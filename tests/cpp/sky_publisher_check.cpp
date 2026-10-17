// Checks of the sky publisher's fast path and its writer thread.
//   g++ -std=c++17 -O3 -march=haswell -pthread sky_publisher_check.cpp -o spc && ./spc [--bench]
// (also builds without -march, then the scalar path is checked against itself)
//
// 1. StokesIFitsOrder (AVX2 when available) is bit-identical to the scalar
//    reference for several grids and binnings, with signed zeros, denormals,
//    infinities and NaNs in the input.
// 2. SkyPublishWorker publishes what it is given, drops (and counts) images
//    offered while it is busy, and never blocks the caller.
// 3. --bench: time of the Stokes I transform for 128 channels of 128 x 128.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "../../src/ex/sky_publisher.hpp"

static int g_fail = 0;
#define CHECK(c, ...)                         \
  do {                                        \
    if (!(c)) {                               \
      std::printf("FAIL: " __VA_ARGS__);      \
      std::printf("\n");                      \
      ++g_fail;                               \
    }                                         \
  } while (0)

static std::vector<float> Image(int nchan, int grid, unsigned seed) {
  std::vector<float> img(size_t(nchan) * grid * grid * 4);
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(-1e4f, 1e4f);
  for (auto& x : img) x = u(rng);
  const float special[] = {0.f, -0.f, std::numeric_limits<float>::denorm_min(),
                           -std::numeric_limits<float>::denorm_min(), 1e38f, -1e38f,
                           std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()};
  for (size_t i = 0; i < img.size(); i += 97) img[i] = special[(i / 97) % 8];
  for (size_t i = 0; i + 1 < img.size(); i += 4 * 61) img[i] = img[i + 1] = -0.f;  // XX + YY = -0
  return img;
}

static void CheckIdentical() {
  for (int grid : {8, 16, 32, 64, 128}) {
    for (int nbin : {1, 2, 4}) {
      const int nchan = 8;
      auto img = Image(nchan, grid, unsigned(grid * 10 + nbin));
      std::vector<float> a(size_t(nchan / nbin) * grid * grid), b(a.size());
      SkyPublisher::StokesIFitsOrder(img.data(), nchan, grid, nbin, a.data());
      SkyPublisher::StokesIFitsOrderScalar(img.data(), nchan, grid, nbin, b.data());
      CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
            "fast path differs from the scalar reference (grid %d, nbin %d)", grid, nbin);
    }
  }
  std::printf("identical: checked grids 8-128, nbin 1/2/4 (%s path)\n",
#if defined(__AVX2__)
              "AVX2"
#else
              "scalar"
#endif
  );
}

static void CheckWorker() {
  const int nchan = 4, grid = 16, nslots = 8;
  const std::string name = "sky_publisher_check_" + std::to_string(getpid());
  SkyPublishWorker w(std::make_unique<SkyPublisher>(name, nslots, nchan, grid, 1));
  auto img = Image(nchan, grid, 7);
  int submitted = 0, refused = 0;
  for (int n = 0; n < 50; ++n) {
    SkyPublisher::ImageInfo info;
    info.seq_start = 1000 + n;
    if (w.TrySubmit(img.data(), nchan, info)) {
      ++submitted;
    } else {
      ++refused;
    }
    // every other image: wait until the worker is done, so some are accepted
    if (n % 2 == 0) {
      while (w.InFlight()) std::this_thread::yield();
    }
  }
  while (w.InFlight()) std::this_thread::yield();
  CHECK(w.Published() == uint64_t(submitted), "published %llu, submitted %d",
        (unsigned long long)w.Published(), submitted);
  CHECK(w.Dropped() == uint64_t(refused), "dropped %llu, refused %d", (unsigned long long)w.Dropped(),
        refused);
  CHECK(!w.Failed(), "worker failed: %s", w.Error().c_str());
  // the last published slot holds the transform of img
  const std::string path = "/dev/shm/" + name;
  FILE* f = std::fopen(path.c_str(), "rb");
  CHECK(f != nullptr, "ring missing");
  if (f) {
    std::vector<char> ring(SkyPublisher::kHeaderBytes + nslots * 4096 * 2);
    size_t got = std::fread(ring.data(), 1, ring.size(), f);
    std::fclose(f);
    uint64_t count, slot_bytes;
    std::memcpy(&count, ring.data() + 40, 8);
    std::memcpy(&slot_bytes, ring.data() + 16, 8);
    CHECK(count == w.Published(), "ring count %llu", (unsigned long long)count);
    const size_t off = SkyPublisher::kHeaderBytes + ((count - 1) % nslots) * slot_bytes;
    std::vector<float> ref(size_t(nchan) * grid * grid);
    SkyPublisher::StokesIFitsOrderScalar(img.data(), nchan, grid, 1, ref.data());
    CHECK(got >= off + SkyPublisher::kSlotHeader + ref.size() * 4 &&
              std::memcmp(ring.data() + off + SkyPublisher::kSlotHeader, ref.data(), ref.size() * 4) == 0,
          "published image differs");
  }
  std::printf("worker: %d published, %d dropped while busy\n", submitted, refused);
  unlink(path.c_str());
}

static void Bench() {
  const int nchan = 128, grid = 128;
  auto img = Image(nchan, grid, 3);
  std::vector<float> out(size_t(nchan) * grid * grid);
  std::vector<char> junk(64 << 20);
  for (int fast = 1; fast >= 0; --fast) {
    std::vector<double> ms;
    for (int it = 0; it < 100; ++it) {
      std::memset(junk.data(), it, junk.size());  // the imager's buffer arrives cold
      auto t0 = std::chrono::steady_clock::now();
      if (fast) {
        SkyPublisher::StokesIFitsOrder(img.data(), nchan, grid, 1, out.data());
      } else {
        SkyPublisher::StokesIFitsOrderScalar(img.data(), nchan, grid, 1, out.data());
      }
      ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    std::printf("bench %s: median %.2f ms, p90 %.2f ms\n", fast ? "fast  " : "scalar", ms[50], ms[90]);
  }
}

int main(int argc, char** argv) {
  CheckIdentical();
  CheckWorker();
  if (argc > 1 && std::string(argv[1]) == "--bench") Bench();
  std::printf(g_fail ? "FAILED (%d)\n" : "OK\n", g_fail);
  return g_fail ? 1 : 0;
}
