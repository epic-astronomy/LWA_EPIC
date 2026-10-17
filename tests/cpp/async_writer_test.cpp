// The disk saver's writer (src/ex/async_writer.hpp) never blocks the pipeline.
//
// A writer job opens a FIFO for writing, which blocks like a write to a hung
// NFS mount until a reader appears. Meanwhile the "pipeline" keeps pushing
// accumulations: every push must return at once (the queue fills, then drops),
// and destroying the writer while its job is still blocked must not hang.
// A second part writes into a directory that does not exist: the job fails,
// and the next jobs still run.
//
//   g++ -std=c++17 -O2 -pthread -o async_writer_test tests/cpp/async_writer_test.cpp
//   ./async_writer_test <scratch directory>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <string>
#include <thread>

#include "../../src/ex/async_writer.hpp"

using Clock = std::chrono::steady_clock;

static int failures = 0;
#define CHECK(cond)                                                   \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::fprintf(stderr, "FAILED line %d: %s\n", __LINE__, #cond); \
      ++failures;                                                     \
    }                                                                 \
  } while (0)

struct Job {
  std::string path;
};

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s scratch_dir\n", argv[0]);
    return 2;
  }
  const std::string dir = argv[1];
  const std::string fifo = dir + "/hung.fits";
  ::unlink(fifo.c_str());
  if (::mkfifo(fifo.c_str(), 0600) != 0) {
    std::perror("mkfifo");
    return 2;
  }

  double worst_ms = 0;
  {
    BoundedWorker<Job> writer(
        2,
        [](Job& j) {
          int fd = ::open(j.path.c_str(), O_WRONLY);  // blocks: no reader
          if (fd >= 0) ::close(fd);
        },
        std::chrono::milliseconds(500));
    // 200 accumulations arrive while the output hangs
    int accepted = 0;
    for (int i = 0; i < 200; ++i) {
      auto t0 = Clock::now();
      accepted += writer.TryPush(Job{fifo});
      double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      if (ms > worst_ms) worst_ms = ms;
      if (i == 0) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    // one job blocked in open(), two queued, the rest dropped
    CHECK(writer.Busy());
    CHECK(accepted == 3);
    CHECK(writer.Dropped() == 197);
    CHECK(writer.Done() == 0);
    std::printf("hung output: %d accepted, %zu dropped, slowest push %.3f ms\n",
                accepted, writer.Dropped(), worst_ms);
    CHECK(worst_ms < 5.0);
    // destructor: grace period, then the blocked worker is detached
  }

  // destruction while the job is blocked: timed
  double destroy_s;
  {
    auto* w = new BoundedWorker<Job>(
        1,
        [](Job& j) {
          int fd = ::open(j.path.c_str(), O_WRONLY);
          if (fd >= 0) ::close(fd);
        },
        std::chrono::milliseconds(500));
    w->TryPush(Job{fifo});
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    auto t0 = Clock::now();
    delete w;
    destroy_s = std::chrono::duration<double>(Clock::now() - t0).count();
  }
  std::printf("writer destroyed while blocked in %.2f s\n", destroy_s);
  CHECK(destroy_s < 2.0);

  // unwritable output: each job fails on its own, the worker carries on
  {
    int failed = 0, ok = 0;
    {
      BoundedWorker<Job> writer(8, [&](Job& j) {
        int fd = ::open(j.path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) {
          ++failed;
          return;
        }
        ::close(fd);
        ::unlink(j.path.c_str());
        ++ok;
      });
      writer.TryPush(Job{dir + "/missing/dir/x.fits"});
      writer.TryPush(Job{dir + "/missing/dir/y.fits"});
      writer.TryPush(Job{dir + "/ok.fits"});
      for (int i = 0; i < 200 && writer.Done() < 3; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::printf("unwritable output: %d failed, %d written\n", failed, ok);
    CHECK(failed == 2 && ok == 1);
  }

  // rate-limited reports
  {
    RateLimit rl(std::chrono::seconds(60));
    size_t held = 99;
    CHECK(rl.Due(&held) && held == 0);
    CHECK(!rl.Due(&held));
    CHECK(!rl.Due(&held));
  }

  // let the detached writers go: give the FIFO readers so open() returns
  for (int i = 0; i < 2; ++i) {
    int fd = ::open(fifo.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd >= 0) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      ::close(fd);
    }
  }
  ::unlink(fifo.c_str());
  std::printf(failures ? "FAILED\n" : "OK\n");
  return failures ? 1 : 0;
}
