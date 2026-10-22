/*
 Copyright (c) 2026 The EPIC++ authors

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

#ifndef SRC_EX_SKY_PUBLISHER_HPP_
#define SRC_EX_SKY_PUBLISHER_HPP_

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

/**
 * @brief Publishes Stokes I images to a shared-memory ring for the all-sky
 * monitor (src/python/skymon).
 *
 * The ring lives in /dev/shm/<name>. The writer never waits for a reader: a
 * reader that falls behind loses the oldest images and notices from the slot
 * locks. The layout is documented in src/python/skymon/shm.py and must match
 * it.
 *
 * Each image is written as XX + YY in the order of the saved FITS images,
 * [chan][y][x], so the monitor can use the FITS WCS (epic_geometry.apply_wcs)
 * directly. The imager's host buffer is [chan][a][b][4]; save_output
 * transposes it to [b][a], applies fftshift and flips b, so FITS pixel
 * (x = i + 1, y = j + 1) is buffer element a = (i + N/2) % N,
 * b = (N - 1 - j + N/2) % N.
 */
class SkyPublisher {
 public:
  static constexpr char kMagic[8] = {'E', 'P', 'I', 'C', 'S', 'K', 'Y', '1'};
  static constexpr uint32_t kVersion = 1;
  static constexpr size_t kHeaderBytes = 4096;
  static constexpr size_t kSlotHeader = 256;

  struct ImageInfo {
    uint64_t seq_start{0};
    uint32_t nseqs{0};
    uint32_t chan0{0};  // absolute NDP channel of input channel 0
    double grid_res{1.0};
    double img_len_ms{0};
    /// calibration gain table the image was made with (0: nominal phases)
    uint64_t gain_version{0};
    /// 1: pix holds the all-sky monitor's pixel spectra [nchan / nbin][npix],
    /// regridded on the GPU with pixel table table_version (sky_table.hpp),
    /// published instead of the image; 0: the image
    uint32_t layout{0};
    uint32_t npix{0};
    uint64_t table_version{0};
    const float* pix{nullptr};
    /// layout 1: bit s set when bright source s (sky_sources.hpp, skymon
    /// shm.PEEL_NAMES) was subtracted from the spectra; its peak per channel,
    /// [kPeelSources][nchan / nbin] float, then follows the spectra in pix,
    /// and after the peaks the subtracted term [nchan / nbin][npix] (the
    /// sources at each pixel; spectra + term is the unpeeled sky)
    uint32_t peel_mask{0};
    /// the subtracted term follows the peaks (the imager sends it every few images)
    bool peel_term{false};
  };
  static constexpr int kPeelSources = 5;

  /**
   * @param p_name ring name in /dev/shm
   * @param p_nslots number of images the ring holds
   * @param p_max_nchan input channels (before binning)
   * @param p_grid image side in pixels
   * @param p_nbin input channels summed into one published channel
   */
  SkyPublisher(const std::string& p_name, int p_nslots, int p_max_nchan, int p_grid,
               int p_nbin = 1)
      : m_nslots(p_nslots), m_grid(p_grid), m_nbin(p_nbin < 1 ? 1 : p_nbin) {
    // Publish() indexes the ring modulo the slot count: none would divide by zero
    if (m_nslots < 1 || m_grid < 1 || p_max_nchan < 1) {
      throw std::invalid_argument("sky publisher: slots, grid and channels must be positive");
    }
    if (p_max_nchan % m_nbin != 0) {
      throw std::invalid_argument("sky publisher: channels not divisible by the binning");
    }
    m_out_nchan = p_max_nchan / m_nbin;
    size_t data = sizeof(float) * size_t(m_out_nchan) * m_grid * m_grid;
    m_slot_bytes = (kSlotHeader + data + 4095) / 4096 * 4096;
    m_size = kHeaderBytes + m_slot_bytes * m_nslots;
    m_path = "/dev/shm/" + p_name;
    int fd = open(m_path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd < 0) throw std::runtime_error("sky publisher: cannot open " + m_path);
    if (ftruncate(fd, static_cast<off_t>(m_size)) != 0) {
      close(fd);
      throw std::runtime_error("sky publisher: cannot size " + m_path);
    }
    void* p = mmap(nullptr, m_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) throw std::runtime_error("sky publisher: mmap failed");
    m_base = static_cast<char*>(p);
    std::memset(m_base, 0, kHeaderBytes);
    std::memcpy(m_base, kMagic, 8);
    put<uint32_t>(8, kVersion);
    put<uint32_t>(12, static_cast<uint32_t>(m_nslots));
    put<uint64_t>(16, m_slot_bytes);
    put<uint64_t>(24, kHeaderBytes);
    put<uint32_t>(32, static_cast<uint32_t>(m_out_nchan));
    put<uint32_t>(36, static_cast<uint32_t>(m_grid));
    put<uint64_t>(40, 0);
    put<uint32_t>(48, static_cast<uint32_t>(getpid()));
    put<uint64_t>(56, static_cast<uint64_t>(
                          std::chrono::system_clock::now().time_since_epoch().count()));
    const char* mode = std::getenv("EPIC_PHASE_CENTER");
    m_phase_mode = (mode && std::string(mode) == "zenith") ? 1 : 0;
  }

  ~SkyPublisher() {
    if (m_base) munmap(m_base, m_size);
  }
  SkyPublisher(const SkyPublisher&) = delete;
  SkyPublisher& operator=(const SkyPublisher&) = delete;

  /**
   * @brief Publish one image.
   * @param p_img host image [nchan][grid][grid][4] (XX, YY, Re XY, Im XY)
   * @param p_nchan channels in p_img (a multiple of the binning)
   */
  void Publish(const float* p_img, int p_nchan, const ImageInfo& p_info) {
    const int nout = p_nchan / m_nbin;
    if (nout > m_out_nchan || nout < 1) return;
    const uint64_t n = m_count;
    char* slot = m_base + kHeaderBytes + (n % m_nslots) * m_slot_bytes;
    auto* lock = reinterpret_cast<std::atomic<uint64_t>*>(slot);
    lock->store(2 * n + 1, std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_release);
    put_at<uint64_t>(slot, 8, p_info.seq_start);
    put_at<uint32_t>(slot, 16, p_info.nseqs);
    put_at<uint32_t>(slot, 20, p_info.chan0);
    put_at<uint32_t>(slot, 24, static_cast<uint32_t>(nout));
    put_at<uint32_t>(slot, 28, static_cast<uint32_t>(m_nbin));
    put_at<uint32_t>(slot, 32, static_cast<uint32_t>(m_grid));
    put_at<uint32_t>(slot, 36, m_phase_mode);
    put_at<double>(slot, 40, p_info.grid_res);
    put_at<double>(slot, 48, p_info.img_len_ms);
    put_at<uint64_t>(slot, 56, p_info.gain_version);
    float* out = reinterpret_cast<float*>(slot + kSlotHeader);
    const bool pixels = p_info.layout == 1 && p_info.pix &&
                        size_t(nout) * (p_info.npix + kPeelSources) * sizeof(float) + kSlotHeader <= m_slot_bytes;
    const uint32_t peel = pixels ? p_info.peel_mask : 0u;
    const bool term = peel && p_info.peel_term && size_t(nout) * (2 * size_t(p_info.npix) + kPeelSources) * sizeof(float) +
                                      kSlotHeader <= m_slot_bytes;
    put_at<uint32_t>(slot, 64, pixels ? 1u : 0u);
    put_at<uint32_t>(slot, 68, pixels ? p_info.npix : 0u);
    put_at<uint64_t>(slot, 72, pixels ? p_info.table_version : 0u);
    put_at<uint32_t>(slot, 80, peel);
    put_at<uint32_t>(slot, 84, term ? 1u : 0u);
    if (pixels) {
      const size_t nf = peel ? (term ? 2 * size_t(p_info.npix) : size_t(p_info.npix)) + kPeelSources
                             : size_t(p_info.npix);
      std::memcpy(out, p_info.pix, size_t(nout) * nf * sizeof(float));
    } else {
      StokesIFitsOrder(p_img, p_nchan, m_grid, m_nbin, out);
    }
    std::atomic_thread_fence(std::memory_order_release);
    lock->store(2 * n + 2, std::memory_order_release);
    m_count = n + 1;
    reinterpret_cast<std::atomic<uint64_t>*>(m_base + 40)
        ->store(m_count, std::memory_order_release);
  }

  /**
   * @brief XX + YY of a [nchan][N][N][4] buffer in FITS order [nchan/nbin][y][x],
   * summing nbin adjacent channels.
   *
   * With AVX2 and N a multiple of 16, 8 x 8 blocks are summed and transposed in
   * registers (about 2 ms instead of 4-9 ms for 128 channels of 128 x 128).
   * The result is bit-identical to StokesIFitsOrderScalar: every output is
   * 0 + (XX + YY) summed over the nbin channels in the same order.
   */
  static void StokesIFitsOrder(const float* p_img, int p_nchan, int p_grid, int p_nbin,
                               float* p_out) {
#if defined(__AVX2__)
    if (p_grid % 16 == 0) {
      StokesIFitsOrderAvx2(p_img, p_nchan, p_grid, p_nbin, p_out);
      return;
    }
#endif
    StokesIFitsOrderScalar(p_img, p_nchan, p_grid, p_nbin, p_out);
  }

  /// Reference implementation (any grid size)
  static void StokesIFitsOrderScalar(const float* p_img, int p_nchan, int p_grid, int p_nbin,
                                     float* p_out) {
    const int N = p_grid;
    const int half = N / 2;
    const int nout = p_nchan / p_nbin;
    for (int co = 0; co < nout; ++co) {
      float* o = p_out + size_t(co) * N * N;
      for (int j = 0; j < N; ++j) {
        const int b = (N - 1 - j + half) % N;
        for (int i = 0; i < N; ++i) {
          const int a = (i + half) % N;
          float s = 0.f;
          for (int k = 0; k < p_nbin; ++k) {
            const float* px =
                p_img + ((size_t(co * p_nbin + k) * N + a) * N + b) * 4;
            s += px[0] + px[1];
          }
          o[size_t(j) * N + i] = s;
        }
      }
    }
  }

#if defined(__AVX2__)
  /// XX + YY of 8 consecutive pixels [b][4] -> 8 floats in pixel order
  static inline __m256 XxPlusYy8(const float* p_px) {
    const __m256 v0 = _mm256_loadu_ps(p_px), v1 = _mm256_loadu_ps(p_px + 8);
    const __m256 v2 = _mm256_loadu_ps(p_px + 16), v3 = _mm256_loadu_ps(p_px + 24);
    // hadd: [p0 XX+YY, p0 XY, p2, p2 | p1, p1, p3, p3] (XX + YY evaluated as XX + YY)
    const __m256 h01 = _mm256_hadd_ps(v0, v1);
    const __m256 h23 = _mm256_hadd_ps(v2, v3);
    const __m256 s = _mm256_shuffle_ps(h01, h23, _MM_SHUFFLE(2, 0, 2, 0));  // [p0 p2 p4 p6 | p1 p3 p5 p7]
    return _mm256_permutevar8x32_ps(s, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
  }

  static void StokesIFitsOrderAvx2(const float* p_img, int p_nchan, int p_grid, int p_nbin,
                                   float* p_out) {
    const int N = p_grid;  // a multiple of 16, so i runs over 8 consecutive values per block
    const int half = N / 2;
    const int nout = p_nchan / p_nbin;
    const __m256 zero = _mm256_setzero_ps();
    for (int co = 0; co < nout; ++co) {
      float* o = p_out + size_t(co) * N * N;
      for (int a0 = 0; a0 < N; a0 += 8) {
        const int i0 = (a0 + half) % N;
        for (int b0 = 0; b0 < N; b0 += 8) {
          __m256 r[8];  // r[t][u]: pixel (a0 + t, b0 + u)
          for (int t = 0; t < 8; ++t) {
            __m256 acc = zero;
            for (int k = 0; k < p_nbin; ++k) {
              const float* px =
                  p_img + ((size_t(co * p_nbin + k) * N + a0 + t) * N + b0) * 4;
              acc = _mm256_add_ps(acc, XxPlusYy8(px));
            }
            r[t] = acc;
          }
          // 8 x 8 transpose: c[u][t] = r[t][u]
          const __m256 t0 = _mm256_unpacklo_ps(r[0], r[1]), t1 = _mm256_unpackhi_ps(r[0], r[1]);
          const __m256 t2 = _mm256_unpacklo_ps(r[2], r[3]), t3 = _mm256_unpackhi_ps(r[2], r[3]);
          const __m256 t4 = _mm256_unpacklo_ps(r[4], r[5]), t5 = _mm256_unpackhi_ps(r[4], r[5]);
          const __m256 t6 = _mm256_unpacklo_ps(r[6], r[7]), t7 = _mm256_unpackhi_ps(r[6], r[7]);
          const __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44), s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
          const __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44), s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
          const __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44), s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
          const __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44), s7 = _mm256_shuffle_ps(t5, t7, 0xEE);
          const __m256 c[8] = {_mm256_permute2f128_ps(s0, s4, 0x20), _mm256_permute2f128_ps(s1, s5, 0x20),
                               _mm256_permute2f128_ps(s2, s6, 0x20), _mm256_permute2f128_ps(s3, s7, 0x20),
                               _mm256_permute2f128_ps(s0, s4, 0x31), _mm256_permute2f128_ps(s1, s5, 0x31),
                               _mm256_permute2f128_ps(s2, s6, 0x31), _mm256_permute2f128_ps(s3, s7, 0x31)};
          for (int u = 0; u < 8; ++u) {
            const int j = (N - 1 - (b0 + u) + half) % N;
            _mm256_storeu_ps(o + size_t(j) * N + i0, c[u]);
          }
        }
      }
    }
  }
#endif

  uint64_t Count() const { return m_count; }
  const std::string& Path() const { return m_path; }

 private:
  template <typename T>
  void put(size_t p_off, T p_v) {
    std::memcpy(m_base + p_off, &p_v, sizeof(T));
  }
  template <typename T>
  static void put_at(char* p_base, size_t p_off, T p_v) {
    std::memcpy(p_base + p_off, &p_v, sizeof(T));
  }

  int m_nslots;
  int m_grid;
  int m_nbin;
  int m_out_nchan{0};
  size_t m_slot_bytes{0};
  size_t m_size{0};
  std::string m_path;
  char* m_base{nullptr};
  uint64_t m_count{0};
  uint32_t m_phase_mode{0};
};

/**
 * @brief Runs SkyPublisher::Publish on its own thread, one image at a time.
 *
 * The imager's thread hands over an image with TrySubmit and returns at once.
 * While an image is still being published, further images are dropped and
 * counted rather than queued, so a slow or starved publisher costs the monitor
 * images but never delays the imager. The caller keeps the image's buffer
 * alive until InFlight() is false; the worker only reads it.
 *
 * The thread inherits the CPU affinity of the thread that constructs it.
 */
class SkyPublishWorker {
 public:
  explicit SkyPublishWorker(std::unique_ptr<SkyPublisher> p_pub)
      : m_pub(std::move(p_pub)), m_thread([this] { Serve(); }) {}

  ~SkyPublishWorker() {
    {
      std::lock_guard<std::mutex> lk(m_mtx);
      m_stop = true;
    }
    m_cv.notify_all();
    m_thread.join();
  }
  SkyPublishWorker(const SkyPublishWorker&) = delete;
  SkyPublishWorker& operator=(const SkyPublishWorker&) = delete;

  /// Hand over an image; false (counted as dropped) while the previous one is in flight.
  bool TrySubmit(const float* p_img, int p_nchan, const SkyPublisher::ImageInfo& p_info) {
    if (m_in_flight.load(std::memory_order_acquire) || m_failed.load(std::memory_order_acquire)) {
      ++m_dropped;
      return false;
    }
    {
      std::lock_guard<std::mutex> lk(m_mtx);
      m_img = p_img;
      m_nchan = p_nchan;
      m_info = p_info;
      m_in_flight.store(true, std::memory_order_release);
    }
    m_cv.notify_one();
    return true;
  }

  /// Count an image that was not offered because the previous one is in flight.
  void CountDropped() { ++m_dropped; }

  /// True from TrySubmit until the worker has finished reading the image.
  bool InFlight() const { return m_in_flight.load(std::memory_order_acquire); }
  /// The publisher failed (the error is in Error()); nothing more is published.
  bool Failed() const { return m_failed.load(std::memory_order_acquire); }
  std::string Error() const {
    std::lock_guard<std::mutex> lk(m_mtx);
    return m_error;
  }
  uint64_t Published() const { return m_published.load(); }
  uint64_t Dropped() const { return m_dropped.load(); }
  /// Duration of the last publish, ms
  double LastMs() const { return m_last_ms.load(); }
  const std::string& Path() const { return m_pub->Path(); }

 private:
  void Serve() {
    for (;;) {
      const float* img;
      int nchan;
      SkyPublisher::ImageInfo info;
      {
        std::unique_lock<std::mutex> lk(m_mtx);
        m_cv.wait(lk, [&] { return m_stop || m_in_flight.load(std::memory_order_acquire); });
        if (m_stop) return;
        img = m_img;
        nchan = m_nchan;
        info = m_info;
      }
      const auto t0 = std::chrono::steady_clock::now();
      try {
        m_pub->Publish(img, nchan, info);
        ++m_published;
      } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(m_mtx);
        m_error = e.what();
        m_failed.store(true, std::memory_order_release);
      }
      m_last_ms.store(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      m_in_flight.store(false, std::memory_order_release);
    }
  }

  std::unique_ptr<SkyPublisher> m_pub;
  mutable std::mutex m_mtx;
  std::condition_variable m_cv;
  bool m_stop{false};
  const float* m_img{nullptr};
  int m_nchan{0};
  SkyPublisher::ImageInfo m_info;
  std::string m_error;
  std::atomic<bool> m_in_flight{false};
  std::atomic<bool> m_failed{false};
  std::atomic<uint64_t> m_published{0};
  std::atomic<uint64_t> m_dropped{0};
  std::atomic<double> m_last_ms{0};
  std::thread m_thread;  // last: starts after the members it uses
};

#endif  // SRC_EX_SKY_PUBLISHER_HPP_
