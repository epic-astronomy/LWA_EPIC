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

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

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
  };

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
    StokesIFitsOrder(p_img, p_nchan, m_grid, m_nbin, out);
    std::atomic_thread_fence(std::memory_order_release);
    lock->store(2 * n + 2, std::memory_order_release);
    m_count = n + 1;
    reinterpret_cast<std::atomic<uint64_t>*>(m_base + 40)
        ->store(m_count, std::memory_order_release);
  }

  /**
   * @brief XX + YY of a [nchan][N][N][4] buffer in FITS order [nchan/nbin][y][x],
   * summing nbin adjacent channels.
   */
  static void StokesIFitsOrder(const float* p_img, int p_nchan, int p_grid, int p_nbin,
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

#endif  // SRC_EX_SKY_PUBLISHER_HPP_
