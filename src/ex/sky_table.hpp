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
#ifndef SRC_EX_SKY_TABLE_HPP_
#define SRC_EX_SKY_TABLE_HPP_

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

/**
 * @brief The all-sky monitor's pixel table, read by the imager.
 *
 * With FX imaging the imager regrids each image to the monitor's sky pixels
 * on the GPU (FxImager::SetSkyTable) and publishes their spectra instead of
 * the image. The monitor writes the table (src/python/skymon/shm.py,
 * SkyTableWriter) whenever it updates its pixels: the image position of every
 * pixel slot at two times, between which the imager interpolates for each
 * image exactly as the monitor would (geometry.SkyGrid.taps).
 *
 * /dev/shm/<ring>_table, little endian:
 *   0  char[8]  magic "EPICSKYT"
 *   8  u32      layout version (1)
 *  12  u32      capacity P (pixel slots)
 *  16  u64      seq: odd while the table is written (seqlock)
 *  24  u64      version: increases with every table
 *  32  u32      grid (image size the positions refer to)
 *  36  u32      reserved
 *  40  f64      grid_res (deg)
 *  48  f64      t_a (unix): positions xy_a hold at t_a
 *  56  f64      t_b (unix): positions xy_b hold at t_b
 * 128  f64      xy_a[P][2], then xy_b[P][2]: 1-based FITS pixel
 *               coordinates (x, y); NaN for a free slot
 */
struct SkyTableSnapshot {
  uint64_t version{0};
  uint32_t grid{0};
  uint32_t capacity{0};
  double grid_res{0};
  double t_a{0}, t_b{0};
  std::vector<double> xy_a, xy_b;
};

class SkyTableReader {
 public:
  static constexpr char kMagic[8] = {'E', 'P', 'I', 'C', 'S', 'K', 'Y', 'T'};
  static constexpr size_t kHeader = 128;

  explicit SkyTableReader(std::string p_path) : m_path(std::move(p_path)) {}
  ~SkyTableReader() { Close(); }
  SkyTableReader(const SkyTableReader&) = delete;
  SkyTableReader& operator=(const SkyTableReader&) = delete;

  const std::string& Path() const { return m_path; }

  /// A new table since the last call (a consistent copy), or nothing
  std::optional<SkyTableSnapshot> Poll() {
    if (!Map()) return std::nullopt;
    for (int attempt = 0; attempt < 4; ++attempt) {
      const uint64_t s0 = Load<uint64_t>(16);
      if (s0 & 1) continue;  // being written
      std::atomic_thread_fence(std::memory_order_acquire);
      const uint64_t version = Load<uint64_t>(24);
      if (version == m_version) return std::nullopt;
      SkyTableSnapshot t;
      t.version = version;
      t.capacity = Load<uint32_t>(12);
      t.grid = Load<uint32_t>(32);
      t.grid_res = Load<double>(40);
      t.t_a = Load<double>(48);
      t.t_b = Load<double>(56);
      const size_t n = size_t(t.capacity) * 2;
      if (kHeader + 2 * n * sizeof(double) > m_size) {
        Close();  // the monitor made a larger table: map it again next time
        return std::nullopt;
      }
      t.xy_a.resize(n);
      t.xy_b.resize(n);
      std::memcpy(t.xy_a.data(), m_base + kHeader, n * sizeof(double));
      std::memcpy(t.xy_b.data(), m_base + kHeader + n * sizeof(double), n * sizeof(double));
      std::atomic_thread_fence(std::memory_order_acquire);
      if (Load<uint64_t>(16) != s0) continue;  // written meanwhile
      m_version = version;
      return t;
    }
    return std::nullopt;
  }

 private:
  bool Map() {
    if (m_base) return true;
    const int fd = open(m_path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    struct stat st {};
    if (fstat(fd, &st) != 0 || size_t(st.st_size) < kHeader) {
      close(fd);
      return false;
    }
    void* p = mmap(nullptr, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    close(fd);
    if (p == MAP_FAILED) return false;
    if (std::memcmp(p, kMagic, 8) != 0) {
      munmap(p, st.st_size);
      return false;
    }
    m_base = static_cast<const char*>(p);
    m_size = st.st_size;
    return true;
  }
  void Close() {
    if (m_base) munmap(const_cast<char*>(m_base), m_size);
    m_base = nullptr;
    m_size = 0;
  }
  template <typename T>
  T Load(size_t p_off) const {
    T v;
    std::memcpy(&v, m_base + p_off, sizeof(T));
    return v;
  }

  std::string m_path;
  const char* m_base{nullptr};
  size_t m_size{0};
  uint64_t m_version{0};
};

#endif  // SRC_EX_SKY_TABLE_HPP_
