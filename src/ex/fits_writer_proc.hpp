/*
 Copyright (c) 2023 Karthik Reddy

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

#ifndef SRC_EX_FITS_WRITER_PROC_HPP_
#define SRC_EX_FITS_WRITER_PROC_HPP_

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <variant>

extern char** environ;

/**
 * @brief The long-accumulation writer (src/python/epic_fits_writer.py) in a
 * process of its own.
 *
 * Writing an accumulation from the embedded interpreter held its lock for
 * seconds, which stalled every pipeline stage that calls into Python and
 * exhausted the correlator's buffers. This class starts the writer once, at
 * the first job, sends each accumulation over a socket (a line of JSON and the
 * raw float32 data) and waits for the file name, on the caller's thread: the
 * disk saver's writer thread, never the pipeline. A writer that dies or does
 * not answer within the timeout is ended and started again with the next job;
 * that job is reported as not written.
 *
 * The writer is `$EPIC_PYTHON -m epic_fits_writer` (python3 by default), run
 * with the imager's environment and working directory, so it finds the
 * modules the embedded interpreter finds (PYTHONPATH in the container).
 */
class FitsWriterProc {
 public:
  explicit FitsWriterProc(std::chrono::seconds p_timeout = std::chrono::seconds(900))
      : m_timeout(p_timeout) {
    const char* py = std::getenv("EPIC_PYTHON");
    m_python = (py && *py) ? py : "python3";
  }
  ~FitsWriterProc() { Stop(); }
  FitsWriterProc(const FitsWriterProc&) = delete;
  FitsWriterProc& operator=(const FitsWriterProc&) = delete;

  /**
   * @brief Write one accumulation; returns the FITS path, or "" with the reason in *p_err
   * @param p_data float32 [nchan][grid][grid][4], the imager's order
   */
  template <typename Dict>  // dict_t (types.hpp): a map of string to std::variant
  std::string Write(size_t p_grid, size_t p_nchan, const float* p_data, const std::string& p_out_dir,
                    const Dict& p_meta, std::string* p_err) {
    if (m_fd < 0 && !Start(p_err)) return "";
    const size_t nbytes = p_grid * p_grid * p_nchan * 4 * sizeof(float);
    std::string hdr = "{\"grid_size\":" + std::to_string(p_grid) + ",\"nchan\":" + std::to_string(p_nchan) +
                      ",\"out_dir\":" + Quote(p_out_dir) + ",\"nbytes\":" + std::to_string(nbytes) +
                      ",\"meta\":" + MetaJson(p_meta) + "}\n";
    if (!SendAll(hdr.data(), hdr.size()) || !SendAll(p_data, nbytes)) {
      Fail("the writer stopped reading", p_err);
      return "";
    }
    std::string line;
    if (!ReadLine(&line, p_err)) return "";
    const std::string key = "\"filename\": \"";
    const auto at = line.find(key);
    if (at == std::string::npos) {
      Fail("unexpected reply: " + line, p_err);
      return "";
    }
    std::string name = Unquote(line, at + key.size());
    if (name.empty() && p_err) *p_err = "the writer could not write it (see its messages above)";
    return name;
  }

  /// JSON object of the metadata: integers stay integers, floating-point values keep a decimal point
  template <typename Dict>
  static std::string MetaJson(const Dict& p_meta) {
    std::string s = "{";
    bool first = true;
    for (const auto& [k, v] : p_meta) {
      if (!first) s += ",";
      first = false;
      s += Quote(k) + ":";
      std::visit(
          [&s](auto&& x) {
            using T = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<T, std::string>) {
              s += Quote(x);
            } else if constexpr (std::is_floating_point_v<T>) {
              const double d = static_cast<double>(x);
              if (std::isnan(d)) {
                s += "NaN";
              } else if (std::isinf(d)) {
                s += d > 0 ? "Infinity" : "-Infinity";
              } else {
                char b[40];
                std::snprintf(b, sizeof(b), "%.17g", d);
                std::string t = b;
                if (t.find_first_of(".eE") == std::string::npos) t += ".0";
                s += t;
              }
            } else {
              s += std::to_string(+x);  // + promotes uint8_t to a number
            }
          },
          v);
    }
    return s + "}";
  }

 private:
  std::string m_python;
  std::chrono::seconds m_timeout;
  int m_fd{-1};
  pid_t m_pid{-1};
  std::string m_pending;  // bytes read past the last reply

  static std::string Quote(const std::string& p) {
    std::string s = "\"";
    for (unsigned char c : p) {
      if (c == '"' || c == '\\') {
        s += '\\';
        s += static_cast<char>(c);
      } else if (c < 0x20) {
        char b[8];
        std::snprintf(b, sizeof(b), "\\u%04x", c);
        s += b;
      } else {
        s += static_cast<char>(c);
      }
    }
    return s + "\"";
  }

  // the string starting at p_at of a json.dumps reply (escapes of " and \ only matter here)
  static std::string Unquote(const std::string& p, size_t p_at) {
    std::string s;
    for (size_t i = p_at; i < p.size(); ++i) {
      if (p[i] == '\\' && i + 1 < p.size()) {
        s += p[++i];
      } else if (p[i] == '"') {
        break;
      } else {
        s += p[i];
      }
    }
    return s;
  }

  bool Start(std::string* p_err) {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, sv) != 0) {
      if (p_err) *p_err = std::string("socketpair: ") + std::strerror(errno);
      return false;
    }
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    // the child's end becomes its stdin and stdout (dup2 clears close-on-exec);
    // the imager's other descriptors (sockets, devices) are closed in the child
    posix_spawn_file_actions_adddup2(&fa, sv[1], 0);
    posix_spawn_file_actions_adddup2(&fa, sv[1], 1);
#if defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 34))
    posix_spawn_file_actions_addclosefrom_np(&fa, 3);
#endif
    std::string mod = "epic_fits_writer";
    std::string flag = "-m";
    char* argv[] = {m_python.data(), flag.data(), mod.data(), nullptr};
    pid_t pid;
    const int rc = posix_spawnp(&pid, m_python.c_str(), &fa, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    close(sv[1]);
    if (rc != 0) {
      close(sv[0]);
      if (p_err) *p_err = "cannot start " + m_python + " -m epic_fits_writer: " + std::strerror(rc);
      return false;
    }
    m_fd = sv[0];
    m_pid = pid;
    m_pending.clear();
    return true;
  }

  void Stop() {
    if (m_fd >= 0) {
      close(m_fd);  // the writer sees the end of its input and exits
      m_fd = -1;
    }
    if (m_pid > 0) {
      for (int i = 0; i < 50; ++i) {  // up to 5 s for the current file
        if (waitpid(m_pid, nullptr, WNOHANG) == m_pid) {
          m_pid = -1;
          return;
        }
        usleep(100000);
      }
      kill(m_pid, SIGKILL);
      waitpid(m_pid, nullptr, 0);
      m_pid = -1;
    }
  }

  void Fail(const std::string& p_why, std::string* p_err) {
    if (p_err) *p_err = p_why;
    if (m_pid > 0) kill(m_pid, SIGKILL);
    Stop();
  }

  bool SendAll(const void* p, size_t n) {
    const char* c = static_cast<const char*>(p);
    while (n > 0) {
      const ssize_t k = send(m_fd, c, n, MSG_NOSIGNAL);
      if (k < 0 && errno == EINTR) continue;
      if (k <= 0) return false;
      c += k;
      n -= static_cast<size_t>(k);
    }
    return true;
  }

  bool ReadLine(std::string* p_line, std::string* p_err) {
    const auto deadline = std::chrono::steady_clock::now() + m_timeout;
    while (true) {
      const auto nl = m_pending.find('\n');
      if (nl != std::string::npos) {
        *p_line = m_pending.substr(0, nl);
        m_pending.erase(0, nl + 1);
        return true;
      }
      const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline -
                                                                              std::chrono::steady_clock::now());
      if (left.count() <= 0) {
        Fail("no answer within " + std::to_string(m_timeout.count()) + " s; writer restarted", p_err);
        return false;
      }
      pollfd pfd{m_fd, POLLIN, 0};
      const int r = poll(&pfd, 1, static_cast<int>(std::min<int64_t>(left.count(), 1000)));
      if (r < 0 && errno == EINTR) continue;
      if (r < 0) {
        Fail(std::string("poll: ") + std::strerror(errno), p_err);
        return false;
      }
      if (r == 0) continue;
      char buf[4096];
      const ssize_t k = recv(m_fd, buf, sizeof(buf), 0);
      if (k < 0 && errno == EINTR) continue;
      if (k <= 0) {
        Fail("the writer exited", p_err);
        return false;
      }
      m_pending.append(buf, static_cast<size_t>(k));
    }
  }
};

#endif  // SRC_EX_FITS_WRITER_PROC_HPP_
