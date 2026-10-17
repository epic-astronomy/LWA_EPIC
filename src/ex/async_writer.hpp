/*
 Copyright (c) 2023 The EPIC++ authors

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

#ifndef SRC_EX_ASYNC_WRITER_HPP_
#define SRC_EX_ASYNC_WRITER_HPP_

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

/**
 * @brief A bounded queue served by one worker thread.
 *
 * TryPush never blocks: when the queue is full the job is dropped and counted.
 * A job that blocks (a write to a hung NFS mount) therefore stalls only the
 * worker, never the caller. On destruction the worker is given a grace period
 * to finish its current job; a worker still blocked after it is detached. The
 * state it uses is shared, so a detached worker never touches freed memory.
 */
template <class Job>
class BoundedWorker {
 private:
  struct State {
    std::mutex mtx;
    std::condition_variable cv;
    std::deque<Job> queue;
    bool stop{false};
    std::function<void(Job&)> fn;
    std::atomic<size_t> dropped{0};
    std::atomic<size_t> done{0};
    std::atomic<bool> busy{false};
  };
  std::shared_ptr<State> m_state;
  size_t m_capacity;
  std::thread m_thread;
  std::future<void> m_finished;
  std::chrono::milliseconds m_grace;

  static void Serve(std::shared_ptr<State> st, std::promise<void> finished) {
    for (;;) {
      Job job;
      {
        std::unique_lock<std::mutex> lk(st->mtx);
        st->cv.wait(lk, [&] { return st->stop || !st->queue.empty(); });
        if (st->queue.empty()) break;  // stopped and drained
        job = std::move(st->queue.front());
        st->queue.pop_front();
        st->busy = true;
      }
      try {
        st->fn(job);
      } catch (...) {
        // the job reports its own errors; nothing may escape the thread
      }
      st->busy = false;
      ++st->done;
    }
    finished.set_value();
  }

 public:
  BoundedWorker(size_t p_capacity, std::function<void(Job&)> p_fn,
                std::chrono::milliseconds p_grace = std::chrono::seconds(5))
      : m_state(std::make_shared<State>()),
        m_capacity(p_capacity > 0 ? p_capacity : 1),
        m_grace(p_grace) {
    m_state->fn = std::move(p_fn);
    std::promise<void> finished;
    m_finished = finished.get_future();
    m_thread = std::thread(Serve, m_state, std::move(finished));
  }

  BoundedWorker(const BoundedWorker&) = delete;
  BoundedWorker& operator=(const BoundedWorker&) = delete;

  /// Queue a job; false (and the job dropped) when the queue is full.
  bool TryPush(Job&& p_job) {
    {
      std::lock_guard<std::mutex> lk(m_state->mtx);
      if (m_state->stop || m_state->queue.size() >= m_capacity) {
        ++m_state->dropped;
        return false;
      }
      m_state->queue.push_back(std::move(p_job));
    }
    m_state->cv.notify_one();
    return true;
  }

  size_t Dropped() const { return m_state->dropped; }
  size_t Done() const { return m_state->done; }
  bool Busy() const { return m_state->busy; }
  size_t Pending() const {
    std::lock_guard<std::mutex> lk(m_state->mtx);
    return m_state->queue.size();
  }

  ~BoundedWorker() {
    {
      std::lock_guard<std::mutex> lk(m_state->mtx);
      m_state->stop = true;
    }
    m_state->cv.notify_all();
    if (m_finished.wait_for(m_grace) == std::future_status::ready) {
      m_thread.join();
    } else {
      m_thread.detach();  // blocked in I/O: leave it, the process may exit
    }
  }
};

/**
 * @brief Allows one report per interval and counts the ones held back.
 */
class RateLimit {
 private:
  std::chrono::steady_clock::duration m_every;
  std::chrono::steady_clock::time_point m_last{};
  bool m_first{true};
  size_t m_held{0};
  std::mutex m_mtx;

 public:
  explicit RateLimit(std::chrono::seconds p_every) : m_every(p_every) {}

  /// True when a report is due; *p_held is then the number held back since the last.
  bool Due(size_t* p_held) {
    std::lock_guard<std::mutex> lk(m_mtx);
    auto now = std::chrono::steady_clock::now();
    if (m_first || now - m_last >= m_every) {
      m_first = false;
      m_last = now;
      *p_held = m_held;
      m_held = 0;
      return true;
    }
    ++m_held;
    return false;
  }
};

#endif  // SRC_EX_ASYNC_WRITER_HPP_
