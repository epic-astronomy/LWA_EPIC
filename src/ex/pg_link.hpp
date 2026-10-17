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

#ifndef SRC_EX_PG_LINK_HPP_
#define SRC_EX_PG_LINK_HPP_

#include <glog/logging.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "pqxx/pqxx"

/**
 * @brief Allows one log report per interval and counts the ones held back.
 */
class LogThrottle {
 private:
  std::chrono::steady_clock::duration m_every;
  std::chrono::steady_clock::time_point m_last{};
  bool m_first{true};
  size_t m_held{0};
  std::mutex m_mtx;

 public:
  explicit LogThrottle(std::chrono::seconds p_every) : m_every(p_every) {}

  /// True when a report is due; *p_held is then the number held back since.
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

/**
 * @brief The connection string of the node's own database.
 *
 * An explicit string wins; otherwise EPIC_NODE_DSN; otherwise "" and libpq
 * reads PGHOST/PGHOSTADDR, PGPORT, PGDATABASE, PGUSER, PGPASSFILE from the
 * environment. A connection timeout of 5 s is added unless one is set.
 */
inline std::string NodeDbConnString(const std::string& p_conn_str = "") {
  std::string conn = p_conn_str;
  if (conn.empty()) {
    const char* env = std::getenv("EPIC_NODE_DSN");
    conn = env != nullptr ? env : "";
  }
  if (conn.find("connect_timeout") == std::string::npos &&
      std::getenv("PGCONNECT_TIMEOUT") == nullptr) {
    // only for key=value strings; a URI keeps whatever it says
    if (conn.rfind("postgres", 0) != 0) {
      conn += (conn.empty() ? "" : " ");
      conn += "connect_timeout=5";
    }
  }
  return conn;
}

/**
 * @brief Writes to a PostgreSQL database off the caller's thread, and never
 * brings the imager down.
 *
 * Submit() queues a write (a function run in its own transaction, together
 * with the prepared statements it needs) and returns at once. One writer
 * thread connects, prepares the statements on each new connection, and runs
 * the writes in order.
 *
 *  - Database unavailable: the writes wait in the queue (up to p_capacity)
 *    while the connection is retried every p_retry; beyond that, new writes
 *    are dropped. Logs are rate-limited.
 *  - Connection lost during a write: the write is put back and retried after
 *    reconnecting.
 *  - A write the server refuses (a missing partition, say): the write is
 *    dropped and the error logged (rate-limited); later writes go on.
 *  - Every statement has a time limit (p_statement_timeout_ms), so a hung
 *    database holds only the writer thread, never the pipeline.
 *
 * On destruction the writer is given a grace period to drain; if it is still
 * blocked it is detached. Its state is shared, so a detached writer never
 * touches freed memory.
 */
class PgLink {
 public:
  using Work = std::function<void(pqxx::work&)>;
  using Statements = std::vector<std::pair<std::string, std::string>>;

 private:
  struct Job {
    Statements stmts;
    Work fn;
    std::string what;
  };

  struct State {
    std::string name;
    std::string conn_str;
    std::chrono::seconds retry;
    int statement_timeout_ms;
    size_t capacity;

    std::mutex mtx;
    std::condition_variable cv;
    std::deque<Job> queue;
    bool stop{false};

    std::atomic<bool> connected{false};
    std::atomic<size_t> written{0};
    std::atomic<size_t> dropped{0};
    std::atomic<size_t> failed{0};
    std::atomic<size_t> reconnects{0};

    LogThrottle down_log{std::chrono::seconds(300)};
    LogThrottle drop_log{std::chrono::seconds(300)};
    LogThrottle error_log{std::chrono::seconds(300)};
  };

  std::shared_ptr<State> m_state;
  std::thread m_thread;
  std::future<void> m_finished;
  std::chrono::milliseconds m_grace;

  // ------------------------------------------------------- writer thread
  static std::unique_ptr<pqxx::connection> Connect(State* st) {
    try {
      auto conn = std::make_unique<pqxx::connection>(st->conn_str);
      pqxx::nontransaction(*conn).exec0(
          "SET statement_timeout = " + std::to_string(st->statement_timeout_ms));
      if (st->reconnects++ > 0) {
        LOG(INFO) << st->name << ": database connection restored";
      }
      st->connected = true;
      return conn;
    } catch (const std::exception& e) {
      st->connected = false;
      size_t held;
      if (st->down_log.Due(&held)) {
        LOG(WARNING) << st->name << ": database unavailable (retrying every "
                     << st->retry.count() << " s, " << Pending(st)
                     << " writes waiting): " << e.what();
      }
      return nullptr;
    }
  }

  static size_t Pending(State* st) {
    std::lock_guard<std::mutex> lk(st->mtx);
    return st->queue.size();
  }

  static void Serve(std::shared_ptr<State> st, std::promise<void> finished) {
    std::unique_ptr<pqxx::connection> conn;
    std::set<std::string> prepared;
    for (;;) {
      Job job;
      {
        std::unique_lock<std::mutex> lk(st->mtx);
        st->cv.wait(lk, [&] { return st->stop || !st->queue.empty(); });
        if (st->queue.empty()) break;  // stopped and drained
      }

      if (!conn || !conn->is_open()) {
        conn.reset();
        prepared.clear();
        conn = Connect(st.get());
        if (!conn) {
          // wait for the next attempt (or the end); writes keep queueing
          std::unique_lock<std::mutex> lk(st->mtx);
          st->cv.wait_for(lk, st->retry, [&] { return st->stop; });
          if (st->stop) break;  // leave the queue: nothing can write it
          continue;
        }
      }

      {
        std::lock_guard<std::mutex> lk(st->mtx);
        if (st->queue.empty()) continue;
        job = std::move(st->queue.front());
        st->queue.pop_front();
      }

      try {
        for (auto& [id, sql] : job.stmts) {
          if (prepared.count(id) == 0) {
            conn->prepare(id, sql);
            prepared.insert(id);
          }
        }
        pqxx::work tx(*conn);
        job.fn(tx);
        tx.commit();
        ++st->written;
      } catch (const pqxx::broken_connection& e) {
        // not written: put it back and reconnect
        st->connected = false;
        conn.reset();
        {
          std::lock_guard<std::mutex> lk(st->mtx);
          st->queue.push_front(std::move(job));
        }
        size_t held;
        if (st->down_log.Due(&held)) {
          LOG(WARNING) << st->name << ": lost the database connection: "
                       << e.what();
        }
      } catch (const std::exception& e) {
        ++st->failed;
        if (conn && !conn->is_open()) {
          st->connected = false;
          conn.reset();
        }
        size_t held;
        if (st->error_log.Due(&held)) {
          LOG(ERROR) << st->name << ": " << job.what
                     << " not written: " << e.what() << " (" << held
                     << " more since the last report, " << st->failed
                     << " in all)";
        }
      } catch (...) {
        ++st->failed;  // nothing may escape the thread
      }
    }
    finished.set_value();
  }

 public:
  /**
   * @param p_name Name used in the logs (the kernel)
   * @param p_conn_str libpq connection string; "" for NodeDbConnString()
   * @param p_capacity Writes kept while the database is unavailable
   * @param p_retry Interval between connection attempts
   * @param p_statement_timeout_ms Time limit of every statement
   */
  explicit PgLink(std::string p_name, std::string p_conn_str = "",
                  size_t p_capacity = 1024,
                  std::chrono::seconds p_retry = std::chrono::seconds(10),
                  int p_statement_timeout_ms = 10000,
                  std::chrono::milliseconds p_grace = std::chrono::seconds(5))
      : m_state(std::make_shared<State>()), m_grace(p_grace) {
    m_state->name = std::move(p_name);
    m_state->conn_str = NodeDbConnString(p_conn_str);
    m_state->retry = p_retry;
    m_state->statement_timeout_ms = p_statement_timeout_ms;
    m_state->capacity = p_capacity > 0 ? p_capacity : 1;
    std::promise<void> finished;
    m_finished = finished.get_future();
    m_thread = std::thread(Serve, m_state, std::move(finished));
  }

  PgLink(const PgLink&) = delete;
  PgLink& operator=(const PgLink&) = delete;

  /**
   * @brief Queue a write. Never blocks.
   *
   * @param p_what What is written, for the logs
   * @param p_stmts Prepared statements (id, SQL) the write uses
   * @param p_fn Runs the write in the given transaction
   * @return false when the queue is full and the write was dropped
   */
  bool Submit(std::string p_what, Statements p_stmts, Work p_fn) {
    bool queued = false;
    size_t pending = 0;
    {
      std::lock_guard<std::mutex> lk(m_state->mtx);
      if (!m_state->stop && m_state->queue.size() < m_state->capacity) {
        m_state->queue.push_back(
            Job{std::move(p_stmts), std::move(p_fn), std::move(p_what)});
        queued = true;
      }
      pending = m_state->queue.size();
    }
    if (queued) {
      m_state->cv.notify_one();
      return true;
    }
    ++m_state->dropped;
    size_t held;
    if (m_state->drop_log.Due(&held)) {
      LOG(ERROR) << m_state->name << ": " << pending
                 << " writes waiting for the database; dropping new ones ("
                 << held << " more since the last report, "
                 << m_state->dropped << " in all)";
    }
    return false;
  }

  bool Connected() const { return m_state->connected; }
  size_t Written() const { return m_state->written; }
  size_t Dropped() const { return m_state->dropped; }
  size_t Failed() const { return m_state->failed; }
  size_t Pending() const { return Pending(m_state.get()); }

  /// Wait (for tests and shutdown) until the queue is empty or the time is up.
  bool Drain(std::chrono::milliseconds p_max) {
    auto until = std::chrono::steady_clock::now() + p_max;
    while (std::chrono::steady_clock::now() < until) {
      if (Pending() == 0) {
        // the last job may still be running: give it the time of a write
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        if (Pending() == 0) return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return Pending() == 0;
  }

  ~PgLink() {
    {
      std::lock_guard<std::mutex> lk(m_state->mtx);
      m_state->stop = true;
    }
    m_state->cv.notify_all();
    if (m_finished.wait_for(m_grace) == std::future_status::ready) {
      m_thread.join();
    } else {
      m_thread.detach();  // blocked in the database: the process may exit
    }
    size_t left = Pending(m_state.get());
    if (left > 0) {
      LOG(WARNING) << m_state->name << ": " << left
                   << " writes not written at shutdown";
    }
  }
};

#endif  // SRC_EX_PG_LINK_HPP_
