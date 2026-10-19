// CPUs reserved for a busy-polling kernel (the packet receiver), kept free of
// the helper threads.
//
// The packet receiver polls its receive queue on one pinned CPU. Helper
// threads (the OpenMP workers of par_helpers.hpp, the calibration tap's
// publisher, threads started by libraries) take the process's CPUs, and one on
// the receiver's CPU, or on its hyperthread sibling, stalls it long enough for
// the queue to overflow: on epic6 a fifth of the packets and a sixth of the
// gulps were lost every 10 s, in step with the tap's 200 MB tables. Reserve
// takes a CPU and its siblings out of the set the helpers use (Free).
#ifndef SRC_EX_CPU_RESERVE_HPP_
#define SRC_EX_CPU_RESERVE_HPP_

#include <sched.h>

#include <cstdio>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>

namespace cpus {

namespace detail {
inline void AddList(const std::string& p_list, cpu_set_t* p_set) {
  std::stringstream ss(p_list);
  std::string part;
  while (std::getline(ss, part, ',')) {
    if (part.empty() || part == "\n") continue;
    const auto dash = part.find('-');
    const int lo = std::stoi(part.substr(0, dash));
    const int hi = dash == std::string::npos ? lo : std::stoi(part.substr(dash + 1));
    for (int c = lo; c <= hi; ++c) CPU_SET(c, p_set);
  }
}
inline std::mutex& Mutex() {
  static std::mutex m;
  return m;
}
inline cpu_set_t& Reserved() {
  static cpu_set_t s = [] {
    cpu_set_t z;
    CPU_ZERO(&z);
    return z;
  }();
  return s;
}
}  // namespace detail

/// CPUs of the main thread when first called (/proc/self/status Cpus_allowed_list)
inline const cpu_set_t& Process() {
  static const cpu_set_t set = [] {
    cpu_set_t s;
    CPU_ZERO(&s);
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
      if (line.rfind("Cpus_allowed_list:", 0) == 0) detail::AddList(line.substr(line.find(':') + 1), &s);
    }
    if (CPU_COUNT(&s) == 0) sched_getaffinity(0, sizeof(s), &s);
    return s;
  }();
  return set;
}

/// Keep the helpers off p_cpu and its hyperthread siblings
inline void Reserve(int p_cpu) {
  Process();  // the process's set, before anything narrows the main thread
  std::lock_guard<std::mutex> lk(detail::Mutex());
  CPU_SET(p_cpu, &detail::Reserved());
  char path[128];
  std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", p_cpu);
  std::ifstream f(path);
  std::string list;
  if (std::getline(f, list)) detail::AddList(list, &detail::Reserved());
}

/// The process's CPUs without the reserved ones (all of them if nothing else is left)
inline cpu_set_t Free() {
  cpu_set_t s = Process();
  std::lock_guard<std::mutex> lk(detail::Mutex());
  cpu_set_t out;
  CPU_ZERO(&out);
  for (int c = 0; c < CPU_SETSIZE; ++c) {
    if (CPU_ISSET(c, &s) && !CPU_ISSET(c, &detail::Reserved())) CPU_SET(c, &out);
  }
  return CPU_COUNT(&out) ? out : s;
}

/// Move the calling thread to Free()
inline void UseFree() {
  const cpu_set_t s = Free();
  sched_setaffinity(0, sizeof(s), &s);
}

}  // namespace cpus

#endif  // SRC_EX_CPU_RESERVE_HPP_
