// Minimal glog stand-in for building the streamcal I/O tests without the submodule.
#pragma once
#include <cstdlib>
#include <iostream>
struct NullLog { template <class T> NullLog& operator<<(const T&) { return *this; } };
struct FatalLog {
  ~FatalLog() { std::cerr << std::endl; std::abort(); }
  template <class T> FatalLog& operator<<(const T& v) { std::cerr << v; return *this; }
};
#define LOG_INFO std::cerr << "[INFO] "
#define LOG_WARNING std::cerr << "[WARNING] "
#define LOG_ERROR std::cerr << "[ERROR] "
#define LOG_FATAL FatalLog() << "[FATAL] "
#define LOG(sev) LOG_##sev
#define LOG_IF(sev, cond) if (!(cond)) {} else LOG(sev)
#define VLOG(n) NullLog()
