// Minimal glog stand-in for building the streamcal I/O tests without the submodule.
#pragma once
#include <iostream>
struct NullLog { template <class T> NullLog& operator<<(const T&) { return *this; } };
#define LOG(sev) std::cerr << "[" #sev "] "
#define VLOG(n) NullLog()
