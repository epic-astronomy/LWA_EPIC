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

#ifndef SRC_RAFT_KERNELS_INDEX_FETCHER_HPP_
#define SRC_RAFT_KERNELS_INDEX_FETCHER_HPP_

#include <glog/logging.h>

#include <chrono>
#include <cmath>
#include <memory>
#include <raft>
#include <raftio>
#include <variant>

#include "../ex/buffer.hpp"
#include "../ex/constants.h"
#include "../ex/orm_types.hpp"
#include "../ex/py_funcs.hpp"
#include "../ex/tensor.hpp"
#include "../ex/types.hpp"

class IndexFetcherRft : public raft::kernel {
 private:
  unsigned int m_refresh_interval{10};
  using high_res_tp =
      typename std::chrono::time_point<std::chrono::high_resolution_clock>;
  high_res_tp m_prev_refresh;

  bool is_refresh_required() {
    auto now = std::chrono::high_resolution_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now - m_prev_refresh)
            .count() >= m_refresh_interval) {
      m_prev_refresh = now;
      return true;
    }

    return false;
  }

  uint64_t m_tstart;

 public:
  explicit IndexFetcherRft(unsigned int p_refresh_interval = 10)
      : raft::kernel(), m_refresh_interval(p_refresh_interval) {
    input.addPort<uint64_t>("tstart");
    output.addPort<EpicPixelTableMetaRows>("meta_pixel_rows");
    m_prev_refresh = std::chrono::high_resolution_clock::now();
  }

  raft::kstatus run() override {
    input["tstart"].pop(m_tstart);
    if (m_tstart == 0) {
      // output["meta_pixel_rows"].push(EpicPixelTableMetaRows());
      return raft::proceed;
    }
    if (is_refresh_required()) {
      output["meta_pixel_rows"].push(create_dummy_meta(128, 128));
      VLOG(2) << "FIRED INDEX FETCHER";
    }
    return raft::proceed;
  }
};
#endif  // SRC_RAFT_KERNELS_INDEX_FETCHER_HPP_
