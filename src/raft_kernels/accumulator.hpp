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

#ifndef SRC_RAFT_KERNELS_ACCUMULATOR_HPP_
#define SRC_RAFT_KERNELS_ACCUMULATOR_HPP_
#include <glog/logging.h>

#include <chrono>
#include <cmath>
#include <memory>
#include <raft>
#include <raftio>
#include <variant>

#include "../ex/buffer.hpp"
#include "../ex/constants.h"
#include "../ex/py_funcs.hpp"
#include "../ex/tensor.hpp"
#include "../ex/types.hpp"

template <class _Pld>
class AccumulatorRft : public raft::kernel {
 private:
  bool m_is_first_gulp{true};
  // gulp size in ms
  size_t m_gulp_size{40};
  // Number of gulps to accumulate
  size_t m_naccum{4};
  static constexpr size_t m_nbuffers{20};
  size_t m_xdim{128};
  size_t m_ydim{128};
  size_t m_in_nchan{32};

  PSTensor<float> m_in_tensor;
  PSTensor<float> m_out_tensor;

  _Pld m_cur_buf;
  size_t m_accum_count{0};

 public:
  AccumulatorRft(size_t p_xdim, size_t p_ydim, size_t p_nchan, size_t p_naccum)
      : raft::kernel(),
        m_naccum(p_naccum),
        m_xdim(p_xdim),
        m_ydim(p_ydim),
        m_in_nchan(p_nchan),
        m_in_tensor(p_nchan, p_xdim, p_xdim),
        m_out_tensor(p_nchan, p_xdim, p_ydim) {
    input.addPort<_Pld>("in_img");
    output.addPort<_Pld>("out_img");
  }

  void increment_count() { m_accum_count++; }

  raft::kstatus run() override {
    if (m_accum_count == 0) {
      // store the current gulp
      input["in_img"].pop(m_cur_buf);
      m_in_tensor.assign_data(m_cur_buf.get_mbuf()->get_data_ptr());
    } else {
      // add the next gulp to the current state
      _Pld pld2;
      input["in_img"].pop(pld2);
      m_out_tensor.assign_data(pld2.get_mbuf()->get_data_ptr());

      m_in_tensor += m_out_tensor;
    }

    m_accum_count++;

    if (m_accum_count == m_naccum) {
      output["out_img"].push(m_cur_buf);
      m_cur_buf = _Pld();
      m_accum_count = 0;

      m_in_tensor.dissociate_data();
      m_out_tensor.dissociate_data();
    }

    return raft::proceed;
  }
};

#endif  // SRC_RAFT_KERNELS_ACCUMULATOR_HPP_
