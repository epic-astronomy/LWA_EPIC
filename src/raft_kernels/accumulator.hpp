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
#include "../ex/metrics.hpp"
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
  uint64_t _seq_end{0};
  int64_t m_init_chan0{0};
  bool m_freq_change{false};

  PSTensor<float> m_in_tensor;
  PSTensor<float> m_out_tensor;

  _Pld m_cur_buf;
  size_t m_accum_count{0};

  unsigned int m_rt_gauge_id{0};
  Timer m_timer;

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
    m_rt_gauge_id = PrometheusExporter::AddRuntimeSummaryLabel(
        {{"type", "exec_time"},
         {"kernel", "accumulator"},
         {"units", "s"},
         {"kernel_id", std::to_string(this->get_id())}});
  }

  void increment_count() { m_accum_count++; }

  raft::kstatus run() override {
    m_timer.Tick();
    if (m_accum_count == 0) {
      // store the current gulp
      input["in_img"].pop(m_cur_buf);
      m_in_tensor.assign_data(m_cur_buf.get_mbuf()->GetDataPtr());
      m_init_chan0 =
          std::get<int64_t>(m_cur_buf.get_mbuf()->GetMetadataRef()["chan0"]);
      _seq_end =
          std::get<uint64_t>(m_cur_buf.get_mbuf()->GetMetadataRef()["seq_end"]);
    } else {
      // add the next gulp to the current state
      _Pld pld2;
      input["in_img"].pop(pld2);
      auto chan0 =
          std::get<int64_t>(pld2.get_mbuf()->GetMetadataRef()["chan0"]);
      auto new_seq_end =
          std::get<uint64_t>(pld2.get_mbuf()->GetMetadataRef()["seq_end"]);
      auto nseqs = std::get<int>(pld2.get_mbuf()->GetMetadataRef()["nseqs"]);
      // if there is a change in the frequency or
      // if the image arrives with a gap
      // ignore the current accumulation and start a new one
      if (chan0 != m_init_chan0 || (new_seq_end - _seq_end) > nseqs) {
        m_in_tensor.dissociate_data();
        m_out_tensor.dissociate_data();

        m_cur_buf = pld2;
        m_in_tensor.assign_data(m_cur_buf.get_mbuf()->GetDataPtr());
        m_init_chan0 = chan0;
        m_accum_count = 1;

        return raft::proceed;
      }
      _seq_end = new_seq_end;
      m_out_tensor.assign_data(pld2.get_mbuf()->GetDataPtr());

      m_in_tensor += m_out_tensor;
    }

    ++m_accum_count;

    if (m_accum_count == m_naccum) {
      auto& meta = m_cur_buf.get_mbuf()->GetMetadataRef();
      if (_seq_end > 0) {
        meta["seq_end"] = _seq_end;
      }
      meta["img_len_ms"] = std::get<double>(meta["gulp_len_ms"]) * m_naccum;
      output["out_img"].push(m_cur_buf);
      m_cur_buf = _Pld();
      m_accum_count = 0;
      _seq_end = 0;

      m_in_tensor.dissociate_data();
      m_out_tensor.dissociate_data();
    }

    m_timer.Tock();
    PrometheusExporter::ObserveRunTimeValue(m_rt_gauge_id, m_timer.Duration());
    return raft::proceed;
  }
};

#endif  // SRC_RAFT_KERNELS_ACCUMULATOR_HPP_
