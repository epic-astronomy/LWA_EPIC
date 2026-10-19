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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <raft>
#include <raftio>
#include <variant>

#include "../ex/buffer.hpp"
#include "../ex/constants.h"
#include "../ex/lf_buf_mngr.hpp"
#include "../ex/par_helpers.hpp"
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
  // Fewest images for a partial sum to be written
  size_t m_min_accum{1};
  // Images that may be missing between two in a sum (lost packets leave
  // single gulps empty); a longer gap ends the sum
  size_t m_max_gap{48};
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
  /// The sums are made in buffers of their own: the first image of a sum
  /// may be shared with other kernels (the video reads the imager's buffer)
  std::unique_ptr<LFBufMngr<AlignedBuffer<float>>> m_sum_bufs;
  size_t m_nfloat{0};
  /// threads of the copies and additions (memory-bound; a 256 x 256 image
  /// of 192 channels is 200 MB, added 24 times a second)
  static constexpr int m_nthreads{4};

  /// Start a sum with this image: a copy of its data and metadata
  void StartSum(_Pld& p_first) {
    auto buf = m_sum_bufs->acquire_buf();
    CHECK(static_cast<bool>(buf)) << "Accumulator buffer allocation failed";
    buf.get_mbuf()->GetMetadataRef() = p_first.get_mbuf()->GetMetadataRef();
    par::Copy(buf.get_mbuf()->GetDataPtr(), p_first.get_mbuf()->GetDataPtr(), m_nfloat, m_nthreads);
    m_cur_buf = buf;
    m_in_tensor.assign_data(m_cur_buf.get_mbuf()->GetDataPtr());
  }

  unsigned int m_rt_gauge_id{0};
  Timer m_timer;

 public:
  /**
   * p_min_accum: the fewest images a partial sum (ended by a gap or a change
   * of frequency) must hold to be written; <= 0 means 10% of p_naccum.
   */
  AccumulatorRft(size_t p_xdim, size_t p_ydim, size_t p_nchan, size_t p_naccum,
                 int p_min_accum = -1, int p_max_gap = 48)
      : raft::kernel(),
        m_naccum(p_naccum),
        m_min_accum(p_min_accum > 0
                        ? static_cast<size_t>(p_min_accum)
                        : std::max<size_t>(1, (p_naccum + 9) / 10)),
        m_max_gap(static_cast<size_t>(std::max(0, p_max_gap))),
        m_xdim(p_xdim),
        m_ydim(p_ydim),
        m_in_nchan(p_nchan),
        m_in_tensor(p_nchan, p_xdim, p_xdim),
        m_out_tensor(p_nchan, p_xdim, p_ydim) {
    input.addPort<_Pld>("in_img");
    output.addPort<_Pld>("out_img");
    m_nfloat = p_nchan * p_xdim * p_ydim * NSTOKES;
    // one sum being made, one being written
    m_sum_bufs = std::make_unique<LFBufMngr<AlignedBuffer<float>>>(3, m_nfloat, 5, false);
    m_rt_gauge_id = PrometheusExporter::AddRuntimeSummaryLabel(
        {{"type", "exec_time"},
         {"kernel", "accumulator"},
         {"units", "s"},
         {"kernel_id", std::to_string(this->get_id())}});
  }

  void increment_count() { m_accum_count++; }

  /// Send the current sum on with its true length; it holds m_accum_count images.
  void Emit() {
    auto& meta = m_cur_buf.get_mbuf()->GetMetadataRef();
    if (_seq_end > 0) {
      meta["seq_end"] = _seq_end;
    }
    meta["img_len_ms"] = std::get<double>(meta["img_len_ms"]) * m_accum_count;
    meta["nimg"] = static_cast<int>(m_accum_count);
    output["out_img"].push(m_cur_buf);
  }

  raft::kstatus run() override {
    m_timer.Tick();
    if (m_accum_count == 0) {
      // start a sum with this image
      _Pld first;
      input["in_img"].pop(first);
      StartSum(first);
      first = _Pld();
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
      // A change in frequency or a gap in the images ends the current sum. It
      // is written as a partial sum (with the number of images it holds) when
      // it holds at least m_min_accum images, otherwise dropped; the new image
      // starts the next sum.
      if (chan0 != m_init_chan0 || (new_seq_end - _seq_end) > uint64_t(nseqs) * (1 + m_max_gap)) {
        m_in_tensor.dissociate_data();
        m_out_tensor.dissociate_data();
        // why the sum ends: a change of frequency, or a gap in the images
        LOG(INFO) << "accumulator: sum of " << m_accum_count << " of " << m_naccum << " images ended by "
                  << (chan0 != m_init_chan0 ? "a change of frequency" : "a gap") << " (seq_end " << _seq_end
                  << " -> " << new_seq_end << ", step " << int64_t(new_seq_end - _seq_end) << " for " << nseqs
                  << " sequences per image); " << (m_accum_count >= m_min_accum ? "written" : "dropped");
        if (m_accum_count >= m_min_accum) {
          Emit();
        }
        StartSum(pld2);
        pld2 = _Pld();
        m_init_chan0 = chan0;
        _seq_end = new_seq_end;
        m_accum_count = 0;
      } else {
        _seq_end = new_seq_end;
        par::AddInPlace(m_cur_buf.get_mbuf()->GetDataPtr(), pld2.get_mbuf()->GetDataPtr(), m_nfloat,
                        m_nthreads);
      }
    }

    ++m_accum_count;

    if (m_accum_count == m_naccum) {
      Emit();
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
