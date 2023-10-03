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

#ifndef SRC_RAFT_KERNELS_CHAN_REDUCER_HPP_
#define SRC_RAFT_KERNELS_CHAN_REDUCER_HPP_

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

/**
 * @brief Raft kernel bin the channels in the input image
 *
 * @tparam _PldIn Input payload type
 * @tparam BufferMngr Buffer manager type
 * @tparam _PldOut Output payload type
 */
template <typename _PldIn, class BufferMngr, typename _PldOut = _PldIn>
class ChanReducerRft : public raft::kernel {
 private:
  /// @brief Number of channels to combine
  const int m_ncombine{4};
  float m_norm{1};
  bool m_is_chan_avg{false};
  std::unique_ptr<BufferMngr> m_buf_mngr{nullptr};
  static constexpr size_t m_nbufs{50};
  static constexpr size_t m_max_buf_reqs{10};
  size_t m_xdim{128};
  size_t m_ydim{128};
  size_t m_in_nchan{128};
  size_t m_out_nchan{32};
  PSTensor<float> m_in_tensor;
  PSTensor<float> m_out_tensor;
  static constexpr int NSTOKES{4};

 public:
  /**
   * @brief Construct a new ChanReducerRft object
   *
   * @param p_ncombine Channel binning factor
   * @param p_xdim X side of the image
   * @param p_ydim Y side of the image
   * @param p_in_nchan Number of input channels to the reducer
   */
  ChanReducerRft(int p_ncombine, int p_xdim, int p_ydim, int p_in_nchan)
      : raft::kernel(),
        m_ncombine(p_ncombine),
        m_xdim(p_xdim),
        m_ydim(p_ydim),
        m_in_nchan(p_in_nchan),
        m_in_tensor(PSTensor<float>(m_in_nchan, m_xdim, m_ydim)),
        m_out_tensor(
            PSTensor<float>(size_t(p_in_nchan / p_ncombine), m_xdim, m_ydim)) {
    input.addPort<_PldIn>("in_img");
    output.addPort<_PldOut>("out_img");
    output.addPort<uint64_t>("seq_start_id");

    if (m_in_nchan % m_ncombine != 0) {
      LOG(FATAL) << "The number of output channels: " << m_in_nchan
                 << " cannot be binned by a factor of " << m_ncombine << ". ";
    }

    m_out_nchan = m_in_nchan / m_ncombine;
    m_buf_mngr.reset(new BufferMngr(m_nbufs,
                                    m_xdim * m_ydim * m_out_nchan * NSTOKES,
                                    m_max_buf_reqs, false));
  }

  raft::kstatus run() override {
    _PldIn in_pld;
    input["in_img"].pop(in_pld);

    auto out_pld = m_buf_mngr->acquire_buf();
    if (!out_pld) {
      LOG(FATAL) << "Memory buffer full in ChanReducer";
    }

    auto& out_meta = out_pld.get_mbuf()->GetMetadataRef();

    out_meta = in_pld.get_mbuf()->GetMetadataRef();
    out_meta["nchan"] = (uint8_t)m_out_nchan;
    out_meta["chan_width"] = BANDWIDTH * m_ncombine;

    m_in_tensor.assign_data(in_pld.get_mbuf()->GetDataPtr());
    m_out_tensor.assign_data(out_pld.get_mbuf()->GetDataPtr());

    m_in_tensor.combine_channels(&m_out_tensor);

    output["out_img"].push(out_pld);
    output["seq_start_id"].push(std::get<uint64_t>(out_meta["seq_start"]));

    return raft::proceed;
  }
};

#endif  // SRC_RAFT_KERNELS_CHAN_REDUCER_HPP_
