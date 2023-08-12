#include "../ex/buffer.hpp"
#include "../ex/constants.h"
#include "../ex/py_funcs.hpp"
#include "../ex/tensor.hpp"
#include "../ex/types.hpp"
#include <chrono>
#include <cmath>
#include <glog/logging.h>
#include <memory>
#include <raft>
#include <raftio>
#include <variant>

/**
 * @brief Raft kernel bin the channels in the input image
 *
 * @tparam _PldIn Input payload type
 * @tparam BufferMngr Buffer manager type
 * @tparam _PldOut Output payload type
 */
template<typename _PldIn, class BufferMngr, typename _PldOut = _PldIn>
class ChanReducer_rft : public raft::kernel
{
  protected:
    /// @brief Number of channels to combine
    const int m_ncombine{ 4 };
    float m_norm{ 1 };
    bool m_is_chan_avg{ false };
    std::unique_ptr<BufferMngr> m_buf_mngr{ nullptr };
    static constexpr size_t m_nbufs{ 20 };
    static constexpr size_t m_max_buf_reqs{ 10 };
    size_t m_xdim{ 128 };
    size_t m_ydim{ 128 };
    size_t m_in_nchan{ 128 };
    size_t m_out_nchan{ 32 };
    PSTensor<float> m_in_tensor;
    PSTensor<float> m_out_tensor;
    static constexpr int NSTOKES{ 4 };

    using high_res_tp = typename std::chrono::time_point<std::chrono::high_resolution_clock>;
    high_res_tp m_prev_refresh;
    long int m_refresh_interval{ 10 };

    bool is_require_refresh()
    {
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::seconds>(now - m_prev_refresh).count() >= m_refresh_interval) {
            m_prev_refresh = now;
            return true;
        }

        return false;
    }

  public:
    /**
     * @brief Construct a new ChanReducer_rft object
     *
     * @param p_ncombine Channel binning factor
     * @param p_xdim X side of the image
     * @param p_ydim Y side of the image
     * @param p_in_nchan Number of input channels to the reducer
     */
    ChanReducer_rft(int p_ncombine, int p_xdim, int p_ydim, int p_in_nchan, int p_refresh_interval = 10)
      : raft::kernel()
      , m_ncombine(p_ncombine)
      , m_xdim(p_xdim)
      , m_ydim(p_ydim)
      , m_in_nchan(p_in_nchan)
      , m_in_tensor(PSTensor<float>(m_in_nchan, m_xdim, m_ydim))
      , m_out_tensor(PSTensor<float>(size_t(p_in_nchan / p_ncombine), m_xdim, m_ydim))
      , m_refresh_interval(p_refresh_interval)
    {
        input.addPort<_PldIn>("in_img");
        output.addPort<_PldOut>("out_img");
        output.addPort<uint64_t>("seq_start_id");

        if (m_in_nchan % m_ncombine != 0) {
            LOG(FATAL) << "The number of output channels: " << m_in_nchan << " cannot be binned by a factor of " << m_ncombine << ". ";
        }

        m_out_nchan = m_in_nchan / m_ncombine;
        m_buf_mngr.reset(
          new BufferMngr(
            m_nbufs, m_xdim * m_ydim * m_out_nchan * NSTOKES, m_max_buf_reqs, false));

        m_prev_refresh = std::chrono::high_resolution_clock::now();
    }

    virtual raft::kstatus run() override
    {
        _PldIn in_pld;
        input["in_img"].pop(in_pld);

        auto out_pld = m_buf_mngr->acquire_buf();
        if (!out_pld) {
            LOG(FATAL) << "Memory buffer full in ChanReducer";
        }

        auto& out_meta = out_pld.get_mbuf()->get_metadataref();

        out_meta = in_pld.get_mbuf()->get_metadataref();
        out_meta["nchan"] = (uint8_t)m_out_nchan;
        out_meta["chan_width"] = BANDWIDTH * m_ncombine;

        m_in_tensor.assign_data(in_pld.get_mbuf()->get_data_ptr());
        m_out_tensor.assign_data(out_pld.get_mbuf()->get_data_ptr());

        m_in_tensor.combine_channels(m_out_tensor);

        output["out_img"].push(out_pld);
        if (is_require_refresh()) { // update once very `m_refresh_interval` seconds
            auto tstart = std::get<uint64_t>(out_meta["seq_start"]);
            output["seq_start_id"].push(tstart);
        } else {
            output["seq_start_id"].push(0);
        }

        return raft::proceed;
    }
};