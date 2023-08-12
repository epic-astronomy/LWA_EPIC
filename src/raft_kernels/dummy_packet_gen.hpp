#include "../ex/buffer.hpp"
#include "../ex/constants.h"
#include "../ex/py_funcs.hpp"
#include "../ex/types.hpp"
#include <chrono>
#include <glog/logging.h>
#include <memory>
#include <raft>
#include <raftio>

template<class Payload, class BufferMngr>
class dummy_pkt_gen : public raft::kernel
{
  private:
    size_t m_n_pkts{ 3 };
    const int m_ngulps{ 20 };
    const int m_ngulps_per_seq{ 1000 };
    const int m_nchan_in{ 128 };
    const int m_chan0{ 1128 + 600 }; // 1128+600
    uint64_t m_time_from_unix_epoch_s{ 0 };
    uint64_t m_time_tag0{ 0 };
    std::unique_ptr<BufferMngr> m_buf_mngr{ nullptr };

  public:
    dummy_pkt_gen(size_t p_n_pkts = 1, std::string utcstart = "2023_06_19T00_00_00"s)
      : raft::kernel()
      , m_n_pkts(p_n_pkts)
    {
        VLOG(3) << "Dummy pkt constructor";
        m_buf_mngr = std::make_unique<BufferMngr>(m_ngulps, m_ngulps_per_seq * SINGLE_SEQ_SIZE);
        output.addPort<Payload>("gulp");

        if (utcstart == "") {
            m_time_from_unix_epoch_s = get_ADP_time_from_unix_epoch();
        } else {
            m_time_from_unix_epoch_s = get_time_from_unix_epoch(utcstart);
        }
        m_time_tag0 = m_time_from_unix_epoch_s * FS;
    }

    virtual raft::kstatus run() override
    {
        for (size_t i = 0; i < m_n_pkts; ++i) {
            VLOG(3) << "Generating a gulp";
            auto pld = m_buf_mngr.get()->acquire_buf();

            auto start = std::chrono::high_resolution_clock::now();
            // get_40ms_gulp(pld.get_mbuf()->get_data_ptr());
            LOG(INFO) << "Gulp gen duration: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
            std::this_thread::sleep_for(std::chrono::milliseconds(40));

            LOG(INFO) << "Sending gulp at: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count();

            auto& mref = pld.get_mbuf()->get_metadataref();
            uint64_t seq_start = 329008696996015680;
            mref["seq_start"] = seq_start;
            mref["time_tag"] = m_time_tag0 + std::get<uint64_t>(mref["seq_start"]) * SEQ_MULT_S;
            mref["seq_end"] = uint64_t(m_ngulps_per_seq + seq_start);
            int nseqs = m_ngulps_per_seq;
            mref["nseqs"] = nseqs;
            mref["gulp_len_ms"] = (m_ngulps_per_seq)*SAMPLING_LEN_uS * 1e3;
            mref["nchan"] = uint8_t(m_nchan_in);
            mref["chan0"] = int64_t(m_chan0); // to meet alignment requirements
            mref["data_order"] = "t_maj"s;
            mref["nbytes"] = m_nchan_in * LWA_SV_NPOLS * nseqs * LWA_SV_NSTANDS * 1 /*bytes*/;

            output["gulp"].push(pld);
        }
        VLOG(2) << "Stopping gulp gen";
        return raft::stop;
    }
};