#include "../ex/constants.h"
#include "../ex/types.hpp"
#include "../ex/buffer.hpp"
#include "../ex/py_funcs.hpp"
#include <chrono>
#include <glog/logging.h>
#include <memory>
#include <raft>
#include <raftio>

template<class Payload, class BufferMngr>
class dummy_pkt_gen : public raft::kernel
{
    private:
    unsigned int m_n_pkts{3};
    const int m_ngulps{20};
    const int m_ngulps_per_seq{1000};
    std::unique_ptr<BufferMngr> m_buf_mngr{nullptr};
  public:
    dummy_pkt_gen(unsigned int p_n_pkts=1)
      : raft::kernel(),
      m_n_pkts(p_n_pkts)
    {
        VLOG(3)<<"Dummy pkt constructor";
        m_buf_mngr = std::make_unique<BufferMngr>(m_ngulps, m_ngulps_per_seq*SINGLE_SEQ_SIZE);
        output.addPort<Payload>("gulp");
    }

    virtual raft::kstatus run() override
    {
        for (int i = 0; i < m_n_pkts; ++i) {
            VLOG(3)<<"Generating a gulp";
            auto pld = m_buf_mngr.get()->acquire_buf();
            get_40ms_gulp(pld.get_mbuf()->get_data_ptr());

            auto& mref=pld.get_mbuf()->get_metadataref();
            mref["seq_start"] = uint64_t(329008696996015680);
            mref["seq_end"] = uint64_t(m_ngulps_per_seq+329008696996015680);
            int nseqs = m_ngulps_per_seq;
            mref["nseqs"] = nseqs;
            mref["gulp_len_ms"] = (m_ngulps_per_seq) * SAMPLING_LEN_uS * 1e3;
            mref["nchan"] = uint8_t(128);
            mref["chan0"] = int64_t(1128); // to meet alignment requirements
            mref["data_order"] =  "t_maj"s;
            mref["nbytes"] = 128 * LWA_SV_NPOLS * nseqs * LWA_SV_NSTANDS * 1/*bytes*/;


            output["gulp"].push(pld);
        }
        VLOG(2)<<"Stopping gulp gen";
        return raft::stop;
    }
};