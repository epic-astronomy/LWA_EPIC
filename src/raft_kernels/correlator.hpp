#include "../ex/constants.h"
#include "../ex/types.hpp"
#include <chrono>
#include <glog/logging.h>
#include <memory>
#include <raft>
#include <raftio>

template<class _Payload, class _Correlator>
class Correlator_rft : public raft::kernel
{
  private:
    std::unique_ptr<_Correlator> m_correlator{ nullptr };
    int m_nchan{ 0 };
    uint64_t m_chan0{ 0 };
    int m_ngulps_per_img{ 1 };
    int m_gulp_counter{ 1 };
    bool m_is_first{ false };
    bool m_is_last{ false };
    uint64_t m_seq_start_id{ 0 };

  public:
    Correlator_rft(std::unique_ptr<_Correlator>& p_correlator)
      : m_correlator(std::move(p_correlator))
      , raft::kernel()
    {
        m_ngulps_per_img = m_correlator.get()->get_ngulps_per_img();
        input.addPort<_Payload>("gulp");
        output.addPort<_Payload>("img");
    }

    virtual raft::status run() override
    {
        _Payload pld;
        input["gulp"].pop(pld);
        m_is_first = m_gulp_counter == 1 ? true : false;
        m_is_last = m_gulp_counter == m_ngulps_per_img ? true : false;

        if (m_is_first) {
            auto& first_metadata = pld.get_mbuf()->get_metadataref();
            m_seq_start_id = std::any_cast<uint64_t>(first_metadata["seq_start"]);
        }

        float out_ptr = nullptr;
        if (m_is_last) {
            // prepare the metadata for the last gulp
            auto buf = m_correlator.get()->get_buffer();
            auto& last_metadata = buf.get_mbuf()->get_metadataref();
            last_metadata = pld.get_mbuf()->get_metadataref();
            last_metadata["seq_start"] = m_seq_start_id;
            last_metadata["nseqs"] = std::any_cast<int>(last_metadata["nseqs"]) * m_ngulps_per_img;
            last_metadata["gulp_len_ms"] = std::any_cast<int>(last_metadata["gulp_len_ms"]) * m_ngulps_per_img;

            m_correlator.get()->process_gulp(
              pld.get_mbuf()->get_data_ptr(), buf.get_mbuf()->get_data_ptr(), m_is_first, m_is_last);
            
            m_gulp_counter = 1;
            output["gulp"].push(buf);

            return raft::proceed;
        }

        // If it's anything other than the final gulp, proceed right away. The process_gulp function
        // images the data in streams with asynchronous reads/writes. Hence the next gulp won't have to
        // wait for the current one to complete thereby keeping the GPU completely occupied.
        m_correlator.process_gulp(pld.get_mbuf()->get_data_ptr(), nullptr, m_is_first, m_is_last);
        ++m_gulp_counter;
        return raft::proceed;
    }
};