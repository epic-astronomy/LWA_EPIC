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
    int m_grid_size{ 0 };
    float m_grid_res{ 0 };
    int m_npols{ 0 };
    int m_support{ 0 };
    uint64_t m_seq_start_id{ 0 };

  public:
    Correlator_rft(std::unique_ptr<_Correlator>& p_correlator)
      : m_correlator(std::move(p_correlator))
      , raft::kernel()
    {
        m_ngulps_per_img = m_correlator.get()->get_ngulps_per_img();
        m_grid_res = m_correlator.get()->get_grid_res();
        m_grid_size = m_correlator.get()->get_grid_size();
        m_npols = m_correlator.get()->get_npols();
        m_support = m_correlator.get()->get_support();
        input.addPort<_Payload>("gulp");
        output.addPort<_Payload>("img");
    }

    virtual raft::status run() override
    {
        _Payload pld;
        input["gulp"].pop(pld);

        if (!pld) {
            return raft::proceed;
        }

        auto& gulp_metadata = pld.get_mbuf()->get_metadataref();
        auto nchan = std::any_cast<uint8_t>(metadata["nchan"]);
        auto chan0 = std::any_cast<int64_t>(metadata["chan0"]);

        // initialization or change in the spectral window
        if (m_correlator.get()->reset(nchan, chan0)) {
            m_gulp_counter = 1;
        }

        m_is_first = m_gulp_counter == 1 ? true : false;
        m_is_last = m_gulp_counter == m_ngulps_per_img ? true : false;

        if (m_is_first) {
            m_seq_start_id = std::any_cast<uint64_t>(gulp_metadata["seq_start"]);
        }

        float out_ptr = nullptr;
        if (m_is_last) {
            // prepare the metadata for the image
            auto buf = m_correlator.get()->get_empty_buffer();
            auto& img_metadata = buf.get_mbuf()->get_metadataref();
            img_metadata = gulp_metadata; // pld.get_mbuf()->get_metadataref();
            img_metadata["seq_start"] = m_seq_start_id;
            img_metadata["nseqs"] = std::any_cast<int>(img_metadata["nseqs"]) * m_ngulps_per_img;
            img_metadata["gulp_len_ms"] = std::any_cast<int>(img_metadata["gulp_len_ms"]) * m_ngulps_per_img;
            img_metadata["grid_size"] = m_grid_size;
            img_metadata["grid_res"] = m_grid_res;
            img_metadata["npols"] = m_npols;
            img_metadata["support_size"] = m_support;

            m_correlator.get()->process_gulp(
              pld.get_mbuf()->get_data_ptr(),
              buf.get_mbuf()->get_data_ptr(),
              m_is_first,
              m_is_last);

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