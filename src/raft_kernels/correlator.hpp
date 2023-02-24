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
    int m_gulp_counter{ 0 };
    bool is_first{false};
    bool is_last{false};

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
    }
};