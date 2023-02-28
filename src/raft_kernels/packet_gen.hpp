#include "../ex/constants.h"
#include "../ex/types.hpp"
#include <chrono>
#include <glog/logging.h>
#include <memory>
#include <raft>
#include <raftio>

/**
 * @brief Raft kernel to generate gulps with support for timed and untimed modes.
 *
 * @tparam PktAssmblr Packet assembler kernel
 */
template<class PktAssmblr>
class GulpGen_rft : public raft::kernel
{
  using payload_t = typename PktAssmblr::payload_t;
  private:
    std::unique_ptr<PktAssmblr> m_assmblr;
    /// Duration of the generator in seconds
    int m_timer;
    /// Flag to keep the program running forever
    bool m_perpetual{ false };
    /// Flag to terminate the generator
    bool m_terminate{ false };
    std::chrono::time_point<std::chrono::steady_clock> m_start_time;
    double m_gulp_duration;
    bool m_start_set{ false };

  public:
    /**
     * @brief Construct a new GulpGen_rft object
     *
     * @param p_assmblr Pointer to the packet assembler
     * @param p_timer_s Total run time of the program in seconds. A negative value will run it forever.
     *
     */
    GulpGen_rft(
      std::unique_ptr<PktAssmblr>& p_assmblr,
      int p_timer_s)
      : m_assmblr(std::move(p_assmblr))
      , m_timer(p_timer_s)
      , raft::kernel()
    {
        if (m_timer < 0)
            m_perpetual = true;

        LOG_IF(FATAL, m_timer == 0) << "Run time for the gulp generator cannot be zero.";

        m_gulp_duration = m_assmblr.get()->get_nseq_per_gulp() * 40e-6;

        if (m_timer > 0 && m_timer < m_gulp_duration) {
            LOG(WARNING) << "The specified runtime of " << m_timer << " s is less than the gulp size (" << m_gulp_duration << "). Adjusting the run time to " << m_gulp_duration << " s.";
            m_timer = m_assmblr.get()->get_nseq_per_gulp();
        }
        LOG(INFO) << "Initializing the Gulp generator.";
        LOG_IF(INFO, m_perpetual) << "Running in continuous mode.";
        LOG_IF(INFO, !m_perpetual) << "Running in timed mode for " << p_timer_s << " seconds.";

        output.addPort<payload_t>("gulp");
    }

    virtual raft::kstatus run()
    {
        if (!m_start_set) {
            m_start_time = std::chrono::steady_clock::now();
            m_start_set = true;
        }
        while (
          m_perpetual ||
          std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - m_start_time).count() < m_timer) {
            VLOG(2)<<"DURATION: "<<std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - m_start_time).count() ;
            auto gulp = m_assmblr.get()->get_gulp();
            if(!gulp){
              VLOG(2)<<"Null gulp";
              continue;
            }
            output["gulp"].push(gulp);

            if (m_terminate)
                break;
        }
        LOG(INFO)<<"Stopping gulp gen";
        return raft::stop;
    }
};