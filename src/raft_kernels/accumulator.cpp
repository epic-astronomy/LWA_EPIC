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

template<class _Pld>
class Accumulator_rft : public raft::kernel
{
  protected:
    bool m_is_first_gulp{ true };
    // gulp size in ms
    size_t m_gulp_size{ 40 };
    // Number of gulps to accumulate
    size_t m_naccum{ 4 };
    static constexpr size_t m_nbuffers{ 20 };
    size_t m_xdim{ 128 };
    size_t m_ydim{ 128 };
    size_t m_in_nchan{ 32 };

    PSTensor<float> m_in_tensor;
    PSTensor<float> m_out_tensor;

    _Pld m_cur_buf;
    size_t m_accum_count{ 0 };

  public:
    Accumulator_rft(size_t p_xdim, size_t p_ydim, size_t p_nchan, size_t p_naccum)
      : m_xdim(p_xdim)
      , m_ydim(p_ydim)
      , m_naccum(p_naccum)
      , m_in_nchan(p_nchan)
      , m_in_tensor(p_nchan, p_xdim, p_xdim)
      , m_out_tensor(p_nchan, p_xdim, p_ydim)
      , raft::kernel()
    {
        input.addPort<_Pld>("in_img");
        output.addPort<_Pld>("out_img");
    }

    void increment_count(){
        m_accum_count++;
    }

    virtual raft::kstatus run() override
    {
        if (m_accum_count == 0) {
            // store the current gulp
            input["in_img"].pop(m_cur_buf);
            m_in_tensor.assign_data(m_cur_buf.get_mbuf()->get_data_ptr());
        } else {
            // add the next gulp to the current state
            _Pld pld2;
            input["in_img"].pop(pld2);
            m_out_tensor.assign_data(pld2.get_mbuf()->get_data_ptr());

            m_in_tensor += m_out_tensor;
        }

        m_accum_count++;

        if (m_accum_count == m_naccum) {
            output["out_img"].push(m_cur_buf);
            m_cur_buf = _Pld();
            m_accum_count = 0;

            m_in_tensor.dissociate_data();
            m_out_tensor.dissociate_data();
        }

        return raft::proceed;
    }
};