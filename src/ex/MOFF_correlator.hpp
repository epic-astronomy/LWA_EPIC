#ifndef MOFF_CORRELATOR_HPP
#define MOFF_CORRELATOR_HPP

#include "MOFF_cu_handler.h"
#include "buffer.hpp"
#include "hwy/highway.h"
#include "lf_buf_mngr.hpp"
#include "packet_assembler.hpp"
#include "py_funcs.hpp"
#include <iostream>
#include <memory>

// namespace hn = hwy::HWY_NAMESPACE;

/**
 * @brief Interface for the optimized MOFF correlator
 *
 * @tparam Dtype Data type
 * @tparam BuffMngr Buffer manager type
 */
template<typename Dtype, typename BuffMngr>
class MOFFCorrelator : public MOFFCuHandler
{
    using mbuf_t = typename BuffMngr::mbuf_t;
    using payload_t = Payload<mbuf_t>;
    /// Data type to store the auxiliary data for imaging. The precision will be fixed to float. 
    using aux_t = hwy::AlignedFreeUniquePtr<float[]>;

    // private:
    // void m_reset();

  protected:
    /// Buffer manager to store the imaging outputs
    std::unique_ptr<BuffMngr> m_mbuf_mngr{ nullptr };

    // std::unique_ptr<MOFFCuHandler> m_cu_handler{nullptr};

    /**Raw antenna grid positions
    The positions must be scaled with 1/wavelength
    and summed with half the grid size to center them*/
    aux_t m_raw_ant_pos{ nullptr };

    /// Frequency dependent antenna positions
    aux_t m_ant_pos_freq{ nullptr };

    /// 2D array to store the GCF kernel
    aux_t m_gcf_kernel2D{ nullptr };

    /// Array to store frequency dependent phases
    aux_t m_phases{ nullptr };

    int m_grid_size;
    double m_grid_res;
    int m_support_size;
    int m_gcf_tex_dim;
    bool m_rm_autocorrs{ false };
    int m_nchan{ 0 };
    int m_chan0{ 0 };
    size_t m_nseq_per_gulp;
    int m_npol{ 2 };
    float m_delta;
    float m_accum_time;

    bool is_raw_ant_pos_set{ false };

    /// @brief Reset the frequency-based antenna positions array
    /// Automatically called upon change in any of the imaging parameters
    /// @param p_grid_size Size of the grid
    /// @param p_grid_res Resolution of the grid in degrees
    /// @param p_nchan Total number of channels
    /// @param p_chan0 Channel number of the first channel
    void reset_antpos(int p_grid_size, double p_grid_res, int p_nchan, int p_chan0);
    /// @brief Reset the phases array
    /// Automatically called upon any change in imaging parameters
    /// @param p_nchan Number of channels
    /// @param p_chan0 First channel number
    void reset_phases(int p_nchan, int p_chan0);
    /// @brief Reset the GCF kernel
    /// @param p_gcf_tex_dim Resolution of the kernel.
    /// @note This is different from the support size
    void reset_gcf_kernel2D(int p_gcf_tex_dim);

  public:
  /**
   * @brief Construct a new MOFFCorrelator object
   * 
   * @param p_accum_time Total accumulation time
   * @param p_npol Number of polarizations to process
   * @param p_grid_size Size of the grid in pixels
   * @param p_grid_res Resolution of the grid in degrees
   * @param p_support Support size
   * @param p_rm_autocorr Flag to remove autocorrelations
   */
    MOFFCorrelator(float p_accum_time, int p_npol, int p_grid_size, double p_grid_res, int p_support, bool p_rm_autocorr = false)
      : m_accum_time(p_accum_time)
      , m_npol(p_npol)
      , m_grid_size(p_grid_size)
      , m_grid_res(p_grid_res)
      , m_support_size(p_support)
      // , m_gcf_tex_dim(p_gcf_tex_dim)
      , m_rm_autocorrs(p_rm_autocorr){
          // if (((p_grid_size) & (p_grid_size - 1)) != 0) {
          //     throw(InvalidSize("The grid size must be a power of 2"));
          // }
      };
    /**
     * @brief Reset the auxiliary data required for imaging
     * 
     * @param p_nchan Total channels
     * @param p_chan0 First channel's number
     * @param p_npol Total number of polarizations
     * @param p_grid_size Size of the grid in pixels
     * @param p_grid_res Resolution of the grid in degrees
     * @param p_gcf_kernel_dim Size of the kernel
     * @param p_nseq_per_gulp Number of sequences per gulp
     */
    void reset(int p_nchan, int p_chan0, int p_npol, int p_grid_size, double p_grid_res, int p_gcf_kernel_dim, size_t p_nseq_per_gulp);
};

template<typename Dtype, typename BuffMngr>
void
MOFFCorrelator<Dtype, BuffMngr>::reset(int p_nchan, int p_chan0, int p_npol, int p_grid_size, double p_grid_res, int p_gcf_kernel_dim, size_t p_nseq_per_gulp)
{
    bool chan_flag = false;
    bool gcf_flag = false;
    if (p_nchan != m_nchan || p_chan0 != m_chan0) {
        m_nchan = p_nchan;
        m_chan0 = p_chan0;
        m_nseq_per_gulp = p_nseq_per_gulp;
        m_grid_size = p_grid_size;
        m_grid_res = p_grid_res;

        std::cout << "resetting antpos\n";
        reset_antpos(m_grid_size, m_grid_res, p_nchan, p_chan0);
        std::cout << "resetting phases\n";
        // try{
        reset_phases(p_nchan, p_chan0);
        // }catch(const std::exception &e){
        //   std::cout<<"test\n";
        //   std::cerr<<e.what()<<"\n";
        // }
        chan_flag = true;
    }

    if (m_gcf_tex_dim != p_gcf_kernel_dim) {
        std::cout << "resetting kernel\n";
        m_gcf_tex_dim = p_gcf_kernel_dim;
        reset_gcf_kernel2D(p_gcf_kernel_dim);
        gcf_flag = true;
    }
    std::cout << "sending to gpu\n";

    if (chan_flag || gcf_flag) {
        this->reset_data(
          p_nchan,
          m_nseq_per_gulp,
          m_gcf_tex_dim,
          m_ant_pos_freq.get(),
          m_phases.get(),
          m_gcf_kernel2D.get());
    }
}

template<typename Dtype, typename BuffMngr>
void
MOFFCorrelator<Dtype, BuffMngr>::reset_antpos(int p_grid_size, double p_grid_res, int p_nchan, int p_chan0)
{
    int pitch = 3 * LWA_SV_NSTANDS;
    float half_grid = float(p_grid_size) / 2.0f;
    m_grid_size = p_grid_size;
    m_grid_res = p_grid_res;
    m_nchan = p_nchan;
    m_chan0 = p_chan0;

    if (!is_raw_ant_pos_set) {
        m_raw_ant_pos.reset();
        m_raw_ant_pos = std::move(hwy::AllocateAligned<float>(pitch));
    }

    m_ant_pos_freq.reset();
    m_ant_pos_freq = std::move(hwy::AllocateAligned<float>(pitch * p_nchan));
    if (!m_ant_pos_freq) {
        std::cout << "wth\n";
    }

    m_delta = get_lwasv_locs<float>(m_raw_ant_pos.get(), p_grid_size, p_grid_res);
    auto chan0 = p_chan0;
    for (auto chan = 0; chan < p_nchan; ++chan) {
        double wavenumber = double((p_chan0 + chan) * BANDWIDTH) / double(SOL);
        for (auto pos = 0; pos < pitch; ++pos) {
            // std::cout<<chan<<" "<<pos<<"\n";
            m_ant_pos_freq[chan * pitch + pos] = half_grid + m_raw_ant_pos[pos] * wavenumber;
        }
    }

    // std::cout<<"antpos[0] CPU: "<<m_ant_pos_freq[0]<<" "<<m_ant_pos_freq[1]<<" "<<m_ant_pos_freq[2]<<"\n";
    printf("antpos_raw[0] CPU: %f %f %f\n", m_raw_ant_pos[0], m_raw_ant_pos[1], m_raw_ant_pos[2]);
    printf("antpos[0] CPU: %f %f %f\n", m_ant_pos_freq[0], m_ant_pos_freq[1], m_ant_pos_freq[2]);
    std::cout << "chan0: " << float((p_chan0)*BANDWIDTH) << "\n";

    // if (uint64_t(m_raw_ant_pos.get()) % HWY_LANES(T)) {
    //     hn::ScalableTag<uint8_t> tag8;
    //     auto half_grid = hwy::Set(tag8, float(p_grid_size) / 2.0);
    //     for (auto i = 0; i < p_nchan; ++i) {

    //     }
    // }
}

template<typename Dtype, typename BuffMngr>
void
MOFFCorrelator<Dtype, BuffMngr>::reset_phases(int p_nchan, int p_chan0)
{

    int pitch = 2 * LWA_SV_NSTANDS;
    m_phases.reset();
    m_phases = std::move(hwy::AllocateAligned<float>(pitch * LWA_SV_NPOLS * p_nchan));

    get_lwasv_phases<float>(m_phases.get(), p_nchan, p_chan0);
    std::cout << "Phases[0] cpu: " << m_phases[0] << " " << m_phases[1] << "\n";
}

template<typename Dtype, typename BuffMngr>
void
MOFFCorrelator<Dtype, BuffMngr>::reset_gcf_kernel2D(int p_gcf_tex_dim)
{
    m_gcf_tex_dim = p_gcf_tex_dim;
    m_gcf_kernel2D.reset();
    m_gcf_kernel2D = std::move(hwy::AllocateAligned<float>(p_gcf_tex_dim * p_gcf_tex_dim));

    prolate_spheroidal_to_tex2D<float>(
      ProSphPars::m,
      ProSphPars::n,
      ProSphPars::alpha,
      m_gcf_kernel2D.get(),
      p_gcf_tex_dim,
      ProSphPars::c);
}

using MOFFCorrelator_t = MOFFCorrelator<uint8_t, default_buf_mngr_t>;

#endif // MOFF_CORRELATOR_HPP