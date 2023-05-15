#ifndef MOFF_CORRELATOR_HPP
#define MOFF_CORRELATOR_HPP

#include "MOFF_cu_handler.h"
#include "buffer.hpp"
#include "constants.h"
#include "hwy/highway.h"
#include "lf_buf_mngr.hpp"
// #include "packet_assembler.hpp"
#include "py_funcs.hpp"
#include "types.hpp"
#include <cmath>
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
  public:
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
    // int m_nchan_in{ 132 };
    int m_nchan_out{ 112 };
    int m_chan0{ 0 };
    size_t m_nseq_per_gulp;
    IMAGING_POL_MODE m_pol_mode{ DUAL_POL };
    float m_delta;
    float m_accum_time;
    int m_ngulps_per_img;

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
    /**
     * @brief Set the up GPU for imaging
     *
     * Allocate memory block for the output image and setup texture(s) for GCF
     */
    void setup_GPU();

  public:
    /**
     * @brief Construct a new MOFFCorrelator object
     *
     * @param p_desc MOFFCorrelator description
     */
    MOFFCorrelator(MOFFCorrelatorDesc p_desc);
    /**
     * @brief Reset the auxiliary data required for imaging
     *
     * @param p_nchan Total channels
     * @param p_chan0 First channel's number
     */
    bool reset(int p_nchan, int p_chan0);
    // , int p_npol, int p_grid_size, double p_grid_res, int p_gcf_kernel_dim);

    size_t get_nseq_per_gulp() { return m_nseq_per_gulp; }
    size_t get_ngulps_per_img() { return m_ngulps_per_img; }
    size_t get_grid_size() { return m_grid_size; }
    double get_grid_res() { return m_grid_res; }
    int get_npols() { return m_pol_mode & m_pol_mode; }
    int get_support() { return m_support_size; }
    int get_gcf_kernel_size(){return m_gcf_tex_dim;}
    float get_scaling_length(){return m_delta;}
    payload_t get_empty_buffer();
};

template<typename Dtype, typename BuffMngr>
typename MOFFCorrelator<Dtype, BuffMngr>::payload_t
MOFFCorrelator<Dtype, BuffMngr>::get_empty_buffer()
{
    return m_mbuf_mngr.get()->acquire_buf();
}

template<typename Dtype, typename BuffMngr>
MOFFCorrelator<Dtype, BuffMngr>::MOFFCorrelator(MOFFCorrelatorDesc p_desc)
{
    m_accum_time = p_desc.accum_time_ms;
    LOG_IF(FATAL, m_accum_time <= 0) << "Total accumulation time must be >0.";

    m_pol_mode = p_desc.pol_mode;

    m_gcf_tex_dim = p_desc.gcf_kernel_dim;
    LOG_IF(FATAL, m_gcf_tex_dim <= 0) << "GCF texture size must be >0.";

    m_grid_size = p_desc.img_size;
    LOG_IF(FATAL, m_grid_size <= 0) << "Grid size must be >0.";

    m_grid_res = p_desc.grid_res_deg;
    LOG_IF(FATAL, m_grid_res <= 0) << "Grid resolution must be >0.";

    m_support_size = p_desc.support_size;
    LOG_IF(FATAL, m_support_size <= 0) << "Gridding support must be >0.";

    m_nchan_out = p_desc.nchan_out;
    LOG_IF(FATAL, m_nchan_out <= 0) << "Number of output channels must be >0.";

    m_out_img_desc.img_size = p_desc.img_size;
    m_out_img_desc.nchan_out = m_nchan_out;
    m_out_img_desc.pol_mode = m_pol_mode;

    if ((m_support_size & (m_support_size - 1)) != 0) {
        LOG(WARNING) << "Support is not a power of 2.";
        m_support_size = std::pow(2, int(std::log2(m_support_size)));
        LOG(WARNING) << "Adjusting it to " << m_support_size;
    }

    m_nseq_per_gulp = p_desc.nseq_per_gulp;
    LOG_IF(FATAL, m_nseq_per_gulp <= 0) << "Number of sequences per gulp must be >0.";

    m_rm_autocorrs = p_desc.is_remove_autocorr;
    LOG_IF(ERROR, m_rm_autocorrs) << "Autocorrelation removal is not supported yet.";

    m_nstreams = p_desc.nstreams;
    LOG_IF(FATAL, (m_nstreams < 1 || m_nstreams > MAX_GULP_STREAMS)) << "Number of streams must  be between 1 and " << MAX_GULP_STREAMS;
    LOG_IF(FATAL, m_nchan_out % m_nstreams != 0) << "The number of output channels must be divisible by the number of streams to process a gulp.";

    // TODO: Check if the GPU can image the specified number of output channels

    float gulp_len_ms = float(m_nseq_per_gulp * SAMPLING_LEN_uS * 1e-3);
    float ngulps = m_accum_time / gulp_len_ms;
    m_ngulps_per_img = std::ceil(ngulps);

    LOG_IF(WARNING, std::abs(std::ceil(ngulps) - ngulps) > 1e-5) << "The accumulation time (" << m_accum_time << " ms) is not an integer multiple of the gulp size (" << gulp_len_ms << " ms). Adjusting it to " << m_ngulps_per_img * gulp_len_ms << " ms";

    VLOG(3)<<"Setting up GPU for imaging.";
    setup_GPU();
    VLOG(3)<<"Done";
    LOG_IF(FATAL, p_desc.device_id < 0) << "Invalid GPU device ID: " << p_desc.device_id;
    m_device_id = p_desc.device_id;

    VLOG(3)<<"Setting up the buffer manager";
    LOG_IF(FATAL, p_desc.nbuffers <= 0) << "Total numbers of buffers must be >0";
    LOG_IF(FATAL, p_desc.buf_size <= 0) << "Buffer size must be at least one byte.";
    m_mbuf_mngr = std::make_unique<BuffMngr>(p_desc.nbuffers, p_desc.buf_size, p_desc.max_tries_acq_buf, p_desc.page_lock_bufs);
    VLOG(3)<<"Done setting up the correlator";
};

template<typename Dtype, typename BuffMngr>
bool
MOFFCorrelator<Dtype, BuffMngr>::reset(int p_nchan, int p_chan0)
{
    if (p_nchan == m_nchan_in && p_chan0 == m_chan0) {
        return false;
    }

    if (p_nchan != m_nchan_in) {
        m_nchan_in = p_nchan;
        this->m_f_eng_bytes = m_nchan_in * LWA_SV_INP_PER_CHAN * m_nseq_per_gulp;
        this->m_nbytes_f_eng_per_stream = m_f_eng_bytes / m_nstreams;
        LOG(INFO) << "Allocating F-eng data on the GPU. Size: " << this->m_f_eng_bytes << " bytes";
        this->allocate_f_eng_gpu(m_f_eng_bytes);
    }

    m_chan0 = p_chan0;

    VLOG(3) << "Resetting antpos. Grid size: " << m_grid_size << " grid res: " << m_grid_res << " nchan: " << p_nchan << " chan0: " << p_chan0;
    reset_antpos(m_grid_size, m_grid_res, p_nchan, p_chan0);
    VLOG(3) << "Resetting phases";
    reset_phases(p_nchan, p_chan0);
    // chan_flag = true;

    VLOG(3) << "Sending imaging context information to gpu";
    this->reset_data(
      p_nchan,
      m_nseq_per_gulp,
      m_ant_pos_freq.get(),
      m_phases.get());

    return true;
}

template<typename Dtype, typename BuffMngr>
void
MOFFCorrelator<Dtype, BuffMngr>::reset_antpos(int p_grid_size, double p_grid_res, int p_nchan, int p_chan0)
{
    int pitch = 3 * LWA_SV_NSTANDS;
    float half_grid = float(p_grid_size) / 2.0f;
    // m_grid_size = p_grid_size;
    // m_grid_res = p_grid_res;
    // m_nchan_in = p_nchan;
    // m_chan0 = p_chan0;

    if (!is_raw_ant_pos_set) {
        m_raw_ant_pos.reset();
        m_raw_ant_pos = std::move(hwy::AllocateAligned<float>(pitch));
    }

    m_ant_pos_freq.reset();
    m_ant_pos_freq = std::move(hwy::AllocateAligned<float>(pitch * p_nchan));
    CHECK_NE(m_ant_pos_freq.get(), static_cast<float*>(NULL)) << "Unable to allocate antenna position memory";

    m_delta = get_lwasv_locs<float>(m_raw_ant_pos.get(), p_grid_size, p_grid_res);
    auto chan0 = p_chan0;
    for (auto chan = 0; chan < p_nchan; ++chan) {
        double wavenumber = double((p_chan0 + chan) * BANDWIDTH) / double(SOL);
        for (auto pos = 0; pos < pitch; ++pos) {
            m_ant_pos_freq[chan * pitch + pos] = half_grid + m_raw_ant_pos[pos] * wavenumber;
        }
    }

    printf("antpos_raw[0] CPU: %f %f %f\n", m_raw_ant_pos[0], m_raw_ant_pos[1], m_raw_ant_pos[2]);
    printf("antpos[0] CPU: %f %f %f\n", m_ant_pos_freq[0], m_ant_pos_freq[1], m_ant_pos_freq[2]);
    DLOG(INFO) << "chan0: " << float((p_chan0)*BANDWIDTH) << "\n";
}

template<typename Dtype, typename BuffMngr>
void
MOFFCorrelator<Dtype, BuffMngr>::reset_phases(int p_nchan, int p_chan0)
{

    int pitch = 2 * LWA_SV_NSTANDS;
    m_phases.reset();
    m_phases = std::move(hwy::AllocateAligned<float>(pitch * LWA_SV_NPOLS * p_nchan));

    get_lwasv_phases<float>(m_phases.get(), p_nchan, p_chan0);
    DLOG(INFO) << "Phases[0] cpu: " << m_phases[0] << " " << m_phases[1]<<" "<<m_phases[2]<<" "<<m_phases[3];
}

template<typename Dtype, typename BuffMngr>
void
MOFFCorrelator<Dtype, BuffMngr>::reset_gcf_kernel2D(int p_gcf_tex_dim)
{
    // m_gcf_tex_dim = p_gcf_tex_dim;
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

template<typename Dtype, typename BuffMngr>
void
MOFFCorrelator<Dtype, BuffMngr>::setup_GPU()
{
    VLOG(2)<<"Allocating output image";
    this->allocate_out_img(m_nchan_out * std::pow(m_grid_size, 2) * std::pow(int(m_pol_mode), 2) * sizeof(float));
    VLOG(2)<<"Initializing GCF texture";
    reset_gcf_kernel2D(m_gcf_tex_dim);
    this->reset_gcf_tex(m_gcf_tex_dim, m_gcf_kernel2D.get());

    // calculate the appropriate offsets to image the gulp in streams
    this->m_nchan_per_stream = m_nchan_out / m_nstreams;
    this->m_nbytes_f_eng_per_stream = m_f_eng_bytes / m_nstreams;
    this->m_nbytes_out_img_per_stream = m_out_img_bytes / m_nstreams;

    VLOG(2)<<"Setting up streams and initializing the imaging kernel";
    create_gulp_custreams();
    set_imaging_kernel();
    set_img_grid_dim();
}
using float_buf_mngr_t = LFBufMngr<AlignedBuffer<float>>;
using MOFFCorrelator_t = MOFFCorrelator<uint8_t, float_buf_mngr_t>;

#endif // MOFF_CORRELATOR_HPP