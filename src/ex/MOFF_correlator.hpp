/*
 Copyright (c) 2023 The EPIC++ authors

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.
 */

#ifndef SRC_EX_MOFF_CORRELATOR_HPP_
#define SRC_EX_MOFF_CORRELATOR_HPP_

#include <hwy/highway.h>

#include <cmath>
#include <cstdlib>
#include <string>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "./MOFF_cu_handler.h"
#include "./buffer.hpp"
#include "./constants.h"
#include "./lf_buf_mngr.hpp"
#include "./metrics.hpp"
// #include "packet_assembler.hpp"

#include "./py_funcs.hpp"
#include "./types.hpp"
#include "./gain_table.hpp"
#include "./proj_table.hpp"

// namespace hn = hwy::HWY_NAMESPACE;

/**
 * @brief Interface for the optimized MOFF correlator
 *
 * @tparam Dtype Data type
 * @tparam BuffMngr Buffer manager type
 */
template <typename Dtype, typename BuffMngr>
class MOFFCorrelator : public MOFFCuHandler {
 public:
  using mbuf_t = typename BuffMngr::mbuf_t;
  using payload_t = Payload<mbuf_t>;
  /// Data type to store the auxiliary data for imaging. The precision will be
  /// fixed to float.
  using aux_t = hwy::AlignedFreeUniquePtr<float[]>;

  // private:
  // void m_reset();

 protected:
  /// Buffer manager to store the imaging outputs
  std::unique_ptr<BuffMngr> m_mbuf_mngr{nullptr};

  // std::unique_ptr<MOFFCuHandler> m_cu_handler{nullptr};

  /**Raw antenna grid positions
  The positions must be scaled with 1/wavelength
  and summed with half the grid size to center them*/
  aux_t m_raw_ant_pos{nullptr};

  /// Frequency dependent antenna positions
  aux_t m_ant_pos_freq{nullptr};

  /// 2D array to store the GCF kernel
  aux_t m_gcf_kernel2D{nullptr};

  /// Array to store frequency dependent phases
  aux_t m_phases{nullptr};

  /// Phases from GetLwasvPhases (geometry x cable model) before calibration
  aux_t m_phases_nominal{nullptr};

  /// Version of the calibration gain table folded into m_phases (0: none)
  uint64_t m_gain_version{0};
  /// Version of the interference projector in use (0: none)
  uint64_t m_proj_version{0};

  /// Calibration options (gain table and tap), from MOFFCorrelatorDesc
  MOFFCorrelatorDesc m_cal_desc;

  /// Array to store the averaged gridding kernel
  aux_t m_correction_kernel_h{nullptr};

  /// Array to store the correction grid
  aux_t m_correction_grid_h{nullptr};

  int m_grid_size;
  double m_grid_res;
  // int m_support_size;
  int m_gcf_tex_dim;
  bool m_rm_autocorrs{false};
  // int m_nchan_in{ 132 };
  int m_nchan_out{128};
  int m_chan0{0};
  size_t m_nseq_per_gulp;
  IMAGING_POL_MODE m_pol_mode{DUAL_POL};
  float m_delta;
  float m_accum_time;
  int m_ngulps_per_img;
  int m_kernel_oversampling_factor{2};
  /// Use complex w-projected gridding kernels (--wproj)
  bool m_use_wproj{false};
  int m_support_oversample{5};

  bool is_raw_ant_pos_set{false};

  /// @brief Reset the frequency-based antenna positions array
  /// Automatically called upon change in any of the imaging parameters
  /// @param p_grid_size Size of the grid
  /// @param p_grid_res Resolution of the grid in degrees
  /// @param p_nchan Total number of channels
  /// @param p_chan0 Channel number of the first channel
  void ResetAntpos(int p_grid_size, double p_grid_res, int p_nchan,
                   int p_chan0);
  /// @brief Reset the phases array
  /// Automatically called upon any change in imaging parameters
  /// @param p_nchan Number of channels
  /// @param p_chan0 First channel number
  void ResetPhases(int p_nchan, int p_chan0);
  /// @brief Reset the GCF kernel
  /// @param p_gcf_tex_dim Resolution of the kernel.
  /// @note This is different from the support size
  void ResetGcfKernel2D(int p_gcf_tex_dim);
  /**
   * @brief Set the up GPU for imaging
   *
   * Allocate memory block for the output image and setup texture(s) for GCF
   */
  void SetupGpu();

  void ResetCorrectionGrid(int p_nchan);

 public:
  /**
   * @brief Construct a new MOFFCorrelator object
   *
   * @param p_desc MOFFCorrelator description
   */
  explicit MOFFCorrelator(MOFFCorrelatorDesc p_desc);
  /**
   * @brief Reset the auxiliary data required for imaging
   *
   * @param p_nchan Total channels
   * @param p_chan0 First channel's number
   */
  bool ResetImagingConfig(int p_nchan, int p_chan0);
  // , int p_npol, int p_grid_size, double p_grid_res, int p_gcf_kernel_dim);

  size_t GetNumSeqPerGulp() { return m_nseq_per_gulp; }
  size_t GetNumGulpsPerImg() { return m_ngulps_per_img; }
  int GetNumStreams() const { return this->m_nstreams; }
  size_t GetGridSize() { return m_grid_size; }
  double GetGridRes() { return m_grid_res; }
  int GetNumPols() { return m_pol_mode & m_pol_mode; }
  int GetSupportSize() { return m_support_size; }
  int GetGcfKernelSize() { return m_gcf_tex_dim; }
  float GetScalingLen() { return m_delta; }
  /// Calibration options (gain table path, tap settings)
  const MOFFCorrelatorDesc& GetCalDesc() const { return m_cal_desc; }
  /// Version of the gain table applied to the current phases (0: nominal)
  uint64_t GetGainVersion() const { return m_gain_version; }
  /// Fold a calibration table into the phases and upload them. Returns false
  /// if the table does not match the current spectral window.
  bool ApplyGainTable(const GainSnapshot& p_snap) {
    if (!m_phases_nominal || p_snap.chan0 != m_chan0 || int(p_snap.nchan) != m_nchan_in) {
      return false;
    }
    ApplyGainCorrections(m_phases_nominal.get(), p_snap, m_phases.get());
    this->UpdatePhases(m_nchan_in, m_phases.get());
    m_gain_version = p_snap.valid ? p_snap.version : 0;
    return true;
  }
  /// Version of the projector applied to the voltages (0: none)
  uint64_t GetProjVersion() const { return m_proj_version; }
  /// Upload an interference projector (between images). Returns false if it
  /// does not match the current spectral window.
  bool ApplyProjTable(const ProjSnapshot& p_snap) {
    if (!this->m_rfi || p_snap.chan0 != m_chan0 || int(p_snap.nchan) != m_nchan_in) {
      return false;
    }
    if (p_snap.valid) {
      this->m_rfi->SetProjector(p_snap.ut.data(), p_snap.vt.data(), p_snap.q, m_nchan_in,
                                p_snap.sun.empty() ? nullptr : p_snap.sun.data());
    } else {
      this->m_rfi->ClearProjector();
    }
    m_proj_version = p_snap.valid ? p_snap.version : 0;
    return true;
  }
  payload_t GetEmptyBuf();
};

template <typename Dtype, typename BuffMngr>
typename MOFFCorrelator<Dtype, BuffMngr>::payload_t
MOFFCorrelator<Dtype, BuffMngr>::GetEmptyBuf() {
  return m_mbuf_mngr.get()->acquire_buf();
}

template <typename Dtype, typename BuffMngr>
MOFFCorrelator<Dtype, BuffMngr>::MOFFCorrelator(MOFFCorrelatorDesc p_desc) {
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

  m_kernel_oversampling_factor = p_desc.kernel_oversampling_factor;
  m_use_wproj = p_desc.use_wproj;

  // Allocate the arrays for the over-sampled correction kernel and grid
  m_support_oversample = m_kernel_oversampling_factor > 1
                             ? m_support_size * m_kernel_oversampling_factor
                             : m_support_size;
  // (int(m_support_size/2)+m_kernel_oversampling_factor/2)*2+1;
  m_correction_grid_h = std::move(
      hwy::AllocateAligned<float>(m_grid_size * m_grid_size * m_nchan_out));

  m_correction_kernel_h = std::move(hwy::AllocateAligned<float>(
      m_nchan_out * m_support_oversample * m_support_oversample));

  m_out_img_desc.img_size = p_desc.img_size;
  m_out_img_desc.nchan_out = m_nchan_out;
  m_out_img_desc.pol_mode = m_pol_mode;

  m_nseq_per_gulp = p_desc.nseq_per_gulp;
  LOG_IF(FATAL, m_nseq_per_gulp <= 0)
      << "Number of sequences per gulp must be >0.";

  m_cal_desc = p_desc;
  m_rm_autocorrs = p_desc.is_remove_autocorr;
  this->SetRemoveAutocorrs(m_rm_autocorrs);
  LOG_IF(INFO, m_rm_autocorrs) << "Removing antenna auto-correlations";
  // 16-bit accumulation: launches of 100 sequences keep the image within ~0.6%
  // of 32-bit accumulation at ~95% of the 40 ms budget (tests/gpu results).
  this->SetSeqChunk(p_desc.seq_chunk >= 0 ? p_desc.seq_chunk
                                          : (p_desc.use_bf16_accum ? 100 : 0));

  m_nstreams = p_desc.nstreams;
  LOG_IF(FATAL, (m_nstreams < 1 || m_nstreams > MAX_GULP_STREAMS))
      << "Number of streams must  be between 1 and " << MAX_GULP_STREAMS;
  LOG_IF(FATAL, m_nchan_out % m_nstreams != 0)
      << "The number of output channels must be divisible by the number of "
         "streams to process a gulp.";

  // TODO(karthik): Check if the GPU can image specified channels

  float gulp_len_ms =
      static_cast<float>(m_nseq_per_gulp * SAMPLING_LEN_uS * 1e-3);
  float ngulps = m_accum_time / gulp_len_ms;
  m_ngulps_per_img = std::ceil(ngulps);

  LOG_IF(WARNING, std::abs(std::ceil(ngulps) - ngulps) > 1e-5)
      << "The accumulation time (" << m_accum_time
      << " ms) is not an integer multiple of the gulp size (" << gulp_len_ms
      << " ms). Adjusting it to " << m_ngulps_per_img * gulp_len_ms << " ms";

  LOG_IF(FATAL, p_desc.device_id < 0)
      << "Invalid GPU device ID: " << p_desc.device_id;
  this->m_device_id = p_desc.device_id;
  this->m_gulp_exec_gauge_id = PrometheusExporter::AddRuntimeSummaryLabel(
      {{"type", "exec_time"},
       {"units", "ms"},
       {"kernel", "BlockEpicImager"},
       {"device", "GPU-" + std::to_string(this->m_device_id)}});

  VLOG(3) << "Setting up the buffer manager";
  LOG_IF(FATAL, p_desc.nbuffers <= 0) << "Total numbers of buffers must be >0";
  LOG_IF(FATAL, p_desc.buf_size <= 0)
      << "Buffer size must be at least one byte.";
  m_mbuf_mngr = std::make_unique<BuffMngr>(p_desc.nbuffers, p_desc.buf_size,
                                           p_desc.max_tries_acq_buf,
                                           p_desc.page_lock_bufs);
  VLOG(3) << "Done setting up the correlator";

  this->use_bf16_accum = p_desc.use_bf16_accum;
  VLOG(3) << "Setting up GPU for imaging.";
  SetupGpu();
  if (!p_desc.proj_table_path.empty() || p_desc.impulse_sigma > 0.f) {
    this->EnableRfiStage(p_desc.impulse_sigma);
  }
  if (p_desc.caltap_gpu) {
    TccTapOptions tap;
    tap.path = p_desc.caltap_path;
    tap.interval_s = p_desc.caltap_interval_s;
    tap.min_quality = p_desc.caltap_min_quality;
    tap.record_dir = p_desc.caltap_record_dir;
    this->EnableTccTap(tap, m_nchan_out, int(m_nseq_per_gulp));
  }
  VLOG(3) << "Done";
}

template <typename Dtype, typename BuffMngr>
bool MOFFCorrelator<Dtype, BuffMngr>::ResetImagingConfig(int p_nchan,
                                                         int p_chan0) {
  if (p_nchan == m_nchan_in && p_chan0 == m_chan0) {
    return false;
  }

  if (p_nchan != m_nchan_in) {
    m_nchan_in = p_nchan;
    if (m_nchan_out > m_nchan_in) {
      LOG(FATAL) << "Number of output channels exceeds the input channels";
    }
    this->m_f_eng_bytes = m_nchan_in * LWA_SV_INP_PER_CHAN * m_nseq_per_gulp;
    // Each stream copies only the channels it images (the data are channel
    // major). Splitting the input channels instead hands a stream channels
    // copied on other streams, which races when nchan_in != nchan_out.
    this->m_nbytes_f_eng_per_stream =
        (m_nchan_out / m_nstreams) * LWA_SV_INP_PER_CHAN * m_nseq_per_gulp;
    VLOG(3) << "Allocating F-eng data on the GPU. Size: " << this->m_f_eng_bytes
            << " bytes";
    this->AllocateFEngGpu(m_f_eng_bytes);
  }

  m_chan0 = p_chan0;
  if (this->m_rfi) {
    this->m_rfi->ClearProjector();  // a projector for the old window does not apply
  }
  m_proj_version = 0;

  // Reset the gridding kernel because effective area changes with freq
  auto aeff = GetAEff(p_chan0);
  m_gcf_tex_dim = sqrt(aeff) * 10;  // in decimeters
  VLOG(3) << "Initializing GCF texture. New Aeff: " << aeff << " sq. m "
            << m_gcf_tex_dim;
  ResetGcfKernel2D(m_gcf_tex_dim);
  this->ResetGcfTex(m_gcf_tex_dim, m_gcf_kernel2D.get());

  VLOG(3) << "Resetting antpos. Grid size: " << m_grid_size
          << " grid res: " << m_grid_res << " nchan: " << p_nchan
          << " chan0: " << p_chan0;
  ResetAntpos(m_grid_size, m_grid_res, p_nchan, p_chan0);
  VLOG(3) << "Resetting phases";
  ResetPhases(p_nchan, p_chan0);
  // chan_flag = true;

  VLOG(3) << "Sending imaging context information to gpu";
  this->ResetData(p_nchan, m_nseq_per_gulp, m_ant_pos_freq.get(),
                  m_phases.get());

  // compute gcf elements on a finer grid
  // int orig_support = m_support_size;
  // int finer_support =
  // (int(orig_support/2)+m_kernel_oversampling_factor/2)*2+1; m_support_size =
  // finer_support;
  const char* corr_env = std::getenv("EPIC_GCF_CORRECTION");
  const bool legacy_correction =
      corr_env != nullptr && std::string(corr_env) == "legacy";
  if (legacy_correction) {
    // Previous method: average oversampled kernel. It omits the one-cell
    // integration of GcfPixelIntegral and leaves the image too faint towards
    // the horizon (~33% at 30 deg elevation for support 5).
    this->ResetGcfElem(m_nchan_out, m_support_oversample, m_chan0,
                       m_delta / static_cast<float>(m_kernel_oversampling_factor),
                       m_grid_size);
    ResetCorrectionGrid(m_nchan_out);
  }
  // Kernel elements used for gridding
  this->ResetGcfElem(m_nchan_out, m_support_size, m_chan0, m_delta,
                     m_grid_size);
  size_t nelem = size_t(m_nchan_out) * LWA_SV_NSTANDS * m_support_size *
                 m_support_size;
  std::vector<float> real_elems(nelem);
  this->GetGcfElem(real_elems.data(), m_nchan_out, m_support_size);
  if (!legacy_correction) {
    // Correction from the actual per-stand kernels (epic_geometry)
    GetGriddingCorrection(real_elems.data(), m_ant_pos_freq.get(), m_nchan_out,
                          m_chan0, m_support_size, m_grid_size,
                          m_kernel_oversampling_factor, m_correction_grid_h.get());
    this->SetCorrectionGrid(m_correction_grid_h.get(), m_grid_size, m_nchan_out);
  }
  if (m_use_wproj) {
    // Fold each stand's residual w into its kernel (complex weights) using the
    // real elements just computed on the GPU. Done once per frequency setup.
    std::vector<float> cplx_elems(2 * nelem);
    GetWprojGcfElems(real_elems.data(), m_ant_pos_freq.get(), m_nchan_out,
                     m_chan0, m_support_size, m_grid_size, m_grid_res,
                     cplx_elems.data());
    this->SetGcfElemCplx(cplx_elems.data(), m_nchan_out, m_support_size);
    LOG(INFO) << "Using w-projected gridding kernels";
  } else {
    this->SetGcfElemCplx(nullptr, m_nchan_out, m_support_size);
  }

  return true;
}

template <typename Dtype, typename BuffMngr>
void MOFFCorrelator<Dtype, BuffMngr>::ResetAntpos(int p_grid_size,
                                                  double p_grid_res,
                                                  int p_nchan, int p_chan0) {
  int pitch = 3 * LWA_SV_NSTANDS;
  float half_grid = static_cast<float>(p_grid_size) / 2.0f;
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
  CHECK_NE(m_ant_pos_freq.get(), static_cast<float*>(NULL))
      << "Unable to allocate antenna position memory";

  m_delta = GetLwasvLocs<float>(m_raw_ant_pos.get(), p_grid_size, p_grid_res);
  // auto chan0 = p_chan0;
  for (auto chan = 0; chan < p_nchan; ++chan) {
    auto wavenumber = static_cast<double>((p_chan0 + chan) * BANDWIDTH) /
                      static_cast<double>(SOL);
    for (auto pos = 0; pos < pitch; ++pos) {
      m_ant_pos_freq[chan * pitch + pos] =
          half_grid + m_raw_ant_pos[pos] * wavenumber;
    }
  }

  // printf("antpos_raw[0] CPU: %f %f %f\n", m_raw_ant_pos[0], m_raw_ant_pos[1],
  // m_raw_ant_pos[2]); printf("antpos[0] CPU: %f %f %f\n", m_ant_pos_freq[0],
  // m_ant_pos_freq[1], m_ant_pos_freq[2]);
  DLOG(INFO) << "chan0: " << float((p_chan0)*BANDWIDTH) << "\n";
}

template <typename Dtype, typename BuffMngr>
void MOFFCorrelator<Dtype, BuffMngr>::ResetPhases(int p_nchan, int p_chan0) {
  int pitch = 2 * LWA_SV_NSTANDS;
  m_phases.reset();
  m_phases =
      std::move(hwy::AllocateAligned<float>(pitch * LWA_SV_NPOLS * p_nchan));

  GetLwasvPhases<float>(m_phases.get(), p_nchan, p_chan0);
  // keep the uncalibrated phases: calibration tables multiply these
  m_phases_nominal.reset();
  m_phases_nominal =
      std::move(hwy::AllocateAligned<float>(pitch * LWA_SV_NPOLS * p_nchan));
  std::copy(m_phases.get(), m_phases.get() + pitch * LWA_SV_NPOLS * p_nchan,
            m_phases_nominal.get());
  m_gain_version = 0;
  DLOG(INFO) << "Phases[0] cpu: " << m_phases[0] << " " << m_phases[1] << " "
             << m_phases[2] << " " << m_phases[3];
}

template <typename Dtype, typename BuffMngr>
void MOFFCorrelator<Dtype, BuffMngr>::ResetGcfKernel2D(int p_gcf_tex_dim) {
  // m_gcf_tex_dim = p_gcf_tex_dim;
  m_gcf_kernel2D.reset();
  m_gcf_kernel2D =
      std::move(hwy::AllocateAligned<float>(p_gcf_tex_dim * p_gcf_tex_dim));
  // GaussianToTex2D(m_gcf_kernel2D.get(), 0.15, p_gcf_tex_dim);

  ProlateSpheroidalToTex2D<float>(ProSphPars::m, ProSphPars::n,
                                  ProSphPars::alpha, m_gcf_kernel2D.get(),
                                  p_gcf_tex_dim, ProSphPars::c);
}

template <typename Dtype, typename BuffMngr>
void MOFFCorrelator<Dtype, BuffMngr>::SetupGpu() {
  VLOG(2) << "Allocating output image";
  this->m_out_img_bytes = m_nchan_out *
                          std::pow(m_grid_size, 2)
                          //* std::pow(int(m_pol_mode), 2)
                          * sizeof(float) * 4 /*XX_re, YY_re*, X*Y, XY* */;
  this->AllocateOutImg(this->m_out_img_bytes);
  // VLOG(2) << "Initializing GCF texture";
  // ResetGcfKernel2D(m_gcf_tex_dim);
  // this->ResetGcfTex(m_gcf_tex_dim, m_gcf_kernel2D.get());

  // calculate the appropriate offsets to image the gulp in streams
  this->m_nchan_per_stream = m_nchan_out / m_nstreams;
  // this->m_nbytes_f_eng_per_stream = m_f_eng_bytes / m_nstreams;
  this->m_nbytes_out_img_per_stream = m_out_img_bytes / m_nstreams;

  VLOG(2) << "Setting up streams and initializing the imaging kernel";
  CreateGulpCuStreams();
  SetImagingKernel();
  SetImgGridDim();
}

template <typename Dtype, typename BuffMngr>
void MOFFCorrelator<Dtype, BuffMngr>::ResetCorrectionGrid(int p_nchan) {
  this->GetCorrectionKernel(m_correction_kernel_h.get(), m_support_oversample,
                            p_nchan);
  GetCorrectionGrid<float>(
      m_correction_kernel_h.get(), m_correction_grid_h.get(), m_grid_size,
      m_support_oversample, p_nchan, m_kernel_oversampling_factor);
  this->SetCorrectionGrid(m_correction_grid_h.get(), m_grid_size, p_nchan);
}
using float_buf_mngr_t = LFBufMngr<AlignedBuffer<float>>;
using MOFFCorrelator_t = MOFFCorrelator<uint8_t, float_buf_mngr_t>;

#endif  // SRC_EX_MOFF_CORRELATOR_HPP_
