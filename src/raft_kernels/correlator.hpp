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

#ifndef SRC_RAFT_KERNELS_CORRELATOR_HPP_
#define SRC_RAFT_KERNELS_CORRELATOR_HPP_
#include <glog/logging.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <set>
#include <utility>
#include <variant>

#include "../ex/cal_tap.hpp"
#include "../ex/constants.h"
#include "../ex/gain_table.hpp"
#include "../ex/proj_table.hpp"
#include "../ex/sky_table.hpp"
#include "../ex/metrics.hpp"
#include "../ex/station_desc.hpp"
#include "../ex/types.hpp"
#include "raft"
#include "raftio"

template <class _Payload, class _Correlator>
class CorrelatorRft : public raft::kernel {
 private:
  std::unique_ptr<_Correlator> m_correlator{nullptr};
  int m_nchan{0};
  uint64_t m_chan0{0};
  int m_ngulps_per_img{1};
  int m_gulp_counter{1};
  /// FX imaging: the image made at the last image boundary, handed on at the
  /// next one, so that its copy to the host overlaps the next image
  typename _Correlator::payload_t m_pending;
  bool m_is_first{false};
  bool m_is_last{false};
  int m_grid_size{0};
  float m_grid_res{0};
  int m_npols{0};
  int m_support{0};
  uint64_t m_seq_start_id{0};
  float m_delta{1};
  unsigned int m_rt_gauge_id{0};
  std::set<int> m_hc_chans;
  int m_is_GPU_setup_once{false};
  std::set<int> m_incl_freqs;
  std::string m_session_id; //changes with frequency
  /// Calibration gains from streamcal (null: off)
  std::unique_ptr<GainTableReader> m_gain_reader{nullptr};
  /// FX: the all-sky monitor's pixel table
  std::unique_ptr<SkyTableReader> m_sky_table{nullptr};
  uint64_t m_sky_table_version{0};
  /// Interference projector from streamcal (null: off)
  std::unique_ptr<ProjTableReader> m_proj_reader{nullptr};
  /// CPU X-engine on a slice of every gulp for streamcal (null: off; the GPU
  /// tap, --caltap_gpu, lives in the correlator)
  std::unique_ptr<CalTap> m_cal_tap{nullptr};

  bool cal_impulse_on() const { return m_correlator.get()->GetCalDesc().impulse_sigma > 0.f; }

  template <typename T, typename Map>
  static T MetaOr(Map& p_meta, const std::string& p_key, T p_default) {
    auto it = p_meta.find(p_key);
    if (it == p_meta.end()) {
      return p_default;
    }
    if (auto* v = std::get_if<T>(&it->second)) {
      return *v;
    }
    return p_default;
  }

 public:
  explicit CorrelatorRft(std::unique_ptr<_Correlator>* p_correlator)
      : raft::kernel(),
        m_correlator(std::move(*p_correlator)),
        m_hc_chans(GetHealthCheckChans<LWA_SV>()) {
    m_ngulps_per_img = m_correlator.get()->GetNumGulpsPerImg();
    m_grid_res = m_correlator.get()->GetGridRes();
    m_grid_size = m_correlator.get()->GetGridSize();
    m_npols = m_correlator.get()->GetNumPols();
    m_support = m_correlator.get()->GetSupportSize();
    m_delta = m_correlator.get()->GetScalingLen();
    input.addPort<_Payload>("gulp");
    // using out_t = typ
    output.addPort<typename _Correlator::payload_t>("img");
    output.addPort<typename _Correlator::payload_t>("img_stream");

    const auto& cal = m_correlator.get()->GetCalDesc();
    if (!cal.gain_table_path.empty()) {
      m_gain_reader = std::make_unique<GainTableReader>(cal.gain_table_path);
      LOG(INFO) << "Calibration gains will be read from " << cal.gain_table_path;
    }
    if (!cal.proj_table_path.empty()) {
      m_proj_reader = std::make_unique<ProjTableReader>(cal.proj_table_path);
      LOG(INFO) << "Interference projector will be read from " << cal.proj_table_path;
    }
    if (cal.use_fx && !cal.sky_table_path.empty()) {
      m_sky_table = std::make_unique<SkyTableReader>(cal.sky_table_path);
      LOG(INFO) << "Images regridded on the GPU to the all-sky monitor's pixels once it writes "
                << cal.sky_table_path;
    }
    if (cal.caltap_nseq > 0 && !cal.caltap_path.empty()) {
      CalTapOptions o;
      o.path = cal.caltap_path;
      o.nseq = cal.caltap_nseq;
      o.interval_s = cal.caltap_interval_s;
      o.nthreads = cal.caltap_threads;
      o.batch = cal.caltap_batch;
      m_cal_tap = std::make_unique<CalTap>(o);
      LOG(INFO) << "Calibration tap: " << o.nseq << " sequences per gulp, " << o.interval_s << " s tables to "
                << o.path;
    }
    if (cal.caltap_gpu) {
      LOG(INFO) << "Calibration tap on the GPU (tensor-core correlator), " << cal.caltap_interval_s
                << " s tables to " << cal.caltap_path;
    }

    m_rt_gauge_id = PrometheusExporter::AddRuntimeSummaryLabel(
        {{"type", "info"},
         {"data", "channel"},
         {"units", "channel_number"},
         {"kernel_id", std::to_string(this->get_id())}});
  }

  void SetInclChans(const std::vector<int>& p_incl_freqs){
    for(const auto& it: p_incl_freqs){
      m_incl_freqs.insert(it);
    }
  }

  raft::kstatus run() override {
    VLOG(2) << "Inside correlator rft";
    _Payload pld;
    input["gulp"].pop(pld);
    VLOG(2) << "Payload received";
    if (!pld) {
      return raft::proceed;
    }

    auto& gulp_metadata = pld.get_mbuf()->GetMetadataRef();
    VLOG(2) << "Acquiring metadata";
    for (auto it = gulp_metadata.begin(); it != gulp_metadata.end(); ++it) {
      VLOG(3) << "Key: " << it->first;
    }
    auto nchan = std::get<uint8_t>(gulp_metadata["nchan"]);
    auto chan0 = std::get<int64_t>(gulp_metadata["chan0"]);

    VLOG(2) << "nchan: " << int(nchan) << " chan0: " << chan0;
    PrometheusExporter::ObserveRunTimeValue(m_rt_gauge_id, chan0);
    
    // check the include frequencies
    if( m_incl_freqs.count(chan0)!=0 || m_hc_chans.count(chan0)!=0){
      typename _Correlator::payload_t _empty;
      auto signal = static_cast<raft::signal>(INVALID_FREQ);
      output["img"].push(_empty, signal);
      output["img_stream"].push(_empty, signal);

      return raft::proceed;
    }
    // initialization or change in the spectral window
    // do not do any initializations for health check frequencies
    if ((m_hc_chans.count(chan0) == 0 || !m_is_GPU_setup_once) &&
      m_correlator.get()->ResetImagingConfig(nchan, chan0)) {
      m_is_GPU_setup_once = true;
      if (m_gain_reader) {
        m_gain_reader->Reset();  // tables for the old window do not apply
      }
      if (m_proj_reader) {
        m_proj_reader->Reset();
      }
      DLOG(INFO) << "Resetting GPU setup";
      m_delta = m_correlator.get()->GetScalingLen();
      m_gulp_counter = 1;
      m_session_id = GetRandomUuid();
      DLOG(INFO) << "Done resetting";
    }


    m_is_first = m_gulp_counter == 1 ? true : false;
    m_is_last = m_gulp_counter == m_ngulps_per_img ? true : false;

    VLOG(2) << "Setting the start id";
    if (m_is_first) {
      m_seq_start_id = std::get<uint64_t>(gulp_metadata["seq_start"]);
      // Gains change only at image boundaries, so every image is made with a
      // single gain table (recorded in its metadata as gain_version).
      if (m_gain_reader) {
        if (auto snap = m_gain_reader->Poll(chan0, nchan, m_seq_start_id)) {
          if (m_correlator.get()->ApplyGainTable(*snap)) {
            LOG(INFO) << "Applied calibration gain table version " << snap->version
                      << (snap->valid ? "" : " (revert to nominal phases)");
          }
        } else if (!m_gain_reader->LastError().empty()) {
          LOG_EVERY_N(WARNING, 100) << "Gain table rejected: " << m_gain_reader->LastError();
        }
      }
      // and the monitor's pixel table
      if (m_sky_table) {
        if (auto t = m_sky_table->Poll()) {
          const int nb = m_correlator.get()->GetCalDesc().sky_nbin;
          if (t->capacity == 0 || t->capacity > 65536 || int(t->grid) != m_grid_size) {
            LOG_EVERY_N(WARNING, 10) << "Sky pixel table v" << t->version << " not used: " << t->capacity
                                     << " slots for a " << t->grid << "-pixel grid";
          } else {
            m_correlator.get()->SetSkyTable(t->xy_a.data(), t->xy_b.data(), int(t->capacity), t->t_a, t->t_b, nb);
            if (m_sky_table_version == 0) {
              LOG(INFO) << "Regridding to the all-sky monitor's " << t->capacity << " pixel slots on the GPU";
            }
            m_sky_table_version = t->version;
          }
        }
      }
      // the projector likewise changes only between images
      if (m_proj_reader) {
        if (auto snap = m_proj_reader->Poll(chan0, nchan, m_seq_start_id)) {
          if (m_correlator.get()->ApplyProjTable(*snap)) {
            LOG_EVERY_N(INFO, 30) << "Applied interference projector version " << snap->version << " (" << snap->q
                                  << " components per polarisation" << (snap->sun.empty() ? "" : ", Sun exemption")
                                  << ")";
          }
        } else if (!m_proj_reader->LastError().empty()) {
          LOG_EVERY_N(WARNING, 100) << "Projector rejected: " << m_proj_reader->LastError();
        }
      }
    }
    // The tap stages its slice while ProcessGulp runs (WaitCopied below).
    bool tapped = false;
    const float quality = float(MetaOr<double>(gulp_metadata, "data_quality", 1.0));
    if (m_cal_tap) {
      tapped = m_cal_tap->Offer(pld.get_mbuf()->GetDataPtr(), std::get<uint64_t>(gulp_metadata["seq_start"]), chan0,
                                nchan, MetaOr<int>(gulp_metadata, "nseqs", 0), quality);
    }
    // The GPU tap correlates the gulp from the device copy ProcessGulp makes.
    m_correlator.get()->ArmTccTap(std::get<uint64_t>(gulp_metadata["seq_start"]), chan0, nchan,
                                  MetaOr<int>(gulp_metadata, "nseqs", 0), quality);

    if (m_is_last) {
      VLOG(3) << "Last gulp. Preparing metadata";
      // prepare the metadata for the image
      auto buf = m_correlator.get()->GetEmptyBuf();
      CHECK(static_cast<bool>(buf)) << "Correlator buffer allocation failed";
      auto& img_metadata = buf.get_mbuf()->GetMetadataRef();
      img_metadata = gulp_metadata;  // pld.get_mbuf()->GetMetadataRef();
      img_metadata["seq_start"] = m_seq_start_id;
      img_metadata["nseqs"] =
          std::get<int>(img_metadata["nseqs"]) * m_ngulps_per_img;
      img_metadata["img_len_ms"] =
          std::get<double>(img_metadata["gulp_len_ms"]) * m_ngulps_per_img;
      img_metadata["grid_size"] = m_grid_size;
      img_metadata["grid_res"] = m_grid_res;
      img_metadata["npols"] = m_npols;
      img_metadata["support_size"] = m_support;
      img_metadata["nchan"] = nchan;
      img_metadata["chan0"] = chan0;
      img_metadata["session_id"] = m_session_id;
      img_metadata["gain_version"] = m_correlator.get()->GetGainVersion();
      m_correlator.get()->SetImageTime(double(m_seq_start_id) * (double(SEQ_MULT_S) / double(FS)));
      // img_metadata["cfreq"] = int((chan0+ceil(nchan/2f))*BANDWIDTH);
      VLOG(3)
          << "Processing gulp at: "
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::high_resolution_clock::now().time_since_epoch())
                 .count();
      DLOG(INFO) << "Processing the gulp";
      if (m_correlator.get()->UsesFx() && m_pending) {
        m_correlator.get()->FinishFxImage();
        output["img"].push(m_pending);
        output["img_stream"].push(m_pending);
        m_pending = typename _Correlator::payload_t();
      }
      m_correlator.get()->ProcessGulp(
          pld.get_mbuf()->GetDataPtr(), buf.get_mbuf()->GetDataPtr(),
          m_is_first, m_is_last, static_cast<int>(chan0), m_delta);
      DLOG(INFO) << "Done processing the gulp";
      if (m_correlator.get()->SkyDone()) {
        // regridded to the monitor's pixels on the GPU: the spectra follow the image in buf
        img_metadata["sky_npix"] = uint64_t(m_correlator.get()->SkyPixels());
        img_metadata["sky_table_version"] = m_sky_table_version;
        img_metadata["sky_pix_offset"] = uint64_t(m_correlator.get()->SkyPixelOffset());
        img_metadata["sky_nchan"] = uint64_t(nchan / std::max(1, m_correlator.get()->GetCalDesc().sky_nbin));
      }
      // interference stage (only when configured, so other images are unchanged)
      if (m_proj_reader || cal_impulse_on()) {
        img_metadata["proj_version"] = m_correlator.get()->GetProjVersion();
        img_metadata["proj_q"] = uint64_t(m_correlator.get()->GetProjQ());
        // fraction of (stream sub-band, sequence) pairs the impulse detector zeroed
        img_metadata["impulse_flag_frac"] =
            double(m_correlator.get()->GetImpulseFlagged()) /
            double(std::max<size_t>(1, size_t(m_ngulps_per_img) * m_correlator.get()->GetNumSeqPerGulp() *
                                           m_correlator.get()->GetNumStreams()));
      }
      if (tapped) {
        m_cal_tap->WaitCopied();
      }
      m_gulp_counter = 1;
      if (m_correlator.get()->UsesFx()) {
        m_pending = buf;  // handed on at the next image boundary
        return raft::proceed;
      }
      VLOG(3) << "Pushing the output image";
      output["img"].push(buf);
      output["img_stream"].push(buf);

      return raft::proceed;
    }

    // If it's anything other than the final gulp, proceed right away. The
    // ProcessGulp function images the data in streams with asynchronous
    // reads/writes. Hence the next gulp won't have to wait for the current one
    // to complete thereby keeping the GPU completely occupied.

    m_correlator.get()->ProcessGulp(pld.get_mbuf()->GetDataPtr(),
                                    static_cast<float*>(nullptr), m_is_first,
                                    m_is_last);
    if (tapped) {
      m_cal_tap->WaitCopied();
    }
    ++m_gulp_counter;
    return raft::proceed;
  }
};

#endif  // SRC_RAFT_KERNELS_CORRELATOR_HPP_
