// Copyright (c) 2026 The EPIC++ authors
//
// Interference stage before imaging, on the imaging GPU. For the channels of
// one imaging stream, after the gulp has been copied to the device:
//   1. RfiProject: every sequence x (512 inputs) of a channel becomes
//      x - Ut (Vt^H x), with the projector published by streamcal
//      (proj_table.hpp); the result is re-quantised to 4+4 bits with a
//      uniform dither (rounding alone costs 3.5% of visibility S/N, dithered
//      1.1%) and written to a second gulp buffer, which the imaging kernel
//      reads. The band power of each (channel, sequence) is recorded.
//   2. ImpulseFlag (optional): per stream sub-band, sequences whose band power
//      exceeds the gulp median by impulse_sigma robust standard deviations
//      (1.4826 x MAD) are zeroed.
// The raw gulp buffer is left untouched, so the calibration tap correlates the
// unprojected voltages and streamcal keeps seeing the interference it tracks.
// Each imaging stream processes the same channels in every gulp, so the second
// buffer needs no synchronisation beyond stream order.
#ifndef SRC_EX_RFI_STAGE_H_
#define SRC_EX_RFI_STAGE_H_

#include <cuda_runtime_api.h>

#include <complex>
#include <cstdint>

#include "./constants.h"

/// Most projector components per channel the imager applies
constexpr int kProjQmax = 4;

class RfiStage {
 public:
  RfiStage(int p_device, float p_impulse_sigma);
  ~RfiStage();
  RfiStage(const RfiStage&) = delete;
  RfiStage& operator=(const RfiStage&) = delete;

  /// (Re)allocate for gulps of p_nchan channels x p_nseq sequences; clears the projector
  void Resize(int p_nchan, int p_nseq);
  /// Upload a projector, [p_nchan][kProjQmax][512] each (synchronous: call
  /// between gulps). p_q = 0 switches the projection off.
  void SetProjector(const std::complex<float>* p_ut, const std::complex<float>* p_vt, int p_q, int p_nchan);
  void ClearProjector() { m_q = 0; }
  int Q() const { return m_q; }
  /// Anything to do for the next gulp (projection or impulse detection)
  bool Active() const { return m_q > 0 || m_sigma > 0.f; }
  /// Start a gulp (advances the dither sequence)
  void NextGulp() { ++m_gulp; }
  /// Process channels [p_chan_offset, p_chan_offset + p_nchan) of the device
  /// gulp on stream p_stream (imaging stream index p_si)
  void Run(const uint8_t* p_gulp, int p_chan_offset, int p_nchan, cudaStream_t p_stream, int p_si);
  /// Queue the read-back of stream p_si's impulse count at the end of an image
  void EndImage(int p_si, cudaStream_t p_stream);
  /// After the streams have synchronised: (stream, sequence) pairs zeroed in the image
  uint64_t CollectFlagged();
  /// The processed gulp, same layout as the raw one
  uint8_t* Out() const { return m_out; }

 private:
  int m_device;
  float m_sigma;
  int m_nchan{0}, m_nseq{0};
  int m_q{0};
  uint32_t m_gulp{0};
  uint8_t* m_out{nullptr};
  float2* m_ut{nullptr};
  float2* m_vt{nullptr};
  float* m_power{nullptr};
  uint32_t* m_cnt_d{nullptr};
  uint32_t* m_cnt_h{nullptr};
};

#endif  // SRC_EX_RFI_STAGE_H_
