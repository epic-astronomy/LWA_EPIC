// Copyright (c) 2026 The EPIC++ authors
//
// GPU calibration tap: correlates every gulp on the imaging GPU with the ASTRON
// Tensor-Core Correlator (libtcc) and publishes the integrated visibilities in
// the table the CPU calibrator reads (vis_table.hpp).
//
// The gulp is correlated from the copy the imager already holds on the device,
// so the tap adds no host-to-device traffic. Per gulp, on the tap's own stream:
//   1. wait until every imaging stream has copied its channels (AfterCopy),
//   2. reorder the 4-bit gulp [chan][seq][stand][pol] into the TCC 8-bit
//      layout [chan][seq / 16][stand][pol][16][re, im] (zero padded to a
//      multiple of 16 sequences),
//   3. correlate, accumulating into the current visibility buffer.
// Before the next gulp is copied over the device buffer, each imaging stream
// waits for step 2 of the previous gulp (BeforeCopy), about 0.2 ms after its
// copy; the 35 ms imaging kernels are never delayed in practice.
// After `interval_s` of F-engine time the accumulator is copied to the host on
// a second stream while the next integration starts in the other buffer, and
// a publisher thread converts it to the table.
//
// The rest of epic++ sees only this header; the implementation is tcc_tap.cu,
// which needs EPIC_WITH_TCC (cmake -DEPIC_TCC_ROOT=...).
#ifndef SRC_EX_TCC_TAP_H_
#define SRC_EX_TCC_TAP_H_

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>
#include <string>

struct TccTapOptions {
  std::string path;          // output table, e.g. /dev/shm/epic_vis_gpu0
  float interval_s{10.f};    // integration per published table
  float min_quality{0.98f};  // skip gulps with less valid data than this
  std::string record_dir;    // also write every table to <record_dir>/vis_<version>.bin ("" = off)
};

class TccTap {
 public:
  /// Compiles the correlator for `p_nchan` channels of `p_nseq` sequences per
  /// gulp (a few seconds, NVRTC) and allocates its buffers on `p_device`.
  TccTap(TccTapOptions p_opts, int p_device, int p_nchan, int p_nseq, int p_nstreams);
  ~TccTap();
  TccTap(const TccTap&) = delete;
  TccTap& operator=(const TccTap&) = delete;

  /// Describe the next gulp before it is processed. Returns true if it will
  /// be correlated (right shape, enough valid data).
  bool Arm(uint64_t p_seq_start, int64_t p_chan0, int p_nchan, int p_nseq, float p_quality);
  /// Imaging stream `p_stream` is about to overwrite the device gulp buffer.
  void BeforeCopy(cudaStream_t p_stream);
  /// Imaging stream number `p_i` has queued the copy of its channels.
  void AfterCopy(int p_i, cudaStream_t p_stream);
  /// All copies are queued: reorder and correlate the armed gulp held at `p_gulp`.
  void Launch(const uint8_t* p_gulp);

  uint64_t Version() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> m;
};

#endif  // SRC_EX_TCC_TAP_H_
