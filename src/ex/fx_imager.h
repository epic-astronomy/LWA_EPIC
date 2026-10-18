// FX imaging on the GPU: Tensor-Core Correlator visibilities gridded onto a
// lag grid with each pair's kernel cross-correlation, one cuFFT per image.
//
// Summed over the gulps of an image, an EPIC image is the transform of the
// gridded aperture's auto-correlation, which is the visibilities gridded with
// each pair's kernel cross-correlation:
//   I(k) = sum_D L(D) exp(2 pi i k D / N),
//   L(D) = sum_ab g_a g_b* R_ab sum_{c_a,i - c_b,j = D} K_a(i) K_b*(j),
// R_ab = sum_t V_a V_b*; leaving out a = b is --remove_autocorr. Only the
// differences of the stands' grid cells enter, so an N-point lag grid with the
// aperture cells of the E-field imager gives an N x N image of the same sky
// with pixels 128 / N of the 128-point ones (tests/gpu/fx_imager.cu checks
// N = 128 against the E-field kernel and N = 256 against its polyphase form).
//
// The correlation runs once per gulp (about 0.5 ms for 128 channels x 1000
// sequences on an RTX 4090), the gridding, transform and combination once per
// image (about 20 ms at N = 256), so the cost hardly depends on the image
// length. The image has the E-field imager's layout and scale:
// out[chan][x][y] = {XX, YY, Re XY*, Im XY*} times the gridding correction.
#ifndef SRC_EX_FX_IMAGER_H_
#define SRC_EX_FX_IMAGER_H_

#include <cuda_runtime_api.h>

#include <cstdint>
#include <memory>

class FxImager {
 public:
  /// p_n: image size (a power of two, at least 128); p_support: kernel support (5)
  FxImager(int p_device, int p_nchan, int p_nseq, int p_n, int p_support, bool p_remove_autocorr);
  ~FxImager();
  FxImager(const FxImager&) = delete;
  FxImager& operator=(const FxImager&) = delete;

  /// Geometry of the gridding, all on the device: stand positions in grid
  /// cells [chan][stand][3] (x, y, w), kernel weights [chan][stand][S^2]
  /// (complex p_gcf_cplx if not null, else real p_gcf_real) and the gridding
  /// correction [chan][N][N] (null: none). Copied, so the callers may free them.
  void SetGeometry(const float* p_antpos, const float* p_gcf_real, const float* p_gcf_cplx,
                   const float* p_correction);

  /// Correlate one gulp [chan][seq][input] (CNib) after p_ready (null: no
  /// wait); p_add accumulates onto the earlier gulps of the image. Records the
  /// event after which the gulp buffer may be overwritten (InputFree).
  void Correlate(const uint8_t* p_gulp, cudaEvent_t const* p_ready, int p_nready, bool p_add);
  /// Image the integrated visibilities with the per-stand phases and gains
  /// [chan][stand][Xre, Xim, Yre, Yim] into p_out (device, [chan][N][N][4])
  /// and copy it to p_host (pinned or pageable) when not null.
  void Image(const float* p_phases, float* p_out, float* p_host);
  /// Wait for the image (and its copy) to complete
  void Sync();

  /// Recorded after the gulp has been read; streams writing the gulp buffer wait for it
  cudaEvent_t InputFree() const;
  cudaStream_t Stream() const;
  int N() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> m;
};

#endif  // SRC_EX_FX_IMAGER_H_
