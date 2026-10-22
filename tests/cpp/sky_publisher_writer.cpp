// Writes test images through SkyPublisher; read back by tests/python/test_skymon.py.
//   sky_publisher_writer <ring name> <nchan> <grid> <nbin> <nimages> [peel]
// With "peel" the images are published as pixel spectra (layout 1) of 10 pixels:
// spectra c*100 + p, peaks [5][nchan/nbin] 1000*s + c (source 0 only peeled), and
// the subtracted term -(c*100 + p) - 0.5.
// Element (chan c, a, b, pol k) of image n holds n*1e5 + c*1e4 + a*100 + b + 0.1*k,
// seq_start of image n is 1000000 + 2000 n, and its gain_version is 7 + n.
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "../../src/ex/sky_publisher.hpp"

int main(int argc, char** argv) {
  if (argc < 6) {
    std::fprintf(stderr, "usage: %s name nchan grid nbin nimages\n", argv[0]);
    return 2;
  }
  const int nchan = std::atoi(argv[2]), grid = std::atoi(argv[3]);
  const int nbin = std::atoi(argv[4]), nimg = std::atoi(argv[5]);
  const bool peel = argc > 6 && std::string(argv[6]) == "peel";
  const int nout = nchan / nbin, npix = 10;
  std::vector<float> pix(size_t(nout) * (2 * npix + SkyPublisher::kPeelSources));
  for (int c = 0; c < nout; ++c)
    for (int p = 0; p < npix; ++p) {
      pix[size_t(c) * npix + p] = c * 100.f + p;
      pix[size_t(nout) * (npix + SkyPublisher::kPeelSources) + size_t(c) * npix + p] = -(c * 100.f + p) - 0.5f;
    }
  for (int s = 0; s < SkyPublisher::kPeelSources; ++s)
    for (int c = 0; c < nout; ++c) pix[size_t(nout) * npix + size_t(s) * nout + c] = 1000.f * s + c;
  SkyPublisher pub(argv[1], 4, nchan, grid, nbin);
  std::vector<float> img(size_t(nchan) * grid * grid * 4);
  for (int n = 0; n < nimg; ++n) {
    for (int c = 0; c < nchan; ++c)
      for (int a = 0; a < grid; ++a)
        for (int b = 0; b < grid; ++b)
          for (int k = 0; k < 4; ++k)
            img[((size_t(c) * grid + a) * grid + b) * 4 + k] =
                n * 1e5f + c * 1e4f + a * 100.f + b + 0.1f * k;
    SkyPublisher::ImageInfo info;
    info.seq_start = 1000000ULL + 2000ULL * n;
    info.nseqs = 2000;
    info.chan0 = 2336;
    info.grid_res = 1.056;
    info.img_len_ms = 83.592;
    info.gain_version = 7ULL + uint64_t(n);
    if (peel) {
      info.layout = 1;
      info.npix = npix;
      info.table_version = 3;
      info.pix = pix.data();
      info.peel_mask = 1u;
    }
    pub.Publish(img.data(), nchan, info);
  }
  std::printf("%llu\n", static_cast<unsigned long long>(pub.Count()));
  return 0;
}
