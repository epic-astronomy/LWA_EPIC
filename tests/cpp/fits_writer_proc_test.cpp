// Writes accumulations through FitsWriterProc; checked by tests/python/test_fits_writer.py.
//   fits_writer_proc_test <out dir> <njobs> [kill]
// Image n (float32 [nchan 2][grid 8][grid 8][4]) holds n*1000 + c*100 + a*10 + b + 0.1*k.
// With "kill", the writer is killed after the first job, so the second starts a new one.
// Prints one line per job: the returned path, or "ERROR <reason>".
#include <signal.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "../../src/ex/fits_writer_proc.hpp"

// as dict_t in src/ex/types.hpp, which needs CUDA
using dict_t = std::unordered_map<std::string, std::variant<int, int64_t, uint64_t, uint8_t, uint16_t, double, float, std::string>>;

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s outdir njobs [kill]\n", argv[0]);
    return 2;
  }
  const std::string out = argv[1];
  const int njobs = std::atoi(argv[2]);
  const bool kill_after_first = argc > 3 && std::string(argv[3]) == "kill";
  const size_t grid = 8, nchan = 2;
  FitsWriterProc proc;
  std::vector<float> img(nchan * grid * grid * 4);
  for (int n = 0; n < njobs; ++n) {
    for (size_t c = 0; c < nchan; ++c)
      for (size_t a = 0; a < grid; ++a)
        for (size_t b = 0; b < grid; ++b)
          for (size_t k = 0; k < 4; ++k)
            img[((c * grid + a) * grid + b) * 4 + k] = n * 1000.f + c * 100.f + a * 10.f + b + 0.1f * k;
    dict_t meta;
    meta["time_tag"] = uint64_t(392536608000000000ULL) + uint64_t(n) * 196000000ULL * 300ULL;
    meta["img_len_ms"] = 300000.0;
    meta["nimg"] = int(7178);
    meta["grid_size"] = int(grid);
    meta["grid_res"] = 1.0;
    meta["chan0"] = int64_t(3136);
    meta["nchan"] = uint8_t(nchan);
    meta["chan_width"] = int(1);
    meta["cfreq"] = 77.304;
    meta["gain_version"] = uint64_t(9);
    meta["support"] = uint16_t(5);
    meta["note"] = std::string("a \"quoted\" note");
    std::string why;
    const std::string path = proc.Write(grid, nchan, img.data(), out, meta, &why);
    if (path.empty()) {
      std::printf("ERROR %s\n", why.c_str());
    } else {
      std::printf("%s\n", path.c_str());
    }
    std::fflush(stdout);
    if (kill_after_first && n == 0) {
      // the writer is this process's child: kill it as a crash would
      std::string cmd = "pkill -KILL -P " + std::to_string(getpid()) + " -f epic_fits_writer";
      if (std::system(cmd.c_str()) != 0) std::fprintf(stderr, "pkill found no writer\n");
      usleep(200000);
    }
  }
  return 0;
}
