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

#ifndef SRC_EX_OPTION_PARSER_HPP_
#define SRC_EX_OPTION_PARSER_HPP_
#include <cxxopts.hpp>
#include <iostream>
#include <iterator>
#include <string>
#include <sstream>
#include <vector>

#include "./constants.h"
#include "./metrics.hpp"
#include "./host_helpers.h"
#include "git.h"

cxxopts::Options GetEpicOptions() {
  cxxopts::Options options("epic++", "EPIC dual-pol imager");

  options.add_options("Online data processing")(
      "addr", "F-Engine UDP Stream Address",
      cxxopts::value<std::vector<std::string>>()->default_value(
          "239.168.40.11,239.168.40.12"))(
      "port", "F-Engine UDP Stream Port",
      cxxopts::value<std::vector<int>>()->default_value("10000,10000"))(
      "printendpoints",
      "Print the IP/port values and their channels for each endpoint in the "
      "LWA-SV station and exit",
      cxxopts::value<bool>()->default_value("false"))(
      "runtime", "EPIC run duration. Provide -1 to run continuously",
      cxxopts::value<int>()->default_value("5"))
      ("include_chans","Image only certain frequencies. Specify the value(s) of chan0.", cxxopts::value<std::vector<int>>());
  // ("utcstart", "F-Engine UDP Stream Start Time")

  options.add_options("Benchmark (no F-engine, database, disk or video)")(
      "bench", "Run the imager on generated gulps: generator -> correlator (with the calibration tap and gain table) -> sink",
      cxxopts::value<bool>()->default_value("false"))(
      "bench_data", "File of simulated gulps (CHAN_MAJOR CNib); empty: zero gulps",
      cxxopts::value<std::string>()->default_value(""))(
      "bench_chan0", "First channel of the generated gulps",
      cxxopts::value<int64_t>()->default_value("3104"))(
      "bench_seq0", "First F-engine sequence (from the Unix epoch); 0: now",
      cxxopts::value<uint64_t>()->default_value("0"))(
      "bench_ngulps", "Gulps to generate (0: unlimited)",
      cxxopts::value<int64_t>()->default_value("2400"))(
      "bench_realtime", "Pace the gulps at the F-engine rate instead of as fast as possible",
      cxxopts::value<bool>()->default_value("false"))(
      "bench_rotate", "Rotation of each channel's sequences per pass through the data file",
      cxxopts::value<int>()->default_value("50"))(
      "bench_dump", "Directory for image dumps (empty: none)",
      cxxopts::value<std::string>()->default_value(""))(
      "bench_dump_every", "Dump every Nth image",
      cxxopts::value<int>()->default_value("0"))(
      "bench_report", "Images per statistics line",
      cxxopts::value<int>()->default_value("120"));

  options.add_options("Offline data processing")(
      "offline", "Load numpy-TBN data from disk",
      cxxopts::value<bool>()->default_value("false"))(
      "npytbnfile", "numpy-TBN data path", cxxopts::value<std::string>());

  options.add_options("Imaging options")(
      "imagesize", "1-D image size (can only be 64 or 128)",
      cxxopts::value<int>()->default_value("128"))(
      "imageres", "Pixel resolution in degrees",
      cxxopts::value<float>()->default_value("1.0"))(
      "nts", "Number of timestamps per span",
      cxxopts::value<int>()->default_value("1000"))(
      "seq_accum", "Duration of the sequence accumulation in milliseconds",
      cxxopts::value<int>()->default_value("40"))(
      "nimg_accum", "Number of images to accumulate before saving to the disk",
      cxxopts::value<int>()->default_value("1"))(
      "nimg_accum_min",
      "Fewest images a partial accumulation, ended by a gap in the images or a "
      "change of frequency, must hold to be saved (with its true length). "
      "-1: 10% of --nimg_accum.",
      cxxopts::value<int>()->default_value("-1"))(
      "channels", "Number of channels in the output image",
      cxxopts::value<int>()->default_value("128"))(
      "support", "Support size of the kernel. Must be a non-zero power of 2",
      cxxopts::value<int>()->default_value("2"))(
      "aeff", "Antenna effective area (experimental) in sq. m",
      cxxopts::value<std::vector<float>>()->default_value("25,16"))(
      "kernel_oversample",
      "Factor to over sample the kernel. Must be a power of 2.",
      cxxopts::value<int>()->default_value("2"))(
      "accum_16bit",
      "Use 16-bit precision for on-chip memory accumulation. Faster but less "
      "precise.",
      cxxopts::value<bool>()->default_value("false"))(
      "chan_nbin", "Binning factor for the number of channels",
      cxxopts::value<int>()->default_value("4"))(
      "wproj",
      "Fold each stand's residual w-term into complex per-antenna gridding "
      "kernels (w-projection). The phase centre is set with the "
      "EPIC_PHASE_CENTER environment variable (plane|zenith).",
      cxxopts::value<bool>()->default_value("false"))(
      "remove_autocorr",
      "Remove the antenna auto-correlations from the images, per stand "
      "(Thyagarajan et al. 2017, sec. 3.2). Without it the images carry a "
      "smooth positive background equal to the total antenna power.",
      cxxopts::value<bool>()->default_value("false"))(
      "seq_chunk",
      "Sequences per imaging kernel launch. The 16-bit on-chip accumulator "
      "(--accum_16bit) under-sums long runs (~24% low over 1000 sequences), "
      "so each gulp is imaged in launches of this many sequences. -1: 100 "
      "with --accum_16bit, otherwise the whole gulp.",
      cxxopts::value<int>()->default_value("-1"))(
      "gain_table",
      "Shared-memory gain table written by the CPU calibrator (streamcal), "
      "e.g. /dev/shm/epic_gains_gpu{gpu}. Read at image boundaries and "
      "multiplied into the per-antenna phases. Empty: no calibration.",
      cxxopts::value<std::string>()->default_value(""))(
      "proj_table",
      "Shared-memory interference projector written by streamcal (daemon "
      "--proj), e.g. /dev/shm/epic_proj_gpu{gpu}. Read at image boundaries; "
      "every voltage sequence x becomes x - U (V^H x) before imaging, while "
      "the calibration tap keeps the raw voltages. Empty: off.",
      cxxopts::value<std::string>()->default_value(""))(
      "impulse_sigma",
      "Zero voltage sequences whose band power exceeds the gulp median by "
      "this many robust standard deviations, per stream sub-band, before "
      "imaging (broadband impulses). 0: off.",
      cxxopts::value<float>()->default_value("0"))(
      "caltap_nseq",
      "Calibration tap: sequences of every gulp correlated on the CPU and "
      "published for streamcal (0: off; 50 = 5% of a 1000-sequence gulp).",
      cxxopts::value<int>()->default_value("0"))(
      "caltap_out",
      "Calibration tap output file ({gpu} is replaced by the device id).",
      cxxopts::value<std::string>()->default_value("/dev/shm/epic_vis_gpu{gpu}"))(
      "caltap_interval",
      "Calibration tap integration per published table, in seconds.",
      cxxopts::value<float>()->default_value("10"))(
      "caltap_threads",
      "Calibration tap X-engine threads.",
      cxxopts::value<int>()->default_value("2"))(
      "caltap_batch",
      "Calibration tap: gulps staged per X-engine call (amortises the accumulator updates).",
      cxxopts::value<int>()->default_value("8"))(
      "caltap_gpu",
      "Calibration tap on the GPU: correlate every gulp with the tensor-core "
      "correlator (all samples, ~0.5 ms per gulp beside the imaging kernels) "
      "instead of --caltap_nseq sequences on the CPU. Needs epic++ built with "
      "-DEPIC_TCC_ROOT.",
      cxxopts::value<bool>()->default_value("false"))(
      "caltap_min_quality",
      "GPU calibration tap: skip gulps with less valid data than this fraction.",
      cxxopts::value<float>()->default_value("0.98"))(
      "caltap_record_dir",
      "GPU calibration tap: also write every published table to this directory "
      "(vis_<version>.bin, 128-byte header + R). Empty: off.",
      cxxopts::value<std::string>()->default_value(""))(
      "vistap",
      "Run only the F-Engine receiver and the GPU calibration tap (no imaging, "
      "database or video) for --runtime seconds; tables go to --caltap_out "
      "and --caltap_record_dir. For calibration data takes.",
      cxxopts::value<bool>()->default_value("false"))
      ("out_dir","Output directory to store the long accumulation files. Defaults to the current directory.",cxxopts::value<std::string>()->default_value("./"));

  options.add_options("All-sky monitor")(
      "sky_shm",
      "Publish every image's Stokes I (XX + YY) at full resolution to the "
      "shared-memory ring /dev/shm/<name> for the all-sky monitor "
      "(src/python/skymon). Empty: off.",
      cxxopts::value<std::string>()->default_value(""))(
      "sky_shm_slots", "Images held by the sky ring",
      cxxopts::value<int>()->default_value("64"))(
      "sky_nbin", "Channels summed per published channel",
      cxxopts::value<int>()->default_value("1"));

  options.add_options("Execution options")(
      "nstreams", "Number of cuda streams to process images",
      cxxopts::value<int>()->default_value("8"))(
      "ngpus", "number of GPUs to simultaneously run EPIC",
      cxxopts::value<int>()->default_value("1"))(
      "gpu_ids",
      "GPU ID for each EPIC instance. Defaults to automatic assignment",
      cxxopts::value<std::vector<int>>());

  options.add_options("Metrics collection")("h,help", "Print usage")(
      "metrics_bind_addr",
      "Address to expose Prometheus metrics. Defaults to 127.0.0.1:8080",
      cxxopts::value<std::string>()->default_value("127.0.0.1:8080"))(
      "disable_metrics", "Disable writing metrics to the Prometheus endpoint",
      cxxopts::value<bool>()->default_value("false"))(
      "cpu_layout",
      "CPUs of the pipeline kernels for GPU 0 (GPU g adds g times the block "
      "size). wide: one CPU per kernel on 3-11. compact: 3-7, with capture "
      "on 3, correlator on 4, channel reducer on 5, video on 6 and the "
      "pixel extractor, watch-list fetcher, database writer, accumulator and "
      "disk writer sharing 7, which leaves 8-11 to other programs",
      cxxopts::value<std::string>()->default_value("wide"));

  options.add_options("Database Operations")(
      "elev_limit_deg",
      "Extract pixels into the DB only if the source is above this elevation "
      "(degrees). "
      "Defaults to 10 deg",
      cxxopts::value<float>()->default_value("10.0"))(
      "watchdog_addr", "Address for the EPIC watchdog services.",
      cxxopts::value<std::string>()->default_value("localhost:2023"))
      ("epic_data_schema","Schema name for all epic data tables.",
      cxxopts::value<std::string>()->default_value("public"));

  options.add_options("Live Streaming")(
      "video_size", "Stream video dimensions. Defaults to 512",
      cxxopts::value<int>()->default_value("512"))(
      "stream_url", "RTMP endpoint to stream the video to",
      cxxopts::value<std::string>()->default_value(
          "rtmp://127.0.0.1:1857/live/epic"))(
            "stream_cmap", "Colormap for the output video stream. See https://ffmpeg.org/ffmpeg-filters.html#pseudocolor for available ones.",
            cxxopts::value<std::string>()->default_value("magma")
          );

  return options;
}

std::optional<std::string> ValidateOptions(const cxxopts::ParseResult& result) {
  using std::string_literals::operator""s;
  auto ports = result["port"].as<std::vector<int>>();
  for (auto& port : ports) {
    if (port < 1 || port > 32768) {
      return "Invalid port number:  "s + std::to_string(port) +
             ". Port must be a number in 1-32768"s;
    }
  }

  int image_size = result["imagesize"].as<int>();
  if (!(image_size == 64 || image_size == 128)) {
    return "Invalid image size: "s + std::to_string(image_size) +
           ". Image size can only be 64 or 128"s;
  }

  int nts = result["nts"].as<int>() * SAMPLING_LEN_uS * 1e-3;
  int accumulate = result["seq_accum"].as<int>();

  if (accumulate < nts) {
    return "Sequence accumulation time must be greater than the gulp size."s;
  }

  int channels = result["channels"].as<int>();
  if (channels <= 0) {
    return "The number of output channels must be at least 1"s;
  }

  int ndevices = result["ngpus"].as<int>();

  if (ndevices <= 0) {
    return "ngpus must be greater than 0";
  }

  if (ndevices > GetNumGpus()) {
    return "Y'all must be kidding. You said EPIC must be run on " +
           std::to_string(ndevices) + " gpu(s) but only " +
           std::to_string(GetNumGpus()) + " available!";
  }

  if (channels > MAX_CHANNELS_4090) {
    return "RTX 4090 only supports output channels upto "s +
           std::to_string(MAX_CHANNELS_4090);
  }

  auto aeffs = result["aeff"].as<std::vector<float>>();
  if (aeffs.size() < ndevices) {
    return "Effective area must be specified for each frequency window";
  }

  for (auto aeff : aeffs) {
    if (aeff <= 0) {
      return "Antenna effective cannot be smaller than or equal to zero:  "s +
             std::to_string(aeff);
    }
  }

  if (result["gpu_ids"].count() > 0) {
    auto gpu_ids = result["gpu_ids"].as<std::vector<int>>();
    if (gpu_ids.size() > 0 && gpu_ids.size() != ndevices) {
      "A gpu ID must be specifed for each EPIC instance";
    }
  }

  int support = result["support"].as<int>();
  if (support <= 0 || support > MAX_ALLOWED_SUPPORT_SIZE) {
    return "Invalid support size: "s + std::to_string(support) +
           ". Support can only be between 1-"s +
           std::to_string(MAX_ALLOWED_SUPPORT_SIZE);
  }

  int kos = result["kernel_oversample"].as<int>();
  if ((kos & (kos - 1)) != 0) {
    return "Kernel oversampling factor must be a power of 2";
  }

  int nbin = result["chan_nbin"].as<int>();
  int nchan = result["channels"].as<int>();

  if (nchan % nbin != 0) {
    return "Number of channels must be an integral multiple of the binning "
           "factor.";
  }

  int nstreams = result["nstreams"].as<int>();

  if (nstreams <= 0) {
    return "The number of streams must be greater than 0";
  }

  if (result["elev_limit_deg"].as<float>() < 0) {
    return "Elevation limit must be between 0 and 90 degrees";
  }

  if (auto l = result["cpu_layout"].as<std::string>(); l != "wide" && l != "compact") {
    return "cpu_layout must be wide or compact";
  }

  return {};
}

template<typename T>
std::string VJoin(std::vector<T> inp_vec, const char* delimiter=","){
  if(inp_vec.size()==0){
    return "";
  }

  if(inp_vec.size()==1){
    return std::to_string(inp_vec[0]);
  }
  std::stringstream result;
  std::copy(inp_vec.begin(), inp_vec.end(), std::ostream_iterator<T>(result, delimiter));
  return result.str();
}

template<>
std::string VJoin<>(std::vector<std::string> inp_vec, const char* delimiter){
  if(inp_vec.size()==0){
    return "";
  }

  if(inp_vec.size()==1){
    return inp_vec[0];
  }
  std::stringstream result;
  std::copy(inp_vec.begin(), inp_vec.end(), std::ostream_iterator<std::string>(result, delimiter));
  return result.str();
}

void InitInfoMetric(const cxxopts::ParseResult& result){
//git_CommitSHA1()
  using sVec = typename std::vector<std::string>;
  using iVec = typename std::vector<int>;
  PrometheusExporter::AddInfoLabels({
    {"F_engines", VJoin(result["addr"].as<sVec>())},
    {"ports",VJoin(result["port"].as<iVec>())},
    {"image_size",std::to_string(result["imagesize"].as<int>())},
    {"imageres", std::to_string(result["imageres"].as<float>())},
    {"nts",std::to_string(result["nts"].as<int>())},
    {"seq_accum",std::to_string(result["seq_accum"].as<int>())},
    {"nimg_accum",std::to_string(result["nimg_accum"].as<int>())},
    {"nimg_accum_min",std::to_string(result["nimg_accum_min"].as<int>())},
    {"channels",std::to_string(result["channels"].as<int>())},
    {"support",std::to_string(result["support"].as<int>())},
    {"kernel_oversample",std::to_string(result["kernel_oversample"].as<int>())},
    {"accum_16bit",std::to_string(result["accum_16bit"].as<bool>())},
    {"chan_nbin",std::to_string(result["chan_nbin"].as<int>())},
    {"wproj",std::to_string(result["wproj"].as<bool>())},
    {"remove_autocorr",std::to_string(result["remove_autocorr"].as<bool>())},
    {"seq_chunk",std::to_string(result["seq_chunk"].as<int>())},
    {"gain_table",result["gain_table"].as<std::string>()},
    {"proj_table",result["proj_table"].as<std::string>()},
    {"impulse_sigma",std::to_string(result["impulse_sigma"].as<float>())},
    {"caltap_nseq",std::to_string(result["caltap_nseq"].as<int>())},
    {"caltap_interval",std::to_string(result["caltap_interval"].as<float>())},
    {"caltap_gpu",std::to_string(result["caltap_gpu"].as<bool>())},
    {"nstreams",std::to_string(result["nstreams"].as<int>())},
    {"ngpus",std::to_string(result["ngpus"].as<int>())},
    {"gpu_ids",VJoin(result["gpu_ids"].as<iVec>())},
    {"elev_limit_deg",std::to_string(result["elev_limit_deg"].as<float>())},
    {"video_size",std::to_string(result["video_size"].as<int>())},
    {"stream_cmap",result["stream_cmap"].as<std::string>()},
    {"imager_build_hash",git_CommitSHA1()}
  });
}
#endif  // SRC_EX_OPTION_PARSER_HPP_
