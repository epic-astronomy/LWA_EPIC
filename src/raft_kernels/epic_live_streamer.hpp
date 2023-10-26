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

#ifndef SRC_RAFT_KERNELS_EPIC_LIVE_STREAMER_HPP_
#define SRC_RAFT_KERNELS_EPIC_LIVE_STREAMER_HPP_
#include <algorithm>
#include <memory>
#include <string>
#include <variant>

#include "../ex/constants.h"
#include "../ex/video_streaming.hpp"
#include "glog/logging.h"
// #include "raft"
// #include "raftio"

/* template <typename _Pld> */
class EpicLiveStream: protected Streamer /* : raft::kernel */ {
 private:
  std::unique_ptr<uint8_t[]> m_frame_buf;
  std::unique_ptr<float[]> m_raw_frame;
  int64_t m_chan0{0};
  int m_chan_stream_no{0};
  int m_grid_size{128};
  int m_vid_size{512};
  int m_fps{10};
  std::string m_stream_key;
  FILE* m_pipeout;
  bool m_is_pipeopen{false};
  int m_npixels{128 * 128};

 public:
  EpicLiveStream(int p_grid_size = 128, int p_vid_size = 1280,
                 int p_chan_stream_no = 0,
                 std::string p_stream_key = "xmxb-5619-03rx-v7t7-3g3d",
                 int p_fps = 25)
      : Streamer(p_fps, p_grid_size, p_grid_size, p_fps),
        m_grid_size(p_grid_size),
        m_vid_size(p_vid_size),
        m_chan_stream_no(p_chan_stream_no),
        m_fps(p_fps),
        m_stream_key(p_stream_key),
        m_npixels(p_grid_size * p_grid_size) {
    // input.addPort<_Pld>("in_img");
    m_raw_frame = std::make_unique<float[]>(p_grid_size * p_grid_size);
    m_frame_buf = std::make_unique<uint8_t[]>(p_grid_size * p_grid_size);
  }


  void CopyToFrameBuf(float* p_data_ptr, int p_chan_no) {
    int offset = (p_chan_no - 1) * m_npixels * NSTOKES;

    for (int i = 0; i < m_npixels; ++i) {
      m_raw_frame.get()[i] = p_data_ptr[i * NSTOKES + offset];  // this is XX*
    }
    auto max_val =
        *std::max_element(m_raw_frame.get(), m_raw_frame.get() + m_npixels);

    // normalize and write to the frame buffer
    // FFT shift and transpose
    for (int i = 0; i < m_grid_size; i++) {
      int ii = (i + m_grid_size / 2) % m_grid_size;
      for (int j = 0; j < m_grid_size; j++) {
        int jj = (j + m_grid_size / 2) % m_grid_size;
        this->frame->data[0][jj * m_grid_size + ii] =
            m_raw_frame.get()[i * m_grid_size + j] * 255 / max_val;
      }
    }
    // std::transform(m_raw_frame.get(), m_raw_frame.get() + m_npixels,
    //                m_frame_buf.get(),
    //                [=](float d) { return (d * 255) / max_val; });
  }

  // raft::kstatus run() override {
  //   _Pld pld;
  //   input["in_img"].pop(pld);
  //   const auto& meta = pld.get_mbuf()->GetMetadataRef();
  //   auto chan0 = std::get<int64_t>(meta["chan0"]);
  //   auto cfreq = std::get<double>(meta["cfreq"]);

  //   ResetPipe(chan0, cfreq);

  //   CopyToFrameBuf(pld.get_mbuf()->GetDataPtr(), m_chan_stream_no);

  //   fwrite(m_frame_buf.get(), 1, m_npixels, m_pipeout);
  // }

  void stream(int64_t chan0, double cfreq, float* data_ptr) {
    // ResetPipe(chan0, cfreq);

    CopyToFrameBuf(data_ptr, m_chan_stream_no);
    this->StreamImage();

    //fwrite(m_frame_buf.get(), 1, m_npixels, m_pipeout);
  }
};

#endif  // SRC_RAFT_KERNELS_EPIC_LIVE_STREAMER_HPP_
/* std::string GetStreamCmd(double cfreq) {
    char t[1000];
    snprintf(
        t, sizeof(t),
        R"(ffmpeg -hide_banner -nostats -threads 10 -loglevel error -y -f lavfi -i anullsrc \
      -f rawvideo -r %d -video_size %dx%d -pixel_format gray -i pipe: \
      -f lavfi -i color=size=%dx%d:rate=%d:duration=10:color=black \
      -filter_complex "[2:v]format=rgba,colorchannelmixer=aa=0.0,drawtext=text='%.2f MHz':fontcolor=white:fontsize=h/15:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-1.1*text_w):y=(h-1.3*text_h),scale=1280:-1[B]; \
      [1:v][B]scale2ref[A][C]; \
      [A][C]overlay[out]" \
      -b:v 4500k -maxrate 4500k -bufsize 4500k  -crf 18 -vcodec libx264 -pix_fmt yuv420p \
      -map [out] -map 0:a -shortest \
      -f flv rtmp://a.rtmp.youtube.com/live2/%s)",
        m_fps, m_grid_size, m_grid_size, m_vid_size, m_vid_size, m_fps, cfreq,
        m_stream_key.c_str());

    return std::string(t);
    // std::string("ffmpeg -y -f lavfi -i aevalsrc=0  -f rawvideo -r ") +
    //     std::to_string(fps) + " -video_size " + std::to_string(width) + "x" +
    //     std::to_string(height) +
    //     " -pixel_format gray -threads 0  -i pipe: -vcodec libx264 -b:v 4500k
    //     "
    //     "-maxrate 4500k -bufsize 4500k  -crf 18 -vf \"drawtext=text='38.1 MHz
    //     "
    //     "%{pts \\: "
    //     "hms}':fontcolor=white:fontsize=h/"
    //     "15:box=1:boxcolor=black@0.5:boxborderw=5:x=(w-1.2*text_w):y=(h-1.2*"
    //     "text_h),scale=1280:-1\" -pix_fmt yuv420p -f flv " +
    //     "rtmp://a.rtmp.youtube.com/live2/xmxb-5619-03rx-v7t7-3g3d";
  } */

  // void ResetPipe(int64_t p_chan0, double p_cfreq) {
  //   if (p_chan0 == m_chan0) {
  //     return;
  //   }
  //   m_chan0 = p_chan0;
  //   if (m_is_pipeopen) {
  //     fflush(m_pipeout);
  //     pclose(m_pipeout);
  //   }
  //   m_pipeout = popen(GetStreamCmd(p_cfreq).c_str(), "w");
  //   m_is_pipeopen = true;
  // }