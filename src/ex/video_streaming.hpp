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

#ifndef SRC_EX_VIDEO_STREAMING_HPP_
#define SRC_EX_VIDEO_STREAMING_HPP_
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/codec.h>
#include <libavcodec/codec_par.h>
#include <libavcodec/packet.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/dict.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/log.h>
#include <libavutil/mathematics.h>
#include <libavutil/rational.h>
#include <libavutil/samplefmt.h>
#include <libswscale/swscale.h>
}

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <thread>

#include "glog/logging.h"
#include "cxxopts"

class Streamer {
  using Status_t = std::optional<std::string>;

 private:
  int m_fps;
  int m_width;
  int m_height;
  int m_time_base_den;
  std::string m_stream_url;
  int m_log_level{AV_LOG_DEBUG};
  int m_npixels_grid;
  int m_npixels_vid;
  int _frame_counter{0};
  AVRational dst_fps;
  AVFormatContext *outputContext{nullptr};
  AVStream *videoStream{nullptr};
  AVCodecContext *codecContext{nullptr};
  AVDictionary *codec_options{nullptr};
  AVCodec *videoCodec;
  int m_vid_frame_size{0};
  uint64_t m_frame_PTS{0};
  uint8_t *framebuf{nullptr};
  std::string out_rtmp_url;
  AVPacket *pkt;

  std::unique_ptr<uint8_t[]> m_frame_buf;
  std::unique_ptr<float[]> m_raw_frame;
  int64_t m_chan0{0};
  int m_chan_stream_no{1};
  int m_grid_size{128};
  int m_vid_size{512};
  // int m_fps{10};
  std::string m_stream_key;
  FILE *m_pipeout;
  bool m_is_pipeopen{false};
  SwsContext *m_sws_context;
  // int m_npixels{128 * 128};

 protected:
  AVFrame *frame{nullptr};
  void CheckError(const Status_t &p_status);

  Status_t InitOutputContext();
  Status_t InitVideoStream();
  Status_t InitVideoCodecContext();
  Status_t InitVideoEncoder();
  Status_t InitVideoFrameBuf();
  Status_t InitOutput();
  Status_t InitVideoPkt();

 public:
  Streamer(int p_fps = 10, int p_width = 512, int p_height = 512,
           int p_time_base_den = 10,
           std::string p_stream_url = "rtmp://127.0.0.1:1857/live/epic",
           int p_log_level = AV_LOG_ERROR)
      : m_fps(p_fps),
        m_width(p_width),
        m_height(p_height),
        m_time_base_den(p_time_base_den),
        m_stream_url(p_stream_url),
        m_log_level(p_log_level),
        dst_fps({p_fps, 1}) {
    m_npixels_vid = m_width * m_height;
    m_npixels_grid = m_grid_size * m_grid_size;

    m_raw_frame = std::make_unique<float[]>(m_npixels_grid);
    m_frame_buf = std::make_unique<uint8_t[]>(m_npixels_grid);

    for (int i = 0; i < m_npixels_grid; ++i) {
      m_raw_frame.get()[i] = 0;
    }
    VLOG(3) << p_fps << p_width << p_height << p_stream_url << p_time_base_den;
    VLOG(3) << "1";
    CheckError(InitOutputContext());
    VLOG(3) << "2";
    CheckError(InitVideoCodecContext());
    VLOG(3) << "3";
    CheckError(InitVideoStream());
    VLOG(3) << "4";
    CheckError(InitVideoEncoder());
    VLOG(3) << "5";
    CheckError(InitVideoFrameBuf());
    VLOG(3) << "6";
    CheckError(InitOutput());
    VLOG(3) << "7";
    CheckError(InitVideoPkt());
    VLOG(3) << "8";

    av_log_set_level(m_log_level);
    if (m_log_level != AV_LOG_QUIET) {
      av_dump_format(outputContext, 0, m_stream_url.c_str(), 1);
    }
  }
  void StreamImage();
  void CopyToFrameBuf(float *p_data_ptr, int p_chan_no) {
    int offset = (p_chan_no - 1) * m_npixels_grid * NSTOKES;

    for (int i = 0; i < m_npixels_grid; ++i) {
      m_raw_frame.get()[i] = p_data_ptr[i * NSTOKES + offset];  // this is XX*
    }
    auto max_val = *std::max_element(m_raw_frame.get(),
                                     m_raw_frame.get() + m_npixels_grid);

    // normalize and write to the frame buffer
    // FFT shift and transpose
    for (int i = 0; i < m_grid_size; i++) {
      int ii = (i + m_grid_size / 2) % m_grid_size;
      for (int j = 0; j < m_grid_size; j++) {
        int jj = (j + m_grid_size / 2) % m_grid_size;
        m_frame_buf[ii * m_grid_size + jj] =
            m_raw_frame.get()[j * m_grid_size + i] * 255 / max_val;
      }
    }

    int pad = 4;
    for (int i = 0; i < m_grid_size; ++i) {
      for (int j = 0; j < m_grid_size; ++j) {
        if (((i < pad || i >= (m_grid_size - pad))) ||
            ((j < pad || j >= (m_grid_size - pad)))) {
          m_frame_buf.get()[i * m_grid_size + j] = 0;
        }
      }
    }
    max_val = *std::max_element(m_frame_buf.get(),
                                m_frame_buf.get() + m_npixels_grid);
    for (int i = 0; i < m_grid_size; ++i) {
      for (int j = 0; j < m_grid_size; ++j) {
        m_frame_buf.get()[i * m_grid_size + j] =
            m_frame_buf.get()[i * m_grid_size + j] * (255.f) / max_val;
      }
    }
    // for (int i = 0; i < m_npixels; ++i) {
    //   frame->data[1][i] = 128;
    //   frame->data[2][i] = 128;
    // }
    // std::transform(m_raw_frame.get(), m_raw_frame.get() + m_npixels,
    //                m_frame_buf.get(),
    //                [=](float d) { return (d * 255) / max_val; });
  }
  void Stream(int64_t chan0, double cfreq, float *data_ptr) {
    // ResetPipe(chan0, cfreq);
    // LOG(INFO)<<"streaming";
    CopyToFrameBuf(data_ptr, m_chan_stream_no);
    StreamImage();

    // fwrite(m_frame_buf.get(), 1, m_npixels, m_pipeout);
  }
};

// Status_t InitOutputContext();
// Status_t InitVideoStream();
// Status_t InitVideoCodecContext();
// Status_t InitVideoEncoder();
// Status_t InitVideoFrameBuf();
// Status_t InitOutput();
// Status_t InitVideoPkt();
void Streamer::CheckError(const Status_t &p_status) {
  if (p_status.value_or("none") != "none") {
    LOG(FATAL) << p_status.value();
  }
}
Streamer::Status_t Streamer::InitOutputContext() {
  if (avformat_alloc_output_context2(&outputContext, nullptr, "flv",
                                     m_stream_url.c_str()) < 0) {
    Status_t{"Error: Could not allocate output context"};
  }
  return {};
}

Streamer::Status_t Streamer::InitVideoCodecContext() {
  videoCodec = avcodec_find_encoder(AV_CODEC_ID_H264);
  codecContext = avcodec_alloc_context3(videoCodec);
  codecContext->codec_id = AV_CODEC_ID_H264;
  codecContext->codec_type = AVMEDIA_TYPE_VIDEO;
  codecContext->width = m_width;
  codecContext->height = m_height;
  codecContext->gop_size = 12;
  codecContext->framerate = dst_fps;
  codecContext->bit_rate = 4500e3;
  codecContext->time_base = {1, m_time_base_den};  // av_inv_q(dst_fps);
  codecContext->pix_fmt = AV_PIX_FMT_YUV420P;
  codecContext->thread_count = 1;
  if (outputContext->oformat->flags & AVFMT_GLOBALHEADER) {
    codecContext->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }

  m_sws_context = sws_getContext(m_grid_size, m_grid_size, AV_PIX_FMT_GRAY8,
                                 m_width, m_height, AV_PIX_FMT_YUV420P,
                                 SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
  return {};
}

Streamer::Status_t Streamer::InitVideoStream() {
  videoStream = avformat_new_stream(outputContext, videoCodec);
  if (!videoStream) {
    return Status_t{"Error: Could not create video stream"};
  }
  avcodec_parameters_from_context(videoStream->codecpar, codecContext);
  return {};
}

Streamer::Status_t Streamer::InitVideoEncoder() {
  av_dict_set(&codec_options, "profile", "high444", 0);
  av_dict_set(&codec_options, "preset", "ultrafast", 0);
  av_dict_set(&codec_options, "tune", "zerolatency", 0);

  videoCodec = avcodec_find_encoder(codecContext->codec_id);
  if (!videoCodec) {
    return Status_t{"Error: Could not find video codec"};
  }

  if (avcodec_open2(codecContext, videoCodec, &codec_options) < 0) {
    return Status_t{"Error: Could not open video codec\n"};
  }

  av_dict_free(&codec_options);
  videoStream->codecpar->extradata = codecContext->extradata;
  videoStream->codecpar->extradata_size = codecContext->extradata_size;
  return {};
}

Streamer::Status_t Streamer::InitVideoFrameBuf() {
  frame = av_frame_alloc();
  if (!frame) {
    return Status_t{"Error: Could not allocate video frame\n"};
  }
  m_vid_frame_size =
      av_image_get_buffer_size(codecContext->pix_fmt, m_width, m_height, 1);
  framebuf = static_cast<uint8_t *>(av_malloc(m_vid_frame_size));
  av_image_fill_arrays(frame->data, frame->linesize, framebuf,
                       codecContext->pix_fmt, m_width, m_height, 1);
  frame->width = codecContext->width;
  frame->height = codecContext->height;
  frame->format = codecContext->pix_fmt;

  if (av_frame_get_buffer(frame, 0) < 0) {
    return Status_t{"Error: Could not allocate buffer for video frame\n"};
  }
  // for (int i = 0; i < m_npixels / 4; ++i) {
  //   frame->data[1][i] = 128;
  //   frame->data[2][i] = 128;
  // }

  return {};
}

Streamer::Status_t Streamer::InitOutput() {
  // Open RTMP output
  VLOG(3) << "open";
  if (avio_open(&outputContext->pb, outputContext->url, AVIO_FLAG_WRITE) < 0) {
    return Status_t{"Error: Could not open RTMP output"};
  }

  // Write header
  VLOG(3) << "header";
  if (avformat_write_header(outputContext, nullptr) < 0) {
    return Status_t{"Error: Could not write header to RTMP stream"};
  }

  return {};
}

Streamer::Status_t Streamer::InitVideoPkt() {
  pkt = av_packet_alloc();
  return {};
}

void Streamer::StreamImage() {
  // for (int y = 0; y < codecContext->height; ++y) {
  //   for (int x = 0; x < codecContext->width; ++x) {
  //     // Fill Y plane with a gradient
  //     frame->data[0][y * frame->linesize[0] + x] = (x + y + i * 3);

  //     frame->data[1][y / 2 * frame->linesize[1] + x / 2] = 128;
  //     frame->data[2][y / 2 * frame->linesize[2] + x / 2] = 128;
  //   }
  // }
  // frame->pts += 1;
  const int stride[] = {static_cast<int>(m_grid_size)};
  auto *_frame_ptr = m_frame_buf.get();
  sws_scale(m_sws_context, &(_frame_ptr), stride, 0, m_grid_size,
            frame->data, frame->linesize);
  VLOG(3) << "sending frame to encoder";
  if (avcodec_send_frame(codecContext, frame) < 0) {
    CheckError("Error: Failed to send frame for encoding");
  }
  VLOG(3) << "receiving";
  while (avcodec_receive_packet(codecContext, pkt) == 0) {
    // Set PTS and DTS (decoding timestamp) for the packet
    pkt->pts = pkt->dts = _frame_counter * codecContext->time_base.den / m_fps;
    // Write packet to output
    av_packet_rescale_ts(pkt, codecContext->time_base, videoStream->time_base);
    av_interleaved_write_frame(outputContext, pkt);

    // Free packet data
    av_packet_unref(pkt);
    ++_frame_counter;
  }
  // m_frame_PTS+=av_rescale_q(1, codecContext->time_base, dst_fps);
}

#endif  // SRC_EX_VIDEO_STREAMING_HPP_
