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
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/channel_layout.h>
#include <libavutil/dict.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libavutil/log.h>
#include <libavutil/mathematics.h>
#include <libavutil/opt.h>
#include <libavutil/rational.h>
#include <libavutil/samplefmt.h>
#include <libswscale/swscale.h>
}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <set>
#include <thread>

#include "./constants.h"
#include "glog/logging.h"

// https://github.com/joncampbell123/composite-video-simulator/issues/5
#ifdef av_err2str
#undef av_err2str
#include <string>
av_always_inline std::string av_err2string(int errnum) {
    char str[AV_ERROR_MAX_STRING_SIZE];
    return av_make_error_string(str, AV_ERROR_MAX_STRING_SIZE, errnum);
}
#define av_err2str(err) av_err2string(err).c_str()
#endif  // av_err2str


class Streamer {
  using Status_t = std::optional<std::string>;

 private:
  float m_fps;
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
  std::string m_cmap;
  int m_atadenoise_radius{9};
  int m_atadenoise_frame_counter{0};
  int m_tmideq_radius{9};
  int m_tmideq_counter{0};

  std::unique_ptr<float[]> m_frame_buf;
  std::unique_ptr<float[]> m_raw_frame;
  int64_t m_chan0{0};
  int m_chan_stream_no{1};
  int m_grid_size{128};
  int m_vid_size{512};
  // int m_fps{10};
  std::string m_stream_key;
  FILE *m_pipeout;
  bool m_is_pipeopen{false};
  // SwsContext *m_sws_context;

  AVFilterGraph *filterGraph;
  AVFilter *buffersrc;
  AVFilter *buffersink;
  AVFilterContext *buffersrcContext{nullptr};
  AVFilterContext *buffersinkContext{nullptr};
  AVFilterContext *pseudocolorContext{nullptr};
  AVFilterContext *scaleContext{nullptr};
  AVFilterContext *histeqContext{nullptr};
  AVFilterContext *textContext{nullptr};
  AVFilterContext *textContextLogo{nullptr};
  AVFilterContext *geqContext{nullptr};
  AVFilterContext *atadenoiseContext{nullptr};
  AVFilterContext *tmideqContext{nullptr};
  // int m_npixels{128 * 128};

 protected:
  AVFrame *frame{nullptr};
  AVFrame *scaledFrame{nullptr};
  void CheckError(const Status_t &p_status);

  Status_t InitOutputContext();
  Status_t InitVideoStream();
  Status_t InitVideoCodecContext();
  Status_t InitFilterGraph();
  Status_t InitVideoEncoder();
  Status_t InitVideoFrameBuf();
  Status_t InitOutput();
  Status_t InitVideoPkt();
  Status_t ValidateCmap(std::string p_cmap);

 public:
  Streamer(float p_fps = 6.25, int p_vid_width = 512, int p_vid_height = 512,
           int p_grid_size = 128, float p_time_base_den = 6.25,
           std::string p_stream_url = "rtmp://127.0.0.1:1857/live/epic",
           std::string p_cmap = "magma", int p_log_level = AV_LOG_ERROR)
      : m_fps(p_fps),
        m_width(p_vid_width),
        m_height(p_vid_height),
        m_grid_size(p_grid_size),
        m_time_base_den(p_time_base_den),
        m_stream_url(p_stream_url),
        m_log_level(p_log_level),
        dst_fps({int(p_fps * 1000), 1000}) {
    m_npixels_vid = m_width * m_height;
    m_npixels_grid = m_grid_size * m_grid_size;

    ValidateCmap(p_cmap);
    m_cmap = p_cmap;

    m_raw_frame = std::make_unique<float[]>(m_npixels_grid);
    m_frame_buf = std::make_unique<float[]>(m_npixels_grid);

    for (int i = 0; i < m_npixels_grid; ++i) {
      m_raw_frame.get()[i] = 0;
    }

    CheckError(InitOutputContext());
    CheckError(InitVideoCodecContext());
    CheckError(InitVideoStream());
    CheckError(InitFilterGraph());
    CheckError(InitVideoEncoder());
    CheckError(InitVideoFrameBuf());
    CheckError(InitOutput());
    CheckError(InitVideoPkt());

    av_log_set_level(m_log_level);
    if (m_log_level != AV_LOG_QUIET) {
      av_dump_format(outputContext, 0, m_stream_url.c_str(), 1);
    }
  }
  void StreamImage();
  void CopyToFrameBuf(const float *p_data_ptr, int p_chan_no,
                      bool p_is_hc = false) {
    int offset = (p_chan_no - 1) * m_npixels_grid * NSTOKES;

    if (p_is_hc) {
      for (int i = 0; i < m_grid_size; ++i) {
        for (int j = 0; j < m_grid_size; ++j) {
          auto val = m_frame_buf.get()[i * m_grid_size + j];
          frame->data[0][i * m_grid_size + j] = 128;
        }
      }
      return;
    }

    for (int i = 0; i < m_npixels_grid; ++i) {
      // I = XX* + YY*
      m_raw_frame.get()[i] = p_data_ptr[i * NSTOKES + offset] + p_data_ptr[i * NSTOKES + offset + 1];
    }

    // normalize and write to the frame buffer
    // FFT shift and transpose
    for (int i = 0; i < m_grid_size; i++) {
      int ii = (i + m_grid_size / 2) % m_grid_size;
      for (int j = 0; j < m_grid_size; j++) {
        int jj = (j + m_grid_size / 2) % m_grid_size;
        m_frame_buf[ii * m_grid_size + jj] =
            m_raw_frame.get()[j * m_grid_size + i];
      }
    }

    // check for the normalization above 30 degrees
    // radius = grid_size * 0.44 * cos(30 deg)
    //        = grid_size * 0.38
    int xcen = m_grid_size / 2;
    int ycen = xcen;
    float radius2 = (m_grid_size * 0.38) * (m_grid_size * 0.38);
    float min_val = *std::min_element(m_frame_buf.get(), m_frame_buf.get()+m_grid_size*m_grid_size);
    //float max_val_full = *std::max_element(m_frame_buf.get(), m_frame_buf.get()+m_grid_size*m_grid_size);
    float max_val = 0;


    for (int i = 0; i < m_grid_size; ++i) {
      for (int j = 0; j < m_grid_size; ++j) {
        if (((i - xcen) * (i - xcen) + (j - ycen) * (j - ycen)) <= radius2) {
          max_val = std::max(max_val, m_frame_buf.get()[i * m_grid_size + j]);
        }
      }
    }


    //max_val *= 0.8; // try to brighten the fainter features

    auto *frame16 = reinterpret_cast<uint16_t*>(frame->data[0]);
    for (int i = 0; i < m_grid_size; ++i) {
      for (int j = 0; j < m_grid_size; ++j) {
        auto val = m_frame_buf.get()[i * m_grid_size + j];
        if (val > max_val) {
          frame16[i * m_grid_size + j] = max_val;
        } else {
          frame16[i * m_grid_size + j] = (val-min_val) * (1023.f) / (max_val-min_val);
        }
      }
    }
  }

  void StreamEmpty(){
    std::string text = "text='Checking system health | %{gmtime\:%Y/%m/%d %H\\:%M\\:%S}'";
    if (avfilter_graph_send_command(filterGraph, "text", "reinit",
                                      text.c_str(), NULL, 0, 0) < 0) {
        CheckError(Status_t{"unable to dynamically set text: " + text});
    }

    //copy a constant value into the famebuffer
    for (int i = 0; i < m_grid_size; ++i) {
      for (int j = 0; j < m_grid_size; ++j) {
        auto val = m_frame_buf.get()[i * m_grid_size + j];
        frame->data[0][i * m_grid_size + j] = 128;
      }
    }

    StreamImage();
  }

  void Stream(int64_t chan0, double cfreq, const float *data_ptr,
              bool is_hc_freq) {
    if (m_chan0 != chan0) {
      m_chan0 = chan0;
      std::string text = "%{gmtime\:%Y/%m/%d %H\\:%M\\:%S}'";
      char buf[1024];
      std::snprintf(buf, sizeof(buf), "%.2f MHz", cfreq);
      text = "text='" + std::string(buf) + " |  " + text;
      if (is_hc_freq) {
        text = "text='Checking system health...'";
      }
      if (avfilter_graph_send_command(filterGraph, "text", "reinit",
                                      text.c_str(), NULL, 0, 0) < 0) {
        CheckError(Status_t{"unable to dynamically set text: " + text});
      }
    }

    CopyToFrameBuf(data_ptr, m_chan_stream_no, is_hc_freq);
    StreamImage();

  }
};


void Streamer::CheckError(const Status_t &p_status) {
  if (p_status.value_or("none") != "none") {
    LOG(FATAL) << p_status.value();
  }
}

Streamer::Status_t Streamer::ValidateCmap(std::string p_cmap) {
  const auto valid_cmaps = std::set<std::string>{"magma","inferno","plasma","viridis","turbo","cividis","range1","range2","shadows","highlights","solar","nominal","preferred","total","spectral","cool","heat","fiery","blues","green","helix"};

  if(valid_cmaps.count(p_cmap)<1){
    return Status_t{"Invalid cmap specfified. See https://ffmpeg.org/ffmpeg-filters.html#pseudocolor for available ones"};
  }
  return {};
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
  codecContext->time_base = {
      1000, int(m_time_base_den * 1000)};  // av_inv_q(dst_fps);
  codecContext->pix_fmt = AV_PIX_FMT_YUV420P10LE;
  // codecContext->thread_count = 2;
  if (outputContext->oformat->flags & AVFMT_GLOBALHEADER) {
    codecContext->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }

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
  av_dict_set(&codec_options, "preset", "medium", 0);
  av_dict_set(&codec_options, "tune", "film", 0);
  av_dict_set(&codec_options, "crf", "15", 0);
  av_dict_set(&codec_options, "qp", "0", 0);

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

Streamer::Status_t Streamer::InitFilterGraph() {
  filterGraph = avfilter_graph_alloc();
  if (!filterGraph) {
    return Status_t{"Failed to allocate filter graph."};
  }
  const AVFilter *bufsrc = avfilter_get_by_name("buffer");
  const AVFilter *bufsink = avfilter_get_by_name("buffersink");
  const AVFilter *pseudocolor = avfilter_get_by_name("pseudocolor");
  const AVFilter *scaleFilter = avfilter_get_by_name("scale");
  //const AVFilter *histeqFilter = avfilter_get_by_name("hqdn3d");
  const AVFilter *geqFilter = avfilter_get_by_name("geq");
  const AVFilter *textFilter = avfilter_get_by_name("drawtext");
  const AVFilter *atadenoiseFilter = avfilter_get_by_name("atadenoise");
  const AVFilter *tmideqFilter = avfilter_get_by_name("tmidequalizer");

  buffersrcContext = avfilter_graph_alloc_filter(filterGraph, bufsrc, "in");
  buffersinkContext = avfilter_graph_alloc_filter(filterGraph, bufsink, "out");
  pseudocolorContext =
      avfilter_graph_alloc_filter(filterGraph, pseudocolor, "pscolor");
  // histeqContext =
  //     avfilter_graph_alloc_filter(filterGraph, histeqFilter, "histeq");
  scaleContext = avfilter_graph_alloc_filter(filterGraph, scaleFilter, "scale");
  geqContext = avfilter_graph_alloc_filter(filterGraph,geqFilter,"geq");
  textContext = avfilter_graph_alloc_filter(filterGraph, textFilter, "text");
  textContextLogo =
      avfilter_graph_alloc_filter(filterGraph, textFilter, "textlogo");
  atadenoiseContext = avfilter_graph_alloc_filter(filterGraph, atadenoiseFilter, "atadenoiser");
  tmideqContext = avfilter_graph_alloc_filter(filterGraph, tmideqFilter, "tmideq");

  // const AVFilter *lutyuvFilter = avfilter_get_by_name("lutyuv");
  // auto *lutyuvContext = avfilter_graph_alloc_filter(filterGraph, lutyuvFilter, "eq");

  char src_buf[1024];
  std::snprintf(src_buf, sizeof(src_buf),
                "width=%d:height=%d:pix_fmt=yuv420p10le:time_base=1/%d",
                m_grid_size, m_grid_size, m_time_base_den);

  char scale_buf[1024];
  std::snprintf(scale_buf, sizeof(scale_buf),
                "w=%d:h=%d:sws_flags=lanczos:sws_dither=auto:in_range=auto:out_range=limited", m_width,
                m_height);

  char geq_buf[1024];
  // assume the resolution is 1.056 deg at 128
  float sll = 1./(2 * m_grid_size * std::sin( PI * 1.056 * m_grid_size/128.f/360.f));
  int sky_size = m_width * sll;//360/(2*3.14) /(m_grid_size/128 * 1 ) * m_width/m_grid_size;
  int boundary_color = 32+1.5*m_width/2; //start with this value and go to black color at the edge
  std::snprintf(geq_buf,sizeof(geq_buf),"'if(gt(sqrt((X-W/2)^2+(Y-H/2)^2),%d),(%d-sqrt((X-W/2)^2+(Y-H/2)^2)),p(X,Y))'", sky_size,boundary_color);
  
  char ata_buf[1024];
  std::snprintf(ata_buf,sizeof(ata_buf),"s=%d",m_atadenoise_radius);
  if (avfilter_init_str(buffersrcContext, src_buf) < 0 ||
      avfilter_init_str(buffersinkContext, "") < 0 ||
      avfilter_init_str(geqContext, geq_buf) <0 ||
      avfilter_init_str(pseudocolorContext, ("preset="+m_cmap).c_str()) < 0 ||
      avfilter_init_str(scaleContext, scale_buf) < 0 ||
      avfilter_init_str(
          textContext,
          "text='%{gmtime\:%Y/%m/%d %H\\:%M\\:%S}':fontsize=h/"
          "25:font=Times:fontcolor=black:box=1:boxcolor=white@0.7:boxborderw="
          "5:x=(w-1.05*text_w):y=(h-1.4*text_h)") < 0 ||
      avfilter_init_str(
          textContextLogo,
          "text='EPIC TV':fontsize=h/"
          "20:font=Times:fontcolor=black:box=1:boxcolor=white@0.7:boxborderw="
          "10:x=(16):y=(16)") < 0  
      || avfilter_init_str(atadenoiseContext,ata_buf)<0
      || avfilter_init_str(tmideqContext,"")<0
      ) {
    // Handle error
    return Status_t{"Unable to init buffer sink>[..filters]->src context"};
  }

  if (avfilter_link(buffersrcContext, 0, scaleContext, 0) < 0) {
    // Handle error
    return Status_t{"Unable to link src/scale"};
  }

  if (avfilter_link(scaleContext, 0, geqContext, 0) < 0) {
    // Handle error
    return Status_t{"Unable to link scale/geq"};
  }
  if (avfilter_link(geqContext, 0, pseudocolorContext, 0) < 0) {
    // Handle error
    return Status_t{"Unable to link geq/pscolor"};
  }
  if (avfilter_link(pseudocolorContext, 0, textContext, 0) < 0) {
    // Handle error
    return Status_t{"Unable to link pscolor/text"};
  }
  if (avfilter_link(textContext, 0, textContextLogo, 0) < 0) {
    // Handle error
    return Status_t{"Unable to link text/text"};
  }
  if (avfilter_link(textContextLogo, 0, atadenoiseContext, 0) < 0) {
    // Handle error
    return Status_t{"Unable to link text/atadenoise"};
  }
  if (avfilter_link(atadenoiseContext, 0, tmideqContext, 0) < 0) {
    // Handle error
    return Status_t{"Unable to link atadenoise/tmideq"};
  }
  if (avfilter_link(tmideqContext, 0, buffersinkContext, 0) < 0) {
    // Handle error
    return Status_t{"Unable to link tmideq/sink"};
  }

  // Configure the filter graph
  if (avfilter_graph_config(filterGraph, nullptr) < 0) {
    return Status_t{"Failed to configure filter graph."};
  }
  return {};
}

Streamer::Status_t Streamer::InitVideoFrameBuf() {
  frame = av_frame_alloc();
  scaledFrame = av_frame_alloc();
  if (!frame) {
    return Status_t{"Error: Could not allocate video frame\n"};
  }
  m_vid_frame_size = av_image_get_buffer_size(codecContext->pix_fmt,
                                              m_grid_size, m_grid_size, 1);
  framebuf = static_cast<uint8_t *>(av_malloc(m_vid_frame_size));
  av_image_fill_arrays(frame->data, frame->linesize, framebuf,
                       codecContext->pix_fmt, m_grid_size, m_grid_size, 1);
  frame->width = m_grid_size;
  frame->height = m_grid_size;
  frame->format = codecContext->pix_fmt;

  if (av_frame_get_buffer(frame, 0) < 0) {
    return Status_t{"Error: Could not allocate buffer for video frame\n"};
  }

  return {};
}

Streamer::Status_t Streamer::InitOutput() {
  // Open RTMP output
  if (avio_open(&outputContext->pb, outputContext->url, AVIO_FLAG_WRITE) < 0) {
    return Status_t{"Error: Could not open RTMP output"};
  }

  // Write header
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
  int ret;
  ret = av_buffersrc_write_frame(buffersrcContext, frame);
  if(ret<0){
    fprintf(stderr, "Error writing frame: %s\n", av_err2str(ret));
    CheckError("Error: Failed write frame");
  }
  // The atadenoise and tmidequalizer each have a radius.
  // So they need to consume a few frames before they start
  // to create output frames. So ignore the -11 return code
  // up to the sum of radii number of frames.
  ret = av_buffersink_get_frame(buffersinkContext, scaledFrame);
  if(ret==-11){ // Resource temporarily unavailable error
    if(m_atadenoise_frame_counter>=(m_atadenoise_radius+m_tmideq_radius)){
      CheckError("Error: Failed to get frame");
    }
    m_atadenoise_frame_counter++;
    return;
  }
  if(ret<0){
    fprintf(stderr, "Error getting frame: %s\n", av_err2str(ret));
    CheckError("Error: Failed to get frame");
  }
  ret=avcodec_send_frame(codecContext, scaledFrame);
  if ( ret< 0) {
    fprintf(stderr, "Error getting frame from buffersink: %s\n", av_err2str(ret));
    CheckError("Error: Failed to send frame for encoding");
  }

  av_frame_unref(scaledFrame);

  while (avcodec_receive_packet(codecContext, pkt) == 0) {
    // Set PTS and DTS (decoding timestamp) for the packet
    pkt->pts = pkt->dts = _frame_counter * m_time_base_den / m_fps;
    // Write packet to output
    av_packet_rescale_ts(pkt, codecContext->time_base, videoStream->time_base);
    av_interleaved_write_frame(outputContext, pkt);

    // Free packet data
    // av_frame_unref(scaledFrame);
    // av_frame_unref(frame);
    av_free_packet(pkt);
    av_packet_unref(pkt);
    // av_free(pkt);
    ++_frame_counter;
  }
}

#endif  // SRC_EX_VIDEO_STREAMING_HPP_