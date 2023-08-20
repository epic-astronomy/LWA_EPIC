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

#ifndef SRC_RAFT_KERNELS_DISK_SAVER_HPP_
#define SRC_RAFT_KERNELS_DISK_SAVER_HPP_
#include <glog/logging.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <raft>
#include <raftio>
#include <string>

#include "../ex/constants.h"
#include "../ex/py_funcs.hpp"
#include "../ex/types.hpp"

template <class Payload>
class DiskSaver_rft : public raft::kernel {
  std::string m_img_suffix;

 public:
  explicit DiskSaver_rft(std::string p_img_suffix = "0") : raft::kernel() {
    input.addPort<Payload>("image");

    m_img_suffix = p_img_suffix;
  }

  raft::kstatus run() override {
    using std::string_literals::operator""s;
    Payload pld;
    input["image"].pop(pld);

    if (!pld) {
      LOG(WARNING) << "Empty image received.";
      return raft::proceed;
    }

    auto& img_metadata = pld.get_mbuf()->get_metadataref();
    for (auto it = img_metadata.begin(); it != img_metadata.end(); ++it) {
      VLOG(3) << it->first << std::endl;
    }
    auto imsize = std::get<int>(img_metadata["grid_size"]);
    auto nchan = std::get<uint8_t>(img_metadata["nchan"]);
    save_image(imsize, nchan, pld.get_mbuf()->get_data_ptr(),
               "test_image_"s + m_img_suffix, img_metadata);

    return raft::proceed;
  }
};

#endif  // SRC_RAFT_KERNELS_DISK_SAVER_HPP_
