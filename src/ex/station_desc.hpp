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

#ifndef SRC_EX_STATION_DESC_HPP_
#define SRC_EX_STATION_DESC_HPP_
#include <glog/logging.h>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "./constants.h"
#include "./helper_traits.hpp"
#include "./packet_assembler.hpp"

#ifdef _USE_VMA_

int GetChan0Vma(std::string ip, int port) {
  // std::cout<<"receiving\n";
  auto receiver =
      VMAReceiver<uint8_t, AlignedBuffer, MultiCastUDPSocket, REG_COPY>();
  uint8_t* buf;
  receiver.set_address(ip, port);
  receiver.bind_socket();
  int nbytes = 0;
  while (nbytes <= 0) {
    nbytes = receiver.recv_packet(buf);
  }
  // std::cout<<nbytes;
  const chips_hdr_type* pkt_hdr = reinterpret_cast<chips_hdr_type*>(buf);
  int chan0 = ntohs(pkt_hdr->chan0);
  if (chan0 == 0) {
    LOG(WARNING) << "Empty packet received: " << nbytes << " bytes";
  }
  return chan0;
}

#endif

int GetChan0(std::string ip, int port) {
  auto receiver = verbs_receiver_t();
  uint8_t* buf;
  receiver.set_address(ip, port);
  std::cout << "Set address\n";
  receiver.bind_socket();
  std::cout << "Socket bound\n";
  receiver.init_receiver();
  std::cout << "Initializing\n";
  int nbytes = receiver.recv_packet(buf, ChipsOffset_t::value);
  // std::cout<<nbytes;
  const chips_hdr_type* pkt_hdr = reinterpret_cast<chips_hdr_type*>(buf);
  return (ntohs(pkt_hdr->chan0));
}

// 239.168.40.11 4015/4016
// 239.168.40.12 4015/4016
// 239.168.40.13 4015/4016
// 239.168.40.14 4015/4016
// 239.168.40.15 4015/4016
// 239.168.40.16 4015/4016

/**
 * @brief Print all end points and their frequencies
 *
 * @tparam _St Station to describe. Is a STATION enum type
 */
template <STATIONS _St>
void PrintStationEndPoints() {}

/**
 * @brief Print endpoints for the LWA-SV station
 *
 * @tparam
 */
template <>
void PrintStationEndPoints<LWA_SV>() {
  auto address_v = std::vector<std::string>{"239.168.40.11", "239.168.40.12",
                                            "239.168.40.13", "239.168.40.14",
                                            "239.168.40.15", "239.168.40.16"};
  auto port_v = std::vector<int>{4015, 4016};

  for (auto& port : port_v) {
    for (auto& ip : address_v) {
      auto chan0 = GetChan0Vma(ip, port);
      std::cout << ip << ":" << port << " " << std::setprecision(4)
                << chan0 * BANDWIDTH * 1e-6 << " MHz\n";
    }
  }
}


#endif  // SRC_EX_STATION_DESC_HPP_
