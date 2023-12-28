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

//#define _USE_VMA_ 1

#ifndef SRC_EX_PACKET_RECEIVER_HPP_
#define SRC_EX_PACKET_RECEIVER_HPP_

#include <glog/logging.h>
#ifdef _USE_VMA_
#include <mellanox/vma_extra.h>
#endif
#include <memory>

#include "./bf_ibverbs.hpp"
#include "./buffer.hpp"
#include "./constants.h"
#include "./exceptions.hpp"
#include "./formats.h"
#include "./helper_traits.hpp"
#include "./sockets.h"

template <typename Dtype, template <class> class Buffer, class Socket,
          bool zcopy = false>
class GenericPktReceiver : protected Buffer<Dtype>, public Socket {
 public:
  GenericPktReceiver(size_t p_buf_size = MAX_PACKET_SIZE,
                     bool p_page_lock = false)
      : Buffer<Dtype>(p_buf_size) {}
  virtual size_t recv_packet(Dtype*& p_out_buf, int p_buf_offset = 0) = 0;
};

#ifdef _USE_VMA_
template <typename Dtype, template <class> class Buffer, class Socket,
          bool Zcopy = false>
class VMAReceiver : public GenericPktReceiver<Dtype, Buffer, Socket, Zcopy> {
 protected:
  vma_api_t* m_api{nullptr};
  vma_packet_t* m_pkt{nullptr};
  vma_packets_t* m_pkts{nullptr};
  int m_flags;
  void free_pkts();

 public:
  static constexpr int type = VMA;
  explicit VMAReceiver(size_t buf_size = MAX_PACKET_SIZE);
  void init_receiver(int p_offset = 0) {}
  size_t recv_packet(Dtype*& p_out_buf, int p_buf_offset = 0);
};
#endif

template <typename Dtype, template <class> class Buffer, class Socket,
          bool Zcopy = true>
class VerbsReceiver : public GenericPktReceiver<Dtype, Buffer, Socket, Zcopy> {
  static_assert(std::is_same<Dtype, uint8_t>::value,
                "Verbs receiver only supports uint8_t data type");

 protected:
  std::unique_ptr<Verbs> m_verbs{nullptr};
  int m_hdr_offset{0};
  void check_connection();
  int m_pkt_size;

 public:
  static constexpr int type = VERBS;
  explicit VerbsReceiver(size_t buf_size = MAX_PACKET_SIZE);
  void init_receiver(int p_offset = 0);
  size_t recv_packet(Dtype*& p_out_buf, int /**/);
};

///////////////////////////////   DEFINITIONS
///////////////////////////////////////////////////
#ifdef _USE_VMA_
template <typename Dtype, template <class> class Buffer, class Socket,
          bool Zcopy>
VMAReceiver<Dtype, Buffer, Socket, Zcopy>::VMAReceiver(size_t buf_size)
    : GenericPktReceiver<Dtype, Buffer, Socket, Zcopy>(buf_size, true) {
  m_api = vma_get_api();
};

template <typename Dtype, template <class> class Buffer, class Socket,
          bool Zcopy>
size_t VMAReceiver<Dtype, Buffer, Socket, Zcopy>::recv_packet(
    Dtype*& p_out_buf, int p_buf_offset) {
  CHECK(this->m_is_bound)
      << "Failed to fetch data. Did you forget to bind the socket?";
  size_t nbytes;
  int flags = 0;
  Dtype* pkt_buffer = this->m_buffer.get() + p_buf_offset;
  if (!Zcopy) {
    nbytes = ::recvfrom(this->m_sockfd, pkt_buffer, this->m_bufsize, flags,
                        NULL, NULL);
    p_out_buf = pkt_buffer;
    return nbytes;
  }
  free_pkts();
  // flags = 0;
  nbytes = m_api->recvfrom_zcopy(this->m_sockfd, pkt_buffer, this->m_bufsize,
                                 &flags, NULL, NULL);

  if (flags & MSG_VMA_ZCOPY) {
    m_pkt = &(reinterpret_cast<vma_packets_t*>(pkt_buffer))->pkts[0];
    p_out_buf = reinterpret_cast<Dtype*>(m_pkt->iov[0].iov_base);
  } else {
    p_out_buf = pkt_buffer;
  }

  return nbytes;
};

template <typename Dtype, template <class> class Buffer, class Socket,
          bool Zcopy>
void VMAReceiver<Dtype, Buffer, Socket, Zcopy>::free_pkts() {
  if (m_pkt && Zcopy) {
    m_api->free_packets(this->m_sockfd, m_pkt,
                        1);  // ((vma_packets_t*)(buf_start))->n_packet_num);
    m_pkt = 0;
  }
}
#endif

template <typename Dtype, template <class> class Buffer, class Socket,
          bool Zcopy>
VerbsReceiver<Dtype, Buffer, Socket, Zcopy>::VerbsReceiver(size_t p_buf_size)
    : m_pkt_size(nearest_integral_vec_size<Dtype>(p_buf_size)),
      GenericPktReceiver<Dtype, Buffer, Socket, Zcopy>(
          nearest_integral_vec_size<Dtype>(p_buf_size) * BF_VERBS_NPKTBUF *
              BF_VERBS_NQP,
          false) {
  DLOG(INFO) << "p_buf_size: " << p_buf_size;
}

template <typename Dtype, template <class> class Buffer, class Socket,
          bool Zcopy>
void VerbsReceiver<Dtype, Buffer, Socket, Zcopy>::check_connection() {
  // CHECK(this->m_is_bound)<<"Invalid socket. Did you set the address and port?
  // Did you bind?";
  //   if (!this->m_is_bound) {
  //       throw(NotConnected("Invalid socket. Did you set the address and port?
  //       Did you bind?"));
  //   }
}

template <typename Dtype, template <class> class Buffer, class Socket,
          bool Zcopy>
void VerbsReceiver<Dtype, Buffer, Socket, Zcopy>::init_receiver(int p_offset) {
  check_connection();
  m_verbs.reset(new Verbs(this->m_sockfd, this->m_pkt_size,
                          this->m_buffer.get(), p_offset));
}

template <typename Dtype, template <class> class Buffer, class Socket,
          bool Zcopy>
size_t VerbsReceiver<Dtype, Buffer, Socket, Zcopy>::recv_packet(
    Dtype*& p_out_buf, int /*unused*/) {
  // check_connection();
  return m_verbs.get()->recv_packet(p_out_buf);
}

#endif  // SRC_EX_PACKET_RECEIVER_HPP_
