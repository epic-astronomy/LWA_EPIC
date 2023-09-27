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

#ifndef SRC_EX_SOCKETS_H_
#define SRC_EX_SOCKETS_H_

#include <arpa/inet.h>
// #include <mellanox/vma_extra.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <iostream>
#include <string>

#include "./constants.h"

/**
 * @brief Socket class for multicast UDP
 *
 */
class MultiCastUDPSocket {
 protected:
  /// Socket file descriptor
  int m_sockfd;
  sockaddr_in m_sockaddr;
  int DEFAULT_SOCK_BUF_SIZE{256 * 1024 * 1024};
  size_t DEFAULT_TIMEOUT{MAX_TIMEOUT};
  ip_mreq m_mreq;
  /// IP address to bind to
  std::string m_address;
  /// Port number
  int m_port;
  /// Flag if the socket is bound
  bool m_is_bound{false};
  /// @brief Construct the necessary objects for connection
  void bootstrap();

 public:
  /**
   * @brief Construct a new Multi Cast UDP Socket object
   *
   */
  MultiCastUDPSocket();
  /**
   * @brief Construct a new Multi Cast UDP Socket object
   *
   * @param p_address Address/host name of the server
   * @param p_port Port to connect to
   */
  MultiCastUDPSocket(std::string p_address, int p_port);
  ~MultiCastUDPSocket();
  /**
   * @brief Bind socket to the address.
   *
   */
  void bind_socket();
  /**
   * @brief Set the address and port to connect to
   *
   * @param p_address Address/host name of the server
   * @param p_port Port number to connect to
   */
  void set_address(std::string p_address, int p_port);
  /**
   * @brief Get the socket file descriptor object
   *
   * @return int
   */
  int get_fd();
};

#endif  // SRC_EX_SOCKETS_H_
