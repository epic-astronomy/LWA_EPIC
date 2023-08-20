/*
 Copyright (c) 2023 The EPIC++ authors

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 the Software, and to permit persons to whom the Software is furnished to do so,
 subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

// #include "ex/sockets.h"
// #include "ex/exceptions.hpp"
// #include <sstream>
// #include <unistd.h>

// MultiCastUDPSocket::MultiCastUDPSocket()
// {
//     bootstrap();
// };

// MultiCastUDPSocket::MultiCastUDPSocket(std::string p_address, int p_port)
// {
//     bootstrap();
//     set_address(p_address, p_port);
//     bind_socket();
// }

// void
// MultiCastUDPSocket::bootstrap()
// {
//     m_sockfd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
//     // sockaddr_in sockaddr;
//     bzero(&m_sockaddr, sizeof(m_sockaddr));
//     m_sockaddr.sin_family = AF_INET;

//     setsockopt(m_sockfd, SOL_SOCKET, SO_RCVBUF, &DEFAULT_SOCK_BUF_SIZE, sizeof(DEFAULT_SOCK_BUF_SIZE));
//     setsockopt(m_sockfd, SOL_SOCKET, SO_SNDBUF, &DEFAULT_SOCK_BUF_SIZE, sizeof(DEFAULT_SOCK_BUF_SIZE));
//     setsockopt(m_sockfd, SOL_SOCKET, SO_RCVTIMEO, &DEFAULT_TIMEOUT, sizeof(DEFAULT_TIMEOUT));

//     memset(&m_mreq, 0, sizeof(ip_mreq));
// }

// void
// MultiCastUDPSocket::set_address(std::string p_address, int p_port)
// {
//     std::cout<<"set address1\n";
//     m_address = p_address;
//     m_port = p_port;
//     m_sockaddr.sin_port = htons(m_port);
//     m_sockaddr.sin_addr.s_addr = inet_addr(m_address.c_str());
//     std::cout<<"set address2\n";


//     m_mreq.imr_multiaddr.s_addr = inet_addr(m_address.c_str());
//     m_mreq.imr_interface.s_addr = htonl(INADDR_ANY);
//     setsockopt(m_sockfd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &m_mreq, sizeof(ip_mreq));
// };

// void
// MultiCastUDPSocket::bind_socket()
// {
//     auto rc = bind(m_sockfd, (struct sockaddr*)&m_sockaddr, sizeof(sockaddr));
//     if (rc == 0) {
//         m_is_bound = true;
//     } else {

//         throw(BindFailure(
//           string_format("Failed to bind %d : %s", errno, strerror(errno))));
//     }
// };

// int
// MultiCastUDPSocket::get_fd()
// {
//     return m_sockfd;
// }

// MultiCastUDPSocket::~MultiCastUDPSocket()
// {
//     if (m_is_bound) {
//         close(m_sockfd);
//     }
// }

