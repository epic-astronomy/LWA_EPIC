#ifndef _SOCKETS_H
#define _SOCKETS_H

#include "constants.h"
#include <arpa/inet.h>
#include <iostream>
#include <mellanox/vma_extra.h>
#include <netdb.h>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <sys/types.h>


/**
 * @brief Socket class for multicast UDP
 * 
 */
class MultiCastUDPSocket
{
  protected:
    /// Socket file descriptor
    int m_sockfd;
    sockaddr_in m_sockaddr;
    int DEFAULT_SOCK_BUF_SIZE{ 256 * 1024 * 1024 };
    size_t DEFAULT_TIMEOUT{ MAX_TIMEOUT };
    ip_mreq m_mreq;
    /// IP address to bind to
    std::string m_address;
    /// Port number
    int m_port;
    /// Flag if the socket is bound
    bool m_is_bound{ false };
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

#endif