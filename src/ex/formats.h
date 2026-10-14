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

#ifndef SRC_EX_FORMATS_H_
#define SRC_EX_FORMATS_H_
#include <cstdint>

/**
 * @brief CHIPS packet structure
 *
 */
struct __attribute__((packed)) chips_hdr_type {
  uint8_t roach;     ///< 1-based ROACH ID
  uint8_t gbe;       ///< (AKA tuning)
  uint8_t nchan;     ///< Number of channels
  uint8_t nsubband;  ///< Number of subbands (11)
  uint8_t subband;   ///< Subband number 0-11
  uint8_t nroach;    ///< Number of ROACH boards (16)
  // Note: Big endian
  uint16_t chan0;  ///< First chan in  the packet
  uint64_t seq;  ///< Sequence number. Number of 40 us sequences from the start
                 ///< of the service.
};

/**
 * @brief NDP (ZCU102) F-Engine packet structure. Same as bifrost's snap2
 * format. Each board sends its 32 inputs for a stream's channels split into
 * blocks of nchan channels.
 *
 */
struct __attribute__((packed)) snap2_hdr_type {
  // Note: Big endian
  uint64_t seq;            ///< Spectrum number since the Unix epoch
  uint32_t sync_time;      ///< Unix sync time
  uint16_t npol;           ///< Number of inputs in this packet (32)
  uint16_t npol_tot;       ///< Total number of inputs (512)
  uint16_t nchan;          ///< Number of channels in this packet
  uint16_t nchan_tot;      ///< Number of channels in this stream
  uint32_t chan_block_id;  ///< Index of this block of channels in the stream
  uint32_t chan0;          ///< First channel in this packet
  uint32_t pol0;           ///< First input in this packet
};

#endif  // SRC_EX_FORMATS_H_
