#ifndef FORMATS
#define FORMATS
#include <cstdint>

/**
 * @brief CHIPS packet structure
 * 
 */
struct __attribute__((packed)) chips_hdr_type
{
    uint8_t roach;    ///< 1-based ROACH ID
    uint8_t gbe;      ///< (AKA tuning)
    uint8_t nchan;    ///< Number of channels 
    uint8_t nsubband; ///< Number of subbands (11)
    uint8_t subband;  ///< Subband number 0-11
    uint8_t nroach;   ///< Number of ROACH boards (16)
    // Note: Big endian
    uint16_t chan0; ///< First chan in  the packet
    uint64_t seq;   ///< Sequence number. Number of 40 us sequences from the start of the service.
};

#endif // FORMATS