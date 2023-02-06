#ifndef TYPES_H
#define TYPES_H

#include "constants.h"
#include <endian.h>
#include <cstdint>

/**
 * @brief Complex nibble. Independent of the host's endianness.
 *
 */
struct __attribute__((aligned(1))) cnib
{
#if __BYTE_ORDER == __BIG_ENDIAN
    signed char im : 4, re : 4;
#elif __BYTE_ORDER == __LITTLE_ENDIAN
    signed char re : 4, im : 4;
#else
    static_assert(false, "Unkonwn endianness. Alien!");
#endif
};

/**
 * @brief Complex nibble vector with two members, X and Y,
 * one for each polarization.
 *
 * @relatesalso MOFFCuHandler
 */
struct __attribute__((aligned(2))) cnib2
{
    cnib X, Y;
};

// template<typename T>
// using const_restrict_q = const __restrict__ T;

// template<typename T>
// using no_q = T;

// template<PKT_DATA_ORDER Order = CHAN_MAJOR, template<typename> typename Qualifier = const_restrict_q>
// struct ImagingData
// {
//     uint8_t* m_f_eng;
//     Qualifier<float*> m_ant_pos;
//     Qualifier<float*> m_phases;
//     size_t m_ngulps;
//     // size_t m_grid_size;
//     size_t m_nchan;
//     size_t m_nstands;
//     size_t m_npols;

//   public:
//     // all sizes in elements

//     // number of elements in the ant pos vector for a given freq/chan
//     const size_t m_ant_pos_seq_size;
//     // number of elements in the f-eng sample for a given freq/chan and time (both pols)
//     const size_t m_f_eng_sample_size;
//     // number of elements in the phases vector for a given freq/chan
//     const size_t m_phases_seq_size;
//     // dimension along the time axis
//     const size_t m_f_eng_time_size;
//     // dimension along the freq/chan axis
//     const size_t m_f_eng_freq_size;

//     ImagingData(uint8_t* p_f_eng, float* p_ant_pos, float* p_phases, size_t p_ngulps, size_t p_nchan, size_t p_nstands, size_t p_npols = 2)
//       : m_f_eng(p_f_eng)
//       , m_ant_pos(p_ant_pos)
//       , m_phases(p_phases)
//       , m_ngulps(p_ngulps)
//       //   , m_grid_size(p_grid_size)
//       , m_nchan(p_nchan)
//       , m_nstands(p_nstands)
//       , m_npols(p_npols)
//       , m_ant_pos_seq_size(m_nstands * 3 /*dimensions*/)
//       , m_phases_seq_size(m_nstands * m_npols * 2 /*real_imag*/)
//       , m_f_eng_sample_size(m_nstands * m_npols)
//       , m_f_eng_time_size([&](PKT_DATA_ORDER _order, size_t _f_eng_sample_size, size_t _nchan) {
//           if (_order == CHAN_MAJOR)
//               return _f_eng_sample_size;
//           else if (_order == TIME_MAJOR)
//               return _f_eng_sample_size * _nchan;
//           else
//               return size_t(0);
//       }(Order, m_f_eng_sample_size, m_nchan))
//       , m_f_eng_freq_size([&](PKT_DATA_ORDER _order, size_t _f_eng_sample_size, size_t _nchan) {
//           if (_order == CHAN_MAJOR)
//               return _f_eng_sample_size * _nchan;
//           else if (_order == TIME_MAJOR)
//               return _f_eng_sample_size;
//           else
//               return size_t(0);
//       }(Order, m_f_eng_sample_size, m_nchan)){
//           // m_ant_pos_seq_size = m_nstands * 3 /*dimensions*/;
//           // m_phases_seq_size = m_nstands * m_npols * 2 /*real_imag*/;

//           // if (Order == CHAN_MAJOR) {
//           //     m_f_eng_time_size = m_grid_size * m_grid_size * m_npol;
//           //     m_f_eng_freq_size = m_grid_size * m_grid_size * m_nchan * m_npol;
//           // } else {
//           //     m_f_eng_time_size = m_grid_size * m_grid_size * m_nchan * m_npol;
//           //     m_f_eng_freq_size = m_grid_size * m_grid_size * m_npol;
//           // }
//       };
    
//     __device__ uint8_t* get_f_eng(size_t gulp, size_t chan_idx)
//     {
//         return m_f_eng + m_f_eng_time_size * gulp + m_f_eng_freq_size * m_nchan;
//     };

//     __device__ float* get_ant_pos(size_t chan_idx)
//     {
//         return m_ant_pos + chan_idx * m_ant_pos_seq_size;
//     };

//     __device__ float* get_phases(size_t chan_idx)
//     {
//         return m_phases + chan_idx * m_phases_seq_size;
//     };
// };
#endif // TYPES_H