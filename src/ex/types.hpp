#ifndef TYPES_H
#define TYPES_H

#include "constants.h"
#include <endian.h>
#include <cstdint>
#include <unordered_map>
#include <any>
#include <cstring>

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

/// Python dict-like data structure to describe Meta data
typedef std::unordered_map<std::string, std::any> dict_t;

#endif // TYPES_H