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

#ifndef SRC_EX_CONSTANTS_H_
#define SRC_EX_CONSTANTS_H_

#define PI 3.141592653589793238
#define SOL 299792458.0

enum RESULT { SUCCESS = 0, FAILURE = -1 };

enum RECEIVERS { VMA, VERBS };

enum CONSTANTS {
  ZCOPY = true,
  REG_COPY = false,
  MAX_PACKET_SIZE = 9000,
  MAX_TIMEOUT = 2000,
  LOCKED = 1,
  UNLOCKED = 0,
  UNLOCK = 0,
  NROACH_BOARDS = 16,
  // this is ~2x the typical size
  SINGLE_SEQ_SIZE = MAX_PACKET_SIZE * NROACH_BOARDS,
  GULP_1K_SEQ_SIZE = 1000 * SINGLE_SEQ_SIZE,
  BANDWIDTH = 25000,
  FS = static_cast<int>(196e6),
  SEQ_MULT_S = FS / BANDWIDTH,
  SAMPLING_LEN_uS = 40,  // micro-seconds
  NSEQ_PER_SEC = static_cast<int>(1e6 / SAMPLING_LEN_uS),
  LWA_SV_NSTANDS = 256,
  LWA_SV_NPOLS = 2,
  LWA_SV_INP_PER_CHAN = LWA_SV_NSTANDS * LWA_SV_NPOLS,
  ALLOWED_PKT_DROP = 50,
  CHIPS_NINPUTS_PER_CHANNEL = 32,
  MAX_GULP_STREAMS = 8,
  GULP_COUNT_START = 1,
  MAX_CHANNELS_PER_F_ENGINE = 132,
  MAX_CHANNELS_4090 = 128,
  MAX_THREADS_PER_BLOCK = 1024,
  MAX_ALLOWED_SUPPORT_SIZE = 9,
  NSTOKES = 4
};

enum STATIONS { LWA_SV = 1 };

/**
 * @brief Supported packet data arrangements for each gulp
 *
 */
enum PKT_DATA_ORDER {
  /// Channel major. Gulp dimensions: chan, time, ant, pol, complex
  CHAN_MAJOR,
  /// Time Major.  Gulp dimensions: time, chan, ant, pol, complex. Defult in
  /// bifrost
  TIME_MAJOR,
};

enum IMAGING_POL_MODE { SINGLE_POL, DUAL_POL };

enum IMAGE_SIZE { HALF = 64, FULL = 128 };

/**
 * @brief Parameters for the prolate spheroid GCF
 *
 */
struct ProSphPars {
  static constexpr int m = 0;
  static constexpr int n = 0;
  static constexpr float alpha = 0.5;
  static constexpr float c = 5.356 * PI / 2.0;
};

#endif  // SRC_EX_CONSTANTS_H_
