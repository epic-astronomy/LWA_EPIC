#ifndef CONSTANTS
#define CONSTANTS

#define PI 3.141592653589793238
#define SOL 299792458.0

enum RESULT
{
    SUCCESS = 0,
    FAILURE = -1
};

enum RECEIVERS{
    VMA, VERBS
};

enum CONSTANTS
{
    ZCOPY = true,
    REG_COPY = false,
    MAX_PACKET_SIZE = 9000,
    MAX_TIMEOUT = 2000,
    LOCKED = 1,
    UNLOCKED = 0,
    UNLOCK = 0,
    NROACH_BOARDS = 16,
    SINGLE_SEQ_SIZE = MAX_PACKET_SIZE * NROACH_BOARDS, // this is ~2x the typical size
    GULP_1K_SEQ_SIZE = 1000 * SINGLE_SEQ_SIZE,
    BANDWIDTH = 25000,
    FS = int(196e6),
    SEQ_MULT_S = FS/BANDWIDTH,
    SAMPLING_LEN_uS = 40, // micro-seconds
    NSEQ_PER_SEC = int(1e6 / SAMPLING_LEN_uS),
    LWA_SV_NSTANDS = 256,
    LWA_SV_NPOLS = 2,
    LWA_SV_INP_PER_CHAN = LWA_SV_NSTANDS * LWA_SV_NPOLS,
    ALLOWED_PKT_DROP=50,
    CHIPS_NINPUTS_PER_CHANNEL=32,
    MAX_GULP_STREAMS=8,
    GULP_COUNT_START=1,
    MAX_CHANNELS_PER_F_ENGINE=132,
    MAX_CHANNELS_4090=128,
    MAX_THREADS_PER_BLOCK=1024,
    MAX_ALLOWED_SUPPORT_SIZE=9,
    NSTOKES=4
};

/**
 * @brief Supported packet data arrangements for each gulp
 * 
 */
enum PKT_DATA_ORDER{
    /// Channel major. Gulp dimensions: chan, time, ant, pol, complex
    CHAN_MAJOR, 
    /// Time Major.  Gulp dimensions: time, chan, ant, pol, complex. Defult in bifrost
    TIME_MAJOR,
};

enum IMAGING_POL_MODE{
    SINGLE_POL, DUAL_POL
};

enum IMAGE_SIZE{
    HALF=64, FULL=128
};

/**
 * @brief Parameters for the prolate spheroid GCF
 * 
 */
struct ProSphPars{
    static constexpr int m=0;
    static constexpr int n=0;
    static constexpr float alpha=0.5;
    static constexpr float c=5.356 * PI / 2.0;
};



#endif /* CONSTANTS */
