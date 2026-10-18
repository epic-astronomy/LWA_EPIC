// A payload released by assignment (`pld = Payload()`) returns its buffer to
// the pool, like one released by going out of scope. Assignment used to drop
// the reference without unlocking the buffer, so a kernel that released its
// input that way (the database writer, the long-integration writer) drained
// its pool and stopped the imager.
//
//   make payload_release_test && ./payload_release_test   (in the build directory)
#include <cstdio>

#include "../../src/ex/lf_buf_mngr.hpp"

static int failures = 0;
#define EXPECT(cond)                                                  \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::fprintf(stderr, "FAILED line %d: %s\n", __LINE__, #cond); \
      ++failures;                                                     \
    }                                                                 \
  } while (0)

using mngr_t = LFBufMngr<AlignedBuffer<float>>;
using pld_t = Payload<mngr_t::mbuf_t>;

int main() {
  constexpr size_t nbufs = 4;
  mngr_t mngr(nbufs, 16, 2, false);

  // released by assignment, many times the pool size
  for (int i = 0; i < 100; ++i) {
    pld_t pld = mngr.acquire_buf();
    EXPECT(static_cast<bool>(pld));
    pld = pld_t();
    EXPECT(!pld);
  }

  // a buffer still referenced elsewhere stays locked
  pld_t held[nbufs];
  for (auto& h : held) h = mngr.acquire_buf();
  pld_t copy = held[0];
  held[0] = pld_t();
  EXPECT(!static_cast<bool>(mngr.acquire_buf()));  // all four in use
  copy = pld_t();                                   // last reference: unlocked
  pld_t again = mngr.acquire_buf();
  EXPECT(static_cast<bool>(again));

  // assigning one live payload over another unlocks the overwritten buffer
  again = held[1];
  held[1] = pld_t();
  EXPECT(static_cast<bool>(mngr.acquire_buf()));

  std::printf("%s\n", failures ? "FAILED" : "OK");
  return failures ? 1 : 0;
}
