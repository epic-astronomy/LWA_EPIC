#ifndef LF_BUF_MNGR
#define LF_BUF_MNGR

#include "buffer.hpp"
#include "constants.h"
#include <atomic>
#include <cstring>
#include <memory>
#include <vector>
#include "exceptions.hpp"
#include <glog/logging.h>


/**
 * @brief Lock-free buffer manager.
 * 
 * The manager maintains a pool of buffers and a cursor to the latest position where an unlocked
 * buffer was found. Any request to acquire a buffer starts from the cursor and iterates through all
 * the buffer until one can be locked.
 * 
 * @tparam Buffer Type of the buffer
 */
template<class Buffer>
class LFBufMngr // : Buffer<T, Allocator>
{
  public:
    using mbuf_t = ManagedBuf<Buffer>;
    using mbuf_sptr_t = std::shared_ptr<mbuf_t>;

  private:
  /// vector to hold shared pointer to managed buffers
    std::vector<mbuf_sptr_t> m_buf_vec;
    /// Number of buffers in the manager
    size_t m_nbufs;
    /// Number of elements in each buffer
    size_t m_buf_size;
    /// Maximum number of allowed cycles through all the buffers before a nullptr is returned
    size_t m_max_iters;
    // bool m_is_reserved{ false };
    /// Cursor (0-based) to the latest position where an unlocked buffer was found.
    std::atomic_ullong m_cursor{ 0 };
    /// Flag if the buffers have page locked memories
    bool m_page_lock{true};

  public:
  /**
   * @brief Construct a new lock-free buffer manager object
   * 
   * @param p_nbufs Number of buffers to allocate
   * @param p_buf_size Number of elements in each buffer
   * @param p_max_tries Maximum number cycles through all the buffers before an empty buffer is returned
   * @param p_page_lock Flage to indicate whether to page lock the buffer memory
   */
    LFBufMngr(size_t p_nbufs = 1, size_t p_buf_size = MAX_PACKET_SIZE, size_t p_max_tries = 5, bool p_page_lock=true);
    /**
     * @brief Acquire a managed buffer
     * 
     * @param p_max_tries Unused
     * @return Payload<mbuf_t> 
     */
    Payload<mbuf_t> acquire_buf(size_t p_max_tries = 5);
};



template<typename Buffer>
LFBufMngr<Buffer>::LFBufMngr(size_t p_nbufs, size_t p_buf_size, size_t p_max_tries, bool p_page_lock)
  : m_nbufs(p_nbufs)
  , m_buf_size(p_buf_size)
  , m_max_iters(p_nbufs * p_max_tries)
  , m_page_lock(p_page_lock)
{
    DLOG(INFO)<<"Nbufs in LFBuffer manager: "<<p_nbufs;
    DLOG(INFO)<<"Buf size: "<<p_buf_size;
    CHECK(p_nbufs>0)<<"Number of buffers must be at least one";
    CHECK(p_buf_size>=0)<<"Buffer size must be non-negative";
    
    // allocate space for buffers
    m_buf_vec.reserve(p_nbufs);
    for (size_t i = 0; i < p_nbufs; ++i) {
        m_buf_vec[i] = std::make_shared<mbuf_t>(p_buf_size, p_page_lock);
    }
    DLOG(INFO)<<"Allocated space for the buffer";
}

template<class Buffer>
Payload<typename LFBufMngr<Buffer>::mbuf_t>
LFBufMngr<Buffer>::acquire_buf(size_t p_max_tries)
{
    auto cur_cursor = m_cursor.load(); // postion where buffer was available previously
    auto orig_cursor = cur_cursor;
    size_t counter = 0;
    DLOG(INFO)<<"Initial cursor: "<<cur_cursor;
    while (true) {
        ++counter;
        cur_cursor = (cur_cursor+1)% m_nbufs;

        if (m_buf_vec[cur_cursor].get()->lock()) {
            break;
        }
        if (counter > m_max_iters) {
            //Buffer still full after max_tries
            return mbuf_sptr_t(nullptr);
        }
    }
    // CAS if it was not changed by another thread already
    m_cursor.compare_exchange_strong(orig_cursor, cur_cursor, std::memory_order_acq_rel);
    return Payload<mbuf_t>(m_buf_vec[cur_cursor]);
};

template class LFBufMngr<AlignedBuffer<uint8_t>>;


#endif // LF_BUF_MNGR