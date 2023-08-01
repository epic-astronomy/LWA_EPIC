#ifndef _BUFFER_H
#define _BUFFER_H

#include "constants.h"
#include "exceptions.hpp"
#include "helper_traits.hpp"
#include "host_helpers.h"
#include "hwy/aligned_allocator.h"
#include "types.hpp"
#include <any>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <glog/logging.h>
#include <map>
#include <sys/mman.h>
#include <thread>
#include <unordered_map>
#include <variant>
#include <type_traits>

/**
 * @brief Mixin class to hold the metadata for buffers
 *
 * @tparam Num Unused.
 */
template<typename Num = double>
class BufMetaData
{
    using var_t = std::variant<int64_t, uint64_t, uint8_t, uint16_t, double, float, std::string>;
    using varmap_t = dict_t; // std::map<std::string, var_t>;

  protected:
    std::map<std::string, Num> m_meta_num;
    std::map<std::string, std::string> m_meta_str;
    varmap_t m_metadata;

  public:
    /**
     * @brief Get a numeric value from the map
     * @deprecated Use `get_metadataref` to directly fetch/store metadata
     * @param key Name of the key
     * @return Num
     */
    Num get_meta_num(std::string key);
    /**
     * @brief Get a pointer to the metadata object
     * @deprecated Use `get_metadataref`
     * @return varmap_t*
     */
    varmap_t* get_metadata();
    /**
     * @brief Get a reference to the metadata object
     *
     * @return varmap_t&
     */
    varmap_t& get_metadataref()
    {
        return m_metadata;
    }
    /**
     * @brief Get the meta str object
     * @deprecated Use `get_metadataref`
     * @param key
     * @return std::string
     */
    std::string get_meta_str(std::string key);
    /**
     * @brief Set the meta num object
     * @deprecated Use `get_metadataref`
     * @param key
     * @param value
     */
    void set_meta_num(std::string key, Num value);
    /**
     * @brief Set the meta str object
     * @deprecated Use `get_metadataref`
     *
     * @param key
     * @param value
     */
    void set_meta_str(std::string key, std::string value);
};

/**
 * @brief A generic class for buffer types
 *
 * @tparam Buffer Buffer type
 */
template<typename Buffer>
class GenericBuffer : public BufMetaData<double>
{
    static_assert(is_unique_ptr<Buffer>::value, "Buffer must be a std::unique_ptr");
    using elem_t = typename std::pointer_traits<Buffer>::element_type;

  protected:
    /// Main buffer object
    Buffer m_buffer{ nullptr };
    /// Size of the buffer in number of elements
    size_t m_bufsize;
    /// Flag if the buffer memory is allocated
    bool m_is_allocated{ false };
    /// @brief Reset the buffer metadata and memset all the elements to 0
    void m_reset_buf();

  public:
    GenericBuffer(){};
    /**
     * @brief Construct a new Generic Buffer object
     *
     * @param p_buf_size Number of elements in the buffer
     */
    GenericBuffer(size_t p_buf_size);
    /** 
     * @brief Allocator for the buffer. This function must be implemented by all the derived
     * classes
     *
     * @param buf_size Number of elements in the buffer
     * @param p_reallocate Flag to indicate whether to release the old memory and reallocate
     * @param p_page_lock Whether to page lock the allocated buffer. cuHostRegister will be used
     */
    virtual void allocate(size_t buf_size, bool p_reallocate = false, bool p_page_lock = true) = 0;
    /**
     * @brief Get the buffer size
     *
     * @return  Number of elements in the buffer
     */
    size_t buf_size() { return m_bufsize; };
    /**
     * @brief Get the buffer
     *
     * @return Pointer to the buffer
     */
    elem_t* get_data_ptr();
    // double get_meta_num(std::string key);
    // std::string get_meta_str(std::string key);
    // void set_meta_num(std::string key, double value);
    // void set_meta_str(std::string key, std::string value);
};

/**
 * @brief Aligned buffer type.
 *
 * The allocated buffer is guaranteed to be aligned to the maximum available vector size.
 *
 * @tparam dtype Type of data. Must be one a POD type
 */
template<typename dtype>
class AlignedBuffer : public GenericBuffer<hwy::AlignedFreeUniquePtr<dtype[]>>
{
  protected:
    /// Flag if the memory is page locked
    bool m_page_lock;

  public:
    AlignedBuffer(){};
    AlignedBuffer(size_t p_buf_size, bool p_page_lock = true);
    /// <inheritdoc />
    void allocate(size_t buf_size, bool p_reallocate = false, bool page_lock = true);
    ~AlignedBuffer();
};

/**
 * @brief Unaligned buffer type.
 *
 * @tparam dtype
 */
template<typename dtype>
class UnalignedBuffer : protected GenericBuffer<std::unique_ptr<dtype[], void (*)(void*)>>
{
  public:
    UnalignedBuffer(){};
    UnalignedBuffer(size_t p_buf_size);
    /// <inheritdoc />
    void allocate(size_t buf_size, bool p_reallocate = false);
};

/**
 * @brief Payload object that can be passed around kernels
 *
 * @tparam MBuf Type of the managed buffer
 */
template<typename MBuf>
struct Payload
{
  private:
    /// Shared pointer to the managed buffer
    std::shared_ptr<MBuf> m_mbuf{ nullptr };
    /// @brief Unlock and return the buffer to the pool
    /// @param p_reset Flag to indicate a data reset. Only resets the metadata for now.
    void unlock(bool p_reset = false);
    /// Declared to ensure payload can unlock the buffer
    friend MBuf; // obnoxious! There is a better way, but works for now

  public:
    /**
     * @brief Construct a new Payload object
     *
     * @param p_mbuf shared pointer to the managed buffer
     */
    Payload(std::shared_ptr<MBuf> p_mbuf);
    Payload()
    {
        VLOG(1) << "Creating an empty payload";
    };
    Payload(const Payload& p_pld)
    {
        VLOG(1) << "Copying payload " << m_mbuf.use_count();
        VLOG_IF(2, m_mbuf.get() != nullptr) << " with ID " << m_mbuf.get()->m_id;
        this->m_mbuf = p_pld.m_mbuf;
        VLOG(2) << "Copied payload " << m_mbuf.use_count();
        VLOG_IF(2, m_mbuf.get() != nullptr) << " with ID " << m_mbuf.get()->m_id;
    }
    ~Payload();
    // void mbuf_shared_count() const{
    //     DLOG(INFO)<<"MBuf shared count: "<<m_mbuf.use_count();
    // }
    /**
     * @brief Get the managed buffer in this payload
     *
     * @return MBuf* Pointer to the shared buffer
     */
    MBuf* get_mbuf();
    /**
     * @brief Check if the buffer is valid
     *
     * @return bool
     */
    operator bool() const { return bool(m_mbuf); }
};

/**
 * @brief A wrapper to manage the buffer
 *
 * @tparam Buffer
 */
template<typename Buffer>
struct ManagedBuf : public Buffer
{
  private:
    // friend Payload<ManagedBuf<Buffer>>;
    std::atomic<uint8_t> m_lock{ 0 };
    /// <inheritdoc />
    friend void Payload<ManagedBuf<Buffer>>::unlock(bool p_reset);

    using mbuf_t = ManagedBuf<Buffer>;
    using mbuf_sptr_t = std::shared_ptr<mbuf_t>;

  public:
    const size_t m_id;
    /**
     * @brief Construct a new managed buffer
     *
     * @param p_size Buffer size
     * @param p_page_lock Flag to indicate if the buffer has to be page locked
     */
    ManagedBuf(size_t p_size, bool p_page_lock = true, size_t p_id = 0)
      : Buffer(p_size, p_page_lock)
      , m_id(p_id){};

    /**
     * @brief Construct a new Managed Buf object using a config object
     * defined in the buffer class
     * 
     * @tparam _t Buffer
     * @param p_config Config object 
     * @param p_id ID for the buffer
     * 
     * @relates LFBuffer
     */
    template<typename _t = Buffer,
             std::enable_if_t<std::is_class_v<typename _t::config_t>, bool> = false>
    ManagedBuf(typename _t::config_t p_config, size_t p_id=0)
        : Buffer(p_config)
        , m_id(p_id){};
    // void unlock();
    /**
     * @brief Attempt to acquire a lock on the buffer
     *
     * @return True if successful else false
     */
    bool lock();
    // ~ManagedBuf(){
    //     DLOG(INFO)<<"D ManagedBuf";
    // };
};

template<typename Buffer>
typename GenericBuffer<Buffer>::elem_t*
GenericBuffer<Buffer>::get_data_ptr()
{
    return this->m_buffer.get();
}

template<typename Buffer>
void
GenericBuffer<Buffer>::m_reset_buf()
{
    this->m_meta_num.clear();
    this->m_meta_str.clear();
    this->m_metadata.clear();
    if (m_is_allocated) {
        // TODO: The current compiler on EPIC does not generate
        //  movntdq instructions for this step.
        //  If this needs to be further optimized, replace the fill function
        // with vector instructions
        //  When the size is a multiple of the vector size
        //  we should be able to generate movntdq instructions
        // thereby avoiding cache pollution
        // Start by using memset. See if it produces movntdq by default
        // std::fill(m_buffer.get(), m_buffer.get() + m_bufsize, 0);
        // memset(m_buffer.get(),0, m_bufsize*sizeof(elem_t));
    }
}

template<typename Num>
Num
BufMetaData<Num>::get_meta_num(std::string key)
{
    auto value = m_meta_num.find(key);
    if (value == m_meta_num.end()) {
        throw(InvalidKey(key));
    }
    return m_meta_num[key];
}

template<typename Num>
typename BufMetaData<Num>::varmap_t*
BufMetaData<Num>::get_metadata()
{
    return &m_metadata;
}

template<typename Num>
std::string
BufMetaData<Num>::get_meta_str(std::string key)
{
    auto value = m_meta_str.find(key);
    if (value == m_meta_str.end()) {
        throw(InvalidKey(key));
    }
    return m_meta_str[key];
}

template<typename Num>
void
BufMetaData<Num>::set_meta_num(std::string key, Num value)
{
    m_meta_num[key] = value;
}

template<typename Num>
void
BufMetaData<Num>::set_meta_str(std::string key, std::string value)
{
    m_meta_str[key] = value;
}

template<typename dtype>
AlignedBuffer<dtype>::AlignedBuffer(size_t p_buf_size, bool p_page_lock)
{
    this->m_bufsize = p_buf_size;
    this->allocate(p_buf_size, false, p_page_lock);
    this->m_is_allocated = true;
}

template<typename dtype>
void
AlignedBuffer<dtype>::allocate(size_t p_buf_size, bool p_reallocate, bool p_page_lock)
{
    m_page_lock = p_page_lock;
    if (this->m_is_allocated && !p_reallocate) {
        throw(MemoryAllocationFailure(p_buf_size, true));
    }
    this->m_bufsize = p_buf_size;
    this->m_is_allocated = true;
    this->m_buffer.reset();
    this->m_buffer = std::move(hwy::AllocateAligned<dtype>(p_buf_size));
    std::fill(this->m_buffer.get(), this->m_buffer.get() + p_buf_size, 0);
    if (!this->m_buffer) {
        throw(MemoryAllocationFailure(p_buf_size));
    }
    if (m_page_lock) {
        auto res = cu_mlock(this->m_buffer.get(), p_buf_size * sizeof(dtype));
        CHECK(res == 0) << "Unable to mlock the buffer";
    }
}

template<typename dtype>
AlignedBuffer<dtype>::~AlignedBuffer()
{
    if (m_page_lock) {
        // DLOG(INFO)<<"D Aligned buffer";
        auto res = cu_munlock(this->m_buffer.get());
        // DLOG(INFO)<<"Unregistering buffer";
        // this->m_buffer.reset();
        // DLOG(INFO)<<"Buffer reset";
        CHECK(res == 0) << "Unable to unregister the buffer";
    }
}

template<typename dtype>
UnalignedBuffer<dtype>::UnalignedBuffer(size_t p_buf_size)
{
    this->m_bufsize = p_buf_size;
    this->allocate(p_buf_size);
    this->m_is_allocated = true;
}

template<typename dtype>
void
UnalignedBuffer<dtype>::allocate(size_t p_buf_size, bool p_reallocate)
{
    if (this->m_is_allocated && !p_reallocate) {
        throw(MemoryAllocationFailure(p_buf_size, true));
    }
    this->m_bufsize = p_buf_size;
    this->is_allocated = true;
    this->m_buffer.reset();
    this->m_buffer = std::move(std::unique_ptr<dtype, void (*)(void*)>((dtype*)(malloc(p_buf_size)), free));
    if (!this->m_buffer) {
        throw(MemoryAllocationFailure(p_buf_size));
    }
}

template<typename Buffer>
bool
ManagedBuf<Buffer>::lock()
{
    uint8_t expected_state = 0, wanted_state = 1;
    return m_lock.compare_exchange_strong(expected_state, wanted_state, std::memory_order_acq_rel);
}

template<typename MBuf>
Payload<MBuf>::Payload(std::shared_ptr<MBuf> p_mbuf)
//   : m_mbuf(p_mbuf)
{
    VLOG(1)<<"C payload";
    if (p_mbuf) {
        VLOG(2) << "Creating a payload with a managed buffer ID " << p_mbuf.get()->m_id;
        m_mbuf = p_mbuf;
    }
};

template<typename Mbuf>
Payload<Mbuf>::~Payload()
{
    // If the buffer is in use, there will be at least two references to it.
    // One with the manager and the rest amongst payloads. When the final
    // reference count drops to two, unlock the buffer upon destruction.
    // This allows the same buffer to be passed around multiple processing blocks
    // without unlocking it.
    // std::cout<<"Mbuf has "<<m_mbuf.use_count()<<" pointers\n";
    VLOG_IF(1, m_mbuf!=nullptr) << "D Payload shared mbuf count: " << m_mbuf.use_count() << " ID: " << m_mbuf.get()->m_id;
    // DLOG(INFO)<<"D Payload";
    if (m_mbuf && m_mbuf.use_count() == 2) {
        VLOG(2) << "unlocking";
        unlock(true);
        VLOG(2) << "Unlocked";
    }
}

template<typename Mbuf>
void
Payload<Mbuf>::unlock(bool p_reset)
{
    m_mbuf.get()->m_lock.store(0, std::memory_order::memory_order_relaxed);
    if (p_reset) {
        m_mbuf.get()->m_reset_buf();
    }
}

template<typename Mbuf>
Mbuf*
Payload<Mbuf>::get_mbuf()
{
    if (m_mbuf) {
        return m_mbuf.get();
    } else {
        return nullptr;
    }
}

#endif