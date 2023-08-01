#include "../ex/buffer.hpp"
#include "../ex/constants.h"
#include "../ex/orm_types.hpp"
#include "../ex/py_funcs.hpp"
#include "../ex/tensor.hpp"
#include "../ex/types.hpp"
#include <chrono>
#include <cmath>
#include <glog/logging.h>
#include <memory>
#include <raft>
#include <raftio>
#include <variant>

template<typename _PldIn, typename _PldOut, class BufferMngr, typename BufConfig>
class PixelExtractor : public raft::kernel
{
  protected:
    EpicPixelTableMetaRows m_pixmeta_rows;
    EpicPixelTableMetaRows _dummy_meta;
    std::unique_ptr<BufferMngr> m_buf_mngr{ nullptr };
    bool is_meta_initialized{ false };
    constexpr static size_t m_nbufs{ 20 };
    constexpr static size_t m_maxiters{ 5 };
    PSTensor<float> m_img_tensor;
    size_t m_xdim;
    size_t m_ydim;
    size_t m_nchan;

  public:
    PixelExtractor(BufConfig p_config, const EpicPixelTableMetaRows& p_init_metarows, size_t p_xdim, size_t p_ydim, size_t p_inchan)
      : m_pixmeta_rows(p_init_metarows)
      , m_img_tensor(p_inchan, p_xdim, p_ydim)
      , m_xdim(p_xdim)
      , m_ydim(p_ydim)
      , m_nchan(p_inchan)
      , raft::kernel()
    {
        input.addPort<EpicPixelTableMetaRows>("meta_pixel_rows");
        input.addPort<_PldIn>("in_img");

        output.addPort<_PldOut>("out_pix_rows");
        output.addPort<_PldIn>("out_img");

        m_buf_mngr.reset();
        m_buf_mngr = std::make_unique<BufferMngr>(m_nbufs, m_maxiters, p_config);
        for (int i = 0; i < m_nbufs; ++i) {
            _PldOut pld = m_buf_mngr.get()->acquire_buf();
            if (!pld) {
                LOG(FATAL)<<"Null payload. Unable to allocate memory for the pixel data buffer";
            }
            pld.get_mbuf()->copy_meta(m_pixmeta_rows);
        }
    }

    virtual raft::kstatus run() override
    {
        // check if there are updates to the pixel meta rows
        while (input["meta_pixel_rows"].size() > 0) {
          LOG(INFO)<<"WHILE LOOP";
            input["meta_pixel_rows"].pop(_dummy_meta);
            if (_dummy_meta.meta_version != -1 && input["meta_pixel_rows"].size() == 0) {
                m_pixmeta_rows = _dummy_meta;
            }
        }

        if (input["in_img"].size() == 0) {
            return raft::proceed;
        }

        _PldOut out_pix_rows = m_buf_mngr.get()->acquire_buf();
        // this takes care of resizing the data and meta vectors
        out_pix_rows.get_mbuf()->copy_meta(m_pixmeta_rows);

        _PldIn in_img;
        input["in_img"].pop(in_img);
        m_img_tensor.assign_data(in_img.get_mbuf()->get_data_ptr());

        // Extract pixels into the output payload
        LOG(INFO)<<"Extracting pixels into the payload";
        LOG(INFO)<<"Ncoords: "<<m_pixmeta_rows.pixel_coords_sft.size()<<" "<<out_pix_rows.get_mbuf()->m_nchan;
        m_img_tensor.extract_pixels(
          m_pixmeta_rows, out_pix_rows.get_mbuf()->pixel_values.get());

        output["out_pix_rows"].push(out_pix_rows);
        output["out_img"].push(in_img);

        return raft::proceed;
    }
};