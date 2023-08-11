#ifndef ORM_TYPES
#define ORM_TYPES

#include "hwy/aligned_allocator.h"
#include "tensor.hpp"
#include <algorithm>
#include <string>
#include <vector>
#include "types.hpp"

struct EpicPixelTableMetaRows
{

    constexpr static int NSTOKES{ 4 };
    std::string uuid;
    std::string src_name;
    std::vector<std::pair<int, int>> pixel_coords;
    std::vector<std::pair<int, int>> pixel_coords_sft;
    std::vector<std::pair<float, float>> pixel_lm;
    std::vector<std::pair<float, float>> pixel_skypos;
    std::vector<std::pair<int, int>> pixel_offst;

    size_t m_ncoords;
    size_t meta_version{ -1 };

    EpicPixelTableMetaRows(int ncoords)
    {
        pixel_coords.reserve(ncoords);
        pixel_coords_sft.reserve(ncoords);
        pixel_lm.reserve(ncoords);
        pixel_skypos.reserve(ncoords);
        pixel_offst.reserve(ncoords);

        m_ncoords = ncoords;
    }

    EpicPixelTableMetaRows(){}

    void transform_pix_coords(int xdim, int ydim)
    {
        for (int i = 0; i < m_ncoords; ++i) {
            int x = pixel_coords[i].first;
            //The array y-index starts from the top while the image from the bottom
            int y = ydim -1 - pixel_coords[i].second;

            // transpose
            std::swap(x, y);

            // circshift
            x = (x + xdim / 2) % xdim;
            y = (y + ydim / 2) % ydim;
            pixel_coords_sft.push_back(std::pair<int, int>(x, y));
        }
    }

    bool diff(const EpicPixelTableMetaRows& rhs)
    {
        return (m_ncoords != rhs.m_ncoords) ||
               (pixel_coords != rhs.pixel_coords) ||
               (pixel_lm != rhs.pixel_lm) ||
               (pixel_skypos != rhs.pixel_skypos) ||
               (pixel_offst != rhs.pixel_offst);
    }
};

template<typename _Dtype>
struct EpicPixelTableDataRows : EpicPixelTableMetaRows
{
    struct _config
    {
        int ncoords;
        int nchan;
        bool check_opts()
        {
            return ncoords < 0 ? false : true;
            return nchan < 0 ? false : true;
        }
    };

    using config_t = _config;
    size_t m_nchan{ 32 };
    dict_t m_img_metadata;
    std::string m_uuid;

    hwy::AlignedFreeUniquePtr<_Dtype[]> pixel_values;
    EpicPixelTableDataRows(config_t config)
      : EpicPixelTableMetaRows(config.ncoords)
    {
        pixel_values = std::move(
          hwy::AllocateAligned<_Dtype>(config.ncoords * NSTOKES * config.nchan));
        m_nchan = config.nchan;
    }

    void copy_meta(const EpicPixelTableMetaRows& meta)
    {
        if (meta_version == meta.meta_version) {
            return;
        }

        if (m_ncoords < meta.m_ncoords) {
            pixel_values.reset();
            pixel_values = std::move(
              hwy::AllocateAligned<_Dtype>(m_ncoords * NSTOKES * m_nchan));
        }
        pixel_coords = meta.pixel_coords;
        pixel_lm = meta.pixel_lm;
        pixel_skypos = meta.pixel_skypos;
        pixel_offst = meta.pixel_offst;

        meta_version = meta.meta_version;
        m_ncoords = m_ncoords;
    }

    void m_reset_buf() {}
};

EpicPixelTableMetaRows
create_dummy_meta(int xdim, int ydim)
{
    EpicPixelTableMetaRows meta(1);
    meta.pixel_coords.push_back(std::pair<int, int>(32, 33));
    meta.meta_version = 42;
    meta.transform_pix_coords(xdim, ydim);

    return meta;
}

#endif /* ORM_TYPES */
