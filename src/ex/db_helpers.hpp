#include "constants.h"
#include <cstring>
#include <iostream>
#include <pqxx/pqxx>
#include <string_view>
#include <vector>
#include <py_funcs.hpp>


std::string
get_pixel_insert_stmnt_1(pqxx::placeholders<unsigned int>& name)
{
    std::string stmnt = "(";

    stmnt += name.get() + ","; // uuid
    name.next();

    stmnt += name.get() + ","; // pixel_values byte array
    name.next();

    stmnt += "point(" + name.get() + ","; // pixel coord start
    name.next();
    stmnt += name.get() + "),"; // pixel coord end
    name.next();

    stmnt += "point(" + name.get() + ","; // pixel lm start
    name.next();
    stmnt += name.get() + "),"; // pixel lm end
    name.next();

    stmnt += name.get() + ","; // source name(s)
    name.next();

    stmnt += "point(" + name.get() + ","; // pixel offset start
    name.next();
    stmnt += name.get() + ")"; // pixel offset end
    name.next();

    stmnt += ")";

    return stmnt;
}

std::string
get_pixel_insert_stmnt_n(int nrows)
{
    pqxx::placeholders name;
    std::string stmnt = "INSERT INTO epic_pixels (id, pixel_values, pixel_coord, pixel_lm, source_name,  pix_ofst) VALUES ";
    for (int i = 0; i < nrows; ++i) {
        stmnt += get_pixel_insert_stmnt_1(name);
        if (i < (nrows - 1)) {
            stmnt += ",";
        }
    }
    return stmnt;
}

std::string
get_img_meta_insert_stmnt_1(pqxx::placeholders<unsigned int>& name)
{
    std::string stmnt = "(";
    int ncols = 7;
    for (int i = 0; i < ncols; ++i) {
        stmnt += name.get();
        stmnt += ",";
        name.next();
    }

    stmnt += "point(" + name.get() + ",";
    name.next();
    stmnt += name.get() + ")";
    stmnt += ")";
    name.next();

    return stmnt;
}

std::string
get_img_meta_insert_stmnt_n(int nrows)
{
    pqxx::placeholders name;
    std::string stmnt = "INSERT INTO epic_img_metadata (id, img_time, n_chan, n_pol, chan0, chan_bw, epic_version, img_size) VALUES ";
    for (int i = 0; i < nrows; ++i) {
        stmnt += get_img_meta_insert_stmnt_1(name);
        if (i < (nrows - 1)) {
            stmnt += ",";
        }
    }
    return stmnt;
}

template<typename _Pld>
void
ingest_payload(
  _Pld& pld,
  pqxx::work& work,
  int npix_per_src,
  std::string pix_stmnt_id = "insert_pixels",
  std::string meta_stmnt_id = "insert_meta")
{
    ingest_pixels_src_n(pld, work, npix_per_src, pix_stmnt_id);
    ingest_meta(pld, work, npix_per_src, meta_stmnt_id);
}


template<typename _PgT>
void
ingest_pixels_src_1(_PgT& data, pqxx::work& work, int src_idx, int nchan, int npix_per_src, std::string stmnt_id)
{
    pqxx::params pars;
    pars.reserve(9);
    int nelem_per_src = nchan * NSTOKES * npix_per_src;
    int nelem_per_pix = nchan * NSTOKES;

    for (int i = 0; i < npix_per_src; ++i) {
        int coord_idx = src_idx * npix_per_src + i;
        auto pix_coord = data.pixel_coords[coord_idx];
        auto pix_lm = data.pixel_lm[coord_idx];
        auto pix_offst = data.pixel_offst[coord_idx];

            pqxx::params pix_pars(
              data.m_uuid,
              std::basic_string_view<std::byte>(
                reinterpret_cast<std::byte*>(
                  data.pixel_values.get() + src_idx * nelem_per_src + i * nelem_per_pix),
                  nelem_per_pix * sizeof(float)),
              pix_coord.first,
              pix_coord.second,
              pix_lm.first,
              pix_lm.second,
              data.source_ids[src_idx],
              pix_offst.first,
              pix_offst.second);

            pars.append(pix_pars);
    }
    work.exec_prepared0(stmnt_id, pars);
}

template<typename _Pld>
void
ingest_pixels_src_n(_Pld& pld, pqxx::work& work, int npix_per_src, const std::string& stmnt)
{
    auto& pix_data = *(pld.get_mbuf());
    int nsrc = pix_data.nsrcs;
    int ncoords = pix_data.m_ncoords;
    int nchan = pix_data.m_nchan;

    assert((ncoords == (nsrc * npix_per_src)) && "The number of extracted pixels do not match the value expected from the source number");

    for (int src_idx = 0; src_idx < nsrc; ++src_idx) {
        ingest_pixels_src_1(pix_data, work, src_idx, nchan, npix_per_src, stmnt);
    }
}

template<typename _Pld>
void
ingest_meta(_Pld& pld, pqxx::work& work, int npix_per_src, const std::string& stmnt)
{
    auto& meta = pld.get_mbuf()->m_img_metadata;
    auto& pix_data = *(pld.get_mbuf());
    auto time_tag = std::get<uint64_t>(meta["time_tag"]);
    auto img_len_ms = std::get<double>(meta["img_len_ms"]);
    auto nchan = pix_data.m_nchan;
    auto chan0 = std::get<int64_t>(meta["chan0"]);
    auto bw = std::get<int>(meta["chan_width"]);
    auto grid_size = std::get<int>(meta["grid_size"]);
    pqxx::params pars(
        pix_data.m_uuid,
        meta2pgtime(time_tag, img_len_ms),
        nchan,
        int(NSTOKES),
        int(chan0),
        bw,
        EPIC_VERSION,
        grid_size,
        grid_size
    );

    work.exec_prepared0(stmnt, pars);
}
