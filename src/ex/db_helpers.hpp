/*
 Copyright (c) 2023 The EPIC++ authors

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 the Software, and to permit persons to whom the Software is furnished to do so,
 subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef SRC_EX_DB_HELPERS_HPP_
#define SRC_EX_DB_HELPERS_HPP_
#include <pqxx/pqxx>
#include <cstring>
#include <iostream>
#include <string_view>
#include <string>
#include <vector>

#include "./constants.h"
#include "./py_funcs.hpp"

std::string
get_pixel_insert_stmnt_1(pqxx::placeholders<unsigned int>* name_ptr) {
    std::string stmnt = "(";

    auto& name = *name_ptr;

    stmnt += name.get() + ",";  // uuid
    name.next();

    stmnt += name.get() + ",";  // pixel_values byte array
    name.next();

    stmnt += "point(" + name.get() + ",";  // pixel coord start
    name.next();
    stmnt += name.get() + "),";  // pixel coord end
    name.next();

    stmnt += "point(" + name.get() + ",";  // pixel lm start
    name.next();
    stmnt += name.get() + "),";  // pixel lm end
    name.next();

    stmnt += name.get() + ",";  // source name(s)
    name.next();

    stmnt += "point(" + name.get() + ",";  // pixel offset start
    name.next();
    stmnt += name.get() + ")";  // pixel offset end
    name.next();

    stmnt += ")";

    return stmnt;
}

std::string
get_pixel_insert_stmnt_n(int nrows) {
    pqxx::placeholders name;
    std::string stmnt = "INSERT INTO epic_pixels";
    stmnt += "(id, pixel_values, pixel_coord, pixel_lm";
    stmnt += ", source_name,  pix_ofst)";
    stmnt += "VALUES ";
    for (int i = 0; i < nrows; ++i) {
        stmnt += get_pixel_insert_stmnt_1(&name);
        if (i < (nrows - 1)) {
            stmnt += ",";
        }
    }
    return stmnt;
}

std::string
get_img_meta_insert_stmnt_1(pqxx::placeholders<unsigned int>* name_ptr) {
    auto& name = *name_ptr;
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
    name.next();

    stmnt += "," + name.get();
    name.next();

    stmnt += ")";
    return stmnt;
}

std::string
get_img_meta_insert_stmnt_n(int n_images) {
    pqxx::placeholders name;
    std::string stmnt = "INSERT INTO epic_img_metadata";
    stmnt += "(id, img_time, n_chan, n_pol, chan0, chan_bw, epic_version";
    stmnt += ", img_size, npix_kernel) VALUES ";
    for (int i = 0; i < n_images; ++i) {
        stmnt += get_img_meta_insert_stmnt_1(&name);
        if (i < (n_images - 1)) {
            stmnt += ",";
        }
    }
    return stmnt;
}

template <typename _Pld>
void ingest_payload(
    _Pld* pld_ptr,
    pqxx::work* work_ptr,
    int npix_per_src,
    std::string pix_stmnt_id = "insert_pixels",
    std::string meta_stmnt_id = "insert_meta") {
    ingest_pixels_src_n(pld_ptr, work_ptr, npix_per_src, pix_stmnt_id);
    ingest_meta(pld_ptr, work_ptr, npix_per_src, meta_stmnt_id);
}

template <typename _PgT>
void ingest_pixels_src_1(const _PgT& data
    , pqxx::work* work_ptr
    , int src_idx, int nchan
    , int npix_per_src
    , std::string stmnt_id) {
    // auto& data = *data_ptr;
    auto& work = *work_ptr;
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
                    data.pixel_values.get()
                    + src_idx * nelem_per_src + i * nelem_per_pix),
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

/**
 * @brief Insert pixel data for n sources in to the db
 *
 * @tparam _Pld
 * @param pld
 * @param work
 * @param npix_per_src
 * @param stmnt
 */
template <typename _Pld>
void ingest_pixels_src_n(_Pld* pld_ptr, pqxx::work* work_ptr
    , int npix_per_src, const std::string& stmnt) {
    auto& pld = *pld_ptr;
    auto& pix_data = *(pld.get_mbuf());
    int nsrc = pix_data.nsrcs;
    int nchan = pix_data.m_nchan;

    assert((pix_data.m_ncoords == (nsrc * npix_per_src))
        && "Mismatch between the extracted pixels and the source number");

    for (int src_idx = 0; src_idx < nsrc; ++src_idx) {
        ingest_pixels_src_1(pix_data, work_ptr, src_idx
            , nchan, npix_per_src, stmnt);
    }
}

/**
 * @brief Insert metadata into the db
 *
 * @tparam _Pld Payload type
 * @param pld Payload object
 * @param work pqxx transaction object
 * @param npix_per_src Number of pixels per source (kernel_size * kernel_size)
 * @param stmnt Prepared statement id
 */
template <typename _Pld>
void ingest_meta(_Pld* pld_ptr, pqxx::work* work_ptr
    , int npix_per_src, const std::string& stmnt) {
    auto& pld = *pld_ptr;
    auto& work = *work_ptr;
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
        static_cast<int>(NSTOKES),
        static_cast<int>(chan0),
        bw,
        EPIC_VERSION,
        grid_size,
        grid_size,
        npix_per_src);

    work.exec_prepared0(stmnt, pars);
}

#endif  // SRC_EX_DB_HELPERS_HPP_
