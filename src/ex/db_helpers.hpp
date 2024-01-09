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

#ifndef SRC_EX_DB_HELPERS_HPP_
#define SRC_EX_DB_HELPERS_HPP_
#include <algorithm>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "./constants.h"
#include "./py_funcs.hpp"
#include "git.h"
#include "pqxx/pqxx"

std::string GetSinglePixelInsertStmnt(
    pqxx::placeholders<unsigned int>* name_ptr) {
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

  stmnt += name.get() + ",";  // source name
  name.next();

  stmnt += "point(" + name.get() + ",";  // pixel offset start
  name.next();
  stmnt += name.get() + ")";  // pixel offset end
  name.next();

  stmnt += ")";

  return stmnt;
}

std::string GetMultiPixelInsertStmnt(int nrows) {
  pqxx::placeholders name;
  std::string stmnt = "INSERT INTO epic_pixels";
  stmnt += "(id, pixel_values, pixel_coord, pixel_lm";
  stmnt += ", source_name,  pix_ofst)";
  stmnt += "VALUES ";
  for (int i = 0; i < nrows; ++i) {
    stmnt += GetSinglePixelInsertStmnt(&name);
    if (i < (nrows - 1)) {
      stmnt += ",";
    }
  }
  return stmnt;
}

std::string GetSingleImgMetaInsertStmnt(
    pqxx::placeholders<unsigned int>* name_ptr) {
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

  stmnt += "," + name.get();
  name.next();

  stmnt += "," + name.get();
  name.next();

  stmnt += ")";
  return stmnt;
}

std::string GetMultiImgMetaInsertStmnt(int n_images) {
  pqxx::placeholders name;
  std::string stmnt = "INSERT INTO epic_img_metadata";
  stmnt += "(id, img_time, n_chan, n_pol, chan0, chan_bw, epic_version";
  stmnt += ", img_size, npix_kernel, int_time, source_names) VALUES ";
  for (int i = 0; i < n_images; ++i) {
    stmnt += GetSingleImgMetaInsertStmnt(&name);
    if (i < (n_images - 1)) {
      stmnt += ",";
    }
  }
  return stmnt;
}

template <typename _Pld>
void IngestPayload(_Pld* pld_ptr, pqxx::work* work_ptr, int npix_per_src,
                   std::string pix_stmnt_id = "insert_pixels",
                   std::string meta_stmnt_id = "insert_meta") {
  IngestPixelsMultiSrc(pld_ptr, work_ptr, npix_per_src, pix_stmnt_id);
  IngestMetadata(pld_ptr, work_ptr, npix_per_src, meta_stmnt_id);
}

template <typename _PgT>
void IngestPixelsSingleSrc(const _PgT& data, pqxx::work* work_ptr, int src_idx,
                           int nchan, int npix_per_src, std::string stmnt_id) {
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
            reinterpret_cast<std::byte*>(data.pixel_values.get() +
                                         src_idx * nelem_per_src +
                                         i * nelem_per_pix),
            nelem_per_pix * sizeof(float)),
        pix_coord.first, pix_coord.second, pix_lm.first, pix_lm.second,
        data.source_ids[coord_idx], pix_offst.first, pix_offst.second);

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
void IngestPixelsMultiSrc(_Pld* pld_ptr, pqxx::work* work_ptr, int npix_per_src,
                          const std::string& stmnt) {
  auto& pld = *pld_ptr;
  auto& pix_data = *(pld.get_mbuf());
  int nsrc = pix_data.nsrcs;
  int nchan = pix_data.m_nchan;

  assert((pix_data.m_ncoords == (nsrc * npix_per_src)) &&
         "Mismatch between the extracted pixels and the source number");

  for (int src_idx = 0; src_idx < nsrc; ++src_idx) {
    IngestPixelsSingleSrc(pix_data, work_ptr, src_idx, nchan, npix_per_src,
                          stmnt);
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
void IngestMetadata(_Pld* pld_ptr, pqxx::work* work_ptr, int npix_per_src,
                    const std::string& stmnt) {
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
  auto int_time = std::get<double>(meta["img_len_ms"])/1000;
  auto source_name_arr = pix_data.source_name_arr;
  pqxx::params pars(pix_data.m_uuid, Meta2PgTime(time_tag, img_len_ms), nchan,
                    static_cast<int>(NSTOKES), static_cast<int>(chan0), bw,
                    git_CommitSHA1(), grid_size, grid_size, npix_per_src, int_time,
                    source_name_arr);

  work.exec_prepared0(stmnt, pars);
}

std::string GetFileMetaInsertStmt() {
  const std::vector<std::string> cols{
      "file_name",   "chan_width",   "nchan",        "support_size",
      "gulp_len_ms", "image_len_ms", "epoch_time_s", "grid_size",
      "grid_res",    "cfreq_mhz",    "epic_version"};

  pqxx::placeholders name;
  std::string stmnt =
      "INSERT INTO epic_files_metadata (";  // filename,...) VALUES ($1, $2...)
  stmnt +=
      std::accumulate(std::next(cols.begin()), cols.end(), cols[0],
                      [](std::string a, std::string b) { return a + "," + b; });
  stmnt += ") VALUES (";
  std::vector<std::string> placeholders(cols.size());
  std::generate(placeholders.begin(), placeholders.end(), [&]() {
    auto ph = name.get();
    name.next();
    return ph;
  });
  stmnt += std::accumulate(
      std::next(placeholders.begin()), placeholders.end(), placeholders[0],
      [](std::string a, std::string b) { return a + "," + b; });
  stmnt += ")";

  return stmnt;
}

template <typename _Pld>
void InsertFilenametoDb(_Pld* pld_ptr, pqxx::work* work_ptr,
                        const std::string& insert_stmt) {
  //  "filename",    "chan_width",   "nchan",        "support_size",
  //  "gulp_len_ms", "image_len_ms", "epoch_time_s", "grid_size",
  //  "grid_res",    "cfreq_mhz",    "epic_version"

  auto& meta = pld_ptr->get_mbuf()->GetMetadataRef();
  auto filename = std::get<std::string>(meta["filename"]);
  auto chan_width = std::get<int>(meta["chan_width"]);
  int nchan = std::get<uint8_t>(meta["nchan"]);
  auto support = std::get<int>(meta["support_size"]);
  auto gulp_len = std::get<double>(meta["gulp_len_ms"]);
  auto image_len = std::get<double>(meta["img_len_ms"]);
  auto epoch_time_s = std::get<double>(meta["epoch_time_s"]);
  auto grid_size = std::get<int>(meta["grid_size"]);
  auto grid_res = std::get<float>(meta["grid_res"]);
  auto cfreq_mhz = std::get<double>(meta["cfreq"]);

  pqxx::params pars(filename, chan_width, nchan, support, gulp_len, image_len,
                    epoch_time_s, grid_size, grid_res, cfreq_mhz,
                    git_CommitSHA1());

  (*work_ptr).exec_prepared0(insert_stmt, pars);
}

class PgDbConnectMixin {
 protected:
  bool is_db_alive{false};
  std::unique_ptr<pqxx::connection> m_pg_conn;
  std::unique_ptr<pqxx::work> m_db_T;  // transaction
  std::string m_conn_str;

 public:
  explicit PgDbConnectMixin(std::string p_conn_str) : m_conn_str(p_conn_str) {
    try {
      m_pg_conn = std::make_unique<pqxx::connection>(p_conn_str);
    } catch (const std::exception& e) {
      LOG(FATAL) << e.what();
    }

    is_db_alive = true;
    m_db_T = std::make_unique<pqxx::work>(*(m_pg_conn.get()));
  }

  void prepare_stmt(std::string p_name, std::string p_stmt) {
    if (is_db_alive) {
      m_pg_conn.get()->prepare(p_name, p_stmt);
    } else {
      LOG(FATAL) << "Unable to prepare stament";
    }
  }
};

#endif  // SRC_EX_DB_HELPERS_HPP_
