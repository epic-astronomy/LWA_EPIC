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
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "./constants.h"
#include "./pg_link.hpp"
#include "./py_funcs.hpp"
#include "git.h"
#include "pqxx/pqxx"
#include "glog/logging.h"

std::string GetSinglePixelInsertStmnt(
    pqxx::placeholders<unsigned int>* name_ptr) {
  std::string stmnt = "(";

  auto& name = *name_ptr;

  stmnt += name.get() + ",";  // img_time
  name.next();

  stmnt += name.get() + ",";  // session_id
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
  stmnt += name.get() + "),";  // pixel offset end
  name.next();

  stmnt += "point(" + name.get() + ",";  // source pixel position start
  name.next();
  stmnt += name.get() + ")";  // source pixel position end
  name.next();

  stmnt += ")";

  return stmnt;
}

std::string GetMultiPixelInsertStmnt(int nrows, std::string schema="public") {
  pqxx::placeholders name;
  std::string stmnt = "INSERT INTO ";
  stmnt +=  schema + ".epic_pixels";
  stmnt += "(img_time, chan0, pixel_values, pixel_coord, pixel_lm";
  stmnt += ", source_name,  pix_offset, source_pix) ";
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
  //int ncols = 7;
  for (int i = 0; i < 9; ++i) {
    stmnt += name.get(); //session_id,chan0,n_pol,n_chan,chan_bw_hz, epic_version, npix_kernel, int_time, source_name
    stmnt += ",";
    name.next();
  }

  stmnt += "point(" + name.get() + ","; //img_size
  name.next();
  stmnt += name.get() + ")";
  name.next();

  stmnt += "," + name.get() + "," + name.get();
  name.next();

  // stmnt += "," + name.get(); // npix_kernel
  // name.next();

  // stmnt += "," + name.get(); // int_time
  // name.next();

  // stmnt += "," + name.get(); // source_names
  // name.next();

  // stmnt += "," + name.get(); // session_id
  // name.next();

  stmnt += ")";
  return stmnt;
}

std::string GetMultiImgMetaInsertStmnt(int n_images, std::string schema="public") {
  pqxx::placeholders name;
  std::string stmnt = "INSERT INTO ";
  stmnt +=  schema + ".epic_img_metadata2(session_id,session_start,session_end,chan0,n_pol,n_chan,chan_bw_hz, epic_version, npix_kernel, int_time, source_name,img_size) SELECT \
  session_id::uuid, session_start::timestamp without time zone, session_end::timestamp without time zone, chan0::int, n_pol::int, n_chan::int, chan_bw_hz::int, epic_version\
  ,npix_kernel::int, int_time::double precision, unnest(source_names::text[]), img_size  FROM ( VALUES ";
  for (int i = 0; i < n_images; ++i) {
    stmnt += GetSingleImgMetaInsertStmnt(&name);
    if (i < (n_images - 1)) {
      stmnt += ",";
    }
  }
  stmnt += " ) AS img(session_id,chan0,n_pol,n_chan,chan_bw_hz, epic_version, npix_kernel, int_time, source_names,img_size,session_start,session_end)";
  stmnt += "ON CONFLICT ON CONSTRAINT epic_img_metadata2_pkey ";
  stmnt += "DO UPDATE SET session_end=excluded.session_end";
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
                           int nchan, int npix_per_src, std::string stmnt_id,
                           std::string time_stamp, int chan0/*std::string session_id*/) {
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
    auto src_pix = data.source_pix[coord_idx];

    pqxx::params pix_pars(
        //data.m_uuid,
        time_stamp,
        chan0,
        std::basic_string_view<std::byte>(
            reinterpret_cast<std::byte*>(data.pixel_values.get() +
                                         src_idx * nelem_per_src +
                                         i * nelem_per_pix),
            nelem_per_pix * sizeof(float)),
        pix_coord.first, pix_coord.second, pix_lm.first, pix_lm.second,
        data.source_ids[coord_idx], pix_offst.first, pix_offst.second,
        src_pix.first, src_pix.second);

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
  auto& meta = pld.get_mbuf()->m_img_metadata;
  auto time_tag = std::get<uint64_t>(meta["time_tag"]);
  auto img_len_ms = std::get<double>(meta["img_len_ms"]);
  auto session_id = std::get<std::string>(meta["session_id"]);
  auto chan0 = std::get<int64_t>(meta["chan0"]);
  auto time_stamp = Meta2PgTime(time_tag, img_len_ms);
  int nsrc = pix_data.nsrcs;
  int nchan = pix_data.m_nchan;

  assert((pix_data.m_ncoords == (nsrc * npix_per_src)) &&
         "Mismatch between the extracted pixels and the source number");

  for (int src_idx = 0; src_idx < nsrc; ++src_idx) {
    IngestPixelsSingleSrc(pix_data, work_ptr, src_idx, nchan, npix_per_src,
                          stmnt, time_stamp, chan0/*session_id*/);
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
  auto session_id = std::get<std::string>(meta["session_id"]);
  //session_id,chan0,n_pol,n_chan,chan_bw_hz, epic_version, npix_kernel, int_time, source_names,img_size,session_start,session_end
  pqxx::params pars(session_id, static_cast<int>(chan0),
                    static_cast<int>(NSTOKES), nchan, bw,
                    git_CommitSHA1(), npix_per_src, int_time,
                    source_name_arr, grid_size, grid_size, Meta2PgTime(time_tag, img_len_ms));

  work.exec_prepared0(stmnt, pars);
}

/**
 * @brief The rows of one image for epic_pixels and epic_img_metadata2, copied
 * out of the pipeline's buffers so that they can be written later, on the
 * database writer's thread.
 *
 * The member names are those of EpicPixelTableDataRows, so
 * IngestPixelsSingleSrc() takes either.
 */
struct PixelJob {
  std::string time_stamp;
  std::string session_id;
  int64_t chan0{0};
  int nchan{0};
  int nsrc{0};
  int npix_per_src{0};
  int chan_bw{0};
  int grid_size{0};
  double int_time{0};
  std::string source_name_arr;
  std::unique_ptr<float[]> pixel_values;
  std::vector<std::pair<int, int>> pixel_coords;
  std::vector<std::pair<float, float>> pixel_lm;
  std::vector<std::pair<int, int>> pixel_offst;
  std::vector<std::pair<float, float>> source_pix;
  std::vector<std::string> source_ids;
};

/**
 * @brief Copy one image's pixel rows and session metadata out of a payload.
 *
 * @param pld_ptr Payload from the pixel extractor
 * @param npix_per_src Pixels per source (kernel_dim * kernel_dim)
 */
template <typename _Pld>
std::shared_ptr<PixelJob> MakePixelJob(_Pld* pld_ptr, int npix_per_src) {
  auto& pix_data = *(pld_ptr->get_mbuf());
  auto& meta = pix_data.m_img_metadata;
  auto job = std::make_shared<PixelJob>();
  auto time_tag = std::get<uint64_t>(meta["time_tag"]);
  auto img_len_ms = std::get<double>(meta["img_len_ms"]);
  job->time_stamp = Meta2PgTime(time_tag, img_len_ms);
  job->session_id = std::get<std::string>(meta["session_id"]);
  job->chan0 = std::get<int64_t>(meta["chan0"]);
  job->chan_bw = std::get<int>(meta["chan_width"]);
  job->grid_size = std::get<int>(meta["grid_size"]);
  job->int_time = img_len_ms / 1000;
  job->nchan = pix_data.m_nchan;
  job->nsrc = pix_data.nsrcs;
  job->npix_per_src = npix_per_src;
  job->source_name_arr = pix_data.source_name_arr;

  size_t ncoords = size_t(job->nsrc) * npix_per_src;
  assert((pix_data.m_ncoords == ncoords) &&
         "Mismatch between the extracted pixels and the source number");
  size_t nvalues = ncoords * job->nchan * NSTOKES;
  job->pixel_values = std::make_unique<float[]>(nvalues);
  std::copy(pix_data.pixel_values.get(), pix_data.pixel_values.get() + nvalues,
            job->pixel_values.get());
  job->pixel_coords.assign(pix_data.pixel_coords.begin(),
                           pix_data.pixel_coords.begin() + ncoords);
  job->pixel_lm.assign(pix_data.pixel_lm.begin(),
                       pix_data.pixel_lm.begin() + ncoords);
  job->pixel_offst.assign(pix_data.pixel_offst.begin(),
                          pix_data.pixel_offst.begin() + ncoords);
  job->source_pix.assign(pix_data.source_pix.begin(),
                         pix_data.source_pix.begin() + ncoords);
  job->source_ids.assign(pix_data.source_ids.begin(),
                         pix_data.source_ids.begin() + ncoords);
  return job;
}

/**
 * @brief Insert one image: its pixel rows source by source, then the session
 * row (epic_img_metadata2), in the given transaction.
 */
inline void IngestPixelJob(const PixelJob& job, pqxx::work* work_ptr,
                           const std::string& pix_stmnt_id,
                           const std::string& meta_stmnt_id) {
  for (int src_idx = 0; src_idx < job.nsrc; ++src_idx) {
    IngestPixelsSingleSrc(job, work_ptr, src_idx, job.nchan, job.npix_per_src,
                          pix_stmnt_id, job.time_stamp, job.chan0);
  }
  // session_id,chan0,n_pol,n_chan,chan_bw_hz, epic_version, npix_kernel,
  // int_time, source_names,img_size,session_start,session_end
  pqxx::params pars(job.session_id, static_cast<int>(job.chan0),
                    static_cast<int>(NSTOKES), job.nchan, job.chan_bw,
                    git_CommitSHA1(), job.npix_per_src, job.int_time,
                    job.source_name_arr, job.grid_size, job.grid_size,
                    job.time_stamp);
  work_ptr->exec_prepared0(meta_stmnt_id, pars);
}

std::string GetFileMetaInsertStmt(std::string schema="public") {
  const std::vector<std::string> cols{
      "file_name",   "chan_width",   "nchan",        "support_size",
      "gulp_len_ms", "image_len_ms", "epoch_time_s", "grid_size",
      "grid_res",    "cfreq_mhz",    "epic_version"};

  pqxx::placeholders name;
  std::string stmnt ="INSERT INTO ";
  stmnt += schema + ".epic_files_metadata2 (";  // filename,...) VALUES ($1, $2...)
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

inline void InsertFileMetaToDb(dict_t& meta, pqxx::work* work_ptr,
                               const std::string& insert_stmt) {
  //  "filename",    "chan_width",   "nchan",        "support_size",
  //  "gulp_len_ms", "image_len_ms", "epoch_time_s", "grid_size",
  //  "grid_res",    "cfreq_mhz",    "epic_version"

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
                    static_cast<int>(epoch_time_s), grid_size, grid_res, cfreq_mhz,
                    git_CommitSHA1());

  (*work_ptr).exec_prepared0(insert_stmt, pars);
}

template <typename _Pld>
void InsertFilenametoDb(_Pld* pld_ptr, pqxx::work* work_ptr,
                        const std::string& insert_stmt) {
  InsertFileMetaToDb(pld_ptr->get_mbuf()->GetMetadataRef(), work_ptr,
                     insert_stmt);
}

/**
 * @brief A synchronous connection for kernels that write in their own
 * thread. A database that is down never stops the imager: the connection is
 * attempted again by Reconnect(), and errors are the caller's to log.
 * Kernels in the pipeline use PgLink (pg_link.hpp) instead, which writes off
 * the pipeline's threads.
 */
class PgDbConnectMixin {
 protected:
  bool is_db_alive{false};
  // Each write opens its own short transaction. A transaction opened here and
  // kept open would hold locks on the tables until its first commit.
  std::unique_ptr<pqxx::connection> m_pg_conn;
  std::string m_conn_str;
  std::map<std::string, std::string> m_stmts;

 public:
  explicit PgDbConnectMixin(std::string p_conn_str)
      : m_conn_str(NodeDbConnString(p_conn_str)) {
    Reconnect();
  }

  bool Reconnect() {
    try {
      m_pg_conn = std::make_unique<pqxx::connection>(m_conn_str);
      for (auto& [id, sql] : m_stmts) {
        m_pg_conn->prepare(id, sql);
      }
      is_db_alive = true;
    } catch (const std::exception& e) {
      LOG(WARNING) << "Database unavailable: " << e.what();
      is_db_alive = false;
    }
    return is_db_alive;
  }

  void prepare_stmt(std::string p_name, std::string p_stmt) {
    m_stmts[p_name] = p_stmt;
    if (!is_db_alive) return;  // prepared on reconnecting
    try {
      m_pg_conn->prepare(p_name, p_stmt);
    } catch (const std::exception& e) {
      LOG(WARNING) << "Unable to prepare statement " << p_name << ": "
                   << e.what();
      is_db_alive = false;
    }
  }
};

#endif  // SRC_EX_DB_HELPERS_HPP_
