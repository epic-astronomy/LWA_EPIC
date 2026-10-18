/*
 Test of the imager's writes to the node database (src/ex/pg_link.hpp and
 the statements of src/ex/db_helpers.hpp), against a throwaway PostgreSQL
 server holding the per-node schema.

 Built with the imager when configured with -DEPIC_BUILD_DB_TESTS=ON:
   cmake --build <build> --target node_db_test
   EPIC_TEST_DSN="host=127.0.0.1 port=... dbname=epic user=epic password=..." \
   EPIC_TEST_ADMIN_DSN="host=... user=postgres" <build>/node_db_test
 Exit status 0 when every check passes. The schema must hold a partition of
 sevilleta.epic_pixels for the current time (the maintenance job's premake).

 Checks:
  1. database unavailable: Submit() never blocks; writes beyond the queue are
     dropped and counted; destruction does not hang;
  2. images and files written in the imager's format: pixel rows, the
     session upsert, the file row and its generated time;
  3. a write the server refuses (no partition) is dropped, later ones go on;
  4. a connection killed by the server: the write is retried and lands.
*/
#include <pqxx/pqxx>  // first, as in main.cpp

#include <cuda_runtime.h>  // types.hpp names CUDA types (no CUDA calls here)

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../../src/ex/db_helpers.hpp"
#include "../../src/ex/pg_link.hpp"

namespace {
int g_failures = 0;

void Check(bool ok, const std::string& what) {
  std::cout << (ok ? "PASS " : "FAIL ") << what << std::endl;
  if (!ok) ++g_failures;
}

std::string UtcNow(int offset_s = 0) {
  std::time_t t = std::time(nullptr) + offset_s;
  char buf[64];
  std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::gmtime(&t));
  return buf;
}

std::shared_ptr<PixelJob> FakeImage(const std::string& ts,
                                    const std::string& session, int nsrc,
                                    int kdim, int nchan, int chan0) {
  auto job = std::make_shared<PixelJob>();
  job->time_stamp = ts;
  job->session_id = session;
  job->chan0 = chan0;
  job->nchan = nchan;
  job->nsrc = nsrc;
  job->npix_per_src = kdim * kdim;
  job->chan_bw = 23926;
  job->grid_size = 128;
  job->int_time = 0.0836;
  job->source_name_arr = nsrc == 2 ? "{sun,\"Cyg A\"}" : "{sun}";
  size_t ncoords = size_t(nsrc) * kdim * kdim;
  size_t nvalues = ncoords * nchan * NSTOKES;
  job->pixel_values = std::make_unique<float[]>(nvalues);
  for (size_t i = 0; i < nvalues; ++i) job->pixel_values[i] = float(i % 977);
  for (size_t i = 0; i < ncoords; ++i) {
    int s = int(i) / (kdim * kdim), k = int(i) % (kdim * kdim);
    job->pixel_coords.emplace_back(60 + k % kdim, 60 + k / kdim);
    job->pixel_lm.emplace_back(0.01f * k, -0.01f * k);
    job->pixel_offst.emplace_back(k % kdim - kdim / 2, k / kdim - kdim / 2);
    job->source_pix.emplace_back(62.25f, 61.75f);
    job->source_ids.push_back(s == 0 ? "sun" : "Cyg A");
  }
  return job;
}

void SubmitImage(PgLink* link, std::shared_ptr<PixelJob> job,
                 const std::string& schema) {
  int kdim = 1;
  while (kdim * kdim < job->npix_per_src) ++kdim;
  std::string pix_id = "insert_pixels_" + std::to_string(kdim);
  std::string meta_id = "insert_meta_1";
  link->Submit("image " + job->time_stamp,
               {{pix_id, GetMultiPixelInsertStmnt(job->npix_per_src, schema)},
                {meta_id, GetMultiImgMetaInsertStmnt(1, schema)}},
               [job, pix_id, meta_id](pqxx::work& tx) {
                 IngestPixelJob(*job, &tx, pix_id, meta_id);
               });
}

long Count(pqxx::connection& c, const std::string& sql) {
  pqxx::nontransaction tx(c);
  return tx.query_value<long>(sql);
}
}  // namespace

int main() {
  google::InitGoogleLogging("node_db_test");
  FLAGS_logtostderr = true;
  const std::string schema = "sevilleta";
  const char* dsn_env = std::getenv("EPIC_TEST_DSN");
  const char* admin_env = std::getenv("EPIC_TEST_ADMIN_DSN");
  if (!dsn_env || !admin_env) {
    std::cerr << "EPIC_TEST_DSN and EPIC_TEST_ADMIN_DSN are required\n";
    return 2;
  }
  const std::string dsn = std::string(dsn_env) + " application_name=node_db_test";
  pqxx::connection admin(admin_env);

  // ------------------------------------------- 1. database unavailable
  {
    auto link = std::make_unique<PgLink>(
        "test (no database)", "host=127.0.0.1 port=1 dbname=epic user=epic", 100,
        std::chrono::seconds(10), 10000, std::chrono::milliseconds(300));
    auto job = FakeImage(UtcNow(), "00000000-0000-0000-0000-000000000001", 2, 5,
                         32, 672);
    double worst_ms = 0;
    for (int i = 0; i < 2000; ++i) {
      auto s = std::chrono::steady_clock::now();
      SubmitImage(link.get(), job, schema);
      worst_ms = std::max(worst_ms, std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - s)
                                        .count());
    }
    Check(worst_ms < 5.0, "Submit() returns at once without a database (worst " +
                              std::to_string(worst_ms) + " ms)");
    Check(link->Dropped() == 1900 && link->Pending() == 100,
          "100 writes kept, 1900 dropped (" + std::to_string(link->Pending()) +
              " kept, " + std::to_string(link->Dropped()) + " dropped)");
    Check(!link->Connected(), "not connected");
    auto d0 = std::chrono::steady_clock::now();
    link.reset();
    double destroy_s = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - d0).count();
    Check(destroy_s < 2.0, "destroyed with writes pending in " +
                               std::to_string(destroy_s) + " s");
  }

  // ------------------------------------- 2. writes in the imager's format
  {
    pqxx::nontransaction(admin).exec0(
        "TRUNCATE sevilleta.epic_pixels, sevilleta.epic_img_metadata2, "
        "sevilleta.epic_files_metadata2");
    PgLink link("test", dsn, 1200);
    const std::string session = "6f1c2a0e-6b0e-4a51-9a39-6a4c8d3c0d11";
    const int nimg = 50, nsrc = 2, kdim = 5, nchan = 32;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < nimg; ++i) {
      SubmitImage(&link, FakeImage(UtcNow(i - nimg), session, nsrc,
                                             kdim, nchan, 672),
                  schema);
    }
    dict_t meta;
    meta["filename"] = std::string("/data/epic1/EPIC_1792000000.000_17.609MHz.fits");
    meta["chan_width"] = 23926;
    meta["nchan"] = uint8_t(32);
    meta["support_size"] = 5;
    meta["gulp_len_ms"] = 41.8;
    meta["img_len_ms"] = 83.6;
    meta["epoch_time_s"] = 1792000000.0;
    meta["grid_size"] = 128;
    meta["grid_res"] = 1.056f;
    meta["cfreq"] = 17.609389;
    auto m = std::make_shared<dict_t>(meta);
    link.Submit("file", {{"insert_file_meta", GetFileMetaInsertStmt(schema)}},
                [m](pqxx::work& tx) {
                  InsertFileMetaToDb(*m, &tx, "insert_file_meta");
                });
    Check(link.Drain(std::chrono::seconds(30)), "queue drained");
    double secs = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - t0).count();
    std::cout << "INFO " << nimg << " images (" << nimg * nsrc * kdim * kdim
              << " rows) written in " << secs << " s, "
              << 1000.0 * secs / nimg << " ms per image" << std::endl;
    Check(link.Written() == size_t(nimg + 1) && link.Failed() == 0,
          "all writes done (" + std::to_string(link.Written()) + " written, " +
              std::to_string(link.Failed()) + " failed)");
    Check(Count(admin, "SELECT count(*) FROM sevilleta.epic_pixels") ==
              long(nimg) * nsrc * kdim * kdim,
          "pixel rows");
    Check(Count(admin,
                "SELECT count(*) FROM sevilleta.epic_pixels WHERE "
                "length(pixel_values) = " +
                    std::to_string(nchan * NSTOKES * sizeof(float)) +
                    " AND source_pix ~= point(62.25, 61.75) AND chan0 = 672") ==
              long(nimg) * nsrc * kdim * kdim,
          "pixel values, source position and chan0 as written");
    Check(Count(admin, "SELECT count(*) FROM sevilleta.epic_img_metadata2") == 2,
          "one session row per source (upsert)");
    Check(Count(admin,
                "SELECT count(*) FROM sevilleta.epic_img_metadata2 WHERE "
                "session_end > session_start AND source_name IN ('sun','Cyg A')") ==
              2,
          "session_end moved on by later images");
    Check(Count(admin,
                "SELECT count(*) FROM sevilleta.epic_files_metadata2 WHERE "
                "epoch_time = to_timestamp(1792000000) AT TIME ZONE 'UTC' "
                "AND cfreq_mhz > 17.6") == 1,
          "file row with its generated time");
  }

  // ------------------------------ 3. a refused write does not stop the rest
  {
    PgLink link("test", dsn, 100);
    SubmitImage(&link, FakeImage("2000-01-01 00:00:00", "00000000-0000-0000-0000-0000000000aa", 1, 5, 32, 672), schema);
    SubmitImage(&link, FakeImage(UtcNow(), "00000000-0000-0000-0000-0000000000bb", 1, 5, 32, 672), schema);
    link.Drain(std::chrono::seconds(10));
    Check(link.Failed() == 1 && link.Written() == 1,
          "image without a partition dropped, next one written (" +
              std::to_string(link.Failed()) + " failed, " +
              std::to_string(link.Written()) + " written)");
  }

  // ------------------------------------- 4. connection killed mid-stream
  {
    PgLink link("test", dsn, 100);
    SubmitImage(&link, FakeImage(UtcNow(), "00000000-0000-0000-0000-0000000000cc", 1, 5, 32, 672), schema);
    link.Drain(std::chrono::seconds(10));
    long killed = Count(admin,
        "SELECT count(*) FROM (SELECT pg_terminate_backend(pid) FROM "
        "pg_stat_activity WHERE application_name = 'node_db_test') s");
    SubmitImage(&link, FakeImage(UtcNow(), "00000000-0000-0000-0000-0000000000cc", 1, 5, 32, 672), schema);
    bool drained = link.Drain(std::chrono::seconds(30));
    Check(killed >= 1 && drained && link.Written() == 2 && link.Failed() == 0,
          "write after a killed connection retried and written (" +
              std::to_string(link.Written()) + " written)");
  }

  std::cout << (g_failures == 0 ? "ALL PASSED" : "FAILURES: " + std::to_string(g_failures))
            << std::endl;
  return g_failures == 0 ? 0 : 1;
}
