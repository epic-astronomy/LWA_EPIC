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

#ifndef SRC_EX_PY_FUNCS_HPP_
#define SRC_EX_PY_FUNCS_HPP_

#include <glog/logging.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <utility>
#include <variant>

#include "./constants.h"
#include "./orm_types.hpp"
#include "./types.hpp"

namespace py = pybind11;
using namespace py::literals;

template <typename T>
using np_array = py::array_t<T, py::array::c_style | py::array::forcecast>;

// prolate spheroid eigen (characteristic) value
// m,n mode:  parameters. n>=m
// c: spheroidal parameter
double ProSphCv(const py::module_& scipy_spl, int m, int n, float c) {
  return scipy_spl.attr("pro_cv")(m, n, c).cast<double>();
}

// Prolate spheroidal angular function pro_ang1 for precomputed characteristic
// value
double ProSphAng1Cv(const py::module_& scipy_spl, int m, int n, float c,
                    float cv, float x) {
  for (auto it : scipy_spl.attr("pro_ang1_cv")(m, n, c, cv, x)) {
    // unsure how to access the first element of the tuple
    // .first and [0] throws a seg fault.
    return it.cast<double>();
  }

  return nan("");
}

/**
 * @brief Create a generic 2D texture for a prolate spheroidal kernel
 *
 * The output texture represents only the u>=0 and v>=0 part of the kernel.
 * Because the kernel is symmetric, one can obtain the kernel value at negative
 * coordinates simply by passing in their absolute values. That means u and v
 * must be normalized with half the support size. Furthermore, the dim parameter
 * does not need to equal the half support size. As long as it's a large enough
 * number, for example, 512, tex2D should provide reasonably accurate values
 * with interpolation.
 *
 * @tparam T Data type for the texture
 * @param m Mode parameter m
 * @param n Mode parameter n
 * @param alpha Order of the wave functions
 * @param out Output array to store the texture
 * @param dim Size of the texture
 * @param c Spheroidal parameter
 */
template <typename T>
void ProlateSpheroidalToTex2D(int m, int n, float alpha, T* out, int dim,
                              float c = 5.356 * PI / 2.0) {
  py::gil_scoped_acquire acquire;
  auto scipy_spl = py::module_::import("scipy.special");
  auto cv = ProSphCv(scipy_spl, m, n, c);
  for (auto i = dim - 1; i >= 0; --i) {  // for a left-bottom origin
    for (auto j = 0; j < dim; ++j) {
      T u = T(i) / T(dim);
      T v = T(j) / T(dim);

      out[i * dim + j] = ::pow((1 - u * u), alpha) * ::pow((1 - v * v), alpha) *
                         ProSphAng1Cv(scipy_spl, m, n, c, cv, u) *
                         ProSphAng1Cv(scipy_spl, m, n, c, cv, v);

      // if (i == 0) {
      //     std::cout << out[j] << ",";
      // }

      if (i == dim - 1 || j == dim - 1) {
        out[i * dim + j] = 0;
      }
    }
  }
}

template <typename T>
void GaussianToTex2D(T* out, float sigma, int dim) {
  for (auto i = dim - 1; i >= 0; --i) {  // for a left-bottom origin
    for (auto j = 0; j < dim; ++j) {
      T u = T(i) / T(dim);
      T v = T(j) / T(dim);

      out[i * dim + j] = exp(-(u * u + v * v) / (2 * sigma * sigma));

      if (i == 0) {
        std::cout << out[j] << ",";
      }

      if (i == dim - 1 || j == dim - 1) {
        out[i * dim + j] = 0;
      }
    }
  }
}

template <typename T>
void ProlateSpheroidalTotex1D(int m, int n, float alpha, T* out, int dim,
                              float c = 5.356 * PI / 2.0) {
  py::gil_scoped_acquire acquire;
  auto scipy_spl = py::module_::import("scipy.special");
  float half_dim = dim / 2;
  auto cv = ProSphCv(scipy_spl, m, n, c);
  for (auto i = 0; i < dim; ++i) {
    T u = T(i) / dim;
    out[i] =
        ::pow((1 - u * u), alpha) * ProSphAng1Cv(scipy_spl, m, n, c, cv, u);
  }
}

/**
 * @brief Generate antenna locations for the specified channels
 *
 * @tparam T Output data precision
 * @param out_ptr Pointer to the output data buffer
 * @param grid_size 1D size of the image
 * @param grid_resolution Image resolution in degrees
 * @return Sampling length
 */
template <typename T>
double GetLwasvLocs(T* out_ptr, int grid_size, double grid_resolution) {
  py::gil_scoped_acquire acquire;
  auto np = py::module_::import("numpy");
  VLOG(2) << "after numpy\n";
  VLOG(2) << "Grid size: " << grid_size << " grid res: " << grid_resolution;

  auto ret_dict = py::module_::import("epic_utils")
                      .attr("gen_loc_lwasv")(grid_size, grid_resolution);
  VLOG(2) << "Generated locations.";

  double delta = ret_dict["delta"].cast<double>();
  // dimensions: NSTANDS, 3
  auto locs_arr =
      ret_dict["locations"]
          .cast<
              py::array_t<double, py::array::c_style | py::array::forcecast>>();
  auto loc_ptr = static_cast<double*>(locs_arr.request().ptr);

  for (auto i = 0; i < LWA_SV_NSTANDS * 3; ++i) {
    out_ptr[i] = static_cast<T>(loc_ptr[i]);
  }
  return delta;
}

/**
 * @brief Generate phases for the specified channels
 *
 * @tparam T Output data precision
 * @param out_ptr Pointer to the output buffer
 * @param nchan Number of channels
 * @param chan0 Initial channel number
 */
template <typename T>
void GetLwasvPhases(T* out_ptr, int nchan, int chan0) {
  py::gil_scoped_acquire acquire;
  auto np = py::module_::import("numpy");
  auto phases_arr =
      py::module_::import("epic_utils")
          .attr("gen_phases_lwasv")(nchan, chan0)
          .cast<py::array_t<std::complex<double>,
                            py::array::c_style | py::array::forcecast>>();
  VLOG(3) << "Received phases data";
  // return

  auto phases_ptr = static_cast<double*>(phases_arr.request().ptr);
  int nvalues = LWA_SV_NSTANDS * LWA_SV_NPOLS * nchan * 2 /*complex*/;

  for (int i = 0; i < nvalues; ++i) {
    out_ptr[i] = static_cast<T>(phases_ptr[i]);
  }
}

/**
 * @brief Generate a 40 ms gulp from a TBN file
 *
 * @tparam T Output data precision
 * @param out_ptr Pointer to the output buffer
 */
template <typename T>
void Get40msGulp(T* out_ptr) {
  VLOG(3) << "Grabbing a 40ms gulp";
  py::gil_scoped_acquire acquire;
  auto gulp_dict = py::module_::import("epic_utils").attr("get_40ms_gulp")();

  auto meta_arr =
      gulp_dict["meta"]
          .cast<
              py::array_t<double, py::array::c_style | py::array::forcecast>>();
  auto data_arr =
      gulp_dict["data"]
          .cast<py::array_t<std::complex<double>,
                            py::array::c_style | py::array::forcecast>>();

  // tstart, chan0, size, ntime, nchan, nstand, npol
  auto meta_ptr = static_cast<double*>(meta_arr.request().ptr);

  // dimensions ntime, nchan, nstand, npol
  auto data_ptr = static_cast<double*>(data_arr.request().ptr);

  // copy it to the out_ptr
  auto out_nib2 = reinterpret_cast<CNib*>(out_ptr);

  VLOG(3) << "Copying " << meta_ptr[2] << " elements into the output array";
  int ncomplex_vals = meta_ptr[2] / 2;
  CHECK(static_cast<int>(meta_ptr[2]) % 2 == 0)
      << "Invalid gulp array. The total number of elements is an odd number.";
  for (auto i = 0; i < ncomplex_vals; ++i) {
    out_nib2[i].re = data_ptr[2 * i];
    out_nib2[i].im = data_ptr[2 * i + 1];
    // if(i==1){
    //     VLOG(3)<<"GUlp data: "<<data_ptr
    // }
  }
}

/**
 * @brief Saves image to the disk as a FITS file
 *
 * @tparam T Input data precision
 * @param grid_size 1D size of the image
 * @param nchan Number of channels
 * @param data Pointer to the image data
 * @param filename Output filename
 * @param metadata Image metadata object
 */
template <typename T>
void __attribute__((visibility("hidden")))
SaveImageToDisk(size_t grid_size, size_t nchan, T* data, std::string filename,
                const dict_t& metadata) {
  py::gil_scoped_acquire acquire;
  VLOG(2) << "type of output data type: " << sizeof(T);
  auto result = py::array_t<T>(grid_size * grid_size * nchan * 4, data);

  py::dict meta_dict;
  for (auto it = metadata.begin(); it != metadata.end(); ++it) {
    std::visit([&](auto&& v) { meta_dict[it->first.c_str()] = v; }, it->second);
  }

  VLOG(2) << "Sending to saver";
  auto utils =
      py::module_::import("epic_utils")
          .attr("save_output")(result, grid_size, nchan, filename, meta_dict);
  // for (int i = 0; i < 10; ++i) {
  //     std::cout << data[i] << std::endl;
  // }
}

/**
 * @brief Generate a correction grid. This is a channel-wise squared iFFT of an
 * average of all, optionally, oversampled antenna kernels.
 *
 * @tparam T Precision of the grid
 * @param correction_kernel Channel-wise average of the oversampled antenna
 * kernels
 * @param out_correction_grid Output correction grid
 * @param grid_size Size of the grid
 * @param support Support size
 * @param nchan Number of channels
 * @param oversample Number of time to oversample the image grid.
 */
template <typename T>
void GetCorrectionGrid(T* correction_kernel, T* out_correction_grid,
                       int grid_size, int support, int nchan,
                       int oversample = 4) {
  py::gil_scoped_acquire acquire;
  auto corr_ker_arr =
      py::array_t<float>(support * support * nchan, correction_kernel);
  // auto corr_grid_arr = py::array_t<T>(grid_size * grid_size * nchan,
  // out_correction_grid);

  auto corr_grid_res =
      py::module_::import("epic_utils")
          .attr("get_correction_grid")(corr_ker_arr, grid_size, support, nchan,
                                       oversample);

  auto corr_grid_arr = corr_grid_res.cast<
      py::array_t<double, py::array::c_style | py::array::forcecast>>();

  auto* corr_grid_ptr = static_cast<double*>(corr_grid_arr.request().ptr);

  for (int i = 0; i < grid_size * grid_size * nchan; ++i) {
    out_correction_grid[i] = corr_grid_ptr[i];
  }
}

/**
 * @brief Get time from unix epoch object from ADP control
 *
 * @return double
 */
double GetAdpTimeFromUnixEpoch() {
  py::gil_scoped_acquire acquire;
  return py::module_::import("epic_utils")
      .attr("get_ADP_time_from_unix_epoch")()
      .cast<double>();
}

/**
 * @brief Convert a utc timestamp into time from unix epoch
 *
 * @param utcstart UTC timestamp string
 * @return double
 */
double GetTimeFromUnixEpoch(std::string utcstart) {
  py::gil_scoped_acquire acquire;
  return py::module_::import("epic_utils")
      .attr("get_time_from_unix_epoch")(utcstart)
      .cast<double>();
}

/**
 * @brief Generate a random uuid
 *
 * @return std::string
 */
std::string GetRandomUuid() {
  py::gil_scoped_acquire acquire;
  return py::module_::import("epic_utils")
      .attr("get_random_uuid")()
      .cast<std::string>();
}

/**
 * @brief Convert a time tag into a Postgres formatted timestamp
 *
 * @param time_tag Time tag
 * @param img_len_ms Length of the accumulation in ms
 * @return std::string
 */
std::string Meta2PgTime(uint64_t time_tag, double img_len_ms) {
  py::gil_scoped_acquire acquire;
  return py::module_::import("epic_utils")
      .attr("meta2pgtime")(time_tag, img_len_ms)
      .cast<std::string>();
}

EpicPixelTableMetaRows get_watch_indices(uint64_t seq_start_no, int grid_size,
                                         float grid_res, float elev_limit,
                                         std::string watchdog_endpoint) {
  py::gil_scoped_acquire acquire;
  auto ret_dict =
      py::module_::import("pixel_extractor")
          .attr("get_pixel_indices")(seq_start_no, grid_size, grid_res,
                                     elev_limit, watchdog_endpoint);

  // WORKAROUND: Simply doing the .cast<int>() results in a core dumped error.
  // But fetching it as an array seems to work fine
  int nsrc = (static_cast<int*>(
      ret_dict["nsrc"].cast<np_array<int>>().request().ptr))[0];
  auto ncoords = (static_cast<int*>(
      ret_dict["ncoords"].cast<np_array<int>>().request().ptr))[0];
  auto kernel_dim = (static_cast<int*>(
      ret_dict["kernel_dim"].cast<np_array<int>>().request().ptr))[0];
  VLOG(3) << nsrc << " " << ncoords << " " << kernel_dim;

  EpicPixelTableMetaRows watch_indices(ncoords, nsrc, kernel_dim);

  // create accessors for different indices
  // pixel coords
  VLOG(3) << "Reading pixel indices";
  auto* x_idx = static_cast<double*>(
      ret_dict["pix_x"].cast<np_array<double>>().request().ptr);
  auto* y_idx = static_cast<double*>(
      ret_dict["pix_y"].cast<np_array<double>>().request().ptr);

  // pixel lm values
  VLOG(3) << "Reading lm coords";
  auto* l_idx = static_cast<double*>(
      ret_dict["l"].cast<np_array<double>>().request().ptr);
  auto* m_idx = static_cast<double*>(
      ret_dict["m"].cast<np_array<double>>().request().ptr);

  // pixel offset
  VLOG(3) << "Reading offsets";
  auto* pix_ofst_x = static_cast<double*>(
      ret_dict["pix_ofst_x"].cast<np_array<double>>().request().ptr);
  auto* pix_ofst_y = static_cast<double*>(
      ret_dict["pix_ofst_y"].cast<np_array<double>>().request().ptr);

  for (int i = 0; i < ncoords; ++i) {
    watch_indices.pixel_coords[i] = std::pair<int, int>(x_idx[i], y_idx[i]);
    watch_indices.pixel_lm[i] = std::pair<float, float>(l_idx[i], m_idx[i]);
    watch_indices.pixel_offst[i] =
        std::pair<int, int>(pix_ofst_x[i], pix_ofst_y[i]);
    VLOG(3) << i << " " << x_idx[i] << " " << y_idx[i] << " " << l_idx[i] << " "
            << m_idx[i] << " " << pix_ofst_x[i] << " " << pix_ofst_y[i] << " "
            << watch_indices.pixel_coords.size() << " "
            << watch_indices.pixel_lm.size() << " "
            << watch_indices.pixel_offst.size();
  }

  // copy the src ids
  VLOG(3) << "Reading source ids";
  auto* src_ids = static_cast<double*>(
      ret_dict["src_ids"].cast<np_array<double>>().request().ptr);
  VLOG(3) << "Adding";
  for (int i = 0; i < nsrc; ++i) {
    watch_indices.source_ids[i] = src_ids[i];
  }
  VLOG(3) << "Setting the meta version";
  unsigned int _seed;
  watch_indices.meta_version = rand_r(&_seed);
  VLOG(3) << "Transforming the coords";
  watch_indices.TransformPixCoords(grid_size, grid_size);

  VLOG(3) << "Returning watch indices";

  return watch_indices;
}

#endif  // SRC_EX_PY_FUNCS_HPP_
