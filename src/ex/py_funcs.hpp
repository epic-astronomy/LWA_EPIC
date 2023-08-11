#ifndef PY_FUNCS
#define PY_FUNCS

#include "constants.h"
#include "types.hpp"
#include <chrono>
#include <cmath>
#include <glog/logging.h>
#include <iostream>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <variant>

namespace py = pybind11;
using namespace py::literals;

// prolate spheroid eigen (characteristic) value
// m,n mode:  parameters. n>=m
// c: spheroidal parameter
double
pro_sph_cv(py::module_& scipy_spl, int m, int n, float c)
{
    return scipy_spl.attr("pro_cv")(m, n, c).cast<double>();
}

// Prolate spheroidal angular function pro_ang1 for precomputed characteristic value
double
pro_sph_ang1_cv(py::module_& scipy_spl, int m, int n, float c, float cv, float x)
{
    for (auto it : scipy_spl.attr("pro_ang1_cv")(m, n, c, cv, x)) {
        // unsure how to access the first element of the tuple
        // .first and [0] throws a seg fault.
        return it.cast<double>();
    }
}

/**
 * @brief Create a generic 2D texture for a prolate spheroidal kernel
 *
 * The output texture represents only the u>=0 and v>=0 part of the kernel.
 * Because the kernel is symmetric, one can obtain the kernel value at negative coordinates
 * simply by passing in their absolute values. That means u and v must be normalized
 * with half the support size. Furthermore, the dim parameter does not need to equal
 * the half support size. As long as it's a large enough number, for example, 512,
 * tex2D should provide reasonably accurate values with interpolation.
 *
 * @tparam T Data type for the texture
 * @param m Mode parameter m
 * @param n Mode parameter n
 * @param alpha Order of the wave functions
 * @param out Output array to store the texture
 * @param dim Size of the texture
 * @param c Spheroidal parameter
 */
template<typename T>
void
prolate_spheroidal_to_tex2D(int m, int n, float alpha, T* out, int dim, float c = 5.356 * PI / 2.0)
{
    py::gil_scoped_acquire acquire;
    auto scipy_spl = py::module_::import("scipy.special");
    auto cv = pro_sph_cv(scipy_spl, m, n, c);
    for (auto i = dim - 1; i >= 0; --i) { // for a left-bottom origin
        for (auto j = 0; j < dim; ++j) {
            T u = T(i) / T(dim);
            T v = T(j) / T(dim);

            out[i * dim + j] = ::pow((1 - u * u), alpha) * ::pow((1 - v * v), alpha) * pro_sph_ang1_cv(scipy_spl, m, n, c, cv, u) * pro_sph_ang1_cv(scipy_spl, m, n, c, cv, v);

            // if (i == 0) {
            //     std::cout << out[j] << ",";
            // }

            if (i == dim - 1 || j == dim - 1) {
                out[i * dim + j] = 0;
            }
        }
    }
}

template<typename T>
void
gaussian_to_tex2D(T* out, float sigma, int dim)
{
    for (auto i = dim - 1; i >= 0; --i) { // for a left-bottom origin
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

template<typename T>
void
prolate_spheroidal_to_tex1D(int m, int n, float alpha, T* out, int dim, float c = 5.356 * PI / 2.0)
{
    py::gil_scoped_acquire acquire;
    auto scipy_spl = py::module_::import("scipy.special");
    float half_dim = dim / 2;
    auto cv = pro_sph_cv(scipy_spl, m, n, c);
    for (auto i = 0; i < dim; ++i) {
        T u = T(i) / dim;
        out[i] = ::pow((1 - u * u), alpha) * pro_sph_ang1_cv(scipy_spl, m, n, c, cv, u);
    }
}

template<typename T>
double
get_lwasv_locs(T* out_ptr, int grid_size, double grid_resolution)
{
    py::gil_scoped_acquire acquire;
    auto np = py::module_::import("numpy");
    LOG(INFO) << "after numpy\n";
    LOG(INFO) << "Grid size: " << grid_size << " grid res: " << grid_resolution;

    auto ret_dict = py::module_::import("epic_utils")
                      .attr("gen_loc_lwasv")(grid_size, grid_resolution);
    LOG(INFO) << "Generated locations.";

    double delta = ret_dict["delta"].cast<double>();
    // dimensions: NSTANDS, 3
    auto locs_arr = ret_dict["locations"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    auto loc_ptr = static_cast<double*>(locs_arr.request().ptr);

    for (auto i = 0; i < LWA_SV_NSTANDS * 3; ++i) {
        out_ptr[i] = static_cast<T>(loc_ptr[i]);
    }
    std::cout << "returning antpos\n";
    return delta;
}

template<typename T>
void
get_lwasv_phases(T* out_ptr, int nchan, int chan0)
{
    py::gil_scoped_acquire acquire;
    auto np = py::module_::import("numpy");
    auto phases_arr = py::module_::import("epic_utils")
                        .attr("gen_phases_lwasv")(nchan, chan0)
                        .cast<py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>>();
    DLOG(INFO) << "Received phases data";
    // return

    auto phases_ptr = static_cast<double*>(phases_arr.request().ptr);
    int nvalues = LWA_SV_NSTANDS * LWA_SV_NPOLS * nchan * 2 /*complex*/;

    for (int i = 0; i < nvalues; ++i) {
        out_ptr[i] = static_cast<T>(phases_ptr[i]);
    }
}

template<typename T>
void
get_40ms_gulp(T* out_ptr)
{
    VLOG(3) << "Grabbing a 40ms gulp";
    py::gil_scoped_acquire acquire;
    auto gulp_dict = py::module_::import("epic_utils")
                       .attr("get_40ms_gulp")();

    auto meta_arr = gulp_dict["meta"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    auto data_arr = gulp_dict["data"].cast<py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>>();

    // tstart, chan0, size, ntime, nchan, nstand, npol
    auto meta_ptr = static_cast<double*>(meta_arr.request().ptr);

    // dimensions ntime, nchan, nstand, npol
    auto data_ptr = static_cast<double*>(data_arr.request().ptr);

    // copy it to the out_ptr
    auto out_nib2 = reinterpret_cast<cnib*>(out_ptr);

    VLOG(3) << "Copying " << meta_ptr[2] << " elements into the output array";
    int ncomplex_vals = meta_ptr[2] / 2;
    CHECK(int(meta_ptr[2]) % 2 == 0) << "Invalid gulp array. The total number of elements is an odd number.";
    size_t i;
    for (i = 0; i < ncomplex_vals; ++i) {
        out_nib2[i].re = data_ptr[2 * i];
        out_nib2[i].im = data_ptr[2 * i + 1];
        // if(i==1){
        //     VLOG(3)<<"GUlp data: "<<data_ptr
        // }
    }

    VLOG(3) << "Copied: " << i << " complex vals";
}

template<typename T>
void
save_image(size_t grid_size, size_t nchan, T* data, std::string filename, dict_t& metadata)
{
    py::gil_scoped_acquire acquire;
    VLOG(3) << "type of output data type: " << sizeof(T);
    auto result = py::array_t<T>(grid_size * grid_size * nchan * 4, data);

    py::dict meta_dict;
    for (auto it = metadata.begin(); it != metadata.end(); ++it) {

        std::visit([&](auto&& v) {
            meta_dict[it->first.c_str()] = v;
        },
                   it->second);
    }

    LOG(INFO) << "Sending to saver";
    auto utils = py::module_::import("epic_utils").attr("save_output")(result, grid_size, nchan, filename, meta_dict);
    // for (int i = 0; i < 10; ++i) {
    //     std::cout << data[i] << std::endl;
    // }
}

template<typename T>
void
get_correction_grid(T* correction_kernel, T* out_correction_grid, int grid_size, int support, int nchan, int oversample = 4)
{
    py::gil_scoped_acquire acquire;
    auto corr_ker_arr = py::array_t<float>(support * support * nchan, correction_kernel);
    // auto corr_grid_arr = py::array_t<T>(grid_size * grid_size * nchan, out_correction_grid);

    auto corr_grid_res = py::module_::import("epic_utils").attr("get_correction_grid")(corr_ker_arr, grid_size, support, nchan, oversample);

    auto corr_grid_arr = corr_grid_res.cast<
      py::array_t<
        double,
        py::array::c_style | py::array::forcecast>>();

    auto* corr_grid_ptr = static_cast<double*>(corr_grid_arr.request().ptr);

    for (int i = 0; i < grid_size * grid_size * nchan; ++i) {
        out_correction_grid[i] = corr_grid_ptr[i];
    }
}

double
get_ADP_time_from_unix_epoch()
{
    py::gil_scoped_acquire acquire;
    return py::module_::import("epic_utils")
      .attr("get_ADP_time_from_unix_epoch")()
      .cast<double>();
}

double
get_time_from_unix_epoch(std::string utcstart)
{
    py::gil_scoped_acquire acquire;
    return py::module_::import("epic_utils")
      .attr("get_time_from_unix_epoch")(utcstart)
      .cast<double>();
}

std::string get_random_uuid(){
    py::gil_scoped_acquire acquire;
    return py::module_::import("epic_utils").attr("get_random_uuid")().cast<std::string>();
}

#endif // PY_FUNCS