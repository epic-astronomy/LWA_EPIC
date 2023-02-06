#ifndef PY_FUNCS
#define PY_FUNCS

#include "constants.h"
#include <cmath>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace py::literals;

// auto scipy_spl = py::module_::import("scipy.special");

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
        // .first and [0] throw a seg fault.
        return it.cast<double>();
    }
}

// Create a generic 2D texture for a prolate spheroidal kernel
// The output texture represents only the u>=0 and v>=0 part of the kernel.
// Because the kernel is symmetric, one can obtain the kernel value at negative coordinates
// simply by passing in their absolute values. That means u and v must be normalized
// with half the support size. Furthermore, the dim parameter does not need to equal
// the half support size. As long as it's a large enough number, for example, 512,
// tex2D should provide reasonably accurate values with interpolation.
template<typename T>
void
prolate_spheroidal_to_tex2D(int m, int n, float alpha, T* out, int dim, float c = 5.356 * PI / 2.0)
{
    // py::initialize_interpreter(); // guard{};

    auto scipy_spl = py::module_::import("scipy.special");
    // float half_dim = dim / 2;
    auto cv = pro_sph_cv(scipy_spl, m, n, c);
    for (auto i = dim-1; i >=0; --i) { // for a left-bottom origin
        for (auto j = 0; j < dim; ++j) {
            T u = T(i) / dim;
            T v = T(j) / dim;

            // std::cout<<i<<" "<<j<<std::endl;

            out[i * dim + j] = ::pow((1 - u * u), alpha) * ::pow((1 - v * v), alpha) * pro_sph_ang1_cv(scipy_spl, m, n, c, cv, u) * pro_sph_ang1_cv(scipy_spl, m, n, c, cv, v);

            if(i==0){
                std::cout<<out[j]<<",";
            }

            if(i==dim-1 || j==dim-1){
                out[i * dim + j]=0;
            }
        }
    }
    
    // py::finalize_interpreter(); // guard{};
}

template<typename T>
void
prolate_spheroidal_to_tex1D(int m, int n, float alpha, T* out, int dim, float c = 5.356 * PI / 2.0)
{
    // py::initialize_interpreter();
    auto scipy_spl = py::module_::import("scipy.special");
    float half_dim = dim / 2;
    auto cv = pro_sph_cv(scipy_spl, m, n, c);
    for (auto i = 0; i < dim; ++i) {
        T u = T(i) / dim;
        out[i] = ::pow((1 - u * u), alpha) * pro_sph_ang1_cv(scipy_spl, m, n, c, cv, u);
    }
    // py::finalize_interpreter(); // guard{};
}

template<typename T>
double
get_lwasv_locs(T* out_ptr, int grid_size, double grid_resolution)
{
    // py::initialize_interpreter();
    auto np = py::module_::import("numpy");
    std::cout << "after numpy\n";

    auto ret_dict = py::module_::import("epic_utils")
                      .attr("gen_loc_lwasv")(grid_size, grid_resolution);

    double delta = ret_dict["delta"].cast<double>();
    // dimensions: NSTANDS, 3
    auto locs_arr = ret_dict["locations"].cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    auto loc_ptr = static_cast<double*>(locs_arr.request().ptr);

    for (auto i = 0; i < LWA_SV_NSTANDS * 3; ++i) {
        out_ptr[i] = static_cast<T>(loc_ptr[i]);
    }
    // py::finalize_interpreter(); // guard{};
    std::cout << "returning antpos\n";
    return delta;
}

template<typename T>
void
get_lwasv_phases(T* out_ptr, int nchan, int chan0)
{
    // py::initialize_interpreter();
    std::cout << "before numpy\n";

    auto np = py::module_::import("numpy");
    std::cout << "after numpy\n";
    // dimensions: chan, pol, ant, real_imag
    auto phases_arr = py::module_::import("epic_utils")
                        .attr("gen_phases_lwasv")(nchan, chan0)
                        .cast<py::array_t<std::complex<double>, py::array::c_style | py::array::forcecast>>();
    std::cout << "Received phases\n";
    // return

    auto phases_ptr = static_cast<double*>(phases_arr.request().ptr);
    int nvalues = LWA_SV_NSTANDS * LWA_SV_NPOLS * nchan * 2 /*complex*/;

    for (int i = 0; i < nvalues; ++i) {
        out_ptr[i] = static_cast<T>(phases_ptr[i]);
        // if(i<10 && i%2==0){
        //     std::cout<<phases_ptr[i]<<" "<<phases_ptr[i+1]<<std::endl;
        // }
    }
    // py::finalize_interpreter(); // guard{};
}

template<typename T>
void save_image(size_t grid_size, size_t nchan, T* data, std::string filename){
    auto result = py::array_t<T>(grid_size*grid_size*nchan, data);
    auto utils = py::module_::import("epic_utils").attr("save_output")(result, grid_size, nchan, filename);
    for(int i=0;i<10;++i){
        std::cout<<data[i]<<std::endl;
    }
}

// template pro_sp_to_tex2D<float>;
// template pro_sp_to_tex1D<float>;
// template pro_sp_to_tex2D<double>;
// template pro_sp_to_tex1D<double>;

#endif // PY_FUNCS