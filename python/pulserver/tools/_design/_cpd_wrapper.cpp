#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cstdlib>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <cstring> // memset

extern "C" {
#include "cpdlib/misc.h"
#include "cpdlib/udcpd.h"
#include "cpdlib/vdcpd.h"
}

// verbose is defined in misc.c
extern "C" int verbose;

namespace py = pybind11;

/*
  This module provides two wrappers:
    - gen_udcpd(nt, FOVRatio, feasible_points, shapeOpt, verbose, C, mindist_scale)
    - gen_vdcpd(nt, FOVRatio, feasible_points, shapeOpt, verbose, C, mindist_scale, Rmax, vd_exp)

  Both accept feasible_points as a 2D NumPy array (ny x nz). The wrapper forces/coerces
  a Fortran-contiguous double array to match the MATLAB MEX linearization before copying
  to the int mask expected by the underlying C functions. The C functions write a Fortran-
  contiguous int buffer, which we copy into a C-contiguous numpy.int32 array to return.
*/

static void validate_and_build_dims(const py::array_t<double, py::array::f_style | py::array::forcecast>& feasible_f,
                                    int nt,
                                    long dims[3],
                                    ssize_t &ny,
                                    ssize_t &nz)
{
    if (nt <= 0) throw std::invalid_argument("nt must be positive");
    if (feasible_f.ndim() != 2) throw std::invalid_argument("feasible_points must be a 2D array");

    ny = feasible_f.shape(0);
    nz = feasible_f.shape(1);

    dims[Y_DIM] = static_cast<long>(ny);
    dims[Z_DIM] = static_cast<long>(nz);
    dims[T_DIM] = static_cast<long>(nt);

    if (!all_positive(3, dims)) {
        throw std::runtime_error("ERROR: Got empty feasible points array!");
    }
}

py::array_t<int32_t> gen_udcpd_py(int nt,
                                  double FOVRatio,
                                  py::array_t<double, py::array::f_style | py::array::forcecast> feasible_points_f,
                                  long shapeOpt,
                                  int verbose_arg,
                                  double C,
                                  double mindist_scale)
{
    // Validate and get shapes
    ssize_t ny, nz;
    long dims[3];
    validate_and_build_dims(feasible_points_f, nt, dims, ny, nz);

    // Verbosity flag
    verbose = verbose_arg;

    const size_t feasiblePointsSize = static_cast<size_t>(ny) * static_cast<size_t>(nz);
    const double *feasible_ptr = feasible_points_f.data(); // Fortran-contiguous double buffer

    // Allocate and copy feasible points into int buffer
    int *feasiblePoints = static_cast<int*>( malloc(feasiblePointsSize * sizeof(int)) );
    if (!feasiblePoints) throw std::bad_alloc();
    copy_array(feasiblePointsSize, feasible_ptr, feasiblePoints);

    // Allocate Fortran-ordered output buffer expected by the C routine
    const size_t outSize = feasiblePointsSize * static_cast<size_t>(nt);
    static_assert(sizeof(int) == 4, "This wrapper assumes sizeof(int) == 4. Adjust if different.");
    int *out_buf = static_cast<int*>( malloc(outSize * sizeof(int)) );
    if (!out_buf) {
        free(feasiblePoints);
        throw std::bad_alloc();
    }
    std::memset(out_buf, 0, outSize * sizeof(int));

    // Call underlying C function
    genUDCPD(dims, out_buf, feasiblePoints, FOVRatio, C, shapeOpt, mindist_scale);

    // free feasiblePoints temporary
    free(feasiblePoints);

    // Prepare C-contiguous result array (int32)
    std::vector<ssize_t> shape = { static_cast<ssize_t>(ny), static_cast<ssize_t>(nz), static_cast<ssize_t>(nt) };
    py::array_t<int32_t> result(shape); // C-contiguous by default
    int32_t *result_rw = result.mutable_data();

    // Copy from Fortran-ordered out_buf into C-order result
    const size_t ny_sz = static_cast<size_t>(ny);
    const size_t nz_sz = static_cast<size_t>(nz);
    const size_t nt_sz = static_cast<size_t>(nt);
    for (size_t i = 0; i < ny_sz; ++i) {
        for (size_t j = 0; j < nz_sz; ++j) {
            for (size_t k = 0; k < nt_sz; ++k) {
                size_t f_idx = i + j * ny_sz + k * (ny_sz * nz_sz);
                size_t c_idx = i * (nz_sz * nt_sz) + j * nt_sz + k;
                result_rw[c_idx] = static_cast<int32_t>( out_buf[f_idx] );
            }
        }
    }

    free(out_buf);
    return result;
}


py::array_t<int32_t> gen_vdcpd_py(int nt,
                                  double FOVRatio,
                                  py::array_t<double, py::array::f_style | py::array::forcecast> feasible_points_f,
                                  long shapeOpt,
                                  int verbose_arg,
                                  double C,
                                  double mindist_scale,
                                  double Rmax,
                                  double vd_exp)
{
    // Validate and get shapes
    ssize_t ny, nz;
    long dims[3];
    validate_and_build_dims(feasible_points_f, nt, dims, ny, nz);

    // Verbosity flag
    verbose = verbose_arg;

    const size_t feasiblePointsSize = static_cast<size_t>(ny) * static_cast<size_t>(nz);
    const double *feasible_ptr = feasible_points_f.data(); // Fortran-contiguous double buffer

    // Allocate and copy feasible points into int buffer
    int *feasiblePoints = static_cast<int*>( malloc(feasiblePointsSize * sizeof(int)) );
    if (!feasiblePoints) throw std::bad_alloc();
    copy_array(feasiblePointsSize, feasible_ptr, feasiblePoints);

    // Allocate Fortran-ordered output buffer expected by the C routine
    const size_t outSize = feasiblePointsSize * static_cast<size_t>(nt);
    static_assert(sizeof(int) == 4, "This wrapper assumes sizeof(int) == 4. Adjust if different.");
    int *out_buf = static_cast<int*>( malloc(outSize * sizeof(int)) );
    if (!out_buf) {
        free(feasiblePoints);
        throw std::bad_alloc();
    }
    std::memset(out_buf, 0, outSize * sizeof(int));

    // Call underlying C function for variable-density variant
    // Expected signature (as in original MEX):
    // genVDCPD(long dims[3], int *out, int *feasiblePoints, double FOVRatio, double C, long shapeOpt, double mindist_scale, double vd_exp, double Rmax);
    genVDCPD(dims, out_buf, feasiblePoints, FOVRatio, C, shapeOpt, mindist_scale, vd_exp, Rmax);

    // free feasiblePoints temporary
    free(feasiblePoints);

    // Prepare C-contiguous result array (int32)
    std::vector<ssize_t> shape = { static_cast<ssize_t>(ny), static_cast<ssize_t>(nz), static_cast<ssize_t>(nt) };
    py::array_t<int32_t> result(shape); // C-contiguous by default
    int32_t *result_rw = result.mutable_data();

    // Copy from Fortran-ordered out_buf into C-order result
    const size_t ny_sz = static_cast<size_t>(ny);
    const size_t nz_sz = static_cast<size_t>(nz);
    const size_t nt_sz = static_cast<size_t>(nt);
    for (size_t i = 0; i < ny_sz; ++i) {
        for (size_t j = 0; j < nz_sz; ++j) {
            for (size_t k = 0; k < nt_sz; ++k) {
                size_t f_idx = i + j * ny_sz + k * (ny_sz * nz_sz);
                size_t c_idx = i * (nz_sz * nt_sz) + j * nt_sz + k;
                result_rw[c_idx] = static_cast<int32_t>( out_buf[f_idx] );
            }
        }
    }

    free(out_buf);
    return result;
}


PYBIND11_MODULE(_cpd_wrapper, m) {
    m.doc() = "pybind11 bindings for CPD (uniform and variable density)";
    
    m.def("gen_udcpd",
          &gen_udcpd_py,
          py::arg("nt"),
          py::arg("FOVRatio"),
          py::arg("feasible_points"),
          py::arg("shapeOpt"),
          py::arg("verbose"),
          py::arg("C"),
          py::arg("mindist_scale"),
          "Generate complementary Poisson-disc sampling pattern (uniform density). Returns int32 NumPy array shape (ny, nz, nt), C-contiguous.");

    m.def("gen_vdcpd",
          &gen_vdcpd_py,
          py::arg("nt"),
          py::arg("FOVRatio"),
          py::arg("feasible_points"),
          py::arg("shapeOpt"),
          py::arg("verbose"),
          py::arg("C"),
          py::arg("mindist_scale"),
          py::arg("Rmax"),
          py::arg("vd_exp"),
          "Generate variable-density complementary Poisson-disc sampling pattern. Returns int32 NumPy array shape (ny, nz, nt), C-contiguous.");
}