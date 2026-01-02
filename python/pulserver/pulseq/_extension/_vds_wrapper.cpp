#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <vector>
#include <cstdlib>

#include "vdslib/vds.h"

namespace py = pybind11;

// Low-level wrapper that calls the C calc_vds function as-is.
// Units follow the original C implementation (G/cm, G/cm/s, seconds, /cm).
// Returns a dict with xgrad, ygrad (1D float64 arrays) and numgrad (int).
py::dict calc_vds_raw(
    double slewmax,        // [G/cm/s]
    double gradmax,        // [G/cm]
    double Tgsample,       // [s]
    double Tdsample,       // [s]
    int    Ninterleaves,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &fov_coeffs,
    double krmax,          // [1/cm]
    int    ngmax           // max gradient samples to allocate
) {
    // Validate inputs
    if (Ninterleaves <= 0) {
        throw std::invalid_argument("Ninterleaves must be > 0");
    }
    if (ngmax <= 0) {
        throw std::invalid_argument("ngmax must be > 0");
    }
    if (fov_coeffs.ndim() != 1) {
        throw std::invalid_argument("fov must be a 1-D array of coefficients");
    }

    // Extract coeffs
    auto buf = fov_coeffs.request();
    const int numfov = static_cast<int>(buf.shape[0]);
    if (numfov <= 0) {
        throw std::invalid_argument("fov must contain at least one coefficient");
    }
    const double* fov_ptr = static_cast<const double*>(buf.ptr);

    // Outputs from C
    double* xgrad_ptr = nullptr;
    double* ygrad_ptr = nullptr;
    int numgrad = 0;

    // Call the C API
    calc_vds(slewmax, gradmax, Tgsample, Tdsample,
             Ninterleaves,
             fov_ptr, numfov,
             krmax, ngmax,
             &xgrad_ptr, &ygrad_ptr, &numgrad);

    if (numgrad <= 0 || xgrad_ptr == nullptr || ygrad_ptr == nullptr) {
        // Ensure we don't leak in case one pointer was allocated
        if (xgrad_ptr) std::free(xgrad_ptr);
        if (ygrad_ptr) std::free(ygrad_ptr);
        throw std::runtime_error("calc_vds returned no data (numgrad <= 0 or null pointers)");
    }

    // Wrap into NumPy arrays by copying, then free C buffers
    py::array_t<double> xgrad({numgrad});
    py::array_t<double> ygrad({numgrad});

    std::memcpy(xgrad.mutable_data(), xgrad_ptr, sizeof(double) * static_cast<size_t>(numgrad));
    std::memcpy(ygrad.mutable_data(), ygrad_ptr, sizeof(double) * static_cast<size_t>(numgrad));

    std::free(xgrad_ptr);
    std::free(ygrad_ptr);

    // Return dictionary to be future-proof (you can add more fields later)
    py::dict out;
    out["xgrad"] = std::move(xgrad); // [G/cm]
    out["ygrad"] = std::move(ygrad); // [G/cm]
    out["numgrad"] = numgrad;
    return out;
}

PYBIND11_MODULE(_vds_wrapper, m) {
    m.doc() = "Pybind11 wrapper for Hargreaves VDS (calc_vds) C implementation. "
              "This is a raw interface with original units (G/cm, G/cm/s).";
    m.def("calc_vds_raw", &calc_vds_raw,
          py::arg("slewmax"),
          py::arg("gradmax"),
          py::arg("Tgsample"),
          py::arg("Tdsample"),
          py::arg("Ninterleaves"),
          py::arg("fov"),
          py::arg("krmax"),
          py::arg("ngmax")
    );
}