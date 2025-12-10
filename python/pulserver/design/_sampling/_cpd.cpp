#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

extern "C" {
#include "cpdlib/misc.h"
#include "cpdlib/udcpd.h"
#include "cpdlib/vdcpd.h"
}

// verbose is defined in misc.c
extern "C" int verbose;

namespace py = pybind11;

static void check_feasible(const py::array &arr) {
    if (arr.ndim() != 2) {
        throw std::invalid_argument("feasible_points must be 2D with shape (ny, nz)");
    }
}

py::array gen_udcpd(py::array feasible_points,
                    long nt,
                    double fov_ratio,
                    double C,
                    int shape_opt,
                    double mindist_scale = 1.0,
                    int verbose_flag = 0) {
    check_feasible(feasible_points);
    py::array_t<int, py::array::c_style | py::array::forcecast> feas(feasible_points);
    const long ny = feas.shape(0);
    const long nz = feas.shape(1);

    long dims[DIMS];
    dims[Y_DIM] = ny;
    dims[Z_DIM] = nz;
    dims[T_DIM] = nt;

    py::array_t<int> out({ny, nz, nt});  // y-fastest, then z, then t
    verbose = verbose_flag;

    genUDCPD(dims,
             out.mutable_data(),
             feas.data(),
             fov_ratio,
             C,
             shape_opt,
             mindist_scale);

    return out;
}

py::array gen_vdcpd(py::array feasible_points,
                    long nt,
                    double fov_ratio,
                    double C,
                    int shape_opt,
                    double mindist_scale,
                    double vd_exp,
                    int maxR,
                    int verbose_flag = 0) {
    check_feasible(feasible_points);
    py::array_t<int, py::array::c_style | py::array::forcecast> feas(feasible_points);
    const long ny = feas.shape(0);
    const long nz = feas.shape(1);

    long dims[DIMS];
    dims[Y_DIM] = ny;
    dims[Z_DIM] = nz;
    dims[T_DIM] = nt;

    py::array_t<int> out({ny, nz, nt});  // y-fastest, then z, then t
    verbose = verbose_flag;

    genVDCPD(dims,
             out.mutable_data(),
             feas.data(),
             static_cast<float>(fov_ratio),
             static_cast<float>(C),
             shape_opt,
             static_cast<float>(mindist_scale),
             static_cast<float>(vd_exp),
             maxR);

    return out;
}

PYBIND11_MODULE(_cpd, m) {
    m.doc() = "pybind11 bindings for CPD (uniform and variable density)";

    m.def("gen_udcpd", &gen_udcpd,
          py::arg("feasible_points"),
          py::arg("nt"),
          py::arg("fov_ratio"),
          py::arg("C"),
          py::arg("shape_opt"),
          py::arg("mindist_scale") = 1.0,
          py::arg("verbose") = 0);

    m.def("gen_vdcpd", &gen_vdcpd,
          py::arg("feasible_points"),
          py::arg("nt"),
          py::arg("fov_ratio"),
          py::arg("C"),
          py::arg("shape_opt"),
          py::arg("mindist_scale"),
          py::arg("vd_exp"),
          py::arg("maxR"),
          py::arg("verbose") = 0);
}