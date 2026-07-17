// Thin pybind11 bindings over a vendored MRArbGrad core
// (see cxx/arbgrad/vendor/mrarbgrad/NOTICE.md).
//
// Deliberately narrow surface: each binding produces exactly one base-shot
// gradient waveform (nStack fixed to 1, iAcq fixed to 0, standard
// non-rotated orientation) plus the number of shots required for Nyquist
// coverage. Shot-to-shot rotation, golden-angle ordering, and gradient
// reversal are Python-side policy (see python/pulserver/arbgrad/) — this
// wrapper never touches the upstream solver's global tuning state.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "traj/Rosette.h"
#include "traj/VDSpiral.h"

namespace py = pybind11;

namespace {

py::array_t<double> vv3_to_numpy(const vv3 &samples) {
    const auto n = static_cast<py::ssize_t>(samples.size());
    py::array_t<double> arr({n, static_cast<py::ssize_t>(3)});
    auto buf = arr.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < n; ++i) {
        buf(i, 0) = samples[static_cast<size_t>(i)].x;
        buf(i, 1) = samples[static_cast<size_t>(i)].y;
        buf(i, 2) = samples[static_cast<size_t>(i)].z;
    }
    return arr;
}

py::array_t<double> v3_to_numpy(const v3 &value) {
    py::array_t<double> arr(3);
    auto buf = arr.mutable_unchecked<1>();
    buf(0) = value.x;
    buf(1) = value.y;
    buf(2) = value.z;
    return arr;
}

// Constructs `TrajT` with nStack fixed to 1 and returns the single base-shot
// waveform (iAcq=0): (k0, gradient, n_shots).
template <typename TrajT, typename... ShapeArgs>
py::tuple base_waveform(const MrTraj::GeoPara &geo, const MrTraj::GradPara &grad, ShapeArgs &&...shape_args) {
    TrajT traj(geo, grad, static_cast<i64>(1), std::forward<ShapeArgs>(shape_args)...);

    v3 m0;
    vv3 gro;
    if (!traj.getGrad(&m0, &gro, 0)) {
        throw std::runtime_error("MRArbGrad base-waveform computation failed");
    }

    return py::make_tuple(v3_to_numpy(m0), vv3_to_numpy(gro), static_cast<int64_t>(traj.getNAcq()));
}

} // namespace

// Note: a plain constant-pitch (Archimedean) spiral is the special case
// kRhoPhi0 == kRhoPhi1 of VDSpiral's trajectory function (verified against
// vendor/mrarbgrad/traj/VDSpiral.h: VDSpiral_TrajFunc::getK reduces to
// rho = kRhoPhi0 * phi exactly when the two are equal), so no separate
// constant-pitch binding is provided.
static py::tuple vdspiral_waveform(double fov_m, int64_t n_pix, double slew_limit_hz_per_pix_per_s,
                                    double grad_limit_hz_per_pix, double dt, double k_rho_phi0, double k_rho_phi1) {
    MrTraj::GeoPara geo{fov_m, n_pix};
    MrTraj::GradPara grad{slew_limit_hz_per_pix_per_s, grad_limit_hz_per_pix, dt};
    return base_waveform<VDSpiral>(geo, grad, k_rho_phi0, k_rho_phi1);
}

static py::tuple rosette_waveform(double fov_m, int64_t n_pix, double slew_limit_hz_per_pix_per_s,
                                   double grad_limit_hz_per_pix, double dt, double om1, double om2, double t_max) {
    MrTraj::GeoPara geo{fov_m, n_pix};
    MrTraj::GradPara grad{slew_limit_hz_per_pix_per_s, grad_limit_hz_per_pix, dt};
    return base_waveform<Rosette>(geo, grad, om1, om2, t_max);
}

PYBIND11_MODULE(_arbgrad_wrapper, m) {
    m.doc() = "Base-waveform-only arbitrary-gradient design (spiral/rosette) "
              "over a vendored MRArbGrad core. Returns (k0, gradient, n_shots) in native "
              "Hz/pix units; shot ordering/rotation is handled in pulserver.arbgrad.";

    m.def("vdspiral_waveform", &vdspiral_waveform, py::arg("fov_m"), py::arg("n_pix"),
          py::arg("slew_limit_hz_per_pix_per_s"), py::arg("grad_limit_hz_per_pix"), py::arg("dt"),
          py::arg("k_rho_phi0"), py::arg("k_rho_phi1"));

    m.def("rosette_waveform", &rosette_waveform, py::arg("fov_m"), py::arg("n_pix"),
          py::arg("slew_limit_hz_per_pix_per_s"), py::arg("grad_limit_hz_per_pix"), py::arg("dt"),
          py::arg("om1"), py::arg("om2"), py::arg("t_max"));
}
