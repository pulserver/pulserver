// Thin pybind11 bindings over a vendored MRArbGrad core
// (see cxx/arbgrad/vendor/mrarbgrad/NOTICE.md).
//
// Built-in bindings produce one base-shot waveform (nStack fixed to 1,
// iAcq fixed to 0); sampled_trajectory_gradient exposes the same solver for
// user-provided NumPy paths. Shot ordering and rotation remain Python policy.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "traj/Rosette.h"
#include "traj/VDSpiral.h"

namespace py = pybind11;

namespace
{

py::array_t<double> vv3_to_numpy(const vv3 &samples)
{
    const auto n = static_cast<py::ssize_t>(samples.size());
    py::array_t<double> arr({n, static_cast<py::ssize_t>(3)});
    auto buf = arr.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < n; ++i)
    {
        buf(i, 0) = samples[static_cast<size_t>(i)].x;
        buf(i, 1) = samples[static_cast<size_t>(i)].y;
        buf(i, 2) = samples[static_cast<size_t>(i)].z;
    }
    return arr;
}

py::array_t<double> v3_to_numpy(const v3 &value)
{
    py::array_t<double> arr(3);
    auto buf = arr.mutable_unchecked<1>();
    buf(0) = value.x;
    buf(1) = value.y;
    buf(2) = value.z;
    return arr;
}

vv3 numpy_to_vv3(const py::array_t<double, py::array::c_style | py::array::forcecast> &trajectory)
{
    const auto buf = trajectory.unchecked<2>();
    if (buf.shape(0) < 4 || (buf.shape(1) != 2 && buf.shape(1) != 3))
    {
        throw std::invalid_argument("trajectory must have shape (n, 2) or (n, 3), with n >= 4");
    }

    vv3 samples;
    samples.reserve(static_cast<size_t>(buf.shape(0)));
    for (py::ssize_t i = 0; i < buf.shape(0); ++i)
    {
        const double x = buf(i, 0);
        const double y = buf(i, 1);
        const double z = buf.shape(1) == 3 ? buf(i, 2) : 0.0;
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
        {
            throw std::invalid_argument("trajectory samples must be finite");
        }
        if (i > 0)
        {
            const v3 delta = v3(x, y, z) - samples.back();
            if (v3::norm(delta) == 0.0)
            {
                throw std::invalid_argument(
                    "trajectory cannot contain consecutive duplicate samples");
            }
        }
        samples.emplace_back(x, y, z);
    }
    return samples;
}

py::array_t<double> sampled_trajectory_gradient(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &trajectory,
    double max_slew,
    double max_grad,
    double dt,
    int oversampling,
    bool start_at_zero,
    bool end_at_zero)
{
    if (!(max_slew > 0.0) || !(max_grad > 0.0) || !(dt > 0.0))
    {
        throw std::invalid_argument("max_slew, max_grad, and dt must be positive");
    }
    if (oversampling < 2)
    {
        throw std::invalid_argument("oversampling must be >= 2");
    }

    vv3 samples = numpy_to_vv3(trajectory);
    const auto reserve = static_cast<i64>(std::max<size_t>(10000, samples.size() * 4));
    Mag solver(reserve, reserve);
    solver.init(
        samples,
        max_slew,
        max_grad,
        dt,
        static_cast<i64>(oversampling),
        start_at_zero ? 0.0 : max_grad,
        end_at_zero ? 0.0 : max_grad);

    vv3 gradient;
    if (!solver.solve(&gradient) || gradient.size() < 2)
    {
        throw std::runtime_error("MRArbGrad sampled-trajectory computation failed");
    }

    if (start_at_zero)
    {
        vv3 ramp;
        Mag::ramp_front(&ramp, gradient.front(), v3(), max_slew, dt);
        gradient.insert(gradient.begin(), ramp.begin(), ramp.end());
    }
    if (end_at_zero)
    {
        vv3 ramp;
        Mag::ramp_back(&ramp, gradient.back(), v3(), max_slew, dt);
        gradient.insert(gradient.end(), ramp.begin(), ramp.end());
    }
    return vv3_to_numpy(gradient);
}

// Constructs `TrajT` with nStack fixed to 1 and returns the single base-shot
// waveform (iAcq=0): (k0, gradient, n_shots).
template <typename TrajT, typename... ShapeArgs>
py::tuple base_waveform(
    const MrTraj::GeoPara &geo,
    const MrTraj::GradPara &grad,
    ShapeArgs &&...shape_args)
{
    TrajT traj(geo, grad, static_cast<i64>(1), std::forward<ShapeArgs>(shape_args)...);

    v3 m0;
    vv3 gro;
    if (!traj.getGrad(&m0, &gro, 0))
    {
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
static py::tuple vdspiral_waveform(
    double fov_m,
    int64_t n_pix,
    double slew_limit_hz_per_pix_per_s,
    double grad_limit_hz_per_pix,
    double dt,
    double k_rho_phi0,
    double k_rho_phi1)
{
    MrTraj::GeoPara geo{fov_m, n_pix};
    MrTraj::GradPara grad{slew_limit_hz_per_pix_per_s, grad_limit_hz_per_pix, dt};
    return base_waveform<VDSpiral>(geo, grad, k_rho_phi0, k_rho_phi1);
}

static py::tuple rosette_waveform(
    double fov_m,
    int64_t n_pix,
    double slew_limit_hz_per_pix_per_s,
    double grad_limit_hz_per_pix,
    double dt,
    double om1,
    double om2,
    double t_max)
{
    MrTraj::GeoPara geo{fov_m, n_pix};
    MrTraj::GradPara grad{slew_limit_hz_per_pix_per_s, grad_limit_hz_per_pix, dt};
    return base_waveform<Rosette>(geo, grad, om1, om2, t_max);
}

PYBIND11_MODULE(_arbgrad_wrapper, m)
{
    m.doc() = "Arbitrary-gradient design over a vendored MRArbGrad core: built-in "
              "spiral/rosette base waveforms plus sampled NumPy trajectories.";

    m.def(
        "vdspiral_waveform",
        &vdspiral_waveform,
        py::arg("fov_m"),
        py::arg("n_pix"),
        py::arg("slew_limit_hz_per_pix_per_s"),
        py::arg("grad_limit_hz_per_pix"),
        py::arg("dt"),
        py::arg("k_rho_phi0"),
        py::arg("k_rho_phi1"));

    m.def(
        "rosette_waveform",
        &rosette_waveform,
        py::arg("fov_m"),
        py::arg("n_pix"),
        py::arg("slew_limit_hz_per_pix_per_s"),
        py::arg("grad_limit_hz_per_pix"),
        py::arg("dt"),
        py::arg("om1"),
        py::arg("om2"),
        py::arg("t_max"));

    m.def(
        "sampled_trajectory_gradient",
        &sampled_trajectory_gradient,
        py::arg("trajectory"),
        py::arg("max_slew"),
        py::arg("max_grad"),
        py::arg("dt"),
        py::arg("oversampling") = 8,
        py::arg("start_at_zero") = true,
        py::arg("end_at_zero") = true,
        "Generate a gradient waveform for sampled k-space coordinates under vector limits.");
}
