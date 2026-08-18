/**
 * @file module.cpp
 * @brief `pulserver._ext` -- every compiled kernel Pulserver ships, in one
 *        extension module.
 *
 * The five groups are separate submodules, so a caller still writes
 * `from pulserver._ext import pulseg`, but they are linked together: the C
 * library underneath (`pulseg`, and the `pulseq` reader below it) is present
 * once, where a module per group had to carry its own copy of whatever it
 * touched.
 *
 * Each submodule is registered in `sys.modules` under its dotted name, which
 * is what lets `from pulserver._ext.pulseg import _find_tr` and
 * `monkeypatch.setattr("pulserver._ext.recon_cpu", ...)` resolve it the way
 * they would a Python one.
 */

#include <string>

#include <pybind11/pybind11.h>

#include "bindings.hpp"

namespace py = pybind11;

namespace
{

    py::module_ attach(py::module_& parent, const char* name)
    {
        py::module_ sub = parent.def_submodule(name);
        const std::string dotted = "pulserver._ext." + std::string(name);
        sub.attr("__name__") = dotted;
        py::module_::import("sys").attr("modules")[py::str(dotted)] = sub;
        return sub;
    }

} // namespace

PYBIND11_MODULE(_ext, m)
{
    m.doc() = "Pulserver's compiled kernels. Private: see pulserver.pypulseq, "
              "pulserver.design and pulserver.recon for the public surface.";
    m.attr("__name__") = "pulserver._ext";

    py::module_ pulseg = attach(m, "pulseg");
    pulseg.doc() = "The PulSeg representation, structure detection and the safety engine.";
    pulserver_bind_pulseg(pulseg);

    py::module_ pulseqpp = attach(m, "pulseqpp");
    pulseqpp.doc() = "pulseq::Sequence -- the design and write path.";
    pulserver_bind_pulseqpp(pulseqpp);

    py::module_ arbgrad = attach(m, "arbgrad");
    arbgrad.doc() = "Arbitrary-gradient design: spiral, rosette and sampled trajectories.";
    pulserver_bind_arbgrad(arbgrad);

    py::module_ sampling = attach(m, "sampling");
    sampling.doc() = "Deterministic sampling-pattern kernels.";
    pulserver_bind_sampling(sampling);

    py::module_ recon_cpu = attach(m, "recon_cpu");
    recon_cpu.doc() = "CPU kernels the reconstruction calls through zero-copy views.";
    pulserver_bind_recon_cpu(recon_cpu);
}
