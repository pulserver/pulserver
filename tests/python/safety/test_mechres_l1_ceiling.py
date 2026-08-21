"""The L1 ceiling may make the gate cheaper, never more permissive.

|S_ax(f)| is at most the integral of |g_ax| over the canonical TR, at every
frequency: the phase factor has unit modulus, so integrating it away can only
shrink the result. That ceiling is frequency-independent and costs one pass
over the events, which lets the gate prove a whole band quiet and skip the
sidelobe probes inside it rather than evaluating them.

A ceiling that is not really a ceiling would show up as a violation the gate
fails to report. These tests put the threshold a hair either side of the
sequence's own measured peak, where a skip that swallowed anything would flip
the verdict, and check the flip lands exactly on the peak.
"""

import numpy as np
import pytest
from pulserver._ext.pulseg import _calc_mech_resonances, _check_safety

import pulserver.pypulseq as pp

GAMMA_HZ_PER_MT_PER_M = 42.576e3

#: ``SA_AEQ_TRAIN_SHAPE``: a stated band amplitude is a plateau, and the
#: criterion runs in equivalent-sinusoid amplitude. Mirrored, not recomputed.
TRAIN_SHAPE = 0.8106

SYSTEM = pp.Opts(max_grad=50, grad_unit="mT/m", max_slew=200, slew_unit="T/m/s")

#: Bands wide and narrow, on and off the loud lines.
BANDS = [(500.0, 600.0), (550.0, 650.0), (600.0, 1000.0), (515.0, 1650.0)]

CASES = [
    ("mprage3D_sequence", {"n_x": 96, "n_y": 32, "n_z": 16}),
    ("gre2D_sequence", {"n_x": 128, "n_y": 96, "tr": 9e-3, "te": 4.5e-3}),
    (
        "gre3D_sequence",
        {"n_x": 128, "n_y": 64, "n_z": 16, "readout_bandwidth_hz": 125e3},
    ),
    ("epi2D_sequence", {"n_x": 64, "n_y": 64, "readout_bandwidth_hz": 250e3}),
    ("gre_spiral2D_sequence", {"n_x": 128, "n_arms": 8, "tr": 25e-3}),
    ("gre_radial2D_sequence", {"n_x": 128, "tr": 10e-3}),
]


def _build(module, **kwargs):
    from importlib import import_module

    made = getattr(import_module("pulserver.app"), module).main(
        system=SYSTEM, n_dummy=0, **kwargs
    )
    return made[0] if isinstance(made, tuple) else made


def _peak_a_eq(structure, band):
    """The loudest in-band line, taken from the path that never skips."""
    spectra = _calc_mech_resonances(
        structure.collection,
        0,
        0,
        0,
        target_resolution_hz=1.0 / structure.tr_duration,
        max_freq_hz=3000.0,
        forbidden_bands=[(band[0], band[1], 0.0)],
    )
    amps = np.asarray(spectra["candidate_grad_amps"], float)
    return float(amps.max()) / GAMMA_HZ_PER_MT_PER_M if amps.size else 0.0


def _passes(collection, band, plateau_mt_per_m):
    try:
        _check_safety(
            collection,
            forbidden_bands=[
                (band[0], band[1], plateau_mt_per_m * GAMMA_HZ_PER_MT_PER_M)
            ],
        )
    except RuntimeError:
        return False
    return True


@pytest.mark.parametrize("module,kwargs", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("band", BANDS, ids=lambda b: f"{b[0]:.0f}_{b[1]:.0f}")
def test_the_ceiling_never_swallows_a_violation(module, kwargs, band):
    sequence = _build(module, **kwargs)
    structure = sequence._structure_for("bound")
    peak = _peak_a_eq(structure, band)
    if peak <= 0.0:
        pytest.skip("no candidate line inside this band")

    plateau = peak / TRAIN_SHAPE
    assert not _passes(structure.collection, band, plateau * 0.98)
    assert _passes(structure.collection, band, plateau * 1.02)
