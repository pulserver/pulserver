"""The multi-echo Cartesian gradient echoes of the sequence zoo.

Both dimensionalities in one file: the echo train is what these two modules
add over their single-echo counterparts, so the checks concentrate on it --
per-echo centering, the ``ECO`` counter, the declared echo times, and the
monopolar/bipolar polarity contract.
"""

from __future__ import annotations

import numpy as np
import pytest

import pulserver.pypulseq as pp
from pulserver.app import gre_multiecho2D_sequence, gre_multiecho3D_sequence

N_X = 32
N_ECHOES = 3


def design_2d(**kwargs):
    kwargs.setdefault("n_echoes", N_ECHOES)
    kwargs.setdefault("n_acs", 4)
    kwargs.setdefault("n_dummy", 1)
    kwargs.setdefault("tr", 40e-3)
    return gre_multiecho2D_sequence.main(n_x=N_X, n_y=8, **kwargs)


def design_3d(**kwargs):
    kwargs.setdefault("n_echoes", N_ECHOES)
    kwargs.setdefault("n_acs", 4)
    kwargs.setdefault("n_acs_z", 2)
    kwargs.setdefault("n_dummy", 1)
    kwargs.setdefault("tr", 30e-3)
    return gre_multiecho3D_sequence.main(
        n_x=N_X, n_y=8, n_z=4, slab_thickness=32e-3, **kwargs
    )


def straddles_centre(seq, count=N_ECHOES):
    """Whether each of the first ``count`` acquisitions crosses k = 0 between
    its two central samples, and symmetrically.

    An even-sampled readout has no sample *at* k = 0: the two central ones sit
    at -dk/2 and +dk/2, equidistant. Which of them ``argmin`` returns is
    decided by the last bits of a gradient amplitude -- and a designed sequence
    is held at the precision the file writes -- so the index is not the claim.
    Straddling the centre is.
    """
    k_traj_adc, *_ = seq.calculate_kspace()
    labels = seq.evaluate_labels(evolution="adc")
    n_samples = k_traj_adc.shape[1] // len(labels["ECO"])
    kx = k_traj_adc[0].reshape(-1, n_samples)
    out = []
    for echo in range(count):
        left, right = kx[echo, n_samples // 2 - 1], kx[echo, n_samples // 2]
        step = abs(right - left)
        # Straddling, and the midpoint on the origin to a thousandth of a
        # sample -- what survives holding the gradients at file precision,
        # which costs about 3e-5 of a step.
        out.append(left * right < 0.0 and abs(left + right) / 2.0 < 1e-3 * step)
    return out


@pytest.mark.parametrize("design", [design_2d, design_3d], ids=["2d", "3d"])
def test_the_sequence_is_valid_pulseq(design):
    is_ok, error_report = design().check_timing()
    assert is_ok, error_report


@pytest.mark.parametrize("design", [design_2d, design_3d], ids=["2d", "3d"])
@pytest.mark.parametrize("monopolar", [True, False], ids=["monopolar", "bipolar"])
def test_every_echo_is_centred_on_its_readout(design, monopolar):
    """The train's whole contract: each echo crosses k = 0 at the centre
    sample, whichever way its lobe is played."""
    seq = design(monopolar=monopolar)
    assert straddles_centre(seq) == [True] * N_ECHOES


@pytest.mark.parametrize("design", [design_2d, design_3d], ids=["2d", "3d"])
def test_each_acquisition_carries_its_echo_index(design):
    seq = design()
    labels = seq.evaluate_labels(evolution="adc")
    eco = labels["ECO"].reshape(-1, N_ECHOES)
    assert np.array_equal(eco, np.tile(np.arange(N_ECHOES), (eco.shape[0], 1)))


@pytest.mark.parametrize("design", [design_2d, design_3d], ids=["2d", "3d"])
def test_the_declared_echo_times_are_where_the_echoes_actually_fall(design):
    seq = design()
    declared = seq.get_definition("TE")
    assert len(declared) == N_ECHOES

    k_traj_adc, _, t_excitation, _, t_adc = seq.calculate_kspace()
    labels = seq.evaluate_labels(evolution="adc")
    n_samples = k_traj_adc.shape[1] // len(labels["ECO"])
    kx = k_traj_adc[0].reshape(-1, n_samples)
    times = t_adc.reshape(-1, n_samples)

    # Relative to the excitation the echoes belong to -- the dummies excite
    # too, so the first excitation of the scan is not it.
    def measured_te(echo):
        instant = times[echo][int(np.argmin(np.abs(kx[echo])))]
        return instant - t_excitation[t_excitation < instant].max()

    measured = [measured_te(echo) for echo in range(N_ECHOES)]
    # The echo instant is measured on the dwell grid, so agreement is to one
    # sample, not to the femtosecond.
    dwell = float(np.diff(times[0]).max())
    assert measured == pytest.approx(declared, abs=dwell)


def test_a_bipolar_train_is_shorter_than_a_monopolar_one():
    monopolar = design_2d(monopolar=True)
    bipolar = design_2d(monopolar=False)
    assert bipolar.duration()[0] < monopolar.duration()[0]
    assert bipolar.get_definition("TE")[-1] < monopolar.get_definition("TE")[-1]


def test_the_echo_spacing_is_uniform():
    for seq in (design_2d(), design_3d()):
        te = seq.get_definition("TE")
        spacings = np.diff(te)
        assert spacings == pytest.approx([spacings[0]] * (N_ECHOES - 1))


def test_the_protocol_reports_the_echo_range():
    system = pp.Opts()
    for module in (gre_multiecho2D_sequence, gre_multiecho3D_sequence):
        report = module.PLUGIN.validate_protocol(
            system, module.PLUGIN.get_default_protocol(system)
        )
        assert report["valid"] is True
        assert "echoes" in report["info"]
