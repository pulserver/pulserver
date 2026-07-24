"""Unit tests for pulserver.pulses."""

from __future__ import annotations

import numpy as np
import pytest

pp = pytest.importorskip("pypulseq")

from pulserver.design._rf import _auxiliary_helpers as pulses  # noqa: E402


def test_fat_shift_hz() -> None:
    assert pulses.fat_shift_hz(3.0) == pytest.approx(-441.0, abs=1.0)
    assert pulses.fat_shift_hz(1.5) == pytest.approx(-220.4, abs=1.0)


def test_fat_saturation_module() -> None:
    fs = pulses.fat_saturation(pp.Opts(), freq_offset_hz=-441.0, voxel_size_m=3e-3)
    assert fs.rf.freq_offset == pytest.approx(-441.0)
    assert fs.rf.use == "saturation"
    assert [lab.label for lab in fs.labels] == ["NOPOS", "NOROT"]
    assert all(lab.type == "labelset" and lab.value == 1 for lab in fs.labels)
    assert len(fs.spoiler) == 3
    assert fs.duration_s > 8e-3
    # flip angle from envelope integral ~ 110 deg
    flip = np.rad2deg(2 * np.pi * np.abs(np.trapz(fs.rf.signal, fs.rf.t)))
    assert flip == pytest.approx(110.0, rel=0.02)


def test_fat_saturation_block_roundtrip(tmp_path) -> None:
    import pulserver.io as pio
    import pulserver.pypulseq as ps

    opts = pp.Opts()
    fs = pulses.fat_saturation(opts, freq_offset_hz=-441.0, voxel_size_m=3e-3)
    seq = ps.Sequence(opts)
    seq.add_block(fs.rf, *fs.labels)
    seq.add_block(*fs.spoiler)
    out = tmp_path / "fatsat.seq"
    pio.write(seq, output=str(out), remove_duplicates=False, check_timing=False)
    content = out.read_text()
    assert "NOPOS" in content and "NOROT" in content


def test_bloch_siegert_envelope_and_kbs() -> None:
    bs = pulses.bloch_siegert(pp.Opts(), freq_offset_hz=4000.0, peak_b1_hz=500.0)
    sig = np.abs(np.asarray(bs.rf.signal))
    n = sig.size
    # symmetric Fermi with a flat top at the peak amplitude
    assert np.allclose(sig, sig[::-1], rtol=1e-9)
    assert sig.max() == pytest.approx(500.0, rel=1e-6)
    center = sig[int(0.4 * n) : int(0.6 * n)]
    assert np.all(center > 0.99 * sig.max())
    assert sig[0] < 0.05 * sig.max()
    assert bs.rf.freq_offset == pytest.approx(4000.0)
    # K_BS matches the direct integral
    dwell = 1e-5
    kbs_direct = np.sum((2 * np.pi * sig) ** 2 / (2 * 2 * np.pi * 4000.0)) * dwell
    assert bs.kbs_rad == pytest.approx(kbs_direct, rel=1e-9)
    assert bs.kbs_rad_per_gauss2 > 0


def test_bloch_siegert_rejects_on_resonance() -> None:
    with pytest.raises(ValueError):
        pulses.bloch_siegert(pp.Opts(), freq_offset_hz=0.0)
