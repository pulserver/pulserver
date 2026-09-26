"""What the cache plays, simulated by KomaMRI: the exported file gives the design's signal, and pypulseqpp's Bloch simulation of the cache gives KomaMRI's.

Runs where ``PULSERVER_KOMA_PROJECT`` names a Julia project holding KomaMRI,
such as ``tests/koma`` once instantiated, with ``julia`` on the path.
"""

import math
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pypulseqpp as pp
import pytest
from _zoo import SMALL
from pypulseqpp import sequences

from pulserver import ir, virtual

PROJECT = os.environ.get("PULSERVER_KOMA_PROJECT")
KOMA = Path(__file__).parent / "koma"
FIXTURES = Path(__file__).parent / "fixtures" / "sequences"
SYSTEM = pp.Opts(B0=3.0)
#: Relative to the peak sample of the design's signal. The text format writes a
#: turned gradient's amplitude to six significant figures, and the phase that
#: rounding accrues over a spoiler's moment stays well below this.
TOLERANCE = 1e-3
#: Relative to the peak sample of KomaMRI's signal: the two simulators step
#: through an RF pulse on different grids, and KomaMRI plays the 10 ns ramp the
#: file holds where the cache plays a gradient's step.
ENGINE_TOLERANCE = 1e-2
NAMES = sorted(
    p.stem for p in FIXTURES.glob("*.seq") if not p.stem.endswith("_b")
) + sorted(SMALL)

pytestmark = pytest.mark.skipif(
    PROJECT is None, reason="PULSERVER_KOMA_PROJECT names no Julia project"
)


def _phantom():
    """Return spins filling an elliptic cylinder 20 cm long around the isocentre.

    Planes of spins lie 2 mm apart within 3 cm of the isocentre, where the
    slices of the 2D sequences lie, and 10 mm apart beyond, past the edges of
    the slabs. Density, T1, T2 and off-resonance each vary across the object,
    none of them symmetrically, so a turned or mistimed readout, a shifted
    slice or slab and a wrong pulse each change the signal. Positions are in
    m, relaxation times in s and the off-resonance in Hz.
    """
    grid = np.linspace(-0.08, 0.08, 17)
    beyond = np.linspace(0.04, 0.1, 7)
    planes = np.concatenate([-beyond[::-1], np.linspace(-0.03, 0.03, 31), beyond])
    x, y, z = np.meshgrid(grid, grid, planes, indexing="ij")
    inside = (x / 0.09) ** 2 + (y / 0.07) ** 2 <= 1.0
    x, y, z = x[inside], y[inside], z[inside]
    return SimpleNamespace(
        positions=np.column_stack([x, y, z]),
        density=1.0 + 0.5 * (x > 0.02) - 0.3 * (y < -0.03) + 0.2 * (z > 0.0),
        t1=0.6 + 5.0 * (x + 0.08),
        t2=0.04 + 0.5 * (y + 0.08),
        off_resonance=20.0 * y / 0.07 + 10.0 * z / 0.03,
    )


def _write_phantom(spins, path):
    """Write the spins as ``simulate.jl`` reads them: their count, then x, y, z, density, T1, T2 and the off-resonance in rad/s."""
    columns = (
        *spins.positions.T,
        spins.density,
        spins.t1,
        spins.t2,
        2.0 * np.pi * spins.off_resonance,
    )
    with path.open("wb") as file:
        np.array([spins.density.size], dtype="<i8").tofile(file)
        for column in columns:
            column.astype("<f8").tofile(file)


def _as_komamri_plays_it(exported, path):
    """Write the exported file with each RF pulse in KomaMRI's convention.

    KomaMRI adds a pulse's phase shape and phase offset to its field with the
    opposite sign to pypulseqpp's Bloch simulation, and refers the phase the
    frequency offset f accrues to the pulse's centre t_c. With the samples
    conjugated and the phase offset phi replaced by -phi - 2 pi f t_c, KomaMRI plays
    the field pypulseqpp plays of the pulse. The offset is written within one
    turn of zero, where the text format keeps it to its six significant figures
    of a radian or less.
    """
    seq = pp.Sequence(SYSTEM)
    seq.read(str(exported))
    turned = pp.Sequence(SYSTEM)
    for index in range(1, len(seq.block_events) + 1):
        block = seq.get_block(index)
        events = [e for e in (block.gx, block.gy, block.gz, block.adc) if e is not None]
        if block.rf is not None:
            rf = block.rf
            events.append(
                SimpleNamespace(
                    type="rf",
                    signal=np.conj(np.asarray(rf.signal)),
                    t=np.asarray(rf.t),
                    shape_dur=rf.shape_dur,
                    delay=rf.delay,
                    freq_offset=rf.freq_offset,
                    phase_offset=math.remainder(
                        -rf.phase_offset - 2.0 * np.pi * rf.freq_offset * rf.center,
                        2.0 * np.pi,
                    ),
                    freq_ppm=0.0,
                    phase_ppm=0.0,
                    center=rf.center,
                    use=rf.use,
                    dead_time=rf.dead_time,
                    ringdown_time=rf.ringdown_time,
                )
            )
        turned.add_block(*events, pp.make_delay(block.block_duration))
    turned.write(str(path))


def _design(name, directory):
    """Write the design ``name`` names into ``directory`` and return its first file."""
    if name in SMALL:
        path = directory / "scan.seq"
        getattr(sequences, name)(**SMALL[name]).write(str(path))
        return path
    shutil.copytree(FIXTURES, directory, dirs_exist_ok=True)
    return directory / f"{name}.seq"


def _as_komamri_reads_it(first, directory):
    """Return the design's files in play order, each as KomaMRI is to read it.

    KomaMRI drops the ppm term of an RF or ADC offset, so a file carrying one
    is given to it as Pulseq 1.4.1, whose writer resolves ppm offsets at the
    field the cache is converted at.
    """
    files = []
    for index, (path, seq) in enumerate(pp.io.read_chain(first)):
        libraries = seq.libraries()
        if np.any(libraries.rf[:, 6:8]) or np.any(libraries.adc[:, 3:5]):
            path = directory / f"resolved_{index}.seq"
            seq.write_v141(str(path), gamma=SYSTEM.gamma, field=SYSTEM.B0)
        files.append(str(path.resolve()))
    return files


@pytest.fixture(scope="module")
def simulated(tmp_path_factory):
    """The signals of each design, by name, of its cache, and of that cache in pypulseqpp's Bloch simulation.

    KomaMRI simulates the design, the exported cache and the exported cache in
    its own RF convention, each without the ADC's offsets; the last is
    demodulated by the receiver phase the export returns, as the playout
    demodulates. pypulseqpp simulates the cache on the same spins.
    """
    root = tmp_path_factory.mktemp("koma")
    spins = _phantom()
    _write_phantom(spins, root / "phantom.bin")
    cases, received, engine = [], {}, {}
    for name in NAMES:
        case = root / name
        (case / "design").mkdir(parents=True)
        first = _design(name, case / "design")
        ir.convert(first, SYSTEM)
        received[name] = np.concatenate(
            virtual.export(first, case / "exported.seq", SYSTEM)
        )
        _as_komamri_plays_it(case / "exported.seq", case / "komamri.seq")
        files = _as_komamri_reads_it(first, case)
        (case / "design.txt").write_text("\n".join(files) + "\n")
        isochromats = pp.Isochromats(
            spins.positions,
            proton_density=spins.density,
            t1=spins.t1,
            t2=spins.t2,
            off_resonance=spins.off_resonance,
        )
        engine[name] = np.concatenate(virtual.simulate(first, isochromats), axis=1)[0]
        cases.append(case)
    subprocess.run(
        [
            "julia",
            f"--project={PROJECT}",
            str(KOMA / "simulate.jl"),
            str(root / "phantom.bin"),
            *map(str, cases),
        ],
        check=True,
    )
    signals = {}
    for case in cases:
        design, exported, komamri = (
            np.fromfile(case / f"{kind}.sig", dtype="<c16")
            for kind in ("design", "exported", "komamri")
        )
        signals[case.name] = SimpleNamespace(
            design=design,
            exported=exported,
            komamri=komamri * np.exp(1j * received[case.name]),
            engine=engine[case.name],
        )
    return signals


@pytest.mark.parametrize("name", NAMES)
def test_komamri_simulates_the_signal_of_the_design_from_the_exported_cache(
    simulated, name
):
    design, exported = simulated[name].design, simulated[name].exported

    assert exported.size == design.size
    np.testing.assert_allclose(
        exported, design, rtol=0, atol=TOLERANCE * np.abs(design).max()
    )


@pytest.mark.parametrize("name", NAMES)
def test_komamri_and_pypulseqpps_bloch_simulation_give_one_signal_of_the_cache(
    simulated, name
):
    komamri, engine = simulated[name].komamri, simulated[name].engine

    assert engine.size == komamri.size
    np.testing.assert_allclose(
        engine, komamri, rtol=0, atol=ENGINE_TOLERANCE * np.abs(komamri).max()
    )
