"""What the envelope of a design costs, against the check of the scan it stands for, and what generate spends.

The slew-rate search of the design service reads the envelope of a design
(:func:`pulserver.design.repetition_envelope`) in place of checking the scan,
and checks and converts the scan once. This measures the envelope per block
of the scan on phase-encoded lines, and one ``generate`` of two shipped 3D
scans at 7 T limits, counting the scans it checks.

Usage: python scripts/envelope_throughput.py [views ...]
"""

from __future__ import annotations

import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pypulseqpp as pp
from pypulseqpp import safety

from pulserver import _plugins
from pulserver.design import repetition_envelope
from pulserver.host import DesignStore, _slew
from pulserver.host import _service as service
from pulserver.protocol import PROTOCOL_BEGIN, PROTOCOL_END

SYSTEM = pp.Opts(
    B0=7.0, max_grad=100, grad_unit="mT/m", max_slew=200, slew_unit="T/m/s"
)
NERVE = safety.ChronaxieModel(chronaxie=359e-6, rheobase=71.62162)

#: A 7 T head gradient designed at the coil's slew rate, and its nerve model.
LIMITS = {
    "B0": 7.0,
    "max_grad": 100.0,
    "grad_unit": "mT/m",
    "max_slew": 200.0,
    "slew_unit": "T/m/s",
    "design_max_grad": 100.0,
    "design_max_slew": 200.0,
    "rf_raster_time": 2e-6,
    "grad_raster_time": 4e-6,
    "adc_raster_time": 2e-6,
    "block_duration_raster": 4e-6,
    "max_b1": 25.0,
    "b1_unit": "uT",
    "design_rf_dead_time": 232e-6,
    "design_rf_ringdown_time": 4e-6,
    "pns_chronaxie": 359e-6,
    "pns_rheobase": 71.62162,
    "pns_limit": 0.98,
}

#: Sagittal 0.8 mm MPRAGE and a 3D FSE, as prescribed at a 7 T console.
SCANS = {
    "mprage3d": "TE: 2076\nTR: 3800000\nprep_time: 1100000\nflip: 8.0\nexcitation: 2\n"
    "bandwidth: 500000.0\nfov: 220.0\nphase_fov: 220.0\nnx: 274\nny: 274\n"
    "nslices: 176\nslice_thickness: 0.8\nRy: 1\nRz: 2\nuser4_value: 24\n"
    "user5_value: 24\nfov_rotation_11: 0.0\nfov_rotation_12: 1.0\n"
    "fov_rotation_21: -1.0\nfov_rotation_22: 0.0\n",
    "fse3d": "TE: 54000\nTR: 3431000\netl: 80\nbandwidth: 500000.0\nfov: 210.0\n"
    "phase_fov: 159.6\nnx: 276\nny: 208\nnslices: 192\nslice_thickness: 0.8\n"
    "Ry: 2\nRz: 2\n",
}


def lines(views: int) -> pp.Sequence:
    """Return a spoiled gradient echo of ``views`` phase-encoded lines, five blocks each."""
    seq = pp.Sequence(SYSTEM)
    pulse = pp.make_block_pulse(np.deg2rad(10), duration=0.5e-3, system=SYSTEM)
    readout = pp.make_trapezoid("x", flat_area=2e3, flat_time=2e-3, system=SYSTEM)
    encode = pp.make_trapezoid("y", area=1e3, duration=1e-3, system=SYSTEM)
    partition = pp.make_trapezoid("z", area=1e3, duration=1e-3, system=SYSTEM)
    rng = np.random.default_rng(0)
    for ky, kz in rng.uniform(-1.0, 1.0, (views, 2)):
        seq.add_block(pulse)
        seq.add_block(pp.scale_grad(encode, ky), pp.scale_grad(partition, kz))
        seq.add_block(readout)
        seq.add_block(pp.scale_grad(encode, -ky), pp.scale_grad(partition, -kz))
        seq.add_block(pp.make_delay(5e-3))
    return seq


def best(run, repeats: int = 5) -> float:
    """Return the shortest of ``repeats`` timings of ``run``, in s."""
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        timings.append(time.perf_counter() - start)
    return min(timings)


def read(sequence: pp.Sequence) -> None:
    safety.check_max_slew(sequence, SYSTEM)
    safety.check_pns(sequence, NERVE)


def per_block(views: list[int]) -> None:
    print(
        f"{'views':>6} {'blocks':>7} {'envelope':>9} {'fold us/block':>14} "
        f"{'read scan ms':>13} {'read envelope ms':>17}"
    )
    for count in views:
        scan = lines(count)
        blocks = len(scan.block_definitions())
        envelope = repetition_envelope(scan)
        fold = best(lambda scan=scan: repetition_envelope(scan))
        whole = best(lambda scan=scan: read(scan))
        bound = best(lambda envelope=envelope: read(envelope))
        print(
            f"{count:>6} {blocks:>7} {len(envelope.block_definitions()):>9} "
            f"{1e6 * fold / blocks:>14.2f} {1e3 * whole:>13.2f} {1e3 * bound:>17.2f}"
        )


def generate(name: str, values: str) -> None:
    """Time one generate from a fresh process state, as a forked design server answers it."""
    checked = []
    check_and_convert = service._check_and_convert

    def counted(*args, **kwargs):
        checked.append(args[0])
        return check_and_convert(*args, **kwargs)

    service._check_and_convert = counted
    service._LOWERED = _slew.Remembered()
    block = f"{PROTOCOL_BEGIN}\n{values}{PROTOCOL_END}\n"
    common = {"plugins": (_plugins.SEQUENCES,), "plugin": name, "limits": LIMITS}
    service.call("list", plugins=common["plugins"], plugin=name)
    with tempfile.TemporaryDirectory() as root:
        store = DesignStore(Path(root), "scan")
        start = time.perf_counter()
        status, reply = service.call("generate", block=block, store=store, **common)
        elapsed = time.perf_counter() - start
        manifest = store.manifest(reply.split()[1]) if status == 0 else {}
    service._check_and_convert = check_and_convert
    print(
        f"{name}: generate {elapsed:.2f} s, {len(checked)} scan(s) checked, "
        f"{manifest.get('design_max_slew_t_per_m_per_s', float('nan')):.1f} T/m/s"
        if status == 0
        else f"{name}: {reply.strip()}"
    )


def main(arguments: list[str]) -> int:
    warnings.simplefilter("ignore")
    per_block([int(a) for a in arguments] or [64, 256, 1024, 4096])
    for name, values in SCANS.items():
        generate(name, values)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
