"""Write tests/utils/expected/arms_scan.seq: a scan of distinct arbitrary arms.

Eight repetitions, each an RF block, one 1024-sample arbitrary gradient on x
and y, and an ADC block. Every arm differs, carries content inside the
550-650 Hz and 1100-1250 Hz bands, and is stored sample for sample, so the
mechanical-resonance scan window prices it from its FFT record. The C tests
under tests/ctests/ read this file; run this script to rebuild it.
"""

from pathlib import Path

import numpy as np

import pulserver.pypulseq as pp

OUT = Path(__file__).resolve().parent / "expected" / "arms_scan.seq"
N_ARMS = 8
N_SAMPLES = 1024
RASTER_S = 4e-6
AMPLITUDE_HZ_PER_M = 0.2e6


def arm(k: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(N_SAMPLES) * RASTER_S
    envelope = np.sin(np.pi * np.arange(N_SAMPLES) / (N_SAMPLES - 1)) ** 2
    tones = [580.0 + 12.0 * k, 1150.0 + 15.0 * k, 300.0 + 50.0 * k]
    w = np.zeros(N_SAMPLES)
    for f in tones:
        w += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    w += 0.5 * np.sin(
        2 * np.pi * (200.0 * t + 1.0e5 * t * t) + rng.uniform(0, 2 * np.pi)
    )
    w *= envelope
    return w / np.max(np.abs(w)) * AMPLITUDE_HZ_PER_M


def main() -> None:
    system = pp.Opts(
        max_grad=40,
        grad_unit="mT/m",
        max_slew=150,
        slew_unit="T/m/s",
        grad_raster_time=RASTER_S,
        rf_raster_time=2e-6,
        adc_raster_time=2e-6,
        block_duration_raster=RASTER_S,
    )
    seq = pp.Sequence(system=system)
    rng = np.random.default_rng(7)
    for k in range(N_ARMS):
        rf = pp.make_block_pulse(flip_angle=0.1, duration=200e-6, system=system)
        gx = pp.make_arbitrary_grad(channel="x", waveform=arm(k, rng), system=system)
        gy = pp.make_arbitrary_grad(
            channel="y", waveform=0.7 * arm(k + 100, rng), system=system
        )
        adc = pp.make_adc(num_samples=N_SAMPLES, dwell=RASTER_S, system=system)
        seq.add_block(rf)
        seq.add_block(gx, gy)
        seq.add_block(adc)
    seq.write(str(OUT))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
