#!/usr/bin/env python3
"""Generate data-driven signal-evolution figures for the SEQDESC docs."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pulserver.recon.simulation import (
    BSSFP,
    FSE,
    SPGR,
    SSFPEcho,
    SSFPFID,
    EventType,
    RfDefinition,
    RfShape,
    RfUse,
    SequenceDescription,
    SequenceEvent,
    TissueProperties,
)

OUT = Path(__file__).resolve().parents[1] / "explanations" / "assets" / "reconstruction"
RF_SHAPE = RfShape(4, np.array([0.0, 1.0, 1.0, 0.0], np.float32))
RF_DEFINITION = RfDefinition(
    id=0,
    bandwidth_hz=1000.0,
    num_bands=1,
    band_frequency_offsets_hz=(0.0,) * 8,
    band_bandwidth_hz=1000.0,
    total_b1sq_power=0.0,
    magnitude=RF_SHAPE,
)


def _rf(timestamp_us: float, flip_deg: float, phase_rad: float, use: RfUse):
    # trapz([0, 1, 1, 0], dt=1 us) = 2 us.
    amplitude_hz = np.deg2rad(flip_deg) / (2.0 * np.pi * 2.0e-6)
    return SequenceEvent(
        EventType.RF,
        timestamp_us,
        (0.0, float(use), amplitude_hz, phase_rad, 0.0, -1.0, 0.0),
    )


def _adc(timestamp_us: float):
    return SequenceEvent(
        EventType.ADC,
        timestamp_us,
        (1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
    )


def _description(events, duration_us):
    return SequenceDescription(
        subsequence_index=0,
        tr_duration_us=duration_us,
        events=tuple(events),
        rf_definitions={0: RF_DEFINITION},
        shim_definitions={},
    )


def _fse():
    events = [_rf(1000.0, 90.0, np.pi / 2.0, RfUse.EXCITATION)]
    for echo in range(1, 9):
        events.append(_rf(echo * 10000.0, 180.0, 0.0, RfUse.REFOCUSING))
        events.append(_adc(echo * 10000.0 + 5000.0))
    return _description(events, 90000.0)


def _mprage():
    events = [_rf(1000.0, 180.0, 0.0, RfUse.INVERSION)]
    for readout in range(12):
        rf_time = 100000.0 + readout * 25000.0
        events.append(_rf(rf_time, 8.0, np.pi / 2.0, RfUse.EXCITATION))
        events.append(_adc(rf_time + 5000.0))
    return _description(events, 420000.0)


def _steady_state(phase_cycle=False):
    if not phase_cycle:
        return _description(
            [
                _rf(1000.0, 25.0, 0.0, RfUse.EXCITATION),
                _adc(5000.0),
            ],
            10000.0,
        )
    return _description(
        [
            _rf(1000.0, 40.0, 0.0, RfUse.EXCITATION),
            _adc(3500.0),
            _rf(6000.0, 40.0, np.pi, RfUse.EXCITATION),
            _adc(8500.0),
        ],
        10000.0,
    )


def plot_fse():
    tissue = TissueProperties(
        t1_ms=np.array([700.0, 1000.0, 1400.0], np.float32),
        t2_ms=np.array([40.0, 70.0, 110.0], np.float32),
    )
    signal = FSE().simulate(_fse(), tissue, record="echo").signal.abs().cpu()
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    for index, t2_ms in enumerate((40, 70, 110)):
        ax.plot(np.arange(1, 9), signal[:, index], "o-", label=f"T2 = {t2_ms} ms")
    ax.set(
        xlabel="echo number",
        ylabel=r"$|F_0^+|$",
        title="FSE zoo policy: crusher -> refocusing RF -> crusher",
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fse_signal_evolution.png", dpi=170)
    plt.close(fig)


def plot_mprage():
    tissue = TissueProperties(
        t1_ms=np.array([700.0, 1000.0, 1400.0], np.float32),
        t2_ms=70.0,
    )
    signal = SPGR().simulate(_mprage(), tissue, record="all").signal.abs().cpu()
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    for index, t1_ms in enumerate((700, 1000, 1400)):
        ax.plot(
            np.arange(1, 13),
            signal[:, index],
            "o-",
            label=f"T1 = {t1_ms} ms",
        )
    ax.set(
        xlabel="SPGR readout in inversion cycle",
        ylabel=r"$|F_0^+|$",
        title="MPRAGE zoo policy: inversion followed by spoiled GRE readouts",
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "mprage_signal_evolution.png", dpi=170)
    plt.close(fig)


def plot_steady_states():
    tissue = TissueProperties(t1_ms=1000.0, t2_ms=70.0)
    description = _steady_state()
    cases = [
        ("SPGR", SPGR(), description),
        ("SSFP-Echo", SSFPEcho(), description),
        ("SSFP-Fid", SSFPFID(), description),
        ("bSSFP", BSSFP(), _steady_state(phase_cycle=True)),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    for label, interpreter, desc in cases:
        repetitions = 80 if label != "bSSFP" else 40
        signal = (
            interpreter.simulate(desc, tissue, repetitions=repetitions)
            .signal[:, 0]
            .abs()
            .cpu()
        )
        ax.plot(np.arange(1, signal.numel() + 1), signal, label=label)
    ax.set(
        xlabel="recorded ADC",
        ylabel=r"$|F_0^+|$",
        title="Steady-state zoo policies from serialized event streams",
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "steady_state_signal_evolution.png", dpi=170)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plot_fse()
    plot_mprage()
    plot_steady_states()


if __name__ == "__main__":
    main()
