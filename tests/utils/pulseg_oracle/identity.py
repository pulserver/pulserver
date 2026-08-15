"""Normalized block identity, computed from content.

Two blocks share an identity exactly when the PulSeg normalization rules
say they are the same base block: waveform SHAPES by value (never by
file-local id, so identities compare across files of a chain), gradient
trapezoids by timing, extended/arbitrary gradients by delay plus time
structure, ADC windows by count/dwell/delay, and the block duration.
Amplitudes, phase/frequency offsets, rotations, labels and triggers are
execution parameters and stay out.
"""

from __future__ import annotations

import numpy as np


def _shape_key(values) -> tuple:
    """A waveform shape by value, rounded to spec-irrelevant precision."""
    if values is None:
        return ()
    return tuple(np.round(np.asarray(values, dtype=float), 9).tolist())


def _rf_key(rf) -> tuple:
    if rf is None:
        return ()
    signal = np.asarray(rf.signal, dtype=complex)
    peak = float(np.abs(signal).max()) or 1.0
    return (
        "rf",
        _shape_key(np.round(np.abs(signal) / peak, 6)),
        _shape_key(np.round(np.angle(signal), 6)),
        _shape_key(rf.t),
        round(float(rf.delay), 9),
    )


def _grad_key(grad) -> tuple:
    """Gradient identity at segment level, the interpreter's reading.

    Trapezoids carry their four times (the spec's timing identity).
    Sampled gradients carry presence and delay only: pulserver extends the
    spec's co-timed shot-variant model to variants with their OWN time
    structures -- an explicit-arm stack's per-arm rewinders differ in
    vertex times, and each instance selects its shape through the instance
    table. This is a declared derived-class extension (see
    docs/PULSEG_CONFORMANCE.md section 5)."""
    if grad is None:
        return ()
    if grad.type == "trap":
        return (
            "trap",
            round(float(grad.delay), 9),
            round(float(grad.rise_time), 9),
            round(float(grad.flat_time), 9),
            round(float(grad.fall_time), 9),
        )
    return ("grad", round(float(grad.delay), 9))


def _adc_key(adc) -> tuple:
    if adc is None:
        return ()
    return (
        "adc",
        int(adc.num_samples),
        round(float(adc.dwell), 12),
        round(float(adc.delay), 9),
    )


def block_signature(block) -> tuple:
    """The normalized identity of one block view (``Sequence.get_block``)."""
    return (
        round(float(block.block_duration), 9),
        _rf_key(block.rf),
        _grad_key(block.gx),
        _grad_key(block.gy),
        _grad_key(block.gz),
        _adc_key(block.adc),
    )


def structural_signature(block) -> tuple:
    """The identity SEGMENTS repeat on.

    Two relaxations of the full signature, both spec-sanctioned:

    - the ADC stays out -- acquisition presence is instance metadata (a
      dummy shot is the same unit played without acquiring); the window's
      identity is checked separately across the instances that acquire;
    - a PURE DELAY matches any pure delay, whatever its duration -- the
      spec's reserved implicit-delay blocks carry duration per instance in
      ``block_duration``, so a TI/TR fill that varies is one position.
    """
    full = block_signature(block)
    if block.rf is None and block.gx is None and block.gy is None and \
            block.gz is None and block.adc is None:
        return ("delay",)
    return full[:5]


def once_class(block) -> int:
    """The block's play-once class: the running ONCE label state it sets.

    Returns 1/0 when the block SETS the flag, -1 when it leaves it alone --
    the caller carries the running state, exactly as an interpreter does.
    """
    for kind, name, value in block.labels:
        if kind == "SET" and name == "ONCE":
            return int(value)
    return -1


def sequence_signatures(seq) -> list[tuple]:
    """One signature per block, in stream order."""
    return [block_signature(seq.get_block(i)) for i in range(1, seq.num_blocks + 1)]
