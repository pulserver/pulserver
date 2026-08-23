"""The synthetic `.seq` corpus for the C safety/structure ctests.

Two lanes:

* :data:`TEXT_CASES` -- files whose exact event table *is* the test input.
  Most are deliberately invalid (gradient amplitude/slew violations,
  non-connecting extended gradients, waveforms that start high), which no
  sequence designer should be able to express; they are therefore written
  as exact text, not built through ``pulserver.pypulseq``. Each entry
  documents the property its consumers assert.
* :func:`write_built_cases` -- valid sequences produced through
  ``pulserver.pypulseq`` builders (the 32x32 blipped GRE, the NextSequence
  dedup chain, the corrupted-signature specimen).
* :data:`MALFORMED_CASES` and the corrupted specimen -- files that must *not*
  read, written into ``expected/malformed/`` so that the sweeps over
  ``expected/`` keep their premise that everything in it reads.

``write_all`` writes both lanes into a directory. Output is deterministic:
running it twice leaves the tree unchanged.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Text lane
# ---------------------------------------------------------------------------

_RASTERS = """\
[DEFINITIONS]
AdcRasterTime 1e-07
BlockDurationRaster 1e-05
GradientRasterTime 1e-05
RadiofrequencyRasterTime 1e-06
"""

#: The three 2-sample shapes every hand-specified RF pulse rides on:
#: flat magnitude, zero phase, and a linear time ramp giving the duration.
_RF_SHAPES = """\
[SHAPES]
shape_id 1
num_samples 2
1
1
shape_id 2
num_samples 2
0
0
shape_id 3
num_samples 2
0
1000
"""

#: 3-sample extended-gradient shapes: a triangle (0-1-0) and its time axis.
_TRIANGLE_SHAPES = """\
[SHAPES]
shape_id 1
num_samples 3
0
1
0
shape_id 2
num_samples 3
0
10
20
"""


def _seq(purpose: str, body: str, *, definitions: str = _RASTERS) -> str:
    return (
        "# Pulseq sequence file\n"
        f"# {purpose}\n"
        "# Written by tests/utils/synthetic_fixtures.py\n"
        "\n"
        "[VERSION]\n"
        "major 1\n"
        "minor 5\n"
        "revision 1\n"
        "\n" + definitions + "\n" + body
    )


def _shapes3(*samples: str) -> str:
    out = ["[SHAPES]"]
    for i, s in enumerate(samples, start=1):
        out.append(f"shape_id {i}")
        out.append(f"num_samples {len(s.split())}")
        out.extend(s.split())
    return "\n".join(out) + "\n"


# -- gradient-safety family (test_safety_grad.c) ----------------------------

_GRAD_CASES: dict[str, str] = {}

_GRAD_CASES["01_grad_amplitude_violation"] = _seq(
    "A 1 ms flat trapezoid whose 20 Hz/m plateau exceeds a max_grad set below it.",
    """\
[BLOCKS]
1 100   0   1   0   0  0  0

[TRAP]
 1           20  10  980  10   0
""",
)

_GRAD_CASES["01_ok_trap_extended_trap"] = _seq(
    "Trapezoid, connecting extended triangle, trapezoid: a legal chain on one axis.",
    """\
[BLOCKS]
1 100   0   1   0   0  0  0
2  20   0   2   0   0  0  0
3 100   0   1   0   0  0  0

[GRADIENTS]
2       100000            0            0 1 2 0

[TRAP]
 1  1.20482e+06 170  660 170   0

"""
    + _TRIANGLE_SHAPES,
)

_GRAD_CASES["02_slew_violation"] = _seq(
    "A 0.2 Hz/m trapezoid with 10 ms ramps: amplitude is legal, the slew is not.",
    """\
[BLOCKS]
1 300   0   1   0   0  0  0

[TRAP]
 1          0.2 1000 1000 1000   0
""",
)

_GRAD_CASES["02_fail_trap_then_startshigh"] = _seq(
    "An extended gradient that starts at full amplitude right after a trapezoid ends at zero.",
    """\
[BLOCKS]
1 100   0   1   0   0  0  0
2  20   0   2   0   0  0  0

[GRADIENTS]
2       100000       100000            0 1 2 0

[TRAP]
 1  1.20482e+06 170  660 170   0

"""
    + _shapes3("1 1 0", "0 10 20"),
)

_GRAD_CASES["03_grad_rss_violation"] = _seq(
    "Two axes at 13 Hz/m each: each within max_grad, their root-sum-square above it.",
    """\
[BLOCKS]
1 100   0   1   1   0  0  0

[TRAP]
 1            13  10  980  10   0
""",
)

_GRAD_CASES["03_fail_startshigh_first"] = _seq(
    "The very first block opens with an extended gradient already at full amplitude.",
    """\
[BLOCKS]
1  20   0   1   0   0  0  0

[GRADIENTS]
1       100000       100000            0 1 2 0

"""
    + _shapes3("1 1 0", "0 10 20"),
)

_GRAD_CASES["04_slew_rss_violation"] = _seq(
    "Two axes whose per-axis slew is legal but whose root-sum-square slew is not.",
    """\
[BLOCKS]
1 300   0   1   1   0  0  0

[TRAP]
 1         0.13 1000 1000 1000   0
""",
)

_GRAD_CASES["04_fail_delay_then_allhigh"] = _seq(
    "A pure delay, then all three axes open at full amplitude at once.",
    """\
[BLOCKS]
1 100   0   0   0   0  0  0
2  20   0   1   0   0  0  0

[GRADIENTS]
1       100000       100000       100000 1 2 0

"""
    + _shapes3("1 1 1", "0 10 20"),
)

_GRAD_CASES["05_ok_extended_with_delay"] = _seq(
    "A single delayed extended triangle: legal, and the arbitrary-gradient binary source.",
    """\
[BLOCKS]
1  30   0   1   0   0  0  0

[GRADIENTS]
1       100000            0            0 1 2 100

"""
    + _TRIANGLE_SHAPES,
)

_GRAD_CASES["06_fail_delay_then_startshigh"] = _seq(
    "A pure delay, then an extended gradient that opens at full amplitude.",
    """\
[BLOCKS]
1 100   0   0   0   0  0  0
2  20   0   1   0   0  0  0

[GRADIENTS]
1       100000       100000            0 1 2 0

"""
    + _shapes3("1 1 0", "0 10 20"),
)

_GRAD_CASES["07_fail_nonconnecting"] = _seq(
    "Adjacent extended gradients whose boundary amplitudes do not connect (ends at 1, restarts at 2x).",
    """\
[BLOCKS]
1  20   0   1   0   0  0  0
2  20   0   2   0   0  0  0

[GRADIENTS]
1       100000            0       100000 1 2 0
2       200000       200000            0 3 2 0

"""
    + _shapes3("0 1 1", "0 10 20", "1 0.5 0"),
)


def _rot_definitions() -> str:
    return _RASTERS + "RequiredExtensions ROTATIONS\n"


def _rot_case(purpose: str, body: str) -> str:
    return _seq(purpose, body, definitions=_rot_definitions())


_GRAD_CASES["08_ok_rot_identity"] = _rot_case(
    "The legal trap/extended/trap chain with an identity rotation on every block.",
    """\
[BLOCKS]
1 100   0   1   0   0  0  1
2  20   0   2   0   0  0  1
3 100   0   1   0   0  0  1

[GRADIENTS]
2       100000            0            0 1 2 0

[TRAP]
 1  1.20482e+06 170  660 170   0

[EXTENSIONS]
1 1 1 0

extension ROTATIONS 1
1  1 0 0 0

"""
    + _TRIANGLE_SHAPES,
)

_GRAD_CASES["09_fail_rot_identity"] = _rot_case(
    "The starts-high violation under an identity rotation: rotating does not excuse it. Also the rotation-quaternion binary source.",
    """\
[BLOCKS]
1 100   0   1   0   0  0  1
2  20   0   2   0   0  0  1

[GRADIENTS]
2       100000       100000            0 1 2 0

[TRAP]
 1  1.20482e+06 170  660 170   0

[EXTENSIONS]
1 1 1 0

extension ROTATIONS 1
1  1 0 0 0

"""
    + _shapes3("1 1 0", "0 10 20"),
)

_GRAD_CASES["10_fail_rot_first_block"] = _rot_case(
    "The first block opens at full amplitude, rotated.",
    """\
[BLOCKS]
1  20   0   1   0   0  0  1

[GRADIENTS]
1       100000       100000            0 1 2 0

[EXTENSIONS]
1 1 1 0

extension ROTATIONS 1
1  1 0 0 0

"""
    + _shapes3("1 1 0", "0 10 20"),
)

_GRAD_CASES["11_fail_rot_allhigh"] = _rot_case(
    "Delay, then all three axes open at full amplitude on a rotated block.",
    """\
[BLOCKS]
1 100   0   0   0   0  0  0
2  20   0   1   0   0  0  1

[GRADIENTS]
1       100000       100000       100000 1 2 0

[EXTENSIONS]
1 1 1 0

extension ROTATIONS 1
1  1 0 0 0

"""
    + _shapes3("1 1 1", "0 10 20"),
)

_GRAD_CASES["12_ok_rot_extended_delay"] = _rot_case(
    "A delayed extended triangle under a rotation: legal.",
    """\
[BLOCKS]
1  30   0   1   0   0  0  1

[GRADIENTS]
1       100000            0            0 1 2 100

[EXTENSIONS]
1 1 1 0

extension ROTATIONS 1
1  1 0 0 0

"""
    + _TRIANGLE_SHAPES,
)

_GRAD_CASES["13_fail_rot_delay_then_startshigh"] = _rot_case(
    "Delay, then a rotated extended gradient that opens at full amplitude.",
    """\
[BLOCKS]
1 100   0   0   0   0  0  0
2  20   0   1   0   0  0  1

[GRADIENTS]
1       100000       100000            0 1 2 0

[EXTENSIONS]
1 1 1 0

extension ROTATIONS 1
1  1 0 0 0

"""
    + _shapes3("1 1 0", "0 10 20"),
)

_GRAD_CASES["14_fail_rot_nonconnecting"] = _rot_case(
    "Non-connecting extended gradients under one shared rotation.",
    """\
[BLOCKS]
1  20   0   1   0   0  0  1
2  20   0   2   0   0  0  1

[GRADIENTS]
1       100000            0       100000 1 2 0
2       200000       200000            0 3 2 0

[EXTENSIONS]
1 1 1 0

extension ROTATIONS 1
1  1 0 0 0

"""
    + _shapes3("0 1 1", "0 10 20", "1 0.5 0"),
)

_GRAD_CASES["15_ok_rot_same_rotation"] = _rot_case(
    "Connecting extended gradients that share one non-identity rotation: legal.",
    """\
[BLOCKS]
1  20   0   1   0   0  0  1
2  20   0   2   0   0  0  1

[GRADIENTS]
1       100000            0       100000 1 2 0
2       100000       100000            0 3 2 0

[EXTENSIONS]
1 1 1 0

extension ROTATIONS 1
1  0.707107 0 0 0.707107

"""
    + _shapes3("0 1 1", "0 10 20", "1 1 0"),
)

_GRAD_CASES["16_fail_rot_diff_rotation_1"] = _rot_case(
    "Gradients connect in amplitude but the rotation changes mid-flow (identity, then 90 deg).",
    """\
[BLOCKS]
1  20   0   1   0   0  0  1
2  20   0   2   0   0  0  2

[GRADIENTS]
1       100000            0       100000 1 2 0
2       100000       100000            0 3 2 0

[EXTENSIONS]
1 1 1 0
2 1 2 0

extension ROTATIONS 1
1  1 0 0 0
2  0.707107 0 0 0.707107

"""
    + _shapes3("0 1 1", "0 10 20", "1 1 0"),
)

_GRAD_CASES["17_fail_rot_diff_rotation_2"] = _rot_case(
    "Gradients connect in amplitude but the rotation changes mid-flow (90 deg, then identity).",
    """\
[BLOCKS]
1  20   0   1   0   0  0  1
2  20   0   2   0   0  0  2

[GRADIENTS]
1       100000            0       100000 1 2 0
2       100000       100000            0 3 2 0

[EXTENSIONS]
1 1 1 0
2 1 2 0

extension ROTATIONS 1
1  0.707107 0 0 0.707107
2  1 0 0 0

"""
    + _shapes3("0 1 1", "0 10 20", "1 1 0"),
)


# -- RF-statistics family (test_rf_stats.c, test_ptx_getters.c) -------------

_RF_CASES: dict[str, str] = {}

_RF_CASES["00_basic_rfstat"] = _seq(
    "One 500 us block pulse at 500 Hz: the single-channel RF-statistics baseline.",
    """\
[BLOCKS]
1 100   1   0   0   0  0  0

[RF]
1          500 1 2 3 500 0 0 0 0 0 e

"""
    + _RF_SHAPES,
)


def _rf_interleaved(rf_rows: list[str], block_rf_ids: list[int]) -> str:
    """Blocks alternating an RF id with a readout trapezoid, 100 raster units each."""
    lines = ["[BLOCKS]"]
    for i, rf_id in enumerate(block_rf_ids, start=1):
        if rf_id:
            lines.append(f"{i:>2d} 100   {rf_id:>2d}   0   0   0  0  0")
        else:
            lines.append(f"{i:>2d} 100   0   1   0   0  0  0")
    lines.append("")
    lines.append("[RF]")
    lines.extend(rf_rows)
    lines.append("")
    lines.append("[TRAP]")
    lines.append(" 1  1.20482e+06 170  660 170   0")
    lines.append("")
    return "\n".join(lines) + _RF_SHAPES


def _mrf_body() -> str:
    # One inversion at 500 Hz, then a 10-step amplitude ramp (25..250 Hz)
    # interleaved with readouts, played twice: a two-TR MR-fingerprinting
    # miniature whose per-TR RF amplitudes vary but repeat exactly.
    rf_rows = ["1          500 1 2 3 500 0 0 0 0 0 i"]
    for i in range(10):
        rf_rows.append(f"{i + 2:>2d}          {25 * (i + 1):>3d} 1 2 3 500 0 0 0 0 0 e")
    ids: list[int] = []
    for _tr in range(2):
        ids.append(1)
        for i in range(10):
            ids.extend((i + 2, 0))
    return _rf_interleaved(rf_rows, ids)


_RF_CASES["01_rfamp_ok_mrfingerprinting"] = _seq(
    "Two identical TRs whose interior RF amplitude ramps 25..250 Hz: variable amplitude that repeats per TR.",
    _mrf_body(),
)


def _vfa_body() -> str:
    # A single 80-block train ramping 25..250 Hz in tiers of four pulses:
    # the amplitude pattern never repeats, so no TR periodicity exists.
    rf_rows = [
        f"{i + 1:>2d}          {25 * (i + 1):>3d} 1 2 3 500 0 0 0 0 0 e"
        for i in range(10)
    ]
    ids: list[int] = []
    for tier in range(10):
        for _rep in range(4):
            ids.extend((tier + 1, 0))
    return _rf_interleaved(rf_rows, ids)


_RF_CASES["02_rfamp_fail_vfa"] = _seq(
    "A variable-flip-angle train whose amplitude schedule never repeats: no TR periodicity to find.",
    _vfa_body(),
)

_SHIM_A = (
    "1 16 1 -0 1 -0.392699 1 -0.785398 1 -1.1781 1 -1.5708 1 -1.9635 1 -2.35619"
    " 1 -2.74889 1 -3.14159 1 2.74889 1 2.35619 1 1.9635 1 1.5708 1 1.1781"
    " 1 0.785398 1 0.392699"
)
_SHIM_B = (
    "2 16 1 -0 1 -0.785398 1 -1.5708 1 -2.35619 1 -3.14159 1 2.35619 1 1.5708"
    " 1 0.785398 1 2.44929e-16 1 -0.785398 1 -1.5708 1 -2.35619 1 -3.14159"
    " 1 2.35619 1 1.5708 1 0.785398"
)


def _pnp_body() -> str:
    # Two identical TRs: one unshimmed 500 Hz pulse, then ten shimmed 250 Hz
    # pulses alternating between two 16-channel shim sets, each followed by a
    # readout. The shim alternation repeats exactly per TR.
    lines = ["[BLOCKS]"]
    n = 1
    for _tr in range(2):
        lines.append(f"{n:>2d} 100   1   0   0   0  0  0")
        n += 1
        for i in range(10):
            shim = 1 + (i % 2)
            lines.append(f"{n:>2d} 100   2   0   0   0  0  {shim}")
            n += 1
            lines.append(f"{n:>2d} 100   0   1   0   0  0  0")
            n += 1
    body = "\n".join(lines)
    return (
        body
        + """

[RF]
1          500 1 2 3 500 0 0 0 0 0 e
2          250 1 2 3 500 0 0 0 0 0 e

[TRAP]
 1  1.20482e+06 170  660 170   0

[EXTENSIONS]
1 1 1 0
2 1 2 0

extension RF_SHIMS 1
"""
        + _SHIM_A
        + "\n"
        + _SHIM_B
        + "\n\n"
        + _RF_SHAPES
    )


_RF_CASES["03_rfshim_ok_pnpmrfingerprinting"] = _seq(
    "Two identical TRs alternating between two 16-channel shim sets: shim variation that repeats per TR.",
    _pnp_body(),
)


def _shim_fail_body() -> str:
    # Twenty RF/readout pairs: the first ten under shim set 1, the last ten
    # under shim set 2. The shim schedule never repeats, so no periodicity.
    lines = ["[BLOCKS]"]
    n = 1
    for i in range(20):
        shim = 1 if i < 10 else 2
        lines.append(f"{n:>2d} 100   1   0   0   0  0  {shim}")
        n += 1
        lines.append(f"{n:>2d} 100   0   1   0   0  0  0")
        n += 1
    body = "\n".join(lines)
    return (
        body
        + """

[RF]
1          250 1 2 3 500 0 0 0 0 0 e

[TRAP]
 1  1.20482e+06 170  660 170   0

[EXTENSIONS]
1 1 1 0
2 1 2 0

extension RF_SHIMS 1
"""
        + _SHIM_A
        + "\n"
        + _SHIM_B
        + "\n\n"
        + _RF_SHAPES
    )


_RF_CASES["04_rfshim_fail_gre"] = _seq(
    "A GRE-like train that switches shim set halfway and never returns: no shim periodicity.",
    _shim_fail_body(),
)

_RF_CASES["05_rf_ladder_no_adc"] = _seq(
    "An RF amplitude ladder (125/250/500 Hz) with slice gradients and no ADC anywhere.",
    """\
[BLOCKS]
1 100   1   0   0   0  0  0
2 100   2   1   0   0  0  0
3 100   2   1   0   0  0  0
4 100   3   0   0   0  0  0

[RF]
1          125 1 2 3 500 0 0 0 0 0 e
2          250 1 2 3 500 0 0 0 0 0 e
3          500 1 2 3 500 0 0 0 0 0 r

[TRAP]
 1  1.20482e+06 170  660 170   0

"""
    + _RF_SHAPES,
)

_RF_CASES["06_rfamp_two_tier_trs"] = _seq(
    "Two structurally identical 6-block TRs whose RF amplitude differs (150 vs 225 Hz): "
    "accepted only when variable RF amplitude is allowed, and the 225 Hz tier is the "
    "worst-B1rms envelope.",
    """\
[BLOCKS]
 1 100   0   0   0   0  0  0
 2 100   1   1   0   0  0  0
 3 100   1   1   0   0  0  0
 4 100   1   1   0   0  0  0
 5 100   1   1   0   0  0  0
 6 1000   0   0   0   0  0  0
 7 100   0   0   0   0  0  0
 8 100   2   1   0   0  0  0
 9 100   2   1   0   0  0  0
10 100   2   1   0   0  0  0
11 100   2   1   0   0  0  0
12 1000   0   0   0   0  0  0

[RF]
1          150 1 2 3 500 0 0 0 0 0 e
2          225 1 2 3 500 0 0 0 0 0 e

[TRAP]
 1  1.20482e+06 170  660 170   0

"""
    + _RF_SHAPES,
)

_RF_CASES["07_rfstat_cp_8ch_180"] = _seq(
    "The 500 Hz baseline pulse under an 8-channel CP shim (1/sqrt(8) per channel): "
    "quadrature aggregation must reproduce the single-channel statistics.",
    """\
[BLOCKS]
1 100   1   0   0   0  0  1

[RF]
1          500 1 2 3 500 0 0 0 0 0 e

[EXTENSIONS]
1 1 1 0

extension RF_SHIMS 1
1 8 0.353553 -0 0.353553 -0.785398 0.353553 -1.5708 0.353553 -2.35619 0.353553 -3.14159 0.353553 2.35619 0.353553 1.5708 0.353553 0.785398

"""
    + _RF_SHAPES,
)


# -- interior-delay pair (test_sequences.c delay classification) ------------


def _interior_delay(third_gap_raster: int) -> str:
    lines = ["[BLOCKS]"]
    n = 1
    for tr in range(3):
        gap = third_gap_raster if tr == 2 else 30
        for dur, rf, adc in ((20, 1, 0), (gap, 0, 0), (32, 0, 1), (200, 0, 0)):
            lines.append(f"{n:>2d} {dur:>3d}   {rf}   0   0   0  {adc}  0")
            n += 1
    body = "\n".join(lines)
    return (
        body
        + """

[RF]
1      138.889 1 2 3 100 0 0 0 0 0 e

[ADC]
1 32 10000 0 0 0 0 0 0

[SHAPES]
shape_id 1
num_samples 2
1
1
shape_id 2
num_samples 2
0
0
shape_id 3
num_samples 2
0
20
"""
    )


_DELAY_DEFS_TEMPLATE = """\
[DEFINITIONS]
AdcRasterTime 1e-07
BlockDurationRaster 1e-05
GradientRasterTime 1e-05
Name delay_test
RadiofrequencyRasterTime 1e-05
TotalDuration {total}
"""

_OTHER_CASES: dict[str, str] = {}

_OTHER_CASES["99_interior_delay_static"] = _seq(
    "Three TRs of RF / interior delay / ADC / trailing delay, the interior gap identical in every TR: a static delay.",
    _interior_delay(30),
    definitions=_DELAY_DEFS_TEMPLATE.format(total="0.00846"),
)

_OTHER_CASES["99_interior_delay_dynamic"] = _seq(
    "Three TRs of RF / interior delay / ADC / trailing delay, the third TR's interior gap doubled: a dynamic delay position.",
    _interior_delay(60),
    definitions=_DELAY_DEFS_TEMPLATE.format(total="0.00876"),
)


# -- TRID label fixtures (test_trid_labels.c) -------------------------------


def _trid(purpose: str, blocks: str, extensions: str) -> str:
    return _seq(
        purpose,
        "[BLOCKS]\n" + blocks + "\n[EXTENSIONS]\n" + extensions,
    )


_OTHER_CASES["trid_two"] = _trid(
    "Two TRID-labeled groups: group 1 spans blocks 1-2 (200 us), group 2 is block 3 alone (50 us).",
    """\
1 10   0   0   0   0  0  1
2 10   0   0   0   0  0  0
3 5    0   0   0   0  0  2
""",
    """\
1 1 1 0
2 1 2 0

extension LABELSET 1
1 1 TRID
2 2 TRID
""",
)

_OTHER_CASES["trid_repeated_tr"] = _trid(
    "One TRID group, three identical TRs back to back: repeats of the SAME TRID across "
    "a TR boundary must split into separate occurrences rather than merge into one run.",
    """\
1 10   0   0   0   0  0  1
2 10   0   0   0   0  0  1
3 10   0   0   0   0  0  1
""",
    """\
1 1 1 0

extension LABELSET 1
1 1 TRID
""",
)

_OTHER_CASES["trid_mismatch"] = _trid(
    "TRID 1 appears twice (blocks 1 and 3) with different durations: ill-formed labeling "
    "that pulseg_get_tr_groups() must reject with PULSEG_ERR_TRID_STRUCTURAL_MISMATCH.",
    """\
1 10   0   0   0   0  0  1
2 5    0   0   0   0  0  2
3 20   0   0   0   0  0  3
""",
    """\
1 1 1 0
2 1 2 0
3 1 3 0

extension LABELSET 1
1 1 TRID
2 2 TRID
3 1 TRID
""",
)


# -- raster-alignment family (test_raster.c) --------------------------------
#
# Every file here declares the standard rasters -- RF 1 us, gradient and block
# 10 us -- and is played against a system whose rasters are twice as coarse.
# That pairing passes the declared-versus-system grid check, which accepts
# either direction, so what each case exercises is the per-event alignment
# check and nothing before it.

_RASTER_CASES: dict[str, str] = {}

_RASTER_CASES["20_raster_ok_on_coarse_system"] = _seq(
    "Every time a multiple of a system raster twice as coarse as the declared one.",
    """\
[BLOCKS]
1 110   1   1   0   0  0  0

[RF]
1          500 1 2 3 500 4 0 0 0 0 e

[TRAP]
 1          0.2  20  960  20   0

"""
    + _RF_SHAPES,
)

_RASTER_CASES["21_raster_rf_delay_misaligned"] = _seq(
    "RF delayed 3 us: a whole number of the declared 1 us raster, not of a 2 us one.",
    """\
[BLOCKS]
1 100   1   0   0   0  0  0

[RF]
1          500 1 2 3 500 3 0 0 0 0 e

"""
    + _RF_SHAPES,
)

_RASTER_CASES["22_raster_trap_ramp_misaligned"] = _seq(
    "A 10 us trapezoid ramp: on the declared 10 us gradient raster, not on a 20 us one.",
    """\
[BLOCKS]
1 100   0   1   0   0  0  0

[TRAP]
 1          0.2  10  980  10   0
""",
)

_RASTER_CASES["23_raster_block_duration_misaligned"] = _seq(
    "A 30 us block: three ticks of the declared raster, one and a half of a 20 us one.",
    """\
[BLOCKS]
1   3   0   0   0   0  0  0
""",
)

_RASTER_CASES["24_raster_block_duration_overrun"] = _seq(
    "A 1 ms trapezoid in a 40 us block: every time aligned, the gradient still overruns.",
    """\
[BLOCKS]
1   4   0   1   0   0  0  0

[TRAP]
 1          0.2  20  960  20   0
""",
)


_RASTER_CASES["28_unknown_extension_type"] = _seq(
    "An extension type this reader has no name for, whose reference it cannot resolve.",
    """\
[BLOCKS]
1 100   0   1   0   0  0  1

[TRAP]
 1          0.2  20  960  20   0

[EXTENSIONS]
1 1 7 0

extension VENDOR_THING 1
1  1 2 3
""",
)


# -- malformed family (files that must NOT read) ---------------------------
#
# Structurally invalid: an id naming a library row that is not there. Every
# other case in this corpus reads, and the sweeps over ``expected/`` assume it,
# so these live in their own directory -- a file that must be refused cannot sit
# in a set whose premise is that reading succeeds.

MALFORMED_CASES: dict[str, str] = {}

MALFORMED_CASES["26_dangling_rotation_ref"] = _seq(
    "An extension row naming rotation 99 in a file that defines one rotation.",
    """\
[BLOCKS]
1 100   0   1   0   0  0  1

[TRAP]
 1          0.2  20  960  20   0

[EXTENSIONS]
1 1 99 0

extension ROTATIONS 1
1  1 0 0 0
""",
    definitions=_RASTERS + "RequiredExtensions ROTATIONS\n",
)

MALFORMED_CASES["27_dangling_extension_link"] = _seq(
    "An extension row whose `next` link points past the end of the chain library.",
    """\
[BLOCKS]
1 100   0   1   0   0  0  1

[TRAP]
 1          0.2  20  960  20   0

[EXTENSIONS]
1 1 1 99

extension ROTATIONS 1
1  1 0 0 0
""",
    definitions=_RASTERS + "RequiredExtensions ROTATIONS\n",
)

MALFORMED_CASES["25_dangling_rf_id"] = _seq(
    "A block naming RF 1 in a file with no [RF] section at all.",
    """\
[BLOCKS]
1 100   1   1   0   0  0  0

[TRAP]
 1          0.2  20  960  20   0
""",
)


#: name -> exact file text for every hand-specified case.
TEXT_CASES: dict[str, str] = {
    **_GRAD_CASES,
    **_RF_CASES,
    **_RASTER_CASES,
    **_OTHER_CASES,
}


# ---------------------------------------------------------------------------
# Built lane
# ---------------------------------------------------------------------------


def build_pe_blip():
    """A 32x32 Cartesian GRE written WITHOUT deduplication.

    Its consumers (k-trajectory memoization, moments table, k-space center
    derivation) need a file where one physical readout appears under many
    gradient ids -- the dedup-off write is the point of the fixture.
    """
    from pulserver.app import gre2D_sequence

    return gre2D_sequence.main(
        fov=0.256,
        n_x=32,
        n_y=32,
        flip_angle_deg=15.0,
        te=5e-3,
        tr=2.0,
        n_dummy=0,
    )


def build_nav_gre():
    """A navigated Cartesian miniature: each TR acquires an x readout and
    then a z navigator readout, so the scan is not one Cartesian readout
    axis and automatic labeling must refuse it.
    """
    import pulserver.pypulseq as pp

    system = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    seq = pp.Sequence(system=system)
    n_read, n_pe, fov = 32, 4, 0.22
    for line in range(n_pe):
        seq.add_block(
            pp.make_block_pulse(flip_angle=0.2, duration=200e-6, system=system)
        )
        seq.add_block(
            pp.make_trapezoid(
                channel="y",
                area=(line - n_pe / 2) / fov,
                duration=400e-6,
                system=system,
            )
        )
        gx = pp.make_trapezoid(
            channel="x", flat_area=n_read / fov, flat_time=320e-6, system=system
        )
        seq.add_block(
            gx,
            pp.make_adc(
                num_samples=n_read, duration=320e-6, delay=gx.rise_time, system=system
            ),
        )
        gz = pp.make_trapezoid(
            channel="z", flat_area=n_read / fov, flat_time=320e-6, system=system
        )
        seq.add_block(
            gz,
            pp.make_adc(
                num_samples=n_read, duration=320e-6, delay=gz.rise_time, system=system
            ),
        )
    seq.set_definition("Name", "gre_nav_2d")
    return seq


def write_built_cases(directory: Path, corpus_dir: Path) -> list[str]:
    """Write the builder-produced fixtures. Returns the file names written.

    Parameters
    ----------
    directory : Path
        Output directory (``tests/utils/expected``).
    corpus_dir : Path
        The zoo corpus (``tests/python/fixtures``); source of the
        corrupted-signature specimen.
    """
    from pulserver.app import gre2D_sequence

    written: list[str] = []

    blip = build_pe_blip()
    blip.write(directory / "gre_32x32_pe_blip.seq", remove_duplicates=False)
    written.append("gre_32x32_pe_blip.seq")

    build_nav_gre().write(directory / "gre_nav_2d.seq")
    written.append("gre_nav_2d.seq")

    # A NextSequence chain of two identical small GREs: the cross-subsequence
    # dedup case (test_pulsegen_cache).  The same sequence is written twice;
    # only the lead file carries the chain pointer.
    pair = gre2D_sequence.main(n_x=32, n_y=8, te=5e-3, tr=30e-3, n_dummy=2)
    pair.write(directory / "dedup_gre_pair_b.seq")
    pair.set_definition("NextSequence", "dedup_gre_pair_b.seq")
    pair.write(directory / "dedup_gre_pair.seq")
    written.extend(["dedup_gre_pair_b.seq", "dedup_gre_pair.seq"])

    written.append(write_corrupted(directory, corpus_dir / "gre_2d.seq"))
    return written


def write_corrupted(directory: Path, source: Path) -> str:
    """Write ``malformed/gre_2d_corrupted.seq``: `source` with block 1 pointing
    at a dangling extension chain, signature left untouched.

    Two consumers, two properties: the signature-checking loader must report a
    mismatch, and the reader must refuse the dangling chain id outright.
    """
    text = source.read_text()
    lines = text.splitlines(keepends=True)
    in_blocks = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "[BLOCKS]":
            in_blocks = True
            continue
        if in_blocks and stripped and not stripped.startswith("#"):
            fields = line.split()
            fields[-1] = "99"
            lines[i] = " ".join(fields) + "\n"
            break
    malformed = directory / "malformed"
    malformed.mkdir(exist_ok=True)
    out = malformed / "gre_2d_corrupted.seq"
    out.write_text("".join(lines))
    return f"malformed/{out.name}"


def write_all(directory: Path, corpus_dir: Path) -> list[str]:
    """Write every synthetic fixture into `directory`; returns the names."""
    directory.mkdir(exist_ok=True)
    written: list[str] = []
    for name, text in sorted(TEXT_CASES.items()):
        (directory / f"{name}.seq").write_text(text)
        written.append(f"{name}.seq")
    malformed = directory / "malformed"
    malformed.mkdir(exist_ok=True)
    for name, text in sorted(MALFORMED_CASES.items()):
        (malformed / f"{name}.seq").write_text(text)
        written.append(f"malformed/{name}.seq")
    written.extend(write_built_cases(directory, corpus_dir))
    return written
