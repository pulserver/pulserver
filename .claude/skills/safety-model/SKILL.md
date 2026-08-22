---
name: safety-model
description: >
  Change a safety check — gradient amplitude, slew, continuity, PNS
  (Irnich/SAFE) or mechanical resonance — or add a new one. Use when touching
  the canonical TR window, a nerve model, the forbidden-band criterion, or any
  fast path in the safety engine.
---

# Changing a safety check

These are the checks a scanner runs before it will play a sequence, and
Pulserver runs the same compiled code at design time so a refusal happens at
the desk rather than at the magnet. Read
`docs/explanations/safety/` before changing the physics, and
`docs/explanations/performance/index.md` before changing how it is computed.

## Two invariants that must survive any change

**The design-time answer and the scanner's answer are the same code.** Python
calls the engine through bindings; the interpreter links it. Never add a
second implementation of a model on either side — that is how the two come to
disagree.

**The verdict is an estimate that runs before the scanner's own gate.** It
never replaces the predownload check or the hardware monitor. Do not write
messages or documentation implying it is authoritative.

## The window everything runs on

Checks evaluate one **canonical TR**, not the whole scan, and the window is
built per task:

- gradient-side checks (amplitude, slew, PNS, resonance) use the
  **worst-case envelope** — at each block position, the largest amplitude
  that position takes across instances, sign kept, so a pass of the envelope
  is a pass of every instance;
- RF checks (SAR, coil heating) walk the instances and take the **worst
  repetition**, not the first and not the mean.

Both are evaluated periodically: the history at the start of the window wraps
around from its end, so the boundary between repetitions is handled and the
peak found inside one window is the steady-state peak.

If you change how the window is built, the bounding claim must still hold:
`test_the_worst_case_tr_bounds_every_instance_it_stands_for` evaluates real
instances against the envelope, and an integer `tr=` selector exists so anyone
can re-check it on their own sequence.

## Adding or changing a nerve model

The core computes dG/dt and nothing else; a model is injected as
`required_padding(dt)` and `evaluate(...)`. Keep it that way — nerve models
and their coefficients are the part most likely to be vendor-specific or
revised, and the padding query is the minimum the core needs in order to hand
the model a correctly warmed-up waveform.

A model that publishes a linear kernel gets the assembled fast path (each
distinct shape convolved once, occurrences added as scaled, shifted copies);
one that does not — SAFE is nonlinear — takes the exact route. Both must
agree with the direct convolution.

## Fast paths need a differential twin

Every optimisation here is asserted equal to the calculation it replaces:

| Fast path | Test |
|---|---|
| memoized PNS vs. exact convolution | `run_pns_memo_equivalence`, `tests/ctests/test_safety_grad.c` |
| compiled SAFE vs. upstream PyPulseq | `test_the_c_safe_model_matches_upstreams_python_one` |
| plotted resonance lines vs. the gate's verdict | `test_the_drawn_lines_and_the_predownload_gate_reach_the_same_verdict` |
| the L1 ceiling's probe skip vs. the full walk | `test_the_ceiling_never_swallows_a_violation` |
| the rank (SVD) basis vs. the encoding that needs none | `test_a_multishot_scan_reads_the_same_however_its_arms_are_encoded` |
| `tr=None` vs. upstream, to the bit | `test_pns_over_the_timeline_is_upstreams_answer_exactly` |

Add the equivalent test with any new fast path. A fast answer that disagrees
with the slow one is not an optimisation, it is a different check.

## Mechanical resonance specifics

The criterion is the **equivalent sustained amplitude** at the TR harmonics
inside a guarded band — the amplitude of the pure sinusoid delivering the same
coherent drive, in the vendor's own units. Two guards make it a verdict: a
frequency guard of half the narrowest band's width, and the threshold itself
— a band's own stated amplitude converted from plateau to equivalent sinusoid
(`SA_AEQ_TRAIN_SHAPE`), or `SA_AEQ_POLICY_MT_PER_M` where the band states
literal zero. Changing either changes which sequences are refused, so re-run
the verdicts over every shipped plugin.

No inner periodicity is ever declared: an echo train or a slice loop is just
more materialised events inside the one known period, and the comb emerges
from the coherent sum. Do not add explicit period factors.

## Finish by

```bash
bash scripts/run_tests.sh --only=native     # the C safety suites
bash scripts/run_tests.sh -k safety         # the Python gate tests
bash scripts/format_and_lint.sh
```

and, if the gate's cost moved, regenerating
`docs/explanations/performance/full_benchmark` with
`python docs/_bench/bench_full.py`.
