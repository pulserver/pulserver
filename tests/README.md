# The test suite: what is tested where, and why

One entry point runs everything: `bash scripts/run_tests.sh` (pytest drives
the Python tests plus native lanes that run the C suites, every C++ case,
and the Nim bridge). `bash scripts/regenerate_fixtures.sh` rebuilds every
checked-in fixture deterministically.

## Why several lanes exist: parallel implementations, not duplication

The repository deliberately carries more than one implementation of some
machinery, and each lane tests a *different* body of code:

| Code under test | Lane | Files |
|---|---|---|
| `csrc` C89 scanner-side parser + pulseg IR + safety engine | C (minunit) | `ctests/` (13 suites) |
| `cxx/pulseq` design/recon C++ library (pulseqpp) | C++ (GoogleTest) | `cpptests/test_pulseq_*.cpp` |
| `cxx/recon` seqfile reader | C++ | `cpptests/test_sequence_*.cpp` |
| Python surface over pulseqpp (`pulserver.pypulseq`) | Python | `python/test_pypulseq_*.py` |
| fastseq compiled kernels (vs their kept `_py` twins) | Python differential | `python/test_fastseq_*.py` |
| design / sampling / app.sequence / app.recon / recon stacks | Python | remaining `python/` |
| Nim bridge (host binary + bundle) | Nim via pytest | `python/native/test_nimtests.py` |

So "k-space appears three times" is three layers of one implementation:
`test_pulseq_ktraj.cpp` covers the trajectory core against the sequences' own
numbers, `test_pulseq_trajectory.cpp`/`test_pulseq_kspace.cpp` the C++ surface
over it, `test_pypulseq_kspace.py` the Python binding surface.

## Where the references come from

There are **no stored truth files**. Reference values are one of:

1. **The spec-first oracle** (`utils/pulseg_oracle/`): independently computes
   PulSeg content from the parsed file; `python/test_pulseg_vs_oracle.py`
   holds the C interpreter against it *in process* across the whole corpus
   (instance table, label columns, segment hydration, composed TR waveforms).
2. **Physics invariants**, asserted directly: symmetric readouts put the echo
   at N/2 and centre-out ones at sample 0; a refocusing pulse negates k;
   adjoint operators satisfy `<Ax,y> = <x,A†y>`; every phase-encode line of a
   full Cartesian scan is visited exactly once; slice thickness re-measured
   from the RF spectrum lands on the prescription; `f = 1/(2·ESP)` maps echo
   spacing to acoustic frequency.
3. **Differential twins**: compiled kernels against the Python they replace
   (fastseq), the binary reader against the text reader, memoized PNS against
   exact evaluation, write→read round trips.
4. **Protocol-derived numbers** measured from the corpus fixtures themselves
   (e.g. the EPI acoustic line at ~1236 Hz = 1/(2·ESP) of that protocol; the
   calibration lead's ky=0 line at index 7 of its +8Δk→−7Δk sweep). When a
   fixture protocol changes, these numbers change with it — the comments
   state the derivation so they can be recomputed, not guessed.

## Fixtures

* `python/fixtures/` — the zoo corpus, one small protocol per sequence-zoo
  slot plus chains and parser edges (`python/fixture_corpus.py`).
* `utils/expected/` — synthetic C-test specimens (`utils/synthetic_fixtures.py`):
  numbered gradient/RF safety cases (most deliberately invalid, written as
  exact event-table text), TRID specimens, the dedup chain, the blip GRE
  written without deduplication, the navigated GRE, the corrupted-signature
  file, and `binary/` encoding pairs.

`python/test_fixture_corpus.py` asserts the checked-in files are exactly what
today's builders write, so fixture drift is a visible diff, never silent.

## Safety lane (`python/safety/`)

Drives the *compiled* safety engine (the predownload gate) through the
bindings — no Python re-implementation in between. Forbidden-band tests use
a **synthetic** lockout table; no vendor resonance data is quoted anywhere
in this repository. Verdicts are corpus-physics-derived: sustained readout
combs inside a band (bSSFP ~1176 Hz, EPI ~1158 Hz) must flag; broadband
transients and out-of-band combs must pass.

## Echo flags: what the corpus should report, and why

The sequence description marks an ADC as the echo when the window reaches
the centre of the space it sweeps. Corpus verdicts, all physics-derived:
Cartesian/FSE/bSSFP trains 1 (the centre line); multi-echo GRE one per
echo; spiral and stack-of-spirals arms all of them (each arm starts at
k = 0, and a held partition encode is an outer encode that must not veto
the in-plane echo); PROPELLER one per blade (its ky = 0 line); **ZTE every view** -- a
centre-out acquisition opens at |k| = G·t_dead, after the dead time, so it
never samples the centre, but its first sample is the k-space anchor an
FOV shift demodulates against and a simulation dates the echo from, and
that is what the flag quotes. The numerical error floor is capped at error
scale so a real dead-time gap is never normalised away as if it were
integration error; the centre-out rule is explicit instead, and applies
only when the window comes as close to the centre as its TR ever does.

`adc_role` is a within-group statement: one excitation's acquisitions are
ranked against each other (an EPI or multi-echo train gets one
`ECHO_CENTER` and the rest `NON_CENTER`), while a scan that excites once
per readout gives every acquisition `SINGLE` -- ranking separate shots
against each other would demote all but one of them.

## Layout note (recon)

The two MRI physics families answer in different image layouts:
DeepInverse Cartesian physics uses `(batch, h, w)`, while MRI-NUFFT hands
its SENSE adjoint back as `(batch, 1, h, w)`. Both are self-consistent and
adjoint; the Toeplitz and streaming stacks are built on the MRI-NUFFT
convention. Tests that wrap a base physics (SMS) must therefore build
images in that base's own layout, which an adjoint round trip reports.

## Known open items

* The corpus slice-selective pulse measures 4.834 mm against a 5 mm
  prescription (−6 dB width, apodization-dependent); the spectral-thickness
  test tolerates 5%.
