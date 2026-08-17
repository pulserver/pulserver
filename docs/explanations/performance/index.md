# Throughput and footprint

A clinical protocol is not a demonstration. A 3D acquisition is hundreds of
thousands to millions of blocks, and it has to be built while an operator
waits, checked against the hardware, and parsed by a scanner with a few
seconds and a few tens of megabytes to spare. This page is how Pulserver
makes that affordable — and, just as important, the evidence that none of the
speed changes an answer: every fast path is held equal to the plain
calculation it replaces, by tests that run over the whole zoo.

Two sequences carry the page, chosen because each is the informative extreme
of its question. **MPRAGE** carries the throughput story: at 512 × 1024 × 512
it is the largest protocol in the zoo, and its Cartesian and stack-of-spirals
variants between them exercise every case the design path has — a line
readout rescaled per shot, an arm rotated per shot, and arms written out as
distinct waveforms. **EPI** carries the safety story: a blipped echo train is
the case where events genuinely interact — through the nerve's memory and
through coherent spectral summation — so it is where a shortcut would be
easiest to get wrong, and where the equivalence tests bite hardest.

## Design: a compiled core under a Python loop

A design loop runs Python once per repetition, so what it costs is decided by
what each call it makes does — not by the loop. Everything on that path is
compiled:

- **Events are compact compiled objects.** `make_trapezoid` and friends hand
  back an object whose fields read at native speed, and `scale_grad` — the
  call a phase-encode loop makes twice per shot — copies the event and
  rescales it in compiled code, without touching a waveform sample.
- **A block is one call.** `add_block(*events)` registers every event of a
  block in a single compiled call.
- **Finding what repeats is compiled.** Deduplicating the events of a
  million-block scan — the step that turns a scan into a small library of
  shapes plus instance tables — is a rounding pass and a uniqueness pass over
  large numeric arrays, in compiled kernels kept beside plain NumPy
  implementations that the tests hold them equal to.
- **Writing is compiled, and has a binary form.** Serialising the text `.seq`
  is dominated by number formatting; the binary form of the same sequence
  writes and parses an order of magnitude faster, and a reader tells the two
  apart by content, not file name.

Measured on the MPRAGE cases — 512 partitions, 1024 views per inversion
train, 2 099 200 blocks each, single core:

| Case | Design | Rate | Distinct shapes | Peak memory |
|---|---:|---:|---:|---:|
| Cartesian, 512² in-plane | 4.1 s | 1.9 µs/block | 12 | 0.8 GB |
| Stack of spirals, arms rotated | 5.8 s | 2.8 µs/block | 15 | 1.6 GB |
| Stack of spirals, arms written out | 8.7 s | 4.1 µs/block | 5 130 | 1.5 GB |

(An off-isocentre shift, applied to the finished two-million-block scan,
costs a further 1.2–1.8 s in every case.)

A few microseconds per block, holding across two orders of magnitude of
protocol size, is the whole design story: the console asks, the answer comes
back on the download clock, and it does so from an ordinary Python program.
The shapes column is the representation at work — a two-million-block scan is
a dozen distinct waveforms plus instance tables. The rotated-versus-written-out
pair shows why the `ROTATIONS` extension matters at scale: a rotated arm is
one row against a stored shape, where a written-out arm is a new waveform to
deduplicate, write and parse — 5 130 shapes instead of 15, and twice the
design time.

The counters are part of the same story: the design loop knows which line,
partition, slice and echo every acquisition is — they are what it iterates
over — so it writes them into the file as labels while it builds.
{meth}`~pulserver.pypulseq.Sequence.auto_label` can recover them from the
gradients instead, at the cost of a full k-space evaluation of the scan; the
zoo runs it as an independent check that the gradients encode what the loop
believed, which is where a whole-scan cost belongs.

## The canonical TR: one window, built for the task

Every expensive analysis in Pulserver runs over **one window** — the
{doc}`structural TR <../sequence_model/tr_and_segmentation>` — instead of
over the scan. The scan is the window repeated; the window is where the work
is. But "the TR" is not one waveform: it is *built*, differently, for what
each consumer needs, and always so that it bounds the scan.

- **For the gradient-side checks** — amplitude, slew, PNS, mechanical
  resonance — the window is the **worst-case envelope**: at every block
  position, the largest amplitude that position takes across all instances of
  the TR, with its sign kept. No single repetition of the real scan looks
  like this window, deliberately: it is constructed so that a pass of the
  envelope is a pass of every instance.
- **For the RF checks** — SAR, coil heating — the worst instance is not the
  envelope but the worst *repetition*: a train whose flip angle ramps has its
  worst B1rms somewhere in the middle, so the check walks the TR instances
  and takes the worst one, not the first and not the mean.
- **For looking**, `seq.plot(tr=...)` draws the same windows the checks use —
  the envelope, or any actual instance by index.

The bound is not asserted, it is tested:
`test_the_worst_case_tr_bounds_every_instance_it_stands_for` evaluates actual
instances against the envelope, and an integer `tr=` exists precisely so
anyone can re-check the claim on their own sequence:

```python
ok, norm, *_ = seq.calculate_pns(hardware, tr="worst_case")  # the gate's window
ok0, norm0, *_ = seq.calculate_pns(hardware, tr=0)           # one real instance
```

Because a TR plays back to back with copies of itself, both windows are
evaluated *periodically*: the history a nerve model or a spectrum needs at
the start of the window is wrapped around from its end, so the boundary
between repetitions is handled rather than ignored, and the peak found inside
one window is the peak of the steady-state scan.

## Safety: where the speed comes from, shown on EPI

An EPI shot is the hard case for both analyses. Its blip train switches a
readout gradient dozens of times a few hundred microseconds apart, so the
nerve response never returns to baseline between events; and its echo train
is exactly the kind of inner periodicity that turns into a sharp spectral
comb.

![EPI representative TR: the blipped echo train the checks run on](../assets/representative_tr/epi_2d_tr.png)

### PNS: assembled from per-shape responses, not re-convolved

The plain evaluation materialises the slew over the whole canonical TR and
convolves it with the nerve kernel — millions of samples for a
multi-second window. But the window is built from a handful of gradient
shapes repeated at different amplitudes, and for a *linear* nerve model
(Irnich publishes its kernel) convolution distributes over that sum: each
distinct shape is convolved **once**, and every occurrence becomes a scaled,
time-shifted add of that stored response. Cost stops scaling with the TR
length and starts scaling with the number of distinct shapes.

The boundaries are where such a scheme would naïvely go wrong, and where this
one is exact by construction. Each block's slice of the window contributes
its slew *zero-extended on both sides* — opening with the step up from zero
and closing with the step back down — so at the seam between two blocks the
closing step of one and the opening step of the next add to exactly the
difference the directly rendered waveform would have. The blip riding on the
readout ramp, the plateau handed from one block to the next: all of it sums
back, in floating point, to the rendered window's own numbers. Two further
guards keep the gate honest: the templates are slices of *the very waveform
the exact path would convolve* (no second renderer to drift), and every
occurrence is checked to really be a scaled copy of its template before it is
accepted — one failed check and the exact route runs instead. A model that
does not publish a kernel — SAFE is nonlinear — always takes the exact route.

On a Cartesian protocol this took the evaluation from about 2 s to under
100 ms; the two routes are asserted to agree, per sequence family, in the C
test suite (`run_pns_memo_equivalence` in `tests/ctests/test_safety_grad.c`),
to a tolerance of a few floating-point rounding steps — the only difference
between the routes being the order of two multiplications.

![EPI PNS over the worst-case TR: per-blip peaks riding a sustained plateau](../assets/pns_safety/epi_2d_pns.png)

### Mechanical resonance: evaluated at chosen frequencies, never rendered

The {doc}`structural evaluation <../safety/mechanical_resonance>` never
builds a waveform at all: each gradient definition's spectrum is computed
analytically once and reused for every occurrence — a small cache keyed by
definition, hit hundreds of times in an echo train — and the coherent sum is
evaluated **only at the frequencies that matter**: the TR harmonics falling
inside a guarded band for the verdict, plus a dense comb computed by a
chirp-z transform when a plot is asked for. The gate's cost is set by the
band table and the complexity of one window, not by the scan length: a
64-TR scan and an 8192-TR scan of the same sequence cost the same.
On a Cartesian protocol, evaluating the display comb this way took the
analysis from about 5.5 s to a quarter of a second; the gate itself runs in
tens of milliseconds.

![EPI drive spectrum: the echo-train comb against the forbidden bands](../assets/mechanical_resonance/current_epi.png)

### The equivalence tests, by name

Fast paths that "should" agree with the plain ones eventually don't, so the
agreement is not a design intention here — it is pinned by tests that run
over the zoo:

| Claim | Test |
|---|---|
| `tr=None` is upstream PyPulseq *to the bit* — a script that ran against PyPulseq gets PyPulseq's numbers | `test_pns_over_the_timeline_is_upstreams_answer_exactly`, `test_gradient_spectrum_over_the_timeline_is_upstreams_answer_exactly` |
| the worst-case envelope bounds every real repetition | `test_the_worst_case_tr_bounds_every_instance_it_stands_for` |
| the assembled PNS response equals the convolved one | `run_pns_memo_equivalence` (C suite) |
| the compiled SAFE model is upstream's Python one | `test_the_c_safe_model_matches_upstreams_python_one` |
| the plotted resonance lines are the predownload gate's verdict | `test_the_drawn_lines_and_the_predownload_gate_reach_the_same_verdict` |

A fast estimate that disagreed with the slow one would not be an
optimization; it would be a different check.

## Why the scanner side stays small

Three properties, none accidental, all consequences of the
{doc}`representation <../sequence_model/pulseg_representation>`:

- **The scan is stored as references into libraries, not as events.** A block
  is a row of indices plus its per-instance parameters, so a
  hundred-thousand-block scan is tens of megabytes with every waveform stored
  once.
- **The structure is detected once, from identities already computed.**
  Periodicity is tested on the normalized block identities deduplication
  produced anyway, so finding the TR costs a fraction of the parse.
- **The checks walk the instance table.** Amplitude, slew and continuity are
  one pass over rows with the shape library resident; the two expensive
  analyses are expensive per *shape*, not per block — which is the whole
  previous section.

## Reproducing it

```bash
python docs/_bench/bench_mprage.py            # the design-throughput table
python docs/_bench/bench_mprage.py --scale=0.25   # a quarter-size sweep
```

and for the safety paths, the equivalence tests named above run with the
ordinary test suite. The numbers on this page were measured on the tree this
documentation was built from; re-measure rather than quote them when the
question is whether a change made something slower.
