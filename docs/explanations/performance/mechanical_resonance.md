# Mechanical resonance

{doc}`The verdict itself <../safety/mechanical_resonance>` asks whether a
sequence drives the magnet's structure inside a band it must not be driven in.
That is a question about a *spectrum*, and a spectrum of a scan is the one
analysis that looks least affordable: minutes of gradient waveform, Fourier
transformed, read against a band table.

It is affordable because the evaluation never renders a waveform and never
computes a whole spectrum. It computes, analytically, the spectrum of each
gradient *definition* — once — and then evaluates the coherent sum **only at
the frequencies that actually matter**.

![EPI drive spectrum: the echo-train comb against the forbidden bands](../assets/mechanical_resonance/current_epi.png)

## Nothing is rendered

Each gradient definition in the canonical TR is a piecewise-linear shape, and
a piecewise-linear shape has a closed-form transform. It is computed once per
definition and cached under it, so an echo train that plays one blip a hundred
times hits the cache ninety-nine times. The occurrences differ by amplitude
and by time offset, which are a scale and a phase on the stored transform —
not a reason to transform anything again.

## Only the frequencies that matter

A TR played back to back with copies of itself puts energy at the harmonics of
its own period, and nowhere else. So the verdict does not need a spectrum: it
needs the harmonics $k/T_\text{TR}$ that fall inside a guarded band, and the
coherent sum evaluated at those. The band table decides how many there are,
and the complexity of one window decides what each costs.

Asking for a *picture* is a different question, and it gets a different
answer: a dense comb across the whole displayed range, computed by a chirp-z
transform.

## What that costs

The same EPI protocol at four scan lengths, with the repetition held fixed:

| Blocks | TRs | Over the timeline | Gate (banded harmonics) | Display comb |
|---:|---:|---:|---:|---:|
| 297 | 3 | 24 ms | 40 ms | 7.6 ms |
| 1 188 | 12 | 80 ms | 36 ms | 8.1 ms |
| 4 752 | 48 | 263 ms | 29 ms | 6.0 ms |
| 14 256 | 144 | 1 139 ms | 36 ms | 8.7 ms |

The two right-hand columns do not move: a 3-TR scan and a 144-TR scan of the
same sequence cost the same, because they *are* the same sequence and the
verdict is a property of the repetition. The left-hand column is upstream
PyPulseq's `calc_gradient_spectrum` over the timeline — which is what `tr=None`
is, to the bit — and it grows with the scan, as a transform of the scan must.

The gate costing more than the picture is not a mistake. The comb is one
chirp-z transform at a fixed resolution; the gate evaluates the coherent sum
at every guarded harmonic, and refines each candidate with sub-points before
it will call a peak a peak. It is the more careful of the two, and it is the
one the scanner runs.

## What is still allowed to scale

The candidate harmonics are $k/T_\text{TR}$ inside a fixed-width band, so
there are exactly as many of them as the TR is long: doubling $T_\text{TR}$
doubles the work. Waveform length, by contrast, barely registers — it enters
only through the size of the transform each definition needs, and that is
computed once per definition rather than once per occurrence.

## The equivalence tests

| Claim | Test |
|---|---|
| `tr=None` is upstream PyPulseq to the bit | `test_gradient_spectrum_over_the_timeline_is_upstreams_answer_exactly` |
| the plotted resonance lines are the predownload gate's verdict | `test_the_drawn_lines_and_the_predownload_gate_reach_the_same_verdict` |
| the worst-case envelope bounds every real repetition | `test_the_worst_case_tr_bounds_every_instance_it_stands_for` |

The second one matters more than it looks: a picture and a verdict computed by
different code would eventually disagree, and the disagreement would surface
as a sequence that plots clean and is refused at the console.
