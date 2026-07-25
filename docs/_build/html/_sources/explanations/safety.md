# Safety

Safety is evaluated from the same PulSeg definitions and instances that will
be executed. The host rejects an invalid prescription before it is cached; the
scanner remains the authority for vendor hardware limits.

## Gradient amplitude and slew

Each base-gradient definition memoises its normalised peak, maximum internal
slew, first sample and last sample (per multishot variant). Per-instance
amplitude then scales those cached quantities. This makes the amplitude and
internal-slew checks a walk over compact instance tables, rather than repeated
waveform expansion.

The boundary check uses the cached last sample of one definition and first
sample of the next after applying their instance amplitudes and rotations. It
also checks the leading and trailing transition to zero. Thus a waveform can
be individually legal yet still be rejected if joining adjacent blocks exceeds
the raster-step slew limit. Limits are evaluated as a gradient-vector norm,
not as three unrelated channels.

## Mechanical resonance

Mechanical resonance is a coherent-frequency constraint, not an acoustic
energy estimate. For each protected band, Pulserver evaluates the complex
Fourier response of every unique gradient shape and sums the scaled,
time-shifted occurrences over the canonical safety window. Repeated outer TRs
are folded analytically. The resulting equivalent sinusoidal amplitude is
compared with the band limit.

No dense gradient waveform is required for this gate. Transform results are
memoised by `(gradient definition, frequency)`, so cost follows the number of
definitions and occurrences in one canonical window, not the duration of a
scan. The public plotting API may also generate a dense display spectrum; that
display work is not part of the gate.

![Representative EPI segment/TR](assets/representative_tr/epi_2d_tr.png)

![Mechanical-resonance verdict for EPI](assets/mechanical_resonance/current_epi.png)

## Peripheral nerve stimulation

PNS depends on the pointwise maximum of a time-domain convolution and cannot
be reduced to independent per-definition peaks. Pulserver materialises the
gradient slew for one canonical TR, circularly pads it with the model's
required history, and evaluates the steady-state peak. The core supplies the
vendor-neutral slew waveform; the PNS model supplies the padding and threshold
calculation. The bundled chronaxie/rheobase path matches the predownload
implementation. `Sequence.pns(model="safe")` delegates to upstream PyPulseq
when SAFE hardware parameters are supplied.

The SAFE model is intentionally not plotted here: only the configured SAFE
hardware model can produce a meaningful SAFE curve. The PNS figures below use
the explicitly stated representative chronaxie/rheobase parameters and are an
authoring inspection, not a scanner verdict.

![EPI PNS inspection](assets/pns_safety/epi_2d_pns.png)

Measurement of these paths belongs in {doc}`benchmarks`, not in this page.
