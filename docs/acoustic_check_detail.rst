
Acoustic Check — Detailed
=========================

.. contents::
   :local:
   :depth: 1


Two-Scale Spectral Analysis
----------------------------

The acoustic check operates at **two complementary time scales**
to capture both transient and steady-state resonance excitation:

.. code-block:: text

   Scale 1 — Sliding Window (short time, within one TR)
   ─────────────────────────────────────────────────────
   Detects transient gradient bursts that excite mechanical
   resonances within a single TR period.

   Scale 2 — Full-TR Harmonic Sampling (long time, across TRs)
   ────────────────────────────────────────────────────────────
   Detects steady-state resonance build-up from the periodic
   repetition of the TR waveform over many TRs.

Both scales run on the **worst-case** (position-max amplitude)
gradient waveform for each unique TR pattern.


----


Scale 1 — Sliding Window Setup
-------------------------------

**FFT configuration** — ``acoustic_support_init``:

.. code-block:: text

   Input:  target_window_size, target_spectral_resolution, max_freq
   Output: nfft (next power of 2), nwin, nfreq, hop_size, cos_window

   nfft = next_pow2(ceil(1 / (resolution × dt)))
   nwin = min(nfft, target_window_size / dt)
   nfreq = nfft / 2 + 1  (capped at max_freq / df)
   hop_size = nwin / 2   (50% overlap)

**Per-window processing** — ``compute_window_spectrum``:

.. code-block:: text

   1. Extract nwin samples from waveform
   2. Apply cosine window (Hann-like):  w[i] = 1 − cos(2π·i / nwin)
   3. Zero-pad to nfft
   4. Real FFT → complex spectrum
   5. Magnitude:  |X[k]| = √(re² + im²)


----


Scale 1 — Sliding Window Loop
------------------------------

``compute_sliding_window_spectra``:

.. code-block:: text

   ┌───────────────────────────────────────────────────┐
   │  Waveform: ════════════════════════════════════   │
   │                                                   │
   │  Window 0:  [████████]                            │
   │  Window 1:      [████████]                        │
   │  Window 2:          [████████]                    │
   │      …                    …                       │
   │  Window N:                        [████████]      │
   │                                                   │
   │  Per window:                                      │
   │    • Track max |waveform| = envelope amplitude    │
   │    • Compute FFT spectrum                         │
   │    • Run candidate peak detection                 │
   │    • Check peaks against forbidden bands          │
   └───────────────────────────────────────────────────┘

   Axes Gx, Gy, Gz processed independently.


----


Scale 2 — Full-TR Harmonic Sampling
-------------------------------------

**Rationale**: when a TR repeats ``num_trs`` times, only discrete
harmonics of the fundamental frequency ``f₀ = 1/TR`` survive.
Energy at other frequencies cancels over many repetitions.

**Algorithm** — ``compute_sequence_spectrum``:

.. code-block:: text

   1. FFT of the full one-TR waveform (all samples, zero-padded)
      → raw_spectrum[k]  at frequencies  k × df

   2. Compute fundamental:  f₀ = 1e6 / TR_duration_us

   3. Sample spectrum at harmonic frequencies:
      h = 1, 2, 3, …, num_harmonics
      f_h = h × f₀

   4. For each harmonic f_h:
      • Find nearest FFT bin:  bin = f_h / df
      • Interpolate between floor(bin) and ceil(bin):
        S(f_h) = S[floor] + frac × (S[ceil] − S[floor])

   5. Scale by  1 / num_trs  (amplitude per repetition cycle)

   → Output: harmonic spectrum at multiples of f₀ only.

.. code-block:: text

   Frequency axis:
   ──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──→
     f₀ 2f₀ 3f₀ 4f₀  …
      ↑   ↑   ↑   ↑
      Sampled harmonics — only these matter for steady-state.


----


Candidate Peak Detection
--------------------------

The same algorithm is used for both scales.

**Algorithm** — ``detect_resonances``:

.. code-block:: text

   Input:  spectrum[0..N-1]

   1. Normalize:  norm[k] = spectrum[k] / max(spectrum)

   2. Log-scale:  log_val[k] = log₁₀(SCALE × norm[k] + 1)
                  where SCALE = 10.0

   3. Compute mean of log values:  mean_log

   4. Threshold:  T = mean_log + 2.25  (PEAK_LOG10_THRESHOLD)

   5. Find local maxima:
      • log_val[k] > T
      • log_val[k] ≥ log_val[k-1]
      • log_val[k] ≥ log_val[k+1]

   → Output: list of (frequency_index, amplitude) pairs

.. code-block:: text

   log₁₀ spectrum
   │          ╱╲
   │         ╱  ╲         ╱╲
   │ ·······╱····╲·······╱··╲·····  ← threshold (mean + 2.25)
   │       ╱      ╲     ╱    ╲
   │ ─────╱────────╲───╱──────╲───
   │          ↑               ↑
   │      candidate       candidate
   └──────────────────────────────→ freq

**Design rationale**: the log₁₀ transform with additive threshold
naturally adapts to the overall spectral level — "peaks" are
defined relative to the spectrum's own average, not an absolute
level.  The scale factor (10×) spreads the log range for robust
thresholding.


----


Violation Check (Double Check)
-------------------------------

After identifying candidate peaks, a **two-stage** verification
determines if any peak actually violates safety limits:

**Algorithm** — ``check_acoustic_violations``:

.. code-block:: text

   Stage 1 — Peak Detection
   ─────────────────────────
   Run detect_resonances(spectrum) → candidate peak list

   Stage 2 — Forbidden Band × Envelope Check
   ───────────────────────────────────────────
   For each candidate peak:
     For each forbidden band [freq_min, freq_max]:
       If peak frequency ∈ [freq_min, freq_max]:
         If max_envelope > band.max_amplitude:
           → VIOLATION

.. code-block:: text

   ┌─────────────────────────────────────────────┐
   │                                             │
   │   Forbidden band:  [500 Hz ─── 700 Hz]      │
   │   Band limit:       0.025 T/m               │
   │                                             │
   │   Candidate peak at 620 Hz  ← in band!     │
   │   Window envelope = 0.031 T/m               │
   │                     ↑                       │
   │               0.031 > 0.025  → VIOLATION    │
   │                                             │
   └─────────────────────────────────────────────┘

The **envelope** (max absolute gradient amplitude within the
current analysis window) is the key metric — not the spectral
amplitude of the peak itself.  This ensures that even a modest
spectral peak is flagged if the gradient is strong enough to
excite a mechanical resonance.

**Worst peak tracking**: across all candidate peaks within all
forbidden bands, the one with the highest
``max_envelope / band.max_amplitude`` ratio is reported as the
worst offender in the diagnostic output.


----


Per-Axis Processing & Output
------------------------------

.. code-block:: text

   pulseqlib_get_tr_acoustic_spectra(waveforms, ...)
   │
   ├── Sliding window (Scale 1):
   │   ├── Gx: compute_sliding_window_spectra → spectra + peaks
   │   ├── Gy: compute_sliding_window_spectra → spectra + peaks
   │   └── Gz: compute_sliding_window_spectra → spectra + peaks
   │
   └── Sequence spectrum (Scale 2):
       ├── Gx: compute_sequence_spectrum → harmonics + peaks
       ├── Gy: compute_sequence_spectrum → harmonics + peaks
       └── Gz: compute_sequence_spectrum → harmonics + peaks

   Output per TR pattern:
   ├── frequency_bins[nfreq]           — shared frequency axis
   ├── spectra[3][num_windows][nfreq]  — per-axis sliding-window
   ├── envelope[3][num_windows]        — per-axis max |gradient|
   ├── num_peaks, peak_freqs, peak_amps — detected candidates
   └── sequence_harmonics[num_harmonics] — full-TR harmonic spectrum

   Axes are independent — a violation on any single axis
   is sufficient to flag the TR pattern.

