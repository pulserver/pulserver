
Safety Checks Overview
======================

.. contents::
   :local:
   :depth: 1


Consistency Checks  *(lightweight — per collection)*
-----------------------------------------------------

Run once at load time.  Two checks validate the internal structure:

**Segment walk** — For each region (prep, cooldown, 2nd main TR),
walk blocks and verify that ``block_table[pos].id`` matches the
segment definition's expected sequence of unique block indices.

**RF amplitude periodicity** — Verify that the RF amplitude pattern
within one TR is identical across all "pure main" TR instances
(excluding TRs adjacent to non-degenerate prep/cooldown).

.. code-block:: text

   Cost:  O(num_blocks)  — single linear pass, no waveform rendering.


----


Gradient Safety  *(per subsequence, per unique block/definition)*
-----------------------------------------------------------------

All three gradient checks run **per subsequence** and exploit the
unique block/definition representation.  All parameters used
(``first_value``, ``last_value``, ``slew_rate``, ``max_amplitude``)
are **parsed directly from the gradient waveforms** at definition
time — no external metadata.

**1. Max amplitude** — Per subsequence, iterate block table entries.
Look up ``gx/gy/gz`` amplitudes → compute GSOS.  Compare against
``max_grad`` limit.

.. code-block:: text

   Cost:  O(num_blocks) per subsequence — scalar from table, no waveform.

**2. Gradient continuity** — Dry-run the block cursor through each
subsequence (all reps).  At each step, use  ``first_value`` /
``last_value`` from the gradient definition × amplitude × rotation
matrix → physical-axis boundary values.  Check jump against
``max_slew × grad_raster_time``.  Gradient must return to zero at
subsequence boundaries.

.. code-block:: text

   Cost:  O(total_played_blocks) per subsequence — precomputed
          first/last, no waveform reconstruction.

**3. Max slew rate** — Per subsequence, iterate **unique gradient
definitions** only.  ``slew_rate[shot] × max_amplitude[shot]``
compared against ``max_slew / √3``.

.. code-block:: text

   Cost:  O(num_unique_grads × num_shots) per subsequence — typically
          a handful.  Independent of sequence length.


----


Acoustic & PNS  *(per unique TR pattern)*
------------------------------------------

Waveform-level checks run on reconstructed TR gradient waveforms.
The key efficiency insight:

.. code-block:: text

   Total cost = (1 prep + K unique TR patterns + 1 cooldown)
                × (acoustic + PNS)

   where K = # unique shot-index patterns (typically 1–4).

**Prep / Cooldown** — If non-degenerate: render one waveform (prep +
first TR, or last TR + cooldown), run acoustic + PNS with
``num_trs = 1``.

**Main TRs** — ``find_unique_shot_trs`` builds a fingerprint of
shot indices per TR, deduplicates.  For each unique group:

- Render **worst-case** (position-max amplitude) waveform for one TR
- Run acoustic check (sliding window + full-TR harmonic analysis)
- Run PNS check (convolution with exponential kernel)

.. code-block:: text

   Result:  cost nearly independent of num_trs.
   A 1000-TR sequence with 2 unique patterns costs the same
   as a 10-TR sequence with 2 unique patterns.


----


Worst-Case Waveform Construction
---------------------------------

For each block position and shot index, the maximum
``|amplitude|`` across all TRs in the group is found.  The
**sign** of each block is preserved from the first TR instance —
this ensures the waveform has a physically consistent polarity
while using worst-case magnitude.

.. code-block:: text

   Per block position:
     max_amp = max(|amplitude|)  across all TRs in group
     sign    = sign(amplitude)   from the first TR
     waveform = sign × max_amp × normalized_shape

This construction is conservative: any real TR instance will have
amplitudes ≤ the worst-case waveform at every block position.


----


RF Stats  *(per unique RF definition — fast)*
----------------------------------------------

Computed once per unique RF definition at load time:

.. code-block:: text

   Per unique RF pulse:
   ├── flip angle (integral of magnitude)
   ├── max amplitude (from RF table scan)
   ├── abswidth, effwidth, dtycyc, maxpw
   ├── bandwidth (FFT-based)
   ├── isodelay
   └── duration

All RF parameters are **parsed directly from the waveforms**:
flip angle from magnitude integral, bandwidth from FFT of the
RF pulse, isodelay from waveform center-of-mass, abswidth /
effwidth / dtycyc / maxpw from envelope statistics.

RF stats are bounded to the **generalized TR pattern**: the set of
unique RF definitions is determined by the (typically few) distinct
RF events in the sequence, not by ``num_trs``.

.. code-block:: text

   Cost:  O(num_unique_rf)  — each RF definition processed once.
          One FFT per unique RF pulse for bandwidth estimation.


----


Safety Pipeline Summary
-----------------------

.. code-block:: text

   pulseqlib_check_safety(collection)
   │
   └── Per subsequence:
       │
       ├── 1. check_max_grad        block table scan         O(num_blocks)
       ├── 2. check_grad_continuity  cursor walk, all reps   O(played blocks)
       ├── 3. check_max_slew         unique grad defs only   O(few)
       │
       ├── 4a. Prep (if non-degenerate)
       │   └── acoustic + PNS on prep+TR waveform
       │
       ├── 4b. Main TRs
       │   ├── find_unique_shot_trs → K groups
       │   └── For each of K groups:
       │       └── acoustic + PNS on worst-case waveform
       │
       └── 4c. Cooldown (if non-degenerate)
           └── acoustic + PNS on TR+cooldown waveform

   Heavy computation (acoustic FFTs, PNS convolutions) runs
   at most  (2 + K) × num_subsequences  times.

   All waveform-derived parameters used throughout the pipeline
   (first/last values, slew rates, isodelay, bandwidth, k-space
   crossings, etc.) are parsed from the waveforms themselves.

