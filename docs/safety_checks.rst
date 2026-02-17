
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


Gradient Safety  *(per unique block/definition — fast)*
-------------------------------------------------------

All three gradient checks exploit the unique block/definition
representation to avoid redundant per-TR computation:

**1. Max amplitude** — Iterate block table entries.  For each block,
look up ``gx/gy/gz`` amplitudes → compute GSOS.  Compare against
``max_grad`` limit.

.. code-block:: text

   Cost:  O(num_blocks)  — scalar amplitude from table, no waveform.

**2. Gradient continuity** — Dry-run the block cursor through the
full collection (all reps, all subsequences).  At each step, use
``first_value`` / ``last_value`` from the gradient definition ×
amplitude × rotation matrix → physical-axis boundary values.
Check jump against ``max_slew × grad_raster_time``.

.. code-block:: text

   Cost:  O(total_played_blocks)  — uses precomputed first/last values,
          no waveform reconstruction.

**3. Max slew rate** — Iterate **unique gradient definitions** only.
``slew_rate[shot] × max_amplitude[shot]`` compared against
``max_slew / √3``.

.. code-block:: text

   Cost:  O(num_unique_grads × num_shots)  — typically a handful.
          Independent of sequence length.


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
   ├── 1. check_max_grad          per block table entry     O(num_blocks)
   ├── 2. check_grad_continuity   cursor walk               O(total blocks)
   ├── 3. check_max_slew          per unique grad def       O(few)
   │
   └── 4. Per subsequence:
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

