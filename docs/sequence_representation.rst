
Sequence Representation & Caching
=================================

.. contents::
   :local:
   :depth: 1


Seq File Concatenation
----------------------

Multiple ``.seq`` files are concatenated into a **linked-list collection**:

.. code-block:: text

   Collection
   ├── Subsequence 0  (e.g., calibration)
   ├── Subsequence 1  (e.g., main imaging)
   └── Subsequence 2  (e.g., noise scan)

- Each subsequence is a self-contained ``.seq`` file with its own:

  - Block table, gradient table, RF table, rotation matrices
  - TR descriptor (prep / main / cooldown structure)
  - Segment table and segment definitions

- A **block cursor** walks the collection in order, transparently
  crossing subsequence boundaries.


----


Unique Blocks & Event Definitions
-----------------------------------

Each block is a tuple
``(gx_def, gy_def, gz_def, rf_def, adc_def, ext_def, duration)``.
Identical tuples share one ``block_definition``.

.. code-block:: text

   Block table (one entry per played block)
   ┌──────┬──────┬──────┬──────┬──────┐
   │ B0   │ B1   │ B2   │ B0   │ B1   │  ← definition IDs
   └──────┴──────┴──────┴──────┴──────┘
             ↓ deduplication
   Block definitions:  {B0, B1, B2}  —  3 unique out of 5

Each definition ID references the **timing skeleton** (shape) of an
event, separated from the per-instance amplitude/phase that lives
in the table:

.. code-block:: text

   grad_definition (timing skeleton):
     type (trap / arbitrary), delay,
     rise / flat / fall  (trapezoid)  OR  shape_id + num_samples (arb),
     per-shot stats:  max_amplitude, min_amplitude, slew_rate,
                      energy, first_value, last_value

   rf_definition:
     mag_shape_id, phase_shape_id, time_shape_id, delay
     stats:  flip_angle, bandwidth, abswidth, effwidth,
             dtycyc, maxpw, duration, isodelay

   adc_definition:
     num_samples, dwell_time, delay

   block_definition:
     duration_us, gx_id, gy_id, gz_id, rf_id   (→ definition IDs)

- All waveform-derived parameters (``first_value``, ``last_value``,
  ``slew_rate``, ``bandwidth``, ``isodelay``, etc.) are **parsed
  directly from the waveforms** at definition time — no external
  metadata required.

- Gradient stats computed **once per unique gradient definition**,
  not per block instance.


----


Segments
--------

Contiguous groups of blocks forming playable units.
Identified by walking the TR with a state machine:

.. code-block:: text

   Segment boundary criterion:
   ───────────────────────────
   A split candidate is a block boundary where all gradient axes
   have zero amplitude on both sides (|last_value| ≈ 0 AND
   |first_value| ≈ 0, within max_slew × grad_raster tolerance).

   State machine:
   1. SEEKING_FIRST_ADC:  record last zero-gradient candidate
      before each RF.  When the first ADC is found, split at
      that candidate (if any)  →  isolates pre-ADC segment.

   2. SEEKING_BOUNDARY:  after the first ADC, record each new
      zero-gradient candidate.  On the next RF, split at the
      most recent candidate  →  isolates each RF–ADC pair.

   3. OPTIMIZED_MODE:  if no zero-gradient candidate exists
      between an ADC and the next RF, stop splitting
      (remaining blocks form one segment).

   In summary: boundaries are placed at the last zero-gradient
   gap before an RF, after a preceding RF–ADC pair.


----


Unique TR Patterns
-------------------

For acoustic / PNS checks, TRs that differ only in the shot
index of their gradients are grouped.  A **fingerprint matrix**
``[tr_size × 3 axes]`` of shot indices is built, then deduplicated.

.. code-block:: text

   TR 0:  shot (0,0,0)  →  group A
   TR 1:  shot (1,0,0)  →  group B
   TR 2:  shot (0,0,0)  →  group A   (skip — already checked)
   TR 3:  shot (1,0,0)  →  group B   (skip — already checked)

   → Only 2 unique patterns checked, regardless of num_trs.


----


Prep / Cooldown and Degenerate Cases
-------------------------------------

Each subsequence can have optional **prep** and **cooldown** blocks:

.. code-block:: text

   ┌──────────┬──────┬──────┬──────┬──────┬──────────────┐
   │   Prep   │ TR 0 │ TR 1 │  …   │ TR N │   Cooldown   │
   └──────────┴──────┴──────┴──────┴──────┴──────────────┘

- **Degenerate prep/cooldown**: when the prep (or cooldown) blocks
  are structurally identical to a normal TR, they are marked
  ``degenerate = 1``.  No special handling needed — they are just
  additional TR instances.

- **Non-degenerate prep/cooldown**: different structure (e.g., driven-equilibrium or gradients ramp-down).  Checked separately for acoustic / PNS
  as single-occurrence waveforms (``num_trs = 1``).

- **RF periodicity** is verified over the "pure main" TRs —
  those not adjacent to non-degenerate prep/cooldown regions.


----


Binary Caching
--------------

After the first parse, the entire ``sequence_descriptor`` is
serialized to a **versioned binary cache** file:

.. code-block:: text

   Header   [magic + version (v2)]
   ├── Rotation matrices
   ├── Block table + definitions
   ├── Gradient table + definitions
   │     └── includes max_amplitude, min_amplitude, slew_rate per shot
   ├── RF table + definitions + stats
   ├── ADC table
   ├── TR descriptor (prep/main/cooldown sizes, degenerate flags)
   └── Segment table + definitions

- On subsequent loads, binary read replaces the full ``.seq`` parse
  → **near-instant startup**.

- Version field ensures automatic invalidation when the format
  changes (currently v2).

- Segment timing (RF/ADC anchors, k-space zero crossings) is
  recomputed after cache load — lightweight compared to parsing.

