
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

- Each subsequence has its own ``grad_raster_time`` — no global
  constraint.


----


Unique Block / Segment / TR Representation
------------------------------------------

Three levels of deduplication compress the representation:

**1. Unique Blocks** — Each block is a tuple
``(gx_def, gy_def, gz_def, rf_def, adc_def, ext_def, duration)``.
Identical tuples share one ``block_definition``.

.. code-block:: text

   Block table (one entry per played block)
   ┌──────┬──────┬──────┬──────┬──────┐
   │ B0   │ B1   │ B2   │ B0   │ B1   │  ← definition IDs
   └──────┴──────┴──────┴──────┴──────┘
             ↓ deduplication
   Block definitions:  {B0, B1, B2}  —  3 unique out of 5

- Gradient stats (slew rate, max/min amplitude) computed **once
  per unique gradient definition**, not per block instance.

**2. Segments** — Contiguous groups of blocks that share the same
timing skeleton.  Identified by walking the TR and splitting at
RF / ADC boundaries.

**3. Unique TR Patterns** — For acoustic / PNS checks, TRs that
differ only in the shot index of their gradients are grouped.
A **fingerprint matrix** ``[tr_size × 3 axes]`` of shot indices
is built, then deduplicated.

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

- **Non-degenerate prep/cooldown**: different structure (e.g., fat-sat
  or driven-equilibrium).  Checked separately for acoustic / PNS
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

