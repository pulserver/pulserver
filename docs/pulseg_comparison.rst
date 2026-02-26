==================================
pulseqlib ↔ pulseg comparison
==================================

This note maps the **pulseg** intermediate representation (spec v1.0)
onto the **pulseqlib** C-library data model and assesses whether a
light rename or restructure could make them consistent.

----

pulseg data model (from ``docs/spec.md`` + ``fromSeq.m``)
----------------------------------------------------------

.. code-block:: text

   BaseBlock               Pulseq block with normalised waveform amplitudes.
                            Two blocks are "the same" if their gradient shape
                            IDs, RF shape IDs, ADC params, and timing match.

   VirtualSegment          Ordered list of BaseBlock IDs, identified by a
                            user-provided TRID label in the .seq file.
                            All instances share the same block structure.

   SegmentInstance          One runtime occurrence of a VirtualSegment.
                            Carries concrete amplitudes, phases, frequency
                            offsets, gradient energy, rotation, trigger flag.

   Loop table (psq.loop)   N×23 matrix: one row per block in execution order.
                            Columns: segmentID, parentBlockID, rfamp, rfphs,
                            rffreq, gx_amp, gx_energy, gy_amp, gy_energy,
                            gz_amp, gz_energy, recphs, duration, trigger, R(9).

Detection is **label-driven**: the TRID label written by the sequence
author defines segment boundaries.  No automatic TR detection.

----

pulseqlib data model
--------------------

.. code-block:: text

   collection              All loaded .seq files (chain).
   └── subsequence[]       One per .seq file in the chain.
       ├── segment[]       Contiguous blocks where gradient boundary
       │   └── block[]     amplitudes prevent interruption.
       ├── TR structure    prep → main TRs → cooldown (auto-detected).
       └── freq-mod lib    Precomputed freq-mod waveforms + cache.

Detection is **automatic**: segments are found from gradient boundary
conditions; TRs are found by detecting the periodic block pattern.

At runtime the **cursor** walks blocks in execution order:

  ``cursor_next`` → ``cursor_get_info``  (segment/TR context)
                   → ``get_block_instance``  (resolved amplitudes)

----

Concept mapping
---------------

=========================================  =========================================
pulseg                                     pulseqlib
=========================================  =========================================
BaseBlock                                  Internal deduped block shape (factored
                                           amplitude, same shape-equality criterion).
                                           Exposed read-only via ``block_info``.

VirtualSegment                             **segment** — ordered block list, unique
                                           ID, exposed via ``segment_info``.

SegmentInstance                            Cursor iteration: ``block_instance`` gives
                                           resolved amplitudes/phases/rotation;
                                           ``cursor_info`` gives segment context.

Loop table (psq.loop)                      Cursor walk (``cursor_next`` loop).  Each
                                           step yields the equivalent of one row.

TRID label (user-provided)                 ``segment_id`` (auto-detected).  User
                                           labels are NOT required.

--                                         **subsequence** — no pulseg equivalent
                                           (pulseg handles only single .seq files).

--                                         **TR structure** — no pulseg equivalent.
                                           (pulseg does not detect periodicity.)

--                                         **freq-mod library** — no pulseg equiv.
=========================================  =========================================

----

Structural comparison
---------------------

+---------------------------+-------------------+------------------------------+
| Aspect                    | pulseg            | pulseqlib                    |
+===========================+===================+==============================+
| Segment detection         | TRID labels       | Gradient boundary analysis   |
|                           | (explicit)        | (automatic)                  |
+---------------------------+-------------------+------------------------------+
| Block deduplication       | Shape comparison   | Shape + hash (at parse time)|
+---------------------------+-------------------+------------------------------+
| Dynamics storage          | Flat N×23 table    | Cursor iterator + structs   |
+---------------------------+-------------------+------------------------------+
| Multi-sequence support    | Single .seq        | Chained collection           |
+---------------------------+-------------------+------------------------------+
| TR detection              | None               | Automatic (prep/main/cool)   |
+---------------------------+-------------------+------------------------------+
| Rotation model            | Per-segment        | Per-block (rotation event)   |
|                           | (last block wins)  |                              |
+---------------------------+-------------------+------------------------------+
| Variable delay            | Tracked separately | Block duration is per-       |
|                           | (isVariableDelay)  | instance (resolved by cursor)|
+---------------------------+-------------------+------------------------------+
| Gradient energy (heating) | Computed per-block | Separate waveform extraction |
|                           | in loop table      | (``get_tr_gradient_waveforms``)|
+---------------------------+-------------------+------------------------------+
| Frequency modulation      | Not handled        | Full library + binary cache  |
+---------------------------+-------------------+------------------------------+
| Acoustic / PNS            | Not handled        | Built-in checks + plotting   |
+---------------------------+-------------------+------------------------------+

----

Assessment
----------

**pulseqlib is a strict superset of pulseg.**

The three pulseg concepts (BaseBlock, VirtualSegment, SegmentInstance)
map one-to-one onto pulseqlib concepts that already exist:

- BaseBlock = deduped block (normalised shapes, amplitude factored out).
- VirtualSegment = segment (same structure: ordered list of block IDs).
- SegmentInstance = one cursor step (same data: amplitudes + context).

pulseqlib adds three layers of hierarchy that pulseg does not have:

1. **Subsequence** (multi-.seq chaining).
2. **TR structure** (automatic prep/main/cooldown detection).
3. **Freq-mod library** (gradient frequency-modulation precomputation).

These are purely additive — they sit above the segment level and do not
contradict pulseg's model.

----

Rename / refactor needed?
--------------------------

**No.**

The existing pulseqlib names are already a natural match:

====================  ================  ==========================================
pulseg name           pulseqlib name    Comment
====================  ================  ==========================================
BaseBlock             block / blk       Same dedup criterion, same normalisation.
VirtualSegment        segment / seg     Same concept: ordered block list.
SegmentInstance       cursor position   Resolved at iteration time, not stored.
====================  ================  ==========================================

Possible cosmetic renames (NOT recommended):

- ``segment`` → ``virtual_segment``: adds verbosity; "segment" already
  implies a template that is instantiated at runtime.
- ``block_info`` → ``base_block_info``: similarly verbose; the "info"
  suffix already signals it describes the template, not an instance.

The only genuine divergence is that pulseg requires explicit
**TRID labels** from the sequence author while pulseqlib auto-detects
segments.  This is a design choice, not a naming issue: pulseqlib
chose automatic detection to avoid depending on author-placed labels,
which may be absent or inconsistent.  If label-driven segmentation is
ever desired, it could be offered as an alternative path without
changing the public API surface.
