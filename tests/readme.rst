==========================
Test Suite Reference Guide
==========================

This document describes every test case in the pulseqlib / pulserver test
suite, the MATLAB generators that produce the test data, and the expected
results for each test.

.. contents:: Table of Contents
   :local:
   :depth: 2


Directory Layout
================

::

    tests/
    ├── ctests/            C unit tests (minunit framework)
    ├── pytests/           Python unit tests (pytest)
    ├── mtests/            MATLAB unit tests (functiontests)
    ├── generators/        MATLAB .m scripts that produce test data
    ├── data/              Generated .seq / .csv / .txt test data
    └── readme.rst         This file


Generators
==========

All generators are MATLAB ``.m`` scripts in ``tests/generators/``.  They
use the `Pulseq MATLAB toolbox <https://github.com/pulseq/pulseq>`_
(``mr.*``) and write ``.seq`` files (and, for the segmentation generator,
``.csv`` / ``.txt`` ground-truth files) into the working directory.  A CI
workflow (``generate-testdata.yml``) runs all five generators and commits
the output into ``tests/data/``.

generate_grad_continuity_test_cases.m
-------------------------------------

Generates ``.seq`` files covering **gradient continuity** edge cases
(trapezoid ↔ extended-trapezoid transitions, delay blocks, rotation
events).

.. important::

   Before running this script locally, the gradient-continuity check in
   ``mr.Sequence/setBlock.m`` (lines 1032–1096) must be commented out,
   because the script intentionally creates sequences that violate
   continuity.

**Output files:**

========================================= ===========================================
File                                      Description
========================================= ===========================================
``01_ok_trap_extended_trap.seq``           Trap → extended trap, valid continuity
``02_fail_trap_then_startshigh.seq``       Trap followed by high-start extended, fail
``03_fail_startshigh_first.seq``           First block starts high, fail
``04_fail_delay_then_allhigh.seq``         Delay then all-high extended, fail
``05_ok_extended_with_delay.seq``          Extended with delay, valid
``06_fail_delay_then_startshigh.seq``      Delay then high-start extended, fail
``07_fail_nonconnecting.seq``              Non-connecting extended gradients, fail
``08_ok_rot_identity.seq``                 Identity rotation, valid
``09_fail_rot_identity.seq``               Identity rotation with high start, fail
``10_fail_rot_first_block.seq``            Rotation on first block, high start, fail
``11_fail_rot_allhigh.seq``                Rotation all-high, fail
``12_ok_rot_extended_delay.seq``           Rotation with extended delay, valid
``13_fail_rot_delay_then_startshigh.seq``  Rotation: delay then high-start, fail
``14_fail_rot_nonconnecting.seq``          Rotation: non-connecting, fail
``15_ok_rot_same_rotation.seq``            Same rotation across blocks, valid
``16_fail_rot_diff_rotation_1.seq``        Different rotations, fail (case 1)
``17_fail_rot_diff_rotation_2.seq``        Different rotations, fail (case 2)
========================================= ===========================================


generate_grad_limits_test_cases.m
---------------------------------

Generates ``.seq`` files covering **gradient amplitude and slew-rate**
limit violations (single-axis and RSS combined).

**Output files:**

====================================  =============================================
File                                  Description
====================================  =============================================
``01_grad_amplitude_violation.seq``   Single-axis amplitude exceeds limit (20 T/m)
``02_slew_violation.seq``             Single-axis slew rate exceeds limit
``03_grad_rss_violation.seq``         RSS combined gradient amplitude violation
``04_slew_rss_violation.seq``         RSS combined slew-rate violation
====================================  =============================================


generate_once_flag_test_cases.m
-------------------------------

Generates ``.seq`` files to test the **ONCE flag** mechanism (preparation
and cooldown block detection, validity of once-flag patterns).

**Output files:**

===================================================== ==============================================
File                                                  Description
===================================================== ==============================================
``01_single_tr_valid_once.seq``                       Single TR with valid ONCE blocks
``02_dual_tr_valid_once.seq``                         Two TRs with valid ONCE blocks
``03_multi_tr_valid_once.seq``                        Multi-TR with valid ONCE blocks
``04_multi_tr_valid_once_degenerate.seq``             Multi-TR valid ONCE, degenerate case
``05_multi_tr_once_prep_only.seq``                    Multi-TR with ONCE prep blocks only
``06_multi_tr_once_cooldown_only.seq``                Multi-TR with ONCE cooldown blocks only
``07_single_tr_nonvalid_once.seq``                    Single TR with invalid ONCE pattern
``08_prep_too_long.seq``                              Prep region exceeds allowed length
``09_cooldown_too_long.seq``                          Cooldown region exceeds allowed length
``10_multi_tr_nonvalid_once_in_the_middle.seq``       Multi-TR invalid ONCE in middle of sequence
===================================================== ==============================================

.. note::

   ``11_multi_tr_valid_once_in_the_middle.seq`` is present in
   ``tests/data/`` from a prior generator revision and is used by
   ``test_structure_valid_once_in_middle``.  It is not produced by the
   current version of this generator.


generate_rf_test_cases.m
------------------------

Generates ``.seq`` files covering **RF safety** cases: periodic and
non-periodic RF amplitude patterns, and RF-shim periodicity.

**Output files:**

============================================= ==============================================
File                                          Description
============================================= ==============================================
``01_rfamp_ok_mrfingerprinting.seq``          Periodic RF amplitude (MR fingerprinting),
                                              pass consistency
``02_rfamp_fail_vfa.seq``                     Non-periodic RF amplitude (VFA),
                                              fail RF periodicity check
``03_rfshim_ok_pnpmrfingerprinting.seq``      Periodic RF shim pattern (PnP-MRF),
                                              pass consistency
``04_rfshim_fail_gre.seq``                    Non-periodic RF shim pattern (GRE),
                                              fail RF-shim periodicity check
============================================= ==============================================


generate_segmentation_test_sequences.m
--------------------------------------

Generates ``.seq`` files (and per-sequence ground-truth ``.csv`` / ``.txt``
metadata) for **TR detection, segmentation, multi-shot patterns, and
gradient amplitude verification**.

**Output files** (each with associated ``_blocks.csv``, ``_meta.txt``,
and axis-specific waveform ``.csv`` files):

======================================================== ==============================
File pattern                                             Description
======================================================== ==============================
``bssfp_2d_<N>avg.seq``                                  2D bSSFP, N averages
``gre_2d_<S>sl_<N>avg.seq``                              2D GRE, S slices, N averages
``fse_2d_<S>sl_<N>avg.seq``                              2D FSE, S slices, N averages
``epi_2d_<S>sl_<N>avg.seq``                              2D EPI, S slices, N averages
``mprage_3d_<N>avg.seq``                                 3D MPRAGE, N averages
``mprage_noncart_3d_<K>shots[_rotext]_<N>avg.seq``       3D non-Cartesian MPRAGE
======================================================== ==============================

Specific parameter combinations generated:

- bSSFP: 1 avg, 3 avg
- GRE: {1,3} slices × {1,3} avg
- FSE: {1,3} slices × {1,3} avg
- EPI: {1,3} slices × {1,3} avg
- MPRAGE: 1 avg, 3 avg
- MPRAGE non-Cartesian: {1,3} avg × {240,2048} shots × {rotext, no rotext}


C Test Suite (``tests/ctests/``)
================================

Built with cmake.  Uses the
`minunit <https://github.com/siu/minunit>`_ single-header framework.
Run from the build directory with ``./bin/run_tests``.

test_error.c — Error/diagnostic helpers (self-contained)
---------------------------------------------------------

Tests the public API functions that do **not** require a loaded collection.
No external data files needed.

====================================  ==================  ==========================================
Test function                         Data file           Expected result
====================================  ==================  ==========================================
``test_error_message_ok``             *(none)*            ``PULSEQLIB_SUCCESS`` message is non-empty
``test_error_message_known_codes``    *(none)*            All known error codes return a message
``test_error_hint_known_codes``       *(none)*            All known error codes return a hint
``test_error_message_unknown_code``   *(none)*            Unknown code returns a valid string
``test_succeeded_failed_macros``      *(none)*            ``SUCCEEDED/FAILED`` macro correctness
``test_diagnostic_init``              *(none)*            Diagnostic init zeroes message
``test_format_error_basic``           *(none)*            ``format_error`` returns >0 chars
``test_format_error_null_diag``       *(none)*            ``format_error`` works with NULL diag
``test_format_error_tiny_buffer``     *(none)*            Tiny buffer → NUL-terminated, no crash
``test_opts_init_values``             *(none)*            ``opts_init`` matches test constants
``test_block_instance_init``          *(none)*            ``BLOCK_INSTANCE_INIT`` zeroes fields
``test_diagnostic_init_macro``        *(none)*            ``DIAGNOSTIC_INIT`` zeroes message
``test_rf_stats_init``                *(none)*            ``RF_STATS_INIT`` zeroes fields
``test_scan_time_info_init``          *(none)*            ``SCAN_TIME_INFO_INIT`` zeroes fields
``test_read_null_out``                *(none)*            NULL ``out_coll`` → error
``test_collection_free_null``         *(none)*            ``free(NULL)`` → no crash
``test_check_consistency_null``       *(none)*            NULL coll → error
``test_getters_null_coll``            *(none)*            All getters with NULL coll → error
``test_cursor_null_coll``             *(none)*            ``get_block_instance(NULL)`` → error
====================================  ==================  ==========================================


test_load.c — File loading
--------------------------

==========================  =======================================  ======================================
Test function               Data file                                Expected result
==========================  =======================================  ======================================
``test_load_ok``            ``01_ok_trap_extended_trap.seq``          Load succeeds; ≥1 subsequence; duration >0
``test_load_file_not_found`` ``does_not_exist.seq``                   Load fails; coll remains NULL
``test_free_after_load``    ``01_ok_trap_extended_trap.seq``          Load + free without crash
``test_null_pointer``       *(none)*                                 NULL output pointer → error
==========================  =======================================  ======================================

**Generator:** ``generate_grad_continuity_test_cases.m``


test_structure.c — TR detection and ONCE-flag validation
--------------------------------------------------------

======================================  ===============================================  =============================================
Test function                           Data file                                        Expected result
======================================  ===============================================  =============================================
``test_structure_ok_basic``             ``01_ok_trap_extended_trap.seq``                  ≥1 subsequence; ≥1 TR; positive duration
``test_structure_valid_once_in_middle`` ``11_multi_tr_valid_once_in_the_middle.seq``      Load succeeds (not rejected by ONCE check)
``test_structure_invalid_once_in_middle`` ``10_multi_tr_nonvalid_once_in_the_middle.seq`` ``PULSEQLIB_ERR_INVALID_ONCE_FLAGS``
``test_structure_valid_once_boundary``  ``03_multi_tr_valid_once.seq``                    Load succeeds (not rejected by ONCE check)
======================================  ===============================================  =============================================

**Generators:** ``generate_grad_continuity_test_cases.m``,
``generate_once_flag_test_cases.m``


test_cursor.c — Block cursor / iterator
----------------------------------------

==============================  =======================================  =============================================
Test function                   Data file                                Expected result
==============================  =======================================  =============================================
``test_cursor_ok_walk``         ``01_ok_trap_extended_trap.seq``          Walks ≥1 block; each duration >0
``test_cursor_mark_reset``      ``01_ok_trap_extended_trap.seq``          Mark → advance 2 → reset → delivers all blocks
``test_cursor_instance_fields`` ``01_ok_trap_extended_trap.seq``          Rotation matrix ≈ identity; ``rf_shim_id`` = −1
``test_cursor_info``            ``01_ok_trap_extended_trap.seq``          ``subseq_idx`` ≥ 0; ``segment_id`` ≥ 0
==============================  =======================================  =============================================

**Generator:** ``generate_grad_continuity_test_cases.m``


test_consistency.c — Consistency checks
----------------------------------------

==============================  =======================================  =============================================
Test function                   Data file                                Expected result
==============================  =======================================  =============================================
``test_consistency_ok_passes``  ``01_ok_trap_extended_trap.seq``          ``check_consistency`` succeeds
``test_consistency_rfamp_fail`` ``02_rfamp_fail_vfa.seq``                 ``PULSEQLIB_ERR_CONSISTENCY_RF_PERIODIC``
``test_consistency_rfshim_fail``  ``04_rfshim_fail_gre.seq``              ``PULSEQLIB_ERR_CONSISTENCY_RF_SHIM_PERIODIC``
==============================  =======================================  =============================================

**Generators:** ``generate_grad_continuity_test_cases.m``,
``generate_rf_test_cases.m``


test_safety_grad.c — Gradient amplitude / slew-rate limits
----------------------------------------------------------

===================================  =======================================  =============================================
Test function                        Data file                                Expected result
===================================  =======================================  =============================================
``test_grad_ok_trap_exceeds_default`` ``01_ok_trap_extended_trap.seq``         ``PULSEQLIB_ERR_MAX_SLEW_EXCEEDED`` (default limits)
``test_grad_ok_trap_generous_passes`` ``01_ok_trap_extended_trap.seq``         Pass with 100 mT/m, 500 T/m/s limits
``test_grad_amplitude_violation_file`` ``01_grad_amplitude_violation.seq``     ``PULSEQLIB_ERR_MAX_GRAD_EXCEEDED`` (tight limit)
``test_grad_slew_violation_file``    ``02_grad_slew_violation.seq``            Fail with tight slew limit
===================================  =======================================  =============================================

**Generators:** ``generate_grad_continuity_test_cases.m``,
``generate_grad_limits_test_cases.m``


test_safety_acoustic.c — Acoustic spectrum checks (stub)
---------------------------------------------------------

*No test cases yet.*  Acoustic test cases will be written manually.


test_safety_pns.c — Peripheral nerve stimulation checks (stub)
--------------------------------------------------------------

*No test cases yet.*  PNS test cases will be written manually.


test_rf_stats.c — RF statistics
-------------------------------

==============================  =======================================  =============================================
Test function                   Data file                                Expected result
==============================  =======================================  =============================================
``test_rf_stats_no_rf``         ``01_ok_trap_extended_trap.seq``          ``get_rf_stats`` fails (no RF events)
``test_rf_array_no_rf``         ``01_ok_trap_extended_trap.seq``          ``get_rf_array`` returns 0 pulses
``test_rf_stats_periodic_rf``   ``01_rfamp_ok_mrfingerprinting.seq``     flip >0, duration >0, amplitude >0, samples >0
``test_rf_array_periodic_rf``   ``01_rfamp_ok_mrfingerprinting.seq``     ≥1 pulse; first pulse flip >0
==============================  =======================================  =============================================

**Generators:** ``generate_grad_continuity_test_cases.m``,
``generate_rf_test_cases.m``

.. note::

   ``test_rf_stats_periodic_rf`` and ``test_rf_array_periodic_rf`` skip
   gracefully if ``01_rfamp_ok_mrfingerprinting.seq`` fails to load
   (e.g., if the file is stale).  They will fully exercise the RF stats
   API after the generator workflow regenerates the data.


test_waveforms.c — TR gradient waveform extraction
---------------------------------------------------

===========================  =======================================  =============================================
Test function                Data file                                Expected result
===========================  =======================================  =============================================
``test_waveforms_ok_smoke``  ``01_ok_trap_extended_trap.seq``          At least one axis has samples; GX time monotonic
===========================  =======================================  =============================================

**Generator:** ``generate_grad_continuity_test_cases.m``


test_segments.c — Segment-level queries
----------------------------------------

===========================  =======================================  =============================================
Test function                Data file                                Expected result
===========================  =======================================  =============================================
``test_segments_ok_smoke``   ``01_ok_trap_extended_trap.seq``          ≥1 segment; each segment: blocks >0, duration >0,
                                                                      block start times non-decreasing, durations >0
===========================  =======================================  =============================================

**Generator:** ``generate_grad_continuity_test_cases.m``


test_freq_mod.c — Frequency modulation collection
--------------------------------------------------

=====================================  =======================================  =============================================
Test function                          Data file                                Expected result
=====================================  =======================================  =============================================
``test_freq_mod_build_ok``             ``01_ok_trap_extended_trap.seq``          Build succeeds; zero shift → zero waveforms
``test_freq_mod_rotation_all_cases``   *(none — pure math)*                     All four norot×rotation combos match
                                                                                analytical values
``test_freq_mod_build_with_fov_rotation`` ``01_ok_trap_extended_trap.seq``       Build with NULL and identity fov_rotation
                                                                                both succeed
=====================================  =======================================  =============================================

**Generator:** ``generate_grad_continuity_test_cases.m``


test_labels.c — ADC label tables
---------------------------------

===========================  =======================================  =============================================
Test function                Data file                                Expected result
===========================  =======================================  =============================================
``test_labels_disabled``     ``01_ok_trap_extended_trap.seq``          ``parse_labels=0``: getters fail or return empty
``test_labels_enabled_smoke`` ``01_ok_trap_extended_trap.seq``         ``parse_labels=1``: ``num_label_columns`` ≥ 0
===========================  =======================================  =============================================

**Generator:** ``generate_grad_continuity_test_cases.m``


Python Test Suite (``tests/pytests/``)
======================================

Run with ``pytest tests/pytests/``.  All tests use an in-memory
``simple_gre_seq`` fixture (16-PE 2D GRE built with pypulseq), so no
external data files are required.

conftest.py — Fixture
---------------------

- ``simple_gre_seq``:  Builds a 16-PE 2D GRE sequence in memory using
  ``pypulseq``.  Parameters: flip 15°, 3 ms sinc RF, 5 mm slice,
  128-sample ADC, 3.2 ms readout.

test_sequence_collection.py
---------------------------

**TestConstruction**

==============================  ===========  ==========================================
Test function                   Data file    Expected result
==============================  ===========  ==========================================
``test_from_sequence_object``   *(fixture)*  ``num_sequences == 1``
``test_from_sequence_list``     *(fixture)*  ``num_sequences == 1``
``test_from_file``              *(fixture)*  Write to tmp → load → ``num_sequences == 1``
``test_invalid_type_raises``    *(none)*     ``TypeError`` for integer input
==============================  ===========  ==========================================

**TestAccessors**

==============================  ===========  ==========================================
Test function                   Data file    Expected result
==============================  ===========  ==========================================
``test_num_blocks``             *(fixture)*  ``num_blocks(0) > 0``
``test_get_block``              *(fixture)*  ``get_block(0, 0)`` is not None
``test_get_sequence_copy``      *(fixture)*  ``get_sequence(0)`` returns ``pp.Sequence``
``test_get_sequence_out_of_range`` *(fixture)* ``get_sequence(99)`` raises ``IndexError``
``test_num_segments``           *(fixture)*  ``num_segments(0) ≥ 1``
``test_tr_size``                *(fixture)*  ``tr_size(0) ≥ 1``
``test_segment_size``           *(fixture)*  Each segment size ≥ 1
==============================  ===========  ==========================================

**TestReport**

==============================  ===========  ==========================================
Test function                   Data file    Expected result
==============================  ===========  ==========================================
``test_report_returns_list``    *(fixture)*  Returns list with ≥1 element
``test_report_fields``          *(fixture)*  Has ``num_blocks``, ``segments``,
                                             ``tr_size``, ``tr_duration_s``
``test_report_print``           *(fixture)*  ``do_print=True`` returns string with
                                             ``"Subsequence"``
==============================  ===========  ==========================================

**TestCheck**

==============================  ===========  ==========================================
Test function                   Data file    Expected result
==============================  ===========  ==========================================
``test_check_passes``           *(fixture)*  ``check()`` does not raise
``test_check_with_pns``         *(fixture)*  ``check(stim_threshold=1e9, …)`` passes
==============================  ===========  ==========================================


MATLAB Test Suite (``tests/mtests/``)
=====================================

Run with ``runtests('test_sequence_collection')`` in MATLAB.  Uses
the ``functiontests`` framework.  All tests build sequences in-memory
with ``mr.*``; no external data files required.

test_sequence_collection.m
--------------------------

======================================  ===========  ==========================================
Test function                           Data file    Expected result
======================================  ===========  ==========================================
``test_construction_from_sequence``     *(in-memory)* ``NumSequences > 0``
``test_construction_from_cell``         *(in-memory)* ``NumSequences == 1``
``test_construction_from_file``         *(in-memory)* Write → load → ``NumSequences > 0``
``test_num_segments``                   *(in-memory)* ``numel(report.segments) ≥ 1``
``test_tr_size``                        *(in-memory)* ``report.tr_size ≥ 1``
``test_report``                         *(in-memory)* ``report`` returns struct/object
``test_check_passes``                   *(in-memory)* ``check()`` does not throw
======================================  ===========  ==========================================


Data File → Generator Cross-Reference
======================================

Quick lookup: which generator produces which data file.

**From** ``generate_grad_continuity_test_cases.m``:

- ``01_ok_trap_extended_trap.seq`` through ``17_fail_rot_diff_rotation_2.seq``

**From** ``generate_grad_limits_test_cases.m``:

- ``01_grad_amplitude_violation.seq`` through ``04_slew_rss_violation.seq``

**From** ``generate_once_flag_test_cases.m``:

- ``01_single_tr_valid_once.seq`` through ``10_multi_tr_nonvalid_once_in_the_middle.seq``

**From** ``generate_rf_test_cases.m``:

- ``01_rfamp_ok_mrfingerprinting.seq`` through ``04_rfshim_fail_gre.seq``

**From** ``generate_segmentation_test_sequences.m``:

- ``bssfp_2d_*.seq``, ``gre_2d_*.seq``, ``fse_2d_*.seq``,
  ``epi_2d_*.seq``, ``mprage_3d_*.seq``, ``mprage_noncart_3d_*.seq``
- Associated ``*_blocks.csv``, ``*_meta.txt``, ``*_segments.txt``,
  ``*_anchors.txt``, ``*_scan_table.csv``

**Legacy / prior revision:**

- ``11_multi_tr_valid_once_in_the_middle.seq`` (base branch, not from
  current generators)
