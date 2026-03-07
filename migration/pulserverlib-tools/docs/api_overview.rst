==========================
pulseqlib C API Overview
==========================

This document summarises every public type, macro, and function
exposed by **pulseqlib**.  All symbols live in
``csrc/pulseqlib_types.h`` (types / macros) and
``csrc/pulseqlib_methods.h`` (functions).

The library is strict **ANSI C89** and compiles cleanly with
``gcc -std=c89 -pedantic -Wall -Wextra -Werror``.

----

Data model
----------

.. code-block:: text

   collection            1 loaded sequence (possibly chained from multiple .seq files)
   └── subsequence[]     1 per .seq file in the chain
       ├── segment[]     contiguous non-interruptible hardware play units
       │   └── block[]   elementary timed events (RF, grad, ADC, …)
       ├── TR structure   periodic repetition pattern (prep → main TRs → cooldown)
       └── freq-mod collection   precomputed gradient frequency-modulation waveforms

A **segment** groups consecutive blocks whose gradient boundary
amplitudes are non-zero; the hardware plays an entire segment without
interruption.  **TRs** are the periodic repetition unit detected
automatically by the library.

----

Opaque handles
--------------

==========================================  ========================================
Type                                        Description
==========================================  ========================================
``pulseqlib_collection``                    Loaded sequence (all subsequences).
``pulseqlib_freq_mod_collection``            Freq-mod waveforms for all subsequences.
==========================================  ========================================

Both are heap-allocated by the library and freed by the caller via
``pulseqlib_collection_free()`` / ``pulseqlib_freq_mod_collection_free()``.

----

Flat info structs (batch getters)
---------------------------------

These replace ~70 individual accessor functions.  Each is filled by a
single ``pulseqlib_get_*`` call and carries an ``_INIT`` macro for
C89-safe zero-initialisation.

====================================  ===  ======================================
Struct                                Flds Populated by
====================================  ===  ======================================
``pulseqlib_collection_info``           5  ``pulseqlib_get_collection_info``
``pulseqlib_subseq_info``              18  ``pulseqlib_get_subseq_info``
``pulseqlib_segment_info``             11  ``pulseqlib_get_segment_info``
``pulseqlib_block_info``               26  ``pulseqlib_get_block_info``
``pulseqlib_adc_def``                   2  ``pulseqlib_get_adc_def``
====================================  ===  ======================================

----

Configuration and input types
-----------------------------

====================================  ======================================
Type                                  Purpose
====================================  ======================================
``pulseqlib_opts``                    Scanner limits and raster times.
``pulseqlib_diagnostic``              Error code + human-readable message.
``pulseqlib_forbidden_band``          Acoustic forbidden-frequency band.
``pulseqlib_pns_params``              PNS model (chronaxie, rheobase, alpha).
``pulseqlib_scan_time_info``          Duration + segment boundary count.
====================================  ======================================

----

Output / result types
---------------------

====================================  ======================================
Type                                  Purpose
====================================  ======================================
``pulseqlib_rf_stats``                Per-RF statistics (flip, BW, SAR…).
``pulseqlib_block_instance``          Resolved amplitudes at cursor position.
``pulseqlib_cursor_info``             Segment/TR boundary flags + scan pos.
``pulseqlib_grad_axis_waveform``      Single-axis waveform + segment labels.
``pulseqlib_tr_gradient_waveforms``   GX/GY/GZ waveforms for one TR.
``pulseqlib_acoustic_spectra``        Spectrograms + peak masks (3 axes).
``pulseqlib_pns_result``              Convolved slew-rate waveforms.
``pulseqlib_label_limit``             Per-label min/max.
``pulseqlib_label_limits``            All 10 Pulseq label limits.
====================================  ======================================

----

Public macros
-------------

**Error codes** (``PULSEQLIB_SUCCESS``, ``PULSEQLIB_ERR_*``)
   ~40 codes grouped by category (generic, parse, TR, segmentation,
   acoustic, PNS, collection, consistency).  Check with
   ``PULSEQLIB_SUCCEEDED(rc)`` / ``PULSEQLIB_FAILED(rc)``.

**Cursor states**
   ``PULSEQLIB_CURSOR_BLOCK`` (0) — block available.
   ``PULSEQLIB_CURSOR_DONE`` (1)  — iteration complete.

**TR region selectors** (for ``pulseqlib_get_freq_mod_count_tr``,
``pulseqlib_get_rf_array``)
   ``PULSEQLIB_TR_REGION_PREP`` (0),
   ``PULSEQLIB_TR_REGION_MAIN`` (1),
   ``PULSEQLIB_TR_REGION_COOLDOWN`` (2).

**Axis indices**
   ``PULSEQLIB_GRAD_AXIS_X`` (0),
   ``PULSEQLIB_GRAD_AXIS_Y`` (1),
   ``PULSEQLIB_GRAD_AXIS_Z`` (2).

**Size limits**
   ``PULSEQLIB_MAX_GRAD_SHOTS`` (16) — max multi-shot waveforms.
   ``PULSEQLIB_DIAG_MSG_LEN`` (256) — diagnostic message buffer.

**Initialiser macros** (every public struct has a ``_INIT`` companion)
   ``PULSEQLIB_OPTS_INIT``, ``PULSEQLIB_DIAGNOSTIC_INIT``,
   ``PULSEQLIB_COLLECTION_INFO_INIT``, etc.

----

Functions by category (51 total)
--------------------------------

Read / load
~~~~~~~~~~~

==============================================  =========================================
Function                                        Notes
==============================================  =========================================
``pulseqlib_read``                              Load from disk (with optional cache/MD5).
``pulseqlib_read_from_buffers``                 Load from in-memory buffers (wrappers).
==============================================  =========================================

Options / diagnostics
~~~~~~~~~~~~~~~~~~~~~

==============================================  =========================================
Function                                        Notes
==============================================  =========================================
``pulseqlib_opts_init``                         Fill ``pulseqlib_opts`` from scanner HW.
``pulseqlib_diagnostic_init``                   Zero a diagnostic struct.
``pulseqlib_get_error_message``                 Human-readable string for error code.
``pulseqlib_get_error_hint``                    Fix-suggestion string for error code.
``pulseqlib_format_error``                      Format code + diagnostic into buffer.
==============================================  =========================================

Checks
~~~~~~

==============================================  =========================================
Function                                        Notes
==============================================  =========================================
``pulseqlib_check_consistency``                 Re-run internal consistency validation.
``pulseqlib_check_safety``                      Gradient limits + acoustic + PNS.
==============================================  =========================================

Scan time
~~~~~~~~~

==============================================  =========================================
Function                                        Notes
==============================================  =========================================
``pulseqlib_peek_scan_time``                    Fast estimate (definitions only).
``pulseqlib_get_scan_time``                     Accurate (from loaded collection).
==============================================  =========================================

Collection lifetime
~~~~~~~~~~~~~~~~~~~

==============================================  =========================================
Function                                        Notes
==============================================  =========================================
``pulseqlib_collection_free``                   Free collection and all owned memory.
==============================================  =========================================

Batch getters (metadata)
~~~~~~~~~~~~~~~~~~~~~~~~

==============================================  =========================================
Function                                        Notes
==============================================  =========================================
``pulseqlib_get_collection_info``               Collection-level summary (5 fields).
``pulseqlib_get_subseq_info``                   Per-subsequence metadata (18 fields).
``pulseqlib_get_segment_info``                  Per-segment metadata (11 fields).
``pulseqlib_get_block_info``                    Per-block metadata (26 fields).
``pulseqlib_get_adc_def``                       Per-ADC definition (2 fields).
==============================================  =========================================

Segment tables
~~~~~~~~~~~~~~

==============================================  =========================================
Function                                        Notes
==============================================  =========================================
``pulseqlib_get_prep_segment_table``            Copy prep segment IDs to buffer.
``pulseqlib_get_main_segment_table``            Copy main segment IDs to buffer.
``pulseqlib_get_cooldown_segment_table``        Copy cooldown segment IDs to buffer.
==============================================  =========================================

RF getters (waveform + statistics)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

==============================================  =========================================
Function                                        Notes
==============================================  =========================================
``pulseqlib_get_rf_stats``                      Per-RF definition statistics.
``pulseqlib_get_tr_rf_ids``                     RF def IDs for each block in one TR.
``pulseqlib_get_rf_array``                      Ordered RF stats for a TR region.
``pulseqlib_get_rf_magnitude``                  Decompressed magnitude (multi-channel).
``pulseqlib_get_rf_phase``                      Decompressed phase (multi-channel).
``pulseqlib_get_rf_time_us``                    RF time-point array.
==============================================  =========================================

Gradient getters (waveform data)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

==============================================  =========================================
Function                                        Notes
==============================================  =========================================
``pulseqlib_get_grad_amplitude``                Decompressed gradient (multi-shot).
``pulseqlib_get_grad_initial_amplitude_hz_per_m`` Initial amplitude of event.
``pulseqlib_get_grad_initial_shot_id``          Initial shot ID.
``pulseqlib_get_grad_time_us``                  Gradient time-point array.
==============================================  =========================================

Label getters
~~~~~~~~~~~~~

==============================================  =========================================
Function                                        Notes
==============================================  =========================================
``pulseqlib_get_label_limits``                  Min/max per label type.
``pulseqlib_get_adc_label``                     Label values for one ADC occurrence.
==============================================  =========================================

Block cursor / iterator
~~~~~~~~~~~~~~~~~~~~~~~

==============================================  =========================================
Function                                        Notes
==============================================  =========================================
``pulseqlib_cursor_next``                       Advance to next block.
``pulseqlib_cursor_reset``                      Rewind to last mark (PMC rescan).
``pulseqlib_cursor_mark``                       Bookmark current position.
``pulseqlib_get_block_instance``                Resolved block at cursor position.
``pulseqlib_cursor_get_info``                   Segment/TR context at cursor position.
==============================================  =========================================

Frequency modulation
~~~~~~~~~~~~~~~~~~~~

====================================================  =============================================
Function                                              Notes
====================================================  =============================================
``pulseqlib_get_freq_mod_count``                      Count RF+ADC events (whole sequence).
``pulseqlib_get_freq_mod_count_tr``                   Count RF+ADC events (specific region).
``pulseqlib_build_freq_mod_collection``               Build collection for all subsequences.
``pulseqlib_update_freq_mod_collection``              Recompute one subsequence (PMC shift).
``pulseqlib_freq_mod_collection_get``                 Look up waveform (subseq idx + scan pos).
``pulseqlib_freq_mod_collection_write_cache``         Persist collection to single binary file.
``pulseqlib_freq_mod_collection_read_cache``          Restore collection from binary file.
``pulseqlib_freq_mod_collection_free``                Free collection.
====================================================  =============================================

TR gradient waveforms (plotting)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

==============================================  =========================================
Function                                        Notes
==============================================  =========================================
``pulseqlib_get_tr_gradient_waveforms``         Per-axis waveforms + segment labels.
``pulseqlib_tr_gradient_waveforms_free``        Free waveform arrays.
==============================================  =========================================

Acoustic spectra (plotting)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

==============================================  =========================================
Function                                        Notes
==============================================  =========================================
``pulseqlib_calc_acoustic_spectra``             Spectrograms + peak masks.
``pulseqlib_acoustic_spectra_free``             Free spectral arrays.
==============================================  =========================================

PNS (plotting)
~~~~~~~~~~~~~~

==============================================  =========================================
Function                                        Notes
==============================================  =========================================
``pulseqlib_calc_pns``                          Convolved slew-rate waveforms.
``pulseqlib_pns_result_free``                   Free PNS result arrays.
==============================================  =========================================

----

Typical usage pattern
---------------------

.. code-block:: c

   pulseqlib_opts       opts = PULSEQLIB_OPTS_INIT;
   pulseqlib_diagnostic diag = PULSEQLIB_DIAGNOSTIC_INIT;
   pulseqlib_collection *coll = NULL;

   /* 1. Load */
   pulseqlib_read(&coll, &diag, "scan.seq", &opts, 1, 1, 0, 1);

   /* 2. Query metadata */
   pulseqlib_collection_info ci = PULSEQLIB_COLLECTION_INFO_INIT;
   pulseqlib_get_collection_info(coll, &ci);

   /* 3. Build freq-mod collection (with caching) */
   pulseqlib_freq_mod_collection *fmc = NULL;
   if (PULSEQLIB_FAILED(pulseqlib_freq_mod_collection_read_cache(
           &fmc, "scan.fmod.bin", coll, shift))) {
       pulseqlib_build_freq_mod_collection(&fmc, coll, shift, fov_rotation);
       pulseqlib_freq_mod_collection_write_cache(fmc, "scan.fmod.bin");
   }

   /* 4. Iterate blocks */
   pulseqlib_cursor_reset(coll);
   while (pulseqlib_cursor_next(coll) == PULSEQLIB_CURSOR_BLOCK) {
       pulseqlib_cursor_info    info = PULSEQLIB_CURSOR_INFO_INIT;
       pulseqlib_block_instance inst = PULSEQLIB_BLOCK_INSTANCE_INIT;
       pulseqlib_cursor_get_info(coll, &info);
       pulseqlib_get_block_instance(coll, &inst);
       /* … program hardware … */
   }

   /* 5. Clean up */
   pulseqlib_freq_mod_collection_free(fmc);
   pulseqlib_collection_free(coll);
