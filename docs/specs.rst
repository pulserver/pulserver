======================================================
Pulseq Processing and Execution System: Design Specification
======================================================

**Document Version:** 5.1 (Definitive with Appendix)
**Date:** 2025-11-18 13:57:56 UTC
**Author:** GitHub Copilot (based on discussion with @mcencini)

**Change Summary (v5.1):** Added Appendix with detailed descriptions of the computationally efficient streaming algorithms (Streamed Convolution for PNS, Streamed Goertzel for Acoustics) and the rationale for using a multi-TR analysis window. No other sections were modified.

1. System Architecture Overview
-------------------------------

This document specifies a complete system for the processing and execution of Pulseq MRI sequences. It is designed for a resource-constrained ANSI C environment and operates in two primary modes:

*   **Offline Mode:** The system processes a pre-existing ``.seq`` file from local storage.
*   **Online Server Mode:** The system acts as a client to a remote Pulseq generation server (e.g., a Python-based ISMRMRD Server). It sends sequence parameters (e.g., in JSON/TOML) and receives a stream of sequence text, which it writes to a local file before processing.

Regardless of the mode, the file is processed through the following stages: Data Ingestion, Safety Validation, Instruction Building, and Real-Time Execution.

2. Stage 0: Data Ingestion and Caching Strategy
------------------------------------------------

This stage ensures a consistent, efficient data source.

2.1. Universal Binary Conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The system **always** converts the input ``.seq`` file to a structured binary cache. This is a one-time cost per file modification.

*   **Process:** The converter reads the ``.seq`` file, decompresses any native Pulseq derivative-compressed shapes, and writes all data (definitions, shapes, event loop) into a single binary cache file.
*   **Structure:** The cache contains a **Master Index** at the header, which holds the byte offset, record count, and record size for each major data section. The rest of the file consists of the raw binary **Data Blobs**.

2.2. Global Data Loading Mode Switch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A global switch (``g_lazy_load_enabled``) controls how the binary cache is used:

*   **Lazy-Loading Mode (``true``):** Default mode. Only the Master Index is loaded into RAM. When data is needed (e.g., a shape or an event loop row), its location is calculated from the Master Index (``offset = section_start_offset + (record_index * record_size)``), and the data is read from disk via ``fseek()`` and ``fread()``.
*   **Full In-Memory Caching Mode (``false``):** The entire binary cache is loaded into RAM for debugging/benchmarking.

----

3. Stage 1: Safety Validation
----------------------------

This stage is the most critical and is performed in a precise order to ensure all dependencies are met.

3.1. RF Safety Analysis (Based on First TR of Each Sequence Object)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This is performed first for each ``.seq`` object.

1.  **Base RF Analysis:** For each unique RF pulse definition, compute flip angle, bandwidth, and isodelay.
2.  **Vendor Struct Population:** Populate a vendor-specific ``RF_PULSE`` struct and run the vendor's ``rfstat()`` algorithm on it.
3.  **First TR Instantiation:** Create an array of ``RF_PULSE`` structs representing every RF event in the **first TR**.
4.  **Instance Scaling:** Scale the ``act_fa`` (actual flip angle) field of each struct in the array based on the ``amplitude`` of its corresponding instance in the first TR.
5.  **Static Storage of Amplitudes:** The RF amplitudes from this first TR are **stored statically**. **Crucially, these stored values will be used at runtime, overriding any other RF amplitudes from the main event loop to provide a hard safety lock against accidental violations.**
6.  **Vendor API Calls:** Pass the fully scaled TR array to vendor routines:
    *   ``findB1max()``: Determines the peak B1 amplitude required for scaling.
    *   ``minseqrfamp()``: Calculates RF hardware heating to determine ``min_TR_hw``.
    *   ``maxsar()``: Calculates patient SAR to determine ``min_TR_sar``.

3.2. Full Pass Gradient Analysis and TR Duration Checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A single full pass is now performed over the entire sequence data to gather all gradient-related metrics.

1.  **Global `Gmax` Check:** Find the maximum ``amplitude`` across all gradient events in the entire sequence and check against the absolute hardware limit.
2.  **Max SOS Energy Instance:** For each block position in the TR structure, find the instance with the maximum Sum-of-Squares (SOS) energy. This is done by pre-calculating the energy of the *normalized shape* and then scaling it by ``amplitude^2`` for each instance. The amplitudes of these max-energy instances are stored for later use in `pulsegen`.
3.  **Max Slew Rate Instance:** For each block position, find the instance with the maximum effective slew rate (``amplitude * base_peak_slew``, where ``base_peak_slew`` is pre-calculated per ``BlockGroup`` type). These instances are used to define the synthetic "Slew-Rate Worst-Case TR".
4.  **Total ADC Count:** During this same pass, count the total number of ADC events to allocate memory for raw data acquisition.
5.  **TR Duration Calculation and Final Check:**
    *   Use a ``minseq(segment_idx)`` function to find the duration of each ``Segment`` definition.
    *   Sum these durations based on the TR definition to calculate ``min_TR_grad``.
    *   The final check: The user-defined ``TR_duration`` must be greater than or equal to ``min_TR_hw``, ``min_TR_sar``, and ``min_TR_grad``.

3.3. Dynamic Gradient Safety Checks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These checks use the "Slew-Rate Worst-Case" instances.

1.  **Waveform Generation and Validation:** Interpolate the gradient waveforms for the `smax` instances of each ``BlockGroup``. Check that these start/end with zero. Verify that no ``BlockGroup`` crosses a ``Segment`` boundary and that segments also have zero-crossings.
2.  **Peak Slew Check:** Compute the peak slew rate for each interpolated `smax` ``BlockGroup`` and compare against the system limit.
3.  **Streaming Acoustic/PNS Analysis:**
    *   **No Explicit TR Waveform is Built.** The system uses the TR definition *in terms of blocks* to orchestrate the streaming of data.
    *   **Acoustics:** A chunked analysis is performed. Data from the `smax` instance of each block is streamed into a buffer for a **Goertzel algorithm** analysis, checking only forbidden frequency bins. A violation is flagged immediately. The TR is conceptually repeated to ensure analysis covers TR boundaries correctly, relative to the spectrogram window length.
    *   **PNS:** A chunked convolution is performed. Slew rate data from the `smax` instance of each block is streamed into a history buffer for convolution with the nerve kernel. A violation is flagged immediately. The TR is conceptually repeated to ensure analysis covers TR boundaries correctly, relative to the PNS kernel length.

----

4. Stage 2: Instruction Building (`pulsegen`)
---------------------------------------------

This stage translates the validated definitions into hardware instruction templates.

4.1. Global ADC and Filter Preparation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.  **Find Max Samples:** Determine the maximum ``nSamples`` across all ADC events to configure the primary receive filter.
2.  **Create Echo Filters:** For each unique ADC definition (`nSamples`, `dwellTime`), create a dedicated hardware filter (`echo_filt[p]`). Store its hardware slot index in a LUT, ensuring a consistent mapping that matches the definition used at runtime.

4.2. Segment Instruction Template Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each ``Segment`` definition:

1.  **Initialize Timeline:** A ``segment_time_us`` variable is initialized to `0`.
2.  **Iterate Over Blocks:** For each ``Block`` in the ``Segment``'s definition:
    a. **Determine Initial Gradient Amplitude:** The initial amplitude for gradient instructions is set from the **Max SOS Energy** instance found for this block position during the validation pass.
    b. **Create Instructions:** For each `Event` within the block:
        *   Calculate start time: ``event_start_time = segment_time_us + event.delay``.
        *   Call the appropriate ``make...`` function, providing ``Shape`` data, start time, initial amplitude (for gradients), and the correct echo filter index (for ADCs).
    c. **Advance Timeline:** Update ``segment_time_us`` by the block's duration.

----

5. Stage 3: Real-Time Execution and Iteration
----------------------------------------------

This stage executes the scan using the pre-built templates.

5.1. Runtime Parameter Parsing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The real-time loop reads parameters on a per-instance basis from the event loop data. For each block instance, it parses:

*   **RF:** The **statically stored** amplitude determined during the first-TR safety analysis. The ``phase`` and ``frequencyOffset`` are read dynamically from the current event loop entry.
*   **Gradients:** The ``amplitude`` (respecting the "Last-Block-Rules" policy), ``rotation`` events, and any other dynamic parameters are read from the current event loop entry.
*   **Delays:** For variable delay blocks, the specific duration is read.

5.2. Real-Time Execution Logic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.  **Initialize Iterators:** A primary iterator (`block_instance_ptr`) points to the current entry in the event loop. State trackers (`current_tr_index`, `current_segment_in_tr_index`) are initialized.
2.  **Process One Segment Instance:**
    a. **Identify Template:** Read the ``COREID`` to select the `Segment` instruction template.
    b. **Update Instructions:** Iterate through the blocks of the ``Segment`` template. For each, use `block_instance_ptr` to read the dynamic parameters specified above and update the template instructions.
    c. **Advance Iterator:** Increment `block_instance_ptr` after processing each block instance.
    d. **Execute Segment:** Trigger the hardware to execute the fully updated `Segment` template.
    e. **Update State:** Update the TR and segment-in-TR counters. At the TR boundary, pause for any external synchronization hooks.

----

Appendix: Computationally Efficient Algorithms
=============================================

This appendix details the memory- and CPU-efficient streaming algorithms used in the Safety Validation stage.

A.1 Streamed Time-Domain Convolution for PNS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PNS check requires convolving the gradient slew-rate waveform with a nerve stimulation kernel (defined by factors like chronaxie and the gradient raster time, `dt`). A naive convolution would require storing the entire multi-megasample slew-rate waveform and its convolved output, which is unfeasible.

*   **Algorithm:** A streaming direct time-domain convolution is used.
*   **Implementation:**
    1.  A circular history buffer of size ``M`` (the exact length of the nerve kernel) is allocated.
    2.  As each new slew-rate sample, `x[n]`, is generated from the `smax` instances, it is pushed into the history buffer.
    3.  A single output sample, `y[n]`, is calculated by taking the dot product of the current state of the history buffer and the (time-reversed) nerve kernel.
    4.  The output `y[n]` is immediately compared against the PNS threshold. If it exceeds the limit, the validation fails instantly. The output sample is then discarded.
*   **Benefit:** This approach has a constant memory footprint of `M` samples and a constant CPU cost per sample, regardless of the sequence length. It never retains the full convolved slew-rate array.

A.2 Streamed Goertzel Algorithm for Acoustics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The acoustic check requires ensuring that no significant energy is present in a few specific, forbidden frequency bands. A full FFT is inefficient as it calculates the entire spectrum, while we only need the power at a few points.

*   **Algorithm:** The Goertzel algorithm is a technique to compute the power at a single frequency bin, making it far more efficient than an FFT for sparse spectral analysis.
*   **Implementation:**
    1.  The forbidden frequency bands are converted to a small, sparse list of integer frequency bin indices.
    2.  A Short-Time Fourier Transform (STFT) window buffer of size ``L`` is allocated.
    3.  Data from the `smax` gradient waveform instances is streamed through this buffer.
    4.  For each filled window of data, the system iterates *only* through the small list of forbidden bin indices and executes the Goertzel algorithm for each one.
    5.  The resulting power is compared against the acoustic limit. If exceeded, validation fails instantly.
*   **Benefit:** This avoids the high CPU cost of a full FFT and the high memory cost of storing a full spectrogram. Memory usage is constant (size ``L``) and CPU usage is proportional only to the number of forbidden frequencies, not the full spectral width.

A.3 Rationale for Multi-TR Analysis Window
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

History-dependent phenomena like PNS and acoustics cannot be accurately assessed by analyzing a single TR in isolation, as this would treat the TR boundary as a zero-crossing and ignore "memory" effects from the end of one TR carrying over to the start of the next.

*   **Problem:** A nerve has a stimulation "memory" defined by its kernel length. An acoustic resonance has a "memory" defined by the analysis window. If the TR is shorter than this memory, an analysis of a single TR will be incorrect.
*   **Solution:** To model the steady-state behavior of a repeating sequence, the worst-case TR is conceptually repeated. The streaming analysis is performed on this continuous signal.
*   **Implementation:** A minimum of **two TRs** (``N=2``) is used for the analysis window. This ensures that at the boundary point between the first and second conceptual TR, the algorithm's history buffer or analysis window is filled with valid data from the end of the TR, correctly simulating the transition. The formal requirement is that the total analysis duration (``N * TR_duration``) must be sufficiently longer than the analysis kernel/window to allow for proper boundary analysis. Using at least two TRs is a robust and simple-to-implement heuristic that satisfies this condition for typical kernel/window sizes.
