/**
 * @file pulseq_adapter.h
 * @brief Adapt a standard-toolkit `ExternalSequence` (pulseq/pulseq,
 * `external/pulseq` submodule) onto a `pulseg_pulseq_file` raw model, so it
 * can be handed to `pulseg_convert_collection()` -- the same conversion path
 * used for files loaded through pulseg's own C reader
 * (`pulseg_pulseq_file_read()`).
 *
 * Rationale (Stage 3 of the vendor-neutral refactor): pulseg ships its own
 * from-scratch Pulseq `.seq` parser (csrc/src/io/pulseg_parse.c) because it
 * predates this integration and needs a couple of pulserver-specific custom
 * LABELSET names (PMC, TRID -- see below). This adapter lets callers use the
 * upstream reference implementation instead where that bespoke parser is not
 * required, while still funneling into the same dedup/structure/safety code.
 *
 * Known limitations (found during the Stage 3 Step 0/3 investigation,
 * documented rather than silently "fixed"):
 *
 * 1. PMC / TRID labels: `ExternalSequence::decodeLabel()` (see
 *    `external/pulseq/src/ExternalSequence.cpp`, `LABELMAP_IGNORE(PMC)` /
 *    `LABELMAP_IGNORE(TRID)`) *deliberately* discards these two
 *    pulserver-specific LABELSET/LABELINC names -- they never reach
 *    `m_labelsetLibrary` / `m_labelincLibrary`, so there is no public API to
 *    recover them from a decoded block. This adapter therefore does its own
 *    lightweight re-scan of the raw `[LABELSET]` / `[LABELINC]` text
 *    sections (see `pulseq_adapter.cpp`, `rescan_ignored_labels()`) to
 *    recover just these two names and re-inject them -- exactly the
 *    "re-scan the .seq text" fallback anticipated by the Stage 3 plan.
 *    NOTE: ROTATIONS and RF_SHIMS themselves are NOT in this situation --
 *    `ExternalSequence` has full native support for both (`EXT_ROTATION`,
 *    `EXT_RF_SHIM`, `m_rotationLibrary`, `m_rfShimLibrary`), contrary to
 *    what an older version of the toolkit might have required.
 * 2. Non-uniform RF time raster: `ExternalSequence::decodeBlock()` resamples
 *    any non-uniform RF `timeShape` onto a uniform dwell-time grid *before*
 *    exposing `rfAmplitude` / `rfPhase` (nearest-neighbour resampling, see
 *    the `waveform_t` handling in `decodeBlock()`). The original non-uniform
 *    time shape is not recoverable afterwards, so this adapter always emits
 *    RF events with `time_shape` absent (implicit uniform raster at
 *    `opts.rf_raster_us` / the file's own `RadiofrequencyRasterTime`).
 *    Sequences that rely on a genuinely non-uniform RF raster must go
 *    through pulseg's native reader (`pulseg_pulseq_file_read()`).
 *    NOTE: non-uniform *gradient* rasters ("extended trapezoids", Pulseq's
 *    `mr.makeExtendedTrapezoid`) are NOT in this situation -- ExternalSequence
 *    keeps their genuine (time, amplitude) sample pairs available via
 *    `SeqBlock::GetExtTrapGradTimes/GetExtTrapGradShape` without resampling,
 *    and this adapter reconstructs the raw (raster-index-normalized)
 *    time_shape from them (undoing ExternalSequence's absolute-microsecond
 *    conversion, which uses the file's own `GradientRasterTime`).
 * 3. ADC per-sample phase-modulation shapes (`ADCEvent::phaseModulationShape`)
 *    are not populated (rare pTx receive-side feature); adapter output
 *    always has this absent (shape id 0).
 *
 * None of the above affect block count, timing, RF/gradient/ADC amplitudes,
 * rotations, triggers, soft delays, RF shims, or the standard MDH label set
 * -- the cpptest (`tests/cpptests/test_pulseq_adapter_equivalence.cpp`)
 * checks exactly those against the native C reader on a fixture with no
 * non-uniform-raster events.
 */

#ifndef PULSEG_PULSEQ_ADAPTER_H
#define PULSEG_PULSEQ_ADAPTER_H

#include <string>

#include "pulseg_io.h"

class ExternalSequence; // external/pulseq/src/ExternalSequence.h

namespace pulseg
{
    namespace adapter
    {

        /**
         * @brief Build a pulseg_pulseq_file from an already-loaded
         * ExternalSequence.
         *
         * @p out is allocated by this function (via PULSEG_ALLOC, matching
         * pulseg_pulseq_file_free()'s expectation that the struct itself --
         * not just its inner arrays -- was heap-allocated; a stack or
         * `new`-allocated pulseg_pulseq_file must NOT be passed to
         * pulseg_pulseq_file_free()). On success the caller owns `*out` and
         * must free it with pulseg_pulseq_file_free(). On failure `*out` is
         * set to NULL (nothing to free).
         *
         * @param[in]  seq      An ExternalSequence that has already had
         *                      load() / load_from_buffer() called successfully.
         * @param[in]  opts     Library options (raster times, gamma, etc.);
         *                      copied into out->opts exactly as
         *                      pulseg_pulseq_file_init() would.
         * @param[out] out      Receives a freshly heap-allocated file.
         * @param[out] err_msg  Optional; on failure, set to a human-readable
         *                      reason.
         * @param[in]  source_path  Optional path to the same .seq file @p seq
         *                      was loaded from. When provided, enables the
         *                      PMC/TRID re-scan fallback described above
         *                      (limitation 1); when null/empty, PMC/TRID
         *                      LABELSET/LABELINC entries are simply absent
         *                      from the result (e.g. when @p seq was loaded
         *                      via load_from_buffer() with no backing file).
         * @return true on success, false on failure.
         */
        bool from_external_sequence(
            ExternalSequence &seq,
            const pulseg_opts &opts,
            pulseg_pulseq_file **out,
            std::string *err_msg,
            const char *source_path = nullptr);

    } // namespace adapter
} // namespace pulseg

#endif // PULSEG_PULSEQ_ADAPTER_H
