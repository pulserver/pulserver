/**
 * @file pulseg_io.h
 * @brief pulseg's view of the raw Pulseq file model, plus scan-time peek.
 *
 * The raw Pulseq reader is a standalone module (src/c/include/pulseq/) that
 * knows nothing about pulseg.  This header re-exports it for pulseg-side code
 * and adds the pulseg-level entry points that only need a parsed file.
 *
 * Dependency direction is strictly one-way: pulseg includes pulseq, never the
 * reverse.
 */

#ifndef PULSEG_IO_H
#define PULSEG_IO_H

#include <stdio.h>

#include "pulseq.h"

#include "pulseg_config.h"
#include "pulseg_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* ================================================================== */
    /*  Options initializer                                               */
    /* ================================================================== */

    /**
     * @brief Fill a pulseg_opts struct with scanner parameters.
     *
     * The @c vendor field is set to @c PULSEG_VENDOR (compile-time
     * default).  Override it after calling this function if needed.
     *
     * The raster arguments are the SYSTEM (scanner hardware) rasters.  A
     * sequence's own [DEFINITIONS] rasters are validated against these for
     * playability; they are not the rasters the file's waveforms are written
     * on.  See pulseq_raster.
     */
    void pulseg_opts_init_full(
        pulseg_opts *opts,
        float gamma_hz_per_t,
        float b0_t,
        float max_grad_hz_per_m,
        float max_slew_hz_per_m_per_s,
        float rf_raster_us,
        float grad_raster_us,
        float adc_raster_us,
        float block_raster_us,
        float peak_log10_threshold,
        float peak_norm_scale,
        float peak_eps);

    /**
     * @brief Legacy initializer using default peak-detection parameters.
     */
    void pulseg_opts_init(
        pulseg_opts *opts,
        float gamma_hz_per_t,
        float b0_t,
        float max_grad_hz_per_m,
        float max_slew_hz_per_m_per_s,
        float rf_raster_us,
        float grad_raster_us,
        float adc_raster_us,
        float block_raster_us);

    /**
     * @brief Extract the design-time rasters a pulseq parse should use from
     * a pulseg_opts.
     *
     * pulseq consults these only for rasters the .seq file itself omits.
     *
     * @param[out] raster  Receives the design-time rasters (zeroed if @p opts
     *                     is NULL).
     * @param[in]  opts    Options carrying the system rasters.
     */
    void pulseg_opts_get_design_raster(pulseq_raster *raster, const pulseg_opts *opts);

    /* ================================================================== */
    /*  Scan-time peek (fast estimate from definitions only)              */
    /* ================================================================== */

    /**
     * @brief Peek at scan time without full sequence loading.
     *
     * Reads only the [DEFINITIONS] sections from a (possibly chained)
     * .seq file to obtain @c TotalDuration.  The result is an
     * approximation: dead time between segments is not accounted for
     * and @c total_segment_boundaries is left at 0.
     *
     * Both @c total_duration_us and @c total_segment_boundaries are populated.
     *
     * @param[out] info       Receives scan time summary.
     * @param[in]  file_path  Path to the first .seq file (may be chained).
     * @param[in]  opts       Library options.
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */
    int pulseg_peek_scan_time(
        pulseg_scan_time_info *info,
        const char *file_path,
        const pulseg_opts *opts);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_IO_H */
