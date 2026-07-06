/**
 * @file pulseg_io.h
 * @brief Pulseq .seq reader and scan-time-peek entry points.
 *
 * Split out of the former pulseg_methods.h (Stage 1 layout normalization).
 * All functions use the pulseg_ prefix and are declared extern "C" when
 * compiled with a C++ compiler.
 */

#ifndef PULSEG_IO_H
#define PULSEG_IO_H

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
     * @c num_reps controls the number of repetitions the consumer
     * intends to play (>= 1).  For subsequences whose
     * @c IgnoreAverages definition is set, the multiplier is clamped to 1.
     *
     * Both @c total_duration_us and @c total_segment_boundaries are populated.
     *
     * @param[out] info       Receives scan time summary.
     * @param[in]  file_path  Path to the first .seq file (may be chained).
     * @param[in]  opts       Library options.
     * @param[in]  num_reps   Number of repetitions (>= 1).
     * @return PULSEG_SUCCESS on success, negative error code on failure.
     */
    int pulseg_peek_scan_time(
        pulseg_scan_time_info *info,
        const char *file_path,
        const pulseg_opts *opts,
        int num_reps);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_IO_H */
