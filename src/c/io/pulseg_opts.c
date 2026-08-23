/* pulseg_opts.c -- pulseg_opts construction and the definitions-only
 * scan-time peek.
 *
 * These sit directly on top of the standalone pulseq reader: the peek walks a
 * "next sequence" chain reading nothing but each file's [DEFINITIONS], which
 * is why it needs no collection and no conversion.  Both moved here when the
 * parser became the vendor-neutral pulseq module (it must not know about
 * pulseg_opts).
 */

#include <string.h>

#include "pulseg_internal.h"

/* ================================================================== */
/*  Opts init                                                         */
/* ================================================================== */

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
    float peak_eps)
{
    if (!opts)
        return;

    opts->vendor = PULSEG_VENDOR;
    opts->gamma_hz_per_t = gamma_hz_per_t;
    opts->b0_t = b0_t;
    opts->max_grad_hz_per_m = max_grad_hz_per_m;
    opts->max_slew_hz_per_m_per_s = max_slew_hz_per_m_per_s;
    opts->rf_raster_us = rf_raster_us;
    opts->grad_raster_us = grad_raster_us;
    opts->adc_raster_us = adc_raster_us;
    opts->block_raster_us = block_raster_us;
    opts->peak_log10_threshold = peak_log10_threshold;
    opts->peak_norm_scale = peak_norm_scale;
    opts->peak_eps = peak_eps;
    opts->peak_prominence = PULSEG_PEAK_PROMINENCE_DEFAULT;
    opts->vendor_rf_stats_fn = NULL;
    opts->vendor_rf_stats_ctx = NULL;
    opts->label_column_map[0] = 0; /* identity default: SLC, PHS, REP; */
    opts->label_column_map[1] = 1; /* vendor layers override before parsing */
    opts->label_column_map[2] = 2;
    strncpy(opts->cache_ext, PULSEG_CACHE_EXT_DEFAULT, sizeof(opts->cache_ext) - 1);
    opts->cache_ext[sizeof(opts->cache_ext) - 1] = '\0';
    opts->vendor_section_write_fn = NULL;
    opts->vendor_section_ctx = NULL;
    opts->allow_variable_rf_amplitude = 1;
}

void pulseg_opts_init(
    pulseg_opts *opts,
    float gamma_hz_per_t,
    float b0_t,
    float max_grad_hz_per_m,
    float max_slew_hz_per_m_per_s,
    float rf_raster_us,
    float grad_raster_us,
    float adc_raster_us,
    float block_raster_us)
{
    pulseg_opts_init_full(
        opts,
        gamma_hz_per_t,
        b0_t,
        max_grad_hz_per_m,
        max_slew_hz_per_m_per_s,
        rf_raster_us,
        grad_raster_us,
        adc_raster_us,
        block_raster_us,
        PULSEG_PEAK_LOG10_THRESHOLD_DEFAULT,
        PULSEG_PEAK_NORM_SCALE_DEFAULT,
        PULSEG_PEAK_EPS_DEFAULT);
}

void pulseg_opts_get_design_raster(pulseq_raster *raster, const pulseg_opts *opts)
{
    if (!raster)
        return;

    if (!opts)
    {
        raster->rf_us = 0.0f;
        raster->grad_us = 0.0f;
        raster->block_us = 0.0f;
        return;
    }

    raster->rf_us = opts->rf_raster_us;
    raster->grad_us = opts->grad_raster_us;
    raster->block_us = opts->block_raster_us;
}

/* ================================================================== */
/*  Scan-time peek                                                    */
/* ================================================================== */

int pulseg_peek_scan_time(
    pulseg_scan_time_info *info,
    const char *file_path,
    const pulseg_opts *opts)
{
    pulseq_raster raster;
    char *current_path;
    char *base_path;
    int count = 0;
    int max_depth = 1000;

    if (!info || !file_path || !opts)
        return PULSEG_ERR_NULL_POINTER;

    info->total_duration_us = 0.0f;
    info->total_segment_boundaries = 0;

    pulseg_opts_get_design_raster(&raster, opts);

    base_path = pulseq_path_dirname(file_path);
    if (!base_path)
        return PULSEG_ERR_ALLOC_FAILED;

    current_path = (char *)PULSEG_ALLOC(strlen(file_path) + 1);
    if (!current_path)
    {
        PULSEG_FREE(base_path);
        return PULSEG_ERR_ALLOC_FAILED;
    }
    strcpy(current_path, file_path);

    while (current_path && current_path[0] != '\0' && count < max_depth)
    {
        pulseq_file temp;
        int result;

        pulseq_file_init(&temp, &raster);
        result = pulseq_read_definitions_only(&temp, current_path);
        if (PULSEG_FAILED(result))
        {
            pulseq_file_free(&temp);
            PULSEG_FREE(current_path);
            PULSEG_FREE(base_path);
            return result;
        }

        info->total_duration_us += temp.reserved_definitions_library.total_duration * 1e6f;
        count++;

        PULSEG_FREE(current_path);
        current_path = NULL;

        if (temp.reserved_definitions_library.next_sequence[0] != '\0')
        {
            current_path =
                pulseq_path_join(base_path, temp.reserved_definitions_library.next_sequence);
            if (!current_path)
            {
                pulseq_file_free(&temp);
                PULSEG_FREE(base_path);
                return PULSEG_ERR_ALLOC_FAILED;
            }
        }
        pulseq_file_free(&temp);
    }

    PULSEG_FREE(current_path);
    PULSEG_FREE(base_path);
    return (count > 0) ? PULSEG_SUCCESS : PULSEG_ERR_COLLECTION_EMPTY;
}
