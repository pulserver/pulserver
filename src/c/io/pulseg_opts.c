/* pulseg_opts.c -- the options a conversion is given. */

#include <string.h>

#include "pulseg_internal.h"

/* ================================================================== */
/*  Opts init                                                         */
/* ================================================================== */

void pulseg_opts_init(
    pulseg_opts *opts,
    float rf_raster_us,
    float grad_raster_us,
    float adc_raster_us,
    float block_raster_us)
{
    if (!opts)
        return;

    opts->vendor = PULSEG_VENDOR;
    opts->rf_raster_us = rf_raster_us;
    opts->grad_raster_us = grad_raster_us;
    opts->adc_raster_us = adc_raster_us;
    opts->block_raster_us = block_raster_us;
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
    opts->structure_only = 0;
    opts->borrow_buffer_shapes = 0;
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
