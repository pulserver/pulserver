/**
 * @file pulseg_io.h
 * @brief The options a conversion is given, and pulseg's view of the raw
 *        Pulseq file model.
 *
 * The Pulseq file model is a standalone module (src/c/include/pulseq/) that
 * knows nothing about pulseg. This header re-exports it for pulseg-side code.
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
    void pulseg_opts_init(
        pulseg_opts *opts,
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

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_IO_H */
