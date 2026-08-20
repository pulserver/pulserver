/**
 * @file pulseg_raster.c
 * @brief Raster alignment of every event time against the scanner's rasters.
 *
 * pulseg_check_raster_alignment() asks one question of a collection: can the
 * hardware start and stop every event where the sequence says?  A time that is
 * not an integer multiple of the raster it is played against cannot be, and
 * pulseg_check_raster_times() -- which compares the file's declared rasters
 * with the scanner's -- does not answer it: the two grids may be integer
 * multiples in the direction that lets a finer declared raster through, and a
 * 15 us event on a 5 us declared grid is then still unplayable on a 10 us one.
 *
 * The cost model is the gradient library's, not the block table's.  Every time
 * field lives in a deduplicated definition -- rf, gradient, ADC -- and a base
 * block carries its own duration, so the whole check is a pass over four
 * definition arrays whose size is the number of *distinct* events in the scan.
 *
 * Dead times and ringdown are deliberately absent: they are limits on what a
 * transmit chain can do, not on what the sequencer can address, and the
 * interpreter does not enforce them.  The design side checks those.
 */

#include <limits.h>
#include <math.h>

#include "pulseg_internal.h"
#include "pulseg.h"

/* Times that must land exactly are integer microseconds, so the arithmetic is
 * integer too.  A raster is a float because a scanner may declare a fraction
 * of a microsecond (a 100 ns ADC raster is ordinary), hence the conversion to
 * nanoseconds first. */

/**
 * @brief A raster in whole nanoseconds, or 0 if it is not one.
 */
static int raster_in_ns(float raster_us)
{
    double ns;
    int rounded;

    ns = (double)raster_us * 1000.0;
    if (ns <= 0.0 || ns > (double)INT_MAX)
        return 0;
    rounded = (int)(ns + 0.5);
    if (fabs(ns - (double)rounded) > 1e-3)
        return 0;
    return rounded;
}

/**
 * @brief Whether a whole-microsecond time is a multiple of a raster.
 *
 * A raster that divides one microsecond admits every microsecond, which is
 * the answer for a sub-microsecond ADC raster and avoids scaling the time up
 * at all.  Otherwise the modulo runs in microseconds when the raster is a
 * whole number of them, and only a raster that is neither -- 300 ns, say --
 * pays the conversion, where the guard keeps a long block in range.
 */
static int us_on_raster(int value_us, int rast_ns)
{
    if (rast_ns <= 0)
        return 1;
    if (1000 % rast_ns == 0)
        return 1;
    if (rast_ns % 1000 == 0)
        return (value_us % (rast_ns / 1000)) == 0;
    if (value_us > INT_MAX / 1000 || value_us < -(INT_MAX / 1000))
        return 1;
    return ((value_us * 1000) % rast_ns) == 0;
}

/**
 * @brief Whether a sample time carried by a shape lands on a raster.
 *
 * The time shape is stored in units of the raster the sequence declared, so
 * this compares a real microsecond time against the scanner's raster and has
 * to tolerate the shape codec's own rounding.
 */
static int time_on_raster(double value_us, double raster_us)
{
    double ratio, rounded;

    if (raster_us <= 0.0)
        return 1;
    ratio = value_us / raster_us;
    rounded = (double)((long)(ratio >= 0.0 ? ratio + 0.5 : ratio - 0.5));
    return fabs(ratio - rounded) <= 1e-6 * (fabs(rounded) + 1.0);
}

/**
 * @brief Report a misaligned value and return the alignment error code.
 */
static int misaligned(
    pulseg_diagnostic *diag,
    const char *event,
    const char *field,
    int index,
    double value_us,
    float raster_us)
{
    if (diag)
    {
        diag->code = PULSEG_ERR_RASTER_ALIGNMENT;
        pulseg__diag_printf(
            diag,
            "%s[%d].%s=%.3fus not on %.3fus raster",
            event,
            index,
            field,
            value_us,
            (double)raster_us);
    }
    return PULSEG_ERR_RASTER_ALIGNMENT;
}

/**
 * @brief Every sample of a time shape, against @p raster_us.
 *
 * @param shape_id  1-based pulseq shape id; <= 0 means the event has no time
 *                  shape and sits on the raster by construction.
 * @param declared_us  The raster the shape's units are counted in.
 * @param[out] end_us  Last sample time, when the shape has one.
 */
static int check_time_shape(
    const pulseg_sequence_descriptor *desc,
    pulseg_diagnostic *diag,
    const char *event,
    int index,
    int shape_id,
    float declared_us,
    float raster_us,
    double *end_us)
{
    pulseq_shape decompressed;
    int i, rc;

    if (shape_id <= 0 || shape_id > desc->num_shapes || !desc->shapes)
        return PULSEG_SUCCESS;

    decompressed.samples = NULL;
    if (!pulseq_decompress_shape(&decompressed, &desc->shapes[shape_id - 1], declared_us))
        return PULSEG_SUCCESS;

    rc = PULSEG_SUCCESS;
    for (i = 0; i < decompressed.num_uncompressed_samples; ++i)
    {
        if (!time_on_raster((double)decompressed.samples[i], (double)raster_us))
        {
            rc = misaligned(
                diag,
                event,
                "time shape sample",
                index,
                (double)decompressed.samples[i],
                raster_us);
            break;
        }
    }
    if (end_us && decompressed.num_uncompressed_samples > 0)
        *end_us = (double)decompressed.samples[decompressed.num_uncompressed_samples - 1];

    PULSEG_FREE(decompressed.samples);
    return rc;
}

/**
 * @brief How long a shape-carried event lasts, in microseconds.
 */
static double shape_duration_us(
    const pulseg_sequence_descriptor *desc,
    int shape_id,
    int time_shape_id,
    float declared_us)
{
    pulseq_shape decompressed;
    double end;

    if (time_shape_id > 0 && time_shape_id <= desc->num_shapes && desc->shapes)
    {
        decompressed.samples = NULL;
        if (pulseq_decompress_shape(&decompressed, &desc->shapes[time_shape_id - 1], declared_us))
        {
            end = (decompressed.num_uncompressed_samples > 0)
                ? (double)decompressed.samples[decompressed.num_uncompressed_samples - 1]
                : 0.0;
            PULSEG_FREE(decompressed.samples);
            return end;
        }
    }

    if (shape_id > 0 && shape_id <= desc->num_shapes && desc->shapes)
        return (double)desc->shapes[shape_id - 1].num_uncompressed_samples * (double)declared_us;

    return 0.0;
}

/**
 * @brief How long an arbitrary gradient definition lasts, in microseconds.
 *
 * A time shape carries the sample times in raster units, so the waveform ends
 * at its last entry however few samples the magnitude shape holds; without one
 * the samples sit on the raster and the count gives the duration.
 */
static double grad_shape_duration_us(
    const pulseg_sequence_descriptor *desc,
    const pulseg_grad_definition *g)
{
    double end;

    if (g->unused_or_time_shape_id > 0)
    {
        end = shape_duration_us(desc, 0, g->unused_or_time_shape_id, desc->grad_raster_us);
        if (end > 0.0)
            return end;
    }
    return (double)g->fall_time_or_num_uncompressed_samples * (double)desc->grad_raster_us;
}

/* ================================================================== */
/*  Per-library passes                                                */
/* ================================================================== */

static int check_rf_definitions(
    const pulseg_sequence_descriptor *desc,
    pulseg_diagnostic *diag,
    const pulseg_opts *opts)
{
    const pulseg_rf_definition *rf;
    int i, rc, rast_ns;

    rast_ns = raster_in_ns(opts->rf_raster_us);
    for (i = 0; i < desc->num_unique_rfs; ++i)
    {
        rf = &desc->rf_definitions[i];
        if (rf->delay < 0)
            return misaligned(diag, "rf", "delay", i, (double)rf->delay, opts->rf_raster_us);
        if (!us_on_raster(rf->delay, rast_ns))
            return misaligned(diag, "rf", "delay", i, (double)rf->delay, opts->rf_raster_us);

        rc = check_time_shape(
            desc,
            diag,
            "rf",
            i,
            rf->time_shape_id,
            desc->rf_raster_us,
            opts->rf_raster_us,
            NULL);
        if (PULSEG_FAILED(rc))
            return rc;
    }
    return PULSEG_SUCCESS;
}

static int check_grad_definitions(
    const pulseg_sequence_descriptor *desc,
    pulseg_diagnostic *diag,
    const pulseg_opts *opts)
{
    const pulseg_grad_definition *g;
    int i, rc, rast_ns;

    rast_ns = raster_in_ns(opts->grad_raster_us);
    for (i = 0; i < desc->num_unique_grads; ++i)
    {
        g = &desc->grad_definitions[i];
        if (g->delay < 0 || !us_on_raster(g->delay, rast_ns))
            return misaligned(diag, "grad", "delay", i, (double)g->delay, opts->grad_raster_us);

        if (g->type == 0)
        {
            if (!us_on_raster(g->rise_time_or_unused, rast_ns))
                return misaligned(
                    diag,
                    "grad",
                    "rise_time",
                    i,
                    (double)g->rise_time_or_unused,
                    opts->grad_raster_us);
            if (!us_on_raster(g->flat_time_or_unused, rast_ns))
                return misaligned(
                    diag,
                    "grad",
                    "flat_time",
                    i,
                    (double)g->flat_time_or_unused,
                    opts->grad_raster_us);
            if (!us_on_raster(g->fall_time_or_num_uncompressed_samples, rast_ns))
                return misaligned(
                    diag,
                    "grad",
                    "fall_time",
                    i,
                    (double)g->fall_time_or_num_uncompressed_samples,
                    opts->grad_raster_us);
        }
        else
        {
            rc = check_time_shape(
                desc,
                diag,
                "grad",
                i,
                g->unused_or_time_shape_id,
                desc->grad_raster_us,
                opts->grad_raster_us,
                NULL);
            if (PULSEG_FAILED(rc))
                return rc;
        }
    }
    return PULSEG_SUCCESS;
}

/*
 * The ADC start time goes against the RF raster, not the ADC one: the dwell is
 * what the digitiser counts in, the start is what the sequencer has to
 * address.  Pulseq says so explicitly (doc/pulseq_shapes_and_times.pdf) and
 * upstream's own checker follows it.
 */
static int check_adc_definitions(
    const pulseg_sequence_descriptor *desc,
    pulseg_diagnostic *diag,
    const pulseg_opts *opts)
{
    const pulseg_adc_definition *adc;
    int i, delay_rast_ns, dwell_rast_ns;

    delay_rast_ns = raster_in_ns(opts->rf_raster_us);
    dwell_rast_ns = raster_in_ns(opts->adc_raster_us);

    for (i = 0; i < desc->num_unique_adcs; ++i)
    {
        adc = &desc->adc_definitions[i];
        if (adc->delay < 0 || !us_on_raster(adc->delay, delay_rast_ns))
            return misaligned(diag, "adc", "delay", i, (double)adc->delay, opts->rf_raster_us);

        if (dwell_rast_ns > 0 && adc->dwell_time > 0 && (adc->dwell_time % dwell_rast_ns) != 0)
            return misaligned(
                diag,
                "adc",
                "dwell",
                i,
                (double)adc->dwell_time / 1000.0,
                opts->adc_raster_us);
    }
    return PULSEG_SUCCESS;
}

/**
 * @brief Block durations on the block raster, and events inside their block.
 *
 * A block longer than its events is ordinary -- that is how a TR filler is
 * written -- so the comparison is one-sided: nothing may end after the block
 * does.  RF ringdown and ADC dead time are not counted, being transmit-chain
 * margins rather than sequencer addresses.
 */
static int check_base_blocks(
    const pulseg_sequence_descriptor *desc,
    pulseg_diagnostic *diag,
    const pulseg_opts *opts)
{
    const pulseg_base_block *blk;
    const pulseg_grad_definition *g;
    const pulseg_adc_definition *adc;
    const pulseg_rf_definition *rf;
    int i, n, rast_ns, grad_id;
    double end, duration;

    rast_ns = raster_in_ns(opts->block_raster_us);

    for (i = 0; i < desc->num_unique_blocks; ++i)
    {
        blk = &desc->base_blocks[i];
        if (!us_on_raster(blk->duration_us, rast_ns))
            return misaligned(
                diag,
                "block",
                "duration",
                i,
                (double)blk->duration_us,
                opts->block_raster_us);

        duration = (double)blk->duration_us;

        if (blk->rf_id >= 0 && blk->rf_id < desc->num_unique_rfs)
        {
            rf = &desc->rf_definitions[blk->rf_id];
            end = (double)rf->delay +
                shape_duration_us(desc, rf->mag_shape_id, rf->time_shape_id, desc->rf_raster_us);
            if (end > duration + 1e-3)
            {
                if (diag)
                {
                    diag->code = PULSEG_ERR_BLOCK_DURATION_OVERRUN;
                    pulseg__diag_printf(
                        diag,
                        "block[%d] rf ends at %.3fus > %.3fus",
                        i,
                        end,
                        duration);
                }
                return PULSEG_ERR_BLOCK_DURATION_OVERRUN;
            }
        }

        for (n = 0; n < 3; ++n)
        {
            grad_id = (n == 0) ? blk->gx_id : ((n == 1) ? blk->gy_id : blk->gz_id);
            if (grad_id < 0 || grad_id >= desc->num_unique_grads)
                continue;
            g = &desc->grad_definitions[grad_id];
            if (g->type == 0)
                end = (double)g->delay + (double)g->rise_time_or_unused +
                    (double)g->flat_time_or_unused +
                    (double)g->fall_time_or_num_uncompressed_samples;
            else
                end = (double)g->delay + grad_shape_duration_us(desc, g);
            if (end > duration + 1e-3)
            {
                if (diag)
                {
                    diag->code = PULSEG_ERR_BLOCK_DURATION_OVERRUN;
                    pulseg__diag_printf(
                        diag,
                        "block[%d] g%c ends at %.3fus > %.3fus",
                        i,
                        (n == 0) ? 'x' : ((n == 1) ? 'y' : 'z'),
                        end,
                        duration);
                }
                return PULSEG_ERR_BLOCK_DURATION_OVERRUN;
            }
        }

        if (blk->adc_id >= 0 && blk->adc_id < desc->num_unique_adcs)
        {
            adc = &desc->adc_definitions[blk->adc_id];
            end = (double)adc->delay + (double)adc->num_samples * (double)adc->dwell_time / 1000.0;
            if (end > duration + 1e-3)
            {
                if (diag)
                {
                    diag->code = PULSEG_ERR_BLOCK_DURATION_OVERRUN;
                    pulseg__diag_printf(
                        diag,
                        "block[%d] adc ends at %.3fus > %.3fus",
                        i,
                        end,
                        duration);
                }
                return PULSEG_ERR_BLOCK_DURATION_OVERRUN;
            }
        }
    }
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Entry point                                                       */
/* ================================================================== */

int pulseg_check_raster_alignment(
    const pulseg_collection *coll,
    pulseg_diagnostic *diag,
    const pulseg_opts *opts)
{
    const pulseg_sequence_descriptor *desc;
    int s, rc;

    if (!coll || !opts)
    {
        if (diag)
        {
            pulseg_diagnostic_init(diag);
            diag->code = PULSEG_ERR_NULL_POINTER;
        }
        return PULSEG_ERR_NULL_POINTER;
    }
    if (diag)
        pulseg_diagnostic_init(diag);

    for (s = 0; s < coll->num_subsequences; ++s)
    {
        desc = &coll->descriptors[s];

        rc = check_rf_definitions(desc, diag, opts);
        if (PULSEG_FAILED(rc))
            return rc;

        rc = check_grad_definitions(desc, diag, opts);
        if (PULSEG_FAILED(rc))
            return rc;

        rc = check_adc_definitions(desc, diag, opts);
        if (PULSEG_FAILED(rc))
            return rc;

        rc = check_base_blocks(desc, diag, opts);
        if (PULSEG_FAILED(rc))
            return rc;
    }

    return PULSEG_SUCCESS;
}
