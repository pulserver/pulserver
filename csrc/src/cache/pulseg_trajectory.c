/**
 * @file pulseg_trajectory.c
 * @brief K-space trajectory derivation and the TRAJECTORY cache section.
 *
 * Integrates the canonical gradient waveforms into k-space, slices out the
 * ADC-sampled window around each readout's k-zero anchor, and deduplicates
 * the result into a library of unique per-axis shots plus a per-ADC table of
 * shot ids, amplitudes, rotations and labels.
 *
 * Recon reads this back from the cache file alone -- it never has the .seq --
 * so the section is self-contained, rotation library included.
 */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "pulseg_internal.h"
#include "pulseg.h"

/* ================================================================== */
/*  Constants                                                         */
/* ================================================================== */

#define TRAJ_DEDUP_TOL 1e-6f

/* ================================================================== */
/*  Helpers                                                           */
/* ================================================================== */

/* Compare two k-space shot shapes; returns 1 if identical within tolerance. */
static int shots_equal(const float *a, const float *b, int n)
{
    int i;
    for (i = 0; i < n; ++i)
    {
        float d = a[i] - b[i];
        if (d > TRAJ_DEDUP_TOL || d < -TRAJ_DEDUP_TOL)
            return 0;
    }
    return 1;
}

/* Add a shot to the library if not already present.
 * Returns the shot index, or -1 on allocation failure. */
static int kshot_library_add(pulseg_kshot_library *lib, const float *k, int n)
{
    int i;
    pulseg_kshot *shot;

    /* Check for duplicate */
    for (i = 0; i < lib->num_shots; ++i)
    {
        if (lib->shots[i].num_samples == n && shots_equal(lib->shots[i].k, k, n))
            return i;
    }

    /* Grow array */
    {
        pulseg_kshot *new_shots;
        new_shots = (pulseg_kshot *)realloc(
            lib->shots,
            (size_t)(lib->num_shots + 1) * sizeof(pulseg_kshot));
        if (!new_shots)
            return -1;
        lib->shots = new_shots;
    }

    shot = &lib->shots[lib->num_shots];
    shot->num_samples = n;
    shot->k = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    if (!shot->k)
        return -1;
    memcpy(shot->k, k, (size_t)n * sizeof(float));

    return lib->num_shots++;
}

/* Analytic g(t_us) sampler for the cartesian classifier.  Evaluates
 * the gradient definition at an exact time (us, relative to block
 * start) WITHOUT going through the integrator's coarse uniform raster.
 * The raster snap can place a ramp endpoint mid-sample (e.g. flat top
 * 600us with raster 20us puts the fall transition between samples 29
 * and 30, so a sample at 600us reads ~2/3 amp instead of amp), giving
 * a false non-constant classification on plain trapezoid readouts.
 * Returns 0 outside the active gradient window.
 *
 * Mirrors testutils.TruthBuilder.sampleGradAtTimes (analytic trap +
 * linear interp ARB with zero outside). */
static float sample_grad_axis_at(
    const pulseg_sequence_descriptor *desc,
    int grad_event_id,
    const float *arb_xp,
    const float *arb_fp,
    int arb_n, /* may be NULL */
    float t_us)
{
    const pulseg_grad_definition *gd;
    float amp, ti;
    int shot_idx;

    if (grad_event_id < 0 || grad_event_id >= desc->grad_table_size)
        return 0.0f;
    amp = desc->grad_table[grad_event_id].amplitude;
    shot_idx = desc->grad_table[grad_event_id].shot_index;
    gd = &desc->grad_definitions[desc->grad_table[grad_event_id].id];
    ti = t_us - (float)gd->delay;

    if (gd->type == 0)
    {
        float r = (float)gd->rise_time_or_unused;
        float fl = (float)gd->flat_time_or_unused;
        float fa = (float)gd->fall_time_or_num_uncompressed_samples;
        if (ti <= 0.0f)
            return 0.0f;
        if (r > 0.0f && ti < r)
            return amp * (ti / r);
        if (ti < r + fl)
            return amp;
        if (fa > 0.0f && ti < r + fl + fa)
            return amp * (1.0f - (ti - r - fl) / fa);
        return 0.0f;
    }
    /* ARB: caller pre-decompressed shape into (arb_xp, arb_fp).  Linear
     * interp with zero outside [arb_xp[0], arb_xp[n-1]]. */
    (void)shot_idx;
    if (!arb_xp || !arb_fp || arb_n < 2)
        return 0.0f;
    if (t_us < arb_xp[0] || t_us > arb_xp[arb_n - 1])
        return 0.0f;
    {
        int lo = 0, hi = arb_n - 1, mid;
        float frac;
        while (hi - lo > 1)
        {
            mid = (lo + hi) >> 1;
            if (arb_xp[mid] <= t_us)
                lo = mid;
            else
                hi = mid;
        }
        if (arb_xp[hi] == arb_xp[lo])
            return arb_fp[lo];
        frac = (t_us - arb_xp[lo]) / (arb_xp[hi] - arb_xp[lo]);
        return arb_fp[lo] * (1.0f - frac) + arb_fp[hi] * frac;
    }
}

/* Decompress an ARB gradient into (xp_us, fp_hzm, n) suitable for use
 * by sample_grad_axis_at and the integrator's expand step.  Returns 1
 * if the axis is ARB and the buffers were allocated (caller frees);
 * 0 if the axis is TRAP or idle (buffers untouched); -1 on error.
 *
 * Time grid is in microseconds relative to block start (delay-shifted),
 * matching what sample_grad_axis_at expects. */
static int decompress_block_arb(
    const pulseg_sequence_descriptor *desc,
    float **out_xp,
    float **out_fp,
    int *out_n,
    int grad_event_id,
    float grad_raster_us)
{
    const pulseg_grad_definition *gd;
    pulseq_shape decomp = {0, 0, NULL};
    pulseq_shape decomp_t = {0, 0, NULL};
    float amp;
    int shot_idx, sid_one_based, sid;
    float *xp = NULL, *fp = NULL;
    int nsrc, i, rc = 0;

    *out_xp = NULL;
    *out_fp = NULL;
    *out_n = 0;
    if (grad_event_id < 0 || grad_event_id >= desc->grad_table_size)
        return 0;
    gd = &desc->grad_definitions[desc->grad_table[grad_event_id].id];
    if (gd->type == 0)
        return 0;
    amp = desc->grad_table[grad_event_id].amplitude;
    shot_idx = desc->grad_table[grad_event_id].shot_index;
    if (shot_idx < 0 || shot_idx >= PULSEG_MAX_GRAD_SHOTS)
        return 0;
    sid_one_based = gd->shot_shape_ids[shot_idx];
    if (sid_one_based <= 0)
        return 0;
    sid = sid_one_based - 1;
    if (sid < 0 || sid >= desc->num_shapes)
        return 0;

    if (!pulseq_decompress_shape(&decomp, &desc->shapes[sid], 1.0f))
        return -1;
    nsrc = decomp.num_samples;
    xp = (float *)PULSEG_ALLOC((size_t)nsrc * sizeof(float));
    fp = (float *)PULSEG_ALLOC((size_t)nsrc * sizeof(float));
    if (!xp || !fp)
    {
        rc = -1;
        goto fail;
    }

    if (gd->unused_or_time_shape_id > 0 && gd->unused_or_time_shape_id <= desc->num_shapes)
    {
        int ts_idx = gd->unused_or_time_shape_id - 1;
        if (!pulseq_decompress_shape(&decomp_t, &desc->shapes[ts_idx], grad_raster_us))
        {
            rc = -1;
            goto fail;
        }
        for (i = 0; i < nsrc && i < decomp_t.num_samples; ++i)
            xp[i] = (float)gd->delay + decomp_t.samples[i];
        for (; i < nsrc; ++i)
            xp[i] = (float)gd->delay + ((float)i + 0.5f) * grad_raster_us;
    }
    else
    {
        /* Uniform ARB: pulseq stores samples at raster centres
         * (mr.makeArbitraryGrad sets tt = ((1:N) - 0.5) * raster), so
         * sample i lives at t = delay + (i + 0.5) * raster, not i*raster. */
        for (i = 0; i < nsrc; ++i)
            xp[i] = (float)gd->delay + ((float)i + 0.5f) * grad_raster_us;
    }
    for (i = 0; i < nsrc; ++i)
        fp[i] = amp * decomp.samples[i];

    *out_xp = xp;
    *out_fp = fp;
    *out_n = nsrc;
    if (decomp.samples)
        PULSEG_FREE(decomp.samples);
    if (decomp_t.samples)
        PULSEG_FREE(decomp_t.samples);
    return 1;

fail:
    if (xp)
        PULSEG_FREE(xp);
    if (fp)
        PULSEG_FREE(fp);
    if (decomp.samples)
        PULSEG_FREE(decomp.samples);
    if (decomp_t.samples)
        PULSEG_FREE(decomp_t.samples);
    return rc;
}

/* ================================================================== */
/*  Compute trajectory for a single block (block-level k-space)       */
/* ================================================================== */

/*
 * For a single block with an ADC event:
 * 1. Get gradient waveforms at block level (uniform raster)
 * 2. Integrate to get k-space trajectory
 * 3. Crop to ADC window
 * 4. Resample to ADC dwell time
 * 5. Center to k-zero anchor
 *
 * Returns per-axis k-space arrays of length adc_num_samples.
 */
/* Sum block durations desc->block_table[num_prep .. block_table_idx-1]
 * (us) -- the time offset of block_table_idx from the start of the main
 * TR window, on the same raster the canonical k-space arrays were built
 * on (pulseg__calc_segment_timing, pulseg_structure.c). */
static float traj_block_time_offset_in_tr_us(
    const pulseg_sequence_descriptor *desc,
    int num_prep,
    int block_table_idx)
{
    float t = 0.0f;
    int i;
    for (i = num_prep; i < block_table_idx; ++i)
    {
        const pulseg_block_table_element *b = &desc->block_table[i];
        t += (b->duration_us >= 0) ? (float)b->duration_us
                                   : (float)desc->base_blocks[b->id].duration_us;
    }
    return t;
}

/* Slice + linearly resample one axis of the retained full-TR canonical
 * (ZERO_VAR) k-space array onto the ADC sample centres. NO re-centering:
 * k=0 is physically set by the excitation reset baked into the canonical
 * array (pulseg__calc_segment_timing); per-line/per-shot offsets are
 * applied recon-side from the table's gradient-amplitude metadata. */
static void traj_slice_canonical_axis(
    float *out_k,
    int adc_num_samples,
    const float *canonical_k,
    int n_samples,
    float dt_us,
    float block_time_offset_us,
    int adc_delay_us,
    float adc_dwell_us)
{
    int i, lo, hi;
    float t_us, fi, frac;

    for (i = 0; i < adc_num_samples; ++i)
    {
        t_us = block_time_offset_us + (float)adc_delay_us + ((float)i + 0.5f) * adc_dwell_us;
        fi = t_us / dt_us;
        if (fi <= 0.0f)
        {
            out_k[i] = canonical_k[0];
            continue;
        }
        lo = (int)fi;
        if (lo >= n_samples - 1)
        {
            out_k[i] = canonical_k[n_samples - 1];
            continue;
        }
        hi = lo + 1;
        frac = fi - (float)lo;
        out_k[i] = canonical_k[lo] * (1.0f - frac) + canonical_k[hi] * frac;
    }
}

static int compute_block_kspace(
    const pulseg_sequence_descriptor *desc,
    float *out_kx,
    float *out_ky,
    float *out_kz, /* [adc_num_samples] */
    int *out_num_samples,
    /* g(t) constant during ADC window?  Used by caller as the
     * cartesian-shot classifier (truth uses the same criterion).
     * Pass NULL pointers if not needed. */
    int *out_gx_const,
    int *out_gy_const,
    int *out_gz_const,
    int block_table_idx,
    int kzero_index,
    pulseg_diagnostic *diag)
{
    const pulseg_block_table_element *bte;
    const pulseg_adc_definition *adc_def;
    int adc_def_idx;
    int adc_num_samples, adc_delay_us;
    float adc_dwell_us;
    float grad_raster_us;
    int num_prep, tr_size, pos_in_tr;
    int i;

    (void)kzero_index; /* no longer used -- canonical array carries its own k=0 */

    bte = &desc->block_table[block_table_idx];
    /* block_table[].adc_id holds the RAW seq ADC index (per-instance);
     * the deduped index into desc->adc_definitions[] lives in
     * base_blocks[bte->id].adc_id. */
    adc_def_idx = desc->base_blocks[bte->id].adc_id;
    if (adc_def_idx < 0 || adc_def_idx >= desc->num_unique_adcs)
    {
        if (diag)
            sprintf(diag->message, "Block %d has no ADC event", block_table_idx);
        return PULSEG_ERR_INVALID_ARGUMENT;
    }

    adc_def = &desc->adc_definitions[adc_def_idx];
    adc_num_samples = adc_def->num_samples;
    adc_dwell_us = (float)adc_def->dwell_time * 1e-3f; /* ns -> us */
    adc_delay_us = adc_def->delay;

    grad_raster_us = desc->grad_raster_us;

    /* ---- Classify each axis: constant g(t) during the ADC window? ----
     * Sampled analytically (TRAP exact, ARB via decompressed shape) --
     * matches the truth integrator's criterion. Used by the caller as the
     * cartesian-shot (-1) classifier; unrelated to the k VALUE computation
     * below (which now slices the retained canonical array). */
    {
        /* Per-axis breakpoint sources (ARB only; TRAP is analytic). */
        float *bp_x = NULL, *fpx = NULL;
        int nx = 0;
        float *bp_y = NULL, *fpy = NULL;
        int ny = 0;
        float *bp_z = NULL, *fpz = NULL;
        int nz = 0;

        int rx = decompress_block_arb(desc, &bp_x, &fpx, &nx, bte->gx_id, grad_raster_us);
        int ry = decompress_block_arb(desc, &bp_y, &fpy, &ny, bte->gy_id, grad_raster_us);
        int rz = decompress_block_arb(desc, &bp_z, &fpz, &nz, bte->gz_id, grad_raster_us);
        (void)rx;
        (void)ry;
        (void)rz;

        /* Per-axis cartesian classifier: g(t) constant across the
         * ACTIVE ADC window?  Sample g ANALYTICALLY at t = adc.delay +
         * (i + 0.5) * dwell.  Mirrors TruthBuilder.exportTrajectory. */
        {
            float xmin = 0, xmax = 0, ymin = 0, ymax = 0, zmin = 0, zmax = 0;
            float xs, ys, zs;
            int gxc, gyc, gzc;
            int j;
            float t_us, gv;
            int first = 1;
            for (j = 0; j < adc_num_samples; ++j)
            {
                t_us = (float)adc_delay_us + ((float)j + 0.5f) * adc_dwell_us;
                gv = sample_grad_axis_at(desc, bte->gx_id, bp_x, fpx, nx, t_us);
                if (first || gv < xmin)
                    xmin = gv;
                if (first || gv > xmax)
                    xmax = gv;
                gv = sample_grad_axis_at(desc, bte->gy_id, bp_y, fpy, ny, t_us);
                if (first || gv < ymin)
                    ymin = gv;
                if (first || gv > ymax)
                    ymax = gv;
                gv = sample_grad_axis_at(desc, bte->gz_id, bp_z, fpz, nz, t_us);
                if (first || gv < zmin)
                    zmin = gv;
                if (first || gv > zmax)
                    zmax = gv;
                first = 0;
            }
            xs = (xmax > -xmin) ? xmax : -xmin;
            if (xs < 0)
                xs = -xs;
            ys = (ymax > -ymin) ? ymax : -ymin;
            if (ys < 0)
                ys = -ys;
            zs = (zmax > -zmin) ? zmax : -zmin;
            if (zs < 0)
                zs = -zs;
            gxc = (xs < 1e-9f) ? 1 : ((xmax - xmin) / xs < 1e-3f);
            gyc = (ys < 1e-9f) ? 1 : ((ymax - ymin) / ys < 1e-3f);
            gzc = (zs < 1e-9f) ? 1 : ((zmax - zmin) / zs < 1e-3f);
            /* Radial classifier: when this block has a
             * rotation, never collapse an active axis to cartesian (-1) --
             * the rotated frame needs the real shot. Inactive (absent)
             * axes stay cartesian regardless (g(t) is identically zero). */
            if (bte->rotation_id >= 0)
            {
                if (bte->gx_id >= 0)
                    gxc = 0;
                if (bte->gy_id >= 0)
                    gyc = 0;
                if (bte->gz_id >= 0)
                    gzc = 0;
            }
            if (getenv("PULSEG_TRAJ_DEBUG"))
            {
                fprintf(
                    stderr,
                    "  cls blk=%d xmin=%.1f xmax=%.1f xs=%.1f gxc=%d  ymin=%.1f ymax=%.1f gyc=%d\n",
                    block_table_idx,
                    xmin,
                    xmax,
                    xs,
                    gxc,
                    ymin,
                    ymax,
                    gyc);
            }
            if (out_gx_const)
                *out_gx_const = gxc;
            if (out_gy_const)
                *out_gy_const = gyc;
            if (out_gz_const)
                *out_gz_const = gzc;
        }

        if (bp_x)
            PULSEG_FREE(bp_x);
        if (fpx)
            PULSEG_FREE(fpx);
        if (bp_y)
            PULSEG_FREE(bp_y);
        if (fpy)
            PULSEG_FREE(fpy);
        if (bp_z)
            PULSEG_FREE(bp_z);
        if (fpz)
            PULSEG_FREE(fpz);
    }

    /* ---- k VALUE computation: slice the retained full-TR canonical
     * (ZERO_VAR) array instead of re-integrating this block's actual
     * gradient waveform. NO re-centering -- k=0 is already
     * physically set by the excitation reset baked into the canonical
     * array by pulseg__calc_segment_timing. ----
     * Only valid for blocks inside the main TR window (num_prep ..
     * num_prep+tr_size); prep/cooldown blocks fall back to all-zero,
     * mirroring the pre-existing seg_time_offset=0 degradation for
     * out-of-window segments in calc_segment_timing. */
    num_prep = desc->tr_descriptor.num_prep_blocks;
    tr_size = desc->tr_descriptor.tr_size;
    pos_in_tr = block_table_idx - num_prep;

    /* The canonical array covers exactly one tr_size-block window. When the
     * main TR pattern repeats back-to-back within a single pass (num_trs > 1
     * -- e.g. a degenerate-prep sequence that plays one identical
     * prep+readout+navigator unit per slice, with slice as the outer loop),
     * wrap any repeat's position back into that window so repeat #2, #3, ...
     * reuse the same canonical samples instead of falling "out of window"
     * and silently degrading to an all-zero trajectory. Blocks beyond the
     * last repeat (real cooldown) are deliberately left unwrapped, so they
     * still degrade as documented below. */
    if (tr_size > 0 && pos_in_tr >= 0)
    {
        int num_trs = (desc->tr_descriptor.num_trs > 0) ? desc->tr_descriptor.num_trs : 1;
        int repeat_span = tr_size * num_trs;
        if (pos_in_tr < repeat_span)
            pos_in_tr = pos_in_tr % tr_size;
    }

    if (desc->has_canonical_kspace && pos_in_tr >= 0 && pos_in_tr < tr_size)
    {
        int local_block_table_idx = num_prep + pos_in_tr;
        float block_time_offset_us =
            traj_block_time_offset_in_tr_us(desc, num_prep, local_block_table_idx);

        traj_slice_canonical_axis(
            out_kx,
            adc_num_samples,
            desc->canonical_kx,
            desc->canonical_kspace_num_samples,
            desc->canonical_kspace_dt_us,
            block_time_offset_us,
            adc_delay_us,
            adc_dwell_us);
        traj_slice_canonical_axis(
            out_ky,
            adc_num_samples,
            desc->canonical_ky,
            desc->canonical_kspace_num_samples,
            desc->canonical_kspace_dt_us,
            block_time_offset_us,
            adc_delay_us,
            adc_dwell_us);
        traj_slice_canonical_axis(
            out_kz,
            adc_num_samples,
            desc->canonical_kz,
            desc->canonical_kspace_num_samples,
            desc->canonical_kspace_dt_us,
            block_time_offset_us,
            adc_delay_us,
            adc_dwell_us);
    }
    else
    {
        for (i = 0; i < adc_num_samples; ++i)
        {
            out_kx[i] = 0.0f;
            out_ky[i] = 0.0f;
            out_kz[i] = 0.0f;
        }
    }

    *out_num_samples = adc_num_samples;

    return PULSEG_SUCCESS;
}

/* Assigns a 3-column vendor-mapped label-table value (D3) into the named
 * field it represents. state_idx is a Pulseq label state-array index
 * (0=SLC,1=PHS,2=REP,3=AVG,4=SEG,5=SET,6=ECO,7=PAR,8=LIN,9=ACQ), taken
 * from desc->label_column_map -- not hardcoded to GE's [lin,slc,eco]. */
static void assign_traj_label_by_state_index(
    pulseg_traj_table_entry *entry,
    int state_idx,
    int value)
{
    switch (state_idx)
    {
    case 0:
        entry->slc = value;
        break;
    case 1:
        entry->phs = value;
        break;
    case 2:
        entry->rep = value;
        break;
    case 3:
        entry->avg = value;
        break;
    case 4:
        entry->seg = value;
        break;
    case 5:
        entry->set = value;
        break;
    case 6:
        entry->eco = value;
        break;
    case 7:
        entry->par = value;
        break;
    case 8:
        entry->lin = value;
        break;
    case 9:
        entry->acq = value;
        break;
    default:
        break;
    }
}

/* ================================================================== */
/*  Public: compute trajectory                                        */
/* ================================================================== */

static int pulseg_compute_trajectory(
    const pulseg_collection *coll,
    pulseg_trajectory *out,
    pulseg_diagnostic *diag,
    int subseq_idx)
{
    const pulseg_sequence_descriptor *desc;
    int n, b;
    int num_adc_events;
    int adc_idx;
    int label_ncols;
    int *label_buf = NULL;
    pulseg_traj_table_entry *table = NULL;
    float *kx_buf = NULL;
    float *ky_buf = NULL;
    float *kz_buf = NULL;
    int max_adc_samples;
    /* Memoization: per-block_table-idx cached shot IDs and kzero so we
     * skip redundant compute_block_kspace calls when the same block is
     * referenced multiple times by the scan table (e.g. an EPI readout
     * block that recurs once per phase-encode line).  Cache is keyed by
     * (b, kzero); -2 means "not computed yet". */
    int *cached_kx_id = NULL;
    int *cached_ky_id = NULL;
    int *cached_kz_id = NULL;
    int *cached_kzero = NULL;
    int rc;

    if (!coll || !out)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    memset(out, 0, sizeof(*out));
    desc = &coll->descriptors[subseq_idx];

    /* Count ADC events from scan table.
     * Gate by the PER-INSTANCE block_table[b].adc_id (-1 for dummy ADC
     * placeholders such as mr.makeDelay used to skip acquisition).
     * base_blocks[].adc_id is the deduped adc_def index and is
     * shared between dummy and real instances when block dedup ignores
     * the ADC slot, so it over-counts here. The deduped index is still
     * used downstream for the adc_definitions[] lookup. */
    num_adc_events = 0;
    for (n = 0; n < desc->exec_stream_len; ++n)
    {
        b = desc->exec_stream_block_idx[n];
        if (desc->block_table[b].adc_id >= 0)
            ++num_adc_events;
    }

    if (num_adc_events == 0)
    {
        /* No ADC events — trajectory is empty */
        return PULSEG_SUCCESS;
    }

    /* Find max ADC sample count */
    max_adc_samples = 0;
    {
        int a;
        for (a = 0; a < desc->num_unique_adcs; ++a)
        {
            if (desc->adc_definitions[a].num_samples > max_adc_samples)
                max_adc_samples = desc->adc_definitions[a].num_samples;
        }
    }

    /* Allocate work buffers */
    kx_buf = (float *)PULSEG_ALLOC((size_t)max_adc_samples * sizeof(float));
    ky_buf = (float *)PULSEG_ALLOC((size_t)max_adc_samples * sizeof(float));
    kz_buf = (float *)PULSEG_ALLOC((size_t)max_adc_samples * sizeof(float));
    label_ncols = desc->label_num_columns;
    if (label_ncols > 0)
        label_buf = (int *)PULSEG_ALLOC((size_t)label_ncols * sizeof(int));
    table = (pulseg_traj_table_entry *)PULSEG_ALLOC(
        (size_t)num_adc_events * sizeof(pulseg_traj_table_entry));

    if (!kx_buf || !ky_buf || !kz_buf || !table)
        goto compute_fail;

    /* Allocate memoization cache and seed with sentinel -2 = uncomputed. */
    if (desc->num_blocks > 0)
    {
        int ci;
        cached_kx_id = (int *)PULSEG_ALLOC((size_t)desc->num_blocks * sizeof(int));
        cached_ky_id = (int *)PULSEG_ALLOC((size_t)desc->num_blocks * sizeof(int));
        cached_kz_id = (int *)PULSEG_ALLOC((size_t)desc->num_blocks * sizeof(int));
        cached_kzero = (int *)PULSEG_ALLOC((size_t)desc->num_blocks * sizeof(int));
        if (!cached_kx_id || !cached_ky_id || !cached_kz_id || !cached_kzero)
            goto compute_fail;
        for (ci = 0; ci < desc->num_blocks; ++ci)
        {
            cached_kx_id[ci] = -2;
            cached_ky_id[ci] = -2;
            cached_kz_id[ci] = -2;
            cached_kzero[ci] = -1;
        }
    }

    memset(table, 0, (size_t)num_adc_events * sizeof(pulseg_traj_table_entry));

    /* Initialize kshot library */
    out->kshots.num_shots = 0;
    out->kshots.shots = NULL;

    /* Iterate ADC scan-table entries.  Trajectory shape depends only
     * on the base block definition (which is content-deduped on
     * (gx_id, gy_id, gz_id, adc_id, duration)), so we memoize the
     * computed shot IDs by block_def index (bte->id) to avoid redoing
     * the same cumsum for every recurrence of the readout block.  The
     * kzero anchor is per-occurrence (segment-derived); if a second
     * occurrence of the same block_def uses a different kzero we
     * recompute. */
    {
        adc_idx = 0;
        for (n = 0; n < desc->exec_stream_len; ++n)
        {
            const pulseg_block_table_element *bte;
            int adc_def_idx, adc_nsamples, kzero;
            int kx_id, ky_id, kz_id;

            b = desc->exec_stream_block_idx[n];
            bte = &desc->block_table[b];

            if (bte->adc_id < 0)
                continue;
            adc_def_idx = desc->base_blocks[bte->id].adc_id;
            if (adc_def_idx < 0)
                continue;

            adc_nsamples = desc->adc_definitions[adc_def_idx].num_samples;

            /* Find k-zero index from segment timing.
             * Look up which segment this block belongs to and find the
             * ADC anchor with matching block offset. */
            kzero = adc_nsamples / 2; /* default: center */
            {
                int seg_idx = desc->exec_stream_seg_id[n];
                if (seg_idx >= 0 && seg_idx < desc->num_unique_segments)
                {
                    const pulseg_virtual_segment *seg_def = &desc->segment_definitions[seg_idx];
                    const pulseg_segment_timing *tim = &seg_def->timing;
                    int a;
                    for (a = 0; a < tim->num_adc_anchors; ++a)
                    {
                        if (tim->adc_anchors[a].block_offset == (b - seg_def->start_block))
                        {
                            kzero = tim->adc_anchors[a].kzero_index;
                            break;
                        }
                    }
                }
            }

            /* Memo lookup keyed by block_table index + kzero.
             * (Not bte->id: base_blocks are deduped on the
             * underlying gradient definition ids — shape only — so two
             * scans with different per-slice rotations or amplitudes
             * share bte->id but produce different physical waveforms.
             * Block-table indices are unique per scan-table row;
             * cross-block content equivalence is recovered by
             * kshot_library_add which dedups identical k arrays.) */
            if (cached_kx_id && b >= 0 && b < desc->num_blocks && cached_kx_id[b] != -2 &&
                cached_kzero[b] == kzero)
            {
                kx_id = cached_kx_id[b];
                ky_id = cached_ky_id[b];
                kz_id = cached_kz_id[b];
            }
            else
            {
                int gx_const = 0, gy_const = 0, gz_const = 0;
                rc = compute_block_kspace(
                    desc,
                    kx_buf,
                    ky_buf,
                    kz_buf,
                    &adc_nsamples,
                    &gx_const,
                    &gy_const,
                    &gz_const,
                    b,
                    kzero,
                    diag);
                if (PULSEG_FAILED(rc))
                    goto compute_fail;

                /* Per-axis dedup into kshot library.  An axis is
                 * cartesian (kshot_id=-1) when g(t) is constant during
                 * the active ADC window; the mrdserver reconstructs
                 * coordinates from the gradient amplitude metadata
                 * (and applies any rotation) without needing a kshot. */
                if (gx_const)
                {
                    kx_id = -1;
                }
                else
                {
                    kx_id = kshot_library_add(&out->kshots, kx_buf, adc_nsamples);
                    if (kx_id < 0)
                        goto compute_fail;
                }
                if (gy_const)
                {
                    ky_id = -1;
                }
                else
                {
                    ky_id = kshot_library_add(&out->kshots, ky_buf, adc_nsamples);
                    if (ky_id < 0)
                        goto compute_fail;
                }
                if (gz_const)
                {
                    kz_id = -1;
                }
                else
                {
                    kz_id = kshot_library_add(&out->kshots, kz_buf, adc_nsamples);
                    if (kz_id < 0)
                        goto compute_fail;
                }

                if (cached_kx_id && b >= 0 && b < desc->num_blocks)
                {
                    cached_kx_id[b] = kx_id;
                    cached_ky_id[b] = ky_id;
                    cached_kz_id[b] = kz_id;
                    cached_kzero[b] = kzero;
                }
            }

            /* Populate table entry */
            table[adc_idx].kx_shot_id = kx_id;
            table[adc_idx].ky_shot_id = ky_id;
            table[adc_idx].kz_shot_id = kz_id;

            /* Gradient amplitudes */
            table[adc_idx].gx_amplitude = (bte->gx_id >= 0 && bte->gx_id < desc->grad_table_size)
                ? desc->grad_table[bte->gx_id].amplitude
                : 0.0f;
            table[adc_idx].gy_amplitude = (bte->gy_id >= 0 && bte->gy_id < desc->grad_table_size)
                ? desc->grad_table[bte->gy_id].amplitude
                : 0.0f;
            table[adc_idx].gz_amplitude = (bte->gz_id >= 0 && bte->gz_id < desc->grad_table_size)
                ? desc->grad_table[bte->gz_id].amplitude
                : 0.0f;

            /* Rotation */
            table[adc_idx].rotation_id = bte->rotation_id;

            /* Metadata: center_sample, sample_time */
            table[adc_idx].center_sample = kzero;
            table[adc_idx].sample_time_us =
                (float)desc->adc_definitions[adc_def_idx].dwell_time * 1e-3f;
            table[adc_idx].flags = 0;
            table[adc_idx].off = (desc->off_table && adc_idx < desc->label_num_entries)
                ? (desc->off_table[adc_idx] ? 1 : 0)
                : 0;

            /* Labels from label table */
            if (label_buf && adc_idx < desc->label_num_entries)
            {
                rc = pulseg_get_adc_label(coll, label_buf, subseq_idx, adc_idx);
                if (PULSEG_SUCCEEDED(rc) && label_ncols >= 10)
                {
                    /* Full Pulseq label mapping: col order matches label_table columns */
                    table[adc_idx].lin = label_buf[0];
                    table[adc_idx].slc = label_buf[1];
                    table[adc_idx].eco = label_buf[2];
                    table[adc_idx].rep = label_buf[3];
                    table[adc_idx].phs = label_buf[4];
                    table[adc_idx].set = label_buf[5];
                    table[adc_idx].seg = label_buf[6];
                    table[adc_idx].avg = label_buf[7];
                    table[adc_idx].par = label_buf[8];
                    table[adc_idx].acq = label_buf[9];

                    /* Override rep with the actual average (NEX) loop index from
                     * the scan table; the seqfile label table cannot know NEX. */
                    if (desc->exec_stream_avg_id)
                        table[adc_idx].rep = desc->exec_stream_avg_id[n];

                    /* Map Pulseq boolean flags to ISMRMRD flag bits
                     * Column indices (0-based) match PULSEG__* macros - 1.
                     * Flags 1-18 (FIRST/LAST) are NOT set here; they are computed
                     * per-encoding-space in the enrichment layer using actual min/max. */
                    if (label_ncols > 10 && label_buf[10])   /* NAV  col11 */
                        table[adc_idx].flags |= (1UL << 22); /* ACQ_IS_NAVIGATION_DATA (23) */
                    if (label_ncols > 11 && label_buf[11])   /* REV  col12 */
                        table[adc_idx].flags |= (1UL << 21); /* ACQ_IS_REVERSE (22) */
                    if (label_ncols > 13 && label_buf[13])   /* REF  col14 */
                        table[adc_idx].flags |= (1UL << 19); /* ACQ_IS_PARALLEL_CALIBRATION (20) */
                    if (label_ncols > 14 && label_buf[14])   /* IMA  col15 */
                        table[adc_idx].flags |=
                            (1UL << 20); /* ACQ_IS_PARALLEL_CALIBRATION_AND_IMAGING (21) */
                    if (label_ncols > 15 && label_buf[15])   /* NOISE col16 */
                        table[adc_idx].flags |= (1UL << 18); /* ACQ_IS_NOISE_MEASUREMENT (19) */
                }
                else if (PULSEG_SUCCEEDED(rc) && label_ncols >= 3)
                {
                    /* 3-column vendor-mapped label table (D3): column
                     * meaning comes from desc->label_column_map (GE
                     * default [lin,slc,eco]), not a hardcoded GEHC order. */
                    int col;
                    for (col = 0; col < 3; ++col)
                        assign_traj_label_by_state_index(
                            &table[adc_idx],
                            desc->label_column_map[col],
                            label_buf[col]);
                }
            }

            /* NAV flag from block table (block-level nav_flag) */
            if (bte->nav_flag)
                table[adc_idx].flags |= (1UL << 22); /* ACQ_IS_NAVIGATION_DATA */

            /* Encoding space ref: local index 0 = normal, 1 = navigator */
            table[adc_idx].encoding_space_ref = (table[adc_idx].flags & (1UL << 22)) ? 1 : 0;

            ++adc_idx;
        }
    }

    /* FIRST_IN / LAST_IN flags (bits 0-17, ISMRMRD flags 1-18) are computed
     * per-encoding-space in the enrichment layer (enrich_ismrmrd_acquisition)
     * using actual label_limits min/max.  Only LAST_IN_MEASUREMENT is set here
     * since it is a scan-global property known at cache build time. */
    if (adc_idx > 0)
        table[adc_idx - 1].flags |= (1UL << 24); /* LAST_IN_MEASUREMENT (25) */

    out->num_adc_events = adc_idx;
    out->table = table;
    /* keep `table` valid until the navigator/label-limits passes below
     * have finished reading it; nulled at end-of-function for cleanup. */

    /* Detect whether any ADC carries the navigator flag.
     * Note: read via out->table since the local `table` was nulled above
     * after ownership transfer. */
    {
        int has_nav = 0, num_es_local, es, i;
        for (i = 0; i < adc_idx; ++i)
        {
            if (out->table[i].flags & (1UL << 22))
            {
                has_nav = 1;
                break;
            }
        }
        num_es_local = has_nav ? 2 : 1;

        out->num_encoding_spaces = num_es_local;
        out->encoding_spaces = (pulseg_encoding_space *)PULSEG_ALLOC(
            (size_t)num_es_local * sizeof(pulseg_encoding_space));
        if (!out->encoding_spaces)
            goto compute_fail;
        memset(out->encoding_spaces, 0, (size_t)num_es_local * sizeof(pulseg_encoding_space));

        /* ES 0: normal scans. Geometry (fov/matrix) is no longer embedded
         * here -- the recon reads it from DEFINITIONS(0) by subseq_idx
         *. */
        out->encoding_spaces[0].subseq_idx = subseq_idx;
        out->encoding_spaces[0].nav_subseq_offset = has_nav ? 1 : 0;
        out->encoding_spaces[0].geometry_tag = 0;

        /* ES 1 (if nav): navigator scans -- DEFINITIONS' NavFOV/NavMatrix
         * apply (geometry_tag == 1). */
        if (has_nav)
        {
            out->encoding_spaces[1].subseq_idx = subseq_idx;
            out->encoding_spaces[1].nav_subseq_offset = 0;
            out->encoding_spaces[1].geometry_tag = 1;
        }

        /* Compute per-encoding-space label_limits from table entries */
        for (es = 0; es < num_es_local; ++es)
        {
            pulseg_label_limits *ll = &out->encoding_spaces[es].label_limits;
            int first = 1;
            for (i = 0; i < adc_idx; ++i)
            {
                if (table[i].encoding_space_ref != es)
                    continue;
                if (first)
                {
                    ll->slc.min = ll->slc.max = table[i].slc;
                    ll->phs.min = ll->phs.max = table[i].phs;
                    ll->rep.min = ll->rep.max = table[i].rep;
                    ll->avg.min = ll->avg.max = table[i].avg;
                    ll->seg.min = ll->seg.max = table[i].seg;
                    ll->set.min = ll->set.max = table[i].set;
                    ll->eco.min = ll->eco.max = table[i].eco;
                    ll->par.min = ll->par.max = table[i].par;
                    ll->lin.min = ll->lin.max = table[i].lin;
                    ll->acq.min = ll->acq.max = table[i].acq;
                    first = 0;
                }
                else
                {
#define LLUP(fld) \
    do \
    { \
        if (table[i].fld < ll->fld.min) \
            ll->fld.min = table[i].fld; \
        if (table[i].fld > ll->fld.max) \
            ll->fld.max = table[i].fld; \
    } while (0)
                    LLUP(slc);
                    LLUP(phs);
                    LLUP(rep);
                    LLUP(avg);
                    LLUP(seg);
                    LLUP(set);
                    LLUP(eco);
                    LLUP(par);
                    LLUP(lin);
                    LLUP(acq);
#undef LLUP
                }
            }
        }
    }

    /* copy this subsequence's rotation-matrix library onto the
     * trajectory itself (table[].rotation_id already indexes it directly --
     * no offsetting needed here; pulseg_merge_trajectory offsets it when
     * appending a second subsequence's trajectory). */
    if (desc->num_rotations > 0 && desc->rotation_matrices)
    {
        out->rotation_matrices =
            (float(*)[9])PULSEG_ALLOC((size_t)desc->num_rotations * sizeof(float[9]));
        if (!out->rotation_matrices)
            goto compute_fail;
        memcpy(
            out->rotation_matrices,
            desc->rotation_matrices,
            (size_t)desc->num_rotations * sizeof(float[9]));
        out->num_rotations = desc->num_rotations;
    }

    PULSEG_FREE(kx_buf);
    PULSEG_FREE(ky_buf);
    PULSEG_FREE(kz_buf);
    PULSEG_FREE(label_buf);
    PULSEG_FREE(cached_kx_id);
    PULSEG_FREE(cached_ky_id);
    PULSEG_FREE(cached_kz_id);
    PULSEG_FREE(cached_kzero);
    return PULSEG_SUCCESS;

compute_fail:
    PULSEG_FREE(kx_buf);
    PULSEG_FREE(ky_buf);
    PULSEG_FREE(kz_buf);
    PULSEG_FREE(label_buf);
    PULSEG_FREE(cached_kx_id);
    PULSEG_FREE(cached_ky_id);
    PULSEG_FREE(cached_kz_id);
    PULSEG_FREE(cached_kzero);
    /* If table ownership was already transferred to `out`, avoid double-free. */
    if (out && out->table == table)
        table = NULL;
    PULSEG_FREE(table);
    pulseg_trajectory_free(out);
    return PULSEG_ERR_ALLOC_FAILED;
}

/* ================================================================== */
/*  Free trajectory                                                   */
/* ================================================================== */

void pulseg_trajectory_free(pulseg_trajectory *traj)
{
    int i;
    if (!traj)
        return;

    if (traj->kshots.shots)
    {
        for (i = 0; i < traj->kshots.num_shots; ++i)
            PULSEG_FREE(traj->kshots.shots[i].k);
        free(traj->kshots.shots);
    }
    traj->kshots.shots = NULL;
    traj->kshots.num_shots = 0;

    PULSEG_FREE(traj->encoding_spaces);
    traj->encoding_spaces = NULL;
    traj->num_encoding_spaces = 0;

    PULSEG_FREE(traj->table);
    traj->table = NULL;
    traj->num_adc_events = 0;

    PULSEG_FREE(traj->rotation_matrices);
    traj->rotation_matrices = NULL;
    traj->num_rotations = 0;
}

/* ================================================================== */
/*  Merge trajectory (append src into dst)                            */
/* ================================================================== */

static int pulseg_merge_trajectory(pulseg_trajectory *dst, const pulseg_trajectory *src)
{
    int kshot_offset, es_offset, rotation_offset, i;

    if (!dst || !src)
        return PULSEG_ERR_NULL_POINTER;

    kshot_offset = dst->kshots.num_shots;
    es_offset = dst->num_encoding_spaces;
    rotation_offset = dst->num_rotations;

    /* ---- Append rotation-matrix library ---- */
    if (src->num_rotations > 0)
    {
        int new_count = rotation_offset + src->num_rotations;
        float(*new_rot)[9] =
            (float(*)[9])realloc(dst->rotation_matrices, (size_t)new_count * sizeof(float[9]));
        if (!new_rot)
            return PULSEG_ERR_ALLOC_FAILED;
        memcpy(
            new_rot[rotation_offset],
            src->rotation_matrices,
            (size_t)src->num_rotations * sizeof(float[9]));
        dst->rotation_matrices = new_rot;
        dst->num_rotations = new_count;
    }

    /* ---- Append kshots ---- */
    if (src->kshots.num_shots > 0)
    {
        int new_count = kshot_offset + src->kshots.num_shots;
        pulseg_kshot *new_shots =
            (pulseg_kshot *)realloc(dst->kshots.shots, (size_t)new_count * sizeof(pulseg_kshot));
        if (!new_shots)
            return PULSEG_ERR_ALLOC_FAILED;
        dst->kshots.shots = new_shots;
        for (i = 0; i < src->kshots.num_shots; ++i)
        {
            pulseg_kshot *d = &dst->kshots.shots[kshot_offset + i];
            const pulseg_kshot *s = &src->kshots.shots[i];
            d->num_samples = s->num_samples;
            d->k = (float *)PULSEG_ALLOC((size_t)s->num_samples * sizeof(float));
            if (!d->k)
                return PULSEG_ERR_ALLOC_FAILED;
            memcpy(d->k, s->k, (size_t)s->num_samples * sizeof(float));
        }
        dst->kshots.num_shots = new_count;
    }

    /* ---- Append encoding spaces ---- */
    if (src->num_encoding_spaces > 0)
    {
        int new_count = es_offset + src->num_encoding_spaces;
        pulseg_encoding_space *new_es = (pulseg_encoding_space *)realloc(
            dst->encoding_spaces,
            (size_t)new_count * sizeof(pulseg_encoding_space));
        if (!new_es)
            return PULSEG_ERR_ALLOC_FAILED;
        dst->encoding_spaces = new_es;
        memcpy(
            &dst->encoding_spaces[es_offset],
            src->encoding_spaces,
            (size_t)src->num_encoding_spaces * sizeof(pulseg_encoding_space));
        dst->num_encoding_spaces = new_count;
    }

    /* ---- Append table entries (adjust kshot IDs + encoding_space_ref) ---- */
    if (src->num_adc_events > 0)
    {
        int old_count = dst->num_adc_events;
        int new_count = old_count + src->num_adc_events;
        pulseg_traj_table_entry *new_table = (pulseg_traj_table_entry *)realloc(
            dst->table,
            (size_t)new_count * sizeof(pulseg_traj_table_entry));
        if (!new_table)
            return PULSEG_ERR_ALLOC_FAILED;
        dst->table = new_table;
        memcpy(
            &dst->table[old_count],
            src->table,
            (size_t)src->num_adc_events * sizeof(pulseg_traj_table_entry));

        for (i = 0; i < src->num_adc_events; ++i)
        {
            pulseg_traj_table_entry *e = &dst->table[old_count + i];
            if (e->kx_shot_id >= 0)
                e->kx_shot_id += kshot_offset;
            if (e->ky_shot_id >= 0)
                e->ky_shot_id += kshot_offset;
            if (e->kz_shot_id >= 0)
                e->kz_shot_id += kshot_offset;
            e->encoding_space_ref += es_offset;
            if (e->rotation_id >= 0)
                e->rotation_id += rotation_offset;
        }
        dst->num_adc_events = new_count;
    }

    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Write trajectory cache (TRAJECTORY section)                       */
/* ================================================================== */

#define CACHE_ENDIAN_MARKER 0x01020304
#define CACHE_SECTION_TRAJECTORY 6

static int pulseg_write_trajectory_cache(
    const pulseg_trajectory *traj,
    const char *seq_path,
    const char *cache_ext)
{
    char *cache_path;
    FILE *f;
    int marker, num_sections;
    int version_major, version_minor, version_revision, vendor, stored_size;
    int do_swap;
    long entries_pos, data_start, data_end, hdr_ns_pos;
    int i, found_idx;
    int entries_buf[16 * 3]; /* up to 16 sections x 3 ints each */

    if (!traj || !seq_path)
        return PULSEG_ERR_NULL_POINTER;

    cache_path = pulseg__make_cache_path(seq_path, cache_ext);
    if (!cache_path)
        return PULSEG_ERR_ALLOC_FAILED;

    f = fopen(cache_path, "r+b");
    if (!f)
    {
        PULSEG_FREE(cache_path);
        return PULSEG_ERR_FILE_READ_FAILED;
    }

    /* Read header */
    if (!pulseg__read4(f, &marker, 1))
        goto tw_fail;
    do_swap = 0;
    if (marker != CACHE_ENDIAN_MARKER)
    {
        pulseg__swap4(&marker);
        if (marker != CACHE_ENDIAN_MARKER)
            goto tw_fail;
        do_swap = 1;
    }
    if (!pulseg__read4(f, &version_major, 1))
        goto tw_fail;
    if (!pulseg__read4(f, &version_minor, 1))
        goto tw_fail;
    if (!pulseg__read4(f, &version_revision, 1))
        goto tw_fail;
    if (!pulseg__read4(f, &vendor, 1))
        goto tw_fail;
    if (!pulseg__read4(f, &stored_size, 1))
        goto tw_fail;
    hdr_ns_pos = ftell(f);
    if (!pulseg__read4(f, &num_sections, 1))
        goto tw_fail;
    if (do_swap)
    {
        pulseg__swap4(&version_major);
        pulseg__swap4(&version_minor);
        pulseg__swap4(&version_revision);
        pulseg__swap4(&vendor);
        pulseg__swap4(&stored_size);
        pulseg__swap4(&num_sections);
    }
    if (num_sections <= 0 || num_sections > 15)
        goto tw_fail;

    entries_pos = ftell(f);
    if (entries_pos < 0)
        goto tw_fail;

    /* Read existing section entries */
    for (i = 0; i < num_sections; ++i)
    {
        if (!pulseg__read4(f, &entries_buf[i * 3], 3))
            goto tw_fail;
        if (do_swap)
            pulseg__swap4_array(&entries_buf[i * 3], 3);
    }

    /* Check if the trajectory section already exists */
    found_idx = -1;
    for (i = 0; i < num_sections; ++i)
    {
        if (entries_buf[i * 3] == CACHE_SECTION_TRAJECTORY)
        {
            found_idx = i;
            break;
        }
    }
    if (found_idx < 0)
    {
        found_idx = num_sections;
        entries_buf[found_idx * 3] = CACHE_SECTION_TRAJECTORY;
        num_sections++;
    }

    /* Seek to end, write trajectory data */
    fseek(f, 0, SEEK_END);
    data_start = ftell(f);
    if (data_start < 0)
        goto tw_fail;

    /* Write kshot library */
    if (!pulseg__write4(f, &traj->kshots.num_shots, 1))
        goto tw_fail;
    for (i = 0; i < traj->kshots.num_shots; ++i)
    {
        if (!pulseg__write4(f, &traj->kshots.shots[i].num_samples, 1))
            goto tw_fail;
        if (traj->kshots.shots[i].num_samples > 0)
        {
            if (!pulseg__write4(f, traj->kshots.shots[i].k, traj->kshots.shots[i].num_samples))
                goto tw_fail;
        }
    }

    /* Write encoding spaces */
    if (!pulseg__write4(f, &traj->num_encoding_spaces, 1))
        goto tw_fail;
    for (i = 0; i < traj->num_encoding_spaces; ++i)
    {
        const pulseg_encoding_space *es = &traj->encoding_spaces[i];
        if (!pulseg__write4(f, &es->subseq_idx, 1))
            goto tw_fail;
        if (!pulseg__write4(f, &es->nav_subseq_offset, 1))
            goto tw_fail;
        if (!pulseg__write4(f, &es->geometry_tag, 1))
            goto tw_fail;
        if (!pulseg__write4(f, &es->label_limits, sizeof(pulseg_label_limits) / sizeof(int)))
            goto tw_fail;
    }

    /* Write trajectory table */
    if (!pulseg__write4(f, &traj->num_adc_events, 1))
        goto tw_fail;
    for (i = 0; i < traj->num_adc_events; ++i)
    {
        const pulseg_traj_table_entry *e = &traj->table[i];
        /* Write as contiguous 17 ints/floats:
         * 3 shot_ids + 3 amplitudes + rotation_id + 10 labels */
        if (!pulseg__write4(f, &e->kx_shot_id, 1))
            goto tw_fail;
        if (!pulseg__write4(f, &e->ky_shot_id, 1))
            goto tw_fail;
        if (!pulseg__write4(f, &e->kz_shot_id, 1))
            goto tw_fail;
        if (!pulseg__write4(f, &e->gx_amplitude, 1))
            goto tw_fail;
        if (!pulseg__write4(f, &e->gy_amplitude, 1))
            goto tw_fail;
        if (!pulseg__write4(f, &e->gz_amplitude, 1))
            goto tw_fail;
        if (!pulseg__write4(f, &e->rotation_id, 1))
            goto tw_fail;
        if (!pulseg__write4(f, &e->slc, 1))
            goto tw_fail;
        if (!pulseg__write4(f, &e->seg, 1))
            goto tw_fail;
        if (!pulseg__write4(f, &e->rep, 1))
            goto tw_fail;
        if (!pulseg__write4(f, &e->avg, 1))
            goto tw_fail;
        if (!pulseg__write4(f, &e->set, 1))
            goto tw_fail;
        if (!pulseg__write4(f, &e->eco, 1))
            goto tw_fail;
        if (!pulseg__write4(f, &e->phs, 1))
            goto tw_fail;
        if (!pulseg__write4(f, &e->lin, 1))
            goto tw_fail;
        if (!pulseg__write4(f, &e->par, 1))
            goto tw_fail;
        if (!pulseg__write4(f, &e->acq, 1))
            goto tw_fail;
        /* new fields: flags (as 2 ints for portability), center_sample,
         * sample_time_us, encoding_space_ref.
         * On 32-bit targets unsigned long is 32 bits, so flags_hi is always 0. */
        {
            int flags_lo = (int)(e->flags & 0xFFFFFFFFUL);
            int flags_hi = 0;
            if (sizeof(unsigned long) > 4)
            {
                flags_hi = (int)((e->flags >> 16) >> 16);
            }
            if (!pulseg__write4(f, &flags_lo, 1))
                goto tw_fail;
            if (!pulseg__write4(f, &flags_hi, 1))
                goto tw_fail;
        }
        if (!pulseg__write4(f, &e->center_sample, 1))
            goto tw_fail;
        if (!pulseg__write4(f, &e->sample_time_us, 1))
            goto tw_fail;
        if (!pulseg__write4(f, &e->encoding_space_ref, 1))
            goto tw_fail;
        if (!pulseg__write4(f, &e->off, 1))
            goto tw_fail;
    }

    /* rotation-matrix library, folded in so the recon reader
     * is self-contained (no separate ROTATIONS-section read). */
    if (!pulseg__write4(f, &traj->num_rotations, 1))
        goto tw_fail;
    if (traj->num_rotations > 0)
    {
        if (!pulseg__write4(f, traj->rotation_matrices, traj->num_rotations * 9))
            goto tw_fail;
    }

    data_end = ftell(f);
    if (data_end < 0)
        goto tw_fail;

    entries_buf[found_idx * 3 + 1] = (int)data_start;
    entries_buf[found_idx * 3 + 2] = (int)(data_end - data_start);

    /* Patch num_sections */
    if (fseek(f, hdr_ns_pos, SEEK_SET) != 0)
        goto tw_fail;
    if (!pulseg__write4(f, &num_sections, 1))
        goto tw_fail;

    /* Rewrite all section entries */
    if (fseek(f, entries_pos, SEEK_SET) != 0)
        goto tw_fail;
    for (i = 0; i < num_sections; ++i)
    {
        if (!pulseg__write4(f, &entries_buf[i * 3], 3))
            goto tw_fail;
    }

    fclose(f);
    PULSEG_FREE(cache_path);
    return PULSEG_SUCCESS;

tw_fail:
    fclose(f);
    PULSEG_FREE(cache_path);
    return PULSEG_ERR_FILE_READ_FAILED;
}

/* ================================================================== */
/*  Compute + merge + append trajectory for a whole collection        */
/* ================================================================== */

int pulseg__save_trajectory_cache_section(const pulseg_collection *coll, const char *seq_path)
{
    pulseg_trajectory acc, one;
    pulseg_diagnostic diag;
    int i, have, rc;

    if (!coll || !seq_path)
        return PULSEG_ERR_NULL_POINTER;

    have = 0;
    memset(&acc, 0, sizeof(acc));

    for (i = 0; i < coll->num_subsequences; ++i)
    {
        memset(&one, 0, sizeof(one));
        pulseg_diagnostic_init(&diag);
        rc = pulseg_compute_trajectory(coll, &one, &diag, i);
        if (PULSEG_FAILED(rc))
        {
            if (have)
                pulseg_trajectory_free(&acc);
            return rc;
        }
        if (!have)
        {
            acc = one;
            have = 1;
        }
        else
        {
            rc = pulseg_merge_trajectory(&acc, &one);
            pulseg_trajectory_free(&one);
            if (PULSEG_FAILED(rc))
            {
                pulseg_trajectory_free(&acc);
                return rc;
            }
        }
    }

    if (have)
    {
        rc = pulseg_write_trajectory_cache(
            &acc,
            seq_path,
            coll->num_subsequences > 0 ? coll->descriptors[0].cache_ext : NULL);
        pulseg_trajectory_free(&acc);
        return rc;
    }

    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Load trajectory from cache (TRAJECTORY section)                   */
/* ================================================================== */

/* Shared body of pulseg_load_trajectory_cache():
 * parses the TRAJECTORY section from an already-open file handle. Takes
 * ownership of @p f (closes it on every return path). */
static int traj_load_from_open_file(pulseg_trajectory *out, FILE *f)
{
    int marker, num_sections;
    int version_major, version_minor, version_revision, vendor, stored_size;
    int do_swap, i, found;
    int section_id, section_offset, section_size;

    /* Read header */
    if (!pulseg__read4(f, &marker, 1))
    {
        fclose(f);
        return PULSEG_ERR_FILE_READ_FAILED;
    }
    do_swap = 0;
    if (marker != CACHE_ENDIAN_MARKER)
    {
        pulseg__swap4(&marker);
        if (marker != CACHE_ENDIAN_MARKER)
        {
            fclose(f);
            return PULSEG_ERR_FILE_READ_FAILED;
        }
        do_swap = 1;
    }
    if (!pulseg__read4(f, &version_major, 1))
    {
        fclose(f);
        return PULSEG_ERR_FILE_READ_FAILED;
    }
    if (!pulseg__read4(f, &version_minor, 1))
    {
        fclose(f);
        return PULSEG_ERR_FILE_READ_FAILED;
    }
    if (!pulseg__read4(f, &version_revision, 1))
    {
        fclose(f);
        return PULSEG_ERR_FILE_READ_FAILED;
    }
    if (!pulseg__read4(f, &vendor, 1))
    {
        fclose(f);
        return PULSEG_ERR_FILE_READ_FAILED;
    }
    if (!pulseg__read4(f, &stored_size, 1))
    {
        fclose(f);
        return PULSEG_ERR_FILE_READ_FAILED;
    }
    if (!pulseg__read4(f, &num_sections, 1))
    {
        fclose(f);
        return PULSEG_ERR_FILE_READ_FAILED;
    }
    if (do_swap)
    {
        pulseg__swap4(&version_major);
        pulseg__swap4(&version_minor);
        pulseg__swap4(&version_revision);
        pulseg__swap4(&vendor);
        pulseg__swap4(&stored_size);
        pulseg__swap4(&num_sections);
    }

    /* Find the trajectory section */
    found = 0;
    section_offset = 0;
    section_size = 0;
    for (i = 0; i < num_sections; ++i)
    {
        if (!pulseg__read4(f, &section_id, 1))
        {
            fclose(f);
            return PULSEG_ERR_FILE_READ_FAILED;
        }
        if (!pulseg__read4(f, &section_offset, 1))
        {
            fclose(f);
            return PULSEG_ERR_FILE_READ_FAILED;
        }
        if (!pulseg__read4(f, &section_size, 1))
        {
            fclose(f);
            return PULSEG_ERR_FILE_READ_FAILED;
        }
        if (do_swap)
        {
            pulseg__swap4(&section_id);
            pulseg__swap4(&section_offset);
            pulseg__swap4(&section_size);
        }
        if (section_id == CACHE_SECTION_TRAJECTORY)
        {
            found = 1;
            break;
        }
    }

    if (!found)
    {
        fclose(f);
        return PULSEG_ERR_FILE_READ_FAILED;
    }

    /* Seek to trajectory data */
    if (fseek(f, section_offset, SEEK_SET) != 0)
    {
        fclose(f);
        return PULSEG_ERR_FILE_READ_FAILED;
    }

    /* Read kshot library */
    if (!pulseg__read4(f, &out->kshots.num_shots, 1))
        goto lr_fail;
    if (do_swap)
        pulseg__swap4(&out->kshots.num_shots);

    if (out->kshots.num_shots > 0)
    {
        out->kshots.shots =
            (pulseg_kshot *)PULSEG_ALLOC((size_t)out->kshots.num_shots * sizeof(pulseg_kshot));
        if (!out->kshots.shots)
            goto lr_fail;
        memset(out->kshots.shots, 0, (size_t)out->kshots.num_shots * sizeof(pulseg_kshot));

        for (i = 0; i < out->kshots.num_shots; ++i)
        {
            if (!pulseg__read4(f, &out->kshots.shots[i].num_samples, 1))
                goto lr_fail;
            if (do_swap)
                pulseg__swap4(&out->kshots.shots[i].num_samples);

            if (out->kshots.shots[i].num_samples > 0)
            {
                out->kshots.shots[i].k =
                    (float *)PULSEG_ALLOC((size_t)out->kshots.shots[i].num_samples * sizeof(float));
                if (!out->kshots.shots[i].k)
                    goto lr_fail;
                if (!pulseg__read4(f, out->kshots.shots[i].k, out->kshots.shots[i].num_samples))
                    goto lr_fail;
                if (do_swap)
                    pulseg__swap4_array(out->kshots.shots[i].k, out->kshots.shots[i].num_samples);
            }
        }
    }

    /* Read encoding spaces */
    if (!pulseg__read4(f, &out->num_encoding_spaces, 1))
        goto lr_fail;
    if (do_swap)
        pulseg__swap4(&out->num_encoding_spaces);

    if (out->num_encoding_spaces > 0)
    {
        out->encoding_spaces = (pulseg_encoding_space *)PULSEG_ALLOC(
            (size_t)out->num_encoding_spaces * sizeof(pulseg_encoding_space));
        if (!out->encoding_spaces)
            goto lr_fail;

        for (i = 0; i < out->num_encoding_spaces; ++i)
        {
            pulseg_encoding_space *es = &out->encoding_spaces[i];
            if (!pulseg__read4(f, &es->subseq_idx, 1))
                goto lr_fail;
            if (!pulseg__read4(f, &es->nav_subseq_offset, 1))
                goto lr_fail;
            if (!pulseg__read4(f, &es->geometry_tag, 1))
                goto lr_fail;
            if (!pulseg__read4(f, &es->label_limits, sizeof(pulseg_label_limits) / sizeof(int)))
                goto lr_fail;
            /* swap all fields: 3 ints + 20 ints (label_limits) = 23 words */
            if (do_swap)
                pulseg__swap4_array(&es->subseq_idx, 23);
        }
    }

    /* Read trajectory table */
    if (!pulseg__read4(f, &out->num_adc_events, 1))
        goto lr_fail;
    if (do_swap)
        pulseg__swap4(&out->num_adc_events);

    if (out->num_adc_events > 0)
    {
        out->table = (pulseg_traj_table_entry *)PULSEG_ALLOC(
            (size_t)out->num_adc_events * sizeof(pulseg_traj_table_entry));
        if (!out->table)
            goto lr_fail;

        for (i = 0; i < out->num_adc_events; ++i)
        {
            pulseg_traj_table_entry *e = &out->table[i];
            if (!pulseg__read4(f, &e->kx_shot_id, 1))
                goto lr_fail;
            if (!pulseg__read4(f, &e->ky_shot_id, 1))
                goto lr_fail;
            if (!pulseg__read4(f, &e->kz_shot_id, 1))
                goto lr_fail;
            if (!pulseg__read4(f, &e->gx_amplitude, 1))
                goto lr_fail;
            if (!pulseg__read4(f, &e->gy_amplitude, 1))
                goto lr_fail;
            if (!pulseg__read4(f, &e->gz_amplitude, 1))
                goto lr_fail;
            if (!pulseg__read4(f, &e->rotation_id, 1))
                goto lr_fail;
            if (!pulseg__read4(f, &e->slc, 1))
                goto lr_fail;
            if (!pulseg__read4(f, &e->seg, 1))
                goto lr_fail;
            if (!pulseg__read4(f, &e->rep, 1))
                goto lr_fail;
            if (!pulseg__read4(f, &e->avg, 1))
                goto lr_fail;
            if (!pulseg__read4(f, &e->set, 1))
                goto lr_fail;
            if (!pulseg__read4(f, &e->eco, 1))
                goto lr_fail;
            if (!pulseg__read4(f, &e->phs, 1))
                goto lr_fail;
            if (!pulseg__read4(f, &e->lin, 1))
                goto lr_fail;
            if (!pulseg__read4(f, &e->par, 1))
                goto lr_fail;
            if (!pulseg__read4(f, &e->acq, 1))
                goto lr_fail;
            if (do_swap)
                pulseg__swap4_array(&e->kx_shot_id, 17);
            /* new fields */
            {
                int flags_lo = 0, flags_hi = 0;
                if (!pulseg__read4(f, &flags_lo, 1))
                    goto lr_fail;
                if (!pulseg__read4(f, &flags_hi, 1))
                    goto lr_fail;
                if (do_swap)
                {
                    pulseg__swap4(&flags_lo);
                    pulseg__swap4(&flags_hi);
                }
                e->flags = (unsigned long)(unsigned int)flags_lo;
                if (sizeof(unsigned long) > 4)
                {
                    e->flags |= ((unsigned long)(unsigned int)flags_hi << 16) << 16;
                }
            }
            if (!pulseg__read4(f, &e->center_sample, 1))
                goto lr_fail;
            if (!pulseg__read4(f, &e->sample_time_us, 1))
                goto lr_fail;
            if (!pulseg__read4(f, &e->encoding_space_ref, 1))
                goto lr_fail;
            if (!pulseg__read4(f, &e->off, 1))
                goto lr_fail;
            if (do_swap)
                pulseg__swap4_array(&e->center_sample, 4);
        }
    }

    /* folded-in rotation-matrix library. */
    if (!pulseg__read4(f, &out->num_rotations, 1))
        goto lr_fail;
    if (do_swap)
        pulseg__swap4(&out->num_rotations);
    if (out->num_rotations > 0)
    {
        out->rotation_matrices =
            (float(*)[9])PULSEG_ALLOC((size_t)out->num_rotations * sizeof(float[9]));
        if (!out->rotation_matrices)
            goto lr_fail;
        if (!pulseg__read4(f, out->rotation_matrices, out->num_rotations * 9))
            goto lr_fail;
        if (do_swap)
            pulseg__swap4_array(out->rotation_matrices, out->num_rotations * 9);
    }

    fclose(f);
    return PULSEG_SUCCESS;

lr_fail:
    fclose(f);
    pulseg_trajectory_free(out);
    return PULSEG_ERR_FILE_READ_FAILED;
}

int pulseg_load_trajectory_cache(pulseg_trajectory *out, const char *cache_path)
{
    FILE *f;

    if (!out || !cache_path)
        return PULSEG_ERR_NULL_POINTER;
    memset(out, 0, sizeof(*out));

    f = fopen(cache_path, "rb");
    if (!f)
        return PULSEG_ERR_FILE_READ_FAILED;

    return traj_load_from_open_file(out, f);
}
