/* pulseg_seqdesc.c -- Sequence description walker (Section 5)
 *
 * Implements:
 *   pulseg_get_sequence_description()
 *   pulseg_sequence_description_free()
 *   pulseg_get_sequence_parameters()
 *
 * Emits a compact per-block row table over the full pass (canonical TR).
 * One row per block: RF rows carry rf_def_id, rf_use, act_amplitude,
 * phase_offset, freq_offset, rf_shim_id; ADC rows carry adc_role and
 * phase_offset and the real-instance echo flag; OTHER rows are zero-padded.
 * freq/phase already include ppm terms (folded in at dedup time by
 * pulseg_dedup.c).
 */

#include "pulseg_internal.h"
#include "pulseg.h"
#include <string.h>
#include <math.h>

/* ================================================================== */
/*  Internal helpers                                                  */
/* ================================================================== */

/* The canonical TR window: the repetition that acquires at the most positions,
 * earliest one on a tie.  A description is read back position by position, so
 * a window that misses a readout the scan plays elsewhere describes a
 * sequence that was never run -- which happens whenever the opening
 * repetitions are dummies, or a repetition drops a readout it takes up again
 * later.  Falls back to the first window when nothing acquires at all
 * (RF-only scans). */
static void seqdesc__select_canonical_window(
    const pulseg_sequence_descriptor *desc,
    int *start_block,
    int *block_count)
{
    int tr_size = desc->tr_descriptor.tr_size;
    int start, i, acquires, best;

    *start_block = 0;
    *block_count = tr_size;

    if (*block_count > desc->num_blocks)
        *block_count = desc->num_blocks;
    if (*block_count < 0)
        *block_count = 0;
    if (tr_size <= 0)
        return;

    best = 0;
    for (start = 0; start + tr_size <= desc->num_blocks; start += tr_size)
    {
        acquires = 0;
        for (i = start; i < start + tr_size; ++i)
        {
            if (desc->block_table[i].adc_id >= 0)
                ++acquires;
        }
        if (acquires > best)
        {
            best = acquires;
            *start_block = start;
            if (acquires == tr_size)
                return;
        }
    }
}

/*
 * Build ADC k=0 anchors and role assignments from canonical-pass waveforms.
 * Uses pulseg__get_gradient_waveforms_range (ZERO_VAR mode) which is
 * anchor-independent and works for both degenerate and non-degenerate sequences.
 * RF refocusing events flip the k-space sign; excitation events reset k to 0.
 * Mirrors TruthBuilder's buildSubseqAdcTiming algorithm exactly.
 * Returns 0 on success, negative error code on failure.
 * On success, caller must free *out_kzero and *out_roles.
 */
static int seqdesc__build_adc_anchors_from_canonical(
    const pulseg_collection *coll,
    float **out_kzero,
    int **out_roles,
    float **out_anchor_kxyz,
    const pulseg_sequence_descriptor *desc,
    int subseq_idx,
    int block_start,
    int n_walk)
{
    pulseg__uniform_grad_waveforms uw = PULSEG__UNIFORM_GRAD_WAVEFORMS_INIT;
    pulseg_diagnostic diag;
    float *kzero_us = NULL;
    int *roles = NULL;
    float *anchor_kxyz = NULL;
    float *krss_vals = NULL;
    float *kx_vals = NULL;
    float *ky_vals = NULL;
    float *kz_vals = NULL;
    int *refocus_samples = NULL;
    int *excite_samples = NULL;
    int n_refocus = 0, n_excite = 0;
    int n_samples;
    float dt;
    int rc, i, j, ai;

    /* Suppress unused-parameter warnings; desc is used directly. */
    (void)coll;
    (void)subseq_idx;

    if (!out_kzero || !out_roles)
        return PULSEG_ERR_NULL_POINTER;

    *out_kzero = NULL;
    *out_roles = NULL;
    if (out_anchor_kxyz)
        *out_anchor_kxyz = NULL;

    pulseg_diagnostic_init(&diag);

    /* Step 1: Get anchor-independent uniform-raster gradient waveforms.
     * ZERO_VAR zeros variable-amplitude gradients (PE), keeping the
     * readout / slice gradients that determine the k-space trajectory shape. */
    rc = pulseg__get_gradient_waveforms_range(
        desc,
        &uw,
        &diag,
        block_start,
        n_walk,
        PULSEG_AMP_ZERO_VAR,
        NULL,
        0,
        NULL);
    if (rc != PULSEG_SUCCESS)
        return rc;

    n_samples = uw.num_samples;
    dt = uw.raster_us;

    if (n_samples < 2 || dt <= 0.0f)
    {
        pulseg__uniform_grad_waveforms_free(&uw);
        return PULSEG_ERR_INVALID_ARGUMENT;
    }

    /* Step 2: Allocate output arrays and initialise default roles */
    kzero_us = (float *)PULSEG_ALLOC((size_t)n_walk * sizeof(float));
    roles = (int *)PULSEG_ALLOC((size_t)n_walk * sizeof(int));
    if (out_anchor_kxyz)
        anchor_kxyz = (float *)PULSEG_ALLOC((size_t)n_walk * 3U * sizeof(float));
    if (!kzero_us || !roles || (out_anchor_kxyz && !anchor_kxyz))
    {
        PULSEG_FREE(kzero_us);
        PULSEG_FREE(roles);
        PULSEG_FREE(anchor_kxyz);
        pulseg__uniform_grad_waveforms_free(&uw);
        return PULSEG_ERR_ALLOC_FAILED;
    }
    if (anchor_kxyz)
        memset(anchor_kxyz, 0, (size_t)n_walk * 3U * sizeof(float));

    for (i = 0; i < n_walk; ++i)
    {
        const pulseg_block_table_element *bte = &desc->block_table[block_start + i];
        kzero_us[i] = -1.0f;
        roles[i] = -1;
        /* Use bte->adc_id >= 0 as the presence test (matches block_table_element
         * semantics); base_blocks[bte->id].adc_id used for definition lookup. */
        if (bte->adc_id >= 0)
            roles[i] = bte->nav_flag ? PULSEG_ADC_ROLE_NON_ACQUIRED : PULSEG_ADC_ROLE_SINGLE;
    }

    /* Step 3: Walk blocks to collect RF event sample indices.
     * Refocusing pulses flip k-space sign; excitation pulses reset k to 0.
     * RF isocenter time = block_start_us + rfdef->delay + rfdef->stats.isodelay_us
     * (matches safety.c and TruthBuilder's block.rf.delay + mr.calcRfCenter). */
    {
        int n_rf_cap = 0, n_ex_cap = 0;
        float blk_start_us;
        const pulseg_block_table_element *bte;
        const pulseg_rf_table_element *rte;
        const pulseg_rf_definition *rdef;
        int rf_def_id, rf_use;

        /* Count refocusing and excitation events */
        for (i = 0; i < n_walk; ++i)
        {
            bte = &desc->block_table[block_start + i];
            if (bte->rf_id < 0 || bte->rf_id >= desc->rf_table_size)
                continue;
            rte = &desc->rf_table[bte->rf_id];
            if (rte->rf_use == PULSEG_RF_USE_REFOCUSING)
                n_rf_cap++;
            else if (rte->rf_use == PULSEG_RF_USE_EXCITATION)
                n_ex_cap++;
        }

        if (n_rf_cap > 0)
        {
            refocus_samples = (int *)PULSEG_ALLOC((size_t)n_rf_cap * sizeof(int));
            if (!refocus_samples)
            {
                PULSEG_FREE(kzero_us);
                PULSEG_FREE(roles);
                PULSEG_FREE(anchor_kxyz);
                pulseg__uniform_grad_waveforms_free(&uw);
                return PULSEG_ERR_ALLOC_FAILED;
            }
        }
        if (n_ex_cap > 0)
        {
            excite_samples = (int *)PULSEG_ALLOC((size_t)n_ex_cap * sizeof(int));
            if (!excite_samples)
            {
                PULSEG_FREE(kzero_us);
                PULSEG_FREE(roles);
                PULSEG_FREE(anchor_kxyz);
                PULSEG_FREE(refocus_samples);
                pulseg__uniform_grad_waveforms_free(&uw);
                return PULSEG_ERR_ALLOC_FAILED;
            }
        }

        blk_start_us = 0.0f;
        for (i = 0; i < n_walk; ++i)
        {
            bte = &desc->block_table[block_start + i];

            if (bte->rf_id >= 0 && bte->rf_id < desc->rf_table_size)
            {
                rte = &desc->rf_table[bte->rf_id];
                rf_def_id = rte->id;
                rf_use = rte->rf_use;

                if (rf_def_id >= 0 && rf_def_id < desc->num_unique_rfs)
                {
                    int iso_sample;
                    float iso_us;
                    rdef = &desc->rf_definitions[rf_def_id];
                    iso_us = blk_start_us + (float)rdef->delay + (float)rdef->stats.duration_us -
                        (float)rdef->stats.isodelay_us;
                    iso_sample = (int)(iso_us / dt);
                    if (iso_sample < 0)
                        iso_sample = 0;
                    if (iso_sample >= n_samples)
                        iso_sample = n_samples - 1;

                    if (rf_use == PULSEG_RF_USE_REFOCUSING && refocus_samples)
                        refocus_samples[n_refocus++] = iso_sample;
                    else if (rf_use == PULSEG_RF_USE_EXCITATION && excite_samples)
                        excite_samples[n_excite++] = iso_sample;
                }
            }

            {
                int blk_dur = (bte->duration_us >= 0) ? bte->duration_us
                                                      : desc->base_blocks[bte->id].duration_us;
                blk_start_us += (float)blk_dur;
            }
        }
    }

    /* Step 4: Compute krss via O(n) cumulative trapezoidal integration.
     * Matches TruthBuilder: integrate forward, then apply RF events at
     * each sample: excitation resets k=0; refocusing negates k. */
    krss_vals = (float *)PULSEG_ALLOC((size_t)n_samples * sizeof(float));
    if (anchor_kxyz)
    {
        kx_vals = (float *)PULSEG_ALLOC((size_t)n_samples * sizeof(float));
        ky_vals = (float *)PULSEG_ALLOC((size_t)n_samples * sizeof(float));
        kz_vals = (float *)PULSEG_ALLOC((size_t)n_samples * sizeof(float));
    }
    if (!krss_vals || (anchor_kxyz && (!kx_vals || !ky_vals || !kz_vals)))
    {
        PULSEG_FREE(kzero_us);
        PULSEG_FREE(roles);
        PULSEG_FREE(anchor_kxyz);
        PULSEG_FREE(krss_vals);
        PULSEG_FREE(kx_vals);
        PULSEG_FREE(ky_vals);
        PULSEG_FREE(kz_vals);
        PULSEG_FREE(refocus_samples);
        PULSEG_FREE(excite_samples);
        pulseg__uniform_grad_waveforms_free(&uw);
        return PULSEG_ERR_ALLOC_FAILED;
    }

    {
        float kx = 0.0f, ky = 0.0f, kz = 0.0f;
        int ex_cursor = 0;
        int ref_cursor = 0;

        krss_vals[0] = 0.0f;
        if (kx_vals)
        {
            kx_vals[0] = 0.0f;
            ky_vals[0] = 0.0f;
            kz_vals[0] = 0.0f;
        }

        for (j = 1; j < n_samples; ++j)
        {
            /* Trapezoidal step: k[j] = k[j-1] + 0.5*(g[j-1]+g[j])*dt */
            kx += 0.5f * (uw.gx[j - 1] + uw.gx[j]) * dt;
            ky += 0.5f * (uw.gy[j - 1] + uw.gy[j]) * dt;
            kz += 0.5f * (uw.gz[j - 1] + uw.gz[j]) * dt;

            /* Apply excitation events: reset k to 0 */
            while (ex_cursor < n_excite && excite_samples[ex_cursor] == j)
            {
                kx = 0.0f;
                ky = 0.0f;
                kz = 0.0f;
                ex_cursor++;
            }
            /* Apply refocusing events: negate k (spin-echo sign flip) */
            while (ref_cursor < n_refocus && refocus_samples[ref_cursor] == j)
            {
                kx = -kx;
                ky = -ky;
                kz = -kz;
                ref_cursor++;
            }

            krss_vals[j] = (float)sqrt((double)kx * kx + (double)ky * ky + (double)kz * kz);
            if (kx_vals)
            {
                kx_vals[j] = kx;
                ky_vals[j] = ky;
                kz_vals[j] = kz;
            }
        }
    }

    /* Step 5: For each ADC block, find min-krss sample in ADC window.
     * ADC onset = block_start_us + adef->delay  (delay in us).
     * ADC duration = num_samples * dwell_time * 1e-3  (dwell in ns -> us). */
    /* Step 5: For each ADC block, find min-krss sample in ADC window.
     * ADC onset = block_start_us + adef->delay  (delay in us).
     * ADC duration = num_samples * dwell_time * 1e-3  (dwell in ns -> us).
     * Use base_blocks[bte->id].adc_id for the definition lookup (same as
     * safety.c) — the bte->adc_id is a collection-relative adc_table index and
     * adc_table[].id may exceed num_unique_adcs in multi-subseq collections. */
    {
        float blk_start_us = 0.0f;

        for (i = 0; i < n_walk; ++i)
        {
            const pulseg_block_table_element *bte = &desc->block_table[block_start + i];

            if (bte->adc_id >= 0)
            {
                int bdef_id = bte->id;
                int adc_def_id = (bdef_id >= 0 && bdef_id < desc->num_unique_blocks)
                    ? desc->base_blocks[bdef_id].adc_id
                    : -1;
                if (adc_def_id >= 0 && adc_def_id < desc->num_unique_adcs)
                {
                    const pulseg_adc_definition *adef = &desc->adc_definitions[adc_def_id];
                    float adc_onset = blk_start_us + (float)adef->delay;
                    float adc_dur = (float)adef->num_samples * (float)adef->dwell_time * 1.0e-3f;
                    float adc_end = adc_onset + adc_dur;
                    int rs = (int)(adc_onset / dt);
                    int re = (int)(adc_end / dt);
                    if (rs < 0)
                        rs = 0;
                    if (re >= n_samples)
                        re = n_samples - 1;

                    if (rs <= re)
                    {
                        int min_idx = rs;
                        float min_krss = krss_vals[rs];
                        for (ai = rs + 1; ai <= re; ++ai)
                        {
                            if (krss_vals[ai] < min_krss)
                            {
                                min_krss = krss_vals[ai];
                                min_idx = ai;
                            }
                        }
                        kzero_us[i] = (float)min_idx * dt;
                        if (anchor_kxyz)
                        {
                            anchor_kxyz[i * 3 + 0] = kx_vals[min_idx];
                            anchor_kxyz[i * 3 + 1] = ky_vals[min_idx];
                            anchor_kxyz[i * 3 + 2] = kz_vals[min_idx];
                        }
                    }
                }
            }

            {
                int blk_dur = (bte->duration_us >= 0) ? bte->duration_us
                                                      : desc->base_blocks[bte->id].duration_us;
                blk_start_us += (float)blk_dur;
            }
        }
    }

    /* Step 6: Classify ADC roles within the canonical window.
     * ADC blocks with kzero closest to k=0 get ECHO_CENTER or SINGLE;
     * others get NON_CENTER. Uses relative tolerance on max krss. */
    {
        int n_acq = 0, n_center = 0;
        float max_krss = 0.0f, min_krss_overall = 1e30f, rel_tol = 1e-3f;
        int group_acq = 0, largest_group = 0;

        /* ECHO_CENTER and NON_CENTER describe positions WITHIN one echo
         * group -- the acquisitions of a single excitation. A scan that
         * excites once per readout (a GRE, a radial or ZTE view) has one
         * acquisition per group, and each is its own centre: comparing it
         * against the readouts of other shots would rank separate shots
         * against each other and demote all but one of them. */
        for (i = 0; i < n_walk; ++i)
        {
            const pulseg_block_table_element *grp = &desc->block_table[block_start + i];
            if (grp->rf_id >= 0 && grp->rf_id < desc->rf_table_size &&
                desc->rf_table[grp->rf_id].rf_use == PULSEG_RF_USE_EXCITATION)
            {
                if (group_acq > largest_group)
                    largest_group = group_acq;
                group_acq = 0;
            }
            if (roles[i] > PULSEG_ADC_ROLE_NON_ACQUIRED && kzero_us[i] >= 0.0f)
                group_acq++;
        }
        if (group_acq > largest_group)
            largest_group = group_acq;

        for (i = 0; largest_group > 1 && i < n_walk; ++i)
        {
            if (roles[i] > PULSEG_ADC_ROLE_NON_ACQUIRED && kzero_us[i] >= 0.0f)
            {
                int raster_idx = (int)(kzero_us[i] / dt);
                float krss_at_adc;
                if (raster_idx < 0)
                    raster_idx = 0;
                if (raster_idx >= n_samples)
                    raster_idx = n_samples - 1;
                krss_at_adc = krss_vals[raster_idx];
                n_acq++;
                if (krss_at_adc < min_krss_overall)
                    min_krss_overall = krss_at_adc;
                if (krss_at_adc > max_krss)
                    max_krss = krss_at_adc;
            }
        }

        if (n_acq > 1 && max_krss > 0.0f)
        {
            float threshold = min_krss_overall + rel_tol * max_krss;

            for (i = 0; i < n_walk; ++i)
            {
                if (roles[i] > PULSEG_ADC_ROLE_NON_ACQUIRED && kzero_us[i] >= 0.0f)
                {
                    int raster_idx = (int)(kzero_us[i] / dt);
                    if (raster_idx < 0)
                        raster_idx = 0;
                    if (raster_idx >= n_samples)
                        raster_idx = n_samples - 1;
                    if (krss_vals[raster_idx] <= threshold)
                        n_center++;
                }
            }

            if (n_center == 1)
            {
                for (i = 0; i < n_walk; ++i)
                {
                    if (roles[i] > PULSEG_ADC_ROLE_NON_ACQUIRED && kzero_us[i] >= 0.0f)
                    {
                        int raster_idx = (int)(kzero_us[i] / dt);
                        if (raster_idx < 0)
                            raster_idx = 0;
                        if (raster_idx >= n_samples)
                            raster_idx = n_samples - 1;
                        roles[i] = (krss_vals[raster_idx] <= threshold)
                            ? PULSEG_ADC_ROLE_ECHO_CENTER
                            : PULSEG_ADC_ROLE_NON_CENTER;
                    }
                }
            }
            else if (n_center > 1)
            {
                for (i = 0; i < n_walk; ++i)
                {
                    if (roles[i] > PULSEG_ADC_ROLE_NON_ACQUIRED && kzero_us[i] >= 0.0f)
                    {
                        int raster_idx = (int)(kzero_us[i] / dt);
                        if (raster_idx < 0)
                            raster_idx = 0;
                        if (raster_idx >= n_samples)
                            raster_idx = n_samples - 1;
                        roles[i] = (krss_vals[raster_idx] <= threshold)
                            ? PULSEG_ADC_ROLE_SINGLE
                            : PULSEG_ADC_ROLE_NON_CENTER;
                    }
                }
            }
            else
            {
                /* n_center == 0: closest to k=0 gets ECHO_CENTER */
                int closest_idx = -1;
                for (i = 0; i < n_walk; ++i)
                {
                    if (roles[i] > PULSEG_ADC_ROLE_NON_ACQUIRED && kzero_us[i] >= 0.0f)
                    {
                        int raster_idx = (int)(kzero_us[i] / dt);
                        if (raster_idx < 0)
                            raster_idx = 0;
                        if (raster_idx >= n_samples)
                            raster_idx = n_samples - 1;
                        if (closest_idx < 0)
                        {
                            closest_idx = i;
                        }
                        else
                        {
                            int cr = (int)(kzero_us[closest_idx] / dt);
                            if (cr < 0)
                                cr = 0;
                            if (cr >= n_samples)
                                cr = n_samples - 1;
                            if (krss_vals[raster_idx] < krss_vals[cr])
                                closest_idx = i;
                        }
                    }
                }
                if (closest_idx >= 0)
                {
                    int closest_raster = (int)(kzero_us[closest_idx] / dt);
                    int n_tied = 0;
                    int role_for_center;
                    if (closest_raster < 0)
                        closest_raster = 0;
                    if (closest_raster >= n_samples)
                        closest_raster = n_samples - 1;
                    for (i = 0; i < n_walk; ++i)
                    {
                        if (roles[i] > PULSEG_ADC_ROLE_NON_ACQUIRED && kzero_us[i] >= 0.0f)
                        {
                            int raster_idx = (int)(kzero_us[i] / dt);
                            if (raster_idx < 0)
                                raster_idx = 0;
                            if (raster_idx >= n_samples)
                                raster_idx = n_samples - 1;
                            if (krss_vals[raster_idx] == krss_vals[closest_raster])
                                n_tied++;
                        }
                    }
                    role_for_center =
                        (n_tied == 1) ? PULSEG_ADC_ROLE_ECHO_CENTER : PULSEG_ADC_ROLE_SINGLE;
                    for (i = 0; i < n_walk; ++i)
                    {
                        if (roles[i] > PULSEG_ADC_ROLE_NON_ACQUIRED && kzero_us[i] >= 0.0f)
                        {
                            int raster_idx = (int)(kzero_us[i] / dt);
                            if (raster_idx < 0)
                                raster_idx = 0;
                            if (raster_idx >= n_samples)
                                raster_idx = n_samples - 1;
                            roles[i] = (krss_vals[raster_idx] == krss_vals[closest_raster])
                                ? role_for_center
                                : PULSEG_ADC_ROLE_NON_CENTER;
                        }
                    }
                }
            }
        }
    }

    /* Clean up temporaries */
    PULSEG_FREE(krss_vals);
    PULSEG_FREE(kx_vals);
    PULSEG_FREE(ky_vals);
    PULSEG_FREE(kz_vals);
    PULSEG_FREE(refocus_samples);
    PULSEG_FREE(excite_samples);
    pulseg__uniform_grad_waveforms_free(&uw);

    *out_kzero = kzero_us;
    *out_roles = roles;
    if (out_anchor_kxyz)
        *out_anchor_kxyz = anchor_kxyz;
    return PULSEG_SUCCESS;
}

/*
 * Determine which canonical ADC positions are true echoes in at least one
 * real scan-table instance. This is deliberately separate from the ZERO_VAR
 * canonical-TR analyzer above: a varying gradient being replaced by zero does
 * not imply that zero is present in the acquired schedule.
 *
 * Each actual TR/pass is inspected because shot-index deduplication does not
 * capture per-instance gradient-amplitude changes. For each instance we
 * integrate the ACTUAL gradient waveform, including excitation resets and
 * refocusing sign changes, then reduce the minimum |k| into the canonical ADC
 * position. The ZERO_VAR path contributes only its numerical k=0 floor; it
 * never counts as an acquired instance. Rotation need not be applied because
 * an orthonormal rotation preserves |k|.
 */
static int seqdesc__build_adc_echo_flags(
    int **out_echoes,
    const pulseg_sequence_descriptor *desc,
    int block_start,
    int n_walk,
    const float *canonical_anchor_kxyz)
{
    int *echoes = NULL;
    float *min_krss_by_position = NULL;
    int *opens_at_closest = NULL;
    float canonical_floor = 1.0e30f;
    float global_scale = 0.0f;
    int num_instances;
    int variant;
    int i;
    int rc = PULSEG_SUCCESS;

    if (!out_echoes || !desc)
        return PULSEG_ERR_NULL_POINTER;
    *out_echoes = NULL;

    echoes = (int *)PULSEG_ALLOC((size_t)n_walk * sizeof(int));
    min_krss_by_position = (float *)PULSEG_ALLOC((size_t)n_walk * sizeof(float));
    opens_at_closest = (int *)PULSEG_ALLOC((size_t)n_walk * sizeof(int));
    if (!echoes || !min_krss_by_position || !opens_at_closest)
    {
        PULSEG_FREE(echoes);
        PULSEG_FREE(min_krss_by_position);
        PULSEG_FREE(opens_at_closest);
        return PULSEG_ERR_ALLOC_FAILED;
    }
    memset(opens_at_closest, 0, (size_t)n_walk * sizeof(int));
    memset(echoes, 0, (size_t)n_walk * sizeof(int));
    for (i = 0; i < n_walk; ++i)
        min_krss_by_position[i] = 1.0e30f;

    num_instances = desc->tr_descriptor.num_trs;
    if (num_instances < 1)
        num_instances = 1;

    for (i = 0; i < n_walk; ++i)
    {
        const pulseg_block_table_element *bte = &desc->block_table[block_start + i];
        if (bte->adc_id >= 0 && !bte->nav_flag)
        {
            if (canonical_anchor_kxyz)
            {
                double kx = canonical_anchor_kxyz[i * 3 + 0];
                double ky = canonical_anchor_kxyz[i * 3 + 1];
                double kz = canonical_anchor_kxyz[i * 3 + 2];
                float norm = (float)sqrt(kx * kx + ky * ky + kz * kz);
                if (norm < canonical_floor)
                    canonical_floor = norm;
            }
        }
    }
    if (canonical_floor == 1.0e30f)
        canonical_floor = 0.0f;

    for (variant = 0; variant < num_instances; ++variant)
    {
        pulseg__uniform_grad_waveforms uw = PULSEG__UNIFORM_GRAD_WAVEFORMS_INIT;
        pulseg_diagnostic diag;
        float *krss = NULL;
        int *refocus_samples = NULL;
        int *excite_samples = NULL;
        int actual_start;
        int rep_idx;
        int n_refocus = 0;
        int n_excite = 0;
        int n_refocus_cap = 0;
        int n_excite_cap = 0;
        int n_samples;
        float dt;
        float blk_start_us;
        float max_krss;
        int j;

        /* Instance windows are anchored at block 0 whatever window was
         * chosen as canonical: every TR of the table is an instance. */
        rep_idx = variant;
        actual_start = rep_idx * desc->tr_descriptor.tr_size;

        pulseg_diagnostic_init(&diag);
        rc = pulseg__get_gradient_waveforms_range(
            desc,
            &uw,
            &diag,
            actual_start,
            n_walk,
            PULSEG_AMP_ACTUAL,
            NULL,
            0,
            NULL);
        if (rc != PULSEG_SUCCESS)
            goto echo_variant_done;

        n_samples = uw.num_samples;
        dt = uw.raster_us;
        if (n_samples < 2 || dt <= 0.0f)
        {
            rc = PULSEG_ERR_INVALID_ARGUMENT;
            goto echo_variant_done;
        }

        /* Collect the RF state changes for this actual representative. */
        for (i = 0; i < n_walk; ++i)
        {
            const pulseg_block_table_element *bte = &desc->block_table[actual_start + i];
            if (bte->rf_id >= 0 && bte->rf_id < desc->rf_table_size)
            {
                int use = desc->rf_table[bte->rf_id].rf_use;
                if (use == PULSEG_RF_USE_REFOCUSING)
                    n_refocus_cap++;
                else if (use == PULSEG_RF_USE_EXCITATION)
                    n_excite_cap++;
            }
        }
        if (n_refocus_cap > 0)
            refocus_samples = (int *)PULSEG_ALLOC((size_t)n_refocus_cap * sizeof(int));
        if (n_excite_cap > 0)
            excite_samples = (int *)PULSEG_ALLOC((size_t)n_excite_cap * sizeof(int));
        if ((n_refocus_cap > 0 && !refocus_samples) || (n_excite_cap > 0 && !excite_samples))
        {
            rc = PULSEG_ERR_ALLOC_FAILED;
            goto echo_variant_done;
        }

        blk_start_us = 0.0f;
        for (i = 0; i < n_walk; ++i)
        {
            const pulseg_block_table_element *bte = &desc->block_table[actual_start + i];
            if (bte->rf_id >= 0 && bte->rf_id < desc->rf_table_size)
            {
                const pulseg_rf_table_element *rte = &desc->rf_table[bte->rf_id];
                int rf_def_id = rte->id;
                if (rf_def_id >= 0 && rf_def_id < desc->num_unique_rfs)
                {
                    const pulseg_rf_definition *rdef = &desc->rf_definitions[rf_def_id];
                    float iso_us = blk_start_us + (float)rdef->delay +
                        (float)rdef->stats.duration_us - (float)rdef->stats.isodelay_us;
                    int iso_sample = (int)(iso_us / dt);
                    if (iso_sample < 0)
                        iso_sample = 0;
                    if (iso_sample >= n_samples)
                        iso_sample = n_samples - 1;
                    if (rte->rf_use == PULSEG_RF_USE_REFOCUSING)
                        refocus_samples[n_refocus++] = iso_sample;
                    else if (rte->rf_use == PULSEG_RF_USE_EXCITATION)
                        excite_samples[n_excite++] = iso_sample;
                }
            }
            {
                int dur = (bte->duration_us >= 0) ? bte->duration_us
                                                  : desc->base_blocks[bte->id].duration_us;
                blk_start_us += (float)dur;
            }
        }

        krss = (float *)PULSEG_ALLOC((size_t)n_samples * 3U * sizeof(float));
        if (!krss)
        {
            rc = PULSEG_ERR_ALLOC_FAILED;
            goto echo_variant_done;
        }

        {
            float kx = 0.0f;
            float ky = 0.0f;
            float kz = 0.0f;
            int ex_cursor = 0;
            int ref_cursor = 0;

            krss[0] = 0.0f;
            krss[n_samples] = 0.0f;
            krss[2 * n_samples] = 0.0f;
            for (j = 1; j < n_samples; ++j)
            {
                kx += 0.5f * (uw.gx[j - 1] + uw.gx[j]) * dt;
                ky += 0.5f * (uw.gy[j - 1] + uw.gy[j]) * dt;
                kz += 0.5f * (uw.gz[j - 1] + uw.gz[j]) * dt;

                while (ex_cursor < n_excite && excite_samples[ex_cursor] == j)
                {
                    kx = 0.0f;
                    ky = 0.0f;
                    kz = 0.0f;
                    ex_cursor++;
                }
                while (ref_cursor < n_refocus && refocus_samples[ref_cursor] == j)
                {
                    kx = -kx;
                    ky = -ky;
                    kz = -kz;
                    ref_cursor++;
                }
                krss[j] = kx;
                krss[n_samples + j] = ky;
                krss[2 * n_samples + j] = kz;
            }
        }

        max_krss = 0.0f;
        for (j = 0; j < n_samples; ++j)
        {
            float kx = krss[j];
            float ky = krss[n_samples + j];
            float kz = krss[2 * n_samples + j];
            float mag = (float)sqrt((double)kx * kx + (double)ky * ky + (double)kz * kz);
            if (mag > max_krss)
                max_krss = mag;
        }
        if (max_krss > global_scale)
            global_scale = max_krss;

        /* Reduce this real variant into the per-position minimum |k|. */
        blk_start_us = 0.0f;
        for (i = 0; i < n_walk; ++i)
        {
            const pulseg_block_table_element *bte = &desc->block_table[actual_start + i];
            if (bte->adc_id >= 0 && !bte->nav_flag)
            {
                int bdef_id = bte->id;
                int adc_def_id = (bdef_id >= 0 && bdef_id < desc->num_unique_blocks)
                    ? desc->base_blocks[bdef_id].adc_id
                    : -1;
                if (adc_def_id >= 0 && adc_def_id < desc->num_unique_adcs)
                {
                    const pulseg_adc_definition *adef = &desc->adc_definitions[adc_def_id];
                    float onset = blk_start_us + (float)adef->delay;
                    float duration = (float)adef->num_samples * (float)adef->dwell_time * 1.0e-3f;
                    int rs = (int)(onset / dt);
                    int re = (int)((onset + duration) / dt);
                    float min_krss;

                    if (rs < 0)
                        rs = 0;
                    if (re >= n_samples)
                        re = n_samples - 1;
                    if (rs <= re)
                    {
                        /* The echo is where the readout reaches the centre of
                         * the space IT sweeps. A window tracing a planar (or
                         * 3D) path -- a spiral or rosette arm -- is a
                         * self-contained acquisition of that plane, and an
                         * axis held constant across it (a stack's partition
                         * encode) is an outer encode that must not veto the
                         * in-plane echo. A one-dimensional window (a Cartesian
                         * or EPI line, a radial spoke) keeps the full-|k|
                         * criterion: there the held offsets are coordinates of
                         * the very plane the train assembles, and only the
                         * centre line is the echo. */
                        float ex0 = krss[rs];
                        float ey0 = krss[n_samples + rs];
                        float ez0 = krss[2 * n_samples + rs];
                        float ux = krss[re] - ex0;
                        float uy = krss[n_samples + re] - ey0;
                        float uz = krss[2 * n_samples + re] - ez0;
                        float span_x_min = 0.0f, span_x_max = 0.0f;
                        float span_y_min = 0.0f, span_y_max = 0.0f;
                        float span_z_min = 0.0f, span_z_max = 0.0f;
                        float cross_max = 0.0f;
                        float u_norm =
                            (float)sqrt((double)ux * ux + (double)uy * uy + (double)uz * uz);
                        int planar;

                        for (j = rs; j <= re; ++j)
                        {
                            float dx = krss[j] - ex0;
                            float dy = krss[n_samples + j] - ey0;
                            float dz = krss[2 * n_samples + j] - ez0;
                            float cx = dy * uz - dz * uy;
                            float cy = dz * ux - dx * uz;
                            float cz = dx * uy - dy * ux;
                            float cross =
                                (float)sqrt((double)cx * cx + (double)cy * cy + (double)cz * cz);
                            if (cross > cross_max)
                                cross_max = cross;
                            if (dx < span_x_min)
                                span_x_min = dx;
                            if (dx > span_x_max)
                                span_x_max = dx;
                            if (dy < span_y_min)
                                span_y_min = dy;
                            if (dy > span_y_max)
                                span_y_max = dy;
                            if (dz < span_z_min)
                                span_z_min = dz;
                            if (dz > span_z_max)
                                span_z_max = dz;
                        }
                        planar =
                            (max_krss > 0.0f && u_norm > 0.0f &&
                             cross_max > 1e-3f * u_norm * max_krss);

                        {
                            float span_scale = span_x_max - span_x_min;
                            float sy = span_y_max - span_y_min;
                            float sz = span_z_max - span_z_min;
                            int use_x = 1, use_y = 1, use_z = 1;
                            int arg_min = rs;

                            if (sy > span_scale)
                                span_scale = sy;
                            if (sz > span_scale)
                                span_scale = sz;
                            if (planar && span_scale > 0.0f)
                            {
                                use_x = (span_x_max - span_x_min) > 1e-3f * span_scale;
                                use_y = (span_y_max - span_y_min) > 1e-3f * span_scale;
                                use_z = (span_z_max - span_z_min) > 1e-3f * span_scale;
                            }

                            min_krss = 0.0f;
                            for (j = rs; j <= re; ++j)
                            {
                                double acc = 0.0;
                                float mag;
                                if (use_x)
                                    acc += (double)krss[j] * krss[j];
                                if (use_y)
                                    acc += (double)krss[n_samples + j] * krss[n_samples + j];
                                if (use_z)
                                    acc +=
                                        (double)krss[2 * n_samples + j] * krss[2 * n_samples + j];
                                mag = (float)sqrt(acc);
                                if (j == rs || mag < min_krss)
                                {
                                    min_krss = mag;
                                    arg_min = j;
                                }
                            }
                            if (min_krss < min_krss_by_position[i])
                            {
                                min_krss_by_position[i] = min_krss;
                                opens_at_closest[i] = (arg_min == rs);
                            }
                        }
                    }
                }
            }
            {
                int dur = (bte->duration_us >= 0) ? bte->duration_us
                                                  : desc->base_blocks[bte->id].duration_us;
                blk_start_us += (float)dur;
            }
        }

    echo_variant_done:
        if (krss)
            PULSEG_FREE(krss);
        if (refocus_samples)
            PULSEG_FREE(refocus_samples);
        if (excite_samples)
            PULSEG_FREE(excite_samples);
        pulseg__uniform_grad_waveforms_free(&uw);
        if (rc != PULSEG_SUCCESS)
            break;
    }

    if (rc != PULSEG_SUCCESS)
    {
        PULSEG_FREE(echoes);
        PULSEG_FREE(min_krss_by_position);
        PULSEG_FREE(opens_at_closest);
        return rc;
    }

    {
        /*
         * Two ways a readout carries the echo.
         *
         * It crosses the centre. Raster integration rarely lands on
         * mathematical zero exactly, so the canonical anchor supplies the
         * error already accepted there -- capped at error scale, because a
         * real distance from the centre is the physics of the sequence and
         * must not be normalised away as if it were integration error.
         *
         * Or it OPENS at its closest approach and only recedes: a
         * centre-out acquisition. A ZTE cannot sample k = 0 at all -- the
         * acquisition opens at |k| = G*t_dead, after the dead time -- yet
         * its first sample is still the k-space anchor everything
         * downstream needs: the phase reference an FOV shift demodulates
         * against, and the time origin a simulation gives the echo. So the
         * first sample is quoted as the echo, provided this window comes as
         * close to the centre as the TR ever does; a readout that opens at
         * its own closest approach far out (an outer line of a centre-out
         * train) is not the train's echo and stays unflagged.
         */
        float error_scale = 1.0e-3f * global_scale;
        float floor = canonical_floor < error_scale ? canonical_floor : error_scale;
        float threshold = floor + 1.0e-5f * global_scale + 1.0e-6f;
        float closest = 1.0e30f;

        for (i = 0; i < n_walk; ++i)
        {
            if (min_krss_by_position[i] < closest)
                closest = min_krss_by_position[i];
        }
        for (i = 0; i < n_walk; ++i)
        {
            if (min_krss_by_position[i] > 1.0e29f)
                continue;
            if (min_krss_by_position[i] <= threshold)
                echoes[i] = 1;
            else if (opens_at_closest[i] && min_krss_by_position[i] <= 1.1f * closest + threshold)
                echoes[i] = 1;
        }
    }
    PULSEG_FREE(min_krss_by_position);
    PULSEG_FREE(opens_at_closest);
    *out_echoes = echoes;
    return PULSEG_SUCCESS;
}

/*
 * Compute pass-relative start time (us) for every block in [0, n_walk).
 * Returns a malloc'd array of length n_walk; caller must free.
 *
 * The walker emits one Section-5 subsequence per .seq descriptor and
 * covers the entire pass (prep + main + cooldown), matching the
 * MATLAB TruthBuilder convention required by the Bloch state-machine
 * simulator. The legacy canonical-TR variant (used by safety/getters)
 * is intentionally NOT used here.
 */
static float *seqdesc__build_block_start_us(
    const pulseg_sequence_descriptor *desc,
    int start_block,
    int n_walk)
{
    float *t;
    float acc;
    int i;

    if (n_walk <= 0)
        return NULL;
    t = (float *)PULSEG_ALLOC((size_t)n_walk * sizeof(float));
    if (!t)
        return NULL;

    acc = 0.0f;
    for (i = 0; i < n_walk; ++i)
    {
        int blk = start_block + i;
        t[i] = acc;
        if (blk >= 0 && blk < desc->num_blocks)
        {
            const pulseg_block_table_element *bte = &desc->block_table[blk];
            int dur =
                (bte->duration_us >= 0) ? bte->duration_us : desc->base_blocks[bte->id].duration_us;
            acc += (float)dur;
        }
    }
    return t;
}

static int seqdesc__adc_role_is_te_bearing(int adc_role)
{
    return adc_role == PULSEG_ADC_ROLE_SINGLE || adc_role == PULSEG_ADC_ROLE_ECHO_CENTER;
}

/* ================================================================== */
/*  pulseg_sequence_description_free                               */
/* ================================================================== */

void pulseg_sequence_description_free(pulseg_sequence_description *desc)
{
    if (!desc)
        return;
    if (desc->rows)
        PULSEG_FREE(desc->rows);
    memset(desc, 0, sizeof(*desc));
}

/* ================================================================== */
/*  pulseg_get_sequence_description                                */
/* ================================================================== */

int pulseg_get_sequence_description(
    pulseg_sequence_description *out,
    const pulseg_collection *coll,
    int subseq_idx)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_descriptor *trd;
    const pulseg_block_table_element *bte;
    const pulseg_rf_definition *rfdef;
    const pulseg_adc_definition *adcdef;

    float *blk_start = NULL;
    float *adc_kzero = NULL;
    float *adc_anchor_kxyz = NULL;
    int *adc_roles = NULL;
    int *adc_echoes = NULL;
    int n_walk, start_block, i, rf_def_id, adc_def_id;
    int ret = PULSEG_SUCCESS;

    if (!out || !coll)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;

    memset(out, 0, sizeof(*out));
    out->subseq_idx = subseq_idx;

    desc = &coll->descriptors[subseq_idx];
    trd = &desc->tr_descriptor;

    seqdesc__select_canonical_window(desc, &start_block, &n_walk);

    out->tr_duration_us = trd->tr_duration_us;

    /* Step 1: per-block start times (legacy, kept for TE computation) */
    blk_start = seqdesc__build_block_start_us(desc, start_block, n_walk);
    if (!blk_start)
    {
        ret = PULSEG_ERR_ALLOC_FAILED;
        goto cleanup;
    }

    /* Step 2: ADC k=0 times + roles from canonical-pass waveforms */
    ret = seqdesc__build_adc_anchors_from_canonical(
        coll,
        &adc_kzero,
        &adc_roles,
        &adc_anchor_kxyz,
        desc,
        subseq_idx,
        start_block,
        n_walk);
    if (ret != PULSEG_SUCCESS)
    {
        goto cleanup;
    }

    /* Step 3: true-echo flags reduced over real scan-table instances. */
    ret = seqdesc__build_adc_echo_flags(&adc_echoes, desc, start_block, n_walk, adc_anchor_kxyz);
    if (ret != PULSEG_SUCCESS)
        goto cleanup;

    /* Step 4: allocate compact row table */
    out->rows = (pulseg_seq_event *)PULSEG_ALLOC((size_t)n_walk * sizeof(pulseg_seq_event));
    if (!out->rows)
    {
        ret = PULSEG_ERR_ALLOC_FAILED;
        goto cleanup;
    }
    memset(out->rows, 0, (size_t)n_walk * sizeof(pulseg_seq_event));
    out->num_rows = n_walk;

    /* Step 5: fill one row per canonical-window block */
    for (i = 0; i < n_walk; ++i)
    {
        pulseg_seq_event *row = &out->rows[i];
        bte = &desc->block_table[start_block + i];

        /* ---- RF block ---- */
        if (bte->rf_id >= 0 && bte->rf_id < desc->rf_table_size)
        {
            const pulseg_rf_table_element *rfe = &desc->rf_table[bte->rf_id];
            rf_def_id = rfe->id;

            if (rf_def_id < 0 || rf_def_id >= desc->num_unique_rfs)
            {
                /* Malformed rf_def_id — emit OTHER */
                row->type = PULSEG_SEQ_EVENT_OTHER;
                row->timestamp_us = blk_start[i];
                continue;
            }

            rfdef = &desc->rf_definitions[rf_def_id];

            /* RF isocenter time = block_start + delay + duration - isodelay */
            row->timestamp_us = blk_start[i] + (float)rfdef->delay + rfdef->stats.duration_us -
                (float)rfdef->stats.isodelay_us;
            row->type = PULSEG_SEQ_EVENT_RF;
            row->params[0] = (float)rf_def_id;
            row->params[1] = (float)rfe->rf_use;
            row->params[2] = rfe->amplitude;           /* act_amplitude_hz — ppm already folded */
            row->params[3] = rfe->phase_offset;        /* phase_offset_rad — ppm already folded */
            row->params[4] = rfe->freq_offset;         /* freq_offset_hz   — ppm already folded */
            row->params[5] = (float)(bte->rf_shim_id); /* -1 if none */

            /* params[6]: amplitude (Hz/m) of the slice-selection gradient.
             * Valid only when exactly one gradient axis is nonzero and that
             * gradient is a trapezoid.  0.0 otherwise.                     */
            {
                float ss_amp = 0.0f;
                int n_active = 0;
                int ss_gt_id = -1; /* grad_table index of the SS gradient */

#define _CHECK_GRAD(bt_grad_id) \
    do \
    { \
        int _gt = (bt_grad_id); \
        if (_gt >= 0 && _gt < desc->grad_table_size) \
        { \
            int _gd = desc->grad_table[_gt].id; \
            if (_gd >= 0 && _gd < desc->num_unique_grads && \
                desc->grad_definitions[_gd].type == PULSEQ_GRAD_TRAP) \
            { \
                n_active++; \
                ss_gt_id = _gt; \
            } \
        } \
    } while (0)

                _CHECK_GRAD(bte->gx_id);
                _CHECK_GRAD(bte->gy_id);
                _CHECK_GRAD(bte->gz_id);
#undef _CHECK_GRAD

                if (n_active == 1)
                {
                    const pulseg_grad_table_element *gte = &desc->grad_table[ss_gt_id];
                    /* grad_table[].amplitude is the physical amplitude in Hz/m
                     * (from the Pulseq GRADIENTS/TRAP table, gamma-scaled).  */
                    ss_amp = gte->amplitude;
                    if (ss_amp < 0.0f)
                        ss_amp = -ss_amp;
                }
                row->params[6] = ss_amp;
            }
        }
        /* ---- ADC block ---- */
        else if (bte->adc_id >= 0 && bte->adc_id < desc->adc_table_size)
        {
            adc_def_id = desc->adc_table[bte->adc_id].id;

            if (adc_kzero[i] < 0.0f)
            {
                ret = PULSEG_ERR_INVALID_ARGUMENT;
                goto cleanup;
            }

            if (adc_def_id >= 0 && adc_def_id < desc->num_unique_adcs)
                adcdef = &desc->adc_definitions[adc_def_id];
            else
                adcdef = NULL;

            row->type = PULSEG_SEQ_EVENT_ADC;
            row->timestamp_us = adc_kzero[i];
            row->params[0] = (float)adc_roles[i];
            row->params[1] = adcdef ? desc->adc_table[bte->adc_id].phase_offset : 0.0f;
            row->params[2] = (float)adc_echoes[i];
        }
        /* ---- OTHER block (delay / gradient only) ---- */
        else
        {
            row->type = PULSEG_SEQ_EVENT_OTHER;
            row->timestamp_us = blk_start[i];
        }
    }

cleanup:
    if (blk_start)
        PULSEG_FREE(blk_start);
    if (adc_kzero)
        PULSEG_FREE(adc_kzero);
    if (adc_anchor_kxyz)
        PULSEG_FREE(adc_anchor_kxyz);
    if (adc_roles)
        PULSEG_FREE(adc_roles);
    if (adc_echoes)
        PULSEG_FREE(adc_echoes);
    if (ret != PULSEG_SUCCESS)
        pulseg_sequence_description_free(out);

    return ret;
}

/* ================================================================== */
/*  pulseg_get_sequence_parameters                                 */
/* ================================================================== */

int pulseg_get_sequence_parameters(pulseg_sequence_parameters *out, const pulseg_collection *coll)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_descriptor *trd;
    pulseg_scan_time_info st = PULSEG_SCAN_TIME_INFO_INIT;
    int ss;
    float fa_max = 0.0f;

    if (!out || !coll)
        return PULSEG_ERR_NULL_POINTER;
    memset(out, 0, sizeof(*out));

    out->min_te_us = 1e30f;
    out->min_tr_us = 1e30f;
    out->num_subseqs = coll->num_subsequences;
    if (pulseg_get_scan_time(coll, &st) == PULSEG_SUCCESS)
        out->total_scan_time_us = st.total_duration_us;

    for (ss = 0; ss < coll->num_subsequences; ++ss)
    {
        float *blk_start = NULL;
        float *adc_kzero = NULL;
        int *adc_roles = NULL;
        int n_walk_p, start_block, ip, rc;
        float pass_dur_us = 0.0f;
        desc = &coll->descriptors[ss];
        trd = &desc->tr_descriptor;

        seqdesc__select_canonical_window(desc, &start_block, &n_walk_p);

        blk_start = seqdesc__build_block_start_us(desc, start_block, n_walk_p);
        if (!blk_start)
            return PULSEG_ERR_ALLOC_FAILED;

        rc = seqdesc__build_adc_anchors_from_canonical(
            coll,
            &adc_kzero,
            &adc_roles,
            NULL,
            desc,
            ss,
            start_block,
            n_walk_p);
        if (rc != PULSEG_SUCCESS)
        {
            PULSEG_FREE(blk_start);
            return rc;
        }

        /* Pass duration */
        for (ip = 0; ip < n_walk_p; ++ip)
        {
            const pulseg_block_table_element *bte = &desc->block_table[start_block + ip];
            int dur =
                (bte->duration_us >= 0) ? bte->duration_us : desc->base_blocks[bte->id].duration_us;
            pass_dur_us += (float)dur;
        }

        if (pass_dur_us > 0.0f)
        {
            if (pass_dur_us < out->min_tr_us)
                out->min_tr_us = pass_dur_us;
            if (pass_dur_us > out->max_tr_us)
                out->max_tr_us = pass_dur_us;
        }

        /* TE: compute as RF excitation isocenter to ADC center time */
        {
            float last_exc_isocenter_us = -1e30f; /* last excitation RF isocenter time */
            for (ip = 0; ip < n_walk_p; ++ip)
            {
                const pulseg_block_table_element *bte = &desc->block_table[start_block + ip];
                float rf_isocenter_us;

                /* Process RF: track excitation isocenter times */
                if (bte->rf_id >= 0 && bte->rf_id < desc->rf_table_size)
                {
                    int rf_def_id = desc->rf_table[bte->rf_id].id;
                    int rf_use = desc->rf_table[bte->rf_id].rf_use;
                    if (rf_def_id >= 0 && rf_def_id < desc->num_unique_rfs)
                    {
                        const pulseg_rf_definition *rd = &desc->rf_definitions[rf_def_id];
                        double ratio_d = 1.0;
                        if (rf_use == PULSEG_RF_USE_EXCITATION)
                        {
                            rf_isocenter_us = blk_start[ip] + (float)rd->delay +
                                rd->stats.duration_us - (float)rd->stats.isodelay_us;
                            last_exc_isocenter_us = rf_isocenter_us;
                        }

                        if (rd->stats.base_amplitude_hz > 0.0f)
                        {
                            double amp = (double)desc->rf_table[bte->rf_id].amplitude;
                            if (amp < 0.0)
                                amp = -amp;
                            ratio_d = amp / (double)rd->stats.base_amplitude_hz;
                        }
                        {
                            double fa_d = (double)rd->stats.flip_angle_rad * ratio_d *
                                (180.0 / 3.14159265358979323846);
                            float fa = (float)fa_d;
                            if (fa > fa_max)
                                fa_max = fa;
                        }
                    }
                }

                /* Process ADC: compute TE from last excitation isocenter */
                if (bte->adc_id >= 0 && bte->adc_id < desc->adc_table_size &&
                    seqdesc__adc_role_is_te_bearing(adc_roles[ip]))
                {
                    if (adc_kzero[ip] < 0.0f)
                    {
                        PULSEG_FREE(adc_roles);
                        PULSEG_FREE(adc_kzero);
                        PULSEG_FREE(blk_start);
                        return PULSEG_ERR_INVALID_ARGUMENT;
                    }
                    if (last_exc_isocenter_us > 0.0f && adc_kzero[ip] > last_exc_isocenter_us)
                    {
                        float te_est = adc_kzero[ip] - last_exc_isocenter_us;
                        if (te_est < out->min_te_us)
                            out->min_te_us = te_est;
                    }
                }
            }
        }

        PULSEG_FREE(adc_roles);
        PULSEG_FREE(adc_kzero);
        PULSEG_FREE(blk_start);

        (void)trd;
    }

    if (out->min_te_us > 1e29f)
        out->min_te_us = 0.0f;
    if (out->min_tr_us > 1e29f)
        out->min_tr_us = 0.0f;
    out->max_flip_angle_deg = fa_max;

    return PULSEG_SUCCESS;
}
