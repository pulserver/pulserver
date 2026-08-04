/**
 * @file pulseg_safety.c
 * @brief Gradient safety gate: amplitude, slew, continuity, acoustic
 *        resonance and PNS.
 *
 * pulseg_check_safety() runs the whole suite and returns on the first
 * violation. The acoustic analysis is structural rather than simulated: it
 * derives the canonical TR's gradient spectrum analytically and evaluates
 * A_eq at guarded in-band harmonics -- see
 * docs/explanations/mechanical_resonance_safety.md for the model.
 *
 * PNS is delegated to a caller-supplied pulseg_pns_model, so no vendor
 * nerve-stimulation formula lives here. RF/SAR safety is deliberately out of
 * scope (vendor-proprietary).
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "pulseg_internal.h"
#include "pulseg.h"

#include "external_kiss_fft.h"
#include "external_kiss_fftr.h"

/* ================================================================== */
/*  Canonical-TR window selection                                     */
/* ================================================================== */

/* Helper: select the canonical TR window for a given canonical_tr_idx */
static void select_canonical_tr_window_idx(
    const struct pulseg_sequence_descriptor *desc,
    int *start_block,
    int *block_count,
    int *amplitude_mode,
    int *num_instances,
    float *tr_duration_us,
    int canonical_tr_idx)
{
    const struct pulseg_tr_descriptor *trd = &desc->tr_descriptor;
    int has_nd_prep = (trd->num_prep_blocks > 0 && !trd->degenerate_prep);
    int has_nd_cool = (trd->num_cooldown_blocks > 0 && !trd->degenerate_cooldown);

    if (has_nd_prep || has_nd_cool)
    {
        /* Non-degenerate: canonical_tr_idx selects pass.
         * Pass-expanded waveform already bakes in num_averages via
         * _build_pass_expanded_block_order; num_instances counts how
         * many times this pass-expanded waveform repeats. */
        int pass_len = desc->pass_len;
        *start_block = canonical_tr_idx * pass_len;
        *block_count = pass_len;
        *amplitude_mode = PULSEG_AMP_MAX_POS;
        *num_instances = (desc->num_passes > 1) ? desc->num_passes : 1;
        *tr_duration_us = 0.0f;
        {
            int n;
            for (n = 0; n < pass_len; ++n)
            {
                int idx;
                const struct pulseg_block_table_element *bte;
                const struct pulseg_base_block *bdef;
                idx = *start_block + n;
                bte = &desc->block_table[idx];
                bdef = &desc->base_blocks[bte->id];
                *tr_duration_us +=
                    (bte->duration_us >= 0) ? (float)bte->duration_us : (float)bdef->duration_us;
            }
        }
        return;
    }

    /* Degenerate: canonical_tr_idx selects imaging TR.
     * No pass expansion occurs, so averages must be accounted for here. */
    {
        int num_avgs = (desc->num_averages > 1) ? desc->num_averages : 1;
        *start_block =
            trd->num_prep_blocks + trd->imaging_tr_start + canonical_tr_idx * trd->tr_size;
        *block_count = trd->tr_size;
        *amplitude_mode = PULSEG_AMP_MAX_POS;
        *num_instances = trd->num_trs * num_avgs;
        *tr_duration_us = trd->tr_duration_us;
    }
}

static int build_pass_expanded_block_order(
    const struct pulseg_sequence_descriptor *desc,
    int **out_block_order,
    int *out_block_count,
    float *out_duration_us,
    int pass_base)
{
    const struct pulseg_tr_descriptor *trd;
    int prep_blk, img_len, cool_blk, num_avgs, exp_count;
    int *block_order;
    int avg_i, pos_i, n;

    if (!desc || !out_block_order || !out_block_count)
    {
        return PULSEG_ERR_INVALID_ARGUMENT;
    }

    trd = &desc->tr_descriptor;
    prep_blk = trd->num_prep_blocks;
    img_len = trd->num_trs * trd->tr_size;
    cool_blk = trd->num_cooldown_blocks;
    num_avgs = (desc->num_averages > 0) ? desc->num_averages : 1;
    exp_count = prep_blk + num_avgs * img_len + cool_blk;

    if (exp_count <= 0)
    {
        return PULSEG_ERR_TR_NO_BLOCKS;
    }

    block_order = (int *)PULSEG_ALLOC((size_t)exp_count * sizeof(int));
    if (!block_order)
    {
        return PULSEG_ERR_ALLOC_FAILED;
    }

    n = 0;
    for (pos_i = 0; pos_i < prep_blk; ++pos_i)
        block_order[n++] = pass_base + pos_i;
    for (avg_i = 0; avg_i < num_avgs; ++avg_i)
        for (pos_i = 0; pos_i < img_len; ++pos_i)
            block_order[n++] = pass_base + prep_blk + pos_i;
    for (pos_i = 0; pos_i < cool_blk; ++pos_i)
        block_order[n++] = pass_base + prep_blk + img_len + pos_i;

    if (out_duration_us)
    {
        *out_duration_us = 0.0f;
        for (pos_i = 0; pos_i < exp_count; ++pos_i)
        {
            int blk_idx;
            const struct pulseg_block_table_element *bte;
            const struct pulseg_base_block *bdef;

            blk_idx = block_order[pos_i];
            if (blk_idx < 0 || blk_idx >= desc->num_blocks)
            {
                PULSEG_FREE(block_order);
                return PULSEG_ERR_INVALID_ARGUMENT;
            }
            bte = &desc->block_table[blk_idx];
            bdef = &desc->base_blocks[bte->id];
            *out_duration_us +=
                (bte->duration_us >= 0) ? (float)bte->duration_us : (float)bdef->duration_us;
        }
    }

    *out_block_order = block_order;
    *out_block_count = exp_count;
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Acoustic spectra free                                             */
/* ================================================================== */

void pulseg_mech_resonances_spectra_free(pulseg_mech_resonances_spectra *s)
{
    if (!s)
        return;

    if (s->spectrum_full_gx)
        PULSEG_FREE(s->spectrum_full_gx);
    if (s->spectrum_full_gy)
        PULSEG_FREE(s->spectrum_full_gy);
    if (s->spectrum_full_gz)
        PULSEG_FREE(s->spectrum_full_gz);

    if (s->analytical_peak_freqs)
        PULSEG_FREE(s->analytical_peak_freqs);
    if (s->analytical_peak_amp_gx)
        PULSEG_FREE(s->analytical_peak_amp_gx);
    if (s->analytical_peak_amp_gy)
        PULSEG_FREE(s->analytical_peak_amp_gy);
    if (s->analytical_peak_amp_gz)
        PULSEG_FREE(s->analytical_peak_amp_gz);
    if (s->analytical_peak_phase_gx)
        PULSEG_FREE(s->analytical_peak_phase_gx);
    if (s->analytical_peak_phase_gy)
        PULSEG_FREE(s->analytical_peak_phase_gy);
    if (s->analytical_peak_phase_gz)
        PULSEG_FREE(s->analytical_peak_phase_gz);
    if (s->analytical_peak_widths_hz)
        PULSEG_FREE(s->analytical_peak_widths_hz);
    if (s->candidate_freqs)
        PULSEG_FREE(s->candidate_freqs);
    if (s->candidate_amps_gx)
        PULSEG_FREE(s->candidate_amps_gx);
    if (s->candidate_amps_gy)
        PULSEG_FREE(s->candidate_amps_gy);
    if (s->candidate_amps_gz)
        PULSEG_FREE(s->candidate_amps_gz);
    if (s->candidate_grad_amps)
        PULSEG_FREE(s->candidate_grad_amps);
    if (s->candidate_grad_amps_gx)
        PULSEG_FREE(s->candidate_grad_amps_gx);
    if (s->candidate_grad_amps_gy)
        PULSEG_FREE(s->candidate_grad_amps_gy);
    if (s->candidate_grad_amps_gz)
        PULSEG_FREE(s->candidate_grad_amps_gz);
    if (s->candidate_violations)
        PULSEG_FREE(s->candidate_violations);
    if (s->component_freqs_hz)
        PULSEG_FREE(s->component_freqs_hz);
    if (s->component_amps)
        PULSEG_FREE(s->component_amps);
    if (s->component_phases_rad)
        PULSEG_FREE(s->component_phases_rad);
    if (s->component_widths_hz)
        PULSEG_FREE(s->component_widths_hz);
    if (s->component_axes)
        PULSEG_FREE(s->component_axes);
    if (s->component_def_ids)
        PULSEG_FREE(s->component_def_ids);
    if (s->component_contrib_ids)
        PULSEG_FREE(s->component_contrib_ids);
    if (s->component_run_ids)
        PULSEG_FREE(s->component_run_ids);
    if (s->surviving_freqs_hz)
        PULSEG_FREE(s->surviving_freqs_hz);
    if (s->envelope_freqs_hz)
        PULSEG_FREE(s->envelope_freqs_hz);
    if (s->envelope_amp_gx)
        PULSEG_FREE(s->envelope_amp_gx);
    if (s->envelope_amp_gy)
        PULSEG_FREE(s->envelope_amp_gy);
    if (s->envelope_amp_gz)
        PULSEG_FREE(s->envelope_amp_gz);

    memset(s, 0, sizeof(*s));
}

/* ================================================================== */
/*  Structural acoustic analysis — data types                         */
/* ================================================================== */

/** Max vertices in piecewise-linear (trap / few-vertex arb) waveform envelope. */
#define SA_MAX_PWL_VERTICES 16

/* --- Equivalent-sustained-drive (A_eq) mechanical-resonance criterion ---
 * (docs/explanations/mechanical_resonance_safety.md).  Sharp-line model: the sequence drive is the
 * Fourier series of the canonical (outer) TR, sampled at the TR harmonics
 * k / T_TR that fall inside a guarded forbidden band.  The per-axis
 * equivalent-sustained amplitude of a spectral line is
 *   A_eq(f_L) = (2 / T_TR) * |sum_events amp * transform(f_L)
 *                              * e^{-j2 pi f_L t_event} * D_reps(f_L)|,
 * the amplitude of the pure sinusoidal gradient delivering the same sustained
 * on-resonance drive.  Inner periodicities (echo train, slices) are NOT known
 * explicitly: they emerge from the coherent sum of the individual event
 * instances materialised inside the canonical TR (the only period we know). */

/** Hardware-anchored readout-scale floor for epsilon, used ONLY when the
 *  vendor band is literal zero-tolerance: eps = SA_AEQ_K_GMAX * G_max
 *  (~4 mT/m at G_max = 50 mT/m). Any nonzero vendor limit is trusted as-is,
 *  even below this floor. The "is-this-a-readout" filter that generalises
 *  the vendor's single-frequency zero-tolerance to arbitrary sequences:
 *  eps = 0 is unusable (every GRE has a weak harmonic in any band wider
 *  than its comb spacing). */
#define SA_AEQ_K_GMAX 0.08f

/** Frequency guard multiplier on the resonance HWHM.  guard = mult * HWHM,
 *  HWHM = min_band_width / 2 (the narrowest band is the sharpest resonance the
 *  vendor identified; wide bands are keep-out ranges).  A TR-harmonic line
 *  counts against a band iff it lies within [f_min - guard, f_max + guard]. */
#define SA_GUARD_HWHM_MULT 1.0f

/** Minimum upper frequency (Hz) for the display/analytical spectrum grid,
 * decoupled from the forbidden-band range so the plotted A_eq comb reaches the
 * validated grad_spectrum default (3000 Hz). The A_eq verdict itself only ever
 * evaluates TR-harmonic lines inside a guarded band, so this bound never widens
 * what can be flagged; it only sets how far the display grid is computed. */
#define SA_MIN_ANALYSIS_FREQ_HZ 3000.0f

/** Finite-outer-rep fix (docs/explanations/mechanical_resonance_safety.md, "Stage 4"): fixed
 *  point budget (per side, i.e. per coarse TR harmonic there are
 *  2*SA_MECHRES_MAX_DM_SUBDIV/2 extra samples total) for probing the
 *  outer-repeat (M = num_instances) Dirichlet comb's sidelobes, which live
 *  near each of the two adjacent exact TR harmonics. M itself is NOT the
 *  cost driver: this budget is a fixed constant-factor overhead per coarse
 *  candidate, independent of how large M is (geometric spacing resolves
 *  the near-lobe region with a fixed point count regardless of M -- see
 *  the sample-placement comment at the sub-loop itself). Each sample DOES
 *  require a fresh sa_eval_axis_spectrum call (reusing the coarse point's
 *  value is a no-op, not a cache hit -- see PLAN doc "Bug 1"), which is
 *  what the sa_eval_pwl_transform memoization below amortizes. */
#define SA_MECHRES_MAX_DM_SUBDIV 16

/** Segments between exact sin/cos re-anchors of the uniform-raster phase
 *  recurrence in sa_eval_pwl_transform(). Bounds accumulated rotation drift
 *  to a fixed budget no matter how many samples an arbitrary waveform has,
 *  at a cost of 2 transcendentals per this many segments (< 1% of the
 *  general path's 4-per-segment). */
#define SA_PWL_REANCHOR 256

/** Largest number of occurrences-per-repeat that sa_compress_axis_events()
 *  will try when looking for interleaved equally-spaced trains of one
 *  gradient definition (e.g. 2 for a prephaser/rewinder pair sharing a
 *  definition, or an echo-train length for a definition reused once per
 *  echo). Only bounds how much compression is FOUND -- never correctness:
 *  a structure wider than this simply falls back to consecutive run-length
 *  encoding. The search is O(occurrences * this) once per axis. */
#define SA_MAX_TRAIN_STRIDE 32

/**
 * A gradient event within the canonical TR.
 * Each event is a single waveform (or sub-event from decomposed arbitrary)
 * played at a specific time with a specific amplitude.
 */
typedef struct
{
    int def_id;                              /**< gradient definition id                */
    int def_index;                           /**< index into grad_definitions[]         */
    double start_time_us;                    /**< event start time within the TR (us)   */
    float amplitude;                         /**< worst-case positional amplitude (Hz/m), signed */
    int pwl_num_vertices;                    /**< >0 -> use piecewise-linear W(f)       */
    float pwl_times_us[SA_MAX_PWL_VERTICES]; /**< vertex times (us from event start) */
    float pwl_values[SA_MAX_PWL_VERTICES];   /**< vertex amplitudes (normalised)     */
    int num_reps;                            /**< repetition count (1=once, N=imaging repeat) */
    double rep_period_us;                    /**< repetition period in us (0 if num_reps==1) */
    /* Occurrence-train compression (sa_compress_axis_events).  A train stands
     * for train_len occurrences of the SAME definition at start_time_us +
     * j*train_period_us, j = 0..train_len-1.  train_len == 1 is a plain
     * single occurrence and every field below is inert. */
    int train_len;          /**< occurrences fused into this event (1 = none)  */
    double train_period_us; /**< uniform spacing of the train (0 if len == 1)  */
    float *train_amps;      /**< [train_len] per-occurrence amplitudes, or NULL
                                 when every occurrence has `amplitude` (the
                                 constant-amplitude case, evaluated in closed
                                 form instead of term by term) */
    /* Raw-sample model (for many-sample arb without sub-period): the true
     * complex event transform is evaluated by direct demodulation of the
     * samples at cell centres.  arb_num_samples == 0 -> use the PWL model. */
    float *arb_samples;  /**< normalised amplitudes at cell centres [arb_num_samples] */
    float *arb_times_us; /**< sample times (us from event start) [arb_num_samples]    */
    int arb_num_samples; /**< raw sample count (0 = not used, use PWL)                */
} sa_event;

/** Per-axis event list. */
typedef struct
{
    int num_events;
    sa_event *events; /**< allocated [num_events] */
} sa_axis_events;

/**
 * W_k(f) memoization (docs/explanations/mechanical_resonance_safety.md, "Stage 4"): caches
 * sa_eval_event_transform()'s result -- the base-waveform Fourier response,
 * a pure function of (definition shape, frequency) -- keyed by def_id, for
 * the duration of ONE sa_eval_axis_spectrum() call (which is itself already
 * scoped to a single fixed frequency, so def_id alone is a sufficient key;
 * no separate frequency field needed). Multiple sa_event occurrences that
 * share a def_id (the common case: a handful of unique gradient shapes
 * reused across many materialized occurrences) hit this cache instead of
 * repeating the O(vertices) sa_eval_pwl_transform integral. NULL disables
 * caching (used at call sites outside the hot per-candidate-frequency
 * loop, where memoizing a single lookup isn't worth the bookkeeping).
 */
typedef struct
{
    int def_id;
    float re, im;
} sa_transform_cache_entry;

typedef struct
{
    sa_transform_cache_entry *entries;
    int count;
    int capacity;
} sa_transform_cache;

/** Linear scan: def_id counts per axis are small (a handful of unique
 *  gradient shapes even for a long/complex hyper-TR, per the design doc's
 *  own cost model), so a hash table would be overhead without benefit. */
static int sa_transform_cache_lookup(
    const sa_transform_cache *cache,
    float *out_re,
    float *out_im,
    int def_id)
{
    int i;
    if (!cache)
        return 0;
    for (i = 0; i < cache->count; ++i)
    {
        if (cache->entries[i].def_id == def_id)
        {
            *out_re = cache->entries[i].re;
            *out_im = cache->entries[i].im;
            return 1;
        }
    }
    return 0;
}

static void sa_transform_cache_insert(sa_transform_cache *cache, int def_id, float re, float im)
{
    if (!cache || cache->count >= cache->capacity)
        return; /* cache disabled or full: caller just recomputes next time */
    cache->entries[cache->count].def_id = def_id;
    cache->entries[cache->count].re = re;
    cache->entries[cache->count].im = im;
    cache->count++;
}

/** Structural analysis: event lists for all three axes. */
typedef struct
{
    sa_axis_events axes[3]; /**< 0=gx, 1=gy, 2=gz */
} sa_structural_events;

static void sa_free_structural_events(sa_structural_events *se)
{
    int ax, k;
    if (!se)
        return;
    for (ax = 0; ax < 3; ++ax)
    {
        if (se->axes[ax].events)
        {
            for (k = 0; k < se->axes[ax].num_events; ++k)
            {
                if (se->axes[ax].events[k].arb_samples)
                    PULSEG_FREE(se->axes[ax].events[k].arb_samples);
                if (se->axes[ax].events[k].arb_times_us)
                    PULSEG_FREE(se->axes[ax].events[k].arb_times_us);
                if (se->axes[ax].events[k].train_amps)
                    PULSEG_FREE(se->axes[ax].events[k].train_amps);
            }
            PULSEG_FREE(se->axes[ax].events);
        }
        se->axes[ax].events = NULL;
        se->axes[ax].num_events = 0;
    }
}

/* ================================================================== */
/*  Structural acoustic analysis — occurrence extraction              */
/* ================================================================== */

/**
 * Extract grad-def occurrences for one axis within the canonical TR.
 * Collects (def_id, def_index, start_time_us, amplitude) for each block
 * that has a gradient event on the given axis.
 * @param axis  0=gx, 1=gy, 2=gz
 * Returns number of occurrences, or -1 on allocation failure.
 * Caller must free *out_events.
 */
typedef struct
{
    int def_id;
    int def_index;
    double start_time_us;
    float amplitude;
} sa_raw_occurrence;

static int sa_extract_raw_occurrences(
    sa_raw_occurrence **out_occ,
    const struct pulseg_sequence_descriptor *desc,
    int start_block,
    int block_count,
    int axis)
{
    int i, n, cap;
    /* Accumulated in double, not float: block durations are whole
     * microseconds, so a double running sum is EXACT across a hyper-TR of
     * millions of us, whereas a float sum quantises to 0.5 us past ~7 s and
     * jitters the event start times. Exactness matters twice over -- it
     * removes that jitter from the e^{-j 2 pi f t} phase, and it lets
     * sa_compress_axis_events() recognise a constant occurrence spacing by
     * near-exact comparison instead of a loose tolerance. */
    double time_us;
    sa_raw_occurrence *occ;

    cap = (block_count > 0) ? block_count : 16;
    occ = (sa_raw_occurrence *)PULSEG_ALLOC((size_t)cap * sizeof(sa_raw_occurrence));
    if (!occ)
        return -1;
    n = 0;
    time_us = 0.0;

    for (i = 0; i < block_count; ++i)
    {
        const struct pulseg_block_table_element *bte = &desc->block_table[start_block + i];
        const struct pulseg_base_block *bdef = &desc->base_blocks[bte->id];
        int grad_table_idx;
        double blk_dur;

        switch (axis)
        {
        case 0:
            grad_table_idx = bte->gx_id;
            break;
        case 1:
            grad_table_idx = bte->gy_id;
            break;
        case 2:
            grad_table_idx = bte->gz_id;
            break;
        default:
            grad_table_idx = -1;
            break;
        }

        if (grad_table_idx >= 0)
        {
            const struct pulseg_grad_table_element *gte = &desc->grad_table[grad_table_idx];
            if (n >= cap)
            {
                sa_raw_occurrence *tmp;
                cap *= 2;
                tmp = (sa_raw_occurrence *)PULSEG_ALLOC((size_t)cap * sizeof(sa_raw_occurrence));
                if (!tmp)
                {
                    PULSEG_FREE(occ);
                    return -1;
                }
                memcpy(tmp, occ, (size_t)n * sizeof(sa_raw_occurrence));
                PULSEG_FREE(occ);
                occ = tmp;
            }
            occ[n].def_id = gte->id;
            occ[n].def_index = gte->id;
            occ[n].start_time_us = time_us;
            occ[n].amplitude = gte->amplitude;
            ++n;
        }

        blk_dur = (bte->duration_us >= 0) ? (double)bte->duration_us : (double)bdef->duration_us;
        time_us += blk_dur;
    }

    *out_occ = occ;
    return n;
}

/* ================================================================== */
/*  Structural acoustic analysis — build events per axis              */
/* ================================================================== */

static int sa_compare_int(const void *a, const void *b)
{
    return (*(const int *)a) - (*(const int *)b);
}

/**
 * Build event list for one axis from raw occurrences.
 *
 * For each unique def_id:
 *   - Trapezoid: one event per occurrence, PWL from rise/flat/fall.
 *   - Arbitrary with few samples: one event per occurrence, PWL from vertices.
 *   - Arbitrary with many samples: one event per occurrence carrying the raw
 *     samples; the exact complex transform is demodulated directly at each
 *     in-band line (no template/sub-period assumption).
 */
static int sa_build_axis_events(
    sa_axis_events *ae,
    const sa_raw_occurrence *occ,
    int num_occ,
    const struct pulseg_sequence_descriptor *desc)
{
    int i, j, n_events, cap, did, idx, s_idx, nv;
    int *unique_ids;
    int num_unique;
    float raster;

    ae->num_events = 0;
    ae->events = NULL;
    if (num_occ <= 0)
        return PULSEG_SUCCESS;

    raster = desc->grad_raster_us;

    /* Collect unique def_ids */
    unique_ids = (int *)PULSEG_ALLOC((size_t)num_occ * sizeof(int));
    if (!unique_ids)
        return PULSEG_ERR_ALLOC_FAILED;
    for (i = 0; i < num_occ; ++i)
        unique_ids[i] = occ[i].def_id;
    qsort(unique_ids, (size_t)num_occ, sizeof(int), sa_compare_int);
    num_unique = 1;
    for (i = 1; i < num_occ; ++i)
        if (unique_ids[i] != unique_ids[i - 1])
            unique_ids[num_unique++] = unique_ids[i];

    /* Initial allocation — may grow for decomposed arbitraries */
    cap = num_occ * 2;
    ae->events = (sa_event *)PULSEG_ALLOC((size_t)cap * sizeof(sa_event));
    if (!ae->events)
    {
        PULSEG_FREE(unique_ids);
        return PULSEG_ERR_ALLOC_FAILED;
    }
    n_events = 0;

    for (idx = 0; idx < num_unique; ++idx)
    {
        const struct pulseg_grad_definition *gdef;
        /* Shared PWL template (trap / few-vertex arb) */
        int pwl_nv;
        float pwl_t[SA_MAX_PWL_VERTICES];
        float pwl_v[SA_MAX_PWL_VERTICES];
        /* Shared raw-sample store (many-sample arb): the exact complex event
         * transform is demodulated directly from these samples at their
         * cell-centre times. */
        float *shared_arb_samples;
        float *shared_arb_times;
        int shared_arb_n;
        int use_arb;

        did = unique_ids[idx];
        gdef = NULL;
        pwl_nv = 0;
        shared_arb_samples = NULL;
        shared_arb_times = NULL;
        shared_arb_n = 0;
        use_arb = 0;

        /* Find first occurrence of this def to get gdef */
        for (j = 0; j < num_occ; ++j)
        {
            if (occ[j].def_id == did)
            {
                gdef = &desc->grad_definitions[occ[j].def_index];
                break;
            }
        }
        if (!gdef)
            continue;

        /* --- Compute shared PWL parameters for this def --- */

        if (gdef->type == 0)
        {
            /* Trapezoid: 4-vertex PWL */
            float tr = (float)gdef->rise_time_or_unused;
            float tf = (float)gdef->flat_time_or_unused;
            float td = (float)gdef->fall_time_or_num_uncompressed_samples;
            pwl_nv = 4;
            pwl_t[0] = 0.0f;
            pwl_v[0] = 0.0f;
            pwl_t[1] = tr;
            pwl_v[1] = 1.0f;
            pwl_t[2] = tr + tf;
            pwl_v[2] = 1.0f;
            pwl_t[3] = tr + tf + td;
            pwl_v[3] = 0.0f;
        }
        else if (gdef->type == 1)
        {
            /* Arbitrary waveform */
            int shape_id = gdef->shot_shape_ids[0];
            int time_shape_id = gdef->unused_or_time_shape_id;

            if (shape_id > 0 && shape_id <= desc->num_shapes)
            {
                pulseq_shape decomp_wave;
                decomp_wave.samples = NULL;
                decomp_wave.num_samples = 0;
                decomp_wave.num_uncompressed_samples = 0;

                if (pulseq_decompress_shape(&decomp_wave, &desc->shapes[shape_id - 1], 1.0f) &&
                    decomp_wave.num_uncompressed_samples > 0)
                {
                    int num_samp = decomp_wave.num_uncompressed_samples;
                    float *wave_samp = decomp_wave.samples;

                    /* Decompress time shape if present */
                    pulseq_shape decomp_time;
                    float *time_us = NULL;
                    int has_time = 0;
                    decomp_time.samples = NULL;
                    decomp_time.num_samples = 0;
                    decomp_time.num_uncompressed_samples = 0;

                    if (time_shape_id > 0 && time_shape_id <= desc->num_shapes)
                    {
                        if (pulseq_decompress_shape(
                                &decomp_time,
                                &desc->shapes[time_shape_id - 1],
                                raster) &&
                            decomp_time.num_uncompressed_samples > 0)
                        {
                            has_time = 1;
                            time_us = decomp_time.samples;
                        }
                    }

                    if (num_samp < PULSEG_MIN_ARBITRARY_SAMPLES)
                    {
                        /* Few-sample arb: PWL from vertices */
                        nv = num_samp;
                        if (nv > SA_MAX_PWL_VERTICES)
                            nv = SA_MAX_PWL_VERTICES;
                        pwl_nv = nv;
                        for (s_idx = 0; s_idx < nv; ++s_idx)
                        {
                            if (has_time && time_us)
                                pwl_t[s_idx] = time_us[s_idx];
                            else
                                pwl_t[s_idx] = 0.5f * raster + (float)s_idx * raster;
                            pwl_v[s_idx] = wave_samp[s_idx];
                        }
                    }
                    else
                    {
                        /* Many-sample arb: keep the raw samples and their
                         * cell-centre times so the exact complex transform is
                         * demodulated directly at each (sparse) in-band line.
                         * This is exact for onset/offset envelopes and multi-
                         * period structure alike — no template/sub-period
                         * assumption.  FAIL CLOSED on OOM. */
                        int m;
                        shared_arb_samples =
                            (float *)PULSEG_ALLOC((size_t)num_samp * sizeof(float));
                        shared_arb_times = (float *)PULSEG_ALLOC((size_t)num_samp * sizeof(float));
                        if (!shared_arb_samples || !shared_arb_times)
                        {
                            if (shared_arb_samples)
                                PULSEG_FREE(shared_arb_samples);
                            if (shared_arb_times)
                                PULSEG_FREE(shared_arb_times);
                            if (decomp_time.samples)
                                PULSEG_FREE(decomp_time.samples);
                            if (decomp_wave.samples)
                                PULSEG_FREE(decomp_wave.samples);
                            PULSEG_FREE(unique_ids);
                            PULSEG_FREE(ae->events);
                            ae->events = NULL;
                            return PULSEG_ERR_ALLOC_FAILED;
                        }
                        for (m = 0; m < num_samp; ++m)
                        {
                            shared_arb_samples[m] = wave_samp[m];
                            if (has_time && time_us)
                                shared_arb_times[m] = time_us[m];
                            else
                                shared_arb_times[m] = 0.5f * raster + (float)m * raster;
                        }
                        shared_arb_n = num_samp;
                        use_arb = 1;
                        pwl_nv = 0;
                    }

                    if (decomp_time.samples)
                        PULSEG_FREE(decomp_time.samples);
                }
                if (decomp_wave.samples)
                    PULSEG_FREE(decomp_wave.samples);
            }
        }

        /* --- Emit events for each occurrence of this def --- */
        for (j = 0; j < num_occ; ++j)
        {
            if (occ[j].def_id != did)
                continue;

            {
                /* Single event (trap / few-vertex arb PWL / many-sample arb) */
                if (n_events >= cap)
                {
                    sa_event *tmp;
                    cap *= 2;
                    tmp = (sa_event *)PULSEG_ALLOC((size_t)cap * sizeof(sa_event));
                    if (!tmp)
                    {
                        if (shared_arb_samples)
                            PULSEG_FREE(shared_arb_samples);
                        if (shared_arb_times)
                            PULSEG_FREE(shared_arb_times);
                        PULSEG_FREE(unique_ids);
                        PULSEG_FREE(ae->events);
                        ae->events = NULL;
                        return PULSEG_ERR_ALLOC_FAILED;
                    }
                    memcpy(tmp, ae->events, (size_t)n_events * sizeof(sa_event));
                    PULSEG_FREE(ae->events);
                    ae->events = tmp;
                }
                ae->events[n_events].def_id = did;
                ae->events[n_events].def_index = occ[j].def_index;
                ae->events[n_events].start_time_us = occ[j].start_time_us + (double)gdef->delay;
                ae->events[n_events].amplitude = occ[j].amplitude;
                ae->events[n_events].train_len = 1;
                ae->events[n_events].train_period_us = 0.0;
                ae->events[n_events].train_amps = NULL;
                if (use_arb && shared_arb_samples && shared_arb_times)
                {
                    float *smp_copy = (float *)PULSEG_ALLOC((size_t)shared_arb_n * sizeof(float));
                    float *tim_copy = (float *)PULSEG_ALLOC((size_t)shared_arb_n * sizeof(float));
                    if (!smp_copy || !tim_copy)
                    {
                        if (smp_copy)
                            PULSEG_FREE(smp_copy);
                        if (tim_copy)
                            PULSEG_FREE(tim_copy);
                        PULSEG_FREE(shared_arb_samples);
                        PULSEG_FREE(shared_arb_times);
                        PULSEG_FREE(unique_ids);
                        PULSEG_FREE(ae->events);
                        ae->events = NULL;
                        return PULSEG_ERR_ALLOC_FAILED;
                    }
                    memcpy(smp_copy, shared_arb_samples, (size_t)shared_arb_n * sizeof(float));
                    memcpy(tim_copy, shared_arb_times, (size_t)shared_arb_n * sizeof(float));
                    ae->events[n_events].arb_samples = smp_copy;
                    ae->events[n_events].arb_times_us = tim_copy;
                    ae->events[n_events].arb_num_samples = shared_arb_n;
                    ae->events[n_events].pwl_num_vertices = 0;
                }
                else
                {
                    ae->events[n_events].pwl_num_vertices = pwl_nv;
                    if (pwl_nv > 0)
                    {
                        memcpy(
                            ae->events[n_events].pwl_times_us,
                            pwl_t,
                            (size_t)pwl_nv * sizeof(float));
                        memcpy(
                            ae->events[n_events].pwl_values,
                            pwl_v,
                            (size_t)pwl_nv * sizeof(float));
                    }
                    ae->events[n_events].arb_samples = NULL;
                    ae->events[n_events].arb_times_us = NULL;
                    ae->events[n_events].arb_num_samples = 0;
                }
                n_events++;
            }
        }
        /* Free shared raw-sample template for this def_id */
        if (shared_arb_samples)
        {
            PULSEG_FREE(shared_arb_samples);
            shared_arb_samples = NULL;
        }
        if (shared_arb_times)
        {
            PULSEG_FREE(shared_arb_times);
            shared_arb_times = NULL;
        }
    }

    ae->num_events = n_events;
    PULSEG_FREE(unique_ids);
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Structural acoustic analysis — occurrence-train compression       */
/* ================================================================== */

/**
 * May occurrences @p a and @p b belong to the same compressed train?
 *
 * Same definition (hence same base waveform and same W_d(f)) and same outer
 * repetition tagging.  Including the (num_reps, rep_period_us) pair in the
 * key is what stops a train from straddling the prep / imaging / cooldown
 * regions that the NEX tagging distinguishes: events in different regions
 * carry different tags and therefore never fuse.
 */
static int sa_train_compatible(const sa_event *a, const sa_event *b)
{
    return a->def_id == b->def_id && a->def_index == b->def_index &&
           a->num_reps == b->num_reps && a->rep_period_us == b->rep_period_us &&
           a->train_len == 1 && b->train_len == 1;
}

/**
 * Fuse runs of equally-spaced occurrences of one definition into single
 * "train" events.
 *
 * The A_eq sum over one axis is
 *   S_ax(f) = sum_k A_k W_{d(k)}(f) e^{-j 2 pi f t_k},
 * and W_d(f) is already memoized per definition (sa_transform_cache).  What
 * is left, and what dominates for any sequence with many repeats, is one
 * e^{-j 2 pi f t_k} -- a sin/cos pair -- per occurrence per frequency.  This
 * pass attacks that term structurally.
 *
 * Method: take each definition's occurrences in time order, differentiate to
 * get the spacings, and describe them as interleaved arithmetic progressions.
 * A train of length L starting at t_0 with spacing D contributes
 *   W_d(f) e^{-j 2 pi f t_0} * sum_{j<L} A_j z^j,   z = e^{-j 2 pi f D},
 * which sa_eval_event_spectrum() evaluates in closed form (constant A: a
 * Dirichlet/geometric sum, O(1)) or by Horner (varying A: L complex
 * multiply-adds and a SINGLE transcendental, instead of L of them).
 *
 * Consecutive differencing alone is not enough, because a definition used
 * more than once per repeat interleaves its occurrences: a gy prephaser and
 * its rewinder are usually the SAME definition at two different amplitudes,
 * so the spacing alternates short/long and no run longer than 2 exists.  The
 * scan therefore tries strides m = 1..SA_MAX_TRAIN_STRIDE and takes the
 * smallest m for which the occurrence list splits into m subsequences that
 * each have constant stride-m spacing -- i.e. m occurrences per repeat.  m
 * is discovered, never assumed; the repeat period is whatever the spacing
 * turns out to be, and each emitted train is an exact arithmetic
 * progression, so a wrong guess cannot produce a wrong answer, only a
 * missed compression.
 *
 * Nothing about the TR's internal layout is assumed -- no period is named,
 * no nesting is detected, no shape is special-cased.  An irregular order
 * such as [0,1,2,3,1,2,3] or [0,1,1,2,3,1,1,2,3] compresses as well as a
 * regular one, because what is scanned is the per-DEFINITION spacing, whose
 * entropy is far below that of the block sequence.  A pathological TR in
 * which no definition's occurrences are equally spaced at any stride falls
 * back to consecutive run-length encoding and, failing that, to trains of
 * length 1 -- i.e. exactly the pre-compression event list, at the cost of
 * one bounded scan.
 *
 * Only ever applied when the caller wants no display products: the
 * per-event component export reports one term per materialised occurrence,
 * which a train deliberately no longer is.
 */
/**
 * Does [beg,end) split into @p m interleaved arithmetic progressions, taken
 * with stride @p m?  Requires at least two occurrences per progression, so a
 * "split" always describes strictly fewer trains than occurrences.
 */
static int sa_stride_splits(const sa_event *ev, int beg, int end, int m)
{
    int total = end - beg;
    int phase;

    if (m < 1 || total < 2 * m || (total % m) != 0)
        return 0;

    for (phase = 0; phase < m; ++phase)
    {
        int first = beg + phase;
        double dt = ev[first + m].start_time_us - ev[first].start_time_us;
        double tol;
        int q;

        if (!(dt > 0.0))
            return 0;
        tol = 1.0e-9 * dt;
        for (q = first + m; q + m < end; q += m)
        {
            double d = ev[q + m].start_time_us - ev[q].start_time_us;
            if (fabs(d - dt) > tol)
                return 0;
        }
    }
    return 1;
}

/**
 * Emit one train covering occurrences beg, beg+stride, ... (@p len of them).
 * Returns 0 on allocation failure, leaving @p out untouched.
 */
static int sa_emit_train(
    sa_event *out,
    const sa_event *ev,
    int beg,
    int stride,
    int len)
{
    int j, varying = 0;

    /* Shallow copy: the head occurrence keeps ownership of its arb arrays. */
    *out = ev[beg];
    if (len <= 1)
        return 1;

    for (j = 1; j < len; ++j)
        if (ev[beg + j * stride].amplitude != ev[beg].amplitude)
        {
            varying = 1;
            break;
        }

    out->train_len = len;
    out->train_period_us = ev[beg + stride].start_time_us - ev[beg].start_time_us;
    if (varying)
    {
        float *amps = (float *)PULSEG_ALLOC((size_t)len * sizeof(float));
        if (!amps)
            return 0;
        for (j = 0; j < len; ++j)
            amps[j] = ev[beg + j * stride].amplitude;
        out->train_amps = amps;
    }
    return 1;
}

static int sa_compress_axis_events(sa_axis_events *ae)
{
    int i, j, n_out;
    sa_event *out;
    char *absorbed;

    if (!ae->events || ae->num_events < 2)
        return PULSEG_SUCCESS;

    out = (sa_event *)PULSEG_ALLOC((size_t)ae->num_events * sizeof(sa_event));
    absorbed = (char *)PULSEG_ALLOC((size_t)ae->num_events);
    if (!out || !absorbed)
    {
        if (out)
            PULSEG_FREE(out);
        if (absorbed)
            PULSEG_FREE(absorbed);
        return PULSEG_ERR_ALLOC_FAILED;
    }
    memset(absorbed, 0, (size_t)ae->num_events);

    n_out = 0;
    i = 0;
    while (i < ae->num_events)
    {
        /* Maximal class of occurrences that may share a train at all. */
        int cls_end = i + 1;
        int m, m_best;

        while (cls_end < ae->num_events &&
               sa_train_compatible(&ae->events[i], &ae->events[cls_end]))
            ++cls_end;

        /* Smallest stride that describes the whole class as m interleaved
         * arithmetic progressions (m = occurrences of this definition per
         * repeat).  m == 1 is the plain equally-spaced case. */
        m_best = 0;
        for (m = 1; m <= SA_MAX_TRAIN_STRIDE && 2 * m <= cls_end - i; ++m)
            if (sa_stride_splits(ae->events, i, cls_end, m))
            {
                m_best = m;
                break;
            }

        if (m_best > 0)
        {
            int len = (cls_end - i) / m_best;
            for (m = 0; m < m_best; ++m)
            {
                if (!sa_emit_train(&out[n_out], ae->events, i + m, m_best, len))
                    goto oom;
                n_out++;
            }
            /* Heads are i .. i+m_best-1; everything after them is folded in. */
            for (j = i + m_best; j < cls_end; ++j)
                absorbed[j] = 1;
            i = cls_end;
            continue;
        }

        /* No stride describes the whole class: fall back to run-length
         * encoding of consecutive equally-spaced occurrences within it. */
        {
            int run_end = i + 1;

            if (i + 1 < cls_end)
            {
                double dt = ae->events[i + 1].start_time_us - ae->events[i].start_time_us;
                if (dt > 0.0)
                {
                    /* Near-exact: start times are exact double sums of
                     * whole-us block durations, so a genuine constant spacing
                     * compares bit-equal.  The relative slack only absorbs
                     * the last-bit noise a non-integral block duration could
                     * introduce; it is ~1e-5 us at MPRAGE scale, orders of
                     * magnitude below the 1 us pulseq time quantum, so it can
                     * never fuse occurrences that are really unequally
                     * spaced. */
                    double tol = 1.0e-9 * dt;
                    run_end = i + 2;
                    while (run_end < cls_end &&
                           fabs((ae->events[run_end].start_time_us -
                                 ae->events[run_end - 1].start_time_us) -
                                dt) <= tol)
                        ++run_end;
                }
            }

            if (!sa_emit_train(&out[n_out], ae->events, i, 1, run_end - i))
                goto oom;
            n_out++;
            for (j = i + 1; j < run_end; ++j)
                absorbed[j] = 1;
            i = run_end;
        }
    }

    /* Every absorbed occurrence carried its own duplicate of the definition's
     * arb samples/times; the train keeps the head's copy and drops the rest. */
    for (j = 0; j < ae->num_events; ++j)
    {
        if (!absorbed[j])
            continue;
        if (ae->events[j].arb_samples)
            PULSEG_FREE(ae->events[j].arb_samples);
        if (ae->events[j].arb_times_us)
            PULSEG_FREE(ae->events[j].arb_times_us);
    }

    PULSEG_FREE(absorbed);
    PULSEG_FREE(ae->events);
    ae->events = out;
    ae->num_events = n_out;
    return PULSEG_SUCCESS;

oom:
    /* Fail closed, leaving `ae` exactly as it was found: no arb array is
     * released until the whole compressed list has been built. */
    for (j = 0; j < n_out; ++j)
        if (out[j].train_amps)
            PULSEG_FREE(out[j].train_amps);
    PULSEG_FREE(out);
    PULSEG_FREE(absorbed);
    return PULSEG_ERR_ALLOC_FAILED;
}

/**
 * Build event lists for all three axes.
 */
static int sa_build_structural_events(
    sa_structural_events *se,
    const struct pulseg_sequence_descriptor *desc,
    int start_block,
    int block_count)
{
    int ax, result;
    sa_raw_occurrence *occ;
    int num_occ;

    memset(se, 0, sizeof(*se));

    for (ax = 0; ax < 3; ++ax)
    {
        occ = NULL;
        num_occ = sa_extract_raw_occurrences(&occ, desc, start_block, block_count, ax);
        if (num_occ < 0)
        {
            sa_free_structural_events(se);
            return PULSEG_ERR_ALLOC_FAILED;
        }
        result = sa_build_axis_events(&se->axes[ax], occ, num_occ, desc);
        PULSEG_FREE(occ);
        if (PULSEG_FAILED(result))
        {
            sa_free_structural_events(se);
            return result;
        }
    }
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Structural acoustic analysis — analytical spectrum evaluation     */
/* ================================================================== */

/**
 * True complex Fourier transform of a piecewise-linear waveform (trap /
 * extended-trap / ramp / coarse-PWL) defined by n_vtx vertices
 * (t_us[i], v[i]); times in microseconds.  Returns the RAW complex integral
 *   G(f) = integral g(t) e^{-j 2 pi f t} dt   (in [value units] * us),
 * with NO |G(0)| normalization and NO zero-area fallback: this is the true
 * complex transform of the base waveform (the A_eq path needs true units, not
 * a peak-anchored |G(f)|/|G(0)|). The caller scales by 1e-6 to get
 * [value units] * s and multiplies by the event amplitude to obtain F_k(f).
 *
 * Per linear segment [t_k, t_{k+1}] with values [v_k, v_{k+1}]:
 *   G_k(f) = e^{-j omega t_k} [a I_0 + b I_1],  a = v_k, b = (v_{k+1}-v_k)/T_k,
 *   omega = 2 pi f (rad/us), I_0 = int_0^T e^{-j omega tau} dtau,
 *   I_1 = int_0^T tau e^{-j omega tau} dtau.
 * At omega -> 0 the segment reduces to the DC area integral (real).
 */
static void sa_eval_pwl_transform(
    float *out_re,
    float *out_im,
    float f_hz,
    const float *t_us,
    const float *v,
    int n_vtx)
{
    double omega, g_re, g_im;
    int k;

    g_re = 0.0;
    g_im = 0.0;

    if (n_vtx < 2)
    {
        *out_re = 0.0f;
        *out_im = 0.0f;
        return;
    }

    omega = 2.0 * M_PI * (double)f_hz * 1.0e-6;

    if (omega < 1.0e-12 && omega > -1.0e-12)
    {
        /* DC: G(0) = integral g dt = sum of trapezoid areas (real). */
        for (k = 0; k < n_vtx - 1; ++k)
        {
            double dt = (double)(t_us[k + 1] - t_us[k]);
            g_re += 0.5 * ((double)v[k] + (double)v[k + 1]) * dt;
        }
        *out_re = (float)g_re;
        *out_im = 0.0f;
        return;
    }

    /* --- Uniform-raster fast path ---------------------------------------
     * An arbitrary gradient sampled on the gradient raster (the common case:
     * sa_build_axis_events() synthesises cell-centre times as
     * 0.5*raster + m*raster whenever the definition carries no time shape)
     * has a CONSTANT segment length Tk. Two consequences, both exact:
     *   1. cos/sin(omega*Tk) and hence the c0/c1 segment integrals are the
     *      same for every segment -- hoist them out of the loop.
     *   2. tk is an arithmetic progression, so e^{-j omega tk} advances by a
     *      constant rotation e^{-j omega dt} per segment -- a complex
     *      multiply instead of a sin/cos pair.
     * That takes the inner loop from 4 transcendentals per segment to none,
     * which is what makes long arbitrary waveforms (spiral/rosette readouts,
     * thousands of samples per shot, re-evaluated at every candidate
     * frequency) affordable. The recurrence is re-anchored to an exact
     * sin/cos every SA_PWL_REANCHOR steps so accumulated rotation drift
     * stays bounded regardless of sample count.
     * Detection is a plain scan for constant spacing -- no assumption about
     * the waveform's content. The tolerance absorbs the float representation
     * error of the stored times while staying far below the 1 us pulseq time
     * quantum, so it can never merge two genuinely different rasters.
     * Non-uniform (explicit time-shape) definitions fall through to the
     * general loop below, unchanged. */
    if (n_vtx > 2)
    {
        double dt0 = (double)(t_us[1] - t_us[0]);
        double tol = 1.0e-4 * fabs(dt0) + 1.0e-6 * fabs((double)t_us[n_vtx - 1]);
        int uniform = (dt0 > 1.0e-12);

        for (k = 1; uniform && k < n_vtx - 1; ++k)
        {
            double d = (double)(t_us[k + 1] - t_us[k]);
            if (fabs(d - dt0) > tol)
                uniform = 0;
        }

        if (uniform)
        {
            double wT = omega * dt0;
            double cos_wT = cos(wT);
            double sin_wT = sin(wT);
            double inv_dt = 1.0 / dt0;
            double c0_re = sin_wT / omega;
            double c0_im = (cos_wT - 1.0) / omega;
            double I0mT_re = c0_re - dt0 * cos_wT;
            double I0mT_im = c0_im + dt0 * sin_wT;
            double c1_re = I0mT_im / omega;
            double c1_im = -I0mT_re / omega;
            double t0 = (double)t_us[0];
            double cos_ph = cos(omega * t0);
            double sin_ph = sin(omega * t0);
            int since_anchor = 0;

            for (k = 0; k < n_vtx - 1; ++k)
            {
                double ak = (double)v[k];
                double bk = ((double)v[k + 1] - ak) * inv_dt;
                double x_re = ak * c0_re + bk * c1_re;
                double x_im = ak * c0_im + bk * c1_im;

                /* multiply by e^{-j omega tk} */
                g_re += x_re * cos_ph + x_im * sin_ph;
                g_im += x_im * cos_ph - x_re * sin_ph;

                /* advance to t_{k+1} = t_k + dt0 */
                if (++since_anchor >= SA_PWL_REANCHOR)
                {
                    double tt = omega * (t0 + (double)(k + 1) * dt0);
                    cos_ph = cos(tt);
                    sin_ph = sin(tt);
                    since_anchor = 0;
                }
                else
                {
                    double nc = cos_ph * cos_wT - sin_ph * sin_wT;
                    double ns = sin_ph * cos_wT + cos_ph * sin_wT;
                    cos_ph = nc;
                    sin_ph = ns;
                }
            }

            *out_re = (float)g_re;
            *out_im = (float)g_im;
            return;
        }
    }

    for (k = 0; k < n_vtx - 1; ++k)
    {
        double tk = (double)t_us[k];
        double Tk = (double)(t_us[k + 1] - t_us[k]);
        double ak = (double)v[k];
        double bk;
        double wTk, cos_wTk, sin_wTk, cos_ph, sin_ph;
        double c0_re, c0_im, c1_re, c1_im;
        double x_re, x_im, seg_re, seg_im;

        if (Tk < 1.0e-12)
            continue;

        bk = ((double)v[k + 1] - ak) / Tk;
        wTk = omega * Tk;
        cos_wTk = cos(wTk);
        sin_wTk = sin(wTk);
        cos_ph = cos(omega * tk);
        sin_ph = sin(omega * tk);

        c0_re = sin_wTk / omega;
        c0_im = (cos_wTk - 1.0) / omega;
        {
            double I0mT_re = c0_re - Tk * cos_wTk;
            double I0mT_im = c0_im + Tk * sin_wTk;
            c1_re = I0mT_im / omega;
            c1_im = -I0mT_re / omega;
        }

        x_re = ak * c0_re + bk * c1_re;
        x_im = ak * c0_im + bk * c1_im;

        /* multiply by e^{-j omega tk} */
        seg_re = x_re * cos_ph + x_im * sin_ph;
        seg_im = x_im * cos_ph - x_re * sin_ph;

        g_re += seg_re;
        g_im += seg_im;
    }

    *out_re = (float)g_re;
    *out_im = (float)g_im;
}

/**
 * True complex transform of one event's base waveform at frequency f (no
 * amplitude scaling, no start-time phase — the caller adds those):
 *   T(f) = integral shape(tau) e^{-j 2 pi f tau} dtau   ([normalised] * us),
 * tau measured from the event start.  Arbitrary waveforms use their raw
 * samples (cell-centre times) directly as PWL vertices — the exact transform
 * of the sample interpolant, evaluated at the sparse in-band lines only.
 */
static void sa_eval_event_transform(
    const sa_event *ev,
    float *out_re,
    float *out_im,
    float f_hz,
    sa_transform_cache *cache)
{
    if (sa_transform_cache_lookup(cache, out_re, out_im, ev->def_id))
        return;

    if (ev->arb_num_samples >= 2)
        sa_eval_pwl_transform(
            out_re,
            out_im,
            f_hz,
            ev->arb_times_us,
            ev->arb_samples,
            ev->arb_num_samples);
    else if (ev->pwl_num_vertices >= 2)
        sa_eval_pwl_transform(
            out_re,
            out_im,
            f_hz,
            ev->pwl_times_us,
            ev->pwl_values,
            ev->pwl_num_vertices);
    else
    {
        *out_re = 0.0f;
        *out_im = 0.0f;
    }

    sa_transform_cache_insert(cache, ev->def_id, *out_re, *out_im);
}

/**
 * Geometric (Dirichlet) sum of N unit phasors spaced by @p phi radians:
 *   sum_{n=0}^{N-1} e^{j n phi}
 *     = e^{j (N-1) phi/2} * sin(N phi/2) / sin(phi/2).
 * The removable singularity at phi = 2 pi m (all phasors aligned) has the
 * limit N.
 */
static void sa_geometric_sum(double *out_re, double *out_im, double phi, int N)
{
    double half = phi * 0.5;
    double sin_half = sin(half);
    double ratio, center;

    if (N <= 1)
    {
        *out_re = (N < 1) ? 0.0 : 1.0;
        *out_im = 0.0;
        return;
    }
    if (fabs(sin_half) < 1.0e-12)
    {
        *out_re = (double)N;
        *out_im = 0.0;
        return;
    }
    ratio = sin((double)N * half) / sin_half;
    center = (double)(N - 1) * half;
    *out_re = ratio * cos(center);
    *out_im = ratio * sin(center);
}

/**
 * Complex spectral contribution of one event at frequency f, in true units:
 *   a_k(f) = A_k * T_k(f) * 1e-6 * e^{-j 2 pi f t_k}   (Hz/m * s),
 * where A_k is the event amplitude (Hz/m), T_k the base-waveform transform
 * (in [normalised]*us, hence the 1e-6 us->s), and t_k the event start time.
 *
 * For a compressed train (train_len L > 1, spacing D — see
 * sa_compress_axis_events) the event stands for L occurrences and this
 * returns their coherent sum,
 *   T_k(f) * 1e-6 * e^{-j 2 pi f t_0} * sum_{j<L} A_j z^j,  z = e^{-j 2 pi f D},
 * evaluated in closed form when the amplitudes are constant and by Horner
 * when they are not.  Either way the transcendental count is independent of
 * L: one sin/cos pair for z plus one for the common start-time phase.
 */
static void sa_eval_event_spectrum(
    const sa_event *ev,
    float *out_re,
    float *out_im,
    float f_hz,
    sa_transform_cache *cache)
{
    float tr_re, tr_im;
    double sum_re, sum_im, base_re, base_im;
    double phase, cos_ph, sin_ph;

    sa_eval_event_transform(ev, &tr_re, &tr_im, f_hz, cache);

    /* sum_{j<L} A_j z^j -- the amplitude-weighted train sum, still relative
     * to the train's own start time. */
    if (ev->train_len > 1)
    {
        double phi = -2.0 * M_PI * (double)f_hz * ev->train_period_us * 1.0e-6;
        if (ev->train_amps)
        {
            double zr = cos(phi);
            double zi = sin(phi);
            int j;
            sum_re = (double)ev->train_amps[ev->train_len - 1];
            sum_im = 0.0;
            for (j = ev->train_len - 2; j >= 0; --j)
            {
                double nr = sum_re * zr - sum_im * zi + (double)ev->train_amps[j];
                double ni = sum_re * zi + sum_im * zr;
                sum_re = nr;
                sum_im = ni;
            }
        }
        else
        {
            sa_geometric_sum(&sum_re, &sum_im, phi, ev->train_len);
            sum_re *= (double)ev->amplitude;
            sum_im *= (double)ev->amplitude;
        }
    }
    else
    {
        sum_re = (double)ev->amplitude;
        sum_im = 0.0;
    }

    /* times the base-waveform transform, in Hz/m * s */
    base_re = (sum_re * (double)tr_re - sum_im * (double)tr_im) * 1.0e-6;
    base_im = (sum_re * (double)tr_im + sum_im * (double)tr_re) * 1.0e-6;

    phase = -2.0 * M_PI * (double)f_hz * ev->start_time_us * 1.0e-6;
    cos_ph = cos(phase);
    sin_ph = sin(phase);

    *out_re = (float)(base_re * cos_ph - base_im * sin_ph);
    *out_im = (float)(base_re * sin_ph + base_im * cos_ph);
}

/**
 * Complex spectral line of one event at frequency f, including the Dirichlet
 * repetition kernel for events repeated N times with period T:
 *   line_k(f) = a_k(f) * D_{N_k}(f),
 *   D_N(f,T)  = sum_{n=0}^{N-1} e^{-j 2 pi f n T}
 *             = e^{-j(N-1)phi/2} sin(N phi/2)/sin(phi/2),  phi = -2 pi f T.
 * (N==1 -> D_1 = 1.)
 *
 * This is the OUTER (NEX / num_averages) repeat.  When the event is also a
 * compressed intra-TR train, sa_eval_event_spectrum() has already applied
 * that train's own sum; the two kernels simply multiply, since the train
 * lies wholly inside the imaging block that NEX repeats.
 */
static void sa_eval_event_line(
    const sa_event *ev,
    float *out_re,
    float *out_im,
    float f_hz,
    sa_transform_cache *cache)
{
    float d_re, d_im;
    int N = ev->num_reps;
    if (N < 1)
        N = 1;

    sa_eval_event_spectrum(ev, &d_re, &d_im, f_hz, cache);

    if (N > 1 && ev->rep_period_us > 0.0)
    {
        double phi = -2.0 * M_PI * (double)f_hz * ev->rep_period_us * 1.0e-6;
        double dk_re, dk_im, new_re, new_im;

        sa_geometric_sum(&dk_re, &dk_im, phi, N);

        new_re = (double)d_re * dk_re - (double)d_im * dk_im;
        new_im = (double)d_re * dk_im + (double)d_im * dk_re;
        d_re = (float)new_re;
        d_im = (float)new_im;
    }

    *out_re = d_re;
    *out_im = d_im;
}

/**
 * Complex structural drive spectrum of one axis at frequency f:
 *   S_ax(f) = sum_k line_k(f).
 * The per-axis equivalent-sustained amplitude of the spectral line at f is
 *   A_eq(f) = (2 / T_TR) * |S_ax(f)|   (Hz/m),
 * computed by the caller with the canonical-TR period T_TR.
 */
static void sa_eval_axis_spectrum(
    const sa_axis_events *ae,
    float *out_re,
    float *out_im,
    float f_hz)
{
    /* double: the per-axis sum is a coherent sum of many oppositely-signed
     * contributions, so it routinely lands deep in cancellation. A float
     * accumulator loses the low bits of the survivors and is the dominant
     * error term there -- costs nothing to carry the running sum in double
     * and only narrow it on the way out. */
    double sum_re, sum_im;
    int k;
    sa_transform_cache cache;
    sa_transform_cache_entry *cache_entries = NULL;

    /* One call = one fixed frequency, so def_id alone is a sufficient
     * memo key (docs/explanations/mechanical_resonance_safety.md, "Stage 4") -- events
     * sharing a def_id (the common case: a handful of unique gradient
     * shapes reused across many materialized occurrences) skip the
     * O(vertices) sa_eval_pwl_transform integral after the first hit.
     * Sized to num_events (worst case: every event a distinct def_id);
     * alloc failure or num_events==0 just disables caching (cache.capacity
     * stays 0), never a correctness issue -- sa_transform_cache_lookup/
     * insert are both no-ops against an empty/absent cache. */
    cache.count = 0;
    cache.capacity = 0;
    cache.entries = NULL;
    if (ae->num_events > 0)
    {
        cache_entries = (sa_transform_cache_entry *)PULSEG_ALLOC(
            (size_t)ae->num_events * sizeof(sa_transform_cache_entry));
        if (cache_entries)
        {
            cache.entries = cache_entries;
            cache.capacity = ae->num_events;
        }
    }

    sum_re = 0.0;
    sum_im = 0.0;
    for (k = 0; k < ae->num_events; ++k)
    {
        float d_re, d_im;
        sa_eval_event_line(&ae->events[k], &d_re, &d_im, f_hz, &cache);
        sum_re += (double)d_re;
        sum_im += (double)d_im;
    }

    if (cache_entries)
        PULSEG_FREE(cache_entries);

    *out_re = (float)sum_re;
    *out_im = (float)sum_im;
}

/* ================================================================== */
/*  Structural acoustic analysis — top-level violation check          */
/* ================================================================== */

/**
 * A_eq mechanical-resonance verdict (docs/explanations/mechanical_resonance_safety.md).
 *
 *   1. Build the event model of the canonical (outer) TR — every gradient
 *      instance materialised at its time within the TR (inner periodicities
 *      such as echo trains / slices are NOT declared; they emerge from the
 *      coherent sum of the instances).  NEX (num_avgs) repetition is tagged
 *      as a Dirichlet kernel on the imaging events.
 *   2. (display) A_eq at every TR harmonic k/T_TR up to freq_max, for plots.
 *   3. (verdict) For each forbidden band, enumerate the TR-harmonic lines that
 *      fall inside the guarded range [f_min-guard, f_max+guard], evaluate
 *      A_eq(f_L) = (2/T_TR)|S_ax(f_L)| per axis, and flag the band iff the
 *      max-axis A_eq exceeds eps = max(band limit, k*G_max).
 * The outermost TR is treated as an infinite-rep (Dirac) comb: only its
 * harmonics carry sustained drive, hence lines live exactly at k/T_TR.
 */
static float sa_eps_for_band(const pulseg_forbidden_band *band, float g_max_hz_per_m)
{
    /* The floor only rescues a literal zero-tolerance band (unusable as a
     * threshold — see SA_AEQ_K_GMAX comment). Any vendor-specified nonzero
     * limit is trusted as-is, even if tighter than the floor. */
    if (band->max_amplitude_hz_per_m <= 0.0f)
        return SA_AEQ_K_GMAX * g_max_hz_per_m;
    return band->max_amplitude_hz_per_m;
}

/**
 * Finite-outer-rep Dirichlet ratio (docs/explanations/mechanical_resonance_safety.md
 * §1): |D_M(x)| / M, where D_M(x) = sin(M*pi*x) / sin(pi*x) is the
 * Dirichlet kernel for M coherent repeats of period T_TR, and
 * x = f * T_TR (dimensionless: integer x = exact TR harmonics, fractional
 * x = the sidelobes between them that only exist for finite M).
 *
 * This ratio is exactly 1.0 at integer x (main lobes -- the M-independent
 * regression identity: at exact TR harmonics this fix must reduce to
 * today's infinite-comb formula) and < 1.0 elsewhere, decaying as M grows
 * (large-M sidelobes shrink toward the M=infinity/Dirac-comb limit, which
 * is why large-M scans reproduce today's verdicts).
 *
 * The caller must evaluate S_TR(f) FRESH at the fractional x (S_TR is an
 * oscillating function of f in its own right -- it is NOT bounded by, or
 * safely approximated from, its value at a nearby exact TR harmonic; an
 * earlier version of this fix tried reusing the coarse S_TR value scaled
 * by this ratio, which is a mathematical no-op since the ratio is always
 * <=1 and can therefore never expose a violation the coarse point didn't
 * already show -- caught via numeric exploration during implementation).
 * This ratio only supplies the Dirichlet attenuation factor; cost is kept
 * independent of M by capping how many fractional points get evaluated
 * per coarse interval (SA_MECHRES_MAX_DM_SUBDIV), not by skipping the
 * fresh S_TR evaluation.
 */
static double sa_dirichlet_ratio(double x, int M)
{
    double s, num;
    if (M <= 1)
        return 1.0;
    s = sin(M_PI * x);
    /* Removable singularity at integer x (s -> 0): the true limit is 1.0
     * (D_M(x)->M), not 0/0. 1e-6 is far tighter than any x offset this
     * function is ever called with (sub-divisions are 1/SA_MECHRES_MAX_DM_SUBDIV
     * apart at the very finest), so this only triggers at genuine integers. */
    if (fabs(s) < 1e-6)
        return 1.0;
    num = sin((double)M * M_PI * x);
    return fabs(num / (s * (double)M));
}

static int sa_check_structural_violations(
    pulseg_mech_resonances_spectra *spectra,
    const struct pulseg_sequence_descriptor *desc,
    int start_block,
    int block_count,
    int num_instances,
    float tr_duration_us,
    int num_forbidden_bands,
    const pulseg_forbidden_band *forbidden_bands,
    float peak_log10_threshold,
    float peak_norm_scale,
    float peak_eps,
    float peak_prominence,
    int num_avgs,
    float g_max_hz_per_m,
    int compute_dense_envelope,
    int compute_display_products,
    int compress_trains)
{
    sa_structural_events se;
    int result, ax, i, b, k, ci;
    double T_s, f1_hz;
    float freq_max, guard, min_bw;
    int m_max;

    /* analytical display grid (TR harmonics; display-only, not the verdict) */
    int num_ana;
    float *ana_freqs;
    float *ana_amps[3];
    float *ana_phases[3];
    float *ana_widths;

    /* candidate selection (in-band TR-harmonic lines; the verdict) */
    int cand_cap, num_cand;
    float *cand_freqs;
    float *cand_amps[3];
    float *cand_grad_amps;
    float *cand_grad_amps_ax[3];
    int *cand_violations;
    float *surviving_freqs_hz;

    /* Component-level export arrays */
    int max_component_terms, num_component_terms;
    float *component_freqs_hz, *component_amps, *component_phases_rad, *component_widths_hz;
    int *component_axes, *component_def_ids, *component_contrib_ids, *component_run_ids;

    /* Dense analytic envelope (display-only; plotting API only, see
     * compute_dense_envelope) */
    int num_env;
    float *env_freqs;
    float *env_amps[3];

    (void)peak_log10_threshold;
    (void)peak_norm_scale;
    (void)peak_eps;
    (void)peak_prominence;

    memset(&se, 0, sizeof(se));
    num_ana = 0;
    ana_freqs = NULL;
    ana_widths = NULL;
    cand_cap = 0;
    num_cand = 0;
    cand_freqs = NULL;
    cand_grad_amps = NULL;
    cand_violations = NULL;
    surviving_freqs_hz = NULL;
    max_component_terms = 0;
    num_component_terms = 0;
    component_freqs_hz = NULL;
    component_amps = NULL;
    component_phases_rad = NULL;
    component_widths_hz = NULL;
    component_axes = NULL;
    component_def_ids = NULL;
    component_contrib_ids = NULL;
    component_run_ids = NULL;
    num_env = 0;
    env_freqs = NULL;
    for (ax = 0; ax < 3; ++ax)
    {
        ana_amps[ax] = NULL;
        ana_phases[ax] = NULL;
        env_amps[ax] = NULL;
        cand_amps[ax] = NULL;
        cand_grad_amps_ax[ax] = NULL;
    }

    result = sa_build_structural_events(&se, desc, start_block, block_count);
    if (PULSEG_FAILED(result))
        return result;

    /* --- Tag imaging events with repetition info ---
     * For non-degenerate sequences (num_avgs > 1), imaging events repeat
     * num_avgs times with period = imaging pass duration.  Cooldown events
     * are shifted to their position in the expanded pass.  This lets the
     * Dirichlet kernel handle repetition analytically (O(1) per event per
     * frequency) instead of enumerating all N copies (O(N)). */
    if (num_avgs > 1)
    {
        const struct pulseg_tr_descriptor *trd = &desc->tr_descriptor;
        int prep_blk = trd->num_prep_blocks;
        int img_len = trd->num_trs * trd->tr_size;
        /* double, to match the exact whole-us accumulation in
         * sa_extract_raw_occurrences() -- these bounds are compared against
         * event start times, and the cooldown shift below is added to them. */
        double prep_dur_us = 0.0;
        double img_dur_us = 0.0;
        double img_end_us;
        int bi;

        for (bi = 0; bi < prep_blk; ++bi)
        {
            int idx = start_block + bi;
            const struct pulseg_block_table_element *bte = &desc->block_table[idx];
            const struct pulseg_base_block *bdef = &desc->base_blocks[bte->id];
            prep_dur_us +=
                (bte->duration_us >= 0) ? (double)bte->duration_us : (double)bdef->duration_us;
        }
        for (bi = 0; bi < img_len; ++bi)
        {
            int idx = start_block + prep_blk + bi;
            const struct pulseg_block_table_element *bte = &desc->block_table[idx];
            const struct pulseg_base_block *bdef = &desc->base_blocks[bte->id];
            img_dur_us +=
                (bte->duration_us >= 0) ? (double)bte->duration_us : (double)bdef->duration_us;
        }
        img_end_us = prep_dur_us + img_dur_us;

        for (ax = 0; ax < 3; ++ax)
        {
            int k;
            for (k = 0; k < se.axes[ax].num_events; ++k)
            {
                sa_event *ev = &se.axes[ax].events[k];
                if (ev->start_time_us >= prep_dur_us && ev->start_time_us < img_end_us)
                {
                    ev->num_reps = num_avgs;
                    ev->rep_period_us = img_dur_us;
                }
                else if (ev->start_time_us >= img_end_us)
                {
                    ev->start_time_us += (double)(num_avgs - 1) * img_dur_us;
                    ev->num_reps = 1;
                    ev->rep_period_us = 0.0;
                }
                else
                {
                    ev->num_reps = 1;
                    ev->rep_period_us = 0.0;
                }
            }
        }
    }
    else
    {
        for (ax = 0; ax < 3; ++ax)
        {
            int k;
            for (k = 0; k < se.axes[ax].num_events; ++k)
            {
                se.axes[ax].events[k].num_reps = 1;
                se.axes[ax].events[k].rep_period_us = 0.0;
            }
        }
    }

    /* --- Compress equally-spaced occurrence trains ---
     * Runs AFTER the NEX tagging above, so the (num_reps, rep_period_us)
     * pair is part of the fusion key and no train can straddle the prep /
     * imaging / cooldown boundary the tagging just drew.
     *
     * Independent of compute_display_products, so the plotting API can draw
     * exactly the lines the headless gate decides on (the default) while a
     * caller chasing a discrepancy can turn compression off and get the
     * uncompressed reference evaluation of the same math.  The one visible
     * difference is the component export: with compression on, a component
     * term covers a whole equally-spaced train rather than one materialised
     * occurrence. */
    if (compress_trains)
    {
        for (ax = 0; ax < 3; ++ax)
        {
            result = sa_compress_axis_events(&se.axes[ax]);
            if (PULSEG_FAILED(result))
            {
                sa_free_structural_events(&se);
                return result;
            }
        }
    }

    /* --- Canonical-TR period & frequency guard --- */
    if (tr_duration_us <= 0.0f)
    {
        sa_free_structural_events(&se);
        return PULSEG_SUCCESS;
    }
    T_s = (double)tr_duration_us * 1.0e-6;
    f1_hz = 1.0 / T_s;

    if (spectra->num_freq_bins > 0)
        freq_max =
            spectra->freq_min_hz + (float)(spectra->num_freq_bins - 1) * spectra->freq_spacing_hz;
    else
        freq_max = 0.0f;

    /* guard = HWHM = min_band_width / 2 (narrowest band = sharpest resonance;
     * wide bands are keep-out ranges scanned at the same guard). */
    min_bw = -1.0f;
    for (b = 0; b < num_forbidden_bands; ++b)
    {
        float w = forbidden_bands[b].freq_max_hz - forbidden_bands[b].freq_min_hz;
        if (w > 0.0f && (min_bw < 0.0f || w < min_bw))
            min_bw = w;
    }
    guard = (min_bw > 0.0f) ? (SA_GUARD_HWHM_MULT * min_bw * 0.5f) : 0.0f;

    /* =====================================================================
     * (A) Analytical display grid — A_eq at each TR harmonic up to freq_max.
     *     Display-only (never the verdict, which comes from (B) alone):
     *     analytical_peak_* is read by the plotting API and by nothing on the
     *     PSD predownload path. Skipped when no display grid was requested
     *     (freq_max == 0, i.e. target_resolution_hz <= 0) and, regardless of
     *     resolution, whenever the caller asked for no display products
     *     (pulseg_check_safety) — for a long TR this grid is freq_max*T_TR
     *     harmonics (~21k for a 7 s MPRAGE hyper-TR at 3 kHz), each a full
     *     three-axis sa_eval_axis_spectrum, and the scanner must not pay for
     *     an array it immediately frees.
     * ===================================================================== */
    m_max = (compute_display_products && freq_max > 0.0f) ? (int)((double)freq_max / f1_hz) : 0;
    if (m_max > 0)
    {
        ana_freqs = (float *)PULSEG_ALLOC((size_t)m_max * sizeof(float));
        ana_widths = (float *)PULSEG_ALLOC((size_t)m_max * sizeof(float));
        if (!ana_freqs || !ana_widths)
            goto alloc_fail;
        for (ax = 0; ax < 3; ++ax)
        {
            ana_amps[ax] = (float *)PULSEG_ALLOC((size_t)m_max * sizeof(float));
            ana_phases[ax] = (float *)PULSEG_ALLOC((size_t)m_max * sizeof(float));
            if (!ana_amps[ax] || !ana_phases[ax])
                goto alloc_fail;
        }
        for (i = 0; i < m_max; ++i)
        {
            float f_hz = (float)((double)(i + 1) * f1_hz);
            ana_freqs[i] = f_hz;
            if (num_instances > 1)
                ana_widths[i] = 1.2067091288032284f * ((float)f1_hz / (float)num_instances);
            else
                ana_widths[i] = (float)f1_hz;
            for (ax = 0; ax < 3; ++ax)
            {
                float sre, sim;
                if (se.axes[ax].num_events == 0)
                {
                    ana_amps[ax][i] = 0.0f;
                    ana_phases[ax][i] = 0.0f;
                    continue;
                }
                sa_eval_axis_spectrum(&se.axes[ax], &sre, &sim, f_hz);
                ana_amps[ax][i] = (float)(2.0 / T_s * sqrt((double)(sre * sre + sim * sim)));
                ana_phases[ax][i] = (float)atan2((double)sim, (double)sre);
            }
        }
        num_ana = m_max;
    }

    /* =====================================================================
     * (A2) Dense analytic envelope — the SAME S_ax(f) transform as (A),
     *      evaluated on a uniform grid (spectrum_full's freq_min_hz/
     *      freq_spacing_hz) instead of only at TR harmonics k/T_TR. Display-
     *      only, and only ever requested by the plotting API
     *      (pulseg_calc_mech_resonances) -- never by pulseg_check_safety, so
     *      this never runs on the PSD predownload path. Because (A)'s values
     *      are this same function sampled at k/T_TR, this array reproduces
     *      them exactly at those frequencies -- a true matched envelope, not
     *      an interpolation and not a separately-windowed/normalised FFT.
     * ===================================================================== */
    if (compute_dense_envelope && spectra->num_freq_bins > 0)
    {
        num_env = spectra->num_freq_bins;
        env_freqs = (float *)PULSEG_ALLOC((size_t)num_env * sizeof(float));
        if (!env_freqs)
            goto alloc_fail;
        for (ax = 0; ax < 3; ++ax)
        {
            env_amps[ax] = (float *)PULSEG_ALLOC((size_t)num_env * sizeof(float));
            if (!env_amps[ax])
                goto alloc_fail;
        }
        for (i = 0; i < num_env; ++i)
        {
            float f_hz = spectra->freq_min_hz + (float)i * spectra->freq_spacing_hz;
            env_freqs[i] = f_hz;
            for (ax = 0; ax < 3; ++ax)
            {
                float sre, sim;
                if (se.axes[ax].num_events == 0)
                {
                    env_amps[ax][i] = 0.0f;
                    continue;
                }
                sa_eval_axis_spectrum(&se.axes[ax], &sre, &sim, f_hz);
                env_amps[ax][i] = (float)(2.0 / T_s * sqrt((double)(sre * sre + sim * sim)));
            }
        }
    }

    /* =====================================================================
     * (B) Candidate selection — TR-harmonic lines inside a guarded band.
     *     f_L = k / T_TR for integer k in [(f_min-guard)T, (f_max+guard)T];
     *     A_eq(f_L) = (2/T_TR)|S_ax(f_L)|; violation iff max-axis A_eq > eps.
     * ===================================================================== */
    for (b = 0; b < num_forbidden_bands; ++b)
    {
        double lo = ((double)forbidden_bands[b].freq_min_hz - (double)guard) * T_s;
        double hi = ((double)forbidden_bands[b].freq_max_hz + (double)guard) * T_s;
        int klo = (int)ceil(lo);
        int khi = (int)floor(hi);
        if (klo < 1)
            klo = 1;
        if (khi >= klo)
            cand_cap += (khi - klo + 1);
    }

    if (cand_cap > 0)
    {
        cand_freqs = (float *)PULSEG_ALLOC((size_t)cand_cap * sizeof(float));
        cand_grad_amps = (float *)PULSEG_ALLOC((size_t)cand_cap * sizeof(float));
        cand_violations = (int *)PULSEG_ALLOC((size_t)cand_cap * sizeof(int));
        surviving_freqs_hz = (float *)PULSEG_ALLOC((size_t)cand_cap * sizeof(float));
        if (!cand_freqs || !cand_grad_amps || !cand_violations || !surviving_freqs_hz)
            goto alloc_fail;
        for (ax = 0; ax < 3; ++ax)
        {
            cand_amps[ax] = (float *)PULSEG_ALLOC((size_t)cand_cap * sizeof(float));
            cand_grad_amps_ax[ax] = (float *)PULSEG_ALLOC((size_t)cand_cap * sizeof(float));
            if (!cand_amps[ax] || !cand_grad_amps_ax[ax])
                goto alloc_fail;
        }

        /* component-term export sizing (bounded). Display-only, like (A):
         * the per-event breakdown of each candidate line is consumed by the
         * plotting/report tooling, never by the verdict, so a caller that
         * asked for no display products allocates none and the export loop
         * below short-circuits on its first bounds check. */
        for (ax = 0; compute_display_products && ax < 3; ++ax)
            max_component_terms += se.axes[ax].num_events;
        if (max_component_terms > 0)
            max_component_terms *= cand_cap;
        if (max_component_terms > 100000)
            max_component_terms = 100000;
        if (max_component_terms > 0)
        {
            component_freqs_hz = (float *)PULSEG_ALLOC((size_t)max_component_terms * sizeof(float));
            component_amps = (float *)PULSEG_ALLOC((size_t)max_component_terms * sizeof(float));
            component_phases_rad =
                (float *)PULSEG_ALLOC((size_t)max_component_terms * sizeof(float));
            component_widths_hz =
                (float *)PULSEG_ALLOC((size_t)max_component_terms * sizeof(float));
            component_axes = (int *)PULSEG_ALLOC((size_t)max_component_terms * sizeof(int));
            component_def_ids = (int *)PULSEG_ALLOC((size_t)max_component_terms * sizeof(int));
            component_contrib_ids = (int *)PULSEG_ALLOC((size_t)max_component_terms * sizeof(int));
            component_run_ids = (int *)PULSEG_ALLOC((size_t)max_component_terms * sizeof(int));
            if (!component_freqs_hz || !component_amps || !component_phases_rad ||
                !component_widths_hz || !component_axes || !component_def_ids ||
                !component_contrib_ids || !component_run_ids)
                goto alloc_fail;
        }

        ci = 0;
        for (b = 0; b < num_forbidden_bands; ++b)
        {
            float eps = sa_eps_for_band(&forbidden_bands[b], g_max_hz_per_m);
            double lo = ((double)forbidden_bands[b].freq_min_hz - (double)guard) * T_s;
            double hi = ((double)forbidden_bands[b].freq_max_hz + (double)guard) * T_s;
            int klo = (int)ceil(lo);
            int khi = (int)floor(hi);
            int kk;
            float fwhm;
            if (klo < 1)
                klo = 1;
            if (num_instances > 1)
                fwhm = 1.2067091288032284f * ((float)f1_hz / (float)num_instances);
            else
                fwhm = (float)f1_hz;
            for (kk = klo; kk <= khi; ++kk)
            {
                float f_hz = (float)((double)kk * f1_hz);
                float max_ga = 0.0f;
                cand_freqs[ci] = f_hz;
                surviving_freqs_hz[ci] = f_hz;
                for (ax = 0; ax < 3; ++ax)
                {
                    float sre, sim, aeq;
                    if (se.axes[ax].num_events == 0)
                    {
                        cand_amps[ax][ci] = 0.0f;
                        cand_grad_amps_ax[ax][ci] = 0.0f;
                        continue;
                    }
                    sa_eval_axis_spectrum(&se.axes[ax], &sre, &sim, f_hz);
                    aeq = (float)(2.0 / T_s * sqrt((double)(sre * sre + sim * sim)));
                    cand_amps[ax][ci] = aeq;
                    cand_grad_amps_ax[ax][ci] = aeq;
                    if (aeq > max_ga)
                        max_ga = aeq;

                    for (k = 0; k < se.axes[ax].num_events; ++k)
                    {
                        float lre, lim;
                        if (num_component_terms >= max_component_terms)
                            break;
                        sa_eval_event_line(&se.axes[ax].events[k], &lre, &lim, f_hz, NULL);
                        component_freqs_hz[num_component_terms] = f_hz;
                        component_amps[num_component_terms] =
                            (float)(2.0 / T_s * sqrt((double)(lre * lre + lim * lim)));
                        component_phases_rad[num_component_terms] =
                            (float)atan2((double)lim, (double)lre);
                        component_widths_hz[num_component_terms] = fwhm;
                        component_axes[num_component_terms] = ax;
                        component_def_ids[num_component_terms] = se.axes[ax].events[k].def_id;
                        component_contrib_ids[num_component_terms] = k;
                        component_run_ids[num_component_terms] = 0;
                        num_component_terms++;
                    }
                }

                /* ---- Finite-outer-rep fix (docs/explanations/mechanical_resonance_safety.md, "Stage 4") ----
                 * The outer repeat (num_instances = M) is no longer treated as an
                 * infinite Dirac comb: for M>1, real candidate frequencies exist
                 * between this coarse TR harmonic (kk/T_TR, exact -- already
                 * max_ga above, unchanged) and the next one, at fractional
                 * harmonics (kk + j/subdiv)/T_TR. These are NOT found by scaling
                 * the coarse point's already-computed amplitude by the Dirichlet
                 * ratio (a ratio <=1 can never exceed the coarse value it scales,
                 * which would make this a mathematical no-op -- caught via a
                 * numeric exploration during implementation, see
                 * docs/explanations/mechanical_resonance_safety.md). S_TR(f) must be
                 * evaluated FRESH at each fractional frequency (it is an
                 * oscillating function of f in its own right, not bounded by
                 * its value at nearby coarse samples) and only THEN attenuated
                 * by the Dirichlet ratio. Cost stays independent of M because
                 * the number of coarse candidates (klo..khi) already doesn't
                 * scale with M, and subdiv is capped at
                 * SA_MECHRES_MAX_DM_SUBDIV regardless of how large M is --
                 * this adds a bounded constant-factor number of extra
                 * sa_eval_axis_spectrum calls per coarse candidate, not a
                 * per-M-scaling cost. M=1 (e.g. a single-pass hyper-TR) takes
                 * this branch's early-out and is untouched: no sidelobes exist
                 * for a single repeat, and the coarse-only max_ga above is
                 * already the correct/final verdict, matching today's
                 * exact-harmonic-only behavior. */
                if (num_instances > 1)
                {
                    /* Geometrically-spaced sample points concentrated near
                     * EACH of the two adjacent main lobes (kk and kk+1),
                     * where the Dirichlet sidelobes actually live for large
                     * M (the first and largest sidelobe sits within
                     * ~1/(2M) of its lobe). Uniform spacing across the
                     * whole [kk,kk+1] interval was tried first and
                     * confirmed (via numeric exploration) to completely
                     * miss the sidelobes once M exceeds the sample count --
                     * e.g. M=64 with 16 uniform points reproduced the M=1
                     * (no-sidelobe) result exactly, because every sample
                     * landed in the far sidelobe region where the
                     * attenuation is negligible. Geometric spacing needs
                     * only a fixed point count per side to resolve the
                     * near-lobe region regardless of how large M is: delta_p
                     * = (0.5/M) * 2^p starting right next to the lobe and
                     * doubling outward, clamped to stay inside the
                     * interval. */
                    int npts_per_side = SA_MECHRES_MAX_DM_SUBDIV / 2;
                    int side, p;
                    for (side = 0; side < 2; ++side)
                    {
                        for (p = 0; p < npts_per_side; ++p)
                        {
                            double delta = (0.5 / (double)num_instances) * pow(2.0, (double)p);
                            double x_sub;
                            float f_sub_hz;
                            double ratio;
                            if (delta >= 0.5)
                                delta = 0.5 - 1e-6;
                            x_sub = (side == 0) ? ((double)kk + delta) : ((double)(kk + 1) - delta);
                            f_sub_hz = (float)(x_sub * f1_hz);
                            ratio = sa_dirichlet_ratio(x_sub, num_instances);
                            for (ax = 0; ax < 3; ++ax)
                            {
                                float sre, sim, aeq_sub;
                                if (se.axes[ax].num_events == 0)
                                    continue;
                                sa_eval_axis_spectrum(&se.axes[ax], &sre, &sim, f_sub_hz);
                                aeq_sub =
                                    (float)(2.0 / T_s * sqrt((double)(sre * sre + sim * sim)) * ratio);
                                if (aeq_sub > max_ga)
                                    max_ga = aeq_sub;
                            }
                        }
                    }
                }

                cand_grad_amps[ci] = max_ga;
                cand_violations[ci] = (max_ga > eps) ? 1 : 0;
                ci++;
            }
        }
        num_cand = ci;
    }

    /* --- Assign to output spectra struct (ownership transfer) --- */
    spectra->num_analytical_peaks = num_ana;
    spectra->analytical_peak_freqs = ana_freqs;
    ana_freqs = NULL;
    spectra->analytical_peak_amp_gx = ana_amps[0];
    ana_amps[0] = NULL;
    spectra->analytical_peak_amp_gy = ana_amps[1];
    ana_amps[1] = NULL;
    spectra->analytical_peak_amp_gz = ana_amps[2];
    ana_amps[2] = NULL;
    spectra->analytical_peak_phase_gx = ana_phases[0];
    ana_phases[0] = NULL;
    spectra->analytical_peak_phase_gy = ana_phases[1];
    ana_phases[1] = NULL;
    spectra->analytical_peak_phase_gz = ana_phases[2];
    ana_phases[2] = NULL;
    spectra->analytical_peak_widths_hz = ana_widths;
    ana_widths = NULL;

    spectra->num_candidates = num_cand;
    spectra->candidate_freqs = cand_freqs;
    cand_freqs = NULL;
    spectra->candidate_amps_gx = cand_amps[0];
    cand_amps[0] = NULL;
    spectra->candidate_amps_gy = cand_amps[1];
    cand_amps[1] = NULL;
    spectra->candidate_amps_gz = cand_amps[2];
    cand_amps[2] = NULL;
    spectra->candidate_grad_amps = cand_grad_amps;
    cand_grad_amps = NULL;
    spectra->candidate_grad_amps_gx = cand_grad_amps_ax[0];
    cand_grad_amps_ax[0] = NULL;
    spectra->candidate_grad_amps_gy = cand_grad_amps_ax[1];
    cand_grad_amps_ax[1] = NULL;
    spectra->candidate_grad_amps_gz = cand_grad_amps_ax[2];
    cand_grad_amps_ax[2] = NULL;
    spectra->candidate_violations = cand_violations;
    cand_violations = NULL;

    spectra->num_component_terms = num_component_terms;
    spectra->component_freqs_hz = component_freqs_hz;
    component_freqs_hz = NULL;
    spectra->component_amps = component_amps;
    component_amps = NULL;
    spectra->component_phases_rad = component_phases_rad;
    component_phases_rad = NULL;
    spectra->component_widths_hz = component_widths_hz;
    component_widths_hz = NULL;
    spectra->component_axes = component_axes;
    component_axes = NULL;
    spectra->component_def_ids = component_def_ids;
    component_def_ids = NULL;
    spectra->component_contrib_ids = component_contrib_ids;
    component_contrib_ids = NULL;
    spectra->component_run_ids = component_run_ids;
    component_run_ids = NULL;

    spectra->num_surviving_freqs = num_cand;
    spectra->surviving_freqs_hz = surviving_freqs_hz;
    surviving_freqs_hz = NULL;

    spectra->num_envelope_bins = num_env;
    spectra->envelope_freqs_hz = env_freqs;
    env_freqs = NULL;
    spectra->envelope_amp_gx = env_amps[0];
    env_amps[0] = NULL;
    spectra->envelope_amp_gy = env_amps[1];
    env_amps[1] = NULL;
    spectra->envelope_amp_gz = env_amps[2];
    env_amps[2] = NULL;

    sa_free_structural_events(&se);
    return PULSEG_SUCCESS;

alloc_fail:
    if (ana_freqs)
        PULSEG_FREE(ana_freqs);
    if (ana_widths)
        PULSEG_FREE(ana_widths);
    if (cand_freqs)
        PULSEG_FREE(cand_freqs);
    if (cand_grad_amps)
        PULSEG_FREE(cand_grad_amps);
    if (cand_violations)
        PULSEG_FREE(cand_violations);
    if (surviving_freqs_hz)
        PULSEG_FREE(surviving_freqs_hz);
    if (env_freqs)
        PULSEG_FREE(env_freqs);
    for (ax = 0; ax < 3; ++ax)
    {
        if (ana_amps[ax])
            PULSEG_FREE(ana_amps[ax]);
        if (ana_phases[ax])
            PULSEG_FREE(ana_phases[ax]);
        if (cand_amps[ax])
            PULSEG_FREE(cand_amps[ax]);
        if (cand_grad_amps_ax[ax])
            PULSEG_FREE(cand_grad_amps_ax[ax]);
        if (env_amps[ax])
            PULSEG_FREE(env_amps[ax]);
    }
    if (component_freqs_hz)
        PULSEG_FREE(component_freqs_hz);
    if (component_amps)
        PULSEG_FREE(component_amps);
    if (component_phases_rad)
        PULSEG_FREE(component_phases_rad);
    if (component_widths_hz)
        PULSEG_FREE(component_widths_hz);
    if (component_axes)
        PULSEG_FREE(component_axes);
    if (component_def_ids)
        PULSEG_FREE(component_def_ids);
    if (component_contrib_ids)
        PULSEG_FREE(component_contrib_ids);
    if (component_run_ids)
        PULSEG_FREE(component_run_ids);
    sa_free_structural_events(&se);
    return PULSEG_ERR_ALLOC_FAILED;
}

/* ================================================================== */
/*  Sequence spectrum (full-TR FFT)                                   */
/* ================================================================== */

/* Computes the full-TR magnitude spectrum (display-only).  The
 * canonical mechanical-resonance verdict comes from
 * sa_check_structural_violations. */
static int compute_sequence_spectrum(
    float *full_spectrum,
    int *out_num_freq_bins_full,
    float *out_freq_res_full,
    const float *waveform,
    int num_samples,
    float grad_raster_us,
    float target_spectral_res_hz,
    float max_frequency)
{
    int nfft, nfreq, output_bins_full, max_idx;
    float freq_res, max_freq, mean, fft_norm;
    float *work;
    float *cos_win;
    kiss_fft_cpx *fft_out;
    kiss_fftr_cfg cfg;
    int min_nfft, i;
    int result;

    result = PULSEG_SUCCESS;
    work = NULL;
    cos_win = NULL;
    fft_out = NULL;
    cfg = NULL;

    if (!waveform || num_samples <= 0)
        return PULSEG_ERR_INVALID_ARGUMENT;
    if (target_spectral_res_hz <= 0.0f)
        return PULSEG_ERR_INVALID_ARGUMENT;
    if (grad_raster_us <= 0.0f)
        return PULSEG_ERR_INVALID_ARGUMENT;

    min_nfft = (int)ceil((double)(1.0e6 / (grad_raster_us * target_spectral_res_hz)));
    nfft = (min_nfft < num_samples) ? (int)pulseg__next_pow2((size_t)num_samples)
                                    : (int)pulseg__next_pow2((size_t)min_nfft);
    nfreq = nfft / 2 + 1;
    freq_res = (float)(1.0e6 / (grad_raster_us * (double)nfft));
    max_freq = (max_frequency > 0.0f) ? max_frequency : (float)(5.0e5 / grad_raster_us);
    max_idx = (int)(max_freq / freq_res);
    if (max_idx >= nfreq)
        output_bins_full = nfreq;
    else if (max_idx < 1)
        output_bins_full = 1;
    else
        output_bins_full = max_idx + 1;

    work = (float *)PULSEG_ALLOC((size_t)nfft * sizeof(float));
    cos_win = (float *)PULSEG_ALLOC((size_t)num_samples * sizeof(float));
    fft_out = (kiss_fft_cpx *)PULSEG_ALLOC((size_t)nfreq * sizeof(kiss_fft_cpx));
    if (!work || !cos_win || !fft_out)
    {
        result = PULSEG_ERR_ALLOC_FAILED;
        goto fail;
    }

    cfg = kiss_fftr_alloc(nfft, 0, NULL, NULL);
    if (!cfg)
    {
        result = PULSEG_ERR_ALLOC_FAILED;
        goto fail;
    }

    for (i = 0; i < num_samples; ++i)
        cos_win[i] = (float)(0.5 * (1.0 - cos(2.0 * M_PI * (double)(i + 1) / (double)num_samples)));

    for (i = 0; i < num_samples; ++i)
        work[i] = waveform[i];
    for (i = num_samples; i < nfft; ++i)
        work[i] = 0.0f;

    mean = 0.0f;
    for (i = 0; i < num_samples; ++i)
        mean += work[i];
    mean /= (float)num_samples;
    for (i = 0; i < num_samples; ++i)
        work[i] -= mean;
    for (i = 0; i < num_samples; ++i)
        work[i] *= cos_win[i];

    kiss_fftr(cfg, work, fft_out);
    fft_norm = 1.0f / (float)nfft;
    for (i = 0; i < nfreq; ++i)
    {
        fft_out[i].r *= fft_norm;
        fft_out[i].i *= fft_norm;
    }

    if (full_spectrum)
    {
        for (i = 0; i < output_bins_full; ++i)
            full_spectrum[i] =
                (float)sqrt((double)(fft_out[i].r * fft_out[i].r + fft_out[i].i * fft_out[i].i));
    }

    if (out_num_freq_bins_full)
        *out_num_freq_bins_full = output_bins_full;
    if (out_freq_res_full)
        *out_freq_res_full = freq_res;

fail:
    if (work)
        PULSEG_FREE(work);
    if (cos_win)
        PULSEG_FREE(cos_win);
    if (fft_out)
        PULSEG_FREE(fft_out);
    if (cfg)
        kiss_fftr_free(cfg);
    return result;
}

/* ================================================================== */
/*  Mechanical resonance spectra (static helper from uniform waveforms)           */
/* ================================================================== */

static int calc_mech_resonances_from_uniform(
    pulseg_mech_resonances_spectra *spectra,
    pulseg_diagnostic *diag,
    const pulseg__uniform_grad_waveforms *waveforms,
    float target_spectral_resolution_hz,
    float max_frequency_hz,
    int num_trs,
    float tr_duration_us,
    int num_forbidden_bands,
    const pulseg_forbidden_band *forbidden_bands,
    float peak_log10_threshold,
    float peak_norm_scale,
    float peak_eps,
    float peak_prominence,
    const struct pulseg_sequence_descriptor *desc,
    int start_block,
    int block_count,
    int num_avgs,
    float g_max_hz_per_m,
    int compute_dense_envelope,
    int compute_display_products,
    int compress_trains)
{
    pulseg_diagnostic local_diag;
    int max_samples, result;
    int num_freq_bins_full;
    float freq_res_full;
    int compute_full_spectrum;

    if (!diag)
    {
        pulseg_diagnostic_init(&local_diag);
        diag = &local_diag;
    }
    else
    {
        pulseg_diagnostic_init(diag);
    }

    if (!waveforms || !spectra)
    {
        diag->code = PULSEG_ERR_NULL_POINTER;
        return diag->code;
    }

    memset(spectra, 0, sizeof(*spectra));

    max_samples = waveforms->num_samples;
    if (max_samples <= 0)
    {
        diag->code = PULSEG_ERR_MECH_RESONANCES_NO_WAVEFORM;
        return diag->code;
    }

    spectra->freq_min_hz = 0.0f;
    spectra->freq_spacing_hz = 0.0f;
    spectra->num_freq_bins = 0;
    spectra->num_instances = num_trs;

    /* Full-TR magnitude spectrum (kissfft over the uniform waveform).
     * Display-only: spectrum_full_* is a plotting product, and the A_eq
     * verdict below is computed from the structural event model, not from
     * this FFT. Requires BOTH a real resolution and a caller that wants
     * display products — the plotting path (pulseg_calc_mech_resonances)
     * wants them, the headless PSD predownload gate (pulseg_check_safety)
     * does not and must not pay for a spectrum it never reads. */
    compute_full_spectrum = (target_spectral_resolution_hz > 0.0f) && compute_display_products;
    if (compute_full_spectrum)
    {
        /* First pass: determine output bin count + frequency resolution. */
        result = compute_sequence_spectrum(
            NULL,
            &num_freq_bins_full,
            &freq_res_full,
            waveforms->gx,
            waveforms->num_samples,
            waveforms->raster_us,
            target_spectral_resolution_hz,
            max_frequency_hz);
        if (PULSEG_FAILED(result))
        {
            pulseg_mech_resonances_spectra_free(spectra);
            diag->code = result;
            return result;
        }

        spectra->freq_spacing_hz = freq_res_full;
        spectra->num_freq_bins = num_freq_bins_full;

        spectra->spectrum_full_gx =
            (float *)PULSEG_ALLOC((size_t)num_freq_bins_full * sizeof(float));
        spectra->spectrum_full_gy =
            (float *)PULSEG_ALLOC((size_t)num_freq_bins_full * sizeof(float));
        spectra->spectrum_full_gz =
            (float *)PULSEG_ALLOC((size_t)num_freq_bins_full * sizeof(float));
        if (!spectra->spectrum_full_gx || !spectra->spectrum_full_gy || !spectra->spectrum_full_gz)
        {
            pulseg_mech_resonances_spectra_free(spectra);
            diag->code = PULSEG_ERR_ALLOC_FAILED;
            return diag->code;
        }

        result = compute_sequence_spectrum(
            spectra->spectrum_full_gx,
            NULL,
            NULL,
            waveforms->gx,
            waveforms->num_samples,
            waveforms->raster_us,
            target_spectral_resolution_hz,
            max_frequency_hz);
        if (PULSEG_FAILED(result))
        {
            pulseg_mech_resonances_spectra_free(spectra);
            diag->code = result;
            return result;
        }
        result = compute_sequence_spectrum(
            spectra->spectrum_full_gy,
            NULL,
            NULL,
            waveforms->gy,
            waveforms->num_samples,
            waveforms->raster_us,
            target_spectral_resolution_hz,
            max_frequency_hz);
        if (PULSEG_FAILED(result))
        {
            pulseg_mech_resonances_spectra_free(spectra);
            diag->code = result;
            return result;
        }
        result = compute_sequence_spectrum(
            spectra->spectrum_full_gz,
            NULL,
            NULL,
            waveforms->gz,
            waveforms->num_samples,
            waveforms->raster_us,
            target_spectral_resolution_hz,
            max_frequency_hz);
        if (PULSEG_FAILED(result))
        {
            pulseg_mech_resonances_spectra_free(spectra);
            diag->code = result;
            return result;
        }
    }

    /* ---- structural acoustic analysis (canonical verdict) ---- */
    if (desc && block_count > 0)
    {
        result = sa_check_structural_violations(
            spectra,
            desc,
            start_block,
            block_count,
            num_trs,
            tr_duration_us,
            num_forbidden_bands,
            forbidden_bands,
            peak_log10_threshold,
            peak_norm_scale,
            peak_eps,
            peak_prominence,
            num_avgs,
            g_max_hz_per_m,
            compute_dense_envelope,
            compute_display_products,
            compress_trains);
        if (PULSEG_FAILED(result))
        {
            pulseg_mech_resonances_spectra_free(spectra);
            diag->code = result;
            return result;
        }
    }

    diag->code = PULSEG_SUCCESS;
    return PULSEG_SUCCESS;
}

/* Select canonical TR window for safety/plotting wrappers.
 * Canonical geometry is always extracted with AMP_MAX_POS.
 * Non-degenerate prep/cooldown: full-pass canonical TR (pass-expanded).
 * Degenerate prep/cooldown: imaging TR canonical window (no pass expansion). */
static void select_canonical_tr_window(
    const pulseg_sequence_descriptor *desc,
    int *start_block,
    int *block_count,
    int *amplitude_mode,
    int *num_instances,
    float *tr_duration_us)
{
    const pulseg_tr_descriptor *trd;
    int has_nd_prep, has_nd_cool, n;

    trd = &desc->tr_descriptor;
    has_nd_prep = (trd->num_prep_blocks > 0 && !trd->degenerate_prep);
    has_nd_cool = (trd->num_cooldown_blocks > 0 && !trd->degenerate_cooldown);

    if (has_nd_prep || has_nd_cool)
    {
        *start_block = 0;
        *block_count = desc->pass_len;
        *amplitude_mode = PULSEG_AMP_MAX_POS;
        *num_instances = (desc->num_passes > 1) ? desc->num_passes : 1;

        *tr_duration_us = 0.0f;
        for (n = 0; n < desc->pass_len; ++n)
        {
            const pulseg_block_table_element *bte = &desc->block_table[n];
            const pulseg_base_block *bdef = &desc->base_blocks[bte->id];
            *tr_duration_us +=
                (bte->duration_us >= 0) ? (float)bte->duration_us : (float)bdef->duration_us;
        }
        return;
    }

    *start_block = trd->num_prep_blocks + trd->imaging_tr_start;
    *block_count = trd->tr_size;
    *amplitude_mode = PULSEG_AMP_MAX_POS;
    /* F8.2: align with select_canonical_tr_window_idx's degenerate
     * branch (:55), which multiplies by num_averages; display-only
     * (min-2 clamp + FWHM), does not change the structural/spectral verdict. */
    {
        int num_avgs = (desc->num_averages > 1) ? desc->num_averages : 1;
        *num_instances = trd->num_trs * num_avgs;
    }
    *tr_duration_us = trd->tr_duration_us;
}

/* Find unique shot-index patterns across pass-expanded canonical windows.
 * Returns number of unique pass patterns, 0 on allocation/shape failure.
 * Caller must free out arrays when return > 0. */
int pulseg__find_unique_shot_passes(
    const pulseg_sequence_descriptor *desc,
    int **out_unique_pass_indices,
    int **out_pass_group_labels)
{
    int num_passes, pass_size, num_cols;
    int *rows;
    int *unique_defs;
    int *event_table;
    int *result_indices;
    int p, pos, col, bt_pos, block_idx, raw_id;
    int num_unique, g, match, c;
    const pulseg_block_table_element *bte;
    const pulseg_grad_table_element *gte;

    *out_unique_pass_indices = NULL;
    *out_pass_group_labels = NULL;

    num_passes = (desc->num_passes > 1) ? desc->num_passes : 1;
    pass_size = (num_passes > 0) ? (desc->exec_stream_len / num_passes) : 0;
    if (num_passes <= 0 || pass_size <= 0 || !desc->exec_runs)
        return 0;

    num_cols = pass_size * 3;
    rows = (int *)PULSEG_ALLOC((size_t)num_passes * (size_t)num_cols * sizeof(int));
    unique_defs = (int *)PULSEG_ALLOC((size_t)num_passes * sizeof(int));
    event_table = (int *)PULSEG_ALLOC((size_t)num_passes * sizeof(int));
    if (!rows || !unique_defs || !event_table)
    {
        if (rows)
            PULSEG_FREE(rows);
        if (unique_defs)
            PULSEG_FREE(unique_defs);
        if (event_table)
            PULSEG_FREE(event_table);
        return 0;
    }

    for (p = 0; p < num_passes; ++p)
    {
        col = 0;
        for (pos = 0; pos < pass_size; ++pos)
        {
            bt_pos = p * pass_size + pos;
            if (bt_pos < 0 || bt_pos >= desc->exec_stream_len)
            {
                rows[p * num_cols + col++] = -1;
                rows[p * num_cols + col++] = -1;
                rows[p * num_cols + col++] = -1;
                continue;
            }
            block_idx = pulseg__exec_block_idx(desc, bt_pos);
            if (block_idx < 0 || block_idx >= desc->num_blocks)
            {
                rows[p * num_cols + col++] = -1;
                rows[p * num_cols + col++] = -1;
                rows[p * num_cols + col++] = -1;
                continue;
            }

            bte = &desc->block_table[block_idx];

            raw_id = bte->gx_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size)
            {
                gte = &desc->grad_table[raw_id];
                rows[p * num_cols + col] = gte->shot_index;
            }
            else
                rows[p * num_cols + col] = -1;
            col++;

            raw_id = bte->gy_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size)
            {
                gte = &desc->grad_table[raw_id];
                rows[p * num_cols + col] = gte->shot_index;
            }
            else
                rows[p * num_cols + col] = -1;
            col++;

            raw_id = bte->gz_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size)
            {
                gte = &desc->grad_table[raw_id];
                rows[p * num_cols + col] = gte->shot_index;
            }
            else
                rows[p * num_cols + col] = -1;
            col++;
        }
    }

    num_unique = 0;
    for (p = 0; p < num_passes; ++p)
    {
        match = -1;
        for (g = 0; g < num_unique; ++g)
        {
            int rep = unique_defs[g];
            int equal = 1;
            for (c = 0; c < num_cols; ++c)
            {
                if (rows[p * num_cols + c] != rows[rep * num_cols + c])
                {
                    equal = 0;
                    break;
                }
            }
            if (equal)
            {
                match = g;
                break;
            }
        }
        if (match >= 0)
        {
            event_table[p] = match;
        }
        else
        {
            unique_defs[num_unique] = p;
            event_table[p] = num_unique;
            num_unique++;
        }
    }

    result_indices = (int *)PULSEG_ALLOC((size_t)num_unique * sizeof(int));
    if (!result_indices)
    {
        PULSEG_FREE(rows);
        PULSEG_FREE(unique_defs);
        PULSEG_FREE(event_table);
        return 0;
    }
    for (g = 0; g < num_unique; ++g)
        result_indices[g] = unique_defs[g];

    *out_unique_pass_indices = result_indices;
    *out_pass_group_labels = event_table;

    PULSEG_FREE(rows);
    PULSEG_FREE(unique_defs);
    return num_unique;
}

/* ================================================================== */
/*  Acoustic spectra (public wrapper)                                 */
/* ================================================================== */

int pulseg_calc_mech_resonances(
    const pulseg_collection *coll,
    pulseg_mech_resonances_spectra *spectra,
    pulseg_diagnostic *diag,
    int subseq_idx,
    int canonical_tr_idx,
    const pulseg_opts *opts,
    float target_resolution_hz,
    float max_freq_hz,
    int num_forbidden_bands,
    const pulseg_forbidden_band *forbidden_bands,
    int compress_trains)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_descriptor *trd;
    pulseg__uniform_grad_waveforms uw;
    pulseg_diagnostic local_diag;
    int rc, start_block, block_count, amplitude_mode, num_instances;
    int sa_start_block, sa_block_count, num_avgs;
    int *block_order;
    int has_nd_prep, has_nd_cool;
    float tr_duration_us;
    float peak_log10_threshold;
    float peak_norm_scale;
    float peak_eps;
    float peak_prominence;

    memset(&uw, 0, sizeof(uw));
    block_order = NULL;
    if (!diag)
    {
        pulseg_diagnostic_init(&local_diag);
        diag = &local_diag;
    }
    else
    {
        pulseg_diagnostic_init(diag);
    }
    if (!coll || !spectra)
    {
        diag->code = PULSEG_ERR_NULL_POINTER;
        return diag->code;
    }
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
    {
        diag->code = PULSEG_ERR_INVALID_ARGUMENT;
        return diag->code;
    }

    peak_log10_threshold = PULSEG_PEAK_LOG10_THRESHOLD_DEFAULT;
    peak_norm_scale = PULSEG_PEAK_NORM_SCALE_DEFAULT;
    peak_eps = PULSEG_PEAK_EPS_DEFAULT;
    peak_prominence = PULSEG_PEAK_PROMINENCE_DEFAULT;
    if (opts)
    {
        peak_log10_threshold = opts->peak_log10_threshold;
        peak_norm_scale = opts->peak_norm_scale;
        peak_eps = opts->peak_eps;
        peak_prominence = opts->peak_prominence;
    }

    desc = &coll->descriptors[subseq_idx];
    trd = &desc->tr_descriptor;
    /* Select the canonical TR window for the given canonical_tr_idx */
    if (canonical_tr_idx < 0 || canonical_tr_idx >= desc->tr_descriptor.num_trs)
    {
        diag->code = PULSEG_ERR_INVALID_ARGUMENT;
        return diag->code;
    }
    /* Use a new helper to select the correct window for the canonical_tr_idx */
    select_canonical_tr_window_idx(
        desc,
        &start_block,
        &block_count,
        &amplitude_mode,
        &num_instances,
        &tr_duration_us,
        canonical_tr_idx);

    sa_start_block = start_block;
    sa_block_count = block_count;
    num_avgs = 1;

    has_nd_prep = (trd->num_prep_blocks > 0 && !trd->degenerate_prep);
    has_nd_cool = (trd->num_cooldown_blocks > 0 && !trd->degenerate_cooldown);
    if (has_nd_prep || has_nd_cool)
    {
        num_avgs = (desc->num_averages > 1) ? desc->num_averages : 1;
        rc = build_pass_expanded_block_order(
            desc,
            &block_order,
            &block_count,
            &tr_duration_us,
            start_block);
        if (PULSEG_FAILED(rc))
        {
            diag->code = rc;
            return rc;
        }
        start_block = 0;
    }

    rc = pulseg__get_gradient_waveforms_range(
        desc,
        &uw,
        diag,
        start_block,
        block_count,
        amplitude_mode,
        NULL,
        0,
        block_order);
    if (PULSEG_FAILED(rc))
    {
        if (block_order)
            PULSEG_FREE(block_order);
        return rc;
    }
    rc = calc_mech_resonances_from_uniform(spectra, diag, &uw,
                                           target_resolution_hz, max_freq_hz,
                                           num_instances,
                                           tr_duration_us,
                                           num_forbidden_bands, forbidden_bands,
                                           peak_log10_threshold, peak_norm_scale, peak_eps,
                                           peak_prominence,
                                           desc, sa_start_block, sa_block_count, num_avgs,
                                           /* Real G_max, not 0: sa_eps_for_band() falls back to
                                            * SA_AEQ_K_GMAX * G_max for a zero-tolerance band, and
                                            * a 0 here collapsed that floor to 0, so every
                                            * candidate of a zero-tolerance band came back flagged.
                                            * Vendor ESP tables are exactly that shape (amplitude
                                            * column 0.0), so the plotting API could not reproduce
                                            * the headless verdict for any real lockout table. */
                                           (opts ? opts->max_grad_hz_per_m : 0.0f),
                                           (target_resolution_hz > 0.0f) ? 1 : 0,
                                           /* dense envelope: plotting API only, see
                                            * calc_mech_resonances_from_uniform doc */
                                           1 /* display products: this IS the plotting API */,
                                           compress_trains);
    pulseg__uniform_grad_waveforms_free(&uw);
    if (block_order)
        PULSEG_FREE(block_order);
    return rc;
}

/* ================================================================== */
/*  PNS                                                               */
/* ================================================================== */

/* FIX: output before inputs, C89-compliant declarations */
static void compute_slew_rate(
    float *slew_out,
    const float *waveform,
    int num_samples,
    float dt_us,
    float gamma_hz_per_tesla)
{
    int i;
    float dt_s;
    float inv_g;

    dt_s = dt_us * 1e-6f;
    inv_g = 1.0f / gamma_hz_per_tesla;

    for (i = 0; i < num_samples - 1; ++i)
        slew_out[i] = ((waveform[i + 1] - waveform[i]) * inv_g) / dt_s;
}

/* Builds one axis's circularly-padded dG/dt: appends `pad` extra samples
 * (wrapped from the start of the waveform) before differentiating, so the
 * injected PNS model sees a fully "warmed up" history. Neutral -- `pad`
 * comes from the model's required_padding() query, not from any
 * vendor-specific kernel knowledge here. `padded_scratch` must have room
 * for (num_samples + pad) floats; `dgdt_out` for (num_samples + pad - 1). */
static void build_padded_dgdt(
    float *dgdt_out,
    float *padded_scratch,
    const float *waveform,
    int num_samples,
    int pad,
    float grad_raster_us,
    float gamma_hz_per_tesla)
{
    int i, padded_len;

    padded_len = num_samples + pad;
    for (i = 0; i < num_samples; ++i)
        padded_scratch[i] = waveform[i];
    for (i = 0; i < pad; ++i)
        padded_scratch[num_samples + i] = waveform[i % num_samples];

    compute_slew_rate(dgdt_out, padded_scratch, padded_len, grad_raster_us, gamma_hz_per_tesla);
}

static int calc_pns_from_uniform(
    pulseg_pns_result *result,
    pulseg_diagnostic *diag,
    float gamma_hz_per_tesla,
    const pulseg__uniform_grad_waveforms *waveforms,
    const pulseg_pns_model *model)
{
    pulseg_diagnostic local_diag;
    int max_samples, pad, n;
    float *padded_scratch;
    float *dgdt_x;
    float *dgdt_y;
    float *dgdt_z;
    float *out_x;
    float *out_y;
    float *out_z;
    int rc;

    padded_scratch = NULL;
    dgdt_x = NULL;
    dgdt_y = NULL;
    dgdt_z = NULL;
    out_x = NULL;
    out_y = NULL;
    out_z = NULL;
    rc = PULSEG_SUCCESS;

    if (!diag)
    {
        pulseg_diagnostic_init(&local_diag);
        diag = &local_diag;
    }
    else
    {
        pulseg_diagnostic_init(diag);
    }

    if (!waveforms || !model || !model->evaluate || !model->required_padding || !result)
    {
        diag->code = PULSEG_ERR_NULL_POINTER;
        return diag->code;
    }

    memset(result, 0, sizeof(*result));

    max_samples = waveforms->num_samples;
    if (max_samples <= 1)
    {
        diag->code = PULSEG_ERR_PNS_NO_WAVEFORM;
        return diag->code;
    }

    pad = model->required_padding(model->ctx, waveforms->raster_us);
    if (pad < 0)
    {
        diag->code = PULSEG_ERR_PNS_INVALID_PARAMS;
        return diag->code;
    }
    n = max_samples + pad - 1;

    padded_scratch = (float *)PULSEG_ALLOC((size_t)(max_samples + pad) * sizeof(float));
    dgdt_x = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    dgdt_y = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    dgdt_z = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    out_x = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    out_y = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    out_z = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    if (!padded_scratch || !dgdt_x || !dgdt_y || !dgdt_z || !out_x || !out_y || !out_z)
    {
        rc = PULSEG_ERR_ALLOC_FAILED;
        goto fail;
    }

    build_padded_dgdt(
        dgdt_x,
        padded_scratch,
        waveforms->gx,
        max_samples,
        pad,
        waveforms->raster_us,
        gamma_hz_per_tesla);
    build_padded_dgdt(
        dgdt_y,
        padded_scratch,
        waveforms->gy,
        max_samples,
        pad,
        waveforms->raster_us,
        gamma_hz_per_tesla);
    build_padded_dgdt(
        dgdt_z,
        padded_scratch,
        waveforms->gz,
        max_samples,
        pad,
        waveforms->raster_us,
        gamma_hz_per_tesla);

    rc = model->evaluate(
        model->ctx,
        dgdt_x,
        dgdt_y,
        dgdt_z,
        n,
        waveforms->raster_us,
        out_x,
        out_y,
        out_z);
    if (PULSEG_FAILED(rc))
    {
        diag->code = rc;
        goto fail;
    }

    result->num_samples = n;
    result->slew_x_hz_per_m_per_s = out_x;
    out_x = NULL;
    result->slew_y_hz_per_m_per_s = out_y;
    out_y = NULL;
    result->slew_z_hz_per_m_per_s = out_z;
    out_z = NULL;

    rc = PULSEG_SUCCESS;
    diag->code = rc;

fail:
    if (padded_scratch)
        PULSEG_FREE(padded_scratch);
    if (dgdt_x)
        PULSEG_FREE(dgdt_x);
    if (dgdt_y)
        PULSEG_FREE(dgdt_y);
    if (dgdt_z)
        PULSEG_FREE(dgdt_z);
    if (out_x)
        PULSEG_FREE(out_x);
    if (out_y)
        PULSEG_FREE(out_y);
    if (out_z)
        PULSEG_FREE(out_z);
    return rc;
}

/* ================================================================== */
/*  PNS (public wrapper)                                              */
/* ================================================================== */

int pulseg_calc_pns(
    const pulseg_collection *coll,
    pulseg_pns_result *result,
    pulseg_diagnostic *diag,
    int subseq_idx,
    int canonical_tr_idx,
    const pulseg_opts *opts,
    const pulseg_pns_model *model)
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_descriptor *trd;
    pulseg__uniform_grad_waveforms uw;
    pulseg_diagnostic local_diag;
    int rc, start_block, block_count, amplitude_mode, num_instances;
    int *block_order;
    int has_nd_prep, has_nd_cool;
    float tr_duration_us;

    memset(&uw, 0, sizeof(uw));
    block_order = NULL;
    if (!diag)
    {
        pulseg_diagnostic_init(&local_diag);
        diag = &local_diag;
    }
    else
    {
        pulseg_diagnostic_init(diag);
    }
    if (!coll || !result || !model || !opts)
    {
        /* F8.1: opts->gamma_hz_per_t is dereferenced below (calc_pns_from_uniform) with
         * no prior NULL check -- public API crash on NULL opts. */
        diag->code = PULSEG_ERR_NULL_POINTER;
        return diag->code;
    }
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
    {
        diag->code = PULSEG_ERR_INVALID_ARGUMENT;
        return diag->code;
    }
    desc = &coll->descriptors[subseq_idx];
    trd = &desc->tr_descriptor;
    if (canonical_tr_idx < 0 || canonical_tr_idx >= desc->tr_descriptor.num_trs)
    {
        diag->code = PULSEG_ERR_INVALID_ARGUMENT;
        return diag->code;
    }
    select_canonical_tr_window_idx(
        desc,
        &start_block,
        &block_count,
        &amplitude_mode,
        &num_instances,
        &tr_duration_us,
        canonical_tr_idx);
    (void)num_instances;
    (void)tr_duration_us;

    has_nd_prep = (trd->num_prep_blocks > 0 && !trd->degenerate_prep);
    has_nd_cool = (trd->num_cooldown_blocks > 0 && !trd->degenerate_cooldown);
    if (has_nd_prep || has_nd_cool)
    {
        rc = build_pass_expanded_block_order(desc, &block_order, &block_count, NULL, start_block);
        if (PULSEG_FAILED(rc))
        {
            diag->code = rc;
            return rc;
        }
        start_block = 0;
    }

    rc = pulseg__get_gradient_waveforms_range(
        desc,
        &uw,
        diag,
        start_block,
        block_count,
        amplitude_mode,
        NULL,
        0,
        block_order);
    if (PULSEG_FAILED(rc))
    {
        if (block_order)
            PULSEG_FREE(block_order);
        return rc;
    }
    rc = calc_pns_from_uniform(result, diag, opts->gamma_hz_per_t, &uw, model);
    pulseg__uniform_grad_waveforms_free(&uw);
    if (block_order)
        PULSEG_FREE(block_order);
    return rc;
}

/* ================================================================== */
/*  PNS result free                                                   */
/* ================================================================== */

void pulseg_pns_result_free(pulseg_pns_result *r)
{
    if (!r)
        return;
    if (r->slew_x_hz_per_m_per_s)
    {
        PULSEG_FREE(r->slew_x_hz_per_m_per_s);
        r->slew_x_hz_per_m_per_s = NULL;
    }
    if (r->slew_y_hz_per_m_per_s)
    {
        PULSEG_FREE(r->slew_y_hz_per_m_per_s);
        r->slew_y_hz_per_m_per_s = NULL;
    }
    if (r->slew_z_hz_per_m_per_s)
    {
        PULSEG_FREE(r->slew_z_hz_per_m_per_s);
        r->slew_z_hz_per_m_per_s = NULL;
    }
    r->num_samples = 0;
}

/* ================================================================== */
/*  Collection-level safety check                                     */
/* ================================================================== */

static int check_max_grad(
    const pulseg_collection *coll,
    pulseg_diagnostic *diag,
    const pulseg_opts *opts)
{
    int s, b, raw_id;
    int worst_subseq, worst_block;
    float gx_amp, gy_amp, gz_amp, gsos, gsos_max, limit_sq, hz_per_mt;
    const pulseg_sequence_descriptor *desc;
    const pulseg_block_table_element *bte;

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

    /* ---- max gradient amplitude (GSOS) check ----
     *
     * `opts->max_grad_hz_per_m` is the per-axis limit already derated by sqrt(3)
     * upstream (in pulserver_init_opts) so that, under an arbitrary rotation,
     * no single physical axis exceeds the scanner's hardware gmax. The vector
     * magnitude (GSOS) of the *unrotated* waveform, however, is the quantity
     * that bounds every rotated axis component; hence the proper GSOS bound is
     * sqrt(3) * per-axis-derated = physical gmax. Compare GSOS^2 against
     * 3 * (per-axis derated)^2.
     */
    gsos_max = 0.0f;
    limit_sq = 3.0f * opts->max_grad_hz_per_m * opts->max_grad_hz_per_m;
    worst_subseq = 0;
    worst_block = 0;

    for (s = 0; s < coll->num_subsequences; ++s)
    {
        desc = &coll->descriptors[s];
        for (b = 0; b < desc->num_blocks; ++b)
        {
            bte = &desc->block_table[b];

            gx_amp = 0.0f;
            raw_id = bte->gx_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size)
                gx_amp = desc->grad_table[raw_id].amplitude;

            gy_amp = 0.0f;
            raw_id = bte->gy_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size)
                gy_amp = desc->grad_table[raw_id].amplitude;

            gz_amp = 0.0f;
            raw_id = bte->gz_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size)
                gz_amp = desc->grad_table[raw_id].amplitude;

            gsos = gx_amp * gx_amp + gy_amp * gy_amp + gz_amp * gz_amp;
            if (gsos > gsos_max)
            {
                gsos_max = gsos;
                worst_subseq = s;
                worst_block = b;
            }
        }
    }

    if (limit_sq > 0.0f && gsos_max > limit_sq)
    {
        hz_per_mt = opts->gamma_hz_per_t * 0.001f;
        if (diag)
        {
            float physical_limit_hz_per_m = (float)sqrt(3.0) * opts->max_grad_hz_per_m;
            diag->code = PULSEG_ERR_MAX_GRAD_EXCEEDED;
            pulseg__diag_printf(
                diag,
                "amp=%.2fmT/m>%.2fmT/m,s=%d,b=%d",
                (double)((float)sqrt((double)gsos_max) / hz_per_mt),
                (double)(physical_limit_hz_per_m / hz_per_mt),
                worst_subseq,
                worst_block);
        }
        return PULSEG_ERR_MAX_GRAD_EXCEEDED;
    }

    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Gradient continuity check (cursor dry-run over n repetitions)     */
/* ================================================================== */

static int check_grad_continuity(
    pulseg_collection *coll,
    pulseg_diagnostic *diag,
    const pulseg_opts *opts)
{
    pulseg_block_cursor saved_cursor;
    const pulseg_sequence_descriptor *desc;
    const pulseg_block_table_element *bte;
    const pulseg_base_block *bdef;
    const pulseg_grad_definition *gdef;
    const pulseg_grad_table_element *gte;
    int n, raw_id, rot_id, status, cur_seq;
    int grad_def_ids[3];
    int shot_idx[3];
    float amp[3], first_val[3], last_val[3];
    float first_phys[3], last_phys[3], prev_phys[3];
    float max_allowed, grad_raster_s, step, hz_per_mt;

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

    /* save cursor state */
    saved_cursor = coll->block_cursor;
    coll->block_cursor.sequence_index = 0;
    coll->block_cursor.exec_stream_position = 0;
    coll->block_cursor.from_last_reset = 0;

    prev_phys[0] = 0.0f;
    prev_phys[1] = 0.0f;
    prev_phys[2] = 0.0f;
    cur_seq = 0;
    status = PULSEG_CURSOR_BLOCK;

    desc = &coll->descriptors[0];
    grad_raster_s = desc->grad_raster_us * 1e-6f;
    max_allowed = opts->max_slew_hz_per_m_per_s * grad_raster_s;

    while (status != PULSEG_CURSOR_DONE)
    {
        /* detect subsequence change */
        if (coll->block_cursor.sequence_index != cur_seq)
        {
            /* end-of-subsequence: prev must ramp to zero */
            for (n = 0; n < 3; ++n)
            {
                step = prev_phys[n];
                if (step < 0.0f)
                    step = -step;
                if (step > max_allowed)
                {
                    hz_per_mt = opts->gamma_hz_per_t * 0.001f;
                    if (diag)
                    {
                        diag->code = PULSEG_ERR_GRAD_DISCONTINUITY;
                        pulseg__diag_printf(
                            diag,
                            "step=%.2fmT/m>%.2fmT/m,a=%d,bnd=1",
                            (double)(step / hz_per_mt),
                            (double)(max_allowed / hz_per_mt),
                            n);
                    }
                    coll->block_cursor = saved_cursor;
                    return PULSEG_ERR_GRAD_DISCONTINUITY;
                }
            }

            cur_seq = coll->block_cursor.sequence_index;
            prev_phys[0] = 0.0f;
            prev_phys[1] = 0.0f;
            prev_phys[2] = 0.0f;

            desc = &coll->descriptors[cur_seq];
            grad_raster_s = desc->grad_raster_us * 1e-6f;
            max_allowed = opts->max_slew_hz_per_m_per_s * grad_raster_s;
        }

        /* read current block */
        {
            int bt_idx = pulseg__exec_block_idx(desc, coll->block_cursor.exec_stream_position);
            bte = &desc->block_table[bt_idx];
        }
        bdef = &desc->base_blocks[bte->id];

        /* grad table: amplitude + shot_index */
        grad_def_ids[0] = bdef->gx_id;
        grad_def_ids[1] = bdef->gy_id;
        grad_def_ids[2] = bdef->gz_id;

        raw_id = bte->gx_id;
        if (raw_id >= 0 && raw_id < desc->grad_table_size)
        {
            gte = &desc->grad_table[raw_id];
            amp[0] = gte->amplitude;
            shot_idx[0] = gte->shot_index;
        }
        else
        {
            amp[0] = 0.0f;
            shot_idx[0] = 0;
        }

        raw_id = bte->gy_id;
        if (raw_id >= 0 && raw_id < desc->grad_table_size)
        {
            gte = &desc->grad_table[raw_id];
            amp[1] = gte->amplitude;
            shot_idx[1] = gte->shot_index;
        }
        else
        {
            amp[1] = 0.0f;
            shot_idx[1] = 0;
        }

        raw_id = bte->gz_id;
        if (raw_id >= 0 && raw_id < desc->grad_table_size)
        {
            gte = &desc->grad_table[raw_id];
            amp[2] = gte->amplitude;
            shot_idx[2] = gte->shot_index;
        }
        else
        {
            amp[2] = 0.0f;
            shot_idx[2] = 0;
        }

        /* first_value / last_value from grad definitions, scaled by amplitude */
        for (n = 0; n < 3; ++n)
        {
            if (grad_def_ids[n] >= 0 && grad_def_ids[n] < desc->num_unique_grads)
            {
                gdef = &desc->grad_definitions[grad_def_ids[n]];
                first_val[n] = gdef->first_value[shot_idx[n]] * amp[n];
                last_val[n] = gdef->last_value[shot_idx[n]] * amp[n];
            }
            else
            {
                first_val[n] = 0.0f;
                last_val[n] = 0.0f;
            }
        }

        /* transform logical -> physical */
        rot_id = bte->rotation_id;
        if (rot_id >= 0 && rot_id < desc->num_rotations)
        {
            pulseg__apply_rotation(first_phys, desc->rotation_matrices[rot_id], first_val, 1);
            pulseg__apply_rotation(last_phys, desc->rotation_matrices[rot_id], last_val, 1);
        }
        else
        {
            first_phys[0] = first_val[0];
            first_phys[1] = first_val[1];
            first_phys[2] = first_val[2];
            last_phys[0] = last_val[0];
            last_phys[1] = last_val[1];
            last_phys[2] = last_val[2];
        }

        /* continuity check */
        for (n = 0; n < 3; ++n)
        {
            step = first_phys[n] - prev_phys[n];
            if (step < 0.0f)
                step = -step;
            if (step > max_allowed)
            {
                hz_per_mt = opts->gamma_hz_per_t * 0.001f;
                if (diag)
                {
                    diag->code = PULSEG_ERR_GRAD_DISCONTINUITY;
                    pulseg__diag_printf(
                        diag,
                        "step=%.2fmT/m>%.2fmT/m,a=%d,p=%d",
                        (double)(step / hz_per_mt),
                        (double)(max_allowed / hz_per_mt),
                        n,
                        coll->block_cursor.exec_stream_position);
                }
                coll->block_cursor = saved_cursor;
                return PULSEG_ERR_GRAD_DISCONTINUITY;
            }
        }

        prev_phys[0] = last_phys[0];
        prev_phys[1] = last_phys[1];
        prev_phys[2] = last_phys[2];

        /* advance cursor */
        status = pulseg_cursor_next(coll);
    }

    /* final subsequence trailing edge */
    for (n = 0; n < 3; ++n)
    {
        step = prev_phys[n];
        if (step < 0.0f)
            step = -step;
        if (step > max_allowed)
        {
            hz_per_mt = opts->gamma_hz_per_t * 0.001f;
            if (diag)
            {
                diag->code = PULSEG_ERR_GRAD_DISCONTINUITY;
                pulseg__diag_printf(
                    diag,
                    "step=%.2fmT/m>%.2fmT/m,a=%d,tail=1",
                    (double)(step / hz_per_mt),
                    (double)(max_allowed / hz_per_mt),
                    n);
            }
            coll->block_cursor = saved_cursor;
            return PULSEG_ERR_GRAD_DISCONTINUITY;
        }
    }

    coll->block_cursor = saved_cursor;
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Max slew rate check (per unique block definition)                 */
/* ================================================================== */

static int check_max_slew(
    const pulseg_collection *coll,
    pulseg_diagnostic *diag,
    const pulseg_opts *opts)
{
    /* ---- max slew rate (GSOS) check ----
     *
     * `opts->max_slew_hz_per_m_per_s` is the per-axis limit already derated by
     * sqrt(3) upstream (in pulserver_init_opts), mirroring max_grad.  The GSOS
     * bound (vector magnitude of per-axis slew rates) therefore corresponds to
     * sqrt(3) * derated = physical slew limit.  Compare GSOS_slew^2 against
     * 3 * (derated)^2.
     *
     * Slew per axis at each block = slew_rate_normalised * amplitude, where
     * slew_rate_normalised (1/s) comes from grad_definitions and amplitude from
     * grad_table (the per-block-instance value).  This mirrors check_max_grad
     * which also iterates block_table for per-instance amplitudes.
     */
    int s, b, n, raw_id, def_idx, shot_idx;
    float slew_sq, slew_sq_max, limit_sq, amp;
    float axis_slew[3];
    int raw_ids[3];
    int worst_subseq, worst_block;
    const pulseg_sequence_descriptor *desc;
    const pulseg_block_table_element *bte;
    const pulseg_grad_definition *gdef;

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

    slew_sq_max = 0.0f;
    limit_sq = 3.0f * opts->max_slew_hz_per_m_per_s * opts->max_slew_hz_per_m_per_s;
    worst_subseq = 0;
    worst_block = 0;

    for (s = 0; s < coll->num_subsequences; ++s)
    {
        desc = &coll->descriptors[s];

        for (b = 0; b < desc->num_blocks; ++b)
        {
            bte = &desc->block_table[b];
            raw_ids[0] = bte->gx_id;
            raw_ids[1] = bte->gy_id;
            raw_ids[2] = bte->gz_id;

            for (n = 0; n < 3; ++n)
            {
                axis_slew[n] = 0.0f;
                raw_id = raw_ids[n];
                if (raw_id < 0 || raw_id >= desc->grad_table_size)
                    continue;

                def_idx = desc->grad_table[raw_id].id;
                shot_idx = desc->grad_table[raw_id].shot_index;
                amp = desc->grad_table[raw_id].amplitude;
                if (amp < 0.0f)
                    amp = -amp;

                if (def_idx < 0 || def_idx >= desc->num_unique_grads)
                    continue;

                gdef = &desc->grad_definitions[def_idx];
                if (shot_idx >= 0 && shot_idx < gdef->num_shots)
                    axis_slew[n] = gdef->slew_rate[shot_idx] * amp;
            }

            slew_sq = axis_slew[0] * axis_slew[0] + axis_slew[1] * axis_slew[1] +
                axis_slew[2] * axis_slew[2];
            if (slew_sq > slew_sq_max)
            {
                slew_sq_max = slew_sq;
                worst_subseq = s;
                worst_block = b;
            }
        }
    }

    if (slew_sq_max > limit_sq)
    {
        if (diag)
        {
            float physical_limit = (float)sqrt(3.0) * opts->max_slew_hz_per_m_per_s;
            diag->code = PULSEG_ERR_MAX_SLEW_EXCEEDED;
            pulseg__diag_printf(
                diag,
                "slew_rss=%.2f>%.2fHz/m/s,s=%d,b=%d",
                (double)sqrt((double)slew_sq_max),
                (double)physical_limit,
                worst_subseq,
                worst_block);
        }
        return PULSEG_ERR_MAX_SLEW_EXCEEDED;
    }

    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Gradient presence (skip gate for gradient-only safety checks)     */
/* ================================================================== */

/* Returns 1 if any block in the collection has a nonzero gx/gy/gz
 * amplitude, 0 if every gradient channel is silent throughout (e.g. an
 * RF-only or pure-delay sequence). Mirrors the block_table walk in
 * check_max_grad() but stops at the first nonzero sample. */
static int collection_has_gradient(const pulseg_collection *coll)
{
    int s, b, raw_id;
    const pulseg_sequence_descriptor *desc;
    const pulseg_block_table_element *bte;

    for (s = 0; s < coll->num_subsequences; ++s)
    {
        desc = &coll->descriptors[s];
        for (b = 0; b < desc->num_blocks; ++b)
        {
            bte = &desc->block_table[b];

            raw_id = bte->gx_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size &&
                desc->grad_table[raw_id].amplitude != 0.0f)
                return 1;

            raw_id = bte->gy_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size &&
                desc->grad_table[raw_id].amplitude != 0.0f)
                return 1;

            raw_id = bte->gz_id;
            if (raw_id >= 0 && raw_id < desc->grad_table_size &&
                desc->grad_table[raw_id].amplitude != 0.0f)
                return 1;
        }
    }
    return 0;
}

/* ================================================================== */
/*  Safety check                                                      */
/* ================================================================== */
int pulseg__check_safety_profiled(
    pulseg_collection *coll,
    pulseg_diagnostic *diag,
    const pulseg_opts *opts,
    int num_forbidden_bands,
    const pulseg_forbidden_band *forbidden_bands,
    const pulseg_pns_model *pns_model,
    float pns_threshold_percent,
    pulseg__safety_profile_fn profile_fn,
    void *profile_ctx)
{
    int rc, s, u, i;
    int ci, b;
    int num_unique_trs;
    int *unique_tr_indices;
    int *tr_group_labels;
    const pulseg_sequence_descriptor *desc;
    const pulseg_tr_descriptor *trd;
    pulseg__uniform_grad_waveforms uw;
    pulseg_mech_resonances_spectra spectra;
    pulseg_pns_result pns_result;
    int start_block, block_count, amplitude_mode, num_instances;
    int sa_start_block, sa_block_count, num_avgs;
    int *block_order;
    int has_nd_prep, has_nd_cool;
    float tr_duration_us;
    float pns_combined, max_pns;
    float cf_hz, ca_hz_per_m;
    int fbi;
    float mr_max_freq_hz, mr_target_res_hz, mr_min_band_width_hz, mr_width;

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

    /* No gradient event anywhere in the collection (e.g. an RF-only or
     * pure-delay sequence): every check below operates on gx/gy/gz, so
     * there is nothing to validate. Skip continuity/slew/mech-resonance/
     * PNS entirely rather than running them over a silent waveform that
     * may span the full sequence duration (RF/SAR safety is evaluated
     * separately and is unaffected by this skip). */
    if (profile_fn)
        profile_fn(profile_ctx, PULSEG__SAFETY_PROFILE_GRAD_PRESENCE, 1);
    rc = collection_has_gradient(coll);
    if (profile_fn)
        profile_fn(profile_ctx, PULSEG__SAFETY_PROFILE_GRAD_PRESENCE, 0);
    if (!rc)
        return PULSEG_SUCCESS;

    /* ---- 1. max gradient amplitude ---- */
    if (profile_fn)
        profile_fn(profile_ctx, PULSEG__SAFETY_PROFILE_MAX_GRAD, 1);
    rc = check_max_grad(coll, diag, opts);
    if (profile_fn)
        profile_fn(profile_ctx, PULSEG__SAFETY_PROFILE_MAX_GRAD, 0);
    if (PULSEG_FAILED(rc))
        return rc;

    /* ---- 2. gradient continuity ---- */
    if (profile_fn)
        profile_fn(profile_ctx, PULSEG__SAFETY_PROFILE_CONTINUITY, 1);
    rc = check_grad_continuity(coll, diag, opts);
    if (profile_fn)
        profile_fn(profile_ctx, PULSEG__SAFETY_PROFILE_CONTINUITY, 0);
    if (PULSEG_FAILED(rc))
        return rc;

    /* ---- 3. max slew rate ---- */
    if (profile_fn)
        profile_fn(profile_ctx, PULSEG__SAFETY_PROFILE_MAX_SLEW, 1);
    rc = check_max_slew(coll, diag, opts);
    if (profile_fn)
        profile_fn(profile_ctx, PULSEG__SAFETY_PROFILE_MAX_SLEW, 0);
    if (PULSEG_FAILED(rc))
        return rc;

    /* Safety path = plotting path: when forbidden bands are configured, drive
     * the structural analysis with a real resolution and max frequency instead
     * of (0,0) (which leaves num_freq_bins == 0 and makes
     * sa_check_structural_violations() a silent no-op). The analysis window is
     * DECOUPLED from the band range (SA_MIN_ANALYSIS_FREQ_HZ): candidate
     * selection normalizes against the peak spectral content over the analyzed
     * range, so it must span the sequence's true dominant feature — which for
     * non-EPI sequences (e.g. GRE) sits above the band — or weak in-band
     * content is spuriously promoted. The candidate→band comparison itself
     * stays band-restricted, so widening the analysis window never widens what
     * can be flagged. This matches the validated grad_spectrum defaults
     * (max_frequency 3000 Hz). */
    mr_max_freq_hz = 0.0f;
    mr_target_res_hz = 0.0f;
    if (num_forbidden_bands > 0)
    {
        mr_min_band_width_hz = -1.0f;
        for (fbi = 0; fbi < num_forbidden_bands; ++fbi)
        {
            if (forbidden_bands[fbi].freq_max_hz > mr_max_freq_hz)
                mr_max_freq_hz = forbidden_bands[fbi].freq_max_hz;

            mr_width = forbidden_bands[fbi].freq_max_hz - forbidden_bands[fbi].freq_min_hz;
            if (mr_min_band_width_hz < 0.0f || mr_width < mr_min_band_width_hz)
                mr_min_band_width_hz = mr_width;
        }
        mr_max_freq_hz *= 1.2f;
        /* Decouple normalization range from band range (see
         * SA_MIN_ANALYSIS_FREQ_HZ): ensure the analyzed spectrum reaches the
         * validated plotting-path range so the per-axis gate and FFT-promotion
         * normalizers capture the true dominant peak, not just the near-band
         * content. */
        if (mr_max_freq_hz < SA_MIN_ANALYSIS_FREQ_HZ)
            mr_max_freq_hz = SA_MIN_ANALYSIS_FREQ_HZ;

        mr_target_res_hz = mr_min_band_width_hz / 4.0f;
        if (mr_target_res_hz < 1.0f)
            mr_target_res_hz = 1.0f;
        else if (mr_target_res_hz > 5.0f)
            mr_target_res_hz = 5.0f;
    }

    /* ---- 4. per-subsequence canonical-TR acoustic + PNS ---- */
    for (s = 0; s < coll->num_subsequences; ++s)
    {
        desc = &coll->descriptors[s];
        trd = &desc->tr_descriptor;
        unique_tr_indices = NULL;
        tr_group_labels = NULL;

        select_canonical_tr_window(
            desc,
            &start_block,
            &block_count,
            &amplitude_mode,
            &num_instances,
            &tr_duration_us);
        (void)amplitude_mode;
        sa_start_block = start_block;
        sa_block_count = block_count;
        num_avgs = 1;
        block_order = NULL;
        has_nd_prep = (trd->num_prep_blocks > 0 && !trd->degenerate_prep);
        has_nd_cool = (trd->num_cooldown_blocks > 0 && !trd->degenerate_cooldown);
        if (has_nd_prep || has_nd_cool)
        {
            num_avgs = (desc->num_averages > 1) ? desc->num_averages : 1;
            rc = build_pass_expanded_block_order(
                desc,
                &block_order,
                &block_count,
                &tr_duration_us,
                start_block);
            if (PULSEG_FAILED(rc))
            {
                if (unique_tr_indices)
                    PULSEG_FREE(unique_tr_indices);
                if (tr_group_labels)
                    PULSEG_FREE(tr_group_labels);
                if (block_order)
                    PULSEG_FREE(block_order);
                return rc;
            }
            start_block = 0;
        }

        /* Evaluate one canonical TR per shot-ID combination. */
        unique_tr_indices = NULL;
        tr_group_labels = NULL;
        if ((trd->num_prep_blocks > 0 && !trd->degenerate_prep) ||
            (trd->num_cooldown_blocks > 0 && !trd->degenerate_cooldown))
        {
            num_unique_trs =
                pulseg__find_unique_shot_passes(desc, &unique_tr_indices, &tr_group_labels);
        }
        else
        {
            num_unique_trs =
                pulseg__find_unique_shot_trs(desc, &unique_tr_indices, &tr_group_labels);
        }
        if (num_unique_trs <= 0)
            num_unique_trs = 1;

        for (u = 0; u < num_unique_trs; ++u)
        {
            memset(&uw, 0, sizeof(uw));
            if (profile_fn)
                profile_fn(profile_ctx, PULSEG__SAFETY_PROFILE_WAVEFORM_EXTRACT, 1);
            rc = pulseg__get_gradient_waveforms_range(
                desc,
                &uw,
                diag,
                start_block,
                block_count,
                PULSEG_AMP_MAX_POS,
                tr_group_labels,
                u,
                block_order);
            if (profile_fn)
                profile_fn(profile_ctx, PULSEG__SAFETY_PROFILE_WAVEFORM_EXTRACT, 0);
            if (PULSEG_FAILED(rc))
            {
                if (unique_tr_indices)
                    PULSEG_FREE(unique_tr_indices);
                if (tr_group_labels)
                    PULSEG_FREE(tr_group_labels);
                if (block_order)
                    PULSEG_FREE(block_order);
                return rc;
            }

            if (num_forbidden_bands > 0)
            {
                if (profile_fn)
                    profile_fn(profile_ctx, PULSEG__SAFETY_PROFILE_MECH_RESONANCE, 1);
                /* Headless A_eq analysis. The verdict comes only from the
                 * in-guarded-band TR-harmonic lines (candidate_*), which this
                 * call still computes in full at the band-derived resolution/
                 * max frequency (>= SA_MIN_ANALYSIS_FREQ_HZ). Covered by
                 * test_safety_grad.c Suite D.
                 * Both display flags are 0: the dense analytic envelope, the
                 * full-TR FFT, the analytical TR-harmonic grid and the
                 * per-event component export are plotting products (see
                 * pulseg_calc_mech_resonances / the python mechres_plots
                 * tooling), and nothing downstream of this call reads them —
                 * the loop below touches only spectra.candidate_*. This is
                 * the real PSD predownload path and must never pay for
                 * arrays it immediately frees. */
                memset(&spectra, 0, sizeof(spectra));
                rc = calc_mech_resonances_from_uniform(
                    &spectra,
                    diag,
                    &uw,
                    mr_target_res_hz,
                    mr_max_freq_hz,
                    num_instances,
                    tr_duration_us,
                    num_forbidden_bands,
                    forbidden_bands,
                    opts->peak_log10_threshold,
                    opts->peak_norm_scale,
                    opts->peak_eps,
                    opts->peak_prominence,
                    desc,
                    sa_start_block,
                    sa_block_count,
                    num_avgs,
                    opts->max_grad_hz_per_m,
                    0 /* compute_dense_envelope: never on the PSD path */,
                    0 /* compute_display_products: never on the PSD path */,
                    1 /* compress_trains: this is the path being optimised */);

                /* Safety path: fail fast on first violating candidate.
                 * Pattern: for each candidate, scan union of all bands. */
                if (!PULSEG_FAILED(rc) && spectra.num_candidates > 0 && spectra.candidate_freqs &&
                    spectra.candidate_grad_amps_gx && spectra.candidate_grad_amps_gy &&
                    spectra.candidate_grad_amps_gz)
                {
                    float *cga[3];
                    int axi;
                    float guard_hz, mbw;
                    /* Guard mirrors sa_check_structural_violations: HWHM of the
                     * narrowest positive-width band. */
                    mbw = -1.0f;
                    for (b = 0; b < num_forbidden_bands; ++b)
                    {
                        float w = forbidden_bands[b].freq_max_hz - forbidden_bands[b].freq_min_hz;
                        if (w > 0.0f && (mbw < 0.0f || w < mbw))
                            mbw = w;
                    }
                    guard_hz = (mbw > 0.0f) ? (SA_GUARD_HWHM_MULT * mbw * 0.5f) : 0.0f;
                    cga[0] = spectra.candidate_grad_amps_gx;
                    cga[1] = spectra.candidate_grad_amps_gy;
                    cga[2] = spectra.candidate_grad_amps_gz;
                    for (ci = 0; ci < spectra.num_candidates; ++ci)
                    {
                        cf_hz = spectra.candidate_freqs[ci];

                        for (b = 0; b < num_forbidden_bands; ++b)
                        {
                            float eps_band =
                                sa_eps_for_band(&forbidden_bands[b], opts->max_grad_hz_per_m);
                            if (cf_hz < forbidden_bands[b].freq_min_hz - guard_hz ||
                                cf_hz > forbidden_bands[b].freq_max_hz + guard_hz)
                                continue;
                            for (axi = 0; axi < 3; ++axi)
                            {
                                ca_hz_per_m = cga[axi][ci];
                                if (ca_hz_per_m > eps_band)
                                {
                                    rc = PULSEG_ERR_MECH_RESONANCES_VIOLATION;
                                    if (diag)
                                    {
                                        diag->code = rc;
                                        pulseg__diag_printf(
                                            diag,
                                            "f=%.2fHz,a=%.2f>%.2fHz/m,ax=%d,ss=%d,tr=%d",
                                            (double)cf_hz,
                                            (double)ca_hz_per_m,
                                            (double)eps_band,
                                            axi,
                                            s,
                                            u);
                                    }
                                    break;
                                }
                            }
                            if (PULSEG_FAILED(rc))
                                break;
                        }
                        if (PULSEG_FAILED(rc))
                            break;
                    }
                }

                pulseg_mech_resonances_spectra_free(&spectra);
                if (profile_fn)
                    profile_fn(profile_ctx, PULSEG__SAFETY_PROFILE_MECH_RESONANCE, 0);
                if (PULSEG_FAILED(rc))
                {
                    pulseg__uniform_grad_waveforms_free(&uw);
                    if (unique_tr_indices)
                        PULSEG_FREE(unique_tr_indices);
                    if (tr_group_labels)
                        PULSEG_FREE(tr_group_labels);
                    if (block_order)
                        PULSEG_FREE(block_order);
                    return rc;
                }
            }

            if (pns_model)
            {
                if (profile_fn)
                    profile_fn(profile_ctx, PULSEG__SAFETY_PROFILE_PNS, 1);
                memset(&pns_result, 0, sizeof(pns_result));
                rc = calc_pns_from_uniform(&pns_result, diag, opts->gamma_hz_per_t, &uw, pns_model);
                if (!PULSEG_FAILED(rc) && pns_result.num_samples > 0)
                {
                    max_pns = 0.0f;
                    for (i = 0; i < pns_result.num_samples; ++i)
                    {
                        pns_combined = 0.0f;
                        if (pns_result.slew_x_hz_per_m_per_s)
                            pns_combined += pns_result.slew_x_hz_per_m_per_s[i] *
                                pns_result.slew_x_hz_per_m_per_s[i];
                        if (pns_result.slew_y_hz_per_m_per_s)
                            pns_combined += pns_result.slew_y_hz_per_m_per_s[i] *
                                pns_result.slew_y_hz_per_m_per_s[i];
                        if (pns_result.slew_z_hz_per_m_per_s)
                            pns_combined += pns_result.slew_z_hz_per_m_per_s[i] *
                                pns_result.slew_z_hz_per_m_per_s[i];
                        pns_combined = (float)sqrt((double)pns_combined);
                        if (pns_combined > max_pns)
                            max_pns = pns_combined;
                    }
                    if (max_pns > pns_threshold_percent)
                        rc = PULSEG_ERR_PNS_THRESHOLD_EXCEEDED;
                }
                pulseg_pns_result_free(&pns_result);
                if (profile_fn)
                    profile_fn(profile_ctx, PULSEG__SAFETY_PROFILE_PNS, 0);
                if (PULSEG_FAILED(rc))
                {
                    pulseg__uniform_grad_waveforms_free(&uw);
                    if (unique_tr_indices)
                        PULSEG_FREE(unique_tr_indices);
                    if (tr_group_labels)
                        PULSEG_FREE(tr_group_labels);
                    if (block_order)
                        PULSEG_FREE(block_order);
                    return rc;
                }
            }

            pulseg__uniform_grad_waveforms_free(&uw);
        }

        if (unique_tr_indices)
            PULSEG_FREE(unique_tr_indices);
        if (tr_group_labels)
            PULSEG_FREE(tr_group_labels);
        if (block_order)
            PULSEG_FREE(block_order);
    }

    return PULSEG_SUCCESS;
}

int pulseg_check_safety(
    pulseg_collection *coll,
    pulseg_diagnostic *diag,
    const pulseg_opts *opts,
    int num_forbidden_bands,
    const pulseg_forbidden_band *forbidden_bands,
    const pulseg_pns_model *pns_model,
    float pns_threshold_percent)
{
    return pulseg__check_safety_profiled(
        coll,
        diag,
        opts,
        num_forbidden_bands,
        forbidden_bands,
        pns_model,
        pns_threshold_percent,
        NULL,
        NULL);
}
