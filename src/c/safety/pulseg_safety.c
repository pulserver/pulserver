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
#include "external_svd.h"

/* ================================================================== */
/*  Canonical-TR window selection                                     */
/* ================================================================== */

/* Helper: select the block window of one TR instance. */
static void select_canonical_tr_window_idx(
    const struct pulseg_sequence_descriptor *desc,
    int *start_block,
    int *block_count,
    int *num_instances,
    float *tr_duration_us,
    int canonical_tr_idx)
{
    const struct pulseg_tr_descriptor *trd = &desc->tr_descriptor;

    *start_block = canonical_tr_idx * trd->tr_size;
    *block_count = trd->tr_size;
    *num_instances = trd->num_trs;
    *tr_duration_us = trd->tr_duration_us;
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

/** Least A_eq any train the vendor forbids can produce, as a fraction of the
 *  plateau amplitude the band's amplitude column states.
 *
 *  A band amplitude is the plateau of an alternating readout train at that
 *  echo spacing, while A_eq is the equivalent-sinusoid amplitude of the whole
 *  waveform, so the two need one conversion to be comparable. Over the train
 *  shapes a system can play, A_eq/plateau runs from 8/pi^2 (fully triangular,
 *  ramp-limited) to 4/pi (square). Taking the smallest makes the threshold the
 *  quietest waveform the vendor forbade. */
#define SA_AEQ_TRAIN_SHAPE 0.8106f

/** Epsilon for a band whose amplitude column is zero, in mT/m of A_eq.
 *
 *  A zero column forbids the train at any amplitude, which no threshold can
 *  express, so this is a policy value rather than a vendor one and is
 *  calibrated empirically: across realistic protocols the in-band A_eq of the
 *  families GE steers (bSSFP at minimum TR, an EPI tooth landing in a band,
 *  fast 3D GRE) starts at 8.9 mT/m, while the families it runs unchecked
 *  (radial, spiral, stack-of-stars, long-TR GRE, FSE, spin echo) top out at
 *  6.1. This sits in that gap. It is below the converted vendor rows
 *  (13-24 mT/m), preserving the ordering that an absolute row is the stricter
 *  statement. */
#define SA_AEQ_POLICY_MT_PER_M 7.5f

/** Proton gyromagnetic ratio, for the plotting API when no opts are supplied. */
#define SA_GAMMA_1H_HZ_PER_T 42.576e6f

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

/** Sidelobe probes per side of a coarse TR harmonic, for the outer repeat
 *  (M = num_instances). A scan of M repetitions is not an infinite comb: its
 *  spectrum is the single-TR transform times the Dirichlet kernel
 *  D_M(x) = sin(M pi x) / (M sin(pi x)), whose local maxima between two
 *  harmonics sit at x = k + (j + 1/2)/M and whose peak levels are
 *  2 / (pi (2j + 1)) -- 0.637, 0.212, 0.127, 0.091, 0.071, ... These levels
 *  do not depend on M; only their spacing does, which is why a fixed count
 *  per side resolves the region for any M. Four probes cover every lobe
 *  above 9% of the main lobe; the residual envelope past that is under 7%,
 *  so reaching eps there would take a single-TR transform more than an order
 *  of magnitude larger than at the probed lobes.
 *
 *  Probe placement is load-bearing, not a resolution knob: sampling at
 *  integer multiples of 1/M instead lands on the kernel's exact NULLS, where
 *  D_M is zero and the probe cannot report anything. Each probe needs a
 *  fresh sa_eval_axis_spectrum call -- the single-TR transform oscillates in
 *  its own right and is not bounded by its value at a neighbouring harmonic,
 *  so scaling the coarse value by D_M is a no-op (D_M <= 1 can never expose
 *  a violation the coarse point did not already show). */
#define SA_MECHRES_SIDELOBES_PER_SIDE 4

/** Fewest distinct waveforms at one position before a rank basis is worth
 *  looking for. Below this the decomposition costs more than the transforms
 *  it would save. */
#define SA_SVD_MIN_WAVEFORMS 4

/** Largest rank kept, as a fraction of the waveform count: a basis that is
 *  not substantially smaller than the set it stands for buys nothing, so the
 *  attempt is abandoned rather than half-taken. */
#define SA_SVD_MAX_RANK_FRACTION 0.5

/** How much of each waveform's own L1 norm the discarded tail of the
 *  decomposition is allowed to reach. The tail is not dropped -- it is
 *  bounded and added back as a magnitude -- so this only sets how much
 *  conservatism the compression introduces, and at this level it is far
 *  below the last bit of the amplitudes being compared. */
#define SA_SVD_RESIDUAL_FRACTION 1.0e-6f

/** Work budget for one decomposition, in multiply-adds. A set that would
 *  cost more than this to decompose is evaluated waveform by waveform
 *  instead: the basis is an accelerator, and an accelerator that costs more
 *  than the thing it replaces is not one. */
#define SA_SVD_MAX_WORK 4.0e8

/** Interleaved Horner chains used to evaluate a compressed train's
 *  amplitude-weighted sum (sa_eval_event_spectrum). Power of two. A single
 *  chain runs at the latency of one complex multiply-add (~12 cycles per
 *  term) however much the core could otherwise retire; 8 lanes bring a
 *  1024-term train to under a cycle per term, and past that the returns are
 *  small while the fixed recombination cost keeps growing for trains too
 *  short to fill the lanes. */
#define SA_HORNER_LANES 8

/** Segments between exact sin/cos re-anchors of the uniform-raster phase
 *  recurrence in sa_eval_pwl_transform(). Bounds accumulated rotation drift
 *  to a fixed budget no matter how many samples an arbitrary waveform has,
 *  at a cost of 2 transcendentals per this many segments (< 1% of the
 *  general path's 4-per-segment). */
#define SA_PWL_REANCHOR 256

/** Interleaved copies of that recurrence in sa_eval_pwl_transform's uniform
 *  fast path. Power of two, and must divide SA_PWL_REANCHOR so the re-anchor
 *  budget stays one exact sin/cos pair per SA_PWL_REANCHOR segments. */
#define SA_PWL_LANES 8

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
    int w_key;                               /**< base-waveform identity, for the W(f) caches */
    double start_time_us;                    /**< event start time within the TR (us)   */
    float amplitude;                         /**< amplitude of this occurrence (Hz/m), signed */
    int pwl_num_vertices;                    /**< >0 -> use piecewise-linear W(f)       */
    float pwl_times_us[SA_MAX_PWL_VERTICES]; /**< vertex times (us from event start) */
    float pwl_values[SA_MAX_PWL_VERTICES];   /**< vertex amplitudes (normalised)     */
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
 * a pure function of (base waveform, frequency) -- keyed by w_key, for
 * the duration of ONE sa_eval_axis_spectrum() call (which is itself already
 * scoped to a single fixed frequency, so w_key alone is a sufficient key;
 * no separate frequency field needed). Multiple sa_event occurrences that
 * share a waveform (the common case: a handful of unique gradient shapes
 * reused across many materialized occurrences) hit this cache instead of
 * repeating the O(vertices) sa_eval_pwl_transform integral. NULL disables
 * caching (used at call sites outside the hot per-candidate-frequency
 * loop, where memoizing a single lookup isn't worth the bookkeeping).
 *
 * The key is the waveform, not the definition: a definition is deduplicated
 * on its timing and sample count, so a materialised multishot readout plays
 * many distinct shapes under one definition id, and keying on the definition
 * would hand every arm the first arm's transform.
 */
typedef struct
{
    int w_key;
    float re, im;
} sa_transform_cache_entry;

typedef struct
{
    sa_transform_cache_entry *entries;
    int count;
    int capacity;
} sa_transform_cache;

/** Linear scan: waveform counts per axis are small (a handful of unique
 *  gradient shapes even for a long/complex hyper-TR, per the design doc's
 *  own cost model), so a hash table would be overhead without benefit. */
static int sa_transform_cache_lookup(
    const sa_transform_cache *cache,
    float *out_re,
    float *out_im,
    int w_key)
{
    int i;
    if (!cache)
        return 0;
    for (i = 0; i < cache->count; ++i)
    {
        if (cache->entries[i].w_key == w_key)
        {
            *out_re = cache->entries[i].re;
            *out_im = cache->entries[i].im;
            return 1;
        }
    }
    return 0;
}

static void sa_transform_cache_insert(sa_transform_cache *cache, int w_key, float re, float im)
{
    if (!cache || cache->count >= cache->capacity)
        return; /* cache disabled or full: caller just recomputes next time */
    cache->entries[cache->count].w_key = w_key;
    cache->entries[cache->count].re = re;
    cache->entries[cache->count].im = im;
    cache->count++;
}

/**
 * A rank basis standing in for one axis's distinct waveforms at one varying
 * position.
 *
 * The waveforms at such a position are rarely independent: a multishot
 * readout written out shot by shot is one arm turned through a set of angles,
 * so its span is two-dimensional however many arms there are. Writing
 * g_k(t) = sum_r c_{k,r} v_r(t) and using the linearity of the transform,
 *
 *     G_k(f) = sum_r c_{k,r} V_r(f)
 *
 * turns num_events transforms per frequency into `rank` of them plus a
 * combination. The tail the truncation leaves out is not discarded: its
 * magnitude is bounded once, at build time, and added to every magnitude the
 * basis produces, so a compressed position can only ever read louder than an
 * uncompressed one.
 */
typedef struct sa_svd_basis_s sa_svd_basis;
static void sa_free_svd_basis(sa_svd_basis *b);

struct sa_svd_basis_s
{
    int rank;             /**< 0 = no basis; evaluate the waveforms directly */
    int num_events;       /**< waveforms the basis stands for                */
    sa_axis_events basis; /**< [rank] synthetic unit-amplitude events         */
    float *coeff;         /**< [num_events*rank] c_{k,r}                      */
    float *residual;      /**< [num_events] ceiling on the discarded tail     */
    float *basis_re;      /**< [rank] scratch, one frequency                  */
    float *basis_im;
};

/**
 * One block position whose gradients are not the same in every TR instance.
 *
 * The instances of a canonical TR share a block structure and a timing, but
 * a position may play a different amplitude (a phase encode), a different
 * definition (a materialised spiral arm) or a different rotation (an arm
 * turned by a rotation extension) in each of them. Such a position cannot
 * join the coherent sum, because there is no single complex contribution it
 * makes; it is bounded instead by the largest magnitude any instance can put
 * there, which is what makes the canonical TR's answer an upper bound for
 * the whole scan rather than one instance's answer.
 *
 * @c shapes holds one event per distinct waveform the position takes on each
 * axis, so the base transform W(f) is evaluated once per waveform however
 * many instances reuse it. @c tuple_* then holds the (waveform, amplitude,
 * rotation) combinations that really occur, and the bound is the largest
 * magnitude among them -- exact for the position, whatever the rotation
 * mixes. A position with more combinations than are worth enumerating falls
 * back to its largest amplitude per axis, combined through @c weight; for a
 * position playing one waveform per axis, which is what a phase encode is,
 * the two are the same number.
 */
typedef struct
{
    sa_axis_events shapes[3]; /**< distinct definitions, unit amplitude, per axis */
    int num_tuples;           /**< distinct instance combinations at this position */
    int *tuple_slot;          /**< [num_tuples*3] index into shapes[axis], -1 = none */
    float *tuple_amp;         /**< [num_tuples*3] the amplitude that instance plays  */
    int *tuple_rot;           /**< [num_tuples] rotation matrix index, -1 = none     */
    float weight[9];          /**< num_tuples == 0: |R| bounded over its rotations   */
    float *w_re[3];           /**< [shapes[ax].num_events] scratch, one frequency    */
    float *w_im[3];
    sa_svd_basis svd[3]; /**< rank basis per axis; rank 0 = evaluate directly   */
} sa_varying_position;

/** Structural analysis: event lists for all three axes. */
typedef struct
{
    sa_axis_events axes[3];       /**< 0=gx, 1=gy, 2=gz; summed coherently */
    sa_varying_position *varying; /**< [num_varying] positions summed as magnitudes */
    int num_varying;
} sa_structural_events;

static void sa_free_axis_events(sa_axis_events *ae)
{
    int k;
    if (!ae)
        return;
    if (ae->events)
    {
        for (k = 0; k < ae->num_events; ++k)
        {
            if (ae->events[k].arb_samples)
                PULSEG_FREE(ae->events[k].arb_samples);
            if (ae->events[k].arb_times_us)
                PULSEG_FREE(ae->events[k].arb_times_us);
            if (ae->events[k].train_amps)
                PULSEG_FREE(ae->events[k].train_amps);
        }
        PULSEG_FREE(ae->events);
    }
    ae->events = NULL;
    ae->num_events = 0;
}

static void sa_free_structural_events(sa_structural_events *se)
{
    int ax, v;
    if (!se)
        return;
    for (ax = 0; ax < 3; ++ax)
        sa_free_axis_events(&se->axes[ax]);
    for (v = 0; v < se->num_varying; ++v)
    {
        sa_varying_position *vp = &se->varying[v];
        for (ax = 0; ax < 3; ++ax)
        {
            sa_free_svd_basis(&vp->svd[ax]);
            sa_free_axis_events(&vp->shapes[ax]);
            if (vp->w_re[ax])
                PULSEG_FREE(vp->w_re[ax]);
            if (vp->w_im[ax])
                PULSEG_FREE(vp->w_im[ax]);
            vp->w_re[ax] = NULL;
            vp->w_im[ax] = NULL;
        }
        if (vp->tuple_slot)
            PULSEG_FREE(vp->tuple_slot);
        if (vp->tuple_amp)
            PULSEG_FREE(vp->tuple_amp);
        if (vp->tuple_rot)
            PULSEG_FREE(vp->tuple_rot);
    }
    if (se->varying)
        PULSEG_FREE(se->varying);
    se->varying = NULL;
    se->num_varying = 0;
}

/* ================================================================== */
/*  Structural acoustic analysis — occurrence extraction              */
/* ================================================================== */

/**
 * Extract grad-def occurrences for one physical axis within the canonical TR.
 * Collects (def_id, def_index, shape_id, start_time_us, amplitude) for each
 * block that drives the given axis, a rotated block through every logical
 * axis its rotation puts there.
 * @param axis  0=gx, 1=gy, 2=gz
 * Returns number of occurrences, or -1 on allocation failure.
 * Caller must free *out_events.
 */
typedef struct
{
    int def_id;
    int def_index;
    int shape_id; /**< pulseq shape this occurrence plays; 0 for a trapezoid */
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
        const float *rotation;
        int raw_ids[3];
        int source;
        double blk_dur;

        raw_ids[0] = bte->gx_id;
        raw_ids[1] = bte->gy_id;
        raw_ids[2] = bte->gz_id;

        /* A rotated block plays a mixture of its logical axes on this
         * physical one, so each of them contributes an occurrence here,
         * scaled by the matrix entry that lands it on this axis. */
        rotation = NULL;
        if (bte->rotation_id >= 0 && bte->rotation_id < desc->num_rotations && !bte->norot_flag)
            rotation = desc->rotation_matrices[bte->rotation_id];

        for (source = 0; source < 3; ++source)
        {
            const struct pulseg_grad_table_element *gte;
            float weight =
                rotation ? rotation[axis * 3 + source] : ((axis == source) ? 1.0f : 0.0f);

            if (weight == 0.0f || raw_ids[source] < 0 || raw_ids[source] >= desc->grad_table_size)
                continue;

            gte = &desc->grad_table[raw_ids[source]];
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
            occ[n].shape_id = gte->shape_id;
            occ[n].start_time_us = time_us;
            occ[n].amplitude = weight * gte->amplitude;
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
 * The base waveform an occurrence plays, as one integer.
 *
 * A trapezoid is described by its corner times, so its definition is its
 * waveform. An arbitrary gradient's waveform is the shape the occurrence
 * carries: definitions deduplicate on timing and sample count, so a
 * materialised multishot readout plays as many shapes as it has shots under
 * a single definition id.
 */
static int sa_waveform_key(
    const struct pulseg_sequence_descriptor *desc,
    const sa_raw_occurrence *occ)
{
    return pulseg__wave_key_flat(desc, occ->def_index, occ->shape_id);
}

/**
 * Build event list for one axis from raw occurrences.
 *
 * For each unique base waveform:
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
    int i, j, n_events, cap, wid, idx, s_idx, nv, occ_shape_id;
    int *unique_ids;
    int num_unique;
    float raster;

    ae->num_events = 0;
    ae->events = NULL;
    if (num_occ <= 0)
        return PULSEG_SUCCESS;

    raster = desc->grad_raster_us;

    /* Collect unique base waveforms */
    unique_ids = (int *)PULSEG_ALLOC((size_t)num_occ * sizeof(int));
    if (!unique_ids)
        return PULSEG_ERR_ALLOC_FAILED;
    for (i = 0; i < num_occ; ++i)
        unique_ids[i] = sa_waveform_key(desc, &occ[i]);
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

        wid = unique_ids[idx];
        gdef = NULL;
        occ_shape_id = 0;
        pwl_nv = 0;
        shared_arb_samples = NULL;
        shared_arb_times = NULL;
        shared_arb_n = 0;
        use_arb = 0;

        /* Find the first occurrence of this waveform to get its definition */
        for (j = 0; j < num_occ; ++j)
        {
            if (sa_waveform_key(desc, &occ[j]) == wid)
            {
                gdef = &desc->grad_definitions[occ[j].def_index];
                occ_shape_id = occ[j].shape_id;
                break;
            }
        }
        if (!gdef)
            continue;

        /* --- Compute the shared PWL parameters for this waveform --- */

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
            /* Arbitrary waveform: the shape these occurrences play. The
             * definition's own representative is the fallback for an
             * occurrence that carries no shape of its own. */
            int shape_id = (occ_shape_id > 0) ? occ_shape_id : gdef->spectral.shape_id;
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
            if (sa_waveform_key(desc, &occ[j]) != wid)
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
                ae->events[n_events].def_id = occ[j].def_id;
                ae->events[n_events].def_index = occ[j].def_index;
                ae->events[n_events].w_key = wid;
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
        /* Free the shared raw-sample template for this waveform */
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
 * Same base waveform, hence the same W_d(f).
 */
static int sa_train_compatible(const sa_event *a, const sa_event *b)
{
    return a->w_key == b->w_key && a->def_index == b->def_index && a->train_len == 1 &&
        b->train_len == 1;
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
static int sa_emit_train(sa_event *out, const sa_event *ev, int beg, int stride, int len)
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
                           fabs(
                               (ae->events[run_end].start_time_us -
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
/*  Structural acoustic analysis — the bound over TR instances        */
/* ================================================================== */

/** Distinct (waveform, amplitude, rotation) combinations one position may
 *  take before it is bounded axis by axis rather than combination by
 *  combination. A multishot readout has as many combinations as it has
 *  shots. A phase encode has as many as it has steps, and for it the two
 *  bounds are the same number, since one waveform's largest amplitude is
 *  its largest contribution. */
#define SA_MAX_POSITION_TUPLES 256

/** Distinct rotations one position may take before every matrix entry is
 *  bounded by 1, which holds for any orthonormal row. */
#define SA_MAX_POSITION_ROTATIONS 8

/** One base waveform a block position plays on one axis. */
typedef struct
{
    int def_index;
    int shape_id;
    float amp_max;   /**< largest |amplitude| any instance plays here */
    float amp_first; /**< the signed amplitude the first instance plays */
    int amp_varies;
} sa_position_waveform;

/** What one block position does across every instance of the canonical TR. */
typedef struct
{
    sa_position_waveform *waveforms[3];
    int num_waveforms[3];
    int cap_waveforms[3];

    int rot_ids[SA_MAX_POSITION_ROTATIONS]; /**< -1 = no rotation applied */
    int num_rot;
    int rot_overflow;

    int num_tuples;
    int cap_tuples;
    int tuple_overflow;
    int *tuple_slot;  /**< [cap_tuples*3] index into waveforms[axis], -1 = none */
    float *tuple_amp; /**< [cap_tuples*3] */
    int *tuple_rot;   /**< [cap_tuples] */
} sa_position_variants;

static void sa_free_position_variants(sa_position_variants *pv, int count)
{
    int i, ax;
    if (!pv)
        return;
    for (i = 0; i < count; ++i)
    {
        for (ax = 0; ax < 3; ++ax)
            if (pv[i].waveforms[ax])
                PULSEG_FREE(pv[i].waveforms[ax]);
        if (pv[i].tuple_slot)
            PULSEG_FREE(pv[i].tuple_slot);
        if (pv[i].tuple_amp)
            PULSEG_FREE(pv[i].tuple_amp);
        if (pv[i].tuple_rot)
            PULSEG_FREE(pv[i].tuple_rot);
    }
    PULSEG_FREE(pv);
}

/** Record one instance's waveform on one axis; returns its slot, or -1. */
static int sa_variants_add_waveform(
    sa_position_variants *pv,
    int axis,
    int def_index,
    int shape_id,
    float amplitude)
{
    int i;
    float mag = (amplitude < 0.0f) ? -amplitude : amplitude;

    for (i = 0; i < pv->num_waveforms[axis]; ++i)
    {
        sa_position_waveform *w = &pv->waveforms[axis][i];
        if (w->def_index == def_index && w->shape_id == shape_id)
        {
            if (mag > w->amp_max)
                w->amp_max = mag;
            if (amplitude != w->amp_first)
                w->amp_varies = 1;
            return i;
        }
    }

    if (pv->num_waveforms[axis] >= pv->cap_waveforms[axis])
    {
        int cap = (pv->cap_waveforms[axis] > 0) ? pv->cap_waveforms[axis] * 2 : 4;
        sa_position_waveform *grown =
            (sa_position_waveform *)PULSEG_ALLOC((size_t)cap * sizeof(sa_position_waveform));
        if (!grown)
            return -1;
        if (pv->waveforms[axis])
        {
            memcpy(
                grown,
                pv->waveforms[axis],
                (size_t)pv->num_waveforms[axis] * sizeof(sa_position_waveform));
            PULSEG_FREE(pv->waveforms[axis]);
        }
        pv->waveforms[axis] = grown;
        pv->cap_waveforms[axis] = cap;
    }

    i = pv->num_waveforms[axis]++;
    pv->waveforms[axis][i].def_index = def_index;
    pv->waveforms[axis][i].shape_id = shape_id;
    pv->waveforms[axis][i].amp_max = mag;
    pv->waveforms[axis][i].amp_first = amplitude;
    pv->waveforms[axis][i].amp_varies = 0;
    return i;
}

static void sa_variants_add_rotation(sa_position_variants *pv, int rot_id)
{
    int i;
    for (i = 0; i < pv->num_rot; ++i)
        if (pv->rot_ids[i] == rot_id)
            return;
    if (pv->num_rot >= SA_MAX_POSITION_ROTATIONS)
    {
        pv->rot_overflow = 1;
        return;
    }
    pv->rot_ids[pv->num_rot++] = rot_id;
}

/** Record one instance's (waveform, amplitude, rotation) combination.
 *  Returns 0 on allocation failure; the overflow past the cap is not one. */
static int sa_variants_add_tuple(
    sa_position_variants *pv,
    const int *slot,
    const float *amp,
    int rot_id)
{
    int i, at;

    if (pv->tuple_overflow)
        return 1;

    for (i = 0; i < pv->num_tuples; ++i)
    {
        if (pv->tuple_rot[i] != rot_id)
            continue;
        if (pv->tuple_slot[i * 3 + 0] == slot[0] && pv->tuple_slot[i * 3 + 1] == slot[1] &&
            pv->tuple_slot[i * 3 + 2] == slot[2] && pv->tuple_amp[i * 3 + 0] == amp[0] &&
            pv->tuple_amp[i * 3 + 1] == amp[1] && pv->tuple_amp[i * 3 + 2] == amp[2])
            return 1;
    }

    if (pv->num_tuples >= SA_MAX_POSITION_TUPLES)
    {
        pv->tuple_overflow = 1;
        return 1;
    }

    if (pv->num_tuples >= pv->cap_tuples)
    {
        int cap = (pv->cap_tuples > 0) ? pv->cap_tuples * 2 : 4;
        int *slots = (int *)PULSEG_ALLOC((size_t)cap * 3 * sizeof(int));
        float *amps = (float *)PULSEG_ALLOC((size_t)cap * 3 * sizeof(float));
        int *rots = (int *)PULSEG_ALLOC((size_t)cap * sizeof(int));
        if (!slots || !amps || !rots)
        {
            if (slots)
                PULSEG_FREE(slots);
            if (amps)
                PULSEG_FREE(amps);
            if (rots)
                PULSEG_FREE(rots);
            return 0;
        }
        if (pv->num_tuples > 0)
        {
            memcpy(slots, pv->tuple_slot, (size_t)pv->num_tuples * 3 * sizeof(int));
            memcpy(amps, pv->tuple_amp, (size_t)pv->num_tuples * 3 * sizeof(float));
            memcpy(rots, pv->tuple_rot, (size_t)pv->num_tuples * sizeof(int));
        }
        if (pv->tuple_slot)
            PULSEG_FREE(pv->tuple_slot);
        if (pv->tuple_amp)
            PULSEG_FREE(pv->tuple_amp);
        if (pv->tuple_rot)
            PULSEG_FREE(pv->tuple_rot);
        pv->tuple_slot = slots;
        pv->tuple_amp = amps;
        pv->tuple_rot = rots;
        pv->cap_tuples = cap;
    }

    at = pv->num_tuples++;
    for (i = 0; i < 3; ++i)
    {
        pv->tuple_slot[at * 3 + i] = slot[i];
        pv->tuple_amp[at * 3 + i] = amp[i];
    }
    pv->tuple_rot[at] = rot_id;
    return 1;
}

/**
 * What every instance of the canonical TR plays at each of its positions.
 *
 * One pass over the scan, the same walk pulseg__compute_variable_grad_flags
 * makes: the execution stream marks where each TR starts, so a block's
 * position within the TR is where it falls after that mark, and the
 * amplitudes it is compared against are the expanded scan's rather than the
 * deduplicated block table's.
 *
 * Block durations are not collected per instance because TR detection has
 * already established they cannot differ: a period is accepted only when
 * every position matches in duration (pulseg__block_defs_structurally_equal),
 * so one instance's timing is every instance's timing.
 */
static int sa_scan_position_variants(
    sa_position_variants *pv,
    const struct pulseg_sequence_descriptor *desc,
    int tr_size)
{
    int si, tr_pos, ax;
    int stream_len;

    stream_len =
        (desc->exec_stream_len > 0) ? desc->exec_stream_len : desc->tr_descriptor.num_trs * tr_size;
    tr_pos = 0;
    for (si = 0; si < stream_len; ++si)
    {
        const struct pulseg_block_table_element *bte;
        int bt_idx;
        int slot[3];
        float amp[3];
        int rot_id;
        int raw_ids[3];

        if (desc->exec_stream_len > 0)
        {
            if (pulseg__exec_tr_start(desc, si))
                tr_pos = 0;
            bt_idx = pulseg__exec_block_idx(desc, si);
        }
        else
        {
            if (si % tr_size == 0)
                tr_pos = 0;
            bt_idx = si;
        }

        if (tr_pos >= tr_size || bt_idx < 0 || bt_idx >= desc->num_blocks)
        {
            ++tr_pos;
            if (tr_pos >= tr_size)
                tr_pos = 0;
            continue;
        }

        bte = &desc->block_table[bt_idx];
        raw_ids[0] = bte->gx_id;
        raw_ids[1] = bte->gy_id;
        raw_ids[2] = bte->gz_id;

        for (ax = 0; ax < 3; ++ax)
        {
            slot[ax] = -1;
            amp[ax] = 0.0f;
            if (raw_ids[ax] >= 0 && raw_ids[ax] < desc->grad_table_size)
            {
                const struct pulseg_grad_table_element *gte = &desc->grad_table[raw_ids[ax]];
                slot[ax] = sa_variants_add_waveform(
                    &pv[tr_pos],
                    ax,
                    gte->id,
                    gte->shape_id,
                    gte->amplitude);
                if (slot[ax] < 0)
                    return PULSEG_ERR_ALLOC_FAILED;
                amp[ax] = gte->amplitude;
            }
        }

        rot_id = bte->rotation_id;
        if (rot_id < 0 || rot_id >= desc->num_rotations || bte->norot_flag)
            rot_id = -1;
        sa_variants_add_rotation(&pv[tr_pos], rot_id);

        if (!sa_variants_add_tuple(&pv[tr_pos], slot, amp, rot_id))
            return PULSEG_ERR_ALLOC_FAILED;

        ++tr_pos;
        if (tr_pos >= tr_size)
            tr_pos = 0;
    }
    return PULSEG_SUCCESS;
}

/** |R[ax][j]| bounded over every rotation the position takes, identity
 *  included for the instances that apply none. */
static void sa_variant_weights(
    float *weight,
    const sa_position_variants *pv,
    const struct pulseg_sequence_descriptor *desc)
{
    int i, ax, j;

    for (i = 0; i < 9; ++i)
        weight[i] = 0.0f;

    if (pv->rot_overflow)
    {
        for (i = 0; i < 9; ++i)
            weight[i] = 1.0f;
        return;
    }

    for (i = 0; i < pv->num_rot; ++i)
    {
        int rot = pv->rot_ids[i];
        if (rot < 0 || rot >= desc->num_rotations)
        {
            for (ax = 0; ax < 3; ++ax)
                if (weight[ax * 3 + ax] < 1.0f)
                    weight[ax * 3 + ax] = 1.0f;
            continue;
        }
        for (ax = 0; ax < 3; ++ax)
            for (j = 0; j < 3; ++j)
            {
                float w = desc->rotation_matrices[rot][ax * 3 + j];
                if (w < 0.0f)
                    w = -w;
                if (w > weight[ax * 3 + j])
                    weight[ax * 3 + j] = w;
            }
    }
}

/** The event index a position's waveform was built into, or -1. */
static int sa_shape_event_index(
    const sa_axis_events *ae,
    const struct pulseg_sequence_descriptor *desc,
    const sa_position_waveform *w)
{
    sa_raw_occurrence probe;
    int key, k;

    probe.def_id = w->def_index;
    probe.def_index = w->def_index;
    probe.shape_id = w->shape_id;
    probe.start_time_us = 0.0;
    probe.amplitude = 0.0f;
    key = sa_waveform_key(desc, &probe);

    for (k = 0; k < ae->num_events; ++k)
        if (ae->events[k].w_key == key)
            return k;
    return -1;
}

/**
 * Build the event model that bounds every instance of the canonical TR.
 *
 * A position every instance plays identically joins the coherent sum, with
 * its rotation folded into the amplitudes it contributes to each physical
 * axis, so the sum is exact wherever the TR really is one waveform. A
 * position that differs between instances becomes a sa_varying_position,
 * whose contribution is taken at its largest magnitude instead of its
 * complex value: the coherence between it and the rest of the TR is what is
 * given up, and it is given up only where there is no single value to be
 * coherent with.
 */
static int sa_build_bounded_events(
    sa_structural_events *se,
    const struct pulseg_sequence_descriptor *desc,
    int start_block,
    int block_count)
{
    sa_position_variants *pv = NULL;
    sa_raw_occurrence *coh[3];
    int num_coh[3];
    int cap_coh[3];
    double *pos_time = NULL;
    int p, ax, j, result;
    double time_us;

    memset(se, 0, sizeof(*se));
    for (ax = 0; ax < 3; ++ax)
    {
        coh[ax] = NULL;
        num_coh[ax] = 0;
        cap_coh[ax] = 0;
    }

    pv = (sa_position_variants *)PULSEG_ALLOC((size_t)block_count * sizeof(sa_position_variants));
    pos_time = (double *)PULSEG_ALLOC((size_t)block_count * sizeof(double));
    if (!pv || !pos_time)
    {
        if (pv)
            PULSEG_FREE(pv);
        if (pos_time)
            PULSEG_FREE(pos_time);
        return PULSEG_ERR_ALLOC_FAILED;
    }
    memset(pv, 0, (size_t)block_count * sizeof(sa_position_variants));

    time_us = 0.0;
    for (p = 0; p < block_count; ++p)
    {
        const struct pulseg_block_table_element *bte = &desc->block_table[start_block + p];
        const struct pulseg_base_block *bdef = &desc->base_blocks[bte->id];
        pos_time[p] = time_us;
        time_us += (bte->duration_us >= 0) ? (double)bte->duration_us : (double)bdef->duration_us;
    }

    result = sa_scan_position_variants(pv, desc, block_count);
    if (PULSEG_FAILED(result))
        goto fail;

    se->varying =
        (sa_varying_position *)PULSEG_ALLOC((size_t)block_count * sizeof(sa_varying_position));
    if (!se->varying)
    {
        result = PULSEG_ERR_ALLOC_FAILED;
        goto fail;
    }
    memset(se->varying, 0, (size_t)block_count * sizeof(sa_varying_position));

    for (p = 0; p < block_count; ++p)
    {
        sa_position_variants *v = &pv[p];
        int multi_waveform = 0;
        int rotated = 0;
        int constant;

        for (ax = 0; ax < 3; ++ax)
            if (v->num_waveforms[ax] > 1)
                multi_waveform = 1;
        if (v->rot_overflow || v->num_rot > 1 || (v->num_rot == 1 && v->rot_ids[0] >= 0))
            rotated = 1;

        constant = (!v->tuple_overflow && v->num_tuples == 1);

        if (v->num_tuples == 0 && !v->tuple_overflow)
            continue; /* no block ever landed here */

        if (constant)
        {
            const float *R = NULL;
            int rot = v->tuple_rot[0];
            if (rot >= 0 && rot < desc->num_rotations)
                R = desc->rotation_matrices[rot];

            for (ax = 0; ax < 3; ++ax)
            {
                for (j = 0; j < 3; ++j)
                {
                    const sa_position_waveform *w;
                    float weight = R ? R[ax * 3 + j] : ((ax == j) ? 1.0f : 0.0f);
                    int slot = v->tuple_slot[j];
                    if (slot < 0 || weight == 0.0f)
                        continue;
                    w = &v->waveforms[j][slot];

                    if (num_coh[ax] >= cap_coh[ax])
                    {
                        int cap = (cap_coh[ax] > 0) ? cap_coh[ax] * 2 : 16;
                        sa_raw_occurrence *grown = (sa_raw_occurrence *)PULSEG_ALLOC(
                            (size_t)cap * sizeof(sa_raw_occurrence));
                        if (!grown)
                        {
                            result = PULSEG_ERR_ALLOC_FAILED;
                            goto fail;
                        }
                        if (coh[ax])
                        {
                            memcpy(grown, coh[ax], (size_t)num_coh[ax] * sizeof(sa_raw_occurrence));
                            PULSEG_FREE(coh[ax]);
                        }
                        coh[ax] = grown;
                        cap_coh[ax] = cap;
                    }
                    coh[ax][num_coh[ax]].def_id = w->def_index;
                    coh[ax][num_coh[ax]].def_index = w->def_index;
                    coh[ax][num_coh[ax]].shape_id = w->shape_id;
                    coh[ax][num_coh[ax]].start_time_us = pos_time[p];
                    coh[ax][num_coh[ax]].amplitude = weight * v->tuple_amp[j];
                    num_coh[ax]++;
                }
            }
            continue;
        }

        /* Varying: one unit-amplitude event per waveform, and either the
         * combinations this position really takes or, when there are more of
         * them than are worth enumerating, its largest amplitude per axis --
         * which for a position playing one waveform per axis is the same
         * bound at a fraction of the cost. */
        {
            sa_varying_position *vp = &se->varying[se->num_varying];
            int by_tuple = (!v->tuple_overflow && (rotated || multi_waveform));

            /* Counted before it is filled, so a failure part way through
             * leaves it for sa_free_structural_events to release. */
            se->num_varying++;

            for (ax = 0; ax < 3; ++ax)
            {
                sa_raw_occurrence *shape_occ;
                int k;
                if (v->num_waveforms[ax] == 0)
                    continue;
                shape_occ = (sa_raw_occurrence *)PULSEG_ALLOC(
                    (size_t)v->num_waveforms[ax] * sizeof(sa_raw_occurrence));
                if (!shape_occ)
                {
                    result = PULSEG_ERR_ALLOC_FAILED;
                    goto fail;
                }
                for (k = 0; k < v->num_waveforms[ax]; ++k)
                {
                    shape_occ[k].def_id = v->waveforms[ax][k].def_index;
                    shape_occ[k].def_index = v->waveforms[ax][k].def_index;
                    shape_occ[k].shape_id = v->waveforms[ax][k].shape_id;
                    shape_occ[k].start_time_us = pos_time[p];
                    shape_occ[k].amplitude = by_tuple ? 1.0f : v->waveforms[ax][k].amp_max;
                }
                result =
                    sa_build_axis_events(&vp->shapes[ax], shape_occ, v->num_waveforms[ax], desc);
                PULSEG_FREE(shape_occ);
                if (PULSEG_FAILED(result))
                    goto fail;

                if (vp->shapes[ax].num_events > 0)
                {
                    vp->w_re[ax] =
                        (float *)PULSEG_ALLOC((size_t)vp->shapes[ax].num_events * sizeof(float));
                    vp->w_im[ax] =
                        (float *)PULSEG_ALLOC((size_t)vp->shapes[ax].num_events * sizeof(float));
                    if (!vp->w_re[ax] || !vp->w_im[ax])
                    {
                        result = PULSEG_ERR_ALLOC_FAILED;
                        goto fail;
                    }
                }
            }

            if (by_tuple)
            {
                int t;
                vp->tuple_slot = (int *)PULSEG_ALLOC((size_t)v->num_tuples * 3 * sizeof(int));
                vp->tuple_amp = (float *)PULSEG_ALLOC((size_t)v->num_tuples * 3 * sizeof(float));
                vp->tuple_rot = (int *)PULSEG_ALLOC((size_t)v->num_tuples * sizeof(int));
                if (!vp->tuple_slot || !vp->tuple_amp || !vp->tuple_rot)
                {
                    result = PULSEG_ERR_ALLOC_FAILED;
                    goto fail;
                }
                for (t = 0; t < v->num_tuples; ++t)
                {
                    vp->tuple_rot[t] = v->tuple_rot[t];
                    for (ax = 0; ax < 3; ++ax)
                    {
                        int slot = v->tuple_slot[t * 3 + ax];
                        vp->tuple_amp[t * 3 + ax] = v->tuple_amp[t * 3 + ax];
                        vp->tuple_slot[t * 3 + ax] = (slot < 0)
                            ? -1
                            : sa_shape_event_index(&vp->shapes[ax], desc, &v->waveforms[ax][slot]);
                    }
                }
                vp->num_tuples = v->num_tuples;
            }
            else
            {
                sa_variant_weights(vp->weight, v, desc);
                vp->num_tuples = 0;
            }
        }
    }

    for (ax = 0; ax < 3; ++ax)
    {
        result = sa_build_axis_events(&se->axes[ax], coh[ax], num_coh[ax], desc);
        if (PULSEG_FAILED(result))
            goto fail;
    }

    for (ax = 0; ax < 3; ++ax)
        if (coh[ax])
            PULSEG_FREE(coh[ax]);
    PULSEG_FREE(pos_time);
    sa_free_position_variants(pv, block_count);
    return PULSEG_SUCCESS;

fail:
    for (ax = 0; ax < 3; ++ax)
        if (coh[ax])
            PULSEG_FREE(coh[ax]);
    if (pos_time)
        PULSEG_FREE(pos_time);
    if (pv)
        sa_free_position_variants(pv, block_count);
    sa_free_structural_events(se);
    return result;
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
            /* SA_PWL_LANES interleaved copies of the recurrence, lane m taking
             * segments m, m+LANES, m+2*LANES, ... Both the phase rotation and
             * the accumulator are serial dependency chains of complex
             * multiply-adds, so one copy runs at their latency (~5 cycles a
             * segment) whatever the core could otherwise retire.
             *
             * This loop is the gate's cost centre whenever a definition has
             * many vertices: it is O(n_vtx) and runs once per definition per
             * candidate frequency, so total gate cost tracks the vertex count
             * of the widest definitions and not the length of the scan. A
             * definition of a few thousand samples therefore costs more here
             * than a scan of millions of blocks made of trapezoids.
             * Cf. SA_HORNER_LANES, the same fix on the other term. */
            double lane_gr[SA_PWL_LANES], lane_gi[SA_PWL_LANES];
            double lane_cos[SA_PWL_LANES], lane_sin[SA_PWL_LANES];
            double wKT = wT * (double)SA_PWL_LANES;
            double cos_wKT = cos(wKT);
            double sin_wKT = sin(wKT);
            int nseg = n_vtx - 1;
            int since_anchor = 0;
            int base, m;

            for (m = 0; m < SA_PWL_LANES; ++m)
            {
                double tm = omega * (t0 + (double)m * dt0);
                lane_cos[m] = cos(tm);
                lane_sin[m] = sin(tm);
                lane_gr[m] = 0.0;
                lane_gi[m] = 0.0;
            }

            for (base = 0; base + SA_PWL_LANES <= nseg; base += SA_PWL_LANES)
            {
                for (m = 0; m < SA_PWL_LANES; ++m)
                {
                    int kk = base + m;
                    double ak = (double)v[kk];
                    double bk = ((double)v[kk + 1] - ak) * inv_dt;
                    double x_re = ak * c0_re + bk * c1_re;
                    double x_im = ak * c0_im + bk * c1_im;

                    /* multiply by e^{-j omega tk} */
                    lane_gr[m] += x_re * lane_cos[m] + x_im * lane_sin[m];
                    lane_gi[m] += x_im * lane_cos[m] - x_re * lane_sin[m];
                }

                /* advance every lane one stride, to t + LANES*dt0. Re-anchored
                 * on the same segment budget as the single chain was, so the
                 * drift bound is unchanged: LANES lanes re-anchoring every
                 * SA_PWL_REANCHOR/LANES strides is one exact sin/cos per
                 * SA_PWL_REANCHOR segments either way. */
                if (++since_anchor >= SA_PWL_REANCHOR / SA_PWL_LANES)
                {
                    for (m = 0; m < SA_PWL_LANES; ++m)
                    {
                        double tt = omega * (t0 + (double)(base + SA_PWL_LANES + m) * dt0);
                        lane_cos[m] = cos(tt);
                        lane_sin[m] = sin(tt);
                    }
                    since_anchor = 0;
                }
                else
                {
                    for (m = 0; m < SA_PWL_LANES; ++m)
                    {
                        double nc = lane_cos[m] * cos_wKT - lane_sin[m] * sin_wKT;
                        lane_sin[m] = lane_sin[m] * cos_wKT + lane_cos[m] * sin_wKT;
                        lane_cos[m] = nc;
                    }
                }
            }

            /* Tail: fewer than LANES segments left, and lane m still holds the
             * phase of segment base+m, so they are taken in lane order. */
            for (k = base; k < nseg; ++k)
            {
                m = k - base;
                {
                    double ak = (double)v[k];
                    double bk = ((double)v[k + 1] - ak) * inv_dt;
                    double x_re = ak * c0_re + bk * c1_re;
                    double x_im = ak * c0_im + bk * c1_im;
                    lane_gr[m] += x_re * lane_cos[m] + x_im * lane_sin[m];
                    lane_gi[m] += x_im * lane_cos[m] - x_re * lane_sin[m];
                }
            }

            for (m = 0; m < SA_PWL_LANES; ++m)
            {
                g_re += lane_gr[m];
                g_im += lane_gi[m];
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

/* ================================================================== */
/*  Tabulated base-waveform transform (chirp-z)                       */
/* ================================================================== */
/*
 * sa_eval_pwl_transform() costs O(vertices) and is called once per gradient
 * definition per evaluated frequency, so a definition of a few thousand
 * samples dominates the whole gate -- and it does so no matter how short the
 * scan is, because neither factor involves the block count.
 *
 * The frequencies it is asked for are not arbitrary. Within one forbidden
 * band the candidates are consecutive TR harmonics, and each is probed at
 * the same fixed set of offsets, so every evaluation sits at
 * (k + offset) / T_TR with k running over an integer range. On that comb the
 * uniform-raster transform reduces to one polynomial evaluated at points
 * spaced evenly in angle:
 *
 *   W(w) = e^{-i w t0} [ A(w) (P - v[n-1] q^{n-1}) + B(w) q^{-1} (P - v[0]) ]
 *
 * with q = e^{-i w dt}, P(q) = sum_k v[k] q^k, and A, B the same segment
 * integrals the direct loop forms (A = c0 - c1/dt, B = c1/dt). P is the only
 * term that costs anything, and a chirp-z transform produces it at every
 * candidate in the band at once -- O((n+m) log(n+m)) against the O(n*m) of
 * asking one frequency at a time.
 *
 * This is a change of summation order, not of model: the same integral of
 * the same sample interpolant, to the same double precision. It is worth
 * doing only where the vertex count is large (SA_CZT_MIN_VERTICES); a
 * trapezoid has four, and for those the direct loop is cheaper than the
 * table, so they never take this path and their results are untouched.
 */

/** Below this many vertices the direct integral is cheaper than tabulating
 *  it, and the tabulation is skipped. A trapezoid is four vertices, so no
 *  sequence built from them ever reaches this path. */
#define SA_CZT_MIN_VERTICES 64

/** One base waveform's transform, tabulated over (offset, candidate). */
typedef struct
{
    int w_key;
    double *re; /* [num_offsets * num_points] */
    double *im;
} sa_w_series;

/** The tabulated transforms for one axis over one forbidden band. */
typedef struct
{
    int num_series;
    sa_w_series *series;
    int num_offsets;
    int num_points;
} sa_w_table;

/** Where in a table to look: which offset from the harmonic, and which
 *  candidate. Bundled rather than passed as three arguments because the
 *  chain that carries it down to the transform is the hot path of the whole
 *  gate, and three more registers on it cost more than the lookup saves on
 *  a sequence that tabulates nothing. NULL means "no table". */
typedef struct
{
    const sa_w_table *table;
    int offset_slot;
    int point;
} sa_w_query;

static void sa_w_table_free(sa_w_table *table)
{
    int i;
    if (!table || !table->series)
        return;
    for (i = 0; i < table->num_series; ++i)
    {
        if (table->series[i].re)
            PULSEG_FREE(table->series[i].re);
        if (table->series[i].im)
            PULSEG_FREE(table->series[i].im);
    }
    PULSEG_FREE(table->series);
    table->series = NULL;
    table->num_series = 0;
}

/** The tabulated value, or 0 if this waveform is not in the table. */
static int sa_w_table_lookup(const sa_w_query *q, int w_key, float *out_re, float *out_im)
{
    const sa_w_table *table = q->table;
    int i;

    if (q->offset_slot < 0 || q->point < 0)
        return 0;
    if (q->offset_slot >= table->num_offsets || q->point >= table->num_points)
        return 0;
    for (i = 0; i < table->num_series; ++i)
    {
        if (table->series[i].w_key == w_key)
        {
            int at = q->offset_slot * table->num_points + q->point;
            *out_re = (float)table->series[i].re[at];
            *out_im = (float)table->series[i].im[at];
            return 1;
        }
    }
    return 0;
}

/** The vertices an event's transform is taken over, or NULL if it has none
 *  on a uniform raster this can exploit. Mirrors the detection in
 *  sa_eval_pwl_transform() exactly, including its tolerance. */
static const float *sa_event_uniform_vertices(
    const sa_event *ev,
    const float **out_times,
    int *out_n,
    double *out_dt)
{
    const float *t_us;
    const float *v;
    int n, k;
    double dt0, tol;

    if (ev->arb_num_samples >= 2)
    {
        t_us = ev->arb_times_us;
        v = ev->arb_samples;
        n = ev->arb_num_samples;
    }
    else if (ev->pwl_num_vertices >= 2)
    {
        t_us = ev->pwl_times_us;
        v = ev->pwl_values;
        n = ev->pwl_num_vertices;
    }
    else
    {
        return NULL;
    }

    if (n < SA_CZT_MIN_VERTICES)
        return NULL;

    dt0 = (double)(t_us[1] - t_us[0]);
    if (dt0 <= 1.0e-12)
        return NULL;
    tol = 1.0e-4 * fabs(dt0) + 1.0e-6 * fabs((double)t_us[n - 1]);
    for (k = 1; k < n - 1; ++k)
    {
        double d = (double)(t_us[k + 1] - t_us[k]);
        if (fabs(d - dt0) > tol)
            return NULL;
    }

    *out_times = t_us;
    *out_n = n;
    *out_dt = dt0;
    return v;
}

/** P(q) -> W(w), the rest of the segment integral, for one frequency. */
static void sa_w_from_poly(
    double *out_re,
    double *out_im,
    double p_re,
    double p_im,
    double omega,
    double dt0,
    double t0,
    double v_first,
    double v_last,
    int n)
{
    double wT = omega * dt0;
    double cos_wT = cos(wT);
    double sin_wT = sin(wT);
    double c0_re = sin_wT / omega;
    double c0_im = (cos_wT - 1.0) / omega;
    double I0mT_re = c0_re - dt0 * cos_wT;
    double I0mT_im = c0_im + dt0 * sin_wT;
    double c1_re = I0mT_im / omega;
    double c1_im = -I0mT_re / omega;
    double inv_dt = 1.0 / dt0;
    double a_re = c0_re - c1_re * inv_dt;
    double a_im = c0_im - c1_im * inv_dt;
    double b_re = c1_re * inv_dt;
    double b_im = c1_im * inv_dt;
    /* q = e^{-i w dt}; the sums run to n-2, so the first and last samples
     * are corrected off the full polynomial. */
    double theta = -omega * dt0;
    double qn_re = cos(theta * (double)(n - 1));
    double qn_im = sin(theta * (double)(n - 1));
    double qi_re = cos(-theta);
    double qi_im = sin(-theta);
    double s0_re = p_re - v_last * qn_re;
    double s0_im = p_im - v_last * qn_im;
    double t_re = p_re - v_first;
    double t_im = p_im;
    double s1_re = t_re * qi_re - t_im * qi_im;
    double s1_im = t_re * qi_im + t_im * qi_re;
    double g_re = a_re * s0_re - a_im * s0_im + b_re * s1_re - b_im * s1_im;
    double g_im = a_re * s0_im + a_im * s0_re + b_re * s1_im + b_im * s1_re;
    double ph = -omega * t0;
    double cp = cos(ph);
    double sp = sin(ph);

    *out_re = g_re * cp - g_im * sp;
    *out_im = g_re * sp + g_im * cp;
}

/** Total offsets a candidate is probed at: the exact harmonic, plus the
 *  sidelobe probes on either side of it. */
#define SA_MAX_OFFSETS (1 + 2 * SA_MECHRES_SIDELOBES_PER_SIDE)

/** Sidelobe probes that fit on one side of a harmonic for this M. The lobes
 *  are at (j + 1/2)/M and only those strictly inside the half-interval
 *  exist, so small M is probed exhaustively and large M is capped. */
static int sa_sidelobes_per_side(int num_instances)
{
    int j;
    if (num_instances <= 1)
        return 0;
    for (j = 0; j < SA_MECHRES_SIDELOBES_PER_SIDE; ++j)
    {
        if (((double)j + 0.5) / (double)num_instances >= 0.5)
            break;
    }
    return j;
}

/** The offsets from a candidate harmonic that the loop below evaluates, in
 *  the order it evaluates them -- slot 0 the harmonic itself, then
 *  side 0 outward, then side 1. Kept here so the table is built over exactly
 *  the frequencies that will be asked for; the arithmetic mirrors the
 *  sub-point loop and must move with it. */
static void sa_build_offsets(double *offsets, int *num_offsets, int num_instances)
{
    int npts_per_side = sa_sidelobes_per_side(num_instances);
    int side, j, at;

    offsets[0] = 0.0;
    at = 1;
    for (side = 0; side < 2; ++side)
    {
        for (j = 0; j < npts_per_side; ++j)
        {
            double delta = ((double)j + 0.5) / (double)num_instances;
            offsets[at++] = (side == 0) ? delta : (1.0 - delta);
        }
    }
    *num_offsets = at;
}

/** Tabulate every wide waveform on one axis over one band of candidates.
 *
 * Failure is not an error: the table is an accelerator, so a waveform that
 * cannot be tabulated (ragged raster, too few vertices, no memory) is simply
 * left out and its transform is taken the direct way. */
static int sa_build_w_table(
    sa_w_table *table,
    const sa_axis_events *ae,
    int klo,
    int num_points,
    const double *offsets,
    int num_offsets,
    double f1_hz)
{
    int k, i, o, j;
    int capacity;

    memset(table, 0, sizeof(*table));
    if (!ae || ae->num_events == 0 || num_points < 1)
        return PULSEG_SUCCESS;

    capacity = ae->num_events;
    table->series = (sa_w_series *)PULSEG_ALLOC((size_t)capacity * sizeof(sa_w_series));
    if (!table->series)
        return PULSEG_SUCCESS;
    memset(table->series, 0, (size_t)capacity * sizeof(sa_w_series));
    table->num_offsets = num_offsets;
    table->num_points = num_points;

    for (k = 0; k < ae->num_events; ++k)
    {
        const sa_event *ev = &ae->events[k];
        const float *t_us = NULL;
        const float *v;
        int n = 0;
        double dt0 = 0.0;
        double dtheta;
        pulseg__czt_plan *plan = NULL;
        double *pr = NULL, *pi = NULL;
        int already = 0;

        for (i = 0; i < table->num_series; ++i)
            if (table->series[i].w_key == ev->w_key)
                already = 1;
        if (already)
            continue;

        v = sa_event_uniform_vertices(ev, &t_us, &n, &dt0);
        if (!v)
            continue;

        /* q advances by this much between adjacent candidates, whatever the
         * offset -- which is why one plan serves every offset. */
        dtheta = -2.0 * M_PI * f1_hz * 1.0e-6 * dt0;
        if (PULSEG_FAILED(pulseg__czt_plan_create(&plan, n, num_points, dtheta)))
            continue;

        pr = (double *)PULSEG_ALLOC((size_t)num_points * sizeof(double));
        pi = (double *)PULSEG_ALLOC((size_t)num_points * sizeof(double));
        table->series[table->num_series].re =
            (double *)PULSEG_ALLOC((size_t)num_offsets * num_points * sizeof(double));
        table->series[table->num_series].im =
            (double *)PULSEG_ALLOC((size_t)num_offsets * num_points * sizeof(double));
        if (!pr || !pi || !table->series[table->num_series].re ||
            !table->series[table->num_series].im)
        {
            if (pr)
                PULSEG_FREE(pr);
            if (pi)
                PULSEG_FREE(pi);
            if (table->series[table->num_series].re)
                PULSEG_FREE(table->series[table->num_series].re);
            if (table->series[table->num_series].im)
                PULSEG_FREE(table->series[table->num_series].im);
            table->series[table->num_series].re = NULL;
            table->series[table->num_series].im = NULL;
            pulseg__czt_plan_free(plan);
            continue;
        }

        for (o = 0; o < num_offsets; ++o)
        {
            double theta0 = -2.0 * M_PI * ((double)klo + offsets[o]) * f1_hz * 1.0e-6 * dt0;
            pulseg__czt_plan_apply(plan, pr, pi, v, theta0);
            for (j = 0; j < num_points; ++j)
            {
                double f_hz = ((double)(klo + j) + offsets[o]) * f1_hz;
                double omega = 2.0 * M_PI * f_hz * 1.0e-6;
                double w_re, w_im;
                sa_w_from_poly(
                    &w_re,
                    &w_im,
                    pr[j],
                    pi[j],
                    omega,
                    dt0,
                    (double)t_us[0],
                    (double)v[0],
                    (double)v[n - 1],
                    n);
                table->series[table->num_series].re[o * num_points + j] = w_re;
                table->series[table->num_series].im[o * num_points + j] = w_im;
            }
        }

        table->series[table->num_series].w_key = ev->w_key;
        table->num_series++;

        PULSEG_FREE(pr);
        PULSEG_FREE(pi);
        pulseg__czt_plan_free(plan);
    }

    return PULSEG_SUCCESS;
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
    sa_transform_cache *cache,
    const sa_w_query *query)
{
    /* The table, when there is one, already holds this definition's
     * transform at this exact frequency -- see the note above it. It is
     * consulted before the per-frequency memo because it is the cheaper of
     * the two and covers whole bands rather than a single frequency.
     *
     * The null test is deliberately here rather than inside the lookup: a
     * sequence whose definitions are all narrow tabulates nothing, and then
     * this must cost one predictable branch on a path taken hundreds of
     * thousands of times, not a call. */
    if (query && sa_w_table_lookup(query, ev->w_key, out_re, out_im))
        return;

    if (sa_transform_cache_lookup(cache, out_re, out_im, ev->w_key))
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

    sa_transform_cache_insert(cache, ev->w_key, *out_re, *out_im);
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
    sa_transform_cache *cache,
    const sa_w_query *query)
{
    float tr_re, tr_im;
    double sum_re, sum_im, base_re, base_im;
    double phase, cos_ph, sin_ph;

    sa_eval_event_transform(ev, &tr_re, &tr_im, f_hz, cache, query);

    /* sum_{j<L} A_j z^j -- the amplitude-weighted train sum, still relative
     * to the train's own start time. */
    if (ev->train_len > 1)
    {
        double phi = -2.0 * M_PI * (double)f_hz * ev->train_period_us * 1.0e-6;
        if (ev->train_amps)
        {
            /* Horner over the train, split into SA_HORNER_LANES independent
             * chains: P(z) = sum_m z^m * P_m(z^L), P_m collecting the terms
             * with j = m (mod LANES). One chain is a serial dependency of
             * complex multiply-adds, so a plain Horner runs at the latency of
             * that chain (~4 cycles/term) no matter how much arithmetic the
             * core could retire; interleaved chains fill it. A train fused by
             * sa_compress_axis_events() out of a long run of equally spaced
             * occurrences is one event of as many terms as the run was long,
             * re-evaluated at every candidate frequency, so this is where the
             * gate's cost sits when the amplitudes vary per occurrence; lanes
             * buy back most of the stall.
             *
             * Reassociation only: the terms and their coefficients are
             * unchanged, and the sum is carried in double throughout, so this
             * moves results by last-bit rounding at most. */
            double zr = cos(phi);
            double zi = sin(phi);
            double lane_re[SA_HORNER_LANES], lane_im[SA_HORNER_LANES];
            double zlr, zli, npr;
            const float *amps = ev->train_amps;
            int L = ev->train_len;
            int m, q, nq, base;

            /* z^LANES, by repeated squaring (LANES is a power of two). */
            zlr = zr;
            zli = zi;
            for (m = 1; m < SA_HORNER_LANES; m <<= 1)
            {
                double t = zlr * zlr - zli * zli;
                zli = 2.0 * zlr * zli;
                zlr = t;
            }

            /* Leading group. The polynomial is padded up to a whole number of
             * groups with zero coefficients, which a Horner chain started at
             * zero absorbs exactly -- so the loop below needs no bounds test. */
            nq = (L + SA_HORNER_LANES - 1) / SA_HORNER_LANES;
            base = (nq - 1) * SA_HORNER_LANES;
            for (m = 0; m < SA_HORNER_LANES; ++m)
            {
                lane_re[m] = (base + m < L) ? (double)amps[base + m] : 0.0;
                lane_im[m] = 0.0;
            }

            /* All lanes advance together: the LANES chains are independent, so
             * their multiply-adds interleave in the pipeline instead of each
             * waiting on the one before it. Running the lanes one after another
             * would be the same instruction count and exactly as slow as the
             * single chain it replaced. */
            for (q = nq - 2; q >= 0; --q)
            {
                base = q * SA_HORNER_LANES;
                for (m = 0; m < SA_HORNER_LANES; ++m)
                {
                    double nr = lane_re[m] * zlr - lane_im[m] * zli + (double)amps[base + m];
                    lane_im[m] = lane_re[m] * zli + lane_im[m] * zlr;
                    lane_re[m] = nr;
                }
            }

            /* recombine: sum_m z^m * P_m, Horner again over the LANES lanes. */
            sum_re = lane_re[SA_HORNER_LANES - 1];
            sum_im = lane_im[SA_HORNER_LANES - 1];
            for (m = SA_HORNER_LANES - 2; m >= 0; --m)
            {
                npr = sum_re * zr - sum_im * zi + lane_re[m];
                sum_im = sum_re * zi + sum_im * zr + lane_im[m];
                sum_re = npr;
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
 * Complex spectral line of one event at frequency f.
 *
 * A compressed intra-TR train has already had its own sum applied by
 * sa_eval_event_spectrum(); the repetition of the canonical TR itself is
 * the Dirichlet kernel of @c num_instances, applied by the caller against
 * the TR period rather than per event.
 */
static void sa_eval_event_line(
    const sa_event *ev,
    float *out_re,
    float *out_im,
    float f_hz,
    sa_transform_cache *cache,
    const sa_w_query *query)
{
    sa_eval_event_spectrum(ev, out_re, out_im, f_hz, cache, query);
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
    float f_hz,
    const sa_w_query *query)
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
        sa_eval_event_line(&ae->events[k], &d_re, &d_im, f_hz, &cache, query);
        sum_re += (double)d_re;
        sum_im += (double)d_im;
    }

    if (cache_entries)
        PULSEG_FREE(cache_entries);

    *out_re = (float)sum_re;
    *out_im = (float)sum_im;
}

/**
 * What the positions that differ between instances can add, per axis, at one
 * frequency.
 *
 * Each such position contributes the largest magnitude any instance can put
 * there, so |S_ax(f)| <= |coherent sum| + this. The maximum is taken over
 * the combinations the position really plays, which is exact for the
 * position however its rotation mixes the axes; a position that fell back to
 * per-axis amplitudes combines them through its |R| weights instead.
 */
/** L1 norm of one event's base waveform: integral |w(t)| dt in us, over the
 *  normalised shape. Trapezoid rule on the PWL vertices, rectangles on raw
 *  arb samples -- the same two waveform models sa_eval_event_transform()
 *  evaluates, so the norm belongs to exactly the function being bounded.
 *  A segment whose endpoints straddle zero is split at the crossing, which
 *  is what keeps this the integral of |w| rather than |integral of w|. */
static double sa_event_base_l1_us(const sa_event *ev)
{
    double total = 0.0;
    int k;

    if (ev->arb_num_samples > 0 && ev->arb_samples && ev->arb_times_us)
    {
        for (k = 0; k < ev->arb_num_samples; ++k)
        {
            double dt;
            if (ev->arb_num_samples == 1)
                dt = 0.0;
            else if (k == 0)
                dt = (double)(ev->arb_times_us[1] - ev->arb_times_us[0]);
            else
                dt = (double)(ev->arb_times_us[k] - ev->arb_times_us[k - 1]);
            total += fabs((double)ev->arb_samples[k]) * dt;
        }
        return total;
    }

    for (k = 0; k + 1 < ev->pwl_num_vertices; ++k)
    {
        double v0 = (double)ev->pwl_values[k];
        double v1 = (double)ev->pwl_values[k + 1];
        double dt = (double)(ev->pwl_times_us[k + 1] - ev->pwl_times_us[k]);
        if (dt <= 0.0)
            continue;
        if ((v0 < 0.0 && v1 > 0.0) || (v0 > 0.0 && v1 < 0.0))
        {
            double denom = fabs(v0) + fabs(v1);
            double t0 = dt * (fabs(v0) / denom);
            total += 0.5 * fabs(v0) * t0 + 0.5 * fabs(v1) * (dt - t0);
        }
        else
        {
            total += 0.5 * (fabs(v0) + fabs(v1)) * dt;
        }
    }
    return total;
}

/** Frequency-independent ceiling on |line_k(f)| for one event, in Hz/m*s.
 *
 * |integral a w(t) e^{-i omega t} dt| <= |a| integral |w| dt, and the train
 * and repeat sums are bounded term by term, so this dominates the event's
 * contribution at EVERY frequency -- there is no band in which it can be
 * exceeded. */
static double sa_event_l1(const sa_event *ev)
{
    double amp_sum = 0.0;
    double base;
    int j;

    base = sa_event_base_l1_us(ev);
    if (base <= 0.0)
        return 0.0;

    if (ev->train_len > 1 && ev->train_amps)
    {
        for (j = 0; j < ev->train_len; ++j)
            amp_sum += fabs((double)ev->train_amps[j]);
    }
    else if (ev->train_len > 1)
    {
        amp_sum = (double)ev->train_len * fabs((double)ev->amplitude);
    }
    else
    {
        amp_sum = fabs((double)ev->amplitude);
    }

    return amp_sum * base * 1.0e-6;
}

/** Release a rank basis. */
static void sa_free_svd_basis(sa_svd_basis *b)
{
    if (!b)
        return;
    sa_free_axis_events(&b->basis);
    if (b->coeff)
        PULSEG_FREE(b->coeff);
    if (b->residual)
        PULSEG_FREE(b->residual);
    if (b->basis_re)
        PULSEG_FREE(b->basis_re);
    if (b->basis_im)
        PULSEG_FREE(b->basis_im);
    memset(b, 0, sizeof(*b));
}

/** The waveform one event carries, whichever of the two models holds it. */
static int sa_svd_waveform(const sa_event *ev, const float **times, const float **values)
{
    if (ev->arb_num_samples > 0)
    {
        if (!ev->arb_samples || !ev->arb_times_us)
            return 0;
        *times = ev->arb_times_us;
        *values = ev->arb_samples;
        return ev->arb_num_samples;
    }
    if (ev->pwl_num_vertices > 0)
    {
        *times = ev->pwl_times_us;
        *values = ev->pwl_values;
        return ev->pwl_num_vertices;
    }
    return 0;
}

/** Can this set be decomposed at all?
 *
 * The decomposition treats the waveforms as rows of one matrix, so they must
 * share a sampling: one model, the same number of points at the same times,
 * played at the same position under the same repetition. That is exactly what
 * a multishot readout gives -- its shots differ in their samples and in
 * nothing else -- and anything that does not is left to the direct path. */
static int sa_svd_set_is_uniform(const sa_axis_events *ae, int *out_n, int *out_is_arb)
{
    const float *t0, *v0, *tk, *vk;
    int k, i, n, n0, is_arb;

    if (!ae || ae->num_events < SA_SVD_MIN_WAVEFORMS)
        return 0;
    n0 = sa_svd_waveform(&ae->events[0], &t0, &v0);
    if (n0 < 2)
        return 0;
    is_arb = ae->events[0].arb_num_samples > 0;
    for (k = 0; k < ae->num_events; ++k)
    {
        const sa_event *ev = &ae->events[k];
        n = sa_svd_waveform(ev, &tk, &vk);
        if (n != n0 || (ev->arb_num_samples > 0) != is_arb)
            return 0;
        if (ev->train_len > 1)
            return 0;
        if (ev->start_time_us != ae->events[0].start_time_us)
            return 0;
        for (i = 0; i < n; ++i)
        {
            if (tk[i] != t0[i])
                return 0;
        }
    }
    *out_n = n0;
    *out_is_arb = is_arb;
    return 1;
}

/** Build a rank basis for one axis's waveforms at one varying position.
 *
 * Never fails the caller: a set that cannot be decomposed, or whose rank is
 * not small enough to pay for itself, simply leaves rank 0 behind and is
 * evaluated the direct way. */
static void sa_try_build_svd(sa_svd_basis *out, const sa_axis_events *ae)
{
    int m, n, r_full, k, i, r, rank, max_rank, is_arb;
    const float *t0 = NULL, *v0 = NULL, *tk = NULL, *vk = NULL;
    float *a = NULL, *u = NULL, *sv = NULL, *v = NULL;
    double dt_max, l1_scale, tail_scale;
    double *tail = NULL;

    memset(out, 0, sizeof(*out));
    if (!sa_svd_set_is_uniform(ae, &n, &is_arb))
        return;

    m = ae->num_events;
    r_full = (m < n) ? m : n;
    max_rank = (int)((double)m * SA_SVD_MAX_RANK_FRACTION);
    if (max_rank < 1)
        return;
    if ((double)m * (double)n * (double)r_full > SA_SVD_MAX_WORK)
        return;

    a = (float *)PULSEG_ALLOC((size_t)m * (size_t)n * sizeof(float));
    u = (float *)PULSEG_ALLOC((size_t)m * (size_t)r_full * sizeof(float));
    sv = (float *)PULSEG_ALLOC((size_t)r_full * sizeof(float));
    v = (float *)PULSEG_ALLOC((size_t)n * (size_t)r_full * sizeof(float));
    tail = (double *)PULSEG_ALLOC((size_t)m * sizeof(double));
    if (!a || !u || !sv || !v || !tail)
        goto done;

    (void)sa_svd_waveform(&ae->events[0], &t0, &v0);
    for (k = 0; k < m; ++k)
    {
        double amp = (double)ae->events[k].amplitude;
        (void)sa_svd_waveform(&ae->events[k], &tk, &vk);
        for (i = 0; i < n; ++i)
            a[k * n + i] = (float)(amp * (double)vk[i]);
    }

    if (svd_decompose(a, (size_t)m, (size_t)n, u, sv, v) != SVD_OK)
        goto done;

    /* |V_r(f)| <= integral |v_r| dt <= dt_max * sqrt(n) for a unit-norm row,
     * which is what turns a discarded singular value into a bound on the
     * transform it would have contributed. */
    dt_max = 0.0;
    for (i = 1; i < n; ++i)
    {
        double d = (double)(t0[i] - t0[i - 1]);
        if (d > dt_max)
            dt_max = d;
    }
    if (dt_max <= 0.0)
        goto done;
    tail_scale = dt_max * sqrt((double)n);

    /* The smallest rank whose discarded tail stays under the tolerance for
     * every waveform in the set. */
    l1_scale = 0.0;
    for (k = 0; k < m; ++k)
    {
        double l1 = sa_event_base_l1_us(&ae->events[k]) * fabs((double)ae->events[k].amplitude);
        if (l1 > l1_scale)
            l1_scale = l1;
    }
    if (l1_scale <= 0.0)
        goto done;

    rank = 0;
    for (r = 1; r <= max_rank; ++r)
    {
        double worst = 0.0;
        for (k = 0; k < m; ++k)
        {
            double t = 0.0;
            for (i = r; i < r_full; ++i)
                t += (double)sv[i] * fabs((double)u[k * r_full + i]);
            t *= tail_scale;
            tail[k] = t;
            if (t > worst)
                worst = t;
        }
        if (worst <= (double)SA_SVD_RESIDUAL_FRACTION * l1_scale)
        {
            rank = r;
            break;
        }
    }
    if (rank < 1)
        goto done;

    out->basis.events = (sa_event *)PULSEG_ALLOC((size_t)rank * sizeof(sa_event));
    out->coeff = (float *)PULSEG_ALLOC((size_t)m * (size_t)rank * sizeof(float));
    out->residual = (float *)PULSEG_ALLOC((size_t)m * sizeof(float));
    out->basis_re = (float *)PULSEG_ALLOC((size_t)rank * sizeof(float));
    out->basis_im = (float *)PULSEG_ALLOC((size_t)rank * sizeof(float));
    if (!out->basis.events || !out->coeff || !out->residual || !out->basis_re || !out->basis_im)
    {
        sa_free_svd_basis(out);
        goto done;
    }
    memset(out->basis.events, 0, (size_t)rank * sizeof(sa_event));
    out->basis.num_events = rank;

    for (r = 0; r < rank; ++r)
    {
        sa_event *be = &out->basis.events[r];
        be->def_id = -1 - r;
        be->def_index = -1;
        be->w_key = -1;
        be->start_time_us = ae->events[0].start_time_us;
        be->amplitude = 1.0f;
        be->train_len = 1;
        if (is_arb)
        {
            be->arb_num_samples = n;
            be->arb_samples = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
            be->arb_times_us = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
            if (!be->arb_samples || !be->arb_times_us)
            {
                sa_free_svd_basis(out);
                goto done;
            }
            for (i = 0; i < n; ++i)
            {
                be->arb_samples[i] = v[i * r_full + r];
                be->arb_times_us[i] = t0[i];
            }
        }
        else
        {
            be->pwl_num_vertices = n;
            for (i = 0; i < n; ++i)
            {
                be->pwl_values[i] = v[i * r_full + r];
                be->pwl_times_us[i] = t0[i];
            }
        }
    }

    for (k = 0; k < m; ++k)
    {
        for (r = 0; r < rank; ++r)
            out->coeff[k * rank + r] = (float)((double)sv[r] * (double)u[k * r_full + r]);
        out->residual[k] = (float)(tail[k] * 1.0e-6);
    }
    out->rank = rank;
    out->num_events = m;

done:
    if (a)
        PULSEG_FREE(a);
    if (u)
        PULSEG_FREE(u);
    if (sv)
        PULSEG_FREE(sv);
    if (v)
        PULSEG_FREE(v);
    if (tail)
        PULSEG_FREE(tail);
}

/** Per-axis frequency-independent ceiling on |S_ax(f)| plus the
 * varying-position bound, in Hz/m*s. Mirrors sa_eval_axis_spectrum() and
 * sa_eval_varying_bound() term for term with |W(f)| replaced by the L1 norm
 * that dominates it everywhere. Computed once per canonical TR; the
 * per-shape scratch it borrows is rewritten by every sa_eval_varying_bound()
 * call, which is what that scratch is for. */
static void sa_axis_l1_sup(
    sa_structural_events *se,
    const struct pulseg_sequence_descriptor *desc,
    double *out_sup)
{
    int v, ax, j, t, k;

    for (ax = 0; ax < 3; ++ax)
    {
        double total = 0.0;
        for (k = 0; k < se->axes[ax].num_events; ++k)
            total += sa_event_l1(&se->axes[ax].events[k]);
        out_sup[ax] = total;
    }

    for (v = 0; v < se->num_varying; ++v)
    {
        sa_varying_position *vp = &se->varying[v];
        double best[3];

        best[0] = 0.0;
        best[1] = 0.0;
        best[2] = 0.0;

        for (j = 0; j < 3; ++j)
        {
            for (k = 0; k < vp->shapes[j].num_events; ++k)
            {
                double l1 = sa_event_l1(&vp->shapes[j].events[k]);
                if (vp->svd[j].rank > 0)
                    l1 += (double)vp->svd[j].residual[k];
                vp->w_re[j][k] = (float)l1;
            }
        }

        if (vp->num_tuples > 0)
        {
            for (t = 0; t < vp->num_tuples; ++t)
            {
                const float *R = NULL;
                int rot = vp->tuple_rot[t];
                if (rot >= 0 && rot < desc->num_rotations)
                    R = desc->rotation_matrices[rot];

                for (ax = 0; ax < 3; ++ax)
                {
                    double mag = 0.0;
                    for (j = 0; j < 3; ++j)
                    {
                        int slot = vp->tuple_slot[t * 3 + j];
                        double w = R ? (double)R[ax * 3 + j] : ((ax == j) ? 1.0 : 0.0);
                        if (slot < 0 || w == 0.0)
                            continue;
                        mag += fabs(w) * fabs((double)vp->tuple_amp[t * 3 + j]) *
                            (double)vp->w_re[j][slot];
                    }
                    if (mag > best[ax])
                        best[ax] = mag;
                }
            }
        }
        else
        {
            double m[3];
            for (j = 0; j < 3; ++j)
            {
                double largest = 0.0;
                for (k = 0; k < vp->shapes[j].num_events; ++k)
                {
                    if ((double)vp->w_re[j][k] > largest)
                        largest = (double)vp->w_re[j][k];
                }
                m[j] = largest;
            }
            for (ax = 0; ax < 3; ++ax)
                best[ax] = (double)vp->weight[ax * 3 + 0] * m[0] +
                    (double)vp->weight[ax * 3 + 1] * m[1] + (double)vp->weight[ax * 3 + 2] * m[2];
        }

        for (ax = 0; ax < 3; ++ax)
            out_sup[ax] += best[ax];
    }
}

static void sa_eval_varying_bound(
    sa_structural_events *se,
    const struct pulseg_sequence_descriptor *desc,
    float f_hz,
    double *out_bound)
{
    int v, ax, j, t, k;

    out_bound[0] = 0.0;
    out_bound[1] = 0.0;
    out_bound[2] = 0.0;

    for (v = 0; v < se->num_varying; ++v)
    {
        sa_varying_position *vp = &se->varying[v];
        double best[3];

        best[0] = 0.0;
        best[1] = 0.0;
        best[2] = 0.0;

        for (j = 0; j < 3; ++j)
        {
            sa_svd_basis *sb = &vp->svd[j];
            if (sb->rank > 0)
            {
                /* rank transforms, then one combination each, in place of a
                 * transform per waveform. */
                int r;
                for (r = 0; r < sb->rank; ++r)
                    sa_eval_event_line(
                        &sb->basis.events[r],
                        &sb->basis_re[r],
                        &sb->basis_im[r],
                        f_hz,
                        NULL,
                        NULL);
                for (k = 0; k < vp->shapes[j].num_events; ++k)
                {
                    double re = 0.0, im = 0.0;
                    for (r = 0; r < sb->rank; ++r)
                    {
                        double c = (double)sb->coeff[k * sb->rank + r];
                        re += c * (double)sb->basis_re[r];
                        im += c * (double)sb->basis_im[r];
                    }
                    vp->w_re[j][k] = (float)re;
                    vp->w_im[j][k] = (float)im;
                }
                continue;
            }
            for (k = 0; k < vp->shapes[j].num_events; ++k)
            {
                float re, im;
                sa_eval_event_line(&vp->shapes[j].events[k], &re, &im, f_hz, NULL, NULL);
                vp->w_re[j][k] = re;
                vp->w_im[j][k] = im;
            }
        }

        if (vp->num_tuples > 0)
        {
            for (t = 0; t < vp->num_tuples; ++t)
            {
                const float *R = NULL;
                int rot = vp->tuple_rot[t];
                if (rot >= 0 && rot < desc->num_rotations)
                    R = desc->rotation_matrices[rot];

                for (ax = 0; ax < 3; ++ax)
                {
                    double sum_re = 0.0;
                    double sum_im = 0.0;
                    double tail = 0.0;
                    double mag;
                    for (j = 0; j < 3; ++j)
                    {
                        int slot = vp->tuple_slot[t * 3 + j];
                        double w = R ? (double)R[ax * 3 + j] : ((ax == j) ? 1.0 : 0.0);
                        double a;
                        if (slot < 0 || w == 0.0)
                            continue;
                        a = w * (double)vp->tuple_amp[t * 3 + j];
                        sum_re += a * (double)vp->w_re[j][slot];
                        sum_im += a * (double)vp->w_im[j][slot];
                        if (vp->svd[j].rank > 0)
                            tail += fabs(a) * (double)vp->svd[j].residual[slot];
                    }
                    mag = sqrt(sum_re * sum_re + sum_im * sum_im) + tail;
                    if (mag > best[ax])
                        best[ax] = mag;
                }
            }
        }
        else
        {
            double m[3];
            for (j = 0; j < 3; ++j)
            {
                double largest = 0.0;
                for (k = 0; k < vp->shapes[j].num_events; ++k)
                {
                    double mag = sqrt(
                        (double)vp->w_re[j][k] * (double)vp->w_re[j][k] +
                        (double)vp->w_im[j][k] * (double)vp->w_im[j][k]);
                    if (vp->svd[j].rank > 0)
                        mag += (double)vp->svd[j].residual[k];
                    if (mag > largest)
                        largest = mag;
                }
                m[j] = largest;
            }
            for (ax = 0; ax < 3; ++ax)
                best[ax] = (double)vp->weight[ax * 3 + 0] * m[0] +
                    (double)vp->weight[ax * 3 + 1] * m[1] + (double)vp->weight[ax * 3 + 2] * m[2];
        }

        for (ax = 0; ax < 3; ++ax)
            out_bound[ax] += best[ax];
    }
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
 *      coherent sum of the instances).
 *   2. (display) A_eq at every TR harmonic k/T_TR up to freq_max, for plots.
 *   3. (verdict) For each forbidden band, enumerate the TR-harmonic lines that
 *      fall inside the guarded range [f_min-guard, f_max+guard], evaluate
 *      A_eq(f_L) = (2/T_TR)|S_ax(f_L)| per axis, and flag the band iff the
 *      max-axis A_eq exceeds eps = max(band limit, k*G_max).
 * The outermost TR is treated as an infinite-rep (Dirac) comb: only its
 * harmonics carry sustained drive, hence lines live exactly at k/T_TR.
 */
static float sa_eps_for_band(const pulseg_forbidden_band *band, float gamma_hz_per_t)
{
    if (band->max_amplitude_hz_per_m > 0.0f)
        return SA_AEQ_TRAIN_SHAPE * band->max_amplitude_hz_per_m;
    return SA_AEQ_POLICY_MT_PER_M * 1.0e-3f * gamma_hz_per_t;
}

/**
 * Finite-outer-rep Dirichlet ratio (docs/explanations/mechanical_resonance_safety.md
 * §1): |D_M(x)| / M, where D_M(x) = sin(M*pi*x) / sin(pi*x) is the
 * Dirichlet kernel for M coherent repeats of period T_TR, and
 * x = f * T_TR (dimensionless: integer x = exact TR harmonics, fractional
 * x = the sidelobes between them that only exist for finite M).
 *
 * Exactly 1.0 at integer x (the main lobes, so an exact TR harmonic reduces
 * to the infinite-comb formula) and < 1.0 elsewhere.
 *
 * The caller must evaluate S_TR(f) FRESH at the fractional x. S_TR
 * oscillates in f in its own right and is neither bounded by nor safely
 * approximated from its value at a neighbouring exact harmonic; scaling that
 * neighbour by this ratio instead can expose nothing, since a factor of at
 * most 1 never exceeds the value it scales. This supplies the attenuation
 * only. Cost is held independent of M by capping how many sidelobes are
 * probed per side (SA_MECHRES_SIDELOBES_PER_SIDE), never by skipping the
 * fresh evaluation.
 */
static double sa_dirichlet_ratio(double x, int M)
{
    double s, num;
    if (M <= 1)
        return 1.0;
    s = sin(M_PI * x);
    /* Removable singularity at integer x (s -> 0): the true limit is 1.0
     * (D_M(x)->M), not 0/0. The nearest offset this is ever called at is
     * half a lobe away, 0.5/M, so this only triggers at genuine integers. */
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
    float gamma_hz_per_t,
    int compute_dense_envelope,
    int compute_display_products,
    int compress_trains,
    int bound_over_instances)
{
    sa_structural_events se;
    int result, ax, i, b, k, ci;
    double T_s, f1_hz;
    float freq_max, guard, min_bw;
    int m_max;
    double bound[3];
    double l1_sup[3];
    double l1_sup_max;

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

    /* Tabulated base-waveform transforms, rebuilt per band. Empty unless a
     * definition is wide enough to be worth it. */
    sa_w_table w_tables[3];
    sa_w_query coarse_query;
    sa_w_query sub_query;
    double offsets[SA_MAX_OFFSETS];
    int num_offsets = 1;

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
        memset(&w_tables[ax], 0, sizeof(w_tables[ax]));
        ana_amps[ax] = NULL;
        ana_phases[ax] = NULL;
        env_amps[ax] = NULL;
        cand_amps[ax] = NULL;
        cand_grad_amps_ax[ax] = NULL;
    }

    bound[0] = 0.0;
    bound[1] = 0.0;
    bound[2] = 0.0;

    /* The bound is over the instances of one canonical TR, so it is offered
     * only for a window that is one: any other window is a caller asking
     * about a specific stretch of blocks, and gets exactly those. */
    if (bound_over_instances && block_count == desc->tr_descriptor.tr_size)
        result = sa_build_bounded_events(&se, desc, start_block, block_count);
    else
        result = sa_build_structural_events(&se, desc, start_block, block_count);
    if (PULSEG_FAILED(result))
        return result;

    /* --- Compress equally-spaced occurrence trains ---
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

    /* --- Rank basis for the varying positions ---
     * Runs after the tagging and compression above, so a basis event inherits
     * the repetition its set was tagged with. A position whose waveforms do
     * not share a sampling, or whose span is not appreciably smaller than
     * their number, keeps rank 0 and is evaluated waveform by waveform. */
    {
        int v;
        for (v = 0; v < se.num_varying; ++v)
        {
            for (ax = 0; ax < 3; ++ax)
                sa_try_build_svd(&se.varying[v].svd[ax], &se.varying[v].shapes[ax]);
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
    /* Frequency-independent ceiling on this TR's drive, from the L1 norm
     * of its own waveforms: |S_ax(f)| <= integral |g_ax| dt at every f.
     * Where the ceiling already sits under a band's epsilon, no line in
     * that band can violate, and the probes that would have looked are
     * skipped rather than evaluated -- a proven no, not a sampled one. */
    sa_axis_l1_sup(&se, desc, l1_sup);
    l1_sup_max = l1_sup[0] > l1_sup[1] ? l1_sup[0] : l1_sup[1];
    if (l1_sup[2] > l1_sup_max)
        l1_sup_max = l1_sup[2];

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
            sa_eval_varying_bound(&se, desc, f_hz, bound);
            for (ax = 0; ax < 3; ++ax)
            {
                float sre, sim;
                if (se.axes[ax].num_events == 0 && bound[ax] == 0.0)
                {
                    ana_amps[ax][i] = 0.0f;
                    ana_phases[ax][i] = 0.0f;
                    continue;
                }
                sa_eval_axis_spectrum(&se.axes[ax], &sre, &sim, f_hz, NULL);
                ana_amps[ax][i] =
                    (float)(2.0 / T_s * (sqrt((double)(sre * sre + sim * sim)) + bound[ax]));
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
            sa_eval_varying_bound(&se, desc, f_hz, bound);
            for (ax = 0; ax < 3; ++ax)
            {
                float sre, sim;
                if (se.axes[ax].num_events == 0 && bound[ax] == 0.0)
                {
                    env_amps[ax][i] = 0.0f;
                    continue;
                }
                sa_eval_axis_spectrum(&se.axes[ax], &sre, &sim, f_hz, NULL);
                env_amps[ax][i] =
                    (float)(2.0 / T_s * (sqrt((double)(sre * sre + sim * sim)) + bound[ax]));
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
            float eps = sa_eps_for_band(&forbidden_bands[b], gamma_hz_per_t);
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

            /* Tabulate the wide definitions over this band before walking it.
             * Nothing below changes behaviour: an axis with no such
             * definition gets an empty table and every lookup misses, which
             * is the direct path unchanged. */
            sa_build_offsets(offsets, &num_offsets, num_instances);
            for (ax = 0; ax < 3; ++ax)
            {
                sa_w_table_free(&w_tables[ax]);
                if (khi >= klo)
                    (void)sa_build_w_table(
                        &w_tables[ax],
                        &se.axes[ax],
                        klo,
                        khi - klo + 1,
                        offsets,
                        num_offsets,
                        f1_hz);
            }

            for (kk = klo; kk <= khi; ++kk)
            {
                float f_hz = (float)((double)kk * f1_hz);
                int point = kk - klo;
                float max_ga = 0.0f;
                cand_freqs[ci] = f_hz;
                surviving_freqs_hz[ci] = f_hz;
                sa_eval_varying_bound(&se, desc, f_hz, bound);
                for (ax = 0; ax < 3; ++ax)
                {
                    float sre, sim, aeq;
                    if (se.axes[ax].num_events == 0 && bound[ax] == 0.0)
                    {
                        cand_amps[ax][ci] = 0.0f;
                        cand_grad_amps_ax[ax][ci] = 0.0f;
                        continue;
                    }
                    coarse_query.table = &w_tables[ax];
                    coarse_query.offset_slot = 0;
                    coarse_query.point = point;
                    sa_eval_axis_spectrum(
                        &se.axes[ax],
                        &sre,
                        &sim,
                        f_hz,
                        w_tables[ax].num_series ? &coarse_query : NULL);
                    aeq = (float)(2.0 / T_s * (sqrt((double)(sre * sre + sim * sim)) + bound[ax]));
                    cand_amps[ax][ci] = aeq;
                    cand_grad_amps_ax[ax][ci] = aeq;
                    if (aeq > max_ga)
                        max_ga = aeq;

                    for (k = 0; k < se.axes[ax].num_events; ++k)
                    {
                        float lre, lim;
                        if (num_component_terms >= max_component_terms)
                            break;
                        sa_eval_event_line(&se.axes[ax].events[k], &lre, &lim, f_hz, NULL, NULL);
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

                /* ---- Finite outer repeat ----
                 * The scan is M = num_instances repetitions of the TR, not an
                 * infinite comb, so real drive exists BETWEEN the coarse TR
                 * harmonics: the single-TR transform is multiplied by the
                 * Dirichlet kernel D_M, whose sidelobes peak at
                 * (kk + (j + 1/2)/M) with levels 2/(pi(2j+1)). Each is probed
                 * with a fresh single-TR transform and only then attenuated by
                 * D_M -- see SA_MECHRES_SIDELOBES_PER_SIDE for why neither the
                 * placement nor the fresh evaluation can be economised away.
                 * M = 1 has no sidelobes and takes no probes, leaving the
                 * exact-harmonic verdict above untouched. */
                {
                    int npts_per_side = sa_sidelobes_per_side(num_instances);
                    int side, j;
                    for (side = 0; side < 2; ++side)
                    {
                        for (j = 0; j < npts_per_side; ++j)
                        {
                            double delta = ((double)j + 0.5) / (double)num_instances;
                            double x_sub;
                            float f_sub_hz;
                            double ratio;
                            x_sub = (side == 0) ? ((double)kk + delta) : ((double)(kk + 1) - delta);
                            f_sub_hz = (float)(x_sub * f1_hz);
                            ratio = sa_dirichlet_ratio(x_sub, num_instances);
                            /* The lobe attenuates by `ratio`, so the ceiling
                             * attenuates with it: under epsilon there is
                             * nothing here to find. The plotting API keeps
                             * evaluating every probe, so the drawn lines keep
                             * their exact amplitudes. */
                            if (!compute_display_products &&
                                ratio * 2.0 / T_s * l1_sup_max <= (double)eps)
                                continue;
                            sa_eval_varying_bound(&se, desc, f_sub_hz, bound);
                            for (ax = 0; ax < 3; ++ax)
                            {
                                float sre, sim, aeq_sub;
                                if (se.axes[ax].num_events == 0 && bound[ax] == 0.0)
                                    continue;
                                sub_query.table = &w_tables[ax];
                                sub_query.offset_slot = 1 + side * npts_per_side + j;
                                sub_query.point = point;
                                sa_eval_axis_spectrum(
                                    &se.axes[ax],
                                    &sre,
                                    &sim,
                                    f_sub_hz,
                                    w_tables[ax].num_series ? &sub_query : NULL);
                                aeq_sub =
                                    (float)(2.0 / T_s * (sqrt((double)(sre * sre + sim * sim)) + bound[ax]) * ratio);
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

    for (ax = 0; ax < 3; ++ax)
        sa_w_table_free(&w_tables[ax]);
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
    for (ax = 0; ax < 3; ++ax)
        sa_w_table_free(&w_tables[ax]);
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
    float gamma_hz_per_t,
    int compute_dense_envelope,
    int compute_display_products,
    int compress_trains,
    int bound_over_instances)
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
            gamma_hz_per_t,
            compute_dense_envelope,
            compute_display_products,
            compress_trains,
            bound_over_instances);
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

/* Select the canonical TR window for the safety gate.
 * Non-degenerate prep/cooldown: full-pass canonical TR (pass-expanded).
 * Degenerate prep/cooldown: imaging TR canonical window (no pass expansion). */
static void select_canonical_tr_window(
    const pulseg_sequence_descriptor *desc,
    int *start_block,
    int *block_count,
    int *num_instances,
    float *tr_duration_us)
{
    const pulseg_tr_descriptor *trd;

    trd = &desc->tr_descriptor;

    *start_block = 0;
    *block_count = trd->tr_size;
    *num_instances = trd->num_trs;
    *tr_duration_us = trd->tr_duration_us;
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
    int amplitude_mode,
    const pulseg_opts *opts,
    float target_resolution_hz,
    float max_freq_hz,
    int num_forbidden_bands,
    const pulseg_forbidden_band *forbidden_bands,
    int compress_trains)
{
    const pulseg_sequence_descriptor *desc;
    pulseg__uniform_grad_waveforms uw;
    pulseg_diagnostic local_diag;
    int rc, start_block, block_count, num_instances, bound_over_instances;
    int sa_start_block, sa_block_count;
    int *block_order;
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
    /* Select the canonical TR window for the given canonical_tr_idx */
    if (canonical_tr_idx < 0 || canonical_tr_idx >= desc->tr_descriptor.num_trs)
    {
        diag->code = PULSEG_ERR_INVALID_ARGUMENT;
        return diag->code;
    }
    bound_over_instances = (amplitude_mode != PULSEG_AMP_ACTUAL);
    select_canonical_tr_window_idx(
        desc,
        &start_block,
        &block_count,
        &num_instances,
        &tr_duration_us,
        bound_over_instances ? 0 : canonical_tr_idx);

    sa_start_block = start_block;
    sa_block_count = block_count;

    rc = pulseg__get_gradient_waveforms_range(
        desc,
        &uw,
        diag,
        start_block,
        block_count,
        bound_over_instances ? PULSEG_AMP_MAX_POS : PULSEG_AMP_ACTUAL,
        NULL,
        0,
        block_order);
    if (PULSEG_FAILED(rc))
    {
        if (block_order)
            PULSEG_FREE(block_order);
        return rc;
    }
    rc = calc_mech_resonances_from_uniform(
        spectra,
        diag,
        &uw,
        target_resolution_hz,
        max_freq_hz,
        num_instances,
        tr_duration_us,
        num_forbidden_bands,
        forbidden_bands,
        peak_log10_threshold,
        peak_norm_scale,
        peak_eps,
        peak_prominence,
        desc,
        sa_start_block,
        sa_block_count,
        /* A real gamma, never 0: sa_eps_for_band() scales the policy epsilon
         * of a zero-tolerance band by it, and a 0 would collapse that epsilon
         * to 0 and flag every candidate. Vendor ESP tables are exactly that
         * shape (amplitude column 0.0), so the plotting API must carry a
         * usable gamma to reproduce the headless verdict. */
        (opts && opts->gamma_hz_per_t > 0.0f) ? opts->gamma_hz_per_t : SA_GAMMA_1H_HZ_PER_T,
        (target_resolution_hz > 0.0f) ? 1 : 0,
        /* dense envelope: plotting API only, see
                                            * calc_mech_resonances_from_uniform doc */
        1 /* display products: this IS the plotting API */,
        compress_trains,
        bound_over_instances);
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

/* `desc` + the block window are optional: pass them and, when the model
 * exposes a kernel, the response is assembled from per-shape convolutions
 * instead of one transform over the whole window (pulseg_pns_memo.c). Pass
 * NULL and the exact full-waveform path always runs. */
static int calc_pns_from_uniform(
    pulseg_pns_result *result,
    pulseg_diagnostic *diag,
    float gamma_hz_per_tesla,
    const pulseg__uniform_grad_waveforms *waveforms,
    const pulseg_pns_model *model,
    const pulseg_sequence_descriptor *desc,
    int block_start,
    int block_count,
    const int *block_order)
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
    float *memo_kernel;
    int memo_kernel_len, memo_applied;
    float memo_scale;
    int rc;

    padded_scratch = NULL;
    dgdt_x = NULL;
    dgdt_y = NULL;
    dgdt_z = NULL;
    out_x = NULL;
    out_y = NULL;
    out_z = NULL;
    memo_kernel = NULL;
    memo_kernel_len = 0;
    memo_applied = 0;
    memo_scale = 1.0f;
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

    out_x = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    out_y = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    out_z = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    if (!out_x || !out_y || !out_z)
    {
        rc = PULSEG_ERR_ALLOC_FAILED;
        goto fail;
    }

    /* Fast path: a model that publishes its kernel is asserting it is a
     * linear filter, which lets the response be assembled from one
     * convolution per distinct gradient shape rather than one transform
     * over the whole window. The builder validates its own preconditions
     * against `waveforms` and reports memo_applied == 0 if they do not
     * hold, in which case the exact path below runs unchanged. */
    if (desc && model->kernel && block_count > 0)
    {
        rc = model->kernel(
            model->ctx,
            waveforms->raster_us,
            &memo_kernel,
            &memo_kernel_len,
            &memo_scale);
        if (PULSEG_FAILED(rc))
        {
            diag->code = rc;
            goto fail;
        }
        rc = pulseg__calc_pns_memoized(
            out_x,
            out_y,
            out_z,
            n,
            &memo_applied,
            desc,
            block_start,
            block_count,
            block_order,
            waveforms,
            gamma_hz_per_tesla,
            memo_kernel,
            memo_kernel_len,
            memo_scale,
            pad);
        PULSEG_FREE(memo_kernel);
        memo_kernel = NULL;
        if (PULSEG_FAILED(rc))
        {
            diag->code = rc;
            goto fail;
        }
    }

    if (memo_applied)
        goto emit;

    padded_scratch = (float *)PULSEG_ALLOC((size_t)(max_samples + pad) * sizeof(float));
    dgdt_x = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    dgdt_y = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    dgdt_z = (float *)PULSEG_ALLOC((size_t)n * sizeof(float));
    if (!padded_scratch || !dgdt_x || !dgdt_y || !dgdt_z)
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

emit:
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
    if (memo_kernel)
        PULSEG_FREE(memo_kernel);
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
/*  PNS over every shape the canonical TR takes                       */
/* ================================================================== */

/** Peak combined response over a result: the number the gate thresholds. */
static float pns_result_peak(const pulseg_pns_result *r)
{
    float best, v;
    int i;

    best = 0.0f;
    for (i = 0; i < r->num_samples; ++i)
    {
        v = 0.0f;
        if (r->slew_x_hz_per_m_per_s)
            v += r->slew_x_hz_per_m_per_s[i] * r->slew_x_hz_per_m_per_s[i];
        if (r->slew_y_hz_per_m_per_s)
            v += r->slew_y_hz_per_m_per_s[i] * r->slew_y_hz_per_m_per_s[i];
        if (r->slew_z_hz_per_m_per_s)
            v += r->slew_z_hz_per_m_per_s[i] * r->slew_z_hz_per_m_per_s[i];
        v = (float)sqrt((double)v);
        if (v > best)
            best = v;
    }
    return best;
}

/**
 * PNS over the canonical TR, worst over every shape the repetitions take.
 *
 * The per-position amplitude maximum bounds repetitions that play the same
 * gradient *definitions* -- the same shape driven harder is a larger response
 * everywhere -- but it does not bound a repetition that plays a *different*
 * definition at that position, because there is no amplitude at which one
 * spiral arm's shape covers another's. So the instances are grouped by the
 * definitions they play and one window is evaluated per group, at that
 * group's own shapes and its own amplitude maximum. A sequence whose
 * repetitions differ only in amplitude has one group, and this is then the
 * single envelope evaluation it has always been.
 *
 * Rotation is not part of the grouping: the extraction does not apply one, so
 * grouping by it would render the same waveform repeatedly. A rotated arm is
 * therefore covered only for a model that treats the axes alike, which the
 * chronaxie model does and a per-axis model does not.
 */
static int calc_pns_over_shape_groups(
    pulseg_pns_result *result,
    pulseg_diagnostic *diag,
    float gamma_hz_per_tesla,
    const pulseg_pns_model *model,
    const pulseg_sequence_descriptor *desc,
    int start_block,
    int block_count)
{
    pulseg__uniform_grad_waveforms uw;
    pulseg_pns_result candidate;
    int *labels;
    int *group_first;
    int num_groups, tr_size, g, rc, group_start, have_best;
    float peak, best_peak;

    labels = NULL;
    group_first = NULL;
    num_groups = 1;
    have_best = 0;
    best_peak = 0.0f;
    tr_size = desc->tr_descriptor.tr_size;

    rc = pulseg__group_tr_instances_by_shape(
        desc,
        &labels,
        &group_first,
        &num_groups,
        PULSEG__MAX_SHAPE_GROUPS);
    if (PULSEG_FAILED(rc))
    {
        /* More shapes than the sweep will enumerate. The envelope alone does
         * not bound them, and saying so is the only honest answer. */
        pulseg__diag_printf(
            diag,
            "PNS: the repetitions play more than %d distinct sets of gradient "
            "waveforms, which is more windows than the check will evaluate, and "
            "one window over all of them would not bound them. Write the "
            "repeated waveform once and turn it with a ROTATIONS extension",
            PULSEG__MAX_SHAPE_GROUPS);
        diag->code = PULSEG_ERR_PNS_INVALID_PARAMS;
        return diag->code;
    }

    memset(result, 0, sizeof(*result));

    for (g = 0; g < num_groups; ++g)
    {
        group_start = (labels && group_first) ? group_first[g] * tr_size : start_block;
        memset(&uw, 0, sizeof(uw));
        rc = pulseg__get_gradient_waveforms_range(
            desc,
            &uw,
            diag,
            group_start,
            block_count,
            PULSEG_AMP_MAX_POS,
            labels,
            g,
            NULL);
        if (PULSEG_FAILED(rc))
            goto done;

        memset(&candidate, 0, sizeof(candidate));
        rc = calc_pns_from_uniform(
            &candidate,
            diag,
            gamma_hz_per_tesla,
            &uw,
            model,
            desc,
            group_start,
            block_count,
            NULL);
        pulseg__uniform_grad_waveforms_free(&uw);
        if (PULSEG_FAILED(rc))
            goto done;

        peak = pns_result_peak(&candidate);
        if (!have_best || peak > best_peak)
        {
            pulseg_pns_result_free(result);
            *result = candidate;
            result->worst_group = g;
            best_peak = peak;
            have_best = 1;
        }
        else
        {
            pulseg_pns_result_free(&candidate);
        }
    }

done:
    if (PULSEG_FAILED(rc))
    {
        pulseg__uniform_grad_waveforms_free(&uw);
        pulseg_pns_result_free(result);
    }
    if (labels)
        PULSEG_FREE(labels);
    if (group_first)
        PULSEG_FREE(group_first);
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
    pulseg__uniform_grad_waveforms uw;
    pulseg_diagnostic local_diag;
    int rc, start_block, block_count, amplitude_mode, num_instances;
    int *block_order;
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
    if (canonical_tr_idx < 0 || canonical_tr_idx >= desc->tr_descriptor.num_trs)
    {
        diag->code = PULSEG_ERR_INVALID_ARGUMENT;
        return diag->code;
    }
    select_canonical_tr_window_idx(
        desc,
        &start_block,
        &block_count,
        &num_instances,
        &tr_duration_us,
        canonical_tr_idx);
    amplitude_mode = PULSEG_AMP_MAX_POS;
    (void)num_instances;
    (void)tr_duration_us;

    /* Index 0 is the request for the worst case rather than for a repetition,
     * and the worst case is worst over every shape the repetitions take. Any
     * other index is that window, on its own shapes. */
    if (canonical_tr_idx == 0)
        return calc_pns_over_shape_groups(
            result,
            diag,
            opts->gamma_hz_per_t,
            model,
            desc,
            start_block,
            block_count);

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
    rc = calc_pns_from_uniform(
        result,
        diag,
        opts->gamma_hz_per_t,
        &uw,
        model,
        desc,
        start_block,
        block_count,
        block_order);
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

int pulseg_check_grad_continuity(
    pulseg_collection *coll,
    pulseg_diagnostic *diag,
    const pulseg_opts *opts)
{
    pulseg_block_cursor saved_cursor;
    const pulseg_sequence_descriptor *desc;
    const pulseg_block_table_element *bte;
    const pulseg_grad_table_element *gte;
    int n, raw_id, rot_id, status, cur_seq;
    int shape_ids[3];
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

        /* grad table: amplitude + shape */

        raw_id = bte->gx_id;
        if (raw_id >= 0 && raw_id < desc->grad_table_size)
        {
            gte = &desc->grad_table[raw_id];
            amp[0] = gte->amplitude;
            shape_ids[0] = gte->shape_id;
        }
        else
        {
            amp[0] = 0.0f;
            shape_ids[0] = 0;
        }

        raw_id = bte->gy_id;
        if (raw_id >= 0 && raw_id < desc->grad_table_size)
        {
            gte = &desc->grad_table[raw_id];
            amp[1] = gte->amplitude;
            shape_ids[1] = gte->shape_id;
        }
        else
        {
            amp[1] = 0.0f;
            shape_ids[1] = 0;
        }

        raw_id = bte->gz_id;
        if (raw_id >= 0 && raw_id < desc->grad_table_size)
        {
            gte = &desc->grad_table[raw_id];
            amp[2] = gte->amplitude;
            shape_ids[2] = gte->shape_id;
        }
        else
        {
            amp[2] = 0.0f;
            shape_ids[2] = 0;
        }

        /* Endpoint values of the shape THIS instance plays, scaled by its own
         * amplitude.  Per shape rather than per definition: continuity is a
         * question about the waveform that actually runs, not about the
         * definition's worst one. */
        for (n = 0; n < 3; ++n)
        {
            first_val[n] = pulseg__grad_shape_first(desc, shape_ids[n]) * amp[n];
            last_val[n] = pulseg__grad_shape_last(desc, shape_ids[n]) * amp[n];
        }

        /* transform logical -> physical */
        rot_id = bte->rotation_id;
        if (rot_id >= 0 && rot_id < desc->num_rotations)
        {
            pulseg__apply_rotation(first_phys, desc->rotation_matrices[rot_id], first_val, 0);
            pulseg__apply_rotation(last_phys, desc->rotation_matrices[rot_id], last_val, 0);
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
     * Slew per axis at each block = slew_rate_normalised * amplitude, both
     * read per block instance from grad_table: the amplitude directly, and
     * the normalised slew (1/s) from the shape that instance names.  This
     * mirrors check_max_grad, which also iterates block_table.
     */
    int s, b, n, raw_id, def_idx;
    float slew_sq, slew_sq_max, limit_sq, amp, shape_slew;
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
                amp = desc->grad_table[raw_id].amplitude;
                if (amp < 0.0f)
                    amp = -amp;

                if (def_idx < 0 || def_idx >= desc->num_unique_grads)
                    continue;

                gdef = &desc->grad_definitions[def_idx];
                /* This instance's own amplitude times the steepest normalised
                 * slew of the shape it actually plays.  Grad definitions are
                 * deduplicated without the magnitude shape id, so the
                 * definition's own ceiling would pair the steepest shape in
                 * the family with the largest amplitude in it -- an upper
                 * bound, but on an object no instance plays.  A trapezoid has
                 * no shape and its one profile is the definition's. */
                shape_slew = pulseg__grad_shape_slew(desc, desc->grad_table[raw_id].shape_id);
                axis_slew[n] = (shape_slew > 0.0f ? shape_slew : gdef->any.max_slew_rate) * amp;
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
    pulseg__uniform_grad_waveforms uw;
    pulseg_mech_resonances_spectra spectra;
    pulseg_pns_result pns_result;
    int start_block, block_count, num_instances;
    int sa_start_block, sa_block_count;
    int *block_order;
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

    /* ---- 0. raster alignment ----
     * Ahead of the gradient-presence skip below: it judges RF, ADC and block
     * durations too, which an RF-only sequence still has. */
    rc = pulseg_check_raster_alignment(coll, diag, opts);
    if (PULSEG_FAILED(rc))
        return rc;

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
    rc = pulseg_check_grad_continuity(coll, diag, opts);
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
        unique_tr_indices = NULL;
        tr_group_labels = NULL;

        select_canonical_tr_window(
            desc,
            &start_block,
            &block_count,
            &num_instances,
            &tr_duration_us);
        sa_start_block = start_block;
        sa_block_count = block_count;
        block_order = NULL;

        /* Evaluate one canonical TR per shot-ID combination. */
        unique_tr_indices = NULL;
        tr_group_labels = NULL;
        num_unique_trs = 0 /* one canonical TR; representatives carry the worst case */;
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
                    opts->gamma_hz_per_t,
                    0 /* compute_dense_envelope: never on the PSD path */,
                    0 /* compute_display_products: never on the PSD path */,
                    1 /* compress_trains: this is the path being optimised */,
                    1 /* bound over every instance of the canonical TR */);

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
                                sa_eps_for_band(&forbidden_bands[b], opts->gamma_hz_per_t);
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
                /* Not the `uw` above: that window carries one group's shapes
                 * at the whole scan's amplitudes, which bounds the
                 * repetitions that share those shapes and no others. */
                rc = calc_pns_over_shape_groups(
                    &pns_result,
                    diag,
                    opts->gamma_hz_per_t,
                    pns_model,
                    desc,
                    start_block,
                    block_count);
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
