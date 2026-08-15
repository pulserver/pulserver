/**
 * @file pulseq_ktraj.c
 * @brief Absolute k at every ADC sample, and where each readout's echo is.
 *
 * See pulseq_ktraj.h for what this is for and why it is not a cumulative sum
 * over a materialised waveform.  The shape of the implementation is:
 *
 *   1. integrate every gradient library row once, at unit amplitude, into a
 *      piecewise-linear breakpoint list with a running integral (`ktraj_wave`);
 *   2. walk the blocks once, accumulating k and recording where each block
 *      starts (`plan_walk_blocks`);
 *   3. per readout, emit `origin + amplitude * cumulative(t)` per axis, then
 *      rotate (`readout_samples`);
 *   4. two memoized sweeps to derive `center_sample` (`derive_centers`).
 *
 * Raw-model units, which this file converts and nothing above it should have
 * to think about again: library times are microseconds, ADC dwell is
 * nanoseconds, block durations are counts of the block-duration raster, and
 * the rasters themselves are microseconds.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "pulseq.h"
#include "pulseq_ktraj.h"

/* ================================================================== */
/*  Raw-model column layouts                                          */
/*                                                                    */
/*  Mirrors what cxx/pulseq/read.cpp converts out of, and what        */
/*  write_text.cpp documents in each section header.  The gradient    */
/*  discriminator is the one to be careful with: read_grad_library()  */
/*  writes 1 for a row that came from [GRADIENTS] and 0 for one from  */
/*  [TRAP].  It is NOT PULSEQ_GRAD_ARB / PULSEQ_GRAD_TRAP, which are  */
/*  a different numbering used by external raw-model producers.       */
/* ================================================================== */

#define KTRAJ_GRAD_IS_ARBITRARY 1

/* [GRADIENTS]: kind amplitude first last amp_shape time_shape delay */
#define KTRAJ_ARB_AMPLITUDE 1
#define KTRAJ_ARB_FIRST 2
#define KTRAJ_ARB_LAST 3
#define KTRAJ_ARB_SHAPE 4
#define KTRAJ_ARB_TIME_SHAPE 5
#define KTRAJ_ARB_DELAY 6

/* [TRAP]: kind amplitude rise flat fall delay */
#define KTRAJ_TRAP_AMPLITUDE 1
#define KTRAJ_TRAP_RISE 2
#define KTRAJ_TRAP_FLAT 3
#define KTRAJ_TRAP_FALL 4
#define KTRAJ_TRAP_DELAY 5

/* [RF]: ampl mag_id phase_id time_id center delay freqPPM phasePPM freq phase */
#define KTRAJ_RF_CENTER 4
#define KTRAJ_RF_DELAY 5
#define KTRAJ_RF_FREQ_PPM 6
#define KTRAJ_RF_PHASE_PPM 7
#define KTRAJ_RF_FREQ 8
#define KTRAJ_RF_PHASE 9

/* [ADC]: num dwell delay freqPPM phasePPM freq phase phase_id */
#define KTRAJ_ADC_NUM 0
#define KTRAJ_ADC_DWELL 1
#define KTRAJ_ADC_DELAY 2

#define KTRAJ_US 1.0e-6
#define KTRAJ_NS 1.0e-9

/* Pulseq's own default when [DEFINITIONS] omits a raster, in microseconds. */
#define KTRAJ_DEFAULT_GRAD_RASTER_US 10.0
#define KTRAJ_DEFAULT_BLOCK_RASTER_US 10.0

/* ================================================================== */
/*  Piecewise-linear waveform with a running integral                 */
/* ================================================================== */

/*
 * Value v[i] at time t[i], linear between, zero outside.  c[i] is the
 * integral from t[0] to t[i], so evaluating the cumulative anywhere costs a
 * binary search plus the area of one partial trapezoid.
 *
 * Amplitude is deliberately kept out of the samples: the shape is what
 * repeats across a scan, and multiplying it in would give every phase-encode
 * step its own numerically distinct array.
 */
typedef struct ktraj_wave
{
    int n;
    double *t;
    double *v;
    double *c;
    /*
     * The integral of `c` -- the *second* antiderivative of the waveform.
     *
     * Only built when the caller asks for window-averaged sampling, which is
     * the whole reason it exists: averaging k over a dwell means integrating
     * k, and k is already an integral.  See wave_mean_cumulative().
     */
    double *c2;
    double amplitude; /* Hz/m */
    double total;     /* c[n-1], seconds */
    int built;
} ktraj_wave;

static void wave_init(ktraj_wave *w)
{
    w->n = 0;
    w->t = NULL;
    w->v = NULL;
    w->c = NULL;
    w->c2 = NULL;
    w->amplitude = 0.0;
    w->total = 0.0;
    w->built = 0;
}

static void wave_free(ktraj_wave *w)
{
    if (!w)
        return;
    PULSEQ_FREE(w->t);
    PULSEQ_FREE(w->v);
    PULSEQ_FREE(w->c);
    PULSEQ_FREE(w->c2);
    wave_init(w);
}

static int wave_alloc(ktraj_wave *w, int n)
{
    w->t = (double *)PULSEQ_ALLOC((size_t)n * sizeof(double));
    w->v = (double *)PULSEQ_ALLOC((size_t)n * sizeof(double));
    w->c = (double *)PULSEQ_ALLOC((size_t)n * sizeof(double));
    if (!w->t || !w->v || !w->c)
    {
        wave_free(w);
        return PULSEQ_ERR_ALLOC_FAILED;
    }
    w->n = n;
    return PULSEQ_SUCCESS;
}

static void wave_integrate(ktraj_wave *w)
{
    int i;
    w->c[0] = 0.0;
    for (i = 1; i < w->n; ++i)
        w->c[i] = w->c[i - 1] + 0.5 * (w->v[i] + w->v[i - 1]) * (w->t[i] - w->t[i - 1]);
    w->total = (w->n > 0) ? w->c[w->n - 1] : 0.0;
}

/* Largest i with t[i] <= x, clamped to [0, n-2]. */
static int wave_segment(const ktraj_wave *w, double x)
{
    int lo = 0;
    int hi = w->n - 1;
    int mid;

    while (hi - lo > 1)
    {
        mid = lo + (hi - lo) / 2;
        if (w->t[mid] <= x)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

/* The waveform at x, holding the end values outside its extent. */
static double wave_value_at(const ktraj_wave *w, double x)
{
    int i;
    double span;

    if (w->n < 2)
        return 0.0;
    if (x <= w->t[0])
        return w->v[0];
    if (x >= w->t[w->n - 1])
        return w->v[w->n - 1];

    i = wave_segment(w, x);
    span = w->t[i + 1] - w->t[i];
    if (span <= 0.0)
        return w->v[i + 1];
    return w->v[i] + (w->v[i + 1] - w->v[i]) * (x - w->t[i]) / span;
}

/* The integral from t[0] to x. */
static double wave_cumulative_at(const ktraj_wave *w, double x)
{
    int i;
    double span, frac, vx;

    if (w->n < 2)
        return 0.0;
    if (x <= w->t[0])
        return 0.0;
    if (x >= w->t[w->n - 1])
        return w->c[w->n - 1];

    i = wave_segment(w, x);
    span = w->t[i + 1] - w->t[i];
    if (span <= 0.0)
        return w->c[i + 1];
    frac = (x - w->t[i]) / span;
    vx = w->v[i] + (w->v[i + 1] - w->v[i]) * frac;
    return w->c[i] + 0.5 * (w->v[i] + vx) * (x - w->t[i]);
}

/*
 * Build the second antiderivative, once, on demand.
 *
 * Exact rather than approximated: on a segment the waveform is linear, so its
 * integral `c` is quadratic and the integral of that is cubic, and a cubic has
 * a closed form.  Approximating here would defeat the purpose -- the whole
 * correction being computed is smaller than one sample step.
 */
static int wave_build_second(ktraj_wave *w)
{
    int i;

    if (w->c2 || w->n < 2)
        return PULSEQ_SUCCESS;

    w->c2 = (double *)PULSEQ_ALLOC((size_t)w->n * sizeof(double));
    if (!w->c2)
        return PULSEQ_ERR_ALLOC_FAILED;

    w->c2[0] = 0.0;
    for (i = 1; i < w->n; ++i)
    {
        double span = w->t[i] - w->t[i - 1];
        double slope = (span > 0.0) ? (w->v[i] - w->v[i - 1]) / span : 0.0;
        w->c2[i] = w->c2[i - 1] + w->c[i - 1] * span + w->v[i - 1] * span * span / 2.0 +
                   slope * span * span * span / 6.0;
    }
    return PULSEQ_SUCCESS;
}

/* The integral of the cumulative, from t[0] to x. */
static double wave_second_at(const ktraj_wave *w, double x)
{
    int i;
    double span, slope, u;

    if (w->n < 2 || !w->c2)
        return 0.0;
    if (x <= w->t[0])
        return 0.0;
    if (x >= w->t[w->n - 1])
    {
        /* Past the end the waveform is silent, so the cumulative holds at its
         * total and the integral of it grows linearly. */
        return w->c2[w->n - 1] + w->c[w->n - 1] * (x - w->t[w->n - 1]);
    }

    i = wave_segment(w, x);
    span = w->t[i + 1] - w->t[i];
    slope = (span > 0.0) ? (w->v[i + 1] - w->v[i]) / span : 0.0;
    u = x - w->t[i];
    return w->c2[i] + w->c[i] * u + w->v[i] * u * u / 2.0 + slope * u * u * u / 6.0;
}

/*
 * The cumulative averaged over one dwell window centred on @p x.
 *
 * An ADC sample is not an instant: it integrates signal for a whole dwell, so
 * the coordinate it belongs to is k averaged across that window rather than k
 * at its midpoint.  For a smooth gradient the difference is
 * (dwell^2 / 24) * dg/dt -- identically zero while the gradient is flat, which
 * is why it goes unnoticed on an ordinary Cartesian readout, and a real
 * systematic offset on a ramp-sampled one.
 *
 * PyPulseq, MRpro and mri-nufft all sample at the midpoint, so this is opt-in:
 * switching it on is a deliberate departure from what those three agree on.
 */
static double wave_mean_cumulative(const ktraj_wave *w, double x, double dwell)
{
    if (w->n < 2 || !w->c2 || !(dwell > 0.0))
        return wave_cumulative_at(w, x);
    return (wave_second_at(w, x + 0.5 * dwell) - wave_second_at(w, x - 0.5 * dwell)) / dwell;
}

/*
 * Flat across [t0, t1]?
 *
 * Checked at the window edges and at every breakpoint strictly inside it,
 * which is exact for a piecewise-linear waveform: a segment can only depart
 * from a constant at its own ends.  The tolerance is relative to the
 * waveform's own scale, so a normalised ramp of 1e-12 does not read as
 * structure.
 */
static int wave_is_flat(const ktraj_wave *w, double t0, double t1)
{
    double lo, hi, edge, scale, tol;
    int i;

    if (w->n < 2)
        return 1;

    lo = wave_value_at(w, t0);
    hi = lo;
    edge = wave_value_at(w, t1);
    if (edge < lo)
        lo = edge;
    if (edge > hi)
        hi = edge;

    scale = 0.0;
    for (i = 0; i < w->n; ++i)
    {
        if (w->t[i] > t0 && w->t[i] < t1)
        {
            if (w->v[i] < lo)
                lo = w->v[i];
            if (w->v[i] > hi)
                hi = w->v[i];
        }
        if (fabs(w->v[i]) > scale)
            scale = fabs(w->v[i]);
    }

    tol = 1.0e-9 * ((scale > 1.0) ? scale : 1.0);
    return (hi - lo) <= tol;
}

/* ================================================================== */
/*  Building a wave from a gradient library row                       */
/* ================================================================== */

static int build_trapezoid(ktraj_wave *w, const PULSEQ_REAL *row)
{
    double rise = (double)row[KTRAJ_TRAP_RISE] * KTRAJ_US;
    double flat = (double)row[KTRAJ_TRAP_FLAT] * KTRAJ_US;
    double fall = (double)row[KTRAJ_TRAP_FALL] * KTRAJ_US;
    double delay = (double)row[KTRAJ_TRAP_DELAY] * KTRAJ_US;
    int n = (flat > 0.0) ? 4 : 3;
    int rc;
    double at;

    rc = wave_alloc(w, n);
    if (PULSEQ_FAILED(rc))
        return rc;

    at = delay;
    w->t[0] = at;
    w->v[0] = 0.0;
    at += rise;
    w->t[1] = at;
    w->v[1] = 1.0;
    if (flat > 0.0)
    {
        at += flat;
        w->t[2] = at;
        w->v[2] = 1.0;
        at += fall;
        w->t[3] = at;
        w->v[3] = 0.0;
    }
    else
    {
        at += fall;
        w->t[2] = at;
        w->v[2] = 0.0;
    }

    w->amplitude = (double)row[KTRAJ_TRAP_AMPLITUDE];
    wave_integrate(w);
    return PULSEQ_SUCCESS;
}

/*
 * An arbitrary gradient at unit amplitude.
 *
 * The shape library holds the waveform already normalised, so the samples are
 * used as they stand.  `first` and `last` are absolute (Hz/m) and close the
 * ends at the block's own boundaries; they are divided back down by the
 * amplitude so they live in the same units as the samples.  With
 * time_shape_id == 0 the samples sit at raster centres, which is the
 * convention the file's own [GRADIENTS] header states.
 */
static int build_arbitrary(ktraj_wave *w, const pulseq_file *seq, const PULSEQ_REAL *row,
                           double grad_raster)
{
    double amplitude = (double)row[KTRAJ_ARB_AMPLITUDE];
    double first = (double)row[KTRAJ_ARB_FIRST];
    double last = (double)row[KTRAJ_ARB_LAST];
    int shape_id = (int)row[KTRAJ_ARB_SHAPE];
    int time_shape_id = (int)row[KTRAJ_ARB_TIME_SHAPE];
    double delay = (double)row[KTRAJ_ARB_DELAY] * KTRAJ_US;
    pulseq_shape amp;
    pulseq_shape tim;
    int have_amp = 0;
    int have_tim = 0;
    int n, nt, i, count, rc;
    double scale, end;

    if (shape_id <= 0 || shape_id > seq->shapes_library_size || !seq->shapes_library)
        return PULSEQ_ERR_INVALID_ARGUMENT;

    memset(&amp, 0, sizeof(amp));
    memset(&tim, 0, sizeof(tim));

    if (!pulseq_decompress_shape(&amp, &seq->shapes_library[shape_id - 1], (PULSEQ_REAL)1.0))
        return PULSEQ_ERR_INVALID_ARGUMENT;
    have_amp = 1;
    n = amp.num_samples;
    if (n <= 0)
    {
        rc = PULSEQ_ERR_INVALID_ARGUMENT;
        goto done;
    }

    nt = 0;
    if (time_shape_id > 0 && time_shape_id <= seq->shapes_library_size)
    {
        if (!pulseq_decompress_shape(&tim, &seq->shapes_library[time_shape_id - 1],
                                     (PULSEQ_REAL)1.0))
        {
            rc = PULSEQ_ERR_INVALID_ARGUMENT;
            goto done;
        }
        have_tim = 1;
        nt = tim.num_samples;
        if (nt < n)
            n = nt;
    }

    /* One breakpoint per sample, plus the two that close the ends. */
    rc = wave_alloc(w, n + 2);
    if (PULSEQ_FAILED(rc))
        goto done;

    scale = (amplitude != 0.0) ? 1.0 / amplitude : 0.0;

    w->t[0] = delay;
    w->v[0] = first * scale;
    for (i = 0; i < n; ++i)
    {
        w->t[i + 1] = (nt > 0) ? (delay + (double)tim.samples[i] * grad_raster)
                               : (delay + ((double)i + 0.5) * grad_raster);
        w->v[i + 1] = (double)amp.samples[i];
    }

    end = (nt > 0) ? w->t[n] : (delay + (double)n * grad_raster);
    count = n + 1;
    if (end > w->t[n])
    {
        w->t[n + 1] = end;
        w->v[n + 1] = last * scale;
        count = n + 2;
    }
    w->n = count;

    w->amplitude = amplitude;
    wave_integrate(w);
    rc = PULSEQ_SUCCESS;

done:
    if (have_amp)
        PULSEQ_FREE(amp.samples);
    if (have_tim)
        PULSEQ_FREE(tim.samples);
    return rc;
}

/* ================================================================== */
/*  The plan                                                          */
/* ================================================================== */

/*
 * The memo key for a readout.  Two blocks agreeing on all of it produce
 * bit-identical samples and the same echo index.
 *
 * The rotation id is load-bearing.  Gradient ids carry shape *and* amplitude
 * here, but not orientation, so two blocks differing only by a per-slice
 * rotation share every gradient id and would collide without it.  The entry
 * k is in the key because the echo index depends on where the readout
 * starts, not only on its shape.
 */
typedef struct ktraj_repeat_key
{
    /*
     * Gradient ids per axis, ADC id, rotation id.
     *
     * An axis the readout does not sweep is zeroed out of the *memo* copy of
     * this key before hashing -- its gradient contributes the same constant
     * to every sample and cannot move the echo, so two readouts differing
     * only in it have the same answer.  That is not a detail: in a blipped
     * Cartesian scan every line carries its own phase-encode gradient id, so
     * keeping the id would give one group per line and no memo at all
     * (measured: 0% on gre_32x32_pe_blip).  The plan's own copy always keeps
     * the true ids -- readout_samples() reads them to find the waveforms.
     */
    int grad[3];
    int adc, rotation;
    /*
     * Which axes the readout sweeps, carried in the key rather than assumed.
     *
     * Zeroing a static axis out of grad[] above means two readouts can now
     * agree on the surviving fields while having disagreed about which axes
     * were static in the first place.  Including the mask makes that
     * impossible instead of merely unlikely, for one int.
     */
    unsigned int swept;
    /*
     * The entry k on the axes the readout sweeps, compared *exactly*.
     *
     * An earlier version rounded these onto a grid, on the theory that two
     * entry points closer than a fraction of the sampling step could not
     * pick different samples.  That is false near a tie, and it produced a
     * wrong answer: in fse_2d_1sl_1avg the two alternating readouts enter
     * 0.0055 apart, well inside a 0.078 quantum, yet their echoes really are
     * samples 127 and 128 -- both are 1.942783 from the origin, on opposite
     * sides.  Merging them handed one readout the other's echo.
     *
     * Exact comparison cannot do that, and it costs nothing in hit rate: an
     * excitation resets k to zero, so every TR reaches its readout by summing
     * the same gradient moments from the same reset and arrives at bit-
     * identical entry points.  Repeats match exactly because they are exact
     * repeats.
     */
    double k0v[3];
} ktraj_repeat_key;

struct pulseq_ktraj_plan
{
    const pulseq_file *seq;
    pulseq_ktraj_opts opts;

    int first_block; /* 0-based inclusive */
    int last_block;  /* 0-based inclusive */

    double grad_raster;  /* s */
    double block_raster; /* s */
    double total_duration;

    ktraj_wave *waves; /* one per gradient library row */
    int num_waves;

    /*
     * Rotation matrices, row-major, one per rotation library row.
     *
     * Built here rather than read from `pulseq_file::rotation_matrix_library`,
     * which the reader allocates but never fills: the .seq stores quaternions,
     * and the only thing that converts them today is pulseg's
     * copy_rotation_library().  pulseq is standalone and must not reach into
     * pulseg, so the conversion is repeated here -- in double, and matching
     * pulseg__quaternion_to_matrix() term for term, because two different
     * rotation conventions in one repository would be a silent wrong-image bug.
     */
    double *rot; /* 9 * num_rot */
    int num_rot;

    /* Row id -> first equivalent row id; see canonicalise_rows(). */
    int *grad_canon;
    int *adc_canon;

    /* Per block in range: k at its start (logical frame) and its start time. */
    double *origin; /* 3 * num_range_blocks */
    double *t_block;
    int num_range_blocks;

    int num_readouts;
    pulseq_ktraj_readout *readouts;
    ktraj_repeat_key *keys;
    /*
     * Each readout's first-sample time and dwell, relative to its own block
     * and in double.
     *
     * Not recoverable from the public struct, and that is the point.  Its t0
     * is an absolute time in PULSEQ_REAL, and on a float32 build a 64-second
     * scan resolves absolute time to about 4 us -- the size of a dwell.
     * Reconstructing the in-block offset as (t0 - block_start) then loses the
     * whole thing to cancellation, which is not a rounding error but a
     * one-sample error in the echo (gre_32x32_pe_blip, every readout).
     * Keeping the small number small avoids the subtraction entirely.
     */
    double *adc_offset;
    double *adc_dwell;
    /* Distinct repeat keys, so a caller can see whether the memo fired. */
    int num_key_groups;

    int num_exc;
    pulseq_ktraj_rf *exc;
    int num_ref;
    pulseq_ktraj_rf *ref;

    double k_center[3];
    /* Readout achieving k_center, or -1 when centres were not derived. */
    int central_readout;
};

/* ------------------------------------------------------------------ */

static double raster_seconds(PULSEQ_REAL declared, double fallback_us)
{
    double us = (double)declared;
    if (!(us > 0.0))
        us = fallback_us;
    return us * KTRAJ_US;
}

/*
 * B0 for the ppm terms on RF and ADC rows.
 *
 * gamma * B0 is a system property, not a file one, so a .seq can only supply
 * it through [DEFINITIONS].  When it does not, the ppm contribution is taken
 * as zero -- which is exact for every file that leaves those columns at 0,
 * i.e. everything written before Pulseq 1.5 and most since.
 */
static double definition_scalar(const pulseq_file *seq, const char *name, double fallback)
{
    int i;
    if (!seq->definitions_library)
        return fallback;
    for (i = 0; i < seq->num_definitions; ++i)
    {
        if (strcmp(seq->definitions_library[i].name, name) == 0 &&
            seq->definitions_library[i].value_size > 0 && seq->definitions_library[i].value &&
            seq->definitions_library[i].value[0])
            return atof(seq->definitions_library[i].value[0]);
    }
    return fallback;
}

static double gamma_b0(const pulseq_file *seq)
{
    /* Pulseq writes B0 in tesla; gamma is the proton value in Hz/T. */
    double b0 = definition_scalar(seq, "B0", 0.0);
    return b0 * 42576000.0;
}

/* Integrate every gradient library row once. */
static int build_waves(pulseq_ktraj_plan *plan)
{
    const pulseq_file *seq = plan->seq;
    int i, rc;

    plan->num_waves = seq->grad_library_size;
    if (plan->num_waves <= 0)
        return PULSEQ_SUCCESS;

    plan->waves = (ktraj_wave *)PULSEQ_ALLOC((size_t)plan->num_waves * sizeof(ktraj_wave));
    if (!plan->waves)
        return PULSEQ_ERR_ALLOC_FAILED;
    for (i = 0; i < plan->num_waves; ++i)
        wave_init(&plan->waves[i]);

    for (i = 0; i < plan->num_waves; ++i)
    {
        const PULSEQ_REAL *row = seq->grad_library[i];
        if ((int)row[0] == KTRAJ_GRAD_IS_ARBITRARY)
            rc = build_arbitrary(&plan->waves[i], seq, row, plan->grad_raster);
        else
            rc = build_trapezoid(&plan->waves[i], row);

        /* A row this file never references, or one with a broken shape id,
         * is not an error until something asks for it -- leave it unbuilt
         * and let the block walk treat it as absent. */
        if (PULSEQ_FAILED(rc))
        {
            if (rc == PULSEQ_ERR_ALLOC_FAILED)
                return rc;
            wave_free(&plan->waves[i]);
            continue;
        }
        plan->waves[i].built = 1;

        if (plan->opts.sample_window_average)
        {
            rc = wave_build_second(&plan->waves[i]);
            if (PULSEQ_FAILED(rc))
                return rc;
        }
    }
    return PULSEQ_SUCCESS;
}

/*
 * Convert every rotation quaternion to a matrix.
 *
 * The same construction as pulseg__quaternion_to_matrix(): (w, x, y, z),
 * normalised first, degenerate rows falling back to the identity.  Kept
 * literally in step with it -- if that one changes, this one has to.
 */
static int build_rotations(pulseq_ktraj_plan *plan)
{
    const pulseq_file *seq = plan->seq;
    int i;

    plan->num_rot = seq->rotation_library_size;
    if (plan->num_rot <= 0 || !seq->rotation_quaternion_library)
    {
        plan->num_rot = 0;
        return PULSEQ_SUCCESS;
    }

    plan->rot = (double *)PULSEQ_ALLOC((size_t)plan->num_rot * 9 * sizeof(double));
    if (!plan->rot)
        return PULSEQ_ERR_ALLOC_FAILED;

    for (i = 0; i < plan->num_rot; ++i)
    {
        const PULSEQ_REAL *q = seq->rotation_quaternion_library[i];
        double *m = plan->rot + (size_t)i * 9;
        double w = (double)q[0];
        double x = (double)q[1];
        double y = (double)q[2];
        double z = (double)q[3];
        double norm = sqrt(w * w + x * x + y * y + z * z);
        double xx, yy, zz, xy, xz, yz, wx, wy, wz, inv;

        if (!(norm > 1.0e-9))
        {
            m[0] = 1.0; m[1] = 0.0; m[2] = 0.0;
            m[3] = 0.0; m[4] = 1.0; m[5] = 0.0;
            m[6] = 0.0; m[7] = 0.0; m[8] = 1.0;
            continue;
        }
        inv = 1.0 / norm;
        w *= inv;
        x *= inv;
        y *= inv;
        z *= inv;

        xx = x * x;
        yy = y * y;
        zz = z * z;
        xy = x * y;
        xz = x * z;
        yz = y * z;
        wx = w * x;
        wy = w * y;
        wz = w * z;

        m[0] = 1.0 - 2.0 * (yy + zz);
        m[1] = 2.0 * (xy - wz);
        m[2] = 2.0 * (xz + wy);
        m[3] = 2.0 * (xy + wz);
        m[4] = 1.0 - 2.0 * (xx + zz);
        m[5] = 2.0 * (yz - wx);
        m[6] = 2.0 * (xz - wy);
        m[7] = 2.0 * (yz + wx);
        m[8] = 1.0 - 2.0 * (xx + yy);
    }
    return PULSEQ_SUCCESS;
}

/* The rotation matrix for a 0-based rotation id, or NULL. */
static const double *rotation_for(const pulseq_ktraj_plan *plan, int rotation_id)
{
    if (!plan->rot || rotation_id < 0 || rotation_id >= plan->num_rot)
        return NULL;
    return plan->rot + (size_t)rotation_id * 9;
}

/*
 * Map each library row to the first row that means the same thing.
 *
 * Library ids are only as good as the writer's deduplication, and this
 * project's plugins ship with it switched off -- gre_32x32_pe_blip carries 32
 * byte-identical readout gradients under 32 different ids.  Keying the memo on
 * the id alone would then see 32 distinct readouts where there is one, and the
 * memo would never fire on exactly the sequences it is meant to help.
 *
 * @p columns lists which of the row's fields actually matter, so a caller can
 * ignore the ones that do not: an ADC's frequency and phase offsets are
 * receiver settings and move no gradient, yet RF spoiling gives every line a
 * different phase and so a different ADC id.
 *
 * One pass with a hash, so this costs the size of the library and not the
 * length of the scan.
 */
static int canonicalise_rows(const PULSEQ_REAL *rows, int n, int width, const int *columns,
                             int num_columns, int **out_canon)
{
    int capacity = 16;
    int *table = NULL;
    int *canon = NULL;
    int i, c;
    unsigned long mask;

    *out_canon = NULL;
    if (n <= 0)
        return PULSEQ_SUCCESS;

    while (capacity < n * 2)
    {
        if (capacity > (1 << 29))
            break;
        capacity <<= 1;
    }
    mask = (unsigned long)(capacity - 1);

    table = (int *)PULSEQ_ALLOC((size_t)capacity * sizeof(int));
    canon = (int *)PULSEQ_ALLOC((size_t)n * sizeof(int));
    if (!table || !canon)
    {
        PULSEQ_FREE(table);
        PULSEQ_FREE(canon);
        return PULSEQ_ERR_ALLOC_FAILED;
    }
    for (i = 0; i < capacity; ++i)
        table[i] = -1;

    for (i = 0; i < n; ++i)
    {
        unsigned long h = 2166136261UL;
        unsigned long slot;

        for (c = 0; c < num_columns; ++c)
        {
            double v = (double)rows[(size_t)i * (size_t)width + (size_t)columns[c]];
            unsigned char bytes[sizeof(double)];
            int b;
            if (v == 0.0)
                v = 0.0; /* -0.0 compares equal, so it must hash equal too */
            memcpy(bytes, &v, sizeof(double));
            for (b = 0; b < (int)sizeof(double); ++b)
            {
                h ^= (unsigned long)bytes[b];
                h *= 16777619UL;
            }
        }

        slot = h & mask;
        canon[i] = i;
        for (;;)
        {
            int occupant = table[slot];
            int same;
            if (occupant < 0)
            {
                table[slot] = i;
                break;
            }
            same = 1;
            for (c = 0; c < num_columns; ++c)
            {
                if (rows[(size_t)i * (size_t)width + (size_t)columns[c]] !=
                    rows[(size_t)occupant * (size_t)width + (size_t)columns[c]])
                {
                    same = 0;
                    break;
                }
            }
            if (same)
            {
                canon[i] = canon[occupant];
                break;
            }
            slot = (slot + 1) & mask;
        }
    }

    PULSEQ_FREE(table);
    *out_canon = canon;
    return PULSEQ_SUCCESS;
}

/* The wave behind a 0-based gradient id, or NULL. */
static const ktraj_wave *wave_for(const pulseq_ktraj_plan *plan, int grad_id)
{
    if (grad_id < 0 || grad_id >= plan->num_waves)
        return NULL;
    if (!plan->waves[grad_id].built)
        return NULL;
    return &plan->waves[grad_id];
}

/*
 * The 0-based rotation id a block carries, or -1.
 *
 * Label/trigger/shim links on the same chain are resolved and ignored by
 * pulseq_get_raw_extension, which walks the whole chain in one pass.
 */
static int block_rotation_id(const pulseq_ktraj_plan *plan, const pulseq_raw_block *raw)
{
    pulseq_raw_extension ext;

    if (raw->ext_count <= 0 || plan->num_rot <= 0)
        return -1;

    pulseq_get_raw_extension(plan->seq, &ext, raw);
    if (ext.rotation_index < 0 || ext.rotation_index >= plan->num_rot)
        return -1;
    return ext.rotation_index;
}

/*
 * How PyPulseq classifies an RF pulse, which is what this has to match:
 * "excitation" and "undefined" both reset k, "refocusing" negates it, and
 * inversion / saturation / preparation / other leave it alone.
 *
 * Treating an undefined use as an excitation matters -- a file that never
 * declared its uses is the common foreign case, and leaving k running through
 * every pulse of one would put the whole scan in the wrong place.  Note this
 * differs from pulseq::block_k_origins() in cxx/pulseq/trajectory.cpp, which
 * only resets on an explicit 'e'; that function answers to the FOV-shift
 * contract, not to PyPulseq's.
 */
#define KTRAJ_RF_NEUTRAL 0
#define KTRAJ_RF_EXCITATION 1
#define KTRAJ_RF_REFOCUSING 2
#define KTRAJ_RF_UNDECLARED 3

/*
 * What a pulse does to k, or KTRAJ_RF_UNDECLARED if the file does not say.
 *
 * An undeclared use is refused rather than guessed at.  PyPulseq and Pulseq
 * both read one as an excitation, and that guess is load-bearing: an
 * excitation resets k to zero, so getting it wrong means k accumulates across
 * the whole scan instead of restarting each TR.  Measured on a 32-line GRE
 * whose pulses carried no use: the origin of the last block came out 2.8e4
 * 1/m against a k-max of the same size.  A number that wrong, arrived at
 * silently, is worse than no number -- and the caller who wrote the sequence
 * is the one who knows what the pulse is for.
 */
static int rf_role(const pulseq_file *seq, int rf_id)
{
    int tag;

    if (rf_id < 0 || rf_id >= seq->rf_library_size)
        return KTRAJ_RF_NEUTRAL;

    tag = seq->rf_use_tags ? seq->rf_use_tags[rf_id] : PULSEQ_RF_USE_UNKNOWN;
    if (tag == PULSEQ_RF_USE_UNKNOWN)
        return KTRAJ_RF_UNDECLARED;
    if (tag == PULSEQ_RF_USE_EXCITATION)
        return KTRAJ_RF_EXCITATION;
    if (tag == PULSEQ_RF_USE_REFOCUSING)
        return KTRAJ_RF_REFOCUSING;
    return KTRAJ_RF_NEUTRAL;
}

/* ------------------------------------------------------------------ */

/*
 * One pass over the blocks: accumulate k, record origins and start times,
 * and inventory the ADC and RF events.
 *
 * A block whose RF resets or reverses k is split at the pulse centre, so
 * crushers on either side of a refocusing pulse -- the ordinary case -- land
 * on the correct side of the sign flip.
 */
static int plan_walk_blocks(pulseq_ktraj_plan *plan)
{
    const pulseq_file *seq = plan->seq;
    double k[3];
    double t = 0.0;
    double gb0 = gamma_b0(seq);
    int b, a, rc;
    int num_adc = 0;
    int num_exc = 0;
    int num_ref = 0;
    int i_adc = 0;
    int i_exc = 0;
    int i_ref = 0;
    pulseq_raw_block raw;

    /* Pass A: count, so the arrays are exact rather than grown. */
    for (b = plan->first_block; b <= plan->last_block; ++b)
    {
        if (!pulseq_get_raw_block_content_ids(seq, &raw, b, 0))
            return PULSEQ_ERR_FILE_READ_FAILED;
        if (raw.adc >= 0)
            num_adc++;
        if (raw.rf >= 0)
        {
            int role = rf_role(seq, raw.rf);
            /* Refused here, in the counting pass, so the failure comes before
             * a single allocation and the caller gets it at plan_create. */
            if (role == KTRAJ_RF_UNDECLARED)
                return PULSEQ_ERR_INVALID_ARGUMENT;
            if (role == KTRAJ_RF_EXCITATION)
                num_exc++;
            else if (role == KTRAJ_RF_REFOCUSING)
                num_ref++;
        }
    }

    plan->num_readouts = num_adc;
    plan->num_exc = num_exc;
    plan->num_ref = num_ref;

    if (num_adc > 0)
    {
        plan->readouts = (pulseq_ktraj_readout *)PULSEQ_ALLOC((size_t)num_adc *
                                                              sizeof(pulseq_ktraj_readout));
        plan->keys =
            (ktraj_repeat_key *)PULSEQ_ALLOC((size_t)num_adc * sizeof(ktraj_repeat_key));
        plan->adc_offset = (double *)PULSEQ_ALLOC((size_t)num_adc * sizeof(double));
        plan->adc_dwell = (double *)PULSEQ_ALLOC((size_t)num_adc * sizeof(double));
        if (!plan->readouts || !plan->keys || !plan->adc_offset || !plan->adc_dwell)
            return PULSEQ_ERR_ALLOC_FAILED;
        memset(plan->readouts, 0, (size_t)num_adc * sizeof(pulseq_ktraj_readout));
        memset(plan->keys, 0, (size_t)num_adc * sizeof(ktraj_repeat_key));
    }
    if (num_exc > 0)
    {
        plan->exc = (pulseq_ktraj_rf *)PULSEQ_ALLOC((size_t)num_exc * sizeof(pulseq_ktraj_rf));
        if (!plan->exc)
            return PULSEQ_ERR_ALLOC_FAILED;
        memset(plan->exc, 0, (size_t)num_exc * sizeof(pulseq_ktraj_rf));
    }
    if (num_ref > 0)
    {
        plan->ref = (pulseq_ktraj_rf *)PULSEQ_ALLOC((size_t)num_ref * sizeof(pulseq_ktraj_rf));
        if (!plan->ref)
            return PULSEQ_ERR_ALLOC_FAILED;
        memset(plan->ref, 0, (size_t)num_ref * sizeof(pulseq_ktraj_rf));
    }

    plan->num_range_blocks = plan->last_block - plan->first_block + 1;
    plan->origin = (double *)PULSEQ_ALLOC((size_t)plan->num_range_blocks * 3 * sizeof(double));
    plan->t_block = (double *)PULSEQ_ALLOC((size_t)plan->num_range_blocks * sizeof(double));
    if (!plan->origin || !plan->t_block)
        return PULSEQ_ERR_ALLOC_FAILED;

    /* Pass B: the walk itself. */
    k[0] = 0.0;
    k[1] = 0.0;
    k[2] = 0.0;

    for (b = plan->first_block; b <= plan->last_block; ++b)
    {
        const ktraj_wave *wave[3];
        int grad_id[3];
        int slot = b - plan->first_block;
        int role = KTRAJ_RF_NEUTRAL;
        int rotation_index = -1;
        double t_center = 0.0;
        double duration;

        if (!pulseq_get_raw_block_content_ids(seq, &raw, b, 1))
            return PULSEQ_ERR_FILE_READ_FAILED;

        plan->t_block[slot] = t;
        for (a = 0; a < 3; ++a)
            plan->origin[slot * 3 + a] = k[a];

        grad_id[0] = raw.gx;
        grad_id[1] = raw.gy;
        grad_id[2] = raw.gz;
        for (a = 0; a < 3; ++a)
            wave[a] = wave_for(plan, grad_id[a]);

        duration = (double)raw.block_duration * plan->block_raster;

        if (raw.rf >= 0 && raw.rf < seq->rf_library_size && seq->rf_library)
        {
            const PULSEQ_REAL *rf = seq->rf_library[raw.rf];
            role = rf_role(seq, raw.rf);
            t_center = ((double)rf[KTRAJ_RF_DELAY] + (double)rf[KTRAJ_RF_CENTER]) * KTRAJ_US;

            if (role != KTRAJ_RF_NEUTRAL)
            {
                pulseq_ktraj_rf *slotp =
                    (role == KTRAJ_RF_EXCITATION) ? &plan->exc[i_exc] : &plan->ref[i_ref];
                double freq = (double)rf[KTRAJ_RF_FREQ] +
                              (double)rf[KTRAJ_RF_FREQ_PPM] * 1.0e-6 * gb0;
                double phase = (double)rf[KTRAJ_RF_PHASE] +
                               (double)rf[KTRAJ_RF_PHASE_PPM] * 1.0e-6 * gb0;

                slotp->block_index = b + 1;
                slotp->t = (PULSEQ_REAL)(t + t_center);
                slotp->freq_offset = (PULSEQ_REAL)freq;
                /* PyPulseq references the phase to the pulse centre. */
                slotp->phase_offset =
                    (PULSEQ_REAL)(phase + 2.0 * 3.14159265358979323846 * freq *
                                              ((double)rf[KTRAJ_RF_CENTER] * KTRAJ_US));
                for (a = 0; a < 3; ++a)
                    slotp->g[a] = (PULSEQ_REAL)(wave[a] ? wave[a]->amplitude *
                                                              wave_value_at(wave[a], t_center)
                                                        : 0.0);
                if (role == KTRAJ_RF_EXCITATION)
                    i_exc++;
                else
                    i_ref++;
            }
        }

        if (raw.adc >= 0 && raw.adc < seq->adc_library_size && seq->adc_library)
        {
            const PULSEQ_REAL *adc = seq->adc_library[raw.adc];
            pulseq_ktraj_readout *ro = &plan->readouts[i_adc];
            ktraj_repeat_key *key = &plan->keys[i_adc];
            double dwell = (double)adc[KTRAJ_ADC_DWELL] * KTRAJ_NS;
            double delay = (double)adc[KTRAJ_ADC_DELAY] * KTRAJ_US;
            int nsamples = (int)adc[KTRAJ_ADC_NUM];
            double win0, win1;

            rotation_index = block_rotation_id(plan, &raw);

            /* Sample centres, which is where Pulseq puts them. */
            win0 = delay + 0.5 * dwell;
            win1 = delay + ((double)nsamples - 0.5) * dwell;

            plan->adc_offset[i_adc] = win0;
            plan->adc_dwell[i_adc] = dwell;

            ro->block_index = b + 1;
            ro->num_samples = nsamples;
            ro->t0 = (PULSEQ_REAL)(t + win0);
            ro->dwell = (PULSEQ_REAL)dwell;
            ro->center_sample = -1;
            ro->rotation_id = rotation_index;
            for (a = 0; a < 3; ++a)
            {
                ro->k0[a] = (PULSEQ_REAL)k[a];
                ro->g_echo[a] = (PULSEQ_REAL)0.0;
                ro->k_echo[a] = (PULSEQ_REAL)0.0;
            }

            /*
             * Constant axes, in the frame k is reported in.
             *
             * The distinction matters and it is the radial case.  A radial
             * spoke is a *flat* readout gradient on one logical axis plus a
             * rotation extension per shot: logically it is as constant as a
             * Cartesian line, and physically it sweeps a line at an arbitrary
             * angle across two or three axes.  Reporting the logical answer
             * beside a rotated k would tell a consumer that a spoke needs no
             * trajectory -- the "constant gradient, so the recon indexes it
             * with a counter and a frequency offset" shortcut -- and radial
             * has no such counter to index with.
             *
             * A rotated axis is constant only if every logical axis feeding
             * it is.  Cancellation between two moving axes could in principle
             * leave a rotated one still, but that is a coincidence rather
             * than a design, and the error is in the safe direction: samples
             * stored that need not have been.
             *
             * With apply_rotation off, k is logical and so is this -- the two
             * always describe the same frame.
             */
            {
                const double *m = plan->opts.apply_rotation
                                      ? rotation_for(plan, rotation_index)
                                      : NULL;
                int flat[3];
                for (a = 0; a < 3; ++a)
                    flat[a] = (!wave[a] || wave_is_flat(wave[a], win0, win1)) ? 1 : 0;

                ro->constant_axes = 0u;
                for (a = 0; a < 3; ++a)
                {
                    int constant = 1;
                    if (m)
                    {
                        int s;
                        for (s = 0; s < 3; ++s)
                        {
                            if (fabs(m[a * 3 + s]) > 1.0e-12 && !flat[s])
                                constant = 0;
                        }
                    }
                    else
                    {
                        constant = flat[a];
                    }
                    if (constant)
                        ro->constant_axes |= (unsigned int)(1u << a);
                }
            }

            /* Canonical ids, so two spellings of the same readout share a
             * key.  The waveform behind a canonical id is identical to the
             * one behind the id it replaced, so lookups stay valid. */
            for (a = 0; a < 3; ++a)
            {
                key->grad[a] = (plan->grad_canon && grad_id[a] >= 0 &&
                                grad_id[a] < seq->grad_library_size)
                                   ? plan->grad_canon[grad_id[a]]
                                   : grad_id[a];
            }
            key->adc = (plan->adc_canon && raw.adc >= 0 && raw.adc < seq->adc_library_size)
                           ? plan->adc_canon[raw.adc]
                           : raw.adc;
            key->rotation = rotation_index;
            i_adc++;
        }
        else
        {
            /* Nothing else needs the rotation: it applies to the block's own
             * gradients, and an origin is only ever read in the logical frame. */
            rotation_index = -1;
        }
        (void)rotation_index;

        /* Accumulate this block's moment into k. */
        if (role != KTRAJ_RF_NEUTRAL)
        {
            for (a = 0; a < 3; ++a)
            {
                double before, mid;
                if (!wave[a])
                {
                    k[a] = (role == KTRAJ_RF_EXCITATION) ? 0.0 : -k[a];
                    continue;
                }
                before = wave_cumulative_at(wave[a], t_center);
                mid = k[a] + wave[a]->amplitude * before;
                mid = (role == KTRAJ_RF_EXCITATION) ? 0.0 : -mid;
                k[a] = mid + wave[a]->amplitude * (wave[a]->total - before);
            }
        }
        else
        {
            for (a = 0; a < 3; ++a)
            {
                if (wave[a])
                    k[a] += wave[a]->amplitude * wave[a]->total;
            }
        }

        /* A background gradient runs for the whole block regardless of what
         * the block drives, so it is added here rather than folded into a
         * wave that may not exist. */
        for (a = 0; a < 3; ++a)
        {
            if (plan->opts.gradient_offset[a] != (PULSEQ_REAL)0.0)
                k[a] += (double)plan->opts.gradient_offset[a] * duration;
        }

        t += duration;
    }

    plan->total_duration = t;
    rc = PULSEQ_SUCCESS;
    return rc;
}

/* ------------------------------------------------------------------ */

/*
 * Absolute k for one readout, axis-major, into out_k.
 *
 * The gradient time base carries `trajectory_delay`: sampling the cumulative
 * at (t + delay) is what shifts the gradient relative to the ADC without
 * moving the ADC, which is the distinction the option exists to make.
 */
static void readout_samples(const pulseq_ktraj_plan *plan, int index, double *out_k,
                            int include_origin)
{
    const pulseq_ktraj_readout *ro = &plan->readouts[index];
    const ktraj_repeat_key *key = &plan->keys[index];
    const ktraj_wave *wave[3];
    int grad_id[3];
    int n = ro->num_samples;
    int a, i;
    double win_start;

    for (a = 0; a < 3; ++a)
    {
        grad_id[a] = key->grad[a];
        wave[a] = wave_for(plan, grad_id[a]);
    }

    /* Time of the first sample centre, relative to the block start. */
    win_start = plan->adc_offset[index];

    for (a = 0; a < 3; ++a)
    {
        double origin = include_origin ? (double)ro->k0[a] : 0.0;
        double amplitude = wave[a] ? wave[a]->amplitude : 0.0;
        double shift = (double)plan->opts.trajectory_delay[a];
        double offset = (double)plan->opts.gradient_offset[a];
        double *dst = out_k + (size_t)a * (size_t)n;

        for (i = 0; i < n; ++i)
        {
            double tt = win_start + (double)i * plan->adc_dwell[index];
            double v = origin;
            if (wave[a])
            {
                v += amplitude *
                     (plan->opts.sample_window_average
                          ? wave_mean_cumulative(wave[a], tt - shift, plan->adc_dwell[index])
                          : wave_cumulative_at(wave[a], tt - shift));
            }
            if (offset != 0.0)
                v += offset * tt;
            dst[i] = v;
        }
    }

    if (plan->opts.apply_rotation)
    {
        const double *m = rotation_for(plan, key->rotation);
        if (m)
        {
            for (i = 0; i < n; ++i)
            {
                double x = out_k[0 * n + i];
                double y = out_k[1 * n + i];
                double z = out_k[2 * n + i];
                out_k[0 * n + i] = m[0] * x + m[1] * y + m[2] * z;
                out_k[1 * n + i] = m[3] * x + m[4] * y + m[5] * z;
                out_k[2 * n + i] = m[6] * x + m[7] * y + m[8] * z;
            }
        }
    }
}

/*
 * Absolute k at one sample of one readout, in the output frame.
 *
 * The same arithmetic readout_samples() does, for a single j -- so a caller
 * that wants one number does not pay for the whole readout.
 */
static void readout_sample_k(const pulseq_ktraj_plan *plan, int index, int sample,
                             double out_k[3])
{
    const pulseq_ktraj_readout *ro = &plan->readouts[index];
    const ktraj_repeat_key *key = &plan->keys[index];
    double tt = plan->adc_offset[index] + (double)sample * plan->adc_dwell[index];
    int a;

    for (a = 0; a < 3; ++a)
    {
        const ktraj_wave *w = wave_for(plan, key->grad[a]);
        double v = (double)ro->k0[a];
        if (w)
        {
            double at = tt - (double)plan->opts.trajectory_delay[a];
            v += w->amplitude * (plan->opts.sample_window_average
                                     ? wave_mean_cumulative(w, at, plan->adc_dwell[index])
                                     : wave_cumulative_at(w, at));
        }
        if (plan->opts.gradient_offset[a] != (PULSEQ_REAL)0.0)
            v += (double)plan->opts.gradient_offset[a] * tt;
        out_k[a] = v;
    }

    if (plan->opts.apply_rotation)
    {
        const double *m = rotation_for(plan, key->rotation);
        if (m)
        {
            double x = out_k[0], y = out_k[1], z = out_k[2];
            out_k[0] = m[0] * x + m[1] * y + m[2] * z;
            out_k[1] = m[3] * x + m[4] * y + m[5] * z;
            out_k[2] = m[6] * x + m[7] * y + m[8] * z;
        }
    }
}

/* The gradient vector at the given time within a readout's block, Hz/m. */
static void readout_gradient_at(const pulseq_ktraj_plan *plan, int index, double t_in_block,
                                double out_g[3])
{
    const ktraj_repeat_key *key = &plan->keys[index];
    int grad_id[3];
    int a;

    for (a = 0; a < 3; ++a)
        grad_id[a] = key->grad[a];

    for (a = 0; a < 3; ++a)
    {
        const ktraj_wave *w = wave_for(plan, grad_id[a]);
        double shift = (double)plan->opts.trajectory_delay[a];
        out_g[a] = w ? w->amplitude * wave_value_at(w, t_in_block - shift) : 0.0;
        out_g[a] += (double)plan->opts.gradient_offset[a];
    }

    if (plan->opts.apply_rotation)
    {
        const double *m = rotation_for(plan, key->rotation);
        if (m)
        {
            double x = out_g[0];
            double y = out_g[1];
            double z = out_g[2];
            out_g[0] = m[0] * x + m[1] * y + m[2] * z;
            out_g[1] = m[3] * x + m[4] * y + m[5] * z;
            out_g[2] = m[6] * x + m[7] * y + m[8] * z;
        }
    }
}

/* ------------------------------------------------------------------ */

static int keys_equal(const ktraj_repeat_key *a, const ktraj_repeat_key *b)
{
    return a->grad[0] == b->grad[0] && a->grad[1] == b->grad[1] && a->grad[2] == b->grad[2] &&
           a->adc == b->adc && a->rotation == b->rotation && a->swept == b->swept &&
           a->k0v[0] == b->k0v[0] && a->k0v[1] == b->k0v[1] && a->k0v[2] == b->k0v[2];
}

static unsigned long key_hash(const ktraj_repeat_key *k)
{
    /*
     * FNV-1a over the integer fields and over the raw bytes of the doubles.
     *
     * Hashing the bytes is sound only because the comparison is exact too:
     * equal doubles have equal bytes, apart from the one pair that does not
     * -- +0.0 and -0.0 -- which is normalised below.  A NaN would hash fine
     * and never compare equal, costing a missed hit and nothing else.
     */
    unsigned long h = 2166136261UL;
    int i, b;

    {
        unsigned long parts[6];
        parts[0] = (unsigned long)(k->grad[0] + 1);
        parts[1] = (unsigned long)(k->grad[1] + 1);
        parts[2] = (unsigned long)(k->grad[2] + 1);
        parts[3] = (unsigned long)(k->adc + 1);
        parts[4] = (unsigned long)(k->rotation + 1);
        parts[5] = (unsigned long)k->swept;
        for (i = 0; i < 6; ++i)
        {
            h ^= parts[i];
            h *= 16777619UL;
        }
    }

    for (i = 0; i < 3; ++i)
    {
        double v = k->k0v[i];
        unsigned char bytes[sizeof(double)];
        if (v == 0.0)
            v = 0.0; /* collapses -0.0, which compares equal but differs in bits */
        memcpy(bytes, &v, sizeof(double));
        for (b = 0; b < (int)sizeof(double); ++b)
        {
            h ^= (unsigned long)bytes[b];
            h *= 16777619UL;
        }
    }
    return h;
}

/*
 * For each readout, the index of the first readout sharing its repeat key.
 *
 * This is the whole speed argument made concrete: without it the "have I seen
 * this readout before" question is a scan over everything already seen, and a
 * million-readout scan pays a million-squared comparisons -- slower than the
 * per-sample work it exists to avoid.  Open addressing with linear probing,
 * sized to twice the readout count and rounded up to a power of two, so the
 * table never exceeds half load and probes stay short.
 *
 * Returns a caller-freed array of length num_readouts.
 */
static int build_repeat_memo(const ktraj_repeat_key *keys, int n, int **out_first)
{
    int capacity = 16;
    int *table;
    int *first;
    int i;
    unsigned long mask;

    *out_first = NULL;
    if (n <= 0)
        return PULSEQ_SUCCESS;

    while (capacity < n * 2)
    {
        if (capacity > (1 << 29))
            break;
        capacity <<= 1;
    }
    mask = (unsigned long)(capacity - 1);

    table = (int *)PULSEQ_ALLOC((size_t)capacity * sizeof(int));
    first = (int *)PULSEQ_ALLOC((size_t)n * sizeof(int));
    if (!table || !first)
    {
        PULSEQ_FREE(table);
        PULSEQ_FREE(first);
        return PULSEQ_ERR_ALLOC_FAILED;
    }
    for (i = 0; i < capacity; ++i)
        table[i] = -1;

    for (i = 0; i < n; ++i)
    {
        unsigned long slot = key_hash(&keys[i]) & mask;
        first[i] = i;
        for (;;)
        {
            int occupant = table[slot];
            if (occupant < 0)
            {
                table[slot] = i;
                break;
            }
            if (keys_equal(&keys[i], &keys[occupant]))
            {
                first[i] = occupant;
                break;
            }
            slot = (slot + 1) & mask;
        }
    }

    PULSEQ_FREE(table);
    *out_first = first;
    return PULSEQ_SUCCESS;
}

/*
 * Derive every readout's center_sample.
 *
 * ### Why the repeat key drops part of the origin
 *
 * The obvious key -- gradient ids, ADC, rotation and the whole entry k -- is
 * correct and almost never fires.  Measured on the fixtures, it hit 0% of
 * readouts everywhere, because in a Cartesian scan every line enters with a
 * different phase-encode prewinder and so no two entry vectors agree.  A memo
 * that never hits is not an optimisation.
 *
 * The origin does not have to be in the key in full.  Writing p for the
 * rotated entry k, v(j) for the rotated sweep the readout adds, and c for the
 * scan's k centre:
 *
 *     |p + v(j) - c|^2  =  sum_a (p_a + v_a(j) - c_a)^2
 *
 * and for an axis where v_a(j) does not change with j, that term is the same
 * for every sample -- it shifts the whole curve and cannot move the minimum.
 * So the answer depends on p only through the axes the readout actually
 * sweeps.  For a Cartesian readout that is the readout axis alone, and its
 * prewinder is identical on every line, so all lines share a key and the memo
 * fires on essentially all of them.  That is the entire speed argument, and
 * it is exact, not an approximation.
 *
 * ### The two sweeps
 *
 * The same split makes the global k centre cheap.  Sweep 1 needs
 * min over readouts and samples of |p + v(j)|^2, which separates:
 *
 *     |p + v(j)|^2  =  sum_{a static} (p_a + v_a)^2   +   sum_{a swept} (p_a + v_a(j))^2
 *                      \___ per readout, O(1) ___/       \___ per key group, O(n) ___/
 *
 * so each group pays one scan over its samples and each readout pays a
 * constant.  Sweep 2 memoizes on the key directly, per the argument above.
 */

/* What a group of readouts sharing a repeat key has in common. */
typedef struct ktraj_group
{
    int rep;              /* representative readout index                  */
    unsigned int swept;   /* axes whose v(j) actually changes              */
    double var_min;       /* min_j over swept axes of |p_a + v_a(j)|       */
    int var_argmin;       /* the j attaining it                            */
    double tie_tol;       /* 1% of this readout's own sampling step        */
    /*
     * Whether k moves forwards through this readout: the sign of the
     * dominant component of (last sample - first sample).  It is the same
     * quantity a REV label carries, and it is what turns "resolve a tie
     * towards increasing k" into an index -- the later sample going forwards,
     * the earlier one going backwards.  See the tie rule in derive_centers().
     */
    int forward;
    int center;           /* sweep-2 answer: index, or -1 for no echo      */
} ktraj_group;

/*
 * Mean distance between consecutive samples of a readout, over the swept axes.
 *
 * The scale every "is this a tie" question in derive_centers() is asked
 * against.  A readout's own step is the only meaningful unit here: k is in
 * 1/m and the field of view is the sequence's business, so an absolute
 * tolerance would mean something different for every scan.
 */
static double sample_step(const double *buf, int n, unsigned int swept)
{
    double total = 0.0;
    int steps = 0;
    int j, a;

    for (j = 1; j < n; ++j)
    {
        double sq = 0.0;
        for (a = 0; a < 3; ++a)
        {
            double dv;
            if (!(swept & (unsigned int)(1u << a)))
                continue;
            dv = buf[a * n + j] - buf[a * n + (j - 1)];
            sq += dv * dv;
        }
        total += sqrt(sq);
        steps++;
    }
    return (steps > 0) ? total / (double)steps : 0.0;
}

/* Whether k advances along the readout: sign of its dominant displacement. */
static int sweeps_forward(const double *buf, int n, unsigned int swept)
{
    double best = 0.0;
    int a;

    if (n < 2)
        return 1;
    for (a = 0; a < 3; ++a)
    {
        double d;
        if (!(swept & (unsigned int)(1u << a)))
            continue;
        d = buf[a * n + (n - 1)] - buf[a * n];
        if (fabs(d) > fabs(best))
            best = d;
    }
    return (best >= 0.0) ? 1 : 0;
}

/* The rotated entry k of a readout -- p in the comment above. */
static void readout_origin(const pulseq_ktraj_plan *plan, int index, double p[3])
{
    const pulseq_ktraj_readout *ro = &plan->readouts[index];
    int a;

    for (a = 0; a < 3; ++a)
        p[a] = (double)ro->k0[a];

    if (plan->opts.apply_rotation)
    {
        const double *m = rotation_for(plan, plan->keys[index].rotation);
        if (m)
        {
            double x = p[0], y = p[1], z = p[2];
            p[0] = m[0] * x + m[1] * y + m[2] * z;
            p[1] = m[3] * x + m[4] * y + m[5] * z;
            p[2] = m[6] * x + m[7] * y + m[8] * z;
        }
    }
}

/*
 * Which axes a readout sweeps, from the origin-free samples.
 *
 * Judged against the largest range across the three axes rather than against
 * an absolute number, because k is in 1/m and a sequence's scale is its own
 * business.  A rotation smears a nominally idle axis by rounding alone, and
 * the relative threshold is what keeps that from reading as a sweep.
 */
static unsigned int swept_axes(const double *v, int n)
{
    double lo[3], hi[3], widest = 0.0;
    unsigned int mask = 0u;
    int a, j;

    for (a = 0; a < 3; ++a)
    {
        lo[a] = v[a * n];
        hi[a] = v[a * n];
        for (j = 1; j < n; ++j)
        {
            if (v[a * n + j] < lo[a])
                lo[a] = v[a * n + j];
            if (v[a * n + j] > hi[a])
                hi[a] = v[a * n + j];
        }
        if (hi[a] - lo[a] > widest)
            widest = hi[a] - lo[a];
    }

    for (a = 0; a < 3; ++a)
    {
        if (hi[a] - lo[a] > 1.0e-9 * widest && hi[a] > lo[a])
            mask |= (unsigned int)(1u << a);
    }
    return mask;
}

static int derive_centers(pulseq_ktraj_plan *plan)
{
    int n_ro = plan->num_readouts;
    int i, a, j, rc, longest = 0;
    double *buf = NULL;
    double *origins = NULL;
    int *shape_first = NULL;
    int *key_first = NULL;
    ktraj_repeat_key *mkeys = NULL;
    ktraj_group *groups = NULL;
    double best_total = 0.0;
    int best_readout = -1;
    int have_center = 0;

    if (n_ro <= 0)
        return PULSEQ_SUCCESS;

    for (i = 0; i < n_ro; ++i)
    {
        if (plan->readouts[i].num_samples > longest)
            longest = plan->readouts[i].num_samples;
    }
    if (longest <= 0)
        return PULSEQ_SUCCESS;

    buf = (double *)PULSEQ_ALLOC((size_t)longest * 3 * sizeof(double));
    origins = (double *)PULSEQ_ALLOC((size_t)n_ro * 3 * sizeof(double));
    groups = (ktraj_group *)PULSEQ_ALLOC((size_t)n_ro * sizeof(ktraj_group));
    mkeys = (ktraj_repeat_key *)PULSEQ_ALLOC((size_t)n_ro * sizeof(ktraj_repeat_key));
    if (!buf || !origins || !groups || !mkeys)
    {
        rc = PULSEQ_ERR_ALLOC_FAILED;
        goto done;
    }

    for (i = 0; i < n_ro; ++i)
        readout_origin(plan, i, &origins[i * 3]);

    /* ---- Phase A: group by shape alone, to learn which axes are swept.
     *
     * This has to come first: the swept set decides which parts of the origin
     * belong in the full key, and it is a property of the gradients and the
     * rotation only, so one representative per shape answers for all of them.
     */
    for (i = 0; i < n_ro; ++i)
    {
        mkeys[i] = plan->keys[i];
        mkeys[i].k0v[0] = mkeys[i].k0v[1] = mkeys[i].k0v[2] = 0.0;
    }

    rc = build_repeat_memo(mkeys, n_ro, &shape_first);
    if (PULSEQ_FAILED(rc))
        goto done;

    for (i = 0; i < n_ro; ++i)
    {
        int nn = plan->readouts[i].num_samples;
        if (shape_first[i] != i || nn <= 0)
            continue;
        readout_samples(plan, i, buf, 0);
        groups[i].swept = swept_axes(buf, nn);
    }

    /* ---- Phase B: the full key -- shape plus the origin on swept axes only.
     *
     * Exactly, not rounded; see the comment on ktraj_repeat_key for the tie
     * this used to get wrong.  Axes the readout does not sweep are left at
     * zero, which is what lets every phase-encode line of a Cartesian scan
     * share one key.
     */
    for (i = 0; i < n_ro; ++i)
    {
        unsigned int swept = groups[shape_first[i]].swept;
        mkeys[i] = plan->keys[i];
        for (a = 0; a < 3; ++a)
        {
            if (swept & (unsigned int)(1u << a))
            {
                mkeys[i].k0v[a] = origins[i * 3 + a];
            }
            else
            {
                /* Contributes the same constant to every sample, so neither
                 * the gradient nor the entry point can change the answer. */
                mkeys[i].k0v[a] = 0.0;
                mkeys[i].grad[a] = -1;
            }
        }
        mkeys[i].swept = swept;
    }

    rc = build_repeat_memo(mkeys, n_ro, &key_first);
    if (PULSEQ_FAILED(rc))
        goto done;

    /* ---- Phase C: per group, the swept-axis minimum. */
    for (i = 0; i < n_ro; ++i)
    {
        int nn = plan->readouts[i].num_samples;
        unsigned int swept;

        if (key_first[i] != i || nn <= 0)
            continue;
        swept = mkeys[i].swept;
        groups[i].swept = swept;

        readout_samples(plan, i, buf, 1);
        groups[i].tie_tol = 1.0e-2 * sample_step(buf, nn, swept);
        groups[i].forward = sweeps_forward(buf, nn, swept);
        groups[i].var_min = 0.0;
        groups[i].var_argmin = 0;

        /*
         * Distances, and a tie resolves along the direction of travel -- the
         * same rule sweep 2 applies, and it has to be the same rule.  A
         * symmetric readout's two central samples are exactly equidistant from
         * the origin, so a strict comparison here would take the earlier one
         * while sweep 2 took the later, and the whole scan's echo index would
         * come out one low.  Measured on bssfp_2d_1sl_1avg, whose readout is
         * symmetric to the last bit: every readout reported 31 of 64 rather
         * than 32.  The sequences where it happened to agree only did so
         * because six printed digits left the two sides unequal.
         */
        for (j = 0; j < nn; ++j)
        {
            double d = 0.0;
            for (a = 0; a < 3; ++a)
            {
                if (swept & (unsigned int)(1u << a))
                    d += buf[a * nn + j] * buf[a * nn + j];
            }
            d = sqrt(d);

            if (j == 0 || d < groups[i].var_min - groups[i].tie_tol)
            {
                groups[i].var_min = d;
                groups[i].var_argmin = j;
            }
            else if (d <= groups[i].var_min + groups[i].tie_tol)
            {
                if (d < groups[i].var_min)
                    groups[i].var_min = d;
                if (groups[i].forward)
                    groups[i].var_argmin = j;
            }
        }
    }

    /* ---- Sweep 1: the scan's closest approach to k = 0.
     *
     * min over samples of |k(j)|^2 splits into the swept axes, which every
     * member of a key group shares, and the static ones, which they do not --
     * a group now spans readouts whose static-axis gradients differ, since
     * those were dropped from the key.  The static part is read off sample 0,
     * where by definition it stays for the whole readout.
     */
    /*
     * A tie here is between whole readouts, and it is resolved in favour of a
     * **forward** one.
     *
     * Which polarity wins no longer decides anyone's echo index -- the tie
     * rule in phase C is geometric, so both polarities of an EPI already agree
     * that the centre lies at the same *point* in k, and each finds it at
     * whichever of its own samples that is.  What this still decides is which
     * readout is reported as `central_readout`, and therefore which one
     * `kSpaceCenterLine` is read off.  A forward one is the readout whose
     * sample indices need no mirroring, so it is the honest thing to quote.
     */
    for (i = 0; i < n_ro; ++i)
    {
        int g = key_first[i];
        int nn = plan->readouts[i].num_samples;
        double first_sample[3];
        double total;
        double tol;

        if (nn <= 0)
            continue;

        readout_sample_k(plan, i, 0, first_sample);
        total = groups[g].var_min * groups[g].var_min;
        for (a = 0; a < 3; ++a)
        {
            if (!(groups[g].swept & (unsigned int)(1u << a)))
                total += first_sample[a] * first_sample[a];
        }
        total = sqrt(total);

        if (!have_center)
        {
            best_total = total;
            best_readout = i;
            have_center = 1;
            continue;
        }

        tol = groups[g].tie_tol;
        if (tol < groups[key_first[best_readout]].tie_tol)
            tol = groups[key_first[best_readout]].tie_tol;

        if (total < best_total - tol)
        {
            best_total = total;
            best_readout = i;
        }
        else if (total <= best_total + tol)
        {
            if (total < best_total)
                best_total = total;
            if (groups[g].forward && !groups[key_first[best_readout]].forward)
                best_readout = i;
        }
    }

    if (best_readout >= 0)
    {
        int g = key_first[best_readout];
        readout_sample_k(plan, best_readout, groups[g].var_argmin, plan->k_center);
        plan->central_readout = best_readout;
    }

    /* ---- Sweep 2: each readout's sample closest to that point. */
    /* -2 = not computed yet, -1 = computed and there is no echo. Two
     * distinct facts, so they get two distinct values. */
    for (i = 0; i < n_ro; ++i)
        groups[i].center = -2;

    for (i = 0; i < n_ro; ++i)
    {
        int g = key_first[i];
        int nn = plan->readouts[i].num_samples;
        double g_echo[3];

        if (nn <= 0)
            continue;

        if (groups[g].center == -2)
        {
            int best = 0;
            double best_d = 0.0;
            double tie_tol = 0.0;
            unsigned int swept = groups[g].swept;

            /*
             * A readout whose k does not move has no echo -- every sample is
             * the same distance from the centre, so there is no sample that
             * is "the" crossing.  A noise scan and a free-induction readout
             * are both like this.
             *
             * It stays -1 rather than becoming 0 or num_samples/2.  Inventing
             * one here would be indistinguishable from a derived answer at
             * every layer above, which is the failure mode this whole file
             * exists to remove: pulseg_structure.c:4108 still falls back to
             * `num_samples / 2` whenever its k-space analysis did not run,
             * and a wrong echo index is a silently mis-reconstructed image.
             */
            if (swept == 0u)
            {
                groups[g].center = -1;
                plan->readouts[i].center_sample = -1;
                continue;
            }

            readout_samples(plan, g, buf, 1);

            /*
             * The tie tolerance, as a fraction of the readout's own step,
             * worked out once per group in phase C.
             *
             * A fully sampled symmetric readout straddles the origin: its two
             * central samples sit at -dk/2 and +dk/2 and are the same distance
             * from it.  Which of them wins is then decided by whatever
             * asymmetry survives the file's six printed digits -- in
             * gre_2d_1sl_1avg the margin is 2e-4 out of a step of 4.5, or
             * 0.005%.  Letting that decide would make the echo index a
             * property of a rounding error.
             *
             * So anything inside this tolerance counts as a tie, and a tie
             * resolves towards **increasing k along the direction of travel**:
             * the later sample on a forward readout, the earlier one on a
             * reverse one.
             *
             * Stated as an index rule it would be two different rules; stated
             * geometrically it is one, and it is the one that survives an EPI.
             * A bipolar train's two polarities are mirror images, so their
             * nearest samples are *different points*: with N = 128 the forward
             * echo is sample 64 and the reverse echo is sample 63, and a
             * reconstruction that mirrors the reverse lines -- which is what
             * the REV label asks it to do -- finds both at
             * 128 - 1 - 63 = 64.  Taking the higher index on both would put
             * the reverse echo at 64 before mirroring and 63 after, so half
             * the lines would be reconstructed one sample off centre.
             *
             * For a unidirectional scan nothing changes: every readout is
             * forward and N samples spanning the origin put DC at N/2, not
             * N/2 - 1.  The asymmetric cases this whole file exists for --
             * partial Fourier, a shifted echo -- are nowhere near a tie and
             * are unaffected either way.
             */
            tie_tol = groups[g].tie_tol;

            for (j = 0; j < nn; ++j)
            {
                double d = 0.0;
                for (a = 0; a < 3; ++a)
                {
                    double v;
                    /* Axes the readout does not sweep shift every sample by
                     * the same amount, so they cannot move the minimum. */
                    if (!(swept & (unsigned int)(1u << a)))
                        continue;
                    v = buf[a * nn + j] - plan->k_center[a];
                    d += v * v;
                }
                d = sqrt(d); /* distances, so the tolerance means something */

                if (j == 0 || d < best_d - tie_tol)
                {
                    best_d = d;
                    best = j;
                }
                else if (d <= best_d + tie_tol)
                {
                    /* Tied with the incumbent: take the sample further along
                     * the direction of travel, which is the later one going
                     * forwards and the incumbent going backwards.  Keep the
                     * smaller distance either way, so a long near-flat run
                     * cannot walk the answer along one tolerance at a time. */
                    if (d < best_d)
                        best_d = d;
                    if (groups[g].forward)
                        best = j;
                }
            }
            groups[g].center = best;
        }

        plan->readouts[i].center_sample = groups[g].center;

        /* No echo, no gradient at the echo and no k at it -- both stay zero
         * rather than being sampled at a made-up time. */
        if (groups[g].center >= 0)
        {
            double t_in_block = plan->adc_offset[i] +
                                (double)groups[g].center * plan->adc_dwell[i];
            double k_echo[3];
            readout_gradient_at(plan, i, t_in_block, g_echo);
            /* Per readout, not per group: members of a group share a shape and
             * their swept-axis entry point, but the axes dropped from the key
             * are exactly the ones whose constant offset differs -- which is
             * the whole phase-encode table. */
            readout_sample_k(plan, i, groups[g].center, k_echo);
            for (a = 0; a < 3; ++a)
            {
                plan->readouts[i].g_echo[a] = (PULSEQ_REAL)g_echo[a];
                plan->readouts[i].k_echo[a] = (PULSEQ_REAL)k_echo[a];
            }
        }
    }

    /* The hit rate is worth having: a scan that should repeat and does not is
     * a wrong key, not a slow run, and the caller cannot see that otherwise. */
    plan->num_key_groups = 0;
    for (i = 0; i < n_ro; ++i)
    {
        if (key_first[i] == i)
            plan->num_key_groups++;
    }

    rc = PULSEQ_SUCCESS;

done:
    PULSEQ_FREE(mkeys);
    PULSEQ_FREE(groups);
    PULSEQ_FREE(shape_first);
    PULSEQ_FREE(key_first);
    PULSEQ_FREE(origins);
    PULSEQ_FREE(buf);
    return rc;
}
/* ================================================================== */
/*  Public interface                                                  */
/* ================================================================== */

void pulseq_ktraj_opts_init(pulseq_ktraj_opts *opts)
{
    int a;
    if (!opts)
        return;
    for (a = 0; a < 3; ++a)
    {
        opts->trajectory_delay[a] = (PULSEQ_REAL)0.0;
        opts->gradient_offset[a] = (PULSEQ_REAL)0.0;
    }
    opts->first_block = 1;
    opts->last_block = 0;
    opts->apply_rotation = 1;
    opts->sample_window_average = 0;
    opts->derive_center_sample = 1;
    opts->tr_period_blocks = 0;
}

int pulseq_ktraj_plan_create(pulseq_ktraj_plan **out, const pulseq_file *seq,
                             const pulseq_ktraj_opts *opts)
{
    pulseq_ktraj_plan *plan;
    pulseq_ktraj_opts defaults;
    int rc;

    if (!out || !seq)
        return PULSEQ_ERR_NULL_POINTER;
    *out = NULL;
    if (seq->num_blocks <= 0 || !seq->block_library)
        return PULSEQ_ERR_EMPTY;

    if (!opts)
    {
        pulseq_ktraj_opts_init(&defaults);
        opts = &defaults;
    }

    plan = (pulseq_ktraj_plan *)PULSEQ_ALLOC(sizeof(pulseq_ktraj_plan));
    if (!plan)
        return PULSEQ_ERR_ALLOC_FAILED;
    memset(plan, 0, sizeof(*plan));

    plan->seq = seq;
    plan->opts = *opts;
    plan->central_readout = -1;

    plan->first_block = (opts->first_block > 0) ? opts->first_block - 1 : 0;
    plan->last_block = (opts->last_block > 0) ? opts->last_block - 1 : seq->num_blocks - 1;
    if (plan->last_block >= seq->num_blocks)
        plan->last_block = seq->num_blocks - 1;
    if (plan->first_block > plan->last_block)
    {
        PULSEQ_FREE(plan);
        return PULSEQ_ERR_INVALID_ARGUMENT;
    }

    plan->grad_raster = raster_seconds(seq->reserved_definitions_library.gradient_raster_time,
                                       KTRAJ_DEFAULT_GRAD_RASTER_US);
    plan->block_raster = raster_seconds(seq->reserved_definitions_library.block_duration_raster,
                                        KTRAJ_DEFAULT_BLOCK_RASTER_US);

    rc = build_rotations(plan);
    if (PULSEQ_FAILED(rc))
    {
        pulseq_ktraj_plan_free(plan);
        return rc;
    }

    {
        /* Every column of a gradient row matters; for an ADC only the three
         * that decide when the samples land. */
        static const int grad_columns[7] = {0, 1, 2, 3, 4, 5, 6};
        static const int adc_columns[3] = {KTRAJ_ADC_NUM, KTRAJ_ADC_DWELL, KTRAJ_ADC_DELAY};

        rc = canonicalise_rows((const PULSEQ_REAL *)seq->grad_library, seq->grad_library_size,
                               7, grad_columns, 7, &plan->grad_canon);
        if (PULSEQ_FAILED(rc))
        {
            pulseq_ktraj_plan_free(plan);
            return rc;
        }
        rc = canonicalise_rows((const PULSEQ_REAL *)seq->adc_library, seq->adc_library_size, 8,
                               adc_columns, 3, &plan->adc_canon);
        if (PULSEQ_FAILED(rc))
        {
            pulseq_ktraj_plan_free(plan);
            return rc;
        }
    }

    rc = build_waves(plan);
    if (PULSEQ_FAILED(rc))
    {
        pulseq_ktraj_plan_free(plan);
        return rc;
    }

    rc = plan_walk_blocks(plan);
    if (PULSEQ_FAILED(rc))
    {
        pulseq_ktraj_plan_free(plan);
        return rc;
    }

    if (plan->opts.derive_center_sample)
    {
        rc = derive_centers(plan);
        if (PULSEQ_FAILED(rc))
        {
            pulseq_ktraj_plan_free(plan);
            return rc;
        }
    }

    *out = plan;
    return PULSEQ_SUCCESS;
}

void pulseq_ktraj_plan_free(pulseq_ktraj_plan *plan)
{
    int i;
    if (!plan)
        return;
    if (plan->waves)
    {
        for (i = 0; i < plan->num_waves; ++i)
            wave_free(&plan->waves[i]);
        PULSEQ_FREE(plan->waves);
    }
    PULSEQ_FREE(plan->rot);
    PULSEQ_FREE(plan->grad_canon);
    PULSEQ_FREE(plan->adc_canon);
    PULSEQ_FREE(plan->origin);
    PULSEQ_FREE(plan->t_block);
    PULSEQ_FREE(plan->readouts);
    PULSEQ_FREE(plan->keys);
    PULSEQ_FREE(plan->adc_offset);
    PULSEQ_FREE(plan->adc_dwell);
    PULSEQ_FREE(plan->exc);
    PULSEQ_FREE(plan->ref);
    PULSEQ_FREE(plan);
}

int pulseq_ktraj_num_readouts(const pulseq_ktraj_plan *plan)
{
    return plan ? plan->num_readouts : 0;
}

int pulseq_ktraj_readout_info(const pulseq_ktraj_plan *plan, int index,
                              pulseq_ktraj_readout *out)
{
    if (!plan || !out)
        return PULSEQ_ERR_NULL_POINTER;
    if (index < 0 || index >= plan->num_readouts)
        return PULSEQ_ERR_INVALID_ARGUMENT;
    *out = plan->readouts[index];
    return PULSEQ_SUCCESS;
}

int pulseq_ktraj_readout_k(const pulseq_ktraj_plan *plan, int index, PULSEQ_REAL *out_k)
{
    double *buf;
    int n, i;

    if (!plan || !out_k)
        return PULSEQ_ERR_NULL_POINTER;
    if (index < 0 || index >= plan->num_readouts)
        return PULSEQ_ERR_INVALID_ARGUMENT;

    n = plan->readouts[index].num_samples;
    if (n <= 0)
        return PULSEQ_SUCCESS;

    buf = (double *)PULSEQ_ALLOC((size_t)n * 3 * sizeof(double));
    if (!buf)
        return PULSEQ_ERR_ALLOC_FAILED;

    readout_samples(plan, index, buf, 1);
    for (i = 0; i < n * 3; ++i)
        out_k[i] = (PULSEQ_REAL)buf[i];

    PULSEQ_FREE(buf);
    return PULSEQ_SUCCESS;
}

int pulseq_ktraj_num_excitations(const pulseq_ktraj_plan *plan)
{
    return plan ? plan->num_exc : 0;
}

int pulseq_ktraj_excitation_info(const pulseq_ktraj_plan *plan, int index,
                                 pulseq_ktraj_rf *out)
{
    if (!plan || !out)
        return PULSEQ_ERR_NULL_POINTER;
    if (index < 0 || index >= plan->num_exc)
        return PULSEQ_ERR_INVALID_ARGUMENT;
    *out = plan->exc[index];
    return PULSEQ_SUCCESS;
}

int pulseq_ktraj_num_refocusings(const pulseq_ktraj_plan *plan)
{
    return plan ? plan->num_ref : 0;
}

int pulseq_ktraj_refocusing_info(const pulseq_ktraj_plan *plan, int index,
                                 pulseq_ktraj_rf *out)
{
    if (!plan || !out)
        return PULSEQ_ERR_NULL_POINTER;
    if (index < 0 || index >= plan->num_ref)
        return PULSEQ_ERR_INVALID_ARGUMENT;
    *out = plan->ref[index];
    return PULSEQ_SUCCESS;
}

int pulseq_ktraj_block_origin(const pulseq_ktraj_plan *plan, int block_index,
                              PULSEQ_REAL out_k[3])
{
    int slot, a;

    if (!plan || !out_k)
        return PULSEQ_ERR_NULL_POINTER;
    slot = block_index - 1 - plan->first_block;
    if (slot < 0 || slot >= plan->num_range_blocks)
        return PULSEQ_ERR_INVALID_ARGUMENT;
    for (a = 0; a < 3; ++a)
        out_k[a] = (PULSEQ_REAL)plan->origin[slot * 3 + a];
    return PULSEQ_SUCCESS;
}

PULSEQ_REAL pulseq_ktraj_total_duration(const pulseq_ktraj_plan *plan)
{
    return plan ? (PULSEQ_REAL)plan->total_duration : (PULSEQ_REAL)0.0;
}

int pulseq_ktraj_num_key_groups(const pulseq_ktraj_plan *plan)
{
    return plan ? plan->num_key_groups : 0;
}

int pulseq_ktraj_center(const pulseq_ktraj_plan *plan, PULSEQ_REAL out_k[3])
{
    int a;
    if (!plan || !out_k)
        return PULSEQ_ERR_NULL_POINTER;
    for (a = 0; a < 3; ++a)
        out_k[a] = (PULSEQ_REAL)plan->k_center[a];
    return PULSEQ_SUCCESS;
}

int pulseq_ktraj_central_readout(const pulseq_ktraj_plan *plan)
{
    return plan ? plan->central_readout : -1;
}

/* The 0-based block containing time @p t, by binary search on t_block. */
static int block_at_time(const pulseq_ktraj_plan *plan, double t)
{
    int lo = 0;
    int hi = plan->num_range_blocks - 1;

    if (t <= plan->t_block[0])
        return 0;
    if (t >= plan->t_block[hi])
        return hi;

    while (hi - lo > 1)
    {
        int mid = lo + (hi - lo) / 2;
        if (plan->t_block[mid] <= t)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

int pulseq_ktraj_at_times(const pulseq_ktraj_plan *plan, const PULSEQ_REAL *times, int count,
                          PULSEQ_REAL *out_k)
{
    int i, a;

    if (!plan || !times || !out_k)
        return PULSEQ_ERR_NULL_POINTER;
    if (count < 0)
        return PULSEQ_ERR_INVALID_ARGUMENT;

    for (i = 0; i < count; ++i)
    {
        double t = (double)times[i];
        int slot, b;
        double local, k[3];
        pulseq_raw_block raw;

        if (t < 0.0)
            t = 0.0;
        if (t > plan->total_duration)
            t = plan->total_duration;

        slot = block_at_time(plan, t);
        b = plan->first_block + slot;
        local = t - plan->t_block[slot];

        if (!pulseq_get_raw_block_content_ids(plan->seq, &raw, b, 1))
            return PULSEQ_ERR_FILE_READ_FAILED;

        for (a = 0; a < 3; ++a)
        {
            int grad_id[3];
            const ktraj_wave *w;
            grad_id[0] = raw.gx;
            grad_id[1] = raw.gy;
            grad_id[2] = raw.gz;
            w = wave_for(plan, grad_id[a]);
            k[a] = plan->origin[slot * 3 + a];
            if (w)
                k[a] += w->amplitude *
                        wave_cumulative_at(w, local - (double)plan->opts.trajectory_delay[a]);
            if (plan->opts.gradient_offset[a] != (PULSEQ_REAL)0.0)
                k[a] += (double)plan->opts.gradient_offset[a] * local;
        }

        if (plan->opts.apply_rotation)
        {
            const double *m = rotation_for(plan, block_rotation_id(plan, &raw));
            if (m)
            {
                double x = k[0], y = k[1], z = k[2];
                k[0] = m[0] * x + m[1] * y + m[2] * z;
                k[1] = m[3] * x + m[4] * y + m[5] * z;
                k[2] = m[6] * x + m[7] * y + m[8] * z;
            }
        }

        for (a = 0; a < 3; ++a)
            out_k[(size_t)a * (size_t)count + (size_t)i] = (PULSEQ_REAL)k[a];
    }
    return PULSEQ_SUCCESS;
}

static int compare_times(const void *a, const void *b)
{
    double x = *(const double *)a;
    double y = *(const double *)b;
    if (x < y)
        return -1;
    if (x > y)
        return 1;
    return 0;
}

int pulseq_ktraj_breakpoints(const pulseq_ktraj_plan *plan, PULSEQ_REAL *out_t, int max)
{
    double *all = NULL;
    int capacity = 0;
    int n = 0;
    int slot, a, i, unique;
    double tol;

    if (!plan)
        return PULSEQ_ERR_NULL_POINTER;

    /* Upper bound: every block contributes its own start plus each axis's
     * breakpoints, and the sequence contributes its end. */
    capacity = plan->num_range_blocks + 1;
    for (slot = 0; slot < plan->num_range_blocks; ++slot)
    {
        pulseq_raw_block raw;
        int grad_id[3];
        if (!pulseq_get_raw_block_content_ids(plan->seq, &raw, plan->first_block + slot, 0))
            return PULSEQ_ERR_FILE_READ_FAILED;
        grad_id[0] = raw.gx;
        grad_id[1] = raw.gy;
        grad_id[2] = raw.gz;
        for (a = 0; a < 3; ++a)
        {
            const ktraj_wave *w = wave_for(plan, grad_id[a]);
            if (w)
                capacity += w->n;
        }
    }

    all = (double *)PULSEQ_ALLOC((size_t)capacity * sizeof(double));
    if (!all)
        return PULSEQ_ERR_ALLOC_FAILED;

    for (slot = 0; slot < plan->num_range_blocks; ++slot)
    {
        pulseq_raw_block raw;
        int grad_id[3];
        double base = plan->t_block[slot];

        all[n++] = base;
        if (!pulseq_get_raw_block_content_ids(plan->seq, &raw, plan->first_block + slot, 0))
        {
            PULSEQ_FREE(all);
            return PULSEQ_ERR_FILE_READ_FAILED;
        }
        grad_id[0] = raw.gx;
        grad_id[1] = raw.gy;
        grad_id[2] = raw.gz;
        for (a = 0; a < 3; ++a)
        {
            const ktraj_wave *w = wave_for(plan, grad_id[a]);
            if (!w)
                continue;
            for (i = 0; i < w->n; ++i)
                all[n++] = base + w->t[i];
        }
    }
    all[n++] = plan->total_duration;

    qsort(all, (size_t)n, sizeof(double), compare_times);

    /* Merge points closer than a tenth of a nanosecond.  Pulseq's own
     * calculateKspacePP rounds to the same 0.1 ns before its unique(), for
     * the same reason: two breakpoints that differ only by the last bit of a
     * printed decimal are one breakpoint. */
    tol = 1.0e-10;
    unique = 0;
    for (i = 0; i < n; ++i)
    {
        if (unique == 0 || all[i] - all[unique - 1] > tol)
            all[unique++] = all[i];
    }

    if (out_t)
    {
        int copy = (unique < max) ? unique : max;
        for (i = 0; i < copy; ++i)
            out_t[i] = (PULSEQ_REAL)all[i];
    }

    PULSEQ_FREE(all);
    return unique;
}

void pulseq_ktraj_free(pulseq_ktraj *traj)
{
    if (!traj)
        return;
    PULSEQ_FREE(traj->readouts);
    PULSEQ_FREE(traj->k);
    PULSEQ_FREE(traj->sample_offset);
    PULSEQ_FREE(traj->excitations);
    PULSEQ_FREE(traj->refocusings);
    memset(traj, 0, sizeof(*traj));
}

int pulseq_ktraj_all(const pulseq_file *seq, const pulseq_ktraj_opts *opts, pulseq_ktraj *out)
{
    pulseq_ktraj_plan *plan = NULL;
    double *buf = NULL;
    int rc, i, a, total, longest;

    if (!out)
        return PULSEQ_ERR_NULL_POINTER;
    memset(out, 0, sizeof(*out));

    rc = pulseq_ktraj_plan_create(&plan, seq, opts);
    if (PULSEQ_FAILED(rc))
        return rc;

    total = 0;
    longest = 0;
    for (i = 0; i < plan->num_readouts; ++i)
    {
        int n = plan->readouts[i].num_samples;
        if (n > 0)
            total += n;
        if (n > longest)
            longest = n;
    }

    out->num_readouts = plan->num_readouts;
    out->total_samples = total;
    out->num_excitations = plan->num_exc;
    out->num_refocusings = plan->num_ref;
    out->total_duration = (PULSEQ_REAL)plan->total_duration;
    for (a = 0; a < 3; ++a)
        out->k_center[a] = (PULSEQ_REAL)plan->k_center[a];

    if (plan->num_readouts > 0)
    {
        out->readouts = (pulseq_ktraj_readout *)PULSEQ_ALLOC(
            (size_t)plan->num_readouts * sizeof(pulseq_ktraj_readout));
        out->sample_offset = (int *)PULSEQ_ALLOC((size_t)plan->num_readouts * sizeof(int));
        if (!out->readouts || !out->sample_offset)
        {
            rc = PULSEQ_ERR_ALLOC_FAILED;
            goto fail;
        }
        memcpy(out->readouts, plan->readouts,
               (size_t)plan->num_readouts * sizeof(pulseq_ktraj_readout));
    }
    if (total > 0)
    {
        out->k = (PULSEQ_REAL *)PULSEQ_ALLOC((size_t)total * 3 * sizeof(PULSEQ_REAL));
        buf = (double *)PULSEQ_ALLOC((size_t)longest * 3 * sizeof(double));
        if (!out->k || !buf)
        {
            rc = PULSEQ_ERR_ALLOC_FAILED;
            goto fail;
        }
    }
    if (plan->num_exc > 0)
    {
        out->excitations =
            (pulseq_ktraj_rf *)PULSEQ_ALLOC((size_t)plan->num_exc * sizeof(pulseq_ktraj_rf));
        if (!out->excitations)
        {
            rc = PULSEQ_ERR_ALLOC_FAILED;
            goto fail;
        }
        memcpy(out->excitations, plan->exc, (size_t)plan->num_exc * sizeof(pulseq_ktraj_rf));
    }
    if (plan->num_ref > 0)
    {
        out->refocusings =
            (pulseq_ktraj_rf *)PULSEQ_ALLOC((size_t)plan->num_ref * sizeof(pulseq_ktraj_rf));
        if (!out->refocusings)
        {
            rc = PULSEQ_ERR_ALLOC_FAILED;
            goto fail;
        }
        memcpy(out->refocusings, plan->ref, (size_t)plan->num_ref * sizeof(pulseq_ktraj_rf));
    }

    total = 0;
    for (i = 0; i < plan->num_readouts; ++i)
    {
        int n = plan->readouts[i].num_samples;
        int j;
        out->sample_offset[i] = total;
        if (n <= 0)
            continue;
        readout_samples(plan, i, buf, 1);
        for (j = 0; j < n * 3; ++j)
            out->k[(size_t)total * 3 + (size_t)j] = (PULSEQ_REAL)buf[j];
        total += n;
    }

    PULSEQ_FREE(buf);
    pulseq_ktraj_plan_free(plan);
    return PULSEQ_SUCCESS;

fail:
    PULSEQ_FREE(buf);
    pulseq_ktraj_free(out);
    pulseq_ktraj_plan_free(plan);
    return rc;
}
