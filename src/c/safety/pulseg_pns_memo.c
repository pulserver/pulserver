/**
 * @file pulseg_pns_memo.c
 * @brief Memoized PNS evaluation for linear (convolutional) nerve models.
 *
 * The straightforward way to evaluate a PNS model is to materialise dG/dt
 * over the whole canonical TR and convolve it with the model kernel. That
 * cost grows with the TR, which on a high-resolution 3D protocol means a
 * multi-second window and millions of samples -- even though the window is
 * built from a handful of gradient shapes repeated at different amplitudes.
 *
 * When the model declares itself linear (pulseg_pns_model::kernel), the
 * response can be assembled instead of convolved. Writing the window as a
 * sum of zero-extended, amplitude-scaled, time-shifted unit shapes
 *
 *     G(t) = sum_i  a_i * g_{k(i)}(t - t_i)
 *
 * and using linearity of both differentiation and convolution,
 *
 *     (k * dG/dt)(t) = sum_i  a_i * (k * dg_{k(i)}/dt)(t - t_i)
 *
 * so each distinct shape is convolved once and every occurrence becomes a
 * scaled add. Cost stops scaling with the TR length and starts scaling with
 * (number of distinct shapes) + (occurrences x kernel length).
 *
 * Two properties make this safe to use on a safety gate:
 *
 *  - The templates are *slices of the very waveform the exact path would
 *    convolve*, not a re-rendering of it, so there is no second copy of the
 *    gradient renderer that could drift away from the first.
 *  - Templates are keyed on pulseg__wave_key, the identity the representation
 *    already carries: the definition fixes the timing skeleton, the shape id
 *    the samples, the duration the span. This path works in the logical
 *    frame, so no rotation participates and a key match is a scalar multiple
 *    by construction. Building with PULSEG_DEBUG_MEMO asserts that against
 *    the materialised samples.
 *
 * Exactness. The window is partitioned into per-block slices, and each
 * slice's contribution to the forward difference is taken zero-extended on
 * both sides: it spans len+1 samples starting one sample *before* the block,
 * opening with w[first] (the step up from zero) and closing with
 * -w[last] (the step back down). Summing those, the seam between two blocks
 * receives (-w_end) + (w_start), which is the same pair of floats the direct
 * forward difference w_start - w_end subtracts -- so the decomposition is
 * exact in floating point, not merely in real arithmetic. The one place this
 * path is *not* bit-identical to convolving the whole waveform is amplitude
 * factorisation: a_i * (template sample) versus the stored (a_i * template)
 * sample, which moves the rounding by up to an ulp per sample.
 */

#include <limits.h>
#include <math.h>

#include "pulseg_internal.h"

/* An occurrence is only accepted as "template scaled by r" if every sample
 * matches to this relative tolerance. Comfortably looser than the ulp-level
 * agreement expected of genuinely proportional slices, comfortably tighter
 * than any real difference in shape. */
#define PNS_MEMO_PROPORTIONAL_RTOL 1e-5f

/* A floor below which no window is worth analysing at all. */
#define PNS_MEMO_MIN_SAMPLES 512

/* Only take the memoized path when it is expected to beat the transform by
 * at least this factor, so a window with little repetition -- where the
 * assembly would be no cheaper -- keeps the simpler exact route. */
#define PNS_MEMO_MIN_SPEEDUP 4

/* Rough operation count of the exact route: three axes, three real
 * transforms each (signal, kernel, inverse), each ~n*log2(nfft) butterflies.
 * Only the ratio against the assembly cost matters, so the constant just has
 * to be in the right ballpark.
 *
 * Counted as a double rather than a wide integer: the library builds as C89,
 * where `long long` does not exist and `long` is only 32 bits on the scanner's
 * PPC target -- too narrow, since 9*n*bits passes two billion on a window of a
 * few million samples. A double holds every integer up to 2^53 exactly, which
 * is far beyond any count this can produce, and both operands of the
 * comparison this feeds are magnitudes rather than exact quantities. */
static double pns_memo_transform_ops(int n, int kernel_len)
{
    double nfft = 1.0;
    double target = (double)n + (double)kernel_len - 1.0;
    int bits = 0;

    while (nfft < target)
    {
        nfft *= 2.0;
        bits++;
    }
    if (bits < 1)
        bits = 1;
    return 9.0 * (double)n * (double)bits;
}

typedef struct
{
    pulseg__wave_key key; /* what makes two slices the same waveform */
    int axis;             /* 0 = x, 1 = y, 2 = z: which output this response feeds */
    int first;            /* block position whose slice defines the shape */
    int pivot;            /* index within that slice of its largest-magnitude sample */
    float pivot_v;        /* that sample's value: the shape's amplitude handle */
    float *resp;          /* [resp_len] kernel convolved with the slice's dG/dt */
    int resp_len;
} pns_memo_template;

typedef struct
{
    int tmpl;   /* index into templates[] */
    int offset; /* uniform sample index of the slice's dG/dt, i.e. block start - 1 */
    float scale;
} pns_memo_occurrence;

/* ================================================================== */
/*  Block geometry                                                    */
/* ================================================================== */

/* Uniform-sample offset and length of every block in the window. Returns 0 if
 * any block boundary misses a whole uniform sample, in which case a template
 * placed there would carry a sub-sample phase and memoization is invalid.
 *
 * The extraction produces duration/raster + 1 samples, so the blocks tile
 * duration/raster of them and the window's final sample is an endpoint no
 * block owns; it is folded into the last block, whose slice is one longer. */
static int pns_memo_block_offsets(
    const pulseg_sequence_descriptor *desc,
    int block_start,
    int block_count,
    const int *block_order,
    float raster_us,
    int num_uniform_samples,
    int *offsets,
    int *lengths)
{
    int n, block_idx, dur_us, raster_halves, dur_halves, t_samples;
    const pulseg_block_table_element *bte;
    const pulseg_base_block *bdef;

    /* Exact integer arithmetic throughout: block durations are integers, and a
     * float accumulator loses microsecond resolution once the window runs
     * into the millions of microseconds a 3D shot spans. The raster is
     * counted in half-microseconds so the half-gradient-raster grid the
     * extraction uses stays an integer too.
     *
     * The running position is accumulated in *samples*, not microseconds. Once
     * a block's duration is known to be a whole number of samples, the sum of
     * those lengths is the same offset that dividing a running microsecond
     * total would give -- and it stays inside an int, because the loop rejects
     * any window whose samples overrun the extraction. That is what lets this
     * hold to C89, where `long long` does not exist and the 32-bit `long` of
     * the scanner's PPC target would overflow on a long window. It also drops
     * a modulo per block: a running total of on-grid durations is on-grid, so
     * the old check against it could never fire on its own. */
    raster_halves = (int)(raster_us * 2.0f + 0.5f);
    if (raster_halves <= 0 || (float)fabs((double)raster_us * 2.0 - raster_halves) > 1e-4)
        return 0;

    t_samples = 0;
    for (n = 0; n < block_count; ++n)
    {
        block_idx = block_order ? block_order[n] : block_start + n;
        bte = &desc->block_table[block_idx];
        bdef = &desc->base_blocks[bte->id];
        dur_us = (bte->duration_us >= 0) ? bte->duration_us : bdef->duration_us;
        /* The upper bound keeps the doubling below in range; a block that long
         * could not fit the window the caller is offering anyway. */
        if (dur_us <= 0 || dur_us > INT_MAX / 2)
            return 0;

        dur_halves = dur_us * 2;
        if (dur_halves % raster_halves != 0)
            return 0;
        offsets[n] = t_samples;
        lengths[n] = dur_halves / raster_halves;
        /* Lengths are strictly positive, so the running total only grows: once
         * it overruns the window it can never come back to the exact total the
         * final check demands, and stopping here bounds the accumulator. */
        t_samples += lengths[n];
        if (t_samples > num_uniform_samples)
            return 0;
    }

    if (offsets[block_count - 1] + lengths[block_count - 1] != num_uniform_samples - 1)
        return 0;
    lengths[block_count - 1] += 1; /* absorb the endpoint sample */

    return 1;
}

/* ================================================================== */
/*  Slice helpers                                                     */
/* ================================================================== */

/* Doubly zero-extended forward difference of one block's slice: len+1
 * samples, opening with the step up into the block and closing with the step
 * back down out of it. Written to land at absolute index (offset - 1), which
 * is where the first of those steps belongs in the full dG/dt.
 *
 * Mirrors compute_slew_rate()'s arithmetic term for term -- (diff * inv_g)
 * / dt_s, not a single fused scale -- so the two round identically. */
static void pns_memo_slice_dgdt(
    float *dgdt_out,
    const float *waveform,
    int offset,
    int len,
    float inv_gamma,
    float dt_s)
{
    int j;

    dgdt_out[0] = (waveform[offset] * inv_gamma) / dt_s;
    for (j = 1; j < len; ++j)
        dgdt_out[j] = ((waveform[offset + j] - waveform[offset + j - 1]) * inv_gamma) / dt_s;
    dgdt_out[len] = ((0.0f - waveform[offset + len - 1]) * inv_gamma) / dt_s;
}

/* Largest-magnitude sample of a slice; -1 if the slice is all zero. */
static int pns_memo_pivot(const float *waveform, int offset, int len)
{
    int j, best;
    float mag, best_mag;

    best = -1;
    best_mag = 0.0f;
    for (j = 0; j < len; ++j)
    {
        mag = (float)fabs(waveform[offset + j]);
        if (mag > best_mag)
        {
            best_mag = mag;
            best = j;
        }
    }
    return best;
}

#ifdef PULSEG_DEBUG_MEMO
/* Assert that a key match really is proportional. The key already implies it
 * -- same definition, same shape, same span, logical frame, whole-sample
 * offsets -- so this is a net for a build under test, not a hot-path guard.
 * It compares against the same materialised waveform the exact path would
 * have convolved. */
static int pns_memo_proportional(
    const float *waveform,
    int offset,
    int ref_offset,
    int len,
    float scale,
    float ref_peak_abs)
{
    int j;
    float expect, diff, tol;

    tol = PNS_MEMO_PROPORTIONAL_RTOL * ref_peak_abs * (float)fabs(scale);
    if (tol <= 0.0f)
        return 0;
    for (j = 0; j < len; ++j)
    {
        expect = waveform[ref_offset + j] * scale;
        diff = waveform[offset + j] - expect;
        if ((float)fabs(diff) > tol)
            return 0;
    }
    return 1;
}
#endif /* PULSEG_DEBUG_MEMO */

/* Time-domain, deliberately: a template is one block long against a kernel
 * spanning tens of chronaxie times, so a transform of the padded template
 * would be mostly padding -- and a direct linear convolution has no
 * wraparound to guard against in the first place. */
static void pns_memo_convolve_direct(
    float *out,
    const float *sig,
    int sig_len,
    const float *kernel,
    int kernel_len)
{
    int i, j, out_len;
    float s;

    out_len = sig_len + kernel_len - 1;
    for (i = 0; i < out_len; ++i)
        out[i] = 0.0f;

    for (i = 0; i < sig_len; ++i)
    {
        s = sig[i];
        if (s == 0.0f)
            continue;
        for (j = 0; j < kernel_len; ++j)
            out[i + j] += s * kernel[j];
    }
}

/* Add `scale` times a template response into one axis, clipping to the
 * output window. `offset` may be -1: the very first block's step-up lands
 * one sample before the window, exactly where the exact path has no sample
 * either. */
static void pns_memo_accumulate(
    float *dst,
    int n,
    const float *src,
    int src_len,
    int offset,
    float scale)
{
    int j, j0, lim;

    j0 = (offset < 0) ? -offset : 0;
    lim = src_len;
    if (offset + lim > n)
        lim = n - offset;
    for (j = j0; j < lim; ++j)
        dst[offset + j] += scale * src[j];
}

static void pns_memo_free_templates(pns_memo_template *t, int count)
{
    int i;
    if (!t)
        return;
    for (i = 0; i < count; ++i)
        if (t[i].resp)
            PULSEG_FREE(t[i].resp);
    PULSEG_FREE(t);
}

/* ================================================================== */
/*  Entry point                                                       */
/* ================================================================== */

int pulseg__calc_pns_memoized(
    float *out_x,
    float *out_y,
    float *out_z,
    int n,
    int *applied,
    const pulseg_sequence_descriptor *desc,
    int block_start,
    int block_count,
    const int *block_order,
    const pulseg__uniform_grad_waveforms *uw,
    float gamma_hz_per_tesla,
    const float *kernel,
    int kernel_len,
    float out_scale,
    int pad)
{
    const float *axis_wave[3];
    float *axis_out[3];
    int *offsets;
    int *lengths;
    pns_memo_template *templates;
    pns_memo_occurrence *occurrences;
    float *scratch;
    int num_templates, num_occurrences, cap;
    int n_uniform, axis, i, t, block_idx, pivot;
    pulseg__wave_key key;
    int rc, max_len, len, off, rep;
    double assembly_ops;
    float inv_gamma, dt_s, scale, pivot_v;
    const pulseg_block_table_element *bte;

    *applied = 0;
    offsets = NULL;
    lengths = NULL;
    templates = NULL;
    occurrences = NULL;
    scratch = NULL;
    rc = PULSEG_SUCCESS;
    num_templates = 0;
    num_occurrences = 0;

    if (!out_x || !out_y || !out_z || !desc || !uw || !kernel || kernel_len <= 0 || !applied)
        return PULSEG_ERR_NULL_POINTER;

    n_uniform = uw->num_samples;
    if (n_uniform < PNS_MEMO_MIN_SAMPLES || block_count <= 0)
        return PULSEG_SUCCESS; /* not applicable; caller uses the exact path */
    if (n != n_uniform + pad - 1)
        return PULSEG_SUCCESS;
    if (!uw->gx || !uw->gy || !uw->gz || uw->raster_us <= 0.0f || gamma_hz_per_tesla <= 0.0f)
        return PULSEG_SUCCESS;

    axis_wave[0] = uw->gx;
    axis_wave[1] = uw->gy;
    axis_wave[2] = uw->gz;
    axis_out[0] = out_x;
    axis_out[1] = out_y;
    axis_out[2] = out_z;

    offsets = (int *)PULSEG_ALLOC((size_t)block_count * sizeof(int));
    lengths = (int *)PULSEG_ALLOC((size_t)block_count * sizeof(int));
    cap = block_count * 3;
    templates = (pns_memo_template *)PULSEG_ALLOC((size_t)cap * sizeof(*templates));
    occurrences = (pns_memo_occurrence *)PULSEG_ALLOC((size_t)cap * sizeof(*occurrences));
    if (!offsets || !lengths || !templates || !occurrences)
    {
        rc = PULSEG_ERR_ALLOC_FAILED;
        goto done;
    }
    for (i = 0; i < cap; ++i)
        templates[i].resp = NULL;

    if (!pns_memo_block_offsets(
            desc,
            block_start,
            block_count,
            block_order,
            uw->raster_us,
            n_uniform,
            offsets,
            lengths))
        goto done; /* off-grid block boundary: not applicable */

    inv_gamma = 1.0f / gamma_hz_per_tesla;
    dt_s = uw->raster_us * 1e-6f;
    max_len = 0;

    /* ---- pass 1: group block/axis slices into templates ---- */
    for (axis = 0; axis < 3; ++axis)
    {
        for (i = 0; i < block_count; ++i)
        {
            block_idx = block_order ? block_order[i] : block_start + i;
            bte = &desc->block_table[block_idx];
            pivot = pns_memo_pivot(axis_wave[axis], offsets[i], lengths[i]);
            if (pivot < 0)
                continue; /* silent slice: contributes nothing at all */
            pivot_v = axis_wave[axis][offsets[i] + pivot];

            /* The logical frame, so no rotation participates: see
             * pulseg__wave_key. */
            if (!pulseg__wave_key_axis(desc, bte, axis, 0, &key))
                goto done; /* non-silent slice with no definition: bail out */

            scale = 1.0f;
            for (t = 0; t < num_templates; ++t)
            {
                if (!pulseg__wave_key_equal(&templates[t].key, &key) || templates[t].axis != axis ||
                    lengths[templates[t].first] != lengths[i])
                    continue;
                scale = pivot_v / templates[t].pivot_v;
#ifdef PULSEG_DEBUG_MEMO
                if (!pns_memo_proportional(
                        axis_wave[axis],
                        offsets[i],
                        offsets[templates[t].first],
                        lengths[i],
                        scale,
                        (float)fabs(templates[t].pivot_v)))
                    continue;
#endif
                break;
            }

            if (t == num_templates)
            {
                templates[t].key = key;
                templates[t].axis = axis;
                templates[t].first = i;
                templates[t].pivot = pivot;
                templates[t].pivot_v = pivot_v;
                templates[t].resp_len = lengths[i] + kernel_len; /* (len+1) + K - 1 */
                if (lengths[i] + 1 > max_len)
                    max_len = lengths[i] + 1;
                num_templates++;
                scale = 1.0f;
            }

            occurrences[num_occurrences].tmpl = t;
            occurrences[num_occurrences].offset = offsets[i] - 1;
            occurrences[num_occurrences].scale = scale;
            num_occurrences++;
        }
    }

    /* Would the assembly actually be cheaper than the transform it replaces?
     * Every occurrence costs its template's response length in fused adds;
     * a window with little repetition has almost as many occurrences as it
     * has samples, and gains nothing. */
    if (num_templates == 0)
        goto done;
    assembly_ops = 0.0;
    for (i = 0; i < num_occurrences; ++i)
        assembly_ops += (double)templates[occurrences[i].tmpl].resp_len;
    /* The wrapped repetitions below place occurrences again, clipped at the
     * end of the output. Counting the placements is integer work against the
     * fused adds it is deciding about, so it is counted rather than guessed. */
    for (rep = n_uniform; rep < n; rep += n_uniform)
    {
        for (i = 0; i < num_occurrences; ++i)
        {
            off = occurrences[i].offset + rep;
            if (off >= n)
                continue;
            len = templates[occurrences[i].tmpl].resp_len;
            if (off + len > n)
                len = n - off;
            assembly_ops += (double)len;
        }
    }
    if (assembly_ops * PNS_MEMO_MIN_SPEEDUP > pns_memo_transform_ops(n, kernel_len))
        goto done;

    /* ---- pass 2: convolve each distinct shape once ---- */
    scratch = (float *)PULSEG_ALLOC((size_t)max_len * sizeof(float));
    if (!scratch)
    {
        rc = PULSEG_ERR_ALLOC_FAILED;
        goto done;
    }
    for (t = 0; t < num_templates; ++t)
    {
        len = lengths[templates[t].first];
        off = offsets[templates[t].first];
        pns_memo_slice_dgdt(scratch, axis_wave[templates[t].axis], off, len, inv_gamma, dt_s);

        templates[t].resp = (float *)PULSEG_ALLOC((size_t)templates[t].resp_len * sizeof(float));
        if (!templates[t].resp)
        {
            rc = PULSEG_ERR_ALLOC_FAILED;
            goto done;
        }
        pns_memo_convolve_direct(templates[t].resp, scratch, len + 1, kernel, kernel_len);
    }

    /* ---- pass 3: accumulate scaled, shifted copies ---- */
    for (axis = 0; axis < 3; ++axis)
        for (i = 0; i < n; ++i)
            axis_out[axis][i] = 0.0f;

    for (i = 0; i < num_occurrences; ++i)
    {
        const pns_memo_template *tp = &templates[occurrences[i].tmpl];
        pns_memo_accumulate(
            axis_out[tp->axis],
            n,
            tp->resp,
            tp->resp_len,
            occurrences[i].offset,
            occurrences[i].scale);
    }

    /* The exact path appends `pad` circularly wrapped samples so the filter is
     * warmed up where the window ends. Replaying the occurrences one window
     * later reproduces that tail -- and `pad` is set by the model's memory,
     * not by the window, so it routinely spans more than one window and the
     * replay has to repeat until it covers the output. Everything past the
     * output is clipped. */
    for (rep = n_uniform; rep < n; rep += n_uniform)
    {
        for (i = 0; i < num_occurrences; ++i)
        {
            const pns_memo_template *tp = &templates[occurrences[i].tmpl];
            off = occurrences[i].offset + rep;
            if (off >= n)
                continue;
            pns_memo_accumulate(
                axis_out[tp->axis],
                n,
                tp->resp,
                tp->resp_len,
                off,
                occurrences[i].scale);
        }
    }

    if (out_scale != 1.0f)
        for (axis = 0; axis < 3; ++axis)
            for (i = 0; i < n; ++i)
                axis_out[axis][i] *= out_scale;

    *applied = 1;

done:
    if (offsets)
        PULSEG_FREE(offsets);
    if (lengths)
        PULSEG_FREE(lengths);
    if (scratch)
        PULSEG_FREE(scratch);
    pns_memo_free_templates(templates, num_templates);
    if (occurrences)
        PULSEG_FREE(occurrences);
    return rc;
}
