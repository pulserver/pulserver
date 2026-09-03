/*
 * pulseg_pns_exact.c -- the stimulation check of a scan with no repetition
 * to stand for it.
 *
 * Past PULSEG__MAX_SHAPE_GROUPS the repetitions play more distinct sets of
 * gradient waveforms than a window per group could carry, so the scan is
 * evaluated as what it is: the response of every block placed on the scan's
 * own timeline. The nerve model is linear, so the response of the scan is
 * the sum of the responses of its blocks, each one exact for the block on
 * its own -- its interior slew through the kernel by one forward and one
 * inverse real FFT, the step it makes at its start against the previous
 * block's end priced as one kernel tap of that size -- rotated into the
 * physical frame and added into an accumulator that spans a chunk of
 * consecutive blocks plus the kernel's memory. The chunks are independent,
 * so they run as ranges under the caller's parallel hook, and the peak of
 * the root-sum-square over the chunks is the scan's peak: no bound, no
 * envelope, and the repetition that holds the peak is the worst case a
 * caller can ask to see.
 *
 * A model that publishes no kernel (SAFE) is not a convolution; for it the
 * same chunks are rendered and handed to the model's own evaluator, one
 * chunk after another.
 */

#include <math.h>
#include <string.h>

#include "external_kiss_fftr.h"
#include "pulseg_internal.h"
#include "pulseg_pns_models.h"

/* The judged peak of the FFT route is raised by this before the verdict:
 * the allowance for single-precision transforms, held by the raster-halving
 * invariant test. */
#define PNS_EXACT_FFT_MARGIN 2e-3
/* Blocks per chunk of the timeline. */
#define PNS_EXACT_CHUNK_BLOCKS 256
/* Distinct block lengths a range keeps FFT plans for at once. */
#define PNS_EXACT_MAX_PLANS 16
/* Memory assumed for a model that publishes no kernel: the blocks before a
 * chunk that are rendered with it so the chunk's first sample carries a
 * warmed-up history. */
#define PNS_EXACT_NO_KERNEL_REACH_US 50000.0

/* ================================================================== */
/*  One block's gradient, one axis, on the raster                     */
/* ================================================================== */

/* Corner-point scratch for the renderer, grown to the largest block met. */
typedef struct
{
    float *ct;
    float *cw;
    float *gt;
    float *gw;
    int corner_cap;
    int grid_cap;
} pns_render_scratch;

static void pns_render_scratch_free(pns_render_scratch *r)
{
    if (r->ct)
        PULSEG_FREE(r->ct);
    if (r->cw)
        PULSEG_FREE(r->cw);
    if (r->gt)
        PULSEG_FREE(r->gt);
    if (r->gw)
        PULSEG_FREE(r->gw);
    memset(r, 0, sizeof(*r));
}

/* The samples of one axis of a block at the raster, the way the window
 * path renders them: the definition's corner points at the given amplitude,
 * interpolated onto the raster from the first corner on -- half a gradient
 * raster into the block for a waveform sampled at cell centres, which
 * out_t0_us reports. The samples land in the scratch's grid buffer, valid
 * until the next call; NULL on failure. */
static const float *pns_render_uniform(
    pns_render_scratch *r,
    const pulseg_sequence_descriptor *desc,
    int def_id,
    int shape_id,
    float amplitude,
    float block_duration_us,
    float raster_us,
    int *out_n,
    float *out_t0_us)
{
    pulseg_grad_table_element gte;
    const pulseg_grad_definition *gdef;
    float amp, t_start, duration;
    int cap, n, n_out, i;

    if (def_id < 0 || def_id >= desc->num_unique_grads)
        return NULL;
    gdef = &desc->grad_definitions[def_id];
    gte.id = def_id;
    gte.shape_id = shape_id;
    gte.amplitude = amplitude;
    amp = amplitude;
    cap = pulseg__count_grad_samples_for_block(desc, gdef, block_duration_us) + 2;
    if (cap > r->corner_cap)
    {
        if (r->ct)
            PULSEG_FREE(r->ct);
        if (r->cw)
            PULSEG_FREE(r->cw);
        r->ct = (float *)PULSEG_ALLOC((size_t)cap * sizeof(float));
        r->cw = (float *)PULSEG_ALLOC((size_t)cap * sizeof(float));
        r->corner_cap = cap;
        if (!r->ct || !r->cw)
            return NULL;
    }
    n = pulseg__fill_grad_waveform_for_block(
        desc,
        r->ct,
        r->cw,
        0,
        gdef,
        &gte,
        0.0f,
        &amp,
        block_duration_us);
    if (n <= 0 || n > cap)
        return NULL;
    t_start = r->ct[0];
    duration = r->ct[n - 1] - t_start;
    if (duration <= 0.0f)
    {
        /* A single corner: one sample, as the window render gives it. */
        n_out = 1;
    }
    else
        n_out = (int)(duration / raster_us) + 1;
    if (n_out > r->grid_cap)
    {
        if (r->gt)
            PULSEG_FREE(r->gt);
        if (r->gw)
            PULSEG_FREE(r->gw);
        r->gt = (float *)PULSEG_ALLOC((size_t)n_out * sizeof(float));
        r->gw = (float *)PULSEG_ALLOC((size_t)n_out * sizeof(float));
        r->grid_cap = n_out;
        if (!r->gt || !r->gw)
            return NULL;
    }
    for (i = 0; i < n_out; ++i)
        r->gt[i] = t_start + (float)i * raster_us;
    if (n_out == 1)
        r->gw[0] = r->cw[0];
    else
        pulseg__interp1_linear(r->gw, r->gt, n_out, r->ct, r->cw, n);
    *out_n = n_out;
    *out_t0_us = t_start;
    return r->gw;
}

/* The slew inside a block: forward differences of its samples, both edge
 * taps zero. The steps at the block's start and end are priced by the
 * caller against the neighbouring blocks. */
static void pns_slice_dgdt(float *dgdt, const float *wave, int n, float raster_us, float gamma)
{
    double inv;
    int i;
    inv = 1.0 / ((double)gamma * ((double)raster_us * 1e-6));
    dgdt[0] = 0.0f;
    for (i = 1; i < n; ++i)
        dgdt[i] = (float)(((double)wave[i] - (double)wave[i - 1]) * inv);
    dgdt[n] = 0.0f;
}

typedef struct
{
    int has_grad[3];
    pulseg__wave_key key[3];
    float amplitude[3];
    int rotated; /* a non-identity rotation applies to this occurrence */
    const float *R;
} pns_occ;

static void pns_resolve_occ(
    const pulseg_sequence_descriptor *desc,
    const pulseg_block_table_element *bte,
    pns_occ *occ)
{
    int a, raw;
    occ->rotated = 0;
    occ->R = NULL;
    if (!bte->norot_flag && bte->rotation_id != -1 && bte->rotation_id < desc->num_rotations &&
        !pulseg__is_identity3(desc->rotation_matrices[bte->rotation_id]))
    {
        occ->rotated = 1;
        occ->R = desc->rotation_matrices[bte->rotation_id];
    }
    for (a = 0; a < 3; ++a)
    {
        occ->has_grad[a] = pulseg__wave_key_axis(desc, bte, a, 0, &occ->key[a]);
        occ->amplitude[a] = 0.0f;
        if (occ->has_grad[a])
        {
            raw = (a == 0) ? bte->gx_id : (a == 1) ? bte->gy_id : bte->gz_id;
            occ->amplitude[a] = desc->grad_table[raw].amplitude;
        }
    }
}

static double pns_block_duration_us(const pulseg_sequence_descriptor *desc, int b)
{
    const pulseg_block_table_element *bte = &desc->block_table[b];
    if (bte->duration_us >= 0)
        return (double)bte->duration_us;
    return (double)desc->base_blocks[bte->id].duration_us;
}

/* Block start and end times on the scan's timeline. */
static void pns_timeline(const pulseg_sequence_descriptor *desc, double *t_start, double *t_end)
{
    double t = 0.0;
    int b;
    for (b = 0; b < desc->num_blocks; ++b)
    {
        t_start[b] = t;
        t += pns_block_duration_us(desc, b);
        t_end[b] = t;
    }
}

/* Into the physical frame the way the window render turns its corner
 * points: the same helper, the same convention. */
static void pns_rotate3(const float *R, const double *v, double *out)
{
    float vin[3], vout[3];
    vin[0] = (float)v[0];
    vin[1] = (float)v[1];
    vin[2] = (float)v[2];
    pulseg__apply_rotation(vout, R, vin, 0);
    out[0] = (double)vout[0];
    out[1] = (double)vout[1];
    out[2] = (double)vout[2];
}

/* ================================================================== */
/*  Exact response peak of a block range, the model's own evaluator    */
/* ================================================================== */

int pulseg__pns_exact_range_peak(
    pulseg_check_plan *plan,
    pulseg_diagnostic *diag,
    const pulseg_sequence_descriptor *desc,
    int subseq_idx,
    int start,
    int count,
    double judge_from_us,
    double judge_until_us,
    const pulseg_pns_model *model,
    float gamma,
    double *out_peak,
    double *out_peak_time_us)
{
    const pulseg__uniform_grad_waveforms *uw;
    const float *wavep[3];
    float *dgdt[3], *resp[3];
    double peak, ss, inv, peak_time;
    int n, i, a, rc, last;

    *out_peak = -1.0;
    if (out_peak_time_us)
        *out_peak_time_us = 0.0;
    for (a = 0; a < 3; ++a)
    {
        dgdt[a] = NULL;
        resp[a] = NULL;
    }
    rc = pulseg__plan_waveforms(
        plan,
        &uw,
        diag,
        desc,
        subseq_idx,
        start,
        count,
        PULSEG_AMP_ACTUAL,
        NULL,
        0);
    if (PULSEG_FAILED(rc))
        return rc;
    n = uw->num_samples;
    if (n < 2 || uw->raster_us <= 0.0f)
        return PULSEG_ERR_INVALID_ARGUMENT;
    wavep[0] = uw->gx;
    wavep[1] = uw->gy;
    wavep[2] = uw->gz;
    inv = 1.0 / ((double)gamma * ((double)uw->raster_us * 1e-6));
    for (a = 0; a < 3; ++a)
    {
        dgdt[a] = (float *)PULSEG_ALLOC((size_t)(n + 1) * sizeof(float));
        resp[a] = (float *)PULSEG_ALLOC((size_t)(n + 1) * sizeof(float));
        if (!dgdt[a] || !resp[a])
        {
            rc = PULSEG_ERR_ALLOC_FAILED;
            goto done;
        }
        dgdt[a][0] = (float)((double)wavep[a][0] * inv);
        for (i = 1; i < n; ++i)
            dgdt[a][i] = (float)(((double)wavep[a][i] - (double)wavep[a][i - 1]) * inv);
        dgdt[a][n] = (float)(-(double)wavep[a][n - 1] * inv);
    }
    rc = model->evaluate(
        model->ctx,
        dgdt[0],
        dgdt[1],
        dgdt[2],
        n + 1,
        uw->raster_us,
        resp[0],
        resp[1],
        resp[2]);
    if (PULSEG_FAILED(rc))
        goto done;
    peak = 0.0;
    peak_time = 0.0;
    i = (int)(judge_from_us / (double)uw->raster_us);
    if (i < 0)
        i = 0;
    last = n + 1;
    if (judge_until_us > 0.0)
    {
        last = (int)(judge_until_us / (double)uw->raster_us);
        if (last > n + 1)
            last = n + 1;
    }
    for (; i < last; ++i)
    {
        ss = sqrt(
            (double)resp[0][i] * resp[0][i] + (double)resp[1][i] * resp[1][i] +
            (double)resp[2][i] * resp[2][i]);
        if (ss > peak)
        {
            peak = ss;
            peak_time = (double)i * (double)uw->raster_us;
        }
    }
    *out_peak = peak;
    if (out_peak_time_us)
        *out_peak_time_us = peak_time;
    rc = PULSEG_SUCCESS;
done:
    for (a = 0; a < 3; ++a)
    {
        if (dgdt[a])
            PULSEG_FREE(dgdt[a]);
        if (resp[a])
            PULSEG_FREE(resp[a]);
    }
    return rc;
}

/* ================================================================== */
/*  The exact scan: every block's response on the timeline             */
/* ================================================================== */

typedef struct
{
    int npts;
    int nfft;
    int nfreq;
    int out_len;
    kiss_fftr_cfg fwd;
    kiss_fftr_cfg inv;
    kiss_fft_cpx *kspec; /* the kernel's spectrum at this length */
} pns_exact_plan;

typedef struct
{
    const pulseg_sequence_descriptor *desc;
    const double *t_start_us;
    const double *t_end_us;
    const float *kernel;
    int kernel_len;
    float out_scale;
    float raster_us;
    float gamma;
    double reach_us;
    int chunk_blocks;
    int num_chunks;
    double *chunk_peak;         /* [num_chunks] */
    double *chunk_peak_time_us; /* [num_chunks] */
    int rc;
    int declined;
} pns_exact_scan_job;

/* Per-range scratch: the FFT plans of the block lengths met so far, and
 * buffers sized to the largest of them. */
typedef struct
{
    pns_exact_plan plans[PNS_EXACT_MAX_PLANS];
    int num_plans;
    float *work;
    float *dgdt;
    kiss_fft_cpx *spec;
    double *resp[3];
    int work_cap;
    int resp_cap;
    float *acc[3];
    long acc_cap;
    pns_render_scratch render;
} pns_exact_scratch;

static void pns_exact_scratch_free(pns_exact_scratch *s)
{
    int k;
    for (k = 0; k < s->num_plans; ++k)
    {
        if (s->plans[k].fwd)
            kiss_fftr_free(s->plans[k].fwd);
        if (s->plans[k].inv)
            kiss_fftr_free(s->plans[k].inv);
        if (s->plans[k].kspec)
            PULSEG_FREE(s->plans[k].kspec);
    }
    s->num_plans = 0;
    if (s->work)
        PULSEG_FREE(s->work);
    if (s->dgdt)
        PULSEG_FREE(s->dgdt);
    if (s->spec)
        PULSEG_FREE(s->spec);
    for (k = 0; k < 3; ++k)
    {
        if (s->resp[k])
            PULSEG_FREE(s->resp[k]);
        if (s->acc[k])
            PULSEG_FREE(s->acc[k]);
    }
    pns_render_scratch_free(&s->render);
    memset(s, 0, sizeof(*s));
}

/* The plan for a block of npts samples, made on first use. Past the plan
 * cap the oldest plan is dropped. */
static const pns_exact_plan *pns_exact_plan_for(
    pns_exact_scratch *s,
    const pns_exact_scan_job *job,
    int npts)
{
    pns_exact_plan *p;
    int k, i;

    for (k = 0; k < s->num_plans; ++k)
        if (s->plans[k].npts == npts)
            return &s->plans[k];
    if (s->num_plans == PNS_EXACT_MAX_PLANS)
    {
        p = &s->plans[0];
        kiss_fftr_free(p->fwd);
        kiss_fftr_free(p->inv);
        PULSEG_FREE(p->kspec);
        memmove(&s->plans[0], &s->plans[1], (size_t)(PNS_EXACT_MAX_PLANS - 1) * sizeof(*p));
        s->num_plans--;
    }
    p = &s->plans[s->num_plans];
    memset(p, 0, sizeof(*p));
    p->npts = npts;
    p->out_len = npts + job->kernel_len;
    p->nfft = (int)pulseg__next_pow2((size_t)p->out_len);
    p->nfreq = p->nfft / 2 + 1;
    p->fwd = kiss_fftr_alloc(p->nfft, 0, NULL, NULL);
    p->inv = kiss_fftr_alloc(p->nfft, 1, NULL, NULL);
    p->kspec = (kiss_fft_cpx *)PULSEG_ALLOC((size_t)p->nfreq * sizeof(kiss_fft_cpx));
    if (!p->fwd || !p->inv || !p->kspec)
        return NULL;
    if (p->nfft > s->work_cap)
    {
        if (s->work)
            PULSEG_FREE(s->work);
        if (s->dgdt)
            PULSEG_FREE(s->dgdt);
        if (s->spec)
            PULSEG_FREE(s->spec);
        s->work = (float *)PULSEG_ALLOC((size_t)p->nfft * sizeof(float));
        s->dgdt = (float *)PULSEG_ALLOC((size_t)(p->nfft + 1) * sizeof(float));
        s->spec = (kiss_fft_cpx *)PULSEG_ALLOC((size_t)p->nfreq * sizeof(kiss_fft_cpx));
        s->work_cap = p->nfft;
        if (!s->work || !s->dgdt || !s->spec)
            return NULL;
    }
    if (p->out_len + 2 > s->resp_cap)
    {
        for (i = 0; i < 3; ++i)
        {
            if (s->resp[i])
                PULSEG_FREE(s->resp[i]);
            s->resp[i] = (double *)PULSEG_ALLOC((size_t)(p->out_len + 2) * sizeof(double));
            if (!s->resp[i])
                return NULL;
        }
        s->resp_cap = p->out_len + 2;
    }
    memset(s->work, 0, (size_t)p->nfft * sizeof(float));
    memcpy(s->work, job->kernel, (size_t)job->kernel_len * sizeof(float));
    kiss_fftr(p->fwd, s->work, p->kspec);
    s->num_plans++;
    return p;
}

/* The last sample of every axis of block b in its logical frame -- what
 * the next block's opening step is measured against, the way the window
 * render joins one block's last corner to the next block's first before
 * turning either. Zero for a block without gradients or for b < 0. */
static int pns_exact_block_last(
    const pns_exact_scan_job *job,
    pns_render_scratch *r,
    int b,
    double *last_phys)
{
    pns_occ occ;
    const pulseg_block_table_element *bte;
    const pulseg_base_block *bdef;
    double last_log[3];
    const float *wave;
    float t0;
    int a, n, def_id;

    last_phys[0] = last_phys[1] = last_phys[2] = 0.0;
    if (b < 0)
        return PULSEG_SUCCESS;
    bte = &job->desc->block_table[b];
    bdef = &job->desc->base_blocks[bte->id];
    pns_resolve_occ(job->desc, bte, &occ);
    last_log[0] = last_log[1] = last_log[2] = 0.0;
    for (a = 0; a < 3; ++a)
    {
        if (!occ.has_grad[a])
            continue;
        def_id = (a == 0) ? bdef->gx_id : (a == 1) ? bdef->gy_id : bdef->gz_id;
        wave = pns_render_uniform(
            r,
            job->desc,
            def_id,
            occ.key[a].shape_id,
            occ.amplitude[a],
            (float)pns_block_duration_us(job->desc, b),
            job->raster_us,
            &n,
            &t0);
        if (!wave)
            return PULSEG_ERR_INVALID_ARGUMENT;
        last_log[a] = (n > 0) ? (double)wave[n - 1] : 0.0;
    }
    memcpy(last_phys, last_log, sizeof(last_log));
    return PULSEG_SUCCESS;
}

/* One chunk of consecutive blocks: the responses of its blocks and of the
 * earlier blocks still ringing into it, accumulated per physical axis over
 * the chunk's span plus the kernel's memory, and the peak of the RSS over
 * the chunk's own span. */
static int pns_exact_chunk(
    const pns_exact_scan_job *job,
    pns_exact_scratch *s,
    int c,
    double *out_peak,
    double *out_peak_time_us)
{
    const pulseg_sequence_descriptor *desc = job->desc;
    float *acc[3];
    double origin, end_us, judged_us, prev_last[3], first_log[3], last_log[3];
    double step, inv, v, ss, peak, peak_time;
    long acc_len, off, idx, judged_len;
    int c0, c1, first, b, a, i, k, n, def_id, rc, shift[3], span;
    float t0;

    c0 = c * job->chunk_blocks;
    c1 = c0 + job->chunk_blocks;
    if (c1 > desc->num_blocks)
        c1 = desc->num_blocks;
    first = c0;
    while (first > 0 && job->t_end_us[first - 1] + job->reach_us > job->t_start_us[c0])
        --first;
    origin = job->t_start_us[c0];
    end_us = job->t_end_us[c1 - 1] + job->reach_us;
    acc_len = (long)((end_us - origin) / (double)job->raster_us) + 2;
    judged_us = (c1 < desc->num_blocks) ? job->t_start_us[c1] : end_us;
    judged_len = (long)((judged_us - origin) / (double)job->raster_us);
    if (c1 >= desc->num_blocks)
        judged_len = acc_len;
    if (judged_len > acc_len)
        judged_len = acc_len;
    if (acc_len > s->acc_cap)
    {
        for (a = 0; a < 3; ++a)
        {
            if (s->acc[a])
                PULSEG_FREE(s->acc[a]);
            s->acc[a] = (float *)PULSEG_ALLOC((size_t)acc_len * sizeof(float));
            if (!s->acc[a])
                return PULSEG_ERR_ALLOC_FAILED;
        }
        s->acc_cap = acc_len;
    }
    for (a = 0; a < 3; ++a)
    {
        acc[a] = s->acc[a];
        memset(acc[a], 0, (size_t)acc_len * sizeof(float));
    }
    inv = 1.0 / ((double)job->gamma * ((double)job->raster_us * 1e-6));
    rc = pns_exact_block_last(job, &s->render, first - 1, prev_last);
    for (b = first; b < c1 && !PULSEG_FAILED(rc); ++b)
    {
        const pulseg_block_table_element *bte = &desc->block_table[b];
        const pulseg_base_block *bdef = &desc->base_blocks[bte->id];
        const pns_exact_plan *plan;
        pns_occ occ;
        double dur;
        int npts;

        dur = pns_block_duration_us(desc, b);
        npts = (int)(dur / (double)job->raster_us) + 1;
        plan = pns_exact_plan_for(s, job, npts);
        if (!plan)
        {
            rc = PULSEG_ERR_ALLOC_FAILED;
            break;
        }
        pns_resolve_occ(desc, bte, &occ);
        span = plan->out_len + 2;
        for (a = 0; a < 3; ++a)
        {
            first_log[a] = 0.0;
            last_log[a] = 0.0;
            shift[a] = 0;
            for (i = 0; i < span; ++i)
                s->resp[a][i] = 0.0;
            if (!occ.has_grad[a])
                continue;
            def_id = (a == 0) ? bdef->gx_id : (a == 1) ? bdef->gy_id : bdef->gz_id;
            {
                const float *wave = pns_render_uniform(
                    &s->render,
                    desc,
                    def_id,
                    occ.key[a].shape_id,
                    occ.amplitude[a],
                    (float)dur,
                    job->raster_us,
                    &n,
                    &t0);
                if (!wave)
                {
                    rc = PULSEG_ERR_INVALID_ARGUMENT;
                    break;
                }
                /* The block's own samples, and nothing past its last one: the
                 * gradient there is the next block's first sample, which its
                 * opening step accounts for. */
                for (i = 0; i < n; ++i)
                    s->work[i] = wave[i];
                first_log[a] = (double)wave[0];
                last_log[a] = (n > 0) ? (double)wave[n - 1] : 0.0;
                shift[a] = (int)floor((double)t0 / (double)job->raster_us + 0.5);
                if (shift[a] < 0)
                    shift[a] = 0;
                if (shift[a] > 2)
                    shift[a] = 2;
            }
            pns_slice_dgdt(s->dgdt, s->work, n, job->raster_us, job->gamma);
            memset(s->work, 0, (size_t)plan->nfft * sizeof(float));
            memcpy(s->work, s->dgdt, (size_t)(n + 1) * sizeof(float));
            kiss_fftr(plan->fwd, s->work, s->spec);
            for (i = 0; i < plan->nfreq; ++i)
            {
                float re, im;
                re = s->spec[i].r * plan->kspec[i].r - s->spec[i].i * plan->kspec[i].i;
                im = s->spec[i].r * plan->kspec[i].i + s->spec[i].i * plan->kspec[i].r;
                s->spec[i].r = re;
                s->spec[i].i = im;
            }
            kiss_fftri(plan->inv, s->spec, s->work);
            for (i = 0; i < plan->out_len && shift[a] + i < span; ++i)
                s->resp[a][shift[a] + i] = (double)s->work[i] / (double)plan->nfft;
            /* The opening step against the previous block's end, one kernel
             * tap of that size at this axis's first sample; the scan's
             * closing step on its last block. Both in the logical frame,
             * which is how the window render joins corners across blocks. */
            step = (first_log[a] - prev_last[a]) * inv;
            for (k = 0; k < job->kernel_len && shift[a] + k < span; ++k)
                s->resp[a][shift[a] + k] += step * (double)job->kernel[k];
            if (b == desc->num_blocks - 1)
            {
                v = -last_log[a] * inv;
                for (k = 0; k < job->kernel_len && shift[a] + n + k < span; ++k)
                    s->resp[a][shift[a] + n + k] += v * (double)job->kernel[k];
            }
        }
        if (PULSEG_FAILED(rc))
            break;
        /* Into the physical frame: the rotation is linear and the kernel is
         * the same on every axis, so rotating the responses is rotating the
         * drive. */
        if (occ.R)
        {
            double vin[3], vout[3];
            for (i = 0; i < span; ++i)
            {
                vin[0] = s->resp[0][i];
                vin[1] = s->resp[1][i];
                vin[2] = s->resp[2][i];
                pns_rotate3(occ.R, vin, vout);
                s->resp[0][i] = vout[0];
                s->resp[1][i] = vout[1];
                s->resp[2][i] = vout[2];
            }
        }
        memcpy(prev_last, last_log, sizeof(last_log));
        off = (long)floor((job->t_start_us[b] - origin) / (double)job->raster_us + 0.5);
        for (a = 0; a < 3; ++a)
        {
            for (i = 0; i < span; ++i)
            {
                idx = off + i;
                if (idx < 0 || idx >= acc_len)
                    continue;
                acc[a][idx] += (float)((double)job->out_scale * s->resp[a][i]);
            }
        }
    }
    peak = 0.0;
    peak_time = origin;
    if (!PULSEG_FAILED(rc))
    {
        for (idx = 0; idx < judged_len; ++idx)
        {
            ss = (double)acc[0][idx] * acc[0][idx] + (double)acc[1][idx] * acc[1][idx] +
                (double)acc[2][idx] * acc[2][idx];
            if (ss > peak)
            {
                peak = ss;
                peak_time = origin + (double)idx * (double)job->raster_us;
            }
        }
        peak = sqrt(peak);
    }
    *out_peak = peak;
    *out_peak_time_us = peak_time;
    return rc;
}

static void pns_exact_scan_range(void *arg, int begin, int end)
{
    pns_exact_scan_job *job = (pns_exact_scan_job *)arg;
    pns_exact_scratch scratch;
    int c, rc;

    memset(&scratch, 0, sizeof(scratch));
    for (c = begin; c < end && !PULSEG_FAILED(job->rc); ++c)
    {
        rc = pns_exact_chunk(job, &scratch, c, &job->chunk_peak[c], &job->chunk_peak_time_us[c]);
        if (PULSEG_FAILED(rc))
            job->rc = rc;
    }
    pns_exact_scratch_free(&scratch);
}

/* The scan's peak through the model's kernel, and the block it falls in. */
static int pns_exact_scan_fft(
    const pulseg_sequence_descriptor *desc,
    const pulseg_pns_model *model,
    pulseg__parallel_for_fn par_fn,
    void *par_ctx,
    float gamma,
    double *out_peak,
    int *out_peak_block)
{
    pns_exact_scan_job job;
    float *kernel = NULL;
    double *t_start = NULL, *t_end = NULL, *peaks = NULL, *times = NULL;
    double best, best_time;
    int kernel_len = 0, rc, c, b;
    float out_scale = 0.0f, raster;

    memset(&job, 0, sizeof(job));
    /* The gradient raster: the kernel is bin-integrated, so the response is
     * the same as at a finer raster to the allowance the raster-halving
     * invariant holds, at half the transform length. */
    raster = desc->grad_raster_us;
    rc = model->kernel(model->ctx, raster, &kernel, &kernel_len, &out_scale);
    if (PULSEG_FAILED(rc) || kernel_len <= 0)
    {
        if (kernel)
            PULSEG_FREE(kernel);
        return PULSEG_FAILED(rc) ? rc : PULSEG_ERR_INVALID_ARGUMENT;
    }
    t_start = (double *)PULSEG_ALLOC((size_t)desc->num_blocks * sizeof(double));
    t_end = (double *)PULSEG_ALLOC((size_t)desc->num_blocks * sizeof(double));
    job.num_chunks = (desc->num_blocks + PNS_EXACT_CHUNK_BLOCKS - 1) / PNS_EXACT_CHUNK_BLOCKS;
    peaks = (double *)PULSEG_ALLOC((size_t)job.num_chunks * sizeof(double));
    times = (double *)PULSEG_ALLOC((size_t)job.num_chunks * sizeof(double));
    if (!t_start || !t_end || !peaks || !times)
    {
        rc = PULSEG_ERR_ALLOC_FAILED;
        goto done;
    }
    pns_timeline(desc, t_start, t_end);
    job.desc = desc;
    job.t_start_us = t_start;
    job.t_end_us = t_end;
    job.kernel = kernel;
    job.kernel_len = kernel_len;
    job.out_scale = out_scale;
    job.raster_us = raster;
    job.gamma = gamma;
    job.reach_us = ((double)kernel_len + 1.0) * (double)raster;
    job.chunk_blocks = PNS_EXACT_CHUNK_BLOCKS;
    job.chunk_peak = peaks;
    job.chunk_peak_time_us = times;
    job.rc = PULSEG_SUCCESS;
    if (par_fn)
        par_fn(par_ctx, job.num_chunks, pns_exact_scan_range, &job);
    else
        pns_exact_scan_range(&job, 0, job.num_chunks);
    rc = job.rc;
    if (PULSEG_FAILED(rc))
        goto done;
    best = -1.0;
    best_time = 0.0;
    for (c = 0; c < job.num_chunks; ++c)
    {
        if (peaks[c] > best)
        {
            best = peaks[c];
            best_time = times[c];
        }
    }
    *out_peak = best * (1.0 + PNS_EXACT_FFT_MARGIN);
    *out_peak_block = 0;
    for (b = 0; b < desc->num_blocks; ++b)
        if (t_start[b] <= best_time)
            *out_peak_block = b;
done:
    if (kernel)
        PULSEG_FREE(kernel);
    if (t_start)
        PULSEG_FREE(t_start);
    if (t_end)
        PULSEG_FREE(t_end);
    if (peaks)
        PULSEG_FREE(peaks);
    if (times)
        PULSEG_FREE(times);
    return rc;
}

/* The scan's peak through a model that publishes no kernel: chunks rendered
 * and handed to the model's evaluator, each opened enough blocks early to
 * carry its history, judged over its own span. Sequential: the plan the
 * rendering runs through is one caller's. */
static int pns_exact_scan_direct(
    pulseg_check_plan *plan,
    pulseg_diagnostic *diag,
    const pulseg_sequence_descriptor *desc,
    int subseq_idx,
    const pulseg_pns_model *model,
    float gamma,
    double *out_peak,
    int *out_peak_block)
{
    double *t_start = NULL, *t_end = NULL;
    double peak, ptime, best = -1.0, best_time = 0.0;
    int c0, c1, first, rc = PULSEG_SUCCESS, b;

    t_start = (double *)PULSEG_ALLOC((size_t)desc->num_blocks * sizeof(double));
    t_end = (double *)PULSEG_ALLOC((size_t)desc->num_blocks * sizeof(double));
    if (!t_start || !t_end)
    {
        rc = PULSEG_ERR_ALLOC_FAILED;
        goto done;
    }
    pns_timeline(desc, t_start, t_end);
    for (c0 = 0; c0 < desc->num_blocks; c0 = c1)
    {
        c1 = c0 + PNS_EXACT_CHUNK_BLOCKS;
        if (c1 > desc->num_blocks)
            c1 = desc->num_blocks;
        first = c0;
        while (first > 0 && t_end[first - 1] + PNS_EXACT_NO_KERNEL_REACH_US > t_start[c0])
            --first;
        rc = pulseg__pns_exact_range_peak(
            plan,
            diag,
            desc,
            subseq_idx,
            first,
            c1 - first,
            t_start[c0] - t_start[first],
            (c1 < desc->num_blocks) ? t_start[c1] - t_start[first] : 0.0,
            model,
            gamma,
            &peak,
            &ptime);
        if (PULSEG_FAILED(rc))
            goto done;
        if (peak > best)
        {
            best = peak;
            best_time = t_start[first] + ptime;
        }
    }
    *out_peak = best;
    *out_peak_block = 0;
    for (b = 0; b < desc->num_blocks; ++b)
        if (t_start[b] <= best_time)
            *out_peak_block = b;
done:
    if (t_start)
        PULSEG_FREE(t_start);
    if (t_end)
        PULSEG_FREE(t_end);
    return rc;
}

int pulseg__pns_exact_scan_peak(
    pulseg_check_plan *plan,
    pulseg_diagnostic *diag,
    const pulseg_sequence_descriptor *desc,
    int subseq_idx,
    const pulseg_pns_model *model,
    pulseg__parallel_for_fn par_fn,
    void *par_ctx,
    float gamma,
    double *out_peak,
    int *out_peak_block)
{
    if (!desc || !model || !out_peak || !out_peak_block)
        return PULSEG_ERR_NULL_POINTER;
    if (desc->num_blocks <= 0 || gamma <= 0.0f)
        return PULSEG_ERR_INVALID_ARGUMENT;
    if (model->kernel)
        return pns_exact_scan_fft(desc, model, par_fn, par_ctx, gamma, out_peak, out_peak_block);
    return pns_exact_scan_direct(
        plan,
        diag,
        desc,
        subseq_idx,
        model,
        gamma,
        out_peak,
        out_peak_block);
}

int pulseg__pns_exact_scan_check(
    pulseg_check_plan *plan,
    pulseg_diagnostic *diag,
    const pulseg_sequence_descriptor *desc,
    int subseq_idx,
    const pulseg_pns_model *model,
    pulseg__parallel_for_fn par_fn,
    void *par_ctx,
    float gamma,
    float threshold_percent,
    double *out_peak,
    int *out_peak_block)
{
    double peak = 0.0;
    int block = 0, rc;

    rc = pulseg__pns_exact_scan_peak(
        plan,
        diag,
        desc,
        subseq_idx,
        model,
        par_fn,
        par_ctx,
        gamma,
        &peak,
        &block);
    if (PULSEG_FAILED(rc))
        return rc;
    if (out_peak)
        *out_peak = peak;
    if (out_peak_block)
        *out_peak_block = block;
    if (peak > (double)threshold_percent)
    {
        if (diag)
        {
            pulseg__diag_printf(
                diag,
                "PNS threshold exceeded (peak %.1f%% > %.1f%%, block %d)",
                peak,
                (double)threshold_percent,
                block);
            diag->code = PULSEG_ERR_PNS_THRESHOLD_EXCEEDED;
        }
        return PULSEG_ERR_PNS_THRESHOLD_EXCEEDED;
    }
    return PULSEG_SUCCESS;
}
