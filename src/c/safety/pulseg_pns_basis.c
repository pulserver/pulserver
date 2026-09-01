/**
 * @file pulseg_pns_basis.c
 * @brief Occurrence-score PNS: a sound per-block bound with no windows.
 *
 * The group sweep answers the PNS question by materialising one waveform
 * window per set of shapes the repetitions play, which prices a scan by how
 * many distinct waveforms it holds. This module prices it by linearity
 * instead. The scan's response is a sum of per-block-occurrence responses,
 *
 *     R(t) = sum_i r_i(t - t_i),   r_i = k * dg_i/dt   (per axis),
 *
 * so by the triangle inequality in R^3 the peak of ||R||_2 is bounded, at any
 * instant, by a sum over the occurrences that have already started: the
 * block playing contributes at most its response envelope
 * E_i(k) = sup over the k-th of K equal windows of its span of ||r_i||_2,
 * and one that ended a gap delta before that window contributes at most
 * its tail peak T_i(delta) = sup_{tau >= delta} ||r_i(end_i + tau)||_2 --
 * what its response can still be that long after it stopped, which for a
 * waveform that ends at rest is a small fraction of its peak. The slide
 * keeps T_i at four gap edges and walks the K windows of every block, so a
 * long readout's peak is met by its neighbours' tails at the gap they
 * really have then, not at the readout's start. Twelve numbers per
 * occurrence and a monotone sweep -- no window, no grouping, no cap on how
 * many distinct waveforms the scan plays.
 *
 * u_i and T_i are bounded from two catalogues, both priced per *distinct*
 * object rather than per occurrence:
 *
 *  - a rank-1 catalogue: sup_tau |k * d(unit waveform)/dt| once per
 *    (definition, shape) identity, scaled by |amplitude| at score time;
 *  - an element catalogue per base block whose occurrences play many
 *    distinct waveform tuples: one (d, npts) element per distinct
 *    (shape, amplitude) tuple the block plays, at the amplitudes it really
 *    plays, each element's own response computed exactly by FFT convolution
 *    with the kernel, streamed one element at a time so no matrix is ever
 *    held, on every core the caller offers.
 *
 * Every element and template is sliced to its interior slew only: the
 * step a block makes at its start -- from the previous block's last sample
 * to its own first -- is priced per occurrence, as one kernel tap of that
 * size, and the scan's closing step on its last block. A gradient that runs
 * continuously across blocks is charged nothing at the seams it does not
 * have; one that starts from rest is charged exactly its start.
 *
 * Per-occurrence norms are frame-invariant -- a block's rotation multiplies
 * its logical response vector by an orthogonal matrix before the norm is
 * taken -- so the score is sound for the rotated scan while every waveform
 * here is rendered in the logical frame, through the very renderer and
 * resampler the exact extraction uses (pulseg__fill_grad_waveform_for_block,
 * pulseg__interpolate_to_uniform, at half the gradient raster). The kernel's
 * taps are bin integrals, so residual raster differences between this
 * pricing and the exact evaluation stay inside the raster-invariance bound
 * held by test_the_response_does_not_move_when_the_raster_is_halved.
 *
 * A scan the module cannot price -- too many basis elements, a failed
 * decomposition -- is *declined*, not refused: the caller falls back to the
 * group sweep unchanged.
 */

#include <math.h>
#include <string.h>

#include "external_kiss_fftr.h"
#include "pulseg_internal.h"
#include "pulseg_pns_models.h"

/* A base block earns an element catalogue once this many distinct element
 * tuples appear at that position; below it the rank-1 catalogue is already
 * tight, one entry per shape. */
#define PNS_BASIS_MIN_ELEMENTS 4

/* Distinct (shape, amplitude) tuples one base block may play before the
 * module declines the scan to the sweep. */
#define PNS_BASIS_MAX_ELEMENTS 262144

/* Relative allowance added to an element peak priced exactly: the
 * single-precision transform's round-off against the direct sum it stands
 * for, and the movement of the response between the gradient raster the
 * elements are priced at and the half raster the exact evaluation streams
 * at, which test_the_response_does_not_move_when_the_raster_is_halved holds
 * below 1e-3. */
#define PNS_EXACT_FFT_MARGIN 2e-3

/* Rendered duration one offender range may span. A dense offence splits
 * into consecutive self-sufficient pieces instead of merging into one
 * scan-length window, so the exact assembly's memory stays bounded and a
 * failing scan is refused on its first piece. A single block longer than
 * the budget still renders whole; the budget caps accumulation, not one
 * block. */
#define PNS_SCORE_RANGE_BUDGET_US 1.0e6

/* ================================================================== */
/*  The score object                                                   */
/* ================================================================== */

/* Gap zones for the slide: an earlier occurrence whose end sits delta
 * before the anchor is priced by its tail peak at the zone's smallest
 * delta. Zone 0 starts at zero gap. */
#define PNS_SCORE_NUM_ZONES PULSEG__PNS_SCORE_ZONES

/* Equal windows a block's own span is priced over. */
#define PNS_SCORE_NUM_WINDOWS PULSEG__PNS_SCORE_WINDOWS

struct pulseg__pns_score
{
    int num_blocks;
    float *u;           /* [num_blocks] per-occurrence response-peak bound, % */
    float *env;         /* [num_blocks * windows] peak over each own window   */
    float *tail;        /* [num_blocks * zones] tail peak at each zone edge   */
    double *t_start_us; /* [num_blocks] block start on the scan timeline      */
    double *t_end_us;   /* [num_blocks] block end                             */
    double reach_us;    /* kernel memory: supports span [start, end + reach]  */
    double zone_edge_us[PNS_SCORE_NUM_ZONES]; /* smallest gap of each zone    */
    int num_bases;
    pulseg__pns_basis_info *bases; /* [num_bases] */
    double build_macs;
};

void pulseg__pns_score_free(pulseg__pns_score *sc)
{
    if (!sc)
        return;
    if (sc->u)
        PULSEG_FREE(sc->u);
    if (sc->env)
        PULSEG_FREE(sc->env);
    if (sc->tail)
        PULSEG_FREE(sc->tail);
    if (sc->t_start_us)
        PULSEG_FREE(sc->t_start_us);
    if (sc->t_end_us)
        PULSEG_FREE(sc->t_end_us);
    if (sc->bases)
        PULSEG_FREE(sc->bases);
    PULSEG_FREE(sc);
}

int pulseg__pns_score_num_bases(const pulseg__pns_score *sc)
{
    return sc ? sc->num_bases : 0;
}

double pulseg__pns_score_block_start_us(const pulseg__pns_score *sc, int block)
{
    if (!sc || block < 0 || block >= sc->num_blocks)
        return 0.0;
    return sc->t_start_us[block];
}

const pulseg__pns_basis_info *pulseg__pns_score_basis_info(const pulseg__pns_score *sc, int index)
{
    if (!sc || index < 0 || index >= sc->num_bases)
        return NULL;
    return &sc->bases[index];
}

int pulseg__pns_score_block_bound(
    const pulseg__pns_score *sc,
    int block,
    float *out_u,
    float *out_env,
    float *out_tail,
    double *out_start_us,
    double *out_end_us)
{
    int z, k;

    if (!sc || block < 0 || block >= sc->num_blocks)
        return PULSEG_ERR_INVALID_ARGUMENT;
    if (out_u)
        *out_u = sc->u[block];
    if (out_env)
        for (k = 0; k < PNS_SCORE_NUM_WINDOWS; ++k)
            out_env[k] = sc->env[(size_t)block * PNS_SCORE_NUM_WINDOWS + k];
    if (out_tail)
        for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
            out_tail[z] = sc->tail[(size_t)block * PNS_SCORE_NUM_ZONES + z];
    if (out_start_us)
        *out_start_us = sc->t_start_us[block];
    if (out_end_us)
        *out_end_us = sc->t_end_us[block];
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Rendering one axis of one block, the extraction's way              */
/* ================================================================== */

/* Vertex list -> uniform raster, exactly as the window extraction does it.
 * Returns a PULSEG_ALLOC'd buffer of *out_n samples, or NULL. */
static float *pns_basis_render_uniform(
    const pulseg_sequence_descriptor *desc,
    int def_id,
    int shape_id,
    float amplitude,
    float block_duration_us,
    float raster_us,
    int *out_n)
{
    pulseg_grad_table_element gte;
    const pulseg_grad_definition *gdef;
    float *time, *wave;
    float amp;
    int cap, n;

    *out_n = 0;
    if (def_id < 0 || def_id >= desc->num_unique_grads)
        return NULL;
    gdef = &desc->grad_definitions[def_id];
    gte.id = def_id;
    gte.shape_id = shape_id;
    gte.amplitude = amplitude;
    amp = amplitude;

    cap = pulseg__count_grad_samples_for_block(desc, gdef, block_duration_us) + 2;
    time = (float *)PULSEG_ALLOC((size_t)cap * sizeof(float));
    wave = (float *)PULSEG_ALLOC((size_t)cap * sizeof(float));
    if (!time || !wave)
    {
        if (time)
            PULSEG_FREE(time);
        if (wave)
            PULSEG_FREE(wave);
        return NULL;
    }

    n = pulseg__fill_grad_waveform_for_block(
        desc,
        time,
        wave,
        0,
        gdef,
        &gte,
        0.0f,
        &amp,
        block_duration_us);
    if (n <= 0 || n > cap ||
        PULSEG_FAILED(pulseg__interpolate_to_uniform(&time, &wave, &n, raster_us)))
    {
        PULSEG_FREE(time);
        PULSEG_FREE(wave);
        return NULL;
    }
    PULSEG_FREE(time);
    *out_n = n;
    return wave;
}

/* Slice forward difference of the block's interior, on the model's tap
 * grid: n+1 taps, the first and last zero, the step into the block and the
 * step out of it being priced per occurrence instead. Each over gamma * dt.
 * Writes into dgdt[n+1]. */
static void pns_basis_slice_dgdt(
    float *dgdt,
    const float *wave,
    int n,
    float raster_us,
    float gamma)
{
    double inv;
    int i;

    inv = 1.0 / ((double)gamma * ((double)raster_us * 1e-6));
    dgdt[0] = 0.0f;
    for (i = 1; i < n; ++i)
        dgdt[i] = (float)(((double)wave[i] - (double)wave[i - 1]) * inv);
    dgdt[n] = 0.0f;
}

/* Where a block's response samples fall: sample m sits m rasters after the
 * block start. Window k of the block's own span runs from
 * floor(k * duration / K / raster); the tail of zone z is every m from
 * floor((duration + edge_z) / raster) on. A window that rounds to no sample
 * of its own is folded into the next by the walk below. */
typedef struct
{
    int m_window[PNS_SCORE_NUM_WINDOWS + 1]; /* [k, k+1) bounds, last = end */
    int m_zone[PNS_SCORE_NUM_ZONES];
} pns_basis_extent;

static void pns_basis_extent_init(
    pns_basis_extent *ex,
    float block_duration_us,
    float raster_us,
    const double *zone_edge_us)
{
    int z, k, m;

    /* Every region opens one sample early: the rendered grid may start up
     * to one sample after the block does, and a sup over a superset of the
     * intended samples is the side to err on. */
    for (k = 0; k <= PNS_SCORE_NUM_WINDOWS; ++k)
    {
        m = (int)((double)k * (double)block_duration_us / (double)PNS_SCORE_NUM_WINDOWS / (double)raster_us) -
            1;
        ex->m_window[k] = m > 0 ? m : 0;
    }
    for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
    {
        m = (int)(((double)block_duration_us + zone_edge_us[z]) / (double)raster_us) - 1;
        ex->m_zone[z] = m > 0 ? m : 0;
    }
}

/* Fold the RSS sample at m into the window and tail sups it belongs to.
 * Samples past the last window's end are the block's tail and count for no
 * window; the peak `sup` takes every sample. */
static void pns_basis_extent_fold(
    const pns_basis_extent *ex,
    int m,
    double ss,
    double *sup,
    double *env,
    double *tail)
{
    int k, z;

    if (ss > *sup)
        *sup = ss;
    for (k = 0; k < PNS_SCORE_NUM_WINDOWS; ++k)
        if (m >= ex->m_window[k] && m < ex->m_window[k + 1] && ss > env[k])
            env[k] = ss;
    if (m >= ex->m_window[PNS_SCORE_NUM_WINDOWS] && ss > env[PNS_SCORE_NUM_WINDOWS - 1])
        env[PNS_SCORE_NUM_WINDOWS - 1] = ss;
    for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
        if (m >= ex->m_zone[z] && ss > tail[z])
            tail[z] = ss;
}

/* sup_tau of the d-axis RSS response of dgdt rows of `len` taps each, with
 * the same sup per own window and per tail zone. All in percent. */
static double pns_basis_response_sups(
    const float *rows,
    int d,
    int len,
    const float *kernel,
    int kernel_len,
    float out_scale,
    const pns_basis_extent *ex,
    double *out_env,
    double *out_tail,
    double *macs)
{
    double sup, acc, ss, env[PNS_SCORE_NUM_WINDOWS], tail[PNS_SCORE_NUM_ZONES];
    int m, j, lo, hi, out_len, a, z, k;

    sup = 0.0;
    for (k = 0; k < PNS_SCORE_NUM_WINDOWS; ++k)
        env[k] = 0.0;
    for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
        tail[z] = 0.0;
    out_len = len + kernel_len - 1;
    for (m = 0; m < out_len; ++m)
    {
        ss = 0.0;
        lo = (m - kernel_len + 1 > 0) ? m - kernel_len + 1 : 0;
        hi = (m < len - 1) ? m : len - 1;
        for (a = 0; a < d; ++a)
        {
            acc = 0.0;
            for (j = lo; j <= hi; ++j)
                acc += (double)rows[(size_t)a * len + j] * (double)kernel[m - j];
            ss += acc * acc;
        }
        pns_basis_extent_fold(ex, m, ss, &sup, env, tail);
        *macs += (double)d * (double)(hi - lo + 1);
    }
    for (k = 0; k < PNS_SCORE_NUM_WINDOWS; ++k)
        out_env[k] = sqrt(env[k]) * (double)out_scale;
    for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
        out_tail[z] = sqrt(tail[z]) * (double)out_scale;
    return sqrt(sup) * (double)out_scale;
}

/* ================================================================== */
/*  Rank-1 catalogue: sup of the unit response per (definition, shape)  */
/* ================================================================== */

typedef struct
{
    int def_id;
    int shape_id;
    double sup;
    double env[PNS_SCORE_NUM_WINDOWS]; /* unit peak over each own window   */
    double tail[PNS_SCORE_NUM_ZONES];  /* unit tail peak at each zone edge */
    float first;                       /* unit waveform's first sample     */
    float last;                        /* unit waveform's last sample      */
} pns_basis_r1_entry;

typedef struct
{
    pns_basis_r1_entry *slots;
    int cap; /* power of two */
} pns_basis_r1;

static int pns_basis_r1_init(pns_basis_r1 *cat, int expected)
{
    int i;

    cat->cap = 16;
    while (cat->cap < 4 * expected)
        cat->cap *= 2;
    cat->slots = (pns_basis_r1_entry *)PULSEG_ALLOC((size_t)cat->cap * sizeof(*cat->slots));
    if (!cat->slots)
        return PULSEG_ERR_ALLOC_FAILED;
    for (i = 0; i < cat->cap; ++i)
        cat->slots[i].def_id = -1;
    return PULSEG_SUCCESS;
}

static void pns_basis_r1_destroy(pns_basis_r1 *cat)
{
    if (cat->slots)
        PULSEG_FREE(cat->slots);
    cat->slots = NULL;
}

/* The unit-amplitude response peak of one identity, computed on first use.
 * Keyed on (definition, shape): the definition fixes the timing skeleton --
 * delay, trapezoid legs, a time shape -- and the shape the samples, so two
 * identities never share a waveform. Trailing block padding is zeros and
 * moves no supremum, so block duration is not part of the key. */
static int pns_basis_r1_get(
    pns_basis_r1 *cat,
    const pulseg_sequence_descriptor *desc,
    const pulseg__wave_key *key,
    float block_duration_us,
    float raster_us,
    float gamma,
    const float *kernel,
    int kernel_len,
    float out_scale,
    const double *zone_edge_us,
    double *out_sup,
    double *out_env,
    double *out_tail,
    float *out_first,
    float *out_last,
    double *macs)
{
    pns_basis_r1_entry *slot;
    pns_basis_extent ex;
    float *wave, *dgdt;
    unsigned h;
    int n, z, k;

    h = (unsigned)key->def_id * 2654435761u ^ (unsigned)key->shape_id * 40503u;
    slot = &cat->slots[h & (unsigned)(cat->cap - 1)];
    while (slot->def_id != -1 && (slot->def_id != key->def_id || slot->shape_id != key->shape_id))
    {
        ++h;
        slot = &cat->slots[h & (unsigned)(cat->cap - 1)];
    }
    if (slot->def_id != -1)
    {
        *out_sup = slot->sup;
        for (k = 0; k < PNS_SCORE_NUM_WINDOWS; ++k)
            out_env[k] = slot->env[k];
        for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
            out_tail[z] = slot->tail[z];
        *out_first = slot->first;
        *out_last = slot->last;
        return PULSEG_SUCCESS;
    }

    wave = pns_basis_render_uniform(
        desc,
        key->def_id,
        key->shape_id,
        1.0f,
        block_duration_us,
        raster_us,
        &n);
    if (!wave)
        return PULSEG_ERR_ALLOC_FAILED;
    dgdt = (float *)PULSEG_ALLOC((size_t)(n + 1) * sizeof(float));
    if (!dgdt)
    {
        PULSEG_FREE(wave);
        return PULSEG_ERR_ALLOC_FAILED;
    }
    pns_basis_slice_dgdt(dgdt, wave, n, raster_us, gamma);
    pns_basis_extent_init(&ex, block_duration_us, raster_us, zone_edge_us);
    slot->sup = pns_basis_response_sups(
        dgdt,
        1,
        n + 1,
        kernel,
        kernel_len,
        out_scale,
        &ex,
        slot->env,
        slot->tail,
        macs);
    slot->def_id = key->def_id;
    slot->shape_id = key->shape_id;
    slot->first = wave[0];
    slot->last = wave[n - 1];
    PULSEG_FREE(wave);
    PULSEG_FREE(dgdt);
    *macs += (double)n;
    *out_sup = slot->sup;
    for (k = 0; k < PNS_SCORE_NUM_WINDOWS; ++k)
        out_env[k] = slot->env[k];
    for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
        out_tail[z] = slot->tail[z];
    *out_first = slot->first;
    *out_last = slot->last;
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  Occurrence geometry                                                */
/* ================================================================== */

typedef struct
{
    int has_grad[3];
    pulseg__wave_key key[3];
    float amplitude[3];
    int rotated; /* a non-identity rotation applies to this occurrence */
} pns_basis_occ;

static void pns_basis_resolve_occ(
    const pulseg_sequence_descriptor *desc,
    const pulseg_block_table_element *bte,
    pns_basis_occ *occ)
{
    int a, raw;

    occ->rotated = 0;
    if (!bte->norot_flag && bte->rotation_id != -1 && bte->rotation_id < desc->num_rotations &&
        !pulseg__is_identity3(desc->rotation_matrices[bte->rotation_id]))
        occ->rotated = 1;
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

/* What varies where, per base block, over the whole scan. */
typedef struct
{
    int seen;
    int rotated;
    int first_shape[3]; /* shape id at the first occurrence, -2 = silent */
    int varies[3];
    int retained[3]; /* joins the decomposition */
    int d;
    int basis_index; /* dense index into the info array, or -1 */
} pns_basis_block_stat;

/* One distinct (shape, amplitude) tuple a base block plays, at the
 * amplitudes it really plays -- element identity is exact equality, so the
 * decomposed object IS the played object and no identification slack enters
 * the bound. */
typedef struct
{
    int shape_id[3];
    float amplitude[3];
    float bound;                      /* response-peak bound, in percent   */
    float env[PNS_SCORE_NUM_WINDOWS]; /* peak bound over each own window   */
    float tail[PNS_SCORE_NUM_ZONES];  /* tail-peak bound at each zone edge */
    float first[3];                   /* rendered first sample per slot    */
    float last[3];                    /* rendered last sample per slot     */
} pns_basis_element;

static int pns_basis_element_matches(
    const pns_basis_element *e,
    const int *shape_id,
    const float *amplitude,
    int d)
{
    int a;

    for (a = 0; a < d; ++a)
        if (e->shape_id[a] != shape_id[a] || e->amplitude[a] != amplitude[a])
            return 0;
    return 1;
}

/* Element lookup must not scan: a scan prices the build at
 * occurrences * elements, which is the quantity this module exists to keep
 * out of the gate. Open addressing over the exact tuple; slots hold element
 * indices, -1 empty. */
typedef struct
{
    int *slot;
    int cap; /* power of two */
} pns_basis_elem_index;

static unsigned pns_basis_tuple_hash(const int *shape_id, const float *amplitude, int d)
{
    unsigned h;
    unsigned bits;
    int a;

    h = 2166136261u;
    for (a = 0; a < d; ++a)
    {
        h = (h ^ (unsigned)shape_id[a]) * 16777619u;
        memcpy(&bits, &amplitude[a], sizeof(bits));
        h = (h ^ bits) * 16777619u;
    }
    return h;
}

static int pns_basis_elem_index_init(pns_basis_elem_index *ix, int expected)
{
    int i;

    ix->cap = 16;
    while (ix->cap < 2 * expected)
        ix->cap *= 2;
    ix->slot = (int *)PULSEG_ALLOC((size_t)ix->cap * sizeof(int));
    if (!ix->slot)
        return PULSEG_ERR_ALLOC_FAILED;
    for (i = 0; i < ix->cap; ++i)
        ix->slot[i] = -1;
    return PULSEG_SUCCESS;
}

static void pns_basis_elem_index_destroy(pns_basis_elem_index *ix)
{
    if (ix->slot)
        PULSEG_FREE(ix->slot);
    ix->slot = NULL;
}

/* Find the tuple's element, or the empty slot it belongs in. Returns the
 * element index (>= 0) or -(slot + 1) for a miss. */
static int pns_basis_elem_index_probe(
    const pns_basis_elem_index *ix,
    const pns_basis_element *elems,
    const int *shape_id,
    const float *amplitude,
    int d)
{
    unsigned h;
    int s;

    h = pns_basis_tuple_hash(shape_id, amplitude, d);
    for (;;)
    {
        s = (int)(h & (unsigned)(ix->cap - 1));
        if (ix->slot[s] == -1)
            return -(s + 1);
        if (pns_basis_element_matches(&elems[ix->slot[s]], shape_id, amplitude, d))
            return ix->slot[s];
        ++h;
    }
}

/* Grow-and-rehash once the table passes half full. */
static int pns_basis_elem_index_maybe_grow(
    pns_basis_elem_index *ix,
    const pns_basis_element *elems,
    int num_elems,
    int d)
{
    pns_basis_elem_index grown;
    int e, probe;

    if (2 * num_elems < ix->cap)
        return PULSEG_SUCCESS;
    if (PULSEG_FAILED(pns_basis_elem_index_init(&grown, ix->cap)))
        return PULSEG_ERR_ALLOC_FAILED;
    for (e = 0; e < num_elems; ++e)
    {
        probe = pns_basis_elem_index_probe(&grown, elems, elems[e].shape_id, elems[e].amplitude, d);
        grown.slot[-probe - 1] = e;
    }
    pns_basis_elem_index_destroy(ix);
    *ix = grown;
    return PULSEG_SUCCESS;
}

/* The retained-slot tuple of one occurrence. */
static void pns_basis_occ_tuple(
    const pns_basis_occ *occ,
    const pns_basis_block_stat *st,
    int *shape_id,
    float *amplitude)
{
    int a, slot;

    slot = 0;
    for (a = 0; a < 3; ++a)
    {
        if (!st->retained[a])
            continue;
        shape_id[slot] = occ->has_grad[a] ? occ->key[a].shape_id : -2;
        amplitude[slot] = occ->has_grad[a] ? occ->amplitude[a] : 0.0f;
        ++slot;
    }
}

/* ================================================================== */
/*  The decomposition of one base block                                */
/* ================================================================== */

/* ================================================================== */
/*  Exact element peaks by FFT convolution                             */
/* ================================================================== */

/* Every element's own response peak, streamed: each retained axis is
 * rendered at the gradient raster -- where a uniform shape is exact as
 * written -- sliced to dgdt, convolved with the kernel (built at that
 * raster) through one forward and one inverse real FFT, and squared into a
 * running RSS; the sup of that RSS is the element's bound, its window and
 * tail sups the envelope and tails. Elements are independent, so the loop
 * runs as ranges under the caller's parallel hook, each range on scratch of
 * its own (a kissfft configuration is not shared between threads). */
typedef struct
{
    const pulseg_sequence_descriptor *desc;
    const pulseg_base_block *bdef;
    const pns_basis_block_stat *st;
    pns_basis_element *elems;
    float dur_us;
    float raster_us;
    float out_scale;
    float gamma;
    int npts;
    int nfft;
    int nfreq;
    int out_len;
    const kiss_fft_cpx *kspec; /* kernel spectrum, shared read-only */
    double norm;
    double transform_work;
    pns_basis_extent extent;
    int rc;       /* first failure any range met */
    int declined; /* a range could not render an element */
    double macs;  /* summed by ranges; a data race only miscounts */
} pns_exact_job;

static void pns_exact_range(void *arg, int begin, int end)
{
    pns_exact_job *job = (pns_exact_job *)arg;
    kiss_fftr_cfg fwd, inv;
    kiss_fft_cpx *spec;
    float *work, *dgdt, *wave;
    double *ss;
    double sup, v, macs;
    double env[PNS_SCORE_NUM_WINDOWS], tail[PNS_SCORE_NUM_ZONES];
    int e, a, slot, i, n2, def_id, z, k;

    fwd = kiss_fftr_alloc(job->nfft, 0, NULL, NULL);
    inv = kiss_fftr_alloc(job->nfft, 1, NULL, NULL);
    spec = (kiss_fft_cpx *)PULSEG_ALLOC((size_t)job->nfreq * sizeof(kiss_fft_cpx));
    work = (float *)PULSEG_ALLOC((size_t)job->nfft * sizeof(float));
    dgdt = (float *)PULSEG_ALLOC((size_t)(job->npts + 1) * sizeof(float));
    ss = (double *)PULSEG_ALLOC((size_t)job->out_len * sizeof(double));
    macs = 0.0;
    if (!fwd || !inv || !spec || !work || !dgdt || !ss)
    {
        job->rc = PULSEG_ERR_ALLOC_FAILED;
        goto done;
    }

    for (e = begin; e < end; ++e)
    {
        pns_basis_element *el = &job->elems[e];

        for (i = 0; i < job->out_len; ++i)
            ss[i] = 0.0;
        slot = 0;
        for (a = 0; a < 3; ++a)
        {
            if (!job->st->retained[a])
                continue;
            if (el->shape_id[slot] == -2)
            {
                ++slot;
                continue;
            }
            def_id = (a == 0) ? job->bdef->gx_id : (a == 1) ? job->bdef->gy_id : job->bdef->gz_id;
            wave = pns_basis_render_uniform(
                job->desc,
                def_id,
                el->shape_id[slot],
                el->amplitude[slot],
                job->dur_us,
                job->raster_us,
                &n2);
            if (!wave)
            {
                job->declined = 1;
                goto done;
            }
            for (i = 0; i < job->npts; ++i)
                work[i] = (i < n2) ? wave[i] : 0.0f;
            el->first[slot] = wave[0];
            el->last[slot] = (n2 > 0) ? wave[n2 - 1] : 0.0f;
            PULSEG_FREE(wave);
            pns_basis_slice_dgdt(dgdt, work, job->npts, job->raster_us, job->gamma);

            memset(work, 0, (size_t)job->nfft * sizeof(float));
            memcpy(work, dgdt, (size_t)(job->npts + 1) * sizeof(float));
            kiss_fftr(fwd, work, spec);
            for (i = 0; i < job->nfreq; ++i)
            {
                float re, im;
                re = spec[i].r * job->kspec[i].r - spec[i].i * job->kspec[i].i;
                im = spec[i].r * job->kspec[i].i + spec[i].i * job->kspec[i].r;
                spec[i].r = re;
                spec[i].i = im;
            }
            kiss_fftri(inv, spec, work);
            for (i = 0; i < job->out_len; ++i)
            {
                v = (double)work[i] * job->norm;
                ss[i] += v * v;
            }
            macs += 2.0 * job->transform_work + (double)job->npts;
            ++slot;
        }
        sup = 0.0;
        for (k = 0; k < PNS_SCORE_NUM_WINDOWS; ++k)
            env[k] = 0.0;
        for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
            tail[z] = 0.0;
        for (i = 0; i < job->out_len; ++i)
            pns_basis_extent_fold(&job->extent, i, ss[i], &sup, env, tail);
        el->bound = (float)(sqrt(sup) * (double)job->out_scale * (1.0 + PNS_EXACT_FFT_MARGIN));
        for (k = 0; k < PNS_SCORE_NUM_WINDOWS; ++k)
            el->env[k] =
                (float)(sqrt(env[k]) * (double)job->out_scale * (1.0 + PNS_EXACT_FFT_MARGIN));
        for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
            el->tail[z] =
                (float)(sqrt(tail[z]) * (double)job->out_scale * (1.0 + PNS_EXACT_FFT_MARGIN));
    }

done:
    job->macs += macs;
    if (fwd)
        kiss_fftr_free(fwd);
    if (inv)
        kiss_fftr_free(inv);
    if (spec)
        PULSEG_FREE(spec);
    if (work)
        PULSEG_FREE(work);
    if (dgdt)
        PULSEG_FREE(dgdt);
    if (ss)
        PULSEG_FREE(ss);
}

static int pns_basis_exact_elements(
    const pulseg_sequence_descriptor *desc,
    const pulseg_base_block *bdef,
    const pns_basis_block_stat *st,
    pns_basis_element *elems,
    int num_elems,
    float dur_us,
    float raster_us,
    const float *kernel,
    int kernel_len,
    float out_scale,
    float gamma,
    const double *zone_edge_us,
    pulseg__parallel_for_fn par_fn,
    void *par_ctx,
    pulseg__pns_basis_info *info,
    int *declined,
    double *macs)
{
    pns_exact_job job;
    kiss_fftr_cfg fwd;
    kiss_fft_cpx *kspec;
    float *work;
    int log2n, rc;

    memset(&job, 0, sizeof(job));
    job.desc = desc;
    job.bdef = bdef;
    job.st = st;
    job.elems = elems;
    job.dur_us = dur_us;
    job.raster_us = raster_us;
    job.out_scale = out_scale;
    job.gamma = gamma;
    job.npts = (int)((double)dur_us / (double)raster_us) + 1;
    pns_basis_extent_init(&job.extent, dur_us, raster_us, zone_edge_us);
    job.out_len = job.npts + 1 + kernel_len - 1;
    job.nfft = (int)pulseg__next_pow2((size_t)job.out_len);
    job.nfreq = job.nfft / 2 + 1;
    log2n = 0;
    while ((1 << log2n) < job.nfft)
        ++log2n;
    job.transform_work = 5.0 * (double)job.nfft * (double)log2n;
    job.norm = 1.0 / (double)job.nfft;
    job.rc = PULSEG_SUCCESS;

    fwd = kiss_fftr_alloc(job.nfft, 0, NULL, NULL);
    kspec = (kiss_fft_cpx *)PULSEG_ALLOC((size_t)job.nfreq * sizeof(kiss_fft_cpx));
    work = (float *)PULSEG_ALLOC((size_t)job.nfft * sizeof(float));
    if (!fwd || !kspec || !work)
    {
        rc = PULSEG_ERR_ALLOC_FAILED;
        goto done;
    }
    memset(work, 0, (size_t)job.nfft * sizeof(float));
    memcpy(work, kernel, (size_t)kernel_len * sizeof(float));
    kiss_fftr(fwd, work, kspec);
    job.kspec = kspec;

    if (par_fn)
        par_fn(par_ctx, num_elems, pns_exact_range, &job);
    else
        pns_exact_range(&job, 0, num_elems);

    rc = job.rc;
    *macs += job.macs;
    if (PULSEG_SUCCEEDED(rc) && job.declined)
        *declined = 1;
    if (PULSEG_SUCCEEDED(rc) && !job.declined)
    {
        info->num_elements = num_elems;
        info->d = st->d;
    }

done:
    if (fwd)
        kiss_fftr_free(fwd);
    if (kspec)
        PULSEG_FREE(kspec);
    if (work)
        PULSEG_FREE(work);
    return rc;
}

/* ================================================================== */
/*  The build                                                          */
/* ================================================================== */

int pulseg__pns_score_build(
    pulseg__pns_score **out,
    const pulseg_sequence_descriptor *desc,
    const pulseg_pns_model *model,
    float gamma_hz_per_tesla,
    int *out_declined,
    double *out_macs)
{
    return pulseg__pns_score_build_ex(
        out,
        desc,
        model,
        gamma_hz_per_tesla,
        NULL,
        NULL,
        out_declined,
        out_macs);
}

int pulseg__pns_score_build_ex(
    pulseg__pns_score **out,
    const pulseg_sequence_descriptor *desc,
    const pulseg_pns_model *model,
    float gamma_hz_per_tesla,
    pulseg__parallel_for_fn par_fn,
    void *par_ctx,
    int *out_declined,
    double *out_macs)
{
    pulseg__pns_score *sc;
    pns_basis_block_stat *stats;
    pns_basis_occ occ;
    pns_basis_r1 r1;
    pns_basis_elem_index elem_ix;
    pns_basis_element *elems;
    float *kernel, *kernel_e;
    float out_scale, out_scale_e, raster_us, elem_raster_us;
    int kernel_e_len;
    double macs, t, sup, term, u_sq;
    double env_sq[PNS_SCORE_NUM_WINDOWS], unit_env[PNS_SCORE_NUM_WINDOWS];
    double tail_sq[PNS_SCORE_NUM_ZONES], unit_tail[PNS_SCORE_NUM_ZONES];
    pns_basis_extent extent;
    float *edge_first, *edge_last;
    float unit_first, unit_last;
    int b, a, z, k, rc, declined, num_basis_blocks, bb, slot, kernel_len;
    int num_elems, elems_cap, found;
    int shape_ids[3];
    float amps[3];

    *out = NULL;
    *out_declined = 0;
    if (out_macs)
        *out_macs = 0.0;
    if (!desc || !model || !model->kernel || desc->num_blocks <= 0 || gamma_hz_per_tesla <= 0.0f)
    {
        *out_declined = 1;
        return PULSEG_SUCCESS;
    }

    macs = 0.0;
    declined = 0;
    kernel = NULL;
    kernel_e = NULL;
    stats = NULL;
    edge_first = NULL;
    edge_last = NULL;
    elems = NULL;
    r1.slots = NULL;
    elem_ix.slot = NULL;
    elem_ix.cap = 0;
    sc = NULL;

    raster_us = 0.5f * desc->grad_raster_us;
    rc = model->kernel(model->ctx, raster_us, &kernel, &kernel_len, &out_scale);
    if (PULSEG_FAILED(rc) || kernel_len <= 0)
    {
        if (kernel)
            PULSEG_FREE(kernel);
        *out_declined = 1;
        return PULSEG_SUCCESS;
    }
    elem_raster_us = desc->grad_raster_us;
    rc = model->kernel(model->ctx, elem_raster_us, &kernel_e, &kernel_e_len, &out_scale_e);
    if (PULSEG_FAILED(rc) || kernel_e_len <= 0)
    {
        if (kernel_e)
            PULSEG_FREE(kernel_e);
        PULSEG_FREE(kernel);
        *out_declined = 1;
        return PULSEG_SUCCESS;
    }

    sc = (pulseg__pns_score *)PULSEG_ALLOC(sizeof(*sc));
    stats = (pns_basis_block_stat *)PULSEG_ALLOC((size_t)desc->num_unique_blocks * sizeof(*stats));
    if (!sc || !stats ||
        PULSEG_FAILED(pns_basis_r1_init(&r1, desc->num_unique_grads + desc->num_shapes + 1)))
    {
        rc = PULSEG_ERR_ALLOC_FAILED;
        goto fail;
    }
    memset(sc, 0, sizeof(*sc));
    memset(stats, 0, (size_t)desc->num_unique_blocks * sizeof(*stats));

    sc->num_blocks = desc->num_blocks;
    sc->u = (float *)PULSEG_ALLOC((size_t)desc->num_blocks * sizeof(float));
    sc->env = (float *)PULSEG_ALLOC(
        (size_t)desc->num_blocks * (size_t)PNS_SCORE_NUM_WINDOWS * sizeof(float));
    sc->tail = (float *)PULSEG_ALLOC(
        (size_t)desc->num_blocks * (size_t)PNS_SCORE_NUM_ZONES * sizeof(float));
    sc->t_start_us = (double *)PULSEG_ALLOC((size_t)desc->num_blocks * sizeof(double));
    sc->t_end_us = (double *)PULSEG_ALLOC((size_t)desc->num_blocks * sizeof(double));
    edge_first = (float *)PULSEG_ALLOC((size_t)desc->num_blocks * 3 * sizeof(float));
    edge_last = (float *)PULSEG_ALLOC((size_t)desc->num_blocks * 3 * sizeof(float));
    if (!sc->u || !sc->env || !sc->tail || !sc->t_start_us || !sc->t_end_us || !edge_first ||
        !edge_last)
    {
        rc = PULSEG_ERR_ALLOC_FAILED;
        goto fail;
    }
    memset(edge_first, 0, (size_t)desc->num_blocks * 3 * sizeof(float));
    memset(edge_last, 0, (size_t)desc->num_blocks * 3 * sizeof(float));
    sc->reach_us = ((double)kernel_len + 1.0) * (double)raster_us;

    /* Zone edges where the kernel has fallen to roughly 1, 1/5, 1/20 and
     * 1/100 of its first tap: the gaps at which a tail is worth its own
     * number. */
    sc->zone_edge_us[0] = 0.0;
    sc->zone_edge_us[1] = 450.0;
    sc->zone_edge_us[2] = 1250.0;
    sc->zone_edge_us[3] = 2700.0;

    /* ---- pass 1: the timeline, and what varies where ---- */
    t = 0.0;
    for (b = 0; b < desc->num_blocks; ++b)
    {
        const pulseg_block_table_element *bte = &desc->block_table[b];
        pns_basis_block_stat *st;

        if (bte->id < 0 || bte->id >= desc->num_unique_blocks)
        {
            *out_declined = 1;
            goto decline;
        }
        sc->t_start_us[b] = t;
        t += (double)desc->base_blocks[bte->id].duration_us;
        sc->t_end_us[b] = t;
        sc->u[b] = 0.0f;
        for (k = 0; k < PNS_SCORE_NUM_WINDOWS; ++k)
            sc->env[(size_t)b * PNS_SCORE_NUM_WINDOWS + k] = 0.0f;
        for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
            sc->tail[(size_t)b * PNS_SCORE_NUM_ZONES + z] = 0.0f;

        pns_basis_resolve_occ(desc, bte, &occ);
        st = &stats[bte->id];
        if (occ.rotated)
            st->rotated = 1;
        for (a = 0; a < 3; ++a)
        {
            int shape = occ.has_grad[a] ? occ.key[a].shape_id : -2;
            if (!st->seen)
                st->first_shape[a] = shape;
            else if (st->first_shape[a] != shape)
                st->varies[a] = 1;
        }
        st->seen = 1;
    }

    /* A varying axis joins the decomposition. Under rotation every
     * gradient-bearing axis joins, because the occurrence norm must cover
     * the whole rotated vector; a multiplicity-1 axis of a never-rotated
     * block stays on the rank-1 catalogue. */
    num_basis_blocks = 0;
    for (bb = 0; bb < desc->num_unique_blocks; ++bb)
    {
        pns_basis_block_stat *st = &stats[bb];
        int nvary = 0;

        st->basis_index = -1;
        if (!st->seen)
            continue;
        for (a = 0; a < 3; ++a)
            if (st->varies[a])
                ++nvary;
        if (!nvary)
            continue;
        st->d = 0;
        for (a = 0; a < 3; ++a)
        {
            st->retained[a] = st->varies[a] || (st->rotated && st->first_shape[a] != -2);
            if (st->retained[a])
                ++st->d;
        }
        st->basis_index = num_basis_blocks++;
    }

    sc->num_bases = num_basis_blocks;
    if (num_basis_blocks > 0)
    {
        sc->bases =
            (pulseg__pns_basis_info *)PULSEG_ALLOC((size_t)num_basis_blocks * sizeof(*sc->bases));
        if (!sc->bases)
        {
            rc = PULSEG_ERR_ALLOC_FAILED;
            goto fail;
        }
        memset(sc->bases, 0, (size_t)num_basis_blocks * sizeof(*sc->bases));
    }

    /* ---- pass 2: decompose each earning base block, then score its
     * occurrences while its element table is in hand ---- */
    for (bb = 0; bb < desc->num_unique_blocks; ++bb)
    {
        pns_basis_block_stat *st = &stats[bb];
        const pulseg_base_block *bdef;
        float dur_us;

        if (st->basis_index < 0)
            continue;
        bdef = &desc->base_blocks[bb];
        dur_us = (float)bdef->duration_us;
        pns_basis_extent_init(&extent, dur_us, raster_us, sc->zone_edge_us);
        sc->bases[st->basis_index].base_id = bb;

        elems_cap = PNS_BASIS_MIN_ELEMENTS;
        elems = (pns_basis_element *)PULSEG_ALLOC((size_t)elems_cap * sizeof(*elems));
        if (!elems)
        {
            rc = PULSEG_ERR_ALLOC_FAILED;
            goto fail;
        }
        num_elems = 0;
        if (PULSEG_FAILED(pns_basis_elem_index_init(&elem_ix, PNS_BASIS_MIN_ELEMENTS)))
        {
            rc = PULSEG_ERR_ALLOC_FAILED;
            goto fail;
        }
        for (b = 0; b < desc->num_blocks; ++b)
        {
            const pulseg_block_table_element *bte = &desc->block_table[b];

            if (bte->id != bb)
                continue;
            pns_basis_resolve_occ(desc, bte, &occ);
            pns_basis_occ_tuple(&occ, st, shape_ids, amps);
            rc = pns_basis_elem_index_maybe_grow(&elem_ix, elems, num_elems, st->d);
            if (PULSEG_FAILED(rc))
                goto fail;
            found = pns_basis_elem_index_probe(&elem_ix, elems, shape_ids, amps, st->d);
            if (found >= 0)
                continue;
            elem_ix.slot[-found - 1] = num_elems;
            if (num_elems == elems_cap)
            {
                pns_basis_element *grown;

                elems_cap *= 2;
                if (elems_cap > PNS_BASIS_MAX_ELEMENTS)
                {
                    *out_declined = 1;
                    goto decline;
                }
                grown = (pns_basis_element *)PULSEG_ALLOC((size_t)elems_cap * sizeof(*grown));
                if (!grown)
                {
                    rc = PULSEG_ERR_ALLOC_FAILED;
                    goto fail;
                }
                memcpy(grown, elems, (size_t)num_elems * sizeof(*grown));
                PULSEG_FREE(elems);
                elems = grown;
            }
            memcpy(elems[num_elems].shape_id, shape_ids, sizeof(shape_ids));
            memcpy(elems[num_elems].amplitude, amps, sizeof(amps));
            elems[num_elems].bound = 0.0f;
            for (k = 0; k < PNS_SCORE_NUM_WINDOWS; ++k)
                elems[num_elems].env[k] = 0.0f;
            for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
                elems[num_elems].tail[z] = 0.0f;
            for (a = 0; a < 3; ++a)
            {
                elems[num_elems].first[a] = 0.0f;
                elems[num_elems].last[a] = 0.0f;
            }
            ++num_elems;
        }

        if (num_elems < PNS_BASIS_MIN_ELEMENTS)
        {
            /* Too few to pay for a factorisation; the rank-1 catalogue
             * carries this block, one entry per shape it plays. */
            sc->bases[st->basis_index].num_elements = num_elems;
            st->basis_index = -1;
            PULSEG_FREE(elems);
            elems = NULL;
            pns_basis_elem_index_destroy(&elem_ix);
            continue;
        }

        /* Every element priced exactly, one at a time. */
        rc = pns_basis_exact_elements(
            desc,
            bdef,
            st,
            elems,
            num_elems,
            dur_us,
            elem_raster_us,
            kernel_e,
            kernel_e_len,
            out_scale_e,
            gamma_hz_per_tesla,
            sc->zone_edge_us,
            par_fn,
            par_ctx,
            &sc->bases[st->basis_index],
            &declined,
            &macs);
        if (PULSEG_FAILED(rc))
            goto fail;
        if (declined)
        {
            *out_declined = 1;
            goto decline;
        }

        for (b = 0; b < desc->num_blocks; ++b)
        {
            const pulseg_block_table_element *bte = &desc->block_table[b];

            if (bte->id != bb)
                continue;
            pns_basis_resolve_occ(desc, bte, &occ);
            pns_basis_occ_tuple(&occ, st, shape_ids, amps);
            found = pns_basis_elem_index_probe(&elem_ix, elems, shape_ids, amps, st->d);
            if (found < 0)
            {
                *out_declined = 1;
                goto decline;
            }
            u_sq = (double)elems[found].bound * (double)elems[found].bound;
            for (k = 0; k < PNS_SCORE_NUM_WINDOWS; ++k)
                env_sq[k] = (double)elems[found].env[k] * (double)elems[found].env[k];
            for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
                tail_sq[z] = (double)elems[found].tail[z] * (double)elems[found].tail[z];
            slot = 0;
            for (a = 0; a < 3; ++a)
            {
                if (!st->retained[a])
                    continue;
                edge_first[3 * b + a] = elems[found].first[slot];
                edge_last[3 * b + a] = elems[found].last[slot];
                ++slot;
            }
            for (a = 0; a < 3; ++a)
            {
                if (st->retained[a] || !occ.has_grad[a])
                    continue;
                rc = pns_basis_r1_get(
                    &r1,
                    desc,
                    &occ.key[a],
                    dur_us,
                    raster_us,
                    gamma_hz_per_tesla,
                    kernel,
                    kernel_len,
                    out_scale,
                    sc->zone_edge_us,
                    &sup,
                    unit_env,
                    unit_tail,
                    &unit_first,
                    &unit_last,
                    &macs);
                if (PULSEG_FAILED(rc))
                    goto fail;
                edge_first[3 * b + a] = unit_first * occ.amplitude[a];
                edge_last[3 * b + a] = unit_last * occ.amplitude[a];
                term = fabs((double)occ.amplitude[a]) * sup;
                u_sq += term * term;
                for (k = 0; k < PNS_SCORE_NUM_WINDOWS; ++k)
                {
                    term = fabs((double)occ.amplitude[a]) * unit_env[k];
                    env_sq[k] += term * term;
                }
                for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
                {
                    term = fabs((double)occ.amplitude[a]) * unit_tail[z];
                    tail_sq[z] += term * term;
                }
            }
            sc->u[b] = (float)sqrt(u_sq);
            for (k = 0; k < PNS_SCORE_NUM_WINDOWS; ++k)
                sc->env[(size_t)b * PNS_SCORE_NUM_WINDOWS + k] = (float)sqrt(env_sq[k]);
            for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
                sc->tail[(size_t)b * PNS_SCORE_NUM_ZONES + z] = (float)sqrt(tail_sq[z]);
        }

        PULSEG_FREE(elems);
        elems = NULL;
        pns_basis_elem_index_destroy(&elem_ix);
    }

    /* ---- pass 3: every block without a basis is pure rank-1 ---- */
    for (b = 0; b < desc->num_blocks; ++b)
    {
        const pulseg_block_table_element *bte = &desc->block_table[b];

        if (stats[bte->id].basis_index >= 0)
            continue;
        pns_basis_resolve_occ(desc, bte, &occ);
        u_sq = 0.0;
        for (k = 0; k < PNS_SCORE_NUM_WINDOWS; ++k)
            env_sq[k] = 0.0;
        for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
            tail_sq[z] = 0.0;
        for (a = 0; a < 3; ++a)
        {
            if (!occ.has_grad[a])
                continue;
            rc = pns_basis_r1_get(
                &r1,
                desc,
                &occ.key[a],
                (float)desc->base_blocks[bte->id].duration_us,
                raster_us,
                gamma_hz_per_tesla,
                kernel,
                kernel_len,
                out_scale,
                sc->zone_edge_us,
                &sup,
                unit_env,
                unit_tail,
                &unit_first,
                &unit_last,
                &macs);
            if (PULSEG_FAILED(rc))
                goto fail;
            edge_first[3 * b + a] = unit_first * occ.amplitude[a];
            edge_last[3 * b + a] = unit_last * occ.amplitude[a];
            term = fabs((double)occ.amplitude[a]) * sup;
            u_sq += term * term;
            for (k = 0; k < PNS_SCORE_NUM_WINDOWS; ++k)
            {
                term = fabs((double)occ.amplitude[a]) * unit_env[k];
                env_sq[k] += term * term;
            }
            for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
            {
                term = fabs((double)occ.amplitude[a]) * unit_tail[z];
                tail_sq[z] += term * term;
            }
        }
        sc->u[b] = (float)sqrt(u_sq);
        for (k = 0; k < PNS_SCORE_NUM_WINDOWS; ++k)
            sc->env[(size_t)b * PNS_SCORE_NUM_WINDOWS + k] = (float)sqrt(env_sq[k]);
        for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
            sc->tail[(size_t)b * PNS_SCORE_NUM_ZONES + z] = (float)sqrt(tail_sq[z]);
    }

    /* ---- pass 4: the steps between blocks, one kernel tap each ---- */
    {
        double step, d, tap_scale, dur;
        int idx;

        tap_scale = (double)out_scale / ((double)gamma_hz_per_tesla * ((double)raster_us * 1e-6));
        for (b = 0; b < desc->num_blocks; ++b)
        {
            step = 0.0;
            for (a = 0; a < 3; ++a)
            {
                d = (double)edge_first[3 * b + a] -
                    (b > 0 ? (double)edge_last[3 * (b - 1) + a] : 0.0);
                step += d * d;
            }
            if (b == desc->num_blocks - 1)
            {
                /* The scan's closing step, on its last block. */
                d = 0.0;
                for (a = 0; a < 3; ++a)
                    d += (double)edge_last[3 * b + a] * (double)edge_last[3 * b + a];
                step = sqrt(step) + sqrt(d);
            }
            else
                step = sqrt(step);
            if (step <= 0.0)
                continue;
            step *= tap_scale;
            dur = sc->t_end_us[b] - sc->t_start_us[b];
            sc->u[b] = (float)((double)sc->u[b] + step * (double)kernel[0]);
            for (k = 0; k < PNS_SCORE_NUM_WINDOWS; ++k)
            {
                idx =
                    (int)((double)k * dur / (double)PNS_SCORE_NUM_WINDOWS / (double)raster_us) - 1;
                if (idx < 0)
                    idx = 0;
                if (idx >= kernel_len)
                    continue;
                sc->env[(size_t)b * PNS_SCORE_NUM_WINDOWS + k] =
                    (float)((double)sc->env[(size_t)b * PNS_SCORE_NUM_WINDOWS + k] + step * (double)kernel[idx]);
            }
            for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
            {
                idx = (int)((dur + sc->zone_edge_us[z]) / (double)raster_us) - 1;
                if (idx < 0)
                    idx = 0;
                if (idx >= kernel_len)
                    continue;
                sc->tail[(size_t)b * PNS_SCORE_NUM_ZONES + z] =
                    (float)((double)sc->tail[(size_t)b * PNS_SCORE_NUM_ZONES + z] + step * (double)kernel[idx]);
            }
        }
    }

    sc->build_macs = macs;
    if (out_macs)
        *out_macs = macs;
    PULSEG_FREE(kernel);
    PULSEG_FREE(kernel_e);
    PULSEG_FREE(edge_first);
    PULSEG_FREE(edge_last);
    PULSEG_FREE(stats);
    pns_basis_r1_destroy(&r1);
    *out = sc;
    return PULSEG_SUCCESS;

decline:
    rc = PULSEG_SUCCESS;
fail:
    if (kernel)
        PULSEG_FREE(kernel);
    if (kernel_e)
        PULSEG_FREE(kernel_e);
    if (edge_first)
        PULSEG_FREE(edge_first);
    if (edge_last)
        PULSEG_FREE(edge_last);
    if (stats)
        PULSEG_FREE(stats);
    pns_basis_r1_destroy(&r1);
    pns_basis_elem_index_destroy(&elem_ix);
    if (elems)
        PULSEG_FREE(elems);
    pulseg__pns_score_free(sc);
    if (out_macs)
        *out_macs = macs;
    return rc;
}

/* ================================================================== */
/*  The sweep                                                          */
/* ================================================================== */

/* One anchor window at a time, occurrences migrating outward through the
 * gap zones as the anchor advances. Anchor (j, k) covers the k-th window of
 * block j's span: block j itself is priced by its envelope there, and every
 * earlier occurrence whose response still reaches the window by its tail
 * peak at its zone's edge, the gap measured to the window's start. Every
 * occurrence enters zone 0 after its own last window and crosses each
 * boundary once, so a sweep over the scan is O(windows * zones * blocks). */
typedef struct
{
    double sums[PNS_SCORE_NUM_ZONES];
    int q[PNS_SCORE_NUM_ZONES]; /* first member of zones >= z */
} pns_score_sweep;

static double pns_score_zone_price(const pulseg__pns_score *sc, int i, int z)
{
    return (double)sc->tail[(size_t)i * PNS_SCORE_NUM_ZONES + z];
}

static void pns_score_sweep_init(pns_score_sweep *sw)
{
    int z;

    for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
    {
        sw->sums[z] = 0.0;
        sw->q[z] = 0;
    }
}

/* Advance to block j (callers pass j = 0, 1, ...) and return the largest
 * covering sum over its windows. */
static double pns_score_sweep_step(const pulseg__pns_score *sc, pns_score_sweep *sw, int j)
{
    double start, cut, total, best, span;
    int z, i, k;

    best = 0.0;
    span = sc->t_end_us[j] - sc->t_start_us[j];
    for (k = 0; k < PNS_SCORE_NUM_WINDOWS; ++k)
    {
        start = sc->t_start_us[j] + span * (double)k / (double)PNS_SCORE_NUM_WINDOWS;
        for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
        {
            cut = (z + 1 < PNS_SCORE_NUM_ZONES) ? start - sc->zone_edge_us[z + 1]
                                                : start - sc->reach_us;
            for (i = sw->q[z]; i < j && sc->t_end_us[i] <= cut; ++i)
            {
                sw->sums[z] -= pns_score_zone_price(sc, i, z);
                if (z + 1 < PNS_SCORE_NUM_ZONES)
                    sw->sums[z + 1] += pns_score_zone_price(sc, i, z + 1);
            }
            sw->q[z] = i;
        }

        total = (double)sc->env[(size_t)j * PNS_SCORE_NUM_WINDOWS + k];
        for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
            total += sw->sums[z];
        if (total > best)
            best = total;
    }

    /* From the next block on, j is an earlier occurrence in zone 0. */
    sw->sums[0] += pns_score_zone_price(sc, j, 0);
    return best;
}

int pulseg__pns_score_evaluate(
    const pulseg__pns_score *sc,
    float *out_max,
    int *out_argmax_block,
    double *out_macs)
{
    pns_score_sweep sw;
    double best, cover;
    int j, best_j;

    if (!sc || !out_max)
        return PULSEG_ERR_NULL_POINTER;

    pns_score_sweep_init(&sw);
    best = 0.0;
    best_j = 0;
    for (j = 0; j < sc->num_blocks; ++j)
    {
        cover = pns_score_sweep_step(sc, &sw, j);
        if (cover > best)
        {
            best = cover;
            best_j = j;
        }
    }

    *out_max = (float)best;
    if (out_argmax_block)
        *out_argmax_block = best_j;
    if (out_macs)
        *out_macs =
            (double)sc->num_blocks * (double)PNS_SCORE_NUM_ZONES * (double)PNS_SCORE_NUM_WINDOWS;
    return PULSEG_SUCCESS;
}

/* Anchors whose covering sum exceeds the threshold, merged into
 * (start, count) block ranges. Each range opens at least one kernel reach
 * of real blocks before its first offending anchor (or at the scan head),
 * so a cold evaluation of the range reproduces the true scan response at
 * every offending instant: contributors older than that cannot reach them,
 * and the under-warmed opening of the window only ever affects instants
 * whose covering sum already cleared the threshold. Writes up to max_ranges
 * (start, count, judge_from) triples: the verdict is read from `judge_from`
 * -- the range's first offending anchor -- onward, the region where the
 * range provably reproduces the whole scan, while the warm-up before it is
 * rendered but never judged. */
int pulseg__pns_score_offenders(
    const pulseg__pns_score *sc,
    float threshold,
    int max_ranges,
    int *out_ranges,
    int *out_num_ranges)
{
    pns_score_sweep sw;
    double cover;
    int j, n, cur_start, cur_end, cur_judge, first;

    if (!sc || !out_ranges || !out_num_ranges || max_ranges <= 0)
        return PULSEG_ERR_NULL_POINTER;

    pns_score_sweep_init(&sw);
    n = 0;
    cur_start = -1;
    cur_end = -1;
    cur_judge = -1;
    for (j = 0; j < sc->num_blocks; ++j)
    {
        cover = pns_score_sweep_step(sc, &sw, j);
        if (cover <= (double)threshold)
            continue;
        first = j;
        while (first > 0 && sc->t_start_us[first] > sc->t_start_us[j] - sc->reach_us)
            --first;
        if (cur_start >= 0 && first <= cur_end + 1 &&
            sc->t_end_us[j] - sc->t_start_us[cur_start] <= PNS_SCORE_RANGE_BUDGET_US)
        {
            if (j > cur_end)
                cur_end = j;
        }
        else
        {
            if (cur_start >= 0)
            {
                if (n == max_ranges)
                    return PULSEG_ERR_INVALID_ARGUMENT;
                out_ranges[3 * n] = cur_start;
                out_ranges[3 * n + 1] = cur_end - cur_start + 1;
                out_ranges[3 * n + 2] = cur_judge;
                ++n;
            }
            cur_start = first;
            cur_end = j;
            cur_judge = j;
        }
    }
    if (cur_start >= 0)
    {
        if (n == max_ranges)
            return PULSEG_ERR_INVALID_ARGUMENT;
        out_ranges[3 * n] = cur_start;
        out_ranges[3 * n + 1] = cur_end - cur_start + 1;
        out_ranges[3 * n + 2] = cur_judge;
        ++n;
    }
    *out_num_ranges = n;
    return PULSEG_SUCCESS;
}

/* ================================================================== */
/*  The check: score, then exact assembly where the score cannot say   */
/* ================================================================== */

/* Offender ranges evaluated exactly before a refusal is allowed. A scan
 * whose bound exceeds the threshold in more regions than this is refused on
 * the bound alone, with the count in the diagnostic. */
#define PNS_SCORE_MAX_OFFENDER_RANGES 64

/* Cold exact evaluation of one block range at the amplitudes it actually
 * plays: extract through the shared window machinery, difference, run the
 * model from silence. The range opens a kernel reach before the instants it
 * exists to judge, so the response there equals the whole scan's. */
int pulseg__pns_exact_range_peak(
    pulseg_check_plan *plan,
    pulseg_diagnostic *diag,
    const pulseg_sequence_descriptor *desc,
    int subseq_idx,
    int start,
    int count,
    double judge_from_us,
    const pulseg_pns_model *model,
    float gamma,
    double *out_peak)
{
    const pulseg__uniform_grad_waveforms *uw;
    const float *wavep[3];
    float *dgdt[3], *resp[3];
    double peak, ss, inv;
    int n, i, a, rc;

    *out_peak = -1.0;
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
    i = (int)(judge_from_us / (double)uw->raster_us);
    if (i < 0)
        i = 0;
    for (; i < n + 1; ++i)
    {
        ss = sqrt(
            (double)resp[0][i] * resp[0][i] + (double)resp[1][i] * resp[1][i] +
            (double)resp[2][i] * resp[2][i]);
        if (ss > peak)
            peak = ss;
    }
    *out_peak = peak;
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

int pulseg__pns_score_check(
    pulseg_check_plan *plan,
    pulseg_diagnostic *diag,
    const pulseg_sequence_descriptor *desc,
    int subseq_idx,
    const pulseg_pns_model *model,
    pulseg__parallel_for_fn par_fn,
    void *par_ctx,
    float gamma_hz_per_tesla,
    float threshold_percent,
    int *out_declined,
    double *out_build_macs,
    double *out_eval_macs)
{
    pulseg__pns_score *score;
    float score_max;
    double peak, eval_macs, judge_us;
    int ranges[3 * PNS_SCORE_MAX_OFFENDER_RANGES];
    int num_ranges, r, rc, argmax;

    *out_declined = 0;
    if (out_build_macs)
        *out_build_macs = 0.0;
    if (out_eval_macs)
        *out_eval_macs = 0.0;

    score = NULL;
    rc = pulseg__pns_score_build_ex(
        &score,
        desc,
        model,
        gamma_hz_per_tesla,
        par_fn,
        par_ctx,
        out_declined,
        out_build_macs);
    if (PULSEG_FAILED(rc) || *out_declined)
        return rc;

    rc = pulseg__pns_score_evaluate(score, &score_max, &argmax, &eval_macs);
    if (PULSEG_FAILED(rc))
        goto done;
    if (out_eval_macs)
        *out_eval_macs = eval_macs;

    if ((double)score_max <= (double)threshold_percent)
    {
        rc = PULSEG_SUCCESS;
        goto done;
    }

    rc = pulseg__pns_score_offenders(
        score,
        threshold_percent,
        PNS_SCORE_MAX_OFFENDER_RANGES,
        ranges,
        &num_ranges);
    if (PULSEG_FAILED(rc))
    {
        if (diag)
        {
            pulseg__diag_printf(
                diag,
                "PNS response bound %.1f%% exceeds %.1f%% in more than %d regions",
                (double)score_max,
                (double)threshold_percent,
                PNS_SCORE_MAX_OFFENDER_RANGES);
            diag->code = PULSEG_ERR_PNS_THRESHOLD_EXCEEDED;
        }
        rc = PULSEG_ERR_PNS_THRESHOLD_EXCEEDED;
        goto done;
    }

    for (r = 0; r < num_ranges; ++r)
    {
        judge_us = pulseg__pns_score_block_start_us(score, ranges[3 * r + 2]) -
            pulseg__pns_score_block_start_us(score, ranges[3 * r]);
        rc = pulseg__pns_exact_range_peak(
            plan,
            diag,
            desc,
            subseq_idx,
            ranges[3 * r],
            ranges[3 * r + 1],
            judge_us,
            model,
            gamma_hz_per_tesla,
            &peak);
        if (PULSEG_FAILED(rc))
            goto done;
        if (out_eval_macs)
            *out_eval_macs += (double)ranges[3 * r + 1];
        if (peak > (double)threshold_percent)
        {
            if (diag)
            {
                pulseg__diag_printf(
                    diag,
                    "PNS threshold exceeded (peak %.1f%% > %.1f%%, blocks %d..%d)",
                    peak,
                    (double)threshold_percent,
                    ranges[3 * r],
                    ranges[3 * r] + ranges[3 * r + 1] - 1);
                diag->code = PULSEG_ERR_PNS_THRESHOLD_EXCEEDED;
            }
            rc = PULSEG_ERR_PNS_THRESHOLD_EXCEEDED;
            goto done;
        }
    }
    rc = PULSEG_SUCCESS;

done:
    pulseg__pns_score_free(score);
    return rc;
}
