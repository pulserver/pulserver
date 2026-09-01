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
 * instant, by a sum over the occurrences that have already started: each
 * contributes at most its own response peak u_i = sup_tau ||r_i(tau)||_2,
 * and one that ended a gap delta earlier contributes at most
 * L_i * K(delta) -- its slew's l1 mass through the largest kernel tap still
 * reachable, since every tap the convolution touches at that distance is
 * K(delta) or smaller. The slide takes min(u_i, L_i K(delta)) over four
 * kernel-decay zones, so a block several chronaxies back is priced by the
 * tail it actually reaches rather than by its peak. Two numbers per
 * occurrence and a monotone sweep -- no window, no grouping, no cap on how
 * many distinct waveforms the scan plays.
 *
 * u_i is bounded from two catalogues, both priced per *distinct* object
 * rather than per occurrence:
 *
 *  - a rank-1 catalogue: sup_tau |k * d(unit waveform)/dt| once per
 *    (definition, shape) identity, scaled by |amplitude| at score time;
 *  - a rank basis per base block whose occurrences play many distinct
 *    waveform tuples: the (d, npts) elements -- one per distinct
 *    (shape, amplitude) tuple the block plays, at the amplitudes it really
 *    plays -- are decomposed jointly, and each element's response peak is
 *    bounded by sum_r |c_r| P_r plus its reconstruction residual under a
 *    closed-form kernel gain. The residual is *added*, so a poor basis
 *    loosens the bound and never unsounds it -- which is also why rank
 *    selection needs no criterion: more rank only tightens, so the rank is
 *    simply min(elements, columns, PULSEG__PNS_BASIS_MAX_RANK).
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

#include "external_svd.h"
#include "pulseg_internal.h"
#include "pulseg_pns_models.h"

/* Basis rank ceiling. Also the count of template convolutions a basis pays,
 * so it bounds the build cost; past it the residual term carries the rest. */
#define PULSEG__PNS_BASIS_MAX_RANK 64

/* A base block earns a decomposition once this many distinct element tuples
 * appear at that position; below it the rank-1 catalogue is already tight
 * (per element, the two bounds coincide at rank == elements). */
#define PNS_BASIS_MIN_ELEMENTS 4

/* Distinct (shape, amplitude) tuples one base block may play before the
 * module declines the scan to the sweep. */
#define PNS_BASIS_MAX_ELEMENTS 262144

/* Flop ceiling for one decomposition (m * n * min(m,n); Golub-Reinsch is a
 * small constant above it); past it the module declines. */
#define PNS_BASIS_MAX_SVD_WORK 4.0e9

/* Bytes the rendered element matrix may occupy; past it the module declines
 * before a single row is rendered. */
#define PNS_BASIS_MAX_ROWS_BYTES 268435456.0

/* Trailing singular directions below this fraction of the leading one carry
 * no bound worth a template convolution; their mass moves to the residual
 * term instead, which is sound by construction. */
#define PNS_BASIS_SIGMA_FLOOR 1e-7

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

/* Kernel-decay zones for the slide: an occurrence whose end sits delta
 * before the anchor is priced min(u, L * K(zone edge)), the edge being the
 * zone's smallest delta. Zone 0 starts at zero gap, where the l1 price
 * L * K(0) is never below u, so u itself is the zone-0 price. */
#define PNS_SCORE_NUM_ZONES 4

struct pulseg__pns_score
{
    int num_blocks;
    float *u;           /* [num_blocks] per-occurrence response-peak bound, % */
    float *l1;          /* [num_blocks] slew l1 mass through the kernel gain  */
    double *t_start_us; /* [num_blocks] block start on the scan timeline      */
    double *t_end_us;   /* [num_blocks] block end                             */
    double reach_us;    /* kernel memory: supports span [start, end + reach]  */
    double zone_edge_us[PNS_SCORE_NUM_ZONES]; /* smallest gap of each zone    */
    double zone_gain[PNS_SCORE_NUM_ZONES];    /* K(edge) * scale, percent per
                                                 unit of l1 mass              */
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
    if (sc->l1)
        PULSEG_FREE(sc->l1);
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

/* Slice forward difference against the model's convention: n+1 taps opening
 * with +w[0] (the step up from zero) and closing with -w[last], each over
 * gamma * dt. Writes into dgdt[n+1]. */
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
    dgdt[0] = (float)((double)wave[0] * inv);
    for (i = 1; i < n; ++i)
        dgdt[i] = (float)(((double)wave[i] - (double)wave[i - 1]) * inv);
    dgdt[n] = (float)(-(double)wave[n - 1] * inv);
}

/* RSS over axes of each axis's slew l1 mass: the price of the closed-form
 * tail bound |r_a(t)| <= K(gap) * sum |dgdt_a|. */
static double pns_basis_rows_l1(const float *rows, int d, int len)
{
    double ss, acc;
    int a, j;

    ss = 0.0;
    for (a = 0; a < d; ++a)
    {
        acc = 0.0;
        for (j = 0; j < len; ++j)
            acc += fabs((double)rows[(size_t)a * len + j]);
        ss += acc * acc;
    }
    return sqrt(ss);
}

/* sup_tau of the d-axis RSS response of dgdt rows of `len` taps each. */
static double pns_basis_response_sup(
    const float *rows,
    int d,
    int len,
    const float *kernel,
    int kernel_len,
    float out_scale,
    double *macs)
{
    double sup, acc, ss;
    int m, j, lo, hi, out_len, a;

    sup = 0.0;
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
        if (ss > sup)
            sup = ss;
        *macs += (double)d * (double)(hi - lo + 1);
    }
    return sqrt(sup) * (double)out_scale;
}

/* Closed-form ceiling on the response of a waveform known only by its
 * sample-domain l2 norm: by summation by parts the response is the waveform
 * convolved with the kernel's own difference, so Cauchy-Schwarz gives
 * sup |k * dw/dt| <= ||w||_2 * ||delta k||_2 / (gamma dt), the edge taps of
 * delta k standing for the zero extension on both sides. RSS across axes
 * follows with the Frobenius norm of the multi-axis residual. */
static double pns_basis_residual_gain(
    const float *kernel,
    int kernel_len,
    float out_scale,
    float raster_us,
    float gamma)
{
    double ss, d;
    int i;

    ss = (double)kernel[0] * (double)kernel[0];
    for (i = 1; i < kernel_len; ++i)
    {
        d = (double)kernel[i] - (double)kernel[i - 1];
        ss += d * d;
    }
    ss += (double)kernel[kernel_len - 1] * (double)kernel[kernel_len - 1];
    return sqrt(ss) * (double)out_scale / ((double)gamma * ((double)raster_us * 1e-6));
}

/* ================================================================== */
/*  Rank-1 catalogue: sup of the unit response per (definition, shape)  */
/* ================================================================== */

typedef struct
{
    int def_id;
    int shape_id;
    double sup;
    double l1; /* slew l1 mass of the unit waveform */
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
    double *out_sup,
    double *out_l1,
    double *macs)
{
    pns_basis_r1_entry *slot;
    float *wave, *dgdt;
    unsigned h;
    int n;

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
        *out_l1 = slot->l1;
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
    slot->sup = pns_basis_response_sup(dgdt, 1, n + 1, kernel, kernel_len, out_scale, macs);
    slot->l1 = pns_basis_rows_l1(dgdt, 1, n + 1);
    slot->def_id = key->def_id;
    slot->shape_id = key->shape_id;
    PULSEG_FREE(wave);
    PULSEG_FREE(dgdt);
    *macs += (double)n;
    *out_sup = slot->sup;
    *out_l1 = slot->l1;
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
    float bound; /* sum_r |c_r| P_r + residual * gain, in percent */
    float l1;    /* slew l1 mass of the element's own dgdt rows   */
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

static int pns_basis_decompose(
    const pns_basis_block_stat *st,
    pns_basis_element *elems,
    int num_elems,
    float *rows, /* [num_elems * d * npts], consumed and scaled in place */
    int npts,
    const float *kernel,
    int kernel_len,
    float out_scale,
    float raster_us,
    float gamma,
    double residual_gain,
    pulseg__pns_basis_info *info,
    int *declined,
    double *macs)
{
    float *u, *s, *v, *dgdt;
    double c_abs, resid, resid_sq, sup, scale, inv_scale;
    size_t total;
    int m, n, k, rank, e, r, j, a, rc;

    m = num_elems;
    n = st->d * npts;
    k = (m < n) ? m : n;

    if ((double)m * (double)n * (double)k > PNS_BASIS_MAX_SVD_WORK)
    {
        *declined = 1;
        return PULSEG_SUCCESS;
    }

    u = (float *)PULSEG_ALLOC((size_t)m * (size_t)k * sizeof(float));
    s = (float *)PULSEG_ALLOC((size_t)k * sizeof(float));
    v = (float *)PULSEG_ALLOC((size_t)n * (size_t)k * sizeof(float));
    dgdt = (float *)PULSEG_ALLOC((size_t)st->d * (size_t)(npts + 1) * sizeof(float));
    if (!u || !s || !v || !dgdt)
    {
        rc = PULSEG_ERR_ALLOC_FAILED;
        goto done;
    }

    /* Condition the factorisation: amplitudes are Hz/m, so raw rows sit at
     * 1e6 and float singular vectors lose digits. The common scale divides
     * out of the rows and multiplies back into every bound, so it changes
     * nothing but the arithmetic's footing. */
    scale = 0.0;
    total = (size_t)m * (size_t)n;
    {
        size_t i;
        double av;
        for (i = 0; i < total; ++i)
        {
            av = fabs((double)rows[i]);
            if (av > scale)
                scale = av;
        }
    }
    if (scale <= 0.0)
        scale = 1.0;
    inv_scale = 1.0 / scale;
    {
        size_t i;
        for (i = 0; i < total; ++i)
            rows[i] = (float)((double)rows[i] * inv_scale);
    }

    rc = svd_decompose(rows, (size_t)m, (size_t)n, u, s, v);
    *macs += (double)m * (double)n * (double)k;
    if (rc != SVD_OK)
    {
        rc = PULSEG_SUCCESS;
        *declined = 1;
        goto done;
    }
    rc = PULSEG_SUCCESS;

    rank = (k < PULSEG__PNS_BASIS_MAX_RANK) ? k : PULSEG__PNS_BASIS_MAX_RANK;
    while (rank > 1 && (double)s[rank - 1] < PNS_BASIS_SIGMA_FLOOR * (double)s[0])
        --rank;

    info->num_elements = m;
    info->d = st->d;
    info->rank = rank;
    info->max_residual = 0.0f;

    /* One template response peak per kept direction. The basis vector is d
     * axis rows of npts samples; its dgdt rows follow the same slice
     * convention as every other waveform here. */
    for (r = 0; r < rank; ++r)
    {
        for (a = 0; a < st->d; ++a)
        {
            float prev, cur;
            prev = 0.0f;
            for (j = 0; j < npts; ++j)
            {
                cur = v[((size_t)a * npts + j) * (size_t)k + r];
                dgdt[a * (npts + 1) + j] =
                    (float)(((double)cur - (double)prev) / ((double)gamma * ((double)raster_us * 1e-6)));
                prev = cur;
            }
            dgdt[a * (npts + 1) + npts] =
                (float)(-(double)prev / ((double)gamma * ((double)raster_us * 1e-6)));
        }
        sup = pns_basis_response_sup(dgdt, st->d, npts + 1, kernel, kernel_len, out_scale, macs);

        for (e = 0; e < m; ++e)
        {
            c_abs = (double)s[r] * (double)u[(size_t)e * k + r];
            if (c_abs < 0.0)
                c_abs = -c_abs;
            elems[e].bound = (float)((double)elems[e].bound + c_abs * sup * scale);
        }
    }

    /* Each element's reconstruction residual, exact from the factorisation:
     * the discarded directions are orthonormal, so a row's residual is
     * sqrt(sum_{r >= rank} (sigma_r u_{e,r})^2). Added under the closed-form
     * gain, so truncation loosens the bound and never unsounds it. (A
     * factorisation that is itself approximate must instead measure the
     * residual by explicit reconstruction; this one is exact to float.) */
    for (e = 0; e < m; ++e)
    {
        resid_sq = 0.0;
        for (r = rank; r < k; ++r)
        {
            c_abs = (double)s[r] * (double)u[(size_t)e * k + r];
            resid_sq += c_abs * c_abs;
        }
        resid = sqrt(resid_sq) * scale;
        if ((float)resid > info->max_residual)
            info->max_residual = (float)resid;
        elems[e].bound = (float)((double)elems[e].bound + resid * residual_gain);
    }

done:
    if (u)
        PULSEG_FREE(u);
    if (s)
        PULSEG_FREE(s);
    if (v)
        PULSEG_FREE(v);
    if (dgdt)
        PULSEG_FREE(dgdt);
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
    pulseg__pns_score *sc;
    pns_basis_block_stat *stats;
    pns_basis_occ occ;
    pns_basis_r1 r1;
    pns_basis_elem_index elem_ix;
    pns_basis_element *elems;
    float *kernel, *rows, *wave;
    float out_scale, raster_us;
    double macs, residual_gain, t, sup, term, u_sq, l1_sq;
    int b, a, rc, declined, num_basis_blocks, bb, slot, kernel_len;
    int num_elems, elems_cap, npts, e_idx, found, n2, def_id;
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
    stats = NULL;
    elems = NULL;
    rows = NULL;
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
    residual_gain =
        pns_basis_residual_gain(kernel, kernel_len, out_scale, raster_us, gamma_hz_per_tesla);

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
    sc->l1 = (float *)PULSEG_ALLOC((size_t)desc->num_blocks * sizeof(float));
    sc->t_start_us = (double *)PULSEG_ALLOC((size_t)desc->num_blocks * sizeof(double));
    sc->t_end_us = (double *)PULSEG_ALLOC((size_t)desc->num_blocks * sizeof(double));
    if (!sc->u || !sc->l1 || !sc->t_start_us || !sc->t_end_us)
    {
        rc = PULSEG_ERR_ALLOC_FAILED;
        goto fail;
    }
    sc->reach_us = ((double)kernel_len + 1.0) * (double)raster_us;

    /* Zone edges chosen where the kernel has fallen to roughly 1, 1/5, 1/20
     * and 1/100 of its first tap; each zone's gain is the tap at its own
     * (smallest-gap) edge, the largest any member's convolution can touch. */
    sc->zone_edge_us[0] = 0.0;
    sc->zone_edge_us[1] = 450.0;
    sc->zone_edge_us[2] = 1250.0;
    sc->zone_edge_us[3] = 2700.0;
    {
        int z, idx;
        for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
        {
            idx = (int)(sc->zone_edge_us[z] / (double)raster_us);
            if (idx >= kernel_len)
                idx = kernel_len - 1;
            sc->zone_gain[z] = (double)kernel[idx] * (double)out_scale;
        }
    }

    /* ---- pass 1: the timeline, and what varies where ---- */
    t = 0.0;
    for (b = 0; b < desc->num_blocks; ++b)
    {
        const pulseg_block_table_element *bte = &desc->block_table[b];
        pns_basis_block_stat *st;

        sc->t_start_us[b] = t;
        t += (double)bte->duration_us;
        sc->t_end_us[b] = t;
        sc->u[b] = 0.0f;
        sc->l1[b] = 0.0f;

        if (bte->id < 0 || bte->id >= desc->num_unique_blocks)
        {
            *out_declined = 1;
            goto decline;
        }
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
        npts = (int)((double)dur_us / (double)raster_us) + 1;
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

        /* Ceilings before any row is rendered: a scan the decomposition will
         * refuse anyway must decline in the time it takes to say so. */
        {
            double dm = (double)num_elems;
            double dn = (double)st->d * (double)npts;
            double dk = (dm < dn) ? dm : dn;

            if (dm * dn * dk > PNS_BASIS_MAX_SVD_WORK || dm * dn * 4.0 > PNS_BASIS_MAX_ROWS_BYTES)
            {
                *out_declined = 1;
                goto decline;
            }
        }

        rows =
            (float *)PULSEG_ALLOC((size_t)num_elems * (size_t)st->d * (size_t)npts * sizeof(float));
        if (!rows)
        {
            rc = PULSEG_ERR_ALLOC_FAILED;
            goto fail;
        }
        for (e_idx = 0; e_idx < num_elems; ++e_idx)
        {
            slot = 0;
            for (a = 0; a < 3; ++a)
            {
                float *dst;
                int i2;

                if (!st->retained[a])
                    continue;
                dst = rows + ((size_t)e_idx * st->d + slot) * (size_t)npts;
                if (elems[e_idx].shape_id[slot] == -2)
                {
                    memset(dst, 0, (size_t)npts * sizeof(float));
                    ++slot;
                    continue;
                }
                def_id = (a == 0) ? bdef->gx_id : (a == 1) ? bdef->gy_id : bdef->gz_id;
                wave = pns_basis_render_uniform(
                    desc,
                    def_id,
                    elems[e_idx].shape_id[slot],
                    elems[e_idx].amplitude[slot],
                    dur_us,
                    raster_us,
                    &n2);
                if (!wave)
                {
                    *out_declined = 1;
                    goto decline;
                }
                for (i2 = 0; i2 < npts; ++i2)
                    dst[i2] = (i2 < n2) ? wave[i2] : 0.0f;
                PULSEG_FREE(wave);
                wave = NULL;
                macs += (double)npts;
                ++slot;
            }
        }

        {
            float *edgdt;
            int slot2;

            edgdt = (float *)PULSEG_ALLOC((size_t)st->d * (size_t)(npts + 1) * sizeof(float));
            if (!edgdt)
            {
                rc = PULSEG_ERR_ALLOC_FAILED;
                goto fail;
            }
            for (e_idx = 0; e_idx < num_elems; ++e_idx)
            {
                for (slot2 = 0; slot2 < st->d; ++slot2)
                    pns_basis_slice_dgdt(
                        edgdt + (size_t)slot2 * (npts + 1),
                        rows + ((size_t)e_idx * st->d + slot2) * (size_t)npts,
                        npts,
                        raster_us,
                        gamma_hz_per_tesla);
                elems[e_idx].l1 = (float)pns_basis_rows_l1(edgdt, st->d, npts + 1);
            }
            PULSEG_FREE(edgdt);
        }

        rc = pns_basis_decompose(
            st,
            elems,
            num_elems,
            rows,
            npts,
            kernel,
            kernel_len,
            out_scale,
            raster_us,
            gamma_hz_per_tesla,
            residual_gain,
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
        PULSEG_FREE(rows);
        rows = NULL;

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
            l1_sq = (double)elems[found].l1 * (double)elems[found].l1;
            for (a = 0; a < 3; ++a)
            {
                double unit_l1;

                if (st->retained[a] || !occ.has_grad[a])
                    continue;
                rc = pns_basis_r1_get(
                    &r1,
                    desc,
                    &occ.key[a],
                    (float)bte->duration_us,
                    raster_us,
                    gamma_hz_per_tesla,
                    kernel,
                    kernel_len,
                    out_scale,
                    &sup,
                    &unit_l1,
                    &macs);
                if (PULSEG_FAILED(rc))
                    goto fail;
                term = fabs((double)occ.amplitude[a]) * sup;
                u_sq += term * term;
                term = fabs((double)occ.amplitude[a]) * unit_l1;
                l1_sq += term * term;
            }
            sc->u[b] = (float)sqrt(u_sq);
            sc->l1[b] = (float)sqrt(l1_sq);
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
        l1_sq = 0.0;
        for (a = 0; a < 3; ++a)
        {
            double unit_l1;

            if (!occ.has_grad[a])
                continue;
            rc = pns_basis_r1_get(
                &r1,
                desc,
                &occ.key[a],
                (float)bte->duration_us,
                raster_us,
                gamma_hz_per_tesla,
                kernel,
                kernel_len,
                out_scale,
                &sup,
                &unit_l1,
                &macs);
            if (PULSEG_FAILED(rc))
                goto fail;
            term = fabs((double)occ.amplitude[a]) * sup;
            u_sq += term * term;
            term = fabs((double)occ.amplitude[a]) * unit_l1;
            l1_sq += term * term;
        }
        sc->u[b] = (float)sqrt(u_sq);
        sc->l1[b] = (float)sqrt(l1_sq);
    }

    sc->build_macs = macs;
    if (out_macs)
        *out_macs = macs;
    PULSEG_FREE(kernel);
    PULSEG_FREE(stats);
    pns_basis_r1_destroy(&r1);
    *out = sc;
    return PULSEG_SUCCESS;

decline:
    rc = PULSEG_SUCCESS;
fail:
    if (kernel)
        PULSEG_FREE(kernel);
    if (stats)
        PULSEG_FREE(stats);
    pns_basis_r1_destroy(&r1);
    pns_basis_elem_index_destroy(&elem_ix);
    if (elems)
        PULSEG_FREE(elems);
    if (rows)
        PULSEG_FREE(rows);
    pulseg__pns_score_free(sc);
    if (out_macs)
        *out_macs = macs;
    return rc;
}

/* ================================================================== */
/*  The sweep                                                          */
/* ================================================================== */

/* One anchor at a time, occurrences migrating outward through the decay
 * zones as the anchor advances. For anchor j the covered instants are those
 * whose last-started block is j; contributors are occurrences at or before
 * j whose response still reaches the anchor, each priced by its zone:
 * min(u, l1 * gain(zone)), with zone 0 priced by u itself. Every occurrence
 * enters at its own anchor and crosses each boundary once, so a sweep over
 * the scan is O(zones * blocks). */
typedef struct
{
    double sums[PNS_SCORE_NUM_ZONES];
    int q[PNS_SCORE_NUM_ZONES]; /* first member of zones >= z */
} pns_score_sweep;

static double pns_score_zone_price(const pulseg__pns_score *sc, int i, int z)
{
    double lp;

    if (z == 0)
        return (double)sc->u[i];
    lp = (double)sc->l1[i] * sc->zone_gain[z];
    return (lp < (double)sc->u[i]) ? lp : (double)sc->u[i];
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

/* Advance to anchor j (callers pass j = 0, 1, ...) and return its covering
 * sum. */
static double pns_score_sweep_step(const pulseg__pns_score *sc, pns_score_sweep *sw, int j)
{
    double start, cut, total;
    int z, i;

    /* The anchor itself enters zone 0. */
    sw->sums[0] += pns_score_zone_price(sc, j, 0);

    start = sc->t_start_us[j];
    for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
    {
        cut =
            (z + 1 < PNS_SCORE_NUM_ZONES) ? start - sc->zone_edge_us[z + 1] : start - sc->reach_us;
        for (i = sw->q[z]; i < j && sc->t_end_us[i] <= cut; ++i)
        {
            sw->sums[z] -= pns_score_zone_price(sc, i, z);
            if (z + 1 < PNS_SCORE_NUM_ZONES)
                sw->sums[z + 1] += pns_score_zone_price(sc, i, z + 1);
        }
        sw->q[z] = i;
    }

    total = 0.0;
    for (z = 0; z < PNS_SCORE_NUM_ZONES; ++z)
        total += sw->sums[z];
    return total;
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
        *out_macs = (double)sc->num_blocks * (double)PNS_SCORE_NUM_ZONES;
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
static int pns_score_exact_range_peak(
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
    rc = pulseg__pns_score_build(
        &score,
        desc,
        model,
        gamma_hz_per_tesla,
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
        rc = pns_score_exact_range_peak(
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
