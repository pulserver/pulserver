/**
 * @file pulseg_check_plan.c
 * @brief The preprocessing the gradient checks share, held across calls.
 *
 * The PNS and mechanical-resonance checks both answer their question from
 * uniform-raster gradient waveforms extracted over a window of the canonical
 * TR, and both need the sequence's repetitions grouped by the set of shapes
 * they play. Extraction is the expensive step -- it renders every block in
 * the window onto a raster -- and grouping is the step that decides how many
 * windows there are.
 *
 * A plan holds both, keyed on what makes a window distinct, so a caller
 * asking several questions of one sequence pays for each window once. The
 * windows PNS and mech-res evaluate are not the same windows, so running the
 * two checks back to back does not hit the cache; running a check and then
 * its plotting counterpart, or re-running a check against a different band
 * table or threshold, does.
 */

#include <string.h>

#include "pulseg/pulseg_collection.h"
#include "pulseg_internal.h"

/* Retained waveform bytes a plan will hold before it starts evicting. */
#define PLAN_DEFAULT_BUDGET_KB 4096

/* Windows tracked at once. A window per shape group per subsequence, for
 * both amplitude modes, with room to spare; beyond this the least recently
 * used entry is dropped whether or not the budget is reached. */
#define PLAN_MAX_WINDOWS 32

typedef struct plan_window
{
    int in_use;
    int subseq_idx;
    int block_start;
    int block_count;
    int amplitude_mode;
    int target_group;
    int grouped; /* whether the extraction was restricted to one group */
    int bytes;
    unsigned long stamp;
    pulseg__uniform_grad_waveforms uw;
} plan_window;

typedef struct plan_groups
{
    int computed;
    int num_groups;
    int *labels;
    int *group_first;
} plan_groups;

struct pulseg_check_plan
{
    const pulseg_collection *coll;
    int budget_bytes;
    int held_bytes;
    unsigned long clock;
    int pinned; /* index of the window most recently handed out, or -1 */
    plan_window windows[PLAN_MAX_WINDOWS];
    plan_groups *groups; /* [coll->num_subsequences] */
};

static int plan_window_bytes(const pulseg__uniform_grad_waveforms *uw)
{
    /* Three axes of float, plus the struct itself. */
    return (int)(3 * (size_t)uw->num_samples * sizeof(float)) + (int)sizeof(plan_window);
}

static void plan_window_release(pulseg_check_plan *plan, int idx)
{
    plan_window *w = &plan->windows[idx];

    if (!w->in_use)
        return;
    pulseg__uniform_grad_waveforms_free(&w->uw);
    plan->held_bytes -= w->bytes;
    memset(w, 0, sizeof(*w));
}

/* Drop least-recently-used windows until the plan is inside its budget and
 * has a free slot. The pinned window -- the one the caller is holding a
 * borrowed pointer to -- is never dropped. */
static int plan_make_room(pulseg_check_plan *plan, int need_slot)
{
    int i, victim;
    unsigned long oldest;

    for (;;)
    {
        int have_slot = 0;
        for (i = 0; i < PLAN_MAX_WINDOWS; ++i)
        {
            if (!plan->windows[i].in_use)
            {
                have_slot = 1;
                break;
            }
        }
        if ((!need_slot || have_slot) && plan->held_bytes <= plan->budget_bytes)
            return have_slot ? i : -1;

        victim = -1;
        oldest = 0;
        for (i = 0; i < PLAN_MAX_WINDOWS; ++i)
        {
            if (!plan->windows[i].in_use || i == plan->pinned)
                continue;
            if (victim < 0 || plan->windows[i].stamp < oldest)
            {
                victim = i;
                oldest = plan->windows[i].stamp;
            }
        }
        if (victim < 0)
            return have_slot ? i : -1; /* nothing evictable left */
        plan_window_release(plan, victim);
    }
}

int pulseg_check_plan_create(
    pulseg_check_plan **out,
    pulseg_diagnostic *diag,
    const pulseg_collection *coll,
    const pulseg_check_plan_config *config)
{
    pulseg_check_plan *plan;
    int budget_kb;

    if (out)
        *out = NULL;
    if (!out || !coll)
    {
        if (diag)
        {
            pulseg_diagnostic_init(diag);
            diag->code = PULSEG_ERR_NULL_POINTER;
        }
        return PULSEG_ERR_NULL_POINTER;
    }

    plan = (pulseg_check_plan *)PULSEG_ALLOC(sizeof(*plan));
    if (!plan)
    {
        if (diag)
        {
            pulseg_diagnostic_init(diag);
            diag->code = PULSEG_ERR_ALLOC_FAILED;
        }
        return PULSEG_ERR_ALLOC_FAILED;
    }
    memset(plan, 0, sizeof(*plan));

    if (coll->num_subsequences > 0)
    {
        size_t n = (size_t)coll->num_subsequences * sizeof(plan_groups);
        plan->groups = (plan_groups *)PULSEG_ALLOC(n);
        if (!plan->groups)
        {
            PULSEG_FREE(plan);
            if (diag)
            {
                pulseg_diagnostic_init(diag);
                diag->code = PULSEG_ERR_ALLOC_FAILED;
            }
            return PULSEG_ERR_ALLOC_FAILED;
        }
        memset(plan->groups, 0, n);
    }

    budget_kb =
        (config && config->cache_budget_kb > 0) ? config->cache_budget_kb : PLAN_DEFAULT_BUDGET_KB;
    plan->coll = coll;
    plan->budget_bytes = budget_kb * 1024;
    plan->pinned = -1;

    *out = plan;
    return PULSEG_SUCCESS;
}

void pulseg_check_plan_destroy(pulseg_check_plan *plan)
{
    int i;

    if (!plan)
        return;

    plan->pinned = -1;
    for (i = 0; i < PLAN_MAX_WINDOWS; ++i)
        plan_window_release(plan, i);
    if (plan->groups)
    {
        for (i = 0; i < plan->coll->num_subsequences; ++i)
        {
            if (plan->groups[i].labels)
                PULSEG_FREE(plan->groups[i].labels);
            if (plan->groups[i].group_first)
                PULSEG_FREE(plan->groups[i].group_first);
        }
        PULSEG_FREE(plan->groups);
    }
    PULSEG_FREE(plan);
}

int pulseg__plan_shape_groups(
    pulseg_check_plan *plan,
    const int **out_labels,
    const int **out_group_first,
    int *out_num_groups,
    pulseg_diagnostic *diag,
    const pulseg_sequence_descriptor *desc,
    int subseq_idx)
{
    plan_groups *g;
    int rc;

    if (!plan || !plan->groups || subseq_idx < 0 || subseq_idx >= plan->coll->num_subsequences)
        return PULSEG_ERR_NULL_POINTER;

    g = &plan->groups[subseq_idx];
    if (!g->computed)
    {
        rc = pulseg__group_tr_instances_by_shape(
            desc,
            &g->labels,
            &g->group_first,
            &g->num_groups,
            PULSEG__MAX_SHAPE_GROUPS);
        if (PULSEG_FAILED(rc))
        {
            if (diag)
            {
                pulseg__diag_printf(
                    diag,
                    "the repetitions play more than %d distinct sets of gradient "
                    "waveforms, which is more windows than the check will evaluate, "
                    "and one window over all of them would not bound them. Write the "
                    "repeated waveform once and turn it with a ROTATIONS extension",
                    PULSEG__MAX_SHAPE_GROUPS);
                diag->code = PULSEG_ERR_PNS_INVALID_PARAMS;
            }
            return PULSEG_ERR_PNS_INVALID_PARAMS;
        }
        g->computed = 1;
    }

    *out_labels = g->labels;
    *out_group_first = g->group_first;
    *out_num_groups = g->num_groups;
    return PULSEG_SUCCESS;
}

int pulseg__plan_waveforms(
    pulseg_check_plan *plan,
    const pulseg__uniform_grad_waveforms **out,
    pulseg_diagnostic *diag,
    const pulseg_sequence_descriptor *desc,
    int subseq_idx,
    int block_start,
    int block_count,
    int amplitude_mode,
    const int *labels,
    int target_group)
{
    plan_window *w;
    int i, slot, rc, grouped;
    *out = NULL;
    if (!plan)
        return PULSEG_ERR_NULL_POINTER;

    if (desc && desc->structure_only)
    {
        if (diag)
        {
            pulseg__diag_printf(
                diag,
                "the collection was converted for structure only and holds no gradient waveforms");
            diag->code = PULSEG_ERR_INVALID_ARGUMENT;
        }
        return PULSEG_ERR_INVALID_ARGUMENT;
    }
    grouped = labels ? 1 : 0;

    for (i = 0; i < PLAN_MAX_WINDOWS; ++i)
    {
        w = &plan->windows[i];
        if (!w->in_use || w->subseq_idx != subseq_idx || w->block_start != block_start ||
            w->block_count != block_count || w->amplitude_mode != amplitude_mode ||
            w->grouped != grouped || (grouped && w->target_group != target_group))
            continue;
        w->stamp = ++plan->clock;
        plan->pinned = i;
        *out = &w->uw;
        return PULSEG_SUCCESS;
    }

    slot = plan_make_room(plan, 1);
    if (slot < 0)
        return PULSEG_ERR_ALLOC_FAILED;

    w = &plan->windows[slot];
    memset(w, 0, sizeof(*w));
    rc = pulseg__get_gradient_waveforms_range(
        desc,
        &w->uw,
        diag,
        block_start,
        block_count,
        amplitude_mode,
        labels,
        target_group,
        NULL);
    if (PULSEG_FAILED(rc))
    {
        pulseg__uniform_grad_waveforms_free(&w->uw);
        memset(w, 0, sizeof(*w));
        return rc;
    }

    w->in_use = 1;
    w->subseq_idx = subseq_idx;
    w->block_start = block_start;
    w->block_count = block_count;
    w->amplitude_mode = amplitude_mode;
    w->target_group = target_group;
    w->grouped = grouped;
    w->bytes = plan_window_bytes(&w->uw);
    w->stamp = ++plan->clock;
    plan->held_bytes += w->bytes;
    plan->pinned = slot;

    (void)plan_make_room(plan, 0);

    *out = &w->uw;
    return PULSEG_SUCCESS;
}
