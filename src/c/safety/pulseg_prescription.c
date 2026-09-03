/*
 * The scanner prescription as the frame of every gradient safety check.
 *
 * The prescription is one rotation the scanner programs over the whole scan,
 * left of every ROTATIONS matrix a block carries and alone on a block that
 * carries none; a block flagged NOROT plays in the logical frame and stays
 * there. While a check runs, each descriptor's rotation table is that
 * composition and every rotating block points into it; on the way out the
 * logical table and the block ids are handed back untouched, so nothing the
 * scanner later plays from the descriptor is changed.
 */

#include <string.h>

#include "pulseg_internal.h"

static void mat3_mul(float *out, const float *a, const float *b)
{
    int i, j, k;
    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j)
        {
            float acc = 0.0f;
            for (k = 0; k < 3; ++k)
                acc += a[i * 3 + k] * b[k * 3 + j];
            out[i * 3 + j] = acc;
        }
}

int pulseg__prescription_enter(pulseg_collection *coll, const pulseg_opts *opts)
{
    int s, b, i;

    if (!coll || !opts || !opts->has_prescription_rotation)
        return PULSEG_SUCCESS;
    if (pulseg__is_identity3(opts->prescription_rotation))
        return PULSEG_SUCCESS;
    for (s = 0; s < coll->num_subsequences; ++s)
    {
        struct pulseg_sequence_descriptor *desc = &coll->descriptors[s];
        float(*composed)[9];
        int *ids;
        int n;

        if (desc->prescription_depth > 0)
        {
            desc->prescription_depth++;
            continue;
        }
        n = desc->num_rotations;
        composed = (float(*)[9])PULSEG_ALLOC((size_t)(n + 1) * sizeof(*composed));
        ids = (int *)PULSEG_ALLOC(
            (size_t)(desc->num_blocks > 0 ? desc->num_blocks : 1) * sizeof(int));
        if (!composed || !ids)
        {
            if (composed)
                PULSEG_FREE(composed);
            if (ids)
                PULSEG_FREE(ids);
            pulseg__prescription_leave(coll, opts);
            return PULSEG_ERR_ALLOC_FAILED;
        }
        for (i = 0; i < n; ++i)
            mat3_mul(composed[i], opts->prescription_rotation, desc->rotation_matrices[i]);
        memcpy(composed[n], opts->prescription_rotation, 9 * sizeof(float));
        for (b = 0; b < desc->num_blocks; ++b)
        {
            pulseg_block_table_element *bte = &desc->block_table[b];
            ids[b] = bte->rotation_id;
            if (!bte->norot_flag && bte->rotation_id < 0)
                bte->rotation_id = n;
        }
        desc->rotation_matrices_logical = desc->rotation_matrices;
        desc->num_rotations_logical = n;
        desc->rotation_id_logical = ids;
        desc->rotation_matrices = composed;
        desc->num_rotations = n + 1;
        desc->prescription_depth = 1;
    }
    return PULSEG_SUCCESS;
}

void pulseg__prescription_leave(pulseg_collection *coll, const pulseg_opts *opts)
{
    int s, b;

    (void)opts;
    if (!coll)
        return;
    for (s = 0; s < coll->num_subsequences; ++s)
    {
        struct pulseg_sequence_descriptor *desc = &coll->descriptors[s];
        if (desc->prescription_depth == 0)
            continue;
        if (--desc->prescription_depth > 0)
            continue;
        for (b = 0; b < desc->num_blocks; ++b)
            desc->block_table[b].rotation_id = desc->rotation_id_logical[b];
        PULSEG_FREE(desc->rotation_id_logical);
        desc->rotation_id_logical = NULL;
        PULSEG_FREE(desc->rotation_matrices);
        desc->rotation_matrices = desc->rotation_matrices_logical;
        desc->num_rotations = desc->num_rotations_logical;
        desc->rotation_matrices_logical = NULL;
        desc->num_rotations_logical = 0;
    }
}
