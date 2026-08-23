/**
 * @file pulseg_chunk.c
 * @brief Planning the chunk split.  See pulseg_chunk.h.
 *
 * The walk is one pass over the exec stream.  Each position contributes a
 * wave key; the key is looked up in the current chunk's set and added if new,
 * which is what makes a chunk's cost its *distinct* waveforms rather than its
 * blocks.  Segment ends are the only legal places to close a chunk, so the
 * planner remembers the last one that still fitted and cuts there.
 */

#include "pulseg_chunk.h"

#include "pulseg_internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================== */
/*  Wave keys                                                         */
/* ================================================================== */

static int wave_key_equal(const pulseg_wave_key *a, const pulseg_wave_key *b)
{
    int i;

    if (a->rotation_id != b->rotation_id || a->num_points != b->num_points)
        return 0;
    for (i = 0; i < 3; ++i)
    {
        if (a->shape_id[i] != b->shape_id[i] || a->grad_def[i] != b->grad_def[i] ||
            a->ratio_q[i] != b->ratio_q[i])
            return 0;
    }
    return 1;
}

/*
 * Build the key for the block at exec-stream position @p pos.
 *
 * Returns 0 when the block drives no gradient at all, in which case it costs
 * nothing to materialise and contributes only its duration.
 */
static int wave_key_for_block(
    const pulseg_sequence_descriptor *desc,
    int block_idx,
    float ratio_tolerance,
    pulseg_wave_key *out)
{
    const pulseg_block_table_element *bte;
    const pulseg_grad_definition *gdef;
    const pulseg_grad_table_element *gte;
    int raw_ids[3];
    float amp[3];
    float scale;
    int axis, n;
    int centre_raster, centre_points;

    memset(out, 0, sizeof(*out));
    out->rotation_id = -1;
    for (axis = 0; axis < 3; ++axis)
        out->grad_def[axis] = -1;

    if (block_idx < 0 || block_idx >= desc->num_blocks)
        return 0;
    bte = &desc->block_table[block_idx];
    raw_ids[0] = bte->gx_id;
    raw_ids[1] = bte->gy_id;
    raw_ids[2] = bte->gz_id;
    out->rotation_id = bte->rotation_id;

    scale = 0.0f;
    n = 0;
    centre_raster = 0;
    centre_points = 0;
    for (axis = 0; axis < 3; ++axis)
    {
        amp[axis] = 0.0f;
        if (raw_ids[axis] < 0 || raw_ids[axis] >= desc->grad_table_size)
            continue;
        gte = &desc->grad_table[raw_ids[axis]];
        if (gte->id < 0 || gte->id >= desc->num_unique_grads)
            continue;

        gdef = &desc->grad_definitions[gte->id];
        out->grad_def[axis] = gte->id;
        out->shape_id[axis] = gte->shape_id;
        amp[axis] = gte->amplitude;
        if ((float)fabs((double)amp[axis]) > scale)
            scale = (float)fabs((double)amp[axis]);
        n++;

        /* Cost of the materialised wave, in corner points.  Nothing is
         * resampled onto a raster -- the rotated combination lives on the
         * union of the axes' own breakpoints -- so a trapezoid costs four
         * points however long it is.
         *
         * The sum over axes bounds that union from above (shared corners
         * collapse), which is the safe direction: it can only make a chunk
         * smaller than it had to be. */
        if (gdef->type == 0)
        {
            out->num_points += 4; /* delay, +rise, +flat, +fall -- all edges */
        }
        else if (gdef->unused_or_time_shape_id > 0)
        {
            out->num_points += gdef->fall_time_or_num_uncompressed_samples;
        }
        else
        {
            /* Uniformly sampled: its centres are the only grid the whole
             * combination can live on, so the wave costs one point per raster
             * tick regardless of what the other axes are.  See
             * wave_needs_centre_raster. */
            centre_raster = 1;
            if (gdef->fall_time_or_num_uncompressed_samples > centre_points)
                centre_points = gdef->fall_time_or_num_uncompressed_samples;
        }
    }

    if (n == 0)
        return 0;

    /* One uniformly sampled axis forces the whole combination onto the raster
     * centre grid, so the corner counts of the other axes stop mattering. */
    if (centre_raster)
        out->num_points = centre_points;

    /*
     * A rotated block materialises all three axes -- the rotation mixes them,
     * so an axis the block did not drive can still come out non-zero.  An
     * unrotated one materialises only what it drives.
     */
    out->num_axes = (out->rotation_id >= 0) ? 3 : n;

    /* Ratios, normalised by the largest magnitude and quantised.  What matters
     * for waveform identity is the direction of the amplitude triple; its
     * length is the scalar setiamp applies. */
    if (scale > 0.0f)
    {
        for (axis = 0; axis < 3; ++axis)
        {
            out->ratio[axis] = amp[axis] / scale;
            if (ratio_tolerance > 0.0f)
                out->ratio_q[axis] =
                    (int)(out->ratio[axis] / ratio_tolerance + (amp[axis] >= 0.0f ? 0.5f : -0.5f));
        }
    }
    return 1;
}

/* ================================================================== */
/*  Planning                                                          */
/* ================================================================== */

void pulseg_free_chunk_plan(pulseg_chunk_plan *plan)
{
    if (!plan)
        return;
    if (plan->chunks)
        PULSEG_FREE(plan->chunks);
    if (plan->waves)
        PULSEG_FREE(plan->waves);
    if (plan->chunk_waves)
        PULSEG_FREE(plan->chunk_waves);
    if (plan->position_wave)
        PULSEG_FREE(plan->position_wave);
    if (plan->wave_slot_points)
        PULSEG_FREE(plan->wave_slot_points);
    if (plan->block_wave_points)
        PULSEG_FREE(plan->block_wave_points);
    plan->mode = PULSEG_WAVE_RESIDENT;
    plan->chunks = NULL;
    plan->waves = NULL;
    plan->chunk_waves = NULL;
    plan->position_wave = NULL;
    plan->wave_slot_points = NULL;
    plan->block_wave_points = NULL;
    plan->num_chunks = 0;
    plan->num_waves = 0;
    plan->num_chunk_waves = 0;
    plan->total_wave_points = 0;
    plan->peak_wave_samples = 0;
    plan->max_wave_points = 0;
    plan->block_stride = 0;
    plan->num_block_points = 0;
    plan->position_count = 0;
}

/* Append @p key to the global list if it is new; return its index either way. */
static int intern_wave(pulseg_wave_key **waves, int *count, int *cap, const pulseg_wave_key *key)
{
    int i;
    pulseg_wave_key *grown;

    for (i = 0; i < *count; ++i)
        if (wave_key_equal(&(*waves)[i], key))
            return i;

    if (*count >= *cap)
    {
        *cap *= 2;
        grown = (pulseg_wave_key *)PULSEG_ALLOC((size_t)(*cap) * sizeof(pulseg_wave_key));
        if (!grown)
            return -1;
        memcpy(grown, *waves, (size_t)(*count) * sizeof(pulseg_wave_key));
        PULSEG_FREE(*waves);
        *waves = grown;
    }
    (*waves)[*count] = *key;
    return (*count)++;
}

int pulseg_plan_chunks(
    const pulseg_collection *coll,
    int subseq_idx,
    const pulseg_chunk_budget *budget,
    pulseg_chunk_plan *out,
    pulseg_diagnostic *diag)
{
    const pulseg_sequence_descriptor *desc;
    pulseg_wave_key *waves = NULL;
    pulseg_chunk *chunks = NULL;
    int *position_wave = NULL;
    int *chunk_waves = NULL;
    int *wave_slot_points = NULL;
    int *block_wave_points = NULL;
    pulseg_wave_key key;
    int cap_waves, num_waves;
    int cap_chunks, num_chunks;
    int cap_cw, num_cw;
    int pos, block_idx, seg_id, prev_seg, k;
    int block_stride, num_block_points;
    int blk_in_seg;

    if (!coll || !budget || !out)
        return PULSEG_ERR_INVALID_ARGUMENT;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;
    if (budget->max_wave_samples <= 0)
    {
        if (diag)
            diag->code = PULSEG_ERR_INVALID_ARGUMENT;
        return PULSEG_ERR_INVALID_ARGUMENT;
    }

    desc = &coll->descriptors[subseq_idx];
    out->mode = PULSEG_WAVE_RESIDENT;
    out->num_chunks = 0;
    out->chunks = NULL;
    out->num_waves = 0;
    out->waves = NULL;
    out->chunk_waves = NULL;
    out->num_chunk_waves = 0;
    out->total_wave_points = 0;
    out->peak_wave_samples = 0;
    out->max_wave_points = 0;
    out->wave_slot_points = NULL;
    out->block_stride = 0;
    out->num_block_points = 0;
    out->block_wave_points = NULL;
    out->position_count = 0;
    out->position_wave = NULL;

    if (desc->exec_stream_len <= 0)
        return PULSEG_SUCCESS;

    /* ---- Pass 1: every distinct waveform in the scan ---- */
    cap_waves = 256;
    num_waves = 0;
    waves = (pulseg_wave_key *)PULSEG_ALLOC((size_t)cap_waves * sizeof(pulseg_wave_key));
    position_wave = (int *)PULSEG_ALLOC((size_t)desc->exec_stream_len * sizeof(int));
    if (!waves || !position_wave)
        goto oom;

    block_stride = 0;
    for (k = 0; k < desc->num_unique_segments; ++k)
        if (desc->segment_definitions[k].num_blocks > block_stride)
            block_stride = desc->segment_definitions[k].num_blocks;
    num_block_points = desc->num_unique_segments * block_stride;
    if (num_block_points > 0)
    {
        block_wave_points = (int *)PULSEG_ALLOC((size_t)num_block_points * sizeof(int));
        if (!block_wave_points)
            goto oom;
        memset(block_wave_points, 0, (size_t)num_block_points * sizeof(int));
    }

    prev_seg = -1;
    blk_in_seg = 0;
    for (pos = 0; pos < desc->exec_stream_len; ++pos)
    {
        position_wave[pos] = -1;
        seg_id = pulseg__exec_seg_id(desc, pos);

        /* Block position within the segment INSTANCE, tracked the same way
         * pass 2 finds instance boundaries: a definition replays unchanged, so
         * only exhausting its block count ends one. */
        if (pos == 0 || seg_id != prev_seg)
            blk_in_seg = 0;
        else if (seg_id >= 0 && seg_id < desc->num_unique_segments &&
                 blk_in_seg >= desc->segment_definitions[seg_id].num_blocks)
            blk_in_seg = 0;
        prev_seg = seg_id;

        block_idx = pulseg__exec_block_idx(desc, pos);
        if (wave_key_for_block(desc, block_idx, budget->ratio_tolerance, &key))
        {
            position_wave[pos] = intern_wave(&waves, &num_waves, &cap_waves, &key);
            if (position_wave[pos] < 0)
                goto oom;
            if (block_wave_points && seg_id >= 0 && seg_id < desc->num_unique_segments &&
                blk_in_seg < block_stride)
            {
                int *slot = &block_wave_points[seg_id * block_stride + blk_in_seg];
                if (key.num_points > *slot)
                    *slot = key.num_points;
            }
        }
        blk_in_seg++;
    }

    out->num_waves = num_waves;
    out->waves = waves;
    out->position_count = desc->exec_stream_len;
    out->position_wave = position_wave;
    out->block_stride = block_stride;
    out->num_block_points = num_block_points;
    out->block_wave_points = block_wave_points;
    for (k = 0; k < num_waves; ++k)
    {
        long pts = (long)waves[k].num_points * (long)waves[k].num_axes;
        out->total_wave_points += pts;
        if (waves[k].num_points > out->max_wave_points)
            out->max_wave_points = waves[k].num_points;
    }

    /* A slot has to hold the wave AND satisfy every instruction pointed at
     * it: playout reads the instruction's own reserved length from wherever
     * the pointer lands, so a slot shorter than one of its blocks runs past
     * the end. */
    if (num_waves > 0)
    {
        wave_slot_points = (int *)PULSEG_ALLOC((size_t)num_waves * sizeof(int));
        if (!wave_slot_points)
            goto oom;
        for (k = 0; k < num_waves; ++k)
            wave_slot_points[k] = waves[k].num_points;
        if (block_wave_points)
        {
            prev_seg = -1;
            blk_in_seg = 0;
            for (pos = 0; pos < desc->exec_stream_len; ++pos)
            {
                int w, reserved;
                seg_id = pulseg__exec_seg_id(desc, pos);
                if (pos == 0 || seg_id != prev_seg)
                    blk_in_seg = 0;
                else if (seg_id >= 0 && seg_id < desc->num_unique_segments &&
                         blk_in_seg >= desc->segment_definitions[seg_id].num_blocks)
                    blk_in_seg = 0;
                prev_seg = seg_id;

                w = position_wave[pos];
                if (w >= 0 && seg_id >= 0 && seg_id < desc->num_unique_segments &&
                    blk_in_seg < block_stride)
                {
                    reserved = block_wave_points[seg_id * block_stride + blk_in_seg];
                    if (reserved > wave_slot_points[w])
                        wave_slot_points[w] = reserved;
                }
                blk_in_seg++;
            }
        }
    }
    out->wave_slot_points = wave_slot_points;

    /* ---- The easy answer ----
     *
     * Everything fits, so it all goes into waveform memory at pulsegen and
     * nothing is uploaded during the scan.  There is no rate to check because
     * there is no work left to be late with.  One chunk is emitted covering
     * the scan so consumers read one shape either way. */
    if (out->total_wave_points <= budget->max_wave_samples)
    {
        chunks = (pulseg_chunk *)PULSEG_ALLOC(sizeof(pulseg_chunk));
        chunk_waves = (int *)PULSEG_ALLOC((size_t)(num_waves > 0 ? num_waves : 1) * sizeof(int));
        if (!chunks || !chunk_waves)
            goto oom;
        for (k = 0; k < num_waves; ++k)
            chunk_waves[k] = k;
        chunks[0].first_position = 0;
        chunks[0].num_positions = desc->exec_stream_len;
        chunks[0].first_wave = 0;
        chunks[0].num_waves = num_waves;
        chunks[0].wave_samples = out->total_wave_points;
        chunks[0].playout_us = 0.0f;
        chunks[0].build_us = 0.0f;
        out->mode = PULSEG_WAVE_RESIDENT;
        out->num_chunks = 1;
        out->chunks = chunks;
        out->chunk_waves = chunk_waves;
        out->num_chunk_waves = num_waves;
        out->peak_wave_samples = out->total_wave_points;
        return PULSEG_SUCCESS;
    }

    /* ---- Pass 2: one chunk per segment instance ----
     *
     * A segment is the unit the work interleaves with: while segment N plays,
     * segment N+1 is materialised.  Looking further ahead buys nothing --
     * segment N+1's waves still have to be ready when it starts -- so the
     * chunk is exactly the segment. */
    out->mode = PULSEG_WAVE_STREAMED;
    cap_chunks = 64;
    cap_cw = 256;
    num_chunks = 0;
    num_cw = 0;
    chunks = (pulseg_chunk *)PULSEG_ALLOC((size_t)cap_chunks * sizeof(pulseg_chunk));
    chunk_waves = (int *)PULSEG_ALLOC((size_t)cap_cw * sizeof(int));
    if (!chunks || !chunk_waves)
        goto oom;

    prev_seg = -1;
    for (pos = 0; pos < desc->exec_stream_len; ++pos)
    {
        int starts_instance;

        seg_id = pulseg__exec_seg_id(desc, pos);
        block_idx = pulseg__exec_block_idx(desc, pos);

        /*
         * A segment INSTANCE, not a segment definition.  A scan usually
         * replays one definition thousands of times, so the id alone never
         * changes and cutting on it would produce a single chunk covering
         * everything -- which is the opposite of streaming.  An instance ends
         * when the definition's block count is exhausted.
         */
        starts_instance = (pos == 0) || (seg_id != prev_seg);
        if (!starts_instance && num_chunks > 0 && seg_id >= 0 && seg_id < desc->num_unique_segments)
        {
            if (chunks[num_chunks - 1].num_positions >=
                desc->segment_definitions[seg_id].num_blocks)
                starts_instance = 1;
        }

        if (starts_instance)
        {
            if (num_chunks >= cap_chunks)
            {
                pulseg_chunk *grown;
                cap_chunks *= 2;
                grown = (pulseg_chunk *)PULSEG_ALLOC((size_t)cap_chunks * sizeof(pulseg_chunk));
                if (!grown)
                    goto oom;
                memcpy(grown, chunks, (size_t)num_chunks * sizeof(pulseg_chunk));
                PULSEG_FREE(chunks);
                chunks = grown;
            }
            chunks[num_chunks].first_position = pos;
            chunks[num_chunks].num_positions = 0;
            chunks[num_chunks].first_wave = num_cw;
            chunks[num_chunks].num_waves = 0;
            chunks[num_chunks].wave_samples = 0;
            chunks[num_chunks].playout_us = 0.0f;
            chunks[num_chunks].build_us = 0.0f;
            num_chunks++;
        }
        prev_seg = seg_id;

        chunks[num_chunks - 1].num_positions++;
        if (block_idx >= 0 && block_idx < desc->num_blocks)
            chunks[num_chunks - 1].playout_us += (float)desc->block_table[block_idx].duration_us;

        if (position_wave[pos] >= 0)
        {
            int w = position_wave[pos];
            int already = 0;
            for (k = chunks[num_chunks - 1].first_wave; k < num_cw; ++k)
                if (chunk_waves[k] == w)
                {
                    already = 1;
                    break;
                }
            if (!already)
            {
                if (num_cw >= cap_cw)
                {
                    int *grown;
                    cap_cw *= 2;
                    grown = (int *)PULSEG_ALLOC((size_t)cap_cw * sizeof(int));
                    if (!grown)
                        goto oom;
                    memcpy(grown, chunk_waves, (size_t)num_cw * sizeof(int));
                    PULSEG_FREE(chunk_waves);
                    chunk_waves = grown;
                }
                chunk_waves[num_cw++] = w;
                chunks[num_chunks - 1].num_waves++;
                chunks[num_chunks - 1].wave_samples +=
                    (long)waves[w].num_points * (long)waves[w].num_axes;
            }
        }
    }

    for (k = 0; k < num_chunks; ++k)
    {
        chunks[k].build_us = (float)chunks[k].wave_samples * budget->build_us_per_sample;
        if (chunks[k].wave_samples > out->peak_wave_samples)
            out->peak_wave_samples = chunks[k].wave_samples;
    }

    out->num_chunks = num_chunks;
    out->chunks = chunks;
    out->chunk_waves = chunk_waves;
    out->num_chunk_waves = num_cw;

    /*
     * One segment's waves have to be built inside the segment before it.  That
     * is the only interval available: the work cannot start earlier, because
     * the buffer half it targets is still playing, and it cannot finish later,
     * because the segment starts.  A miss is reported rather than worked
     * around -- no arrangement of the plan changes which segment the work has
     * to happen in.
     */
    if (budget->build_us_per_sample > 0.0f)
    {
        for (k = 1; k < num_chunks; ++k)
        {
            float allowed = chunks[k - 1].playout_us * budget->headroom;
            if (chunks[k].build_us > allowed)
            {
                if (diag)
                {
                    diag->code = PULSEG_ERR_CHUNK_INFEASIBLE;
                    pulseg__diag_printf(
                        diag,
                        " segment_chunk=%d build_us=%.0f allowed_us=%.0f",
                        k,
                        (double)chunks[k].build_us,
                        (double)allowed);
                    pulseg__diag_printf(
                        diag,
                        " waves=%d points=%ld",
                        chunks[k].num_waves,
                        chunks[k].wave_samples);
                }
                return PULSEG_ERR_CHUNK_INFEASIBLE;
            }
        }
    }

    return PULSEG_SUCCESS;

oom:
    if (waves && waves != out->waves)
        PULSEG_FREE(waves);
    if (position_wave && position_wave != out->position_wave)
        PULSEG_FREE(position_wave);
    if (block_wave_points && block_wave_points != out->block_wave_points)
        PULSEG_FREE(block_wave_points);
    if (wave_slot_points && wave_slot_points != out->wave_slot_points)
        PULSEG_FREE(wave_slot_points);
    if (chunks)
        PULSEG_FREE(chunks);
    if (chunk_waves)
        PULSEG_FREE(chunk_waves);
    if (diag)
        diag->code = PULSEG_ERR_ALLOC_FAILED;
    return PULSEG_ERR_ALLOC_FAILED;
}

/* ================================================================== */
/*  Materialisation                                                   */
/* ================================================================== */

/* Corner points of one axis' normalised waveform: value v[i] at time t[i],
 * linear in between, zero outside.  A trapezoid is four numbers; an arbitrary
 * gradient is its samples with the two boundary values closing the ends. */
typedef struct axis_corners
{
    float *t;
    float *v;
    int n;
} axis_corners;

static void axis_corners_free(axis_corners *c)
{
    if (c->t)
        PULSEG_FREE(c->t);
    if (c->v)
        PULSEG_FREE(c->v);
    c->t = NULL;
    c->v = NULL;
    c->n = 0;
}

static int axis_corners_build(
    const pulseg_sequence_descriptor *desc,
    int grad_def_id,
    int shape_id,
    axis_corners *out)
{
    const pulseg_grad_definition *gd;
    pulseq_shape decomp;
    pulseq_shape decomp_t;
    float delay, rise, flat, fall, raster;
    int i, ns, has_time, k;

    out->t = NULL;
    out->v = NULL;
    out->n = 0;

    if (grad_def_id < 0 || grad_def_id >= desc->num_unique_grads)
        return 1; /* axis not driven */
    gd = &desc->grad_definitions[grad_def_id];
    delay = (float)gd->delay;

    if (gd->type == 0)
    {
        rise = (float)gd->rise_time_or_unused;
        flat = (float)gd->flat_time_or_unused;
        fall = (float)gd->fall_time_or_num_uncompressed_samples;
        out->t = (float *)PULSEG_ALLOC(4 * sizeof(float));
        out->v = (float *)PULSEG_ALLOC(4 * sizeof(float));
        if (!out->t || !out->v)
        {
            axis_corners_free(out);
            return 0;
        }
        out->t[0] = delay;
        out->v[0] = 0.0f;
        out->t[1] = delay + rise;
        out->v[1] = 1.0f;
        out->t[2] = delay + rise + flat;
        out->v[2] = 1.0f;
        out->t[3] = delay + rise + flat + fall;
        out->v[3] = 0.0f;
        out->n = 4;
        return 1;
    }

    if (shape_id <= 0 || shape_id > desc->num_shapes)
        return 1;
    decomp.samples = NULL;
    decomp.num_samples = 0;
    decomp.num_uncompressed_samples = 0;
    if (!pulseq_decompress_shape(&decomp, &desc->shapes[shape_id - 1], 1.0f))
        return 0;
    ns = decomp.num_uncompressed_samples;
    if (ns <= 0)
    {
        if (decomp.samples)
            PULSEG_FREE(decomp.samples);
        return 1;
    }

    raster = desc->grad_raster_us > 0.0f ? desc->grad_raster_us : 4.0f;
    decomp_t.samples = NULL;
    decomp_t.num_samples = 0;
    decomp_t.num_uncompressed_samples = 0;
    has_time = 0;
    if (gd->unused_or_time_shape_id > 0 && gd->unused_or_time_shape_id <= desc->num_shapes)
    {
        if (pulseq_decompress_shape(
                &decomp_t,
                &desc->shapes[gd->unused_or_time_shape_id - 1],
                raster))
            has_time = decomp_t.num_uncompressed_samples >= ns;
    }

    /* Samples, plus the two boundary values that close the event. */
    out->t = (float *)PULSEG_ALLOC((size_t)(ns + 2) * sizeof(float));
    out->v = (float *)PULSEG_ALLOC((size_t)(ns + 2) * sizeof(float));
    if (!out->t || !out->v)
    {
        axis_corners_free(out);
        PULSEG_FREE(decomp.samples);
        if (decomp_t.samples)
            PULSEG_FREE(decomp_t.samples);
        return 0;
    }

    /*
     * Where the samples sit depends on whether a time shape is present, and
     * that decides whether the event's edges need corners of their own.
     *
     *   - no time shape: the samples are at raster CENTRES, (i+0.5)*raster.
     *     The two edges are then times no sample covers, and `first`/`last`
     *     are exactly the values there -- so the corner list is ns + 2.
     *
     *   - time shape: the times are explicit and edge-aligned, so the first
     *     and last samples already ARE the edges.  Adding boundary corners
     *     here would put two points at one time with different values -- a
     *     zero-width discontinuity that the rotation combine would then
     *     propagate onto every output axis.  The list is ns.
     *
     * A trapezoid, further up, is likewise all edges: delay, +rise, +flat,
     * +fall.
     */
    k = 0;
    if (!has_time)
    {
        out->t[k] = delay;
        out->v[k] = pulseg__grad_shape_first(desc, shape_id);
        k++;
    }
    for (i = 0; i < ns; ++i)
    {
        out->t[k] = delay + (has_time ? decomp_t.samples[i] : ((float)i + 0.5f) * raster);
        out->v[k] = decomp.samples[i];
        k++;
    }
    if (!has_time)
    {
        out->t[k] = delay + (float)ns * raster;
        out->v[k] = pulseg__grad_shape_last(desc, shape_id);
        k++;
    }
    out->n = k;

    PULSEG_FREE(decomp.samples);
    if (decomp_t.samples)
        PULSEG_FREE(decomp_t.samples);
    return 1;
}

/* The waveform at @p x, holding the end values outside its own extent. */
static float axis_value_at(const axis_corners *c, float x)
{
    int lo, hi, mid;
    float span;

    if (c->n < 2)
        return 0.0f;
    /* The endpoints carry the event's own boundary values, so they are read
     * off rather than treated as outside; strictly beyond them the axis is
     * silent. */
    if (x < c->t[0] || x > c->t[c->n - 1])
        return 0.0f;
    if (x == c->t[0])
        return c->v[0];
    if (x == c->t[c->n - 1])
        return c->v[c->n - 1];

    lo = 0;
    hi = c->n - 1;
    while (hi - lo > 1)
    {
        mid = (lo + hi) / 2;
        if (c->t[mid] <= x)
            lo = mid;
        else
            hi = mid;
    }
    span = c->t[hi] - c->t[lo];
    if (span <= 0.0f)
        return c->v[hi];
    return c->v[lo] + (c->v[hi] - c->v[lo]) * (x - c->t[lo]) / span;
}

/*
 * Does this key have to come out on raster centres rather than on the union of
 * its corners?
 *
 * Yes as soon as ONE involved axis is a uniformly sampled arbitrary gradient.
 * Such a waveform is stored as samples at raster CENTRES and nothing else --
 * the hardware object carries no explicit first or last sample, those live on
 * the edges -- so a combination involving it can only be expressed on that
 * same centre grid.  Everything else (trapezoids, and arbitrary gradients with
 * an explicit time shape) is edge-aligned and composes exactly on the union of
 * its breakpoints.
 *
 * Both this library and pulsegen have to make the same call, or the RSP swaps
 * in a waveform sampled differently from the one the hardware slot was built
 * for.
 */
static int wave_needs_centre_raster(
    const pulseg_sequence_descriptor *desc,
    const pulseg_wave_key *key)
{
    const pulseg_grad_definition *gd;
    int d;

    for (d = 0; d < 3; ++d)
    {
        if (key->grad_def[d] < 0 || key->grad_def[d] >= desc->num_unique_grads)
            continue;
        gd = &desc->grad_definitions[key->grad_def[d]];
        if (gd->type != 0 && gd->unused_or_time_shape_id <= 0)
            return 1;
    }
    return 0;
}

static int float_cmp(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    if (fa < fb)
        return -1;
    if (fa > fb)
        return 1;
    return 0;
}

int pulseg_materialize_wave(
    const pulseg_collection *coll,
    int subseq_idx,
    const pulseg_wave_key *key,
    int out_axis,
    float *out_time_us,
    float *out_amp,
    int max_points,
    int *out_num_points,
    float *out_peak)
{
    const pulseg_sequence_descriptor *desc;
    axis_corners axis[3];
    float *merged = NULL;
    float rot[9];
    float peak, v, prev;
    int total, n_union, i, d, rc;

    if (!coll || !key || !out_num_points)
        return PULSEG_ERR_NULL_POINTER;
    if (subseq_idx < 0 || subseq_idx >= coll->num_subsequences)
        return PULSEG_ERR_INVALID_ARGUMENT;
    if (out_axis < PULSEG_GRAD_AXIS_X || out_axis > PULSEG_GRAD_AXIS_Z)
        return PULSEG_ERR_INVALID_ARGUMENT;

    desc = &coll->descriptors[subseq_idx];
    *out_num_points = 0;
    if (out_peak)
        *out_peak = 0.0f;

    for (d = 0; d < 3; ++d)
    {
        axis[d].t = NULL;
        axis[d].v = NULL;
        axis[d].n = 0;
    }

    rc = PULSEG_SUCCESS;
    total = 0;
    for (d = 0; d < 3; ++d)
    {
        if (!axis_corners_build(desc, key->grad_def[d], key->shape_id[d], &axis[d]))
        {
            rc = PULSEG_ERR_ALLOC_FAILED;
            goto done;
        }
        total += axis[d].n;
    }
    if (total == 0)
        goto done;

    /*
     * The union of every axis' corner times.  No resampling onto a raster:
     * each axis is piecewise linear, so evaluating all of them at the union of
     * their breakpoints reproduces every one of them exactly, and their
     * rotated combination is piecewise linear on that same grid.  This is what
     * Pulseq's rotate3D does, and it keeps a rotated trapezoid four points
     * wide instead of one point per raster tick.
     */
    if (wave_needs_centre_raster(desc, key))
    {
        /* Centre grid, spanning every involved axis.  The start is raster-
         * aligned downwards so the samples land where the hardware expects
         * them however the axes' delays differ. */
        float raster = desc->grad_raster_us > 0.0f ? desc->grad_raster_us : 4.0f;
        float lo = 0.0f;
        float hi = 0.0f;
        int have = 0;

        for (d = 0; d < 3; ++d)
        {
            if (axis[d].n < 1)
                continue;
            if (!have || axis[d].t[0] < lo)
                lo = axis[d].t[0];
            if (!have || axis[d].t[axis[d].n - 1] > hi)
                hi = axis[d].t[axis[d].n - 1];
            have = 1;
        }
        if (!have || hi <= lo)
            goto done;

        lo = (float)floor((double)(lo / raster)) * raster;
        n_union = (int)ceil((double)((hi - lo) / raster));
        if (n_union < 1)
            n_union = 1;

        merged = (float *)PULSEG_ALLOC((size_t)n_union * sizeof(float));
        if (!merged)
        {
            rc = PULSEG_ERR_ALLOC_FAILED;
            goto done;
        }
        for (i = 0; i < n_union; ++i)
            merged[i] = lo + ((float)i + 0.5f) * raster;
    }
    else
    {
        merged = (float *)PULSEG_ALLOC((size_t)total * sizeof(float));
        if (!merged)
        {
            rc = PULSEG_ERR_ALLOC_FAILED;
            goto done;
        }
        n_union = 0;
        for (d = 0; d < 3; ++d)
            for (i = 0; i < axis[d].n; ++i)
                merged[n_union++] = axis[d].t[i];
        qsort(merged, (size_t)n_union, sizeof(float), float_cmp);

        /* Collapse coincident breakpoints; a shared corner is one corner. */
        total = 0;
        prev = 0.0f;
        for (i = 0; i < n_union; ++i)
        {
            if (i == 0 || merged[i] > prev)
            {
                merged[total++] = merged[i];
                prev = merged[i];
            }
        }
        n_union = total;
    }

    *out_num_points = n_union;
    if (!out_time_us || !out_amp)
        goto done;
    if (max_points < n_union)
    {
        rc = PULSEG_ERR_INDEX;
        goto done;
    }

    if (key->rotation_id >= 0 && key->rotation_id < desc->num_rotations && desc->rotation_matrices)
    {
        for (i = 0; i < 9; ++i)
            rot[i] = desc->rotation_matrices[key->rotation_id][i];
    }
    else
    {
        for (i = 0; i < 9; ++i)
            rot[i] = (i % 4 == 0) ? 1.0f : 0.0f;
    }

    peak = 0.0f;
    for (i = 0; i < n_union; ++i)
    {
        out_time_us[i] = merged[i];
        v = 0.0f;
        for (d = 0; d < 3; ++d)
            v += rot[out_axis * 3 + d] * key->ratio[d] * axis_value_at(&axis[d], merged[i]);
        out_amp[i] = v;
        if ((float)fabs((double)v) > peak)
            peak = (float)fabs((double)v);
    }

    /* Normalise to unit peak; the scalar goes back to the caller, which is
     * what lets one wave serve every instance that shares this key. */
    if (peak > 0.0f)
        for (i = 0; i < n_union; ++i)
            out_amp[i] /= peak;
    if (out_peak)
        *out_peak = peak;

done:
    if (merged)
        PULSEG_FREE(merged);
    for (d = 0; d < 3; ++d)
        axis_corners_free(&axis[d]);
    return rc;
}
