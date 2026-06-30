/*
 * test_seg_helpers.h -- ground-truth file parsers for segmentation tests.
 *
 * Provides parse_meta() for the MATLAB-generated _meta.txt files and
 * parse_tr_waveform() for the binary _tr_waveform.bin files.
 */
#ifndef TEST_SEG_HELPERS_H
#define TEST_SEG_HELPERS_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __GNUC__
#define TSEG_MAYBE_UNUSED __attribute__((unused))
#else
#define TSEG_MAYBE_UNUSED
#endif

/* ------------------------------------------------------------------ */
/*  Meta struct — mirrors quantities from example_check.c steps 5-6   */
/* ------------------------------------------------------------------ */

#define MAX_UNIQUE_ADCS 16
#define MAX_SEGMENTS    16

typedef struct seg_meta {
    /* Phase 1: ADC / TR (step 6) */
    int num_unique_adcs;
    int adc_samples[MAX_UNIQUE_ADCS];
    int adc_dwell_ns[MAX_UNIQUE_ADCS];
    int max_b1_subseq;
    int tr_duration_us;
    /* Phase 2: Segment structure (step 5) */
    int num_segments;
    int segment_num_blocks[MAX_SEGMENTS];
    int num_canonical_trs;
    int fmod_build_mode_tr_scoped;
    /* Phase 5: Segment order (scan-table validation) */
    int num_segment_order_entries;   /* actual count of entries in segment_order */
    int segment_order[MAX_SEGMENTS];
} seg_meta;

#define SEG_META_INIT {0, {0}, {0}, 0, 0, 0, {0}, 0, 0, 0, {0}}

/* ------------------------------------------------------------------ */
/*  parse_meta                                                        */
/* ------------------------------------------------------------------ */

static TSEG_MAYBE_UNUSED int parse_meta(const char* path, seg_meta* out)
{
    FILE* f;
    char line[512];
    char key[64];
    char suffix[32];
    int val, idx, i, n;
    char sval[64];
    seg_meta m = SEG_META_INIT;

    f = fopen(path, "r");
    if (!f) return 0;

    while (fgets(line, sizeof(line), f) != NULL) {
        /* Parse key and first value */
        if (sscanf(line, "%63s", key) < 1)
            continue;

        if (strcmp(key, "fmod_build_mode") == 0) {
            if (sscanf(line, "%63s %63s", key, sval) == 2) {
                m.fmod_build_mode_tr_scoped = (strcmp(sval, "tr_scoped") == 0) ? 1 : 0;
            }
            continue;
        }

        if (sscanf(line, "%63s %d", key, &val) < 2)
            continue;

        if (strcmp(key, "num_unique_adcs") == 0) {
            m.num_unique_adcs = val;
        } else if (sscanf(key, "adc_%d_%31s", &idx, suffix) == 2) {
            if (idx >= 0 && idx < MAX_UNIQUE_ADCS) {
                if (strcmp(suffix, "samples") == 0)
                    m.adc_samples[idx] = val;
                else if (strcmp(suffix, "dwell_ns") == 0)
                    m.adc_dwell_ns[idx] = val;
            }
        } else if (strcmp(key, "max_b1_subseq") == 0) {
            m.max_b1_subseq = val;
        } else if (strcmp(key, "tr_duration_us") == 0) {
            m.tr_duration_us = val;
        } else if (strcmp(key, "num_segments") == 0) {
            m.num_segments = val;
        } else if (sscanf(key, "segment_%d_%31s", &idx, suffix) == 2) {
            if (idx >= 0 && idx < MAX_SEGMENTS) {
                if (strcmp(suffix, "num_blocks") == 0)
                    m.segment_num_blocks[idx] = val;
            }
        } else if (strcmp(key, "num_canonical_trs") == 0) {
            m.num_canonical_trs = val;
        } else if (strcmp(key, "segment_order") == 0) {
            /* Parse multi-value segment_order line:
             * "segment_order 0 1 2 2 2 1" */
            char* pos = line + strlen("segment_order");
            for (n = 0; n < MAX_SEGMENTS && sscanf(pos, " %d%n", &m.segment_order[n], &i) == 1; n++) {
                pos += i;
            }
            m.num_segment_order_entries = n;
        }
    }

    fclose(f);
    *out = m;
    return 1;
}

/* ------------------------------------------------------------------ */
/*  TR waveform struct + binary parser                                */
/* ------------------------------------------------------------------ */

typedef struct seg_tr_waveform {
    int    num_samples;
    float* time_us;
    float* gx;
    float* gy;
    float* gz;
} seg_tr_waveform;

#define SEG_TR_WAVEFORM_INIT {0, NULL, NULL, NULL, NULL}

static TSEG_MAYBE_UNUSED void free_tr_waveform(seg_tr_waveform* w)
{
    if (!w) return;
    free(w->time_us); w->time_us = NULL;
    free(w->gx);      w->gx = NULL;
    free(w->gy);      w->gy = NULL;
    free(w->gz);       w->gz = NULL;
    w->num_samples = 0;
}

#define MAX_CANONICAL_TRS 256

typedef struct seg_tr_waveform_set {
    int num_trs;
    seg_tr_waveform waveforms[MAX_CANONICAL_TRS];
} seg_tr_waveform_set;

#define SEG_TR_WAVEFORM_SET_INIT {0, {{0}}}

static TSEG_MAYBE_UNUSED void free_tr_waveform_set(seg_tr_waveform_set* s)
{
    int i;
    if (!s) return;
    for (i = 0; i < s->num_trs; ++i)
        free_tr_waveform(&s->waveforms[i]);
    s->num_trs = 0;
}

/**
 * Parse binary TR waveform file (multi-TR format).
 * Layout: int32 num_canonical_trs, then for each TR:
 *   int32 num_samples, then 4 contiguous float32 arrays:
 *   time_us[N], gx[N], gy[N], gz[N].
 * Returns 1 on success, 0 on failure.
 */
static TSEG_MAYBE_UNUSED int parse_tr_waveform_set(const char* path, seg_tr_waveform_set* out)
{
    FILE* f;
    int m, i, n;
    size_t ns;
    seg_tr_waveform_set ws = SEG_TR_WAVEFORM_SET_INIT;

    f = fopen(path, "rb");
    if (!f) return 0;

    if (fread(&m, sizeof(int), 1, f) != 1 || m <= 0 || m > MAX_CANONICAL_TRS) {
        fclose(f);
        return 0;
    }
    ws.num_trs = m;

    for (i = 0; i < m; ++i) {
        if (fread(&n, sizeof(int), 1, f) != 1 || n <= 0) {
            free_tr_waveform_set(&ws);
            fclose(f);
            return 0;
        }
        ns = (size_t)n;
        ws.waveforms[i].num_samples = n;
        ws.waveforms[i].time_us = (float*)malloc(ns * sizeof(float));
        ws.waveforms[i].gx      = (float*)malloc(ns * sizeof(float));
        ws.waveforms[i].gy      = (float*)malloc(ns * sizeof(float));
        ws.waveforms[i].gz      = (float*)malloc(ns * sizeof(float));
        if (!ws.waveforms[i].time_us || !ws.waveforms[i].gx ||
            !ws.waveforms[i].gy      || !ws.waveforms[i].gz) {
            free_tr_waveform_set(&ws);
            fclose(f);
            return 0;
        }
        if (fread(ws.waveforms[i].time_us, sizeof(float), ns, f) != ns ||
            fread(ws.waveforms[i].gx,      sizeof(float), ns, f) != ns ||
            fread(ws.waveforms[i].gy,      sizeof(float), ns, f) != ns ||
            fread(ws.waveforms[i].gz,      sizeof(float), ns, f) != ns) {
            free_tr_waveform_set(&ws);
            fclose(f);
            return 0;
        }
    }

    fclose(f);
    *out = ws;
    return 1;
}

/* ------------------------------------------------------------------ */
/*  Phase 3: Block-level ground truth (geninstruction tests)          */
/* ------------------------------------------------------------------ */

#define MAX_BLOCKS 8

/** Per-block timing + per-axis trapezoid corners. */
typedef struct block_meta {
    int num_blocks;

    /* Per-block timing */
    int duration_us[MAX_BLOCKS];
    int start_time_us[MAX_BLOCKS];

    /* RF (block 0) */
    int rf_delay_us;
    int rf_num_samples;
    int rf_is_complex;
    int rf_num_channels;

    /* Trap gradient corners (indexed by [block][axis 0=x,1=y,2=z]).
     * Only filled for blocks/axes that are actual trapezoids. */
    int   has_trap[MAX_BLOCKS][3];
    float trap_amplitude[MAX_BLOCKS][3];
    int   trap_rise_us[MAX_BLOCKS][3];
    int   trap_flat_us[MAX_BLOCKS][3];
    int   trap_fall_us[MAX_BLOCKS][3];
    int   trap_delay_us[MAX_BLOCKS][3];

    /* Arb gradient metadata (indexed by [block][axis]). */
    int has_arb[MAX_BLOCKS][3];
    int arb_num_samples[MAX_BLOCKS][3];
    int arb_delay_us[MAX_BLOCKS][3];

    /* ADC */
    int adc_delay_us;

    /* Segment gap */
    int rf_adc_gap_us;
} block_meta;

#define BLOCK_META_INIT { \
    0, {0}, {0}, \
    0, 0, 0, 0, \
    {{0}}, {{0}}, {{0}}, {{0}}, {{0}}, {{0}}, \
    {{0}}, {{0}}, {{0}}, \
    0, 0 \
}

static TSEG_MAYBE_UNUSED int parse_block_meta(const char* path, block_meta* out)
{
    FILE* f;
    char key[80];
    char val_str[80];
    block_meta m = BLOCK_META_INIT;
    int idx;
    char axis_ch, suffix[40];

    f = fopen(path, "r");
    if (!f) return 0;

    while (fscanf(f, "%79s %79s", key, val_str) == 2) {
        /* Per-block timing */
        if (sscanf(key, "block_%d_duration_us", &idx) == 1 && idx < MAX_BLOCKS) {
            m.duration_us[idx] = atoi(val_str);
            if (idx >= m.num_blocks) m.num_blocks = idx + 1;
        }
        else if (sscanf(key, "block_%d_start_time_us", &idx) == 1 && idx < MAX_BLOCKS) {
            m.start_time_us[idx] = atoi(val_str);
        }
        /* RF (block 0) */
        else if (strcmp(key, "block_0_rf_delay_us") == 0)     m.rf_delay_us     = atoi(val_str);
        else if (strcmp(key, "block_0_rf_num_samples") == 0)   m.rf_num_samples  = atoi(val_str);
        else if (strcmp(key, "block_0_rf_is_complex") == 0)    m.rf_is_complex   = atoi(val_str);
        else if (strcmp(key, "block_0_rf_num_channels") == 0)  m.rf_num_channels = atoi(val_str);
        /* Trap gradients: block_B_gA_suffix */
        else if (sscanf(key, "block_%d_g%c_%39s", &idx, &axis_ch, suffix) == 3
                 && idx < MAX_BLOCKS) {
            int ax = (axis_ch == 'x') ? 0 : (axis_ch == 'y') ? 1 : 2;
            if (strcmp(suffix, "amplitude_hz_m") == 0) {
                m.has_trap[idx][ax] = 1;
                m.trap_amplitude[idx][ax] = (float)atof(val_str);
            }
            else if (strcmp(suffix, "rise_us") == 0)  m.trap_rise_us[idx][ax]  = atoi(val_str);
            else if (strcmp(suffix, "flat_us") == 0)   m.trap_flat_us[idx][ax]  = atoi(val_str);
            else if (strcmp(suffix, "fall_us") == 0)   m.trap_fall_us[idx][ax]  = atoi(val_str);
            else if (strcmp(suffix, "delay_us") == 0)  m.trap_delay_us[idx][ax] = atoi(val_str);
            /* Arb keys */
            else if (strcmp(suffix, "is_arb") == 0)       m.has_arb[idx][ax]         = atoi(val_str);
            else if (strcmp(suffix, "num_samples") == 0)   m.arb_num_samples[idx][ax] = atoi(val_str);
        }
        /* Arb keys without axis in gradient pattern — use block_B_gA_KEY */
        /* ADC */
        else if (strcmp(key, "block_2_adc_delay_us") == 0) m.adc_delay_us = atoi(val_str);
        /* Segment gap */
        else if (strcmp(key, "rf_adc_gap_us") == 0) m.rf_adc_gap_us = atoi(val_str);
    }

    fclose(f);
    *out = m;
    return 1;
}

/* ------------------------------------------------------------------ */
/*  Segment definition binary (Phase 3 ground truth)                 */
/* ------------------------------------------------------------------ */

/*
 * Binary layout written by export_segment_def() in MATLAB:
 *
 *   int32  num_segments
 *   for each segment:
 *     int32  num_blocks
 *     for each block:
 *       uint8  flags  (bit 0=rf, 1=gx, 2=gy, 3=gz, 4=adc,
 *                       5=rotation, 6=digital_out, 7=freq_mod)
 *       float32 rf_delay,  rf_amp,  rf_raster_us
 *       int32   rf_n;  float32 rf_rho[rf_n];  float32 rf_time_s[rf_n]
 *       for axis in {x,y,z}:
 *         float32 grad_delay,  grad_amp
 *         int32   grad_n;  float32 grad_wave[grad_n];  float32 grad_time_s[grad_n]
 *       float32 adc_delay;  int32 adc_def_id
 *       float32 digital_out_delay, digital_out_duration
 *       int32   freq_mod_num_samples;  int32 freq_mod_def_id
 */

#define SEG_DEF_MAX_BLOCKS   80
#define SEG_DEF_MAX_WAVE     4096   /* generous upper bound per axis */

typedef struct seg_block_def {
    /* flags */
    int has_rf;
    int has_grad[3];   /* [0]=x [1]=y [2]=z */
    int has_adc;
    int has_rotation;
    int has_digital_out;
    int has_freq_mod;

    /* RF */
    float rf_delay;
    float rf_amp;
    float rf_raster_us;   /* rfRasterTime in us (new field) */
    int   rf_n;
    float rf_rho[SEG_DEF_MAX_WAVE];
    float rf_time_s[SEG_DEF_MAX_WAVE];   /* sample times in seconds (0-based) */

    /* Gradients (x,y,z) */
    float grad_delay[3];
    float grad_amp[3];
    int   grad_n[3];
    float grad_wave[3][SEG_DEF_MAX_WAVE];
    float grad_time_s[3][SEG_DEF_MAX_WAVE];  /* knot/sample times in seconds (0-based) */

    /* ADC */
    float adc_delay;
    int   adc_def_id;   /* 0-based ADC definition index (-1 if none) */

    /* Digital output */
    float digital_out_delay;
    float digital_out_duration;

    /* Freq-mod */
    int freq_mod_num_samples;
    int freq_mod_def_id;  /* 0-based freq-mod definition index (-1 if none) */

    /* Anchors (us, relative to segment start; -1 if absent) */
    float rf_isocenter_us;
    float adc_kzero_us;
} seg_block_def;

typedef struct seg_def_file {
    int           num_segments;
    int           num_blocks[MAX_SEGMENTS];
    seg_block_def blocks[MAX_SEGMENTS][SEG_DEF_MAX_BLOCKS];

    /* Segment-level gaps (us; -1 if not applicable) */
    float rf_adc_gap_us[MAX_SEGMENTS];
    float adc_adc_gap_us[MAX_SEGMENTS];
} seg_def_file;

#define SEG_DEF_FILE_INIT {0, {0}, {{{0}}}}

static TSEG_MAYBE_UNUSED int parse_seg_def(const char* path, seg_def_file* out)
{
    FILE* f;
    int s, b, ax, n;
    memset(out, 0, sizeof(*out));

    f = fopen(path, "rb");
    if (!f) return 0;

#define RD4(dst)  if (fread(&(dst), 4, 1, f) != 1) { fclose(f); return 0; }
#define RDU1(dst) { unsigned char _u; if (fread(&_u, 1, 1, f) != 1) { fclose(f); return 0; } (dst) = _u; }
#define RDF(dst)  { float _v; if (fread(&_v, sizeof(float), 1, f) != 1) { fclose(f); return 0; } (dst) = _v; }
#define RDFN(arr,n) if ((n) > 0) { if (fread((arr), sizeof(float), (size_t)(n), f) != (size_t)(n)) { fclose(f); return 0; } }

    RD4(out->num_segments);
    if (out->num_segments < 0 || out->num_segments > MAX_SEGMENTS) {
        fprintf(stderr, "[parse_seg_def] %s: num_segments=%d exceeds MAX_SEGMENTS=%d\n",
                path, out->num_segments, MAX_SEGMENTS);
        fclose(f); return 0;
    }

    for (s = 0; s < out->num_segments; ++s) {
        RD4(out->num_blocks[s]);
        if (out->num_blocks[s] < 0 || out->num_blocks[s] > SEG_DEF_MAX_BLOCKS) {
            fprintf(stderr, "[parse_seg_def] %s: segment %d num_blocks=%d exceeds SEG_DEF_MAX_BLOCKS=%d\n",
                path, s, out->num_blocks[s], SEG_DEF_MAX_BLOCKS);
            fclose(f); return 0;
        }

        for (b = 0; b < out->num_blocks[s]; ++b) {
            seg_block_def* blk = &out->blocks[s][b];
            int flags;
            RDU1(flags);
            blk->has_rf          = (flags >> 0) & 1;
            blk->has_grad[0]     = (flags >> 1) & 1;
            blk->has_grad[1]     = (flags >> 2) & 1;
            blk->has_grad[2]     = (flags >> 3) & 1;
            blk->has_adc         = (flags >> 4) & 1;
            blk->has_rotation    = (flags >> 5) & 1;
            blk->has_digital_out = (flags >> 6) & 1;
            blk->has_freq_mod    = (flags >> 7) & 1;

            /* RF */
            RDF(blk->rf_delay);
            RDF(blk->rf_amp);
            RDF(blk->rf_raster_us);   /* new: rfRasterTime in us */
            RD4(n);
            blk->rf_n = n;
            if (n > SEG_DEF_MAX_WAVE) {
                fprintf(stderr, "[parse_seg_def] %s: segment %d block %d rf_n=%d exceeds SEG_DEF_MAX_WAVE=%d\n",
                        path, s, b, n, SEG_DEF_MAX_WAVE);
                fclose(f); return 0;
            }
            RDFN(blk->rf_rho, n);
            RDFN(blk->rf_time_s, n);

            /* Gradients */
            for (ax = 0; ax < 3; ++ax) {
                RDF(blk->grad_delay[ax]);
                RDF(blk->grad_amp[ax]);
                RD4(n);
                blk->grad_n[ax] = n;
                if (n > SEG_DEF_MAX_WAVE) {
                    fprintf(stderr, "[parse_seg_def] %s: segment %d block %d axis %d grad_n=%d exceeds SEG_DEF_MAX_WAVE=%d\n",
                            path, s, b, ax, n, SEG_DEF_MAX_WAVE);
                    fclose(f); return 0;
                }
                RDFN(blk->grad_wave[ax], n);
                RDFN(blk->grad_time_s[ax], n);
            }

            /* ADC */
            RDF(blk->adc_delay);
            RD4(blk->adc_def_id);

            /* Digital output */
            RDF(blk->digital_out_delay);
            RDF(blk->digital_out_duration);

            /* Freq-mod */
            RD4(blk->freq_mod_num_samples);
            RD4(blk->freq_mod_def_id);

            /* Anchors */
            RDF(blk->rf_isocenter_us);
            RDF(blk->adc_kzero_us);
        }

        /* Segment-level gaps */
        RDF(out->rf_adc_gap_us[s]);
        RDF(out->adc_adc_gap_us[s]);
    }

#undef RD4
#undef RDU1
#undef RDF
#undef RDFN

    fclose(f);
    return 1;
}

/* ------------------------------------------------------------------ */
/*  RF magnitude waveform (binary float32)                            */
/* ------------------------------------------------------------------ */

typedef struct rf_mag_waveform {
    int    num_samples;
    float* magnitude;
} rf_mag_waveform;

#define RF_MAG_WAVEFORM_INIT {0, NULL}

static TSEG_MAYBE_UNUSED void free_rf_mag(rf_mag_waveform* w)
{
    if (!w) return;
    free(w->magnitude); w->magnitude = NULL;
    w->num_samples = 0;
}

static TSEG_MAYBE_UNUSED int parse_rf_mag(const char* path, rf_mag_waveform* out)
{
    FILE* f;
    int n;
    size_t ns;
    rf_mag_waveform w = RF_MAG_WAVEFORM_INIT;

    f = fopen(path, "rb");
    if (!f) return 0;

    if (fread(&n, sizeof(int), 1, f) != 1 || n <= 0) { fclose(f); return 0; }
    ns = (size_t)n;

    w.num_samples = n;
    w.magnitude = (float*)malloc(ns * sizeof(float));
    if (!w.magnitude) { fclose(f); return 0; }

    if (fread(w.magnitude, sizeof(float), ns, f) != ns) {
        free_rf_mag(&w);
        fclose(f);
        return 0;
    }

    fclose(f);
    *out = w;
    return 1;
}

/* ------------------------------------------------------------------ */
/*  Arbitrary gradient waveform (binary float32)                      */
/* ------------------------------------------------------------------ */

typedef struct arb_grad_waveform {
    int    num_samples;
    float* amplitude;
    float* time_us;
} arb_grad_waveform;

#define ARB_GRAD_WAVEFORM_INIT {0, NULL, NULL}

static TSEG_MAYBE_UNUSED void free_arb_grad(arb_grad_waveform* w)
{
    if (!w) return;
    free(w->amplitude); w->amplitude = NULL;
    free(w->time_us);   w->time_us = NULL;
    w->num_samples = 0;
}

static TSEG_MAYBE_UNUSED int parse_arb_grad(const char* path, arb_grad_waveform* out)
{
    FILE* f;
    int n;
    size_t ns;
    arb_grad_waveform w = ARB_GRAD_WAVEFORM_INIT;

    f = fopen(path, "rb");
    if (!f) return 0;

    if (fread(&n, sizeof(int), 1, f) != 1 || n <= 0) { fclose(f); return 0; }
    ns = (size_t)n;

    w.num_samples = n;
    w.amplitude = (float*)malloc(ns * sizeof(float));
    w.time_us   = (float*)malloc(ns * sizeof(float));
    if (!w.amplitude || !w.time_us) {
        free_arb_grad(&w);
        fclose(f);
        return 0;
    }

    if (fread(w.amplitude, sizeof(float), ns, f) != ns ||
        fread(w.time_us,   sizeof(float), ns, f) != ns) {
        free_arb_grad(&w);
        fclose(f);
        return 0;
    }

    fclose(f);
    *out = w;
    return 1;
}

/* ------------------------------------------------------------------ */
/*  Frequency modulation definitions (binary)                         */
/* ------------------------------------------------------------------ */

#define MAX_FMOD_DEFS   8
#define MAX_FMOD_SAMPLES 4096

typedef struct fmod_def {
    int   type;           /* 0=RF, 1=ADC */
    int   num_samples;
    float raster_us;
    float duration_us;
    float ref_time_us;
    float ref_integral[3];
    float waveform_gx[MAX_FMOD_SAMPLES];
    float waveform_gy[MAX_FMOD_SAMPLES];
    float waveform_gz[MAX_FMOD_SAMPLES];
} fmod_def;

typedef struct fmod_def_file {
    int      num_defs;
    fmod_def defs[MAX_FMOD_DEFS];
} fmod_def_file;

#define FMOD_DEF_FILE_INIT {0, {{0,0,0,0,0,{0},{0},{0},{0}}}}

static TSEG_MAYBE_UNUSED int parse_fmod_defs(const char* path, fmod_def_file* out)
{
    FILE* f;
    int d, ns;
    memset(out, 0, sizeof(*out));

    f = fopen(path, "rb");
    if (!f) return 0;

    if (fread(&out->num_defs, sizeof(int), 1, f) != 1 ||
        out->num_defs < 0 || out->num_defs > MAX_FMOD_DEFS) {
        fprintf(stderr, "[parse_fmod_defs] num_defs=%d (max=%d) file=%s\n",
                out->num_defs, MAX_FMOD_DEFS, path);
        fclose(f); return 0;
    }

    for (d = 0; d < out->num_defs; ++d) {
        fmod_def* fd = &out->defs[d];
        if (fread(&fd->type, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
        if (fread(&fd->num_samples, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
        ns = fd->num_samples;
        if (ns <= 0 || ns > MAX_FMOD_SAMPLES) {
            fprintf(stderr, "[parse_fmod_defs] def[%d] ns=%d (max=%d) file=%s\n",
                    d, ns, MAX_FMOD_SAMPLES, path);
            fclose(f); return 0;
        }
        if (fread(&fd->raster_us, sizeof(float), 1, f) != 1) { fclose(f); return 0; }
        if (fread(&fd->duration_us, sizeof(float), 1, f) != 1) { fclose(f); return 0; }
        if (fread(&fd->ref_time_us, sizeof(float), 1, f) != 1) { fclose(f); return 0; }
        if (fread(fd->ref_integral, sizeof(float), 3, f) != 3) { fclose(f); return 0; }
        if (fread(fd->waveform_gx, sizeof(float), (size_t)ns, f) != (size_t)ns) { fclose(f); return 0; }
        if (fread(fd->waveform_gy, sizeof(float), (size_t)ns, f) != (size_t)ns) { fclose(f); return 0; }
        if (fread(fd->waveform_gz, sizeof(float), (size_t)ns, f) != (size_t)ns) { fclose(f); return 0; }
    }

    fclose(f);
    return 1;
}

/* ------------------------------------------------------------------ */
/*  Scan table ground truth (binary)                                  */
/* ------------------------------------------------------------------ */

#define MAX_SCAN_TABLE_ENTRIES 4096

typedef struct scan_table_entry {
    int   tr_start_flag;
    int   segment_id;
    int   within_segment_idx;
    float rf_amp_hz;
    float rf_phase_rad;
    float rf_freq_hz;
    float gx_amp_hz_per_m;
    float gy_amp_hz_per_m;
    float gz_amp_hz_per_m;
    int   adc_flag;
    float adc_phase_rad;
    float adc_freq_hz;
    int   digitalout_flag;
    int   trigger_flag;
    float rotmat[9];
    int   freq_mod_id;    /* 0=none, 1-based fmod def index */
    int   block_dur_us;   /* block duration in microseconds */
} scan_table_entry;

typedef struct scan_table_file {
    int               num_entries;
    scan_table_entry  entries[MAX_SCAN_TABLE_ENTRIES];
} scan_table_file;

#define SCAN_TABLE_FILE_INIT {0, {{0}}}

static TSEG_MAYBE_UNUSED int parse_scan_table(const char* path, scan_table_file* out)
{
    FILE* f;
    int i;
    memset(out, 0, sizeof(*out));

    f = fopen(path, "rb");
    if (!f) return 0;

    if (fread(&out->num_entries, sizeof(int), 1, f) != 1 ||
        out->num_entries < 0 || out->num_entries > MAX_SCAN_TABLE_ENTRIES) {
        fclose(f); return 0;
    }

    for (i = 0; i < out->num_entries; ++i) {
        scan_table_entry* e = &out->entries[i];
        if (fread(&e->tr_start_flag,    sizeof(int),   1, f) != 1) { fclose(f); return 0; }
        if (fread(&e->segment_id,       sizeof(int),   1, f) != 1) { fclose(f); return 0; }
        if (fread(&e->within_segment_idx,sizeof(int),  1, f) != 1) { fclose(f); return 0; }
        if (fread(&e->rf_amp_hz,       sizeof(float), 1, f) != 1) { fclose(f); return 0; }
        if (fread(&e->rf_phase_rad,     sizeof(float), 1, f) != 1) { fclose(f); return 0; }
        if (fread(&e->rf_freq_hz,       sizeof(float), 1, f) != 1) { fclose(f); return 0; }
        if (fread(&e->gx_amp_hz_per_m,  sizeof(float), 1, f) != 1) { fclose(f); return 0; }
        if (fread(&e->gy_amp_hz_per_m,  sizeof(float), 1, f) != 1) { fclose(f); return 0; }
        if (fread(&e->gz_amp_hz_per_m,  sizeof(float), 1, f) != 1) { fclose(f); return 0; }
        if (fread(&e->adc_flag,         sizeof(int),   1, f) != 1) { fclose(f); return 0; }
        if (fread(&e->adc_phase_rad,    sizeof(float), 1, f) != 1) { fclose(f); return 0; }
        if (fread(&e->adc_freq_hz,      sizeof(float), 1, f) != 1) { fclose(f); return 0; }
        if (fread(&e->digitalout_flag,  sizeof(int),   1, f) != 1) { fclose(f); return 0; }
        if (fread(&e->trigger_flag,     sizeof(int),   1, f) != 1) { fclose(f); return 0; }
        if (fread(e->rotmat,            sizeof(float), 9, f) != 9) { fclose(f); return 0; }
        if (fread(&e->freq_mod_id,      sizeof(int),   1, f) != 1) { fclose(f); return 0; }
        /* block_dur_us is appended after freq_mod_id; older files may not have it */
        if (fread(&e->block_dur_us,     sizeof(int),   1, f) != 1)
            e->block_dur_us = 0;
    }

    fclose(f);
    return 1;
}

#define MAX_FMOD_PLAN_PROBES   8
#define MAX_FMOD_PLAN_ENTRIES  2048

typedef struct fmod_plan_entry {
    int   def_id;  /* 1-based fmod def id */
    int   num_samples;
    float rotmat[9];
    float phase_active[MAX_FMOD_PLAN_PROBES];
    float phase_extra[MAX_FMOD_PLAN_PROBES];
    float phase_total[MAX_FMOD_PLAN_PROBES];
    float waveforms[MAX_FMOD_PLAN_PROBES][MAX_FMOD_SAMPLES];
} fmod_plan_entry;

typedef struct fmod_plan_file {
    int num_probes;
    float probes[MAX_FMOD_PLAN_PROBES][3];
    int num_plans;
    int max_samples;
    fmod_plan_entry plans[MAX_FMOD_PLAN_ENTRIES];
    int scan_len;
    int scan_to_plan[MAX_SCAN_TABLE_ENTRIES];
} fmod_plan_file;

#define FMOD_PLAN_FILE_INIT {0, {{0}}, 0, 0, {{0}}, 0, {0}}

static TSEG_MAYBE_UNUSED int parse_fmod_plan(const char* path, fmod_plan_file* out)
{
    FILE* f;
    int i, q;
    memset(out, 0, sizeof(*out));

    f = fopen(path, "rb");
    if (!f) return 0;

    if (fread(&out->num_probes, sizeof(int), 1, f) != 1 ||
        out->num_probes < 0 || out->num_probes > MAX_FMOD_PLAN_PROBES) {
        fclose(f); return 0;
    }
    if (out->num_probes > 0) {
        if (fread(out->probes, sizeof(float), (size_t)out->num_probes * 3, f)
            != (size_t)out->num_probes * 3) {
            fclose(f); return 0;
        }
    }

    if (fread(&out->num_plans, sizeof(int), 1, f) != 1 ||
        out->num_plans < 0 || out->num_plans > MAX_FMOD_PLAN_ENTRIES) {
        fclose(f); return 0;
    }
    if (fread(&out->max_samples, sizeof(int), 1, f) != 1 ||
        out->max_samples < 0 || out->max_samples > MAX_FMOD_SAMPLES) {
        fclose(f); return 0;
    }

    for (i = 0; i < out->num_plans; ++i) {
        fmod_plan_entry* e = &out->plans[i];
        if (fread(&e->def_id, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
        if (fread(&e->num_samples, sizeof(int), 1, f) != 1) { fclose(f); return 0; }
        if (e->num_samples < 0 || e->num_samples > out->max_samples) { fclose(f); return 0; }
        if (fread(e->rotmat, sizeof(float), 9, f) != 9) { fclose(f); return 0; }
        if (fread(e->phase_active, sizeof(float), (size_t)out->num_probes, f)
            != (size_t)out->num_probes) { fclose(f); return 0; }
        if (fread(e->phase_extra, sizeof(float), (size_t)out->num_probes, f)
            != (size_t)out->num_probes) { fclose(f); return 0; }
        if (fread(e->phase_total, sizeof(float), (size_t)out->num_probes, f)
            != (size_t)out->num_probes) { fclose(f); return 0; }
        for (q = 0; q < out->num_probes; ++q) {
            if (fread(e->waveforms[q], sizeof(float), (size_t)out->max_samples, f)
                != (size_t)out->max_samples) { fclose(f); return 0; }
        }
    }

    if (fread(&out->scan_len, sizeof(int), 1, f) != 1 ||
        out->scan_len < 0 || out->scan_len > MAX_SCAN_TABLE_ENTRIES) {
        fclose(f); return 0;
    }
    if (out->scan_len > 0) {
        if (fread(out->scan_to_plan, sizeof(int), (size_t)out->scan_len, f)
            != (size_t)out->scan_len) {
            fclose(f); return 0;
        }
    }

    fclose(f);
    return 1;
}

#endif /* TEST_SEG_HELPERS_H */
