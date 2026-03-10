/**
 * @file example_scanloop.c
 * @brief Flat scan-table scan loop with PMC support.
 *
 * Demonstrates the recommended scan-loop architecture:
 *
 *   1. Load the cached sequence collection.
 *   2. Build per-subsequence frequency-modulation libraries,
 *      using a binary cache when available.
 *   3. Walk every block via pulseqlib_cursor_next(); use
 *      pulseqlib_cursor_get_info() for segment / TR boundaries,
 *      trigger flags, NAV status, and the scan-table position
 *      needed for freq-mod lookup.
 *   4. At segment boundaries: set FOV rotation, arm trigger, play.
 *   5. PMC-enabled subsequences: at main-TR boundaries, update the
 *      freq-mod library with the new position; after NAV segments,
 *      evaluate motion and optionally rescan the TR via
 *      pulseqlib_cursor_reset().
 *
 * Compile:
 *   cc -I../../csrc example_scanloop.c ../../csrc/pulseqlib_*.c -lm -o scanloop
 *
 * Run:
 *   ./scanloop path/to/sequence.seq
 */

#include "example_vendorlib.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(rc, diag)                                 \
    do {                                                \
        if (PULSEQLIB_FAILED(rc)) {                     \
            vendor_report_error(rc, (diag));             \
            goto fail;                                  \
        }                                               \
    } while (0)

/* ================================================================== */
/*  Vendor stubs                                                      */
/* ================================================================== */

/** @brief Program one block on the hardware sequencer.
 *
 * This is called once per block during the scan loop.  The
 * block_instance contains per-TR amplitude values (signed, physical
 * units) that are used to update the hardware amplitude registers
 * established during geninstruction:
 *
 *   grad: hw_amp = DAC_MAX × inst->gx_amp_hz_per_m / max_amp
 *   rf:   hw_amp = DAC_MAX × inst->rf_amp_hz       / max_amp
 *
 * where max_amp was stored at geninstruction time (see
 * example_geninstructions.c).  The normalised waveform shapes do
 * not change — only the amplitude scalar is updated each TR.
 */
static void vendor_set_block(const pulseqlib_block_instance* inst,
                             const float* fmod_waveform,
                             int fmod_nsamples,
                             float fmod_phase_rad)
{
    (void)inst; (void)fmod_waveform; (void)fmod_nsamples; (void)fmod_phase_rad;
}

/** @brief Set FOV rotation matrix for the next segment play. */
static void vendor_set_rotation(const float* rot) { (void)rot; }

/** @brief Arm the physio trigger gate for the next segment play. */
static void vendor_set_trigger(void) { }

/** @brief Issue hardware play for the prepared segment. */
static void vendor_play_segment(int seg_idx) { (void)seg_idx; }

/**
 * @brief Evaluate PMC (navigator) feedback.
 *
 * In a real driver this receives a motion estimate, updates the
 * shift / rotation, and returns 0 = accepted or 1 = rescan.
 */
static int vendor_get_pmc_feedback(float* shift)
{
    shift[2] += 0.001f;
    return 0;
}

/* ================================================================== */
/*  Main                                                              */
/* ================================================================== */

int main(int argc, char** argv)
{
    const char*           seq_path;
    pulseqlib_opts        opts = PULSEQLIB_OPTS_INIT;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_collection* coll = NULL;
    int rc, nsub;

    /* Patient-table / prescription shift (metres). */
    float fovshift[3]    = {0.05f, 0.0f, 0.0f};
    /* FOV rotation matrix (3x3 row-major, logical -> physical). */
    float fovrotation[9] = {1,0,0, 0,1,0, 0,0,1};

    /* Per-subsequence freq-mod collection (opaque, heap-allocated). */
    pulseqlib_freq_mod_collection* freqmods = NULL;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <sequence.seq>\n", argv[0]);
        return 1;
    }
    seq_path = argv[1];

    vendor_opts_init(&opts, 42577478.0f, 3.0f, 50.0f, 200.0f);

    /* ============================================================== */
    /*  1. Load (with cache + labels)                                 */
    /* ============================================================== */
    rc = pulseqlib_read(&coll, &diag, seq_path, &opts, 1, 1, 0, 1);
    CHECK(rc, &diag);

    {
        pulseqlib_collection_info ci = PULSEQLIB_COLLECTION_INFO_INIT;
        rc = pulseqlib_get_collection_info(coll, &ci);
        CHECK(rc, &diag);
        nsub = ci.num_subsequences;
        printf("Loaded: %d subsequences, %.2f s\n",
               nsub, ci.total_duration_us / 1e6);
    }

    /* ============================================================== */
    /*  2. Build freq-mod collection (with caching)                   */
    /* ============================================================== */
    {
        char cache_path[512];
        sprintf(cache_path, "%s.fmod.bin", seq_path);

        /* Try loading from cache first. */
        rc = pulseqlib_freq_mod_collection_read_cache(
            &freqmods, cache_path, coll, fovshift);

        if (PULSEQLIB_FAILED(rc)) {
            /* Cache miss or stale: build from scratch and store. */
            rc = pulseqlib_build_freq_mod_collection(
                &freqmods, coll, fovshift, fovrotation);
            CHECK(rc, &diag);

            rc = pulseqlib_freq_mod_collection_write_cache(
                freqmods, cache_path);
            CHECK(rc, &diag);

            printf("  freq-mod: built + cached\n");
        } else {
            printf("  freq-mod: loaded from cache\n");
        }
    }

    /* ============================================================== */
    /*  3. Scan loop                                                  */
    /* ============================================================== */
    {
        int n          = 0;   /* global block counter         */
        int prev_seg   = -1;  /* previous segment id          */
        int rescan     = 0;   /* 1 = PMC requests TR rescan   */

        pulseqlib_cursor_reset(coll);

        while (pulseqlib_cursor_next(coll) == PULSEQLIB_CURSOR_BLOCK) {
            pulseqlib_cursor_info    ci   = PULSEQLIB_CURSOR_INFO_INIT;
            pulseqlib_block_instance inst = PULSEQLIB_BLOCK_INSTANCE_INIT;
            const float* fmod_waveform = NULL;
            int   fmod_nsamples = 0;
            float fmod_phase    = 0.0f;

            rc = pulseqlib_cursor_get_info(coll, &ci);
            if (PULSEQLIB_FAILED(rc)) goto fail;

            /* ------------------------------------------------------ */
            /*  PMC: at main-TR start, update freq-mod and rescan     */
            /* ------------------------------------------------------ */
            if (ci.pmc && ci.tr_start) {
                rc = pulseqlib_update_freq_mod_collection(
                    freqmods, ci.subseq_idx, fovshift);
                if (PULSEQLIB_FAILED(rc)) goto fail;

                if (rescan) {
                    pulseqlib_cursor_reset(coll);
                    rescan = 0;
                    continue;
                }
                pulseqlib_cursor_mark(coll);
            }

            /* ------------------------------------------------------ */
            /*  New segment: play the previous one, set up the next   */
            /* ------------------------------------------------------ */
            if (ci.segment_id != prev_seg) {
                if (prev_seg >= 0)
                    vendor_play_segment(prev_seg);

                vendor_set_rotation(fovrotation);
                if (ci.has_trigger)
                    vendor_set_trigger();

                prev_seg = ci.segment_id;
            }

            /* ------------------------------------------------------ */
            /*  Get block + freq-mod, program hardware                */
            /* ------------------------------------------------------ */
            rc = pulseqlib_get_block_instance(coll, &inst);
            if (PULSEQLIB_FAILED(rc)) goto fail;

            if (freqmods)
                pulseqlib_freq_mod_collection_get(
                    freqmods, ci.subseq_idx, ci.scan_pos,
                    &fmod_waveform, &fmod_nsamples, &fmod_phase);

            vendor_set_block(&inst, fmod_waveform,
                             fmod_nsamples, fmod_phase);
            ++n;

            /* ------------------------------------------------------ */
            /*  Segment end: play + PMC feedback after NAV            */
            /* ------------------------------------------------------ */
            if (ci.segment_end) {
                vendor_play_segment(ci.segment_id);
                prev_seg = -1;

                if (ci.pmc && ci.is_nav)
                    rescan = vendor_get_pmc_feedback(fovshift);
            }
        }

        printf("\nScan loop complete: %d blocks\n", n);
    }

    /* ============================================================== */
    /*  Cleanup                                                       */
    /* ============================================================== */
    pulseqlib_freq_mod_collection_free(freqmods);
    pulseqlib_collection_free(coll);
    return 0;

fail:
    if (freqmods) pulseqlib_freq_mod_collection_free(freqmods);
    if (coll) pulseqlib_collection_free(coll);
    return 1;
}
