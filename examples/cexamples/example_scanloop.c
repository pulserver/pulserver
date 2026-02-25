/**
 * @file example_scanloop.c
 * @brief Flat scan-table scan loop with PMC support.
 *
 * Demonstrates the recommended scan-loop architecture:
 *
 *   1. Load the cached sequence collection.
 *   2. Build per-subsequence frequency-modulation libraries.
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

#include "pulseqlib_methods.h"
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

/** @brief Program one block on the hardware sequencer. */
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
    int rc, nsub, s;

    /* Patient-table / prescription shift (metres). */
    float fovshift[3]    = {0.05f, 0.0f, 0.0f};
    /* FOV rotation matrix (3x3 row-major, logical -> physical). */
    float fovrotation[9] = {1,0,0, 0,1,0, 0,0,1};

    /* Per-subsequence freq-mod libraries (opaque, heap-allocated). */
    pulseqlib_freq_mod_library** freqlibs = NULL;
    int nlibs = 0;

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

    nsub = pulseqlib_get_num_subsequences(coll);
    printf("Loaded: %d subsequences, %.2f s\n",
           nsub, pulseqlib_get_total_duration_us(coll) / 1e6);

    /* ============================================================== */
    /*  2. Build per-subsequence freq-mod libraries                   */
    /* ============================================================== */
    freqlibs = (pulseqlib_freq_mod_library**)calloc(
        (size_t)nsub, sizeof(*freqlibs));
    if (!freqlibs) { rc = PULSEQLIB_ERR_ALLOC_FAILED; goto fail; }
    nlibs = nsub;

    for (s = 0; s < nsub; ++s) {
        rc = pulseqlib_build_freq_mod_library(
            &freqlibs[s], coll, s, fovshift);
        CHECK(rc, &diag);
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
                rc = pulseqlib_update_freq_mod_library(
                    freqlibs[ci.subseq_idx], fovshift);
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

            if (freqlibs[ci.subseq_idx])
                pulseqlib_freq_mod_library_get(
                    freqlibs[ci.subseq_idx], ci.scan_pos,
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
    for (s = 0; s < nlibs; ++s)
        if (freqlibs[s]) pulseqlib_freq_mod_library_free(freqlibs[s]);
    free(freqlibs);
    pulseqlib_collection_free(coll);
    return 0;

fail:
    if (freqlibs) {
        for (s = 0; s < nlibs; ++s)
            if (freqlibs[s]) pulseqlib_freq_mod_library_free(freqlibs[s]);
        free(freqlibs);
    }
    if (coll) pulseqlib_collection_free(coll);
    return 1;
}
