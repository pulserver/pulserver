/**
 * @file example_check.c
 * @brief Load a sequence, cache it, and run all safety checks.
 *
 * Workflow:
 *   1. Initialise vendor opts / PNS params / forbidden bands.
 *   2. Load the .seq file (with binary cache enabled so subsequent
 *      loads skip the text parser).
 *   3. Peek scan time (quick estimate from [DEFINITIONS] only).
 *   4. Run full safety check (gradient limits + acoustic + PNS).
 *   5. Print a summary or report errors.
 *
 * Compile:
 *   cc -I../../csrc example_check.c ../../csrc/pulseqlib_*.c -lm -o check
 *
 * Run:
 *   ./check path/to/sequence.seq
 */

#include "example_vendorlib.h"   /* must come first */
#include "pulseqlib_methods.h"

#include <stdio.h>
#include <string.h>

#define CHECK(rc, diag)                                 \
    do {                                                \
        if (PULSEQLIB_FAILED(rc)) {                     \
            vendor_report_error(rc, (diag));             \
            goto fail;                                  \
        }                                               \
    } while (0)

int main(int argc, char** argv)
{
    const char*            seq_path;
    pulseqlib_opts         opts       = PULSEQLIB_OPTS_INIT;
    pulseqlib_diagnostic   diag       = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_collection*  coll       = NULL;
    pulseqlib_scan_time_info peek_info = PULSEQLIB_SCAN_TIME_INFO_INIT;
    pulseqlib_scan_time_info scan_info = PULSEQLIB_SCAN_TIME_INFO_INIT;
    pulseqlib_pns_params   pns        = PULSEQLIB_PNS_PARAMS_INIT;
    int                    rc;
    int                    num_averages = 1;

    /* -- Command line --------------------------------------------- */
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <sequence.seq> [num_averages]\n", argv[0]);
        return 1;
    }
    seq_path = argv[1];
    if (argc > 2) num_averages = atoi(argv[2]);

    /* -- Step 1: initialise vendor parameters --------------------- */
    vendor_opts_init(&opts);
    vendor_pns_params_init(&pns);

    /*
     * Acoustic forbidden bands.
     *
     * In a real integration these come from the system configuration
     * database.  Here we hardcode one example band.
     */
    pulseqlib_forbidden_band bands[1];
    int num_bands = 0;  /* set to 1 to enable the check below */

    /* Example: forbid 500–600 Hz above 10 mT/m spectral amplitude */
    bands[0].freq_min_hz            = 500.0f;
    bands[0].freq_max_hz            = 600.0f;
    bands[0].max_amplitude_hz_per_m = 10.0e-3f * VENDOR_GAMMA_HZ_PER_T;
    /* num_bands = 1;  -- uncomment to enable */

    /* -- Step 2: quick scan-time peek (before full load) ---------- */
    rc = pulseqlib_peek_scan_time(&peek_info, seq_path, &opts,
                                  num_averages);
    if (PULSEQLIB_SUCCEEDED(rc)) {
        printf("Quick scan-time estimate: %.3f s\n",
               peek_info.total_duration_us / 1e6f);
    } else {
        fprintf(stderr, "Warning: could not peek scan time (rc=%d)\n", rc);
        /* Non-fatal: continue with full load */
    }

    /* -- Step 3: full load (with binary cache) -------------------- */
    rc = pulseqlib_read(
        &coll, &diag, seq_path, &opts,
        /*cache_binary=*/1,          /* write/read .bin cache */
        /*verify_signature=*/1,      /* check MD5 signature   */
        /*parse_labels=*/1,          /* build ADC label table  */
        num_averages);
    CHECK(rc, &diag);

    /* -- Step 4: print structural summary ------------------------- */
    {
        int nsub = pulseqlib_get_num_subsequences(coll);
        int s;
        printf("\nLoaded collection: %d subsequence(s)\n", nsub);

        for (s = 0; s < nsub; ++s) {
            int tr_size     = pulseqlib_get_tr_size(coll, s);
            int num_trs     = pulseqlib_get_num_trs(coll, s);
            int num_prep    = pulseqlib_get_num_prep_blocks(coll, s);
            int num_cool    = pulseqlib_get_num_cooldown_blocks(coll, s);
            float tr_dur_us = pulseqlib_get_tr_duration_us(coll, s);

            printf("  Subseq %d: TR size=%d, #TRs=%d, "
                   "prep=%d, cooldown=%d, TR duration=%.1f us\n",
                   s, tr_size, num_trs, num_prep, num_cool, tr_dur_us);
        }
    }

    /* -- Step 5: accurate scan time ------------------------------- */
    rc = pulseqlib_get_scan_time(coll, num_averages, &scan_info);
    CHECK(rc, &diag);

    printf("\nAccurate scan time: %.3f s  (%d segment boundaries)\n",
           scan_info.total_duration_us / 1e6f,
           scan_info.total_segment_boundaries);

    /* -- Step 6: full safety check -------------------------------- */
    rc = pulseqlib_check_safety(
        coll, &diag, &opts,
        num_bands, bands,
        &pns,
        VENDOR_PNS_THRESHOLD_PCT);

    if (PULSEQLIB_SUCCEEDED(rc)) {
        printf("\nSafety check PASSED.\n");
    } else {
        /* Safety violation — report the specific failure */
        char buf[512];
        pulseqlib_format_error(buf, sizeof(buf), rc, &diag);

        printf("\nSafety check FAILED:\n  %s\n", buf);

        /*
         * The error code tells you which check failed:
         *
         *   PULSEQLIB_ERR_MAX_GRAD_EXCEEDED (-550)
         *   PULSEQLIB_ERR_MAX_SLEW_EXCEEDED (-552)
         *   PULSEQLIB_ERR_GRAD_DISCONTINUITY (-551)
         *   PULSEQLIB_ERR_ACOUSTIC_VIOLATION (-404)
         *   PULSEQLIB_ERR_PNS_THRESHOLD_EXCEEDED (-455)
         *
         * The diagnostic message includes axis and block index.
         */
        switch (rc) {
        case PULSEQLIB_ERR_MAX_GRAD_EXCEEDED:
            printf("  -> Reduce gradient amplitude.\n");
            break;
        case PULSEQLIB_ERR_MAX_SLEW_EXCEEDED:
            printf("  -> Reduce slew rate.\n");
            break;
        case PULSEQLIB_ERR_ACOUSTIC_VIOLATION:
            printf("  -> Adjust echo spacing or gradient waveform shape.\n");
            break;
        case PULSEQLIB_ERR_PNS_THRESHOLD_EXCEEDED:
            printf("  -> Reduce slew rate or TR.\n");
            break;
        default:
            break;
        }

        /*
         * In a real vendor integration, this would call the vendor
         * error API to stop the scan prescription:
         *
         *   vendor_report_error(USE_ERMES, buf,
         *                       VENDOR_ERR_PSD_SAFETY_VIOLATION, 0);
         */
    }

    /* -- Cleanup -------------------------------------------------- */
    pulseqlib_collection_free(coll);
    return (PULSEQLIB_SUCCEEDED(rc)) ? 0 : 1;

fail:
    if (coll) pulseqlib_collection_free(coll);
    return 1;
}
