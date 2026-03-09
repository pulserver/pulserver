/**
 * @file example_check.c
 * @brief Load a sequence, run safety checks, set up echo filters.
 *
 * Assumes globals from example_startup.c are initialised.
 *
 * Workflow:
 *   1. Load with signature check + caching (no label parsing).
 *   2. Consistency check.
 *   3. Hardware safety check (gmax, slewmax, continuity, acoustic, PNS).
 *   4. Build per-TR RF stat arrays; run vendor RF safety + find max B1.
 *   5. Gradient safety per segment.
 *   6. Set up echo filters and data storage dimensions.
 *
 * Compile:
 *   cc -I../../csrc example_check.c ../../csrc/pulseqlib_*.c -lm -o check
 */

#include "example_vendorlib.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CHECK(rc, diag)                                 \
    do {                                                \
        if (PULSEQLIB_FAILED(rc)) {                     \
            vendor_report_error(rc, (diag));             \
            goto fail;                                  \
        }                                               \
    } while (0)

#define MAX_SEGMENTS  256

/* ================================================================== */
/*  Globals — initialised by startup() in production                  */
/* ================================================================== */

#define MAX_FORBIDDEN_BANDS 16

pulseqlib_opts           g_opts;
pulseqlib_diagnostic     g_diag;
pulseqlib_pns_params     g_pns;
pulseqlib_forbidden_band g_bands[MAX_FORBIDDEN_BANDS];
int                      g_num_bands;

/* ================================================================== */
/*  Vendor RF / gradient safety stubs                                 */
/* ================================================================== */

/**
 * @brief Vendor RF safety check.
 *
 * In a real driver this calls the vendor SAR model with the ordered
 * array of RF pulses (each carrying act_amplitude_hz, duration,
 * duty_cycle, num_instances, …).  Returns minimum TR in us.
 */
static float vendor_check_rf_safety(const pulseqlib_rf_stats* pulses,
                                 int num_pulses)
{
    /* Placeholder: return 0 = no constraint */
    (void)pulses; (void)num_pulses;
    return 0.0f;
}

/**
 * @brief Vendor max-B1 finder — returns peak |gamma*B1| (Hz).
 */
static float vendor_find_rf_max(const pulseqlib_rf_stats* pulses,
                             int num_pulses)
{
    float mx = 0.0f;
    int i;
    for (i = 0; i < num_pulses; ++i) {
        if (pulses[i].base_amplitude_hz > mx)
            mx = pulses[i].base_amplitude_hz;
    }
    return mx;
}

/**
 * @brief Vendor gradient safety per segment — returns min duration (us).
 */
static float vendor_check_grad_safety(const pulseqlib_collection* coll,
                                   int seg_idx)
{
    /* Placeholder: return 0 = no constraint */
    (void)coll; (void)seg_idx;
    return 0.0f;
}

/* ================================================================== */
/*  Main                                                              */
/* ================================================================== */

int main(int argc, char** argv)
{
    const char*           seq_path;
    pulseqlib_collection* coll = NULL;
    int rc, s, nsub, nseg;
    int max_b1_subseq = 0;

    if (argc < 2) {
        fprintf(stderr, "Usage: %s <sequence.seq>\n", argv[0]);
        return 1;
    }
    seq_path = argv[1];

    /* -- Startup (in production this is in a separate function) --- */
    vendor_opts_init(&g_opts, 42577478.0f, 3.0f, 50.0f, 200.0f);
    vendor_pns_params_init(&g_pns, &g_opts, 360.0f, 20.0f, 0.333f);
    pulseqlib_diagnostic_init(&g_diag);
    g_num_bands = 0;

    /* ============================================================= */
    /*  1. Load — signature + cache, no label parsing                */
    /* ============================================================= */

    rc = pulseqlib_read(&coll, &g_diag, seq_path, &g_opts,
                        1,   /* cache_binary     */
                        1,   /* verify_signature */
                        0,   /* parse_labels     */
                        1);  /* num_averages     */
    CHECK(rc, &g_diag);

    /* ============================================================= */
    /*  2. Consistency check                                         */
    /* ============================================================= */

    rc = pulseqlib_check_consistency(coll, &g_diag);
    CHECK(rc, &g_diag);

    /* ============================================================= */
    /*  3. Hardware safety (gmax, slewmax, continuity, acoustic, PNS)*/
    /* ============================================================= */

    rc = pulseqlib_check_safety(coll, &g_diag, &g_opts,
                                g_num_bands, g_bands,
                                &g_pns, 100.0f);
    CHECK(rc, &g_diag);
    printf("Hardware safety check PASSED.\n");

    /* ============================================================= */
    /*  4. RF safety — per-region RF arrays via pulseqlib_get_rf_array*/
    /* ============================================================= */
    {
        float max_b1_hz = 0.0f;
        pulseqlib_collection_info ci = PULSEQLIB_COLLECTION_INFO_INIT;

        rc = pulseqlib_get_collection_info(coll, &ci);
        CHECK(rc, &g_diag);
        nsub = ci.num_subsequences;

        for (s = 0; s < nsub; ++s) {
            pulseqlib_subseq_info si = PULSEQLIB_SUBSEQ_INFO_INIT;
            pulseqlib_rf_stats* pulses = NULL;
            int   npulses;
            float min_tr_us, b1;

            rc = pulseqlib_get_subseq_info(coll, s, &si);
            CHECK(rc, &g_diag);

            /* -- Prep (if non-degenerate) ------------------------- */
            if (!si.degenerate_prep) {
                npulses = pulseqlib_get_rf_array(
                    coll, &pulses, s, PULSEQLIB_TR_REGION_PREP);
                if (npulses > 0) {
                    min_tr_us = vendor_check_rf_safety(pulses, npulses);
                    if (min_tr_us > si.tr_duration_us) {
                        free(pulses);
                        fprintf(stderr,
                            "RF safety: subseq %d prep TR too short\n", s);
                        goto fail;
                    }
                    b1 = vendor_find_rf_max(pulses, npulses);
                    if (b1 > max_b1_hz) {
                        max_b1_hz = b1;
                        max_b1_subseq = s;
                    }
                }
                free(pulses);
                pulses = NULL;
            }

            /* -- Cooldown (if non-degenerate) --------------------- */
            if (!si.degenerate_cooldown) {
                npulses = pulseqlib_get_rf_array(
                    coll, &pulses, s, PULSEQLIB_TR_REGION_COOLDOWN);
                if (npulses > 0) {
                    min_tr_us = vendor_check_rf_safety(pulses, npulses);
                    if (min_tr_us > si.tr_duration_us) {
                        free(pulses);
                        fprintf(stderr,
                            "RF safety: subseq %d cooldown TR too short\n", s);
                        goto fail;
                    }
                    b1 = vendor_find_rf_max(pulses, npulses);
                    if (b1 > max_b1_hz) {
                        max_b1_hz = b1;
                        max_b1_subseq = s;
                    }
                }
                free(pulses);
                pulses = NULL;
            }

            /* -- Main TR ------------------------------------------ */
            npulses = pulseqlib_get_rf_array(
                coll, &pulses, s, PULSEQLIB_TR_REGION_MAIN);
            if (npulses > 0) {
                min_tr_us = vendor_check_rf_safety(pulses, npulses);
                if (min_tr_us > si.tr_duration_us) {
                    free(pulses);
                    fprintf(stderr,
                        "RF safety: subseq %d main TR too short\n", s);
                    goto fail;
                }
                b1 = vendor_find_rf_max(pulses, npulses);
                if (b1 > max_b1_hz) {
                    max_b1_hz = b1;
                    max_b1_subseq = s;
                }
            }
            free(pulses);
        }

        printf("Max B1 = %.1f Hz (subseq %d)\n",
               max_b1_hz, max_b1_subseq);
    }

    /* ============================================================= */
    /*  5. Gradient safety per segment                               */
    /* ============================================================= */
    {
        pulseqlib_collection_info ci = PULSEQLIB_COLLECTION_INFO_INIT;
        rc = pulseqlib_get_collection_info(coll, &ci);
        CHECK(rc, &g_diag);
        nseg = ci.num_segments;
    }

    for (s = 0; s < nseg; ++s) {
        pulseqlib_segment_info segi = PULSEQLIB_SEGMENT_INFO_INIT;
        float min_dur;

        rc = pulseqlib_get_segment_info(coll, s, &segi);
        CHECK(rc, &g_diag);

        min_dur = vendor_check_grad_safety(coll, s);

        if (min_dur > (float)segi.duration_us) {
            fprintf(stderr,
                "Gradient safety: segment %d too short "
                "(%.0f us < %.0f us min)\n",
                s, (float)segi.duration_us, min_dur);
            goto fail;
        }
    }
    printf("Gradient safety check PASSED (%d segments).\n", nseg);

    /* ============================================================= */
    /*  6. Echo filters and data storage dimensions                  */
    /* ============================================================= */
    {
        pulseqlib_collection_info ci = PULSEQLIB_COLLECTION_INFO_INIT;
        pulseqlib_subseq_info     si = PULSEQLIB_SUBSEQ_INFO_INIT;
        int max_samples, total_readouts, num_unique_adcs;
        int calibration_samples = 0;
        int a;

        rc = pulseqlib_get_collection_info(coll, &ci);
        CHECK(rc, &g_diag);
        max_samples    = ci.max_adc_samples;
        total_readouts = ci.total_readouts;

        rc = pulseqlib_get_subseq_info(coll, 0, &si);
        CHECK(rc, &g_diag);
        num_unique_adcs = si.num_unique_adcs;

        /*
         * calibration_samples = first ADC sample count from the
         * subsequence that contains max B1 (used for prescan).
         * Walking unique ADCs in that subsequence:
         */
        {
            pulseqlib_subseq_info si_cal = PULSEQLIB_SUBSEQ_INFO_INIT;
            rc = pulseqlib_get_subseq_info(coll, max_b1_subseq, &si_cal);
            CHECK(rc, &g_diag);

            for (a = 0; a < si_cal.num_unique_adcs; ++a) {
                pulseqlib_adc_def ad = PULSEQLIB_ADC_DEF_INIT;
                rc = pulseqlib_get_adc_def(coll, a, &ad);
                CHECK(rc, &g_diag);
                if (ad.num_samples > 0) {
                    calibration_samples = ad.num_samples;
                    break;
                }
            }
        }

        printf("\nEcho filter / data storage setup:\n");
        printf("  max_samples         = %d\n", max_samples);
        printf("  calibration_samples = %d\n", calibration_samples);
        printf("  total_readouts      = %d\n", total_readouts);
        printf("  unique ADC events   = %d\n", num_unique_adcs);

        /*
         * In a real vendor driver you would now:
         *
         * 1. Set up echo filters per unique ADC event:
         *      pulseqlib_adc_def ad;
         *      pulseqlib_get_adc_def(coll, a, &ad);
         *      bw        = <vendor formula from ad.dwell_ns>;
         *      calcfilter(...);
         *
         * 2. Set data storage dimensions from total_readouts,
         *    max_samples, and vendor-specific limits.
         */

        for (a = 0; a < num_unique_adcs; ++a) {
            pulseqlib_adc_def ad = PULSEQLIB_ADC_DEF_INIT;
            rc = pulseqlib_get_adc_def(coll, a, &ad);
            CHECK(rc, &g_diag);
            printf("  ADC %d: dwell=%d ns, nsamples=%d\n",
                   a, ad.dwell_ns, ad.num_samples);
        }
    }

    /* ============================================================= */
    /*  Cleanup                                                      */
    /* ============================================================= */
    pulseqlib_collection_free(coll);
    return 0;

fail:
    if (coll) pulseqlib_collection_free(coll);
    return 1;
}
