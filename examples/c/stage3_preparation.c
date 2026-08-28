/*
 * stage3_preparation.c -- Preparation: read, structure, gate, cache.
 *
 * The operator has committed. This is the last point at which the scan can be
 * refused, and the only stage that pays for the whole sequence: the file is
 * parsed, structured, checked against the hardware, and written back as a
 * cache the later stages read instead of repeating any of it.
 *
 *   stage3_preparation <scan.seq>
 */

#include "vendor.h"

int main(int argc, char **argv)
{
    pulseg_opts opts;
    pulseg_diagnostic diagnostic;
    pulseg_collection *scan = NULL;
    pulseg_check_plan *plan = NULL;
    pulseg_forbidden_band_list bands;
    pulseg_collection_info info = PULSEG_COLLECTION_INFO_INIT;
    pulseg_scan_time_info timing = PULSEG_SCAN_TIME_INFO_INIT;
    int code;

    if (argc != 2)
    {
        (void)fprintf(stderr, "usage: %s <scan.seq>\n", argv[0]);
        return 2;
    }

    vendor_system_limits(&opts);
    vendor_forbidden_bands(&bands);
    pulseg_diagnostic_init(&diagnostic);

    /* 1. Read and structure
     * pulseg_read is pulseq_read composed with pulseg_convert_collection: it
     * follows the NextSequence chain, deduplicates the unique blocks, detects
     * the TR and the segments from block content, and expands the execution
     * stream. Nothing is annotated, and a TRID label is never trusted, so
     * what comes back is what the file plays. */
    code = pulseg_read(&scan, &diagnostic, argv[1], &opts, 0, 0, 1);
    if (PULSEG_FAILED(code))
    {
        vendor_refuse(code, &diagnostic);
        return 1;
    }

    (void)pulseg_get_collection_info(scan, &info);
    (void)pulseg_get_scan_time(scan, &timing);
    vendor_log(
        "read: %d subsequence(s), %d segments, %.1f s",
        info.num_subsequences,
        info.num_segments,
        timing.total_duration_us / 1e6);

    /* 2. Gate it
     * One plan, shared by the checks that would otherwise each extract the
     * same gradient waveforms. Passing NULL would work identically and cost
     * a little more; creating it explicitly is what a stage asking several
     * questions of one sequence does.
     *
     * No PNS model is passed here (NULL skips that check): the model is
     * caller-injected precisely because its coefficients are vendor
     * property, and a real integration would supply its own, or one of the
     * published forms in pulseg_pns_models.h. */
    code = pulseg_check_plan_create(&plan, &diagnostic, scan, NULL);
    if (PULSEG_FAILED(code))
    {
        vendor_refuse(code, &diagnostic);
        pulseg_collection_free(scan);
        return 1;
    }

    code = pulseg_check_safety(scan, &diagnostic, plan, &opts, &bands, NULL, 80.0f);
    if (PULSEG_FAILED(code))
    {
        /* The diagnostic names what failed and where: the axis, the block,
         * the frequency, the amount by which it was over. */
        vendor_refuse(code, &diagnostic);
        pulseg_check_plan_destroy(plan);
        pulseg_collection_free(scan);
        return 1;
    }
    vendor_log("checks: passed");

    /* Each check is also callable on its own, for a platform that gates some
     * of them in hardware and wants only the rest. Re-asking one here costs
     * almost nothing: the plan already holds what it needs. */
    code = pulseg_check_max_slew(scan, &diagnostic, &opts);
    vendor_log("slew alone: %s", PULSEG_SUCCEEDED(code) ? "within limits" : "over");

    pulseg_check_plan_destroy(plan);

    /* 3. Cache it
     * The structure was expensive to derive and the later stages need it, so
     * it is written beside the .seq rather than derived twice. */
    code = pulseg_save_cache(scan, argv[1], &opts);
    if (PULSEG_FAILED(code))
    {
        vendor_refuse(code, &diagnostic);
        pulseg_collection_free(scan);
        return 1;
    }
    vendor_log("cache: written beside the sequence");

    pulseg_collection_free(scan);
    return 0;
}
