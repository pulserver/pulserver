/**
 * @file example_uieval.c
 * @brief Quick scan-time preview for the scanner UI.
 *
 * Called whenever the operator changes a prescription parameter.
 * Uses pulseqlib_peek_scan_time() which only reads the [DEFINITIONS]
 * header — no full parse, no caching, no signature verification.
 *
 * Compile:
 *   cc -I../../csrc example_uieval.c ../../csrc/pulseqlib_*.c -lm -o uieval
 */

#include "example_vendorlib.h"

#include <stdio.h>

/* Globals — initialised by startup() (see example_startup.c) */
pulseqlib_opts       g_opts;
pulseqlib_diagnostic g_diag;

/* Updated by uieval — read by the scanner UI framework */
float scan_duration_us;

/* ================================================================== */
/*  UI evaluation                                                     */
/* ================================================================== */

static int uieval(const char* seq_path, int num_reps)
{
    pulseqlib_scan_time_info info = PULSEQLIB_SCAN_TIME_INFO_INIT;
    int rc;

    rc = pulseqlib_peek_scan_time(&info, seq_path, &g_opts, num_reps);
    if (PULSEQLIB_FAILED(rc)) {
        vendor_report_error(rc, &g_diag);
        return rc;
    }

    scan_duration_us = info.total_duration_us;

    /*
     * In a real driver, the UI framework picks up scan_duration_us
     * and displays it on the operator console.
     */
    return 0;
}

/* ================================================================== */
/*  Main (standalone demo)                                            */
/* ================================================================== */

int main(int argc, char** argv)
{
    const char* path = (argc > 1) ? argv[1] : "sequence.seq";
    int num_reps = 1;

    vendor_opts_init(&g_opts, 42577478.0f, 3.0f, 50.0f, 200.0f);
    pulseqlib_diagnostic_init(&g_diag);

    if (uieval(path, num_reps) == 0)
        printf("Scan duration: %.2f s\n", scan_duration_us / 1e6f);

    return 0;
}