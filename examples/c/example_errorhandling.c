/**
 * @file example_errorhandling.c
 * @brief Minimal error-handling snippet for pulseqlib.
 *
 * Every pulseqlib function returns an int status code:
 *   positive  = success  (check with PULSEQLIB_SUCCEEDED)
 *   negative  = failure  (check with PULSEQLIB_FAILED)
 *
 * On failure, diag.message contains a human-readable explanation.
 * vendor_report_error() in example_vendorlib.h formats and prints it
 * through the vendor error channel.  The specific negative value is
 * an opaque library detail — consumers must not match on it.
 *
 * The CHECK macro below is the only boilerplate needed — it mirrors
 * the pattern used by vendor toolchains (e.g. EPIC_CHECK).
 *
 * Compile:
 *   cc -I../../csrc example_errorhandling.c ../../csrc/pulseqlib_*.c -lm -o errorhandling
 */

#include "example_vendorlib.h"

#include <stdio.h>

/* ------------------------------------------------------------------ */
/*  CHECK macro — use this everywhere                                 */
/* ------------------------------------------------------------------ */

#define CHECK(rc, diag)                                 \
    do {                                                \
        if (PULSEQLIB_FAILED(rc)) {                     \
            vendor_report_error(rc, (diag));             \
            goto fail;                                  \
        }                                               \
    } while (0)

/* ------------------------------------------------------------------ */
/*  Usage example                                                     */
/* ------------------------------------------------------------------ */

int main(int argc, char** argv)
{
    const char*           path = (argc > 1) ? argv[1] : "sequence.seq";
    pulseqlib_opts        opts = PULSEQLIB_OPTS_INIT;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_collection* coll = NULL;
    int rc;

    vendor_opts_init(&opts, 42577478.0f, 3.0f, 50.0f, 200.0f);

    rc = pulseqlib_read(&coll, &diag, path, &opts, 1, 1, 1, 1);
    CHECK(rc, &diag);

    rc = pulseqlib_check_consistency(coll, &diag);
    CHECK(rc, &diag);

    {
        pulseqlib_collection_info ci = PULSEQLIB_COLLECTION_INFO_INIT;
        rc = pulseqlib_get_collection_info(coll, &ci);
        CHECK(rc, &diag);
        printf("OK: %d subsequences, %.1f ms\n",
               ci.num_subsequences,
               ci.total_duration_us / 1000.0f);
    }

    pulseqlib_collection_free(coll);
    return 0;

fail:
    if (coll) pulseqlib_collection_free(coll);
    return 1;
}
