/**
 * @file example_errorhandling.c
 * @brief Demonstrates pulseqlib error handling patterns.
 *
 * Every pulseqlib API function returns an int status code:
 *
 *   - Positive (PULSEQLIB_OK = 1)  → success.
 *   - Negative (PULSEQLIB_ERR_*)   → specific error.
 *
 * Use PULSEQLIB_SUCCEEDED(rc) / PULSEQLIB_FAILED(rc) macros, or
 * compare to individual error codes when you need to branch.
 *
 * On failure the library fills a pulseqlib_diagnostic struct with:
 *   .code     – the same error code
 *   .message  – a human-readable explanation (up to 256 chars)
 *
 * The helper pulseqlib_format_error() combines the error-code
 * description, any hint text, and the diagnostic message into a
 * single formatted string suitable for user-facing error display.
 *
 * Compile:
 *   cc -I../../csrc example_errorhandling.c ../../csrc/pulseqlib_*.c -lm -o errorhandling
 */

#include "example_vendorlib.h"   /* must come first (sets ALLOC/FREE) */
#include "pulseqlib_methods.h"

#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Pattern 1:  Simple "bail-on-first-error" with goto                */
/* ------------------------------------------------------------------ */

static int pattern_bail_on_error(const char* seq_path)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_opts        opts = PULSEQLIB_OPTS_INIT;
    int rc;

    vendor_opts_init(&opts);

    /* --- Load ---------------------------------------------------- */
    rc = pulseqlib_read(&coll, &diag, seq_path, &opts,
                        /*cache_binary=*/1,
                        /*verify_signature=*/1,
                        /*parse_labels=*/1,
                        /*num_averages=*/1);
    if (PULSEQLIB_FAILED(rc)) goto fail;

    /* --- Use the collection -------------------------------------- */
    printf("Loaded %d subsequences, total duration %.1f ms\n",
           pulseqlib_get_num_subsequences(coll),
           pulseqlib_get_total_duration_us(coll) / 1000.0f);

    /* --- Cleanup (success path) ---------------------------------- */
    pulseqlib_collection_free(coll);
    return 0;

fail:
    /* Report through vendor channel and clean up */
    vendor_report_error(rc, &diag);
    if (coll) pulseqlib_collection_free(coll);
    return rc;
}

/* ------------------------------------------------------------------ */
/*  Pattern 2:  CHECK macro (reduces boilerplate)                     */
/* ------------------------------------------------------------------ */

/*
 * A project-wide macro that calls vendor_report_error and jumps
 * to a local `fail:` label.  Real vendor code often has a similar
 * macro (e.g. EM_CHECK, EPIC_CHECK, …).
 */
#define CHECK(rc, diag)                                 \
    do {                                                \
        if (PULSEQLIB_FAILED(rc)) {                     \
            vendor_report_error(rc, (diag));             \
            goto fail;                                  \
        }                                               \
    } while (0)

static int pattern_check_macro(const char* seq_path)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_opts        opts = PULSEQLIB_OPTS_INIT;
    int rc;

    vendor_opts_init(&opts);

    rc = pulseqlib_read(&coll, &diag, seq_path, &opts, 1, 1, 1, 1);
    CHECK(rc, &diag);

    /* Consistency re-check (already done by read, shown for example) */
    rc = pulseqlib_check_consistency(coll, &diag);
    CHECK(rc, &diag);

    printf("Consistency OK\n");
    pulseqlib_collection_free(coll);
    return 0;

fail:
    if (coll) pulseqlib_collection_free(coll);
    return rc;
}

/* ------------------------------------------------------------------ */
/*  Pattern 3:  Switch on specific error codes                        */
/* ------------------------------------------------------------------ */

static int pattern_specific_errors(const char* seq_path)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_opts        opts = PULSEQLIB_OPTS_INIT;
    int rc;

    vendor_opts_init(&opts);

    rc = pulseqlib_read(&coll, &diag, seq_path, &opts, 1, 1, 1, 1);

    if (PULSEQLIB_SUCCEEDED(rc)) {
        printf("Load OK\n");
        pulseqlib_collection_free(coll);
        return 0;
    }

    /* Branch on the error code to give a more specific response */
    switch (rc) {
    case PULSEQLIB_ERR_FILE_NOT_FOUND:
        fprintf(stderr, "File not found: %s\n", seq_path);
        break;

    case PULSEQLIB_ERR_SIGNATURE_MISMATCH:
        fprintf(stderr,
            "Sequence file signature mismatch — the .seq file may "
            "have been modified.\n  Detail: %s\n", diag.message);
        break;

    case PULSEQLIB_ERR_TR_NO_PERIODIC_PATTERN:
    case PULSEQLIB_ERR_TR_PATTERN_MISMATCH:
        fprintf(stderr,
            "Could not identify a repeating TR structure.\n"
            "  Detail: %s\n"
            "  Hint:   %s\n",
            diag.message,
            pulseqlib_get_error_hint(rc));
        break;

    default:
        vendor_report_error(rc, &diag);
        break;
    }

    if (coll) pulseqlib_collection_free(coll);
    return rc;
}

/* ------------------------------------------------------------------ */
/*  Pattern 4:  Using format_error for a full message                 */
/* ------------------------------------------------------------------ */

static int pattern_format_error(const char* seq_path)
{
    pulseqlib_collection* coll = NULL;
    pulseqlib_diagnostic  diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_opts        opts = PULSEQLIB_OPTS_INIT;
    char                  errbuf[512];
    int rc;

    vendor_opts_init(&opts);

    rc = pulseqlib_read(&coll, &diag, seq_path, &opts, 1, 1, 1, 1);

    if (PULSEQLIB_FAILED(rc)) {
        /* Build a single string with code name + diagnostic detail */
        pulseqlib_format_error(errbuf, sizeof(errbuf), rc, &diag);

        /*
         * In a real vendor integration you would pass errbuf to the
         * vendor error reporting API, e.g.:
         *
         *   vendor_report_error(USE_ERMES, errbuf,
         *                       VENDOR_ERR_PSD_PULSEQ_FAILURE, 0);
         */
        fprintf(stderr, "%s\n", errbuf);

        if (coll) pulseqlib_collection_free(coll);
        return rc;
    }

    printf("All good.\n");
    pulseqlib_collection_free(coll);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Main                                                              */
/* ------------------------------------------------------------------ */

int main(int argc, char** argv)
{
    const char* path = (argc > 1) ? argv[1] : "sequence.seq";

    printf("=== Pattern 1: bail on error ===\n");
    pattern_bail_on_error(path);

    printf("\n=== Pattern 2: CHECK macro ===\n");
    pattern_check_macro(path);

    printf("\n=== Pattern 3: specific errors ===\n");
    pattern_specific_errors(path);

    printf("\n=== Pattern 4: format_error ===\n");
    pattern_format_error(path);

    return 0;
}
