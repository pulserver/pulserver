/*
 * safety_gate.c -- the checks on their own.
 *
 * You already have an interpreter, and it almost certainly already checks
 * amplitude and slew. This is for the rest: the acoustic and nerve-stimulation
 * checks, which are more work to write, and the speed that comes from judging
 * one canonical TR instead of the whole scan. The basic checks are here too,
 * so a caller that wants the whole gate from one place can have it.
 *
 * Nothing else is adopted: no cache, no structure on disk, no change to how
 * the sequence is played.
 *
 *   safety_gate <scan.seq>
 */

#include "vendor.h"

int main(int argc, char **argv)
{
    pulseg_opts opts;
    pulseg_diagnostic diagnostic;
    pulseg_collection *scan = NULL;
    pulseg_forbidden_band_list bands;
    int code;

    if (argc != 2)
    {
        (void)fprintf(stderr, "usage: %s <scan.seq>\n", argv[0]);
        return 2;
    }

    vendor_system_limits(&opts);
    vendor_forbidden_bands(&bands);
    pulseg_diagnostic_init(&diagnostic);

    code = pulseg_read(&scan, &diagnostic, argv[1], &opts, 0, 0, 0);
    if (PULSEG_SUCCEEDED(code))
    {
        /* NULL plan: the checks keep their preprocessing private to the call,
         * which is what a caller asking one question wants. */
        code = pulseg_check_safety(scan, &diagnostic, NULL, &opts, &bands, NULL, 80.0f);
        pulseg_collection_free(scan);
    }

    if (PULSEG_FAILED(code))
    {
        vendor_refuse(code, &diagnostic);
        return 1;
    }

    vendor_log("playable");
    return 0;
}
