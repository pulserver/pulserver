/**
 * @file safety_with_external_sequence.c
 * @brief Vendor-neutral one-shot safety check from a `.seq` file path.
 *
 * Demonstrates the iterator-based safety entrypoint
 * `pulseqlib_check_safety_from_file()`, which is the single call a
 * third-party vendor needs to validate a Pulseq sequence:
 *
 *   1. The library reads the (possibly chained) `.seq` file from disk
 *      using the caller's hardware limits (`pulseqlib_opts`).
 *   2. It runs the standard open-source safety pipeline on the loaded
 *      collection: gradient continuity, max gradient amplitude, max
 *      slew rate, structural mechanical-resonance forbidden-band check,
 *      and PNS thresholding using the vendor model selected by
 *      `pns_params.vendor`.
 *   3. The collection is freed before returning.
 *
 * The vendor never holds a `pulseqlib_collection*`. This is the same
 * surface a Pulseq C++ `ExternalSequence`-style integrator would call:
 * point it at the .seq file and ask "is it safe?".
 *
 * Build (linking against the static pulseqlib library):
 *   cc -I../../csrc safety_with_external_sequence.c \
 *      -L<build-dir> -lpulseqlib -lm \
 *      -o safety_with_external_sequence <path-to-some.seq>
 */

#include "pulseqlib_methods.h"
#include "pulseqlib_types.h"

#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv)
{
    pulseqlib_opts opts = PULSEQLIB_OPTS_INIT;
    pulseqlib_diagnostic diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_forbidden_band band = PULSEQLIB_FORBIDDEN_BAND_INIT;
    pulseqlib_pns_params pns = PULSEQLIB_PNS_PARAMS_INIT;
    const char* seq_path;
    int rc;

    if (argc < 2) {
        fprintf(stderr, "usage: %s <sequence.seq>\n", argv[0]);
        return 1;
    }
    seq_path = argv[1];

    /* Scanner limits (typical 3 T whole-body): 80 mT/m, 200 T/m/s. */
    opts.vendor                  = PULSEQLIB_VENDOR_GEHC;
    opts.gamma_hz_per_t          = 42.577478e6f;
    opts.b0_t                    = 3.0f;
    opts.max_grad_hz_per_m       = 0.080f * opts.gamma_hz_per_t;
    opts.max_slew_hz_per_m_per_s = 200.0f * opts.gamma_hz_per_t;
    opts.rf_raster_us            = 1.0f;
    opts.grad_raster_us          = 10.0f;
    opts.adc_raster_us           = 0.1f;
    opts.block_raster_us         = 10.0f;

    /* One example forbidden mechanical-resonance band. */
    band.freq_min_hz             = 580.0f;
    band.freq_max_hz             = 720.0f;
    band.max_amplitude_hz_per_m  = 1.0e4f;

    /* PNS model (GE exponential). */
    pns.vendor                   = PULSEQLIB_VENDOR_GEHC;
    pns.chronaxie_us             = 360.0f;
    pns.rheobase_hz_per_m_per_s  = 20.0f * opts.gamma_hz_per_t;
    pns.alpha                    = 0.324f;

    rc = pulseqlib_check_safety_from_file(
        &diag, seq_path, &opts,
        /*num_forbidden_bands*/ 1, &band,
        &pns, /*pns_threshold_percent*/ 80.0f);

    if (PULSEQLIB_FAILED(rc)) {
        printf("safety check FAILED  rc=%d  msg=\"%s\"\n",
               rc, diag.message);
        return 2;
    }

    printf("safety check PASSED for %s\n", seq_path);
    return 0;
}
