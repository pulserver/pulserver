/**
 * @file safety_with_external_sequence.cpp
 * @brief C++ vendor-neutral one-shot safety check from a `.seq` file path.
 *
 * Mirror of `cexamples/safety_with_external_sequence.c`, demonstrating
 * that the public C API of pulseqlib is directly consumable from C++
 * (the headers are wrapped in `extern "C"` already). The single entry
 * point `pulseqlib_check_safety_from_file()` reads the sequence,
 * validates it (continuity, max grad, max slew, mechanical-resonance
 * forbidden bands, vendor-dispatched PNS), and frees the collection
 * before returning.
 *
 * Build (linking against the static pulseqlib library):
 *   c++ -std=c++17 -I../../csrc safety_with_external_sequence.cpp \
 *       -L<build-dir> -lpulseqlib -lm \
 *       -o safety_with_external_sequence_cpp
 *
 * Run:
 *   ./safety_with_external_sequence_cpp <path-to-some.seq>
 */

/* The pulseqlib public headers already wrap their declarations in
 * extern "C" when compiled with a C++ compiler, so no additional
 * wrapping is needed here. */
#include "pulseqlib_methods.h"
#include "pulseqlib_types.h"

#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <sequence.seq>\n", argv[0]);
        return 1;
    }
    const std::string seq_path = argv[1];

    pulseqlib_opts       opts = PULSEQLIB_OPTS_INIT;
    pulseqlib_diagnostic diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_forbidden_band band = PULSEQLIB_FORBIDDEN_BAND_INIT;
    pulseqlib_pns_params pns  = PULSEQLIB_PNS_PARAMS_INIT;

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

    const int rc = pulseqlib_check_safety_from_file(
        &diag, seq_path.c_str(), &opts,
        /*num_forbidden_bands*/ 1, &band,
        &pns, /*pns_threshold_percent*/ 80.0f);

    if (PULSEQLIB_FAILED(rc)) {
        std::printf("safety check FAILED  rc=%d  msg=\"%s\"\n",
                    rc, diag.message);
        return 2;
    }

    std::printf("safety check PASSED for %s\n", seq_path.c_str());
    return 0;
}
