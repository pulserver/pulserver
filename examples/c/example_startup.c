/**
 * @file example_startup.c
 * @brief Initialise the global pulseqlib configuration at startup.
 *
 * In a real driver these variables are parsed once from the system
 * configuration database and remain valid for the entire scan
 * session.  Every other section (check, geninstructions, scanloop)
 * reads them as globals.
 *
 * Compile:
 *   cc -I../../csrc example_startup.c ../../csrc/pulseqlib_*.c -lm -o startup
 */

#include "example_vendorlib.h"

#include <stdio.h>

/* ================================================================== */
/*  Global state — shared with check / geninstructions / scanloop     */
/* ================================================================== */

#define MAX_FORBIDDEN_BANDS 16

pulseqlib_opts           g_opts;
pulseqlib_diagnostic     g_diag;
pulseqlib_pns_params     g_pns;
pulseqlib_forbidden_band g_bands[MAX_FORBIDDEN_BANDS];
int                      g_num_bands;

/* ================================================================== */
/*  Startup                                                           */
/* ================================================================== */

/**
 * @brief Parse system configuration and fill the global structs.
 *
 * In a real driver the scalar values below come from the scanner's
 * hardware configuration database (e.g. cvinit / SysParms / …).
 * Here we hard-code typical 3 T values for illustration.
 */
static int startup(void)
{
    /* ----- Scanner opts ------------------------------------------ */
    vendor_opts_init(&g_opts,
                     42577478.0f,   /* gamma  (Hz / T)               */
                     3.0f,          /* B0     (T)                    */
                     50.0f,         /* max_grad (mT / m)             */
                     200.0f);       /* max_slew (T / m / s)          */

    /* ----- PNS model --------------------------------------------- */
    vendor_pns_params_init(&g_pns, &g_opts,
                           360.0f,  /* chronaxie (us)                */
                           20.0f,   /* rheobase  (T / m / s)         */
                           0.333f); /* alpha     (dimensionless)     */

    /* ----- Acoustic forbidden bands ------------------------------ */
    /*
     * Up to MAX_FORBIDDEN_BANDS entries.  In a real driver these are
     * read from a vendor-supplied table (e.g. a config file or
     * hardcoded per gradient coil model).
     *
     * Each band specifies [freq_min_hz, freq_max_hz] and the maximum
     * allowed gradient spectral amplitude within that range.
     */
    g_num_bands = 2;

    g_bands[0].freq_min_hz            = 540.0f;
    g_bands[0].freq_max_hz            = 580.0f;
    g_bands[0].max_amplitude_hz_per_m = 0.0f;   /* strictly forbidden */

    g_bands[1].freq_min_hz            = 1060.0f;
    g_bands[1].freq_max_hz            = 1140.0f;
    g_bands[1].max_amplitude_hz_per_m = 5000.0f; /* limited amplitude */

    /* ----- Diagnostic (zero-init) -------------------------------- */
    pulseqlib_diagnostic_init(&g_diag);

    return 0;
}

/* ================================================================== */
/*  Main (sanity print)                                               */
/* ================================================================== */

int main(void)
{
    startup();

    printf("System configuration:\n");
    printf("  gamma     = %.0f Hz/T\n", g_opts.gamma_hz_per_t);
    printf("  B0        = %.1f T\n",    g_opts.b0_t);
    printf("  max_grad  = %.0f Hz/m\n", g_opts.max_grad_hz_per_m);
    printf("  max_slew  = %.0f Hz/m/s\n", g_opts.max_slew_hz_per_m_per_s);
    printf("  rasters   = rf=%.1f  grad=%.1f  adc=%.1f  block=%.1f us\n",
           g_opts.rf_raster_us, g_opts.grad_raster_us,
           g_opts.adc_raster_us, g_opts.block_raster_us);
    printf("  PNS       = chronaxie=%.0f us, rheobase=%.0f Hz/m/s, alpha=%.3f\n",
           g_pns.chronaxie_us, g_pns.rheobase_hz_per_m_per_s, g_pns.alpha);
    printf("  Bands     = %d forbidden\n", g_num_bands);

    return 0;
}