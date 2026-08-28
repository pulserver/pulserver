/*
 * stage1_setup.c -- Setup: limits, design host, protocol, console UI.
 *
 * The first thing an interpreter does, once per session. It states what the
 * hardware can do, opens a connection to the design host, asks that host what
 * parameters its sequence takes, and draws them.
 *
 * Nothing is read from disk here and no sequence exists yet. What comes back
 * is a protocol: the parameters and their defaults, which the console then
 * owns until the operator prescribes a scan.
 *
 *   stage1_setup <pypulseq_host> <plugin.py>
 */

#include "vendor.h"

int main(int argc, char **argv)
{
    pulseg_opts opts;
    pulseg_bridge bridge;
    pulseg_protocol protocol;
    int count, i;

    if (argc != 3)
    {
        (void)fprintf(stderr, "usage: %s <pypulseq_host> <plugin.py>\n", argv[0]);
        return 2;
    }

    /* 1. What the hardware can do
     * Every check and every raster comparison downstream is against this,
     * so it is stated once and passed everywhere. */
    vendor_system_limits(&opts);
    vendor_log("limits: %.1f mT/m, %.0f T/m/s, gradient raster %.0f us",
               (double)(opts.max_grad_hz_per_m / opts.gamma_hz_per_t * 1000.0f),
               (double)(opts.max_slew_hz_per_m_per_s / opts.gamma_hz_per_t),
               (double)opts.grad_raster_us);

    /* 2. Open the design host
     * Sequence generation is Python; everything downstream of a .seq file is
     * this library. The bridge is the seam. It stays open for the session,
     * so the child process is spawned once rather than per prescription.
     *
     * The bridge reports process and pipe failures as -1 with errno set, not
     * as PULSEG_ERR_* codes: what fails here is not a sequence model. */
    if (pulseg_bridge_open_with_opts(&bridge, argv[1], argv[2], &opts) != 0)
    {
        (void)fprintf(stderr, "could not start the design host\n");
        return 1;
    }

    /* 3. Ask what the sequence takes */
    count = pulseg_bridge_list_protocol(&bridge, &protocol);
    if (count < 0)
    {
        (void)fprintf(stderr, "the design host did not answer\n");
        pulseg_bridge_close(&bridge);
        return 1;
    }
    vendor_log("protocol: %d parameters", count);

    /* 4. Draw them
     * The console does not need to know what the sequence is. Each parameter
     * carries a wire name and a declared type, which is enough to build the
     * control, so a new sequence needs no console-side change. */
    for (i = 0; i < protocol.count; ++i)
    {
        int id = protocol.keys[i];
        vendor_ui_declare(pulseg_param_wire_name(id), pulseg_param_get_type(id));
    }

    /* The bridge is deliberately left open across the exam in a real
     * integration; closing here is what ends this example. */
    pulseg_bridge_close(&bridge);
    return 0;
}
