/*
 * stage2_prescription.c -- Prescription check: is what the operator asked for
 * playable, and how long will it take?
 *
 * This runs every time a parameter changes, so it has to be cheap. It never
 * builds a sequence. There are two ways to answer, and which one applies
 * depends on whether a .seq exists yet:
 *
 *   - no file yet: ask the design host to validate the protocol;
 *   - file already on disk: read its declarations without parsing it.
 *
 *   stage2_prescription <pypulseq_host> <plugin.py>   (ask the host)
 *   stage2_prescription <scan.seq>                    (ask the file)
 */

#include "vendor.h"

static int validate_protocol(const char *host, const char *plugin, const pulseg_opts *opts)
{
    pulseg_bridge bridge;
    pulseg_protocol protocol;
    pulseg_text_buffer reason;
    char reason_storage[256];
    float duration_s = 0.0f;
    int verdict;

    if (pulseg_bridge_open_with_opts(&bridge, host, plugin, opts) != 0)
    {
        (void)fprintf(stderr, "could not start the design host\n");
        return 1;
    }
    if (pulseg_bridge_list_protocol(&bridge, &protocol) < 0)
    {
        pulseg_bridge_close(&bridge);
        return 1;
    }

    /* Whatever the operator changed would be written back here, e.g.
     *   pulseg_protocol_set_float(&protocol, pulseg_param_find("fov"), 240.0f);
     * Left at the defaults, this validates the protocol as it arrived. */

    /* The buffer and its capacity travel as one argument, so there is no way
     * to pass a size that does not match the storage. */
    reason.capacity = (int)sizeof reason_storage;
    reason.data = reason_storage;
    reason_storage[0] = '\0';

    verdict = pulseg_bridge_validate(&bridge, &duration_s, &reason, &protocol);
    pulseg_bridge_close(&bridge);

    if (verdict < 0)
    {
        (void)fprintf(stderr, "the design host did not answer\n");
        return 1;
    }
    if (verdict == 0)
    {
        vendor_log("not playable: %s", reason_storage);
        return 1;
    }

    vendor_log("playable: %s", reason_storage);
    vendor_ui_scan_time((double)duration_s);
    return 0;
}

static int peek_file(const char *seq_path, const pulseg_opts *opts)
{
    pulseg_scan_time_info timing = PULSEG_SCAN_TIME_INFO_INIT;
    pulseg_sequence_flags flags;
    int code;

    /* Reads the [DEFINITIONS] section and stops. The whole point is not to
     * pay for a parse: what comes back is what the file declares about
     * itself, and the duration is an approximation: dead time between
     * segments is not accounted for. */
    code = pulseg_peek_scan_time(&timing, seq_path, opts);
    if (PULSEG_FAILED(code))
    {
        vendor_refuse(code, NULL);
        return 1;
    }
    vendor_ui_scan_time(timing.total_duration_us / 1e6);

    code = pulseg_peek_sequence_flags(&flags, seq_path, opts);
    if (PULSEG_FAILED(code))
    {
        vendor_refuse(code, NULL);
        return 1;
    }
    vendor_log("declared flags read from the head of the chain");
    return 0;
}

int main(int argc, char **argv)
{
    pulseg_opts opts;

    vendor_system_limits(&opts);

    if (argc == 3)
        return validate_protocol(argv[1], argv[2], &opts);
    if (argc == 2)
        return peek_file(argv[1], &opts);

    (void)fprintf(stderr,
                  "usage: %s <pypulseq_host> <plugin.py>\n"
                  "       %s <scan.seq>\n",
                  argv[0],
                  argv[0]);
    return 2;
}
