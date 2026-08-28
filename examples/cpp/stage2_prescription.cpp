/*
 * stage2_prescription.cpp -- Prescription check.
 *
 * C++ counterpart of examples/c/stage2_prescription.c.
 *
 *   stage2_prescription <pypulseq_host> <plugin.py>   (ask the host)
 *   stage2_prescription <scan.seq>                    (ask the file)
 */

#include "vendor.hpp"

static int validate_protocol(const char* host, const char* plugin, const pulseg::Opts& opts)
{
    pulseg::Bridge bridge(host, plugin, opts);
    pulseg::Protocol protocol = bridge.list_protocol();

    /* Whatever the operator changed would be written back here, e.g.
     *   protocol.set_float(pulseg::Protocol::param_id("fov"), 240.0f); */

    const pulseg::ValidateResult result = bridge.validate(protocol);
    if (!result.playable)
    {
        vendor::log("not playable: " + result.message);
        return 1;
    }
    vendor::log("playable: " + result.message);
    vendor::ui_scan_time(result.duration_s);
    return 0;
}

static int peek_file(const char* seq_path, const pulseg::Opts& opts)
{
    /* Reads the [DEFINITIONS] section and stops. The duration is an
     * approximation: dead time between segments is not accounted for. */
    const pulseg::ScanTimeInfo timing = pulseg::peek_scan_time(seq_path, opts);
    vendor::ui_scan_time(timing.total_duration_us / 1e6);

    const pulseg_sequence_flags flags = pulseg::peek_sequence_flags(seq_path, opts);
    std::printf("sar burst requested: %d\n", flags.enable_sar_burst_mode);
    return 0;
}

int main(int argc, char** argv)
{
    const pulseg::Opts opts = vendor::system_limits();
    try
    {
        if (argc == 3)
            return validate_protocol(argv[1], argv[2], opts);
        if (argc == 2)
            return peek_file(argv[1], opts);
    }
    catch (const pulseg::BridgeError& error)
    {
        std::fprintf(stderr, "bridge: %s\n", error.what());
        return 1;
    }
    catch (const pulseg::Error& error)
    {
        std::fprintf(stderr, "refused: %s\n", error.what());
        return 1;
    }

    std::fprintf(
        stderr,
        "usage: %s <pypulseq_host> <plugin.py>\n       %s <scan.seq>\n",
        argv[0],
        argv[0]);
    return 2;
}
