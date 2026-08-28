/*
 * stage1_setup.cpp -- Setup: limits, design host, protocol, console UI.
 *
 * C++ counterpart of examples/c/stage1_setup.c.
 *
 *   stage1_setup <pypulseq_host> <plugin.py>
 */

#include "vendor.hpp"

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        std::fprintf(stderr, "usage: %s <pypulseq_host> <plugin.py>\n", argv[0]);
        return 2;
    }

    try
    {
        /* 1. What the hardware can do. Every check and every raster
         * comparison downstream is against this. */
        const pulseg::Opts opts = vendor::system_limits();
        std::printf(
            "limits: %.1f mT/m, %.0f T/m/s, gradient raster %.0f us\n",
            double(opts.max_grad_hz_per_m / opts.gamma_hz_per_t * 1000.0f),
            double(opts.max_slew_hz_per_m_per_s / opts.gamma_hz_per_t),
            double(opts.grad_raster_us));

        /* 2. Open the design host. The child process is spawned here and
         * reaped by the destructor, so it lives as long as the Bridge. */
        pulseg::Bridge bridge(argv[1], argv[2], opts);

        /* 3. Ask what the sequence takes. */
        const pulseg::Protocol protocol = bridge.list_protocol();
        std::printf("protocol: %d parameters\n", protocol.size());

        /* 4. Draw them. A wire name and a declared type is enough to build
         * the control, so a new sequence needs no console-side change. */
        for (int id : protocol.keys())
            vendor::ui_declare(pulseg::Protocol::wire_name(id), pulseg::Protocol::param_type(id));
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
    return 0;
}
