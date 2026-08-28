/*
 * safety_gate.cpp -- the checks on their own.
 *
 * C++ counterpart of examples/c/safety_gate.c.
 *
 *   safety_gate <scan.seq>
 */

#include "vendor.hpp"

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::fprintf(stderr, "usage: %s <scan.seq>\n", argv[0]);
        return 2;
    }

    try
    {
        /* Collection releases the C handle in its destructor, on the throwing
         * path as well as the normal one. */
        pulseg::Collection scan(argv[1], vendor::system_limits());

        /* No plan: the checks keep their preprocessing private to the call. */
        scan.check_safety(vendor::forbidden_bands());
    }
    catch (const pulseg::Error& error)
    {
        std::fprintf(stderr, "refused: %s\n", error.what());
        return 1;
    }

    vendor::log("playable");
    return 0;
}
