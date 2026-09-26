/**
 * @file playout.hpp
 * @brief Both stages of a playout, over a backend that records them.
 */

#pragma once

#include <pybind11/pybind11.h>

#include "pulseg.h"

namespace native
{

/* Run pulseg_playout_prepare() and pulseg_playout_scan() on @p coll over a
 * backend that plays nothing and records what each stage hands it; see
 * pulserver.ir.playout. */
pybind11::dict record_playout(
    pulseg_collection *coll,
    const pulseg_wave_budget &budget,
    const pulseg_playout_options &options);

} // namespace native
