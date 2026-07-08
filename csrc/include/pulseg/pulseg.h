/**
 * @file pulseg.h
 * @brief Umbrella header for the public Pulseg C API.
 *
 * Include this single header in application code; it pulls in every
 * public module header (config/types plus the per-module API headers
 * produced by the Stage 1 split of the former pulseg_methods.h).
 */

#ifndef PULSEG_H
#define PULSEG_H

#include "pulseg_config.h"
#include "pulseg_types.h"

#include "pulseg_io.h"
#include "pulseg_convert.h"
#include "pulseg_collection.h"
#include "pulseg_safety.h"
#include "pulseg_trajectory.h"
#include "pulseg_cache.h"
#include "pulseg_recon.h"
#include "pulseg_freqmod.h"

#include "pulseg_protocol.h"
#include "pulseg_bridge.h"

#endif /* PULSEG_H */
