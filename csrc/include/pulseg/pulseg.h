/**
 * @file pulseg.h
 * @brief Umbrella header for the public pulseg C API.
 *
 * Include this single header in application code; it pulls in every public
 * module header. The complete public API is the contents of
 * @c csrc/include/pulseg/ -- anything under @c csrc/src/ (notably
 * @c pulseg_internal.h) is private and may change without notice.
 *
 * Conventions holding across every header here:
 *   - every entry point is prefixed @c pulseg_ and declared @c extern "C"
 *     when compiled as C++;
 *   - functions return @c PULSEG_SUCCESS or a negative @c PULSEG_ERR_* code,
 *     testable with @c PULSEG_SUCCEEDED() / @c PULSEG_FAILED();
 *   - arguments are ordered (self, outputs, inputs); a pointer and the
 *     capacity that follows it count as one argument;
 *   - physical quantities carry unit suffixes (@c _us, @c _hz,
 *     @c _hz_per_m, ...).
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
