/**
 * @file pulseg.hpp
 * @brief Single include for the C++ interface to the pulseg library.
 *
 *     #include "pulseg.hpp"
 *     using namespace pulseg;
 *
 *     Opts opts;
 *     opts.gamma_hz_per_t          = 42.576e6f;
 *     opts.b0_t                    = 3.0f;
 *     opts.max_grad_hz_per_m       = 42.576e6f * 0.080f / 1.732f;
 *     opts.max_slew_hz_per_m_per_s = 42.576e6f * 250.0f / 1.732f;
 *     opts.rf_raster_us            = 2.0f;
 *     opts.grad_raster_us          = 4.0f;
 *     opts.adc_raster_us           = 2.0f;
 *     opts.block_raster_us         = 4.0f;
 *
 *     Collection scan("scan.seq", opts);
 *     scan.check_safety(bands);
 *
 * Every name here forwards to the C entry point of the same name without the
 * prefix, with the arguments in the same order. See the C API reference for
 * what each does.
 */

#ifndef PULSEG_HPP
#define PULSEG_HPP

#include "error.hpp"
#include "types.hpp"
#include "collection.hpp"
#include "chunk.hpp"
#include "file.hpp"
#include "protocol.hpp"

#endif // PULSEG_HPP
