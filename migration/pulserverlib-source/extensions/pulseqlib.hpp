/**
 * @file pulseqlib.hpp
 * @brief C++11 interface for the pulseqlib library.
 *
 * Single-include header.  Usage:
 *
 *     #include "pulseqlib.hpp"
 *     using namespace pulseqlib;
 *
 *     Opts opts;
 *     opts.gamma_hz_per_t    = 42.576e6f;
 *     opts.b0_t              = 3.0f;
 *     opts.max_grad_hz_per_m = 42.576e6f * 0.040f;
 *     opts.max_slew_hz_per_m_per_s = 42.576e6f * 150.0f;
 *     opts.grad_raster_us    = 4.0f;
 *     opts.rf_raster_us      = 1.0f;
 *     opts.adc_raster_us     = 2.0f;
 *     opts.block_raster_us   = 10.0f;
 *
 *     const char* buf = data.data();
 *     int         len = static_cast<int>(data.size());
 *     Collection coll(&buf, &len, 1, opts);
 *
 *     auto wf = coll.get_tr_gradient_waveforms();
 */

#ifndef PULSEQLIB_HPP
#define PULSEQLIB_HPP

#include "error.hpp"
#include "types.hpp"
#include "collection.hpp"

#endif // PULSEQLIB_HPP
