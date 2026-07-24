/**
 * @file pulseq.hpp
 * @brief C++11 interface for the standalone pulseq (.seq) reader.
 *
 * Single-include header.  Usage:
 *
 *     #include "pulseq.hpp"
 *
 *     pulseq::Raster raster;
 *     raster.rf_us = 1.0f;
 *     raster.grad_us = 4.0f;
 *     raster.block_us = 10.0f;
 *
 *     pulseq::File file = pulseq::File::read("scan.seq", raster);
 *     for (int b = 0; b < file.num_blocks(); ++b)
 *     {
 *         pulseq_raw_block raw = file.raw_block_content_ids(b);
 *         // ...
 *     }
 */

#ifndef PULSEQ_CXX_HPP
#define PULSEQ_CXX_HPP

#include "error.hpp"
#include "types.hpp"
#include "file.hpp"

#endif // PULSEQ_CXX_HPP
