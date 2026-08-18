/**
 * @file shape.hpp
 * @brief Pulseq's shape codec: run-length encoding of the derivative.
 *
 * A gradient or pulse shape is stored in a `.seq` file compressed, and the
 * scheme is chosen for what MR waveforms actually look like: a constant run
 * and a linear ramp both differentiate to one repeated number, so a trapezoid
 * of ten thousand samples is a handful of them.
 *
 * Compression is not free -- it is a pass over every sample, with a
 * quantisation and a correction term -- which is why it belongs *here*, at the
 * end, rather than where a shape is registered.  A scan that registers a
 * waveform per shot registers hundreds of thousands of them and keeps a few:
 * compressing on the way in pays for every candidate, compressing on the way
 * out pays only for the survivors.
 */

#ifndef PULSEQ_CXX_SHAPE_HPP
#define PULSEQ_CXX_SHAPE_HPP

#include <vector>

namespace pulseq
{

    /**
     * @p count samples, run-length encoded on their derivative.
     *
     * Returns the compressed form, or a copy of the samples when compressing
     * them would not make them shorter -- which is Pulseq's own rule, and what
     * decides whether a `[SHAPES]` entry holds a code or a waveform.  Shapes of
     * four samples or fewer are never compressed.
     */
    std::vector<double> compress_shape(const double* samples, int count);

    /**
     * The inverse: @p count encoded samples back to @p num_uncompressed of
     * them.
     *
     * A port of `pulseq_decompress_shape` (src/c/pulseq/pulseq_parse.c) and
     * kept in step with it, so a shape read by the C reader and one decoded
     * here agree sample for sample.  When the two counts are equal the entry
     * was never compressed -- compress_shape's "keep the original if it is no
     * longer" rule -- and the samples are copied through unchanged.
     *
     * Throws std::runtime_error on a malformed run length, which is the only
     * way the encoding can be inconsistent with itself.
     */
    std::vector<double> decompress_shape(const double* samples, int count, int num_uncompressed);

} // namespace pulseq

#endif /* PULSEQ_CXX_SHAPE_HPP */
