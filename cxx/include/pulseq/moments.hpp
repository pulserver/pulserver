/**
 * @file moments.hpp
 * @brief The diffusion b-tensor and the gradient moments, one per excitation.
 *
 * ### Why this is C++ and not C89
 *
 * Unlike the trajectory, nothing on the scanner's pulse-generation side ever
 * wants a b-value -- the interpreter plays gradients, it does not characterise
 * their diffusion weighting.  The consumer is the *reconstruction*: LiveSDK
 * putting a gradient table into the MRD stream, and dipy or MRtrix reading it
 * back.  So this lives in `cxx/pulseq`, which is what LiveSDK links, and works
 * from a `pulseq::Sequence` built straight off the `.seq` rather than from the
 * `.pge` cache.
 *
 * ### The two rotations
 *
 * Conflating them is the trap this file exists to avoid.
 *
 *  - A **`ROTATIONS` extension** turns the gradient inside the *logical*
 *    frame.  It is a property of the waveform, it is written in the `.seq`,
 *    and it is resolved here.
 *  - The **prescription** -- the FOV rotation a radiographer sets on the
 *    console -- is applied by the interpreter and is *not* in the `.seq`.
 *    `NOROT` is how a block opts out of it.
 *
 * A product diffusion sequence puts its diffusion preparation under `NOROT`
 * precisely so the b-matrix does not swing with the prescribed slice
 * orientation.  But the imaging gradients in the very same shot -- prewinder,
 * crushers, readout -- are *not* `NOROT`, and their cross terms with the
 * diffusion lobes are real.  So a shot is genuinely mixed and there is no one
 * frame its tensor lives in.
 *
 * Rather than pick a frame and be wrong, this reports the decomposition.
 * With `k(t) = k_fixed(t) + R k_rot(t)` for a prescription @f$R@f$:
 *
 * @f[
 *   B(R) = B_{fixed} + C R^T + R C^T + R\,B_{rot}\,R^T
 * @f]
 *
 * which @ref compose_btensor evaluates.  Twenty-seven numbers a shot instead
 * of nine, and both degenerate cases fall out of it: an all-`NOROT` shot has
 * `B = B_fixed` and does not move with the prescription at all, while a shot
 * with no `NOROT` has the familiar `B = R B_rot Rᵀ`.  At `R = I` the sum is
 * what MATLAB's `calcMomentsBtensor` returns.
 *
 * ### Exactness
 *
 * A Pulseq gradient is piecewise linear, so `k` is piecewise quadratic and
 * `k_i k_j` piecewise quartic.  Every integral here is closed form on each
 * piece; there is no step size and therefore no discretisation error to
 * report.  Verified against a two-million-point quadrature in
 * `tests/python/test_pypulseq_analysis.py`.
 *
 * ### Where the echo comes from
 *
 * A shot's integral runs from its excitation to its echo, and the echo is the
 * ADC sample nearest k-space zero -- which @ref calculate_kspace already
 * derives while integrating the trajectory (`Readout::center_sample`).  It is
 * read from there rather than re-derived, so this and the reconstruction agree
 * by construction, and a twice-refocused sequence needs no special case.
 * MATLAB instead reconstructs `2*t_refocusing - t_excitation` and carries its
 * own `TODO: fixme for double-refocused sequences`; that formula is only used
 * here as a fallback for a design that has no ADC yet.
 */

#ifndef PULSEQ_CXX_MOMENTS_HPP
#define PULSEQ_CXX_MOMENTS_HPP

#include <array>
#include <vector>

namespace pulseq
{

    class Sequence;
    struct KSpace;

    /** A 3x3 matrix, row-major. */
    using Matrix3 = std::array<double, 9>;

    /** What @ref calc_moments computes and over which blocks. */
    struct MomentsOptions
    {
        /** Compute the b-tensor.  Cheap; on by default. */
        bool calc_b = true;
        /** Compute the first, second and third gradient moments. */
        bool calc_m1 = false;
        bool calc_m2 = false;
        bool calc_m3 = false;

        /** Leading excitations to skip -- dummy shots that acquire nothing. */
        int n_dummy = 0;

        /** 1-based inclusive; 0 for `last_block` means "to the end". */
        int first_block = 1;
        int last_block = 0;

        /**
         * Round the tensors to this many decimal digits before deduplicating.
         *
         * Only the table is affected; the per-shot tensors are reported at
         * full precision either way.  Six digits on a b-value near 1000 s/mm²
         * separates directions that differ at all from ones that differ only
         * in the last bits of an amplitude that was written to the file as a
         * decimal.
         */
        int table_digits = 6;

        /**
         * An already-computed trajectory to take the echoes from.
         *
         * Optional.  When null this calls @ref calculate_kspace itself, which
         * costs one serialise and one parse; a caller that already holds a
         * `KSpace` over the same block range should hand it over instead of
         * paying for a second one.  It must have been computed with
         * `derive_center_sample` on, or there are no echoes in it.
         */
        const KSpace* kspace = nullptr;
    };

    /** The b-tensor of one shot, split by what the prescription rotates. */
    struct BTensorParts
    {
        /** `NOROT` gradients only, s/m².  Prescription-independent. */
        Matrix3 fixed{};
        /** Everything else, s/m².  The prescription turns this. */
        Matrix3 rotatable{};
        /** `(2π)² ∫ k_fixed k_rotᵀ dt`, s/m².  Not symmetric. */
        Matrix3 cross{};
    };

    /** Everything @ref calc_moments produces, one entry per shot. */
    struct Moments
    {
        /** Excitation times, seconds, in playing order. */
        std::vector<double> t_excitation;
        /** Echo times, seconds.  NaN for a shot whose echo was not found. */
        std::vector<double> t_echo;

        /** Empty unless `MomentsOptions::calc_b`. */
        std::vector<BTensorParts> b;

        /**
         * `2π ∫ g (t - t_excitation)ⁿ dt` per axis, in the logical frame.
         *
         * Empty unless the matching `calc_mN` was asked for.  These are *not*
         * split the way the b-tensor is: a moment is linear in the gradient,
         * so a prescription `R` turns the whole vector and there is nothing
         * to decompose.
         */
        std::vector<std::array<double, 3>> m1;
        std::vector<std::array<double, 3>> m2;
        std::vector<std::array<double, 3>> m3;

        /**
         * The distinct b-tensors, in first-seen order.
         *
         * A diffusion scan plays a few dozen directions over many thousands of
         * shots, and what a reconstruction wants is that short list plus a way
         * to index it -- which is also what fits in an MRD header.  The
         * deduplication is done on the computed *values* rather than on a
         * structural key: it makes no assumption that shots share a template,
         * so a free-waveform or variable-b design deduplicates as well as a
         * Stejskal-Tanner one.
         */
        std::vector<BTensorParts> table;
        /** Row of @ref table for each shot, or -1 when `calc_b` was off. */
        std::vector<int> table_index;

        int num_shots() const { return static_cast<int>(t_excitation.size()); }
    };

    /**
     * The b-tensor and gradient moments of every excitation in @p seq.
     *
     * Takes the sequence by non-const reference for the same reason
     * @ref calculate_kspace does: the binary writer records `TotalDuration`
     * before serialising.
     */
    Moments calc_moments(Sequence& seq, const MomentsOptions& options = {});

    /** `B_fixed + C Rᵀ + R Cᵀ + R B_rot Rᵀ`, s/m². */
    Matrix3 compose_btensor(const BTensorParts& parts, const Matrix3& rotation);

    /** The identity, which is the prescription a design script has. */
    const Matrix3& identity3();

}  // namespace pulseq

#endif /* PULSEQ_CXX_MOMENTS_HPP */
