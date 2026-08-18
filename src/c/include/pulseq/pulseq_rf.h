/**
 * @file pulseq_rf.h
 * @brief The spectrum of an RF pulse, and the bandwidth read off it.
 *
 * ### What this is for
 *
 * A slice-selective pulse played on a gradient excites a slab whose thickness
 * is its spectral bandwidth over that gradient's amplitude.  Nothing in a
 * `.seq` file records the thickness, so anything that wants it -- a
 * `SliceThickness` definition, a slice gap, an RF statistics table -- has to
 * take the transform of the pulse.
 *
 * This is that transform, at the pulseq level, because two things in this
 * repository need it and neither may call the other: pulseg's RF statistics
 * (`compute_rf_stats`) and pulseq's own label derivation.  pulseg links pulseq
 * and never the reverse, so the shared piece belongs here.
 *
 * ### Why it is a plan
 *
 * The grid is set by the raster and the wanted spectral resolution alone --
 * 10 Hz over a 1 us raster is a hundred thousand points -- so it does not
 * depend on the pulse.  A sequence with a variable-flip train has hundreds of
 * distinct pulses sharing one grid, and allocating a transform per pulse would
 * cost more than the transforms.  Create one plan, run every pulse through it.
 *
 * ### Agreement with Pulseq
 *
 * The recipe is `mr.calcRfBandwidth`'s, including its defaults: resample the
 * complex pulse onto a uniform grid centred on the pulse centre, transform,
 * and take the outermost frequencies at which the magnitude first exceeds
 * `cutoff` times its peak -- `mr.aux.findFlank` from each end.  Pulseq's
 * defaults are a cutoff of 0.5 and a resolution of 10 Hz.
 *
 * A frequency offset moves the spectrum but not its width, so the bandwidth
 * does not depend on whether the caller folded one into the pulse.  The
 * centre frequency does.
 */

#ifndef PULSEQ_RF_H
#define PULSEQ_RF_H

#include "pulseq_config.h"
#include "pulseq_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /** Pulseq's own defaults, so a caller need not restate them. */
#define PULSEQ_RF_DEFAULT_CUTOFF 0.5
#define PULSEQ_RF_DEFAULT_RESOLUTION_HZ 10.0

    /**
     * @brief A reusable resample-and-transform grid.
     *
     * Opaque; the spectrum it produces is read back through the accessors
     * below and is overwritten by the next @ref pulseq_rf_spectrum_run.
     */
    typedef struct pulseq_rf_spectrum pulseq_rf_spectrum;

    /**
     * @brief Build a plan for pulses on @p raster_us at @p resolution_hz.
     *
     * The grid holds `1 / (resolution_hz * raster)` points, rounded up to a
     * size the transform likes.  @p resolution_hz of 0 means
     * @ref PULSEQ_RF_DEFAULT_RESOLUTION_HZ.
     *
     * @return PULSEQ_SUCCESS, or a negative error code.
     */
    int pulseq_rf_spectrum_create(
        pulseq_rf_spectrum **out,
        PULSEQ_REAL raster_us,
        PULSEQ_REAL resolution_hz);

    /** @brief Release a plan. */
    void pulseq_rf_spectrum_free(pulseq_rf_spectrum *plan);

    /**
     * @brief Transform one pulse.
     *
     * @p re and @p im are the pulse's complex samples -- magnitude and phase
     * already combined -- at times @p t_us, in microseconds, which need not be
     * uniform.  @p center_us is the pulse's centre in the same units: the
     * spectrum is taken about it, which is what makes a pulse's phase ramp
     * read as a frequency offset rather than as a smeared band.
     *
     * Samples outside the plan's window contribute nothing, and a pulse
     * shorter than the window is zero-padded -- which is the resolution being
     * asked for.
     *
     * @return PULSEQ_SUCCESS, or a negative error code.
     */
    int pulseq_rf_spectrum_run(
        pulseq_rf_spectrum *plan,
        const PULSEQ_REAL *re,
        const PULSEQ_REAL *im,
        const PULSEQ_REAL *t_us,
        int count,
        PULSEQ_REAL center_us);

    /** @brief Points in the spectrum. */
    int pulseq_rf_spectrum_size(const pulseq_rf_spectrum *plan);
    /** @brief Frequency axis, Hz, ascending and centred on zero. */
    const PULSEQ_REAL *pulseq_rf_spectrum_freq(const pulseq_rf_spectrum *plan);
    /** @brief Real and imaginary parts of the last transform. */
    const PULSEQ_REAL *pulseq_rf_spectrum_re(const pulseq_rf_spectrum *plan);
    const PULSEQ_REAL *pulseq_rf_spectrum_im(const pulseq_rf_spectrum *plan);

    /**
     * @brief Width of the last transform at @p cutoff of its peak, Hz.
     *
     * @p cutoff of 0 means @ref PULSEQ_RF_DEFAULT_CUTOFF.  @p out_center_hz,
     * when not NULL, receives the midpoint of the two flanks -- the pulse's
     * centre frequency.
     *
     * @return the bandwidth, or 0 when the spectrum is empty or flat.  Zero is
     * "not measurable", never a default: a caller that wants an analytic
     * stand-in has to choose one knowingly.
     */
    PULSEQ_REAL pulseq_rf_bandwidth(
        const pulseq_rf_spectrum *plan,
        PULSEQ_REAL cutoff,
        PULSEQ_REAL *out_center_hz);

#ifdef __cplusplus
}
#endif

#endif /* PULSEQ_RF_H */
