#ifndef PULSEQLIB_MATH_H
#define PULSEQLIB_MATH_H

#include <math.h>

/* 
 * Round time to the nearest next integer multiple of raster.
 * Uses a small epsilon to avoid rounding up values that are already aligned.
 */
float pulseqlib_roundToRaster(float time, float raster);

/* 
 * Generate time axis for uniform raster waveforms.
 * mode 0: Centers (0.5, 1.5, ...) * raster  [Standard Arbitrary]
 * mode 1: Edges (0.0, 1.0, ...) * raster    [Trapezoids / Extended Traps]
 * mode 2: Oversampled (0.0, 0.5, 1.0, ...) * raster [Special Arbitrary Flag]
 */
void pulseqlib_generateTimeAxis(float* out_time, int num_samples, float raster, int mode);

/* 
 * 1D Linear Interpolation.
 * Extrapolation behavior: Clamps to the nearest boundary value (nearest neighbor).
 * Returns 0 on success, non-zero on error.
 */
int pulseqlib_interpolateLinear(float* out_arr, int out_len, const float* out_time, const float* in_arr, int in_len, const float* in_time);

/* 
 * Trapezoidal Integration.
 * Computes the area under the curve defined by val_arr and time_arr.
 */
float pulseqlib_integrateTrapezoid(const float* val_arr, const float* time_arr, int len);

#endif /* PULSEQLIB_MATH_H */