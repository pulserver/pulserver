#include <stdlib.h>
#include <stdio.h>

#include "pulseqlib_math.h"

#define PULSEQ_EPSILON 1e-9

float pulseqlib_roundToRaster(float time, float raster) {
    if (raster <= 0) return time;
    /* Subtract epsilon to handle floating point inaccuracies where time is already a multiple */
    return ceilf(time / raster - PULSEQ_EPSILON) * raster;
}


void pulseqlib_generateTimeAxis(float* out_time, int num_samples, float raster, int mode) {
    int i;
    if (!out_time) return;

    for (i = 0; i < num_samples; i++) {
        if (mode == 0) {
            /* Centers: (i + 0.5) * dt */
            out_time[i] = (((float)i + 0.5f) * raster);
        } else if (mode == 1) {
            /* Edges: i * dt */
            out_time[i] = ((float)i * raster);
        } else if (mode == 2) {
            /* Oversampled: i * 0.5 * dt */
            out_time[i] = ((float)i * 0.5f * raster);
        }
    }
}


int pulseqlib_interpolateLinear(float* out_arr, int out_len, const float* out_time, const float* in_arr, int in_len, const float* in_time) {
    int i, j;
    float t, t0, t1, y0, y1, slope;

    if (!out_arr || !out_time || !in_arr || !in_time || out_len <= 0 || in_len <= 0) {
        return 1;
    }

    /* Optimization: assume out_time is monotonic increasing, so we track j */
    j = 0;

    for (i = 0; i < out_len; i++) {
        t = out_time[i];

        /* Handle extrapolation (clamp to boundaries) */
        if (t <= in_time[0]) {
            out_arr[i] = in_arr[0];
            continue;
        }
        if (t >= in_time[in_len - 1]) {
            out_arr[i] = in_arr[in_len - 1];
            continue;
        }

        /* Find segment [t0, t1] such that t0 <= t < t1 */
        /* Advance j until in_time[j+1] > t */
        while (j < in_len - 1 && in_time[j + 1] <= t) {
            j++;
        }

        /* Safety check for j */
        if (j >= in_len - 1) {
            out_arr[i] = in_arr[in_len - 1];
            continue;
        }

        t0 = in_time[j];
        t1 = in_time[j + 1];
        y0 = in_arr[j];
        y1 = in_arr[j + 1];

        /* Avoid division by zero */
        if (fabs(t1 - t0) < PULSEQ_EPSILON) {
            out_arr[i] = y0;
        } else {
            slope = (y1 - y0) / (t1 - t0);
            out_arr[i] = y0 + slope * (t - t0);
        }
    }

    return 0;
}


float pulseqlib_integrateTrapezoid(const float* val_arr, const float* time_arr, int len) {
    int i;
    float area = 0.0f;

    if (!val_arr || !time_arr || len < 2) return 0.0f;
    for (i = 0; i < len - 1; i++) {
        float dt = time_arr[i + 1] - time_arr[i];
        float avg_height = (val_arr[i] + val_arr[i + 1]) * 0.5f;
        area += avg_height * dt;
    }

    return area;
}