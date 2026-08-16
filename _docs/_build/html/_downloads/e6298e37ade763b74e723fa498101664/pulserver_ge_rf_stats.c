/*
 * pulserver_ge_rf_stats.c -- GE RF envelope statistics implementation.
 *
 * Math lifted verbatim from the pre-Stage-2 pulseg_dedup.c
 * compute_rf_stats() envelope-stat block (dedup.c:1093-1129 pre-Stage-2
 * line numbers): abswidth = mean |B1|, effwidth = mean |B1|^2, dtycyc =
 * fraction of samples above threshold (floored at the longest run),
 * maxpw = longest contiguous run above threshold. All normalised by the
 * uniform-grid sample count n (dimensionless fractions of the RF event).
 */

#include "pulserver_ge_rf_stats.h"

#include "pulseg_types.h" /* PULSEG_SUCCESS / PULSEG_ERR_* */

#define PULSERVER_GE_RF_DTY_THRESHOLD 0.2236f

int pulserver_ge_rf_stats_cb(void *ctx, const pulseg_rf_view *rf, float out_stat[4])
{
    float sum_abs, sum_sq, time_above_threshold, temp_pw, maxpw, rf_abs;
    int i;

    (void)ctx;
    if (!rf || !rf->mag || rf->n <= 0 || !out_stat)
        return PULSEG_ERR_NULL_POINTER;

    sum_abs = 0.0f;
    sum_sq = 0.0f;
    time_above_threshold = 0.0f;
    maxpw = 0.0f;
    temp_pw = 0.0f;
    for (i = 0; i < rf->n; ++i)
    {
        rf_abs = rf->mag[i];
        sum_abs += rf_abs;
        sum_sq += rf_abs * rf_abs;
        if (rf_abs > PULSERVER_GE_RF_DTY_THRESHOLD)
        {
            time_above_threshold += 1.0f;
            temp_pw += 1.0f;
        }
        else
        {
            if (temp_pw > maxpw)
                maxpw = temp_pw;
            temp_pw = 0.0f;
        }
    }
    if (temp_pw > maxpw)
        maxpw = temp_pw;
    if (time_above_threshold < maxpw)
        time_above_threshold = maxpw;

    out_stat[0] = sum_abs / (float)rf->n;             /* abswidth */
    out_stat[1] = sum_sq / (float)rf->n;               /* effwidth */
    out_stat[2] = time_above_threshold / (float)rf->n; /* dtycyc   */
    out_stat[3] = maxpw / (float)rf->n;                /* maxpw    */
    return PULSEG_SUCCESS;
}
