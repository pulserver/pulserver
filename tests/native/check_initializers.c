/*
 * Instantiate every initializer the public headers define.
 *
 * An initializer is the one thing in a header that no other translation unit
 * has to expand, so one left behind when its structure changes compiles
 * everywhere except in the scanner that finally uses it.  Compiled by
 * tests/test_ir.py with warnings as errors, where a missing or excess field
 * is a diagnostic.
 */

#include <stdio.h>

#include "pulseg.h"
#include "pulseg_cache.h"
#include "pulseg_collection.h"
#include "pulseg_io.h"
#include "pulseg_playout.h"
#include "pulseq.h"

int main(void)
{
    pulseg_adc_def adc = PULSEG_ADC_DEF_INIT;
    pulseg_block_info block = PULSEG_BLOCK_INFO_INIT;
    pulseg_block_instance instance = PULSEG_BLOCK_INSTANCE_INIT;
    pulseg_collection_info collection = PULSEG_COLLECTION_INFO_INIT;
    pulseg_corner_point_stream corners = PULSEG_CORNER_POINT_STREAM_INIT;
    pulseg_cursor_info cursor = PULSEG_CURSOR_INFO_INIT;
    pulseg_diagnostic diagnostic = PULSEG_DIAGNOSTIC_INIT;
    pulseg_grouping grouping = PULSEG_GROUPING_INIT;
    pulseg_opts opts = PULSEG_OPTS_INIT;
    pulseg_playout_options playout = PULSEG_PLAYOUT_OPTIONS_INIT;
    pulseg_rf_shim_def shim = PULSEG_RF_SHIM_DEF_INIT;
    pulseg_rf_stats rf = PULSEG_RF_STATS_INIT;
    pulseg_scan_time_info scan_time = PULSEG_SCAN_TIME_INFO_INIT;
    pulseg_segment_info segment = PULSEG_SEGMENT_INFO_INIT;
    pulseg_segment_layout layout = PULSEG_SEGMENT_LAYOUT_INIT;
    pulseg_sequence_flags flags = PULSEG_SEQUENCE_FLAGS_INIT;
    pulseg_subseq_info subsequence = PULSEG_SUBSEQ_INFO_INIT;
    pulseg_text_buffer text = PULSEG_TEXT_BUFFER_INIT;
    pulseg_wave_budget budget = PULSEG_WAVE_BUDGET_INIT;
    pulseg_wave_plan plan = PULSEG_WAVE_PLAN_INIT;
    pulseq_shape shape = PULSEQ_SHAPE_INIT;

    /* Read one field of each, so none is optimised away unexamined. */
    printf(
        "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n",
        adc.num_samples, block.duration_us, instance.duration_us,
        collection.num_subsequences, corners.num_points, cursor.scan_pos,
        diagnostic.code, opts.vendor, playout.prescan_subsequence,
        shim.num_channels, (int)rf.flip_angle_rad, (int)scan_time.total_duration_us,
        segment.num_blocks, layout.num_blocks, flags.enable_sar_burst_mode,
        subsequence.num_trs, text.capacity, (int)budget.max_samples,
        plan.mode, shape.num_samples, grouping.split_by_pulses);
    return 0;
}
