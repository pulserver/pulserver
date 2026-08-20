/**
 * @file timing.hpp
 * @brief Block timing: rasters, block durations, dead times and ringdown.
 *
 * The question the scanner asks at predownload is narrower -- can the
 * sequencer address every event time -- and `pulseg_check_raster_alignment`
 * answers that one in C, over the deduplicated definitions, against the
 * scanner's own rasters. This is the wider question a design tool asks before
 * a file leaves the bench, and it reads the sequence at the precision the file
 * carries rather than the microsecond grid the interpreter quantises to.
 */

#ifndef PULSEQ_TIMING_HPP
#define PULSEQ_TIMING_HPP

#include <string>
#include <vector>

#include "pulseq/sequence.hpp"

namespace pulseq
{

    /**
     * What the sequence is judged against: the system's, not its own.
     *
     * A file records the rasters it was laid out on, but the question a
     * timing check asks is whether the machine that will play it can address
     * those times -- so the rasters come from the system, alongside the dead
     * times a sequence never carries.
     */
    struct TimingLimits
    {
        double rf_raster_time = 1e-6;
        double grad_raster_time = 10e-6;
        double adc_raster_time = 100e-9;
        double block_duration_raster = 10e-6;

        double rf_dead_time = 0.0;
        double rf_ringdown_time = 0.0;
        double adc_dead_time = 0.0;

        /** Block range to judge, 1-based inclusive; 0 means to the end. */
        int first_block = 1;
        int last_block = 0;
    };

    /**
     * One problem with one block, carrying the fields its kind reports.
     *
     * `error_type` names the kind: RASTER, BLOCK_DURATION_MISMATCH,
     * NEGATIVE_DELAY, RF_DEAD_TIME, RF_RINGDOWN_TIME, ADC_DEAD_TIME,
     * POST_ADC_DEAD_TIME, SOFT_DELAY_FACTOR, SOFT_DELAY_DUR_INCONSISTENCY.
     */
    struct TimingFinding
    {
        int block = 0;
        std::string event;
        std::string field;
        std::string error_type;
        std::string raster;
        std::string hint;
        int num_id = 0;
        double value = 0.0;
        double value_rounded = 0.0;
        double error = 0.0;
        double duration = 0.0;
        double dead_time = 0.0;
        double ringdown_time = 0.0;
    };

    /**
     * Every timing problem in @p seq, in block order.
     *
     * Raster alignment is a property of a library row, so it is decided once
     * per distinct event and attributed to each block that plays it; what is
     * genuinely per block -- the stored duration against the events it holds,
     * and the dead-time margins -- is arithmetic over precomputed extents. No
     * block is decoded.
     *
     * @param seq     The sequence to check.
     * @param limits  Dead times and ringdown, which the sequence does not carry.
     * @return One finding per problem; empty when the sequence is clean.
     */
    std::vector<TimingFinding> check_timing(const Sequence& seq, const TimingLimits& limits);

} // namespace pulseq

#endif /* PULSEQ_TIMING_HPP */
