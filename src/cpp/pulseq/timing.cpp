/**
 * @file timing.cpp
 * @brief Block timing checks.  See timing.hpp.
 *
 * Two passes.  The first decides, once per library row, everything that is a
 * property of the event itself: whether its delay lands on a raster, whether
 * that delay is negative, and how long it lasts.  The second walks the block
 * table -- integer columns, no decoding -- attributing each row's verdict to
 * the blocks that play it and asking the questions only a block can answer.
 *
 * The event set is the one upstream PyPulseq walks: RF, the three gradient
 * axes, the ADC, and soft delays.  Triggers and labels are stored per block in
 * a dictionary rather than as bare events, which upstream's field walk steps
 * over, so their times are not judged here either.
 */

#include "pulseq/timing.hpp"

#include "pulseq/shape.hpp"

#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace pulseq
{

    namespace
    {

        /** The gradient axes, in block-table column order. */
        const char* const kAxis[3] = {"gx", "gy", "gz"};

        /** Upstream's tolerance: 1e-9 s, a nanosecond. */
        constexpr double kEps = 1e-9;

        /** What a delay, duration or dwell is measured against. */
        struct Raster
        {
            double time;
            const char* name;
        };

        /**
         * Whether @p value is an integer multiple of @p raster, and by how far
         * it misses.
         *
         * The comparison is upstream's: the ratio must land within 1e-6 of an
         * integer, which is a relative test and so independent of how long the
         * event is.
         */
        bool on_raster(double value, double raster, double* rounded, double* error)
        {
            const double ratio = value / raster;
            /* Ties to even, which is what Python's round() does and so what
             * the value a report prints has to agree with. */
            const double nearest = std::nearbyint(ratio);
            *rounded = nearest * raster;
            *error = value - *rounded;
            return std::fabs(ratio - nearest) < 1e-6;
        }

        /** A RASTER finding for @p value, with the block left to the caller. */
        TimingFinding raster_finding(
            const char* event,
            const char* field,
            double value,
            const Raster& raster)
        {
            TimingFinding f;
            f.event = event;
            f.field = field;
            f.error_type = "RASTER";
            f.raster = raster.name;
            f.value = value;
            on_raster(value, raster.time, &f.value_rounded, &f.error);
            return f;
        }

        TimingFinding negative_delay_finding(const char* event, double value)
        {
            TimingFinding f;
            f.event = event;
            f.field = "delay";
            f.error_type = "NEGATIVE_DELAY";
            f.value = value;
            return f;
        }

        /**
         * What one library row contributes: its problems, its delay, and where
         * it ends relative to the start of its block.
         */
        struct EventVerdict
        {
            std::vector<TimingFinding> findings;
            double extent = 0.0;
        };

        void judge(
            std::vector<TimingFinding>& into,
            const char* event,
            double value,
            const char* field,
            const Raster& raster)
        {
            double rounded = 0.0;
            double error = 0.0;
            if (!on_raster(value, raster.time, &rounded, &error))
                into.push_back(raster_finding(event, field, value, raster));
        }

        /**
         * How long a shape-carried event lasts.
         *
         * A time shape gives the end directly; without one the samples sit on
         * the raster, and a time id of -1 marks a waveform oversampled by two.
         */
        double shape_duration(const Sequence& seq, int amp_shape, int time_shape, double raster)
        {
            const ShapeLibrary& shapes = seq.shape_library();
            if (time_shape > 0 && time_shape <= shapes.size())
            {
                const std::vector<double> tt = decompress_shape(
                    shapes.samples(time_shape),
                    shapes.num_compressed(time_shape),
                    shapes.num_uncompressed(time_shape));
                return tt.empty() ? 0.0 : tt.back() * raster;
            }
            if (amp_shape < 1 || amp_shape > shapes.size())
                return 0.0;
            const int samples = shapes.num_uncompressed(amp_shape);
            if (time_shape == -1)
                return (samples + 1) * raster;
            return samples * raster;
        }

        /** Every RF row's verdict, indexed by row id. */
        std::vector<EventVerdict> judge_rf(const Sequence& seq, const Raster& rf_raster)
        {
            const Table& lib = seq.rf_library();
            std::vector<EventVerdict> out(lib.size() + 1);
            for (int id = 1; id <= lib.size(); ++id)
            {
                const double* row = lib.row(id);
                const double delay = row[5];
                EventVerdict& v = out[id];
                if (delay < -kEps)
                    v.findings.push_back(negative_delay_finding("rf", delay));
                judge(v.findings, "rf", delay, "delay", rf_raster);
                v.extent = delay +
                    shape_duration(
                               seq,
                               static_cast<int>(row[1]),
                               static_cast<int>(row[3]),
                               seq.rf_raster_time());
            }
            return out;
        }

        /** Every gradient id's verdict, indexed by gradient id. */
        std::vector<EventVerdict> judge_gradients(const Sequence& seq, const Raster& grad_raster)
        {
            const int count = seq.num_gradients();
            std::vector<EventVerdict> out(count + 1);
            for (int id = 1; id <= count; ++id)
            {
                const GradKind kind = seq.grad_kind(id);
                const int rid = seq.grad_row(id);
                EventVerdict& v = out[id];
                if (kind == GradKind::Trap)
                {
                    const double* row = seq.trap_library().row(rid);
                    const double delay = row[4];
                    if (delay < -kEps)
                        v.findings.push_back(negative_delay_finding("", delay));
                    judge(v.findings, "", delay, "delay", grad_raster);
                    judge(v.findings, "", row[1], "rise_time", grad_raster);
                    judge(v.findings, "", row[2], "flat_time", grad_raster);
                    judge(v.findings, "", row[3], "fall_time", grad_raster);
                    v.extent = delay + row[1] + row[2] + row[3];
                }
                else if (kind == GradKind::Arbitrary)
                {
                    const double* row = seq.arb_library().row(rid);
                    const double delay = row[5];
                    if (delay < -kEps)
                        v.findings.push_back(negative_delay_finding("", delay));
                    judge(v.findings, "", delay, "delay", grad_raster);
                    v.extent = delay +
                        shape_duration(
                                   seq,
                                   static_cast<int>(row[3]),
                                   static_cast<int>(row[4]),
                                   seq.grad_raster_time());
                }
            }
            return out;
        }

        /**
         * Every ADC row's verdict.
         *
         * The start time is judged against the RF raster and the dwell against
         * the ADC raster: the digitiser counts samples on its own clock, but
         * the sequencer has to address the moment it opens.
         */
        std::vector<EventVerdict> judge_adc(
            const Sequence& seq,
            const Raster& rf_raster,
            const Raster& adc_raster,
            double adc_dead_time)
        {
            const Table& lib = seq.adc_library();
            std::vector<EventVerdict> out(lib.size() + 1);
            for (int id = 1; id <= lib.size(); ++id)
            {
                const double* row = lib.row(id);
                const double delay = row[2];
                EventVerdict& v = out[id];
                if (delay < -kEps)
                    v.findings.push_back(negative_delay_finding("adc", delay));
                judge(v.findings, "adc", delay, "delay", rf_raster);
                judge(v.findings, "adc", row[1], "dwell", adc_raster);
                v.extent = delay + row[0] * row[1] + adc_dead_time;
            }
            return out;
        }

    } // namespace

    std::vector<TimingFinding> check_timing(const Sequence& seq, const TimingLimits& limits)
    {
        const Raster rf_raster = {limits.rf_raster_time, "rf_raster_time"};
        const Raster grad_raster = {limits.grad_raster_time, "grad_raster_time"};
        const Raster adc_raster = {limits.adc_raster_time, "adc_raster_time"};
        const Raster block_raster = {limits.block_duration_raster, "block_duration_raster"};

        const std::vector<EventVerdict> rf = judge_rf(seq, rf_raster);
        const std::vector<EventVerdict> grad = judge_gradients(seq, grad_raster);
        const std::vector<EventVerdict> adc =
            judge_adc(seq, rf_raster, adc_raster, limits.adc_dead_time);

        const int32_t* events = seq.block_events();
        const double* durations = seq.block_durations();
        const int blocks = seq.num_blocks();
        const int first_block = limits.first_block > 1 ? limits.first_block : 1;
        const int last_block =
            (limits.last_block > 0 && limits.last_block < blocks) ? limits.last_block : blocks;

        const int delay_ext = seq.find_extension_type_id("DELAYS");
        const std::vector<SoftDelay>& soft = seq.soft_delay_library();
        std::map<int32_t, double> soft_default;

        std::vector<TimingFinding> report;

        for (int b = first_block - 1; b < last_block; ++b)
        {
            const int32_t* row = events + static_cast<size_t>(b) * BLOCK_WIDTH;
            const int number = b + 1;
            const double stored = durations[b];

            const int32_t rf_id =
                (row[0] > 0 && row[0] < static_cast<int32_t>(rf.size())) ? row[0] : 0;
            const int32_t adc_id =
                (row[4] > 0 && row[4] < static_cast<int32_t>(adc.size())) ? row[4] : 0;

            /* calc_duration starts from the stored duration and takes the
             * maximum with each event's end, so a block longer than what it
             * holds is silent and only an event running past it disagrees. */
            double computed = stored;
            if (rf_id)
            {
                const double end = rf[rf_id].extent + limits.rf_ringdown_time;
                if (end > computed)
                    computed = end;
            }
            for (int axis = 0; axis < 3; ++axis)
            {
                const int32_t id = row[1 + axis];
                if (id > 0 && id < static_cast<int32_t>(grad.size()) && grad[id].extent > computed)
                    computed = grad[id].extent;
            }
            if (adc_id && adc[adc_id].extent > computed)
                computed = adc[adc_id].extent;

            const size_t first = report.size();

            double rounded = 0.0;
            double error = 0.0;
            if (!on_raster(computed, block_raster.time, &rounded, &error))
                report.push_back(raster_finding("block", "duration", computed, block_raster));

            if (std::fabs(computed - stored) > kEps)
            {
                TimingFinding f;
                f.event = "block";
                f.field = "duration";
                f.error_type = "BLOCK_DURATION_MISMATCH";
                f.value = computed;
                f.duration = stored;
                report.push_back(f);
            }

            if (rf_id)
                report.insert(report.end(), rf[rf_id].findings.begin(), rf[rf_id].findings.end());
            /* One gradient row can be played on any axis, so which axis a
             * finding belongs to is known here and not in the verdict. */
            for (int axis = 0; axis < 3; ++axis)
            {
                const int32_t id = row[1 + axis];
                if (id <= 0 || id >= static_cast<int32_t>(grad.size()))
                    continue;
                const size_t named = report.size();
                report.insert(report.end(), grad[id].findings.begin(), grad[id].findings.end());
                for (size_t i = named; i < report.size(); ++i)
                    report[i].event = kAxis[axis];
            }
            if (adc_id)
                report.insert(
                    report.end(),
                    adc[adc_id].findings.begin(),
                    adc[adc_id].findings.end());

            if (rf_id)
            {
                const double delay = seq.rf_library().row(rf_id)[5];
                if (delay - limits.rf_dead_time < -kEps)
                {
                    TimingFinding f;
                    f.event = "rf";
                    f.field = "delay";
                    f.error_type = "RF_DEAD_TIME";
                    f.value = delay;
                    f.dead_time = limits.rf_dead_time;
                    report.push_back(f);
                }
                if (rf[rf_id].extent + limits.rf_ringdown_time - stored > kEps)
                {
                    TimingFinding f;
                    f.event = "rf";
                    f.field = "duration";
                    f.error_type = "RF_RINGDOWN_TIME";
                    f.value = rf[rf_id].extent;
                    f.duration = stored;
                    f.ringdown_time = limits.rf_ringdown_time;
                    report.push_back(f);
                }
            }

            if (adc_id)
            {
                const double* arow = seq.adc_library().row(adc_id);
                if (arow[2] - limits.adc_dead_time < -kEps)
                {
                    TimingFinding f;
                    f.event = "adc";
                    f.field = "delay";
                    f.error_type = "ADC_DEAD_TIME";
                    f.value = arow[2];
                    f.dead_time = limits.adc_dead_time;
                    report.push_back(f);
                }
                if (adc[adc_id].extent > stored + kEps)
                {
                    TimingFinding f;
                    f.event = "adc";
                    f.field = "duration";
                    f.error_type = "POST_ADC_DEAD_TIME";
                    f.value = arow[2] + arow[0] * arow[1];
                    f.duration = stored;
                    f.dead_time = limits.adc_dead_time;
                    report.push_back(f);
                }
            }

            if (delay_ext > 0 && row[5] > 0)
            {
                /* A block decodes to at most one soft delay, the last its
                 * extension chain names, so that is the one judged. */
                int32_t chosen = 0;
                int32_t node = row[5];
                while (node > 0 && node <= seq.extensions_library().size())
                {
                    const int32_t* link = seq.extensions_library().row(node);
                    if (link[0] == delay_ext && link[1] >= 1 &&
                        link[1] <= static_cast<int32_t>(soft.size()))
                        chosen = link[1];
                    node = link[2];
                }

                if (chosen)
                {
                    const SoftDelay& sd = soft[chosen - 1];
                    if (sd.factor == 0.0)
                    {
                        TimingFinding f;
                        f.event = "soft_delay";
                        f.field = "delay";
                        f.error_type = "SOFT_DELAY_FACTOR";
                        f.value = sd.factor;
                        f.hint = sd.hint;
                        f.num_id = sd.num;
                        report.push_back(f);
                    }

                    const double def = (stored - sd.offset) * sd.factor;
                    const std::map<int32_t, double>::iterator seen = soft_default.find(sd.num);
                    if (seen == soft_default.end())
                        soft_default[sd.num] = def;
                    else if (std::fabs(def - seen->second) > 1e-7)
                    {
                        TimingFinding f;
                        f.event = "soft_delay";
                        f.field = "delay";
                        f.error_type = "SOFT_DELAY_DUR_INCONSISTENCY";
                        f.value = def;
                        f.hint = sd.hint;
                        f.num_id = sd.num;
                        report.push_back(f);
                    }
                }
            }

            for (size_t i = first; i < report.size(); ++i)
                report[i].block = number;
        }

        return report;
    }

} // namespace pulseq
