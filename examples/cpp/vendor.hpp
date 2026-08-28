/**
 * @file vendor.hpp
 * @brief The scanner side of an integration, stubbed out.
 *
 * The C++ counterpart of examples/c/vendor.h. Each stub prints what a real
 * implementation would have done. A vendor integration replaces this file.
 */

#ifndef EXAMPLE_VENDOR_HPP
#define EXAMPLE_VENDOR_HPP

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "pulseg/pulseg.hpp"

namespace vendor
{
    /**
     * The hardware's limits.
     *
     * The gradient and slew ceilings are per axis and already derated by
     * sqrt(3), so no physical axis exceeds the amplifier under an arbitrary
     * rotation. The rasters are the scanner's, not the ones a .seq declares.
     */
    inline pulseg::Opts system_limits()
    {
        pulseg::Opts opts;
        opts.gamma_hz_per_t = 42.576e6f;
        opts.b0_t = 3.0f;
        opts.max_grad_hz_per_m = opts.gamma_hz_per_t * 0.080f / 1.732f;
        opts.max_slew_hz_per_m_per_s = opts.gamma_hz_per_t * 250.0f / 1.732f;
        opts.rf_raster_us = 2.0f;
        opts.grad_raster_us = 4.0f;
        opts.adc_raster_us = 2.0f;
        opts.block_raster_us = 4.0f;
        return opts;
    }

    /** The acoustic bands this magnet resonates in. */
    inline std::vector<pulseg::ForbiddenBand> forbidden_bands()
    {
        std::vector<pulseg::ForbiddenBand> bands(2);
        bands[0].freq_min_hz = 550.0f;
        bands[0].freq_max_hz = 620.0f;
        bands[0].max_amplitude_hz_per_m = 0.0f;
        bands[1].freq_min_hz = 1180.0f;
        bands[1].freq_max_hz = 1260.0f;
        bands[1].max_amplitude_hz_per_m = 0.0f;
        return bands;
    }

    /** The scanner's log. */
    inline void log(const std::string& message)
    {
        std::cout << message << '\n';
    }

    /** One control on the console. */
    inline void ui_declare(const std::string& wire_name, int type)
    {
        static const char* type_names[] = {"float", "int", "bool", "list", "text", "config"};
        const char* name = (type >= 0 && type <= 5) ? type_names[type] : "?";
        std::printf("    ui: %-24s (%s)\n", wire_name.c_str(), name);
    }

    /** Where the console shows the scan duration. */
    inline void ui_scan_time(double seconds)
    {
        std::printf("    ui: scan time %d:%02d\n", int(seconds) / 60, int(seconds) % 60);
    }

    /** How many samples the gradient waveform memory holds. */
    inline long waveform_memory_samples()
    {
        return 512L * 1024L;
    }

    /** Upload one waveform; returns the hardware's id for it. */
    inline int load_waveform(int axis, const std::vector<float>& amplitudes)
    {
        static int next_id = 0;
        std::printf(
            "    upload: axis %d, %d points -> hardware id %d\n",
            axis,
            int(amplitudes.size()),
            next_id);
        return next_id++;
    }

    /** Point the sequencer at a prepared segment. */
    inline void begin_segment(int segment_id)
    {
        std::printf("    segment %d\n", segment_id);
    }

    /** Play one block. */
    inline void play_block(const pulseg::BlockInstance& block)
    {
        std::printf(
            "      %6d us  rf %8.1f Hz  g (%9.1f %9.1f %9.1f) Hz/m%s\n",
            block.duration_us,
            double(block.rf_amp_hz),
            double(block.gx_amp),
            double(block.gy_amp),
            double(block.gz_amp),
            block.adc_flag ? "  [acquire]" : "");
    }
} // namespace vendor

#endif // EXAMPLE_VENDOR_HPP
