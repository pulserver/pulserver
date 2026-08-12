/**
 * @file types.hpp
 * @brief C++11 value types and enums wrapping pulseq C structs/macros.
 *
 * pulseq_types.h represents every pseudo-enum as a set of `#define`d ints
 * (an EPIC/C89 constraint: csrc/ is shared, via a symlink farm, into the
 * GE PSD build -- see src_psd/pulseglib-src/ -- so it must stay compilable
 * as plain ANSI C). This header re-expresses them as real, scoped C++
 * enums for consumers of the C++ frontend; the underlying values are kept
 * identical to the macros so `static_cast<int>` round-trips into the C API
 * exactly.
 */

#ifndef PULSEQ_CXX_TYPES_HPP
#define PULSEQ_CXX_TYPES_HPP

#include <vector>

#include "pulseq_types.h"

namespace pulseq
{

    /** Decompressed SHAPES library entry (owning; see decompress_shape()). */
    struct Shape
    {
        int num_uncompressed_samples = 0;
        std::vector<float> samples;
    };

    /** Design-time raster times (us). See pulseq_raster. */
    struct Raster
    {
        float rf_us = 0.0f;
        float grad_us = 0.0f;
        float block_us = 0.0f;

        pulseq_raster to_c() const
        {
            pulseq_raster r;
            r.rf_us = rf_us;
            r.grad_us = grad_us;
            r.block_us = block_us;
            return r;
        }

        static Raster from_c(const pulseq_raster& r)
        {
            Raster out;
            out.rf_us = r.rf_us;
            out.grad_us = r.grad_us;
            out.block_us = r.block_us;
            return out;
        }
    };

    /** RF use tag (the trailing e/r/i/s code on an RF library row). */
    enum class RfUse : int
    {
        Unknown = PULSEQ_RF_USE_UNKNOWN,
        Excitation = PULSEQ_RF_USE_EXCITATION,
        Refocusing = PULSEQ_RF_USE_REFOCUSING,
        Inversion = PULSEQ_RF_USE_INVERSION,
        Saturation = PULSEQ_RF_USE_SATURATION,
    };

    /** Block extension-chain entry type. */
    enum class ExtensionType : int
    {
        List = PULSEQ_EXT_LIST,
        Trigger = PULSEQ_EXT_TRIGGER,
        Rotation = PULSEQ_EXT_ROTATION,
        LabelSet = PULSEQ_EXT_LABELSET,
        LabelInc = PULSEQ_EXT_LABELINC,
        RfShim = PULSEQ_EXT_RF_SHIM,
        Delay = PULSEQ_EXT_DELAY,
        Unknown = PULSEQ_EXT_UNKNOWN,
    };

    /** Numeric id for a LABELSET/LABELINC name (see label_id_for_name()). */
    enum class LabelId : int
    {
        Slc = PULSEQ_LABEL_SLC,
        Seg = PULSEQ_LABEL_SEG,
        Rep = PULSEQ_LABEL_REP,
        Avg = PULSEQ_LABEL_AVG,
        Set = PULSEQ_LABEL_SET,
        Eco = PULSEQ_LABEL_ECO,
        Phs = PULSEQ_LABEL_PHS,
        Lin = PULSEQ_LABEL_LIN,
        Par = PULSEQ_LABEL_PAR,
        Acq = PULSEQ_LABEL_ACQ,
        Nav = PULSEQ_LABEL_NAV,
        Rev = PULSEQ_LABEL_REV,
        Sms = PULSEQ_LABEL_SMS,
        Ref = PULSEQ_LABEL_REF,
        Ima = PULSEQ_LABEL_IMA,
        Noise = PULSEQ_LABEL_NOISE,
        Pmc = PULSEQ_LABEL_PMC,
        NoRot = PULSEQ_LABEL_NOROT,
        NoPos = PULSEQ_LABEL_NOPOS,
        NoScl = PULSEQ_LABEL_NOSCL,
        Once = PULSEQ_LABEL_ONCE,
        Trid = PULSEQ_LABEL_TRID,
        Off = PULSEQ_LABEL_OFF,
    };

    /** Numeric id for a soft-delay hint name (see hint_id_for_name()). */
    enum class HintId : int
    {
        Te = PULSEQ_HINT_TE,
        Tr = PULSEQ_HINT_TR,
        Ti = PULSEQ_HINT_TI,
        Esp = PULSEQ_HINT_ESP,
        RecTime = PULSEQ_HINT_RECTIME,
        T2Prep = PULSEQ_HINT_T2PREP,
        Te2 = PULSEQ_HINT_TE2,
    };

    /** Gradient library row discriminator. */
    enum class GradientType : int
    {
        Trapezoid = PULSEQ_GRAD_TRAP,
        Arbitrary = PULSEQ_GRAD_ARB,
    };

    /** Digital trigger / physio event type. */
    enum class TriggerType : int
    {
        Output = PULSEQ_TRIGGER_TYPE_OUTPUT, ///< TTL / digital output
        Input = PULSEQ_TRIGGER_TYPE_INPUT,   ///< ECG / cardiac gating
    };

    /** Trigger channel for TriggerType::Input events. */
    enum class TriggerChannelInput : int
    {
        Physio1 = PULSEQ_TRIGGER_CHANNEL_INPUT_PHYSIO_1,
        Physio2 = PULSEQ_TRIGGER_CHANNEL_INPUT_PHYSIO_2,
    };

    /** Trigger channel for TriggerType::Output events. */
    enum class TriggerChannelOutput : int
    {
        Osc0 = PULSEQ_TRIGGER_CHANNEL_OUTPUT_OSC_0,
        Osc1 = PULSEQ_TRIGGER_CHANNEL_OUTPUT_OSC_1,
        Ext1 = PULSEQ_TRIGGER_CHANNEL_OUTPUT_EXT_1,
    };

} // namespace pulseq

#endif // PULSEQ_CXX_TYPES_HPP
