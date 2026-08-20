/**
 * @file pulseg_error.c
 * @brief Error-code messages, fix hints, and diagnostic formatting.
 *
 * pulseg_format_error() is what a consumer surfaces to the user: the message
 * for the code, the hint for how to fix it, and whatever context the failing
 * pass wrote into the diagnostic.
 */

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "pulseg_internal.h"

/* ================================================================== */
/*  Diagnostic init                                                   */
/* ================================================================== */
void pulseg_diagnostic_init(pulseg_diagnostic *diag)
{
    if (!diag)
        return;
    diag->code = PULSEG_SUCCESS;
    diag->message[0] = '\0';
}

void pulseg__diag_printf(pulseg_diagnostic *diag, const char *fmt, ...)
{
    va_list ap;
    int used;
    if (!diag)
        return;
    /* Find current length so successive calls append */
    used = (int)strlen(diag->message);
    if (used >= PULSEG_DIAG_MSG_LEN - 1)
        return;
    va_start(ap, fmt);
    vsprintf(diag->message + used, fmt, ap);
    va_end(ap);
    diag->message[PULSEG_DIAG_MSG_LEN - 1] = '\0';
}

/* ================================================================== */
/*  Error messages                                                     */
/* ================================================================== */
const char *pulseg_get_error_message(int code)
{
    switch (code)
    {
    case PULSEG_SUCCESS:
        return "Success";
    case PULSEG_ERR_NULL_POINTER:
        return "Required pointer argument is NULL";
    case PULSEG_ERR_INVALID_ARGUMENT:
        return "Invalid argument value";
    case PULSEG_ERR_ALLOC_FAILED:
        return "Memory allocation failed";
    case PULSEG_ERR_FILE_NOT_FOUND:
        return "Sequence file not found or could not be opened";
    case PULSEG_ERR_FILE_READ_FAILED:
        return "Error reading from sequence file";
    case PULSEG_ERR_UNSUPPORTED_VERSION:
        return "Unsupported sequence file version (requires >= 1.5.0)";
    case PULSEG_ERR_RASTER_MISMATCH:
        return "System and sequence raster times are not integer multiples";
    case PULSEG_ERR_SIGNATURE_MISMATCH:
        return "MD5 signature verification failed";
    case PULSEG_ERR_SIGNATURE_MISSING:
        return "Sequence file has no [SIGNATURE] section or stored hash";
    case PULSEG_ERR_ADC_DEFINITION_CONFLICT:
        return "A non-acquiring block matches more than one ADC definition";
    case PULSEG_ERR_RASTER_ALIGNMENT:
        return "An event time does not land on the system raster it is played against";
    case PULSEG_ERR_BLOCK_DURATION_OVERRUN:
        return "A block event extends past the end of its block";
    case PULSEG_ERR_INDEX:
        return "Index out of range";
    case PULSEG_ERR_TOO_MANY_GRAD_SHOTS:
        return "Number of waveform shots exceeds maximum allowed";
    case PULSEG_ERR_TR_NO_BLOCKS:
        return "Sequence contains no blocks";
    case PULSEG_ERR_TR_NO_IMAGING_REGION:
        return "No imaging region found (preparation + cooldown >= total blocks)";
    case PULSEG_ERR_TR_NO_PERIODIC_PATTERN:
        return "No periodic TR pattern found in imaging region";
    case PULSEG_ERR_TR_PATTERN_MISMATCH:
        return "TR pattern does not repeat consistently across imaging region";
    case PULSEG_ERR_TR_PREP_TOO_LONG:
        return "Non-standard preparation section exceeds duration threshold";
    case PULSEG_ERR_TR_COOLDOWN_TOO_LONG:
        return "Non-standard cooldown section exceeds duration threshold";
    case PULSEG_ERR_SEG_NONZERO_START_GRAD:
        return "TR does not start with zero gradient amplitude";
    case PULSEG_ERR_SEG_NONZERO_END_GRAD:
        return "TR does not end with zero gradient amplitude";
    case PULSEG_ERR_SEG_NO_SEGMENTS_FOUND:
        return "No segment boundaries could be identified in TR";
    case PULSEG_ERR_CHUNK_INFEASIBLE:
        return "No chunk split satisfies both the waveform memory budget and the playout rate";
    case PULSEG_ERR_SEG_ROTATION_MID_GRADIENT:
        return "Rotation state changes across a live gradient (no zero-gradient junction)";
    case PULSEG_ERR_MECH_RESONANCES_NO_WAVEFORM:
        return "No waveform data for mechanical resonance analysis";
    case PULSEG_ERR_MECH_RESONANCES_VIOLATION:
        return "mech-res exceeded";
    case PULSEG_ERR_PNS_INVALID_PARAMS:
        return "Invalid PNS parameters";
    case PULSEG_ERR_PNS_INVALID_CHRONAXIE:
        return "Invalid chronaxie value for PNS";
    case PULSEG_ERR_PNS_INVALID_RHEOBASE:
        return "Invalid rheobase value for PNS";
    case PULSEG_ERR_PNS_NO_WAVEFORM:
        return "No waveform data for PNS analysis";
    case PULSEG_ERR_PNS_FFT_FAILED:
        return "FFT convolution failed during PNS analysis";
    case PULSEG_ERR_PNS_THRESHOLD_EXCEEDED:
        return "PNS threshold exceeded";
    case PULSEG_ERR_MAX_GRAD_EXCEEDED:
        return "gmax exceeded";
    case PULSEG_ERR_NOT_IMPLEMENTED:
        return "Functionality not yet implemented";
    case PULSEG_ERR_GRAD_DISCONTINUITY:
        return "gstep exceeded";
    case PULSEG_ERR_MAX_SLEW_EXCEEDED:
        return "gslew exceeded";
    case PULSEG_ERR_CONSISTENCY_SEG_MISMATCH:
        return "Block definition IDs do not match segment table";
    case PULSEG_ERR_CONSISTENCY_RF_PERIODIC:
        return "RF amplitude pattern is not periodic across canonical TRs";
    case PULSEG_ERR_CONSISTENCY_RF_SHIM_PERIODIC:
        return "RF shim ID pattern is not periodic across canonical TRs";
    default:
        return "Unknown error";
    }
}

/* ================================================================== */
/*  Error hints                                                       */
/* ================================================================== */
const char *pulseg_get_error_hint(int code)
{
    switch (code)
    {
    case PULSEG_SUCCESS:
        return "";
    case PULSEG_ERR_RASTER_MISMATCH:
        return "System raster times and sequence-defined raster times must be "
               "integer multiples of each other for piecewise-constant "
               "interpolation to produce correct waveforms.";
    case PULSEG_ERR_SIGNATURE_MISMATCH:
        return "The .seq file content does not match its stored MD5 hash. "
               "The file may have been modified after export.";
    case PULSEG_ERR_SIGNATURE_MISSING:
        return "The .seq file does not contain a [SIGNATURE] section. "
               "Re-export the sequence to include a signature.";
    case PULSEG_ERR_TR_NO_IMAGING_REGION:
        return "Make sure to use ONCE flags either at beginning (preparation) or end (cooldown) of "
               "the sequence.";
    case PULSEG_ERR_TR_NO_PERIODIC_PATTERN:
        return "This often occurs when phase-encoding gradients are created inside "
               "the sequence loop with varying amplitudes. Instead, create gradient "
               "events ONCE outside the loop and use 'scale' parameter to vary amplitude.";
    case PULSEG_ERR_SEG_NONZERO_START_GRAD:
    case PULSEG_ERR_SEG_NONZERO_END_GRAD:
        return "Each segment must begin and end with gradient amplitudes that can "
               "ramp to/from zero within one gradient raster.";
    case PULSEG_ERR_ADC_DEFINITION_CONFLICT:
        return "A block position holds instances that do not acquire alongside two or "
               "more different ADC events (different num_samples, dwell time, or "
               "delay), so which readout the non-acquiring instances stand in for is "
               "not written anywhere. Give those instances the ADC they belong to, or "
               "keep one ADC structure per block.";
    case PULSEG_ERR_TOO_MANY_GRAD_SHOTS:
        return "The sequence contains a waveform with more than 16 distinct waveform shapes.";
    case PULSEG_ERR_TR_PREP_TOO_LONG:
    case PULSEG_ERR_TR_COOLDOWN_TOO_LONG:
        return "The preparation or cooldown section differs from the main TR pattern and is too "
               "long.";
    case PULSEG_ERR_MAX_GRAD_EXCEEDED:
        return "The gradient sum-of-squares amplitude exceeds the system limit. "
               "See diagnostic message for details.";
    case PULSEG_ERR_NOT_IMPLEMENTED:
        return "This functionality is not yet implemented.";
    case PULSEG_ERR_GRAD_DISCONTINUITY:
        return "The gradient amplitude step between consecutive blocks exceeds "
               "the maximum allowed by the slew rate and raster time. "
               "See diagnostic message for details.";
    case PULSEG_ERR_RASTER_ALIGNMENT:
        return "Every delay, duration, ramp and dwell must be an integer multiple of "
               "the scanner raster it is played against: RF and ADC start times on the "
               "RF raster, gradient times on the gradient raster, ADC dwell on the ADC "
               "raster, block durations on the block duration raster. "
               "See diagnostic message for the event and the offending value.";
    case PULSEG_ERR_BLOCK_DURATION_OVERRUN:
        return "An RF, gradient or ADC event runs past the end of the block that "
               "holds it. A block longer than its events is fine; a block shorter "
               "than them is not playable. See diagnostic message for details.";
    case PULSEG_ERR_MAX_SLEW_EXCEEDED:
        return "A gradient waveform exceeds the per-axis slew rate limit "
               "(derated by 1/sqrt(3) for rotation safety). "
               "See diagnostic message for details.";
    case PULSEG_ERR_CONSISTENCY_SEG_MISMATCH:
        return "Segment definitions are inconsistent with block table. "
               "Try scaling gradients to eps instead of zero to preserve "
               "gradient continuity across segment boundaries";
    case PULSEG_ERR_CONSISTENCY_RF_PERIODIC:
        return "RF amplitudes differ between canonical TR instances that should "
               "be identical. If variable flip angles across canonical units are intended, "
               "split volumes into separate subsequences";
    case PULSEG_ERR_CONSISTENCY_RF_SHIM_PERIODIC:
        return "RF shim IDs differ between canonical TR instances that should "
               "be identical. If variable shimming across canonical units is intended, "
               "split volumes into separate subsequences";
    default:
        return "Check sequence design for structural consistency.";
    }
}
