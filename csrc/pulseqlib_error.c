/* pulseqlib_error.c -- error messages, hints, and lookup tables */

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "pulseqlib_internal.h"

/* ------------------------------------------------------------------ */
/*  Label / hint lookup tables                                        */
/* ------------------------------------------------------------------ */

static const pulseqlib__table_entry label_table[] = {
    { "SLC",   PULSEQLIB__SLC },
    { "SEG",   PULSEQLIB__SEG },
    { "REP",   PULSEQLIB__REP },
    { "AVG",   PULSEQLIB__AVG },
    { "SET",   PULSEQLIB__SET },
    { "ECO",   PULSEQLIB__ECO },
    { "PHS",   PULSEQLIB__PHS },
    { "LIN",   PULSEQLIB__LIN },
    { "PAR",   PULSEQLIB__PAR },
    { "ACQ",   PULSEQLIB__ACQ },
    { "NAV",   PULSEQLIB__NAV },
    { "REV",   PULSEQLIB__REV },
    { "SMS",   PULSEQLIB__SMS },
    { "REF",   PULSEQLIB__REF },
    { "IMA",   PULSEQLIB__IMA },
    { "NOISE", PULSEQLIB__NOISE },
    { "PMC",   PULSEQLIB__PMC },
    { "NOROT", PULSEQLIB__NOROT },
    { "NOPOS", PULSEQLIB__NOPOS },
    { "NOSCL", PULSEQLIB__NOSCL },
    { "ONCE",  PULSEQLIB__ONCE },
    { "TRID",  PULSEQLIB__TRID },
    { NULL, -1 }
};

int pulseqlib__label2enum(const char *label)
{
    int i;
    if (!label) return -1;
    for (i = 0; label_table[i].name != NULL; i++) {
        if (strcmp(label, label_table[i].name) == 0)
            return label_table[i].value;
    }
    return -1;
}

static const pulseqlib__table_entry hint_table[] = {
    { "TE",      PULSEQLIB__HINT_TE },
    { "TR",      PULSEQLIB__HINT_TR },
    { "TI",      PULSEQLIB__HINT_TI },
    { "ESP",     PULSEQLIB__HINT_ESP },
    { "RECTIME", PULSEQLIB__HINT_RECTIME },
    { "T2PREP",  PULSEQLIB__HINT_T2PREP },
    { "TE2",     PULSEQLIB__HINT_TE2 },
    { NULL, -1 }
};

int pulseqlib__hint2enum(const char *hint)
{
    int i;
    if (!hint) return -1;
    for (i = 0; hint_table[i].name != NULL; i++) {
        if (strcmp(hint, hint_table[i].name) == 0)
            return hint_table[i].value;
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/*  Diagnostic init                                                   */
/* ------------------------------------------------------------------ */
void pulseqlib_diagnostic_init(pulseqlib_diagnostic* diag)
{
    if (!diag) return;
    diag->code = PULSEQLIB_SUCCESS;
    diag->message[0] = '\0';
}

void pulseqlib__diag_printf(pulseqlib_diagnostic* diag, const char* fmt, ...)
{
    va_list ap;
    int used;
    if (!diag) return;
    /* Find current length so successive calls append */
    used = (int)strlen(diag->message);
    if (used >= PULSEQLIB_DIAG_MSG_LEN - 1) return;
    va_start(ap, fmt);
    vsprintf(diag->message + used, fmt, ap);
    va_end(ap);
    diag->message[PULSEQLIB_DIAG_MSG_LEN - 1] = '\0';
}

/* ------------------------------------------------------------------ */
/*  Error messages                                                     */
/* ------------------------------------------------------------------ */
const char* pulseqlib_get_error_message(int code)
{
    switch (code) {
        case PULSEQLIB_SUCCESS:                            return "Success";
        case PULSEQLIB_ERR_NULL_POINTER:              return "Required pointer argument is NULL";
        case PULSEQLIB_ERR_INVALID_ARGUMENT:          return "Invalid argument value";
        case PULSEQLIB_ERR_ALLOC_FAILED:              return "Memory allocation failed";
        case PULSEQLIB_ERR_FILE_NOT_FOUND:            return "Sequence file not found or could not be opened";
        case PULSEQLIB_ERR_FILE_READ_FAILED:          return "Error reading from sequence file";
        case PULSEQLIB_ERR_UNSUPPORTED_VERSION:       return "Unsupported sequence file version (requires >= 1.5.0)";
        case PULSEQLIB_ERR_INVALID_PREP_POSITION:     return "Invalid preparation block position";
        case PULSEQLIB_ERR_INVALID_COOLDOWN_POSITION: return "Invalid cooldown block position";
        case PULSEQLIB_ERR_INVALID_ONCE_FLAGS:        return "ONCE flags define non-identical inner-loop repetitions";
        case PULSEQLIB_ERR_RASTER_MISMATCH:            return "System and sequence raster times are not integer multiples";
        case PULSEQLIB_ERR_SIGNATURE_MISMATCH:         return "MD5 signature verification failed";
        case PULSEQLIB_ERR_SIGNATURE_MISSING:          return "Sequence file has no [SIGNATURE] section or stored hash";
        case PULSEQLIB_ERR_ADC_DEFINITION_CONFLICT:    return "Block definition has conflicting ADC definitions across instances";
        case PULSEQLIB_ERR_INDEX:                      return "Index out of range";
        case PULSEQLIB_ERR_TOO_MANY_GRAD_SHOTS:       return "Number of waveform shots exceeds maximum allowed";
        case PULSEQLIB_ERR_TR_NO_BLOCKS:              return "Sequence contains no blocks";
        case PULSEQLIB_ERR_TR_NO_IMAGING_REGION:      return "No imaging region found (preparation + cooldown >= total blocks)";
        case PULSEQLIB_ERR_TR_NO_PERIODIC_PATTERN:    return "No periodic TR pattern found in imaging region";
        case PULSEQLIB_ERR_TR_PATTERN_MISMATCH:       return "TR pattern does not repeat consistently across imaging region";
        case PULSEQLIB_ERR_TR_PREP_TOO_LONG:          return "Non-standard preparation section exceeds duration threshold";
        case PULSEQLIB_ERR_TR_COOLDOWN_TOO_LONG:      return "Non-standard cooldown section exceeds duration threshold";
        case PULSEQLIB_ERR_SEG_NONZERO_START_GRAD:    return "TR does not start with zero gradient amplitude";
        case PULSEQLIB_ERR_SEG_NONZERO_END_GRAD:      return "TR does not end with zero gradient amplitude";
        case PULSEQLIB_ERR_SEG_NO_SEGMENTS_FOUND:     return "No segment boundaries could be identified in TR";
        case PULSEQLIB_ERR_ACOUSTIC_NO_WAVEFORM:      return "No waveform data for acoustic analysis";
        case PULSEQLIB_ERR_ACOUSTIC_VIOLATION:        return "Acoustic resonance violation detected";
        case PULSEQLIB_ERR_PNS_INVALID_PARAMS:        return "Invalid PNS parameters";
        case PULSEQLIB_ERR_PNS_INVALID_CHRONAXIE:     return "Invalid chronaxie value for PNS";
        case PULSEQLIB_ERR_PNS_INVALID_RHEOBASE:      return "Invalid rheobase value for PNS (GE model)";
        case PULSEQLIB_ERR_PNS_NO_WAVEFORM:           return "No waveform data for PNS analysis";
        case PULSEQLIB_ERR_PNS_FFT_FAILED:            return "FFT convolution failed during PNS analysis";
        case PULSEQLIB_ERR_PNS_THRESHOLD_EXCEEDED:    return "PNS threshold exceeded";
        case PULSEQLIB_ERR_MAX_GRAD_EXCEEDED:         return "Maximum gradient amplitude exceeded";
        case PULSEQLIB_ERR_NOT_IMPLEMENTED:           return "Functionality not yet implemented";
        case PULSEQLIB_ERR_GRAD_DISCONTINUITY:        return "Gradient discontinuity between blocks";
        case PULSEQLIB_ERR_MAX_SLEW_EXCEEDED:         return "Maximum slew rate exceeded";
        case PULSEQLIB_ERR_CONSISTENCY_SEG_MISMATCH:  return "Block definition IDs do not match segment table";
        case PULSEQLIB_ERR_CONSISTENCY_RF_PERIODIC:   return "RF amplitude pattern is not periodic across main TRs";
        case PULSEQLIB_ERR_CONSISTENCY_RF_SHIM_PERIODIC: return "RF shim ID pattern is not periodic across main TRs";
        default:                                       return "Unknown error";
    }
}

/* ------------------------------------------------------------------ */
/*  Error hints                                                       */
/* ------------------------------------------------------------------ */
const char* pulseqlib_get_error_hint(int code)
{
    switch (code) {
        case PULSEQLIB_SUCCESS:
            return "";
        case PULSEQLIB_ERR_INVALID_PREP_POSITION:
            return "Ensure that the preparation section is marked with ONCE labels "
                   "and starts at the first block of the sequence.";
        case PULSEQLIB_ERR_INVALID_COOLDOWN_POSITION:
            return "Ensure that the cooldown section is marked with ONCE labels "
                   "and ends at the last block of the sequence.";
        case PULSEQLIB_ERR_INVALID_ONCE_FLAGS:
            return "Ensure that ONCE flags define structurally identical inner-loop "
                   "repetitions. All periods delimited by ONCE sections must have the "
                   "same block structure.";
        case PULSEQLIB_ERR_RASTER_MISMATCH:
            return "System raster times and sequence-defined raster times must be "
                   "integer multiples of each other for piecewise-constant "
                   "interpolation to produce correct waveforms.";
        case PULSEQLIB_ERR_SIGNATURE_MISMATCH:
            return "The .seq file content does not match its stored MD5 hash. "
                   "The file may have been modified after export.";
        case PULSEQLIB_ERR_SIGNATURE_MISSING:
            return "The .seq file does not contain a [SIGNATURE] section. "
                   "Re-export the sequence to include a signature.";
        case PULSEQLIB_ERR_TR_NO_IMAGING_REGION:
            return "Make sure to use ONCE flags either at beginning (preparation) or end (cooldown) of the sequence.";
        case PULSEQLIB_ERR_TR_NO_PERIODIC_PATTERN:
            return "This often occurs when phase-encoding gradients are created inside "
                   "the sequence loop with varying amplitudes. Instead, create gradient "
                   "events ONCE outside the loop and use 'scale' parameter to vary amplitude.";
        case PULSEQLIB_ERR_SEG_NONZERO_START_GRAD:
        case PULSEQLIB_ERR_SEG_NONZERO_END_GRAD:
            return "Each segment must begin and end with gradient amplitudes that can "
                   "ramp to/from zero within one gradient raster.";
        case PULSEQLIB_ERR_ADC_DEFINITION_CONFLICT:
            return "Two instances of the same block definition use different ADC events "
                   "(different num_samples, dwell time, or delay). Ensure all instances "
                   "of a given block use the same ADC structure, or omit the ADC entirely.";
        case PULSEQLIB_ERR_TOO_MANY_GRAD_SHOTS:
            return "The sequence contains a waveform with more than 16 distinct waveform shapes.";
        case PULSEQLIB_ERR_TR_PREP_TOO_LONG:
        case PULSEQLIB_ERR_TR_COOLDOWN_TOO_LONG:
            return "The preparation or cooldown section differs from the main TR pattern and is too long.";
        case PULSEQLIB_ERR_MAX_GRAD_EXCEEDED:
            return "The gradient sum-of-squares amplitude exceeds the system limit. "
                   "See diagnostic message for details.";
        case PULSEQLIB_ERR_NOT_IMPLEMENTED:
            return "This functionality is not yet implemented.";
        case PULSEQLIB_ERR_GRAD_DISCONTINUITY:
            return "The gradient amplitude step between consecutive blocks exceeds "
                   "the maximum allowed by the slew rate and raster time. "
                   "See diagnostic message for details.";
        case PULSEQLIB_ERR_MAX_SLEW_EXCEEDED:
            return "A gradient waveform exceeds the per-axis slew rate limit "
                   "(derated by 1/sqrt(3) for rotation safety). "
                   "See diagnostic message for details.";
        case PULSEQLIB_ERR_CONSISTENCY_SEG_MISMATCH:
            return "Segment definitions are inconsistent with block table. "
                "Try scaling gradients to eps instead of zero to preserve "
                "gradient continuity across segment boundaries";
        case PULSEQLIB_ERR_CONSISTENCY_RF_PERIODIC:
            return "RF amplitudes differ between main TR instances that should "
                "be identical. If variable flip angles across TRs are intended, "
                "split volumes into separate subsequences";
        case PULSEQLIB_ERR_CONSISTENCY_RF_SHIM_PERIODIC:
            return "RF shim IDs differ between main TR instances that should "
                "be identical. If variable shimming across TRs is intended, "
                "split volumes into separate subsequences";
        default:
            return "Check sequence design for structural consistency.";
    }
}