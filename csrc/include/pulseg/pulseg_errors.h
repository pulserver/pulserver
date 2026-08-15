/**
 * @file pulseg_errors.h
 * @brief Negative status codes returned across the pulseg API.
 *
 * Consumers should branch on PULSEG_FAILED() / PULSEG_SUCCEEDED() and surface
 * the diagnostic message string rather than matching individual values -- the
 * specific numbers are not a stability promise.  They live in a public header
 * because public entry points return them, and because modules layered on the
 * * core need to produce them without reaching into the
 * library's private headers.
 */

#ifndef PULSEG_ERRORS_H
#define PULSEG_ERRORS_H

/* Generic errors (-1 to -9) */
#define PULSEG_ERR_NULL_POINTER -1
#define PULSEG_ERR_INVALID_ARGUMENT -2
#define PULSEG_ERR_ALLOC_FAILED -3

/* Parsing / file errors (-10 to -19) */
#define PULSEG_ERR_FILE_NOT_FOUND -10
#define PULSEG_ERR_FILE_READ_FAILED -11
#define PULSEG_ERR_UNSUPPORTED_VERSION -12

/* Unique-block errors (-50 to -59) */
#define PULSEG_ERR_RASTER_MISMATCH -53
#define PULSEG_ERR_SIGNATURE_MISMATCH -54
#define PULSEG_ERR_SIGNATURE_MISSING -55
#define PULSEG_ERR_ADC_DEFINITION_CONFLICT -56
#define PULSEG_ERR_INDEX -57
#define PULSEG_ERR_MISSING_KSPACE_ANCHOR -58

/* TR detection errors (-100 to -199) */
#define PULSEG_ERR_TR_NO_BLOCKS -100
#define PULSEG_ERR_TR_NO_IMAGING_REGION -101
#define PULSEG_ERR_TR_NO_PERIODIC_PATTERN -102
#define PULSEG_ERR_TR_PATTERN_MISMATCH -103
#define PULSEG_ERR_TR_PREP_TOO_LONG -104
#define PULSEG_ERR_TR_COOLDOWN_TOO_LONG -105

/* Segmentation errors (-200 to -299) */
#define PULSEG_ERR_SEG_NONZERO_START_GRAD -200
#define PULSEG_ERR_SEG_NONZERO_END_GRAD -201
#define PULSEG_ERR_SEG_NO_SEGMENTS_FOUND -202
#define PULSEG_ERR_TOO_MANY_GRAD_SHOTS -203
#define PULSEG_ERR_SEG_MULTIPLE_PHYSIO_TRIGGERS -204
#define PULSEG_ERR_SEG_MULTIPLE_NAV_SEGMENTS -205
#define PULSEG_ERR_SEG_ROTATION_MID_GRADIENT -206

/* Chunk planning errors (-250 to -259) */
#define PULSEG_ERR_CHUNK_INFEASIBLE -250

/* Mechanical resonance errors (-400 to -449) */
#define PULSEG_ERR_MECH_RESONANCES_NO_WAVEFORM -402
#define PULSEG_ERR_MECH_RESONANCES_VIOLATION -404

/* PNS errors (-450 to -499) */
#define PULSEG_ERR_PNS_INVALID_PARAMS -450
#define PULSEG_ERR_PNS_INVALID_CHRONAXIE -451
#define PULSEG_ERR_PNS_INVALID_RHEOBASE -452
#define PULSEG_ERR_PNS_NO_WAVEFORM -453
#define PULSEG_ERR_PNS_FFT_FAILED -454
#define PULSEG_ERR_PNS_THRESHOLD_EXCEEDED -455

/* Collection / safety errors (-500 to -559) */
#define PULSEG_ERR_COLLECTION_EMPTY -500
#define PULSEG_ERR_TRID_STRUCTURAL_MISMATCH -502
#define PULSEG_ERR_MAX_GRAD_EXCEEDED -550
#define PULSEG_ERR_GRAD_DISCONTINUITY -551
#define PULSEG_ERR_MAX_SLEW_EXCEEDED -552

/* Consistency errors (-560 to -569) */
#define PULSEG_ERR_CONSISTENCY_SEG_MISMATCH -560
#define PULSEG_ERR_CONSISTENCY_RF_PERIODIC -561
#define PULSEG_ERR_CONSISTENCY_RF_SHIM_PERIODIC -562

/* Sentinel */
#define PULSEG_ERR_NOT_IMPLEMENTED -999

#endif /* PULSEG_ERRORS_H */
