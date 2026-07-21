/**
 * @file pulseq_config.h
 * @brief Platform configuration for the standalone Pulseq module.
 *
 * This header MUST be included (directly or transitively) before any other
 * pulseq header or source file.  It provides:
 *
 *   PULSEQ_ALLOC / PULSEQ_FREE  -- heap allocator hooks (default malloc/free)
 *   PULSEQ_ERR_*                -- error codes returned by the reader
 *
 * The pulseq module is standalone: it pulls in NO pulseg header.  A host
 * library layering on top of it (e.g. pulseg) binds the allocator hooks to
 * its own arena by defining them before including this file.
 */

#ifndef PULSEQ_CONFIG_H
#define PULSEQ_CONFIG_H

/* Suppress -Wfloat-equal for intentional exact float comparisons (shape
 * decompression RLE, zero-detection).  Required when building with GE EPIC's
 * -Werror -Wfloat-equal. */
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wfloat-equal"
#endif

#include <stdlib.h>

/* ================================================================== */
/*  Allocator overrides                                               */
/* ================================================================== */

/*
 * Override PULSEQ_ALLOC / PULSEQ_FREE *before* including this header to use
 * a host-specific allocator, e.g.:
 *
 *   #define PULSEQ_ALLOC(sz)  MyArenaAlloc(sz)
 *   #define PULSEQ_FREE(ptr)  MyArenaFree(ptr)
 *   #include "pulseq_config.h"
 */
#ifndef PULSEQ_ALLOC
#define PULSEQ_ALLOC(sz) malloc(sz)
#endif

#ifndef PULSEQ_FREE
#define PULSEQ_FREE(ptr) free(ptr)
#endif

/* ================================================================== */
/*  Status / error codes                                              */
/* ================================================================== */

/*
 * Every pulseq function returning int uses this convention:
 *   positive = success (PULSEQ_SUCCESS)
 *   negative = failure
 *
 * The specific negative values below are deliberately identical to the
 * corresponding pulseg PULSEG_ERR_* codes, so a host library layering on
 * pulseq can pass them through unmapped.  Consumers should test with
 * PULSEQ_FAILED() / PULSEQ_SUCCEEDED() rather than matching exact values.
 */

#define PULSEQ_SUCCESS 1

#define PULSEQ_SUCCEEDED(code) ((code) > 0)
#define PULSEQ_FAILED(code) ((code) < 0)

/* Generic */
#define PULSEQ_ERR_NULL_POINTER -1
#define PULSEQ_ERR_INVALID_ARGUMENT -2
#define PULSEQ_ERR_ALLOC_FAILED -3

/* File / parsing */
#define PULSEQ_ERR_FILE_NOT_FOUND -10
#define PULSEQ_ERR_FILE_READ_FAILED -11
#define PULSEQ_ERR_UNSUPPORTED_VERSION -12

/* Signature */
#define PULSEQ_ERR_SIGNATURE_MISMATCH -54
#define PULSEQ_ERR_SIGNATURE_MISSING -55

/* File-set ("next sequence" chain) */
#define PULSEQ_ERR_EMPTY -500
#define PULSEQ_ERR_CHAIN_BROKEN -501

#endif /* PULSEQ_CONFIG_H */
