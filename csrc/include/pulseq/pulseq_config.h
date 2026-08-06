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
/*  Sequence-data numeric type                                        */
/* ================================================================== */

/*
 * The type every library row, shape sample and raster time is stored in.
 *
 * It is `float` by default, and that is what the scanner builds: the raw model
 * of a two-million-block scan is tens of megabytes of library rows, and the
 * PSD carries them on a 32-bit target where halving that matters more than the
 * digits do.
 *
 * The host reader wants the digits.  A .seq file writes shape samples to nine
 * significant figures and the binary format carries float64 amplitudes and
 * picosecond times -- neither survives a float32 round trip -- so
 * cxx/pulseq/read_double.cpp builds a second instance of these sources with
 * PULSEQ_DOUBLE_PRECISION, inside a namespace of its own.  Both instances
 * exist in the same process and never meet: the type is part of pulseq_file's
 * layout, so a translation unit gets whichever one it compiled against and no
 * header declares a function taking the other.
 *
 * Define PULSEQ_DOUBLE_PRECISION to select it; it switches the scanf
 * conversion alongside the type, which is the only other thing that has to
 * agree.
 */
/* PULSEQ_REAL_FMT is the scanf conversion that matches PULSEQ_REAL.  Every
 * sscanf reading a sequence value must use it rather than a literal "%f":
 * passing a double* to "%f" is undefined behaviour, and it is the one mistake
 * this typedef makes easy to write and impossible to see. */
#ifdef PULSEQ_DOUBLE_PRECISION
#define PULSEQ_REAL double
#define PULSEQ_REAL_FMT "%lf"
#else
#define PULSEQ_REAL float
#define PULSEQ_REAL_FMT "%f"
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
