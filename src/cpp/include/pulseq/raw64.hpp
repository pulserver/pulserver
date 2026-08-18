/**
 * @file raw64.hpp
 * @brief The standalone C Pulseq reader, instantiated a second time at double
 *        precision, scoped inside `pulseq::raw64`.
 *
 * src/c/pulseq is `float` end to end, and deliberately so: the raw model of
 * a two-million-block scan is tens of megabytes of library rows, and the PSD
 * carries them on a 32-bit target where halving that matters more than the
 * digits do.
 *
 * On the host the digits are what matter.  A .seq writes shape samples to nine
 * significant figures and the binary format carries float64 amplitudes and
 * picosecond times; neither survives a float32 round trip, so a `Sequence` read
 * back through the scanner's reader would not equal the one that was written.
 *
 * Rather than maintain a second parser, the same sources are compiled twice.
 * PULSEQ_DOUBLE_PRECISION selects `double` (and the matching scanf conversion);
 * PULSEQ_NO_EXTERN_C drops the C language linkage that would otherwise emit
 * unmangled symbols and collide with the ordinary build.  What comes out is
 * `pulseq::raw64::pulseq_read`, `pulseq::raw64::pulseq_file` and so on --
 * distinct types and distinct symbols, sharing not one line of source with the
 * float instance beyond the files themselves.
 *
 * The definitions live in ONE translation unit, src/cpp/pulseq/raw64.cpp.  This
 * header carries only the declarations.
 *
 * @warning A translation unit that includes this header must not also include
 * the ordinary pulseq headers.  They share include guards, so whichever comes
 * first wins and the other silently becomes a no-op -- which would leave calls
 * to one instance compiled against the other's struct layout.  The static
 * assertion below catches the ordering that is detectable; keeping the two
 * apart is otherwise the caller's job.  Convert to pulseq::Sequence here and
 * let the rest of the program see only that.
 */

#ifndef PULSEQ_CXX_RAW64_HPP
#define PULSEQ_CXX_RAW64_HPP

#if defined(PULSEQ_CONFIG_H) || defined(PULSEQ_TYPES_H) || defined(PULSEQ_H) || defined(PULSEQ_RF_H)
#error "raw64.hpp must be included before (and never alongside) the float-precision pulseq headers"
#endif

/* The C sources include these, and a system header must never be pulled in
 * inside a namespace.  Their own include guards make the inner includes
 * no-ops, which is exactly the point of listing them here. */
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PULSEQ_DOUBLE_PRECISION 1
#define PULSEQ_NO_EXTERN_C 1

namespace pulseq
{
    /** The raw Pulseq file model and its reader, at double precision. */
    namespace raw64
    {
#include "pulseq.h"
#include "pulseq_config.h"
#include "pulseq_types.h"
    } // namespace raw64
} // namespace pulseq

#endif /* PULSEQ_CXX_RAW64_HPP */
