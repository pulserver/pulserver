/**
 * @file raw64.cpp
 * @brief The single translation unit that defines `pulseq::raw64`.
 *
 * See cxx/include/pulseq/raw64.hpp for why the C reader is compiled twice.
 * This file is the second instance: the same sources, inside the same
 * namespace the header declared them in, with PULSEQ_DOUBLE_PRECISION already
 * set by that header.
 *
 * Including .c files is not a habit worth spreading, but it is the whole
 * mechanism here -- it is what gives the second instance C++ linkage, and so
 * namespace-scoped symbols, without renaming a single function.  The
 * alternative was a per-symbol -D prefix list that would have to be kept in
 * step with the header by hand.
 *
 * external_md5 comes along because pulseq_parse.c calls it for [SIGNATURE]
 * verification; the float build links the ordinary copy, and this one needs
 * its own inside the namespace.
 */

#include "pulseq/raw64.hpp"

namespace pulseq
{
    namespace raw64
    {
#include "external_md5.h"
#include "pulseq_internal.h"

#include "external_md5.c"
#include "pulseq_binary.c"
#include "pulseq_names.c"
#include "pulseq_parse.c"
    }  // namespace raw64
}  // namespace pulseq
