/**
 * @file collection_ext_seq.cpp
 * @brief Out-of-line definition of pulseg::Collection's ExternalSequence
 * constructor (declared in cxx/include/pulseg/collection.hpp).
 *
 * Kept out of collection.hpp so that consumers of pulseg::Collection who
 * don't use this constructor need not have external/pulseq's
 * ExternalSequence.h on their include path; only targets that link the
 * pulseg_pulseq_adapter library pull in that dependency.
 */

#include "collection.hpp"

#include "ExternalSequence.h"
#include "pulseq_adapter.h"

namespace pulseg
{

Collection::Collection(
    ExternalSequence &seq,
    const Opts &opts,
    bool parse_labels,
    int num_averages,
    const char *source_path)
{
    pulseg_opts copts = opts.to_c();

    pulseq_file *raw = nullptr;
    std::string err;
    if (!pulseg::adapter::from_external_sequence(seq, copts, &raw, &err, source_path))
    {
        throw std::runtime_error("pulseq_adapter: " + err);
    }

    coll_ = pulseg_collection_alloc();
    if (!coll_)
    {
        pulseq_file_free(raw);
        throw std::runtime_error("pulseq_adapter: allocation failed");
    }

    pulseg_diagnostic diag;
    pulseg_diagnostic_init(&diag);
    int code =
        pulseg_convert_collection(coll_, &diag, raw, 1, &copts, parse_labels ? 1 : 0, num_averages);

    pulseq_file_free(raw);

    if (PULSEG_FAILED(code))
    {
        PULSEG_FREE(coll_);
        coll_ = nullptr;
        check(code, diag);
    }

    opts_ = copts;
}

} // namespace pulseg
