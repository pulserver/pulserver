/**
 * @file sequence_file_reader.h
 * @brief Populate a SequenceCache directly from the Pulseq seqfile chain.
 *
 * The recon side's data source is the sequence file itself -- text or
 * binary, lazily indexed through pulseq::SequenceFile -- rather than any
 * scanner-written sidecar. One file per subsequence: the lead file names
 * its successor in the `NextSequence` definition and the chain is walked to
 * the end, mirroring how the interpreter builds its collection.
 *
 * Everything downstream of the populated SequenceCache
 * (pre_compute_trajectories, materialize_readout, the enrich_* family,
 * demodulate_fov_shift) is unchanged: those functions consume the struct,
 * not the file.
 *
 * The per-subsequence sequence description (SeqEvent/RfDef) is derived only
 * when that file carries a `TRSize` definition -- the design side states the
 * structural TR, and without it the description is skipped rather than
 * re-derived: it feeds subspace reconstruction and parameter fitting, never
 * basic imaging.
 */

#ifndef SEQUENCE_FILE_READER_H
#define SEQUENCE_FILE_READER_H

#include "sequence_model.h"

#include <string>

namespace mrdserver
{

    /**
     * @brief Build a SequenceCache from a seqfile and its NextSequence chain.
     *
     * @param lead_path  First file of the chain (text or binary Pulseq).
     *                   Successors are resolved relative to its directory.
     * @throws std::runtime_error on I/O or parse errors, or when a chain
     *         link names a file that does not exist.
     */
    SequenceCache read_sequence_files(const std::string& lead_path);

} // namespace mrdserver

#endif /* SEQUENCE_FILE_READER_H */
