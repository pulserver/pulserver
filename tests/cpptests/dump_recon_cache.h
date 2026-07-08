// SPDX-License-Identifier: MIT
//
// Deterministic text dump of every field parsed by
// mrdserver::read_sequence_cache() (trajectory_cache_reader.{h,cpp}).
//
// Used both by the dump_recon_cache CLI tool (Stage 4 Step 0: capture
// golden dumps against the pre-port reader) and by the Stage 4 Step 3
// equivalence gtest (dump the post-port reader's output and diff against
// the committed golden files byte-for-byte).
//
// Shapes are dumped verbatim as parsed -- SEQDESC RF shapes are still
// compressed (delta-RLE) in the cache; this reader never decompresses
// them, so neither does the dump.

#ifndef DUMP_RECON_CACHE_H
#define DUMP_RECON_CACHE_H

#include <iomanip>
#include <ostream>
#include <string>

#include "trajectory_cache_reader.h"

namespace dump_recon_cache
{

    inline void dump_floats(std::ostream &os, const float *v, int n)
    {
        os << "[";
        for (int i = 0; i < n; ++i)
        {
            if (i)
                os << ",";
            os << std::setprecision(9) << v[i];
        }
        os << "]";
    }

    inline void dump_shape(std::ostream &os, const mrdserver::RfShapeSamples &s)
    {
        os << "nunc=" << s.num_uncompressed << " n=" << s.samples.size() << " samples=";
        dump_floats(os, s.samples.empty() ? nullptr : s.samples.data(),
                    static_cast<int>(s.samples.size()));
    }

    inline void dump_sequence_cache(std::ostream &os, const mrdserver::SequenceCache &cache)
    {
        os << std::setprecision(9);

        // ---- per-subsequence DEFINITIONS ----
        os << "DEFINITIONS subseq_count=" << cache.definitions_by_subseq.size() << "\n";
        for (size_t s = 0; s < cache.definitions_by_subseq.size(); ++s)
        {
            os << "subseq[" << s << "]:\n";
            const auto &defs = cache.definitions_by_subseq[s];
            for (const auto &kv : defs)
            {
                os << "  " << kv.first << "=";
                for (size_t i = 0; i < kv.second.size(); ++i)
                {
                    if (i)
                        os << ",";
                    os << kv.second[i];
                }
                os << "\n";
            }
        }

        // ---- KSHOTS ----
        os << "KSHOTS count=" << cache.kshots.size() << "\n";
        for (size_t i = 0; i < cache.kshots.size(); ++i)
        {
            const auto &k = cache.kshots[i].k;
            os << "shot[" << i << "]: n=" << k.size() << " ";
            dump_floats(os, k.empty() ? nullptr : k.data(), static_cast<int>(k.size()));
            os << "\n";
        }

        // ---- ENCODING SPACES ----
        os << "ENCODING_SPACES count=" << cache.encoding_spaces.size() << "\n";
        for (size_t i = 0; i < cache.encoding_spaces.size(); ++i)
        {
            const auto &es = cache.encoding_spaces[i];
            os << "es[" << i << "]: subseq_idx=" << es.subseq_idx
               << " nav_subseq_offset=" << es.nav_subseq_offset
               << " geometry_tag=" << es.geometry_tag
               << " slc=(" << es.label_limits.slc.min << "," << es.label_limits.slc.max << ")"
               << " phs=(" << es.label_limits.phs.min << "," << es.label_limits.phs.max << ")"
               << " rep=(" << es.label_limits.rep.min << "," << es.label_limits.rep.max << ")"
               << " avg=(" << es.label_limits.avg.min << "," << es.label_limits.avg.max << ")"
               << " seg=(" << es.label_limits.seg.min << "," << es.label_limits.seg.max << ")"
               << " set=(" << es.label_limits.set.min << "," << es.label_limits.set.max << ")"
               << " eco=(" << es.label_limits.eco.min << "," << es.label_limits.eco.max << ")"
               << " par=(" << es.label_limits.par.min << "," << es.label_limits.par.max << ")"
               << " lin=(" << es.label_limits.lin.min << "," << es.label_limits.lin.max << ")"
               << " acq=(" << es.label_limits.acq.min << "," << es.label_limits.acq.max << ")"
               << "\n";
        }

        // ---- TRAJECTORY TABLE ----
        os << "TABLE count=" << cache.table.size() << "\n";
        for (size_t i = 0; i < cache.table.size(); ++i)
        {
            const auto &e = cache.table[i];
            os << "row[" << i << "]: kx=" << e.kx_shot_id << " ky=" << e.ky_shot_id
               << " kz=" << e.kz_shot_id << " gx=" << e.gx_amplitude << " gy=" << e.gy_amplitude
               << " gz=" << e.gz_amplitude << " rot=" << e.rotation_id << " slc=" << e.slc
               << " seg=" << e.seg << " rep=" << e.rep << " avg=" << e.avg << " set=" << e.set
               << " eco=" << e.eco << " phs=" << e.phs << " lin=" << e.lin << " par=" << e.par
               << " acq=" << e.acq << " flags=" << e.flags << " center_sample=" << e.center_sample
               << " sample_time_us=" << e.sample_time_us
               << " encoding_space_ref=" << e.encoding_space_ref << " off=" << e.off << "\n";
        }

        // ---- ROTATIONS ----
        os << "ROTATIONS count=" << cache.rotations.size() << "\n";
        for (size_t i = 0; i < cache.rotations.size(); ++i)
        {
            os << "rot[" << i << "]: ";
            dump_floats(os, cache.rotations[i].data(), 9);
            os << "\n";
        }

        // ---- SEQ_PARAMS / SEQ_DESCS ----
        os << "SEQ_PARAMS has_seq_desc=" << (cache.has_seq_desc ? 1 : 0)
           << " min_te_us=" << cache.seq_params.min_te_us
           << " min_tr_us=" << cache.seq_params.min_tr_us
           << " max_tr_us=" << cache.seq_params.max_tr_us
           << " max_flip_angle_deg=" << cache.seq_params.max_flip_angle_deg
           << " total_scan_time_us=" << cache.seq_params.total_scan_time_us
           << " num_subseqs=" << cache.seq_params.num_subseqs << "\n";

        os << "SEQ_DESCS count=" << cache.seq_descs.size() << "\n";
        for (size_t s = 0; s < cache.seq_descs.size(); ++s)
        {
            const auto &sd = cache.seq_descs[s];
            os << "seqdesc[" << s << "]: subseq_idx=" << sd.subseq_idx
               << " tr_duration_us=" << sd.tr_duration_us << "\n";

            os << "  events count=" << sd.events.size() << "\n";
            for (size_t e = 0; e < sd.events.size(); ++e)
            {
                const auto &ev = sd.events[e];
                os << "  event[" << e << "]: type=" << static_cast<int>(ev.type)
                   << " ts=" << ev.timestamp_us << " params=";
                dump_floats(os, ev.params, 7);
                os << "\n";
            }

            os << "  rf_defs count=" << sd.rf_defs.size() << "\n";
            for (size_t r = 0; r < sd.rf_defs.size(); ++r)
            {
                const auto &d = sd.rf_defs[r];
                os << "  rf_def[" << r << "]: id=" << d.rf_def_id
                   << " bandwidth_hz=" << d.bandwidth_hz << " num_bands=" << d.num_bands
                   << " band_freq_offsets_hz=";
                dump_floats(os, d.band_freq_offsets_hz, 8);
                os << " band_bandwidth_hz=" << d.band_bandwidth_hz
                   << " total_b1sq_power=" << d.total_b1sq_power
                   << " mag(" ;
                dump_shape(os, d.mag);
                os << ") has_phase=" << (d.has_phase ? 1 : 0);
                if (d.has_phase)
                {
                    os << " phase(";
                    dump_shape(os, d.phase);
                    os << ")";
                }
                os << " has_time=" << (d.has_time ? 1 : 0);
                if (d.has_time)
                {
                    os << " time(";
                    dump_shape(os, d.time);
                    os << ")";
                }
                os << "\n";
            }

            os << "  rf_shape_tuples count=" << sd.rf_shape_tuples.size() << "\n";
            for (size_t t = 0; t < sd.rf_shape_tuples.size(); ++t)
            {
                const auto &tup = sd.rf_shape_tuples[t];
                os << "  tuple[" << t << "]: id=" << tup.tuple_id
                   << " rf_def_id=" << tup.rf_def_id << " rf_shim_id=" << tup.rf_shim_id
                   << " ss_grad_amp_hz_per_m=" << tup.ss_grad_amp_hz_per_m
                   << " slice_thickness_mm=" << tup.slice_thickness_mm
                   << " slice_selective=" << tup.slice_selective << "\n";
            }
        }
    }

} // namespace dump_recon_cache

#endif // DUMP_RECON_CACHE_H
