/**
 * @file pulseqlib_mex.cpp
 * @brief Modern C++ MEX gateway for pulseqlib (R2018a+).
 *
 * Provides a single MEX entry point that dispatches to the appropriate
 * pulseqlib C function based on a string command argument.
 *
 * Build (from MATLAB):
 *   run('scripts/setup_mex.m')
 *
 * Usage (from MATLAB, via the +pulserver package wrappers):
 *   coll = pulserver.load(seq_bytes, opts);
 *   tr   = pulserver.find_tr(coll);
 *   wf   = pulserver.get_tr_waveforms(coll);
 *   pulserver.check(coll);
 *   pulserver.save_cache(coll, 'out.bin');
 */

#include "mex.hpp"
#include "mexAdapter.hpp"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

/* pulseqlib C headers */
extern "C" {
#include "pulseqlib_types.h"
#include "pulseqlib_methods.h"
}

using namespace matlab::data;
using matlab::mex::ArgumentList;

/* ================================================================== */
/*  Persistent collection store                                       */
/* ================================================================== */

static std::vector<pulseqlib_collection*> g_collections;
static std::vector<pulseqlib_opts>        g_opts_store;
static std::vector<int>                   g_source_sizes;

/* ================================================================== */
/*  Helper: throw MATLAB error                                        */
/* ================================================================== */
static void check_code(int code, const pulseqlib_diagnostic& diag,
                       std::shared_ptr<matlab::engine::MATLABEngine> eng) {
    if (PULSEQLIB_FAILED(code)) {
        ArrayFactory f;
        eng->feval(u"error", 0,
                   {f.createScalar(std::string("pulserver:error")),
                    f.createScalar(std::string(diag.message))});
    }
}

/* ================================================================== */
/*  MEX class                                                         */
/* ================================================================== */

class MexFunction : public matlab::mex::Function {
public:
    void operator()(ArgumentList outputs, ArgumentList inputs) override {
        auto eng = getEngine();
        ArrayFactory factory;

        if (inputs.size() < 1) {
            eng->feval(u"error", 0,
                       {factory.createScalar("pulserver:error"),
                        factory.createScalar("First argument must be a command string")});
            return;
        }

        std::string cmd = std::string(CharArray(inputs[0]).toAscii());

        /* ──────────────────────────────────────────────────────── */
        /*  load — parse .seq bytes and return collection handle    */
        /* ──────────────────────────────────────────────────────── */
        if (cmd == "load") {
            /* inputs: 'load', seq_bytes_cell(cell of uint8 arrays),
             *         gamma, B0, max_grad, max_slew,
             *         rf_raster, grad_raster, adc_raster, block_raster,
             *         parse_labels(logical), num_averages(double) */
            if (inputs.size() < 12) {
                eng->feval(u"error", 0,
                           {factory.createScalar("pulserver:error"),
                            factory.createScalar("load requires 11 args after command")});
                return;
            }

            /* Unpack cell array of byte buffers */
            CellArray cell_bufs = inputs[1];
            int n_seqs = static_cast<int>(cell_bufs.getNumberOfElements());
            std::vector<std::vector<char>> bufs(n_seqs);
            std::vector<const char*> buf_ptrs(n_seqs);
            std::vector<int> buf_sizes(n_seqs);
            for (int i = 0; i < n_seqs; ++i) {
                TypedArray<uint8_t> bytes = cell_bufs[i];
                bufs[i].assign(bytes.begin(), bytes.end());
                buf_ptrs[i]  = bufs[i].data();
                buf_sizes[i] = static_cast<int>(bufs[i].size());
            }

            pulseqlib_opts opts;
            opts.gamma_hz_per_t          = static_cast<float>(TypedArray<double>(inputs[2])[0]);
            opts.b0_t                    = static_cast<float>(TypedArray<double>(inputs[3])[0]);
            opts.max_grad_hz_per_m       = static_cast<float>(TypedArray<double>(inputs[4])[0]);
            opts.max_slew_hz_per_m_per_s = static_cast<float>(TypedArray<double>(inputs[5])[0]);
            opts.rf_raster_us            = static_cast<float>(TypedArray<double>(inputs[6])[0]) * 1e6f;
            opts.grad_raster_us          = static_cast<float>(TypedArray<double>(inputs[7])[0]) * 1e6f;
            opts.adc_raster_us           = static_cast<float>(TypedArray<double>(inputs[8])[0]) * 1e6f;
            opts.block_raster_us         = static_cast<float>(TypedArray<double>(inputs[9])[0]) * 1e6f;
            int parse_labels = TypedArray<bool>(inputs[10])[0] ? 1 : 0;
            int num_averages = static_cast<int>(TypedArray<double>(inputs[11])[0]);

            pulseqlib_collection* coll = nullptr;
            pulseqlib_diagnostic diag;
            pulseqlib_diagnostic_init(&diag);

            int code = pulseqlib_read_from_buffers(
                &coll, &diag, buf_ptrs.data(), buf_sizes.data(), n_seqs,
                &opts, parse_labels, num_averages);
            check_code(code, diag, eng);

            /* Store and return 1-based handle */
            g_collections.push_back(coll);
            g_opts_store.push_back(opts);
            g_source_sizes.push_back((n_seqs > 0) ? buf_sizes[0] : 0);

            outputs[0] = factory.createScalar(
                static_cast<double>(g_collections.size()));
        }

        /* ──────────────────────────────────────────────────────── */
        /*  free — release a collection                             */
        /* ──────────────────────────────────────────────────────── */
        else if (cmd == "free") {
            int idx = static_cast<int>(TypedArray<double>(inputs[1])[0]) - 1;
            if (idx >= 0 && idx < (int)g_collections.size() && g_collections[idx]) {
                pulseqlib_collection_free(g_collections[idx]);
                g_collections[idx] = nullptr;
            }
        }

        /* ──────────────────────────────────────────────────────── */
        /*  find_tr — identify TR structure                         */
        /* ──────────────────────────────────────────────────────── */
        else if (cmd == "find_tr") {
            int idx = static_cast<int>(TypedArray<double>(inputs[1])[0]) - 1;
            pulseqlib_collection* coll = g_collections[idx];

            pulseqlib_subseq_info si = PULSEQLIB_SUBSEQ_INFO_INIT;
            check_code(pulseqlib_get_subseq_info(coll, 0, &si), {}, eng);

            StructArray result = factory.createStructArray(
                {1, 1},
                {"tr_size", "num_trs", "num_prep_blocks", "num_cooldown_blocks",
                 "degenerate_prep", "degenerate_cooldown", "tr_duration_us"});
            result[0]["tr_size"]              = factory.createScalar((double)si.tr_size);
            result[0]["num_trs"]              = factory.createScalar((double)si.num_trs);
            result[0]["num_prep_blocks"]      = factory.createScalar((double)si.num_prep_blocks);
            result[0]["num_cooldown_blocks"]  = factory.createScalar((double)si.num_cooldown_blocks);
            result[0]["degenerate_prep"]      = factory.createScalar(si.degenerate_prep != 0);
            result[0]["degenerate_cooldown"]  = factory.createScalar(si.degenerate_cooldown != 0);
            result[0]["tr_duration_us"]       = factory.createScalar((double)si.tr_duration_us);

            outputs[0] = result;
        }

        /* ──────────────────────────────────────────────────────── */
        /*  find_segments — identify segments within TR             */
        /* ──────────────────────────────────────────────────────── */
        else if (cmd == "find_segments") {
            int idx = static_cast<int>(TypedArray<double>(inputs[1])[0]) - 1;
            pulseqlib_collection* coll = g_collections[idx];

            pulseqlib_subseq_info si = PULSEQLIB_SUBSEQ_INFO_INIT;
            check_code(pulseqlib_get_subseq_info(coll, 0, &si), {}, eng);

            /* Segment definitions */
            int nseg = si.num_prep_segments + si.num_main_segments + si.num_cooldown_segments;
            auto start_blocks = factory.createArray<double>({1, (size_t)nseg});
            auto num_blocks   = factory.createArray<double>({1, (size_t)nseg});
            for (int i = 0; i < nseg; ++i) {
                pulseqlib_segment_info sinfo = PULSEQLIB_SEGMENT_INFO_INIT;
                pulseqlib_get_segment_info(coll, i, &sinfo);
                start_blocks[0][i] = (double)sinfo.start_block + 1;  /* 1-based */
                num_blocks[0][i]   = (double)sinfo.num_blocks;
            }

            /* Segment tables */
            std::vector<int> prep_tbl(si.num_prep_segments);
            std::vector<int> main_tbl(si.num_main_segments);
            std::vector<int> cool_tbl(si.num_cooldown_segments);
            if (si.num_prep_segments > 0)
                pulseqlib_get_prep_segment_table(coll, 0, prep_tbl.data());
            if (si.num_main_segments > 0)
                pulseqlib_get_main_segment_table(coll, 0, main_tbl.data());
            if (si.num_cooldown_segments > 0)
                pulseqlib_get_cooldown_segment_table(coll, 0, cool_tbl.data());

            auto to_double_arr = [&](const std::vector<int>& v) {
                auto a = factory.createArray<double>({1, v.size()});
                for (size_t i = 0; i < v.size(); ++i) a[0][i] = (double)(v[i] + 1);
                return a;
            };

            StructArray result = factory.createStructArray(
                {1, 1},
                {"start_blocks", "num_blocks",
                 "prep_segment_table", "main_segment_table", "cooldown_segment_table"});
            result[0]["start_blocks"]           = start_blocks;
            result[0]["num_blocks"]             = num_blocks;
            result[0]["prep_segment_table"]     = to_double_arr(prep_tbl);
            result[0]["main_segment_table"]     = to_double_arr(main_tbl);
            result[0]["cooldown_segment_table"] = to_double_arr(cool_tbl);

            outputs[0] = result;
        }

        /* ──────────────────────────────────────────────────────── */
        /*  get_tr_waveforms — native-timing waveforms              */
        /* ──────────────────────────────────────────────────────── */
        else if (cmd == "get_tr_waveforms") {
            int idx       = static_cast<int>(TypedArray<double>(inputs[1])[0]) - 1;
            int ss_idx    = (inputs.size() > 2) ? static_cast<int>(TypedArray<double>(inputs[2])[0]) : 0;
            int amp_mode  = (inputs.size() > 3) ? static_cast<int>(TypedArray<double>(inputs[3])[0]) : 0;
            int tr_idx    = (inputs.size() > 4) ? static_cast<int>(TypedArray<double>(inputs[4])[0]) : 0;
            int inc_prep  = (inputs.size() > 5) ? (TypedArray<bool>(inputs[5])[0] ? 1 : 0) : 0;
            int inc_cool  = (inputs.size() > 6) ? (TypedArray<bool>(inputs[6])[0] ? 1 : 0) : 0;
            int col_delay = (inputs.size() > 7) ? (TypedArray<bool>(inputs[7])[0] ? 1 : 0) : 0;

            pulseqlib_collection* coll = g_collections[idx];

            pulseqlib_tr_waveforms cw;
            pulseqlib_diagnostic diag;
            std::memset(&cw, 0, sizeof(cw));
            pulseqlib_diagnostic_init(&diag);

            int code = pulseqlib_get_tr_waveforms(
                coll, ss_idx, amp_mode, tr_idx, inc_prep, inc_cool, col_delay, &cw, &diag);
            check_code(code, diag, eng);

            auto copy_ch = [&](const pulseqlib_channel_waveform& ch) -> StructArray {
                int n = ch.num_samples;
                auto t = factory.createArray<double>({1, (size_t)n});
                auto a = factory.createArray<double>({1, (size_t)n});
                for (int i = 0; i < n; ++i) {
                    t[0][i] = (double)ch.time_us[i];
                    a[0][i] = (double)ch.amplitude[i];
                }
                StructArray s = factory.createStructArray({1, 1}, {"time_us", "amplitude"});
                s[0]["time_us"]   = t;
                s[0]["amplitude"] = a;
                return s;
            };

            StructArray result = factory.createStructArray(
                {1, 1},
                {"gx", "gy", "gz", "rf_mag", "rf_phase", "total_duration_us"});
            result[0]["gx"]       = copy_ch(cw.gx);
            result[0]["gy"]       = copy_ch(cw.gy);
            result[0]["gz"]       = copy_ch(cw.gz);
            result[0]["rf_mag"]   = copy_ch(cw.rf_mag);
            result[0]["rf_phase"] = copy_ch(cw.rf_phase);
            result[0]["total_duration_us"] = factory.createScalar((double)cw.total_duration_us);

            pulseqlib_tr_waveforms_free(&cw);
            outputs[0] = result;
        }

        /* ──────────────────────────────────────────────────────── */
        /*  check — consistency + safety                            */
        /* ──────────────────────────────────────────────────────── */
        else if (cmd == "check") {
            /* inputs: 'check', handle, stim_threshold, decay_constant_us,
             *         pns_threshold_percent, forbidden_bands (Nx3 matrix) */
            int idx = static_cast<int>(TypedArray<double>(inputs[1])[0]) - 1;
            pulseqlib_collection* coll = g_collections[idx];
            pulseqlib_diagnostic diag;
            pulseqlib_diagnostic_init(&diag);

            /* --- consistency ---------------------------------------- */
            int code = pulseqlib_check_consistency(coll, &diag);
            check_code(code, diag, eng);

            /* --- parse optional inputs ------------------------------ */
            float stim_threshold        = 0.0f;
            float decay_constant_us     = 0.0f;
            float pns_threshold_percent = 100.0f;
            int   num_bands = 0;
            std::vector<pulseqlib_forbidden_band> cbands;

            if (inputs.size() > 2)
                stim_threshold = static_cast<float>(
                    TypedArray<double>(inputs[2])[0]);
            if (inputs.size() > 3)
                decay_constant_us = static_cast<float>(
                    TypedArray<double>(inputs[3])[0]);
            if (inputs.size() > 4)
                pns_threshold_percent = static_cast<float>(
                    TypedArray<double>(inputs[4])[0]);
            if (inputs.size() > 5) {
                TypedArray<double> bmat(inputs[5]);
                auto dims = bmat.getDimensions();
                num_bands = static_cast<int>(dims[0]);
                cbands.resize(num_bands);
                for (int i = 0; i < num_bands; ++i) {
                    cbands[i].freq_min_hz            = static_cast<float>(bmat[i][0]);
                    cbands[i].freq_max_hz            = static_cast<float>(bmat[i][1]);
                    cbands[i].max_amplitude_hz_per_m = static_cast<float>(bmat[i][2]);
                }
            }

            /* --- PNS params ----------------------------------------- */
            const pulseqlib_pns_params* pns_ptr = nullptr;
            pulseqlib_pns_params pns;
            if (stim_threshold > 0.0f && decay_constant_us > 0.0f) {
                pns.vendor                  = 0;
                pns.chronaxie_us            = decay_constant_us;
                pns.rheobase_hz_per_m_per_s = stim_threshold;
                pns.alpha                   = 1.0f;
                pns_ptr = &pns;
            }

            /* --- safety --------------------------------------------- */
            code = pulseqlib_check_safety(
                coll, &diag, &g_opts_store[idx],
                num_bands,
                cbands.empty() ? nullptr : cbands.data(),
                pns_ptr, pns_threshold_percent);
            check_code(code, diag, eng);
        }

        /* ──────────────────────────────────────────────────────── */
        /*  save_cache — write binary cache (serialize)             */
        /* ──────────────────────────────────────────────────────── */
        else if (cmd == "save_cache") {
            int idx = static_cast<int>(TypedArray<double>(inputs[1])[0]) - 1;
            std::string path(CharArray(inputs[2]).toAscii());
            pulseqlib_collection* coll = g_collections[idx];

            int code = pulseqlib_save_cache(coll, path.c_str(), g_source_sizes[idx]);
            if (PULSEQLIB_FAILED(code)) {
                eng->feval(u"error", 0,
                           {factory.createScalar("pulserver:error"),
                            factory.createScalar("Failed to write cache file")});
            }
        }

        /* ──────────────────────────────────────────────────────── */
        /*  load_cache — read binary cache (deserialize)            */
        /* ──────────────────────────────────────────────────────── */
        else if (cmd == "load_cache") {
            int idx = static_cast<int>(TypedArray<double>(inputs[1])[0]) - 1;
            std::string path(CharArray(inputs[2]).toAscii());
            pulseqlib_collection* coll = g_collections[idx];

            int code = pulseqlib_load_cache(coll, path.c_str(), g_source_sizes[idx]);
            if (PULSEQLIB_FAILED(code)) {
                eng->feval(u"error", 0,
                           {factory.createScalar("pulserver:error"),
                            factory.createScalar("Failed to read cache file")});
            }
        }

        /* ──────────────────────────────────────────────────────── */
        /*  report — collection / subseq / segment summary          */
        /* ──────────────────────────────────────────────────────── */
        else if (cmd == "report") {
            int idx = static_cast<int>(TypedArray<double>(inputs[1])[0]) - 1;
            pulseqlib_collection* coll = g_collections[idx];

            pulseqlib_collection_info ci = PULSEQLIB_COLLECTION_INFO_INIT;
            pulseqlib_get_collection_info(coll, &ci);

            size_t nss = (size_t)ci.num_subsequences;

            StructArray result = factory.createStructArray(
                {1, nss},
                {"tr_size", "num_trs", "tr_duration_us",
                 "num_prep_blocks", "num_cooldown_blocks",
                 "num_unique_segments", "segments",
                 "prep_segment_table", "main_segment_table",
                 "cooldown_segment_table"});

            for (size_t ss = 0; ss < nss; ++ss) {
                pulseqlib_subseq_info si = PULSEQLIB_SUBSEQ_INFO_INIT;
                pulseqlib_get_subseq_info(coll, (int)ss, &si);

                result[ss]["tr_size"]             = factory.createScalar((double)si.tr_size);
                result[ss]["num_trs"]             = factory.createScalar((double)si.num_trs);
                result[ss]["tr_duration_us"]      = factory.createScalar((double)si.tr_duration_us);
                result[ss]["num_prep_blocks"]     = factory.createScalar((double)si.num_prep_blocks);
                result[ss]["num_cooldown_blocks"] = factory.createScalar((double)si.num_cooldown_blocks);

                int seg_count = si.num_prep_segments + si.num_main_segments + si.num_cooldown_segments;
                result[ss]["num_unique_segments"] = factory.createScalar((double)seg_count);

                /* segment descriptors */
                StructArray segs = factory.createStructArray(
                    {1, (size_t)seg_count}, {"start_block", "num_blocks"});
                for (int j = 0; j < seg_count; ++j) {
                    pulseqlib_segment_info sinfo = PULSEQLIB_SEGMENT_INFO_INIT;
                    pulseqlib_get_segment_info(coll, si.segment_offset + j, &sinfo);
                    segs[j]["start_block"] = factory.createScalar((double)sinfo.start_block + 1);
                    segs[j]["num_blocks"]  = factory.createScalar((double)sinfo.num_blocks);
                }
                result[ss]["segments"] = segs;

                /* segment tables (1-based) */
                auto copy_table = [&](int count, int (*getter)(const pulseqlib_collection*, int, int*)) {
                    std::vector<int> ids(count);
                    if (count > 0) getter(coll, (int)ss, ids.data());
                    auto arr = factory.createArray<double>({1, (size_t)count});
                    for (int k = 0; k < count; ++k) arr[0][k] = (double)(ids[k] + 1);
                    return arr;
                };
                result[ss]["prep_segment_table"]     = copy_table(si.num_prep_segments, pulseqlib_get_prep_segment_table);
                result[ss]["main_segment_table"]     = copy_table(si.num_main_segments, pulseqlib_get_main_segment_table);
                result[ss]["cooldown_segment_table"] = copy_table(si.num_cooldown_segments, pulseqlib_get_cooldown_segment_table);
            }
            outputs[0] = result;
        }

        /* ──────────────────────────────────────────────────────── */
        /*  get_block — per-block metadata                          */
        /* ──────────────────────────────────────────────────────── */
        else if (cmd == "get_block") {
            int idx = static_cast<int>(TypedArray<double>(inputs[1])[0]) - 1;
            int seg = static_cast<int>(TypedArray<double>(inputs[2])[0]);
            int blk = static_cast<int>(TypedArray<double>(inputs[3])[0]);
            pulseqlib_collection* coll = g_collections[idx];

            pulseqlib_block_info info = PULSEQLIB_BLOCK_INFO_INIT;
            pulseqlib_get_block_info(coll, seg, blk, &info);

            StructArray result = factory.createStructArray(
                {1, 1},
                {"duration_us", "start_time_us",
                 "has_gx", "has_gy", "has_gz",
                 "gx_is_trap", "gy_is_trap", "gz_is_trap",
                 "gx_num_samples", "gy_num_samples", "gz_num_samples",
                 "has_rf", "rf_num_samples",
                 "has_adc"});
            result[0]["duration_us"]    = factory.createScalar((double)info.duration_us);
            result[0]["start_time_us"]  = factory.createScalar((double)info.start_time_us);
            result[0]["has_gx"]         = factory.createScalar(info.has_grad[0] != 0);
            result[0]["has_gy"]         = factory.createScalar(info.has_grad[1] != 0);
            result[0]["has_gz"]         = factory.createScalar(info.has_grad[2] != 0);
            result[0]["gx_is_trap"]     = factory.createScalar(info.grad_is_trapezoid[0] != 0);
            result[0]["gy_is_trap"]     = factory.createScalar(info.grad_is_trapezoid[1] != 0);
            result[0]["gz_is_trap"]     = factory.createScalar(info.grad_is_trapezoid[2] != 0);
            result[0]["gx_num_samples"] = factory.createScalar((double)info.grad_num_samples[0]);
            result[0]["gy_num_samples"] = factory.createScalar((double)info.grad_num_samples[1]);
            result[0]["gz_num_samples"] = factory.createScalar((double)info.grad_num_samples[2]);
            result[0]["has_rf"]         = factory.createScalar(info.has_rf != 0);
            result[0]["rf_num_samples"] = factory.createScalar((double)info.rf_num_samples);
            result[0]["has_adc"]        = factory.createScalar(info.has_adc != 0);
            outputs[0] = result;
        }

        /* ──────────────────────────────────────────────────────── */
        /*  pns — peripheral nerve stimulation                      */
        /* ──────────────────────────────────────────────────────── */
        else if (cmd == "pns") {
            int idx    = static_cast<int>(TypedArray<double>(inputs[1])[0]) - 1;
            int ss_idx = static_cast<int>(TypedArray<double>(inputs[2])[0]);
            float chronaxie_us = static_cast<float>(TypedArray<double>(inputs[3])[0]);
            float rheobase     = static_cast<float>(TypedArray<double>(inputs[4])[0]);
            float alpha        = static_cast<float>(TypedArray<double>(inputs[5])[0]);

            pulseqlib_collection* coll = g_collections[idx];

            pulseqlib_pns_params params = PULSEQLIB_PNS_PARAMS_INIT;
            params.chronaxie_us            = chronaxie_us;
            params.rheobase_hz_per_m_per_s = rheobase;
            params.alpha                   = alpha;

            pulseqlib_pns_result cr = PULSEQLIB_PNS_RESULT_INIT;
            pulseqlib_diagnostic diag;
            pulseqlib_diagnostic_init(&diag);

            int code = pulseqlib_calc_pns(&cr, &diag, coll, ss_idx,
                                          &g_opts_store[idx], &params);
            check_code(code, diag, eng);

            int n = cr.num_samples;
            auto slew_x = factory.createArray<double>({1, (size_t)n});
            auto slew_y = factory.createArray<double>({1, (size_t)n});
            auto slew_z = factory.createArray<double>({1, (size_t)n});
            for (int i = 0; i < n; ++i) {
                slew_x[0][i] = (double)cr.slew_x_hz_per_m_per_s[i];
                slew_y[0][i] = (double)cr.slew_y_hz_per_m_per_s[i];
                slew_z[0][i] = (double)cr.slew_z_hz_per_m_per_s[i];
            }

            StructArray result = factory.createStructArray(
                {1, 1}, {"num_samples", "slew_x", "slew_y", "slew_z"});
            result[0]["num_samples"] = factory.createScalar((double)n);
            result[0]["slew_x"]      = slew_x;
            result[0]["slew_y"]      = slew_y;
            result[0]["slew_z"]      = slew_z;

            pulseqlib_pns_result_free(&cr);
            outputs[0] = result;
        }

        /* ──────────────────────────────────────────────────────── */
        /*  grad_spectrum — acoustic spectral analysis              */
        /* ──────────────────────────────────────────────────────── */
        else if (cmd == "grad_spectrum") {
            int idx    = static_cast<int>(TypedArray<double>(inputs[1])[0]) - 1;
            int ss_idx = static_cast<int>(TypedArray<double>(inputs[2])[0]);
            int target_window_size      = static_cast<int>(TypedArray<double>(inputs[3])[0]);
            float target_resolution_hz  = static_cast<float>(TypedArray<double>(inputs[4])[0]);
            float max_freq_hz           = static_cast<float>(TypedArray<double>(inputs[5])[0]);

            pulseqlib_collection* coll = g_collections[idx];

            pulseqlib_acoustic_spectra sp = PULSEQLIB_ACOUSTIC_SPECTRA_INIT;
            pulseqlib_diagnostic diag;
            pulseqlib_diagnostic_init(&diag);

            int code = pulseqlib_calc_acoustic_spectra(
                &sp, &diag, coll, ss_idx, &g_opts_store[idx],
                target_window_size, target_resolution_hz, max_freq_hz,
                0, nullptr);
            check_code(code, diag, eng);

            /* Copy sliding-window spectrograms */
            size_t nw = (size_t)sp.num_windows;
            size_t nf = (size_t)sp.num_freq_bins;

            auto copy_2d = [&](const float* src) {
                auto arr = factory.createArray<double>({nw, nf});
                for (size_t w = 0; w < nw; ++w)
                    for (size_t f = 0; f < nf; ++f)
                        arr[w][f] = (double)src[w * nf + f];
                return arr;
            };
            auto copy_1d = [&](const float* src, size_t len) {
                auto arr = factory.createArray<double>({1, len});
                for (size_t i = 0; i < len; ++i)
                    arr[0][i] = (double)src[i];
                return arr;
            };
            auto copy_int_2d = [&](const int* src) {
                auto arr = factory.createArray<double>({nw, nf});
                for (size_t w = 0; w < nw; ++w)
                    for (size_t f = 0; f < nf; ++f)
                        arr[w][f] = (double)src[w * nf + f];
                return arr;
            };
            auto copy_int_1d = [&](const int* src, size_t len) {
                auto arr = factory.createArray<double>({1, len});
                for (size_t i = 0; i < len; ++i)
                    arr[0][i] = (double)src[i];
                return arr;
            };

            std::vector<std::string> fields = {
                "freq_min_hz", "freq_spacing_hz", "num_freq_bins", "num_windows",
                "spectrogram_gx", "spectrogram_gy", "spectrogram_gz",
                "peaks_gx", "peaks_gy", "peaks_gz",
                "spectrum_full_gx", "spectrum_full_gy", "spectrum_full_gz",
                "peaks_full_gx", "peaks_full_gy", "peaks_full_gz",
                "freq_spacing_seq_hz", "num_freq_bins_seq"
            };
            if (sp.num_freq_bins_seq > 0) {
                fields.push_back("spectrum_seq_gx");
                fields.push_back("spectrum_seq_gy");
                fields.push_back("spectrum_seq_gz");
                fields.push_back("peaks_seq_gx");
                fields.push_back("peaks_seq_gy");
                fields.push_back("peaks_seq_gz");
            }

            StructArray result = factory.createStructArray({1, 1}, fields);
            result[0]["freq_min_hz"]     = factory.createScalar((double)sp.freq_min_hz);
            result[0]["freq_spacing_hz"] = factory.createScalar((double)sp.freq_spacing_hz);
            result[0]["num_freq_bins"]   = factory.createScalar((double)sp.num_freq_bins);
            result[0]["num_windows"]     = factory.createScalar((double)sp.num_windows);

            result[0]["spectrogram_gx"] = copy_2d(sp.spectrogram_gx);
            result[0]["spectrogram_gy"] = copy_2d(sp.spectrogram_gy);
            result[0]["spectrogram_gz"] = copy_2d(sp.spectrogram_gz);
            result[0]["peaks_gx"]       = copy_int_2d(sp.peaks_gx);
            result[0]["peaks_gy"]       = copy_int_2d(sp.peaks_gy);
            result[0]["peaks_gz"]       = copy_int_2d(sp.peaks_gz);

            result[0]["spectrum_full_gx"] = copy_1d(sp.spectrum_full_gx, nf);
            result[0]["spectrum_full_gy"] = copy_1d(sp.spectrum_full_gy, nf);
            result[0]["spectrum_full_gz"] = copy_1d(sp.spectrum_full_gz, nf);
            result[0]["peaks_full_gx"]    = copy_int_1d(sp.peaks_full_gx, nf);
            result[0]["peaks_full_gy"]    = copy_int_1d(sp.peaks_full_gy, nf);
            result[0]["peaks_full_gz"]    = copy_int_1d(sp.peaks_full_gz, nf);

            result[0]["freq_spacing_seq_hz"] = factory.createScalar((double)sp.freq_spacing_seq_hz);
            result[0]["num_freq_bins_seq"]   = factory.createScalar((double)sp.num_freq_bins_seq);

            if (sp.num_freq_bins_seq > 0) {
                size_t nfs = (size_t)sp.num_freq_bins_seq;
                result[0]["spectrum_seq_gx"] = copy_1d(sp.spectrum_seq_gx, nfs);
                result[0]["spectrum_seq_gy"] = copy_1d(sp.spectrum_seq_gy, nfs);
                result[0]["spectrum_seq_gz"] = copy_1d(sp.spectrum_seq_gz, nfs);
                result[0]["peaks_seq_gx"]    = copy_int_1d(sp.peaks_seq_gx, nfs);
                result[0]["peaks_seq_gy"]    = copy_int_1d(sp.peaks_seq_gy, nfs);
                result[0]["peaks_seq_gz"]    = copy_int_1d(sp.peaks_seq_gz, nfs);
            }

            pulseqlib_acoustic_spectra_free(&sp);
            outputs[0] = result;
        }

        /* ──────────────────────────────────────────────────────── */
        /*  Unique-block / segment-block queries                    */
        /* ──────────────────────────────────────────────────────── */
        else if (cmd == "num_unique_blocks") {
            /* inputs: 'num_unique_blocks', handle, seq_idx */
            int idx = static_cast<int>(TypedArray<double>(inputs[1])[0]) - 1;
            pulseqlib_collection* coll = g_collections[idx];
            int seq_idx = static_cast<int>(TypedArray<double>(inputs[2])[0]);
            int n = pulseqlib_get_num_unique_blocks(coll, seq_idx);
            if (n < 0) {
                pulseqlib_diagnostic diag;
                pulseqlib_diagnostic_init(&diag);
                check_code(n, diag, eng);
            }
            outputs[0] = factory.createScalar(static_cast<double>(n));
        }

        else if (cmd == "unique_block_id") {
            /* inputs: 'unique_block_id', handle, seq_idx, blk_def_idx */
            int idx = static_cast<int>(TypedArray<double>(inputs[1])[0]) - 1;
            pulseqlib_collection* coll = g_collections[idx];
            int seq_idx = static_cast<int>(TypedArray<double>(inputs[2])[0]);
            int blk_def_idx = static_cast<int>(TypedArray<double>(inputs[3])[0]);
            int id = pulseqlib_get_unique_block_id(coll, seq_idx, blk_def_idx);
            if (id < 0) {
                pulseqlib_diagnostic diag;
                pulseqlib_diagnostic_init(&diag);
                check_code(id, diag, eng);
            }
            outputs[0] = factory.createScalar(static_cast<double>(id));
        }

        else if (cmd == "segment_block_def_indices") {
            /* inputs: 'segment_block_def_indices', handle, seg_idx */
            int idx = static_cast<int>(TypedArray<double>(inputs[1])[0]) - 1;
            pulseqlib_collection* coll = g_collections[idx];
            int seg_idx = static_cast<int>(TypedArray<double>(inputs[2])[0]);

            pulseqlib_segment_info si;
            pulseqlib_get_segment_info(coll, seg_idx, &si);
            std::vector<int> ids(si.num_blocks);
            if (si.num_blocks > 0)
                pulseqlib_get_segment_block_def_indices(coll, seg_idx, ids.data());

            TypedArray<double> arr = factory.createArray<double>({1, (size_t)si.num_blocks});
            for (int i = 0; i < si.num_blocks; ++i)
                arr[0][i] = static_cast<double>(ids[i]);
            outputs[0] = arr;
        }

        /* ──────────────────────────────────────────────────────── */
        /*  Unknown command                                         */
        /* ──────────────────────────────────────────────────────── */
        else {
            eng->feval(u"error", 0,
                       {factory.createScalar("pulserver:error"),
                        factory.createScalar("Unknown command: " + cmd)});
        }
    }
};
