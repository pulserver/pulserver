/**
 * @file pulseq_adapter.cpp
 * @brief See pulseq_adapter.h for the design rationale and known
 * limitations.
 */

#include "pulseq_adapter.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "ExternalSequence.h"
#include "pulseg_config.h"

// pulseg_pulseq_file's raw arrays are plain malloc'd C arrays freed via
// PULSEG_FREE by pulseg_pulseq_file_free(); PULSEG_ALLOC/PULSEG_FREE resolve
// to malloc/free by default (pulseg_config.h) so building them from C++ with
// the same macros is safe.

namespace pulseg
{
    namespace adapter
    {

        namespace
        {

            constexpr double kTwoPi = 6.283185307179586476925286766558;

            // ---------------------------------------------------------------
            // Growable raw-row builders (converted to malloc'd C arrays at the end)
            // ---------------------------------------------------------------
            struct Builder
            {
                std::vector<std::array<float, 7>> block_rows;
                std::vector<std::array<float, 10>> rf_rows;
                std::vector<int> rf_use_tags;
                std::vector<std::array<float, 7>> grad_rows;
                std::vector<std::array<float, 8>> adc_rows;
                std::vector<std::array<float, 3>> ext_rows; // type, ref(1-based), next(1-based,0=end)
                std::vector<std::array<float, 4>> trigger_rows;
                std::vector<std::array<float, 4>> rotation_rows; // quaternion
                std::vector<std::array<float, 2>> labelset_rows; // value, label_id
                std::vector<std::array<float, 2>> labelinc_rows;
                std::vector<std::array<float, 4>> soft_delay_rows;
                std::vector<pulseg__rf_shim_entry> rf_shim_rows;
                std::vector<pulseg_shape_arbitrary> shapes; // pass-through (already decompressed)
                // Content -> 1-based shape id, so that two blocks referencing
                // the identical underlying Pulseq shape (e.g. the same RF
                // envelope reused across many excitations) collapse onto the
                // SAME shape id. This matters beyond storage: dedup_rf/
                // grad_library (pulseg_dedup.c) key their definition identity
                // on shape-id *numbers*, not shape *content* -- the native
                // .seq format gets this for free because the file itself
                // shares one [SHAPES] entry across events; this adapter must
                // recreate that sharing explicitly since it rebuilds shapes
                // from ExternalSequence's already-resolved per-block arrays.
                std::map<std::vector<float>, int> shape_cache;
            };

            // 1-based id (0 = "no shape") for a pass-through (uncompressed) shape.
            // Interns by content (see Builder::shape_cache) so identical
            // shapes referenced from different blocks share one id.
            int add_shape(Builder &b, const float *data, int n)
            {
                if (!data || n <= 0)
                    return 0;
                std::vector<float> key(data, data + n);
                auto it = b.shape_cache.find(key);
                if (it != b.shape_cache.end())
                    return it->second;

                pulseg_shape_arbitrary s;
                s.num_samples = n;
                s.num_uncompressed_samples = n;
                s.samples = (float *)PULSEG_ALLOC(sizeof(float) * (size_t)n);
                std::memcpy(s.samples, data, sizeof(float) * (size_t)n);
                b.shapes.push_back(s);
                int id = (int)b.shapes.size(); // 1-based
                b.shape_cache.emplace(std::move(key), id);
                return id;
            }

            int add_shape(Builder &b, const std::vector<float> &v)
            {
                return add_shape(b, v.empty() ? nullptr : v.data(), (int)v.size());
            }

            // Appends one extension-chain node (type,ref) to the block's chain and
            // returns the (1-based) id of the newly appended node, threading
            // `head`/`tail` so callers can build a chain incrementally.
            void append_ext(Builder &b, int type, int ref_1based, int &head, int &prev_id)
            {
                std::array<float, 3> row = {(float)type, (float)ref_1based, 0.0f};
                b.ext_rows.push_back(row);
                int new_id = (int)b.ext_rows.size(); // 1-based
                if (prev_id > 0)
                    b.ext_rows[prev_id - 1][2] = (float)new_id;
                else
                    head = new_id;
                prev_id = new_id;
            }

            // ---------------------------------------------------------------
            // PMC / TRID recovery (Stage 3 Step 3.3 STOP-point fallback).
            //
            // ExternalSequence's decodeLabel() (LABELMAP_IGNORE) silently drops
            // any LABELSET/LABELINC entry named PMC or TRID -- they never reach
            // m_labelsetLibrary/m_labelincLibrary, so there is no public API to
            // recover them from a decoded block. This performs a narrow,
            // self-contained re-scan of the raw .seq text: it only extracts the
            // handful of lines needed to correlate "block N references ignored
            // labelset/labelinc id M, name PMC/TRID, value V" -- it does not
            // reimplement the rest of the format.
            // ---------------------------------------------------------------
            struct IgnoredLabelRef
            {
                bool is_labelinc; // false = LABELSET, true = LABELINC
                std::string name; // "PMC" or "TRID"
                int value;
            };

            void strip_comment_and_trim(std::string &line)
            {
                std::size_t h = line.find('#');
                if (h != std::string::npos)
                    line.resize(h);
                std::size_t a = line.find_first_not_of(" \t\r\n");
                std::size_t z = line.find_last_not_of(" \t\r\n");
                if (a == std::string::npos)
                {
                    line.clear();
                    return;
                }
                line = line.substr(a, z - a + 1);
            }

            // Returns, per 0-based block index, the list of ignored (PMC/TRID)
            // label references that apply to that block. Best-effort: on any
            // parse irregularity it simply returns fewer/no entries (PMC/TRID
            // are optional pulserver extensions; their absence does not affect
            // standard MDH labels, rotations, RF shims, or timing).
            std::map<int, std::vector<IgnoredLabelRef>> rescan_ignored_labels(const std::string &path)
            {
                std::map<int, std::vector<IgnoredLabelRef>> result;
                std::ifstream f(path.c_str());
                if (!f.is_open())
                    return result;

                // file-local extension type id -> canonical name ("LABELSET"/"LABELINC"/other)
                std::map<int, std::string> ext_type_name;
                // (is_labelinc, file-local library id) -> (name, value), PMC/TRID only
                std::map<std::pair<bool, int>, std::pair<std::string, int>> ignored_by_id;
                // extensions_library: file-local ext id -> (type_id, ref_id, next_id)
                std::map<int, std::array<int, 3>> ext_table;
                // block index (0-based) -> head extension id (0 = none)
                std::vector<int> block_ext_head;

                std::string line;
                std::string section;
                while (std::getline(f, line))
                {
                    std::string raw = line;
                    strip_comment_and_trim(line);
                    if (line.empty())
                        continue;

                    if (!raw.empty() && raw[0] == '[')
                    {
                        std::size_t e = raw.find(']');
                        section = (e != std::string::npos) ? raw.substr(1, e - 1) : "";
                        continue;
                    }

                    if (line.compare(0, 10, "extension ") == 0)
                    {
                        std::istringstream iss(line);
                        std::string kw, name;
                        int id = 0;
                        iss >> kw >> name >> id;
                        if (id > 0)
                            ext_type_name[id] = name;
                        continue;
                    }

                    if (section == "EXTENSIONS")
                    {
                        std::istringstream iss(line);
                        int id, type, ref, next;
                        if (iss >> id >> type >> ref >> next)
                            ext_table[id] = {type, ref, next};
                    }
                    else if (section == "LABELSET" || section == "LABELINC")
                    {
                        std::istringstream iss(line);
                        int id, value;
                        std::string name;
                        if (iss >> id >> value >> name)
                        {
                            if (name == "PMC" || name == "TRID")
                                ignored_by_id[{section == "LABELINC", id}] = {name, value};
                        }
                    }
                    else if (section == "BLOCKS")
                    {
                        std::istringstream iss(line);
                        int idx;
                        long vals[7];
                        if (iss >> idx >> vals[0] >> vals[1] >> vals[2] >> vals[3] >> vals[4] >> vals[5] >> vals[6])
                        {
                            if ((int)block_ext_head.size() < idx)
                                block_ext_head.resize(idx, 0);
                            block_ext_head[idx - 1] = (int)vals[6];
                        }
                    }
                }

                for (std::size_t bi = 0; bi < block_ext_head.size(); ++bi)
                {
                    int cur = block_ext_head[bi];
                    int guard = 0;
                    while (cur > 0 && guard++ < 4096)
                    {
                        auto et = ext_table.find(cur);
                        if (et == ext_table.end())
                            break;
                        int type_id = et->second[0];
                        int ref_id = et->second[1];
                        int next_id = et->second[2];
                        auto tn = ext_type_name.find(type_id);
                        if (tn != ext_type_name.end())
                        {
                            bool is_inc = (tn->second == "LABELINC");
                            if (tn->second == "LABELSET" || is_inc)
                            {
                                auto ig = ignored_by_id.find({is_inc, ref_id});
                                if (ig != ignored_by_id.end())
                                {
                                    result[(int)bi].push_back(
                                        IgnoredLabelRef{is_inc, ig->second.first, ig->second.second});
                                }
                            }
                        }
                        cur = next_id;
                    }
                }
                return result;
            }

            // ---------------------------------------------------------------
            // Reserved [DEFINITIONS] population (mirrors pulseg_parse.c's
            // read_definitions() key-for-key so downstream timing/geometry
            // resolution is identical regardless of which reader produced the
            // pulseg_pulseq_file).
            // ---------------------------------------------------------------
            void populate_definitions(ExternalSequence &seq, pulseg_pulseq_file *out, Builder &b)
            {
                std::vector<std::string> keys = seq.GetAllDefinitions();
                (void)b;

                out->num_definitions = 0;
                if (keys.empty())
                    return;

                out->definitions_library =
                    (pulseg__definition *)PULSEG_ALLOC(sizeof(pulseg__definition) * keys.size());

                int n = 0;
                for (const auto &key : keys)
                {
                    std::string value = seq.GetDefinitionStr(key);

                    pulseg__definition &def = out->definitions_library[n++];
                    std::memset(def.name, 0, sizeof(def.name));
                    std::strncpy(def.name, key.c_str(), sizeof(def.name) - 1);
                    def.value_size = 0;
                    def.value = nullptr;
                    if (!value.empty())
                    {
                        std::istringstream iss(value);
                        std::string tok;
                        std::vector<std::string> toks;
                        while (iss >> tok)
                            toks.push_back(tok);
                        if (!toks.empty())
                        {
                            def.value = (char **)PULSEG_ALLOC(sizeof(char *) * toks.size());
                            for (std::size_t i = 0; i < toks.size(); ++i)
                            {
                                def.value[i] = (char *)PULSEG_ALLOC(toks[i].size() + 1);
                                std::strcpy(def.value[i], toks[i].c_str());
                            }
                            def.value_size = (int)toks.size();
                        }
                    }

                    if (key == "GradientRasterTime")
                        out->reserved_definitions_library.gradient_raster_time = (float)(std::atof(value.c_str()) * 1e6);
                    else if (key == "RadiofrequencyRasterTime")
                        out->reserved_definitions_library.radiofrequency_raster_time = (float)(std::atof(value.c_str()) * 1e6);
                    else if (key == "AdcRasterTime")
                        out->reserved_definitions_library.adc_raster_time = (float)(std::atof(value.c_str()) * 1e6);
                    else if (key == "BlockDurationRaster")
                        out->reserved_definitions_library.block_duration_raster = (float)(std::atof(value.c_str()) * 1e6);
                    else if (key == "Name")
                    {
                        std::strncpy(out->reserved_definitions_library.name, value.c_str(),
                                      sizeof(out->reserved_definitions_library.name) - 1);
                    }
                    else if (key == "FOV")
                    {
                        std::vector<double> v = seq.GetDefinition(key);
                        if (v.size() >= 3)
                            for (int i = 0; i < 3; ++i)
                                out->reserved_definitions_library.fov[i] = (float)v[i] * 100.0f;
                    }
                    else if (key == "Matrix")
                    {
                        std::vector<double> v = seq.GetDefinition(key);
                        if (v.size() >= 3)
                            for (int i = 0; i < 3; ++i)
                                out->reserved_definitions_library.matrix[i] = (float)v[i];
                    }
                    else if (key == "NavFOV")
                    {
                        std::vector<double> v = seq.GetDefinition(key);
                        if (v.size() >= 3)
                            for (int i = 0; i < 3; ++i)
                                out->reserved_definitions_library.nav_fov[i] = (float)v[i] * 100.0f;
                    }
                    else if (key == "NavMatrix")
                    {
                        std::vector<double> v = seq.GetDefinition(key);
                        if (v.size() >= 3)
                            for (int i = 0; i < 3; ++i)
                                out->reserved_definitions_library.nav_matrix[i] = (float)v[i];
                    }
                    else if (key == "TotalDuration")
                        out->reserved_definitions_library.total_duration = (float)std::atof(value.c_str());
                    else if (key == "NextSequence")
                    {
                        std::strncpy(out->reserved_definitions_library.next_sequence, value.c_str(),
                                      sizeof(out->reserved_definitions_library.next_sequence) - 1);
                    }
                    else if (key == "IgnoreFovShift")
                        out->reserved_definitions_library.ignore_fov_shift = std::atoi(value.c_str());
                    else if (key == "EnablePmc")
                        out->reserved_definitions_library.enable_pmc = std::atoi(value.c_str());
                    else if (key == "IgnoreAverages")
                        out->reserved_definitions_library.ignore_averages = std::atoi(value.c_str());
                    else if (key == "NumGainCalibrationReadouts")
                        out->reserved_definitions_library.num_gain_cal_readouts = std::atoi(value.c_str());
                }
                out->num_definitions = n;
                out->is_definitions_library_parsed = 1;
            }

            // Appends one axis's gradient event (if present) and returns its
            // 1-based library id (0 = absent). `grad_raster_us` is the same
            // raster ExternalSequence used internally to convert its
            // ExtTrap time samples to absolute microseconds (the file's own
            // GradientRasterTime definition, mirrored into
            // out->reserved_definitions_library by populate_definitions);
            // it is needed to undo that conversion back to pulseg's raw
            // (raster-index-normalized) time_shape convention.
            int add_grad_axis(Builder &b, SeqBlock &blk, int channel, float grad_raster_us)
            {
                if (blk.GetEventIndex((Event)(GX + channel)) <= 0)
                    return 0;
                GradEvent &g = blk.GetGradEvent(channel);
                std::array<float, 7> row = {0, 0, 0, 0, 0, 0, 0};
                row[1] = g.amplitude;
                if (g.waveShape == 0)
                {
                    // Trapezoid
                    row[0] = 0;
                    row[2] = (float)g.rampUpTime;
                    row[3] = (float)g.flatTime;
                    row[4] = (float)g.rampDownTime;
                    row[5] = (float)g.delay;
                    row[6] = 0;
                }
                else if (blk.isExtTrapGradient(channel))
                {
                    // Extended trapezoid: genuinely non-uniform raster.
                    // ExternalSequence::decodeExtTrapGradInBlock() already
                    // decompressed both arrays and converted the time
                    // samples to absolute microseconds
                    // (long(0.5 + gradRasterUs * raw_raster_index)); undo
                    // that conversion so pulseg's own get_block() (which
                    // re-multiplies by grad_raster_us after decompressing
                    // the raw shape) recovers the same absolute times.
                    row[0] = 1;
                    row[2] = g.first;
                    row[3] = g.last;
                    row[4] = (float)add_shape(b, blk.GetExtTrapGradShape(channel).data(),
                                               (int)blk.GetExtTrapGradShape(channel).size());
                    {
                        const std::vector<long> &times = blk.GetExtTrapGradTimes(channel);
                        std::vector<float> time_norm(times.size());
                        float raster = (grad_raster_us > 0.0f) ? grad_raster_us : 1.0f;
                        for (std::size_t i = 0; i < times.size(); ++i)
                            time_norm[i] = (float)times[i] / raster;
                        row[5] = (float)add_shape(b, time_norm);
                    }
                    row[6] = (float)g.delay;
                }
                else
                {
                    // Plain arbitrary gradient, uniform raster.
                    row[0] = 1;
                    row[2] = g.first;
                    row[3] = g.last;
                    row[4] = (float)add_shape(b, blk.GetArbGradShapePtr(channel), blk.GetArbGradNumSamples(channel));
                    row[5] = 0; // no time shape -> uniform raster
                    row[6] = (float)g.delay;
                }
                b.grad_rows.push_back(row);
                return (int)b.grad_rows.size();
            }

        } // namespace

        bool from_external_sequence(
            ExternalSequence &seq,
            const pulseg_opts &opts,
            pulseg_pulseq_file **out_ptr,
            std::string *err_msg,
            const char *source_path)
        {
            if (!out_ptr)
            {
                if (err_msg)
                    *err_msg = "out is null";
                return false;
            }
            *out_ptr = nullptr;

            // `out` (not out_ptr) is used by every reference below, matching
            // pulseg_pulseq_file_free()'s contract that the *struct itself*
            // (not just its inner arrays) must be PULSEG_ALLOC'd -- a stack
            // or `new`-allocated instance must never be passed to it.
            pulseg_pulseq_file *out = (pulseg_pulseq_file *)PULSEG_ALLOC(sizeof(pulseg_pulseq_file));
            if (!out)
            {
                if (err_msg)
                    *err_msg = "allocation failed";
                return false;
            }

            auto fail = [&](const std::string &msg) -> bool
            {
                if (err_msg)
                    *err_msg = msg;
                pulseg_pulseq_file_free(out);
                return false;
            };

            pulseg_pulseq_file_init(out, &opts);
            Builder b;

            int n_blocks = seq.GetNumberOfBlocks();
            if (n_blocks <= 0)
                return fail("ExternalSequence has no blocks (call load()/load_from_buffer() first)");

            populate_definitions(seq, out, b);
            float grad_raster_us = (out->reserved_definitions_library.gradient_raster_time > 0.0f)
                                       ? out->reserved_definitions_library.gradient_raster_time
                                       : opts.grad_raster_us;

            std::map<int, std::vector<IgnoredLabelRef>> ignored_labels;
            if (source_path && *source_path)
                ignored_labels = rescan_ignored_labels(source_path);

            for (int bi = 0; bi < n_blocks; ++bi)
            {
                std::unique_ptr<SeqBlock, void (*)(SeqBlock *)> blk(
                    seq.GetBlock(bi), [](SeqBlock *p)
                    { p->free(); delete p; });
                if (!blk)
                    return fail("GetBlock failed at index " + std::to_string(bi));
                if (!seq.decodeBlock(blk.get()))
                    return fail("decodeBlock failed at index " + std::to_string(bi));

                std::array<float, 7> block_row = {0, 0, 0, 0, 0, 0, 0};
                block_row[0] = (float)blk->GetStoredDuration_ru();

                // --- RF ---
                if (blk->isRF())
                {
                    RFEvent &rf = blk->GetRFEvent();
                    std::array<float, 10> row = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                    row[0] = rf.amplitude;
                    row[1] = (float)add_shape(b, blk->GetRFAmplitudePtr(), blk->GetRFLength());

                    // Undo ExternalSequence's TWO_PI phase scaling: pulseg's own
                    // get_block() re-applies it after decompressing the raw
                    // (normalized [0,1]-ish) phase shape.
                    int np = blk->GetRFLength();
                    std::vector<float> phase_norm(np);
                    float *ph = blk->GetRFPhasePtr();
                    for (int i = 0; i < np; ++i)
                        phase_norm[i] = (float)(ph[i] / kTwoPi);
                    row[2] = (float)add_shape(b, phase_norm);

                    row[3] = 0; // no time shape -> uniform RF raster (see header)
                    row[4] = rf.center;
                    row[5] = (float)rf.delay;
                    row[6] = rf.freqPPM;
                    row[7] = rf.phasePPM;
                    row[8] = rf.freqOffset;
                    row[9] = rf.phaseOffset;
                    b.rf_rows.push_back(row);
                    b.rf_use_tags.push_back(
                        rf.use == 'e' ? PULSEG_RF_USE_EXCITATION : rf.use == 'r' ? PULSEG_RF_USE_REFOCUSING
                                                               : rf.use == 'i'   ? PULSEG_RF_USE_INVERSION
                                                               : rf.use == 's'   ? PULSEG_RF_USE_SATURATION
                                                                                 : PULSEG_RF_USE_UNKNOWN);
                    block_row[1] = (float)b.rf_rows.size();
                }

                // --- Gradients ---
                block_row[2] = (float)add_grad_axis(b, *blk, 0, grad_raster_us);
                block_row[3] = (float)add_grad_axis(b, *blk, 1, grad_raster_us);
                block_row[4] = (float)add_grad_axis(b, *blk, 2, grad_raster_us);

                // --- ADC ---
                if (blk->isADC())
                {
                    ADCEvent &adc = blk->GetADCEvent();
                    std::array<float, 8> row = {0, 0, 0, 0, 0, 0, 0, 0};
                    row[0] = (float)adc.numSamples;
                    row[1] = (float)adc.dwellTime;
                    row[2] = (float)adc.delay;
                    row[3] = adc.freqPPM;
                    row[4] = adc.phasePPM;
                    row[5] = adc.freqOffset;
                    row[6] = adc.phaseOffset;
                    row[7] = 0; // phase-modulation shape not supported (see header)
                    b.adc_rows.push_back(row);
                    block_row[5] = (float)b.adc_rows.size();
                }

                // --- Extensions (rotation, trigger, soft delay, rf shim, labels) ---
                int ext_head = 0, ext_prev = 0;

                if (blk->isRotation())
                {
                    RotationEvent &rot = blk->GetRotationEvent();
                    std::array<float, 4> row = {0, 0, 0, 0};
                    for (int i = 0; i < 4; ++i)
                        row[i] = (float)rot.rotQuaternion[i];
                    b.rotation_rows.push_back(row);
                    append_ext(b, PULSEG_PULSEQ_EXT_ROTATION, (int)b.rotation_rows.size(), ext_head, ext_prev);
                }

                if (blk->isTrigger())
                {
                    TriggerEvent &trig = blk->GetTriggerEvent();
                    std::array<float, 4> row = {
                        (float)trig.triggerType, (float)trig.triggerChannel,
                        (float)trig.delay, (float)trig.duration};
                    b.trigger_rows.push_back(row);
                    append_ext(b, PULSEG_PULSEQ_EXT_TRIGGER, (int)b.trigger_rows.size(), ext_head, ext_prev);
                }

                if (blk->isSoftDelay())
                {
                    SoftDelayEvent &sd = blk->GetSoftDelayEvent();
                    int hint_id = pulseg_pulseq_hint_id_for_name(sd.hint);
                    std::array<float, 4> row = {
                        (float)sd.numID, (float)sd.offset, sd.factor, (float)hint_id};
                    b.soft_delay_rows.push_back(row);
                    append_ext(b, PULSEG_PULSEQ_EXT_DELAY, (int)b.soft_delay_rows.size(), ext_head, ext_prev);
                }

                if (blk->hasRfShim())
                {
                    RfShimmingEvent &sh = blk->GetRfShim();
                    pulseg__rf_shim_entry entry;
                    entry.num_channels = sh.nchan;
                    std::memset(entry.values, 0, sizeof(entry.values));
                    int nch = sh.nchan;
                    if (nch > PULSEG_PULSEQ_MAX_RF_SHIM_CHANNELS)
                        nch = PULSEG_PULSEQ_MAX_RF_SHIM_CHANNELS;
                    for (int i = 0; i < nch; ++i)
                    {
                        entry.values[2 * i] = (i < (int)sh.amplitudes.size()) ? sh.amplitudes[i] : 0.0f;
                        entry.values[2 * i + 1] = (i < (int)sh.phases.size()) ? sh.phases[i] : 0.0f;
                    }
                    b.rf_shim_rows.push_back(entry);
                    append_ext(b, PULSEG_PULSEQ_EXT_RF_SHIM, (int)b.rf_shim_rows.size(), ext_head, ext_prev);
                }

                for (LabelEvent &le : blk->GetLabelSetEvents())
                {
                    std::string name = (le.numVal.first >= 0) ? seq.getCounterIdAsString(le.numVal.first) : "";
                    int value = le.numVal.second;
                    if (name.empty())
                    {
                        name = (le.flagVal.first >= 0) ? seq.getFlagIdAsString(le.flagVal.first) : "";
                        value = le.flagVal.second ? 1 : 0;
                    }
                    if (name.empty())
                        continue;
                    int label_id = pulseg_pulseq_label_id_for_name(name.c_str());
                    if (label_id < 0)
                        continue;
                    std::array<float, 2> row = {(float)value, (float)label_id};
                    b.labelset_rows.push_back(row);
                    append_ext(b, PULSEG_PULSEQ_EXT_LABELSET, (int)b.labelset_rows.size(), ext_head, ext_prev);
                }
                for (LabelEvent &le : blk->GetLabelIncEvents())
                {
                    std::string name = (le.numVal.first >= 0) ? seq.getCounterIdAsString(le.numVal.first) : "";
                    if (name.empty())
                        continue;
                    int label_id = pulseg_pulseq_label_id_for_name(name.c_str());
                    if (label_id < 0)
                        continue;
                    std::array<float, 2> row = {(float)le.numVal.second, (float)label_id};
                    b.labelinc_rows.push_back(row);
                    append_ext(b, PULSEG_PULSEQ_EXT_LABELINC, (int)b.labelinc_rows.size(), ext_head, ext_prev);
                }

                // PMC / TRID rescan fallback (see header + rescan_ignored_labels):
                // re-inject the label refs ExternalSequence's own decoder
                // silently dropped for this block.
                auto ig = ignored_labels.find(bi);
                if (ig != ignored_labels.end())
                {
                    for (const IgnoredLabelRef &r : ig->second)
                    {
                        int label_id = pulseg_pulseq_label_id_for_name(r.name.c_str());
                        if (label_id < 0)
                            continue;
                        std::array<float, 2> row = {(float)r.value, (float)label_id};
                        if (r.is_labelinc)
                        {
                            b.labelinc_rows.push_back(row);
                            append_ext(b, PULSEG_PULSEQ_EXT_LABELINC, (int)b.labelinc_rows.size(), ext_head, ext_prev);
                        }
                        else
                        {
                            b.labelset_rows.push_back(row);
                            append_ext(b, PULSEG_PULSEQ_EXT_LABELSET, (int)b.labelset_rows.size(), ext_head, ext_prev);
                        }
                    }
                }

                block_row[6] = (float)ext_head;
                b.block_rows.push_back(block_row);
            }

            // --- Copy growable rows into malloc'd C arrays ---
            auto copy_rows7 = [](const std::vector<std::array<float, 7>> &rows) -> float(*)[7]
            {
                if (rows.empty())
                    return nullptr;
                float(*arr)[7] = (float(*)[7])PULSEG_ALLOC(sizeof(float[7]) * rows.size());
                std::memcpy(arr, rows.data(), sizeof(float[7]) * rows.size());
                return arr;
            };
            auto copy_rows10 = [](const std::vector<std::array<float, 10>> &rows) -> float(*)[10]
            {
                if (rows.empty())
                    return nullptr;
                float(*arr)[10] = (float(*)[10])PULSEG_ALLOC(sizeof(float[10]) * rows.size());
                std::memcpy(arr, rows.data(), sizeof(float[10]) * rows.size());
                return arr;
            };
            auto copy_rows8 = [](const std::vector<std::array<float, 8>> &rows) -> float(*)[8]
            {
                if (rows.empty())
                    return nullptr;
                float(*arr)[8] = (float(*)[8])PULSEG_ALLOC(sizeof(float[8]) * rows.size());
                std::memcpy(arr, rows.data(), sizeof(float[8]) * rows.size());
                return arr;
            };
            auto copy_rows4 = [](const std::vector<std::array<float, 4>> &rows) -> float(*)[4]
            {
                if (rows.empty())
                    return nullptr;
                float(*arr)[4] = (float(*)[4])PULSEG_ALLOC(sizeof(float[4]) * rows.size());
                std::memcpy(arr, rows.data(), sizeof(float[4]) * rows.size());
                return arr;
            };
            auto copy_rows3 = [](const std::vector<std::array<float, 3>> &rows) -> float(*)[3]
            {
                if (rows.empty())
                    return nullptr;
                float(*arr)[3] = (float(*)[3])PULSEG_ALLOC(sizeof(float[3]) * rows.size());
                std::memcpy(arr, rows.data(), sizeof(float[3]) * rows.size());
                return arr;
            };
            auto copy_rows2 = [](const std::vector<std::array<float, 2>> &rows) -> float(*)[2]
            {
                if (rows.empty())
                    return nullptr;
                float(*arr)[2] = (float(*)[2])PULSEG_ALLOC(sizeof(float[2]) * rows.size());
                std::memcpy(arr, rows.data(), sizeof(float[2]) * rows.size());
                return arr;
            };

            out->block_library = copy_rows7(b.block_rows);
            out->num_blocks = (int)b.block_rows.size();
            out->is_block_library_parsed = 1;

            out->rf_library = copy_rows10(b.rf_rows);
            out->rf_library_size = (int)b.rf_rows.size();
            out->is_rf_library_parsed = 1;
            if (!b.rf_use_tags.empty())
            {
                out->rf_use_tags = (int *)PULSEG_ALLOC(sizeof(int) * b.rf_use_tags.size());
                std::memcpy(out->rf_use_tags, b.rf_use_tags.data(), sizeof(int) * b.rf_use_tags.size());
            }

            out->grad_library = copy_rows7(b.grad_rows);
            out->grad_library_size = (int)b.grad_rows.size();
            out->is_grad_library_parsed = 1;

            out->adc_library = copy_rows8(b.adc_rows);
            out->adc_library_size = (int)b.adc_rows.size();
            out->is_adc_library_parsed = 1;

            out->extensions_library = copy_rows3(b.ext_rows);
            out->extensions_library_size = (int)b.ext_rows.size();
            out->is_extensions_library_parsed = 1;
            // Identity file-local-id <-> canonical-type LUT (see pulseq_adapter.cpp
            // design notes): this adapter writes canonical PULSEG_PULSEQ_EXT_*
            // values directly into extensions_library[i][0], so extension_map /
            // extension_lut just need to map each canonical type to itself.
            for (int t = 0; t < 8; ++t)
                out->extension_map[t] = t;
            out->extension_lut_size = 7;
            out->extension_lut = (int *)PULSEG_ALLOC(sizeof(int) * 8);
            for (int t = 0; t < 8; ++t)
                out->extension_lut[t] = t;

            out->trigger_library = copy_rows4(b.trigger_rows);
            out->trigger_library_size = (int)b.trigger_rows.size();

            out->rotation_quaternion_library = copy_rows4(b.rotation_rows);
            out->rotation_library_size = (int)b.rotation_rows.size();

            out->labelset_library = copy_rows2(b.labelset_rows);
            out->labelset_library_size = (int)b.labelset_rows.size();
            out->labelinc_library = copy_rows2(b.labelinc_rows);
            out->labelinc_library_size = (int)b.labelinc_rows.size();

            out->soft_delay_library = copy_rows4(b.soft_delay_rows);
            out->soft_delay_library_size = (int)b.soft_delay_rows.size();

            if (!b.rf_shim_rows.empty())
            {
                out->rf_shim_library = (pulseg__rf_shim_entry *)PULSEG_ALLOC(
                    sizeof(pulseg__rf_shim_entry) * b.rf_shim_rows.size());
                std::memcpy(out->rf_shim_library, b.rf_shim_rows.data(),
                            sizeof(pulseg__rf_shim_entry) * b.rf_shim_rows.size());
            }
            out->rf_shim_library_size = (int)b.rf_shim_rows.size();

            if (!b.shapes.empty())
            {
                out->shapes_library = (pulseg_shape_arbitrary *)PULSEG_ALLOC(
                    sizeof(pulseg_shape_arbitrary) * b.shapes.size());
                std::memcpy(out->shapes_library, b.shapes.data(),
                            sizeof(pulseg_shape_arbitrary) * b.shapes.size());
            }
            out->shapes_library_size = (int)b.shapes.size();
            out->is_shapes_library_parsed = 1;

            *out_ptr = out;
            return true;
        }

    } // namespace adapter
} // namespace pulseg
