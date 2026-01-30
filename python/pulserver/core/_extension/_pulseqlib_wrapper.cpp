#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#include <windows.h>
#endif

extern "C" {
#include "pulseqlib.h"
}
#include "pulseqlib_methods.h"

namespace py = pybind11;

typedef struct FMEMOPEN_HANDLE {
#ifdef _WIN32
    char tmp_file[MAX_PATH];
#endif
    FILE* f;
} FMEMOPEN_HANDLE;

#define FMEMOPEN_HANDLE_INIT {0}

// Cross-platform open-memory-as-file
inline void open_buffer_as_file(FMEMOPEN_HANDLE* handle, char* buffer, size_t size) {
#ifdef _WIN32
    // Create a temporary file
    char tmp_path[MAX_PATH];
    if (!GetTempPathA(MAX_PATH, tmp_path)) return;
    if (!GetTempFileNameA(tmp_path, "psq", 0, handle->tmp_file)) return;
    handle->f = fopen(handle->tmp_file, "wb+");
    if (!handle->f) return;
    fwrite(buffer, 1, size, handle->f);
    rewind(handle->f);
#else
    handle->f = fmemopen(buffer, size, "r");
#endif
}

class _PulserverSeqFile {
public:
    pulseqlib_SeqFile* seq;

    _PulserverSeqFile(const py::bytes& seq_bytes,
                      float gamma,
                      float B0,
                      float max_grad,
                      float max_slew,
                      float rf_raster_time,
                      float grad_raster_time,
                      float adc_raster_time,
                      float block_duration_raster) {
        seq = (pulseqlib_SeqFile*)ALLOC(sizeof(pulseqlib_SeqFile));
        if (!seq) {
            throw std::runtime_error("Failed to allocate SeqFile");
        }

        // Initialize SeqFile with options
        pulseqlib_Opts opts;
        pulseqlib_optsInit(&opts, gamma, B0, max_grad, max_slew,
                           rf_raster_time * 1e6f, grad_raster_time * 1e6f,
                           adc_raster_time * 1e6f, block_duration_raster * 1e6f);
        pulseqlib_seqFileInit(seq, &opts);
        pulseqlib_optsFree(&opts);

        // Copy Python bytes into a buffer
        std::string buffer = seq_bytes;
        FMEMOPEN_HANDLE handle = FMEMOPEN_HANDLE_INIT;
        open_buffer_as_file(&handle, (char*)buffer.data(), buffer.size());
        if (!handle.f) {
            pulseqlib_seqFileFree(seq);
            seq = nullptr;
            throw std::runtime_error("Failed to open buffer as file");
        }
        
        int result = pulseqlib_readSeqFromBuffer(seq, handle.f);
        fclose(handle.f);
#ifdef _WIN32
        DeleteFileA(handle.tmp_file);
#endif

        if (PULSEQLIB_FAILED(result)) {
            const char* errMsg = pulseqlib_getErrorMessage(result);
            pulseqlib_seqFileFree(seq);
            seq = nullptr;
            throw std::runtime_error(std::string("Failed to parse sequence: ") + errMsg);
        }
    }

    ~_PulserverSeqFile() {
        if (seq) {
            pulseqlib_seqFileFree(seq);
            seq = nullptr;
        }
    }
};

static py::dict _get_unique_blocks(_PulserverSeqFile& seqfile) {
    if (!seqfile.seq) {
        throw std::runtime_error("SeqFile pointer is null");
    }

    // Initialize sequence descriptor with memset (C++ compatible)
    pulseqlib_SequenceDescriptor seqDesc;
    memset(&seqDesc, 0, sizeof(seqDesc));

    // Call getUniqueBlocks - it handles all internal allocation
    int result = pulseqlib_getUniqueBlocks(seqfile.seq, &seqDesc);

    py::dict output;

    if (PULSEQLIB_FAILED(result)) {
        output["success"] = false;
        output["error"] = pulseqlib_getErrorMessage(result);
        output["error_code"] = result;
        return output;
    }

    output["success"] = true;
    output["num_prep_blocks"] = seqDesc.numPrepBlocks;
    output["num_cooldown_blocks"] = seqDesc.numCooldownBlocks;
    output["num_blocks"] = seqDesc.numBlocks;
    output["num_unique_blocks"] = seqDesc.numUniqueBlocks;

    // Build block_table list
    py::list blockTableList;
    for (int i = 0; i < seqDesc.numBlocks; ++i) {
        py::dict entry;
        entry["id"] = seqDesc.blockTable[i].ID;
        entry["pure_delay_flag"] = seqDesc.blockTable[i].pureDelayFlag;
        entry["adc_id"] = seqDesc.blockTable[i].adcID;
        entry["trigger_id"] = seqDesc.blockTable[i].triggerID;
        entry["rotation_id"] = seqDesc.blockTable[i].rotationID;
        entry["norot_flag"] = seqDesc.blockTable[i].norotFlag;
        entry["nopos_flag"] = seqDesc.blockTable[i].noposFlag;
        entry["pmc_flag"] = seqDesc.blockTable[i].pmcFlag;
        entry["nav_flag"] = seqDesc.blockTable[i].navFlag;
        blockTableList.append(entry);
    }
    output["block_table"] = blockTableList;

    // Build block_definitions list
    py::list blockDefsList;
    for (int i = 0; i < seqDesc.numUniqueBlocks; ++i) {
        py::dict entry;
        entry["id"] = seqDesc.blockDefinitions[i].ID;
        entry["duration_us"] = seqDesc.blockDefinitions[i].duration_us;
        entry["rf_id"] = seqDesc.blockDefinitions[i].rfID;
        entry["gx_id"] = seqDesc.blockDefinitions[i].gxID;
        entry["gy_id"] = seqDesc.blockDefinitions[i].gyID;
        entry["gz_id"] = seqDesc.blockDefinitions[i].gzID;
        blockDefsList.append(entry);
    }
    output["block_definitions"] = blockDefsList;

    // RF definitions and table
    output["num_unique_rfs"] = seqDesc.numUniqueRFs;
    py::list rfDefsList;
    for (int i = 0; i < seqDesc.numUniqueRFs; ++i) {
        py::dict entry;
        entry["id"] = seqDesc.rfDefinitions[i].ID;
        entry["mag_shape_id"] = seqDesc.rfDefinitions[i].magShapeID;
        entry["phase_shape_id"] = seqDesc.rfDefinitions[i].phaseShapeID;
        entry["time_shape_id"] = seqDesc.rfDefinitions[i].timeShapeID;
        entry["delay"] = seqDesc.rfDefinitions[i].delay;
#ifdef IS_GEHC
        if (IS_GEHC) {
            entry["num_samples"] = seqDesc.rfDefinitions[i].numSamples;
            entry["flip_angle"] = seqDesc.rfDefinitions[i].flipAngle;
            entry["max_amplitude"] = seqDesc.rfDefinitions[i].maxAmplitude;
            entry["duration_us"] = seqDesc.rfDefinitions[i].duration_us;
            entry["area"] = seqDesc.rfDefinitions[i].area;
            entry["abswidth"] = seqDesc.rfDefinitions[i].abswidth;
            entry["effwidth"] = seqDesc.rfDefinitions[i].effwidth;
            entry["dtycyc"] = seqDesc.rfDefinitions[i].dtycyc;
            entry["maxpw"] = seqDesc.rfDefinitions[i].maxpw;
            entry["isodelay_us"] = seqDesc.rfDefinitions[i].isodelay_us;
            entry["bandwidth"] = seqDesc.rfDefinitions[i].bandwidth;
        }
#endif
        rfDefsList.append(entry);
    }
    output["rf_definitions"] = rfDefsList;

    py::list rfTableList;
    for (int i = 0; i < seqDesc.rfTableSize; ++i) {
        py::dict entry;
        entry["id"] = seqDesc.rfTable[i].ID;
        entry["amplitude"] = seqDesc.rfTable[i].amplitude;
        entry["freq_offset"] = seqDesc.rfTable[i].freqOffset;
        entry["phase_offset"] = seqDesc.rfTable[i].phaseOffset;
        rfTableList.append(entry);
    }
    output["rf_table"] = rfTableList;

    // Gradient definitions and table
    output["num_unique_grads"] = seqDesc.numUniqueGrads;
    py::list gradDefsList;
    for (int i = 0; i < seqDesc.numUniqueGrads; ++i) {
        py::dict entry;
        entry["id"] = seqDesc.gradDefinitions[i].ID;
        entry["type"] = seqDesc.gradDefinitions[i].type;
        entry["delay"] = seqDesc.gradDefinitions[i].delay;

        if (seqDesc.gradDefinitions[i].type == 0) {
            // Trapezoid: timing defined by rise/flat/fall
            entry["rise_time"] = seqDesc.gradDefinitions[i].riseTimeOrUnused;
            entry["flat_time"] = seqDesc.gradDefinitions[i].flatTimeOrUnused;
            entry["fall_time"] = seqDesc.gradDefinitions[i].fallTimeOrNumUncompressedSamples;
        } else {
            // Arbitrary/Extended: timing defined by num_samples and time_shape_id
            entry["num_shots"] = seqDesc.gradDefinitions[i].numShots;
            entry["num_samples"] = seqDesc.gradDefinitions[i].fallTimeOrNumUncompressedSamples;
            entry["time_shape_id"] = seqDesc.gradDefinitions[i].unusedOrTimeShapeID;
        }
#ifdef IS_GEHC
        if (IS_GEHC) {
            // Export arrays for multi-shot gradients
            std::vector<int> shotShapeIDs(seqDesc.gradDefinitions[i].shotShapeIDs, 
                                           seqDesc.gradDefinitions[i].shotShapeIDs + seqDesc.gradDefinitions[i].numShots);
            std::vector<float> maxAmplitudes(seqDesc.gradDefinitions[i].maxAmplitude,
                                              seqDesc.gradDefinitions[i].maxAmplitude + seqDesc.gradDefinitions[i].numShots);
            std::vector<float> slewRates(seqDesc.gradDefinitions[i].slewRate,
                                          seqDesc.gradDefinitions[i].slewRate + seqDesc.gradDefinitions[i].numShots);
            std::vector<float> energies(seqDesc.gradDefinitions[i].energy,
                                         seqDesc.gradDefinitions[i].energy + seqDesc.gradDefinitions[i].numShots);
            std::vector<float> firstValues(seqDesc.gradDefinitions[i].firstValue,
                                            seqDesc.gradDefinitions[i].firstValue + seqDesc.gradDefinitions[i].numShots);
            std::vector<float> lastValues(seqDesc.gradDefinitions[i].lastValue,
                                           seqDesc.gradDefinitions[i].lastValue + seqDesc.gradDefinitions[i].numShots);
            entry["max_amplitude"] = maxAmplitudes;
            entry["slew_rate"] = slewRates;
            entry["energy"] = energies;
            if (seqDesc.gradDefinitions[i].type != 0) {
                entry["shot_shape_ids"] = shotShapeIDs;
                entry["first_value"] = firstValues;
                entry["last_value"] = lastValues;
            }
        }
#endif
        gradDefsList.append(entry);
    }
    output["grad_definitions"] = gradDefsList;

    py::list gradTableList;
    for (int i = 0; i < seqDesc.gradTableSize; ++i) {
        py::dict entry;
        entry["id"] = seqDesc.gradTable[i].ID;
        entry["amplitude"] = seqDesc.gradTable[i].amplitude;
        if (seqDesc.gradDefinitions[seqDesc.gradTable[i].ID].type != 0) {
            entry["shot_index"] = seqDesc.gradTable[i].shotIndex;
        }
        gradTableList.append(entry);
    }
    output["grad_table"] = gradTableList;

    // ADC definitions and table
    output["num_unique_adcs"] = seqDesc.numUniqueADCs;
    py::list adcDefsList;
    for (int i = 0; i < seqDesc.numUniqueADCs; ++i) {
        py::dict entry;
        entry["id"] = seqDesc.adcDefinitions[i].ID;
        entry["num_samples"] = seqDesc.adcDefinitions[i].numSamples;
        entry["dwell_time"] = seqDesc.adcDefinitions[i].dwellTime;
        entry["delay"] = seqDesc.adcDefinitions[i].delay;
        adcDefsList.append(entry);
    }
    output["adc_definitions"] = adcDefsList;

    py::list adcTableList;
    for (int i = 0; i < seqDesc.adcTableSize; ++i) {
        py::dict entry;
        entry["id"] = seqDesc.adcTable[i].ID;
        entry["freq_offset"] = seqDesc.adcTable[i].freqOffset;
        entry["phase_offset"] = seqDesc.adcTable[i].phaseOffset;
        adcTableList.append(entry);
    }
    output["adc_table"] = adcTableList;

    // TR descriptor
    py::dict trDesc;
    trDesc["num_prep_blocks"] = seqDesc.numPrepBlocks;
    trDesc["num_cooldown_blocks"] = seqDesc.numCooldownBlocks;
    output["tr_descriptor"] = trDesc;

    // Free all allocated memory in seqDesc
    pulseqlib_sequenceDescriptorFree(&seqDesc);

    return output;
}

static py::dict _find_tr_in_sequence(_PulserverSeqFile& seqfile) {
    if (!seqfile.seq) {
        throw std::runtime_error("SeqFile pointer is null");
    }

    // Initialize sequence descriptor with memset (C++ compatible)
    pulseqlib_SequenceDescriptor seqDesc;
    memset(&seqDesc, 0, sizeof(seqDesc));

    // Step 1: Call getUniqueBlocks
    int result = pulseqlib_getUniqueBlocks(seqfile.seq, &seqDesc);
    
    py::dict output;
    
    if (PULSEQLIB_FAILED(result)) {
        output["success"] = false;
        output["error"] = pulseqlib_getErrorMessage(result);
        output["error_code"] = result;
        return output;
    }

    // Step 2: Call findTRInSequence
    pulseqlib_Diagnostic diag;
    pulseqlib_diagnosticInit(&diag);
    
    int code = pulseqlib_findTRInSequence(&seqDesc, &diag);
    
    output["success"] = PULSEQLIB_SUCCEEDED(code);
    output["tr_size"] = seqDesc.trDescriptor.trSize;
    output["num_trs"] = seqDesc.trDescriptor.numTRs;
    output["degenerate_prep"] = seqDesc.trDescriptor.degeneratePrep;
    output["degenerate_cooldown"] = seqDesc.trDescriptor.degenerateCooldown;
    output["num_prep_blocks"] = seqDesc.trDescriptor.numPrepBlocks;
    output["num_prep_trs"] = seqDesc.trDescriptor.numPrepTRs;
    output["num_cooldown_blocks"] = seqDesc.trDescriptor.numCooldownBlocks;
    output["num_cooldown_trs"] = seqDesc.trDescriptor.numCooldownTRs;
    
    // Include diagnostic info
    py::dict diagDict;
    diagDict["code"] = diag.code;
    diagDict["message"] = pulseqlib_getErrorMessage(diag.code);
    diagDict["hint"] = pulseqlib_getErrorHint(diag.code);
    diagDict["block_index"] = diag.blockIndex;
    diagDict["channel"] = diag.channel;
    diagDict["num_unique_blocks"] = diag.numUniqueBlocks;
    diagDict["imaging_region_length"] = diag.imagingRegionLength;
    diagDict["candidate_pattern_length"] = diag.candidatePatternLength;
    diagDict["mismatch_position"] = diag.mismatchPosition;
    output["diagnostic"] = diagDict;
    
    // Free allocated memory
    pulseqlib_sequenceDescriptorFree(&seqDesc);
    
    return output;
}

static py::dict _find_segments_in_tr(_PulserverSeqFile& seqfile) {
    if (!seqfile.seq) {
        throw std::runtime_error("SeqFile pointer is null");
    }

    // Initialize sequence descriptor with memset (C++ compatible)
    pulseqlib_SequenceDescriptor seqDesc;
    memset(&seqDesc, 0, sizeof(seqDesc));

    // Step 1: Call getUniqueBlocks
    int result = pulseqlib_getUniqueBlocks(seqfile.seq, &seqDesc);
    
    py::dict output;
    
    if (PULSEQLIB_FAILED(result)) {
        output["success"] = false;
        output["error"] = pulseqlib_getErrorMessage(result);
        output["error_code"] = result;
        return output;
    }

    // Step 2: Call findTRInSequence
    pulseqlib_Diagnostic diag;
    pulseqlib_diagnosticInit(&diag);
    
    int code = pulseqlib_findTRInSequence(&seqDesc, &diag);
    
    if (PULSEQLIB_FAILED(code)) {
        output["success"] = false;
        output["error"] = pulseqlib_getErrorMessage(code);
        output["error_code"] = code;
        
        py::dict diagDict;
        diagDict["code"] = diag.code;
        diagDict["message"] = pulseqlib_getErrorMessage(diag.code);
        diagDict["hint"] = pulseqlib_getErrorHint(diag.code);
        output["diagnostic"] = diagDict;
        
        pulseqlib_sequenceDescriptorFree(&seqDesc);
        return output;
    }

    // Step 3: Call findSegmentsInTR
    int numUniqueSegments = pulseqlib_findSegmentsInTR(seqfile.seq, &seqDesc, &diag);

    if (numUniqueSegments == 0 && PULSEQLIB_FAILED(diag.code)) {
        output["success"] = false;
        output["unique_segments"] = py::list();
        output["prep_segment_table"] = std::vector<int>();
        output["main_segment_table"] = std::vector<int>();
        output["cooldown_segment_table"] = std::vector<int>();
        output["error_code"] = diag.code;
        output["error"] = pulseqlib_getErrorMessage(diag.code);
        pulseqlib_sequenceDescriptorFree(&seqDesc);
        return output;
    }

    output["success"] = true;

    // Convert unique segments to Python-friendly format
    py::list uniqueSegmentsList;
    for (int i = 0; i < numUniqueSegments; ++i) {
        py::dict segmentDict;
        segmentDict["start_block"] = seqDesc.segmentDefinitions[i].startBlock;
        segmentDict["num_blocks"] = seqDesc.segmentDefinitions[i].numBlocks;
        
        std::vector<int> indices;
        if (seqDesc.segmentDefinitions[i].uniqueBlockIndices && 
            seqDesc.segmentDefinitions[i].numBlocks > 0) {
            indices.assign(
                seqDesc.segmentDefinitions[i].uniqueBlockIndices,
                seqDesc.segmentDefinitions[i].uniqueBlockIndices + seqDesc.segmentDefinitions[i].numBlocks
            );
        }
        segmentDict["unique_block_indices"] = indices;
        uniqueSegmentsList.append(segmentDict);
    }

    // Convert segment tables to vectors
    std::vector<int> prepTable;
    std::vector<int> mainTable;
    std::vector<int> cooldownTable;

    if (seqDesc.segmentTable.prepSegmentTable && seqDesc.segmentTable.numPrepSegments > 0) {
        prepTable.assign(
            seqDesc.segmentTable.prepSegmentTable,
            seqDesc.segmentTable.prepSegmentTable + seqDesc.segmentTable.numPrepSegments
        );
    }
    if (seqDesc.segmentTable.mainSegmentTable && seqDesc.segmentTable.numMainSegments > 0) {
        mainTable.assign(
            seqDesc.segmentTable.mainSegmentTable,
            seqDesc.segmentTable.mainSegmentTable + seqDesc.segmentTable.numMainSegments
        );
    }
    if (seqDesc.segmentTable.cooldownSegmentTable && seqDesc.segmentTable.numCooldownSegments > 0) {
        cooldownTable.assign(
            seqDesc.segmentTable.cooldownSegmentTable,
            seqDesc.segmentTable.cooldownSegmentTable + seqDesc.segmentTable.numCooldownSegments
        );
    }

    // Build result dictionary
    output["unique_segments"] = uniqueSegmentsList;
    output["prep_segment_table"] = prepTable;
    output["main_segment_table"] = mainTable;
    output["cooldown_segment_table"] = cooldownTable;

    // Free all allocated memory
    pulseqlib_sequenceDescriptorFree(&seqDesc);

    return output;
}

PYBIND11_MODULE(_pulseqlib_wrapper, m) {
    py::class_<_PulserverSeqFile>(m, "_PulserverSeqFile")
        .def(py::init<py::bytes, float, float, float, float, float, float, float, float>())
        ;
    
    m.def("_get_unique_blocks",
          &_get_unique_blocks,
          py::arg("seqfile"),
          R"pbdoc(
            Get unique blocks and event deduplication info from a sequence.
            
            Returns a dict with:
            - num_prep_blocks: number of preparation blocks
            - num_cooldown_blocks: number of cooldown blocks
            - num_blocks: total number of blocks
            - num_unique_blocks: number of unique block definitions
            - block_table: list of per-block info (id, pure_delay_flag, adc_id, trigger_id, rotation_id, rfshim_id, norot_flag, nopos_flag)
            - block_definitions: list of unique block definitions (id, duration_us, rf_id, gx_id, gy_id, gz_id)
            - rf_definitions, rf_table: RF event deduplication
            - grad_definitions, grad_table: Gradient event deduplication  
            - adc_definitions, adc_table: ADC event deduplication
            - tr_descriptor: TR structure info
          )pbdoc");

    m.def("_find_tr_in_sequence",
          &_find_tr_in_sequence,
          py::arg("seqfile"),
          R"pbdoc(
            Find TR pattern in a sequence.
            
            Internally calls getUniqueBlocks then findTRInSequence.
            
            Parameters
            ----------
            seqfile : _PulserverSeqFile
                The sequence file object.
                
            Returns
            -------
            dict
                Dictionary with TR descriptor fields and diagnostic info.
          )pbdoc");

    m.def("_find_segments_in_tr",
          &_find_segments_in_tr,
          py::arg("seqfile"),
          R"pbdoc(
            Get segment definitions within a TR.
            
            Internally calls getUniqueBlocks, findTRInSequence, then findSegmentsInTR.

            Parameters
            ----------
            seqfile : _PulserverSeqFile
                The sequence file object.

            Returns
            -------
            dict
                Dictionary with keys:
                - 'unique_segments': List of dicts with 'start_block', 'num_blocks', 'unique_block_indices'
                - 'prep_segment_table': List mapping prep segments to unique segment IDs
                - 'main_segment_table': List mapping main TR segments to unique segment IDs  
                - 'cooldown_segment_table': List mapping cooldown segments to unique segment IDs
          )pbdoc");
}
