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
        pulseqlib_optsInit(&opts, B0, max_grad, max_slew,
                           rf_raster_time, grad_raster_time,
                           adc_raster_time, block_duration_raster);
        pulseqlib_seqFileInit(seq, &opts);
        pulseqlib_optsFree(&opts);

        // Copy Python bytes into a buffer
        std::string buffer = seq_bytes;
        FMEMOPEN_HANDLE handle = FMEMOPEN_HANDLE_INIT;
        open_buffer_as_file(&handle, (char*)buffer.data(), buffer.size());
        if (!handle.f) {
            pulseqlib_seqFileFree(seq);
            FREE(seq);
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
            FREE(seq);
            seq = nullptr;
            throw std::runtime_error(std::string("Failed to parse sequence: ") + errMsg);
        }
    }

    ~_PulserverSeqFile() {
        if (seq) {
            pulseqlib_seqFileFree(seq);
            FREE(seq);
            seq = nullptr;
        }
    }
};

static py::tuple _get_unique_blocks(_PulserverSeqFile& seqfile, int index_min, int index_max) {
    if (!seqfile.seq) {
        throw std::runtime_error("SeqFile pointer is null");
    }

    const int numBlocks = seqfile.seq->numBlocks;

    int start = (index_min < 0) ? 0 : index_min;
    int end   = (index_max < 0) ? numBlocks : index_max;

    if (start < 0) start = 0;
    if (start > numBlocks) start = numBlocks;
    if (end < start) end = start;
    if (end > numBlocks) end = numBlocks;

    const int rangeCount = end - start;

    std::vector<int> block_durations_us(numBlocks > 0 ? (size_t)numBlocks : 0, -1);
    std::vector<int> unique_defs(rangeCount > 0 ? (size_t)rangeCount : 0);
    std::vector<int> unique_table(numBlocks > 0 ? (size_t)numBlocks : 0, -1);
    std::vector<int> pure_delay_block(numBlocks > 0 ? (size_t)numBlocks : 0, -1);

    int numPrep = 0;
    int numCooldown = 0;

    int k = 0;
    if (numBlocks > 0 && rangeCount > 0) {
        k = pulseqlib_getUniqueBlocks(
            seqfile.seq,
            unique_defs.data(),
            unique_table.data(),
            block_durations_us.data(),
            pure_delay_block.data(),
            &numPrep,
            &numCooldown,
            index_min,
            index_max
        );
        if (k < 0) k = 0;
        if (k > rangeCount) k = rangeCount;
        unique_defs.resize((size_t)k);
    } else {
        unique_defs.clear();
    }

    return py::make_tuple(unique_defs, unique_table, block_durations_us, pure_delay_block, numPrep, numCooldown);
}

static py::dict _find_tr_in_sequence(
    std::vector<int>& unique_block_table,
    std::vector<int>& block_durations_us,
    std::vector<int>& pure_delay_block,
    const int numPrep,
    const int numCooldown
) {
    pulseqlib_TRdescriptor trDesc;
    pulseqlib_Diagnostic diag;
    pulseqlib_diagnosticInit(&diag);
    
    const int n = (int)unique_block_table.size();
    
    py::dict result;
    
    if (n <= 0) {
        diag.code = PULSEQLIB_ERR_TR_NO_BLOCKS;
        result["success"] = false;
        result["tr_size"] = 0;
        result["num_trs"] = 0;
        result["degenerate_prep"] = 0;
        result["degenerate_cooldown"] = 0;
    } else {
        int code = pulseqlib_findTRInSequence(
            &trDesc,
            &diag,
            n,
            unique_block_table.data(), 
            block_durations_us.data(),
            pure_delay_block.data(),
            numPrep,
            numCooldown
        );
        
        result["success"] = PULSEQLIB_SUCCEEDED(code);
        result["tr_size"] = trDesc.trSize;
        result["num_trs"] = trDesc.numTRs;
        result["degenerate_prep"] = trDesc.degeneratePrep;
        result["degenerate_cooldown"] = trDesc.degenerateCooldown;
    }
    
    // Always include diagnostic info
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
    result["diagnostic"] = diagDict;
    
    return result;
}

static py::dict _find_segments_in_tr(
    _PulserverSeqFile& seqfile,
    const int trSize,
    const int numTRs,
    const int numPrepBlocks,
    const int numCooldownBlocks,
    const int degeneratePrep,
    const int degenerateCooldown,
    std::vector<int>& unique_block_table
) {
    if (!seqfile.seq) {
        throw std::runtime_error("SeqFile pointer is null");
    }

    // Build TR descriptor from input parameters
    pulseqlib_TRdescriptor trDesc;
    trDesc.trSize = trSize;
    trDesc.numTRs = numTRs;
    trDesc.numPrepBlocks = numPrepBlocks;
    trDesc.numPrepTRs = (numPrepBlocks > 0 && trSize > 0) ? (numPrepBlocks / trSize) : 0;
    trDesc.numCooldownBlocks = numCooldownBlocks;
    trDesc.numCooldownTRs = (numCooldownBlocks > 0 && trSize > 0) ? (numCooldownBlocks / trSize) : 0;
    trDesc.degeneratePrep = degeneratePrep;
    trDesc.degenerateCooldown = degenerateCooldown;

    // Maximum possible segments: one per block in TR + prep + cooldown
    const int maxSegments = trSize + numPrepBlocks + numCooldownBlocks;
    if (maxSegments <= 0) {
        py::dict result;
        result["unique_segments"] = py::list();
        result["prep_segment_table"] = std::vector<int>();
        result["main_segment_table"] = std::vector<int>();
        result["cooldown_segment_table"] = std::vector<int>();
        return result;
    }

    // Allocate output arrays
    std::vector<pulseqlib_TRsegment> trSegments(maxSegments);
    pulseqlib_SegmentTableResult segmentTable = {0};
    pulseqlib_Diagnostic diag;
    pulseqlib_diagnosticInit(&diag);

    // Initialize segment pointers to NULL
    for (int i = 0; i < maxSegments; ++i) {
        trSegments[i].uniqueBlockIndices = nullptr;
    }

    // Call the C function - NOTE: argument order must match the declaration
    int numUniqueSegments = pulseqlib_findSegmentsInTR(
        seqfile.seq,
        trSegments.data(),
        &segmentTable,
        &diag,              // diagnostic comes BEFORE trDesc
        &trDesc,
        unique_block_table.data()
    );

    // Convert unique segments to Python-friendly format
    py::list uniqueSegmentsList;
    for (int i = 0; i < numUniqueSegments; ++i) {
        py::dict segmentDict;
        segmentDict["start_block"] = trSegments[i].startBlock;
        segmentDict["num_blocks"] = trSegments[i].numBlocks;
        
        std::vector<int> indices;
        if (trSegments[i].uniqueBlockIndices && trSegments[i].numBlocks > 0) {
            indices.assign(
                trSegments[i].uniqueBlockIndices,
                trSegments[i].uniqueBlockIndices + trSegments[i].numBlocks
            );
        }
        segmentDict["unique_block_indices"] = indices;
        uniqueSegmentsList.append(segmentDict);
    }

    // Convert segment tables to vectors
    std::vector<int> prepTable;
    std::vector<int> mainTable;
    std::vector<int> cooldownTable;

    if (segmentTable.prepSegmentTable && segmentTable.numPrepSegments > 0) {
        prepTable.assign(
            segmentTable.prepSegmentTable,
            segmentTable.prepSegmentTable + segmentTable.numPrepSegments
        );
    }
    if (segmentTable.mainSegmentTable && segmentTable.numMainSegments > 0) {
        mainTable.assign(
            segmentTable.mainSegmentTable,
            segmentTable.mainSegmentTable + segmentTable.numMainSegments
        );
    }
    if (segmentTable.cooldownSegmentTable && segmentTable.numCooldownSegments > 0) {
        cooldownTable.assign(
            segmentTable.cooldownSegmentTable,
            segmentTable.cooldownSegmentTable + segmentTable.numCooldownSegments
        );
    }

    // Free allocated memory
    for (int i = 0; i < numUniqueSegments; ++i) {
        if (trSegments[i].uniqueBlockIndices) {
            FREE(trSegments[i].uniqueBlockIndices);
        }
    }
    pulseqlib_segmentTableResultFree(&segmentTable);

    // Build result dictionary
    py::dict result;
    result["unique_segments"] = uniqueSegmentsList;
    result["prep_segment_table"] = prepTable;
    result["main_segment_table"] = mainTable;
    result["cooldown_segment_table"] = cooldownTable;

    return result;
}

PYBIND11_MODULE(_pulseqlib_wrapper, m) {
    py::class_<_PulserverSeqFile>(m, "_PulserverSeqFile")
        .def(py::init<py::bytes, float, float, float, float, float, float, float>())
        ;
    
    m.def("_get_unique_blocks",
          &_get_unique_blocks,
          py::arg("seqfile"),
          py::arg("index_min") = -1,
          py::arg("index_max") = -1);

    m.def("_find_tr_in_sequence",
      &_find_tr_in_sequence,
      py::arg("unique_block_table"),
      py::arg("block_durations_us"),
      py::arg("pure_delay_block"),
      py::arg("numPrep"),
      py::arg("numCooldown"));

    m.def("_find_segments_in_tr",
      &_find_segments_in_tr,
      py::arg("seqfile"),
      py::arg("trSize"),
      py::arg("numTRs"),
      py::arg("numPrepBlocks"),
      py::arg("numCooldownBlocks"),
      py::arg("degeneratePrep"),
      py::arg("degenerateCooldown"),
      py::arg("unique_block_table"),
      R"pbdoc(
        Get segment definitions within a TR.

        Parameters
        ----------
        seqfile : _PulserverSeqFile
            The sequence file object.
        trSize : int
            Size of the TR in number of blocks.
        numTRs : int
            Number of TRs in the sequence.
        numPrepBlocks : int
            Number of preparation blocks before imaging.
        numCooldownBlocks : int
            Number of cooldown blocks after imaging.
        degeneratePrep : int
            Non-zero if preparation blocks are degenerate (identical to main TR).
        degenerateCooldown : int
            Non-zero if cooldown blocks are degenerate (identical to main TR).
        unique_block_table : list[int]
            Array mapping each block to its unique definition index.

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
