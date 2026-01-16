#ifndef PULSEQLIB_METHODS_H
#define PULSEQLIB_METHODS_H

#include <math.h>
#include <stdlib.h>

#include "pulseqlib.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef DETECT_REAL_RF
    #define  DETECT_REAL_RF 1
#endif 

/** 
 * Default ALLOC to malloc if it's not already defined
 *
 * Users are encouraged to replace with vendor-specific implementations if needed:
 * 
 * // pulseqlib_vendor_methods.h (vendor-specific header)
 * 
 * #include "my_vendor_library.h"  // This contains the definition of MyVendorAlloc
 * 
 * // Override ALLOC to use MyVendorAlloc in the vendor environment
 * #define ALLOC(size) MyVendorAlloc(size)  // Replaces malloc with MyVendorAlloc
 * 
 * #include "pulseqlib_methods.h"  // Now include the vendor-agnostic pulseqlib_vendor_methods.h with the overridden ALLOC
 * 
 * // Other vendor-specific declarations can go here
*/
#ifndef ALLOC
    #define ALLOC(size) malloc(size)
#endif

/** 
 * Default FREE to free if it's not already defined
 *
 * Users are encouraged to replace with vendor-specific implementations if needed:
 * 
 * // pulseqlib_vendor_methods.h (vendor-specific header)
 * 
 * #include "my_vendor_library.h"  // This contains the definition of MyVendorAlloc
 * 
 * // Override FREE to use MyVendorFree in the vendor environment
 * #define FREE(ptr) MyVendorFree(ptr)  // Replaces malloc with MyVendorAlloc
 * 
 * #include "pulseqlib_methods.h"  // Now include the vendor-agnostic pulseqlib_vendor_methods.h with the overridden FREE
 * 
 * // Other vendor-specific declarations can go here
*/
#ifndef FREE
  #define FREE(ptr) free(ptr)
#endif

/* Constructor, destructor and reset */
void pulseqlib_optsInit(pulseqlib_Opts* opts, float B0, float max_grad, float max_slew, float rf_raster_time, float grad_raster_time, float adc_raster_time, float block_duration_raster);
void pulseqlib_optsFree(pulseqlib_Opts* opts);

void pulseqlib_seqFileInit(pulseqlib_SeqFile* seq, const pulseqlib_Opts* opts);
void pulseqlib_seqFileFree(pulseqlib_SeqFile* seq);

void pulseqlib_seqBlockInit(pulseqlib_SeqBlock* block);
void pulseqlib_seqBlockFree(pulseqlib_SeqBlock* block);

/* Parsing Sequence */
int pulseqlib_readSeq(pulseqlib_SeqFile* seq, const char* filePath);
int pulseqlib_readSeqFromBuffer(pulseqlib_SeqFile* seq, FILE* f);

/* Getters - to mimic OOP *outputs = obj.func(input), we do func(obj, *outputs, *inputs) */
void pulseqlib_getBlockStatic(const pulseqlib_SeqFile* seq, pulseqlib_SeqBlock* block, const int blockIndex);
void pulseqlib_getBlockDynamic(const pulseqlib_SeqFile* seq, pulseqlib_BlockDynamic* dynamic, const int blockIndex);
void pulseqlib_getBlockDynamicWithoutExtensions(const pulseqlib_SeqFile* seq, pulseqlib_BlockDynamic* dynamic, const int blockIndex);
void pulseqlib_getBlockLabels(const pulseqlib_SeqFile* seq, pulseqlib_BlockLabels* labels, const int blockIndex);
void pulseqlib_getBlock(const pulseqlib_SeqFile* seq, pulseqlib_SeqBlock* block, const int blockIndex);
float pulseqlib_getGradLibraryMaxAmplitude(const pulseqlib_SeqFile* seq);

/**
 * @brief Get human-readable error message for an error code.
 * 
 * @param code The error code.
 * @return Static string describing the error. Never returns NULL.
 */
const char* pulseqlib_getErrorMessage(int code);

/**
 * @brief Get a hint message suggesting how to fix the error.
 * 
 * @param code The error code.
 * @return Static string with suggestions. Never returns NULL.
 */
const char* pulseqlib_getErrorHint(int code);

/**
 * @brief Initialize diagnostic struct to default values.
 */
void pulseqlib_diagnosticInit(pulseqlib_Diagnostic* diag);

/* Segment-specific functions */
int pulseqlib_getUniqueBlocks(
  const pulseqlib_SeqFile* seq, 
  int* uniqueBlockDefs, 
  int* uniqueBlockTable,
  int* blockDurations_us,
  int* pureDelayBlock,
  int* numPrep,
  int* numCooldown, 
  int index_min, 
  int index_max
);

int pulseqlib_findTRInSequence(
  pulseqlib_TRdescriptor* trDesc,
  pulseqlib_Diagnostic* diag,
  int numBlocks,
  int* uniqueBlockTable,
  int* blockDurations_us,
  int* pureDelayBlock,
  int numPrep,
  int numCooldown
);

int pulseqlib_findSegmentsInTR(
  const pulseqlib_SeqFile* seq, 
  pulseqlib_TRsegment* trSegments,
  pulseqlib_SegmentTableResult* segmentTable,
  pulseqlib_Diagnostic* diag,
  const pulseqlib_TRdescriptor* trDesc,
  const int* uniqueBlockTable
);

void pulseqlib_segmentTableResultFree(pulseqlib_SegmentTableResult* result);

#ifdef __cplusplus
}
#endif

#endif /* PULSEQLIB_METHODS_H */
