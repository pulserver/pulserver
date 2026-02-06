#ifndef PULSEQLIB_METHODS_H
#define PULSEQLIB_METHODS_H

#include <math.h>
#include <stdlib.h>

#include "pulseqlib.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SIEMENS 1
#define GEHC 2
#define PHILIPS 3
#define UNITED_IMAGING 4
#define BRUKER 5

#ifndef VENDOR
#define VENDOR GEHC
#endif

#if VENDOR == GEHC
#define  DETECT_REAL_RF 1
#else
#define  DETECT_REAL_RF 0
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
void pulseqlib_optsInit(pulseqlib_Opts* opts, float gamma, float B0, float max_grad, float max_slew, float rf_raster_time, float grad_raster_time, float adc_raster_time, float block_duration_raster);
void pulseqlib_optsFree(pulseqlib_Opts* opts);

void pulseqlib_seqFileInit(pulseqlib_SeqFile* seq, const pulseqlib_Opts* opts);
void pulseqlib_seqFileFree(pulseqlib_SeqFile* seq);
void pulseqlib_seqFileCollectionFree(pulseqlib_SeqFileCollection* collection);

void pulseqlib_seqBlockInit(pulseqlib_SeqBlock* block);
void pulseqlib_seqBlockFree(pulseqlib_SeqBlock* block);

/* Parsing Sequence */
int pulseqlib_readSeq(pulseqlib_SeqFile* seq, const char* filePath);
int pulseqlib_readSeqFromBuffer(pulseqlib_SeqFile* seq, FILE* f);
int pulseqlib_readSeqCollection(pulseqlib_SeqFileCollection* collection, const char* firstFilePath, const pulseqlib_Opts* opts);

/* Getters - to mimic OOP *outputs = obj.func(input), we do func(obj, *outputs, *inputs) */
void pulseqlib_getBlock(const pulseqlib_SeqFile* seq, pulseqlib_SeqBlock* block, const int blockIndex);
float pulseqlib_getGradLibraryMaxAmplitude(const pulseqlib_SeqFile* seq);

const char* pulseqlib_getErrorMessage(int code);

const char* pulseqlib_getErrorHint(int code);

void pulseqlib_diagnosticInit(pulseqlib_Diagnostic* diag);

/* Segment-specific functions */
int pulseqlib_getUniqueBlocks(const pulseqlib_SeqFile* seq, pulseqlib_SequenceDescriptor* seqDesc);

int pulseqlib_findTRInSequence(
  pulseqlib_SequenceDescriptor* seqDesc,
  pulseqlib_Diagnostic* diag
);

int pulseqlib_findSegmentsInTR(
  const pulseqlib_SeqFile* seq,
  pulseqlib_SequenceDescriptor* seqDesc,
  pulseqlib_Diagnostic* diag
);

int pulseqlib_getCollectionDescriptors(
    const pulseqlib_SeqFileCollection* collection,
    pulseqlib_SequenceDescriptorCollection* descCollection,
    pulseqlib_Diagnostic* diag);

void pulseqlib_segmentTableResultFree(pulseqlib_SegmentTableResult* result);
void pulseqlib_sequenceDescriptorFree(pulseqlib_SequenceDescriptor* seqDesc);
void pulseqlib_sequenceDescriptorCollectionFree(pulseqlib_SequenceDescriptorCollection* descCollection);

/**
 * @brief Get maximum number of samples across all unique ADCs in collection.
 * 
 * @param descCollection Sequence descriptor collection
 * @return Maximum number of samples, or 0 if no ADCs defined
 */
int pulseqlib_getMaxADCSamples(const pulseqlib_SequenceDescriptorCollection* descCollection);

/**
 * @brief Get dwell time for a specific ADC by global ADC index.
 * 
 * @param descCollection Sequence descriptor collection
 * @param adcIdx Global ADC index (0-based, across all subsequences)
 * @return Dwell time in nanoseconds, or 0 if invalid index
 */
int pulseqlib_getADCDwell(const pulseqlib_SequenceDescriptorCollection* descCollection, int adcIdx);

/**
 * @brief Get number of samples for a specific ADC by global ADC index.
 * 
 * @param descCollection Sequence descriptor collection
 * @param adcIdx Global ADC index (0-based, across all subsequences)
 * @return Number of samples, or 0 if invalid index
 */
int pulseqlib_getADCNumSamples(const pulseqlib_SequenceDescriptorCollection* descCollection, int adcIdx);

void pulseqlib_trGradientWaveformsFree(pulseqlib_TRGradientWaveforms* waveforms);
int pulseqlib_getTRGradientWaveforms(
    const pulseqlib_SequenceDescriptor* seqDesc,
    pulseqlib_TRGradientWaveforms* waveforms,
    pulseqlib_Diagnostic* diag);

void pulseqlib_trAcousticSpectraFree(pulseqlib_TRAcousticSpectra* spectra);
int pulseqlib_getTRAcousticSpectra(
    const pulseqlib_TRGradientWaveforms* waveforms,
    float gradRasterTime_us,
    int targetWindowSize,
    float targetSpectralResolution_Hz,
    float maxFrequency_Hz,
    int combined,
    int numTRs,
    float trDuration_us,
    int numForbiddenBands,
    const pulseqlib_ForbiddenBand* forbiddenBands,
    int storeResults,
    pulseqlib_TRAcousticSpectra* spectra,
    pulseqlib_Diagnostic* diag);

int pulseqlib_computePNS(
    const float gamma_hz_per_tesla,
    const float pns_threshold,
    const pulseqlib_TRGradientWaveforms* waveforms,
    float gradRasterTime_us,
    const pulseqlib_PNSParams* params,
    int storeWaveforms,
    pulseqlib_PNSResult* result,
    pulseqlib_Diagnostic* diag);

void pulseqlib_pnsResultFree(pulseqlib_PNSResult* result);

/**
 * @brief Get total number of unique segments in collection.
 * 
 * @param descCollection Sequence descriptor collection
 * @return Total number of unique segments, or 0 if none
 */
int pulseqlib_getNumSegments(const pulseqlib_SequenceDescriptorCollection* descCollection);

/**
 * @brief Check if a segment is pure delay.
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @return 1 if pure delay, 0 if not, -1 if invalid index
 */
int pulseqlib_isSegmentPureDelay(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx);

/**
 * @brief Get number of blocks in a segment.
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @return Number of blocks in segment, or -1 if invalid index
 */
int pulseqlib_getSegmentNumBlocks(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx);

/**
 * @brief Get start time of a block within a segment.
 * 
 * First block in segment starts at time 0.
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @return Block start time in microseconds, or -1 if invalid index
 */
int pulseqlib_getBlockStartTime(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx);

/**
 * @brief Get duration of a block.
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @return Block duration in microseconds, or -1 if invalid index
 */
int pulseqlib_getBlockDuration(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx);

#ifdef __cplusplus
}
#endif

#endif /* PULSEQLIB_METHODS_H */
