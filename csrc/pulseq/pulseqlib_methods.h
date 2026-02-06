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

/**
 * @brief Check if a block has RF pulse.
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @return 1 if block has RF, 0 if no RF, -1 if invalid index
 */
int pulseqlib_blockHasRF(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx);

/**
 * @brief Check if a block's RF pulse has uniform raster (time shape defined).
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @return 1 if RF has uniform raster, 0 if not, -1 if invalid index or no RF
 */
int pulseqlib_blockRFHasUniformRaster(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx);

/**
 * @brief Check if a block's RF pulse is complex valued (phase shape defined).
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @return 1 if RF is complex valued, 0 if not, -1 if invalid index or no RF
 */
int pulseqlib_blockRFIsComplex(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx);

/**
 * @brief Get number of RF samples in a block's RF pulse.
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @return Number of RF samples, or -1 if invalid index, no RF, or unable to determine
 */
int pulseqlib_getRFNumSamples(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx);

/**
 * @brief Get RF pulse delay in microseconds.
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @return RF delay in microseconds, or -1 if invalid index or no RF
 */
int pulseqlib_getRFDelay(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx);

/**
 * @brief Get RF magnitude waveform samples rescaled by max amplitude.
 * 
 * Returns pointer to magnitude shape samples scaled by the max amplitude value
 * stored in the RF definition (Hz).
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @param numSamples Output parameter for number of samples (set by function)
 * @return Pointer to magnitude waveform array, or NULL if invalid index, no RF, or no magnitude shape
 */
float* pulseqlib_getRFMagnitude(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int* numSamples);

/**
 * @brief Get RF phase waveform samples in radians.
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @param numSamples Output parameter for number of samples (set by function)
 * @return Pointer to phase waveform array (rad), or NULL if invalid index, no RF, or no phase shape
 */
float* pulseqlib_getRFPhase(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int* numSamples);

/**
 * @brief Get RF time waveform samples in microseconds.
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @param numSamples Output parameter for number of samples (set by function)
 * @return Pointer to time waveform array (us), or NULL if invalid index, no RF, or no time shape
 */
float* pulseqlib_getRFTime(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int* numSamples);

/**
 * @brief Check if a block has gradient on specified axis.
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @param axis Gradient axis (PULSEQLIB_GRAD_AXIS_X, _Y, or _Z)
 * @return 1 if block has gradient on axis, 0 if no gradient, -1 if invalid index/axis
 */
int pulseqlib_blockHasGrad(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int axis);

/**
 * @brief Check if a block's gradient is a trapezoid or has time shape defined.
 * 
 * Returns 1 if gradient is a standard trapezoid (type == 0) or if it has a time shape ID defined.
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @param axis Gradient axis (PULSEQLIB_GRAD_AXIS_X, _Y, or _Z)
 * @return 1 if trapezoid or has time shape, 0 if not, -1 if invalid index or no gradient
 */
int pulseqlib_blockGradIsTrapezoid(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int axis);

/**
 * @brief Get number of gradient waveform samples on a specified axis.
 * 
 * For trapezoids: returns 4 if flatTime > 0, or 3 if flatTime == 0.
 * For arbitrary gradients: returns number of samples in the waveform shape.
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @param axis Gradient axis (PULSEQLIB_GRAD_AXIS_X, _Y, or _Z)
 * @return Number of samples, or -1 if invalid index or no gradient
 */
int pulseqlib_getGradNumSamples(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int axis);

/**
 * @brief Get number of gradient shots on a specified axis.
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @param axis Gradient axis (PULSEQLIB_GRAD_AXIS_X, _Y, or _Z)
 * @return Number of shots, or -1 if invalid index or no gradient
 */
int pulseqlib_getGradNumShots(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int axis);

/**
 * @brief Get gradient delay in microseconds.
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @param axis Gradient axis (PULSEQLIB_GRAD_AXIS_X, _Y, or _Z)
 * @return Gradient delay in microseconds, or -1 if invalid index or no gradient
 */
int pulseqlib_getGradDelay(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int axis);

/**
 * @brief Get gradient amplitude waveform scaled by max amplitude.
 * 
 * Returns array of waveforms for each shot. For trapezoids, generates [0, maxAmp, maxAmp, 0] or [0, maxAmp, 0]
 * depending on flatTime. For arbitrary gradients, decompresses waveform shape and scales by maxAmplitude.
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @param axis Gradient axis (PULSEQLIB_GRAD_AXIS_X, _Y, or _Z)
 * @param numShots Output parameter for number of shots (set by function)
 * @param numSamplesPerShot Output parameter for number of samples per shot (array of size numShots)
 * @return Pointer to array of waveform pointers (float**), one per shot, or NULL if invalid
 */
float** pulseqlib_getGradAmplitude(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int axis, int* numShots, int** numSamplesPerShot);

/**
 * @brief Get gradient amplitude for the max-energy instance of a segment.
 * 
 * Returns the gradient amplitude (Hz/m) from the block table entry of the
 * segment instance that has the maximum total gradient energy. If no gradient
 * is present on the specified axis, returns 1.0.
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @param axis Gradient axis (PULSEQLIB_GRAD_AXIS_X, _Y, or _Z)
 * @return Gradient amplitude (Hz/m), or 1.0 if no gradient on this axis
 */
float pulseqlib_getGradInitialAmplitude(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int axis);

/**
 * @brief Get gradient shot index for the max-energy instance of a segment.
 * 
 * Returns the shot index from the block table entry of the segment instance
 * that has the maximum total gradient energy. If no gradient is present on
 * the specified axis, returns 0.
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @param axis Gradient axis (PULSEQLIB_GRAD_AXIS_X, _Y, or _Z)
 * @return Gradient shot index (0-based), or 0 if no gradient on this axis
 */
int pulseqlib_getGradInitialShotID(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int axis);

/**
 * @brief Get gradient time waveform in microseconds.
 * 
 * For trapezoids, generates cumsum([0, riseTime, flatTime, fallTime]) or cumsum([0, riseTime, fallTime]) if flatTime == 0.
 * For arbitrary gradients, decompresses time shape.
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @param axis Gradient axis (PULSEQLIB_GRAD_AXIS_X, _Y, or _Z)
 * @param numSamples Output parameter for number of samples (set by function)
 * @return Pointer to time waveform array (us), or NULL if invalid index or no gradient
 */
float* pulseqlib_getGradTime(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int axis, int* numSamples);

/**
 * @brief Check if a block has ADC.
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @return 1 if block has ADC, 0 if no ADC, -1 if invalid index
 */
int pulseqlib_blockHasADC(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx);

/**
 * @brief Get ADC delay in microseconds.
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @return ADC delay in microseconds, or -1 if invalid index or no ADC
 */
int pulseqlib_getADCDelay(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx);

/**
 * @brief Get ADC definition index in the SequenceDescriptorCollection's shared ADC library.
 * 
 * This index can be used to access ADC properties (numSamples, dwellTime) from the
 * collection's global ADC definitions array, or to look up the ADC in echo_filters.
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @return ADC definition index in global ADC library, or -1 if invalid index or no ADC
 */
int pulseqlib_getADCLibraryIndex(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx);

/**
 * @brief Check if a block has a trigger event (across all segment instances).
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @return 1 if block has trigger, 0 if no trigger, -1 if invalid index
 */
int pulseqlib_blockHasTrigger(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx);

/**
 * @brief Get trigger delay in microseconds.
 * 
 * Returns the delay from the first encountered trigger event for this block
 * position across all segment instances.
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @return Trigger delay in microseconds, or -1 if invalid index or no trigger
 */
int pulseqlib_getTriggerDelay(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx);

/**
 * @brief Check if a block has a rotation event (across all segment instances).
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @return 1 if block has rotation, 0 if not, -1 if invalid index
 */
int pulseqlib_blockHasRotation(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx);

/**
 * @brief Check if a block has norot flag set (across all segment instances).
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @return 1 if norot, 0 if not, -1 if invalid index
 */
int pulseqlib_blockHasNorot(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx);

/**
 * @brief Check if a block has nopos flag set (across all segment instances).
 * 
 * @param descCollection Sequence descriptor collection
 * @param segmentIdx Global segment index (0-based)
 * @param blockIdx Block index within segment (0-based)
 * @return 1 if nopos, 0 if not, -1 if invalid index
 */
int pulseqlib_blockHasNopos(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx);

#ifdef __cplusplus
}
#endif

#endif /* PULSEQLIB_METHODS_H */
