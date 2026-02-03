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

#ifndef IS_GEHC
    #define  IS_GEHC 1
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

void pulseqlib_seqBlockInit(pulseqlib_SeqBlock* block);
void pulseqlib_seqBlockFree(pulseqlib_SeqBlock* block);

/* Parsing Sequence */
int pulseqlib_readSeq(pulseqlib_SeqFile* seq, const char* filePath);
int pulseqlib_readSeqFromBuffer(pulseqlib_SeqFile* seq, FILE* f);

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

void pulseqlib_segmentTableResultFree(pulseqlib_SegmentTableResult* result);
void pulseqlib_sequenceDescriptorFree(pulseqlib_SequenceDescriptor* seqDesc);

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

#ifdef __cplusplus
}
#endif

#endif /* PULSEQLIB_METHODS_H */
