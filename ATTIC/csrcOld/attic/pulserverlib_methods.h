#ifndef PULSERVERLIB_METHODS_H
#define PULSERVERLIB_METHODS_H

#include "pulserverlib.h"

pulserverlib_Status pulserverlib_segmentLayoutInit(pulserverlib_SegmentLayout* layout, const pulseqlib_SeqFile* seq);
void pulserverlib_segmentLayoutFree(pulserverlib_SegmentLayout* layout);

pulserverlib_Status pulserverlib_concatenateIndexedFloatArrays(const float** arrays, const int* arrayLengths, int numArrays, const int* sequence, int sequenceLength, float** outBuffer, int* outLength);
pulserverlib_Status pulserverlib_computeFirstDifference(const float* array, int length, float** outBuffer, int* outLength);
float pulserverlib_checkMaxGradientMagnitude(const pulseqlib_SeqFile* seq);
pulserverlib_Status pulserverlib_checkMaxSlewRate(const pulseqlib_SeqFile* seq, const float* waveX, int lengthX, const float* waveY, int lengthY, const float* waveZ, int lengthZ, float** slewX, int* slewLengthX, float** slewY, int* slewLengthY, float** slewZ, int* slewLengthZ);

/* Acoustic/TR spectral check -- simplified single-TR analysis (Phase A only). */
int pulserverlib_check_acoustics(
	float *gx, float *gy, float *gz, int N_samples, float dt,
	/* frequency bands (Hz) */
	float *f_low_hz, float *f_high_hz,
	/* max amplitude per band, in same units as gradients (no conversion) */
	float *max_amp, int num_bands,
	float TR_duration, int N_TR, float window_len_sec, float threshold
);

#endif /* PULSERVERLIB_METHODS_H */