#include <math.h>
#include <limits.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "external/kiss_fft.h"
#include "external/kiss_fftr.h"

#include "pulseqlib_methods.h"
#include "pulserverlib_methods.h"

static void pulserverlib_reset_layout(pulserverlib_SegmentLayout* layout) {
	if (!layout) return;
	layout->segments = NULL;
	layout->numSegments = 0;
	layout->tr.ID = 0;
	layout->tr.numSegments = 0;
	layout->tr.segmentIndices = NULL;
}

pulserverlib_Status pulserverlib_segmentLayoutInit(pulserverlib_SegmentLayout* layout, const pulseqlib_SeqFile* seq) {
	int i;
	int j;
	int firstTrId;
	int blockLimit;
	int occurrences;
	int occIndex;
	int uniqueCount;
	pulserverlib_Status status;
	pulseqlib_BlockLabels labels;
	int* coreidValues;
	int* coreidStarts;
	int* coreidSizes;
	pulserverlib_SegmentDefinition* segments;
	int* trIndices;

	if (!layout || !seq) {
		return PULSERVERLIB_STATUS_INVALID_ARGUMENT;
	}

	pulserverlib_reset_layout(layout);

	if (seq->numBlocks <= 0) {
		return PULSERVERLIB_STATUS_NO_BLOCKS;
	}

	status = PULSERVERLIB_STATUS_OK;
	firstTrId = 0;
	blockLimit = seq->numBlocks;
	occurrences = 0;
	coreidValues = NULL;
	coreidStarts = NULL;
	coreidSizes = NULL;
	segments = NULL;
	trIndices = NULL;
	/* Allocate all working data structures once; break handles early failures */
	do {
		for (i = 0; i < seq->numBlocks; ++i) {
			pulseqlib_getBlockLabels(seq, &labels, i);

			if (i == 0) {
				if (labels.flag.trid == 0) {
					status = PULSERVERLIB_STATUS_MISSING_TRID;
					break;
				}
				firstTrId = labels.flag.trid;
				if (labels.flag.coreid == 0) {
					status = PULSERVERLIB_STATUS_MISSING_COREID;
					break;
				}
			} else if (labels.flag.trid != 0 && labels.flag.trid != firstTrId) {
				blockLimit = i;
				break;
			}

			if (labels.flag.coreid != 0) {
				occurrences += 1;
			}
		}
		if (status != PULSERVERLIB_STATUS_OK) {
			break;
		}

		if (blockLimit == 0) {
			status = PULSERVERLIB_STATUS_NO_BLOCKS;
			break;
		}

		if (occurrences == 0) {
			status = PULSERVERLIB_STATUS_MISSING_COREID;
			break;
		}

		coreidValues = (int*)ALLOC(sizeof(int) * occurrences);
		coreidStarts = (int*)ALLOC(sizeof(int) * occurrences);
		coreidSizes = (int*)ALLOC(sizeof(int) * occurrences);
		if (!coreidValues || !coreidStarts || !coreidSizes) {
			status = PULSERVERLIB_STATUS_MEMORY_ERROR;
			break;
		}

		occIndex = 0;
		for (i = 0; i < blockLimit; ++i) {
			pulseqlib_getBlockLabels(seq, &labels, i);
			if (labels.flag.coreid != 0) {
				coreidValues[occIndex] = labels.flag.coreid;
				coreidStarts[occIndex] = i;
				occIndex += 1;
			}
		}
		if (occIndex != occurrences) {
			status = PULSERVERLIB_STATUS_INCONSISTENT_COREID;
			break;
		}

		for (i = 0; i < occurrences - 1; ++i) {
			int span = coreidStarts[i + 1] - coreidStarts[i];
			if (span <= 0) {
				status = PULSERVERLIB_STATUS_INCONSISTENT_COREID;
				break;
			}
			coreidSizes[i] = span;
		}
		if (status != PULSERVERLIB_STATUS_OK) {
			break;
		}

		coreidSizes[occurrences - 1] = blockLimit - coreidStarts[occurrences - 1];
		if (coreidSizes[occurrences - 1] <= 0) {
			status = PULSERVERLIB_STATUS_INCONSISTENT_COREID;
			break;
		}

		segments = (pulserverlib_SegmentDefinition*)ALLOC(sizeof(pulserverlib_SegmentDefinition) * occurrences);
		trIndices = (int*)ALLOC(sizeof(int) * occurrences);
		if (!segments || !trIndices) {
			status = PULSERVERLIB_STATUS_MEMORY_ERROR;
			break;
		}

		uniqueCount = 0;
		for (i = 0; i < occurrences; ++i) {
			int id = coreidValues[i];
			int size = coreidSizes[i];
			int found = -1;

			for (j = 0; j < uniqueCount; ++j) {
				if (segments[j].ID == id) {
					found = j;
					break;
				}
			}

			if (found < 0) {
				segments[uniqueCount].ID = id;
				segments[uniqueCount].offsetBlock = coreidStarts[i];
				segments[uniqueCount].numBlocks = size;
				found = uniqueCount;
				uniqueCount += 1;
			} else if (segments[found].numBlocks != size) {
				status = PULSERVERLIB_STATUS_INCONSISTENT_COREID;
				break;
			}

			trIndices[i] = found;
		}
		if (status != PULSERVERLIB_STATUS_OK) {
			break;
		}

		layout->segments = segments;
		layout->numSegments = uniqueCount;
		layout->tr.ID = firstTrId;
		layout->tr.numSegments = occurrences;
		layout->tr.segmentIndices = trIndices;
		segments = NULL;
		trIndices = NULL;
	} while (0);

	if (coreidValues) FREE(coreidValues);
	if (coreidStarts) FREE(coreidStarts);
	if (coreidSizes) FREE(coreidSizes);
	if (status != PULSERVERLIB_STATUS_OK) {
		if (segments) FREE(segments);
		if (trIndices) FREE(trIndices);
		pulserverlib_segmentLayoutFree(layout);
	}
	return status;
}

void pulserverlib_segmentLayoutFree(pulserverlib_SegmentLayout* layout) {
	if (!layout) return;
	if (layout->segments) {
		FREE(layout->segments);
		layout->segments = NULL;
	}
	if (layout->tr.segmentIndices) {
		FREE(layout->tr.segmentIndices);
		layout->tr.segmentIndices = NULL;
	}
	layout->numSegments = 0;
	layout->tr.numSegments = 0;
	layout->tr.ID = 0;
}

pulserverlib_Status pulserverlib_concatenateIndexedFloatArrays(
	const float** arrays,
	const int* arrayLengths,
	int numArrays,
	const int* sequence,
	int sequenceLength,
	float** outBuffer,
	int* outLength) {
	int i;
	int totalLength;
	float* buffer;
	int offset;

	if (!arrays || !arrayLengths || !sequence || !outBuffer || !outLength) {
		return PULSERVERLIB_STATUS_INVALID_ARGUMENT;
	}

	*outBuffer = NULL;
	*outLength = 0;

	if (sequenceLength <= 0) {
		return PULSERVERLIB_STATUS_OK;
	}

	if (numArrays <= 0) {
		return PULSERVERLIB_STATUS_INVALID_ARGUMENT;
	}

	totalLength = 0;
	for (i = 0; i < sequenceLength; ++i) {
		int index = sequence[i];
		int length;
		if (index < 0 || index >= numArrays) {
			return PULSERVERLIB_STATUS_INVALID_ARGUMENT;
		}
		length = arrayLengths[index];
		if (length < 0) {
			return PULSERVERLIB_STATUS_INVALID_ARGUMENT;
		}
		if (totalLength > INT_MAX - length) {
			return PULSERVERLIB_STATUS_INVALID_ARGUMENT;
		}
		totalLength += length;
	}

	if (totalLength == 0) {
		return PULSERVERLIB_STATUS_OK;
	}

	buffer = (float*)ALLOC(sizeof(float) * totalLength);
	if (!buffer) {
		return PULSERVERLIB_STATUS_MEMORY_ERROR;
	}

	offset = 0;
	for (i = 0; i < sequenceLength; ++i) {
		int index = sequence[i];
		int length = arrayLengths[index];
		if (length > 0) {
			if (!arrays[index]) {
				FREE(buffer);
				return PULSERVERLIB_STATUS_INVALID_ARGUMENT;
			}
			memcpy(buffer + offset, arrays[index], sizeof(float) * length);
			offset += length;
		}
	}

	*outBuffer = buffer;
	*outLength = totalLength;
	return PULSERVERLIB_STATUS_OK;
}

pulserverlib_Status pulserverlib_computeFirstDifference(
	const float* array,
	int length,
	float** outBuffer,
	int* outLength) {
	float* diffs;
	int diffLength;
	int i;

	if (!outBuffer || !outLength) {
		return PULSERVERLIB_STATUS_INVALID_ARGUMENT;
	}

	*outBuffer = NULL;
	*outLength = 0;

	if (length < 0) {
		return PULSERVERLIB_STATUS_INVALID_ARGUMENT;
	}

	if (length == 0) {
		return PULSERVERLIB_STATUS_OK;
	}

	if (!array) {
		return PULSERVERLIB_STATUS_INVALID_ARGUMENT;
	}

	if (length == 1) {
		return PULSERVERLIB_STATUS_OK;
	}

	diffLength = length - 1;
	diffs = (float*)ALLOC(sizeof(float) * diffLength);
	if (!diffs) {
		return PULSERVERLIB_STATUS_MEMORY_ERROR;
	}

	for (i = 0; i < diffLength; ++i) {
		diffs[i] = array[i + 1] - array[i];
	}

	*outBuffer = diffs;
	*outLength = diffLength;
	return PULSERVERLIB_STATUS_OK;
}

float pulserverlib_checkMaxGradientMagnitude(const pulseqlib_SeqFile* seq) {
	float allowedMax;
	float actualMax;

	if (!seq) {
		return -1.0f;
	}

	allowedMax = seq->opts.max_grad / sqrtf(3.0f);
	actualMax = pulseqlib_getGradLibraryMaxAmplitude(seq);

	if (actualMax > allowedMax) {
		return -1.0f;
	}

	return actualMax;
}

pulserverlib_Status pulserverlib_checkMaxSlewRate(
	const pulseqlib_SeqFile* seq,
	const float* waveX,
	int lengthX,
	const float* waveY,
	int lengthY,
	const float* waveZ,
	int lengthZ,
	float** slewX,
	int* slewLengthX,
	float** slewY,
	int* slewLengthY,
	float** slewZ,
	int* slewLengthZ) {
	float* diffX = NULL;
	float* diffY = NULL;
	float* diffZ = NULL;
	int diffLenX = 0;
	int diffLenY = 0;
	int diffLenZ = 0;
	float gradRaster;
	float maxSlew;
	int i;
	pulserverlib_Status status;

	if (!slewX || !slewLengthX || !slewY || !slewLengthY || !slewZ || !slewLengthZ) {
		return PULSERVERLIB_STATUS_INVALID_ARGUMENT;
	}

	*slewX = NULL;
	*slewY = NULL;
	*slewZ = NULL;
	*slewLengthX = 0;
	*slewLengthY = 0;
	*slewLengthZ = 0;

	if (!seq) {
		return PULSERVERLIB_STATUS_INVALID_ARGUMENT;
	}

	gradRaster = seq->opts.grad_raster_time;
	maxSlew = seq->opts.max_slew;

	if (gradRaster <= 0.0f) {
		return PULSERVERLIB_STATUS_INVALID_ARGUMENT;
	}

	if (lengthX < 0 || lengthY < 0 || lengthZ < 0) {
		return PULSERVERLIB_STATUS_INVALID_ARGUMENT;
	}

	if ((lengthX > 0 && !waveX) || (lengthY > 0 && !waveY) || (lengthZ > 0 && !waveZ)) {
		return PULSERVERLIB_STATUS_INVALID_ARGUMENT;
	}

	if (lengthX > 0) {
		if (fabsf(waveX[0]) / gradRaster > maxSlew || fabsf(waveX[lengthX - 1]) / gradRaster > maxSlew) {
			return PULSERVERLIB_STATUS_SAFETY_LIMIT_EXCEEDED;
		}
	}
	if (lengthY > 0) {
		if (fabsf(waveY[0]) / gradRaster > maxSlew || fabsf(waveY[lengthY - 1]) / gradRaster > maxSlew) {
			return PULSERVERLIB_STATUS_SAFETY_LIMIT_EXCEEDED;
		}
	}
	if (lengthZ > 0) {
		if (fabsf(waveZ[0]) / gradRaster > maxSlew || fabsf(waveZ[lengthZ - 1]) / gradRaster > maxSlew) {
			return PULSERVERLIB_STATUS_SAFETY_LIMIT_EXCEEDED;
		}
	}

	status = pulserverlib_computeFirstDifference(waveX, lengthX, &diffX, &diffLenX);
	if (status != PULSERVERLIB_STATUS_OK) {
		return status;
	}
	status = pulserverlib_computeFirstDifference(waveY, lengthY, &diffY, &diffLenY);
	if (status != PULSERVERLIB_STATUS_OK) {
		if (diffX) FREE(diffX);
		return status;
	}
	status = pulserverlib_computeFirstDifference(waveZ, lengthZ, &diffZ, &diffLenZ);
	if (status != PULSERVERLIB_STATUS_OK) {
		if (diffX) FREE(diffX);
		if (diffY) FREE(diffY);
		return status;
	}

	for (i = 0; i < diffLenX; ++i) {
		diffX[i] /= gradRaster;
		if (fabsf(diffX[i]) > maxSlew) {
			if (diffX) FREE(diffX);
			if (diffY) FREE(diffY);
			if (diffZ) FREE(diffZ);
			return PULSERVERLIB_STATUS_SAFETY_LIMIT_EXCEEDED;
		}
	}
	for (i = 0; i < diffLenY; ++i) {
		diffY[i] /= gradRaster;
		if (fabsf(diffY[i]) > maxSlew) {
			if (diffX) FREE(diffX);
			if (diffY) FREE(diffY);
			if (diffZ) FREE(diffZ);
			return PULSERVERLIB_STATUS_SAFETY_LIMIT_EXCEEDED;
		}
	}
	for (i = 0; i < diffLenZ; ++i) {
		diffZ[i] /= gradRaster;
		if (fabsf(diffZ[i]) > maxSlew) {
			if (diffX) FREE(diffX);
			if (diffY) FREE(diffY);
			if (diffZ) FREE(diffZ);
			return PULSERVERLIB_STATUS_SAFETY_LIMIT_EXCEEDED;
		}
	}

	*slewX = diffX;
	*slewY = diffY;
	*slewZ = diffZ;
	*slewLengthX = diffLenX;
	*slewLengthY = diffLenY;
	*slewLengthZ = diffLenZ;

	return PULSERVERLIB_STATUS_OK;
}


/********************* Acoustic checks ****************************/
#define MIN_SPEC_LENGTH 2048

void detrend_signal(float* signal, const int N) {
	int n;
	float mean_val = 0.0f;

	/* Calculate mean value */
	for (n = 0; n < N; n++) {
		mean_val += signal[n];
	}
	mean_val /= N;

	/* Subtract mean value from signal */
	for (n = 0; n < N; n++) {
		signal[n] -= mean_val;
	}
}

/* Tapering functions */
void calc_hanning_window(float* window, const int N) {
	int n;
	for (n = 0; n < N; n++) {
		window[n] = 0.5f * (1.0f - cosf((2.0f * M_PI * n) / (N - 1)));
	}
}

void apply_window(float* signal, const float* window, const int N) {
	int n;
	for (n = 0; n < N; n++) {
		signal[n] *= window[n];
	}
}

int calc_padding_length(const int N_samples) {
	int num_pad_samples;
	int num_samples;

	/* Determine target length */
	if (N_samples < MIN_SPEC_LENGTH) {
		num_samples = MIN_SPEC_LENGTH;
	} else {
		num_samples = N_samples;
	}

	/* Calculate number of padding samples */
	num_pad_samples = num_samples - N_samples;
	if (num_pad_samples < 0) {
		num_pad_samples = 0;
	}

	return num_pad_samples;
}

void apply_padding(float* signal_out, const int n_out, const float* signal_in, const int n_in) {
	int n;

	/* Zero-pad the signal */
	for (n = 0; n < n_in; n++) {
		signal_out[n] = signal_in[n];
	}
	for (n = n_in; n < n_out; n++) {
		signal_out[n] = 0.0f;
	}
}

void calc_logpow_spectrum(
	float* output,
	const kiss_fft_cpx* sx, 
	const kiss_fft_cpx* sy, 
	const kiss_fft_cpx* sz,
	const int N
) {
	int n;
	float power;

	/* Calculate log-power spectrum */
	for (n = 0; n < N; n++) {
		/* Calculate power spectrum at index n */
		power = sx[n].r * sx[n].r + sx[n].i * sx[n].i +
				sy[n].r * sy[n].r + sy[n].i * sy[n].i +
				sz[n].r * sz[n].r + sz[n].i * sz[n].i;

		/* Avoid log of zero by adding a small constant */
		output[n] = (float)log10((double)power + 1e-12f);
	}
}

void clip_logpow_spectrum(float* spectrum, const int N, const float threshold) {
	int n;

	/* Detrend */
	detrend_signal(spectrum, N);

	/* Clip values below threshold */
	for (n = 0; n < N; n++) {
		if (spectrum[n] < threshold) {
			spectrum[n] = 0.0f;
		}
	}
	
}

int pulserverlib_check_acoustics(
	float dt,
	int num_samples,
	float* gx, 
	float* gy, 
	float* gz,
	int num_bands,
	float* fmin, 
	float* fmax, 
	float* max_amp
) {
	int n, m;
	int num_samples_padded;
	int num_pad_samples;
	
	float fs;
	float df;

	float* window;
	float* combined_spectrum;
	
	float maxGrad;
	float* gx_zf;
	float* gy_zf;
	float* gz_zf;
	float* freq_axis;
	
	kiss_fft_cpx* sx;
	kiss_fft_cpx* sy;
	kiss_fft_cpx* sz;
	
	kiss_fftr_cfg fft_cfg;

	/* Determine max gradient amplitude */
	maxGrad = 0.0f;
	for (n = 0; n < num_samples; n++) {
		if (fabsf(gx[n]) > maxGrad) {
			maxGrad = fabsf(gx[n]);
		}
		if (fabsf(gy[n]) > maxGrad) {
			maxGrad = fabsf(gy[n]);
		}
		if (fabsf(gz[n]) > maxGrad) {
			maxGrad = fabsf(gz[n]);
		}
	}
	/* Early exit if max gradient is zero */
	if (maxGrad == 0.0f) {
		return 0; /* No violation */
	}

	/* Sampling frequency */
	fs = 1.0f / dt;

	/* Detrend gradient waveforms */
	detrend_signal(gx, num_samples);
	detrend_signal(gy, num_samples);
	detrend_signal(gz, num_samples);

	/* Taper gradient waveforms */
	window = (float*)ALLOC(sizeof(float) * num_samples);
	if (!window) {
		return -1; /* Error */
	}
	calc_hanning_window(window, num_samples);
	apply_window(gx, window, num_samples);
	apply_window(gy, window, num_samples);
	apply_window(gz, window, num_samples);
	FREE(window);

	/* Determine zerofill */
	num_pad_samples = calc_padding_length(num_samples);
	num_samples_padded = num_samples + num_pad_samples;
	df = fs / num_samples_padded;

	/* Apply zero-fill of gradient waveforms */
	gx_zf = (float*)ALLOC(sizeof(float) * num_samples_padded);
	if (!gx_zf) {
		return -1; /* Error */
	}
	gy_zf = (float*)ALLOC(sizeof(float) * num_samples_padded);
	if (!gy_zf) {
		FREE(gx_zf);
		return -1; /* Error */
	}
	gz_zf = (float*)ALLOC(sizeof(float) * num_samples_padded);
	if (!gz_zf) {
		FREE(gx_zf);
		FREE(gy_zf);
		return -1; /* Error */
	}
	apply_padding(gx_zf, num_samples_padded, gx, num_samples);
	apply_padding(gy_zf, num_samples_padded, gy, num_samples);
	apply_padding(gz_zf, num_samples_padded, gz, num_samples);

	/* Compute frequency axis */
	freq_axis = (float*)ALLOC(sizeof(float) * (	num_samples_padded / 2 + 1));
	if (!freq_axis) {
		/* Free allocated memory */
		FREE(gx_zf);
		FREE(gy_zf);
		FREE(gz_zf);
		return -1; /* Error */
	}
	for (n = 0; n < (num_samples_padded / 2 + 1); n++) {
		freq_axis[n] = (float)n * df;
	}

	/* Compute gx, gy and gz spectra */
	fft_cfg = kiss_fftr_alloc(num_samples_padded, 0, NULL, NULL);
	if (!fft_cfg) {
		/* Free allocated memory */
		FREE(gx_zf);
		FREE(gy_zf);
		FREE(gz_zf);
		FREE(freq_axis);
		return -1; /* Error */
	}
	sx = (kiss_fft_cpx*)ALLOC(sizeof(kiss_fft_cpx) * (num_samples_padded / 2 + 1));
	if (!sx) {
		/* Free allocated memory */
		FREE(gx_zf);
		FREE(gy_zf);
		FREE(gz_zf);
		FREE(freq_axis);
		kiss_fftr_free(fft_cfg);
		return -1; /* Error */
	}
	sy = (kiss_fft_cpx*)ALLOC(sizeof(kiss_fft_cpx) * (num_samples_padded / 2 + 1));
	if (!sy) {
		/* Free allocated memory */
		FREE(gx_zf);
		FREE(gy_zf);
		FREE(gz_zf);
		FREE(freq_axis);
		FREE(sx);
		kiss_fftr_free(fft_cfg);
		return -1; /* Error */
	}
	sz = (kiss_fft_cpx*)ALLOC(sizeof(kiss_fft_cpx) * (num_samples_padded / 2 + 1));
	if (!sz) {
		/* Free allocated memory */
		FREE(gx_zf);
		FREE(gy_zf);
		FREE(gz_zf);
		FREE(freq_axis);
		FREE(sx);
		FREE(sy);
		kiss_fftr_free(fft_cfg);
		return -1; /* Error */
	}
	kiss_fftr(fft_cfg, gx_zf, sx);
	kiss_fftr(fft_cfg, gy_zf, sy);
	kiss_fftr(fft_cfg, gz_zf, sz);
	kiss_fftr_free(fft_cfg);

	/* Compute combined log-power spectrum */
	combined_spectrum = (float*)ALLOC(sizeof(float) * (num_samples_padded / 2 + 1));
	if (!combined_spectrum) {
		/* Free allocated memory */
		FREE(gx_zf);
		FREE(gy_zf);
		FREE(gz_zf);
		FREE(freq_axis);
		FREE(sx);
		FREE(sy);
		FREE(sz);
		return -1; /* Error */
	}
	calc_logpow_spectrum(combined_spectrum, sx, sy, sz, (num_samples_padded / 2 + 1));
	FREE(sx);
	FREE(sy);
	FREE(sz);

	/* Clip combined spectrum */
	clip_logpow_spectrum(combined_spectrum, (num_samples_padded / 2 + 1), 0.75f);

	/* Loop over bands and search for violations */
	for (n = 0; n < num_bands; n++) {
		for (m = 0; m < (num_samples_padded / 2 + 1); m++) {
			if (freq_axis[m] >= fmin[n] && freq_axis[m] <= fmax[n]) {
				if (fabs(combined_spectrum[m]) > 0.0 && sqrtf(3.0f) * maxGrad >= max_amp[n]) {
					/* Free allocated memory */
					FREE(gx_zf);
					FREE(gy_zf);
					FREE(gz_zf);
					FREE(freq_axis);
					FREE(combined_spectrum);
					return 1; /* Violation found */
				}
			}
		}
	}

	/* Free allocated memory */
	FREE(gx_zf);
	FREE(gy_zf);
	FREE(gz_zf);
	FREE(freq_axis);
	FREE(combined_spectrum);

	return 0; /* No violation */
}


