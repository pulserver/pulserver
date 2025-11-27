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
/* Compute max RSS amplitude of three arrays */
float max_rss(float *x, float *y, float *z, int len)
{
    int t;
    float amp, max_amp = 0.0f;
    for(t = 0; t < len; t++) {
        amp = sqrtf(x[t]*x[t] + y[t]*y[t] + z[t]*z[t]);
        if(amp > max_amp) max_amp = amp;
    }
    return max_amp;
}

/* Compute Hann window (Tukey alpha=1) */
void compute_hann(float *hann, int len)
{
    int i;
    for(i=0; i < len; i++) {
        hann[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (len - 1)));
    }
}

/* Acoustic checker - dynamic allocation for arbitrary TR length */
int pulserverlib_check_acoustics(
    float *gx, float *gy, float *gz, int N_samples, float dt,
    float *esp_min_us, float *esp_max_us, float *max_amp_Gcm, int num_bands,
	float TR_duration, int N_TR, float window_len_sec, float threshold, float max_freq
)
{
	/*
	 * Two-level acoustic screening strategy executed on the supplied gradient waveforms.
	 *
	 * Phase A (single-TR sliding windows):
	 *   - Build overlapping windows of length `window_len_sec` converted to samples.
	 *   - For each window, gather raw gradients, compute the time-domain RSS maximum, remove DC,
	 *     apply a Hann taper, and perform three real FFTs (one per axis).
	 *   - Combine the axis FFTs into an RSS power spectrum, integrate the total spectral energy
	 *     over the user-selected frequency span (`max_freq < 0` covers the full Nyquist range),
	 *     and accumulate energy inside each forbidden ESP band. A window is flagged when a band
	 *     carries more than `threshold` of the window's total energy and the time-domain guard
	 *     exceeds the permitted amplitude for that band.
	 *
	 * Phase B (TR harmonic analysis):
	 *   - After all windows pass, detrend the entire TR, perform a full-length FFT, and build the
	 *     RSS power spectrum across the full repetition within the same frequency span. Forbidden
	 *     bands that retain more than `threshold` of the TR energy while exceeding the amplitude
	 *     guard are reported as violations to protect against resonances that emerge only with
	 *     repeated TRs. Harmonic spikes up to `N_TR` at multiples of 1/TR are also examined when
	 *     they fall inside the selected frequency band.
	 *
	 * Return semantics: 0 = safe, 1 = violation detected, -1 = error (invalid input or allocation).
	 */
	int i, w, k, f, harm, bin_idx, bin_start, bin_end, b;
	int nwindows = 0;
	int stride = 0;
	int window_length = 0;
	int start = 0;
	int actual_length = 0;
	float mean_x, mean_y, mean_z;
	float amp_rss;
	int Nfft;
	float amp_rss_TR = 0.0f;
	int violation = 0;
	int status = -1;
	float span = 0.0f;
	float total_power = 0.0f;
	float freq_scale = 0.0f;
	float band_power = 0.0f;
	float power = 0.0f;
	float freq = 0.0f;
	float total_power_tr = 0.0f;
	float freq_scale_tr = 0.0f;
	float band_ratio = 0.0f;
	float nyquist = 0.0f;
	float harmonic_freq = 0.0f;
	float harmonic_power = 0.0f;
	float freq_limit_input = 0.0f;
	int window_max_bin = 0;
	int tr_max_bin = 0;
	int candidate_bin = 0;

	float *hann = NULL;
	float *gxw = NULL;
	float *gyw = NULL;
	float *gzw = NULL;
	float *f_low = NULL;
	float *f_high = NULL;
	float *max_allowed_amp = NULL;
	float *gx_tr = NULL;
	float *gy_tr = NULL;
	float *gz_tr = NULL;
	float *window_power = NULL;
	float *tr_power = NULL;
	kiss_fft_cpx *X = NULL;
	kiss_fft_cpx *Y = NULL;
	kiss_fft_cpx *Z = NULL;
	kiss_fftr_cfg cfg = NULL;
	kiss_fftr_cfg cfg_tr = NULL;
	kiss_fft_cpx *X_tr = NULL;
	kiss_fft_cpx *Y_tr = NULL;
	kiss_fft_cpx *Z_tr = NULL;

	/* Sanity check the required inputs */
	if(!gx || !gy || !gz || !esp_min_us || !esp_max_us || !max_amp_Gcm) return -1;
	if(N_samples <= 0 || num_bands <= 0) return -1;
	if(dt <= 0.0f || window_len_sec <= 0.0f || TR_duration <= 0.0f) return -1;
	if(threshold < 0.0f) threshold = 0.0f;
	if(max_freq < 0.0f) {
		freq_limit_input = -1.0f;
	} else {
		freq_limit_input = max_freq;
	}

	/* Derive FFT window size (clamp to avoid degenerate cases) */
	window_length = (int)(window_len_sec / dt + 0.5f);
	if(window_length < 2) window_length = 2;

	if(TR_duration < 2.0f * window_len_sec) {
		stride = window_length;
	} else {
		stride = window_length / 2;
	}
	if(stride < 1) stride = 1;

	/* Determine how many overlapping windows we will inspect */
	if(N_samples <= window_length) {
		nwindows = 1;
	} else {
		span = (float)(N_samples - window_length);
		nwindows = (int)ceilf(span / (float)stride) + 1;
	}
	if(nwindows <= 0) return -1;

	/*
	 * Allocate once, reuse everywhere: any failure jumps to cleanup thanks to the do-while loop
	 * and the shared `break` statements. This keeps resource handling compact and easy to audit.
	 */
	do {
		/*
		 * Translate the forbidden ESP specification (microsecond ranges and G/cm amplitudes)
		 * into working frequency bands and mT/m amplitudes so we can compare directly against
		 * the FFT output and RSS values.
		 */
		f_low = (float*)ALLOC(sizeof(float) * num_bands);
		f_high = (float*)ALLOC(sizeof(float) * num_bands);
		max_allowed_amp = (float*)ALLOC(sizeof(float) * num_bands);
		if(!f_low || !f_high || !max_allowed_amp) break;

		for(k = 0; k < num_bands; k++) {
			f_low[k]  = 1.0f / (2.0f * esp_max_us[k] * 1e-6f);
			f_high[k] = 1.0f / (2.0f * esp_min_us[k] * 1e-6f);
			max_allowed_amp[k] = max_amp_Gcm[k] * 10.0f;
		}

		/*
		 * Build the Hann taper and allocate per-window working buffers. The same memory is reused
		 * for every iteration of the sliding-window loop to avoid needless churn.
		 */
		hann = (float*)ALLOC(sizeof(float) * window_length);
		if(!hann) break;
		compute_hann(hann, window_length);

		gxw = (float*)ALLOC(sizeof(float) * window_length);
		gyw = (float*)ALLOC(sizeof(float) * window_length);
		gzw = (float*)ALLOC(sizeof(float) * window_length);
		if(!gxw || !gyw || !gzw) break;

		Nfft = window_length;
		X = (kiss_fft_cpx*)ALLOC(sizeof(kiss_fft_cpx) * (Nfft / 2 + 1));
		Y = (kiss_fft_cpx*)ALLOC(sizeof(kiss_fft_cpx) * (Nfft / 2 + 1));
		Z = (kiss_fft_cpx*)ALLOC(sizeof(kiss_fft_cpx) * (Nfft / 2 + 1));
		if(!X || !Y || !Z) break;

		window_power = (float*)ALLOC(sizeof(float) * (Nfft / 2 + 1));
		if(!window_power) break;

		cfg = kiss_fftr_alloc(Nfft, 0, NULL, NULL);
		if(!cfg) break;

		status = 0;
		violation = 0;

		/* Sweep every candidate window, looking for acoustic violations */
		for(w = 0; w < nwindows && !violation; w++) {
			start = w * stride;
			if(start >= N_samples) break;

			actual_length = N_samples - start;
			if(actual_length <= 0) break;
			if(actual_length > window_length) actual_length = window_length;
			total_power = 0.0f;

			/*
			 * Copy raw gradients into the working window. Any portion that extends past the
			 * available samples is zero-padded so the FFT vector length stays fixed and the
			 * frequency resolution remains uniform, even for edge windows.
			 */
			for(i = 0; i < window_length; i++) {
				if(i < actual_length) {
					gxw[i] = gx[start + i];
					gyw[i] = gy[start + i];
					gzw[i] = gz[start + i];
				} else {
					gxw[i] = 0.0f;
					gyw[i] = 0.0f;
					gzw[i] = 0.0f;
				}
			}

			/*
			 * Measure the RSS peak inside this window. This guard captures the largest mechanical
			 * drive present in the time domain before any processing and is paired with the spectral
			 * threshold checks below.
			 */
			amp_rss = max_rss(gxw, gyw, gzw, actual_length);

			/*
			 * Remove the DC component so the FFT measures oscillatory energy only. By centring the
			 * data we prevent strong zero-frequency content from hiding narrow-band resonances.
			 */
			mean_x = mean_y = mean_z = 0.0f;
			for(i = 0; i < actual_length; i++) {
				mean_x += gxw[i];
				mean_y += gyw[i];
				mean_z += gzw[i];
			}
			mean_x /= (float)actual_length;
			mean_y /= (float)actual_length;
			mean_z /= (float)actual_length;

			for(i = 0; i < actual_length; i++) {
				gxw[i] -= mean_x;
				gyw[i] -= mean_y;
				gzw[i] -= mean_z;
			}

			for(i = actual_length; i < window_length; i++) {
				gxw[i] = 0.0f;
				gyw[i] = 0.0f;
				gzw[i] = 0.0f;
			}

			/*
			 * Apply a Hann taper to limit spectral leakage before computing the FFT. This reduces
			 * sidelobe energy so that sharp peaks within forbidden bands stand out clearly.
			 */
			for(i = 0; i < window_length; i++) {
				gxw[i] *= hann[i];
				gyw[i] *= hann[i];
				gzw[i] *= hann[i];
			}

			/* Execute a real FFT for each gradient axis before building the RSS spectrum. */
			kiss_fftr(cfg, gxw, X);
			kiss_fftr(cfg, gyw, Y);
			kiss_fftr(cfg, gzw, Z);

			freq_scale = 1.0f / (dt * (float)Nfft);
			window_max_bin = Nfft / 2;
			if(freq_limit_input >= 0.0f) {
				candidate_bin = (int)(freq_limit_input / freq_scale);
				if(candidate_bin < window_max_bin) {
					window_max_bin = candidate_bin;
					if(window_max_bin < 0) {
						window_max_bin = 0;
					}
				}
			}
			/*
			 * Build the RSS power spectrum once so we can compute energy fractions per band without
			 * recomputing magnitudes.
			 */
			for(f = 0; f <= window_max_bin; f++) {
				power =
					X[f].r * X[f].r + X[f].i * X[f].i +
					Y[f].r * Y[f].r + Y[f].i * Y[f].i +
					Z[f].r * Z[f].r + Z[f].i * Z[f].i;
				window_power[f] = power;
				total_power += power;
			}

			if(total_power <= 0.0f) continue;

			for(k = 0; k < num_bands && !violation; k++) {
				band_power = 0.0f;
				for(f = 0; f <= window_max_bin; f++) {
					freq = (float)f * freq_scale;
					if(freq >= f_low[k] && freq <= f_high[k]) {
						band_power += window_power[f];
					}
				}
				if(band_power > 0.0f) {
					band_ratio = band_power / total_power;
					if(band_ratio > threshold && amp_rss > max_allowed_amp[k]) {
						violation = 1;
						break;
					}
				}
			}
		}

		/* Also inspect the fundamental TR repetition for narrow-band risk */
		if(!violation && N_TR > 1) {
			float mean_tx = 0.0f;
			float mean_ty = 0.0f;
			float mean_tz = 0.0f;
			int max_bin;

			if(N_samples < 2) {
				status = -1;
				break;
			}

			/*
			 * Phase B guards against resonances that appear only when multiple TRs are repeated.
			 * Start by measuring the RSS maximum across the entire TR to serve as the harmonic
			 * time-domain guard.
			 */
			amp_rss_TR = max_rss(gx, gy, gz, N_samples);

			gx_tr = (float*)ALLOC(sizeof(float) * N_samples);
			gy_tr = (float*)ALLOC(sizeof(float) * N_samples);
			gz_tr = (float*)ALLOC(sizeof(float) * N_samples);
			if(!gx_tr || !gy_tr || !gz_tr) {
				status = -1;
				break;
			}

			/*
			 * Copy the full TR into dedicated buffers and subtract the mean so the harmonic FFT
			 * reflects purely oscillatory content. A centred waveform keeps the harmonic bins sharp.
			 */
			for(i = 0; i < N_samples; i++) {
				gx_tr[i] = gx[i];
				gy_tr[i] = gy[i];
				gz_tr[i] = gz[i];
				mean_tx += gx_tr[i];
				mean_ty += gy_tr[i];
				mean_tz += gz_tr[i];
			}
			mean_tx /= (float)N_samples;
			mean_ty /= (float)N_samples;
			mean_tz /= (float)N_samples;

			for(i = 0; i < N_samples; i++) {
				gx_tr[i] -= mean_tx;
				gy_tr[i] -= mean_ty;
				gz_tr[i] -= mean_tz;
			}

			/* Create the FFT plan for the full-length TR and allocate complex output vectors */
			cfg_tr = kiss_fftr_alloc(N_samples, 0, NULL, NULL);
			if(!cfg_tr) {
				status = -1;
				break;
			}

			X_tr = (kiss_fft_cpx*)ALLOC(sizeof(kiss_fft_cpx) * (N_samples / 2 + 1));
			Y_tr = (kiss_fft_cpx*)ALLOC(sizeof(kiss_fft_cpx) * (N_samples / 2 + 1));
			Z_tr = (kiss_fft_cpx*)ALLOC(sizeof(kiss_fft_cpx) * (N_samples / 2 + 1));
			if(!X_tr || !Y_tr || !Z_tr) {
				status = -1;
				break;
			}

			tr_power = (float*)ALLOC(sizeof(float) * (N_samples / 2 + 1));
			if(!tr_power) {
				status = -1;
				break;
			}

			kiss_fftr(cfg_tr, gx_tr, X_tr);
			kiss_fftr(cfg_tr, gy_tr, Y_tr);
			kiss_fftr(cfg_tr, gz_tr, Z_tr);

			max_bin = N_samples / 2;
			freq_scale_tr = 1.0f / (dt * (float)N_samples);
			tr_max_bin = max_bin;
			if(freq_limit_input >= 0.0f) {
				candidate_bin = (int)(freq_limit_input / freq_scale_tr);
				if(candidate_bin < tr_max_bin) {
					tr_max_bin = candidate_bin;
					if(tr_max_bin < 0) {
						tr_max_bin = 0;
					}
				}
			}
			nyquist = freq_scale_tr * (float)tr_max_bin;
			total_power_tr = 0.0f;
			for(f = 0; f <= tr_max_bin; f++) {
				power =
					X_tr[f].r * X_tr[f].r + X_tr[f].i * X_tr[f].i +
					Y_tr[f].r * Y_tr[f].r + Y_tr[f].i * Y_tr[f].i +
					Z_tr[f].r * Z_tr[f].r + Z_tr[f].i * Z_tr[f].i;
				tr_power[f] = power;
				total_power_tr += power;
			}

			if(total_power_tr > 0.0f) {
				for(k = 0; k < num_bands && !violation; k++) {
					band_power = 0.0f;
					for(f = 0; f <= tr_max_bin; f++) {
						freq = (float)f * freq_scale_tr;
						if(freq >= f_low[k] && freq <= f_high[k]) {
							band_power += tr_power[f];
						}
					}
					if(band_power > 0.0f) {
						band_ratio = band_power / total_power_tr;
						if(band_ratio > threshold && amp_rss_TR > max_allowed_amp[k]) {
							violation = 1;
							break;
						}
					}
				}
				if(!violation) {
					/* Option 3: inspect the first N_TR harmonics for concentrated energy spikes. */
					for(harm = 1; harm <= N_TR && !violation; harm++) {
						harmonic_freq = (float)harm / TR_duration;
						if(harmonic_freq <= 0.0f) {
							continue;
						}
						if(harmonic_freq > nyquist) {
							break;
						}
						bin_idx = (int)(harmonic_freq * dt * (float)N_samples + 0.5f);
						if(bin_idx < 0) {
							bin_idx = 0;
						}
						if(bin_idx > tr_max_bin) {
							bin_idx = tr_max_bin;
						}
						bin_start = bin_idx - 1;
						if(bin_start < 0) {
							bin_start = 0;
						}
						bin_end = bin_idx + 1;
						if(bin_end > tr_max_bin) {
							bin_end = tr_max_bin;
						}
						harmonic_power = 0.0f;
						for(b = bin_start; b <= bin_end; b++) {
							harmonic_power += tr_power[b];
						}
						if(harmonic_power <= 0.0f) {
							continue;
						}
						band_ratio = harmonic_power / total_power_tr;
						for(k = 0; k < num_bands; k++) {
							if(harmonic_freq >= f_low[k] && harmonic_freq <= f_high[k]) {
								if(band_ratio > threshold && amp_rss_TR > max_allowed_amp[k]) {
									violation = 1;
									break;
								}
							}
						}
					}
				}
			}
		}

		if(violation) status = 1;

	} while(0);

	if(hann) FREE(hann);
	if(gxw) FREE(gxw);
	if(gyw) FREE(gyw);
	if(gzw) FREE(gzw);
	if(X) FREE(X);
	if(Y) FREE(Y);
	if(Z) FREE(Z);
	if(gx_tr) FREE(gx_tr);
	if(gy_tr) FREE(gy_tr);
	if(gz_tr) FREE(gz_tr);
	if(X_tr) FREE(X_tr);
	if(Y_tr) FREE(Y_tr);
	if(Z_tr) FREE(Z_tr);
	if(window_power) FREE(window_power);
	if(tr_power) FREE(tr_power);
	if(f_low) FREE(f_low);
	if(f_high) FREE(f_high);
	if(max_allowed_amp) FREE(max_allowed_amp);
	if(cfg) FREE(cfg);
	if(cfg_tr) FREE(cfg_tr);

	return status;
}