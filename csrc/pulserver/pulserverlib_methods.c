#include <stdlib.h>
#include <string.h>

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

	for (i = 0; i < seq->numBlocks; ++i) {
		pulseqlib_getBlockLabels(seq, &labels, i);

		if (i == 0) {
			if (labels.flag.trid == 0) {
				status = PULSERVERLIB_STATUS_MISSING_TRID;
				goto cleanup;
			}
			firstTrId = labels.flag.trid;
			if (labels.flag.coreid == 0) {
				status = PULSERVERLIB_STATUS_MISSING_COREID;
				goto cleanup;
			}
		} else {
			if (labels.flag.trid != 0 && labels.flag.trid != firstTrId) {
				blockLimit = i;
				break;
			}
		}

		if (labels.flag.coreid != 0) {
			occurrences += 1;
		}
	}

	if (blockLimit == 0) {
		status = PULSERVERLIB_STATUS_NO_BLOCKS;
		goto cleanup;
	}

	if (occurrences == 0) {
		status = PULSERVERLIB_STATUS_MISSING_COREID;
		goto cleanup;
	}

	coreidValues = (int*)malloc(sizeof(int) * occurrences);
	coreidStarts = (int*)malloc(sizeof(int) * occurrences);
	coreidSizes = (int*)malloc(sizeof(int) * occurrences);
	if (!coreidValues || !coreidStarts || !coreidSizes) {
		status = PULSERVERLIB_STATUS_MEMORY_ERROR;
		goto cleanup;
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
		goto cleanup;
	}

	for (i = 0; i < occurrences - 1; ++i) {
		int span = coreidStarts[i + 1] - coreidStarts[i];
		if (span <= 0) {
			status = PULSERVERLIB_STATUS_INCONSISTENT_COREID;
			goto cleanup;
		}
		coreidSizes[i] = span;
	}
	coreidSizes[occurrences - 1] = blockLimit - coreidStarts[occurrences - 1];
	if (coreidSizes[occurrences - 1] <= 0) {
		status = PULSERVERLIB_STATUS_INCONSISTENT_COREID;
		goto cleanup;
	}

	segments = (pulserverlib_SegmentDefinition*)malloc(sizeof(pulserverlib_SegmentDefinition) * occurrences);
	trIndices = (int*)malloc(sizeof(int) * occurrences);
	if (!segments || !trIndices) {
		status = PULSERVERLIB_STATUS_MEMORY_ERROR;
		goto cleanup;
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
			goto cleanup;
		}

		trIndices[i] = found;
	}

	layout->segments = segments;
	layout->numSegments = uniqueCount;
	layout->tr.ID = firstTrId;
	layout->tr.numSegments = occurrences;
	layout->tr.segmentIndices = trIndices;

	segments = NULL;
	trIndices = NULL;

cleanup:
	if (coreidValues) free(coreidValues);
	if (coreidStarts) free(coreidStarts);
	if (coreidSizes) free(coreidSizes);
	if (status != PULSERVERLIB_STATUS_OK) {
		if (segments) free(segments);
		if (trIndices) free(trIndices);
		pulserverlib_segmentLayoutFree(layout);
	}
	return status;
}

void pulserverlib_segmentLayoutFree(pulserverlib_SegmentLayout* layout) {
	if (!layout) return;
	if (layout->segments) {
		free(layout->segments);
		layout->segments = NULL;
	}
	if (layout->tr.segmentIndices) {
		free(layout->tr.segmentIndices);
		layout->tr.segmentIndices = NULL;
	}
	layout->numSegments = 0;
	layout->tr.numSegments = 0;
	layout->tr.ID = 0;
}