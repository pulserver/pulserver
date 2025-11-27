#ifndef PULSERVERLIB_H
#define PULSERVERLIB_H

#include "pulseqlib.h"

typedef struct pulserverlib_SegmentDefinition {
    int ID;
    int offsetBlock;
    int numBlocks;
} pulserverlib_SegmentDefinition;

typedef struct pulserverlib_TRDefinition {
    int ID;
    int numSegments;
    int* segmentIndices; /* Indices of segments array describing TR playback order */
} pulserverlib_TRDefinition;

typedef struct pulserverlib_SegmentLayout {
    pulserverlib_SegmentDefinition* segments;
    int numSegments;
    pulserverlib_TRDefinition tr;
} pulserverlib_SegmentLayout;

typedef enum pulserverlib_Status {
    PULSERVERLIB_STATUS_OK = 0,
    PULSERVERLIB_STATUS_INVALID_ARGUMENT = 1,
    PULSERVERLIB_STATUS_NO_BLOCKS = 2,
    PULSERVERLIB_STATUS_MISSING_TRID = 3,
    PULSERVERLIB_STATUS_MISSING_COREID = 4,
    PULSERVERLIB_STATUS_INCONSISTENT_COREID = 5,
    PULSERVERLIB_STATUS_MEMORY_ERROR = 6,
    PULSERVERLIB_STATUS_SAFETY_LIMIT_EXCEEDED = 7
} pulserverlib_Status;

#endif /* PULSERVERLIB_H */