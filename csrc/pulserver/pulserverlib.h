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
    pulserverlib_SegmentDefinition* segments; /* Array of segment definitions */
} pulserverlib_TRDefinition;

#endif /* PULSERVERLIB_H */