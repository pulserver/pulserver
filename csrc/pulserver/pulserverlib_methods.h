#ifndef PULSERVERLIB_METHODS_H
#define PULSERVERLIB_METHODS_H

#include "pulserverlib.h"

pulserverlib_Status pulserverlib_segmentLayoutInit(pulserverlib_SegmentLayout* layout, const pulseqlib_SeqFile* seq);
void pulserverlib_segmentLayoutFree(pulserverlib_SegmentLayout* layout);

#endif /* PULSERVERLIB_METHODS_H */