#ifndef PULSERVERLIB_METHODS_H
#define PULSERVERLIB_METHODS_H

#include <stddef.h>

#include "pulseqlib.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PULSERVERLIB_INVALID_INDEX (-1)

typedef struct {
    int first_block_index;
    int block_count;
} pulserverlib_TRSpan;

typedef struct {
    int first_block_offset;
    int block_count;
} pulserverlib_SegmentSpan;

typedef struct {
    int core_id;
    pulserverlib_SegmentSpan span;
} pulserverlib_SegmentDefinition;

typedef struct {
    int definition_index;
    pulserverlib_SegmentSpan span;
} pulserverlib_SegmentInstance;

typedef struct {
    int trid;
    pulserverlib_TRSpan span;
    int num_segment_definitions;
    pulserverlib_SegmentDefinition* segment_definitions;
    int num_segment_instances;
    pulserverlib_SegmentInstance* segment_instances;
} pulserverlib_TRDefinition;

typedef struct {
    const pulseqlib_SeqFile* sequence;
    int num_tr_definitions;
    pulserverlib_TRDefinition* tr_definitions;
} pulserverlib_SequenceLayout;

#ifdef __cplusplus
}
#endif

#endif /* PULSERVERLIB_METHODS_H */