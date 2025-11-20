#ifndef PULSEQLIB_METHODS_H
#define PULSEQLIB_METHODS_H

#include <math.h>
#include <stdlib.h>

#include "pulseqlib.h"

#ifndef DETECT_REAL_RF
    #define  DETECT_REAL_RF 1
#endif 

/** 
 * Default ALLOC to malloc if it's not already defined
 */
#ifndef ALLOC
    #define ALLOC(size) malloc(size)
#endif

/** 
 * Default FREE to free if it's not already defined
 */
#ifndef FREE
  #define FREE(ptr) free(ptr)
#endif

/* Constructor, destructor and reset */
void pulseqlib_seqFileInit(const char* filePath, pulseqlib_SeqFile* seq);
void pulseqlib_seqFileFree(pulseqlib_SeqFile* seq);
void pulseqlib_seqFileReset(pulseqlib_SeqFile* seq);

void pulseqlib_seqBlockInit(pulseqlib_SeqBlock* block); /* Reverted to simple signature */
void pulseqlib_seqBlockFree(pulseqlib_SeqBlock* block);

/* Parsing Sequence */
void pulseqlib_readSeq(pulseqlib_SeqFile* seq, const int readBlocks);

/* Getters */
void pulseqlib_getRawBlockContentIDs(const pulseqlib_SeqFile* seq, const int blockIndex, const int parseExtensions, pulseqlib_RawBlock* block);
void pulseqlib_getBlock(pulseqlib_SeqFile* seq, const int blockIndex, const int parseExtensions, pulseqlib_SeqBlock* block);

#endif /* PULSEQLIB_METHODS_H */
