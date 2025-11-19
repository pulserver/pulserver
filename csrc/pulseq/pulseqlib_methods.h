#ifndef PULSEQLIB_METHODS_H
#define PULSEQLIB_METHODS_H

#include <stdlib.h>

#include "pulseqlib.h"

#ifndef DETECT_REAL_RF
    #define  DETECT_REAL_RF 1
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
void pulseqlib_SeqFileInit(const char* filePath, pulseqlib_SeqFile* seq);
void pulseqlib_SeqFileFree(pulseqlib_SeqFile* seq);
void pulseqlib_SeqFileReset(pulseqlib_SeqFile* seq);

void pulseqlib_SeqBlockInit(pulseqlib_SeqBlock* block);
void pulseqlib_SeqBlockFree(pulseqlib_SeqBlock* block);

/* Parsing Sequence */
void pulseqlib_readSeq(pulseqlib_SeqFile* seq, const int readBlocks);

/* Getters */
void pulseqlib_getRawBlockContentIDs(const pulseqlib_SeqFile* seq, const int blockIndex, const int parseExtensions, pulseqlib_RawBlock* block);
void pulseqlib_getBlock(const pulseqlib_SeqFile* seq, const int blockIndex, const int parseExtensions, pulseqlib_SeqBlock* block);

#endif /* PULSEQLIB_METHODS_H */
