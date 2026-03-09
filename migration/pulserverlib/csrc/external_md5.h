#ifndef MD5_H
#define MD5_H

/*
 * Portable 32-bit unsigned integer type.
 * Works with MSVC <= 2010, VxWorks, and any C89/C99 compiler.
 */
#if defined(_MSC_VER) && _MSC_VER <= 1600
typedef unsigned __int32 uint32;
#elif defined(VXWORKS)
#include "types/vxTypes.h"
typedef unsigned long uint32;
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L
#include <stdint.h>
typedef uint32_t uint32;
#else
/* C89 fallback: unsigned long is at least 32 bits */
typedef unsigned long uint32;
#endif

struct MD5Context
{
    uint32        buf[4];
    uint32        bits[2];
    unsigned char in[64];
};

void MD5Init(struct MD5Context* context);
void MD5Update(struct MD5Context* context, unsigned char const* buf, unsigned len);
void MD5Final(unsigned char digest[16], struct MD5Context* context);
void MD5Transform(uint32 buf[4], uint32 const in[16]);

/*
 * This is needed to make RSAREF happy on some MS-DOS compilers.
 */
typedef struct MD5Context MD5_CTX;

#endif /* !MD5_H */