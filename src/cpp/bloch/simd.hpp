/**
 * @file simd.hpp
 * @brief The AVX2, FMA and AVX-512 a function is compiled for, and whether
 *        the processor it runs on has them.
 */

#ifndef PULSERVER_BLOCH_SIMD_HPP
#define PULSERVER_BLOCH_SIMD_HPP

#if defined(__x86_64__) || defined(_M_X64)
#define BLOCH_X86_64 1
#include <immintrin.h>
#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
/* MSVC compiles AVX2 and AVX-512 intrinsics in any function. */
#define BLOCH_AVX2
#define BLOCH_AVX512
#else
#include <cpuid.h>
#define BLOCH_AVX2 __attribute__((target("avx2,fma")))
#define BLOCH_AVX512 __attribute__((target("avx512f,avx2,fma")))
#endif
#endif

/* A loop whose iterations do not depend on each other through memory, so
 * that it is vectorised without checks for aliasing at run time. */
#if defined(__clang__)
#define BLOCH_INDEPENDENT _Pragma("clang loop vectorize(assume_safety)")
#elif defined(__GNUC__)
#define BLOCH_INDEPENDENT _Pragma("GCC ivdep")
#elif defined(_MSC_VER)
#define BLOCH_INDEPENDENT __pragma(loop(ivdep))
#else
#define BLOCH_INDEPENDENT
#endif

#if defined(__GNUC__)
#define BLOCH_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define BLOCH_INLINE __forceinline
#else
#define BLOCH_INLINE inline
#endif

namespace bloch
{

#ifdef BLOCH_X86_64
    /** Whether the processor has AVX2 and FMA, and the system saves the
     *  256-bit registers they use. */
    inline bool avx2_and_fma()
    {
#if defined(_MSC_VER) && !defined(__clang__)
        int info[4];
        __cpuid(info, 0);
        if (info[0] < 7)
            return false;
        __cpuid(info, 1);
        const unsigned features = static_cast<unsigned>(info[2]);
        const unsigned long long saved = (features & (1u << 27)) ? _xgetbv(0) : 0;
        __cpuidex(info, 7, 0);
        const unsigned extended = static_cast<unsigned>(info[1]);
#else
        unsigned a = 0, b = 0, features = 0, d = 0;
        if (__get_cpuid_max(0, nullptr) < 7 || !__get_cpuid(1, &a, &b, &features, &d))
            return false;
        unsigned low = 0, high = 0;
        if (features & (1u << 27))
            __asm__("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
        const unsigned long long saved = low;
        unsigned extended = 0, c = 0;
        __get_cpuid_count(7, 0, &a, &extended, &c, &d);
#endif
        const bool fma = (features & (1u << 12)) != 0;
        const bool avx = (features & (1u << 28)) != 0;
        const bool avx2 = (extended & (1u << 5)) != 0;
        return fma && avx && avx2 && (saved & 6) == 6;
    }

    /** Whether the processor has AVX-512F besides AVX2 and FMA, and the
     *  system saves the mask and 512-bit registers it uses. */
    inline bool avx512f()
    {
        if (!avx2_and_fma())
            return false;
#if defined(_MSC_VER) && !defined(__clang__)
        int info[4];
        __cpuidex(info, 7, 0);
        const unsigned extended = static_cast<unsigned>(info[1]);
        const unsigned long long saved = _xgetbv(0);
#else
        unsigned a = 0, extended = 0, c = 0, d = 0;
        __get_cpuid_count(7, 0, &a, &extended, &c, &d);
        unsigned low = 0, high = 0;
        __asm__("xgetbv" : "=a"(low), "=d"(high) : "c"(0));
        const unsigned long long saved = low;
#endif
        return (extended & (1u << 16)) != 0 && (saved & 0xE6) == 0xE6;
    }
#endif

} // namespace bloch

#endif /* PULSERVER_BLOCH_SIMD_HPP */
