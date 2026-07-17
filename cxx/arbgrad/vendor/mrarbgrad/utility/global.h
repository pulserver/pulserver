#pragma once

#include <cmath>
#include <ctime>
#include <vector>
#include <list>
#include <string>
#include <cstdio>
#include <cstdint>
#include <stdexcept>

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float f32;
typedef double f64;

typedef std::vector<i64> vi64;
typedef std::vector<f64> vf64;
typedef std::list<i64> li64;
typedef std::list<f64> lf64;
typedef std::vector<i32> vi32;
typedef std::vector<f32> vf32;
typedef std::list<i32> li32;
typedef std::list<f32> lf32;

typedef std::string str;

template<typename T>
inline T round(T x)
{ return (x >= 0) ? std::floor(x + T(0.5)) : std::ceil(x - T(0.5)); }

template<typename T>
inline T gcd(T x, T y)
{ return y==0 ? x : gcd(y, x%y); }

#undef M_PI
#define M_PI (3.1415926535897931e0)
#undef M_SQRT2
#define M_SQRT2 (1.4142135623730951e0)
#undef M_SQRT3
#define M_SQRT3 (1.7320508075688772e0)
#undef M_SQRT5
#define M_SQRT5 (2.2360679774997898e0)
#undef M_SQRT7
#define M_SQRT7 (2.6457513110645907e0)

#define GOLDRAT (1.6180339887498949e0) // ((1e0+std::sqrt(5e0))/2e0)
#define GOLDANG (2.3999632297286531e0) // ((3e0-std::sqrt(5e0))*M_PI)

#define PRINT(X) {printf("%s: %ld\n", #X, (long)(X));}
#define PRINT_F(X) {printf("%s: %.3lf\n", #X, (f64)(X));}
#define PRINT_E(X) {printf("%s: %.3e\n", #X, (f64)(X));}
#define ASSERT(X) {if(!(X)) throw std::runtime_error(#X);}

#define TIC \
    clock_t cTick = std::clock();\

#define TOC \
    cTick = std::clock() - cTick;\
    if (glob_enDbgPrint) printf("Elapsed time: %.3lf ms\n", 1e3*cTick/CLOCKS_PER_SEC);

extern bool glob_enDbgPrint;
