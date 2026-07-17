#pragma once

#include <cmath>
#include <list>
#include <array>
#include "global.h"

class v3;

typedef std::vector<v3> vv3;
typedef std::vector<vv3> vvv3;
typedef std::list<v3> lv3;

class v3
{
public:
    f64 x, y, z;

    v3();
    v3(f64 _);
    v3(f64 x, f64 y, f64 z);
    ~v3();
    v3 operator+(const v3 &rhs) const;
    v3& operator+=(const v3 &rhs);
    v3 operator+(const f64 &rhs) const;
    v3& operator+=(const f64 &rhs);
    v3 operator-(const v3 &rhs) const;
    v3& operator-=(const v3 &rhs);
    v3 operator-(const f64 &rhs) const;
    v3& operator-=(const f64 &rhs);
    v3 operator*(const v3 &rhs) const;
    v3& operator*=(const v3 &rhs);
    v3 operator*(const f64 &rhs) const;
    v3& operator*=(const f64 &rhs);
    v3 operator/(const v3 &rhs) const;
    v3& operator/=(const v3 &rhs);
    v3 operator/(const f64 &rhs) const;
    v3& operator/=(const f64 &rhs);
    bool operator==(const v3 &rhs) const;
    bool operator!=(const v3 &rhs) const;
    f64& operator[](i64 idx);
    f64 operator[](i64 idx) const;
    static f64 norm(const v3& v3In);
    static v3 cross(const v3& v3In0, const v3& v3In1);
    static f64 inner(const v3& v3In0, const v3& v3In1);
    static v3 pow(const v3& v3In, f64 exp);
    static bool rotate
    (
        v3* pv3Dst,
        int iAx, f64 ang,
        const v3& v3Src
    );
    static bool rotate
    (
        vv3* pvv3Dst,
        int iAx, f64 ang,
        const vv3& vv3Src
    );
    static bool rotate
    (
        lv3* plv3Dst,
        int iAx, f64 ang,
        const lv3& lv3Src
    );
    static v3 axisroll(const v3& v3In, i64 lShift);
    static bool saveF64(FILE* pfHdr, FILE* pfBin, const vv3& vv3Data);
    static bool loadF64(FILE* pfHdr, FILE* pfBin, vv3* pvv3Data);
    static bool saveF32(FILE* pfHdr, FILE* pfBin, const vv3& vv3Data);
    static bool loadF32(FILE* pfHdr, FILE* pfBin, vv3* pvv3Data);
private:
    static bool genRotMat(std::array<v3,3>* pav3RotMat, int iAx, f64 ang);
};
