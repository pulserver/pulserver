#include "v3.h"
#include <array>
#include <cstring> // test

v3::v3() :x(0e0), y(0e0), z(0e0) {}
v3::v3(f64 _) :x(_), y(_), z(_) {}
v3::v3(f64 x, f64 y, f64 z) :x(x), y(y), z(z) {}
v3::~v3() {}

v3 v3::operator+(const v3 &rhs) const
{
    return v3
    (
        this->x + rhs.x,
        this->y + rhs.y,
        this->z + rhs.z
    );
}

v3& v3::operator+=(const v3 &rhs)
{
    this->x += rhs.x;
    this->y += rhs.y;
    this->z += rhs.z;
    return *this;
}

v3 v3::operator+(const f64 &rhs) const
{
    return v3
    (
        this->x + rhs,
        this->y + rhs,
        this->z + rhs
    );
}

v3& v3::operator+=(const f64 &rhs)
{
    this->x += rhs;
    this->y += rhs;
    this->z += rhs;
    return *this;
}

v3 v3::operator-(const v3 &rhs) const
{
    return v3
    (
        this->x - rhs.x,
        this->y - rhs.y,
        this->z - rhs.z
    );
}

v3& v3::operator-=(const v3 &rhs)
{
    this->x -= rhs.x;
    this->y -= rhs.y;
    this->z -= rhs.z;
    return *this;
}

v3 v3::operator-(const f64 &rhs) const
{
    return v3
    (
        this->x - rhs,
        this->y - rhs,
        this->z - rhs
    );
}

v3& v3::operator-=(const f64 &rhs)
{
    this->x -= rhs;
    this->y -= rhs;
    this->z -= rhs;
    return *this;
}

v3 v3::operator*(const v3 &rhs) const
{
    return v3
    (
        this->x * rhs.x,
        this->y * rhs.y,
        this->z * rhs.z
    );
}

v3& v3::operator*=(const v3 &rhs)
{
    this->x *= rhs.x;
    this->y *= rhs.y;
    this->z *= rhs.z;
    return *this;
}

v3 v3::operator*(const f64 &rhs) const
{
    return v3
    (
        this->x * rhs,
        this->y * rhs,
        this->z * rhs
    );
}

v3& v3::operator*=(const f64 &rhs)
{
    this->x *= rhs;
    this->y *= rhs;
    this->z *= rhs;
    return *this;
}

v3 v3::operator/(const v3 &rhs) const
{
    return v3
    (
        this->x / rhs.x,
        this->y / rhs.y,
        this->z / rhs.z
    );
}

v3& v3::operator/=(const v3 &rhs)
{
    this->x /= rhs.x;
    this->y /= rhs.y;
    this->z /= rhs.z;
    return *this;
}

v3 v3::operator/(const f64 &rhs) const
{
    return v3
    (
        this->x / rhs,
        this->y / rhs,
        this->z / rhs
    );
}

v3& v3::operator/=(const f64 &rhs)
{
    this->x /= rhs;
    this->y /= rhs;
    this->z /= rhs;
    return *this;
}

bool v3::operator==(const v3 &rhs) const
{
    return bool
    (
        this->x == rhs.x &&
        this->y == rhs.y &&
        this->z == rhs.z
    );
}

bool v3::operator!=(const v3 &rhs) const
{
    return bool
    (
        this->x != rhs.x ||
        this->y != rhs.y ||
        this->z != rhs.z
    );
}

f64& v3::operator[](i64 idx)
{
    if (idx==0 || idx==-3) return x;
    if (idx==1 || idx==-2) return y;
    if (idx==2 || idx==-1) return z;
    throw std::runtime_error("idx");
}

f64 v3::operator[](i64 idx) const
{
    if (idx==0 || idx==-3) return x;
    if (idx==1 || idx==-2) return y;
    if (idx==2 || idx==-1) return z;
    throw std::runtime_error("idx");
}

f64 v3::norm(const v3& v3In)
{
    return sqrt
    (
        v3In.x*v3In.x +
        v3In.y*v3In.y +
        v3In.z*v3In.z
    );
}

v3 v3::cross(const v3& v3In0, const v3& v3In1)
{
    return v3
    (
        v3In0.y*v3In1.z - v3In0.z*v3In1.y,
        -v3In0.x*v3In1.z + v3In0.z*v3In1.x,
        v3In0.x*v3In1.y - v3In0.y*v3In1.x
    );
}

f64 v3::inner(const v3& v3In0, const v3& v3In1)
{
    return f64
    (
        v3In0.x*v3In1.x +
        v3In0.y*v3In1.y +
        v3In0.z*v3In1.z
    );
}

v3 v3::pow(const v3& v3In, f64 exp)
{
    return v3
    (
        std::pow(v3In.x, exp),
        std::pow(v3In.y, exp),
        std::pow(v3In.z, exp)
    );
}

bool v3::genRotMat(std::array<v3,3>* pav3RotMat, int iAx, f64 ang)
{
    if (!pav3RotMat) return false;
    switch (iAx)
    {
    case 0:
        (*pav3RotMat)[0] = v3(1e0, 0e0, 0e0);
        (*pav3RotMat)[1] = v3(0e0, std::cos(ang), -std::sin(ang));
        (*pav3RotMat)[2] = v3(0e0, std::sin(ang), std::cos(ang));
        break;
    case 1:
        (*pav3RotMat)[0] = v3(std::cos(ang), 0e0, std::sin(ang));
        (*pav3RotMat)[1] = v3(0e0, 1e0, 0e0);
        (*pav3RotMat)[2] = v3(-std::sin(ang), 0e0, std::cos(ang));
        break;
    case 2:
        (*pav3RotMat)[0] = v3(std::cos(ang), -std::sin(ang), 0e0);
        (*pav3RotMat)[1] = v3(std::sin(ang), std::cos(ang), 0e0);
        (*pav3RotMat)[2] = v3(0e0, 0e0, 1e0);
        break;
    default:
        return false;
    }

    return true;
}

bool v3::rotate
(
    v3* pv3Dst,
    int iAx, f64 ang,
    const v3& v3Src
)
{
    if (!pv3Dst) return false;
    bool ret = true;

    std::array<v3,3> av3RotMat;
    ret &= genRotMat(&av3RotMat, iAx, ang);

    *pv3Dst = v3
    (
        v3::inner(av3RotMat[0], v3Src),
        v3::inner(av3RotMat[1], v3Src),
        v3::inner(av3RotMat[2], v3Src)
    );

    return ret;
}

bool v3::rotate
(
    vv3* pvv3Dst,
    int iAx, f64 ang,
    const vv3& vv3Src
)
{
    if (!pvv3Dst) return false;
    std::array<v3, 3> av3RotMat;
    if (!genRotMat(&av3RotMat, iAx, ang)) return false;

    if (pvv3Dst->size() != vv3Src.size()) {
        pvv3Dst->resize(vv3Src.size());
    }

    for (size_t i = 0; i < vv3Src.size(); ++i)
    {
        f64 tx = v3::inner(av3RotMat[0], vv3Src[i]);
        f64 ty = v3::inner(av3RotMat[1], vv3Src[i]);
        f64 tz = v3::inner(av3RotMat[2], vv3Src[i]);

        (*pvv3Dst)[i].x = tx;
        (*pvv3Dst)[i].y = ty;
        (*pvv3Dst)[i].z = tz;
    }

    return true;
}

bool v3::rotate
(
    lv3* plv3Dst,
    int iAx, f64 ang,
    const lv3& lv3Src
)
{
    if (!plv3Dst) return false;
    bool ret = true;

    std::array<v3,3> av3RotMat;
    ret &= genRotMat(&av3RotMat, iAx, ang);

    // apply rotation matrix
    lv3 _lv3Dst; // for self-in self-out compatible
    lv3::const_iterator ilv3CoordSrc = lv3Src.begin();
    while (ilv3CoordSrc != lv3Src.end())
    {
        _lv3Dst.push_back
        (
            v3
            (
                v3::inner(av3RotMat[0], *ilv3CoordSrc),
                v3::inner(av3RotMat[1], *ilv3CoordSrc),
                v3::inner(av3RotMat[2], *ilv3CoordSrc)
            )
        );

        ++ilv3CoordSrc;
    }
    plv3Dst->swap(_lv3Dst);

    return ret;
}

v3 v3::axisroll(const v3& v3In, i64 nShift)
{
    v3 v3Ot;
    switch ((nShift%3+3)%3)
    {
    case 1:
        v3Ot.x = v3In.y;
        v3Ot.y = v3In.z;
        v3Ot.z = v3In.x;
        break;
        
    case 2:
        v3Ot.x = v3In.z;
        v3Ot.y = v3In.x;
        v3Ot.z = v3In.y;
        break;
    
    default:
        v3Ot = v3In;
        break;
    }
    return v3Ot;
}

bool v3::saveF64(FILE* pfHdr, FILE* pfBin, const vv3& vv3Data)
{
    bool ret = true;
    i64 lenData = vv3Data.size();
    fprintf(pfHdr, "float64[%ld][3];\n", (long)lenData);

    f64* bufFile = (f64*)malloc(lenData*3*sizeof(f64));
    for(i64 i=0; i<(i64)lenData; ++i)
    {
        for(i64 j=0; j<3; ++j)
        { bufFile[3*i+j] = (f64)vv3Data[i][j]; }
    }
    ret &= (i64)fwrite(bufFile, sizeof(f64), lenData*3, pfBin)==lenData*3;
    free(bufFile);
    return ret;
}

bool v3::loadF64(FILE* pfHdr, FILE* pfBin, vv3* pvv3Data)
{
    bool ret = true;
    pvv3Data->clear();
    i64 lenData = 0;
    {
        long _;
        int nRead = fscanf(pfHdr, "float64[%ld][3];\n", &_);
        lenData = (i64)_;
        if (nRead == EOF) return true; // EOF
        else if (nRead != 1) return false;
    }
    pvv3Data->resize(lenData);

    f64* bufFile = (f64*)malloc(lenData*3*sizeof(f64));
    ret &= (i64)fread(bufFile, sizeof(f64), lenData*3, pfBin)==lenData*3;
    for(i64 i=0; i<lenData; ++i)
    {
        for(i64 j=0; j<3; ++j)
        { (*pvv3Data)[i][j] = (f64)bufFile[3*i+j]; }
    }
    free(bufFile);
    return ret;
}

bool v3::saveF32(FILE* pfHdr, FILE* pfBin, const vv3& vv3Data)
{
    bool ret = true;
    i64 lenData = vv3Data.size();
    fprintf(pfHdr, "float32[%ld][3];\n", (long)lenData);

    f32* bufFile = (f32*)malloc(lenData*3*sizeof(f32));
    for(i64 i=0; i<(i64)lenData; ++i)
    {
        for(i64 j=0; j<3; ++j)
        { bufFile[3*i+j] = (f32)vv3Data[i][j]; }
    }
    ret &= (i64)fwrite(bufFile, sizeof(f32), lenData*3, pfBin)==lenData*3;
    free(bufFile);
    return ret;
}

bool v3::loadF32(FILE* pfHdr, FILE* pfBin, vv3* pvv3Data)
{
    bool ret = true;
    pvv3Data->clear();
    i64 lenData = 0;
    {
        long _;
        int nRead = fscanf(pfHdr, "float32[%ld][3];\n", &_);
        lenData = (i64)_;
        if (nRead == EOF) return true; // EOF
        else if (nRead != 1) return false;
    }
    pvv3Data->resize(lenData);

    f32* bufFile = (f32*)malloc(lenData*3*sizeof(f32));
    ret &= (i64)fread(bufFile, sizeof(f32), lenData*3, pfBin)==lenData*3;
    for(i64 i=0; i<lenData; ++i)
    {
        for(i64 j=0; j<3; ++j)
        { (*pvv3Data)[i][j] = (f64)bufFile[3*i+j]; }
    }
    free(bufFile);
    return ret;
}
