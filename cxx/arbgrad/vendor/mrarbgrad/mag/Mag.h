#pragma once

#include <vector>
#include <list>
#include <tuple>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include "../utility/global.h"
#include "../utility/v3.h"
#include "../traj/TrajFunc.h"
#include "../utility/SplineIntp.h"
#include "../utility/LinIntp.h"

extern i64 gMag_oversamp;
extern bool gMag_enSFS;
extern bool gMag_enGradRep;
extern bool gMag_enTrajRep;
extern i64 gMrTraj_nTrajSampRsv;

// virtual TrajFunc, take in discrete samples and construct a Segmentied Cubic Polynomial function
class Spline_TrajFunc: public TrajFunc
{
public:
    Spline_TrajFunc():
        TrajFunc(0,0),
        m_intpX(gMrTraj_nTrajSampRsv),
        m_intpY(gMrTraj_nTrajSampRsv),
        m_intpZ(gMrTraj_nTrajSampRsv)
    {}

    Spline_TrajFunc(const vv3& vv3K):
        TrajFunc(0,0),
        m_intpX(vv3K.size()),
        m_intpY(vv3K.size()),
        m_intpZ(vv3K.size())
    {
        i64 lNTrajSamp = vv3K.size();

        vf64 vf64P(lNTrajSamp);
        vf64P[0] = 0;
        for (i64 i = 1; i < lNTrajSamp; ++i)
        {
            vf64P[i] = vf64P[i-1] + v3::norm(vv3K[i] - vv3K[i-1]);
        }

        vf64 vf64X(lNTrajSamp), vf64Y(lNTrajSamp), vf64Z(lNTrajSamp);
        for (i64 i = 0; i < lNTrajSamp; ++i)
        {
           vf64X[i] =  vv3K[i].x;
           vf64Y[i] =  vv3K[i].y;
           vf64Z[i] =  vv3K[i].z;
        }

        m_intpX.m_eSearchMode = Intp::ECached;
        m_intpY.m_eSearchMode = Intp::ECached;
        m_intpZ.m_eSearchMode = Intp::ECached;

        m_intpX.fit(vf64P, vf64X); 
        m_intpY.fit(vf64P, vf64Y);
        m_intpZ.fit(vf64P, vf64Z);

        m_p0 = *vf64P.begin();
        m_p1 = *vf64P.rbegin();
    }
    
    bool getK(v3* k, f64 p)
    {
        k->x = m_intpX.eval(p);
        k->y = m_intpY.eval(p);
        k->z = m_intpZ.eval(p);

        return true;
    }

    bool getDkDp(v3* dkdp, f64 p) const
    {
        dkdp->x = m_intpX.eval(p, 1);
        dkdp->y = m_intpY.eval(p, 1);
        dkdp->z = m_intpZ.eval(p, 1);

        return true;
    }

    bool getD2kDp2(v3* d2kdp2, f64 p) const
    {
        d2kdp2->x = m_intpX.eval(p, 2);
        d2kdp2->y = m_intpY.eval(p, 2);
        d2kdp2->z = m_intpZ.eval(p, 2);

        return true;
    }
protected:
    SplineIntp m_intpX, m_intpY, m_intpZ;
};

class Mag
{
public:
    Mag(i64 nSampReserve=0, i64 nTrajSampRsv=0);
    bool init
    (
        TrajFunc* ptTraj,
        f64 sLim, f64 gLim,
        f64 dt=10e-6, i64 oversamp=8, 
        f64 dG0Norm=0e0, f64 dG1Norm=0e0
    );
    bool init
    (
        const vv3& vv3TrajSamp,
        f64 sLim, f64 gLim,
        f64 dt=10e-6, i64 oversamp=8, 
        f64 dG0Norm=0e0, f64 dG1Norm=0e0
    );
    ~Mag();
    bool solve(vv3* plv3G, vf64* pvf64P=NULL);
    template <typename dtype, typename cv3>
    static bool decomp
    (
        std::vector<dtype>* pvfGx,
        std::vector<dtype>* pvfGy,
        std::vector<dtype>* pvfGz,
        const cv3& cv3G,
        bool bResize = false,
        bool bFillZero = true
    );
    static bool ramp_front(vv3* pvv3GRamp, const v3& v3G0, const v3& v3G0Des, f64 sLim, f64 dt);
    static f64 ramp_front(vv3* pvv3GRamp, const v3& v3G0, const v3& v3G0Des, i64 lNSamp, f64 dt);
    static bool ramp_back(vv3* pvv3GRamp, const v3& v3G1, const v3& v3G1Des, f64 sLim, f64 dt);
    static f64 ramp_back(vv3* pvv3GRamp, const v3& v3G1, const v3& v3G1Des, i64 lNSamp, f64 dt);
    static bool revGrad(v3* pv3M0Dst, vv3* pvv3Dst, const v3& v3M0Src, const vv3& vv3Src, f64 dt);
    static v3 calM0(const vv3& vv3G, f64 dt);
private:
    Spline_TrajFunc m_sptfTraj;
    TrajFunc* m_ptfTraj;
    f64 m_sLim, m_gLim;
    f64 m_dt;
    i64 m_oversamp;
    f64 m_g0Norm, m_g1Norm;

    // reserved vector for faster computation
    vf64 m_vf64P_Bac;
    vv3 m_vv3G_Bac;
    vf64 m_vf64GNorm_Bac;
    vf64 m_vf64P_For;
    vv3 m_vv3G_For;
    vf64 m_vf64P;
    vv3 m_vv3G;
    vv3 m_vv3TrajSamp;
    LinIntp m_intp; // SplineIntp m_intp;

    bool sovQDE(f64* psol0, f64* psol1, f64 a, f64 b, f64 c);
    f64 getCurRad(f64 p);
    f64 getDp(const v3& v3GPrev, const v3& v3GThis, f64 dt, f64 pPrev, f64 pThis, f64 signDp);
    bool step(v3* pv3GUnit, f64* gNormMin, f64* gNormMax, f64 p, f64 signDp, const v3& v3G, f64 sLim, f64 dt);
};

// definition must be in `.h` file (compiler limitation)
template <typename dtype, typename cv3>
bool Mag::decomp
(
    std::vector<dtype>* pvfGx,
    std::vector<dtype>* pvfGy,
    std::vector<dtype>* pvfGz,
    const cv3& cv3G,
    bool bResize,
    bool bFillZero
)
{
    if (bResize)
    {
        pvfGx->resize(cv3G.size());
        pvfGy->resize(cv3G.size());
        pvfGz->resize(cv3G.size());
    }
    if (bFillZero)
    {
        std::fill(pvfGx->begin(), pvfGx->end(), (dtype)0);
        std::fill(pvfGy->begin(), pvfGy->end(), (dtype)0);
        std::fill(pvfGz->begin(), pvfGz->end(), (dtype)0);
    }
    typename std::vector<dtype>::iterator ivfGx = pvfGx->begin();
    typename std::vector<dtype>::iterator ivfGy = pvfGy->begin();
    typename std::vector<dtype>::iterator ivfGz = pvfGz->begin();
    typename cv3::const_iterator icv3G = cv3G.begin();
    while (icv3G != cv3G.end())
    {
        *ivfGx = dtype(icv3G->x);
        *ivfGy = dtype(icv3G->y);
        *ivfGz = dtype(icv3G->z);
        ++ivfGx;
        ++ivfGy;
        ++ivfGz;
        ++icv3G;
    }
    return true;
}