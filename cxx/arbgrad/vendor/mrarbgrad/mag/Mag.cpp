#include <cassert>
#include <algorithm>
#include <cstdio>
#include "../utility/global.h"
#include "Mag.h"

i64 gMag_oversamp = -1; // oversample ratio, overwrite the set value
bool gMag_enSFS = false; // Single Forward Sweep flag
bool gMag_enGradRep = true; // Gradient Reparameterization
bool gMag_enTrajRep = true; // use trajectory reparameterization for MAG solver

Mag::Mag(i64 nGradSampRsv, i64 nTrajSampRsv)
{
    // for solver
    if (nGradSampRsv)
    {
        m_vf64P_Bac.reserve(nGradSampRsv * 8);
        m_vv3G_Bac.reserve(nGradSampRsv * 8);
        m_vf64GNorm_Bac.reserve(nGradSampRsv * 8);

        m_vf64P_For.reserve(nGradSampRsv * 8);
        m_vv3G_For.reserve(nGradSampRsv * 8);

        m_vf64P.reserve(nGradSampRsv);
        m_vv3G.reserve(nGradSampRsv);

        m_intp.init(nGradSampRsv * 8);
        m_intp.m_eSearchMode = Intp::ECached;
    }

    // for trajectory reparameterization
    if (gMrTraj_nTrajSampRsv)
    {
        m_vv3TrajSamp.resize(gMrTraj_nTrajSampRsv);
    }
}

bool Mag::init
    (
        TrajFunc* ptTraj,
        f64 sLim, f64 gLim,
        f64 dt, i64 oversamp, 
        f64 g0Norm, f64 g1Norm
    )
{
    m_sLim = sLim;
    m_gLim = gLim;
    m_dt = dt;
    m_oversamp = gMag_oversamp>0?gMag_oversamp:oversamp;
    m_g0Norm = g0Norm;
    m_g1Norm = g1Norm;

    if (gMag_enTrajRep)
    {
        f64 p0 = ptTraj->getP0();
        f64 p1 = ptTraj->getP1();
        for (i64 i = 0; i < gMrTraj_nTrajSampRsv; ++i)
        {
            f64 p = p0 + (p1-p0) * (i)/f64(gMrTraj_nTrajSampRsv-1);
            ptTraj->getK(&m_vv3TrajSamp[i], p);
        }
        m_sptfTraj = Spline_TrajFunc(m_vv3TrajSamp);
        m_ptfTraj = &m_sptfTraj;
    }
    else
    {
        m_ptfTraj = ptTraj;
    }

    return true;
}

bool Mag::init
    (
        const vv3& vv3TrajSamp,
        f64 sLim, f64 gLim,
        f64 dt, i64 oversamp, 
        f64 g0Norm, f64 g1Norm
    )
{
    m_sptfTraj = Spline_TrajFunc(vv3TrajSamp);
    m_ptfTraj = &m_sptfTraj;
    m_sLim = sLim;
    m_gLim = gLim;
    m_dt = dt;
    m_oversamp = gMag_oversamp>0?gMag_oversamp:oversamp;
    m_g0Norm = g0Norm;
    m_g1Norm = g1Norm;

    return true;
}

Mag::~Mag()
{}

bool Mag::sovQDE(f64* psol0, f64* psol1, f64 a, f64 b, f64 c)
{
    f64 delta = b*b - 4e0*a*c;
    if (psol0) *psol0 = (-b-(delta<0?0:std::sqrt(delta)))/(2*a);
    if (psol1) *psol1 = (-b+(delta<0?0:std::sqrt(delta)))/(2*a);
    return delta>=0;
}

f64 Mag::getCurRad(f64 p)
{
    v3 dkdp; m_ptfTraj->getDkDp(&dkdp, p);
    v3 d2kdp2; m_ptfTraj->getD2kDp2(&d2kdp2, p);
    f64 nume = pow(v3::norm(dkdp), 3e0);
    f64 deno = v3::norm(v3::cross(dkdp, d2kdp2));
    return nume/deno;
}

#if 1

f64 Mag::getDp(const v3& v3GPrev, const v3& v3GThis, f64 dt, f64 pPrev, f64 pThis, f64 signDp)
{
    // solve `ΔP` by RK2
    f64 l = v3::norm(v3GThis)*dt;
    // k1
    f64 k1;
    {
        v3 dkdp; m_ptfTraj->getDkDp(&dkdp, pThis);
        f64 dldp = v3::norm(dkdp)*signDp;
        k1 = 1e0/dldp;
    }
    // k2
    f64 k2;
    {
        v3 dkdp; m_ptfTraj->getDkDp(&dkdp, pThis+k1*l);
        f64 dldp = v3::norm(dkdp)*signDp;
        k2 = 1e0/dldp;
    }
    f64 dp = l*(0.5*k1 + 0.5*k2);
    return dp;
}

#else // less accurate due to estimation of PNext

f64 Mag::getDp(const v3& v3GPrev, const v3& v3GThis, f64 dt, f64 pPrev, f64 pThis, f64 signDp)
{
    // solve `ΔP` by RK2
    f64 l = v3::norm(v3GThis)*dt;
    v3 dkdp0; m_ptfTraj->getDkDp(&dkdp0, pThis);
    v3 dkdp1; m_ptfTraj->getDkDp(&dkdp1, pThis*2e0-pPrev);
    f64 dldp0 = v3::norm(dkdp0)*signDp;
    f64 dldp1 = v3::norm(dkdp1)*signDp;
    return l*(1e0/dldp0 + 1e0/dldp1)/2e0;
}

#endif

bool Mag::step(v3* gUnit, f64* gNormMin, f64* gNormMax, f64 p, f64 signDp, const v3& g, f64 sLim, f64 dt)
{
    // current gradient direction
    v3 dkdp; m_ptfTraj->getDkDp(&dkdp, p);
    f64 dldp = v3::norm(dkdp)*signDp;
    if (gUnit) *gUnit = dkdp/dldp;
    
    // current gradient magnitude
    bool isQDESucc = sovQDE
    (
        gNormMin, gNormMax,
        1e0,
        -2e0*v3::inner(g, *gUnit),
        v3::inner(g, g) - std::pow(sLim*dt, 2e0)
    );
    if (gNormMin) *gNormMin = fabs(*gNormMin);
    if (gNormMax) *gNormMax = fabs(*gNormMax);

    return isQDESucc;
}

bool Mag::solve(vv3* pvv3G, vf64* pvf64P)
{
    bool ret = true;
    f64 p0 = m_ptfTraj->getP0();
    f64 p1 = m_ptfTraj->getP1();
    m_vv3G.clear(); if (!pvv3G) pvv3G = &m_vv3G;
    m_vf64P.clear(); if (!pvf64P) pvf64P = &m_vf64P;
    bool isQDESucc = true; (void)isQDESucc;
    i64 nIter = 0;

    // backward
    v3 g1Unit; ret &= m_ptfTraj->getDkDp(&g1Unit, p1);
    g1Unit = g1Unit * (p0>p1?1e0:-1e0);
    g1Unit = g1Unit / v3::norm(g1Unit);
    f64 g1Norm = m_g1Norm;
    g1Norm = std::min(g1Norm, m_gLim);
    g1Norm = std::min(g1Norm, std::sqrt(m_sLim*getCurRad(p1)));
    v3 g1 = g1Unit * g1Norm;

    m_vf64P_Bac.clear(); m_vf64P_Bac.push_back(p1);
    m_vv3G_Bac.clear(); m_vv3G_Bac.push_back(g1);
    m_vf64GNorm_Bac.clear(); m_vf64GNorm_Bac.push_back(v3::norm(g1));
    while (!gMag_enSFS)
    {
        f64 p = m_vf64P_Bac.back();
        v3 g = m_vv3G_Bac.back();
        
        // update grad
        v3 gUnit;
        f64 gNorm;
        isQDESucc = step(&gUnit, NULL, &gNorm, p, (p0-p1)/std::fabs(p0-p1), g, m_sLim, m_dt/m_oversamp);
        gNorm = std::min(gNorm, m_gLim);
        gNorm = std::min(gNorm, std::sqrt(m_sLim*getCurRad(p)));
        g = gUnit*gNorm;

        // update para
        p += getDp(m_vv3G_Bac.back(), g, m_dt/m_oversamp, m_vf64P_Bac.back(), p, (p0-p1)/std::fabs(p0-p1));

        // stop or append
        if (std::fabs(m_vf64P_Bac.back() - p1) >= (1-1e-6)*std::fabs(p0 - p1))
        {
            // printf("bac: dP/dP1 = %lf/%lf\n", dP, dP1); // test
            break;
        }
        else
        {
            // printf("bac: dP = %lf\n", dP); // test
            m_vf64P_Bac.push_back(p);
            m_vv3G_Bac.push_back(g);
            m_vf64GNorm_Bac.push_back(v3::norm(g));
        }
    }

    std::reverse(m_vf64P_Bac.begin(), m_vf64P_Bac.end());
    std::reverse(m_vf64GNorm_Bac.begin(), m_vf64GNorm_Bac.end());

    if (!gMag_enSFS) m_intp.fit(m_vf64P_Bac, m_vf64GNorm_Bac);
    
    nIter += m_vf64P_Bac.size();

    // forward
    v3 g0Unit; ret &= m_ptfTraj->getDkDp(&g0Unit, p0);
    g0Unit = g0Unit * (p1>p0?1e0:-1e0);
    g0Unit = g0Unit / v3::norm(g0Unit);
    f64 g0Norm = m_g0Norm;
    g0Norm = std::min(g0Norm, m_gLim);
    g0Norm = std::min(g0Norm, std::sqrt(m_sLim*getCurRad(p0)));
    g0Norm = std::min(g0Norm, gMag_enSFS?1e15:m_intp.eval(p0));
    v3 g0 = g0Unit * g0Norm;

    m_vf64P_For.clear(); m_vf64P_For.push_back(p0);
    m_vv3G_For.clear(); m_vv3G_For.push_back(g0);
    while (1)
    {
        f64 p = m_vf64P_For.back();
        v3 g = m_vv3G_For.back();

        // update grad
        v3 gUnit;
        f64 gNorm;
        isQDESucc = step(&gUnit, NULL, &gNorm, p, (p1-p0)/std::fabs(p1-p0), g, m_sLim, m_dt/m_oversamp);
        if (gMag_enSFS)
        {
            gNorm = std::min(gNorm, m_gLim);
            // dGNorm = std::min(dGNorm, std::sqrt(m_sLim*getCurRad(dP)));
        }
        else
        {
            f64 gNormBac = m_intp.eval(p);
            gNorm = std::min(gNorm, gNormBac);
            if (gNormBac<=0)
            {
                gNorm *= -1;
                gUnit *= -1;
            }
        }
        g = gUnit*gNorm;

        // update para
        p += getDp(m_vv3G_For.back(), g, m_dt/m_oversamp, m_vf64P_For.back(), p, (p1-p0)/std::fabs(p1-p0));

        // stop or append
        if (std::fabs(m_vf64P_For.back() - p0) >= (1-1e-6)*std::fabs(p1 - p0)) // || dGNorm <= 0)
        {
            // printf("for: dP/dP1 = %lf/%lf\n", dP, dP1); // test
            break;
        }
        else
        {
            // printf("for: dP = %lf\n", dP); // test
            m_vf64P_For.push_back(p);
            m_vv3G_For.push_back(g);
        }
    }
    nIter += m_vf64P_For.size();
    
    if (glob_enDbgPrint)
    {
        i64 MAG_Nit = nIter;
        PRINT(MAG_Nit);
    }

    // deoversamp the para. vec.
    {
        pvf64P->clear();
        i64 n = m_vf64P_For.size();
        for (i64 i = 0; i < n; ++i)
        {
            if (i%m_oversamp==m_oversamp/2) pvf64P->push_back(m_vf64P_For[i]);
        }
    }

    // derive gradient
    if (gMag_enGradRep)
    {
        v3 v3K1, v3K0; 
        vf64::iterator ivf64P = std::next(pvf64P->begin());
        i64 n = pvf64P->size();
        pvv3G->clear();
        pvv3G->reserve(n-1);
        for (i64 i = 1; i < n; ++i)
        {
            ret &= m_ptfTraj->getK(&v3K1, *ivf64P);
            ret &= m_ptfTraj->getK(&v3K0, *std::prev(ivf64P));
            pvv3G->push_back((v3K1 - v3K0)/m_dt);
            ++ivf64P;
        }
    }
    else
    {
        i64 n = m_vv3G_For.size();
        pvv3G->clear();
        pvv3G->reserve(n);
        for (i64 i = 0; i < n; ++i)
        {
            if(i%m_oversamp==0) pvv3G->push_back(m_vv3G_For[i]);
        }
    }

    // [WARN] `p` sequence has 1 more element than `g` sequence
    return ret;
}

bool Mag::ramp_front(vv3* pvv3GRamp, const v3& g0, const v3& g0Des, f64 sLim, f64 dt)
{
    v3 dg = g0Des - g0;
    v3 dgUnit = v3::norm(dg)!=0 ? dg/v3::norm(dg) : v3(0,0,0);
    i64 nSamp = (i64)std::ceil(v3::norm(dg)/(sLim*dt));

    // derive ramp gradient
    pvv3GRamp->clear();
    if (nSamp==0) return true;

    pvv3GRamp->reserve(nSamp);
    pvv3GRamp->push_back(g0Des);
    for (i64 i = nSamp-1; i > 0; --i)
    {
        pvv3GRamp->push_back(g0 + dgUnit * sLim * (dt*i));
    }
    
    return true;
}

f64 Mag::ramp_front(vv3* pvv3GRamp, const v3& g0, const v3& g0Des, i64 nSamp, f64 dt)
{
    v3 dg = g0Des - g0;
    v3 dgUnit = v3::norm(dg)!=0 ? dg/v3::norm(dg) : v3(0,0,0);
    f64 sLim = v3::norm(dg)/(nSamp*dt);

    // derive ramp gradient
    pvv3GRamp->clear();
    if (nSamp==0) return true;

    pvv3GRamp->reserve(nSamp);
    pvv3GRamp->push_back(g0Des);
    for (i64 i = nSamp-1; i > 0; --i)
    {
        pvv3GRamp->push_back(g0 + dgUnit * sLim * (dt*i));
    }

    return sLim;
}

bool Mag::ramp_back(vv3* pvv3GRamp, const v3& g1, const v3& g1Des, f64 sLim, f64 dt)
{
    v3 dg = g1Des - g1;
    v3 dgUnit = v3::norm(dg)!=0 ? dg/v3::norm(dg) : v3(0,0,0);
    i64 nSamp = (i64)std::ceil(v3::norm(dg)/(sLim*dt));

    // derive ramp gradient
    pvv3GRamp->clear();
    if (nSamp==0) return true;

    pvv3GRamp->reserve(nSamp);
    for (i64 i = 1; i < nSamp; ++i)
    {
        pvv3GRamp->push_back(g1 + dgUnit * sLim * (dt*i));
    }
    pvv3GRamp->push_back(g1Des);
    
    return true;
}

f64 Mag::ramp_back(vv3* pvv3GRamp, const v3& g1, const v3& g1Des, i64 nSamp, f64 dt)
{
    v3 dg = g1Des - g1;
    v3 dgUnit = v3::norm(dg)!=0 ? dg/v3::norm(dg) : v3(0,0,0);
    f64 sLim = v3::norm(dg)/(nSamp*dt);

    // derive ramp gradient
    pvv3GRamp->clear();
    if (nSamp==0) return true;

    pvv3GRamp->reserve(nSamp);
    for (i64 i = 1; i < nSamp; ++i)
    {
        pvv3GRamp->push_back(g1 + dgUnit * sLim * (dt*i));
    }
    pvv3GRamp->push_back(g1Des);
    
    return sLim;
}

bool Mag::revGrad(v3* pv3M0Dst, vv3* pvv3Dst, const v3& v3M0Src, const vv3& vv3Src, f64 dt)
{
    bool ret = true;
    i64 nSamp = vv3Src.size();

    ASSERT(nSamp > 1);
    ASSERT((i64)pvv3Dst->capacity() >= nSamp);

    // derive Total M0
    *pv3M0Dst = v3M0Src + calM0(vv3Src, dt);
    
    // reverse gradient
    pvv3Dst->resize(nSamp);
    for (int64_t i = 0; i < nSamp; ++i)
    {
        (*pvv3Dst)[i] = vv3Src[nSamp-1-i]*(-1);
    }

    return ret;
}

v3 Mag::calM0(const vv3& vv3G, f64 dt)
{
    v3 M0 = v3(0,0,0);
    for (int64_t i = 1; i < (i64)vv3G.size(); ++i)
    {
        M0 += (vv3G[i] + vv3G[i-1])*dt/2e0;
    }

    return M0;
}