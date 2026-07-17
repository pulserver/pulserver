#pragma once

#include "TrajFunc.h"
#include "MrTraj_2D.h"

class VDSpiral_TrajFunc: public TrajFunc
{
public:
    VDSpiral_TrajFunc(f64 kRhoPhi0, f64 kRhoPhi1, f64 phi0=0e0):
        TrajFunc(0,0)
    {
        m_kRhoPhi0 = kRhoPhi0;
        m_kRhoPhi1 = kRhoPhi1;
        m_phi0 = phi0;

        m_p0 = 0e0;
        if (m_kRhoPhi0!=m_kRhoPhi1)
        { m_p1 = (std::log(kRhoPhi1)-std::log(kRhoPhi0)) / (2e0*(kRhoPhi1-kRhoPhi0)); }
        else
        { m_p1 = 0.5e0/kRhoPhi0; }
    }

    virtual bool getK(v3* k, f64 p)
    {
        if (k==NULL) return false;
        
        f64& phi = p;
        f64 rho;
        if (m_kRhoPhi0!=m_kRhoPhi1)
        { rho = m_kRhoPhi0*(std::exp(2e0*(m_kRhoPhi1 - m_kRhoPhi0)*phi) - 1e0) / (2e0*(m_kRhoPhi1 - m_kRhoPhi0)); }
        else
        { rho = m_kRhoPhi0*phi; }
        k->x = rho * std::cos(phi + m_phi0);
        k->y = rho * std::sin(phi + m_phi0);
        k->z = 0e0;

        return true;
    }

protected:
    f64 m_kRhoPhi0;
    f64 m_kRhoPhi1;
    f64 m_phi0;
};

class VDSpiral: public MrTraj_2D
{
public:
    VDSpiral(const GeoPara& objGeoPara, const GradPara& objGradPara, i64 nStack, f64 kRhoPhi0, f64 kRhoPhi1):
        MrTraj_2D(objGeoPara,objGradPara,0,0,0,0,v3(),vv3())
    {
        m_ptfBaseTraj = new VDSpiral_TrajFunc(kRhoPhi0, kRhoPhi1);
        ASSERT(m_ptfBaseTraj!=NULL);
        m_nStack = nStack;

        i64 nRot = calNRot(std::max(kRhoPhi0, kRhoPhi1), m_objGeoPara.nPix);
        m_rotang = calRotAng(nRot);
        m_nAcq = nRot*m_nStack;

        calGrad(&m_v3BaseM0PE, &m_vv3BaseGRO, NULL, *m_ptfBaseTraj, m_objGradPara);
        m_nSampMax = m_vv3BaseGRO.size();
    }
    
    virtual ~VDSpiral()
    {
        delete m_ptfBaseTraj;
    }

protected:
    TrajFunc* m_ptfBaseTraj;
};

class VDSpiral_RT: public MrTraj
{
    // TODO: Goldang sampling is incomplete, shuffled sampling is incomplete.
public:
    VDSpiral_RT(const GeoPara& objGeoPara, const GradPara& objGradPara, f64 kRhoPhi0, f64 kRhoPhi1):
    /*
     * nAcq: Num of Acq, used to preallocate an array to store PE M0
     */
        MrTraj(objGeoPara,objGradPara,0,0)
    {
        m_kRhoPhi0 = kRhoPhi0;
        m_kRhoPhi1 = kRhoPhi1;
        m_nAcq = calNRot(std::max(kRhoPhi0, kRhoPhi1), objGeoPara.nPix);
        m_dRotAng = 2e0*M_PI/m_nAcq;
        genPermTab(&m_vi64PermTab, m_nAcq);

        VDSpiral_TrajFunc tf(m_kRhoPhi0, m_kRhoPhi1, 0);
        vv3 vv3GRO; calGrad(NULL, &vv3GRO, NULL, tf, m_objGradPara, 4);
        m_nSampMax = vv3GRO.size() + 0;
    }

    virtual ~VDSpiral_RT()
    {}

    virtual bool getGrad(v3* pv3M0PE, vv3* pvv3GRO, i64 iAcq)
    {
        bool ret = true;
        ASSERT(iAcq >= 0);
        i64 iPhi = m_vi64PermTab[iAcq%m_nAcq];
        f64 phi = iPhi*m_dRotAng;
        VDSpiral_TrajFunc tf(m_kRhoPhi0, m_kRhoPhi1, phi);
        ret &= calGrad(pv3M0PE, pvv3GRO, NULL, tf, m_objGradPara, 2);
        return ret;
    }

protected:
    f64 m_kRhoPhi0;
    f64 m_kRhoPhi1;
    f64 m_dRotAng;
    vi64 m_vi64PermTab;
};