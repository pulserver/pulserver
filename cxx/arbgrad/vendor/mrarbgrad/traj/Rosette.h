#pragma once

#include "TrajFunc.h"
#include "MrTraj_2D.h"

class Rosette_TrajFunc: public TrajFunc
{
public:
    Rosette_TrajFunc(f64 om1, f64 om2, f64 tMax=1e0):
        TrajFunc(0,0)
    {
        /*
         * NOTE:
         * When tMax=1, om1=Npi, om2=(N-2)pi,
         * there will be N petal because om1
         * controls how fast the rho changes.
         */
        m_om1 = om1;
        m_om2 = om2;
        m_tMax = tMax;

        m_p0 = 0e0;
        m_p1 = m_tMax;
    }

    virtual bool getK(v3* k, f64 p)
    {
        if (k==NULL) return false;
        
        f64& t = p;
        f64 dRho = 0.5e0*std::sin(m_om1*t);
        k->x = dRho * std::cos(m_om2*t);
        k->y = dRho * std::sin(m_om2*t);
        k->z = 0e0;

        return true;
    }

protected:
    f64 m_om1, m_om2, m_tMax;
};

class Rosette: public MrTraj_2D
{
public:
    Rosette(const GeoPara& objGeoPara, const GradPara& objGradPara, i64 nStack, f64 om1, f64 om2, f64 tMax):
        MrTraj_2D(objGeoPara,objGradPara,0,0,0,0,v3(),vv3())
    {
        m_ptfBaseTraj = new Rosette_TrajFunc(om1, om2, tMax);
        ASSERT(m_ptfBaseTraj!=NULL);
        m_nStack = nStack;

        i64 nRot = calNRot(m_ptfBaseTraj, 0e0, (M_PI/2e0)/om1, m_objGeoPara.nPix);
        m_rotang = calRotAng(nRot);
        m_nAcq = nRot*m_nStack;
        
        calGrad(&m_v3BaseM0PE, &m_vv3BaseGRO, NULL, *m_ptfBaseTraj, m_objGradPara);
        m_nSampMax = m_vv3BaseGRO.size();
    }
    
    virtual ~Rosette()
    {
        delete m_ptfBaseTraj;
    }

protected:
    TrajFunc* m_ptfBaseTraj;
};

class Rosette_Trad: public MrTraj_2D
{
public:
    Rosette_Trad(const GeoPara& objGeoPara, const GradPara& objGradPara, i64 nStack, f64 om1, f64 om2, f64 tMax, f64 dTE):
        MrTraj_2D(objGeoPara,objGradPara,0,0,0,0,v3(),vv3())
    {
        m_ptfBaseTraj = new Rosette_TrajFunc(om1, om2, tMax);
        ASSERT(m_ptfBaseTraj!=NULL);
        m_nStack = nStack;
        i64 nRot = calNRot(m_ptfBaseTraj, 0e0, (M_PI/2e0)/om1, m_objGeoPara.nPix);
        m_nAcq = nRot*m_nStack;
        m_rotang = calRotAng(nRot);

        // readout
        f64 tAcq = dTE*om1/M_PI;
        m_nSampMax = tAcq/m_objGradPara.dt;
        m_vv3BaseGRO.reserve(m_nSampMax);
        for(i64 i = 0; i < m_nSampMax; ++i)
        {
            m_vv3BaseGRO.push_back(v3());
            m_ptfBaseTraj->getDkDp(&*m_vv3BaseGRO.rbegin(), tMax*i/(f64)m_nSampMax); // derivative to p
            *m_vv3BaseGRO.rbegin() *= tMax/tAcq; // derivative to t
        }

        // calculate M0 of PE
        m_ptfBaseTraj->getK0(&m_v3BaseM0PE);
    }
    
    virtual ~Rosette_Trad()
    {
        delete m_ptfBaseTraj;
    }

protected:
    TrajFunc* m_ptfBaseTraj;
};
