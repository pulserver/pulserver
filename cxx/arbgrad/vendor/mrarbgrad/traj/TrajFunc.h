#pragma once

#include "../utility/global.h"
#include "../utility/v3.h"
// #include "../mag/Mag.h" // VS2010 reports error on this

/* 
 * Single trajectory define by a parameterized function getK()
 * and parameter bounding m_dP0, m_dP1
 */

class TrajFunc
{
public:
    TrajFunc(const f64& m_p0, const f64& m_p1):
        m_p0(m_p0), m_p1(m_p1)
    {}

    virtual ~TrajFunc()
    {}
    
    virtual bool getK(v3* k, f64 p) = 0; // trajectory function
    
    virtual bool getDkDp(v3* dkdp, f64 p) // 1st-ord differentiative of trajectory function
    {
        static const f64 dp = 1e-7;
        v3 v3K_Nx1; getK(&v3K_Nx1, p+dp);
        v3 v3K_Pv1; getK(&v3K_Pv1, p-dp);
        *dkdp = (v3K_Nx1-v3K_Pv1)/(2e0*dp);

        return true;
    }

    virtual bool getD2kDp2(v3* d2kdp2, f64 p) // 2nd-ord differentiative of trajectory function
    {
        static const f64 dp = 1e-3;
        v3 v3K_Nx1; getK(&v3K_Nx1, p+dp);
        v3 v3K_This; getK(&v3K_This, p);
        v3 v3K_Pv1; getK(&v3K_Pv1, p-dp);
        *d2kdp2 = (v3K_Nx1-v3K_This*2e0+v3K_Pv1)/(dp*dp);
    
        return true;
    }

    // get the lower bound of traj. para.
    f64 getP0()
    { return m_p0; }

    // get the higher bound of traj. para.
    f64 getP1()
    { return m_p1; }

    bool getK0(v3* k0)
    { return getK(k0, m_p0); }
    
    bool getK1(v3* k1)
    { return getK(k1, m_p1); }

    // convinient interface
    v3 getK(f64 p)
    {
        v3 k; getK(&k, p);
        return k;
    }

    v3 getDkDp(f64 p)
    {
        v3 dkdp; getDkDp(&dkdp, p);
        return dkdp;
    }

    v3 getD2kDp2(f64 p)
    {
        v3 d2kdp2; getD2kDp2(&d2kdp2, p);
        return d2kdp2;
    }
    v3 getK0(f64 p)
    {
        v3 k0; getK0(&k0);
        return k0;
    }
    v3 getK1(f64 p)
    {
        v3 k1; getK1(&k1);
        return k1;
    }
    
protected:
    f64 m_p0, m_p1;
};
