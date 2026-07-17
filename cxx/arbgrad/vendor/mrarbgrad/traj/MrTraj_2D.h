#pragma once

#include "MrTraj.h"
#include "../mag/Mag.h"

class MrTraj_2D: public MrTraj
{
public:
    MrTraj_2D(const GeoPara& m_objGeoPara, const GradPara& m_objGradPara, const i64& m_nAcq, const i64& m_nSampMax, const i64& m_nStack, const f64& m_rotang, const v3& m_v3BaseM0PE, const vv3& m_vv3BaseGRO):
        MrTraj(m_objGeoPara, m_objGradPara, m_nAcq, m_nSampMax),
        m_nStack(m_nStack),
        m_rotang(m_rotang),
        m_v3BaseM0PE(m_v3BaseM0PE),
        m_vv3BaseGRO(m_vv3BaseGRO)
    {}
    
    virtual ~MrTraj_2D()
    {}

    virtual bool getGrad(v3* pv3M0PE, vv3* pvv3GRO, i64 iAcq)
    {
        bool ret = true;
        i64 nStack = getNStack();
        f64 rotang = getRotAng();
        i64 iStack = iAcq%nStack;
        i64 iRot = iAcq/nStack;
        
        if (pv3M0PE)
        {
            *pv3M0PE = m_v3BaseM0PE;
            pv3M0PE->z += getK0z(iStack, nStack);
            ret &= v3::rotate(pv3M0PE, 2, rotang * iRot, *pv3M0PE);
        }
        if (pvv3GRO)
        {
            ret &= v3::rotate(pvv3GRO, 2, rotang * iRot, m_vv3BaseGRO);
        }

        return ret;
    }

    i64 getNStack()
    { return m_nStack; }

    f64 getRotAng()
    { return m_rotang; }

    void setNStack(i64 nStack)
    { m_nStack = nStack; }

    void setRotAng(f64 rotang)
    { m_rotang = rotang; }

protected:
    i64 m_nStack;
    f64 m_rotang;

    v3 m_v3BaseM0PE;
    vv3 m_vv3BaseGRO;

    static f64 getK0z(i64 iStack, i64 nStack=256)
    { return iStack/(f64)nStack - (nStack/2)/(f64)nStack; }
};
