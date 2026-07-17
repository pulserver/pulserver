#pragma once

#include "Intp.h"

class LinIntp : public Intp
{
public:
    LinIntp(i64 sizCache=0)
    {
        if (sizCache) init(sizCache);
    }

    LinIntp(const vf64& vf64X, const vf64& vf64Y)
    {
        init(vf64X.size());
        fit(vf64X, vf64Y);
    }

    void init(i64 sizCache)
    {
        m_sizCache = sizCache;
        m_vf64Slope.reserve(sizCache-1);

        m_vf64X.reserve(sizCache);
        m_vf64Y.reserve(sizCache);
    }

    virtual bool fit(const vf64& vf64X, const vf64& vf64Y)
    {
        ASSERT(vf64X.size() == vf64Y.size());
        const i64 nSamp = vf64X.size();
        ASSERT(nSamp >= 2);

        m_idxCache = 0;

        m_vf64X = vf64X;
        m_vf64Y = vf64Y;
        m_vf64Slope.resize(nSamp-1);

        for (i64 i=0; i < nSamp-1; ++i)
        {
            const f64 dx = m_vf64X[i+1] - m_vf64X[i];
            m_vf64Slope[i] = (m_vf64Y[i+1] - m_vf64Y[i]) / dx;
        }

        return true;
    }

    virtual f64 eval(f64 x, i64 ord = 0) const
    {
        ASSERT(m_vf64X.size() >= 2);

        const i64 idx = getIdx(x);
        const f64 dx = x - m_vf64X[idx];

        if (ord == 0)
        {
            return m_vf64Y[idx] + m_vf64Slope[idx] * dx;
        }
        if (ord == 1)
        {
            return m_vf64Slope[idx];
        }

        return 0e0;
    }

private:
    i64 m_sizCache;
    vf64 m_vf64Slope;
};
