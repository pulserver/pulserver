#pragma once

#include "Intp.h"

class SplineIntp : public Intp
{
public:
    SplineIntp(i64 sizCache=0)
    {
        if (sizCache) init(sizCache);
    }

    SplineIntp(const vf64& vf64X, const vf64& vf64Y)
    {
        init(vf64X.size());
        fit(vf64X, vf64Y);
    }

    void init(i64 sizCache)
    {
        m_sizCache = sizCache;

        m_vf64X.reserve(sizCache);
        m_vf64Y.reserve(sizCache);

        m_vf64H.reserve(sizCache-1);
        m_vf64Alpha.reserve(sizCache);
        m_vf64L.reserve(sizCache);
        m_vf64Mu.reserve(sizCache);
        m_vf64Z.reserve(sizCache);

        m_vf64A.reserve(sizCache);
        m_vf64B.reserve(sizCache-1);
        m_vf64C.reserve(sizCache);
        m_vf64D.reserve(sizCache-1);
    }

    virtual bool fit(const vf64& vf64X, const vf64& vf64Y)
    {
        ASSERT(vf64X.size() == vf64Y.size());
        const i64 nSamp = vf64X.size();
        ASSERT(nSamp >= 2);

        m_idxCache = 0;

        m_vf64X = vf64X;
        m_vf64Y = vf64Y;

        m_vf64H.resize(nSamp-1);
        m_vf64Alpha.resize(nSamp);
        m_vf64L.resize(nSamp);
        m_vf64Mu.resize(nSamp);
        m_vf64Z.resize(nSamp);

        m_vf64A = vf64Y;
        m_vf64B.resize(nSamp-1);
        m_vf64C.resize(nSamp);
        m_vf64D.resize(nSamp-1);

        m_vf64L[0]  = 1.0;
        m_vf64Mu[0] = 0.0;
        m_vf64Z[0]  = 0.0;
        m_vf64Alpha[0] = 0.0;
        m_vf64Alpha[nSamp-1] = 0.0;

        for (i64 i = 0; i < nSamp-1; ++i)
        {
            m_vf64H[i] = m_vf64X[i+1] - m_vf64X[i];
        }

        // Step 1: Set up the tridiagonal system
        for (i64 i = 1; i < nSamp-1; ++i)
            m_vf64Alpha[i] = (3e0 / m_vf64H[i]) * (m_vf64Y[i+1] - m_vf64Y[i]) - (3e0 / m_vf64H[i-1]) * (m_vf64Y[i] - m_vf64Y[i-1]);

        // Step 2: Solve tridiagonal system for c (second derivatives)
        for (i64 i = 1; i < nSamp-1; ++i)
        {
            m_vf64L[i] = 2e0 * (m_vf64X[i+1] - m_vf64X[i-1]) - m_vf64H[i-1] * m_vf64Mu[i-1];
            m_vf64Mu[i] = m_vf64H[i] / m_vf64L[i];
            m_vf64Z[i] = (m_vf64Alpha[i] - m_vf64H[i-1] * m_vf64Z[i-1]) / m_vf64L[i];
        }

        // Natural spline boundary conditions
        m_vf64L[nSamp-1] = 1.0;
        m_vf64Z[nSamp-1] = 0.0;
        m_vf64C[nSamp-1] = 0.0;

        // Back substitution
        for (i64 i=nSamp-2; i>=0; --i)
        {
            m_vf64C[i] = m_vf64Z[i] - m_vf64Mu[i] * m_vf64C[i+1];
            m_vf64B[i] = (m_vf64A[i+1] - m_vf64A[i]) / m_vf64H[i] - m_vf64H[i] * (m_vf64C[i+1] + 2e0 * m_vf64C[i]) / 3e0;
            m_vf64D[i] = (m_vf64C[i+1] - m_vf64C[i]) / (3e0 * m_vf64H[i]);
        }

        return true;
    }

    virtual f64 eval(f64 x, i64 ord=0) const // order: order of derivation, default is 0 (function value)
    {
        i64 idx = getIdx(x);

        f64 dx = x - m_vf64X[idx];
        if (ord == 0) return
        (
            m_vf64A[idx]
            + m_vf64B[idx] * dx
            + m_vf64C[idx] * dx * dx
            + m_vf64D[idx] * dx * dx * dx
        );
        if (ord == 1) return
        (
            m_vf64B[idx]
            + m_vf64C[idx] * 2e0 * dx
            + m_vf64D[idx] * 3e0 * dx * dx
        );
        if (ord == 2) return
        (
            m_vf64C[idx] * 2e0
            + m_vf64D[idx] * 6e0 * dx
        );
        if (ord == 3) return
        (
            m_vf64D[idx] * 6e0
        );
        return 0e0;
    }

private:
    i64 m_sizCache;
    vf64 m_vf64H, m_vf64Alpha, m_vf64L;
    vf64 m_vf64Mu, m_vf64Z;
    vf64 m_vf64A, m_vf64B, m_vf64C, m_vf64D;
};
