#pragma once

#include <stdint.h>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

class Intp
{
public:
    enum SearchMode
    {
        EBinary = 0,
        ECached,
        EUniform
    } m_eSearchMode;

    Intp() : m_eSearchMode(EBinary), m_idxCache(0) {}
    virtual ~Intp() {}

    virtual bool fit(const vf64& vf64X, const vf64& vf64Y) = 0;

    virtual f64 eval(f64 x, i64 ord = 0) const = 0;

    static bool validate(vf64& vf64X, vf64& vf64Y)
    {
        if (vf64X.size()!=vf64Y.size() || vf64X.size()<2) return false;
        
        if (vf64X.back() < vf64X.front())
        {
            std::reverse(vf64X.begin(), vf64X.end());
            std::reverse(vf64Y.begin(), vf64Y.end());
        }

        i64 nSamp = i64(vf64X.size());
        for (i64 i = 1; i < nSamp; ++i)
        {
            if (vf64X[i] <= vf64X[i-1]) return false;
        }
        return true;
    }

protected:
    vf64 m_vf64X, m_vf64Y;
    mutable i64 m_idxCache;

    i64 getIdx(const f64& x) const
    {
        const i64 nSamp = i64(m_vf64X.size());
        ASSERT(nSamp >= 2);

        i64 idx;

        if (m_eSearchMode == EBinary)
        {
            i64 low = 0;
            i64 high = nSamp - 1;
            while (high - low > 1)
            {
                i64 mid = (low + high) / 2;
                if (m_vf64X[mid] > x) high = mid;
                else low = mid;
            }
            idx = low;
            return idx;
        }
        if (m_eSearchMode == ECached)
        {
            if (m_idxCache < 0) m_idxCache = 0;
            if (m_idxCache > nSamp - 2) m_idxCache = nSamp - 2;
            idx = m_idxCache;
            while (idx > 0 && m_vf64X[idx] > x) --idx;
            while (idx + 1 < nSamp - 1 && m_vf64X[idx + 1] < x) ++idx;
            m_idxCache = idx;
            return idx;
        }
        if (m_eSearchMode == EUniform)
        {
            const f64 x0 = m_vf64X.front();
            const f64 x1 = m_vf64X.back();
            idx = i64((x - x0) / (x1 - x0) * (nSamp - 1));
            if (idx < 0) idx = 0;
            if (idx > nSamp - 2) idx = nSamp - 2;
            return idx;
        }

        throw std::invalid_argument("m_eSearchMode");
    }
};