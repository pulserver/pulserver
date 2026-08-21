/*
 * Standalone dense singular-value decomposition.
 *
 * Adapted from the Golub-Reinsch SVD implementation in CControl by
 * Daniel Martensson.  The original project is distributed under the MIT
 * License.  This file keeps that attribution and license notice below.
 *
 * Modifications in this standalone version include:
 *   - no CControl headers, macros, or helper functions;
 *   - C89-compatible standard-library calls;
 *   - a thin-SVD API using row-major storage;
 *   - transparent support for both m >= n and m < n;
 *   - explicit allocation, argument, and convergence error codes.
 *
 * MIT License
 *
 * Copyright (c) 2022 Daniel Martensson
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include "external_svd.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define SVD_MAX_QR_ITERATIONS 100

/* C89 has sqrt/fabs but not sqrtf/fabsf. */
#define SVD_SQRT(x) ((float)sqrt((double)(x)))
#define SVD_ABS(x) ((float)fabs((double)(x)))

static void bidiagonalize(
    const float *a,
    int rows,
    int cols,
    float *u,
    float *v,
    float *diag,
    float *superdiag);
static int diagonalize(int rows, int cols, float *u, float *v, float *diag, float *superdiag);
static void sort_singular_values(int rows, int cols, float *s, float *u, float *v);
static int svd_tall(const float *a, int rows, int cols, float *u, float *s, float *v);

static int svd_tall(const float *a, int rows, int cols, float *u, float *s, float *v)
{
    float *superdiag;
    int ok;

    superdiag = (float *)malloc((size_t)cols * sizeof(float));
    if (superdiag == NULL)
    {
        return SVD_ERR_ALLOC;
    }

    bidiagonalize(a, rows, cols, u, v, s, superdiag);
    ok = diagonalize(rows, cols, u, v, s, superdiag);
    if (ok)
    {
        sort_singular_values(rows, cols, s, u, v);
    }

    free(superdiag);
    return ok ? SVD_OK : SVD_ERR_CONVERGENCE;
}

int svd_decompose(const float *a, size_t m, size_t n, float *u, float *s, float *v)
{
    size_t i, j;
    size_t k;
    float *at;
    float *ut;
    float *vt;
    int status;

    if (a == NULL || u == NULL || s == NULL || v == NULL || m == 0 || n == 0)
    {
        return SVD_ERR_ARGUMENT;
    }
    if (m > (size_t)INT_MAX || n > (size_t)INT_MAX)
    {
        return SVD_ERR_DIMENSION;
    }

    k = (m < n) ? m : n;

    if (m >= n)
    {
        /* Native Golub-Reinsch layout already is the requested thin SVD. */
        return svd_tall(a, (int)m, (int)n, u, s, v);
    }

    /*
     * For a wide matrix, decompose A^T = Ut S Vt^T.  Then
     * A = Vt S Ut^T, so U = Vt and V = Ut.
     */
    at = (float *)malloc(n * m * sizeof(float));
    ut = (float *)malloc(n * k * sizeof(float));
    vt = (float *)malloc(k * k * sizeof(float));
    if (at == NULL || ut == NULL || vt == NULL)
    {
        free(at);
        free(ut);
        free(vt);
        return SVD_ERR_ALLOC;
    }

    for (i = 0; i < m; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            at[j * m + i] = a[i * n + j];
        }
    }

    status = svd_tall(at, (int)n, (int)m, ut, s, vt);
    if (status == SVD_OK)
    {
        /* U_A = Vt (k x k == m x k here). */
        memcpy(u, vt, m * k * sizeof(float));

        /* V_A = Ut (n x k). */
        memcpy(v, ut, n * k * sizeof(float));
    }

    free(at);
    free(ut);
    free(vt);
    return status;
}

static void bidiagonalize(
    const float *a,
    int rows,
    int cols,
    float *u,
    float *v,
    float *diag,
    float *superdiag)
{
    int i, j, k, next;
    float s, sumsq, dot, scale, denom;
    float *row_i;
    float *p;
    float *vrow;
    float *vp;

    memcpy(u, a, (size_t)rows * (size_t)cols * sizeof(float));
    memset(v, 0, (size_t)cols * (size_t)cols * sizeof(float));

    s = 0.0f;
    scale = 0.0f;

    for (i = 0; i < cols; ++i)
    {
        row_i = u + (size_t)i * (size_t)cols;
        next = i + 1;
        superdiag[i] = scale * s;

        scale = 0.0f;
        for (j = i, p = row_i; j < rows; ++j, p += cols)
        {
            scale += SVD_ABS(p[i]);
        }

        s = 0.0f;
        if (scale != 0.0f)
        {
            sumsq = 0.0f;
            for (j = i, p = row_i; j < rows; ++j, p += cols)
            {
                p[i] /= scale;
                sumsq += p[i] * p[i];
            }

            s = (row_i[i] < 0.0f) ? SVD_SQRT(sumsq) : -SVD_SQRT(sumsq);
            denom = row_i[i] * s - sumsq;
            row_i[i] -= s;

            if (denom != 0.0f)
            {
                for (j = next; j < cols; ++j)
                {
                    dot = 0.0f;
                    for (k = i, p = row_i; k < rows; ++k, p += cols)
                    {
                        dot += p[i] * p[j];
                    }
                    dot /= denom;
                    for (k = i, p = row_i; k < rows; ++k, p += cols)
                    {
                        p[j] += dot * p[i];
                    }
                }
            }

            for (j = i, p = row_i; j < rows; ++j, p += cols)
            {
                p[i] *= scale;
            }
        }
        diag[i] = s * scale;

        s = 0.0f;
        scale = 0.0f;
        if (i >= rows || i == cols - 1)
        {
            continue;
        }

        for (j = next; j < cols; ++j)
        {
            scale += SVD_ABS(row_i[j]);
        }

        if (scale != 0.0f)
        {
            sumsq = 0.0f;
            for (j = next; j < cols; ++j)
            {
                row_i[j] /= scale;
                sumsq += row_i[j] * row_i[j];
            }

            s = (row_i[next] < 0.0f) ? SVD_SQRT(sumsq) : -SVD_SQRT(sumsq);
            denom = row_i[next] * s - sumsq;
            row_i[next] -= s;

            if (denom != 0.0f)
            {
                for (k = next; k < cols; ++k)
                {
                    superdiag[k] = row_i[k] / denom;
                }

                if (i < rows - 1)
                {
                    for (j = next, p = row_i + cols; j < rows; ++j, p += cols)
                    {
                        dot = 0.0f;
                        for (k = next; k < cols; ++k)
                        {
                            dot += row_i[k] * p[k];
                        }
                        for (k = next; k < cols; ++k)
                        {
                            p[k] += dot * superdiag[k];
                        }
                    }
                }
            }

            for (k = next; k < cols; ++k)
            {
                row_i[k] *= scale;
            }
        }
    }

    /* Accumulate the right-hand Householder transformations in V. */
    v[(size_t)(cols - 1) * (size_t)cols + (cols - 1)] = 1.0f;
    if (cols > 1)
    {
        row_i = u + (size_t)(cols - 2) * (size_t)cols;
        vrow = v + (size_t)(cols - 2) * (size_t)cols;
        s = superdiag[cols - 1];

        for (i = cols - 2; i >= 0; --i)
        {
            next = i + 1;

            if (SVD_ABS(s) > FLT_MIN)
            {
                vp = vrow + cols;
                for (j = next; j < cols; ++j, vp += cols)
                {
                    vp[i] = (row_i[j] / row_i[next]) / s;
                }

                for (j = next; j < cols; ++j)
                {
                    dot = 0.0f;
                    for (k = next, vp = vrow + cols; k < cols; ++k, vp += cols)
                    {
                        dot += row_i[k] * vp[j];
                    }
                    for (k = next, vp = vrow + cols; k < cols; ++k, vp += cols)
                    {
                        vp[j] += dot * vp[i];
                    }
                }
            }

            vp = vrow + cols;
            for (j = next; j < cols; ++j, vp += cols)
            {
                vrow[j] = 0.0f;
                vp[i] = 0.0f;
            }
            vrow[i] = 1.0f;
            s = superdiag[i];

            if (i > 0)
            {
                row_i -= cols;
                vrow -= cols;
            }
        }
    }

    /* Accumulate the left-hand Householder transformations in U. */
    row_i = u + (size_t)(cols - 1) * (size_t)cols;
    for (i = cols - 1; i >= 0; --i)
    {
        next = i + 1;
        s = diag[i];

        for (j = next; j < cols; ++j)
        {
            row_i[j] = 0.0f;
        }

        if (SVD_ABS(s) > FLT_MIN)
        {
            for (j = next; j < cols; ++j)
            {
                dot = 0.0f;
                p = row_i + cols;
                for (k = next; k < rows; ++k, p += cols)
                {
                    dot += p[i] * p[j];
                }

                dot = (dot / row_i[i]) / s;
                for (k = i, p = row_i; k < rows; ++k, p += cols)
                {
                    p[j] += dot * p[i];
                }
            }

            for (j = i, p = row_i; j < rows; ++j, p += cols)
            {
                p[i] /= s;
            }
        }
        else
        {
            for (j = i, p = row_i; j < rows; ++j, p += cols)
            {
                p[i] = 0.0f;
            }
        }

        row_i[i] += 1.0f;
        if (i > 0)
        {
            row_i -= cols;
        }
    }
}

static int diagonalize(int rows, int cols, float *u, float *v, float *diag, float *superdiag)
{
    int i, j, k, split;
    int need_cancel;
    int iterations;
    float eps, c, sn;
    float f, g, h;
    float x, y, z;
    float max_norm;
    float *pu;
    float *pv;

    max_norm = 0.0f;
    for (i = 0; i < cols; ++i)
    {
        y = SVD_ABS(diag[i]) + SVD_ABS(superdiag[i]);
        if (y > max_norm)
        {
            max_norm = y;
        }
    }
    eps = max_norm * FLT_EPSILON;
    if (eps < FLT_MIN)
    {
        eps = FLT_MIN;
    }

    for (k = cols - 1; k >= 0; --k)
    {
        iterations = 0;

        for (;;)
        {
            need_cancel = 1;
            split = k;

            while (split >= 0)
            {
                if (SVD_ABS(superdiag[split]) <= eps)
                {
                    need_cancel = 0;
                    break;
                }
                if (split == 0 || SVD_ABS(diag[split - 1]) <= eps)
                {
                    break;
                }
                --split;
            }

            if (need_cancel)
            {
                c = 0.0f;
                sn = 1.0f;
                for (i = split; i <= k; ++i)
                {
                    f = sn * superdiag[i];
                    superdiag[i] *= c;
                    if (SVD_ABS(f) <= eps)
                    {
                        break;
                    }

                    g = diag[i];
                    h = SVD_SQRT(f * f + g * g);
                    if (h == 0.0f)
                    {
                        continue;
                    }
                    diag[i] = h;
                    c = g / h;
                    sn = -f / h;

                    if (split > 0)
                    {
                        for (j = 0, pu = u; j < rows; ++j, pu += cols)
                        {
                            y = pu[split - 1];
                            z = pu[i];
                            pu[split - 1] = y * c + z * sn;
                            pu[i] = -y * sn + z * c;
                        }
                    }
                }
            }

            z = diag[k];
            if (split == k)
            {
                if (z < 0.0f)
                {
                    diag[k] = -z;
                    for (j = 0, pv = v; j < cols; ++j, pv += cols)
                    {
                        pv[k] = -pv[k];
                    }
                }
                break;
            }

            if (iterations++ >= SVD_MAX_QR_ITERATIONS)
            {
                return 0;
            }

            x = diag[split];
            y = diag[k - 1];
            g = superdiag[k - 1];
            h = superdiag[k];

            if (h == 0.0f || y == 0.0f || x == 0.0f)
            {
                /* The split test should normally prevent this.  A tiny
                   denominator keeps the implicit-shift QR step finite. */
                if (h == 0.0f)
                    h = eps;
                if (y == 0.0f)
                    y = eps;
                if (x == 0.0f)
                    x = eps;
            }

            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0f * h * y);
            g = SVD_SQRT(f * f + 1.0f);
            if (f < 0.0f)
            {
                g = -g;
            }
            f = ((x - z) * (x + z) + h * (y / (f + g) - h)) / x;

            c = 1.0f;
            sn = 1.0f;
            for (i = split + 1; i <= k; ++i)
            {
                g = superdiag[i];
                y = diag[i];
                h = sn * g;
                g *= c;

                z = SVD_SQRT(f * f + h * h);
                superdiag[i - 1] = z;
                if (z != 0.0f)
                {
                    c = f / z;
                    sn = h / z;
                }
                else
                {
                    c = 1.0f;
                    sn = 0.0f;
                }

                f = x * c + g * sn;
                g = -x * sn + g * c;
                h = y * sn;
                y *= c;

                for (j = 0, pv = v; j < cols; ++j, pv += cols)
                {
                    x = pv[i - 1];
                    z = pv[i];
                    pv[i - 1] = x * c + z * sn;
                    pv[i] = -x * sn + z * c;
                }

                z = SVD_SQRT(f * f + h * h);
                diag[i - 1] = z;
                if (z != 0.0f)
                {
                    c = f / z;
                    sn = h / z;
                }
                else
                {
                    c = 1.0f;
                    sn = 0.0f;
                }

                f = c * g + sn * y;
                x = -sn * g + c * y;

                for (j = 0, pu = u; j < rows; ++j, pu += cols)
                {
                    y = pu[i - 1];
                    z = pu[i];
                    pu[i - 1] = c * y + sn * z;
                    pu[i] = -sn * y + c * z;
                }
            }

            superdiag[split] = 0.0f;
            superdiag[k] = f;
            diag[k] = x;
        }
    }

    return 1;
}

static void sort_singular_values(int rows, int cols, float *s, float *u, float *v)
{
    int i, j, r;
    int largest;
    float tmp;

    for (i = 0; i < cols - 1; ++i)
    {
        largest = i;
        for (j = i + 1; j < cols; ++j)
        {
            if (s[j] > s[largest])
            {
                largest = j;
            }
        }

        if (largest == i)
        {
            continue;
        }

        tmp = s[i];
        s[i] = s[largest];
        s[largest] = tmp;

        for (r = 0; r < rows; ++r)
        {
            tmp = u[(size_t)r * (size_t)cols + i];
            u[(size_t)r * (size_t)cols + i] = u[(size_t)r * (size_t)cols + largest];
            u[(size_t)r * (size_t)cols + largest] = tmp;
        }

        for (r = 0; r < cols; ++r)
        {
            tmp = v[(size_t)r * (size_t)cols + i];
            v[(size_t)r * (size_t)cols + i] = v[(size_t)r * (size_t)cols + largest];
            v[(size_t)r * (size_t)cols + largest] = tmp;
        }
    }
}
