#ifndef STANDALONE_SVD_H
#define STANDALONE_SVD_H

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

/* Return codes. */
#define SVD_OK 0
#define SVD_ERR_ARGUMENT -1
#define SVD_ERR_ALLOC -2
#define SVD_ERR_CONVERGENCE -3
#define SVD_ERR_DIMENSION -4

    /*
 * Compute the thin singular value decomposition of a real m-by-n matrix A:
 *
 *     A = U * diag(S) * V^T
 *
 * All matrices use row-major storage.
 *
 * Inputs:
 *   a   - m*n input matrix (not modified)
 *   m,n - matrix dimensions, both > 0
 *
 * Outputs, where k = min(m,n):
 *   u   - m*k left singular vectors
 *   s   - k singular values, sorted from largest to smallest
 *   v   - n*k right singular vectors (V, not V^T)
 *
 * The routine allocates only temporary workspace internally.  It depends only
 * on the ISO C library and libm.  The implementation is compatible with C89.
 */
    int svd_decompose(const float *a, size_t m, size_t n, float *u, float *s, float *v);

#ifdef __cplusplus
}
#endif

#endif /* STANDALONE_SVD_H */
