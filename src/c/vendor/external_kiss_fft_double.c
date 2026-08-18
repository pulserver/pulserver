/*
 * external_kiss_fft_double.c -- a second instantiation of the vendored
 * kissfft, in double precision.
 *
 * kissfft picks its arithmetic with `kiss_fft_scalar` at compile time, and
 * the copy beside this one is compiled as float because that is what the
 * display-spectrum path wants. The safety evaluator needs the same transform
 * in double: its sums land in cancellation, which is why every accumulator
 * over there is a double already.
 *
 * So the library is compiled twice. The renames below give the second copy
 * its own symbols, and the guard #undefs let it re-read the same headers --
 * which matters because the EPIC build is an amalgamation, where both copies
 * land in ONE translation unit and the guards would otherwise suppress the
 * second set of declarations.
 *
 * Nothing here is a fork: the sources are included verbatim, so a kissfft
 * update needs no change on this side.
 */

#undef EXTERNAL_KISS_FFT_H
#undef external_kiss_fft_guts_h

/* The float copy leaves this defined when both land in one translation
 * unit, and redefining it would be a warning -- which the PSD build treats
 * as an error. */
#undef kiss_fft_scalar
#define kiss_fft_scalar double

#define kiss_fft_cpx kiss_fft_cpx_dbl
#define kiss_fft_cfg kiss_fft_cfg_dbl
#define kiss_fft_state kiss_fft_state_dbl
#define kiss_fft_alloc kiss_fft_alloc_dbl
#define kiss_fft_stride kiss_fft_stride_dbl
#define kiss_fft kiss_fft_dbl
#define kiss_fft_cleanup kiss_fft_cleanup_dbl
#define kiss_fft_next_fast_size kiss_fft_next_fast_size_dbl

/* kissfft's internals too. They are static, so a separate-TU build would not
 * need this -- but the EPIC amalgamation puts both copies in one TU, where a
 * second definition of the same name is an error however local it is. */
#define kf_bfly2 kf_bfly2_dbl
#define kf_bfly3 kf_bfly3_dbl
#define kf_bfly4 kf_bfly4_dbl
#define kf_bfly5 kf_bfly5_dbl
#define kf_bfly_generic kf_bfly_generic_dbl
/* kf_cexp is deliberately absent: it is a macro in the guts header, whose
 * own definition is re-read below and is identical either way. */
#define kf_factor kf_factor_dbl
#define kf_work kf_work_dbl

#include "external_kiss_fft.c"

/* Entry points that name no kissfft type, so callers need neither this
 * instantiation's header nor its typedefs -- which in the amalgamated build
 * would collide with the float copy's. A complex buffer is passed as
 * interleaved (re, im) doubles, which is exactly kiss_fft_cpx's layout. */

int pulseg__fft_double_size(int at_least)
{
    return kiss_fft_next_fast_size_dbl(at_least);
}

void *pulseg__fft_double_alloc(int nfft, int inverse)
{
    return (void *)kiss_fft_alloc_dbl(nfft, inverse, NULL, NULL);
}

void pulseg__fft_double_run(void *cfg, const double *in_interleaved, double *out_interleaved)
{
    kiss_fft_dbl(
        (kiss_fft_cfg_dbl)cfg,
        (const kiss_fft_cpx_dbl *)in_interleaved,
        (kiss_fft_cpx_dbl *)out_interleaved);
}

void pulseg__fft_double_free(void *cfg)
{
    if (cfg)
        KISS_FFT_FREE(cfg);
}
