#ifndef VDS_H
#define VDS_H

#ifdef __cplusplus
extern "C" {
#endif

// Hargreaves VDS C API (as used via CFFI in libspiral)
void calc_vds(
    double slewmax,       // Maximum slew rate [G/cm/s]
    double gradmax,       // Maximum gradient amplitude [G/cm]
    double Tgsample,      // Gradient sample period [s]
    double Tdsample,      // Data (ADC) sample period [s]
    int    Ninterleaves,  // Number of interleaves
    const double* fov,    // FOV polynomial coefficients
    int    numfov,        // Number of FOV coeffs
    double krmax,         // Max k-space extent [/cm]
    int    ngmax,         // Max number of gradient samples
    double** xgrad,       // [out] x-component of gradient (G/cm)
    double** ygrad,       // [out] y-component of gradient (G/cm)
    int*    numgrad       // [out] number of gradient samples
);

#ifdef __cplusplus
}
#endif

#endif // VDS_H