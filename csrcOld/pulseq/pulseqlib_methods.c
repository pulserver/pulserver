#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pulseqlib_methods.h"

#include "kiss_fft.h"
#include "kiss_fftr.h"

#define INIT_LIBRARY(seq, fieldPtr, sizeField, flagField) \
    do { \
        (seq)->fieldPtr = NULL; \
        (seq)->sizeField = 0; \
        (seq)->flagField = 0; \
    } while (0)

static void seqFileSetDefaults(pulseqlib_SeqFile* seq);
static void seqFileInit(pulseqlib_SeqFile* seq, const pulseqlib_Opts* opts);
static void seqFileReset(pulseqlib_SeqFile* seq);

typedef struct {
    const char *name;
    int value;
} TableEntry;


static const TableEntry label_table[] = {
    { "SLC", SLC }, 
    { "SEG", SEG }, 
    { "REP", REP }, 
    { "AVG", AVG },
    { "SET", SET }, 
    { "ECO", ECO }, 
    { "PHS", PHS }, 
    { "LIN", LIN },
    { "PAR", PAR }, 
    { "ACQ", ACQ }, 
    { "NAV", NAV },
    { "REV", REV }, 
    { "SMS", SMS }, 
    { "REF", REF }, 
    { "IMA", IMA },
    { "NOISE", NOISE }, 
    { "PMC", PMC }, 
    { "NOROT", NOROT },
    { "NOPOS", NOPOS }, 
    { "NOSCL", NOSCL }, 
    { "ONCE", ONCE },
    { "TRID", TRID },
    { NULL, -1 }
};


int label2enum(const char *label) {
    int i;
    if (!label) return -1;
    for (i = 0; label_table[i].name != NULL; i++) {
        if (strcmp(label, label_table[i].name) == 0) return label_table[i].value;
    }
    return -1;
}


static const TableEntry hint_table[] = {
    { "TE", HINT_TE }, 
    { "TR", HINT_TR },
    { "TI", HINT_TI }, 
    { "ESP", HINT_ESP },
    { "RECTIME", HINT_RECTIME },
    { "T2PREP", HINT_T2PREP }, 
    { "TE2", HINT_TE2 },
    { NULL, -1 }
};


int hint2enum(const char *hint) {
    int i;
    if (!hint) return -1;
    for (i = 0; hint_table[i].name != NULL; i++) {
        if (strcmp(hint, hint_table[i].name) == 0) return hint_table[i].value;
    }
    return -1;
}


void pulseqlib_diagnosticInit(pulseqlib_Diagnostic* diag) {
    if (!diag) return;
    diag->code = PULSEQLIB_OK;
    diag->blockIndex = -1;
    diag->channel = -1;
    diag->numUniqueBlocks = 0;
    diag->imagingRegionLength = 0;
    diag->candidatePatternLength = 0;
    diag->mismatchPosition = -1;
    diag->gradientAmplitude = 0.0f;
    diag->maxAllowedAmplitude = 0.0f;
}

const char* pulseqlib_getErrorMessage(int code) {
    switch (code) {
        case PULSEQLIB_OK:
            return "Success";
        
        /* Generic errors */
        case PULSEQLIB_ERR_NULL_POINTER:
            return "Required pointer argument is NULL";
        case PULSEQLIB_ERR_INVALID_ARGUMENT:
            return "Invalid argument value";
        case PULSEQLIB_ERR_ALLOC_FAILED:
            return "Memory allocation failed";
    
        /* Parsing/file errors */
        case PULSEQLIB_ERR_FILE_NOT_FOUND:
            return "Sequence file not found or could not be opened";
        case PULSEQLIB_ERR_FILE_READ_FAILED:
            return "Error reading from sequence file";
        case PULSEQLIB_ERR_UNSUPPORTED_VERSION:
            return "Unsupported sequence file version (requires >= 1.5.0)";
        case PULSEQLIB_ERR_PARSE_FAILED:
            return "Failed to parse sequence data";

        /* Unique block errors */
        case PULSEQLIB_ERR_INVALID_PREP_POSITION:
            return "Invalid preparation block position";
        case PULSEQLIB_ERR_INVALID_COOLDOWN_POSITION:
            return "Invalid cooldown block position";
        case PULSEQLIB_ERR_INVALID_ONCE_FLAGS:
            return "ONCE flags were found outside preparation/cooldown sections";

        /* Too many gradient shots error */
        case PULSEQLIB_ERR_TOO_MANY_GRAD_SHOTS:
            return "Number of waveform shots exceeds maximum allowed";

        /* Incompatible selective excitation errors */
        case PULSEQLIB_ERR_SELEXC_GRAD_SCALING:
            return "Selective excitation block has varying gradient amplitude across instances";
        case PULSEQLIB_ERR_SELEXC_ROTATION:
            return "Selective excitation block has rotation extension";
        
        /* TR detection errors */
        case PULSEQLIB_ERR_TR_NO_BLOCKS:
            return "Sequence contains no blocks";
        case PULSEQLIB_ERR_TR_NO_IMAGING_REGION:
            return "No imaging region found (preparation + cooldown >= total blocks)";
        case PULSEQLIB_ERR_TR_NO_PERIODIC_PATTERN:
            return "No periodic TR pattern found in imaging region";
        case PULSEQLIB_ERR_TR_PATTERN_MISMATCH:
            return "TR pattern does not repeat consistently across imaging region";
        case PULSEQLIB_ERR_TR_PREP_TOO_LONG:
            return "Non-standard preparation section exceeds duration threshold";
        case PULSEQLIB_ERR_TR_COOLDOWN_TOO_LONG:
            return "Non-standard cooldown section exceeds duration threshold";
        
        /* Segmentation errors */
        case PULSEQLIB_ERR_SEG_NONZERO_START_GRAD:
            return "TR does not start with zero gradient amplitude";
        case PULSEQLIB_ERR_SEG_NONZERO_END_GRAD:
            return "TR does not end with zero gradient amplitude";
        case PULSEQLIB_ERR_SEG_NO_SEGMENTS_FOUND:
            return "No segment boundaries could be identified in TR";

        /* Acoustic errors */
        case PULSEQLIB_ERR_ACOUSTIC_INVALID_WINDOW:
            return "Invalid window size for acoustic analysis";
        case PULSEQLIB_ERR_ACOUSTIC_INVALID_RESOLUTION:
            return "Invalid spectral resolution for acoustic analysis";
        case PULSEQLIB_ERR_ACOUSTIC_NO_WAVEFORM:
            return "No waveform data for acoustic analysis";
        case PULSEQLIB_ERR_ACOUSTIC_FFT_FAILED:
            return "FFT computation failed during acoustic analysis";
        case PULSEQLIB_ERR_ACOUSTIC_VIOLATION:
            return "Acoustic resonance violation detected";
            
        /* PNS errors */
        case PULSEQLIB_ERR_PNS_INVALID_PARAMS:
            return "Invalid PNS parameters";
        case PULSEQLIB_ERR_PNS_INVALID_CHRONAXIE:
            return "Invalid chronaxie value for PNS";
        case PULSEQLIB_ERR_PNS_INVALID_RHEOBASE:
            return "Invalid rheobase value for PNS (GE model)";
        case PULSEQLIB_ERR_PNS_NO_WAVEFORM:
            return "No waveform data for PNS analysis";
        case PULSEQLIB_ERR_PNS_FFT_FAILED:
            return "FFT convolution failed during PNS analysis";
        case PULSEQLIB_ERR_PNS_THRESHOLD_EXCEEDED:
            return "PNS threshold exceeded (>100%)";

         case PULSEQLIB_ERR_NOT_IMPLEMENTED:
            return "Functionality not yet implemented";       
        
        default:
            return "Unknown error";
    }
}

const char* pulseqlib_getErrorHint(int code) {
    switch (code) {
        case PULSEQLIB_OK:
            return "";

        case PULSEQLIB_ERR_INVALID_PREP_POSITION:
            return "Ensure that the preparation section is marked with ONCE labels "
                   "and starts at the first block of the sequence.";

        case PULSEQLIB_ERR_INVALID_COOLDOWN_POSITION:
            return "Ensure that the cooldown section is marked with ONCE labels "
                   "and ends at the last block of the sequence.";

        case PULSEQLIB_ERR_INVALID_ONCE_FLAGS:
            return "Ensure that ONCE flags are only used in preparation and cooldown sections.";

        case PULSEQLIB_ERR_TR_NO_IMAGING_REGION:
            return "Make sure to use ONCE flags either at beginning (preparation) or end (cooldown) of the sequence.";
        
        case PULSEQLIB_ERR_TR_NO_PERIODIC_PATTERN:
            return "This often occurs when phase-encoding gradients are created inside "
                   "the sequence loop with varying amplitudes. Instead, create gradient "
                   "events ONCE outside the loop and use 'scale' parameter to vary amplitude. "
                   "Example: use seq.add_block(gx=make_trapezoid(..., scale=pe_scale[n])) "
                   "rather than seq.add_block(gx=make_trapezoid(..., amplitude=pe_amp[n]))";
                
        case PULSEQLIB_ERR_SEG_NONZERO_START_GRAD:
        case PULSEQLIB_ERR_SEG_NONZERO_END_GRAD:
            return "Each segment must begin and end with gradient amplitudes that can "
                   "ramp to/from zero within one gradient raster. Check that spoiler "
                   "gradients or flow-compensation lobes are properly balanced.";
        case PULSEQLIB_ERR_TOO_MANY_GRAD_SHOTS:
            return "The sequence contains a waveform with more than 16 distinct waveform shapes "
                   "(interleaves/shots). Consider representing the waveform using scaling or rotation.";
        case PULSEQLIB_ERR_SELEXC_GRAD_SCALING:
            return "Blocks containing both RF and gradients (spatially selective excitation) require "
                   "constant gradient amplitude across instances for off-isocenter frequency modulation. Use explicit "
                   "multi-shot gradient definitions instead of amplitude scaling.";
        case PULSEQLIB_ERR_SELEXC_ROTATION:
            return "Blocks containing both RF and gradients (spatially selective excitation) cannot "
                   "have rotation extensions because frequency modulation is computed at prep time. "
                   "Use explicit multi-shot gradient definitions with pre-rotated waveforms instead.";
        
        case PULSEQLIB_ERR_TR_PREP_TOO_LONG:
        case PULSEQLIB_ERR_TR_COOLDOWN_TOO_LONG:
            return "The preparation or cooldown section differs from the main TR pattern and is too long to be safe."
                   "If this is intentional (e.g., steady-state preparation), ensure it is marked with appropriate ONCE labels.";

        case PULSEQLIB_ERR_NOT_IMPLEMENTED:
            return "This functionality is not yet implemented.";
        
        default:
            return "Check sequence design for structural consistency.";
    }
}

/**
 * @brief Extract base directory path from a file path.
 */
static char* extractBasePath(const char* filePath) {
    char* basePath;
    const char* lastSlash;
    size_t baseLen;
    
    /* Find last path separator */
    lastSlash = strrchr(filePath, '/');
    if (!lastSlash) {
        lastSlash = strrchr(filePath, '\\');
    }
    
    if (lastSlash) {
        baseLen = (size_t)(lastSlash - filePath + 1);
        basePath = (char*)ALLOC(baseLen + 1);
        if (basePath) {
            strncpy(basePath, filePath, baseLen);
            basePath[baseLen] = '\0';
        }
    } else {
        /* No path separator - use current directory */
        basePath = (char*)ALLOC(3);
        if (basePath) {
            strcpy(basePath, "./");
        }
    }
    
    return basePath;
}

/**
 * @brief Build full path from base path and filename.
 */
static char* buildFullPath(const char* basePath, const char* filename) {
    size_t fullLen;
    char* fullPath;
    
    fullLen = strlen(basePath) + strlen(filename) + 1;
    fullPath = (char*)ALLOC(fullLen);
    if (fullPath) {
        strcpy(fullPath, basePath);
        strcat(fullPath, filename);
    }
    
    return fullPath;
}

/******************************************* Math Helpers *************************************************/

/**
 * @brief Find maximum absolute value in a real-valued array.
 *
 * @param[in]  samples    Array of samples.
 * @param[in]  numSamples Number of samples.
 * @return Maximum absolute value, or 0 if array is empty/NULL.
 */
static float find_max_abs_real(const float* samples, int numSamples)
{
    int i;
    float maxAbs = 0.0f;
    float absVal;
    
    if (!samples || numSamples <= 0) {
        return 0.0f;
    }
    
    for (i = 0; i < numSamples; ++i) {
        absVal = (float)fabs(samples[i]);
        if (absVal > maxAbs) {
            maxAbs = absVal;
        }
    }
    
    return maxAbs;
}

/**
 * @brief Convert magnitude/phase arrays to real/imaginary arrays.
 *
 * @param[in]  magnitude  Magnitude array.
 * @param[in]  phase      Phase array (radians).
 * @param[in]  numSamples Number of samples.
 * @param[out] real       Output real part array (must be pre-allocated).
 * @param[out] imag       Output imaginary part array (must be pre-allocated).
 */
static void mag_phase_to_real_imag(
    const float* magnitude,
    const float* phase,
    int numSamples,
    float* real,
    float* imag)
{
    int i;
    
    if (!magnitude || !phase || !real || !imag || numSamples <= 0) {
        return;
    }
    
    for (i = 0; i < numSamples; ++i) {
        real[i] = magnitude[i] * (float)cos(phase[i]);
        imag[i] = magnitude[i] * (float)sin(phase[i]);
    }
}

/**
 * @brief Trapezoidal integration for real-valued function with uniform raster.
 *
 * Computes integral of f(t) dt using trapezoidal rule.
 *
 * @param[in]  samples    Array of function values.
 * @param[in]  numSamples Number of samples.
 * @param[in]  dt         Time step (seconds).
 * @return Integral value.
 */
static float trapz_real_uniform(const float* samples, int numSamples, float dt)
{
    int i;
    float sum = 0.0f;
    
    if (!samples || numSamples < 2 || dt <= 0.0f) {
        return 0.0f;
    }
    
    /* Trapezoidal rule: sum of (f[i-1] + f[i]) / 2 * dt */
    for (i = 1; i < numSamples; ++i) {
        sum += 0.5f * (samples[i - 1] + samples[i]) * dt;
    }
    
    return sum;
}

/**
 * @brief Trapezoidal integration for real-valued function with non-uniform raster.
 *
 * Computes integral of f(t) dt using trapezoidal rule.
 *
 * @param[in]  samples    Array of function values.
 * @param[in]  time       Array of time points (seconds).
 * @param[in]  numSamples Number of samples.
 * @return Integral value.
 */
static float trapz_real_nonuniform(const float* samples, const float* time, int numSamples)
{
    int i;
    float sum = 0.0f;
    float dt;
    
    if (!samples || !time || numSamples < 2) {
        return 0.0f;
    }
    
    for (i = 1; i < numSamples; ++i) {
        dt = time[i] - time[i - 1];
        if (dt > 0.0f) {
            sum += 0.5f * (samples[i - 1] + samples[i]) * dt;
        }
    }
    
    return sum;
}

/**
 * @brief Trapezoidal integration of |f(t)| for complex-valued function with uniform raster.
 *
 * Computes integral of |f(t)| dt (magnitude, not magnitude squared).
 *
 * @param[in]  samples_re Real part of function values.
 * @param[in]  samples_im Imaginary part of function values.
 * @param[in]  numSamples Number of samples.
 * @param[in]  dt         Time step (seconds).
 * @return Integral value.
 */
static float trapz_complex_mag_uniform(
    const float* samples_re,
    const float* samples_im,
    int numSamples,
    float dt)
{
    int i;
    float sum = 0.0f;
    float mag_prev, mag_curr;
    
    if (!samples_re || !samples_im || numSamples < 2 || dt <= 0.0f) {
        return 0.0f;
    }
    
    mag_prev = (float)sqrt(samples_re[0] * samples_re[0] + samples_im[0] * samples_im[0]);
    
    for (i = 1; i < numSamples; ++i) {
        mag_curr = (float)sqrt(samples_re[i] * samples_re[i] + samples_im[i] * samples_im[i]);
        sum += 0.5f * (mag_prev + mag_curr) * dt;
        mag_prev = mag_curr;
    }
    
    return sum;
}

/**
 * @brief Trapezoidal integration of |f(t)| for complex-valued function with non-uniform raster.
 *
 * Computes integral of |f(t)| dt (magnitude, not magnitude squared).
 *
 * @param[in]  samples_re Real part of function values.
 * @param[in]  samples_im Imaginary part of function values.
 * @param[in]  time       Array of time points (seconds).
 * @param[in]  numSamples Number of samples.
 * @return Integral value.
 */
static float trapz_complex_mag_nonuniform(
    const float* samples_re,
    const float* samples_im,
    const float* time,
    int numSamples)
{
    int i;
    float sum = 0.0f;
    float dt;
    float mag_prev, mag_curr;
    
    if (!samples_re || !samples_im || !time || numSamples < 2) {
        return 0.0f;
    }
    
    mag_prev = (float)sqrt(samples_re[0] * samples_re[0] + samples_im[0] * samples_im[0]);
    
    for (i = 1; i < numSamples; ++i) {
        dt = time[i] - time[i - 1];
        if (dt > 0.0f) {
            mag_curr = (float)sqrt(samples_re[i] * samples_re[i] + samples_im[i] * samples_im[i]);
            sum += 0.5f * (mag_prev + mag_curr) * dt;
            mag_prev = mag_curr;
        }
    }
    
    return sum;
}

/**
 * @brief Compute maximum absolute slew rate for real-valued waveform with uniform raster.
 *
 * @param[in]  samples    Array of function values.
 * @param[in]  numSamples Number of samples.
 * @param[in]  dt         Time step (seconds).
 * @return Maximum absolute slew rate (1/s).
 */
static float max_slew_real_uniform(const float* samples, int numSamples, float dt)
{
    int i;
    float maxSlew = 0.0f;
    float slew;
    
    if (!samples || numSamples < 2 || dt <= 0.0f) {
        return 0.0f;
    }
    
    for (i = 1; i < numSamples; ++i) {
        slew = (float)fabs((samples[i] - samples[i - 1]) / dt);
        if (slew > maxSlew) {
            maxSlew = slew;
        }
    }
    
    return maxSlew;
}

/**
 * @brief Compute maximum absolute slew rate for real-valued waveform with non-uniform raster.
 *
 * @param[in]  samples    Array of function values.
 * @param[in]  time       Array of time points (seconds).
 * @param[in]  numSamples Number of samples.
 * @return Maximum absolute slew rate (1/s).
 */
static float max_slew_real_nonuniform(const float* samples, const float* time, int numSamples)
{
    int i;
    float maxSlew = 0.0f;
    float dt, slew;
    
    if (!samples || !time || numSamples < 2) {
        return 0.0f;
    }
    
    for (i = 1; i < numSamples; ++i) {
        dt = time[i] - time[i - 1];
        if (dt > 0.0f) {
            slew = (float)fabs((samples[i] - samples[i - 1]) / dt);
            if (slew > maxSlew) {
                maxSlew = slew;
            }
        }
    }
    
    return maxSlew;
}

/**
 * @brief Find index of maximum absolute value in a real-valued array.
 *
 * @param[in]  samples    Array of samples.
 * @param[in]  numSamples Number of samples.
 * @return Index of maximum absolute value, or 0 if array is empty/NULL.
 */
static int find_max_abs_index_real(const float* samples, int numSamples)
{
    int i;
    int maxIdx = 0;
    float maxAbs = 0.0f;
    float absVal;
    
    if (!samples || numSamples <= 0) {
        return 0;
    }
    
    for (i = 0; i < numSamples; ++i) {
        absVal = (float)fabs(samples[i]);
        if (absVal > maxAbs) {
            maxAbs = absVal;
            maxIdx = i;
        }
    }
    
    return maxIdx;
}

/**
 * @brief Convert a quaternion to a 3x3 rotation matrix.
 *
 * Quaternion format: [w, x, y, z] where w is the scalar part.
 * The quaternion is normalized before conversion to ensure a unitary rotation matrix.
 * Output matrix is in row-major order.
 *
 * @param[in]  quat   Quaternion as [w, x, y, z].
 * @param[out] matrix Output 3x3 rotation matrix (9 floats, row-major).
 */
static void quaternion_to_matrix(const float* quat, float* matrix)
{
    float w = quat[0];
    float x = quat[1];
    float y = quat[2];
    float z = quat[3];
    
    float norm, invNorm;
    float xx, yy, zz, xy, xz, yz, wx, wy, wz;
    
    /* Normalize quaternion */
    norm = (float)sqrt(w*w + x*x + y*y + z*z);
    if (norm > 1e-9f) {
        invNorm = 1.0f / norm;
        w *= invNorm;
        x *= invNorm;
        y *= invNorm;
        z *= invNorm;
    } else {
        /* Degenerate quaternion - return identity matrix */
        matrix[0] = 1.0f; matrix[1] = 0.0f; matrix[2] = 0.0f;
        matrix[3] = 0.0f; matrix[4] = 1.0f; matrix[5] = 0.0f;
        matrix[6] = 0.0f; matrix[7] = 0.0f; matrix[8] = 1.0f;
        return;
    }
    
    xx = x * x;
    yy = y * y;
    zz = z * z;
    xy = x * y;
    xz = x * z;
    yz = y * z;
    wx = w * x;
    wy = w * y;
    wz = w * z;
    
    /* Row 0 */
    matrix[0] = 1.0f - 2.0f * (yy + zz);
    matrix[1] = 2.0f * (xy - wz);
    matrix[2] = 2.0f * (xz + wy);
    
    /* Row 1 */
    matrix[3] = 2.0f * (xy + wz);
    matrix[4] = 1.0f - 2.0f * (xx + zz);
    matrix[5] = 2.0f * (yz - wx);
    
    /* Row 2 */
    matrix[6] = 2.0f * (xz - wy);
    matrix[7] = 2.0f * (yz + wx);
    matrix[8] = 1.0f - 2.0f * (xx + yy);
}

/******************************************* Interpolation Helpers *************************************************/

/**
 * @brief 1D linear interpolation.
 *
 * Interpolates values from (xp, fp) onto new x-coordinates.
 * Assumes xp is sorted in ascending order.
 * Values outside the range are clamped to boundary values.
 *
 * @param[in]  x          Target x-coordinates (sorted ascending).
 * @param[in]  nx         Number of target points.
 * @param[in]  xp         Source x-coordinates (sorted ascending).
 * @param[in]  fp         Source function values.
 * @param[in]  nxp        Number of source points.
 * @param[out] out        Output interpolated values (must be pre-allocated, size nx).
 */
static void interp1_linear(
    const float* x,
    int nx,
    const float* xp,
    const float* fp,
    int nxp,
    float* out)
{
    int i, j;
    float t;
    
    if (!x || !xp || !fp || !out || nx <= 0 || nxp <= 0) {
        return;
    }
    
    /* Handle single-point source */
    if (nxp == 1) {
        for (i = 0; i < nx; ++i) {
            out[i] = fp[0];
        }
        return;
    }
    
    j = 0; /* Current interval index in xp */
    
    for (i = 0; i < nx; ++i) {
        /* Clamp below range */
        if (x[i] <= xp[0]) {
            out[i] = fp[0];
            continue;
        }
        
        /* Clamp above range */
        if (x[i] >= xp[nxp - 1]) {
            out[i] = fp[nxp - 1];
            continue;
        }
        
        /* Find interval: xp[j] <= x[i] < xp[j+1] */
        while (j < nxp - 2 && xp[j + 1] < x[i]) {
            ++j;
        }
        
        /* Linear interpolation */
        t = (x[i] - xp[j]) / (xp[j + 1] - xp[j]);
        out[i] = fp[j] + t * (fp[j + 1] - fp[j]);
    }
}

/**
 * @brief 1D linear interpolation for complex values.
 *
 * Interpolates complex values from (xp, fp_re, fp_im) onto new x-coordinates.
 * Assumes xp is sorted in ascending order.
 *
 * @param[in]  x          Target x-coordinates (sorted ascending).
 * @param[in]  nx         Number of target points.
 * @param[in]  xp         Source x-coordinates (sorted ascending).
 * @param[in]  fp_re      Source function values (real part).
 * @param[in]  fp_im      Source function values (imaginary part).
 * @param[in]  nxp        Number of source points.
 * @param[out] out_re     Output interpolated real part (must be pre-allocated, size nx).
 * @param[out] out_im     Output interpolated imaginary part (must be pre-allocated, size nx).
 */
static void interp1_linear_complex(
    const float* x,
    int nx,
    const float* xp,
    const float* fp_re,
    const float* fp_im,
    int nxp,
    float* out_re,
    float* out_im)
{
    interp1_linear(x, nx, xp, fp_re, nxp, out_re);
    interp1_linear(x, nx, xp, fp_im, nxp, out_im);
}


/******************************************* FFT Helpers *************************************************/

/**
 * @brief In-place fftshift for complex array.
 *
 * Swaps left and right halves of the array (for centering zero-frequency).
 *
 * @param[in,out] re  Real part array.
 * @param[in,out] im  Imaginary part array.
 * @param[in]     n   Array length.
 */
static void fftshift_complex(float* re, float* im, int n)
{
    int i, half, shift;
    float tmp_re, tmp_im;
    
    if (!re || !im || n <= 1) {
        return;
    }
    
    half = n / 2;
    shift = (n + 1) / 2; /* For odd n, shift = (n+1)/2 */
    
    /* Use rotation algorithm for in-place shift */
    /* This handles both even and odd n correctly */
    for (i = 0; i < half; ++i) {
        tmp_re = re[i];
        tmp_im = im[i];
        re[i] = re[i + shift];
        im[i] = im[i + shift];
        re[i + shift] = tmp_re;
        im[i + shift] = tmp_im;
    }
}

/**
 * @brief Find bandwidth flank position.
 *
 * Finds the x-position where the normalized spectrum first exceeds the cutoff.
 *
 * @param[in] x          Frequency axis array.
 * @param[in] spectrum_re Real part of spectrum.
 * @param[in] spectrum_im Imaginary part of spectrum.
 * @param[in] n          Array length.
 * @param[in] cutoff     Cutoff level (0 to 1, typically 0.5).
 * @param[in] reverse    If non-zero, search from end to beginning.
 * @return Frequency at flank position.
 */
static float find_spectrum_flank(
    const float* x,
    const float* spectrum_re,
    const float* spectrum_im,
    int n,
    float cutoff,
    int reverse
) {
    int i, idx;
    float maxMag, mag, threshold;
    
    if (!x || !spectrum_re || !spectrum_im || n <= 0) {
        return 0.0f;
    }
    
    /* Find max magnitude */
    maxMag = 0.0f;
    for (i = 0; i < n; ++i) {
        mag = (float)sqrt(spectrum_re[i] * spectrum_re[i] + spectrum_im[i] * spectrum_im[i]);
        if (mag > maxMag) {
            maxMag = mag;
        }
    }
    
    if (maxMag < 1e-12f) {
        return 0.0f;
    }
    
    threshold = cutoff * maxMag;
    
    /* Search for first index exceeding threshold */
    if (reverse) {
        for (i = n - 1; i >= 0; --i) {
            mag = (float)sqrt(spectrum_re[i] * spectrum_re[i] + spectrum_im[i] * spectrum_im[i]);
            if (mag > threshold) {
                return x[i];
            }
        }
    } else {
        for (i = 0; i < n; ++i) {
            mag = (float)sqrt(spectrum_re[i] * spectrum_re[i] + spectrum_im[i] * spectrum_im[i]);
            if (mag > threshold) {
                return x[i];
            }
        }
    }
    
    return 0.0f;
}

/**
 * @brief Compute RF bandwidth from uniformly sampled complex signal via FFT.
 *
 * Assumes rf_re and rf_im are already on the FFT grid (nn samples, dt spacing).
 * Performs fftshift, FFT, fftshift, then finds bandwidth at cutoff level.
 *
 * @param[in]     rf_re     Real part of RF signal on FFT grid.
 * @param[in]     rf_im     Imaginary part of RF signal on FFT grid.
 * @param[in]     fft_cfg   Pre-allocated kiss_fft configuration.
 * @param[in]     nn        Number of FFT points.
 * @param[in]     dw        Frequency resolution (Hz).
 * @param[in]     cutoff    Bandwidth cutoff level (0 to 1, typically 0.5).
 * @param[in]     duration  Pulse duration for fallback (s).
 * @param[in]     w         Frequency grid array.
 * @param[in,out] work_re   Work array for real part (size nn).
 * @param[in,out] work_im   Work array for imag part (size nn).
 * @param[in,out] fft_in    FFT input array.
 * @param[in,out] fft_out   FFT output array.
 * @return Bandwidth in Hz.
 */
static float compute_rf_bandwidth_fft(
    const float* rf_re,
    const float* rf_im,
    kiss_fft_cfg fft_cfg,
    int nn,
    float dw,
    float cutoff,
    float duration,
    const float* w,
    float* work_re,
    float* work_im,
    kiss_fft_cpx* fft_in,
    kiss_fft_cpx* fft_out
) {
    int i;
    float w1, w2, bw;
    float fallback_bw;
    
    /* If calc fails, assume it is block pulse and take 99% power bandwidth */
    fallback_bw = (duration > 0.0f) ? (3.12f / duration) : 0.0f;
    
    if (!rf_re || !rf_im || !fft_cfg || nn <= 0) {
        return fallback_bw;
    }
    
    /* Copy input to work arrays */
    for (i = 0; i < nn; ++i) {
        work_re[i] = rf_re[i];
        work_im[i] = rf_im[i];
    }
    
    /* Apply fftshift to input (center the signal) */
    fftshift_complex(work_re, work_im, nn);
    
    /* Copy to FFT input */
    for (i = 0; i < nn; ++i) {
        fft_in[i].r = work_re[i];
        fft_in[i].i = work_im[i];
    }
    
    /* Perform complex FFT */
    kiss_fft(fft_cfg, fft_in, fft_out);
    
    /* Copy FFT output back to work arrays */
    for (i = 0; i < nn; ++i) {
        work_re[i] = fft_out[i].r;
        work_im[i] = fft_out[i].i;
    }
    
    /* Apply fftshift to output */
    fftshift_complex(work_re, work_im, nn);
    
    /* Find bandwidth flanks */
    w1 = find_spectrum_flank(w, work_re, work_im, nn, cutoff, 0);
    w2 = find_spectrum_flank(w, work_re, work_im, nn, cutoff, 1);
    
    bw = w2 - w1;
    
    return (bw > 0.0f) ? bw : fallback_bw;
}

/***************************************************** Private Functions  ***************************************/
int initStandardLibrary(FILE* f, const long* offsets, int numSections, void** target, int* targetCount, int N) 
{
    char line[MAX_LINE_LENGTH];
    int maxIndex = -1;
    int sec, i, j, idx;
    char* p;
    float *array_raw;

    if (!f) return 1;
    for (sec = 0; sec < numSections; sec++) {
        if (offsets[sec] < 0) continue;  /* Skip not found */

        if (fseek(f, offsets[sec], SEEK_SET) != 0) {
            return 1;
        }

        /* Skip the section header line */
        if (!fgets(line, sizeof(line), f)) {
            return 1;
        }

        /* Read until next section or EOF */
        while (fgets(line, sizeof(line), f)) {
            p = line;
            while (*p == ' ' || *p == '\t') p++;
            if (*p == '[' || *p == 'e') break;     /* Next section */
            if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue; /* Skip blank/comment */
            if (sscanf(p, "%d", &idx) == 1) {
                if (idx > maxIndex) maxIndex = idx;
            }
        }
    }

    if (maxIndex <= 0) {
        *target = NULL;
        *targetCount = 0;
        return 0; /* No entries found: treat as empty */
    }

    /* Allocate zero-filled 2D array as a single block */
    array_raw = (float*) ALLOC(sizeof(float) * N * maxIndex);
    if (!array_raw) return 1;
    for (i = 0; i < maxIndex * N; i++) {
        array_raw[i] = 0.0f;
    }
    *target = (void*)array_raw;
    *targetCount = maxIndex;
    return 0;
}


int initDefinitionsLibrary(FILE* f, long offset, pulseqlib_Definition** target, int* targetCount) {
    char line[MAX_LINE_LENGTH];
    int count = 0;
    char* p;
    char* nameToken;
    pulseqlib_Definition* defs;

    if (!f || offset < 0 || !target || !targetCount) return 1;

    if (fseek(f, offset, SEEK_SET) != 0) return 1;

    /* Skip section header line */
    if (!fgets(line, sizeof(line), f)) {
        return 1;
    }

    /* Count valid definition lines */
    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '[' || *p == 'e') break;     /* Next section */
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue; /* Skip blank/comment */        nameToken = strtok(p, " \t\r\n");
        if (nameToken) count++;
    }

    if (count == 0) {
        *target = NULL;
        *targetCount = 0;
        return 1;  /* No Definitions found */
    }

    defs = (pulseqlib_Definition*) ALLOC(sizeof(pulseqlib_Definition) * count);
    if (!defs) return 1;

    *target = defs;
    *targetCount = count;
    return 0;
}


int initShapesLibrary(FILE* f, long offset, pulseqlib_ShapeArbitrary** target, int* targetCount) {
    char line[MAX_LINE_LENGTH];
    int maxIndex = -1;
    int n, i, idx;
    char* p;
    pulseqlib_ShapeArbitrary* shapes = NULL;

    if (!f || !offset || !target || !targetCount) {
        return 1;  /* Invalid arguments */
    }

    if (fseek(f, offset, SEEK_SET) != 0) {
        return 1;  /* Seek failed */
    }

    /* Skip the section header line */
    if (!fgets(line, sizeof(line), f)) {
        return 1;
    }

    /* First pass: find max shape_id and count shapes */
    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '[' || *p == 'e') break;     /* Next section */
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue; /* Skip blank/comment */
        if (strncmp(p, "shape_id", 8) == 0) {
            if (sscanf(p + 8, "%d", &idx) == 1) {
                if (idx > maxIndex) maxIndex = idx;
            }
        }
    }

    if (maxIndex <= 0) {
        *target = NULL;
        *targetCount = 0;
        return 0;
    }

    shapes = (pulseqlib_ShapeArbitrary*) ALLOC(sizeof(pulseqlib_ShapeArbitrary) * maxIndex);
    if (!shapes) return 1;
    for (i = 0; i < maxIndex; i++) {
        shapes[i].numSamples = 0;
        shapes[i].numUncompressedSamples = 0;
        shapes[i].samples = NULL;
    }

    /* Reset file pointer for second pass */
    if (fseek(f, offset, SEEK_SET) != 0) {
        FREE(shapes);
        return 1;
    }

    /* Skip the section header line */
    if (!fgets(line, sizeof(line), f)) {
        return 1;
    }

    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '[' || *p == 'e') break;     /* Next section */
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue; /* Skip blank/comment */
        if (strncmp(p, "shape_id", 8) == 0) {
            if (sscanf(p + 8, "%d", &idx) == 1) {
                shapes[idx - 1].numSamples = 0;
                shapes[idx - 1].numUncompressedSamples = 0;
                shapes[idx - 1].samples = NULL;
            } 
        }
        else if (strncmp(p, "num_samples", 11) == 0) {
            if (sscanf(p + 11, "%d", &n) == 1) {
                shapes[idx - 1].numUncompressedSamples = n;
            }
        }
        else {
            shapes[idx - 1].numSamples++;
        }
    }

    /* Allocate sample arrays */
    for (i = 0; i < maxIndex; i++) {
        int j;
        n = shapes[i].numSamples;
        if (n > 0) {
            shapes[i].samples = (float*) ALLOC(sizeof(float) * n);
            if (!shapes[i].samples) {
                for (j = 0; j < i; j++) {
                    if (shapes[j].samples) FREE(shapes[j].samples);
                }
                FREE(shapes);
                return 1;
            }
        } else {
            shapes[i].samples = NULL;
        }
    }

    *target = shapes;
    *targetCount = maxIndex;
    return 0;
}


int initRfShimLibrary(FILE* f, long offset, pulseqlib_RfShimEntry** target, int* targetCount) {
    char line[MAX_LINE_LENGTH];
    int maxIndex = -1;
    char* p;
    int idx, i;
    pulseqlib_RfShimEntry* array;

    if (!f || !target || !targetCount) return 1;
    if (fseek(f, offset, SEEK_SET) != 0) return 1;

    /* Skip section header line */
    if (!fgets(line, sizeof(line), f)) {
        return 1;
    }

    /* First pass: determine max index */
    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '[' || *p == 'e') break;     /* Next section */
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue; /* Skip blank/comment */
        if (sscanf(p, "%d", &idx) == 1) {
            if (idx > maxIndex) maxIndex = idx;
        }
    }

    if (maxIndex <= 0) {
        *target = NULL;
        *targetCount = 0;
        return 1; /* No entries found */
    }

    /* Allocate array of RfShimEntry */
    array = (pulseqlib_RfShimEntry*) ALLOC(sizeof(pulseqlib_RfShimEntry) * maxIndex);
    if (!array) return 1;

    for (i = 0; i < maxIndex; i++) {
        array[i].nChannels = 0;
    }

    *target = array;
    *targetCount = maxIndex;
    return 0;
}


typedef struct Scale {
    int size;  /**< Number of values to scale */
    const float* values; /**< Array of scaling factors */
} Scale;


int readStandardLibrary(FILE* f, long offset, void* target, int targetCount, int N, Scale scale, int flag) {
    char line[MAX_LINE_LENGTH];
    int idx, parsed, consumed, n, offsetCol;                        
    float vals[MAX_SCALE_SIZE];
    char* scanPtr;
    char* p;
    float v;

    float *array_raw = (float*)target;
    if (!f) return 1;
    if (scale.size > MAX_SCALE_SIZE) return 1;
    if (fseek(f, offset, SEEK_SET) != 0) {
        return 1;
    }

    /* Skip section header line */
    if (!fgets(line, sizeof(line), f)) {
        return 1;
    }

    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '[' || *p == 'e') break;     /* Next section */
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue; /* Skip blank/comment */
        if (sscanf(p, "%d", &idx) != 1) continue;
        if (idx <= 0 || idx > targetCount) continue;

        /* Move pointer past index */
        while (*p && *p != ' ' && *p != '\t') p++;
        while (*p == ' ' || *p == '\t') p++;

        parsed = 0;
        scanPtr = p;

        for (n = 0; n < scale.size; n++) {
            consumed = 0;
            if (sscanf(scanPtr, "%f%n", &v, &consumed) != 1) break;
            vals[n] = v;
            scanPtr += consumed;
            while (*scanPtr == ' ' || *scanPtr == '\t') scanPtr++;
            parsed++;
        }

        if (parsed != scale.size) continue;

        offsetCol = (flag >= 0) ? 1 : 0;
        for (n = 0; n < scale.size; n++) {
            array_raw[(idx - 1) * N + n + offsetCol] = vals[n] * scale.values[n];
        }
        if (flag >= 0) {
            array_raw[(idx - 1) * N + 0] = (float)flag;
        }
    }

    return 0;
}


void getSectionOffsets(pulseqlib_SeqFile* seq, FILE* f) {
    char line[MAX_LINE_LENGTH];
    char* p;
    long pos;

    char extName[EXT_NAME_LENGTH];
    int extId, extEnum;

    if (fseek(f, 0L, SEEK_SET) != 0) return;

    while (fgets(line, sizeof(line), f)) {
        pos = ftell(f);
        if (pos < 0) break;

        p = line;
        while (*p == ' ' || *p == '\t') p++;

        if (*p == '[') {
            if (strncmp(p, "[VERSION]", 9) == 0)
                seq->offsets.version = pos - strlen(line);
            else if (strncmp(p, "[DEFINITIONS]", 13) == 0)
                seq->offsets.definitions = pos - strlen(line);
            else if (strncmp(p, "[BLOCKS]", 8) == 0)
                seq->offsets.blocks = pos - strlen(line);
            else if (strncmp(p, "[RF]", 4) == 0)
                seq->offsets.rf = pos - strlen(line);
            else if (strncmp(p, "[GRADIENTS]", 11) == 0)
                seq->offsets.grad = pos - strlen(line);
            else if (strncmp(p, "[TRAP]", 6) == 0)
                seq->offsets.trap = pos - strlen(line);
            else if (strncmp(p, "[ADC]", 5) == 0)
                seq->offsets.adc = pos - strlen(line);
            else if (strncmp(p, "[EXTENSIONS]", 12) == 0)
                seq->offsets.extensions = pos - strlen(line);
            else if (strncmp(p, "[SHAPES]", 8) == 0)
                seq->offsets.shapes = pos - strlen(line);
            else if (strncmp(p, "[SIGNATURE]", 11) == 0)
                seq->offsets.signature = pos - strlen(line);
        }
        else if (strncmp(p, "extension", 9) == 0 && (*(p + 9) == ' ' || *(p + 9) == '\t')) {
            extId = -1;
            extEnum = EXT_UNKNOWN;
            if (sscanf(p, "extension %31s %d", extName, &extId) == 2) {
                if (strcmp(extName, "TRIGGERS") == 0) extEnum = EXT_TRIGGER;
                else if (strcmp(extName, "ROTATIONS") == 0) extEnum = EXT_ROTATION;
                else if (strcmp(extName, "LABELSET") == 0) extEnum = EXT_LABELSET;
                else if (strcmp(extName, "LABELINC") == 0) extEnum = EXT_LABELINC;
                else if (strcmp(extName, "RF_SHIMS") == 0) extEnum = EXT_RF_SHIM;
                else if (strcmp(extName, "DELAYS") == 0) extEnum = EXT_DELAY;
                switch (extEnum) {
                    case EXT_TRIGGER:
                        seq->offsets.triggers = pos - strlen(line);
                        break;
                    case EXT_ROTATION:
                        seq->offsets.rotations = pos - strlen(line);
                        break;
                    case EXT_LABELSET:
                        seq->offsets.labelset = pos - strlen(line);
                        break;
                    case EXT_LABELINC:
                        seq->offsets.labelinc = pos - strlen(line);
                        break;
                    case EXT_RF_SHIM:
                        seq->offsets.rfshim = pos - strlen(line);
                        break;
                    case EXT_DELAY:
                        seq->offsets.delays = pos - strlen(line);
                        break;
                    default:
                        break;
                }   
                if (extEnum >= 0 && extEnum < EXT_UNKNOWN) {
                    seq->extensionMap[extEnum] = extId;
                }
            }
        }
    }

    /* Set scan_cursor to EOF */
    seq->offsets.scan_cursor = ftell(f);
}


void readVersion(pulseqlib_SeqFile* seq, FILE* f) {
    char line[MAX_LINE_LENGTH];
    int major = 0, minor = 0, revision = 0;
    char key[32];
    int value;
    char* p;

    /* Check if library was already parsed */
    if (seq->isVersionParsed) return;

    /* Go to the correct section */
    if (seq->offsets.version < 0) {
        seq->isVersionParsed = 1;
        return;
    }

    if (fseek(f, seq->offsets.version, SEEK_SET) != 0) {
        return;
    }

    /* Skip section header */
    if (!fgets(line, sizeof(line), f)) {
        return;
    }

    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue; /* Skip blank/comment */
        if (*p == '[') break;
        if (sscanf(p, "%31s %d", key, &value) == 2) {
            if (strcmp(key, "major") == 0) major = value;
            else if (strcmp(key, "minor") == 0) minor = value;
            else if (strcmp(key, "revision") == 0) revision = value;
        }
    }

    seq->versionMajor = major;
    seq->versionMinor = minor;
    seq->versionRevision = revision;
    seq->versionCombined = major * 1000000 + minor * 1000 + revision;
    seq->isVersionParsed = 1;
}


void readDefinitionsLibrary(pulseqlib_SeqFile* seq, FILE* f) {
    int ret;
    char line[MAX_LINE_LENGTH];
    int defIndex = 0;
    char* p;
    char* nameToken;
    char* token;
    char** newArray;
    int i;
    pulseqlib_Definition def;

    /* Check if library was already parsed */
    if (seq->isDefinitionsLibraryParsed) return;

    /* Go to the correct section */
    if (seq->offsets.definitions < 0) {
        seq->isDefinitionsLibraryParsed = 1;
        return;
    }

    /* Preallocate definitions array */
    ret = initDefinitionsLibrary(f, (seq->offsets).definitions, &seq->definitionsLibrary, &seq->numDefinitions);
    if (ret != 0) {
        return;
    }

    /* Second pass — parse values */
    if (fseek(f, seq->offsets.definitions, SEEK_SET) != 0) return;

    /* Skip section header line */
    if (!fgets(line, sizeof(line), f)) {
        return;
    }

    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (isspace((unsigned char)*p)) p++;

        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue;
        if (*p == '[') break;  /* Next section begins */

        def.valueSize = 0;
        def.value = NULL;

        /* Parse name */
        nameToken = strtok(p, " \t\r\n");
        if (!nameToken) continue;
        strncpy(def.name, nameToken, DEFINITION_NAME_LENGTH - 1);
        def.name[DEFINITION_NAME_LENGTH - 1] = '\0';

        /* Parse values */
        while ((token = strtok(NULL, " \t\r\n")) != NULL) {
            newArray = (char**) ALLOC(sizeof(char*) * (def.valueSize + 1));
            for (i = 0; i < def.valueSize; i++) {
                newArray[i] = def.value[i];
            }

            newArray[def.valueSize] = (char*) ALLOC(strlen(token) + 1);
            strcpy(newArray[def.valueSize], token);
            if (def.value) FREE(def.value);
            def.value = newArray;
            def.valueSize++;
        }

        /* Assign parsed definition */
        seq->definitionsLibrary[defIndex++] = def;
    }

    seq->isDefinitionsLibraryParsed = 1;
}


void readDefinitions(pulseqlib_SeqFile* seq) {
    int i;
    char* key;
    char* value;
    float temp[3];

    /* Parse definitionsLibrary */
    for (i = 0; i < seq->numDefinitions; i++) {
        key = seq->definitionsLibrary[i].name;
        value = seq->definitionsLibrary[i].value[0];

        /* Parse required reserved definitions */
        if (strcmp(key, "GradientRasterTime") == 0) {
            seq->reservedDefinitionsLibrary.gradientRasterTime = atof(value) * 1e6; /* Convert to us */
        } else if (strcmp(key, "RadiofrequencyRasterTime") == 0) {
            seq->reservedDefinitionsLibrary.radiofrequencyRasterTime = atof(value) * 1e6; /* Convert to us */
        } else if (strcmp(key, "AdcRasterTime") == 0) {
            seq->reservedDefinitionsLibrary.adcRasterTime = atof(value) * 1e6; /* Convert to us */
        } else if (strcmp(key, "BlockDurationRaster") == 0) {
            seq->reservedDefinitionsLibrary.blockDurationRaster = atof(value) * 1e6; /* Convert to us */
        }

        /* Parse optional reserved definitions */
        else if (strcmp(key, "Name") == 0) {
            strncpy(seq->reservedDefinitionsLibrary.name, value, sizeof(seq->reservedDefinitionsLibrary.name) - 1);
            seq->reservedDefinitionsLibrary.name[sizeof(seq->reservedDefinitionsLibrary.name) - 1] = '\0';
        } else if (strcmp(key, "FOV") == 0) {
            if (sscanf(value, "%f %f %f", &temp[0], &temp[1], &temp[2]) == 3) {
                seq->reservedDefinitionsLibrary.fov[0] = temp[0] * 100.0f; /* Convert to cm */
                seq->reservedDefinitionsLibrary.fov[1] = temp[1] * 100.0f; /* Convert to cm */
                seq->reservedDefinitionsLibrary.fov[2] = temp[2] * 100.0f; /* Convert to cm */
            }
        } else if (strcmp(key, "TotalDuration") == 0) {
            seq->reservedDefinitionsLibrary.totalDuration = atof(value); /* Already in seconds */
        } else if (strcmp(key, "NextSequence") == 0) {
            strncpy(seq->reservedDefinitionsLibrary.nextSequence, value, sizeof(seq->reservedDefinitionsLibrary.nextSequence) - 1);
            seq->reservedDefinitionsLibrary.nextSequence[sizeof(seq->reservedDefinitionsLibrary.nextSequence) - 1] = '\0';
        }
    }

    /* Check for missing required definitions */
    if (seq->reservedDefinitionsLibrary.gradientRasterTime == 0.0f ||
        seq->reservedDefinitionsLibrary.radiofrequencyRasterTime == 0.0f ||
        seq->reservedDefinitionsLibrary.adcRasterTime == 0.0f ||
        seq->reservedDefinitionsLibrary.blockDurationRaster == 0.0f) {
    }
}


void readBlockLibrary(pulseqlib_SeqFile* seq, FILE* f) {
    int ret;
    float block_values[7] = {1, 1, 1, 1, 1, 1, 1};
    Scale blockScale;
    blockScale.size = 7;
    blockScale.values = block_values;
    const char* block_section[] = {"[BLOCKS]"};

    /* Check if library was already parsed */
    if (seq->isBlockLibraryParsed) return;

    /* Go to the correct section */
    if (seq->offsets.blocks < 0) {
        seq->isBlockLibraryParsed = 1;
        return;
    }

    /* Preallocate library */
    ret = initStandardLibrary(f,  &((seq->offsets).blocks), 1, (void**)&seq->blockLibrary, &seq->numBlocks, blockScale.size);
    if (ret != 0) {
        return;
    }

    /* Parse Block library */
    ret = readStandardLibrary(f, seq->offsets.blocks, seq->blockLibrary, seq->numBlocks, blockScale.size, blockScale, -1);
    if (ret != 0) {
        return;
    }

    seq->isBlockLibraryParsed = 1;
}


void readRfLibrary(pulseqlib_SeqFile* seq, FILE* f) {
    int ret;
    float rf_values[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    Scale rfScale;
    rfScale.size = 10;
    rfScale.values = rf_values;
    const char* rf_section[] = {"[RF]"};

    /* Check if library was already parsed */
    if (seq->isRfLibraryParsed) return;

    /* Go to the correct section */
    if (seq->offsets.rf < 0) {
        seq->isRfLibraryParsed = 1;
        return;
    }

    /* Preallocate library */
    ret = initStandardLibrary(f,  &((seq->offsets).rf), 1, (void**)&seq->rfLibrary, &seq->rfLibrarySize, rfScale.size);
    if (ret != 0) {
        return;
    }

    /* Parse RF library */
    ret = readStandardLibrary(f, seq->offsets.rf, seq->rfLibrary, seq->rfLibrarySize, rfScale.size, rfScale, -1);
    if (ret != 0) {
        return;
    }

    seq->isRfLibraryParsed = 1;
}


void readGradLibrary(pulseqlib_SeqFile* seq, FILE* f) {
    int ret;
    long offsets[2] = { seq->offsets.grad, seq->offsets.trap };
    int numSections = 0;
    Scale gradScale;
    static const float gradScaleValues[6] = { 1, 1, 1, 1, 1, 1 };
    Scale trapScale;
    static const float trapScaleValues[5] = { 1, 1, 1, 1, 1 };
    const char* sections[] = { "[GRADIENTS]", "[TRAP]" };

    gradScale.size = 6;
    gradScale.values = gradScaleValues;
    trapScale.size = 5;
    trapScale.values = trapScaleValues;


    if (seq->isGradLibraryParsed) return;

    /* Go to the correct section */
    (seq->offsets).grad = offsets[0];
    (seq->offsets).trap = offsets[1];

    /* If SeqFile does not have gradients, exit*/
    if ((seq->offsets).grad >= 0) numSections++;
    if ((seq->offsets).trap >= 0) numSections++;
    if (numSections == 0) {
        seq->isGradLibraryParsed = 1;
        return;
    }

    /* Preallocate library */
    ret = initStandardLibrary(f, offsets, 2, (void**)&seq->gradLibrary, &seq->gradLibrarySize, gradScale.size + 1);
    if (ret != 0) {
        return;
    }

    /* Parse GRADIENTS library */
    if ((seq->offsets).grad >= 0){
        ret = readStandardLibrary(f, offsets[0], seq->gradLibrary, seq->gradLibrarySize, gradScale.size + 1, gradScale, 1);
        if (ret != 0) {
            return;
        }
    }

    /* Parse TRAP library */
    if ((seq->offsets).trap >= 0){
        ret = readStandardLibrary(f, offsets[1], seq->gradLibrary, seq->gradLibrarySize, gradScale.size + 1, trapScale, 0);
        if (ret != 0) {
            return;
        }
    }

    seq->isGradLibraryParsed = 1;
}


void readAdcLibrary(pulseqlib_SeqFile* seq, FILE* f) {
    int ret;
    Scale adcScale;
    static const float adcScaleValues[8] = { 1, 1, 1, 1, 1, 1, 1, 1 };
    const char* adc_section[] = {"[ADC]"};

    adcScale.size = 8;
    adcScale.values = adcScaleValues;

    /* Check if library was already parsed */
    if (seq->isAdcLibraryParsed) return;

    /* Go to the correct section */
    if (seq->offsets.adc < 0) {
        seq->isAdcLibraryParsed = 1;
        return;
    }

    /* Preallocate library */
    ret = initStandardLibrary(f,  &((seq->offsets).adc), 1, (void**)&seq->adcLibrary, &seq->adcLibrarySize, adcScale.size);
    if (ret != 0) {
        return;
    }

    /* Parse ADC library */
    ret = readStandardLibrary(f, seq->offsets.adc, seq->adcLibrary, seq->adcLibrarySize, adcScale.size, adcScale, -1);
    if (ret != 0) {
        return;
    }

    seq->isAdcLibraryParsed = 1;
}


void readShapesLibrary(pulseqlib_SeqFile* seq, FILE* f) {
    int ret;
    const char* shape_section[] = {"[SHAPES]"};
    char line[MAX_LINE_LENGTH];
    int shapeIndex;
    int sampleIndex;
    long pos;
    char* p;
    float val;
    
    /* Check if library was already parsed */
    if (seq->isShapesLibraryParsed) return;

    /* Go to the correct section */
    if (seq->offsets.shapes < 0) {
        seq->isShapesLibraryParsed = 1;
        return;
    }

    /* Preallocate shapes array */
    ret = initShapesLibrary(f, (seq->offsets).shapes, &seq->shapesLibrary, &seq->shapesLibrarySize);
    if (ret != 0) {
        return;
    }

    /* Second pass: Parse and fill waveform data */
    pos = seq->offsets.shapes;
    if (fseek(f, pos, SEEK_SET) != 0) return;

    /* Skip section header line */
    if (!fgets(line, sizeof(line), f)) {
        return;
    }

    /* Actual parsing */
    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;

        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue; /* Skip blank/comment */
        if (*p == '[') break;

        /* Beginning of waveform: parse shape ID */
        if (strncmp(p, "shape_id", 8) == 0) {
            if (sscanf(p + 8, "%d", &shapeIndex) == 1) sampleIndex = 0;
        }

        /* Number of uncompressed samples: skip (already stored) */
        if (strncmp(p, "num_samples", 11) == 0) {
            continue;
        }

        /* Parse waveform sample value */
        if (shapeIndex > 0 && shapeIndex <= seq->shapesLibrarySize) {
            if (sscanf(p, "%f", &val) == 1){
                if(sampleIndex < seq->shapesLibrary[shapeIndex - 1].numSamples) {
                    seq->shapesLibrary[shapeIndex - 1].samples[sampleIndex++] = val;
                }
            }
        }
    }

    seq->isShapesLibraryParsed = 1;
}


int readLabelLibrary(FILE* f, long offset, void* target, int targetCount, int N, int* isLabelDefined) {
    char line[MAX_LINE_LENGTH];
    char* p;
    int idx, labelCode;
    float val;
    char label[LABEL_NAME_LENGTH];

    float *array_raw = (float*)target;
    if (!f || offset < 0) return 1;
    if (fseek(f, offset, SEEK_SET) != 0) return 1;

    /* Skip section header line */
    if (!fgets(line, sizeof(line), f)) {
        return 1;
    }

    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '[' || *p == 'e') break;     /* Next section */
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue; /* Skip blank/comment */
        if (sscanf(p, "%d %f %31s", &idx, &val, label) != 3) continue;
        if (idx <= 0 || idx > targetCount) continue;
        labelCode = label2enum(label); 
        if (labelCode > 0){ /* bookkeep found labels and flags */
            isLabelDefined[labelCode - 1] = 1;
        }
        array_raw[(idx - 1) * N + 0] = val;
        array_raw[(idx - 1) * N + 1] = (float)labelCode;
    }
    
    return 0;
}


int readDelayLibrary(FILE* f, long offset, void* target, int targetCount, int N, int* isDelayDefined) {
    char line[MAX_LINE_LENGTH];
    char* p;
    int idx, hintCode;
    float numID, offsetVal, scaleVal;
    char hint[SOFT_DELAY_HINT_LENGTH];

    float *array_raw = (float*)target;
    if (!f || offset < 0) return 1;
    if (fseek(f, offset, SEEK_SET) != 0) return 1;
    
    /* Skip section header line */
    if (!fgets(line, sizeof(line), f)) {
        return 1;
    }

    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '[' || *p == 'e') break;     /* Next section */
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue; /* Skip blank/comment */
        if (sscanf(p, "%d %f %f %f %31s", &idx, &numID, &offsetVal, &scaleVal, hint) != 5) continue;
        if (idx <= 0 || idx > targetCount) continue;
        hintCode = hint2enum(hint);
        if (hintCode > 0){ /* bookkeep found hint for UI */
            isDelayDefined[hintCode - 1] = 1;
        }
        array_raw[(idx - 1) * N + 0] = numID;
        array_raw[(idx - 1) * N + 1] = offsetVal;
        array_raw[(idx - 1) * N + 2] = scaleVal;
        array_raw[(idx - 1) * N + 3] = (float)hintCode;
    }

    return 0;
}


int readRfShimLibrary(FILE* f, long offset, pulseqlib_RfShimEntry* target, int targetCount) {
    char line[MAX_LINE_LENGTH];
    char* p;
    int idx, nCh, i, consumed;
    float val;

    if (!f || !target) return 1;
    if (fseek(f, offset, SEEK_SET) != 0) return 1;

    /* Skip section header line */
    if (!fgets(line, sizeof(line), f)) {
        return 1;
    }

    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '[' || *p == 'e') break;     /* Next section */
        if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') continue; /* Skip blank/comment */        if (sscanf(p, "%d %d", &idx, &nCh) != 2) continue;
        if (idx <= 0 || idx > targetCount || nCh <= 0) continue;

        /* Skip past the index and nCh */
        while (*p && *p != ' ') p++; while (*p == ' ') p++;
        while (*p && *p != ' ') p++; while (*p == ' ') p++;

        if (nCh > MAX_RF_SHIM_CHANNELS) return 1; /* Too many channels */
        target[idx - 1].nChannels = nCh;
        for (i = 0; i < 2 * nCh; i++) {
            consumed = 0;
            if (sscanf(p, "%f%n", &val, &consumed) != 1) break;
            target[idx - 1].values[i] = val;
            p += consumed;
            while (*p == ' ' || *p == '\t') p++;
        }

    }

    return 0;
}


void readExtensionsLibrary(pulseqlib_SeqFile* seq, FILE* f) {
    int ret;
    int n;
    Scale extScale;
    static const float extScaleValues[3] = { 1, 1, 1 };
    Scale trigScale;
    static const float trigScaleValues[4] = { 1, 1, 1, 1 };
    Scale rotScale;
    static const float rotScaleValues[4] = { 1, 1, 1, 1 };

    extScale.size = 3;
    extScale.values = extScaleValues;
    trigScale.size = 4;
    trigScale.values = trigScaleValues;
    rotScale.size = 4;
    rotScale.values = rotScaleValues;

    /* Check if library was already parsed */
    if (seq->isExtensionsLibraryParsed) return;

    /* Go to the correct section */
    if (seq->offsets.extensions < 0) {
        seq->isExtensionsLibraryParsed = 1;
        return;
    }

    /* Preallocate library */
    ret = initStandardLibrary(f, &((seq->offsets).extensions), 1, (void**)&seq->extensionsLibrary, &seq->extensionsLibrarySize, extScale.size);
    if (ret != 0) {
        return;
    }

    if (seq->offsets.triggers >= 0){
        ret = initStandardLibrary(f, &((seq->offsets).triggers), 1, (void**)&seq->triggerLibrary, &seq->triggerLibrarySize, trigScale.size);
        if (ret != 0) {
            return;
        }
    }

    if (seq->offsets.rotations >= 0){
        ret = initStandardLibrary(f, &((seq->offsets).rotations), 1, (void**)&seq->rotationQuaternionLibrary, &seq->rotationLibrarySize, rotScale.size);
        if (ret != 0) {
            return;
        }
    }

    if (seq->offsets.labelset >= 0){
        ret = initStandardLibrary(f, &((seq->offsets).labelset), 1, (void**)&seq->labelsetLibrary, &seq->labelsetLibrarySize, 2);
        if (ret != 0) {
            return;
        }
    }

    if (seq->offsets.labelinc >= 0){
        ret = initStandardLibrary(f, &((seq->offsets).labelinc), 1, (void**)&seq->labelincLibrary, &seq->labelincLibrarySize, 2);
        if (ret != 0) {
            return;
        }
    }

    if (seq->offsets.delays >= 0){
        ret = initStandardLibrary(f, &((seq->offsets).delays), 1, (void**)&seq->softDelayLibrary, &seq->softDelayLibrarySize, 4);
        if (ret != 0) {
            return;
        }
    }

    if (seq->offsets.rfshim >= 0){
        ret = initRfShimLibrary(f, seq->offsets.rfshim, &seq->rfShimLibrary, &seq->rfShimLibrarySize);
        if (ret != 0) {
            return;
        }
    }

    /* Parse Extensions library */
    ret = readStandardLibrary(f, seq->offsets.extensions, seq->extensionsLibrary, seq->extensionsLibrarySize, extScale.size, extScale, -1);
    if (ret != 0) {
        return;
    }

    if (seq->offsets.triggers >= 0){
        ret = readStandardLibrary(f, seq->offsets.triggers, seq->triggerLibrary, seq->triggerLibrarySize, trigScale.size, trigScale, -1);
        if (ret != 0) {
            return;
        }
    }

    if (seq->offsets.rotations >= 0){
        ret = readStandardLibrary(f, seq->offsets.rotations, seq->rotationQuaternionLibrary, seq->rotationLibrarySize, rotScale.size, rotScale, -1);
        if (ret != 0) {
            return;
        }
        {
            float quatNorm;
            int n;
            for(n = 1; n < seq->rotationLibrarySize; n++){
                quatNorm = (float)sqrt(pow((double)seq->rotationQuaternionLibrary[n][0], 2) + pow((double)seq->rotationQuaternionLibrary[n][1], 2) + 
                            pow((double)seq->rotationQuaternionLibrary[n][2], 2) + pow((double)seq->rotationQuaternionLibrary[n][3], 2));
                seq->rotationQuaternionLibrary[n][0] = seq->rotationQuaternionLibrary[n][0] / quatNorm; /* manually unroll - with so few entries, more readable than loop */
                seq->rotationQuaternionLibrary[n][1] = seq->rotationQuaternionLibrary[n][1] / quatNorm;
                seq->rotationQuaternionLibrary[n][2] = seq->rotationQuaternionLibrary[n][2] / quatNorm;
                seq->rotationQuaternionLibrary[n][3] = seq->rotationQuaternionLibrary[n][3] / quatNorm;
            }
        }
        
        /* If using matrix format, we'll convert the quaternions to matrices in pulseqlib_seqFile */
    }

    if (seq->offsets.labelset >= 0){
        ret = readLabelLibrary(f, seq->offsets.labelset, seq->labelsetLibrary, seq->labelsetLibrarySize, 2, seq->isLabelDefined);
        if (ret != 0) {
            return;
        }
    }

    if (seq->offsets.labelinc >= 0){
        ret = readLabelLibrary(f, seq->offsets.labelinc, seq->labelincLibrary, seq->labelincLibrarySize, 2, seq->isLabelDefined);
        if (ret != 0) {
            return;
        }
    }

    if (seq->offsets.delays >= 0){
        ret = readDelayLibrary(f, seq->offsets.delays, seq->softDelayLibrary, seq->softDelayLibrarySize, 4, seq->isDelayDefined);
        if (ret != 0) {
            return;
        }
    }

    if (seq->offsets.rfshim >= 0){
        ret = readRfShimLibrary(f, seq->offsets.rfshim, seq->rfShimLibrary, seq->rfShimLibrarySize);
        if (ret != 0) {
            return;
        }
    }

    /* Prepare extensionLUT */
    for (n = 0; n < 8; n++){
        if (seq->extensionLUTSize < seq->extensionMap[n]){
            seq->extensionLUTSize = seq->extensionMap[n];
        }
    }
    if (seq->extensionLUTSize > 0){
        seq->extensionLUT = (int*) ALLOC(sizeof(int) * (seq->extensionLUTSize + 1));
        for (n = 0; n < 8; n++){
            if (seq->extensionMap[n] > 0) seq->extensionLUT[seq->extensionMap[n]] = n;
        }
    }

    seq->isExtensionsLibraryParsed = 1;
}


int decompressShape(pulseqlib_ShapeArbitrary* encoded, pulseqlib_ShapeArbitrary* result, float scale)
{
    int i, rep;
    const float *packed;
    int numPacked, numSamples;
    int countPack = 1;
    int countUnpack = 1;
    float* unpacked;
    
    /* Validate inputs */
    if (!encoded || !result) {
        return 0; /* Invalid inputs */
    }
    
    packed = encoded->samples;
    numPacked = encoded->numSamples;
    numSamples = encoded->numUncompressedSamples;
    
    /* Input shape is uncompressed - copy it */
    if (encoded->numSamples == encoded->numUncompressedSamples) {
        result->numSamples = encoded->numSamples;
        result->numUncompressedSamples = encoded->numUncompressedSamples;
        result->samples = (float*) ALLOC(sizeof(float) * encoded->numSamples);
        if (!result->samples) {
            return 0; /* Allocation failed */
        }
        for (i = 0; i < encoded->numSamples; ++i) {
            result->samples[i] = encoded->samples[i] * scale;
        }
        return 1; /* Success */
    }

    unpacked = (float*) ALLOC(sizeof(float) * numSamples);
    if (unpacked == NULL) {
        return 0; /* Allocation failed */
    }

    while (countPack < numPacked) {
        if (packed[countPack - 1] != packed[countPack]) {
            unpacked[countUnpack - 1] = packed[countPack - 1];
            countPack++;
            countUnpack++;
        } else {
            rep = (int)(packed[countPack + 1]) + 2;
            if (fabsf(packed[countPack + 1] + 2 - (float)rep) > 1e-6f) {
                /* Malformed shape compression format */
                FREE(unpacked);
                return 0; /* Failed */
            }
            for (i = countUnpack - 1; i <= countUnpack + rep - 2; i++) {
                unpacked[i] = packed[countPack - 1];
            }
            countPack += 3;
            countUnpack += rep;
        }
    }

    if (countPack == numPacked) {
        unpacked[countUnpack - 1] = packed[countPack - 1];
    }

    /* Cumulative sum */
    for (i = 1; i < numSamples; i++) {
        unpacked[i] += unpacked[i - 1];
    }

    /* Apply scale factor */
    for (i = 0; i < numSamples; i++) {
        unpacked[i] *= scale;
    }

    result->numSamples = numSamples;
    result->numUncompressedSamples = numSamples;
    result->samples = unpacked;

    return 1; /* Success */
}


/******************************************* Public methods *************************************************/
void pulseqlib_optsInit(pulseqlib_Opts* opts, float gamma, float B0, float max_grad, float max_slew, float rf_raster_time, float grad_raster_time, float adc_raster_time, float block_duration_raster){
    if (!opts) return;
    opts->gamma = gamma;
    opts->B0 = B0;
    opts->max_grad = max_grad;
    opts->max_slew = max_slew;
    opts->rf_raster_time = rf_raster_time;
    opts->grad_raster_time = grad_raster_time;
    opts->adc_raster_time = adc_raster_time;
    opts->block_duration_raster = block_duration_raster;
}


void pulseqlib_optsFree(pulseqlib_Opts* opts) {
    if (!opts) return;
    memset(opts, 0, sizeof(*opts));
}


static void seqFileSetDefaults(pulseqlib_SeqFile* seq) {
    int i;
    if (!seq) return;

    seq->offsets.scan_cursor = 0;
    seq->offsets.version = -1;
    seq->offsets.definitions = -1;
    seq->offsets.blocks = -1;
    seq->offsets.rf = -1;
    seq->offsets.grad = -1;
    seq->offsets.trap = -1;
    seq->offsets.adc = -1;
    seq->offsets.extensions = -1;
    seq->offsets.triggers = -1;
    seq->offsets.rfshim = -1;
    seq->offsets.labelset = -1;
    seq->offsets.labelinc = -1;
    seq->offsets.delays = -1;
    seq->offsets.rotations = -1;
    seq->offsets.shapes = -1;
    seq->offsets.signature = -1;

    seq->isVersionParsed = 0;
    seq->versionCombined = 0;
    seq->versionMajor = 0;
    seq->versionMinor = 0;
    seq->versionRevision = 0;

    INIT_LIBRARY(seq, definitionsLibrary, numDefinitions, isDefinitionsLibraryParsed);
    seq->reservedDefinitionsLibrary = (pulseqlib_ReservedDefinitions){0};

    INIT_LIBRARY(seq, blockLibrary, numBlocks, isBlockLibraryParsed);
    seq->blockIDs = NULL;
    INIT_LIBRARY(seq, rfLibrary, rfLibrarySize, isRfLibraryParsed);
    INIT_LIBRARY(seq, gradLibrary, gradLibrarySize, isGradLibraryParsed);
    INIT_LIBRARY(seq, adcLibrary, adcLibrarySize, isAdcLibraryParsed);
    INIT_LIBRARY(seq, extensionsLibrary, extensionsLibrarySize, isExtensionsLibraryParsed);
    INIT_LIBRARY(seq, triggerLibrary, triggerLibrarySize, isExtensionsLibraryParsed);
    INIT_LIBRARY(seq, rotationQuaternionLibrary, rotationLibrarySize, isExtensionsLibraryParsed);
    INIT_LIBRARY(seq, rotationMatrixLibrary, rotationLibrarySize, isExtensionsLibraryParsed);
    INIT_LIBRARY(seq, labelsetLibrary, labelsetLibrarySize, isExtensionsLibraryParsed);
    INIT_LIBRARY(seq, labelincLibrary, labelincLibrarySize, isExtensionsLibraryParsed);
    for (i = 0; i < 22; i++) {
        seq->isLabelDefined[i] = 0;
    }
    memset(&seq->labelLimits, 0, sizeof(seq->labelLimits));
    for (i = 0; i < 8; i++) {
        seq->isDelayDefined[i] = 0;
        seq->extensionMap[i] = -1;
    }
    INIT_LIBRARY(seq, softDelayLibrary, softDelayLibrarySize, isExtensionsLibraryParsed);
    INIT_LIBRARY(seq, rfShimLibrary, rfShimLibrarySize, isExtensionsLibraryParsed);
    seq->extensionLUTSize = 0;
    seq->extensionLUT = NULL;
    INIT_LIBRARY(seq, shapesLibrary, shapesLibrarySize, isShapesLibraryParsed);
}


static void seqFileInit(pulseqlib_SeqFile* seq, const pulseqlib_Opts* opts) {
    if (!seq) return;
    seq->filePath = NULL;
    seq->opts = *opts;
    seqFileSetDefaults(seq);
}


void pulseqlib_seqFileInit(pulseqlib_SeqFile* seq, const pulseqlib_Opts* opts) {
    seqFileInit(seq, opts);
}


static void seqFileReset(pulseqlib_SeqFile* seq) {
    int i, j;
    if (!seq) return;
    if (seq->isDefinitionsLibraryParsed && seq->definitionsLibrary) {
        for (i = 0; i < seq->numDefinitions; i++) {
            FREE(seq->definitionsLibrary[i].value);
        }
        FREE(seq->definitionsLibrary);
    }
    if (seq->isBlockLibraryParsed) {
        FREE(seq->blockLibrary);
        FREE(seq->blockIDs);
        seq->blockIDs = NULL;
    }
    if (seq->isRfLibraryParsed)         FREE(seq->rfLibrary);
    if (seq->isGradLibraryParsed)       FREE(seq->gradLibrary);
    if (seq->isAdcLibraryParsed)        FREE(seq->adcLibrary);
    if (seq->isExtensionsLibraryParsed) {
        FREE(seq->extensionsLibrary);
        FREE(seq->triggerLibrary);
        
        /* Free both rotation libraries to be safe */
        FREE(seq->rotationQuaternionLibrary);
        FREE(seq->rotationMatrixLibrary);
        
        FREE(seq->labelsetLibrary);
        FREE(seq->labelincLibrary);
        FREE(seq->softDelayLibrary);
        FREE(seq->rfShimLibrary);
    }
    if (seq->isShapesLibraryParsed && seq->shapesLibrary) {
        for (i = 0; i < seq->shapesLibrarySize; i++) {
            FREE(seq->shapesLibrary[i].samples);
            seq->shapesLibrary[i].samples = NULL;
            seq->shapesLibrary[i].numUncompressedSamples = 0;
            seq->shapesLibrary[i].numSamples = 0;
        }
        FREE(seq->shapesLibrary);
    }

    FREE(seq->extensionLUT);
    seq->extensionLUT = NULL;
    
    seqFileSetDefaults(seq);
}


void pulseqlib_seqFileFree(pulseqlib_SeqFile *seq) {
    if (!seq) return;
    seqFileReset(seq);
    if (seq->filePath) {
        FREE(seq->filePath);
        seq->filePath = NULL;
    }
    pulseqlib_optsFree(&seq->opts);
    FREE(seq);
}

void pulseqlib_seqFileCollectionFree(pulseqlib_SeqFileCollection* collection) {
    int i;
    
    if (!collection) return;
    
    if (collection->sequences) {
        for (i = 0; i < collection->numSequences; ++i) {
            pulseqlib_seqFileFree(&collection->sequences[i]);
        }
        FREE(collection->sequences);
    }
    
    if (collection->basePath) {
        FREE(collection->basePath);
    }
    
    collection->numSequences = 0;
    collection->sequences = NULL;
    collection->basePath = NULL;
}

/**
 * @brief Initializes a sequence block with default values.
 *
 * @param[out] block The pre-allocated block structure to initialize
 * @return 1 if successful, 0 if failed
 */
void pulseqlib_seqBlockInit(pulseqlib_SeqBlock* block) {
    pulseqlib_RFEvent rf;
    pulseqlib_GradEvent gx;
    pulseqlib_GradEvent gy;
    pulseqlib_GradEvent gz;
    pulseqlib_ADCEvent adc;
    pulseqlib_TriggerEvent trigger;
    pulseqlib_RotationEvent rotation;
    pulseqlib_FlagEvent flag;
    pulseqlib_LabelEvent labelset;
    pulseqlib_LabelEvent labelinc;
    pulseqlib_SoftDelayEvent delay;
    pulseqlib_RfShimmingEvent rfShimming;
    
    /* Check for null pointer */
    if (!block) return;

    /* Initialize rf Event*/
    rf.type = 0;
    gx.type = 0;
    gy.type = 0;
    gz.type = 0;
    adc.type = 0;
    trigger.type = 0;
    rotation.type = 0;
    delay.type = 0;
    rfShimming.type = 0;

    /* Initialize flag values to 0 */
    flag.trid = -1;
    flag.nav = -1;
    flag.rev = -1;
    flag.sms = -1;
    flag.ref = -1;
    flag.ima = -1;
    flag.noise = -1;
    flag.pmc = -1;
    flag.norot = -1;
    flag.nopos = -1;
    flag.noscl = -1;
    flag.once = -1;
    
    /* Initialize label values to 0 */
    labelset.slc = 0;
    labelset.seg = 0;
    labelset.rep = 0;
    labelset.avg = 0;
    labelset.set = 0;
    labelset.eco = 0;
    labelset.phs = 0;
    labelset.lin = 0;
    labelset.par = 0;
    labelset.acq = 0;
    labelinc.slc = 0;
    labelinc.seg = 0;
    labelinc.rep = 0;
    labelinc.avg = 0;
    labelinc.set = 0;
    labelinc.eco = 0;
    labelinc.phs = 0;
    labelinc.lin = 0;
    labelinc.par = 0;
    labelinc.acq = 0;

    /* Initialize the block */
    block->rf = rf;
    block->gx = gx;
    block->gy = gy;
    block->gz = gz;
    block->adc = adc;
    block->trigger = trigger;
    block->rotation = rotation;
    block->flag = flag;
    block->labelset = labelset;
    block->labelinc = labelinc;
    block->delay = delay;
    block->rfShimming = rfShimming;
}


/**
 * @brief Frees all resources associated with a SeqBlock.
 *
 * This function deallocates memory for all waveform samples and resets the block.
 *
 * @param[in,out] block The SeqBlock to be freed.
 */
void pulseqlib_seqBlockFree(pulseqlib_SeqBlock* block) {
    if (block == 0) return;

    /* RF waveforms */
    if (block->rf.type > 0){
        if (block->rf.magShape.samples) {
            FREE(block->rf.magShape.samples);
            block->rf.magShape.samples = NULL;
        }
        if (block->rf.phaseShape.samples) {
            FREE(block->rf.phaseShape.samples);
            block->rf.phaseShape.samples = NULL;
        }
        if (block->rf.timeShape.samples) {
            FREE(block->rf.timeShape.samples);
            block->rf.timeShape.samples = NULL;
        }
    }

    /* GX waveforms */
    if (block->gx.type > 1){
        if (block->gx.waveShape.samples) {
            FREE(block->gx.waveShape.samples);
            block->gx.waveShape.samples = NULL;
        }
        if (block->gx.timeShape.samples) {
            FREE(block->gx.timeShape.samples);
            block->gx.timeShape.samples = NULL;
        }
    }

    /* GY waveforms */
    if (block->gy.type > 1){
        if (block->gy.waveShape.samples) {
            FREE(block->gy.waveShape.samples);
            block->gy.waveShape.samples = NULL;
        }
        if (block->gy.timeShape.samples) {
            FREE(block->gy.timeShape.samples);
            block->gy.timeShape.samples = NULL;
        }
    }

    /* GZ waveforms */
    if (block->gz.type > 1){
        if (block->gz.waveShape.samples) {
            FREE(block->gz.waveShape.samples);
            block->gz.waveShape.samples = NULL;
        }
        if (block->gz.timeShape.samples) {
            FREE(block->gz.timeShape.samples);
            block->gz.timeShape.samples = NULL;
        }
    }

    /* ADC waveform */
    if (block->adc.type > 0){
        if (block->adc.phaseModulationShape.samples) {
            FREE(block->adc.phaseModulationShape.samples);
            block->adc.phaseModulationShape.samples = NULL;
        }
    }

    /* RF shimming arrays */
    if (block->rfShimming.type > 0){
        if (block->rfShimming.amplitudes) {
            FREE(block->rfShimming.amplitudes);
            block->rfShimming.amplitudes = NULL;
        }
        if (block->rfShimming.phases) {
            FREE(block->rfShimming.phases);
            block->rfShimming.phases = NULL;
        }
    }
}

/**
 * @brief Read SeqFile content from buffer.
 * 
 * @param[in, out] seq The SeqFile structure.
 * @param[in] f The FILE buffer.
 */
int pulseqlib_readSeqFromBuffer(pulseqlib_SeqFile* seq, FILE* f) {
    if (!seq || !f) return PULSEQLIB_ERR_NULL_POINTER;

    seqFileReset(seq);

    if (seq->filePath) {
        FREE(seq->filePath);
        seq->filePath = NULL;
    }

    getSectionOffsets(seq, f);
    readVersion(seq, f);
    if (seq->versionCombined < 1005000) {
        return PULSEQLIB_ERR_UNSUPPORTED_VERSION;
    }
    readDefinitionsLibrary(seq, f); 
    readDefinitions(seq);
    readBlockLibrary(seq, f);
    readRfLibrary(seq, f);
    readGradLibrary(seq, f);
    readAdcLibrary(seq, f);
    readShapesLibrary(seq, f);
    readExtensionsLibrary(seq, f);  
    return PULSEQLIB_OK;    
}

/**
 * @brief Read SeqFile content from file.
 * 
 * @param[in, out] seq The SeqFile structure.
 * @param[in] filePath The path to the sequence file.
 */
int pulseqlib_readSeq(pulseqlib_SeqFile* seq, const char* filePath) {
    FILE* f;
    int code;
    if (!seq || !filePath) return PULSEQLIB_ERR_INVALID_ARGUMENT;
    f = fopen(filePath, "r");
    if (!f) return PULSEQLIB_ERR_FILE_NOT_FOUND;
    code = pulseqlib_readSeqFromBuffer(seq, f);
    fclose(f);
    return code;
}


static int countSequencesInChain(const char* firstFilePath, const pulseqlib_Opts* opts) {
    int count = 0;
    int maxCount = 1000;  /* Prevent infinite loops from circular chains */
    char* currentPath;
    char* basePath;
    char* nextPath;
    pulseqlib_SeqFile tempSeq;
    int result;
    
    basePath = extractBasePath(firstFilePath);
    if (!basePath) return -1;
    
    currentPath = (char*)ALLOC(strlen(firstFilePath) + 1);
    if (!currentPath) {
        FREE(basePath);
        return -1;
    }
    strcpy(currentPath, firstFilePath);
    
    /* Walk the chain, counting sequences */
    while (currentPath && currentPath[0] != '\0' && count < maxCount) {
        /* Initialize temp sequence to read only definitions */
        pulseqlib_seqFileInit(&tempSeq, opts);
        result = pulseqlib_readSeq(&tempSeq, currentPath);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_seqFileFree(&tempSeq);
            FREE(currentPath);
            FREE(basePath);
            return -1;
        }
        
        count++;
        
        /* Prepare for next iteration */
        FREE(currentPath);
        currentPath = NULL;
        
        if (tempSeq.reservedDefinitionsLibrary.nextSequence[0] != '\0') {
            currentPath = buildFullPath(basePath, tempSeq.reservedDefinitionsLibrary.nextSequence);
            if (!currentPath) {
                pulseqlib_seqFileFree(&tempSeq);
                FREE(basePath);
                return -1;
            }
        }
        
        pulseqlib_seqFileFree(&tempSeq);
    }
    
    FREE(basePath);
    
    /* Check if we hit the limit (likely circular) */
    if (count >= maxCount) {
        return -1;
    }
    
    return count;
}


int pulseqlib_readSeqCollection(
    pulseqlib_SeqFileCollection* collection,
    const char* firstFilePath,
    const pulseqlib_Opts* opts)
{
    int numSeq;
    int i;
    char* currentPath;
    char* basePath;
    int result;
    
    if (!collection || !firstFilePath || !opts) {
        return PULSEQLIB_ERR_NULL_POINTER;
    }
    
    /* PASS 1: Count sequences in chain */
    numSeq = countSequencesInChain(firstFilePath, opts);
    if (numSeq <= 0) {
        collection->numSequences = 0;
        collection->sequences = NULL;
        collection->basePath = NULL;
        return (numSeq == 0) ? PULSEQLIB_ERR_COLLECTION_EMPTY : PULSEQLIB_ERR_COLLECTION_CHAIN_BROKEN;
    }
    
    /* Initialize collection base path */
    collection->basePath = extractBasePath(firstFilePath);
    if (!collection->basePath) {
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }
    
    /* PASS 2: Allocate and populate sequence array */
    collection->sequences = (pulseqlib_SeqFile*)ALLOC(numSeq * sizeof(pulseqlib_SeqFile));
    if (!collection->sequences) {
        FREE(collection->basePath);
        collection->basePath = NULL;
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }
    
    currentPath = (char*)ALLOC(strlen(firstFilePath) + 1);
    if (!currentPath) {
        FREE(collection->sequences);
        FREE(collection->basePath);
        collection->sequences = NULL;
        collection->basePath = NULL;
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }
    strcpy(currentPath, firstFilePath);
    
    /* Parse each sequence in the chain */
    for (i = 0; i < numSeq; ++i) {
        /* Initialize and parse current sequence */
        pulseqlib_seqFileInit(&collection->sequences[i], opts);
        result = pulseqlib_readSeq(&collection->sequences[i], currentPath);
        if (PULSEQLIB_FAILED(result)) {
            /* Clean up all previously parsed sequences */
            for (int j = 0; j < i; ++j) {
                pulseqlib_seqFileFree(&collection->sequences[j]);
            }
            FREE(collection->sequences);
            FREE(currentPath);
            FREE(collection->basePath);
            collection->sequences = NULL;
            collection->basePath = NULL;
            return result;
        }
        
        /* Prepare path for next sequence */
        if (i < numSeq - 1) {
            FREE(currentPath);
            currentPath = buildFullPath(
                collection->basePath,
                collection->sequences[i].reservedDefinitionsLibrary.nextSequence);
            if (!currentPath) {
                for (int j = 0; j <= i; ++j) {
                    pulseqlib_seqFileFree(&collection->sequences[j]);
                }
                FREE(collection->sequences);
                FREE(collection->basePath);
                collection->sequences = NULL;
                collection->basePath = NULL;
                return PULSEQLIB_ERR_ALLOC_FAILED;
            }
        }
    }
    
    FREE(currentPath);
    collection->numSequences = numSeq;
    
    return PULSEQLIB_OK;
}


static int getRawBlockContentIDs(const pulseqlib_SeqFile* seq, pulseqlib_RawBlock* block, int blockIndex, int parseExtensions) {
    int nextExtID;
    int extCount;
    float* eventFloat;
    float* extData;

    if (!seq || !block || blockIndex < 0 || blockIndex >= seq->numBlocks) {
        return 0;
    }

    block->block_duration = 0;
    block->rf = -1;
    block->gx = -1;
    block->gy = -1;
    block->gz = -1;
    block->adc = -1;
    block->extCount = 0;

    if (!seq->blockLibrary || !seq->blockLibrary[blockIndex]) {
        return 0;
    }

    eventFloat = seq->blockLibrary[blockIndex];

    block->block_duration = (int)eventFloat[0];
    block->rf = (int)eventFloat[1] - 1;
    block->gx = (int)eventFloat[2] - 1;
    block->gy = (int)eventFloat[3] - 1;
    block->gz = (int)eventFloat[4] - 1;
    block->adc = (int)eventFloat[5] - 1;

    if (!parseExtensions) {
        block->extCount = 0;
        return 1;
    }

    if (!seq->isExtensionsLibraryParsed || !seq->extensionsLibrary || seq->extensionsLibrarySize <= 0) {
        return 1;
    }

    nextExtID = (int)eventFloat[6];
    extCount = 0;

    while (nextExtID > 0 && nextExtID <= seq->extensionsLibrarySize && extCount < MAX_EXTENSIONS_PER_BLOCK) {
        extData = seq->extensionsLibrary[nextExtID - 1];
        block->ext[extCount][0] = (int)extData[0];
        block->ext[extCount][1] = (int)extData[1] - 1;
        nextExtID = (int)extData[2];
        extCount += 1;
    }

    block->extCount = extCount;
    return 1;
}

/******************************************* Extension Parsing *************************************************/
void rawExtensionInit(pulseqlib_RawExtension* rawExt) {
    if (!rawExt) return;
    
    /* Initialize labels to 0 */
    memset(&rawExt->labelset, 0, sizeof(rawExt->labelset));
    memset(&rawExt->labelinc, 0, sizeof(rawExt->labelinc));
    
    /* Initialize flags to -1 (undefined) */
    rawExt->flag.trid = -1;
    rawExt->flag.nav = -1;
    rawExt->flag.rev = -1;
    rawExt->flag.sms = -1;
    rawExt->flag.ref = -1;
    rawExt->flag.ima = -1;
    rawExt->flag.noise = -1;
    rawExt->flag.pmc = -1;
    rawExt->flag.norot = -1;
    rawExt->flag.nopos = -1;
    rawExt->flag.noscl = -1;
    rawExt->flag.once = -1;
    
    /* Initialize indices to -1 (not present) */
    rawExt->rotationIndex = -1;
    rawExt->rfShimIndex = -1;
    rawExt->triggerIndex = -1;
    rawExt->softDelayIndex = -1;
}


void extensionBlockInit(pulseqlib_ExtensionBlock* extBlock) {
    if (!extBlock) return;
    
    memset(&extBlock->labelset, 0, sizeof(extBlock->labelset));
    memset(&extBlock->labelinc, 0, sizeof(extBlock->labelinc));
    
    /* Initialize flags to -1 (undefined) */
    extBlock->flag.trid = -1;
    extBlock->flag.nav = -1;
    extBlock->flag.rev = -1;
    extBlock->flag.sms = -1;
    extBlock->flag.ref = -1;
    extBlock->flag.ima = -1;
    extBlock->flag.noise = -1;
    extBlock->flag.pmc = -1;
    extBlock->flag.norot = -1;
    extBlock->flag.nopos = -1;
    extBlock->flag.noscl = -1;
    extBlock->flag.once = -1;
    
    /* Initialize rotation */
    extBlock->rotation.type = 0;
    memset(&extBlock->rotation.data, 0, sizeof(extBlock->rotation.data));
    
    /* Initialize RF shimming */
    extBlock->rfShimming.type = 0;
    extBlock->rfShimming.nChan = 0;
    extBlock->rfShimming.amplitudes = NULL;
    extBlock->rfShimming.phases = NULL;

    /* Initialize trigger */
    extBlock->trigger.type = 0;
    extBlock->trigger.duration = 0;
    extBlock->trigger.delay = 0;
    extBlock->trigger.triggerType = 0;
    extBlock->trigger.triggerChannel = 0;

    /* Initialize soft delay */
    extBlock->softDelay.type = 0;
    extBlock->softDelay.numID = 0;
    extBlock->softDelay.hintID = 0;
    extBlock->softDelay.offset = 0;
    extBlock->softDelay.factor = 0;
}


void extensionBlockFree(pulseqlib_ExtensionBlock* extBlock) {
    if (!extBlock) return;
    
    if (extBlock->rfShimming.amplitudes) {
        FREE(extBlock->rfShimming.amplitudes);
        extBlock->rfShimming.amplitudes = NULL;
    }
    if (extBlock->rfShimming.phases) {
        FREE(extBlock->rfShimming.phases);
        extBlock->rfShimming.phases = NULL;
    }
    extBlock->rfShimming.type = 0;
    extBlock->rfShimming.nChan = 0;
}


void getRawExtension(const pulseqlib_SeqFile* seq, pulseqlib_RawExtension* rawExt, const pulseqlib_RawBlock* raw) {
    int i;
    int typeIdx;
    int refIdx;
    int extType;
    int labelValue;
    int labelID;
    
    rawExtensionInit(rawExt);
    
    if (!seq || !rawExt || !raw) return;
    if (!seq->isExtensionsLibraryParsed || !seq->extensionLUT) return;
    
    for (i = 0; i < raw->extCount; ++i) {
        typeIdx = raw->ext[i][0];
        refIdx = raw->ext[i][1];
        
        if (typeIdx < 0 || typeIdx > seq->extensionLUTSize) continue;
        extType = seq->extensionLUT[typeIdx];
        if (refIdx < 0) continue;
        
        switch (extType) {
            case EXT_LABELSET:
                if (seq->labelsetLibrary && refIdx < seq->labelsetLibrarySize) {
                    labelValue = (int)seq->labelsetLibrary[refIdx][0];
                    labelID = (int)seq->labelsetLibrary[refIdx][1];
                    switch (labelID) {
                        case SLC: rawExt->labelset.slc = labelValue; break;
                        case SEG: rawExt->labelset.seg = labelValue; break;
                        case REP: rawExt->labelset.rep = labelValue; break;
                        case AVG: rawExt->labelset.avg = labelValue; break;
                        case SET: rawExt->labelset.set = labelValue; break;
                        case ECO: rawExt->labelset.eco = labelValue; break;
                        case PHS: rawExt->labelset.phs = labelValue; break;
                        case LIN: rawExt->labelset.lin = labelValue; break;
                        case PAR: rawExt->labelset.par = labelValue; break;
                        case ACQ: rawExt->labelset.acq = labelValue; break;
                        /* Flags from labelset */
                        case NAV: rawExt->flag.nav = labelValue; break;
                        case REV: rawExt->flag.rev = labelValue; break;
                        case SMS: rawExt->flag.sms = labelValue; break;
                        case REF: rawExt->flag.ref = labelValue; break;
                        case IMA: rawExt->flag.ima = labelValue; break;
                        case NOISE: rawExt->flag.noise = labelValue; break;
                        case PMC: rawExt->flag.pmc = labelValue; break;
                        case NOROT: rawExt->flag.norot = labelValue; break;
                        case NOPOS: rawExt->flag.nopos = labelValue; break;
                        case NOSCL: rawExt->flag.noscl = labelValue; break;
                        case ONCE: rawExt->flag.once = labelValue; break;
                        case TRID: rawExt->flag.trid = labelValue; break;
                        default: break;
                    }
                }
                break;
            case EXT_LABELINC:
                if (seq->labelincLibrary && refIdx < seq->labelincLibrarySize) {
                    labelValue = (int)seq->labelincLibrary[refIdx][0];
                    labelID = (int)seq->labelincLibrary[refIdx][1];
                    switch (labelID) {
                        case SLC: rawExt->labelinc.slc = labelValue; break;
                        case SEG: rawExt->labelinc.seg = labelValue; break;
                        case REP: rawExt->labelinc.rep = labelValue; break;
                        case AVG: rawExt->labelinc.avg = labelValue; break;
                        case SET: rawExt->labelinc.set = labelValue; break;
                        case ECO: rawExt->labelinc.eco = labelValue; break;
                        case PHS: rawExt->labelinc.phs = labelValue; break;
                        case LIN: rawExt->labelinc.lin = labelValue; break;
                        case PAR: rawExt->labelinc.par = labelValue; break;
                        case ACQ: rawExt->labelinc.acq = labelValue; break;
                        default: break;
                    }
                }
                break;
            case EXT_ROTATION:
                rawExt->rotationIndex = refIdx;
                break;
            case EXT_RF_SHIM:
                rawExt->rfShimIndex = refIdx;
                break;
            case EXT_TRIGGER:
                rawExt->triggerIndex = refIdx;
                break;
            case EXT_DELAY:
                rawExt->softDelayIndex = refIdx;
                break;
            default:
                break;
        }
    }
}


static int parseRotationFromRawExtension(const pulseqlib_SeqFile* seq, pulseqlib_ExtensionBlock* extBlock, const pulseqlib_RawExtension* rawExt) {
    int i;
    int refIdx = rawExt->rotationIndex;
    
    if (refIdx < 0) return 1; /* No rotation extension, not an error */
    
#if ROTATION_FORMAT == ROTATION_FORMAT_QUATERNION
    if (seq->rotationQuaternionLibrary && refIdx < seq->rotationLibrarySize) {
        extBlock->rotation.type = 1;
        for (i = 0; i < 4; ++i) {
            extBlock->rotation.data.rotQuaternion[i] = seq->rotationQuaternionLibrary[refIdx][i];
        }
    }
#elif ROTATION_FORMAT == ROTATION_FORMAT_MATRIX
    if (seq->rotationMatrixLibrary && refIdx < seq->rotationLibrarySize) {
        extBlock->rotation.type = 1;
        for (i = 0; i < 9; ++i) {
            extBlock->rotation.data.rotMatrix[i] = seq->rotationMatrixLibrary[refIdx][i];
        }
    }
#endif
    
    return 1;
}


static int parseRfShimFromRawExtension(const pulseqlib_SeqFile* seq, pulseqlib_ExtensionBlock* extBlock, const pulseqlib_RawExtension* rawExt) {
    int refIdx = rawExt->rfShimIndex;
    int i, n;
    const pulseqlib_RfShimEntry* entry;
    float* amplitudes;
    float* phases;
    
    if (refIdx < 0) return 1; /* No RF shim extension, not an error */
    
    if (!seq->rfShimLibrary || refIdx >= seq->rfShimLibrarySize) {
        return 1; /* Invalid reference, skip */
    }
    
    entry = &seq->rfShimLibrary[refIdx];
    n = entry->nChannels;
    
    if (n <= 0 || n > MAX_RF_SHIM_CHANNELS) {
        return 1; /* Invalid channel count */
    }
    
    amplitudes = (float*) ALLOC(sizeof(float) * n);
    phases = (float*) ALLOC(sizeof(float) * n);
    
    if (!amplitudes || !phases) {
        if (amplitudes) FREE(amplitudes);
        if (phases) FREE(phases);
        return 0; /* Allocation failure */
    }
    
    for (i = 0; i < n; ++i) {
        amplitudes[i] = entry->values[2 * i];
        phases[i] = entry->values[2 * i + 1];
    }
    
    extBlock->rfShimming.type = 1;
    extBlock->rfShimming.nChan = n;
    extBlock->rfShimming.amplitudes = amplitudes;
    extBlock->rfShimming.phases = phases;
    
    return 1;
}


static int parseTriggerFromRawExtension(const pulseqlib_SeqFile* seq, pulseqlib_ExtensionBlock* extBlock, const pulseqlib_RawExtension* rawExt) {
    int refIdx = rawExt->triggerIndex;

    if (refIdx < 0) return 1; /* No trigger extension, not an error */
    
    if (!seq->triggerLibrary || refIdx >= seq->triggerLibrarySize) {
        return 1; /* Invalid reference, skip */
    }
     
    /* File format: id type channel delay duration */
    /* Array indices: [0]=type, [1]=channel, [2]=delay, [3]=duration */
    extBlock->trigger.type = 1;
    extBlock->trigger.triggerType = (int)seq->triggerLibrary[refIdx][0];
    extBlock->trigger.triggerChannel = (int)seq->triggerLibrary[refIdx][1];
    extBlock->trigger.delay = seq->triggerLibrary[refIdx][2];
    extBlock->trigger.duration = seq->triggerLibrary[refIdx][3];
    
    return 1;
}

static int parseSoftDelayFromRawExtension(const pulseqlib_SeqFile* seq, pulseqlib_ExtensionBlock* extBlock, const pulseqlib_RawExtension* rawExt) {
    int refIdx = rawExt->softDelayIndex;

    if (refIdx < 0) return 1; /* No soft delay extension, not an error */
    
    if (!seq->softDelayLibrary || refIdx >= seq->softDelayLibrarySize) {
        return 1; /* Invalid reference, skip */
    }
        
    /* readDelayLibrary stores: [0]=numID, [1]=offset, [2]=factor, [3]=hintCode */
    extBlock->softDelay.type = 1;
    extBlock->softDelay.numID = (int)seq->softDelayLibrary[refIdx][0];
    extBlock->softDelay.offset = seq->softDelayLibrary[refIdx][1];
    extBlock->softDelay.factor = seq->softDelayLibrary[refIdx][2];
    extBlock->softDelay.hintID = (int)seq->softDelayLibrary[refIdx][3];
    
    return 1;
}

int parseExtension(const pulseqlib_SeqFile* seq, pulseqlib_ExtensionBlock* extBlock, const pulseqlib_RawExtension* rawExt) {
    if (!extBlock) return 0;
    
    extensionBlockInit(extBlock);
    
    if (!seq || !rawExt) return 1; /* No data to parse, but not an error */
    
    /* Copy labels and flags directly from rawExt (already parsed) */
    extBlock->labelset = rawExt->labelset;
    extBlock->labelinc = rawExt->labelinc;
    extBlock->flag = rawExt->flag;
    
    /* Parse rotation */
    if (!parseRotationFromRawExtension(seq, extBlock, rawExt)) {
        extensionBlockFree(extBlock);
        return 0;
    }
    
    /* Parse RF shimming */
    if (!parseRfShimFromRawExtension(seq, extBlock, rawExt)) {
        extensionBlockFree(extBlock);
        return 0;
    }

    /* Parse trigger */
    if (!parseTriggerFromRawExtension(seq, extBlock, rawExt)) {
        extensionBlockFree(extBlock);
        return 0;
    }

    /* Parse soft delay */
    if (!parseSoftDelayFromRawExtension(seq, extBlock, rawExt)) {
        extensionBlockFree(extBlock);
        return 0;
    }
    
    return 1;
}


void applyExtension(const pulseqlib_ExtensionBlock* extBlock, pulseqlib_SeqBlock* block) {
    int i, n;
    float* amplitudes;
    float* phases;
    
    if (!extBlock || !block) return;
    
    /* Apply labels */
    block->labelset = extBlock->labelset;
    block->labelinc = extBlock->labelinc;
    block->flag = extBlock->flag;
    
    /* Apply rotation */
    block->rotation = extBlock->rotation;
    
    /* Apply RF shimming (need to copy the allocated arrays) */
    if (extBlock->rfShimming.type && extBlock->rfShimming.nChan > 0) {
        n = extBlock->rfShimming.nChan;
        amplitudes = (float*) ALLOC(sizeof(float) * n);
        phases = (float*) ALLOC(sizeof(float) * n);
        
        if (amplitudes && phases) {
            for (i = 0; i < n; ++i) {
                amplitudes[i] = extBlock->rfShimming.amplitudes[i];
                phases[i] = extBlock->rfShimming.phases[i];
            }
            block->rfShimming.type = 1;
            block->rfShimming.nChan = n;
            block->rfShimming.amplitudes = amplitudes;
            block->rfShimming.phases = phases;
        } else {
            if (amplitudes) FREE(amplitudes);
            if (phases) FREE(phases);
        }
    }

    /* Apply trigger */
    block->trigger = extBlock->trigger;

    /* Apply soft delay */
    block->delay = extBlock->softDelay;
}


int parseBlockWithoutExt(const pulseqlib_SeqFile* seq, pulseqlib_SeqBlock* block, const pulseqlib_RawBlock* raw, const pulseqlib_RawExtension* rawExt) {
    float* farray;
    float* trig;
    float* delay;
    int idx;
    int i;
    int* isRealSample = NULL;
    float blockDurationRaster_us, rfRasterTime_us, gradRasterTime_us;
    pulseqlib_ShapeArbitrary shape;

    if (!seq || !raw || !block) {
        return 0;
    }

    if (seq->reservedDefinitionsLibrary.blockDurationRaster > 0.0f) {
        blockDurationRaster_us = seq->reservedDefinitionsLibrary.blockDurationRaster;
    } else {
        blockDurationRaster_us = seq->opts.block_duration_raster;
    }

    if (seq->reservedDefinitionsLibrary.radiofrequencyRasterTime > 0.0f) {
        rfRasterTime_us = seq->reservedDefinitionsLibrary.radiofrequencyRasterTime;
    } else {
        rfRasterTime_us = seq->opts.rf_raster_time;
    }

    if (seq->reservedDefinitionsLibrary.gradientRasterTime > 0.0f) {
        gradRasterTime_us = seq->reservedDefinitionsLibrary.gradientRasterTime;
    } else {
        gradRasterTime_us = seq->opts.grad_raster_time;
    }

    block->duration = raw->block_duration * (int)blockDurationRaster_us;

    /* Parse RF event */
    if (raw->rf >= 0 && seq->rfLibrary && raw->rf < seq->rfLibrarySize) {
        int numRealSamples = 0;
        farray = seq->rfLibrary[raw->rf];
        block->rf.type = 1;
        block->rf.amplitude = farray[0];

        /* Magnitude shape*/
        idx = (int)farray[1];
        if (idx > 0 && seq->isShapesLibraryParsed && idx <= seq->shapesLibrarySize) {
            if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape, 1.0f)) {
                pulseqlib_seqBlockFree(block);
                pulseqlib_seqBlockInit(block);
                return 0;
            }
            block->rf.magShape = shape;
        } else {
            block->rf.magShape.numSamples = 0;
            block->rf.magShape.numUncompressedSamples = 0;
            block->rf.magShape.samples = NULL;
        }

        /* Phase shape */
        idx = (int)farray[2];
        if (idx > 0 && seq->isShapesLibraryParsed && idx <= seq->shapesLibrarySize) {
            if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape, 1.0f)) {
                pulseqlib_seqBlockFree(block);
                pulseqlib_seqBlockInit(block);
                return 0;
            }
            block->rf.phaseShape = shape;
            for (i = 0; i < block->rf.phaseShape.numSamples; i++) {
                block->rf.phaseShape.samples[i] *= TWO_PI;
            }
        } else {
            block->rf.phaseShape.numSamples = 0;
            block->rf.phaseShape.numUncompressedSamples = 0;
            block->rf.phaseShape.samples = NULL;
        }

        if (DETECT_REAL_RF && block->rf.magShape.numSamples > 0 && block->rf.phaseShape.numSamples > 0) {
            isRealSample = (int*) ALLOC(block->rf.magShape.numSamples * sizeof(int));
            if (!isRealSample) {
                pulseqlib_seqBlockFree(block);
                pulseqlib_seqBlockInit(block);
                return 0;
            }
            for (i = 0; i < block->rf.magShape.numSamples; i++) {
                isRealSample[i] = fabs(block->rf.phaseShape.samples[i]) < 1e-6 || fabs(block->rf.phaseShape.samples[i] - M_PI) < 1e-6;
            }
            for (i = 0; i < block->rf.magShape.numSamples; i++) {
                if (isRealSample[i]) {
                    numRealSamples++;
                }
            }
            if (numRealSamples == block->rf.magShape.numSamples) {
                for (i = 0; i < block->rf.magShape.numSamples; i++) {
                    if (fabs(block->rf.phaseShape.samples[i] - M_PI) < 1e-6) {
                        block->rf.magShape.samples[i] *= -1;
                    }
                }
                FREE(block->rf.phaseShape.samples);
                block->rf.phaseShape.numSamples = 0;
                block->rf.phaseShape.numUncompressedSamples = 0;
                block->rf.phaseShape.samples = NULL;
            }
            FREE(isRealSample);
            isRealSample = NULL;
        }

        /* Time shape */
        idx = (int)farray[3];
        if (idx > 0 && seq->isShapesLibraryParsed && idx <= seq->shapesLibrarySize) {
            if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape, rfRasterTime_us)) {
                pulseqlib_seqBlockFree(block);
                pulseqlib_seqBlockInit(block);
                return 0;
            }
            block->rf.timeShape = shape;
        } else {
            block->rf.timeShape.numSamples = 0;
            block->rf.timeShape.numUncompressedSamples = 0;
            block->rf.timeShape.samples = NULL;
        }

        block->rf.center = farray[4];
        block->rf.delay = (int)farray[5];
        block->rf.freqPPM = farray[6];
        block->rf.phasePPM = farray[7];
        block->rf.freqOffset = farray[8];
        block->rf.phaseOffset = farray[9];
    }

    /* Parse Gx gradient */
    if (raw->gx >= 0 && seq->gradLibrary && raw->gx < seq->gradLibrarySize) {
        farray = seq->gradLibrary[raw->gx];
        block->gx.amplitude = farray[1];
        if ((int)farray[0] == 0) {
            block->gx.type = 1;
            block->gx.trap.riseTime = (long)farray[2];
            block->gx.trap.flatTime = (long)farray[3];
            block->gx.trap.fallTime = (long)farray[4];
            block->gx.delay = (int)farray[5];
            block->gx.first = 0;
            block->gx.last = 0;
        } else if ((int)farray[0] == 1) {
            block->gx.type = 2;
            block->gx.first = farray[2];
            block->gx.last = farray[3];
            idx = (int)farray[4];
            if (idx > 0 && seq->isShapesLibraryParsed && idx <= seq->shapesLibrarySize) {
                if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape, 1.0f)) {
                    pulseqlib_seqBlockFree(block);
                    pulseqlib_seqBlockInit(block);
                    return 0;
                }
                block->gx.waveShape = shape;
            }
            idx = (int)farray[5];
            if (idx > 0 && seq->isShapesLibraryParsed && idx <= seq->shapesLibrarySize) {
                if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape, gradRasterTime_us)) {
                    pulseqlib_seqBlockFree(block);
                    pulseqlib_seqBlockInit(block);
                    return 0;
                }
                block->gx.timeShape = shape;
            } else {
                block->gx.timeShape.numSamples = 0;
                block->gx.timeShape.numUncompressedSamples = 0;
                block->gx.timeShape.samples = NULL;
            }
            block->gx.delay = (int)farray[6];
        }
    }

    /* Parse Gy gradient */
    if (raw->gy >= 0 && seq->gradLibrary && raw->gy < seq->gradLibrarySize) {
        farray = seq->gradLibrary[raw->gy];
        block->gy.amplitude = farray[1];
        if ((int)farray[0] == 0) {
            block->gy.type = 1;
            block->gy.trap.riseTime = (long)farray[2];
            block->gy.trap.flatTime = (long)farray[3];
            block->gy.trap.fallTime = (long)farray[4];
            block->gy.delay = (int)farray[5];
            block->gy.first = 0;
            block->gy.last = 0;
        } else if ((int)farray[0] == 1) {
            block->gy.type = 2;
            block->gy.first = farray[2];
            block->gy.last = farray[3];
            idx = (int)farray[4];
            if (idx > 0 && seq->isShapesLibraryParsed && idx <= seq->shapesLibrarySize) {
                if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape, 1.0f)) {
                    pulseqlib_seqBlockFree(block);
                    pulseqlib_seqBlockInit(block);
                    return 0;
                }
                block->gy.waveShape = shape;
            }
            idx = (int)farray[5];
            if (idx > 0 && seq->isShapesLibraryParsed && idx <= seq->shapesLibrarySize) {
                if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape, gradRasterTime_us)) {
                    pulseqlib_seqBlockFree(block);
                    pulseqlib_seqBlockInit(block);
                    return 0;
                }
                block->gy.timeShape = shape;
            } else {
                block->gy.timeShape.numSamples = 0;
                block->gy.timeShape.numUncompressedSamples = 0;
                block->gy.timeShape.samples = NULL;
            }
            block->gy.delay = (int)farray[6];
        }
    }

    /* Parse Gz gradient */
    if (raw->gz >= 0 && seq->gradLibrary && raw->gz < seq->gradLibrarySize) {
        farray = seq->gradLibrary[raw->gz];
        block->gz.amplitude = farray[1];
        if ((int)farray[0] == 0) {
            block->gz.type = 1;
            block->gz.trap.riseTime = (long)farray[2];
            block->gz.trap.flatTime = (long)farray[3];
            block->gz.trap.fallTime = (long)farray[4];
            block->gz.delay = (int)farray[5];
            block->gz.first = 0;
            block->gz.last = 0;
        } else if ((int)farray[0] == 1) {
            block->gz.type = 2;
            block->gz.first = farray[2];
            block->gz.last = farray[3];
            idx = (int)farray[4];
            if (idx > 0 && seq->isShapesLibraryParsed && idx <= seq->shapesLibrarySize) {
                if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape, 1.0f)) {
                    pulseqlib_seqBlockFree(block);
                    pulseqlib_seqBlockInit(block);
                    return 0;
                }
                block->gz.waveShape = shape;
            }
            idx = (int)farray[5];
            if (idx > 0 && seq->isShapesLibraryParsed && idx <= seq->shapesLibrarySize) {
                if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape, gradRasterTime_us)) {
                    pulseqlib_seqBlockFree(block);
                    pulseqlib_seqBlockInit(block);
                    return 0;
                }
                block->gz.timeShape = shape;
            } else {
                block->gz.timeShape.numSamples = 0;
                block->gz.timeShape.numUncompressedSamples = 0;
                block->gz.timeShape.samples = NULL;
            }
            block->gz.delay = (int)farray[6];
        }
    }

    /* Parse ADC event */
    if (raw->adc >= 0 && seq->adcLibrary && raw->adc < seq->adcLibrarySize) {
        farray = seq->adcLibrary[raw->adc];
        block->adc.type = 1;
        block->adc.numSamples = (int)farray[0];
        block->adc.dwellTime = (int)farray[1];
        block->adc.delay = (int)farray[2];
        block->adc.freqPPM = farray[3];
        block->adc.phasePPM = farray[4];
        block->adc.freqOffset = farray[5];
        block->adc.phaseOffset = farray[6];
        idx = (int)farray[7];
        if (idx > 0 && seq->isShapesLibraryParsed && idx <= seq->shapesLibrarySize) {
            if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape, farray[1] * 1e-3f)) {
                pulseqlib_seqBlockFree(block);
                pulseqlib_seqBlockInit(block);
                return 0;
            }
            block->adc.phaseModulationShape = shape;
        } else {
            block->adc.phaseModulationShape.numSamples = 0;
            block->adc.phaseModulationShape.numUncompressedSamples = 0;
            block->adc.phaseModulationShape.samples = NULL;
        }
    }

    /* Parse trigger and soft delay from rawExt if provided */
    if (rawExt) {
        if (rawExt->triggerIndex >= 0 && seq->triggerLibrary) {
            trig = seq->triggerLibrary[rawExt->triggerIndex];
            block->trigger.type = 1;
            block->trigger.duration = (long)trig[3];
            block->trigger.delay = (long)trig[2];
            block->trigger.triggerType = (int)trig[0];
            block->trigger.triggerChannel = (int)trig[1];
        }
        
        if (rawExt->softDelayIndex >= 0 && seq->softDelayLibrary) {
            delay = seq->softDelayLibrary[rawExt->softDelayIndex];
            block->delay.type = 1;
            block->delay.numID = (int)delay[0];
            block->delay.offset = (int)delay[1];
            block->delay.factor = (int)delay[2];
            block->delay.hintID = (int)delay[3];
        }
    }

    return 1;
}


/**
 * @brief Retrieves a block from the sequence file (refactored version).
 *
 * @param[in] seq Pointer to the SeqFile structure.
 * @param[in, out] block Pointer to a pre-allocated SeqBlock to fill.
 * @param[in] blockIndex Index of the block to retrieve.
 */
void pulseqlib_getBlock(const pulseqlib_SeqFile* seq, pulseqlib_SeqBlock* block, const int blockIndex) {
    pulseqlib_RawBlock rawBlock;
    pulseqlib_RawExtension rawExt;
    pulseqlib_ExtensionBlock extBlock;
    int hasExtensions;
    
    /* Check inputs */
    if (!seq || !block || blockIndex < 0 || blockIndex >= seq->numBlocks) {
        return; /* Invalid inputs */
    }

    /* Step 1: Get raw block content IDs (with extensions) */
    if (!getRawBlockContentIDs(seq, &rawBlock, blockIndex, 1)) {
        return;
    }
    
    /* Check if we have extensions to parse */
    hasExtensions = (rawBlock.extCount > 0) && seq->isExtensionsLibraryParsed && seq->extensionLUT;
    
    /* Step 2: Extract raw extension data (if needed) */
    if (hasExtensions) {
        getRawExtension(seq, &rawExt, &rawBlock);
    } else {
        rawExtensionInit(&rawExt);
    }
    
    /* Step 3: Parse main block content (RF, grads, ADC, trigger, delay) */
    if (!parseBlockWithoutExt(seq, block, &rawBlock, hasExtensions ? &rawExt : NULL)) {
        return;
    }
    
    /* Step 4: Parse and apply extensions (if needed) */
    if (hasExtensions) {
        if (parseExtension(seq, &extBlock, &rawExt)) {
            applyExtension(&extBlock, block);
            extensionBlockFree(&extBlock);
        }
    }
}

float pulseqlib_getGradLibraryMaxAmplitude(const pulseqlib_SeqFile* seq) {
    float maxAmplitude;
    int i;

    maxAmplitude = 0.0f;

    if (!seq || !seq->isGradLibraryParsed || !seq->gradLibrary || seq->gradLibrarySize <= 0) {
        return maxAmplitude;
    }

    for (i = 0; i < seq->gradLibrarySize; ++i) {
        float amplitude = seq->gradLibrary[i][1];
        if (amplitude > maxAmplitude) {
            maxAmplitude = amplitude;
        }
    }

    return maxAmplitude;
}

typedef struct {
    size_t hash;
    int row_index;
    int label;
    char used;
} HashEntry;

static size_t hash_row(const int *row, const int numCols)
{
    size_t h = 2166136261UL;
    int i;

    for (i = 0; i < numCols; ++i) {
        h ^= (size_t)row[i];
        h *= 16777619UL;
    }
    return h;
}

/* Compare two arrays element-wise. Returns 1 if equal, 0 otherwise. */
static int array_equal(const int *a, const int *b, const int len)
{
    int i;
    for (i = 0; i < len; ++i)
        if (a[i] != b[i])
            return 0;
    return 1;
}

static size_t next_pow2(size_t x)
{
    size_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

/* Number of columns for each event type (static parameters only) */
#define RF_DEF_COLS   4  /* mag_id, phase_id, time_id, delay */
#define RF_PARAMS_COLS 3 /* amplitude, freqOffset, phaseOffset */
#define GRAD_DEF_COLS 6  /* type, riseTime/first, flatTime/last, fallTime/waveLen, time_id, delay */
#define ADC_DEF_COLS  2  /* numSamples, dwellTime */
#define ADC_PARAMS_COLS 2 /* freqOffset, phaseOffset */
#define BLOCK_DEF_COLS   5 /* duration, uniqueRF, uniqueGx, uniqueGy, uniqueGz */


/**
 * @brief Core deduplication function for integer row arrays.
 *
 * @param[in, out] uniqueDefs Array of indices of unique rows (size >= numRows).
 * @param[in, out] eventTable Mapping from row index to unique ID (size >= numRows).
 * @param[in]  numRows Number of rows.
 * @param[in]  numCols Number of columns per row.
 * @param[in]  intRows Flattened array of integer rows (numRows * numCols).
 * @return Number of unique rows found.
 */
static int deduplicate_int_rows(int* uniqueDefs, int* eventTable, int numRows, int numCols, const int* intRows)
{
    HashEntry* table;
    size_t table_size, mask, pos, t;
    size_t h;
    int numUnique, r;

    if (numRows <= 0) return 0;

    table_size = next_pow2((size_t)numRows * 2);
    mask = table_size - 1;

    table = (HashEntry*)ALLOC(table_size * sizeof(HashEntry));
    if (!table) return 0;

    for (t = 0; t < table_size; ++t) {
        table[t].used = 0;
    }

    numUnique = 0;

    for (r = 0; r < numRows; ++r) {
        h = hash_row(&intRows[r * numCols], numCols);
        pos = h & mask;

        while (1) {
            if (!table[pos].used) {
                table[pos].used = 1;
                table[pos].hash = h;
                table[pos].row_index = r;
                table[pos].label = numUnique;

                uniqueDefs[numUnique] = r;
                eventTable[r] = numUnique;
                numUnique++;
                break;
            }

            if (table[pos].hash == h &&
                array_equal(&intRows[r * numCols],
                            &intRows[table[pos].row_index * numCols],
                            numCols)) {
                eventTable[r] = table[pos].label;
                break;
            }

            pos = (pos + 1) & mask;
        }
    }

    FREE(table);
    return numUnique;
}


/**
 * @brief Build RF static parameter row for deduplication.
 */
static void build_rf_def_row(const pulseqlib_SeqFile* seq, int rfIdx, int* row, float* params)
{
    float gamma = seq->opts.gamma;
    float B0 = seq->opts.B0;
    float* rf = seq->rfLibrary[rfIdx];
    float ppm_to_hz = 1e-6 * gamma * B0;
    row[0] = (int)rf[1]; /* mag_id */
    row[1] = (int)rf[2]; /* phase_id */
    row[2] = (int)rf[3]; /* time_id */
    row[3] = (int)rf[5]; /* delay */
    params[0] = rf[0];   /* amplitude */
    params[1] = rf[8] + ppm_to_hz * rf[6];   /* freqPPM */
    params[2] = rf[9] + ppm_to_hz * rf[7];   /* phasePPM */
}

/**
 * @brief Deduplicate RF library by static parameters only.
 *
 * @param[in]  seq         Pointer to the sequence file.
 * @param[in, out] rfDefinitions  Array of RF definitions (size >= numRows).
 * @param[in, out] rfTable  Mapping from RF index to unique RF ID (size >= numRows).
 * @return Number of unique RF events found.
 */
static int deduplicate_rf_library(const pulseqlib_SeqFile* seq, pulseqlib_RfDefinition* rfDefinitions, pulseqlib_RfTableElement* rfTable)
{
    int (*intRows)[RF_DEF_COLS];
    float (*params)[RF_PARAMS_COLS];
    int* uniqueDefs;
    int* eventTable;
    int numUnique;
    int numRows = seq->rfLibrarySize;
    int i;

    /* Prepare temporary variables */
    if (numRows <= 0) return 0;
    intRows = ALLOC(numRows * sizeof(*intRows));
    if (!intRows) return 0;
    params = ALLOC(numRows * sizeof(*params));
    if (!params) {
        FREE(intRows);
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }
    uniqueDefs = (int*)ALLOC(numRows * sizeof(int));
    if (!uniqueDefs) {
        FREE(intRows);
        FREE(params);
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }
    eventTable = (int*)ALLOC(numRows * sizeof(int));
    if (!eventTable) {
        FREE(intRows);
        FREE(params);
        FREE(uniqueDefs);
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }
    for (i = 0; i < numRows; ++i) {
        build_rf_def_row(seq, i, intRows[i], params[i]);
    }

    /* Deduplicate */
    numUnique = deduplicate_int_rows(uniqueDefs, eventTable, numRows, RF_DEF_COLS, (const int*)intRows);

    /* Copy inside rfDefinitions */
    for (i = 0; i < numUnique; ++i) {
        rfDefinitions[i].ID = uniqueDefs[i];
        rfDefinitions[i].magShapeID = (int)intRows[uniqueDefs[i]][0];
        rfDefinitions[i].phaseShapeID = (int)intRows[uniqueDefs[i]][1];
        rfDefinitions[i].timeShapeID = (int)intRows[uniqueDefs[i]][2];
        rfDefinitions[i].delay = (int)intRows[uniqueDefs[i]][3];
    }

    /* Copy inside rfTable */
    for (i = 0; i < numRows; ++i) {
        rfTable[i].ID = eventTable[i];
        rfTable[i].amplitude = params[i][0];
        rfTable[i].freqOffset = params[i][1];
        rfTable[i].phaseOffset = params[i][2];
    }
    FREE(intRows);
    FREE(params);
    FREE(uniqueDefs);
    FREE(eventTable);

    return numUnique;
}


/**
 * @brief Build Grad static parameter row for deduplication.
 */
static void build_grad_def_row(const pulseqlib_SeqFile* seq, int gradIdx, int* row, float* params)
{
    float* grad = seq->gradLibrary[gradIdx];
    int gradType = (int)grad[0];
    int waveId;

    row[0] = gradType;           /* type */

    if (gradType == 0) {
        /* trapezoid: index 4 is fallTime */
        row[1] = (int)grad[2];   /* riseTime */
        row[2] = (int)grad[3];   /* flatTime */
        row[3] = (int)grad[4];   /* fallTime */
        row[4] = 0;              /* unused */ 
    } else {
        /* arbitrary: index 4 is wave_id, replace with numUncompressedSamples */
        row[1] = 0;              /* unused */
        row[2] = 0;              /* unused */
        waveId = (int)grad[4];
        if (waveId > 0 && seq->isShapesLibraryParsed && waveId <= seq->shapesLibrarySize) {
            row[3] = seq->shapesLibrary[waveId - 1].numUncompressedSamples;
        } else {
            row[3] = 0;
        }
        row[4] = (int)grad[5];       /* time_id */
    }
    row[5] = (int)grad[6];       /* delay */
    *params = grad[1];          /* amplitude */
}


/**
 * @brief Deduplicate Grad library by static parameters only.
 *
 * @param[in]  seq         Pointer to the sequence file.
 * @param[in, out] gradDefinitions  Array of Grad definitions (size >= numRows).
 * @param[in, out] gradTable  Mapping from Grad index to unique Grad ID (size >= numRows).
 * @return Number of unique Grad events found.
 */
static int deduplicate_grad_library(const pulseqlib_SeqFile* seq, pulseqlib_GradDefinition* gradDefinitions, pulseqlib_GradTableElement* gradTable)
{
    int (*intRows)[GRAD_DEF_COLS];
    float* params;
    int* uniqueDefs;
    int* eventTable;
    int numUnique;
    int numRows = seq->gradLibrarySize;
    int i;

    /* Prepare temporary variables */
    if (numRows <= 0) return 0;
    intRows = ALLOC(numRows * sizeof(*intRows));
    if (!intRows) return 0;
    params = (float*)ALLOC(numRows * sizeof(float));
    if (!params) {
        FREE(intRows);
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }
    uniqueDefs = (int*)ALLOC(numRows * sizeof(int));
    if (!uniqueDefs) {
        FREE(intRows);
        FREE(params);
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }
    eventTable = (int*)ALLOC(numRows * sizeof(int));
    if (!eventTable) {
        FREE(intRows);
        FREE(params);
        FREE(uniqueDefs);
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }
    for (i = 0; i < numRows; ++i) {
        build_grad_def_row(seq, i, intRows[i], &params[i]);
    }

    /* Deduplicate */
    numUnique = deduplicate_int_rows(uniqueDefs, eventTable, numRows, GRAD_DEF_COLS, (const int*)intRows);

    /* Copy inside gradDefinitions */
    for (i = 0; i < numUnique; ++i) {
        gradDefinitions[i].ID = uniqueDefs[i];
        gradDefinitions[i].type = (int)intRows[uniqueDefs[i]][0];
        gradDefinitions[i].riseTimeOrUnused = (int)intRows[uniqueDefs[i]][1];
        gradDefinitions[i].flatTimeOrUnused = (int)intRows[uniqueDefs[i]][2];
        gradDefinitions[i].fallTimeOrNumUncompressedSamples = (int)intRows[uniqueDefs[i]][3];
        gradDefinitions[i].unusedOrTimeShapeID = (int)intRows[uniqueDefs[i]][4];
        gradDefinitions[i].delay = (int)intRows[uniqueDefs[i]][5];
    }

    /* Copy inside gradTable */
    for (i = 0; i < numRows; ++i) {
        gradTable[i].ID = eventTable[i];
        gradTable[i].amplitude = params[i];
    }
    FREE(intRows);
    FREE(params);
    FREE(uniqueDefs);
    FREE(eventTable);

    return numUnique;
}


/**
 * @brief Compute shot indices for gradient table entries.
 *
 * For each unique gradient definition, this function identifies distinct waveform
 * shapes (for arbitrary gradients) and assigns a shot index to each gradient
 * library entry. Trapezoids always have numShots=1 and shotIndex=0.
 *
 * @param[in]  seq              Pointer to the sequence file.
 * @param[in]  numUniqueGrads   Number of unique gradient definitions.
 * @param[in,out] gradDefinitions  Array of gradient definitions; numShots and shotShapeIDs are populated.
 * @param[in,out] gradTable     Array of gradient table entries; shotIndex is populated.
 * @return PULSEQLIB_OK on success, or a negative error code on failure.
 */
static int compute_grad_shot_indices(
    const pulseqlib_SeqFile* seq,
    int numUniqueGrads,
    pulseqlib_GradDefinition* gradDefinitions,
    pulseqlib_GradTableElement* gradTable)
{
    int numRows = seq->gradLibrarySize;
    int defIdx, i, j;
    int shapeId;
    int found;
    int shotCount;

    if (numRows <= 0 || numUniqueGrads <= 0) {
        return PULSEQLIB_OK;
    }

    /* Process each unique gradient definition */
    for (defIdx = 0; defIdx < numUniqueGrads; ++defIdx) {
        int gradType = gradDefinitions[defIdx].type;

        /* Initialize shotShapeIDs to zero */
        for (j = 0; j < MAX_GRAD_SHOTS; ++j) {
            gradDefinitions[defIdx].shotShapeIDs[j] = 0;
        }

        /* For trapezoids: single shot, no shape IDs */
        if (gradType == 0) {
            gradDefinitions[defIdx].numShots = 1;

            /* Set shotIndex = 0 for all gradTable entries with this definition */
            for (i = 0; i < numRows; ++i) {
                if (gradTable[i].ID == defIdx) {
                    gradTable[i].shotIndex = 0;
                }
            }
            continue;
        }

        /* For arbitrary gradients: collect unique shape IDs */
        shotCount = 0;

        /* Scan all gradient library entries that map to this definition */
        for (i = 0; i < numRows; ++i) {
            if (gradTable[i].ID != defIdx) {
                continue;
            }

            /* Get the wave_id (shape ID) from gradLibrary[i][4] */
            shapeId = (int)seq->gradLibrary[i][4];

            /* Check if this shapeId is already in our list */
            found = 0;
            for (j = 0; j < shotCount; ++j) {
                if (gradDefinitions[defIdx].shotShapeIDs[j] == shapeId) {
                    found = 1;
                    gradTable[i].shotIndex = j;
                    break;
                }
            }

            /* If not found, add it as a new shot */
            if (!found) {
                if (shotCount >= MAX_GRAD_SHOTS) {
                    return PULSEQLIB_ERR_TOO_MANY_GRAD_SHOTS;
                } else {
                    gradTable[i].shotIndex = shotCount;
                    gradDefinitions[defIdx].shotShapeIDs[shotCount] = shapeId;
                    shotCount++;
                }
            }
        }

        gradDefinitions[defIdx].numShots = shotCount > 0 ? shotCount : 1;
    }

    return PULSEQLIB_OK;
}

/**
 * @brief Normalize a real-valued waveform to unit peak amplitude in place.
 *
 * @param[in,out] waveform   Waveform to normalize.
 * @param[in]     numSamples Number of samples.
 * @return Peak amplitude before normalization.
 */
static float normalize_waveform(float* waveform, int numSamples)
{
    int i;
    float maxAbs;
    
    maxAbs = find_max_abs_real(waveform, numSamples);
    
    if (maxAbs > 1e-9f) {
        for (i = 0; i < numSamples; ++i) {
            waveform[i] /= maxAbs;
        }
    }
    
    return maxAbs;
}

/**
 * @brief Compute statistics for trapezoid gradient.
 *
 * Creates synthetic waveform [0, 1, 1, 0] with times [0, rise, rise+flat, rise+flat+fall].
 * Time units are seconds.
 *
 * @param[in]  riseTime_us  Rise time in microseconds.
 * @param[in]  flatTime_us  Flat time in microseconds.
 * @param[in]  fallTime_us  Fall time in microseconds.
 * @param[out] slewRate     Maximum slew rate (1/s).
 * @param[out] energy       Energy integral (s).
 * @param[out] firstVal     First sample value (always 0).
 * @param[out] lastVal      Last sample value (always 0).
 */
static void compute_trapezoid_stats(
    float riseTime_us,
    float flatTime_us,
    float fallTime_us,
    float* slewRate,
    float* energy,
    float* firstVal,
    float* lastVal
) {
    float riseTime_s = riseTime_us * 1e-6f;
    float flatTime_s = flatTime_us * 1e-6f;
    float fallTime_s = fallTime_us * 1e-6f;
    float slew_rise, slew_fall;
    
    /* First and last values for trapezoid are always zero */
    *firstVal = 0.0f;
    *lastVal = 0.0f;
    
    /* Slew rate: max of rise and fall ramps */
    slew_rise = (riseTime_s > 0.0f) ? (1.0f / riseTime_s) : 0.0f;
    slew_fall = (fallTime_s > 0.0f) ? (1.0f / fallTime_s) : 0.0f;
    *slewRate = (slew_rise > slew_fall) ? slew_rise : slew_fall;
    
    /* Energy: analytical integral of |normalized_waveform|² dt */
    /* For normalized trapezoid with peak = 1:
     *   rise ramp contributes riseTime/3
     *   flat top contributes flatTime
     *   fall ramp contributes fallTime/3
     */
    *energy = riseTime_s / 3.0f + flatTime_s + fallTime_s / 3.0f;
}

/**
 * @brief Compute gradient statistics for all unique gradient definitions.
 *
 * For each gradient definition and each shot, computes:
 * - maxAmplitude: maximum |amplitude| across all instances (Hz/m)
 * - slewRate: maximum |d(normalized_waveform)/dt| in 1/s
 * - energy: integral of |normalized_waveform|^2 dt in s
 * - firstValue: first sample of normalized waveform
 * - lastValue: last sample of normalized waveform
 *
 * @param[in]     seq              Pointer to the sequence file.
 * @param[in,out] gradDefinitions  Array of gradient definitions to populate.
 * @param[in]     numUniqueGrads   Number of unique gradient definitions.
 * @param[in]     gradTable        Array of gradient table elements with amplitude per instance.
 * @param[in]     gradTableSize    Number of entries in the gradient table.
 * @return PULSEQLIB_OK on success, or a negative error code on failure.
 */
static int compute_grad_statistics(
    const pulseqlib_SeqFile* seq,
    pulseqlib_GradDefinition* gradDefinitions,
    int numUniqueGrads,
    const pulseqlib_GradTableElement* gradTable,
    int gradTableSize)
{
    int defIdx, shotIdx, i;
    float gradRasterTime_us;
    pulseqlib_ShapeArbitrary decompressedWave;
    pulseqlib_ShapeArbitrary decompressedTime;
    float* waveform = NULL;
    float* sq_waveform = NULL;
    float* time_us = NULL;
    float riseTime_us, flatTime_us, fallTime_us;
    int numSamples;
    int shapeId, timeId;
    int gradType;
    int hasTimeShape;
    pulseqlib_GradDefinition* gradDef;
    float absAmp;
    
    if (!seq || !gradDefinitions || numUniqueGrads <= 0) {
        return PULSEQLIB_OK;
    }
    
    if (seq->reservedDefinitionsLibrary.gradientRasterTime > 0.0f) {
        gradRasterTime_us = seq->reservedDefinitionsLibrary.gradientRasterTime;
    } else {
        gradRasterTime_us = seq->opts.grad_raster_time;
    }
    
    /* Initialize decompressed shapes */
    decompressedWave.numSamples = 0;
    decompressedWave.numUncompressedSamples = 0;
    decompressedWave.samples = NULL;
    decompressedTime.numSamples = 0;
    decompressedTime.numUncompressedSamples = 0;
    decompressedTime.samples = NULL;
    
    for (defIdx = 0; defIdx < numUniqueGrads; ++defIdx) {
        gradDef = &gradDefinitions[defIdx];
        gradType = gradDef->type;
        
        /* Initialize all stats to zero */
        for (i = 0; i < MAX_GRAD_SHOTS; ++i) {
            gradDef->maxAmplitude[i] = 0.0f;
            gradDef->slewRate[i] = 0.0f;
            gradDef->energy[i] = 0.0f;
            gradDef->firstValue[i] = 0.0f;
            gradDef->lastValue[i] = 0.0f;
        }
        
        /* Compute max amplitude per shot from gradient table */
        if (gradTable && gradTableSize > 0) {
            for (i = 0; i < gradTableSize; ++i) {
                if (gradTable[i].ID == defIdx) {
                    shotIdx = gradTable[i].shotIndex;
                    if (shotIdx >= 0 && shotIdx < MAX_GRAD_SHOTS) {
                        absAmp = gradTable[i].amplitude;
                        if (absAmp < 0.0f) absAmp = -absAmp;
                        if (absAmp > gradDef->maxAmplitude[shotIdx]) {
                            gradDef->maxAmplitude[shotIdx] = absAmp;
                        }
                    }
                }
            }
        }
        
        if (gradType == 0) {
            /* Trapezoid gradient - times already in us */
            riseTime_us = (float)gradDef->riseTimeOrUnused;
            flatTime_us = (float)gradDef->flatTimeOrUnused;
            fallTime_us = (float)gradDef->fallTimeOrNumUncompressedSamples;
            
            compute_trapezoid_stats(
                riseTime_us, flatTime_us, fallTime_us,
                &gradDef->slewRate[0],
                &gradDef->energy[0],
                &gradDef->firstValue[0],
                &gradDef->lastValue[0]);
        } else {
            /* Arbitrary gradient - process each shot */
            timeId = gradDef->unusedOrTimeShapeID;
            
            /* Decompress time shape if present (shared across shots) */
            time_us = NULL;
            hasTimeShape = 0;
            if (timeId > 0 && timeId <= seq->shapesLibrarySize) {
                if (!decompressShape(&seq->shapesLibrary[timeId - 1], &decompressedTime, gradRasterTime_us)) {
                    goto cleanup_error;
                }
                time_us = (float*)ALLOC(decompressedTime.numUncompressedSamples * sizeof(float));
                if (!time_us) {
                    goto cleanup_error;
                }
                for (i = 0; i < decompressedTime.numUncompressedSamples; ++i) {
                    time_us[i] = decompressedTime.samples[i]; /* already in us */
                }
                hasTimeShape = 1;
                FREE(decompressedTime.samples);
                decompressedTime.samples = NULL;
            }
            
            for (shotIdx = 0; shotIdx < gradDef->numShots; ++shotIdx) {
                shapeId = gradDef->shotShapeIDs[shotIdx];
                
                if (shapeId <= 0 || shapeId > seq->shapesLibrarySize) {
                    continue;
                }
                
                /* Decompress waveform shape */
                if (!decompressShape(&seq->shapesLibrary[shapeId - 1], &decompressedWave, 1.0f)) {
                    goto cleanup_error;
                }
                
                numSamples = decompressedWave.numUncompressedSamples;
                
                /* Allocate working arrays */
                waveform = (float*)ALLOC(numSamples * sizeof(float));
                sq_waveform = (float*)ALLOC(numSamples * sizeof(float));
                if (!waveform || !sq_waveform) {
                    goto cleanup_error;
                }
                
                /* Copy and normalize waveform */
                for (i = 0; i < numSamples; ++i) {
                    waveform[i] = decompressedWave.samples[i];
                }
                normalize_waveform(waveform, numSamples);
                
                /* Compute squared waveform for energy */
                for (i = 0; i < numSamples; ++i) {
                    sq_waveform[i] = waveform[i] * waveform[i];
                }
                
                /* First and last values */
                gradDef->firstValue[shotIdx] = waveform[0];
                gradDef->lastValue[shotIdx] = waveform[numSamples - 1];
                
                /* Compute slew rate and energy (all in us) */
                if (hasTimeShape && time_us) {
                    gradDef->slewRate[shotIdx] = max_slew_real_nonuniform(waveform, time_us, numSamples);
                    gradDef->energy[shotIdx] = trapz_real_nonuniform(sq_waveform, time_us, numSamples);
                } else {
                    gradDef->slewRate[shotIdx] = max_slew_real_uniform(waveform, numSamples, gradRasterTime_us);
                    gradDef->energy[shotIdx] = trapz_real_uniform(sq_waveform, numSamples, gradRasterTime_us);
                }

                /* Convert to SI units: slew from 1/us to 1/s, energy from us to s */
                gradDef->slewRate[shotIdx] *= 1e6f;   /* 1/us -> 1/s */
                gradDef->energy[shotIdx] *= 1e-6f;   /* us -> s */
                
                FREE(waveform);
                waveform = NULL;
                FREE(sq_waveform);
                sq_waveform = NULL;
                
                /* Free decompressed waveform */
                FREE(decompressedWave.samples);
                decompressedWave.samples = NULL;
            }
            
            /* Free time shape resources */
            if (time_us) {
                FREE(time_us);
                time_us = NULL;
            }
        }
    }
    
    return PULSEQLIB_OK;

cleanup_error:
    if (waveform) FREE(waveform);
    waveform = NULL;
    if (sq_waveform) FREE(sq_waveform);
    sq_waveform = NULL;
    if (time_us) FREE(time_us);
    time_us = NULL;
    if (decompressedWave.samples) FREE(decompressedWave.samples);
    decompressedWave.samples = NULL;
    if (decompressedTime.samples) FREE(decompressedTime.samples);
    decompressedTime.samples = NULL;
    return PULSEQLIB_ERR_ALLOC_FAILED;
}

/**
 * @brief Compute RF statistics for deduplicated RF definitions (GE-compatible).
 *
 * Workflow:
 * 1. Decompress magnitude, phase, time shapes
 * 2. Detect real-valued RF (phase = 0 or π) and convert to signed magnitude
 * 3. Normalize to max|magnitude| = 1
 * 4. Interpolate to uniform FFT grid (centered at peak)
 * 5. Compute stats on uniform grid
 * 6. Compute bandwidth via FFT
 */
static int compute_rf_statistics(
    const pulseqlib_SeqFile* seq,
    pulseqlib_RfDefinition* rfDefinitions,
    int numUniqueRFs,
    const pulseqlib_RfTableElement* rfTable,
    int rfTableSize)
{
    int defIdx, i;
    pulseqlib_ShapeArbitrary decompMag;
    pulseqlib_ShapeArbitrary decompPhase;
    pulseqlib_ShapeArbitrary decompTime;
    float* magnitude = NULL;
    float* phase = NULL;
    float* time_us = NULL;
    float* time_us_uniform = NULL;
    float* rf_re = NULL;
    float* rf_im = NULL;
    float* rf_re_uniform = NULL;
    float* rf_im_uniform = NULL;
    float* time_centered = NULL;
    int numSamples, numUniformSamples, numRealSamples;
    int magId, phaseId, timeId;
    int result;
    int hasPhase, hasTime;
    int first;
    int last;
    float maxMag;
    float duration;
    float time_center;
    float rfRasterTime_us;
    pulseqlib_RfDefinition* rfDef;
    
    /* Thresholds */
    const float DTY_THRESHOLD = 0.2236f;  /* sqrt(0.05) for 5% power */
    const float MPW_THRESHOLD = 1e-5f;
    
    /* FFT grid parameters */
    int nn;
    float dw = 10.0f;       /* Frequency resolution (Hz) */
    float cutoff = 0.5f;    /* Bandwidth cutoff level */
    
    /* FFT resources (allocated once, reused for all RFs) */
    kiss_fft_cfg fft_cfg = NULL;
    float* tt = NULL;           /* Centered time grid for FFT */
    float* w = NULL;            /* Frequency grid */
    float* rfs_re = NULL;       /* Interpolated RF real part on FFT grid */
    float* rfs_im = NULL;       /* Interpolated RF imag part on FFT grid */
    float* work_re = NULL;      /* Work array for FFT */
    float* work_im = NULL;      /* Work array for FFT */
    kiss_fft_cpx* fft_in = NULL;
    kiss_fft_cpx* fft_out = NULL;
    int fft_ready = 0;
    
    /* Stats computation variables */
    float rf_abs;
    float sum_signed; 
    float sum_signed_re; 
    float sum_signed_im; 
    float sum_abs; 
    float sum_sq; 
    float time_above_threshold;
    float temp_pw;
    float maxpw;
    int fft_start, fft_end, fft_count;
    
    if (!seq || !rfDefinitions || numUniqueRFs <= 0) {
        return PULSEQLIB_OK;
    }

    if (seq->reservedDefinitionsLibrary.radiofrequencyRasterTime > 0.0f) {
        rfRasterTime_us = seq->reservedDefinitionsLibrary.radiofrequencyRasterTime;
    } else {
        rfRasterTime_us = seq->opts.rf_raster_time;
    }
        
    /* Compute FFT size: nn = 1 / (dw * dt) */
    nn = (int)(1.0f / (dw * rfRasterTime_us * 1e-6f));
    nn = kiss_fft_next_fast_size(nn);
    if (nn < 2) nn = 2;
       
    /* Allocate FFT resources once */
    tt = (float*)ALLOC(nn * sizeof(float));
    w = (float*)ALLOC(nn * sizeof(float));
    rfs_re = (float*)ALLOC(nn * sizeof(float));
    rfs_im = (float*)ALLOC(nn * sizeof(float));
    work_re = (float*)ALLOC(nn * sizeof(float));
    work_im = (float*)ALLOC(nn * sizeof(float));
    fft_in = (kiss_fft_cpx*)KISS_FFT_MALLOC(nn * sizeof(kiss_fft_cpx));
    fft_out = (kiss_fft_cpx*)KISS_FFT_MALLOC(nn * sizeof(kiss_fft_cpx));
    fft_cfg = kiss_fft_alloc(nn, 0, NULL, NULL);
    if (tt && w && rfs_re && rfs_im && work_re && work_im && fft_in && fft_out && fft_cfg) {
        for (i = 0; i < nn; ++i) {
            tt[i] = (float)(i - nn / 2) * rfRasterTime_us; /* in us */
            w[i] = (float)(i - nn / 2) * dw;
        }
        fft_ready = 1;
    } else {
        fft_ready = 0;
    }
    if (!fft_ready) {
        goto cleanup_error;
    }
    
    /* Initialize decompressed shape structs */
    decompMag.numSamples = 0;
    decompMag.numUncompressedSamples = 0;
    decompMag.samples = NULL;
    decompPhase.numSamples = 0;
    decompPhase.numUncompressedSamples = 0;
    decompPhase.samples = NULL;
    decompTime.numSamples = 0;
    decompTime.numUncompressedSamples = 0;
    decompTime.samples = NULL;
    
    /* Process each unique RF definition */
    for (defIdx = 0; defIdx < numUniqueRFs; ++defIdx) {
        rfDef = &rfDefinitions[defIdx];

        /* Reset per-RF variables */
        first = -1;
        last = -1;
        
        /* Initialize stats to zero/default */
        rfDef->numSamples = 0;
        rfDef->flipAngle = 0.0f;
        rfDef->maxAmplitude = 0.0f;
        rfDef->area = 0.0f;
        rfDef->abswidth = 0.0f;
        rfDef->effwidth = 0.0f;
        rfDef->dtycyc = 0.0f;
        rfDef->maxpw = 0.0f;
        rfDef->duration_us = 0.0f;
        rfDef->isodelay_us = 0;
        rfDef->bandwidth = 0.0f;
        
        /* Find max amplitude across all instances of this RF definition */
        if (rfTable && rfTableSize > 0) {
            for (i = 0; i < rfTableSize; ++i) {
                if (rfTable[i].ID == defIdx) {
                    float amp = (float)fabs(rfTable[i].amplitude);
                    if (amp > rfDef->maxAmplitude) {
                        rfDef->maxAmplitude = amp;
                    }
                }
            }
        }
        
        magId = rfDef->magShapeID;
        phaseId = rfDef->phaseShapeID;
        timeId = rfDef->timeShapeID;
        
        hasPhase = 0;
        hasTime = 0;
        magnitude = NULL;
        phase = NULL;
        time_us = NULL;
        rf_re = NULL;
        rf_im = NULL;
        time_centered = NULL;
        numSamples = 0;
        duration = 0.0f;
        
        /* ================================================================
         * Step 1: Decompress shapes
         * ================================================================ */
        /* Magnitude shape (required) */
        if (!decompressShape(&(seq->shapesLibrary[magId - 1]), &decompMag, 1.0f)) {
            goto cleanup_error;
        }
        numSamples = decompMag.numUncompressedSamples;
        magnitude = (float*)ALLOC(numSamples * sizeof(float));
        if (!magnitude) {
            FREE(decompMag.samples);
            goto cleanup_error;
        }
        for (i = 0; i < numSamples; ++i) {
            magnitude[i] = decompMag.samples[i];
        }
        FREE(decompMag.samples);
        decompMag.samples = NULL;
  
        /* Store number of samples */
        rfDef->numSamples = numSamples;
        
        /* Phase shape (optional) */
        if (phaseId > 0 && phaseId <= seq->shapesLibrarySize) {
            if (!decompressShape(&(seq->shapesLibrary[phaseId - 1]), &decompPhase, 1.0f)) 
            {
                goto cleanup_error;
            }
            phase = (float*)ALLOC(numSamples * sizeof(float));
            if (!phase) {
                FREE(decompPhase.samples);
                goto cleanup_error;
            }
            for (i = 0; i < numSamples; ++i) {
                phase[i] = decompPhase.samples[i];
            }
            hasPhase = 1;
            FREE(decompPhase.samples);
            decompPhase.samples = NULL;
        }
        
        /* ================================================================
         * Step 2: Detect real-valued RF and convert to signed magnitude
         * ================================================================ */
        if (hasPhase && phase) {
            numRealSamples = 0;
            
            for (i = 0; i < numSamples; ++i) {
                if ((float)fabs(phase[i]) < 1e-6f || 
                    (float)fabs(phase[i] - (float)M_PI) < 1e-6f) {
                    ++numRealSamples;
                }
            }
            
            if (numRealSamples == numSamples) {
                for (i = 0; i < numSamples; ++i) {
                    if ((float)fabs(phase[i] - (float)M_PI) < 1e-6f) {
                        magnitude[i] *= -1.0f;
                    }
                }
                FREE(phase);
                phase = NULL;
                hasPhase = 0;
            }
        }
        
        /* Time shape (optional) */
        if (timeId > 0 && timeId <= seq->shapesLibrarySize) {
            if(!decompressShape(&seq->shapesLibrary[timeId - 1], &decompTime, rfRasterTime_us))
            {
                goto cleanup_error;
            }
            time_us = (float*)ALLOC(numSamples * sizeof(float));
            if (!time_us) {
                FREE(decompTime.samples);
                goto cleanup_error;
            }
                for (i = 0; i < numSamples; ++i) {
                    time_us[i] = decompTime.samples[i];
                }
                hasTime = 1;
            FREE(decompTime.samples);
            decompTime.samples = NULL;
        }
        
        /* Build uniform time array if no time shape */
        if (!hasTime) {
            time_us = (float*)ALLOC(numSamples * sizeof(float));
            if (!time_us) {
                goto cleanup_error;
            }
            for (i = 0; i < numSamples; ++i) {
                time_us[i] = (float)i * rfRasterTime_us;
            }
            hasTime = 1;
        }
        
        /* Compute pulse duration */
        duration = (hasTime && numSamples > 0) ? time_us[numSamples - 1] : (numSamples * rfRasterTime_us);
        rfDef->duration_us = duration;
        
        /* ================================================================
         * Step 3: Normalize to max|magnitude| = 1
         * ================================================================ */
        maxMag = find_max_abs_real(magnitude, numSamples);
        for (i = 0; i < numSamples; ++i) {
            if (fabs(magnitude[i]) >= 0.99999f * maxMag) {
                if (first < 0) {
                    first = i;
                }
                last = i;
            }
        }

        /* Safety fallback (should not happen unless maxMag ~ 0) */
        if (first < 0) {
            first = last = 0; /* or 0 */
        }

        /* Compute time_center */
        if (hasTime && time_us) {
            time_center = 0.5f * (time_us[first] + time_us[last]);
        } else {
            time_center = 0.5f * ((float)(first + last)) * rfRasterTime_us;
        }

        /* Compute isodelay: time from peak to end */
        rfDef->isodelay_us = (int)((duration - time_center));

        /* Normalize waveform */
        if (maxMag > 1e-9f) {
            for (i = 0; i < numSamples; ++i) {
                magnitude[i] /= maxMag;
            }
        }
        
        /* ================================================================
         * Step 4: Compute stats on original RF samples (not interpolated)
         * ================================================================ */
        rf_re = (float*)ALLOC(numSamples * sizeof(float));
        rf_im = (float*)ALLOC(numSamples * sizeof(float));
        if (!rf_re || !rf_im) {
            goto cleanup_error;
        }
        
        /* Build complex RF signal */
        if (hasPhase && phase) {
            for (i = 0; i < numSamples; ++i) {
                rf_re[i] = magnitude[i] * (float)cos(phase[i]);
                rf_im[i] = magnitude[i] * (float)sin(phase[i]);
            }
        } else {
            for (i = 0; i < numSamples; ++i) {
                rf_re[i] = magnitude[i];
                rf_im[i] = 0.0f;
            }
        }

        /* Compute uniform grid parameters (all in us) */
        numUniformSamples = (int)(duration / rfRasterTime_us + 0.5f) + 1;
        if (numUniformSamples < 2) numUniformSamples = 2;
        
        /* Allocate uniform grid arrays */
        time_us_uniform = (float*)ALLOC(numUniformSamples * sizeof(float));
        rf_re_uniform = (float*)ALLOC(numUniformSamples * sizeof(float));
        rf_im_uniform = (float*)ALLOC(numUniformSamples * sizeof(float));
        if (!time_us_uniform || !rf_re_uniform || !rf_im_uniform) {
                goto cleanup_error;
        }

        /* Build uniform time grid (NOT centered), in us */
        if (time_us_uniform && rf_re_uniform && rf_im_uniform && time_us) {
            for (i = 0; i < numUniformSamples; ++i) {
                time_us_uniform[i] = (float)i * rfRasterTime_us;
            }
        } 

        /* Interpolate complex RF to uniform grid */
        interp1_linear_complex(time_us_uniform, numUniformSamples, time_us, rf_re, rf_im, numSamples, rf_re_uniform, rf_im_uniform);
            
        /* Compute stats on original samples using RF raster time */
        sum_signed_re = 0.0f;
        sum_signed_im = 0.0f;
        sum_abs = 0.0f;
        sum_sq = 0.0f;
        time_above_threshold = 0.0f;
        maxpw = 0.0f;
        temp_pw = 0.0f;
            
        for (i = 0; i < numUniformSamples; ++i) {                
            sum_signed_re += rf_re_uniform[i];
            sum_signed_im += rf_im_uniform[i];
            rf_abs = (float)sqrt(rf_re_uniform[i] * rf_re_uniform[i] + rf_im_uniform[i] * rf_im_uniform[i]);
            sum_abs += rf_abs;
            sum_sq += rf_abs * rf_abs;
            
            if (rf_abs > DTY_THRESHOLD) {
                time_above_threshold += 1.0f;
            }   
            if (rf_abs >= MPW_THRESHOLD) {
                temp_pw += 1.0f;
            } else {
                if (temp_pw > maxpw) {
                    maxpw = temp_pw;
                }
                temp_pw = 0.0f;
            }
        }
        if (temp_pw > maxpw) {
            maxpw = temp_pw;
        }

        /* Store unnormalized integral for flip angle */
        sum_signed = (float)sqrt(sum_signed_re * sum_signed_re + sum_signed_im * sum_signed_im) * seq->opts.rf_raster_time * 1e-6f; /* convert to seconds */
        
        /* Normalize by duration for width metrics */
        rfDef->area = sum_signed;
        rfDef->abswidth = sum_abs / numUniformSamples;
        rfDef->effwidth = sum_sq / numUniformSamples;
        rfDef->dtycyc = time_above_threshold / numUniformSamples;
        
        /* Flip angle */
        rfDef->flipAngle = (float)(TWO_PI) * rfDef->maxAmplitude * sum_signed;
        rfDef->maxpw = maxpw / numUniformSamples;
        if (rfDef->dtycyc < rfDef->maxpw) {
            rfDef->dtycyc = rfDef->maxpw;
        }

        if (time_us_uniform) FREE(time_us_uniform);
        if (rf_re_uniform) FREE(rf_re_uniform);
        if (rf_im_uniform) FREE(rf_im_uniform);
        time_us_uniform = NULL;
        rf_re_uniform = NULL;
        rf_im_uniform = NULL;
            
        /* ================================================================
            * Step 5: Interpolate to FFT grid and compute bandwidth (only)
            * ================================================================ */
        if (fft_ready && time_us) {
            /* Center time array at peak */
            time_centered = (float*)ALLOC(numSamples * sizeof(float));
            if (time_centered) {
                for (i = 0; i < numSamples; ++i) {
                    time_centered[i] = (time_us[i] - time_center);
                }
                
                /* Interpolate to uniform FFT grid */
                interp1_linear_complex(tt, nn, time_centered, rf_re, rf_im, numSamples, rfs_re, rfs_im);
                
                /* Compute bandwidth via FFT */
                rfDef->bandwidth = compute_rf_bandwidth_fft(
                    rfs_re, rfs_im, fft_cfg, nn, dw, cutoff, duration * 1e-6f, /* convert to seconds */
                    w, work_re, work_im, fft_in, fft_out);
                
                FREE(time_centered);
                time_centered = NULL;
            }
        }
        if (rf_re) FREE(rf_re);
        if (rf_im) FREE(rf_im);
 
        /* Cleanup per-RF arrays */
        if (magnitude) FREE(magnitude);
        if (phase) FREE(phase);
        if (time_us) FREE(time_us);
    }
    
    /* Cleanup FFT resources */
    if (tt) FREE(tt);
    if (w) FREE(w);
    if (rfs_re) FREE(rfs_re);
    if (rfs_im) FREE(rfs_im);
    if (work_re) FREE(work_re);
    if (work_im) FREE(work_im);
    if (fft_in) KISS_FFT_FREE(fft_in);
    if (fft_out) KISS_FFT_FREE(fft_out);
    if (fft_cfg) kiss_fft_free(fft_cfg);
    
    return PULSEQLIB_OK;

cleanup_error:
    /* Free FFT resources */
    if (tt) FREE(tt);
    if (w) FREE(w);
    if (rfs_re) FREE(rfs_re);
    if (rfs_im) FREE(rfs_im);
    if (work_re) FREE(work_re);
    if (work_im) FREE(work_im);
    if (fft_in) KISS_FFT_FREE(fft_in);
    if (fft_out) KISS_FFT_FREE(fft_out);
    if (fft_cfg) kiss_fft_free(fft_cfg);
    /* Free per-RF resources */
    if (magnitude) FREE(magnitude);
    if (phase) FREE(phase);
    if (time_us) FREE(time_us);
    if (rf_re) FREE(rf_re);
    if (rf_im) FREE(rf_im);
    if (time_us_uniform) FREE(time_us_uniform);
    if (rf_re_uniform) FREE(rf_re_uniform);
    if (rf_im_uniform) FREE(rf_im_uniform);
    if (time_centered) FREE(time_centered);
    return PULSEQLIB_ERR_ALLOC_FAILED;
}


/**
 * @brief Build ADC static parameter row for deduplication.
 */
static void build_adc_def_row(const pulseqlib_SeqFile* seq, int adcIdx, int* row, float* params)
{
    float gamma = seq->opts.gamma;
    float B0 = seq->opts.B0;
    float* adc = seq->adcLibrary[adcIdx];
    float ppm_to_hz = 1e-6 * gamma * B0;
    row[0] = (int)adc[0]; /* numSamples */
    row[1] = (int)adc[1]; /* dwellTime_ns */
    row[2] = (int)adc[2]; /* delay */
    params[0] = adc[5] + ppm_to_hz * adc[3];   /* freqPPM */
    params[1] = adc[6] + ppm_to_hz * adc[4];   /* phasePPM */
}


/**
 * @brief Deduplicate ADC library by static parameters only.
 *
 * @param[in]  seq         Pointer to the sequence file.
 * @param[out] adcDefinitions  Array of indices of unique ADC events (size >= numRows).
 * @param[out] adcTable  Mapping from ADC index to unique ADC ID (size >= numRows).
 * @return Number of unique ADC events found.
 */
static int deduplicate_adc_library(const pulseqlib_SeqFile* seq, pulseqlib_AdcDefinition* adcDefinitions, pulseqlib_AdcTableElement* adcTable)
{
    int (*intRows)[ADC_DEF_COLS];
    float (*params)[ADC_PARAMS_COLS];
    int* uniqueDefs;
    int* eventTable;
    int numUnique;
    int numRows = seq->adcLibrarySize;
    int i;

    /* Prepare temporary variables */
    if (numRows <= 0) return 0;
    intRows = ALLOC(numRows * sizeof(*intRows));
    if (!intRows) return 0;
    params = ALLOC(numRows * sizeof(*params));
    if (!params) {
        FREE(intRows);
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }
    uniqueDefs = (int*)ALLOC(numRows * sizeof(int));
    if (!uniqueDefs) {
        FREE(intRows);
        FREE(params);
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }
    eventTable = (int*)ALLOC(numRows * sizeof(int));
    if (!eventTable) {
        FREE(intRows);
        FREE(params);
        FREE(uniqueDefs);
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }
    for (i = 0; i < numRows; ++i) {
        build_adc_def_row(seq, i, intRows[i], params[i]);
    }

    /* Deduplicate */
    numUnique = deduplicate_int_rows(uniqueDefs, eventTable, numRows, ADC_DEF_COLS, (const int*)intRows);

    /* Copy inside adcDefinitions */
    for (i = 0; i < numUnique; ++i) {
        adcDefinitions[i].ID = uniqueDefs[i];
        adcDefinitions[i].numSamples = (int)intRows[uniqueDefs[i]][0];
        adcDefinitions[i].dwellTime = (int)intRows[uniqueDefs[i]][1];
        adcDefinitions[i].delay = (int)intRows[uniqueDefs[i]][2];
    }

    /* Copy inside adcTable */
    for (i = 0; i < numRows; ++i) {
        adcTable[i].ID = eventTable[i];
        adcTable[i].freqOffset = params[i][0];
        adcTable[i].phaseOffset = params[i][1];
    }
    FREE(intRows);
    FREE(params);
    FREE(uniqueDefs);
    FREE(eventTable);

    return numUnique;
}


/**
 * @brief Check if PMC is enabled in the sequence definitions.
 *
 * @param[in] seq Pointer to the sequence file.
 * @return 1 if PMC is enabled, 0 otherwise.
 */
static int is_pmc_enabled(const pulseqlib_SeqFile* seq)
{
    int i, j;
    
    if (!seq->isDefinitionsLibraryParsed || !seq->definitionsLibrary) {
        return 0;
    }
    
    for (i = 0; i < seq->numDefinitions; ++i) {
        if (strcmp(seq->definitionsLibrary[i].name, "pmcEnabled") == 0) {
            /* Check if value is "True" or "true" or "1" */
            if (seq->definitionsLibrary[i].valueSize > 0 && 
                seq->definitionsLibrary[i].value && 
                seq->definitionsLibrary[i].value[0]) {
                const char* val = seq->definitionsLibrary[i].value[0];
                if (strcmp(val, "True") == 0 || 
                    strcmp(val, "true") == 0 || 
                    strcmp(val, "1") == 0) {
                    return 1;
                }
            }
            break;
        }
    }
    return 0;
}

/**
 * @brief Check if a block definition has spatially selective excitation (RF + gradient).
 *
 * @param[in] blockDef Pointer to the block definition.
 * @return 1 if selective excitation, 0 otherwise.
 */
static int is_selective_excitation_block(const pulseqlib_BlockDefinition* blockDef)
{
    int hasRF, hasGrad;
    
    if (!blockDef) return 0;
    
    hasRF = (blockDef->rfID >= 0);
    hasGrad = (blockDef->gxID >= 0) || (blockDef->gyID >= 0) || (blockDef->gzID >= 0);
    
    return hasRF && hasGrad;
}

/**
 * @brief Copy and convert rotation library from quaternions to matrices.
 *
 * @param[in]  seq      Pointer to the sequence file.
 * @param[out] seqDesc  Pointer to the sequence descriptor.
 * @return PULSEQLIB_OK on success, or a negative error code on failure.
 */
static int copy_rotation_library(const pulseqlib_SeqFile* seq, pulseqlib_SequenceDescriptor* seqDesc)
{
    int i;
    int numRotations = seq->rotationLibrarySize;
    
    seqDesc->numRotations = 0;
    seqDesc->rotationMatrices = NULL;
    
    if (numRotations <= 0 || !seq->rotationQuaternionLibrary) {
        return PULSEQLIB_OK;
    }
    
    seqDesc->rotationMatrices = (float(*)[9])ALLOC(numRotations * sizeof(float[9]));
    if (!seqDesc->rotationMatrices) {
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }
    
    for (i = 0; i < numRotations; ++i) {
        quaternion_to_matrix(seq->rotationQuaternionLibrary[i], seqDesc->rotationMatrices[i]);
    }
    
    seqDesc->numRotations = numRotations;
    return PULSEQLIB_OK;
}

/**
 * @brief Copy trigger library to sequence descriptor.
 *
 * @param[in]  seq      Pointer to the sequence file.
 * @param[out] seqDesc  Pointer to the sequence descriptor.
 * @return PULSEQLIB_OK on success, or a negative error code on failure.
 */
static int copy_trigger_library(const pulseqlib_SeqFile* seq, pulseqlib_SequenceDescriptor* seqDesc)
{
    int i;
    int numTriggers = seq->triggerLibrarySize;
    
    seqDesc->numTriggers = 0;
    seqDesc->triggerEvents = NULL;
    
    if (numTriggers <= 0 || !seq->triggerLibrary) {
        return PULSEQLIB_OK;
    }
    
    seqDesc->triggerEvents = (pulseqlib_TriggerEvent*)ALLOC(numTriggers * sizeof(pulseqlib_TriggerEvent));
    if (!seqDesc->triggerEvents) {
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }
    
    for (i = 0; i < numTriggers; ++i) {
        /* triggerLibrary columns: type, channel, delay, duration */
        seqDesc->triggerEvents[i].type = 1; /* mark as defined */
        seqDesc->triggerEvents[i].triggerType = (int)seq->triggerLibrary[i][0];
        seqDesc->triggerEvents[i].triggerChannel = (int)seq->triggerLibrary[i][1];
        seqDesc->triggerEvents[i].delay = (long)seq->triggerLibrary[i][2];
        seqDesc->triggerEvents[i].duration = (long)seq->triggerLibrary[i][3];
    }
    
    seqDesc->numTriggers = numTriggers;
    return PULSEQLIB_OK;
}

/**
 * @brief Copy and decompress shapes library to sequence descriptor.
 *
 * @param[in]  seq      Pointer to the sequence file.
 * @param[out] seqDesc  Pointer to the sequence descriptor.
 * @return PULSEQLIB_OK on success, or a negative error code on failure.
 */
static int copy_shapes_library(const pulseqlib_SeqFile* seq, pulseqlib_SequenceDescriptor* seqDesc)
{
    int i;
    int numShapes = seq->shapesLibrarySize;
    int numSamples;
    
    seqDesc->numShapes = 0;
    seqDesc->shapes = NULL;
    
    if (numShapes <= 0 || !seq->shapesLibrary) {
        return PULSEQLIB_OK;
    }
    
    seqDesc->shapes = (pulseqlib_ShapeArbitrary*)ALLOC(numShapes * sizeof(pulseqlib_ShapeArbitrary));
    if (!seqDesc->shapes) {
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }
    
    /* Initialize all shapes */
    for (i = 0; i < numShapes; ++i) {
        seqDesc->shapes[i].numSamples = 0;
        seqDesc->shapes[i].numUncompressedSamples = 0;
        seqDesc->shapes[i].samples = NULL;
    }
    
    /* Copy each shape (still compressed) */
    for (i = 0; i < numShapes; ++i) {
        numSamples = seq->shapesLibrary[i].numSamples;
        
        seqDesc->shapes[i].numSamples = numSamples;
        seqDesc->shapes[i].numUncompressedSamples = seq->shapesLibrary[i].numUncompressedSamples;
        seqDesc->shapes[i].samples = NULL;
        
        if (numSamples > 0 && seq->shapesLibrary[i].samples) {
            seqDesc->shapes[i].samples = (float*)ALLOC(numSamples * sizeof(float));
            if (!seqDesc->shapes[i].samples) {
                /* Cleanup on error */
                int j;
                for (j = 0; j < i; ++j) {
                    if (seqDesc->shapes[j].samples) {
                        FREE(seqDesc->shapes[j].samples);
                    }
                }
                FREE(seqDesc->shapes);
                seqDesc->shapes = NULL;
                return PULSEQLIB_ERR_ALLOC_FAILED;
            }
            memcpy(seqDesc->shapes[i].samples, seq->shapesLibrary[i].samples, 
                   numSamples * sizeof(float));
        }
    }
    
    seqDesc->numShapes = numShapes;
    return PULSEQLIB_OK;
}

/**
 * @brief Free all memory allocated in a SequenceDescriptor.
 * 
 * Call this after you're done using the descriptor returned by getUniqueBlocks.
 *
 * @param[in,out] seqDesc  Pointer to the sequence descriptor to free.
 */
void pulseqlib_sequenceDescriptorFree(pulseqlib_SequenceDescriptor* seqDesc)
{
    int i;

    if (!seqDesc) return;

    /* Free block definitions and table */
    if (seqDesc->blockDefinitions) {
        FREE(seqDesc->blockDefinitions);
        seqDesc->blockDefinitions = NULL;
    }
    seqDesc->numUniqueBlocks = 0;

    if (seqDesc->blockTable) {
        FREE(seqDesc->blockTable);
        seqDesc->blockTable = NULL;
    }
    seqDesc->numBlocks = 0;

    /* Free RF definitions and table */
    if (seqDesc->rfDefinitions) {
        FREE(seqDesc->rfDefinitions);
        seqDesc->rfDefinitions = NULL;
    }
    seqDesc->numUniqueRFs = 0;

    if (seqDesc->rfTable) {
        FREE(seqDesc->rfTable);
        seqDesc->rfTable = NULL;
    }
    seqDesc->rfTableSize = 0;

    /* Free gradient definitions and table */
    if (seqDesc->gradDefinitions) {
        FREE(seqDesc->gradDefinitions);
        seqDesc->gradDefinitions = NULL;
    }
    seqDesc->numUniqueGrads = 0;

    if (seqDesc->gradTable) {
        FREE(seqDesc->gradTable);
        seqDesc->gradTable = NULL;
    }
    seqDesc->gradTableSize = 0;

    /* Free ADC definitions and table */
    if (seqDesc->adcDefinitions) {
        FREE(seqDesc->adcDefinitions);
        seqDesc->adcDefinitions = NULL;
    }
    seqDesc->numUniqueADCs = 0;

    if (seqDesc->adcTable) {
        FREE(seqDesc->adcTable);
        seqDesc->adcTable = NULL;
    }
    seqDesc->adcTableSize = 0;

    /* Free rotation matrices */
    if (seqDesc->rotationMatrices) {
        FREE(seqDesc->rotationMatrices);
        seqDesc->rotationMatrices = NULL;
    }
    seqDesc->numRotations = 0;

    /* Free trigger events */
    if (seqDesc->triggerEvents) {
        FREE(seqDesc->triggerEvents);
        seqDesc->triggerEvents = NULL;
    }
    seqDesc->numTriggers = 0;

    /* Free shapes (including sample arrays) */
    if (seqDesc->shapes) {
        for (i = 0; i < seqDesc->numShapes; ++i) {
            if (seqDesc->shapes[i].samples) {
                FREE(seqDesc->shapes[i].samples);
                seqDesc->shapes[i].samples = NULL;
            }
        }
        FREE(seqDesc->shapes);
        seqDesc->shapes = NULL;
    }
    seqDesc->numShapes = 0;

    /* Reset prep/cooldown counts */
    seqDesc->numPrepBlocks = 0;
    seqDesc->numCooldownBlocks = 0;

    /* Free segment definitions */
    if (seqDesc->segmentDefinitions) {
        for (i = 0; i < seqDesc->numUniqueSegments; ++i) {
            if (seqDesc->segmentDefinitions[i].uniqueBlockIndices) {
                FREE(seqDesc->segmentDefinitions[i].uniqueBlockIndices);
            }
            if (seqDesc->segmentDefinitions[i].hasTrigger) {
                FREE(seqDesc->segmentDefinitions[i].hasTrigger);
            }
            if (seqDesc->segmentDefinitions[i].hasRotation) {
                FREE(seqDesc->segmentDefinitions[i].hasRotation);
            }
            if (seqDesc->segmentDefinitions[i].norotFlag) {
                FREE(seqDesc->segmentDefinitions[i].norotFlag);
            }
            if (seqDesc->segmentDefinitions[i].noposFlag) {
                FREE(seqDesc->segmentDefinitions[i].noposFlag);
            }
        }
        FREE(seqDesc->segmentDefinitions);
        seqDesc->segmentDefinitions = NULL;
    }
    seqDesc->numUniqueSegments = 0;
    
    /* Free segment table */
    pulseqlib_segmentTableResultFree(&seqDesc->segmentTable);
}

void pulseqlib_sequenceDescriptorCollectionFree(
    pulseqlib_SequenceDescriptorCollection* descCollection)
{
    int i;
    
    if (!descCollection) return;
    
    if (descCollection->descriptors) {
        for (i = 0; i < descCollection->numSubsequences; ++i) {
            pulseqlib_sequenceDescriptorFree(&descCollection->descriptors[i]);
        }
        FREE(descCollection->descriptors);
    }
    
    if (descCollection->subsequenceInfo) {
        FREE(descCollection->subsequenceInfo);
    }
    
    descCollection->numSubsequences = 0;
    descCollection->descriptors = NULL;
    descCollection->subsequenceInfo = NULL;
    descCollection->totalUniqueSegments = 0;
    descCollection->totalUniqueADCs = 0;
    descCollection->totalBlocks = 0;
    descCollection->totalDuration_us = 0.0f;
}

/**
 * @brief Free temporary arrays used during getUniqueBlocks.
 */
static void free_temp_arrays(
    pulseqlib_RfDefinition* tmpRfDefs,
    pulseqlib_RfTableElement* tmpRfTable,
    pulseqlib_GradDefinition* tmpGradDefs,
    pulseqlib_GradTableElement* tmpGradTable,
    pulseqlib_AdcDefinition* tmpAdcDefs,
    pulseqlib_AdcTableElement* tmpAdcTable,
    pulseqlib_BlockDefinition* tmpBlockDefs,
    pulseqlib_BlockTableElement* tmpBlockTable)
{
    if (tmpRfDefs) FREE(tmpRfDefs);
    if (tmpRfTable) FREE(tmpRfTable);
    if (tmpGradDefs) FREE(tmpGradDefs);
    if (tmpGradTable) FREE(tmpGradTable);
    if (tmpAdcDefs) FREE(tmpAdcDefs);
    if (tmpAdcTable) FREE(tmpAdcTable);
    if (tmpBlockDefs) FREE(tmpBlockDefs);
    if (tmpBlockTable) FREE(tmpBlockTable);
}

/**
 * @brief Get unique blocks from a sequence.
 *
 * Two-step deduplication:
 * 1. Deduplicate event libraries (RF, Grad, ADC) by static parameters
 * 2. Deduplicate blocks using unique event IDs
 *
 * All arrays in seqDesc are allocated internally with exact sizes.
 * The caller must call pulseqlib_sequenceDescriptorFree() when done.
 *
 * @param[in]     seq       Pointer to the sequence file.
 * @param[in,out] seqDesc   Pointer to the sequence descriptor to fill (should be
 *                          initialized with PULSEQLIB_SEQUENCE_DESCRIPTOR_INIT).
 * @return 0 on success, error code otherwise. On error, seqDesc is cleaned up.
 */
int pulseqlib_getUniqueBlocks(const pulseqlib_SeqFile* seq, pulseqlib_SequenceDescriptor* seqDesc) {
    int result;
    int numBlocks;
    int numUniqueRFs;
    int numUniqueGrads;
    int numUniqueADCs;

    /* Loop counters */
    int n;

    /* Temporary max-size arrays for deduplication */
    pulseqlib_RfDefinition* tmpRfDefs = NULL;
    pulseqlib_RfTableElement* tmpRfTable = NULL;
    pulseqlib_GradDefinition* tmpGradDefs = NULL;
    pulseqlib_GradTableElement* tmpGradTable = NULL;
    pulseqlib_AdcDefinition* tmpAdcDefs = NULL;
    pulseqlib_AdcTableElement* tmpAdcTable = NULL;
    pulseqlib_BlockDefinition* tmpBlockDefs = NULL;
    pulseqlib_BlockTableElement* tmpBlockTable = NULL;

    /* Auxiliaries for unique block finder */
    pulseqlib_RawBlock raw;
    int (*intRows)[BLOCK_DEF_COLS] = NULL;
    int *uniqueDefs = NULL;
    int *eventTable = NULL;

    /* Auxiliaries for preparation and cooldown detection */
    pulseqlib_RawExtension ext;
    int norotFlag, noposFlag, onceFlag, pmcFlag, navFlag, onceCounter;
    int hasPrep, hasCooldown;
    int ctrl;

    /* Validate inputs */
    if (!seq || !seqDesc) {
        return PULSEQLIB_ERR_INVALID_ARGUMENT;
    }
    numBlocks = seq->numBlocks;
    if (numBlocks <= 0 || !seq->blockLibrary) {
        return PULSEQLIB_ERR_INVALID_ARGUMENT;
    }

    /* Initialize output counts */
    seqDesc->numPrepBlocks = 0;
    seqDesc->numCooldownBlocks = 0;
    seqDesc->numUniqueRFs = 0;
    seqDesc->numUniqueGrads = 0;
    seqDesc->numUniqueADCs = 0;
    seqDesc->numUniqueBlocks = 0;
    seqDesc->numBlocks = 0;
    seqDesc->rfTableSize = 0;
    seqDesc->gradTableSize = 0;
    seqDesc->adcTableSize = 0;

    /* Store timing rasters: prefer from definitions, fall back to opts */
    if (seq->reservedDefinitionsLibrary.radiofrequencyRasterTime > 0.0f) {
        seqDesc->rfRasterTime_us = seq->reservedDefinitionsLibrary.radiofrequencyRasterTime;
    } else {
        seqDesc->rfRasterTime_us = seq->opts.rf_raster_time;
    }

    if (seq->reservedDefinitionsLibrary.gradientRasterTime > 0.0f) {
        seqDesc->gradRasterTime_us = seq->reservedDefinitionsLibrary.gradientRasterTime;
    } else {
        seqDesc->gradRasterTime_us = seq->opts.grad_raster_time;
    }

    if (seq->reservedDefinitionsLibrary.adcRasterTime > 0.0f) {
        seqDesc->adcRasterTime_us = seq->reservedDefinitionsLibrary.adcRasterTime;
    } else {
        seqDesc->adcRasterTime_us = seq->opts.adc_raster_time;
    }
    
    if (seq->reservedDefinitionsLibrary.blockDurationRaster > 0.0f) {
        seqDesc->blockDurationRaster_us = seq->reservedDefinitionsLibrary.blockDurationRaster;
    } else {
        seqDesc->blockDurationRaster_us = seq->opts.block_duration_raster;
    }

    /* ========== ALLOCATE TEMPORARY MAX-SIZE ARRAYS ========== */
    if (seq->rfLibrarySize > 0) {
        tmpRfDefs = (pulseqlib_RfDefinition*)ALLOC(seq->rfLibrarySize * sizeof(pulseqlib_RfDefinition));
        tmpRfTable = (pulseqlib_RfTableElement*)ALLOC(seq->rfLibrarySize * sizeof(pulseqlib_RfTableElement));
        if (!tmpRfDefs || !tmpRfTable) {
            free_temp_arrays(tmpRfDefs, tmpRfTable, tmpGradDefs, tmpGradTable,
                             tmpAdcDefs, tmpAdcTable, tmpBlockDefs, tmpBlockTable);
            return PULSEQLIB_ERR_ALLOC_FAILED;
        }
    }

    if (seq->gradLibrarySize > 0) {
        tmpGradDefs = (pulseqlib_GradDefinition*)ALLOC(seq->gradLibrarySize * sizeof(pulseqlib_GradDefinition));
        tmpGradTable = (pulseqlib_GradTableElement*)ALLOC(seq->gradLibrarySize * sizeof(pulseqlib_GradTableElement));
        if (!tmpGradDefs || !tmpGradTable) {
            free_temp_arrays(tmpRfDefs, tmpRfTable, tmpGradDefs, tmpGradTable,
                             tmpAdcDefs, tmpAdcTable, tmpBlockDefs, tmpBlockTable);
            return PULSEQLIB_ERR_ALLOC_FAILED;
        }
    }

    if (seq->adcLibrarySize > 0) {
        tmpAdcDefs = (pulseqlib_AdcDefinition*)ALLOC(seq->adcLibrarySize * sizeof(pulseqlib_AdcDefinition));
        tmpAdcTable = (pulseqlib_AdcTableElement*)ALLOC(seq->adcLibrarySize * sizeof(pulseqlib_AdcTableElement));
        if (!tmpAdcDefs || !tmpAdcTable) {
            free_temp_arrays(tmpRfDefs, tmpRfTable, tmpGradDefs, tmpGradTable,
                             tmpAdcDefs, tmpAdcTable, tmpBlockDefs, tmpBlockTable);
            return PULSEQLIB_ERR_ALLOC_FAILED;
        }
    }

    tmpBlockDefs = (pulseqlib_BlockDefinition*)ALLOC(numBlocks * sizeof(pulseqlib_BlockDefinition));
    tmpBlockTable = (pulseqlib_BlockTableElement*)ALLOC(numBlocks * sizeof(pulseqlib_BlockTableElement));
    if (!tmpBlockDefs || !tmpBlockTable) {
        free_temp_arrays(tmpRfDefs, tmpRfTable, tmpGradDefs, tmpGradTable,
                         tmpAdcDefs, tmpAdcTable, tmpBlockDefs, tmpBlockTable);
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }

    /* ========== STEP 1: Deduplicate event libraries into temp arrays ========== */
    if (seq->rfLibrarySize > 0) {
        numUniqueRFs = deduplicate_rf_library(seq, tmpRfDefs, tmpRfTable);
        seqDesc->numUniqueRFs = numUniqueRFs;
        seqDesc->rfTableSize = seq->rfLibrarySize;
#if VENDOR == GEHC
        result = compute_rf_statistics(seq, tmpRfDefs, numUniqueRFs, tmpRfTable, seq->rfLibrarySize);
        if (PULSEQLIB_FAILED(result)) {
            free_temp_arrays(tmpRfDefs, tmpRfTable, tmpGradDefs, tmpGradTable,
                                tmpAdcDefs, tmpAdcTable, tmpBlockDefs, tmpBlockTable);
            return result;
        }
#endif
    }

    if (seq->gradLibrarySize > 0) {
        numUniqueGrads = deduplicate_grad_library(seq, tmpGradDefs, tmpGradTable);
        seqDesc->numUniqueGrads = numUniqueGrads;
        seqDesc->gradTableSize = seq->gradLibrarySize;

        result = compute_grad_shot_indices(seq, numUniqueGrads, tmpGradDefs, tmpGradTable);
        if (PULSEQLIB_FAILED(result)) {
            free_temp_arrays(tmpRfDefs, tmpRfTable, tmpGradDefs, tmpGradTable,
                             tmpAdcDefs, tmpAdcTable, tmpBlockDefs, tmpBlockTable);
            return result;
        }

        result = compute_grad_statistics(seq, tmpGradDefs, numUniqueGrads, tmpGradTable, seq->gradLibrarySize);
        if (PULSEQLIB_FAILED(result)) {
            free_temp_arrays(tmpRfDefs, tmpRfTable, tmpGradDefs, tmpGradTable,
                             tmpAdcDefs, tmpAdcTable, tmpBlockDefs, tmpBlockTable);
            return result;
        }
    }

    if (seq->adcLibrarySize > 0) {
        numUniqueADCs = deduplicate_adc_library(seq, tmpAdcDefs, tmpAdcTable);
        seqDesc->numUniqueADCs = numUniqueADCs;
        seqDesc->adcTableSize = seq->adcLibrarySize;
    }

    /* ========== STEP 2: Build block definition matrix ========== */
    intRows = ALLOC(numBlocks * sizeof(*intRows));
    if (!intRows) {
        free_temp_arrays(tmpRfDefs, tmpRfTable, tmpGradDefs, tmpGradTable,
                         tmpAdcDefs, tmpAdcTable, tmpBlockDefs, tmpBlockTable);
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }

    uniqueDefs = (int*)ALLOC(numBlocks * sizeof(int));
    if (!uniqueDefs) {
        FREE(intRows);
        free_temp_arrays(tmpRfDefs, tmpRfTable, tmpGradDefs, tmpGradTable,
                         tmpAdcDefs, tmpAdcTable, tmpBlockDefs, tmpBlockTable);
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }

    eventTable = (int*)ALLOC(numBlocks * sizeof(int));
    if (!eventTable) {
        FREE(intRows);
        FREE(uniqueDefs);
        free_temp_arrays(tmpRfDefs, tmpRfTable, tmpGradDefs, tmpGradTable,
                         tmpAdcDefs, tmpAdcTable, tmpBlockDefs, tmpBlockTable);
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }

    /* Initialize flags */
    norotFlag = 0;
    noposFlag = 0;
    onceFlag = 0;
    pmcFlag = 1;
    navFlag = 0;
    onceCounter = 0;

    for (n = 0; n < numBlocks; ++n) {
        if (!getRawBlockContentIDs(seq, &raw, n, 1)) {
            FREE(intRows);
            FREE(uniqueDefs);
            FREE(eventTable);
            free_temp_arrays(tmpRfDefs, tmpRfTable, tmpGradDefs, tmpGradTable,
                             tmpAdcDefs, tmpAdcTable, tmpBlockDefs, tmpBlockTable);
            return PULSEQLIB_ERR_INVALID_ARGUMENT;
        }

        /* Map event indices to unique event IDs */
        intRows[n][0] = raw.block_duration >= 0 ? raw.block_duration : 0;
        intRows[n][1] = (raw.rf >= 0 && tmpRfTable) ? tmpRfTable[raw.rf].ID : -1;
        intRows[n][2] = (raw.gx >= 0 && tmpGradTable) ? tmpGradTable[raw.gx].ID : -1;
        intRows[n][3] = (raw.gy >= 0 && tmpGradTable) ? tmpGradTable[raw.gy].ID : -1;
        intRows[n][4] = (raw.gz >= 0 && tmpGradTable) ? tmpGradTable[raw.gz].ID : -1;

        /* Store table ID in block table */
        tmpBlockTable[n].rfID = raw.rf;
        tmpBlockTable[n].gxID = raw.gx;
        tmpBlockTable[n].gyID = raw.gy;
        tmpBlockTable[n].gzID = raw.gz;
        tmpBlockTable[n].adcID = raw.adc;
        
        /* Store pure delay flag */
        tmpBlockTable[n].duration_us = (raw.rf < 0 && raw.gx < 0 && raw.gy < 0 && raw.gz < 0 && raw.adc < 0) ? (int)(raw.block_duration * seqDesc->blockDurationRaster_us) : -1;

        /* Inspect extensions */
        if (raw.extCount > 0 && seq->isExtensionsLibraryParsed && seq->extensionLUT) {
            getRawExtension(seq, &ext, &raw);
            tmpBlockTable[n].rotationID = ext.rotationIndex;
            tmpBlockTable[n].triggerID = ext.triggerIndex;
            norotFlag = (ext.flag.norot >= 0) ? ext.flag.norot : norotFlag;
            noposFlag = (ext.flag.nopos >= 0) ? ext.flag.nopos : noposFlag;
            pmcFlag = (ext.flag.pmc >= 0) ? ext.flag.pmc : pmcFlag;
            navFlag = (ext.flag.nav >= 0) ? ext.flag.nav : navFlag;
            onceFlag = (ext.flag.once >= 0) ? ext.flag.once : onceFlag;
            if (onceFlag > 0) {
                ++onceCounter;
            }
        }
        tmpBlockTable[n].norotFlag = norotFlag;
        tmpBlockTable[n].noposFlag = noposFlag;
        tmpBlockTable[n].pmcFlag = pmcFlag;
        tmpBlockTable[n].onceFlag = onceFlag;
        tmpBlockTable[n].navFlag = navFlag;
    }

    /* Step 3: Deduplicate blocks */
    seqDesc->numUniqueBlocks = deduplicate_int_rows(uniqueDefs, eventTable, numBlocks, BLOCK_DEF_COLS, (const int*)intRows);
    seqDesc->numBlocks = numBlocks;

    /* Copy into tmpBlockDefinitions */
    for (n = 0; n < seqDesc->numUniqueBlocks; ++n) {
        tmpBlockDefs[n].ID = uniqueDefs[n];
        tmpBlockDefs[n].duration_us = (int)(intRows[uniqueDefs[n]][0] * seqDesc->blockDurationRaster_us);
        tmpBlockDefs[n].rfID = (int)intRows[uniqueDefs[n]][1];
        tmpBlockDefs[n].gxID = (int)intRows[uniqueDefs[n]][2];
        tmpBlockDefs[n].gyID = (int)intRows[uniqueDefs[n]][3];
        tmpBlockDefs[n].gzID = (int)intRows[uniqueDefs[n]][4];
    }

    /* Copy block IDs into tmpBlockTable */
    for (n = 0; n < numBlocks; ++n) {
        tmpBlockTable[n].ID = eventTable[n];
    }

    FREE(intRows);
    FREE(uniqueDefs);
    FREE(eventTable);

    /* ========== STEP 4: Allocate output arrays with exact sizes and copy ========== */
    /* RF definitions (exact size) */
    if (seqDesc->numUniqueRFs > 0) {
        seqDesc->rfDefinitions = (pulseqlib_RfDefinition*)ALLOC(seqDesc->numUniqueRFs * sizeof(pulseqlib_RfDefinition));
        if (!seqDesc->rfDefinitions) {
            free_temp_arrays(tmpRfDefs, tmpRfTable, tmpGradDefs, tmpGradTable,
                             tmpAdcDefs, tmpAdcTable, tmpBlockDefs, tmpBlockTable);
            return PULSEQLIB_ERR_ALLOC_FAILED;
        }
        memcpy(seqDesc->rfDefinitions, tmpRfDefs, seqDesc->numUniqueRFs * sizeof(pulseqlib_RfDefinition));
    }

    /* RF table (full library size - indexed by raw library index) */
    if (seqDesc->rfTableSize > 0) {
        seqDesc->rfTable = (pulseqlib_RfTableElement*)ALLOC(seqDesc->rfTableSize * sizeof(pulseqlib_RfTableElement));
        if (!seqDesc->rfTable) {
            free_temp_arrays(tmpRfDefs, tmpRfTable, tmpGradDefs, tmpGradTable,
                             tmpAdcDefs, tmpAdcTable, tmpBlockDefs, tmpBlockTable);
            pulseqlib_sequenceDescriptorFree(seqDesc);
            return PULSEQLIB_ERR_ALLOC_FAILED;
        }
        memcpy(seqDesc->rfTable, tmpRfTable, seqDesc->rfTableSize * sizeof(pulseqlib_RfTableElement));
    }

    /* Gradient definitions (exact size) */
    if (seqDesc->numUniqueGrads > 0) {
        seqDesc->gradDefinitions = (pulseqlib_GradDefinition*)ALLOC(seqDesc->numUniqueGrads * sizeof(pulseqlib_GradDefinition));
        if (!seqDesc->gradDefinitions) {
            free_temp_arrays(tmpRfDefs, tmpRfTable, tmpGradDefs, tmpGradTable,
                             tmpAdcDefs, tmpAdcTable, tmpBlockDefs, tmpBlockTable);
            pulseqlib_sequenceDescriptorFree(seqDesc);
            return PULSEQLIB_ERR_ALLOC_FAILED;
        }
        memcpy(seqDesc->gradDefinitions, tmpGradDefs, seqDesc->numUniqueGrads * sizeof(pulseqlib_GradDefinition));
    }

    /* Gradient table (full library size) */
    if (seqDesc->gradTableSize > 0) {
        seqDesc->gradTable = (pulseqlib_GradTableElement*)ALLOC(seqDesc->gradTableSize * sizeof(pulseqlib_GradTableElement));
        if (!seqDesc->gradTable) {
            free_temp_arrays(tmpRfDefs, tmpRfTable, tmpGradDefs, tmpGradTable,
                             tmpAdcDefs, tmpAdcTable, tmpBlockDefs, tmpBlockTable);
            pulseqlib_sequenceDescriptorFree(seqDesc);
            return PULSEQLIB_ERR_ALLOC_FAILED;
        }
        memcpy(seqDesc->gradTable, tmpGradTable, seqDesc->gradTableSize * sizeof(pulseqlib_GradTableElement));
    }

    /* ADC definitions (exact size) */
    if (seqDesc->numUniqueADCs > 0) {
        seqDesc->adcDefinitions = (pulseqlib_AdcDefinition*)ALLOC(seqDesc->numUniqueADCs * sizeof(pulseqlib_AdcDefinition));
        if (!seqDesc->adcDefinitions) {
            free_temp_arrays(tmpRfDefs, tmpRfTable, tmpGradDefs, tmpGradTable,
                             tmpAdcDefs, tmpAdcTable, tmpBlockDefs, tmpBlockTable);
            pulseqlib_sequenceDescriptorFree(seqDesc);
            return PULSEQLIB_ERR_ALLOC_FAILED;
        }
        memcpy(seqDesc->adcDefinitions, tmpAdcDefs, seqDesc->numUniqueADCs * sizeof(pulseqlib_AdcDefinition));
    }

    /* ADC table (full library size) */
    if (seqDesc->adcTableSize > 0) {
        seqDesc->adcTable = (pulseqlib_AdcTableElement*)ALLOC(seqDesc->adcTableSize * sizeof(pulseqlib_AdcTableElement));
        if (!seqDesc->adcTable) {
            free_temp_arrays(tmpRfDefs, tmpRfTable, tmpGradDefs, tmpGradTable,
                             tmpAdcDefs, tmpAdcTable, tmpBlockDefs, tmpBlockTable);
            pulseqlib_sequenceDescriptorFree(seqDesc);
            return PULSEQLIB_ERR_ALLOC_FAILED;
        }
        memcpy(seqDesc->adcTable, tmpAdcTable, seqDesc->adcTableSize * sizeof(pulseqlib_AdcTableElement));
    }

    /* Block definitions (exact size) */
    if (seqDesc->numUniqueBlocks > 0) {
        seqDesc->blockDefinitions = (pulseqlib_BlockDefinition*)ALLOC(seqDesc->numUniqueBlocks * sizeof(pulseqlib_BlockDefinition));
        if (!seqDesc->blockDefinitions) {
            free_temp_arrays(tmpRfDefs, tmpRfTable, tmpGradDefs, tmpGradTable,
                             tmpAdcDefs, tmpAdcTable, tmpBlockDefs, tmpBlockTable);
            pulseqlib_sequenceDescriptorFree(seqDesc);
            return PULSEQLIB_ERR_ALLOC_FAILED;
        }
        memcpy(seqDesc->blockDefinitions, tmpBlockDefs, seqDesc->numUniqueBlocks * sizeof(pulseqlib_BlockDefinition));
    }

    /* Block table (full numBlocks size) */
    if (numBlocks > 0) {
        seqDesc->blockTable = (pulseqlib_BlockTableElement*)ALLOC(numBlocks * sizeof(pulseqlib_BlockTableElement));
        if (!seqDesc->blockTable) {
            free_temp_arrays(tmpRfDefs, tmpRfTable, tmpGradDefs, tmpGradTable,
                             tmpAdcDefs, tmpAdcTable, tmpBlockDefs, tmpBlockTable);
            pulseqlib_sequenceDescriptorFree(seqDesc);
            return PULSEQLIB_ERR_ALLOC_FAILED;
        }
        memcpy(seqDesc->blockTable, tmpBlockTable, numBlocks * sizeof(pulseqlib_BlockTableElement));
    }

    /* Free temporary arrays - done with them */
    free_temp_arrays(tmpRfDefs, tmpRfTable, tmpGradDefs, tmpGradTable,
                     tmpAdcDefs, tmpAdcTable, tmpBlockDefs, tmpBlockTable);

    /* ========== STEP 5: Copy auxiliary libraries ========== */
    result = copy_rotation_library(seq, seqDesc);
    if (PULSEQLIB_FAILED(result)) {
        pulseqlib_sequenceDescriptorFree(seqDesc);
        return result;
    }

    result = copy_trigger_library(seq, seqDesc);
    if (PULSEQLIB_FAILED(result)) {
        pulseqlib_sequenceDescriptorFree(seqDesc);
        return result;
    }

    result = copy_shapes_library(seq, seqDesc);
    if (PULSEQLIB_FAILED(result)) {
        pulseqlib_sequenceDescriptorFree(seqDesc);
        return result;
    }

    /* ========== STEP 6: Prep/cooldown detection ========== */
    hasPrep = 0;
    hasCooldown = 0;
    for (n = 0; n < seq->labelsetLibrarySize; ++n) {
        if ((int)(seq->labelsetLibrary[n][1]) == ONCE) {
            if ((int)(seq->labelsetLibrary[n][0]) == 1) {
                hasPrep = 1;
            } else if ((int)(seq->labelsetLibrary[n][0]) == 2) {
                hasCooldown = 1;
            }
        }
    }

    if (!hasPrep && !hasCooldown) {
        return PULSEQLIB_OK;
    }

    /* Preparation must start at first block */
    if (hasPrep == 1) {
        getRawBlockContentIDs(seq, &raw, 0, 1);
        getRawExtension(seq, &ext, &raw);
        if (ext.flag.once != 1) {
            pulseqlib_sequenceDescriptorFree(seqDesc);
            return PULSEQLIB_ERR_INVALID_PREP_POSITION;
        }

        ctrl = 0;
        seqDesc->numPrepBlocks = 1;
        while (ctrl == 0 && seqDesc->numPrepBlocks < numBlocks) {
            getRawBlockContentIDs(seq, &raw, seqDesc->numPrepBlocks, 1);
            getRawExtension(seq, &ext, &raw);
            if (ext.flag.once != 0) {
                (seqDesc->numPrepBlocks)++;
            } else {
                ctrl = 1;
            }
        }
    }

    if (hasCooldown == 1) {
        ctrl = 0;
        seqDesc->numCooldownBlocks = 0;
        while (ctrl == 0 && seqDesc->numCooldownBlocks < numBlocks) {
            getRawBlockContentIDs(seq, &raw, numBlocks - 1 - seqDesc->numCooldownBlocks, 1);
            getRawExtension(seq, &ext, &raw);
            if (ext.flag.once != 2) {
                (seqDesc->numCooldownBlocks)++;
            } else {
                ctrl = 1;
            }
        }
        if (ctrl == 0) {
            pulseqlib_sequenceDescriptorFree(seqDesc);
            return PULSEQLIB_ERR_INVALID_COOLDOWN_POSITION;
        }
    }

    if (onceCounter != (seqDesc->numPrepBlocks > 0 ? 1 : 0) + (seqDesc->numCooldownBlocks > 0 ? 1 : 0)) {
        pulseqlib_sequenceDescriptorFree(seqDesc);
        return PULSEQLIB_ERR_INVALID_ONCE_FLAGS;
    }

    return PULSEQLIB_OK;
}

#define PULSEQLIB_PREP_COOLDOWN_THRESHOLD_US 100000 /* 100 ms */
#define PULSEQLIB_SINGLE_TR_MAX_DURATION_US 15000000 /* 15 s */

static long long pulseqlib_sum_durations_us(const int* durations_us, int start, int count)
{
    long long total = 0;
    int i;
    for (i = 0; i < count; ++i) {
        total += (long long)durations_us[start + i];
    }
    return total;
}

/*
 * Find the first repeating segment of a sequence.
 * Equivalent to Python _principal_period behavior (not minimal).
 *
 * seq: pointer to integer array
 * len: number of elements
 *
 * Returns:
 *   Length of first repeating segment if found
 *   len if no repetition is found
 */
int first_repeating_segment(const int *seq, int len)
{
    int start;
    int sublen;
    int L;
    int i;
    int match;
    const int *s;

    if (len <= 1) {
        return len;
    }

    for (start = 0; start < len; start++) {
        s = seq + start;
        sublen = len - start;

        for (L = 1; L <= sublen / 2; L++) {
            match = 1;

            for (i = 0; i < L; i++) {
                if (s[i] != s[i + L]) {
                    match = 0;
                    break;
                }
            }

            if (match) {
                return L;
            }
        }
    }

    return len;
}

/**
 * @brief Detect TR pattern.
 *
 * @param[in, out] seqDesc Pointer to sequence descriptor.
 * @param[in, out] diag Pointer to diagnostic struct (optional, can be NULL).
 * @return PULSEQLIB_OK on success, or negative error code on failure.
 */
int pulseqlib_findTRInSequence(
    pulseqlib_SequenceDescriptor* seqDesc,
    pulseqlib_Diagnostic* diag
) {
    pulseqlib_TRdescriptor* trDesc = &seqDesc->trDescriptor;
    int i, n;
    int imagingStart, imagingEnd, imagingLen;
    int* sequence_pattern;
    int prepDuration_us, cooldownDuration_us;
    int activeDuration_us;
    int* blockDurations_us;
    int found;
    int L;
    pulseqlib_Diagnostic localDiag;
    float trDuration;
    int trStartBlock;

    /* Use local diag if caller doesn't want diagnostics */
    if (!diag) {
        pulseqlib_diagnosticInit(&localDiag);
        diag = &localDiag;
    } else {
        pulseqlib_diagnosticInit(diag);
    }

    /* Initialize */
    found = 0;
    L = 0;

    /* Basic validation */
    if (seqDesc->numBlocks <= 0 || !seqDesc->blockTable || !seqDesc->blockDefinitions) {
        diag->code = (seqDesc->numBlocks <= 0) ? PULSEQLIB_ERR_TR_NO_BLOCKS : PULSEQLIB_ERR_NULL_POINTER;
        return diag->code;
    }
    if (seqDesc->numPrepBlocks < 0 ||seqDesc-> numCooldownBlocks < 0) {
        diag->code = PULSEQLIB_ERR_INVALID_ARGUMENT;
        return diag->code;
    }
    if (seqDesc->numPrepBlocks + seqDesc->numCooldownBlocks > seqDesc->numBlocks) {
        diag->code = PULSEQLIB_ERR_TR_NO_IMAGING_REGION;
        return diag->code;
    }

    /* Fill trDesc with initial values */
    trDesc->trSize = 0;
    trDesc->numTRs = 0;
    trDesc->trDuration_us = 0.0f;
    trDesc->degeneratePrep = 1;
    trDesc->numPrepBlocks = seqDesc->numPrepBlocks;
    trDesc->numPrepTRs = 1;
    trDesc->degenerateCooldown = 1;
    trDesc->numCooldownBlocks = seqDesc->numCooldownBlocks;
    trDesc->numCooldownTRs = 1;

    /* Imaging region is [prepBlocks, numBlocks - cooldownBlocks) */
    imagingStart = seqDesc->numPrepBlocks;
    imagingEnd = seqDesc->numBlocks - seqDesc->numCooldownBlocks; /* exclusive */
    imagingLen = imagingEnd - imagingStart;
    
    diag->imagingRegionLength = imagingLen;
    
    if (imagingLen <= 0) {
        diag->code = PULSEQLIB_ERR_TR_NO_IMAGING_REGION;
        return diag->code;
    }

    /* Count unique blocks for diagnostics */
    {
        int maxUnique = 0;
        for (n = 0; n < seqDesc->numBlocks; ++n) {
            if (seqDesc->blockTable[n].ID > maxUnique) maxUnique = seqDesc->blockTable[n].ID;
        }
        diag->numUniqueBlocks = maxUnique + 1;
    }

    /* To identify TR, pure delay actual duration must be considered */
    sequence_pattern = (int*)ALLOC(seqDesc->numBlocks * sizeof(int));
    if (!sequence_pattern) {
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        return diag->code;
    }
    blockDurations_us = (int*)ALLOC(seqDesc->numBlocks * sizeof(int));
    if (!blockDurations_us) {
        FREE(sequence_pattern);
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        return diag->code;
    }
    for (n = 0; n < seqDesc->numBlocks; ++n) {
        blockDurations_us[n] = seqDesc->blockDefinitions[seqDesc->blockTable[n].ID].duration_us;
        if (seqDesc->blockTable[n].duration_us >= 0) {
            sequence_pattern[n] = blockDurations_us[n];
        } else {
            sequence_pattern[n] = -1 * seqDesc->blockTable[seqDesc->blockTable[n].ID].ID; /* negate to avoid collision with durations */
        }
    }

    /* Try candidate lengths from 1 up to n/2 */
    L = first_repeating_segment(&sequence_pattern[imagingStart], imagingLen);
    diag->candidatePatternLength = L;

    /* Check if pattern found */
    if (L <= 0 || L > imagingLen) {
        found = 0;
    } else {
        found = 1;
    }

    /* Verify consistency over imaging blocks */
    if (found) {
        for (i = 0; i < imagingLen; i++) {
            n = imagingStart + i;
            if (sequence_pattern[n] != sequence_pattern[imagingStart + (i % L)]) 
            {
                diag->mismatchPosition = i;
                diag->blockIndex = n;
                found = 0;
                break;
            }
        }
    }

    /* Fallback for single TR sequences: if no periodic pattern found,
       treat entire sequence as single TR if active duration < 15s */
    if (!found) {
        /* Calculate total duration of non-pure-delay blocks */
        activeDuration_us = 0;
        for (n = 0; n < seqDesc->numBlocks; ++n) {
            if (seqDesc->blockTable[n].duration_us < 0) {
                activeDuration_us += seqDesc->blockDefinitions[seqDesc->blockTable[n].ID].duration_us;
            }
        }

        if (activeDuration_us <= PULSEQLIB_SINGLE_TR_MAX_DURATION_US) {
            /* Treat entire sequence as a single TR */
            trDesc->trSize = seqDesc->numBlocks;
            trDesc->numTRs = 1;
            trDesc->degeneratePrep = 1;
            trDesc->numPrepBlocks = 0;
            trDesc->numPrepTRs = 0;
            trDesc->degenerateCooldown = 1;
            trDesc->numCooldownBlocks = 0;
            trDesc->numCooldownTRs = 0;

            /* Compute TR duration as sum of all block durations */
            trDuration = 0.0f;
            for (i = 0; i < seqDesc->numBlocks; ++i) {
                trDuration += (float)blockDurations_us[i];
            }
            trDesc->trDuration_us = trDuration;

            diag->code = PULSEQLIB_OK;
            FREE(sequence_pattern);
            FREE(blockDurations_us);  /* Don't forget this! */
            return PULSEQLIB_OK; /* SUCCESS - single TR fallback */
        }

        /* Determine which error to report */
        if (diag->mismatchPosition >= 0) {
            diag->code = PULSEQLIB_ERR_TR_PATTERN_MISMATCH;
        } else {
            diag->code = PULSEQLIB_ERR_TR_NO_PERIODIC_PATTERN;
        }
        FREE(sequence_pattern);
        FREE(blockDurations_us);
        return diag->code;
    }

    /* Fill trDesc */
    trDesc->trSize = L;
    trDesc->numTRs = imagingLen / L;

    /* Compute TR duration by summing block durations in one TR period */
    trDuration = 0.0f;
    trStartBlock = imagingStart;
    for (i = 0; i < L; ++i) {
        trDuration += (float)blockDurations_us[trStartBlock + i];
    }
    trDesc->trDuration_us = trDuration;

    /* Safety check for preparation */
    if (seqDesc->numPrepBlocks) {
        if (seqDesc->numPrepBlocks % L == 0) {
            for (n = 0; n < (int)(seqDesc->numPrepBlocks / L); ++n) {
                if (!array_equal(&sequence_pattern[imagingStart], &sequence_pattern[n * L], L)) 
                {
                    prepDuration_us = pulseqlib_sum_durations_us(blockDurations_us, 0, seqDesc->numPrepBlocks);
                    if (prepDuration_us > PULSEQLIB_PREP_COOLDOWN_THRESHOLD_US) 
                    {
                        diag->code = PULSEQLIB_ERR_TR_PREP_TOO_LONG;
                        FREE(sequence_pattern);
                        return diag->code;
                    } else { 
                        trDesc->degeneratePrep = 0;
                        break;
                    }
                }
            }
            if (trDesc->degeneratePrep == 1) 
            {
                trDesc->numPrepBlocks = 0;
                trDesc->numPrepTRs = seqDesc->numPrepBlocks / L;
            }
        } else {
            prepDuration_us = pulseqlib_sum_durations_us(blockDurations_us, 0, seqDesc->numPrepBlocks);
            if (prepDuration_us > PULSEQLIB_PREP_COOLDOWN_THRESHOLD_US)
            {
                diag->code = PULSEQLIB_ERR_TR_PREP_TOO_LONG;
                FREE(sequence_pattern);
                return diag->code;
            } else { 
                trDesc->degeneratePrep = 0;
            }
        }
    }

    /* Safety check for cooldown */
    if (seqDesc->numCooldownBlocks) {
        if (seqDesc->numCooldownBlocks % L == 0) {
            for (n = 0; n < (int)(seqDesc->numCooldownBlocks / L); ++n) {
                if (!array_equal(&sequence_pattern[imagingStart], &sequence_pattern[imagingEnd + n * L], L)) 
                {
                    cooldownDuration_us = pulseqlib_sum_durations_us(blockDurations_us, imagingEnd, seqDesc->numCooldownBlocks);
                    if (cooldownDuration_us > PULSEQLIB_PREP_COOLDOWN_THRESHOLD_US) 
                    {
                        diag->code = PULSEQLIB_ERR_TR_COOLDOWN_TOO_LONG;
                        FREE(sequence_pattern);
                        return diag->code;
                    } else { 
                        trDesc->degenerateCooldown = 0;
                        break; 
                    }
                }
            }
            if (trDesc->degenerateCooldown == 1) 
            {
                trDesc->numCooldownBlocks = 0;
                trDesc->numCooldownTRs = seqDesc->numCooldownBlocks / L;
            }
        } else {
            cooldownDuration_us = pulseqlib_sum_durations_us(blockDurations_us, imagingEnd, seqDesc->numCooldownBlocks);
            if (cooldownDuration_us > PULSEQLIB_PREP_COOLDOWN_THRESHOLD_US) 
            {
                diag->code = PULSEQLIB_ERR_TR_COOLDOWN_TOO_LONG;
                FREE(sequence_pattern);
                return diag->code;
            } else { 
                trDesc->degenerateCooldown = 0;
            }
        }
    }
   
    diag->code = PULSEQLIB_OK;
    FREE(sequence_pattern);
    return PULSEQLIB_OK; /* SUCCESS */
}

/* Get the RF start time with respect to block start */
int get_rf_start_time(pulseqlib_SeqFile const* seq, int rfIndex)
{
    if (!seq || rfIndex < 0 || !seq->rfLibrary || rfIndex >= seq->rfLibrarySize) {
        return 0;
    }
    return (int)(seq->rfLibrary[rfIndex][5]); /* delay */
}

/* Get the RF shape duration with respect to RF start */
int get_rf_duration(pulseqlib_SeqFile const* seq, int rfIndex)
{
    pulseqlib_ShapeArbitrary rf_times;
    pulseqlib_ShapeArbitrary rf_magnitude;
    int rf_raster_us;
    int waveID;
    int num_samples;
    int timeID;
    float rfRasterTime_us;
    if (!seq || rfIndex < 0 || !seq->rfLibrary || rfIndex >= seq->rfLibrarySize) {
        return 0;
    }

    if (seq->reservedDefinitionsLibrary.radiofrequencyRasterTime > 0.0f) {
        rfRasterTime_us = seq->reservedDefinitionsLibrary.radiofrequencyRasterTime;
    } else {
        rfRasterTime_us = seq->opts.rf_raster_time;
    }

    timeID = (int)(seq->rfLibrary[rfIndex][3]);
    if (timeID >= 0){
        decompressShape(&seq->shapesLibrary[timeID], &rf_times, rfRasterTime_us);
        return rf_times.samples[rf_times.numUncompressedSamples - 1] - rf_times.samples[0];
    } else {
        waveID = (int)(seq->rfLibrary[rfIndex][1]);
        decompressShape(&seq->shapesLibrary[waveID], &rf_magnitude, 1.0f);
        num_samples = rf_magnitude.numUncompressedSamples;
        return num_samples * rfRasterTime_us; /* duration in us */}
}

/* Get the ADC start time with respect to block start */
int get_adc_start_time(pulseqlib_SeqFile const* seq, int adcIndex){
    if (!seq || adcIndex < 0 || !seq->adcLibrary || adcIndex >= seq->adcLibrarySize) {
        return 0;
    }
    return (int)(seq->adcLibrary[adcIndex][2]); /* delay */
}

/* Get the readout duration with respect to ADC start */
int get_adc_duration(pulseqlib_SeqFile const* seq, int adcIndex){
    if (!seq || adcIndex < 0 || !seq->adcLibrary || adcIndex >= seq->adcLibrarySize) {
        return 0;
    }
    return (int)(seq->adcLibrary[adcIndex][0] * seq->adcLibrary[adcIndex][1]); /* num_samples * dwell */
}

/* State constants for segment finding state machine */
#define SEGSTATE_SEEKING_FIRST_ADC 0
#define SEGSTATE_SEEKING_BOUNDARY  1
#define SEGSTATE_OPTIMIZED_MODE    2

/* Find segments definitions in a single TR - Single-pass algorithm */
static int findSegmentsInTRInternal(
  const pulseqlib_Opts* opts,
  const pulseqlib_SequenceDescriptor* seqDesc,
  pulseqlib_TRsegment* trSegments,
  const int offset,
  const int trStart,
  const int trSize,
  pulseqlib_Diagnostic* diag
) {
    /* System specs */
    float max_slew;
    float grad_raster_s;
    float maxAllowed;

    /* Gradient amplitude bookkeeping */
    int gradIDs[3];
    float physicalFirst, physicalLast;
    float gradLastCurrent[3];
    float gradFirstNext[3];
    const pulseqlib_BlockDefinition* blockDef;
    const pulseqlib_GradDefinition* gradDef;
    int blockDefID;
    int shotIdx;

    /* Segment building */
    int* segmentStarts;
    int* segmentSizes;
    int numSegments;
    int segmentStart;

    /* State machine */
    int state;
    int candidateBeforeLastRF;
    int savedCandidate;
    int hasSavedCandidate;
    int hasRF, hasADC;
    int isCandidate;
    
    /* Loop counters */
    int numBlocksInTR;
    int n, i;

    /* Parse maximum slew rate from opts */
    max_slew = opts->max_slew;
    grad_raster_s = seqDesc->gradRasterTime_us * 1e-6f;
    maxAllowed = max_slew * grad_raster_s;

    numBlocksInTR = trSize;

    /* Allocate arrays */
    segmentStarts = (int*) ALLOC(numBlocksInTR * sizeof(int));
    segmentSizes = (int*) ALLOC(numBlocksInTR * sizeof(int));
    if (!segmentStarts || !segmentSizes) {
        if (segmentStarts) FREE(segmentStarts);
        if (segmentSizes) FREE(segmentSizes);
        if (diag) diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        return 0;
    }

    /* Check that first block begins with "zero" gradients */
    blockDefID = seqDesc->blockTable[trStart].ID;
    blockDef = &seqDesc->blockDefinitions[blockDefID];
    gradIDs[0] = blockDef->gxID;
    gradIDs[1] = blockDef->gyID;
    gradIDs[2] = blockDef->gzID;
    
    for (i = 0; i < 3; ++i) {
        if (gradIDs[i] < 0) continue;
        gradDef = &seqDesc->gradDefinitions[gradIDs[i]];
        for (shotIdx = 0; shotIdx < gradDef->numShots; ++shotIdx) {
            physicalFirst = gradDef->firstValue[shotIdx] * gradDef->maxAmplitude[shotIdx];
            if (fabs(physicalFirst) > maxAllowed) {
                if (diag) {
                    diag->code = PULSEQLIB_ERR_SEG_NONZERO_START_GRAD;
                    diag->blockIndex = trStart;
                    diag->channel = i;
                    diag->gradientAmplitude = physicalFirst;
                    diag->maxAllowedAmplitude = maxAllowed;
                }
                FREE(segmentStarts);
                FREE(segmentSizes);
                return 0;
            }
        }
    }

    /* Check that last block ends with "zero" gradients */
    blockDefID = seqDesc->blockTable[trStart + numBlocksInTR - 1].ID;
    blockDef = &seqDesc->blockDefinitions[blockDefID];
    gradIDs[0] = blockDef->gxID;
    gradIDs[1] = blockDef->gyID;
    gradIDs[2] = blockDef->gzID;
    
    for (i = 0; i < 3; ++i) {
        if (gradIDs[i] < 0) continue;
        gradDef = &seqDesc->gradDefinitions[gradIDs[i]];
        for (shotIdx = 0; shotIdx < gradDef->numShots; ++shotIdx) {
            physicalLast = gradDef->lastValue[shotIdx] * gradDef->maxAmplitude[shotIdx];
            if (fabs(physicalLast) > maxAllowed) {
                if (diag) {
                    diag->code = PULSEQLIB_ERR_SEG_NONZERO_END_GRAD;
                    diag->blockIndex = trStart + numBlocksInTR - 1;
                    diag->channel = i;
                    diag->gradientAmplitude = physicalLast;
                    diag->maxAllowedAmplitude = maxAllowed;
                }
                FREE(segmentStarts);
                FREE(segmentSizes);
                return 0;
            }
        }
    }

    /* Initialize state machine */
    numSegments = 0;
    segmentStart = trStart;
    state = SEGSTATE_SEEKING_FIRST_ADC;
    candidateBeforeLastRF = -1;
    savedCandidate = -1;
    hasSavedCandidate = 0;

    /* Single pass: check boundaries and RF/ADC events together */
    for (n = trStart; n < trStart + numBlocksInTR; ++n) {
        
        /* Check if there's a boundary candidate between n-1 and n */
        isCandidate = 0;
        if (n > trStart) {
            isCandidate = 1;
            
            /* Get last values of previous block */
            blockDefID = seqDesc->blockTable[n - 1].ID;
            blockDef = &seqDesc->blockDefinitions[blockDefID];
            gradIDs[0] = blockDef->gxID;
            gradIDs[1] = blockDef->gyID;
            gradIDs[2] = blockDef->gzID;
            
            for (i = 0; i < 3; ++i) {
                gradLastCurrent[i] = 0.0f;
                if (gradIDs[i] >= 0) {
                    gradDef = &seqDesc->gradDefinitions[gradIDs[i]];
                    for (shotIdx = 0; shotIdx < gradDef->numShots; ++shotIdx) {
                        physicalLast = gradDef->lastValue[shotIdx] * gradDef->maxAmplitude[shotIdx];
                        if (fabs(physicalLast) > fabs(gradLastCurrent[i])) {
                            gradLastCurrent[i] = physicalLast;
                        }
                    }
                }
            }

            /* Get first values of current block */
            blockDefID = seqDesc->blockTable[n].ID;
            blockDef = &seqDesc->blockDefinitions[blockDefID];
            gradIDs[0] = blockDef->gxID;
            gradIDs[1] = blockDef->gyID;
            gradIDs[2] = blockDef->gzID;
            
            for (i = 0; i < 3; ++i) {
                gradFirstNext[i] = 0.0f;
                if (gradIDs[i] >= 0) {
                    gradDef = &seqDesc->gradDefinitions[gradIDs[i]];
                    for (shotIdx = 0; shotIdx < gradDef->numShots; ++shotIdx) {
                        physicalFirst = gradDef->firstValue[shotIdx] * gradDef->maxAmplitude[shotIdx];
                        if (fabs(physicalFirst) > fabs(gradFirstNext[i])) {
                            gradFirstNext[i] = physicalFirst;
                        }
                    }
                }
            }

            /* Check if this is a valid boundary candidate */
            for (i = 0; i < 3; ++i) {
                if (fabs(gradLastCurrent[i]) > maxAllowed || fabs(gradFirstNext[i]) > maxAllowed) {
                    isCandidate = 0;
                    break;
                }
            }
        }

        /* Get RF/ADC status for current block */
        hasRF = (seqDesc->blockDefinitions[seqDesc->blockTable[n].ID].rfID >= 0);
        hasADC = (seqDesc->blockTable[n].adcID >= 0);

        if (state == SEGSTATE_SEEKING_FIRST_ADC) {
            /* Track the last candidate seen before the most recent RF */
            if (isCandidate) {
                savedCandidate = n;
            }
            if (hasRF) {
                /* When we see RF, the current savedCandidate becomes the prep/excitation boundary */
                candidateBeforeLastRF = savedCandidate;
                savedCandidate = -1;
            }
            if (hasADC) {
                /* Found first ADC. Commit prep segment if we have a valid candidate */
                if (candidateBeforeLastRF > segmentStart) {
                    segmentStarts[numSegments] = segmentStart;
                    segmentSizes[numSegments] = candidateBeforeLastRF - segmentStart;
                    numSegments++;
                    segmentStart = candidateBeforeLastRF;
                }
                state = SEGSTATE_SEEKING_BOUNDARY;
                hasSavedCandidate = 0;
                savedCandidate = -1;
            }
        }
        else if (state == SEGSTATE_SEEKING_BOUNDARY) {
            /* Track boundary candidates */
            if (isCandidate) {
                savedCandidate = n;
                hasSavedCandidate = 1;
            }

            if (hasRF) {
                if (hasSavedCandidate) {
                    /* Commit segment at last saved candidate */
                    segmentStarts[numSegments] = segmentStart;
                    segmentSizes[numSegments] = savedCandidate - segmentStart;
                    numSegments++;
                    segmentStart = savedCandidate;
                    hasSavedCandidate = 0;
                    savedCandidate = -1;
                } else {
                    /* No candidate found between last ADC and this RF → optimized mode */
                    state = SEGSTATE_OPTIMIZED_MODE;
                }
            }
        }
        /* SEGSTATE_OPTIMIZED_MODE: no action, everything goes in current segment */
    }

    /* Commit final segment */
    segmentStarts[numSegments] = segmentStart;
    segmentSizes[numSegments] = trStart + numBlocksInTR - segmentStart;
    numSegments++;

    /* Copy to output */
    for (i = 0; i < numSegments; ++i) {
        trSegments[offset + i].startBlock = segmentStarts[i];
        trSegments[offset + i].numBlocks = segmentSizes[i];
        trSegments[offset + i].uniqueBlockIndices = NULL;
    }

    FREE(segmentStarts);
    FREE(segmentSizes);

    return numSegments;
}

/**
 * @brief Strip leading and trailing pure delay blocks from ALL segments.
 *
 * For each input segment, this function:
 * 1. Creates individual 1-block segments for EACH leading pure delay
 * 2. Creates a segment for the core (non-pure-delay) blocks
 * 3. Creates individual 1-block segments for EACH trailing pure delay
 *
 * @param[in]     rawSegments      Array of raw segments from findSegmentsInTRInternal.
 * @param[in]     numRawSegments   Number of raw segments.
 * @param[in]     blockTable       Block table containing pureDelayFlag per block.
 * @param[out]    outSegments      Output array for expanded segments (must be pre-allocated).
 * @param[in]     maxOutSegments   Maximum number of output segments (size of outSegments).
 * @return Number of output segments, or -1 on error.
 */
static int stripPureDelaysFromSegments(
    const pulseqlib_TRsegment* rawSegments,
    int numRawSegments,
    const pulseqlib_BlockTableElement* blockTable,
    pulseqlib_TRsegment* outSegments,
    int maxOutSegments)
{
    int numOut = 0;
    int segIdx, i;
    int numBlocks;
    int leadingDelays, trailingDelays;
    int coreStart, coreEnd, coreSize;
    const int* indices;
    
    if (numRawSegments == 0) {
        return 0;
    }
    
    for (segIdx = 0; segIdx < numRawSegments; ++segIdx) {
        numBlocks = rawSegments[segIdx].numBlocks;
        indices = rawSegments[segIdx].uniqueBlockIndices;
        
        if (numBlocks == 0 || !indices) {
            continue;
        }
        
        /* Count leading pure delays */
        /* Count leading pure delays */
        leadingDelays = 0;
        for (i = 0; i < numBlocks; ++i) {
            if (blockTable[rawSegments[segIdx].startBlock + i].duration_us >= 0) {
                leadingDelays++;
            } else {
                break;
            }
        }
        
        /* Count trailing pure delays (don't overlap with leading) */
        trailingDelays = 0;
        for (i = numBlocks - 1; i >= leadingDelays; --i) {
            if (blockTable[rawSegments[segIdx].startBlock + i].duration_us >= 0) {
                trailingDelays++;
            } else {
                break;
            }
        }
        
        /* Compute core range */
        coreStart = leadingDelays;
        coreEnd = numBlocks - trailingDelays;
        
        /* Create individual segments for EACH leading pure delay */
        for (i = 0; i < leadingDelays; ++i) {
            if (numOut >= maxOutSegments) {
                return -1;
            }
            outSegments[numOut].startBlock = rawSegments[segIdx].startBlock + i;
            outSegments[numOut].numBlocks = 1;
            outSegments[numOut].uniqueBlockIndices = (int*)ALLOC(sizeof(int));
            if (!outSegments[numOut].uniqueBlockIndices) {
                return -1;
            }
            outSegments[numOut].uniqueBlockIndices[0] = indices[i];
            numOut++;
        }
        
        /* Create segment for core blocks, if any */
        if (coreEnd > coreStart) {
            coreSize = coreEnd - coreStart;
            if (numOut >= maxOutSegments) {
                return -1;
            }
            outSegments[numOut].startBlock = rawSegments[segIdx].startBlock + coreStart;
            outSegments[numOut].numBlocks = coreSize;
            outSegments[numOut].uniqueBlockIndices = (int*)ALLOC(coreSize * sizeof(int));
            if (!outSegments[numOut].uniqueBlockIndices) {
                return -1;
            }
            for (i = 0; i < coreSize; ++i) {
                outSegments[numOut].uniqueBlockIndices[i] = indices[coreStart + i];
            }
            numOut++;
        }
        
        /* Create individual segments for EACH trailing pure delay */
        for (i = 0; i < trailingDelays; ++i) {
            if (numOut >= maxOutSegments) {
                return -1;
            }
            outSegments[numOut].startBlock = rawSegments[segIdx].startBlock + coreEnd + i;
            outSegments[numOut].numBlocks = 1;
            outSegments[numOut].uniqueBlockIndices = (int*)ALLOC(sizeof(int));
            if (!outSegments[numOut].uniqueBlockIndices) {
                return -1;
            }
            outSegments[numOut].uniqueBlockIndices[0] = indices[coreEnd + i];
            numOut++;
        }
    }
    
    return numOut;
}

/**
 * @brief Get segment definitions in TR.
 */
int pulseqlib_findSegmentsInTR(
  const pulseqlib_SeqFile* seq, 
  pulseqlib_SequenceDescriptor* seqDesc,
  pulseqlib_Diagnostic* diag
) {
    const pulseqlib_TRdescriptor* trDesc = &seqDesc->trDescriptor;
    const pulseqlib_BlockTableElement* bte;
    const pulseqlib_BlockDefinition* bdef;
    pulseqlib_TRsegment* trSegments = NULL;
    pulseqlib_TRsegment* trSegmentsRaw = NULL;
    pulseqlib_TRsegment* trSegmentsExpanded = NULL;
    pulseqlib_Diagnostic localDiag;
    int numBlocks;
    int numSegmentsRaw;
    int numPrepSegmentsRaw, numMainSegmentsRaw, numCooldownSegmentsRaw;
    int numPrepSegments, numMainSegments, numCooldownSegments;
    int numSegmentsTotal;
    int numUniqueSegments;
    int found;
    int trStart;
    int trSize;
    int n, b, i;
    int segResult;
    int offset;
    int maxExpandedSegments;
    int pureDelayUniqueIdx;
    int isPureDelay;
    int nb;
    int uniqueIdx;
    int blockTableIdx;
    int blockDefID;
    int shotIdx;
    int axGradIDs[3];
    int axDefIDs[3];
    int ax;
    float* maxEnergy;
    float instanceEnergy, e;
    float amp;

    /* Use local diag if caller doesn't want diagnostics */
    if (!diag) {
        pulseqlib_diagnosticInit(&localDiag);
        diag = &localDiag;
    } else {
        pulseqlib_diagnosticInit(diag);
    }

    if (!seq || !seqDesc) {
        diag->code = PULSEQLIB_ERR_NULL_POINTER;
        return 0;
    }

    /* Initialize counts */
    numPrepSegmentsRaw = 0;
    numMainSegmentsRaw = 0;
    numCooldownSegmentsRaw = 0;
    numSegmentsRaw = 0;

    numBlocks = trDesc->trSize + trDesc->numPrepBlocks + trDesc->numCooldownBlocks;

    /* Allocate temporary storage for raw segments (at most one segment per block) */
    trSegmentsRaw = (pulseqlib_TRsegment*) ALLOC(numBlocks * sizeof(pulseqlib_TRsegment));
    if (!trSegmentsRaw) {
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        return 0;
    }

    /* ========== Find raw segments in each section ========== */

    /* Prep section */
    if (trDesc->degeneratePrep == 0 && trDesc->numPrepBlocks > 0) {
        trStart = 0;
        trSize = trDesc->numPrepBlocks + trDesc->trSize;
        segResult = findSegmentsInTRInternal(&seq->opts, seqDesc, trSegmentsRaw, numSegmentsRaw, trStart, trSize, diag);
        if (segResult == 0 && PULSEQLIB_FAILED(diag->code)) {
            FREE(trSegmentsRaw);
            return 0;
        }
        numPrepSegmentsRaw = segResult;
        numSegmentsRaw += numPrepSegmentsRaw;
    }

    /* Main TR section */
    trStart = trDesc->numPrepBlocks;
    trSize = trDesc->trSize;
    segResult = findSegmentsInTRInternal(&seq->opts, seqDesc, trSegmentsRaw, numSegmentsRaw, trStart, trSize, diag);
    if (segResult == 0 && PULSEQLIB_FAILED(diag->code)) {
        FREE(trSegmentsRaw);
        return 0;
    }
    numMainSegmentsRaw = segResult;
    numSegmentsRaw += numMainSegmentsRaw;

    /* Cooldown section */
    if (trDesc->degenerateCooldown == 0 && trDesc->numCooldownBlocks > 0) {
        trStart = seq->numBlocks - trDesc->numCooldownBlocks - trDesc->trSize;
        trSize = trDesc->numCooldownBlocks + trDesc->trSize;
        segResult = findSegmentsInTRInternal(&seq->opts, seqDesc, trSegmentsRaw, numSegmentsRaw, trStart, trSize, diag);
        if (segResult == 0 && PULSEQLIB_FAILED(diag->code)) {
            FREE(trSegmentsRaw);
            return 0;
        }
        numCooldownSegmentsRaw = segResult;
        numSegmentsRaw += numCooldownSegmentsRaw;
    }

    /* Check if any segments were found */
    if (numSegmentsRaw == 0) {
        diag->code = PULSEQLIB_ERR_SEG_NO_SEGMENTS_FOUND;
        FREE(trSegmentsRaw);
        return 0;
    }

    /* Parse actual segment definition from blockTable (fill uniqueBlockIndices) */
    for (n = 0; n < numSegmentsRaw; ++n) {
        trSegmentsRaw[n].uniqueBlockIndices = (int*) ALLOC(trSegmentsRaw[n].numBlocks * sizeof(int));
        if (!trSegmentsRaw[n].uniqueBlockIndices) {
            diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
            for (i = 0; i < n; ++i) FREE(trSegmentsRaw[i].uniqueBlockIndices);
            FREE(trSegmentsRaw);
            return 0;
        }
        for (i = 0; i < trSegmentsRaw[n].numBlocks; ++i) {
            trSegmentsRaw[n].uniqueBlockIndices[i] = seqDesc->blockTable[trSegmentsRaw[n].startBlock + i].ID;
        }
    }

    /* ========== Strip pure delays from segments ========== */
    /* Maximum expanded segments: each block could become its own segment */
    maxExpandedSegments = numBlocks;
    trSegmentsExpanded = (pulseqlib_TRsegment*) ALLOC(maxExpandedSegments * sizeof(pulseqlib_TRsegment));
    if (!trSegmentsExpanded) {
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        for (n = 0; n < numSegmentsRaw; ++n) FREE(trSegmentsRaw[n].uniqueBlockIndices);
        FREE(trSegmentsRaw);
        return 0;
    }

    /* Process prep section */
    offset = 0;
    numPrepSegments = 0;
    if (numPrepSegmentsRaw > 0) {
        numPrepSegments = stripPureDelaysFromSegments(
            trSegmentsRaw, numPrepSegmentsRaw,
            seqDesc->blockTable, trSegmentsExpanded + offset, maxExpandedSegments - offset);
        if (numPrepSegments == 0 && numPrepSegmentsRaw > 0) {
            diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
            for (n = 0; n < numSegmentsRaw; ++n) FREE(trSegmentsRaw[n].uniqueBlockIndices);
            FREE(trSegmentsRaw);
            FREE(trSegmentsExpanded);
            return 0;
        }
        offset += numPrepSegments;
    }

    /* Process main section */
    numMainSegments = stripPureDelaysFromSegments(
        trSegmentsRaw + numPrepSegmentsRaw, numMainSegmentsRaw,
        seqDesc->blockTable, trSegmentsExpanded + offset, maxExpandedSegments - offset);
    if (numMainSegments == 0 && numMainSegmentsRaw > 0) {
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        for (n = 0; n < offset; ++n) FREE(trSegmentsExpanded[n].uniqueBlockIndices);
        for (n = 0; n < numSegmentsRaw; ++n) FREE(trSegmentsRaw[n].uniqueBlockIndices);
        FREE(trSegmentsRaw);
        FREE(trSegmentsExpanded);
        return 0;
    }
    offset += numMainSegments;

    /* Process cooldown section */
    numCooldownSegments = 0;
    if (numCooldownSegmentsRaw > 0) {
        numCooldownSegments = stripPureDelaysFromSegments(
            trSegmentsRaw + numPrepSegmentsRaw + numMainSegmentsRaw, numCooldownSegmentsRaw,
            seqDesc->blockTable, trSegmentsExpanded + offset, maxExpandedSegments - offset);
        if (numCooldownSegments == 0 && numCooldownSegmentsRaw > 0) {
            diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
            for (n = 0; n < offset; ++n) FREE(trSegmentsExpanded[n].uniqueBlockIndices);
            for (n = 0; n < numSegmentsRaw; ++n) FREE(trSegmentsRaw[n].uniqueBlockIndices);
            FREE(trSegmentsRaw);
            FREE(trSegmentsExpanded);
            return 0;
        }
        offset += numCooldownSegments;
    }

    numSegmentsTotal = numPrepSegments + numMainSegments + numCooldownSegments;

    /* Free raw segments */
    for (n = 0; n < numSegmentsRaw; ++n) {
        FREE(trSegmentsRaw[n].uniqueBlockIndices);
    }
    FREE(trSegmentsRaw);

    /* ========== Allocate output segment tables ========== */
    seqDesc->segmentTable.numPrepSegments = numPrepSegments;
    seqDesc->segmentTable.numMainSegments = numMainSegments;
    seqDesc->segmentTable.numCooldownSegments = numCooldownSegments;

    seqDesc->segmentTable.prepSegmentTable = (numPrepSegments > 0) 
        ? (int*) ALLOC(numPrepSegments * sizeof(int)) : NULL;
    seqDesc->segmentTable.mainSegmentTable = (numMainSegments > 0) 
        ? (int*) ALLOC(numMainSegments * sizeof(int)) : NULL;
    seqDesc->segmentTable.cooldownSegmentTable = (numCooldownSegments > 0) 
        ? (int*) ALLOC(numCooldownSegments * sizeof(int)) : NULL;

    /* Allocate output segments array */
    trSegments = (pulseqlib_TRsegment*) ALLOC(numSegmentsTotal * sizeof(pulseqlib_TRsegment));
    if (!trSegments) {
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        for (n = 0; n < numSegmentsTotal; ++n) FREE(trSegmentsExpanded[n].uniqueBlockIndices);
        FREE(trSegmentsExpanded);
        return 0;
    }

    /* ========== Find unique segments and fill tables ========== */
    /* 
     * Special handling for pure delay segments: ALL pure delay segments 
     * are considered identical regardless of their block ID or duration.
     */
    numUniqueSegments = 0;
    pureDelayUniqueIdx = -1;
        
    for (n = 0; n < numSegmentsTotal; ++n) {
        /* Check if this is a pure delay segment (single block that is pure delay) */
        isPureDelay = (trSegmentsExpanded[n].numBlocks == 1 && 
                        seqDesc->blockTable[trSegmentsExpanded[n].startBlock].duration_us >= 0);
        
        if (isPureDelay) {
            if (pureDelayUniqueIdx == -1) {
                /* Create the unique pure delay segment (use first occurrence) */
                trSegments[numUniqueSegments].numBlocks = 1;
                trSegments[numUniqueSegments].startBlock = trSegmentsExpanded[n].startBlock;
                trSegments[numUniqueSegments].uniqueBlockIndices = (int*)ALLOC(sizeof(int));
                if (!trSegments[numUniqueSegments].uniqueBlockIndices) {
                    diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
                    for (i = 0; i < numUniqueSegments; ++i) FREE(trSegments[i].uniqueBlockIndices);
                    for (i = 0; i < numSegmentsTotal; ++i) FREE(trSegmentsExpanded[i].uniqueBlockIndices);
                    FREE(trSegments);
                    FREE(trSegmentsExpanded);
                    return 0;
                }
                trSegments[numUniqueSegments].uniqueBlockIndices[0] = trSegmentsExpanded[n].uniqueBlockIndices[0];
                pureDelayUniqueIdx = numUniqueSegments;
                numUniqueSegments++;
            }
            found = pureDelayUniqueIdx;
        } else {
            /* Non-pure-delay segment: use normal deduplication */
            found = -1;
            for (i = 0; i < numUniqueSegments; ++i) {
                /* Skip the pure delay segment in comparison */
                if (i == pureDelayUniqueIdx) {
                    continue;
                }
                if (trSegmentsExpanded[n].numBlocks == trSegments[i].numBlocks &&
                    array_equal(trSegmentsExpanded[n].uniqueBlockIndices, 
                                trSegments[i].uniqueBlockIndices, 
                                trSegmentsExpanded[n].numBlocks)) {
                    found = i;
                    break;
                }
            }

            if (found == -1) {
                /* New unique segment */
                trSegments[numUniqueSegments].numBlocks = trSegmentsExpanded[n].numBlocks;
                trSegments[numUniqueSegments].startBlock = trSegmentsExpanded[n].startBlock;
                trSegments[numUniqueSegments].uniqueBlockIndices = 
                    (int*) ALLOC(trSegmentsExpanded[n].numBlocks * sizeof(int));
                if (!trSegments[numUniqueSegments].uniqueBlockIndices) {
                    diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
                    for (i = 0; i < numUniqueSegments; ++i) FREE(trSegments[i].uniqueBlockIndices);
                    for (i = 0; i < numSegmentsTotal; ++i) FREE(trSegmentsExpanded[i].uniqueBlockIndices);
                    FREE(trSegments);
                    FREE(trSegmentsExpanded);
                    return 0;
                }
                for (i = 0; i < trSegmentsExpanded[n].numBlocks; ++i) {
                    trSegments[numUniqueSegments].uniqueBlockIndices[i] = 
                        trSegmentsExpanded[n].uniqueBlockIndices[i];
                }
                found = numUniqueSegments;
                numUniqueSegments++;
            }
        }

        /* Store mapping in the appropriate section table */
        if (n < numPrepSegments) {
            seqDesc->segmentTable.prepSegmentTable[n] = found;
        } else if (n < numPrepSegments + numMainSegments) {
            seqDesc->segmentTable.mainSegmentTable[n - numPrepSegments] = found;
        } else {
            seqDesc->segmentTable.cooldownSegmentTable[n - numPrepSegments - numMainSegments] = found;
        }
    }
    
    seqDesc->segmentTable.numUniqueSegments = numUniqueSegments;

    /* Store results in seqDesc */
    seqDesc->numUniqueSegments = numUniqueSegments;
    
    /* Reallocate trSegments to exact size and store in seqDesc */
    seqDesc->segmentDefinitions = (pulseqlib_TRsegment*) ALLOC(numUniqueSegments * sizeof(pulseqlib_TRsegment));
    if (seqDesc->segmentDefinitions) {
        for (i = 0; i < numUniqueSegments; ++i) {
            seqDesc->segmentDefinitions[i] = trSegments[i];
        }
    }
        FREE(trSegments);

    /* ========== Populate per-block flags (trigger, rotation, norot, nopos) ========== */
    /* 
     * For each unique segment, allocate and zero-init per-block flag arrays.
     * Then scan all expanded segment instances, and for each block in the instance,
     * OR in the flags from the blockTable entry.
     */
    for (i = 0; i < numUniqueSegments; ++i) {
        nb = seqDesc->segmentDefinitions[i].numBlocks;
        seqDesc->segmentDefinitions[i].hasTrigger  = (int*) ALLOC(nb * sizeof(int));
        seqDesc->segmentDefinitions[i].hasRotation  = (int*) ALLOC(nb * sizeof(int));
        seqDesc->segmentDefinitions[i].norotFlag    = (int*) ALLOC(nb * sizeof(int));
        seqDesc->segmentDefinitions[i].noposFlag    = (int*) ALLOC(nb * sizeof(int));
        if (!seqDesc->segmentDefinitions[i].hasTrigger ||
            !seqDesc->segmentDefinitions[i].hasRotation ||
            !seqDesc->segmentDefinitions[i].norotFlag ||
            !seqDesc->segmentDefinitions[i].noposFlag) {
            diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
            for (n = 0; n < numSegmentsTotal; ++n) FREE(trSegmentsExpanded[n].uniqueBlockIndices);
            FREE(trSegmentsExpanded);
            return 0;
        }
        for (n = 0; n < nb; ++n) {
            seqDesc->segmentDefinitions[i].hasTrigger[n]  = 0;
            seqDesc->segmentDefinitions[i].hasRotation[n] = 0;
            seqDesc->segmentDefinitions[i].norotFlag[n]   = 0;
            seqDesc->segmentDefinitions[i].noposFlag[n]   = 0;
        }
    }

    /* Scan all expanded segment instances */
    for (n = 0; n < numSegmentsTotal; ++n) {
        
        /* Look up this instance's unique segment index from the section tables */
        if (n < numPrepSegments) {
            uniqueIdx = seqDesc->segmentTable.prepSegmentTable[n];
        } else if (n < numPrepSegments + numMainSegments) {
            uniqueIdx = seqDesc->segmentTable.mainSegmentTable[n - numPrepSegments];
        } else {
            uniqueIdx = seqDesc->segmentTable.cooldownSegmentTable[n - numPrepSegments - numMainSegments];
        }
        
        instanceEnergy = 0.0f;
        
        /* For each block in this instance */
        for (b = 0; b < trSegmentsExpanded[n].numBlocks; ++b) {
            blockTableIdx = trSegmentsExpanded[n].startBlock + b;
            bte = &seqDesc->blockTable[blockTableIdx];
            blockDefID = bte->ID;
            bdef = &seqDesc->blockDefinitions[blockDefID];
            
            /* OR in flags */
            if (bte->triggerID != -1) {
                seqDesc->segmentDefinitions[uniqueIdx].hasTrigger[b] = 1;
            }
            if (bte->rotationID != -1) {
                seqDesc->segmentDefinitions[uniqueIdx].hasRotation[b] = 1;
            }
            if (bte->norotFlag) {
                seqDesc->segmentDefinitions[uniqueIdx].norotFlag[b] = 1;
            }
            if (bte->noposFlag) {
                seqDesc->segmentDefinitions[uniqueIdx].noposFlag[b] = 1;
            }
            
            /* Accumulate gradient energy for this instance */
            axGradIDs[0] = bte->gxID;
            axGradIDs[1] = bte->gyID;
            axGradIDs[2] = bte->gzID;
            axDefIDs[0] = bdef->gxID;
            axDefIDs[1] = bdef->gyID;
            axDefIDs[2] = bdef->gzID;

            for (ax = 0; ax < 3; ++ax) {
                if (axGradIDs[ax] >= 0 && axGradIDs[ax] < seqDesc->gradTableSize &&
                    axDefIDs[ax] >= 0 && axDefIDs[ax] < seqDesc->numUniqueGrads) {
                    amp = seqDesc->gradTable[axGradIDs[ax]].amplitude;
                    shotIdx = seqDesc->gradTable[axGradIDs[ax]].shotIndex;
                    e = seqDesc->gradDefinitions[axDefIDs[ax]].energy[shotIdx];
                    instanceEnergy += e * amp * amp;
                }
            }
        }

        /* Track the instance with maximum energy */
        if (instanceEnergy > maxEnergy[uniqueIdx]) {
            maxEnergy[uniqueIdx] = instanceEnergy;
            seqDesc->segmentDefinitions[uniqueIdx].maxEnergyStartBlock = trSegmentsExpanded[n].startBlock;
        }
    }
    
    FREE(maxEnergy);

    /* Free expanded segments */
    for (n = 0; n < numSegmentsTotal; ++n) {
        FREE(trSegmentsExpanded[n].uniqueBlockIndices);
    }
    FREE(trSegmentsExpanded);

    diag->code = PULSEQLIB_OK;
    return numUniqueSegments;
}

void pulseqlib_segmentTableResultFree(pulseqlib_SegmentTableResult* result) {
    if (!result) return;
    if (result->prepSegmentTable) FREE(result->prepSegmentTable);
    if (result->mainSegmentTable) FREE(result->mainSegmentTable);
    if (result->cooldownSegmentTable) FREE(result->cooldownSegmentTable);
    result->prepSegmentTable = NULL;
    result->mainSegmentTable = NULL;
    result->cooldownSegmentTable = NULL;
    result->numPrepSegments = 0;
    result->numMainSegments = 0;
    result->numCooldownSegments = 0;
    result->numUniqueSegments = 0;
}

int pulseqlib_getCollectionDescriptors(
    const pulseqlib_SeqFileCollection* collection,
    pulseqlib_SequenceDescriptorCollection* descCollection,
    pulseqlib_Diagnostic* diag)
{
    int i, j;
    int result;
    int adcOffset = 0;
    int segmentOffset = 0;
    int blockOffset = 0;
    pulseqlib_Diagnostic localDiag;
    
    if (!diag) {
        pulseqlib_diagnosticInit(&localDiag);
        diag = &localDiag;
    }
    
    if (!collection || !descCollection) {
        diag->code = PULSEQLIB_ERR_NULL_POINTER;
        return 0;
    }
    
    
    if (collection->numSequences == 0) {
        diag->code = PULSEQLIB_ERR_COLLECTION_EMPTY;
        return 0;
    }
    
    /* Allocate descriptor and info arrays */
    descCollection->descriptors = (pulseqlib_SequenceDescriptor*)ALLOC(
        collection->numSequences * sizeof(pulseqlib_SequenceDescriptor));
    descCollection->subsequenceInfo = (pulseqlib_SubsequenceInfo*)ALLOC(
        collection->numSequences * sizeof(pulseqlib_SubsequenceInfo));
    
    if (!descCollection->descriptors || !descCollection->subsequenceInfo) {
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        if (descCollection->descriptors) FREE(descCollection->descriptors);
        if (descCollection->subsequenceInfo) FREE(descCollection->subsequenceInfo);
        descCollection->descriptors = NULL;
        descCollection->subsequenceInfo = NULL;
        return 0;
    }
    
    /* Initialize collection totals */
    descCollection->numSubsequences = collection->numSequences;
    descCollection->totalDuration_us = 0.0f;
    descCollection->totalUniqueSegments = 0;
    descCollection->totalUniqueADCs = 0;
    descCollection->totalBlocks = 0;
    
    /* Process each subsequence */
    for (i = 0; i < collection->numSequences; ++i) {
        /* Store offsets BEFORE processing this subsequence */
        descCollection->subsequenceInfo[i].sequenceIndex = i;
        descCollection->subsequenceInfo[i].adcIDOffset = adcOffset;
        descCollection->subsequenceInfo[i].segmentIDOffset = segmentOffset;
        descCollection->subsequenceInfo[i].blockIndexOffset = blockOffset;
        
        /* Get unique blocks */
        result = pulseqlib_getUniqueBlocks(
            &collection->sequences[i],
            &descCollection->descriptors[i]);
        if (PULSEQLIB_FAILED(result)) {
            diag->code = result;
            goto cleanup_error;
        }
        
        /* Find TR structure */
        result = pulseqlib_findTRInSequence(
            &descCollection->descriptors[i],
            diag);
        if (PULSEQLIB_FAILED(diag->code)) {
            goto cleanup_error;
        }
        
        /* Find segments */
        result = pulseqlib_findSegmentsInTR(
            &collection->sequences[i],
            &descCollection->descriptors[i],
            diag);
        if (PULSEQLIB_FAILED(diag->code)) {
            goto cleanup_error;
        }
        
        /* Apply segment ID offsets */
        if (segmentOffset > 0) {
            for (j = 0; j < descCollection->descriptors[i].segmentTable.numPrepSegments; ++j) {
                descCollection->descriptors[i].segmentTable.prepSegmentTable[j] += segmentOffset;
            }
            for (j = 0; j < descCollection->descriptors[i].segmentTable.numMainSegments; ++j) {
                descCollection->descriptors[i].segmentTable.mainSegmentTable[j] += segmentOffset;
            }
            for (j = 0; j < descCollection->descriptors[i].segmentTable.numCooldownSegments; ++j) {
                descCollection->descriptors[i].segmentTable.cooldownSegmentTable[j] += segmentOffset;
            }
        }
        
        /* Apply ADC ID offsets */
        if (adcOffset > 0) {
            for (j = 0; j < descCollection->descriptors[i].adcTableSize; ++j) {
                descCollection->descriptors[i].adcTable[j].ID += adcOffset;
            }
            for (j = 0; j < descCollection->descriptors[i].numUniqueADCs; ++j) {
                descCollection->descriptors[i].adcDefinitions[j].ID += adcOffset;
            }
        }
        
        /* Update offsets for next subsequence */
        adcOffset += descCollection->descriptors[i].numUniqueADCs;
        segmentOffset += descCollection->descriptors[i].numUniqueSegments;
        blockOffset += descCollection->descriptors[i].numBlocks;
        
        /* Accumulate duration */
        descCollection->totalDuration_us += 
            descCollection->descriptors[i].trDescriptor.trDuration_us * 
            descCollection->descriptors[i].trDescriptor.numTRs;
    }
    
    /* Store global totals */
    descCollection->totalUniqueSegments = segmentOffset;
    descCollection->totalUniqueADCs = adcOffset;
    descCollection->totalBlocks = blockOffset;
    
    diag->code = PULSEQLIB_OK;
    return collection->numSequences;

cleanup_error:
    for (j = 0; j < i; ++j) {
        pulseqlib_sequenceDescriptorFree(&descCollection->descriptors[j]);
    }
    FREE(descCollection->descriptors);
    FREE(descCollection->subsequenceInfo);
    descCollection->descriptors = NULL;
    descCollection->subsequenceInfo = NULL;
    descCollection->numSubsequences = 0;
    descCollection->totalUniqueSegments = 0;
    descCollection->totalUniqueADCs = 0;
    descCollection->totalBlocks = 0;
    descCollection->totalDuration_us = 0.0f;
    return 0;
}

/***************************************** SeqDescriptor Methods  **********************************/
/**
 * @brief Free memory allocated for TR gradient waveforms.
 */
void pulseqlib_trGradientWaveformsFree(pulseqlib_TRGradientWaveforms* waveforms)
{
    if (!waveforms) return;
    if (waveforms->timeGx) FREE(waveforms->timeGx);
    if (waveforms->timeGy) FREE(waveforms->timeGy);
    if (waveforms->timeGz) FREE(waveforms->timeGz);
    if (waveforms->waveformGx) FREE(waveforms->waveformGx);
    if (waveforms->waveformGy) FREE(waveforms->waveformGy);
    if (waveforms->waveformGz) FREE(waveforms->waveformGz);
    waveforms->timeGx = NULL;
    waveforms->timeGy = NULL;
    waveforms->timeGz = NULL;
    waveforms->waveformGx = NULL;
    waveforms->waveformGy = NULL;
    waveforms->waveformGz = NULL;
    waveforms->numSamplesGx = 0;
    waveforms->numSamplesGy = 0;
    waveforms->numSamplesGz = 0;
}

static int count_grad_samples_for_block(
    const pulseqlib_SequenceDescriptor* seqDesc,
    const pulseqlib_GradDefinition* gradDef,
    const float blockDuration_us)
{
    int count;
    int numSamples;
    float delay_us, riseTime_us, flatTime_us, fallTime_us;
    float duration_us;
    float gradRaster_us;
    float blockDurationRaster_us;
    pulseqlib_ShapeArbitrary decompTime;

    /* No gradient on this channel */
    if (!gradDef) {
        return 2;
    }
    
    /* Initialize */
    count = 0;
    decompTime.samples = NULL;

    /* Parse rasters */
    gradRaster_us = seqDesc->gradRasterTime_us;
    blockDurationRaster_us = seqDesc->blockDurationRaster_us;
    numSamples = gradDef->fallTimeOrNumUncompressedSamples;
        /* Get delay */
    delay_us = (float)gradDef->delay;

    /* If delay > block_duration_raster, write zero at block start */
    if (delay_us > 0.0f) {
        count++;
    }
    
    if (gradDef->type == 0) {
        /* Trapezoid */
        riseTime_us = (float)gradDef->riseTimeOrUnused;
        flatTime_us = (float)gradDef->flatTimeOrUnused;
        fallTime_us = (float)gradDef->fallTimeOrNumUncompressedSamples;
        duration_us = delay_us + riseTime_us + flatTime_us + fallTime_us;
        
        if (flatTime_us > 0) {
            count += 4; /* 4 points: start, rise, flat, fall */
        } else {
            count += 3; /* 3 points: start, rise, fall */
        }            
    } else {        
        if (decompressShape(&seqDesc->shapes[gradDef->unusedOrTimeShapeID - 1], &decompTime, gradRaster_us)) {
            duration_us = delay_us + decompTime.samples[decompTime.numUncompressedSamples - 1];
        } else {
            duration_us = delay_us + 0.5f * gradRaster_us + gradRaster_us * (float)(numSamples - 1);
        }  

        /* Free decompressed shapes */
        if (decompTime.samples) {
            FREE(decompTime.samples);
        }

        count += numSamples;
    }

    /* Add block-end point if waveform ends before block ends */
    if (duration_us > blockDuration_us) {
        count++;
    }
    
    return count;
}

/**
 * @brief Compute position-specific max amplitudes for acoustic worst-case analysis.
 * 
 * For each position within the TR, each gradient axis, and each shot index,
 * finds the maximum |amplitude| across all TR repetitions.
 * 
 * Arrays are sized trSize * MAX_GRAD_SHOTS, indexed as [posInTR * MAX_GRAD_SHOTS + shotIdx]
 */
static int compute_position_max_amplitudes(
    const pulseqlib_SequenceDescriptor* seqDesc,
    float* posMaxAmpGx,
    float* posMaxAmpGy,
    float* posMaxAmpGz)
{
    const pulseqlib_TRdescriptor* trDesc;
    int trStart, trSize, numTRs;
    int trIdx, posInTR, blockIdx;
    const pulseqlib_BlockTableElement* blockTableEntry;
    const pulseqlib_GradTableElement* gradTable;
    float absAmp;
    int gRawID;
    int n, shotIdx, arrIdx;
    
    trDesc = &seqDesc->trDescriptor;
    trSize = trDesc->trSize;
    numTRs = trDesc->numTRs;
    
    /* Initialize to zero (trSize * MAX_GRAD_SHOTS elements per axis) */
    for (n = 0; n < trSize * MAX_GRAD_SHOTS; ++n) {
        posMaxAmpGx[n] = 0.0f;
        posMaxAmpGy[n] = 0.0f;
        posMaxAmpGz[n] = 0.0f;
    }
    
    /* Iterate over all TR repetitions */
    for (trIdx = 0; trIdx < numTRs; ++trIdx) {
        trStart = trDesc->numPrepBlocks + trIdx * trSize;
        
        /* Iterate over each position within the TR */
        for (posInTR = 0; posInTR < trSize; ++posInTR) {
            blockIdx = trStart + posInTR;
            blockTableEntry = &seqDesc->blockTable[blockIdx];
            
            /* Gx */
            gRawID = blockTableEntry->gxID;
            if (gRawID >= 0 && gRawID < seqDesc->gradTableSize) {
                gradTable = &seqDesc->gradTable[gRawID];
                shotIdx = gradTable->shotIndex;
                if (shotIdx >= 0 && shotIdx < MAX_GRAD_SHOTS) {
                    absAmp = gradTable->amplitude;
                    if (absAmp < 0.0f) absAmp = -absAmp;
                    arrIdx = posInTR * MAX_GRAD_SHOTS + shotIdx;
                    if (absAmp > posMaxAmpGx[arrIdx]) {
                        posMaxAmpGx[arrIdx] = absAmp;
                    }
                }
            }
            
            /* Gy */
            gRawID = blockTableEntry->gyID;
            if (gRawID >= 0 && gRawID < seqDesc->gradTableSize) {
                gradTable = &seqDesc->gradTable[gRawID];
                shotIdx = gradTable->shotIndex;
                if (shotIdx >= 0 && shotIdx < MAX_GRAD_SHOTS) {
                    absAmp = gradTable->amplitude;
                    if (absAmp < 0.0f) absAmp = -absAmp;
                    arrIdx = posInTR * MAX_GRAD_SHOTS + shotIdx;
                    if (absAmp > posMaxAmpGy[arrIdx]) {
                        posMaxAmpGy[arrIdx] = absAmp;
                    }
                }
            }
            
            /* Gz */
            gRawID = blockTableEntry->gzID;
            if (gRawID >= 0 && gRawID < seqDesc->gradTableSize) {
                gradTable = &seqDesc->gradTable[gRawID];
                shotIdx = gradTable->shotIndex;
                if (shotIdx >= 0 && shotIdx < MAX_GRAD_SHOTS) {
                    absAmp = gradTable->amplitude;
                    if (absAmp < 0.0f) absAmp = -absAmp;
                    arrIdx = posInTR * MAX_GRAD_SHOTS + shotIdx;
                    if (absAmp > posMaxAmpGz[arrIdx]) {
                        posMaxAmpGz[arrIdx] = absAmp;
                    }
                }
            }
        }
    }
    
    return PULSEQLIB_OK;
}

static int fill_grad_waveform_for_block(
    const pulseqlib_GradDefinition* gradDef,
    const pulseqlib_GradTableElement* gradTableEntry,
    const pulseqlib_SequenceDescriptor* seqDesc,
    float t0,
    const float* positionMaxAmp,
    const float blockDuration_us,
    float* time,
    float* waveform,
    int startIdx)
{
    int i, idx;
    float sign, maxAmp;
    float delay_us;
    int shapeId, timeShapeId, shotIdx;
    int numSamples;
    float riseTime_us, flatTime_us, fallTime_us;
    float t_sample;
    float gradRaster_us;
    float blockDurationRaster_us;
    float blockEnd_us;
    float blockStart_us;
    float lastWrittenTime;
    pulseqlib_ShapeArbitrary decompWave;
    pulseqlib_ShapeArbitrary decompTime;
    int hasTimeShape;
    
    idx = startIdx;
    gradRaster_us = seqDesc->gradRasterTime_us;
    blockDurationRaster_us = seqDesc->blockDurationRaster_us;
    blockStart_us = t0;
    blockEnd_us = t0 + blockDuration_us;
    decompWave.samples = NULL;
    decompTime.samples = NULL;
    
    /* No gradient on this channel */
    if (!gradDef || !gradTableEntry) 
    {
        /* Write zero point at block start (if not duplicate of previous) */
        time[idx] = blockStart_us;
        waveform[idx] = 0.0f;
        idx++;
        
        /* Always write zero point at block end (should never be same as blockStart) */
        time[idx] = blockEnd_us;
        waveform[idx] = 0.0f;
        idx++;
        
        return idx - startIdx;
    }

    /* Initialize last written time to sample before block start */
    lastWrittenTime = t0;
    
    /* Get amplitude and sign from gradTable */
    sign = (gradTableEntry->amplitude >= 0.0f) ? 1.0f : -1.0f;
    shotIdx = gradTableEntry->shotIndex;
    maxAmp = positionMaxAmp[shotIdx];
    delay_us = (float)gradDef->delay;

    /* If delay > block_duration_raster, write zero at block start */
    if (delay_us > 0.0f) {
        t_sample = blockStart_us;
        time[idx] = t_sample;
        waveform[idx] = 0.0f;
        lastWrittenTime = t_sample;
        idx++;
    }
    
    if (gradDef->type == 0) {
        /* Trapezoid */
        riseTime_us = (float)gradDef->riseTimeOrUnused;
        flatTime_us = (float)gradDef->flatTimeOrUnused;
        fallTime_us = (float)gradDef->fallTimeOrNumUncompressedSamples;
                
        if (flatTime_us > 0) {
            /* Full trapezoid: 4 points */
            t_sample = blockStart_us + delay_us;
            time[idx] = t_sample;
            waveform[idx] = 0.0f;
            lastWrittenTime = t_sample;
            idx++;

            t_sample = blockStart_us + delay_us + riseTime_us;
            time[idx] = t_sample;
            waveform[idx] = sign * maxAmp;
            lastWrittenTime = t_sample;
            idx++;
            
            t_sample = blockStart_us + delay_us + riseTime_us + flatTime_us;
            time[idx] = t_sample;
            waveform[idx] = sign * maxAmp;
            lastWrittenTime = t_sample;
            idx++;
            
            t_sample = blockStart_us + delay_us + riseTime_us + flatTime_us + fallTime_us;
            time[idx] = t_sample;
            waveform[idx] = 0.0f;
            lastWrittenTime = t_sample;
            idx++;
        } else {
            /* Triangular: 3 points */
            t_sample = blockStart_us + delay_us;
            time[idx] = t_sample;
            waveform[idx] = 0.0f;
            lastWrittenTime = t_sample;
            idx++;

            t_sample = blockStart_us + delay_us + riseTime_us;
            time[idx] = t_sample;
            waveform[idx] = sign * maxAmp;
            lastWrittenTime = t_sample;
            idx++;
            
            t_sample = blockStart_us + delay_us + riseTime_us + fallTime_us;
            time[idx] = t_sample;
            waveform[idx] = 0.0f;
            lastWrittenTime = t_sample;
            idx++;
        }            
    } else {
        /* Arbitrary or Extended trapezoid */
        numSamples = gradDef->fallTimeOrNumUncompressedSamples;
        timeShapeId = gradDef->unusedOrTimeShapeID;
        shapeId = gradDef->shotShapeIDs[shotIdx];
        
        /* Decompress waveform shape from seqDesc->shapes (1-based ID) */
        if (shapeId <= 0 || shapeId > seqDesc->numShapes) {
            return 0;
        }
        if (!decompressShape(&seqDesc->shapes[shapeId - 1], &decompWave, 1.0f)) {
            return 0;
        }
        
        /* Decompress time shape if present */
        hasTimeShape = 0;
        if (timeShapeId > 0 && timeShapeId <= seqDesc->numShapes) {
            if (decompressShape(&seqDesc->shapes[timeShapeId - 1], &decompTime, gradRaster_us)) {
                hasTimeShape = 1;
            }
        }
                
        if (hasTimeShape) {
            /* Extended trapezoid: time shape stores edge times */
            for (i = 0; i < numSamples; ++i) {
                t_sample = blockStart_us + delay_us + decompTime.samples[i];
                time[idx] = t_sample;
                waveform[idx] = sign * maxAmp * decompWave.samples[i];
                lastWrittenTime = t_sample;
                idx++;
            }
        } else {
            /* Arbitrary gradient: uniform timing, samples at raster center */
            for (i = 0; i < numSamples; ++i) {
                t_sample = blockStart_us + delay_us + 0.5f * gradRaster_us + (float)i * gradRaster_us;
                time[idx] = t_sample;
                waveform[idx] = sign * maxAmp * decompWave.samples[i];
                lastWrittenTime = t_sample;
                idx++;
            }
        }
                
        /* Free decompressed shapes */
        if (decompWave.samples) {
            FREE(decompWave.samples);
        }
        if (decompTime.samples) {
            FREE(decompTime.samples);
        }
    }

    /* Add block-end point if waveform ends before block ends */
    if (blockEnd_us > lastWrittenTime) {
        time[idx] = blockEnd_us;
        waveform[idx] = 0.0f;
        idx++;
    }
    
    return idx - startIdx;
}

static int interpolate_waveform_to_uniform(
    float** time,
    float** waveform,
    int* numSamples,
    float targetRaster_us)
{
    float* timeIn;
    float* waveformIn;
    float* timeOut = NULL;
    float* waveformOut = NULL;
    int numIn, numOut;
    float tStart, tEnd, duration;
    int i;
    
    if (!time || !waveform || !numSamples || *numSamples <= 0) {
        return PULSEQLIB_OK; /* Nothing to do for empty waveform */
    }
    
    timeIn = *time;
    waveformIn = *waveform;
    numIn = *numSamples;
    
    /* Get time range */
    tStart = timeIn[0];
    tEnd = timeIn[numIn - 1];
    duration = tEnd - tStart;
    
    if (duration <= 0.0f) {
        /* Single point or zero duration - nothing to interpolate */
        return PULSEQLIB_OK;
    }
    
    /* Compute number of uniform samples */
    numOut = (int)(duration / targetRaster_us) + 1;
    
    /* Allocate output arrays */
    timeOut = (float*)ALLOC(numOut * sizeof(float));
    waveformOut = (float*)ALLOC(numOut * sizeof(float));
    if (!timeOut || !waveformOut) {
        FREE(timeOut);
        FREE(waveformOut);
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }
    
    /* Generate uniform time grid */
    for (i = 0; i < numOut; ++i) {
        timeOut[i] = tStart + (float)i * targetRaster_us;
    }
    
    /* Interpolate waveform to uniform grid */
    /* interp1_linear(x_query, nx_query, xp_data, fp_data, nxp_data, out) */
    interp1_linear(timeOut, numOut, timeIn, waveformIn, numIn, waveformOut);
    
    /* Free old arrays and replace with new */
    FREE(timeIn);
    FREE(waveformIn);
    
    *time = timeOut;
    *waveform = waveformOut;
    *numSamples = numOut;
    
    return PULSEQLIB_OK;
}

/**
 * @brief Extract concatenated gradient waveforms for a single TR.
 */
int pulseqlib_getTRGradientWaveforms(
    const pulseqlib_SequenceDescriptor* seqDesc,
    pulseqlib_TRGradientWaveforms* waveforms,
    pulseqlib_Diagnostic* diag)
{
    pulseqlib_Diagnostic localDiag;
    const pulseqlib_TRdescriptor* trDesc;
    int trStart, trSize;
    int blockIdx, n;
    int totalSamplesGx, totalSamplesGy, totalSamplesGz;
    int idxGx, idxGy, idxGz;
    int result;
    float t0;
    int blockDefID;
    const pulseqlib_BlockDefinition* blockDef;
    const pulseqlib_BlockTableElement* blockTableEntry;
    float blockDuration_us;
    float targetRaster_us;
    int gxRawID, gyRawID, gzRawID;
    const pulseqlib_GradDefinition* gxDef;
    const pulseqlib_GradDefinition* gyDef;
    const pulseqlib_GradDefinition* gzDef;
    const pulseqlib_GradTableElement* gxTable;
    const pulseqlib_GradTableElement* gyTable;
    const pulseqlib_GradTableElement* gzTable;
    
    /* Position-specific max amplitudes */
    float* posMaxAmpGx = NULL;
    float* posMaxAmpGy = NULL;
    float* posMaxAmpGz = NULL;
    
    /* Use local diag if caller doesn't provide one */
    if (!diag) {
        pulseqlib_diagnosticInit(&localDiag);
        diag = &localDiag;
    } else {
        pulseqlib_diagnosticInit(diag);
    }
    
    /* Validate inputs */
    if (!seqDesc || !waveforms) {
        diag->code = PULSEQLIB_ERR_NULL_POINTER;
        return diag->code;
    }
    
    /* Initialize output */
    waveforms->numSamplesGx = 0;
    waveforms->numSamplesGy = 0;
    waveforms->numSamplesGz = 0;
    waveforms->timeGx = NULL;
    waveforms->timeGy = NULL;
    waveforms->timeGz = NULL;
    waveforms->waveformGx = NULL;
    waveforms->waveformGy = NULL;
    waveforms->waveformGz = NULL;
    
    /* Get TR info */
    trDesc = &seqDesc->trDescriptor;
    trSize = trDesc->trSize;
    
    if (trSize <= 0) {
        diag->code = PULSEQLIB_ERR_TR_NO_BLOCKS;
        return diag->code;
    }
    
    /* Calculate TR start block */
    trStart = trDesc->numPrepBlocks;
    if (trStart + trSize > seqDesc->numBlocks) {
        diag->code = PULSEQLIB_ERR_INVALID_ARGUMENT;
        return diag->code;
    }

    /* Allocate and compute position-specific max amplitudes */
    posMaxAmpGx = (float*)ALLOC(trSize * MAX_GRAD_SHOTS * sizeof(float));
    posMaxAmpGy = (float*)ALLOC(trSize * MAX_GRAD_SHOTS * sizeof(float));
    posMaxAmpGz = (float*)ALLOC(trSize * MAX_GRAD_SHOTS * sizeof(float));
    if (!posMaxAmpGx || !posMaxAmpGy || !posMaxAmpGz) {
        FREE(posMaxAmpGx);
        FREE(posMaxAmpGy);
        FREE(posMaxAmpGz);
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        return diag->code;
    }
    compute_position_max_amplitudes(seqDesc, posMaxAmpGx, posMaxAmpGy, posMaxAmpGz);
        
    /* ========== PASS 1: Count total samples ========== */
    totalSamplesGx = 0;
    totalSamplesGy = 0;
    totalSamplesGz = 0;
    
    for (n = 0; n < trSize; ++n) {
        blockIdx = trStart + n;
        blockTableEntry = &seqDesc->blockTable[blockIdx];
        blockDefID = blockTableEntry->ID;
        blockDef = &seqDesc->blockDefinitions[blockDefID];
        
        /* Get raw gradient table indices from blockTable */
        gxRawID = blockTableEntry->gxID;
        gyRawID = blockTableEntry->gyID;
        gzRawID = blockTableEntry->gzID;
        
        /* Get gradient table entries (contains amplitude and deduplicated ID) */
        gxTable = (gxRawID >= 0 && gxRawID < seqDesc->gradTableSize) ? 
                  &seqDesc->gradTable[gxRawID] : NULL;
        gyTable = (gyRawID >= 0 && gyRawID < seqDesc->gradTableSize) ? 
                  &seqDesc->gradTable[gyRawID] : NULL;
        gzTable = (gzRawID >= 0 && gzRawID < seqDesc->gradTableSize) ? 
                  &seqDesc->gradTable[gzRawID] : NULL;
        
        /* Get gradient definitions using table entry's ID */
        gxDef = (gxTable && gxTable->ID >= 0 && gxTable->ID < seqDesc->numUniqueGrads) ? 
                &seqDesc->gradDefinitions[gxTable->ID] : NULL;
        gyDef = (gyTable && gyTable->ID >= 0 && gyTable->ID < seqDesc->numUniqueGrads) ? 
                &seqDesc->gradDefinitions[gyTable->ID] : NULL;
        gzDef = (gzTable && gzTable->ID >= 0 && gzTable->ID < seqDesc->numUniqueGrads) ? 
                &seqDesc->gradDefinitions[gzTable->ID] : NULL;
        
        totalSamplesGx += count_grad_samples_for_block(seqDesc, gxDef, blockDuration_us);
        totalSamplesGy += count_grad_samples_for_block(seqDesc, gyDef, blockDuration_us);
        totalSamplesGz += count_grad_samples_for_block(seqDesc, gzDef, blockDuration_us);
    }
    
    /* ========== Allocate arrays ========== */
    waveforms->timeGx = (float*)ALLOC(totalSamplesGx * sizeof(float));
    waveforms->waveformGx = (float*)ALLOC(totalSamplesGx * sizeof(float));
    waveforms->timeGy = (float*)ALLOC(totalSamplesGy * sizeof(float));
    waveforms->waveformGy = (float*)ALLOC(totalSamplesGy * sizeof(float));
    waveforms->timeGz = (float*)ALLOC(totalSamplesGz * sizeof(float));
    waveforms->waveformGz = (float*)ALLOC(totalSamplesGz * sizeof(float));
    
    if (!waveforms->timeGx || !waveforms->waveformGx ||
        !waveforms->timeGy || !waveforms->waveformGy ||
        !waveforms->timeGz || !waveforms->waveformGz) {
        FREE(posMaxAmpGx);  /* Don't forget to free these! */
        FREE(posMaxAmpGy);
        FREE(posMaxAmpGz);
        pulseqlib_trGradientWaveformsFree(waveforms);
        diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
        return diag->code;
    }
    
    /* ========== PASS 2: Fill waveforms ========== */
    t0 = 0.0f;
    idxGx = 0;
    idxGy = 0;
    idxGz = 0;

    for (n = 0; n < trSize; ++n) {
        blockIdx = trStart + n;
        blockTableEntry = &seqDesc->blockTable[blockIdx];
        blockDefID = blockTableEntry->ID;
        blockDef = &seqDesc->blockDefinitions[blockDefID];
        
        /* Get block duration: from blockTable if pure delay (>= 0), else from blockDefinitions */
        blockDuration_us = (blockTableEntry->duration_us >= 0) ? (float)blockTableEntry->duration_us : (float)blockDef->duration_us;
        
        /* Get raw gradient table indices from blockTable */
        gxRawID = blockTableEntry->gxID;
        gyRawID = blockTableEntry->gyID;
        gzRawID = blockTableEntry->gzID;
        
        /* Get gradient table entries (contains amplitude and deduplicated ID) */
        gxTable = (gxRawID >= 0 && gxRawID < seqDesc->gradTableSize) ? 
                  &seqDesc->gradTable[gxRawID] : NULL;
        gyTable = (gyRawID >= 0 && gyRawID < seqDesc->gradTableSize) ? 
                  &seqDesc->gradTable[gyRawID] : NULL;
        gzTable = (gzRawID >= 0 && gzRawID < seqDesc->gradTableSize) ? 
                  &seqDesc->gradTable[gzRawID] : NULL;
        
        /* Get gradient definitions using table entry's ID */
        gxDef = (gxTable && gxTable->ID >= 0 && gxTable->ID < seqDesc->numUniqueGrads) ? 
                &seqDesc->gradDefinitions[gxTable->ID] : NULL;
        gyDef = (gyTable && gyTable->ID >= 0 && gyTable->ID < seqDesc->numUniqueGrads) ? 
                &seqDesc->gradDefinitions[gyTable->ID] : NULL;
        gzDef = (gzTable && gzTable->ID >= 0 && gzTable->ID < seqDesc->numUniqueGrads) ? 
                &seqDesc->gradDefinitions[gzTable->ID] : NULL;
        
        /* Fill Gx */
        idxGx += fill_grad_waveform_for_block(
            gxDef, gxTable, seqDesc,
            t0,
            &posMaxAmpGx[n * MAX_GRAD_SHOTS], blockDuration_us,
            waveforms->timeGx, waveforms->waveformGx, idxGx);
        
        /* Fill Gy */
        idxGy += fill_grad_waveform_for_block(
            gyDef, gyTable, seqDesc,
            t0,
            &posMaxAmpGy[n * MAX_GRAD_SHOTS], blockDuration_us,
            waveforms->timeGy, waveforms->waveformGy, idxGy);
        
        /* Fill Gz */
        idxGz += fill_grad_waveform_for_block(
            gzDef, gzTable, seqDesc,
            t0,
            &posMaxAmpGz[n * MAX_GRAD_SHOTS], blockDuration_us,
            waveforms->timeGz, waveforms->waveformGz, idxGz);
        
        /* Update time offset for next block */
        t0 += blockDuration_us;
    }

    /* Cleanup */
    FREE(posMaxAmpGx);
    FREE(posMaxAmpGy);
    FREE(posMaxAmpGz);
    
    /* Store preliminary counts */
    waveforms->numSamplesGx = idxGx;
    waveforms->numSamplesGy = idxGy;
    waveforms->numSamplesGz = idxGz;

    /* Interpolate to uniform raster (0.5 * gradient raster time) */
    targetRaster_us = 0.5f * seqDesc->gradRasterTime_us;
    
    result = interpolate_waveform_to_uniform(
        &waveforms->timeGx, &waveforms->waveformGx, &waveforms->numSamplesGx, targetRaster_us);
    if (PULSEQLIB_FAILED(result)) {
        pulseqlib_trGradientWaveformsFree(waveforms);
        diag->code = result;
        return result;
    }
    
    result = interpolate_waveform_to_uniform(
        &waveforms->timeGy, &waveforms->waveformGy, &waveforms->numSamplesGy, targetRaster_us);
    if (PULSEQLIB_FAILED(result)) {
        pulseqlib_trGradientWaveformsFree(waveforms);
        diag->code = result;
        return result;
    }
    
    result = interpolate_waveform_to_uniform(
        &waveforms->timeGz, &waveforms->waveformGz, &waveforms->numSamplesGz, targetRaster_us);
    if (PULSEQLIB_FAILED(result)) {
        pulseqlib_trGradientWaveformsFree(waveforms);
        diag->code = result;
        return result;
    }

    diag->code = PULSEQLIB_OK;
    return PULSEQLIB_OK;
}

typedef struct AcousticSpectrumSupport {
    int nwin;              /**< Window size (number of samples) */
    int nfft;              /**< FFT size (nwin * oversampling, rounded to power of 2) */
    int nfreq;             /**< Number of frequency bins (nfft/2 + 1) */
    int outputFreqBins;    /**< Number of frequency bins in output (may be less than nfreq) */
    int numWindows;        /**< Total number of windows for this waveform */
    int hopSize;           /**< Hop size between windows (nwin / 2) */
    float gradRaster_us;   /**< Gradient raster time in us */
    float freqResolution;  /**< Frequency resolution in Hz */
    float maxFrequency_Hz; /**< Maximum frequency to include in output (Hz, -1 for all) */
    float* cosWindow;      /**< Precomputed cosine taper window (size: nwin) */
    float* workBuffer;     /**< Working buffer for windowed/padded data (size: nfft) */
    kiss_fftr_cfg fftCfg;  /**< KissFFT real-FFT configuration */
    kiss_fft_cpx* fftOut;  /**< FFT output buffer (size: nfreq) */
} AcousticSpectrumSupport;

#define ACOUSTIC_SPECTRUM_SUPPORT_INIT {0, 0, 0, 0, 0, 0, 0.0f, 0.0f, 0.0f, NULL, NULL, NULL, NULL}

typedef struct AcousticWaveform {
    int numSamples;        /**< Number of samples (padded to integer multiple of nwin, or original if < nwin) */
    int numSamplesOriginal;/**< Original number of samples before padding */
    float* samples;        /**< Waveform samples (may be padded copy or alias) */
    int ownsMemory;        /**< Non-zero if samples was allocated and needs freeing */
} AcousticWaveform;

#define ACOUSTIC_WAVEFORM_INIT {0, 0, NULL, 0}

/* ============== Acoustic Spectrum Analysis Implementation ============== */
static int acousticSpectrumSupportInit(
    AcousticSpectrumSupport* support,
    int numSamples,
    int targetWindowSize,
    float targetSpectralResolution_Hz,
    float gradRaster_us,
    float maxFrequency_Hz)
{
    int nwin, nfft, nfreq, outputFreqBins;
    int hopSize, numWindows;
    int paddedLen;
    int maxIdx;
    int minNfftForResolution;
    float* cosWindow = NULL;
    float* workBuffer = NULL;
    kiss_fft_cpx* fftOut = NULL;
    kiss_fftr_cfg fftCfg = NULL;
    float freqResolution;
    float maxFreqBand;
    int i;
    
    if (!support || numSamples <= 0 || targetWindowSize <= 0 || targetSpectralResolution_Hz <= 0.0f) {
        return PULSEQLIB_ERR_INVALID_ARGUMENT;
    }
    
    /* Initialize to safe state */
    memset(support, 0, sizeof(*support));
    
    /* Determine window size: use target if TR is long enough, else use TR length */
    if (numSamples >= targetWindowSize) {
        nwin = targetWindowSize;
    } else {
        nwin = numSamples;
    }
    
    /* Determine frequency band of interest */
    if (maxFrequency_Hz < 0.0f) {
        /* Full Nyquist bandwidth: 0 to fs/2 = 0 to (1e6 / (2 * gradRaster_us)) Hz */
        maxFreqBand = 5.0e5f / gradRaster_us;  /* Hz */
    } else {
        maxFreqBand = maxFrequency_Hz;
    }
    
    /* Calculate minimum nfft needed to achieve target spectral resolution in frequency band:
     * freqResolution = samplingRate / nfft = (1e6 / gradRaster_us) / nfft
     * Therefore: nfft = (1e6 / gradRaster_us) / targetSpectralResolution_Hz
     */
    minNfftForResolution = (int)ceil((double)(1.0e6f / (gradRaster_us * targetSpectralResolution_Hz)));
    
    /* Ensure nfft is at least nwin (can't zero-pad to smaller than window) */
    if (minNfftForResolution < nwin) {
        nfft = nwin;
    } else {
        /* Round up to next power of 2 for FFT efficiency */
        nfft = next_pow2(minNfftForResolution);
    }
    
    /* Number of frequency bins for real FFT */
    nfreq = nfft / 2 + 1;
    
    /* Actual frequency resolution achieved */
    freqResolution = 1.0e6f / (gradRaster_us * (float)nfft); /* Hz */
    
    /* Determine output frequency bins based on maxFrequency_Hz */
    if (maxFrequency_Hz < 0.0f) {
        /* User wants full spectrum */
        outputFreqBins = nfreq;
    } else {
        /* Find index of frequency closest to maxFrequency_Hz */
        maxIdx = (int)(maxFrequency_Hz / freqResolution + 0.5f);
        if (maxIdx >= nfreq) {
            outputFreqBins = nfreq;
        } else if (maxIdx < 1) {
            outputFreqBins = 1; /* At least DC bin */
        } else {
            outputFreqBins = maxIdx + 1; /* +1 because we include bin at maxIdx */
        }
    }
    
    /* Hop size (50% overlap) */
    hopSize = nwin / 2;
    if (hopSize < 1) hopSize = 1;
    
    /* Number of windows (with 50% overlap) */
    if (numSamples <= nwin) {
        numWindows = 1;
    } else {
        /* Pad to integer multiple of nwin, then count windows with hopSize */
        paddedLen = ((numSamples + nwin - 1) / nwin) * nwin;
        numWindows = (paddedLen - nwin) / hopSize + 1;
    }
    
    /* Allocate cosine taper window */
    cosWindow = (float*)ALLOC(nwin * sizeof(float));
    if (!cosWindow) {
        goto cleanup_error;
    }
    
    /* Compute cosine taper: 0.5 * (1 - cos(2*pi*i/nwin)) for i=1..nwin */
    for (i = 0; i < nwin; ++i) {
        cosWindow[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (float)(i + 1) / (float)nwin));
    }
    
    /* Allocate work buffer (for windowed + zero-padded data) */
    workBuffer = (float*)ALLOC(nfft * sizeof(float));
    if (!workBuffer) {
        goto cleanup_error;
    }
    
    /* Allocate FFT output buffer (full size, we'll subset when copying out) */
    fftOut = (kiss_fft_cpx*)ALLOC(nfreq * sizeof(kiss_fft_cpx));
    if (!fftOut) {
        goto cleanup_error;
    }
    
    /* Initialize KissFFT for real-to-complex forward transform */
    fftCfg = kiss_fftr_alloc(nfft, 0, NULL, NULL);
    if (!fftCfg) {
        goto cleanup_error;
    }
    
    /* Fill support structure */
    support->nwin = nwin;
    support->nfft = nfft;
    support->nfreq = nfreq;
    support->outputFreqBins = outputFreqBins;
    support->numWindows = numWindows;
    support->hopSize = hopSize;
    support->gradRaster_us = gradRaster_us;
    support->freqResolution = freqResolution;
    support->maxFrequency_Hz = maxFrequency_Hz;
    support->cosWindow = cosWindow;
    support->workBuffer = workBuffer;
    support->fftCfg = fftCfg;
    support->fftOut = fftOut;
    
    return PULSEQLIB_OK;

cleanup_error:
    if (cosWindow) FREE(cosWindow);
    if (workBuffer) FREE(workBuffer);
    if (fftOut) FREE(fftOut);
    if (fftCfg) kiss_fftr_free(fftCfg);
    return PULSEQLIB_ERR_ALLOC_FAILED;
}

void acousticSpectrumSupportFree(AcousticSpectrumSupport* support)
{
    if (!support) return;
    
    if (support->cosWindow) {
        FREE(support->cosWindow);
        support->cosWindow = NULL;
    }
    if (support->workBuffer) {
        FREE(support->workBuffer);
        support->workBuffer = NULL;
    }
    if (support->fftOut) {
        FREE(support->fftOut);
        support->fftOut = NULL;
    }
    if (support->fftCfg) {
        kiss_fftr_free(support->fftCfg);
        support->fftCfg = NULL;
    }
    
    memset(support, 0, sizeof(*support));
}

int acousticWaveformInit(
    AcousticWaveform* acoustic,
    const AcousticSpectrumSupport* support,
    const float* waveform,
    int numSamples,
    int targetPaddedLen)
{
    float* paddedSamples = NULL;
    int i;
    
    if (!acoustic || !support || !waveform || numSamples <= 0) {
        return PULSEQLIB_ERR_NULL_POINTER;
    }
    
    /* Initialize to safe state */
    memset(acoustic, 0, sizeof(*acoustic));
    acoustic->numSamplesOriginal = numSamples;
    
    /* Always allocate and pad to targetPaddedLen to ensure all windows can be computed */
    paddedSamples = (float*)ALLOC(targetPaddedLen * sizeof(float));
    if (!paddedSamples) {
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }
    
    /* Copy original samples */
    for (i = 0; i < numSamples; ++i) {
        paddedSamples[i] = waveform[i];
    }
    
    /* Zero-pad the rest */
    for (i = numSamples; i < targetPaddedLen; ++i) {
        paddedSamples[i] = 0.0f;
    }
    
    acoustic->numSamples = targetPaddedLen;
    acoustic->samples = paddedSamples;
    acoustic->ownsMemory = 1;
    
    return PULSEQLIB_OK;
}

void acousticWaveformFree(AcousticWaveform* acoustic)
{
    if (!acoustic) return;
    
    if (acoustic->ownsMemory && acoustic->samples) {
        FREE(acoustic->samples);
    }
    
    memset(acoustic, 0, sizeof(*acoustic));
}

int computeWindowSpectrum(
    float* spectrum,
    AcousticSpectrumSupport* support,
    const AcousticWaveform* acoustic,
    int windowIndex)
{
    int i;
    int startIdx;
    int nwin, nfft, nfreq, outputFreqBins;
    float* workBuffer;
    float* cosWindow;
    const float* samples;
    float mean;
    float fftNorm;
    kiss_fft_cpx* fftOut;
    
    if (!spectrum || !support || !acoustic || !acoustic->samples) {
        return PULSEQLIB_ERR_NULL_POINTER;
    }
    
    if (windowIndex < 0 || windowIndex >= support->numWindows) {
        return PULSEQLIB_ERR_INVALID_ARGUMENT;
    }
    
    nwin = support->nwin;
    nfft = support->nfft;
    nfreq = support->nfreq;
    outputFreqBins = support->outputFreqBins;
    workBuffer = support->workBuffer;
    cosWindow = support->cosWindow;
    fftOut = support->fftOut;
    samples = acoustic->samples;
    
    /* Calculate start index for this window */
    startIdx = windowIndex * support->hopSize;
    
    /* Ensure we don't read past the end */
    if (startIdx + nwin > acoustic->numSamples) {
        /* Shouldn't happen if numWindows is computed correctly, but handle it */
        return PULSEQLIB_ERR_INVALID_ARGUMENT;
    }
    
    /* Step 1: Copy window samples */
    for (i = 0; i < nwin; ++i) {
        workBuffer[i] = samples[startIdx + i];
    }
    
    /* Step 2: Zero-fill from nwin to nfft */
    for (i = nwin; i < nfft; ++i) {
        workBuffer[i] = 0.0f;
    }
    
    /* Step 3: Compute and subtract mean of window samples */
    mean = 0.0f;
    for (i = 0; i < nwin; ++i) {
        mean += workBuffer[i];
    }
    mean /= (float)nwin;
    for (i = 0; i < nwin; ++i) {
        workBuffer[i] -= mean;
    }
    
    /* Step 4: Apply cosine taper to window samples (not zero-filled region) */
    for (i = 0; i < nwin; ++i) {
        workBuffer[i] *= cosWindow[i];
    }
    
    /* Step 5: Compute real FFT */
    kiss_fftr(support->fftCfg, workBuffer, fftOut);

    /* Normalize FFT output by nfft for proper amplitude scaling */
    fftNorm = 1.0f / (float)nfft;
    for (i = 0; i < nfreq; ++i) {
        fftOut[i].r *= fftNorm;
        fftOut[i].i *= fftNorm;
    }
    
    /* Step 6: Compute magnitude spectrum and copy only requested frequency range to output */
    for (i = 0; i < outputFreqBins; ++i) {
        spectrum[i] = (float)sqrt((double)(fftOut[i].r * fftOut[i].r + fftOut[i].i * fftOut[i].i));
    }
    
    return PULSEQLIB_OK;
}

void pulseqlib_trAcousticSpectraFree(pulseqlib_TRAcousticSpectra* spectra)
{
    if (!spectra) return;
    
    if (spectra->frequencies) {
        FREE(spectra->frequencies);
        spectra->frequencies = NULL;
    }
    if (spectra->spectraGx) {
        FREE(spectra->spectraGx);
        spectra->spectraGx = NULL;
    }
    if (spectra->spectraGy) {
        FREE(spectra->spectraGy);
        spectra->spectraGy = NULL;
    }
    if (spectra->spectraGz) {
        FREE(spectra->spectraGz);
        spectra->spectraGz = NULL;
    }

    if (spectra->maxEnvelopeGx) {
        FREE(spectra->maxEnvelopeGx);
        spectra->maxEnvelopeGx = NULL;
    }
    if (spectra->maxEnvelopeGy) {
        FREE(spectra->maxEnvelopeGy);
        spectra->maxEnvelopeGy = NULL;
    }
    if (spectra->maxEnvelopeGz) {
        FREE(spectra->maxEnvelopeGz);
        spectra->maxEnvelopeGz = NULL;
    }
    
    /* Free peak arrays for sliding window */
    if (spectra->peaksGx) {
        FREE(spectra->peaksGx);
        spectra->peaksGx = NULL;
    }
    if (spectra->peaksGy) {
        FREE(spectra->peaksGy);
        spectra->peaksGy = NULL;
    }
    if (spectra->peaksGz) {
        FREE(spectra->peaksGz);
        spectra->peaksGz = NULL;
    }
    
    if (spectra->frequenciesFull) {
        FREE(spectra->frequenciesFull);
        spectra->frequenciesFull = NULL;
    }
    if (spectra->spectraGxFull) {
        FREE(spectra->spectraGxFull);
        spectra->spectraGxFull = NULL;
    }
    if (spectra->spectraGyFull) {
        FREE(spectra->spectraGyFull);
        spectra->spectraGyFull = NULL;
    }
    if (spectra->spectraGzFull) {
        FREE(spectra->spectraGzFull);
        spectra->spectraGzFull = NULL;
    }
    
    if (spectra->frequenciesSeq) {
        FREE(spectra->frequenciesSeq);
        spectra->frequenciesSeq = NULL;
    }
    if (spectra->spectraGxSeq) {
        FREE(spectra->spectraGxSeq);
        spectra->spectraGxSeq = NULL;
    }
    if (spectra->spectraGySeq) {
        FREE(spectra->spectraGySeq);
        spectra->spectraGySeq = NULL;
    }
    if (spectra->spectraGzSeq) {
        FREE(spectra->spectraGzSeq);
        spectra->spectraGzSeq = NULL;
    }
    
    /* Free peak arrays for sequence spectra */
    if (spectra->peaksGxSeq) {
        FREE(spectra->peaksGxSeq);
        spectra->peaksGxSeq = NULL;
    }
    if (spectra->peaksGySeq) {
        FREE(spectra->peaksGySeq);
        spectra->peaksGySeq = NULL;
    }
    if (spectra->peaksGzSeq) {
        FREE(spectra->peaksGzSeq);
        spectra->peaksGzSeq = NULL;
    }
}

/* ============== Acoustic Resonance Detection ============== */
#define PEAK_LOG10_THRESHOLD    2.25f   /* Threshold in log10 domain */
#define PEAK_NORM_SCALE         10.0f   /* Normalization scale factor */
#define PEAK_EPS                1e-30f  /* Epsilon to avoid log(0) - placed INSIDE log */

/**
 * @brief Detect resonance peaks using normalized log-scale criterion
 * 
 * Original criterion (exactly as implemented):
 *   norm[i] = (spectrum[i] / max(spectrum) + eps) * 10.0
 *   criterion[i] = log10(norm[i]) - mean(log10(norm))
 *   peak if criterion[i] > 1.75
 * 
 * Note: eps is placed inside the normalization (before log), not outside,
 * to match the exact behavior of your empirical formula.
 * 
 * @param n      Number of frequency bins
 * @param mag    Input spectrum (magnitude)
 * @param peaks  Output: 1 where peak detected, 0 otherwise
 */
static void detect_resonances(int n, const float *mag, int *peaks) {
    int i;
    float maxVal = 0.0f;
    float sumLog = 0.0f;
    float meanLog;
    float norm;
    float logVal;
    
    /* Initialize output */
    if (n <= 0 || !mag || !peaks) return;
    
    for (i = 0; i < n; ++i) {
        peaks[i] = 0;
    }
    
    if (n < 3) return;
    
    /* Find max */
    for (i = 0; i < n; ++i) {
        if (mag[i] > maxVal) maxVal = mag[i];
    }
    
    if (maxVal <= 0.0f) return;
    
    /* First pass: compute normalized spectrum and sum of logs */
    for (i = 0; i < n; ++i) {
        norm = (mag[i] / maxVal + PEAK_EPS) * PEAK_NORM_SCALE;
        sumLog += (float)log10((double)norm);
    }
    
    meanLog = sumLog / (float)n;
    
    /* Second pass: find local maxima above threshold */
    for (i = 1; i < n - 1; ++i) {
        if (mag[i] > mag[i-1] && mag[i] > mag[i+1]) {
            norm = (mag[i] / maxVal + PEAK_EPS) * PEAK_NORM_SCALE;
            logVal = (float)log10((double)norm);
            
            if (logVal - meanLog > PEAK_LOG10_THRESHOLD) {
                peaks[i] = 1;
            }
        }
    }
}

static int check_acoustic_violations(
    const float* spectrum,
    const float* frequencies,
    int numFreqBins,
    float maxEnvelope,
    const pulseqlib_ForbiddenBand* forbiddenBands,
    int numBands,
    pulseqlib_AcousticViolation* violation,
    int** outPeaks)
{
    int* peaks = NULL;
    int i, b;
    float freq;
    float peakMagnitude;
    float worstPeakFreq;
    int worstBandIndex;
    float worstPeakMagnitude;
    
    /* Initialize violation result */
    violation->detected = 0;
    violation->bandIndex = -1;
    violation->peakFrequency_Hz = 0.0f;
    violation->maxAmplitude = maxEnvelope;
    violation->allowedAmplitude = 0.0f;
    
    /* No bands to check - no violation possible */
    if (numBands <= 0 || !forbiddenBands) {
        if (outPeaks) {
            *outPeaks = NULL;  /* ADD THIS LINE */
        }
        return PULSEQLIB_OK;
    }
    
    /* Allocate peaks array */
    peaks = (int*)ALLOC((size_t)numFreqBins * sizeof(int));
    if (!peaks) {
        return PULSEQLIB_ERR_ALLOC_FAILED;
    }
    
    /* Detect peaks in spectrum */
    detect_resonances(numFreqBins, spectrum, peaks);
    
    /* Find the worst peak in any forbidden band */
    worstPeakFreq = 0.0f;
    worstBandIndex = -1;
    worstPeakMagnitude = 0.0f;
    
    for (i = 0; i < numFreqBins; ++i) {
        if (!peaks[i]) {
            continue;  /* Not a peak */
        }
        
        freq = frequencies[i];
        peakMagnitude = spectrum[i];
        
        /* Check if this peak falls in any forbidden band */
        for (b = 0; b < numBands; ++b) {
            if (freq >= forbiddenBands[b].freqMin_Hz && freq <= forbiddenBands[b].freqMax_Hz) {
                /* Peak is in forbidden band - check if it's the worst one so far */
                if (peakMagnitude > worstPeakMagnitude) {
                    worstPeakMagnitude = peakMagnitude;
                    worstPeakFreq = freq;
                    worstBandIndex = b;
                }
                break;  /* Found a band, no need to check more bands for this peak */
            }
        }
    }
    
    /* Return peaks to caller if requested, otherwise free them */
    if (outPeaks) {
        *outPeaks = peaks;
    } else {
        FREE(peaks);
    }
    
    /* If we found a peak in a forbidden band, check amplitude violation */
    if (worstBandIndex >= 0) {
        violation->peakFrequency_Hz = worstPeakFreq;
        violation->bandIndex = worstBandIndex;
        violation->allowedAmplitude = forbiddenBands[worstBandIndex].maxAmplitude;
        
        /* Check if max amplitude exceeds allowed */
        if (maxEnvelope > forbiddenBands[worstBandIndex].maxAmplitude) {
            violation->detected = 1;
        }
    }
    
    return PULSEQLIB_OK;
}

static int compute_sliding_window_spectra(
    float* spectraOut,
    AcousticSpectrumSupport* support,
    const float* waveform,
    const float* frequencies,
    int numSamples,
    int paddedLen,
    int combined,
    float* outMaxEnvelope,
    int numForbiddenBands,
    const pulseqlib_ForbiddenBand* forbiddenBands,
    int* outPeaks
) {
    AcousticWaveform acoustic;
    int w, i, result;
    int startIdx;
    int windowLen;
    float* windowSpectrum = NULL;
    float maxEnvWindow;
    float absVal;
    pulseqlib_AcousticViolation violation;
    int* windowPeaks;
    float maxEnvOverall = 0.0f;

    memset(&acoustic, 0, sizeof(acoustic));
    
    /* If combined mode, we need a temporary buffer for each window */
    if (combined) {
        windowSpectrum = (float*)ALLOC(support->outputFreqBins * sizeof(float));
        if (!windowSpectrum) {
            return PULSEQLIB_ERR_ALLOC_FAILED;
        }
        
        /* Initialize output to zero (will be updated with max) */
        for (i = 0; i < support->outputFreqBins; ++i) {
            spectraOut[i] = 0.0f;
        }
    }
        
    /* Preprocess waveform (pad if needed) */
    result = acousticWaveformInit(&acoustic, support, waveform, numSamples, paddedLen);
    if (PULSEQLIB_FAILED(result)) {
        if (windowSpectrum) FREE(windowSpectrum);
        return result;
    }

    /* Compute spectrum for each window */
    for (w = 0; w < support->numWindows; ++w) {

        /* Calculate start index and length for this window */
        startIdx = w * support->hopSize;
        windowLen = support->nwin;
        if (startIdx + windowLen > acoustic.numSamples) {
            windowLen = acoustic.numSamples - startIdx;
        }
        
        /* Compute max envelope for this window */
        maxEnvWindow = 0.0f;
        for (i = startIdx; i < startIdx + windowLen; ++i) {
            absVal = (acoustic.samples[i] >= 0.0f) ? acoustic.samples[i] : -acoustic.samples[i];
            if (absVal > maxEnvWindow) {
                maxEnvWindow = absVal;
            }
        }

        if (combined) {
            /* Track overall maximum */
            if (maxEnvWindow > maxEnvOverall) {
                maxEnvOverall = maxEnvWindow;
            }

            /* Compute to temporary buffer */
            result = computeWindowSpectrum(windowSpectrum, support, &acoustic, w);
            if (PULSEQLIB_FAILED(result)) {
                FREE(windowSpectrum);
                acousticWaveformFree(&acoustic);
                return result;
            }
            
            /* Update output with pointwise maximum */
            for (i = 0; i < support->outputFreqBins; ++i) {
                if (windowSpectrum[i] > spectraOut[i]) {
                    spectraOut[i] = windowSpectrum[i];
                }
            }
        } else {
            /* Store per-window max envelope */
            if (outMaxEnvelope) {
                outMaxEnvelope[w] = maxEnvWindow;
            }

            /* Compute directly to output buffer at appropriate offset */
            result = computeWindowSpectrum(
                &spectraOut[w * support->outputFreqBins],
                support,
                &acoustic,
                w);

            /* Check violations in current spectrum */
            if (numForbiddenBands > 0 && forbiddenBands) {
                windowPeaks = NULL;
                result = check_acoustic_violations(
                    &spectraOut[w * support->outputFreqBins],
                    frequencies,
                    support->outputFreqBins,
                    maxEnvWindow,
                    forbiddenBands,
                    numForbiddenBands,
                    &violation,
                    outPeaks ? &windowPeaks : NULL);
                if (PULSEQLIB_FAILED(result)) {
                    if (windowSpectrum) FREE(windowSpectrum);
                    acousticWaveformFree(&acoustic);
                    return result;
                }            
                
                /* Copy peaks to output array if requested */
                if (windowPeaks && outPeaks) {
                    memcpy(&outPeaks[w * support->outputFreqBins], windowPeaks, support->outputFreqBins * sizeof(int));
                    FREE(windowPeaks);
                }
            } else if (outPeaks) {
                detect_resonances(support->outputFreqBins, &spectraOut[w * support->outputFreqBins], &outPeaks[w * support->outputFreqBins]);
            }
        }
    }

    /* Store combined max envelope */
    if (combined && outMaxEnvelope) {
        outMaxEnvelope[0] = maxEnvOverall;
    }
    
    if (windowSpectrum) FREE(windowSpectrum);
    acousticWaveformFree(&acoustic);
    return PULSEQLIB_OK;
}

static int compute_sequence_spectrum(
    float* fullSpectrum,
    float** seqSpectrum,
    float** seqFrequencies,
    const float* waveform,
    int numSamples,
    float gradRasterTime_us,
    float targetSpectralResolution_Hz,
    float maxFrequency,
    float fundamentalFreq,
    int numTRs,
    int* outNumFreqBinsFull,
    float* outFreqResolutionFull,
    int* outNumPicked,
    float* outMaxEnvelope,
    int numForbiddenBands,
    const pulseqlib_ForbiddenBand* forbiddenBands,
    int** outSeqPeaks
) {
    /* All declarations at the top for C89 compliance */
    int nfft, nfreq, outputFreqBinsFull, numPicked;
    int minNfftForResolution;
    float freqResolution;
    float maxFreq;
    float maxEnv;
    int maxIdx;
    float* workBuffer = NULL;
    float* cosWindow = NULL;
    kiss_fft_cpx* fftOut = NULL;
    kiss_fftr_cfg fftCfg = NULL;
    float* pickedMagnitudes = NULL;
    float* pickedFreqs = NULL;
    float mean;
    float scale;
    int i, k, freqIdx;
    float freq, freqLow, freqHigh, t;
    float re_low, im_low, re_high, im_high;
    float re_interp, im_interp;
    float cosArg;
    float normFactor;
    float fftNorm;
    float absVal;
    pulseqlib_AcousticViolation violation;
    int* seqPeaks;
    int result = PULSEQLIB_OK;
    
    if (!waveform || numSamples <= 0) {
        return PULSEQLIB_ERR_INVALID_ARGUMENT;
    }

    /* Compute maximum envelope value */
    maxEnv = 0.0f;
    for (i = 0; i < numSamples; ++i) {
        absVal = (waveform[i] >= 0.0f) ? waveform[i] : -waveform[i];
        if (absVal > maxEnv) {
            maxEnv = absVal;
        }
    }
    if (outMaxEnvelope) {
        *outMaxEnvelope = maxEnv;
    }
    
    /* Calculate minimum nfft needed for target spectral resolution */
    minNfftForResolution = (int)ceil((double)(1.0e6 / (gradRasterTime_us * targetSpectralResolution_Hz)));
    
    /* Ensure nfft is at least numSamples, round to power of 2 */
    if (minNfftForResolution < numSamples) {
        nfft = (int)next_pow2((size_t)numSamples);
    } else {
        nfft = (int)next_pow2((size_t)minNfftForResolution);
    }
    
    /* Number of frequency bins for real FFT */
    nfreq = nfft / 2 + 1;
    
    /* Actual frequency resolution achieved */
    freqResolution = (float)(1.0e6 / (gradRasterTime_us * (double)nfft));
    
    /* Determine output frequency bins based on maxFrequency */
    maxFreq = (maxFrequency > 0.0f) ? maxFrequency : (float)(5.0e5 / gradRasterTime_us);
    maxIdx = (int)(maxFreq / freqResolution + 0.5);
    if (maxIdx >= nfreq) {
        outputFreqBinsFull = nfreq;
    } else if (maxIdx < 1) {
        outputFreqBinsFull = 1;
    } else {
        outputFreqBinsFull = maxIdx + 1;
    }
    
    /* Allocate buffers */
    workBuffer = (float*)ALLOC((size_t)nfft * sizeof(float));
    cosWindow = (float*)ALLOC((size_t)numSamples * sizeof(float));
    fftOut = (kiss_fft_cpx*)ALLOC((size_t)nfreq * sizeof(kiss_fft_cpx));
    
    if (!workBuffer || !cosWindow || !fftOut) {
        result = PULSEQLIB_ERR_ALLOC_FAILED;
        goto cleanup;
    }
    
    /* Initialize FFT */
    fftCfg = kiss_fftr_alloc(nfft, 0, NULL, NULL);
    if (!fftCfg) {
        result = PULSEQLIB_ERR_ALLOC_FAILED;
        goto cleanup;
    }
    
    /* Compute cosine taper window for full TR */
    for (i = 0; i < numSamples; ++i) {
        cosArg = 2.0 * M_PI * (double)(i + 1) / (double)numSamples;
        cosWindow[i] = (float)(0.5 * (1.0 - cos(cosArg)));
    }
    
    /* Copy waveform to work buffer */
    for (i = 0; i < numSamples; ++i) {
        workBuffer[i] = waveform[i];
    }
    
    /* Zero-fill from numSamples to nfft */
    for (i = numSamples; i < nfft; ++i) {
        workBuffer[i] = 0.0f;
    }
    
    /* Compute and subtract mean */
    mean = 0.0f;
    for (i = 0; i < numSamples; ++i) {
        mean += workBuffer[i];
    }
    mean /= (float)numSamples;
    for (i = 0; i < numSamples; ++i) {
        workBuffer[i] -= mean;
    }
    
    /* Apply cosine taper */
    for (i = 0; i < numSamples; ++i) {
        workBuffer[i] *= cosWindow[i];
    }
    
    /* Compute complex FFT */
    kiss_fftr(fftCfg, workBuffer, fftOut);

    /* Normalize FFT output by nfft for proper amplitude scaling */
    fftNorm = 1.0f / (float)nfft;
    for (i = 0; i < nfreq; ++i) {
        fftOut[i].r *= fftNorm;
        fftOut[i].i *= fftNorm;
    }
    
    /* Compute full TR magnitude spectrum if requested */
    if (fullSpectrum) {
        for (i = 0; i < outputFreqBinsFull; ++i) {
            fullSpectrum[i] = (float)sqrt((double)(fftOut[i].r * fftOut[i].r + fftOut[i].i * fftOut[i].i));
        }
    }
    
    /* Compute N-TR sequence spectrum (harmonic sampling) if requested */
    if (fundamentalFreq > 0.0f && seqSpectrum && numTRs > 0) {
        /* Count how many harmonic lines fit in the frequency range */
        numPicked = (int)(maxFreq / fundamentalFreq) + 1;
        
        /* Allocate output arrays */
        pickedMagnitudes = (float*)ALLOC((size_t)numPicked * sizeof(float));
        if (!pickedMagnitudes) {
            result = PULSEQLIB_ERR_ALLOC_FAILED;
            goto cleanup;
        }
        
        /* Only allocate frequencies if caller wants them */
        if (seqFrequencies) {
            pickedFreqs = (float*)ALLOC((size_t)numPicked * sizeof(float));
            if (!pickedFreqs) {
                result = PULSEQLIB_ERR_ALLOC_FAILED;
                goto cleanup;
            }
        }
        
        /* Normalization factor to prevent overflow and keep magnitudes comparable */
        normFactor = (numTRs > 1) ? (1.0f / (float)numTRs) : 1.0f;
        
        /* Sample complex spectrum at harmonic frequencies */
        for (k = 0; k < numPicked; ++k) {
            freq = (float)k * fundamentalFreq;
            if (pickedFreqs) {
                pickedFreqs[k] = freq;
            }
            
            /* Find the FFT bin index for this frequency */
            freqIdx = (int)(freq / freqResolution);
            
            if (freqIdx >= nfreq - 1) {
                /* Beyond Nyquist - set to zero */
                pickedMagnitudes[k] = 0.0f;
            } else if (freqIdx == 0) {
                /* DC bin - no interpolation */
                pickedMagnitudes[k] = (float)sqrt((double)(fftOut[0].r * fftOut[0].r +  fftOut[0].i * fftOut[0].i)) * normFactor;
            } else {
                /* Interpolate complex spectrum linearly between adjacent bins */
                freqLow = (float)freqIdx * freqResolution;
                freqHigh = (float)(freqIdx + 1) * freqResolution;
                
                re_low = fftOut[freqIdx].r;
                im_low = fftOut[freqIdx].i;
                re_high = fftOut[freqIdx + 1].r;
                im_high = fftOut[freqIdx + 1].i;
                
                /* Interpolation parameter */
                t = (freq - freqLow) / (freqHigh - freqLow);
                
                /* Linear interpolation of complex values */
                re_interp = re_low * (1.0f - t) + re_high * t;
                im_interp = im_low * (1.0f - t) + im_high * t;
                
                /* Take magnitude after interpolation and normalize */
                pickedMagnitudes[k] = (float)sqrt((double)(re_interp * re_interp + im_interp * im_interp)) * normFactor;
            }
        }
        
        *seqSpectrum = pickedMagnitudes;
        pickedMagnitudes = NULL;  /* Prevent cleanup from freeing - ownership transferred */
        
        if (seqFrequencies) {
            *seqFrequencies = pickedFreqs;
            pickedFreqs = NULL;  /* Prevent cleanup from freeing - ownership transferred */
        }
        
        if (outNumPicked) {
            *outNumPicked = numPicked;
        }
    }
    
    if (outNumFreqBinsFull) {
        *outNumFreqBinsFull = outputFreqBinsFull;
    }
    if (outFreqResolutionFull) {
        *outFreqResolutionFull = freqResolution;
    }

    /* Acoustic check */
    if (numForbiddenBands > 0 && forbiddenBands) {
        result = check_acoustic_violations(
            *seqSpectrum,
            *seqFrequencies,
            numPicked,  /* FIX: was outputFreqBinsFull, should be numPicked */
            maxEnv,
            forbiddenBands,
            numForbiddenBands,
            &violation, outSeqPeaks);
        if (PULSEQLIB_FAILED(result)) {
            goto cleanup;
        }
    } else if (outSeqPeaks) {
        /* No violation check, but peaks requested - detect directly */
        seqPeaks = (int*)ALLOC((size_t)numPicked * sizeof(int));  /* FIX: removed double assignment */
        if (!seqPeaks) {
            result = PULSEQLIB_ERR_ALLOC_FAILED;
            goto cleanup;
        }
        detect_resonances(numPicked, *seqSpectrum, seqPeaks);  /* FIX: was outputFreqBinsFull, should be numPicked */
        *outSeqPeaks = seqPeaks;
    }

cleanup:
    if (workBuffer) FREE(workBuffer);
    if (cosWindow) FREE(cosWindow);
    if (fftOut) FREE(fftOut);
    if (fftCfg) kiss_fftr_free(fftCfg);
    if (pickedMagnitudes) FREE(pickedMagnitudes);
    if (pickedFreqs) FREE(pickedFreqs);
    
    return result;
}

int pulseqlib_getTRAcousticSpectra(
    const pulseqlib_TRGradientWaveforms* waveforms,
    float gradRasterTime_us,
    int targetWindowSize,
    float targetSpectralResolution_Hz,
    float maxFrequency_Hz,
    int combined,
    int numTRs,
    float trDuration_us,
    int numForbiddenBands,
    const pulseqlib_ForbiddenBand* forbiddenBands,
    int storeResults,  /* NEW parameter */
    pulseqlib_TRAcousticSpectra* spectra,
    pulseqlib_Diagnostic* diag)
{
    AcousticSpectrumSupport support;
    pulseqlib_Diagnostic localDiag;
    int maxSamples;
    int result;
    int outputSpectraSize;
    int paddedLen;
    int i;
    float fundamentalFreq;
    int numFreqBinsFull;
    int numFreqBinsSeq;
    float freqResolutionFull;
    float* seqSpectrumGx = NULL;
    float* seqSpectrumGy = NULL;
    float* seqSpectrumGz = NULL;
    float* seqFrequencies = NULL;
    
    /* Use local diag if caller doesn't provide one */
    if (!diag) {
        pulseqlib_diagnosticInit(&localDiag);
        diag = &localDiag;
    } else {
        pulseqlib_diagnosticInit(diag);
    }
    
    /* Validate inputs */
    if (!waveforms || !spectra) {
        diag->code = PULSEQLIB_ERR_NULL_POINTER;
        return diag->code;
    }
    
    /* Initialize output */
    memset(spectra, 0, sizeof(*spectra));
    memset(&support, 0, sizeof(support));
    
    /* Find maximum number of samples across all axes */
    maxSamples = waveforms->numSamplesGx;
    if (waveforms->numSamplesGy > maxSamples) {
        maxSamples = waveforms->numSamplesGy;
    }
    if (waveforms->numSamplesGz > maxSamples) {
        maxSamples = waveforms->numSamplesGz;
    }
    
    if (maxSamples <= 0) {
        diag->code = PULSEQLIB_ERR_INVALID_ARGUMENT;
        return diag->code;
    }
    
    /* Initialize acoustic support structure for sliding window */
    result = acousticSpectrumSupportInit(
        &support, maxSamples, targetWindowSize, targetSpectralResolution_Hz, gradRasterTime_us, maxFrequency_Hz);
    if (PULSEQLIB_FAILED(result)) {
        diag->code = result;
        return result;
    }
    
    /* Store sliding window output parameters */
    spectra->combined = combined;
    spectra->numWindows = combined ? 1 : support.numWindows;
    spectra->numFreqBins = support.outputFreqBins;
    spectra->freqResolution = support.freqResolution;
    
    /* Determine output size based on combined flag */
    if (combined) {
        outputSpectraSize = support.outputFreqBins;
    } else {
        outputSpectraSize = support.numWindows * support.outputFreqBins;
    }
    
    /* Allocate arrays only if storing results */
    if (storeResults) {
        /* Allocate sliding window output arrays */
        spectra->spectraGx = (float*)ALLOC((size_t)outputSpectraSize * sizeof(float));
        spectra->spectraGy = (float*)ALLOC((size_t)outputSpectraSize * sizeof(float));
        spectra->spectraGz = (float*)ALLOC((size_t)outputSpectraSize * sizeof(float));
        spectra->frequencies = (float*)ALLOC((size_t)support.outputFreqBins * sizeof(float));

        /* Allocate max envelope arrays */
        spectra->maxEnvelopeGx = (float*)ALLOC((size_t)spectra->numWindows * sizeof(float));
        spectra->maxEnvelopeGy = (float*)ALLOC((size_t)spectra->numWindows * sizeof(float));
        spectra->maxEnvelopeGz = (float*)ALLOC((size_t)spectra->numWindows * sizeof(float));

        /* Allocate peak arrays for sliding window (only for non-combined mode) */
        if (!combined) {
            spectra->peaksGx = (int*)ALLOC((size_t)outputSpectraSize * sizeof(int));
            spectra->peaksGy = (int*)ALLOC((size_t)outputSpectraSize * sizeof(int));
            spectra->peaksGz = (int*)ALLOC((size_t)outputSpectraSize * sizeof(int));
            
            if (!spectra->peaksGx || !spectra->peaksGy || !spectra->peaksGz) {
                pulseqlib_trAcousticSpectraFree(spectra);
                acousticSpectrumSupportFree(&support);
                diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
                return diag->code;
            }
        } else {
            spectra->peaksGx = NULL;
            spectra->peaksGy = NULL;
            spectra->peaksGz = NULL;
        }

        if (!spectra->spectraGx || !spectra->spectraGy || !spectra->spectraGz || !spectra->frequencies ||
            !spectra->maxEnvelopeGx || !spectra->maxEnvelopeGy || !spectra->maxEnvelopeGz) {
            pulseqlib_trAcousticSpectraFree(spectra);
            acousticSpectrumSupportFree(&support);
            diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
            return diag->code;
        }

        /* Zero-initialize output arrays */
        memset(spectra->spectraGx, 0, (size_t)outputSpectraSize * sizeof(float));
        memset(spectra->spectraGy, 0, (size_t)outputSpectraSize * sizeof(float));
        memset(spectra->spectraGz, 0, (size_t)outputSpectraSize * sizeof(float));
        memset(spectra->maxEnvelopeGx, 0, (size_t)spectra->numWindows * sizeof(float));
        memset(spectra->maxEnvelopeGy, 0, (size_t)spectra->numWindows * sizeof(float));
        memset(spectra->maxEnvelopeGz, 0, (size_t)spectra->numWindows * sizeof(float));
        
        if (!combined) {
            memset(spectra->peaksGx, 0, (size_t)outputSpectraSize * sizeof(int));
            memset(spectra->peaksGy, 0, (size_t)outputSpectraSize * sizeof(int));
            memset(spectra->peaksGz, 0, (size_t)outputSpectraSize * sizeof(int));
        }

        /* Populate sliding window frequency axis */
        for (i = 0; i < support.outputFreqBins; i++) {
            spectra->frequencies[i] = (float)i * support.freqResolution;
        }
    } else {
        /* Production mode: no storage */
        spectra->spectraGx = NULL;
        spectra->spectraGy = NULL;
        spectra->spectraGz = NULL;
        spectra->frequencies = NULL;
        spectra->maxEnvelopeGx = NULL;
        spectra->maxEnvelopeGy = NULL;
        spectra->maxEnvelopeGz = NULL;
        spectra->peaksGx = NULL;
        spectra->peaksGy = NULL;
        spectra->peaksGz = NULL;
    }

    /* Compute padded length that support expects */
    if (maxSamples <= support.nwin) {
        paddedLen = maxSamples;
    } else {
        paddedLen = ((maxSamples + support.nwin - 1) / support.nwin) * support.nwin;
    }
    
    /* ========== Compute sliding window spectra for each axis ========== */
    
    /* Gx */
    if (waveforms->numSamplesGx > 0) {
        result = compute_sliding_window_spectra(
            storeResults ? spectra->spectraGx : NULL,
            &support, 
            waveforms->waveformGx, 
            spectra->frequencies,  /* Can be NULL in production mode */
            waveforms->numSamplesGx, 
            paddedLen, combined, 
            storeResults ? spectra->maxEnvelopeGx : NULL,
            numForbiddenBands, forbiddenBands, 
            storeResults ? spectra->peaksGx : NULL);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_trAcousticSpectraFree(spectra);
            acousticSpectrumSupportFree(&support);
            diag->code = result;
            return result;
        }
    } else if (storeResults) {
        memset(spectra->spectraGx, 0, (size_t)outputSpectraSize * sizeof(float));
    }
    
    /* Gy */
    if (waveforms->numSamplesGy > 0) {
        result = compute_sliding_window_spectra(
            storeResults ? spectra->spectraGy : NULL,
            &support,
            waveforms->waveformGy, 
            spectra->frequencies,
            waveforms->numSamplesGy, 
            paddedLen, combined, 
            storeResults ? spectra->maxEnvelopeGy : NULL,
            numForbiddenBands, forbiddenBands, 
            storeResults ? spectra->peaksGy : NULL);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_trAcousticSpectraFree(spectra);
            acousticSpectrumSupportFree(&support);
            diag->code = result;
            return result;
        }
    } else if (storeResults) {
        memset(spectra->spectraGy, 0, (size_t)outputSpectraSize * sizeof(float));
    }

    /* Gz */
    if (waveforms->numSamplesGz > 0) {
        result = compute_sliding_window_spectra(
            storeResults ? spectra->spectraGz : NULL,
            &support,
            waveforms->waveformGz, 
            spectra->frequencies,
            waveforms->numSamplesGz, 
            paddedLen, combined, 
            storeResults ? spectra->maxEnvelopeGz : NULL,
            numForbiddenBands, forbiddenBands, 
            storeResults ? spectra->peaksGz : NULL);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_trAcousticSpectraFree(spectra);
            acousticSpectrumSupportFree(&support);
            diag->code = result;
            return result;
        }
    } else if (storeResults) {
        memset(spectra->spectraGz, 0, (size_t)outputSpectraSize * sizeof(float));
    }
    
    /* Cleanup sliding window support */
    acousticSpectrumSupportFree(&support);
    
    /* ========== Compute Sequence Spectra ========== */
    /* Compute fundamental frequency for sequence spectrum (0 to skip) */
    if (numTRs > 1 && trDuration_us > 0.0f) {
        fundamentalFreq = 1.0e6f / trDuration_us;  /* Convert to Hz */
        spectra->numTRs = numTRs;
        spectra->trDuration_us = trDuration_us;
        spectra->fundamentalFreq = fundamentalFreq;
    } else {
        fundamentalFreq = 0.0f;
        spectra->numTRs = 1;
        spectra->trDuration_us = trDuration_us;
        spectra->fundamentalFreq = 0.0f;
    }
    
    /* Compute for Gx (first call also determines sizes) */
    if (waveforms->numSamplesGx > 0) {
        result = compute_sequence_spectrum(
            NULL,  /* Don't output full spectrum yet - need to allocate first */
            (storeResults && fundamentalFreq > 0.0f) ? &seqSpectrumGx : NULL,
            (storeResults && fundamentalFreq > 0.0f) ? &seqFrequencies : NULL,
            waveforms->waveformGx,
            waveforms->numSamplesGx,
            gradRasterTime_us,
            targetSpectralResolution_Hz,
            maxFrequency_Hz,
            fundamentalFreq,
            numTRs,
            &numFreqBinsFull,
            &freqResolutionFull,
            &numFreqBinsSeq,
            NULL,  /* maxEnvelope computed later */
            numForbiddenBands, forbiddenBands,
            (storeResults && fundamentalFreq > 0.0f) ? &spectra->peaksGxSeq : NULL);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_trAcousticSpectraFree(spectra);
            diag->code = result;
            return result;
        }
    } else {
        /* Use maxSamples to determine sizes */
        result = compute_sequence_spectrum(
            NULL, NULL, NULL,
            waveforms->waveformGy ? waveforms->waveformGy : waveforms->waveformGz,
            maxSamples,
            gradRasterTime_us,
            targetSpectralResolution_Hz,
            maxFrequency_Hz,
            0.0f,  /* No sequence spectrum needed for size calculation */
            numTRs,
            &numFreqBinsFull,
            &freqResolutionFull,
            NULL,
            NULL,
            numForbiddenBands, forbiddenBands,
            NULL);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_trAcousticSpectraFree(spectra);
            diag->code = result;
            return result;
        }
    }
    
    /* Store full TR parameters */
    spectra->numFreqBinsFull = numFreqBinsFull;
    spectra->freqResolutionFull = freqResolutionFull;
    
    /* Allocate full TR spectrum arrays only if storing */
    if (storeResults) {
        spectra->spectraGxFull = (float*)ALLOC((size_t)numFreqBinsFull * sizeof(float));
        spectra->spectraGyFull = (float*)ALLOC((size_t)numFreqBinsFull * sizeof(float));
        spectra->spectraGzFull = (float*)ALLOC((size_t)numFreqBinsFull * sizeof(float));
        spectra->frequenciesFull = (float*)ALLOC((size_t)numFreqBinsFull * sizeof(float));
        
        if (!spectra->spectraGxFull || !spectra->spectraGyFull || 
            !spectra->spectraGzFull || !spectra->frequenciesFull) {
            if (seqSpectrumGx) FREE(seqSpectrumGx);
            if (seqFrequencies) FREE(seqFrequencies);
            pulseqlib_trAcousticSpectraFree(spectra);
            diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
            return diag->code;
        }
        
        /* Populate full TR frequency axis */
        for (i = 0; i < numFreqBinsFull; i++) {
            spectra->frequenciesFull[i] = (float)i * freqResolutionFull;
        }
        
        /* Allocate sequence spectrum arrays if computing sequence spectrum */
        if (fundamentalFreq > 0.0f && numFreqBinsSeq > 0) {
            spectra->numFreqBinsSeq = numFreqBinsSeq;
            spectra->spectraGxSeq = (float*)ALLOC((size_t)numFreqBinsSeq * sizeof(float));
            spectra->spectraGySeq = (float*)ALLOC((size_t)numFreqBinsSeq * sizeof(float));
            spectra->spectraGzSeq = (float*)ALLOC((size_t)numFreqBinsSeq * sizeof(float));
            spectra->frequenciesSeq = seqFrequencies;  /* Take ownership from first call */
            seqFrequencies = NULL;
            
            if (!spectra->spectraGxSeq || !spectra->spectraGySeq || !spectra->spectraGzSeq) {
                if (seqSpectrumGx) FREE(seqSpectrumGx);
                pulseqlib_trAcousticSpectraFree(spectra);
                diag->code = PULSEQLIB_ERR_ALLOC_FAILED;
                return diag->code;
            }
            
            /* Copy Gx sequence spectrum from first call */
            if (seqSpectrumGx && waveforms->numSamplesGx > 0) {
                memcpy(spectra->spectraGxSeq, seqSpectrumGx, (size_t)numFreqBinsSeq * sizeof(float));
                FREE(seqSpectrumGx);
                seqSpectrumGx = NULL;
            } else {
                memset(spectra->spectraGxSeq, 0, (size_t)numFreqBinsSeq * sizeof(float));
            }
        }
    } else {
        /* Production mode: no full spectrum storage */
        spectra->spectraGxFull = NULL;
        spectra->spectraGyFull = NULL;
        spectra->spectraGzFull = NULL;
        spectra->frequenciesFull = NULL;
        spectra->spectraGxSeq = NULL;
        spectra->spectraGySeq = NULL;
        spectra->spectraGzSeq = NULL;
        spectra->frequenciesSeq = NULL;
        spectra->peaksGxSeq = NULL;
        spectra->peaksGySeq = NULL;
        spectra->peaksGzSeq = NULL;
        
        /* Free temporary allocations from first call */
        if (seqSpectrumGx) { FREE(seqSpectrumGx); seqSpectrumGx = NULL; }
        if (seqFrequencies) { FREE(seqFrequencies); seqFrequencies = NULL; }
    }
    
    /* Now compute full TR spectrum for Gx */
    if (waveforms->numSamplesGx > 0) {
        result = compute_sequence_spectrum(
            storeResults ? spectra->spectraGxFull : NULL,
            NULL, NULL,  /* Already have sequence spectrum */
            waveforms->waveformGx,
            waveforms->numSamplesGx,
            gradRasterTime_us,
            targetSpectralResolution_Hz,
            maxFrequency_Hz,
            0.0f,  /* Don't recompute sequence spectrum */
            numTRs,
            NULL, NULL, NULL, &spectra->maxEnvelopeGxFull,
            numForbiddenBands, forbiddenBands,
            NULL);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_trAcousticSpectraFree(spectra);
            diag->code = result;
            return result;
        }
    } else if (storeResults) {
        memset(spectra->spectraGxFull, 0, (size_t)numFreqBinsFull * sizeof(float));
    }
    
    /* Compute for Gy */
    if (waveforms->numSamplesGy > 0) {
        result = compute_sequence_spectrum(
            storeResults ? spectra->spectraGyFull : NULL,
            (storeResults && fundamentalFreq > 0.0f) ? &seqSpectrumGy : NULL,
            NULL,  /* Don't need frequencies again */
            waveforms->waveformGy,
            waveforms->numSamplesGy,
            gradRasterTime_us,
            targetSpectralResolution_Hz,
            maxFrequency_Hz,
            fundamentalFreq,
            numTRs,
            NULL, NULL, NULL, &spectra->maxEnvelopeGyFull,
            numForbiddenBands, forbiddenBands, 
            (storeResults && fundamentalFreq > 0.0f) ? &spectra->peaksGySeq : NULL);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_trAcousticSpectraFree(spectra);
            diag->code = result;
            return result;
        }
        
        /* Copy Gy sequence spectrum */
        if (storeResults && seqSpectrumGy && spectra->spectraGySeq) {
            memcpy(spectra->spectraGySeq, seqSpectrumGy, (size_t)numFreqBinsSeq * sizeof(float));
            FREE(seqSpectrumGy);
            seqSpectrumGy = NULL;
        } else if (seqSpectrumGy) {
            FREE(seqSpectrumGy);
            seqSpectrumGy = NULL;
        }
    } else if (storeResults) {
        memset(spectra->spectraGyFull, 0, (size_t)numFreqBinsFull * sizeof(float));
        if (spectra->spectraGySeq) {
            memset(spectra->spectraGySeq, 0, (size_t)numFreqBinsSeq * sizeof(float));
        }
        if (spectra->peaksGySeq) {
            memset(spectra->peaksGySeq, 0, (size_t)numFreqBinsSeq * sizeof(int));
        }
    }
    
    /* Compute for Gz */
    if (waveforms->numSamplesGz > 0) {
        result = compute_sequence_spectrum(
            storeResults ? spectra->spectraGzFull : NULL,
            (storeResults && fundamentalFreq > 0.0f) ? &seqSpectrumGz : NULL,
            NULL,  /* Don't need frequencies again */
            waveforms->waveformGz,
            waveforms->numSamplesGz,
            gradRasterTime_us,
            targetSpectralResolution_Hz,
            maxFrequency_Hz,
            fundamentalFreq,
            numTRs,
            NULL, NULL, NULL, &spectra->maxEnvelopeGzFull,
            numForbiddenBands, forbiddenBands,
            (storeResults && fundamentalFreq > 0.0f) ? &spectra->peaksGzSeq : NULL);
        if (PULSEQLIB_FAILED(result)) {
            pulseqlib_trAcousticSpectraFree(spectra);
            diag->code = result;
            return result;
        }
        
        /* Copy Gz sequence spectrum */
        if (storeResults && seqSpectrumGz && spectra->spectraGzSeq) {
            memcpy(spectra->spectraGzSeq, seqSpectrumGz, (size_t)numFreqBinsSeq * sizeof(float));
            FREE(seqSpectrumGz);
            seqSpectrumGz = NULL;
        } else if (seqSpectrumGz) {
            FREE(seqSpectrumGz);
            seqSpectrumGz = NULL;
        }
    } else if (storeResults) {
        memset(spectra->spectraGzFull, 0, (size_t)numFreqBinsFull * sizeof(float));
        if (spectra->spectraGzSeq) {
            memset(spectra->spectraGzSeq, 0, (size_t)numFreqBinsSeq * sizeof(float));
        }
        if (spectra->peaksGzSeq) {
            memset(spectra->peaksGzSeq, 0, (size_t)numFreqBinsSeq * sizeof(int));
        }
    }
    
    diag->code = PULSEQLIB_OK;
    return PULSEQLIB_OK;
}

/* ============== PNS Computation ============== */

#define PNS_MIN_DURATION_FACTOR 3.0f      /* Minimum signal duration = factor * kernel duration */
#define PNS_KERNEL_DURATION_FACTOR 20.0f  /* Kernel extends to 20 * chronaxie */

#if VENDOR == SIEMENS
#define PNS_SIEMENS_EPS 1e-6f             /* Accuracy for Siemens kernel truncation */
#endif

#if VENDOR == GEHC
/**
 * @brief Build GE PNS kernel: f(tau) = dt/Smin * c / (c + tau)^2
 * IEC 60601-2-33:2022 Eq. AA.21
 * where Smin = rheobase / alpha
 */
static int build_pns_kernel(
    float** kernel,
    int* kernelLen,
    float dt_us,
    const pulseqlib_PNSParams* params)
{
    int n, i;
    float tau;
    float c, dt, c_s, dt_s;
    float* k;
    float denom;
    float Smin;
    
    c = params->chronaxie_us;
    
    if (c <= 0.0f) return PULSEQLIB_ERR_PNS_INVALID_CHRONAXIE;
    if (params->rheobase <= 0.0f) return PULSEQLIB_ERR_PNS_INVALID_RHEOBASE;
    if (params->alpha <= 0.0f) return PULSEQLIB_ERR_PNS_INVALID_PARAMS;
    
    /* Convert to seconds for proper dimensional analysis */
    c_s = c * 1e-6f;      /* chronaxie from µs to s */
    dt_s = dt_us * 1e-6f; /* raster from µs to s */
    
    /* Smin = rheobase / alpha (T/m/s) */
    Smin = params->rheobase / params->alpha;
    
    /* Kernel length: 20 * chronaxie / dt */
    n = (int)(PNS_KERNEL_DURATION_FACTOR * c_s / dt_s) + 1;
    
    k = (float*)ALLOC((size_t)n * sizeof(float));
    if (!k) return PULSEQLIB_ERR_ALLOC_FAILED;
    
    /* f(tau) = dt/Smin * c / (c + tau)^2, tau = dt, 2*dt, ..., n*dt */
    for (i = 0; i < n; ++i) {
        tau = (float)i * dt_s;  /* tau in seconds */
        denom = (c_s + tau) * (c_s + tau);
        k[i] = (dt_s / Smin) * (c_s / denom);
    }
    
    *kernel = k;
    *kernelLen = n;
    return PULSEQLIB_OK;
}

static float get_chronaxie(const pulseqlib_PNSParams* params)
{
    return params->chronaxie_us;
}

/**
 * @brief Compute slew rate (derivative) of waveform
 * s[i] = (g[i+1] - g[i]) / dt
 * Output has length numSamples - 1
 */
static void compute_slew_rate(
    const float* waveform,
    int numSamples,
    float dt_us,
    float gamma_hz_per_tesla,
    float* slewRate)
{
    float grad_i;
    float grad_ip1;
    int i;
    float dt_s = dt_us * 1e-6f;  /* Convert to seconds for T/m/s */
    
    for (i = 0; i < numSamples - 1; ++i) {
        grad_i = waveform[i] / gamma_hz_per_tesla;       /* Convert Hz/m to T/m */
        grad_ip1 = waveform[i + 1] / gamma_hz_per_tesla; /* Convert Hz/m to T/m */
        slewRate[i] = (grad_ip1 - grad_i) / dt_s;
    }
}

/**
 * @brief FFT-based convolution
 * Computes output = signal * kernel (convolution)
 * Output has same length as signal (causal, truncated)
 */
static int convolve_fft(
    const float* signal,
    int signalLen,
    const float* kernel,
    int kernelLen,
    float* output)
{
    int nfft, nfreq;
    int i;
    kiss_fftr_cfg fwdCfg = NULL;
    kiss_fftr_cfg invCfg = NULL;
    float* paddedSignal = NULL;
    float* paddedKernel = NULL;
    kiss_fft_cpx* signalFFT = NULL;
    kiss_fft_cpx* kernelFFT = NULL;
    float* convResult = NULL;
    float re, im;
    float scale;
    int result = PULSEQLIB_OK;
    
    /* FFT size: next power of 2 >= signalLen + kernelLen - 1 */
    nfft = (int)next_pow2((size_t)(signalLen + kernelLen - 1));
    nfreq = nfft / 2 + 1;
    
    /* Allocate buffers */
    paddedSignal = (float*)ALLOC((size_t)nfft * sizeof(float));
    paddedKernel = (float*)ALLOC((size_t)nfft * sizeof(float));
    signalFFT = (kiss_fft_cpx*)ALLOC((size_t)nfreq * sizeof(kiss_fft_cpx));
    kernelFFT = (kiss_fft_cpx*)ALLOC((size_t)nfreq * sizeof(kiss_fft_cpx));
    convResult = (float*)ALLOC((size_t)nfft * sizeof(float));
    
    if (!paddedSignal || !paddedKernel || !signalFFT || !kernelFFT || !convResult) {
        result = PULSEQLIB_ERR_ALLOC_FAILED;
        goto cleanup;
    }
    
    /* Zero-pad signal */
    for (i = 0; i < signalLen; ++i) paddedSignal[i] = signal[i];
    for (i = signalLen; i < nfft; ++i) paddedSignal[i] = 0.0f;
    
    /* Zero-pad kernel */
    for (i = 0; i < kernelLen; ++i) paddedKernel[i] = kernel[i];
    for (i = kernelLen; i < nfft; ++i) paddedKernel[i] = 0.0f;
    
    /* Create FFT configs */
    fwdCfg = kiss_fftr_alloc(nfft, 0, NULL, NULL);
    invCfg = kiss_fftr_alloc(nfft, 1, NULL, NULL);
    if (!fwdCfg || !invCfg) {
        result = PULSEQLIB_ERR_PNS_FFT_FAILED;
        goto cleanup;
    }
    
    /* Forward FFT of signal and kernel */
    kiss_fftr(fwdCfg, paddedSignal, signalFFT);
    kiss_fftr(fwdCfg, paddedKernel, kernelFFT);
    
    /* Multiply in frequency domain */
    for (i = 0; i < nfreq; ++i) {
        re = signalFFT[i].r * kernelFFT[i].r - signalFFT[i].i * kernelFFT[i].i;
        im = signalFFT[i].r * kernelFFT[i].i + signalFFT[i].i * kernelFFT[i].r;
        signalFFT[i].r = re;
        signalFFT[i].i = im;
    }
    
    /* Inverse FFT */
    kiss_fftri(invCfg, signalFFT, convResult);
    
    /* Normalize and copy result (only first signalLen samples) */
    scale = 1.0f / (float)nfft;
    for (i = 0; i < signalLen; ++i) {
        output[i] = convResult[i] * scale;
    }
    
cleanup:
    if (paddedSignal) FREE(paddedSignal);
    if (paddedKernel) FREE(paddedKernel);
    if (signalFFT) FREE(signalFFT);
    if (kernelFFT) FREE(kernelFFT);
    if (convResult) FREE(convResult);
    if (fwdCfg) kiss_fftr_free(fwdCfg);
    if (invCfg) kiss_fftr_free(invCfg);
    
    return result;
}

static int process_pns_axis_circular(
    const float* waveform,
    int numSamples,
    int kernelLen,
    float gradRasterTime_us,
    float gamma_hz_per_tesla,
    const float* kernel,
    float* paddedWaveform,   /* Working buffer: size = numSamples + kernelLen */
    float* slewRate,         /* Working buffer: size = numSamples + kernelLen - 1 */
    float* pnsConv,          /* Working buffer: size = numSamples + kernelLen - 1 */
    float* pnsAxis,          /* Output: size = numSamples + kernelLen - 1 (FULL output) */
    float* pnsTotal,         /* In/Out accumulator: size = numSamples + kernelLen - 1 */
    float* pnsStore,         /* Optional output copy: size = numSamples + kernelLen - 1 or NULL */
    int convOutputLen)       /* Length of pnsConv output (= numSamples + kernelLen - 1) */
{
    int i, paddedLen, slewLen;
    int returnCode;
    
    if (numSamples <= 0 || !waveform) {
        return PULSEQLIB_OK;  /* Nothing to process */
    }
    
    paddedLen = numSamples + kernelLen;
    slewLen = paddedLen - 1;
    
    /* Copy original waveform */
    for (i = 0; i < numSamples; ++i) {
        paddedWaveform[i] = waveform[i];
    }
    
    /* Circular padding: wrap around to beginning */
    for (i = 0; i < kernelLen; ++i) {
        paddedWaveform[numSamples + i] = waveform[i % numSamples];
    }
    
    /* Compute slew rate: s[i] = (g[i+1] - g[i]) / dt */
    compute_slew_rate(paddedWaveform, paddedLen, gradRasterTime_us, gamma_hz_per_tesla, slewRate);
    
    /* Convolve with kernel */
    returnCode = convolve_fft(slewRate, slewLen, kernel, kernelLen, pnsConv);
    if (PULSEQLIB_FAILED(returnCode)) {
        return returnCode;
    }
    
    /* Return FULL convolution output (no extraction/truncation) */
    /* This preserves the inter-TR boundary information */
    for (i = 0; i < slewLen; ++i) {
        pnsAxis[i] = pnsConv[i] * 100.0f;  /* Convert to percent */
        pnsTotal[i] += pnsConv[i] * pnsConv[i];
    }
    
    /* Store if requested */
    if (pnsStore) {
        for (i = 0; i < slewLen; ++i) {
            pnsStore[i] = pnsAxis[i];
        }
    }
    
    return PULSEQLIB_OK;
}

int pulseqlib_computePNS(
    const float gamma_hz_per_tesla,
    const float pns_threshold,
    const pulseqlib_TRGradientWaveforms* waveforms,
    float gradRasterTime_us,
    const pulseqlib_PNSParams* params,
    int storeWaveforms,
    pulseqlib_PNSResult* result,
    pulseqlib_Diagnostic* diag)
{
    pulseqlib_Diagnostic localDiag;
    int maxSamples, paddedLen, slewLen, fullOutputLen;
    int kernelLen;
    float* kernel = NULL;
    float* paddedWaveform = NULL;
    float* slewRate = NULL;
    float* pnsConv = NULL;
    float* pnsAxis = NULL;
    float* pnsX = NULL;
    float* pnsY = NULL;
    float* pnsZ = NULL;
    float* pnsTotal = NULL;
    int i;
    float maxPNS;
    int maxIdx;
    int returnCode = PULSEQLIB_OK;
    
    /* Use local diag if caller doesn't provide one */
    if (!diag) {
        pulseqlib_diagnosticInit(&localDiag);
        diag = &localDiag;
    } else {
        pulseqlib_diagnosticInit(diag);
    }
    
    /* Validate inputs */
    if (!waveforms || !params || !result) {
        diag->code = PULSEQLIB_ERR_NULL_POINTER;
        return diag->code;
    }
    
    /* Initialize result */
    memset(result, 0, sizeof(*result));
    
    /* Find maximum samples across axes */
    maxSamples = waveforms->numSamplesGx;
    if (waveforms->numSamplesGy > maxSamples) maxSamples = waveforms->numSamplesGy;
    if (waveforms->numSamplesGz > maxSamples) maxSamples = waveforms->numSamplesGz;
    
    if (maxSamples <= 1) {
        diag->code = PULSEQLIB_ERR_PNS_NO_WAVEFORM;
        return diag->code;
    }
    
    /* Build PNS kernel */
    returnCode = build_pns_kernel(&kernel, &kernelLen, gradRasterTime_us, params);
    if (PULSEQLIB_FAILED(returnCode)) {
        diag->code = returnCode;
        return returnCode;
    }
    
    /* Compute buffer sizes with circular padding */
    paddedLen = maxSamples + kernelLen;
    slewLen = paddedLen - 1;
    fullOutputLen = slewLen;  /* Full output (no truncation) */
    
    /* Allocate working buffers */
    paddedWaveform = (float*)ALLOC((size_t)paddedLen * sizeof(float));
    slewRate = (float*)ALLOC((size_t)slewLen * sizeof(float));
    pnsConv = (float*)ALLOC((size_t)slewLen * sizeof(float));
    pnsAxis = (float*)ALLOC((size_t)fullOutputLen * sizeof(float));
    pnsTotal = (float*)ALLOC((size_t)fullOutputLen * sizeof(float));
    
    if (!paddedWaveform || !slewRate || !pnsConv || !pnsAxis || !pnsTotal) {
        returnCode = PULSEQLIB_ERR_ALLOC_FAILED;
        goto cleanup;
    }
    
    /* Initialize pnsTotal to zero */
    for (i = 0; i < fullOutputLen; ++i) {
        pnsTotal[i] = 0.0f;
    }
    
    /* Allocate per-axis storage if requested */
    if (storeWaveforms) {
        pnsX = (float*)ALLOC((size_t)fullOutputLen * sizeof(float));
        pnsY = (float*)ALLOC((size_t)fullOutputLen * sizeof(float));
        pnsZ = (float*)ALLOC((size_t)fullOutputLen * sizeof(float));
        if (!pnsX || !pnsY || !pnsZ) {
            returnCode = PULSEQLIB_ERR_ALLOC_FAILED;
            goto cleanup;
        }
        /* Initialize to zero */
        for (i = 0; i < fullOutputLen; ++i) {
            pnsX[i] = 0.0f;
            pnsY[i] = 0.0f;
            pnsZ[i] = 0.0f;
        }
    }
    
    /* Process X axis */
    returnCode = process_pns_axis_circular(
        waveforms->waveformGx, waveforms->numSamplesGx,
        kernelLen, gradRasterTime_us, gamma_hz_per_tesla, kernel,
        paddedWaveform, slewRate, pnsConv, pnsAxis, pnsTotal,
        storeWaveforms ? pnsX : NULL, fullOutputLen);
    if (PULSEQLIB_FAILED(returnCode)) goto cleanup;
    
    /* Process Y axis */
    returnCode = process_pns_axis_circular(
        waveforms->waveformGy, waveforms->numSamplesGy,
        kernelLen, gradRasterTime_us, gamma_hz_per_tesla, kernel,
        paddedWaveform, slewRate, pnsConv, pnsAxis, pnsTotal,
        storeWaveforms ? pnsY : NULL, fullOutputLen);
    if (PULSEQLIB_FAILED(returnCode)) goto cleanup;
    
    /* Process Z axis */
    returnCode = process_pns_axis_circular(
        waveforms->waveformGz, waveforms->numSamplesGz,
        kernelLen, gradRasterTime_us, gamma_hz_per_tesla, kernel,
        paddedWaveform, slewRate, pnsConv, pnsAxis, pnsTotal,
        storeWaveforms ? pnsZ : NULL, fullOutputLen);
    if (PULSEQLIB_FAILED(returnCode)) goto cleanup;
    
    /* Compute sqrt for combined PNS and find maximum */
    maxPNS = 0.0f;
    maxIdx = 0;
    for (i = 0; i < fullOutputLen; ++i) {
        pnsTotal[i] = 100.0f * (float)sqrt((double)pnsTotal[i]);
        if (pnsTotal[i] > maxPNS) {
            maxPNS = pnsTotal[i];
            maxIdx = i;
        }
    }
    
    /* Fill result */
    result->numSamples = fullOutputLen;
    result->pnsTotal = pnsTotal;
    pnsTotal = NULL;  /* Transfer ownership */
    
    result->maxPNS = maxPNS;
    result->maxPNS_index = maxIdx;
    result->maxPNS_time_us = (float)maxIdx * gradRasterTime_us;
    
    if (storeWaveforms) {
        result->pnsX = pnsX; pnsX = NULL;
        result->pnsY = pnsY; pnsY = NULL;
        result->pnsZ = pnsZ; pnsZ = NULL;
    }
    
    /* Set warning if threshold exceeded */
    if (storeWaveforms) {
        returnCode = PULSEQLIB_OK;
    } else if (maxPNS > pns_threshold) {
        returnCode = PULSEQLIB_ERR_PNS_THRESHOLD_EXCEEDED;
    } else {
        returnCode = PULSEQLIB_OK;
    }
    diag->code = returnCode;

    
cleanup:
    if (kernel) FREE(kernel);
    if (paddedWaveform) FREE(paddedWaveform);
    if (slewRate) FREE(slewRate);
    if (pnsConv) FREE(pnsConv);
    if (pnsAxis) FREE(pnsAxis);
    if (pnsTotal) FREE(pnsTotal);
    if (pnsX) FREE(pnsX);
    if (pnsY) FREE(pnsY);
    if (pnsZ) FREE(pnsZ);
    
    return returnCode;
}
#else
int pulseqlib_computePNS(
    const float pns_threshold,
    const pulseqlib_TRGradientWaveforms* waveforms,
    float gradRasterTime_us,
    const pulseqlib_PNSParams* params,
    int storeWaveforms,
    pulseqlib_PNSResult* result,
    pulseqlib_Diagnostic* diag)
{
    return PULSEQLIB_ERR_NOT_IMPLEMENTED;
}
#endif

void pulseqlib_pnsResultFree(pulseqlib_PNSResult* result)
{
    if (!result) return;
    
    if (result->pnsX) { FREE(result->pnsX); result->pnsX = NULL; }
    if (result->pnsY) { FREE(result->pnsY); result->pnsY = NULL; }
    if (result->pnsZ) { FREE(result->pnsZ); result->pnsZ = NULL; }
    if (result->pnsTotal) { FREE(result->pnsTotal); result->pnsTotal = NULL; }
    
    result->numSamples = 0;
    result->maxPNS = 0.0f;
    result->maxPNS_index = 0;
    result->maxPNS_time_us = 0.0f;
}

/* ========== ADC Library Accessors ========== */

int pulseqlib_getMaxADCSamples(const pulseqlib_SequenceDescriptorCollection* descCollection)
{
    int i, j;
    int maxSamples = 0;
    
    if (!descCollection) return 0;
    
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        for (j = 0; j < descCollection->descriptors[i].numUniqueADCs; ++j) {
            if (descCollection->descriptors[i].adcDefinitions[j].numSamples > maxSamples) {
                maxSamples = descCollection->descriptors[i].adcDefinitions[j].numSamples;
            }
        }
    }
    
    return maxSamples;
}

int pulseqlib_getADCDwell(const pulseqlib_SequenceDescriptorCollection* descCollection, int adcIdx)
{
    int i, j;
    int globalIdx = 0;
    int numADCs;
    
    if (!descCollection || adcIdx < 0 || adcIdx >= descCollection->totalUniqueADCs) {
        return 0;
    }

    /* Find which subsequence and local ADC index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numADCs = descCollection->descriptors[i].numUniqueADCs;
        if (adcIdx < globalIdx + numADCs) {
            /* Found the subsequence */
            j = adcIdx - globalIdx;
            return descCollection->descriptors[i].adcDefinitions[j].dwellTime;
        }
        globalIdx += numADCs;
    }
    
    return 0;
}

int pulseqlib_getADCNumSamples(const pulseqlib_SequenceDescriptorCollection* descCollection, int adcIdx)
{
    int i, j;
    int globalIdx = 0;
    int numADCs;
    
    if (!descCollection || adcIdx < 0 || adcIdx >= descCollection->totalUniqueADCs) {
        return 0;
    }

    /* Find which subsequence and local ADC index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numADCs = descCollection->descriptors[i].numUniqueADCs;
        if (adcIdx < globalIdx + numADCs) {
            /* Found the subsequence */
            j = adcIdx - globalIdx;
            return descCollection->descriptors[i].adcDefinitions[j].numSamples;
        }
        globalIdx += numADCs;
    }
    
    return 0;
}

/* ========== Instruction Creation Support ========== */

int pulseqlib_getNumSegments(const pulseqlib_SequenceDescriptorCollection* descCollection)
{
    if (!descCollection) return 0;
    return descCollection->totalUniqueSegments;
}

int pulseqlib_isSegmentPureDelay(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    pulseqlib_TRsegment* segment;
    pulseqlib_BlockDefinition* blockDef;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return -1;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            /* Found the subsequence */
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            /* Pure delay: single block with no RF and no gradients */
            if (segment->numBlocks == 1) {
                blockDef = &descCollection->descriptors[i].blockDefinitions[segment->uniqueBlockIndices[0]];
                if (blockDef->rfID == -1 && blockDef->gxID == -1 && 
                    blockDef->gyID == -1 && blockDef->gzID == -1) {
                    return 1;
                }
            }
            return 0;
        }
        globalIdx += numSegs;
    }
    
    return -1;
}

int pulseqlib_getSegmentNumBlocks(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return -1;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            /* Found the subsequence */
            j = segmentIdx - globalIdx;
            return descCollection->descriptors[i].segmentDefinitions[j].numBlocks;
        }
        globalIdx += numSegs;
    }
    
    return -1;
}

int pulseqlib_getBlockStartTime(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx)
{
    int i, j, k;
    int globalIdx = 0;
    int numSegs;
    pulseqlib_TRsegment* segment;
    int startTime = 0;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return -1;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            /* Found the subsequence */
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                return -1;
            }
            
            /* Sum durations of all blocks before blockIdx */
            for (k = 0; k < blockIdx; ++k) {
                startTime += descCollection->descriptors[i].blockDefinitions[segment->uniqueBlockIndices[k]].duration_us;
            }
            
            return startTime;
        }
        globalIdx += numSegs;
    }
    
    return -1;
}

int pulseqlib_getBlockDuration(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    pulseqlib_TRsegment* segment;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return -1;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            /* Found the subsequence */
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                return -1;
            }
            
            return descCollection->descriptors[i].blockDefinitions[segment->uniqueBlockIndices[blockIdx]].duration_us;
        }
        globalIdx += numSegs;
    }
    
    return -1;
}

int pulseqlib_blockHasRF(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    pulseqlib_TRsegment* segment;
    pulseqlib_BlockDefinition* blockDef;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return -1;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            /* Found the subsequence */
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                return -1;
            }
            
            blockDef = &descCollection->descriptors[i].blockDefinitions[segment->uniqueBlockIndices[blockIdx]];
            return (blockDef->rfID != -1) ? 1 : 0;
        }
        globalIdx += numSegs;
    }
    
    return -1;
}

int pulseqlib_blockRFHasUniformRaster(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    pulseqlib_TRsegment* segment;
    pulseqlib_BlockDefinition* blockDef;
    pulseqlib_RfDefinition* rfDef;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return -1;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            /* Found the subsequence */
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                return -1;
            }
            
            blockDef = &descCollection->descriptors[i].blockDefinitions[segment->uniqueBlockIndices[blockIdx]];
            
            /* Check if block has RF */
            if (blockDef->rfID == -1) {
                return -1;
            }
            
            /* Get RF definition and check timeShapeID */
            rfDef = &descCollection->descriptors[i].rfDefinitions[blockDef->rfID];
            return (rfDef->timeShapeID != 0) ? 1 : 0;
        }
        globalIdx += numSegs;
    }
    
    return -1;
}

int pulseqlib_blockRFIsComplex(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    pulseqlib_TRsegment* segment;
    pulseqlib_BlockDefinition* blockDef;
    pulseqlib_RfDefinition* rfDef;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return -1;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            /* Found the subsequence */
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                return -1;
            }
            
            blockDef = &descCollection->descriptors[i].blockDefinitions[segment->uniqueBlockIndices[blockIdx]];
            
            /* Check if block has RF */
            if (blockDef->rfID == -1) {
                return -1;
            }
            
            /* Get RF definition and check phaseShapeID */
            rfDef = &descCollection->descriptors[i].rfDefinitions[blockDef->rfID];
            return (rfDef->phaseShapeID != 0) ? 1 : 0;
        }
        globalIdx += numSegs;
    }
    
    return -1;
}

int pulseqlib_getRFNumSamples(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    int shapeIdx;
    pulseqlib_TRsegment* segment;
    pulseqlib_BlockDefinition* blockDef;
    pulseqlib_RfDefinition* rfDef;
    pulseqlib_ShapeArbitrary* magShape;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return -1;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            /* Found the subsequence */
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                return -1;
            }
            
            blockDef = &descCollection->descriptors[i].blockDefinitions[segment->uniqueBlockIndices[blockIdx]];
            
            /* Check if block has RF */
            if (blockDef->rfID == -1) {
                return -1;
            }
            
            /* Get RF definition */
            rfDef = &descCollection->descriptors[i].rfDefinitions[blockDef->rfID];
            
            /* Try to get number of samples from magnitude shape (preferred) */
            if (rfDef->magShapeID > 0) {
                shapeIdx = rfDef->magShapeID - 1;
                if (shapeIdx >= 0 && shapeIdx < descCollection->descriptors[i].numShapes) {
                    magShape = &descCollection->descriptors[i].shapes[shapeIdx];
                    if (magShape->numUncompressedSamples > 0) {
                        return magShape->numUncompressedSamples;
                    }
                }
            }
            
            /* Try to get number of samples from phase shape */
            if (rfDef->phaseShapeID > 0) {
                shapeIdx = rfDef->phaseShapeID - 1;
                if (shapeIdx >= 0 && shapeIdx < descCollection->descriptors[i].numShapes) {
                    magShape = &descCollection->descriptors[i].shapes[shapeIdx];
                    if (magShape->numUncompressedSamples > 0) {
                        return magShape->numUncompressedSamples;
                    }
                }
            }
            
            /* Try to get number of samples from time shape */
            if (rfDef->timeShapeID > 0) {
                shapeIdx = rfDef->timeShapeID - 1;
                if (shapeIdx >= 0 && shapeIdx < descCollection->descriptors[i].numShapes) {
                    magShape = &descCollection->descriptors[i].shapes[shapeIdx];
                    if (magShape->numUncompressedSamples > 0) {
                        return magShape->numUncompressedSamples;
                    }
                }
            }
            
            return -1;
        }
        globalIdx += numSegs;
    }
    
    return -1;
}

int pulseqlib_getRFDelay(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    pulseqlib_TRsegment* segment;
    pulseqlib_BlockDefinition* blockDef;
    pulseqlib_RfDefinition* rfDef;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return -1;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            /* Found the subsequence */
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                return -1;
            }
            
            blockDef = &descCollection->descriptors[i].blockDefinitions[segment->uniqueBlockIndices[blockIdx]];
            
            /* Check if block has RF */
            if (blockDef->rfID == -1) {
                return -1;
            }
            
            /* Get RF definition and return delay */
            rfDef = &descCollection->descriptors[i].rfDefinitions[blockDef->rfID];
            return rfDef->delay;
        }
        globalIdx += numSegs;
    }
    
    return -1;
}

float* pulseqlib_getRFMagnitude(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int* numSamples)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    int shapeIdx;
    pulseqlib_TRsegment* segment;
    pulseqlib_BlockDefinition* blockDef;
    pulseqlib_RfDefinition* rfDef;
    pulseqlib_ShapeArbitrary compressedShape;
    pulseqlib_ShapeArbitrary decompressed;
    
    if (!descCollection || !numSamples || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        if (numSamples) *numSamples = 0;
        return NULL;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            /* Found the subsequence */
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                *numSamples = 0;
                return NULL;
            }
            
            blockDef = &descCollection->descriptors[i].blockDefinitions[segment->uniqueBlockIndices[blockIdx]];
            
            /* Check if block has RF */
            if (blockDef->rfID == -1) {
                *numSamples = 0;
                return NULL;
            }
            
            /* Get RF definition */
            rfDef = &descCollection->descriptors[i].rfDefinitions[blockDef->rfID];
            
            /* Check if magShapeID is valid (1-based index) */
            if (rfDef->magShapeID <= 0) {
                *numSamples = 0;
                return NULL;
            }
            
            /* magShapeID is 1-based, convert to 0-based index */
            shapeIdx = rfDef->magShapeID - 1;
            
            if (shapeIdx < 0 || shapeIdx >= descCollection->descriptors[i].numShapes) {
                *numSamples = 0;
                return NULL;
            }
            
            /* Get compressed shape and decompress with maxAmplitude scale */
            compressedShape = descCollection->descriptors[i].shapes[shapeIdx];
            
            /* Initialize decompressed shape */
            decompressed.numSamples = 0;
            decompressed.numUncompressedSamples = 0;
            decompressed.samples = NULL;
            
            /* Decompress using maxAmplitude as scale */
            if (!decompressShape(&compressedShape, &decompressed, rfDef->maxAmplitude)) {
                *numSamples = 0;
                return NULL;
            }
            
            *numSamples = decompressed.numSamples;
            return decompressed.samples;
        }
        globalIdx += numSegs;
    }
    
    *numSamples = 0;
    return NULL;
}

float* pulseqlib_getRFPhase(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int* numSamples)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    int shapeIdx;
    pulseqlib_TRsegment* segment;
    pulseqlib_BlockDefinition* blockDef;
    pulseqlib_RfDefinition* rfDef;
    pulseqlib_ShapeArbitrary compressedShape;
    pulseqlib_ShapeArbitrary decompressed;
    
    if (!descCollection || !numSamples || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        if (numSamples) *numSamples = 0;
        return NULL;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            /* Found the subsequence */
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                *numSamples = 0;
                return NULL;
            }
            
            blockDef = &descCollection->descriptors[i].blockDefinitions[segment->uniqueBlockIndices[blockIdx]];
            
            /* Check if block has RF */
            if (blockDef->rfID == -1) {
                *numSamples = 0;
                return NULL;
            }
            
            /* Get RF definition */
            rfDef = &descCollection->descriptors[i].rfDefinitions[blockDef->rfID];
            
            /* Check if phaseShapeID is valid (1-based index) */
            if (rfDef->phaseShapeID <= 0) {
                *numSamples = 0;
                return NULL;
            }
            
            /* phaseShapeID is 1-based, convert to 0-based index */
            shapeIdx = rfDef->phaseShapeID - 1;
            
            if (shapeIdx < 0 || shapeIdx >= descCollection->descriptors[i].numShapes) {
                *numSamples = 0;
                return NULL;
            }
            
            /* Get compressed shape and decompress with scale 1.0 (phase in rad) */
            compressedShape = descCollection->descriptors[i].shapes[shapeIdx];
            
            /* Initialize decompressed shape */
            decompressed.numSamples = 0;
            decompressed.numUncompressedSamples = 0;
            decompressed.samples = NULL;
            
            /* Decompress with scale 1.0 (already in rad) */
            if (!decompressShape(&compressedShape, &decompressed, 1.0f)) {
                *numSamples = 0;
                return NULL;
            }
            
            *numSamples = decompressed.numSamples;
            return decompressed.samples;
        }
        globalIdx += numSegs;
    }
    
    *numSamples = 0;
    return NULL;
}

float* pulseqlib_getRFTime(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int* numSamples)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    int shapeIdx;
    pulseqlib_TRsegment* segment;
    pulseqlib_BlockDefinition* blockDef;
    pulseqlib_RfDefinition* rfDef;
    pulseqlib_ShapeArbitrary compressedShape;
    pulseqlib_ShapeArbitrary decompressed;
    
    if (!descCollection || !numSamples || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        if (numSamples) *numSamples = 0;
        return NULL;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            /* Found the subsequence */
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                *numSamples = 0;
                return NULL;
            }
            
            blockDef = &descCollection->descriptors[i].blockDefinitions[segment->uniqueBlockIndices[blockIdx]];
            
            /* Check if block has RF */
            if (blockDef->rfID == -1) {
                *numSamples = 0;
                return NULL;
            }
            
            /* Get RF definition */
            rfDef = &descCollection->descriptors[i].rfDefinitions[blockDef->rfID];
            
            /* Check if timeShapeID is valid (1-based index) */
            if (rfDef->timeShapeID <= 0) {
                *numSamples = 0;
                return NULL;
            }
            
            /* timeShapeID is 1-based, convert to 0-based index */
            shapeIdx = rfDef->timeShapeID - 1;
            
            if (shapeIdx < 0 || shapeIdx >= descCollection->descriptors[i].numShapes) {
                *numSamples = 0;
                return NULL;
            }
            
            /* Get compressed shape and decompress with RF raster time scale */
            compressedShape = descCollection->descriptors[i].shapes[shapeIdx];
            
            /* Initialize decompressed shape */
            decompressed.numSamples = 0;
            decompressed.numUncompressedSamples = 0;
            decompressed.samples = NULL;
            
            /* Decompress with RF raster time as scale (already in us) */
            if (!decompressShape(&compressedShape, &decompressed, descCollection->descriptors[i].rfRasterTime_us)) {
                *numSamples = 0;
                return NULL;
            }
            
            *numSamples = decompressed.numSamples;
            return decompressed.samples;
        }
        globalIdx += numSegs;
    }
    
    *numSamples = 0;
    return NULL;
}

static int getGradIDByAxis(const pulseqlib_BlockDefinition* blockDef, int axis)
{
    /* Helper function to get gradient ID by axis */
    switch (axis) {
        case PULSEQLIB_GRAD_AXIS_X:
            return blockDef->gxID;
        case PULSEQLIB_GRAD_AXIS_Y:
            return blockDef->gyID;
        case PULSEQLIB_GRAD_AXIS_Z:
            return blockDef->gzID;
        default:
            return -1;
    }
}

int pulseqlib_blockHasGrad(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int axis)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    pulseqlib_TRsegment* segment;
    pulseqlib_BlockDefinition* blockDef;
    int gradID;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return -1;
    }
    
    if (axis < PULSEQLIB_GRAD_AXIS_X || axis > PULSEQLIB_GRAD_AXIS_Z) {
        return -1;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            /* Found the subsequence */
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                return -1;
            }
            
            blockDef = &descCollection->descriptors[i].blockDefinitions[segment->uniqueBlockIndices[blockIdx]];
            gradID = getGradIDByAxis(blockDef, axis);
            
            return (gradID != -1) ? 1 : 0;
        }
        globalIdx += numSegs;
    }
    
    return -1;
}

int pulseqlib_blockGradIsTrapezoid(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int axis)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    pulseqlib_TRsegment* segment;
    pulseqlib_BlockDefinition* blockDef;
    pulseqlib_GradDefinition* gradDef;
    int gradID;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return -1;
    }
    
    if (axis < PULSEQLIB_GRAD_AXIS_X || axis > PULSEQLIB_GRAD_AXIS_Z) {
        return -1;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            /* Found the subsequence */
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                return -1;
            }
            
            blockDef = &descCollection->descriptors[i].blockDefinitions[segment->uniqueBlockIndices[blockIdx]];
            gradID = getGradIDByAxis(blockDef, axis);
            
            /* Check if block has gradient */
            if (gradID == -1) {
                return -1;
            }
            
            /* Get gradient definition */
            gradDef = &descCollection->descriptors[i].gradDefinitions[gradID];
            
            /* Trapezoid if type == 0 (TRAP) */
            if (gradDef->type == 0) {
                return 1;
            }
            
            /* Also return 1 if timeShapeID is defined (unusedOrTimeShapeID for ARBITRARY) */
            if (gradDef->unusedOrTimeShapeID > 0) {
                return 1;
            }
            
            return 0;
        }
        globalIdx += numSegs;
    }
    
    return -1;
}

int pulseqlib_getGradNumSamples(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int axis)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    int shapeIdx;
    pulseqlib_TRsegment* segment;
    pulseqlib_BlockDefinition* blockDef;
    pulseqlib_GradDefinition* gradDef;
    int gradID;
    pulseqlib_ShapeArbitrary* shape;
    int riseTime, flatTime;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return -1;
    }
    
    if (axis < PULSEQLIB_GRAD_AXIS_X || axis > PULSEQLIB_GRAD_AXIS_Z) {
        return -1;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            /* Found the subsequence */
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                return -1;
            }
            
            blockDef = &descCollection->descriptors[i].blockDefinitions[segment->uniqueBlockIndices[blockIdx]];
            gradID = getGradIDByAxis(blockDef, axis);
            
            /* Check if block has gradient */
            if (gradID == -1) {
                return -1;
            }
            
            /* Get gradient definition */
            gradDef = &descCollection->descriptors[i].gradDefinitions[gradID];
            
            /* Handle trapezoid case */
            if (gradDef->type == 0) {
                flatTime = gradDef->flatTimeOrUnused;
                if (flatTime > 0) {
                    return 4;  /* [0, riseTime, flatTime, fallTime] */
                } else {
                    return 3;  /* [0, riseTime, fallTime] */
                }
            } else {
                /* Handle arbitrary gradient - get number of samples from first shot's shape */
                if (gradDef->numShots > 0 && gradDef->shotShapeIDs[0] > 0) {
                    shapeIdx = gradDef->shotShapeIDs[0] - 1;
                    if (shapeIdx >= 0 && shapeIdx < descCollection->descriptors[i].numShapes) {
                        shape = &descCollection->descriptors[i].shapes[shapeIdx];
                        if (shape->numUncompressedSamples > 0) {
                            return shape->numUncompressedSamples;
                        }
                    }
                }
                return -1;
            }
        }
        globalIdx += numSegs;
    }
    
    return -1;
}

int pulseqlib_getGradNumShots(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int axis)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    pulseqlib_TRsegment* segment;
    pulseqlib_BlockDefinition* blockDef;
    pulseqlib_GradDefinition* gradDef;
    int gradID;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return -1;
    }
    
    if (axis < PULSEQLIB_GRAD_AXIS_X || axis > PULSEQLIB_GRAD_AXIS_Z) {
        return -1;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            /* Found the subsequence */
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                return -1;
            }
            
            blockDef = &descCollection->descriptors[i].blockDefinitions[segment->uniqueBlockIndices[blockIdx]];
            gradID = getGradIDByAxis(blockDef, axis);
            
            /* Check if block has gradient */
            if (gradID == -1) {
                return -1;
            }
            
            /* Get gradient definition and return number of shots */
            gradDef = &descCollection->descriptors[i].gradDefinitions[gradID];
            return gradDef->numShots;
        }
        globalIdx += numSegs;
    }
    
    return -1;
}

int pulseqlib_getGradDelay(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int axis)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    pulseqlib_TRsegment* segment;
    pulseqlib_BlockDefinition* blockDef;
    pulseqlib_GradDefinition* gradDef;
    int gradID;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return -1;
    }
    
    if (axis < PULSEQLIB_GRAD_AXIS_X || axis > PULSEQLIB_GRAD_AXIS_Z) {
        return -1;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            /* Found the subsequence */
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                return -1;
            }
            
            blockDef = &descCollection->descriptors[i].blockDefinitions[segment->uniqueBlockIndices[blockIdx]];
            gradID = getGradIDByAxis(blockDef, axis);
            
            /* Check if block has gradient */
            if (gradID == -1) {
                return -1;
            }
            
            /* Get gradient definition and return delay */
            gradDef = &descCollection->descriptors[i].gradDefinitions[gradID];
            return gradDef->delay;
        }
        globalIdx += numSegs;
    }
    
    return -1;
}

float** pulseqlib_getGradAmplitude(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int axis, int* numShots, int** numSamplesPerShot)
{
    int i, j, k, shot;
    int globalIdx = 0;
    int numSegs;
    int shapeIdx;
    pulseqlib_TRsegment* segment;
    pulseqlib_BlockDefinition* blockDef;
    pulseqlib_GradDefinition* gradDef;
    int gradID;
    float** waveforms;
    float* trapWaveform;
    pulseqlib_ShapeArbitrary compressedShape;
    pulseqlib_ShapeArbitrary decompressed;
    int samplesPerShot;
    int riseTime, flatTime, fallTime;
    
    if (!descCollection || !numShots || !numSamplesPerShot || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        if (numShots) *numShots = 0;
        if (numSamplesPerShot) *numSamplesPerShot = NULL;
        return NULL;
    }
    
    if (axis < PULSEQLIB_GRAD_AXIS_X || axis > PULSEQLIB_GRAD_AXIS_Z) {
        *numShots = 0;
        *numSamplesPerShot = NULL;
        return NULL;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            /* Found the subsequence */
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                *numShots = 0;
                *numSamplesPerShot = NULL;
                return NULL;
            }
            
            blockDef = &descCollection->descriptors[i].blockDefinitions[segment->uniqueBlockIndices[blockIdx]];
            gradID = getGradIDByAxis(blockDef, axis);
            
            /* Check if block has gradient */
            if (gradID == -1) {
                *numShots = 0;
                *numSamplesPerShot = NULL;
                return NULL;
            }
            
            /* Get gradient definition */
            gradDef = &descCollection->descriptors[i].gradDefinitions[gradID];
            
            /* Allocate arrays to store number of samples per shot and waveforms */
            *numSamplesPerShot = (int*) ALLOC(gradDef->numShots * sizeof(int));
            if (!*numSamplesPerShot) {
                *numShots = 0;
                return NULL;
            }
            
            waveforms = (float**) ALLOC(gradDef->numShots * sizeof(float*));
            if (!waveforms) {
                FREE(*numSamplesPerShot);
                *numSamplesPerShot = NULL;
                *numShots = 0;
                return NULL;
            }
            
            *numShots = gradDef->numShots;
            
            /* Handle trapezoid case */
            if (gradDef->type == 0) {
                riseTime = gradDef->riseTimeOrUnused;
                flatTime = gradDef->flatTimeOrUnused;
                fallTime = gradDef->fallTimeOrNumUncompressedSamples;
                
                /* Determine number of points */
                if (flatTime > 0) {
                    samplesPerShot = 4;  /* [0, riseTime, flatTime, fallTime] */
                } else {
                    samplesPerShot = 3;  /* [0, riseTime, fallTime] */
                }
                
                /* Create waveform for each shot */
                for (shot = 0; shot < gradDef->numShots; ++shot) {
                    trapWaveform = (float*) ALLOC(samplesPerShot * sizeof(float));
                    if (!trapWaveform) {
                        /* Cleanup on allocation failure */
                        for (k = 0; k < shot; ++k) {
                            FREE(waveforms[k]);
                        }
                        FREE(waveforms);
                        FREE(*numSamplesPerShot);
                        *numSamplesPerShot = NULL;
                        *numShots = 0;
                        return NULL;
                    }
                    
                    trapWaveform[0] = 0.0f;
                    trapWaveform[1] = gradDef->maxAmplitude[shot];
                    
                    if (flatTime > 0) {
                        trapWaveform[2] = gradDef->maxAmplitude[shot];
                        trapWaveform[3] = 0.0f;
                    } else {
                        trapWaveform[2] = 0.0f;
                    }
                    
                    waveforms[shot] = trapWaveform;
                    (*numSamplesPerShot)[shot] = samplesPerShot;
                }
            } else {
                /* Handle arbitrary gradient - decompress shapes for each shot */
                for (shot = 0; shot < gradDef->numShots; ++shot) {
                    /* Check if this shot has a shape */
                    if (gradDef->shotShapeIDs[shot] <= 0) {
                        waveforms[shot] = NULL;
                        (*numSamplesPerShot)[shot] = 0;
                        continue;
                    }
                    
                    shapeIdx = gradDef->shotShapeIDs[shot] - 1;
                    
                    if (shapeIdx < 0 || shapeIdx >= descCollection->descriptors[i].numShapes) {
                        waveforms[shot] = NULL;
                        (*numSamplesPerShot)[shot] = 0;
                        continue;
                    }
                    
                    compressedShape = descCollection->descriptors[i].shapes[shapeIdx];
                    
                    /* Initialize decompressed shape */
                    decompressed.numSamples = 0;
                    decompressed.numUncompressedSamples = 0;
                    decompressed.samples = NULL;
                    
                    /* Decompress with the appropriate scale for this shot */
                    if (!decompressShape(&compressedShape, &decompressed, gradDef->maxAmplitude[shot])) {
                        waveforms[shot] = NULL;
                        (*numSamplesPerShot)[shot] = 0;
                        continue;
                    }
                    
                    waveforms[shot] = decompressed.samples;
                    (*numSamplesPerShot)[shot] = decompressed.numSamples;
                }
            }
            
            return waveforms;
        }
        globalIdx += numSegs;
    }
    
    *numShots = 0;
    *numSamplesPerShot = NULL;
    return NULL;
}

float pulseqlib_getGradInitialAmplitude(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int axis)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    pulseqlib_TRsegment* segment;
    int blockTableIdx;
    const pulseqlib_BlockTableElement* bte;
    int gradEventID;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return 1.0f;
    }
    
    if (axis < PULSEQLIB_GRAD_AXIS_X || axis > PULSEQLIB_GRAD_AXIS_Z) {
        return 1.0f;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                return 1.0f;
            }
            
            /* Use the max-energy instance's startBlock to index blockTable */
            blockTableIdx = segment->maxEnergyStartBlock + blockIdx;
            bte = &descCollection->descriptors[i].blockTable[blockTableIdx];
            
            /* Get the gradient event ID for the requested axis */
            switch (axis) {
                case PULSEQLIB_GRAD_AXIS_X: gradEventID = bte->gxID; break;
                case PULSEQLIB_GRAD_AXIS_Y: gradEventID = bte->gyID; break;
                case PULSEQLIB_GRAD_AXIS_Z: gradEventID = bte->gzID; break;
                default: return 1.0f;
            }
            
            /* No gradient on this axis */
            if (gradEventID < 0 || gradEventID >= descCollection->descriptors[i].gradTableSize) {
                return 1.0f;
            }
            
            return descCollection->descriptors[i].gradTable[gradEventID].amplitude;
        }
        globalIdx += numSegs;
    }
    
    return 1.0f;
}

int pulseqlib_getGradInitialShotID(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int axis)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    pulseqlib_TRsegment* segment;
    int blockTableIdx;
    const pulseqlib_BlockTableElement* bte;
    int gradEventID;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return 0;
    }
    
    if (axis < PULSEQLIB_GRAD_AXIS_X || axis > PULSEQLIB_GRAD_AXIS_Z) {
        return 0;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                return 0;
            }
            
            /* Use the max-energy instance's startBlock to index blockTable */
            blockTableIdx = segment->maxEnergyStartBlock + blockIdx;
            bte = &descCollection->descriptors[i].blockTable[blockTableIdx];
            
            /* Get the gradient event ID for the requested axis */
            switch (axis) {
                case PULSEQLIB_GRAD_AXIS_X: gradEventID = bte->gxID; break;
                case PULSEQLIB_GRAD_AXIS_Y: gradEventID = bte->gyID; break;
                case PULSEQLIB_GRAD_AXIS_Z: gradEventID = bte->gzID; break;
                default: return 0;
            }
            
            /* No gradient on this axis */
            if (gradEventID < 0 || gradEventID >= descCollection->descriptors[i].gradTableSize) {
                return 0;
            }
            
            return descCollection->descriptors[i].gradTable[gradEventID].shotIndex;
        }
        globalIdx += numSegs;
    }
    
    return 0;
}

float* pulseqlib_getGradTime(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx, int axis, int* numSamples)
{
    int i, j, k;
    int globalIdx = 0;
    int numSegs;
    int shapeIdx;
    pulseqlib_TRsegment* segment;
    pulseqlib_BlockDefinition* blockDef;
    pulseqlib_GradDefinition* gradDef;
    int gradID;
    float* timeWaveform;
    pulseqlib_ShapeArbitrary compressedShape;
    pulseqlib_ShapeArbitrary decompressed;
    int riseTime, flatTime, fallTime;
    float accum;
    
    if (!descCollection || !numSamples || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        if (numSamples) *numSamples = 0;
        return NULL;
    }
    
    if (axis < PULSEQLIB_GRAD_AXIS_X || axis > PULSEQLIB_GRAD_AXIS_Z) {
        *numSamples = 0;
        return NULL;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            /* Found the subsequence */
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                *numSamples = 0;
                return NULL;
            }
            
            blockDef = &descCollection->descriptors[i].blockDefinitions[segment->uniqueBlockIndices[blockIdx]];
            gradID = getGradIDByAxis(blockDef, axis);
            
            /* Check if block has gradient */
            if (gradID == -1) {
                *numSamples = 0;
                return NULL;
            }
            
            /* Get gradient definition */
            gradDef = &descCollection->descriptors[i].gradDefinitions[gradID];
            
            /* Handle trapezoid case */
            if (gradDef->type == 0) {
                riseTime = gradDef->riseTimeOrUnused;
                flatTime = gradDef->flatTimeOrUnused;
                fallTime = gradDef->fallTimeOrNumUncompressedSamples;
                
                /* Determine number of time points */
                if (flatTime > 0) {
                    *numSamples = 4;  /* [0, riseTime, riseTime+flatTime, riseTime+flatTime+fallTime] */
                } else {
                    *numSamples = 3;  /* [0, riseTime, riseTime+fallTime] */
                }
                
                timeWaveform = (float*) ALLOC((*numSamples) * sizeof(float));
                if (!timeWaveform) {
                    *numSamples = 0;
                    return NULL;
                }
                
                /* Build cumulative time array */
                accum = 0.0f;
                timeWaveform[0] = accum;
                accum += (float)riseTime;
                timeWaveform[1] = accum;
                
                if (flatTime > 0) {
                    accum += (float)flatTime;
                    timeWaveform[2] = accum;
                    accum += (float)fallTime;
                    timeWaveform[3] = accum;
                } else {
                    accum += (float)fallTime;
                    timeWaveform[2] = accum;
                }
                
                return timeWaveform;
            } else {
                /* Handle arbitrary gradient - decompress time shape */
                if (gradDef->unusedOrTimeShapeID <= 0) {
                    *numSamples = 0;
                    return NULL;
                }
                
                shapeIdx = gradDef->unusedOrTimeShapeID - 1;
                
                if (shapeIdx < 0 || shapeIdx >= descCollection->descriptors[i].numShapes) {
                    *numSamples = 0;
                    return NULL;
                }
                
                compressedShape = descCollection->descriptors[i].shapes[shapeIdx];
                
                /* Initialize decompressed shape */
                decompressed.numSamples = 0;
                decompressed.numUncompressedSamples = 0;
                decompressed.samples = NULL;
                
                /* Decompress with gradient raster time as scale */
                if (!decompressShape(&compressedShape, &decompressed, descCollection->descriptors[i].gradRasterTime_us)) {
                    *numSamples = 0;
                    return NULL;
                }
                
                *numSamples = decompressed.numSamples;
                return decompressed.samples;
            }
        }
        globalIdx += numSegs;
    }
    
    *numSamples = 0;
    return NULL;
}

int pulseqlib_blockHasADC(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    pulseqlib_TRsegment* segment;
    pulseqlib_BlockDefinition* blockDef;
    pulseqlib_BlockTableElement* blockTableElem;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return -1;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            /* Found the subsequence */
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                return -1;
            }
            
            blockDef = &descCollection->descriptors[i].blockDefinitions[segment->uniqueBlockIndices[blockIdx]];
            blockTableElem = &descCollection->descriptors[i].blockTable[blockDef->ID];
            
            return (blockTableElem->adcID != -1) ? 1 : 0;
        }
        globalIdx += numSegs;
    }
    
    return -1;
}

int pulseqlib_getADCDelay(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    pulseqlib_TRsegment* segment;
    pulseqlib_BlockDefinition* blockDef;
    pulseqlib_BlockTableElement* blockTableElem;
    pulseqlib_AdcDefinition* adcDef;
    int adcID;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return -1;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            /* Found the subsequence */
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                return -1;
            }
            
            blockDef = &descCollection->descriptors[i].blockDefinitions[segment->uniqueBlockIndices[blockIdx]];
            blockTableElem = &descCollection->descriptors[i].blockTable[blockDef->ID];
            
            /* Check if block has ADC */
            if (blockTableElem->adcID == -1) {
                return -1;
            }
            
            /* Get ADC definition and return delay */
            adcID = blockTableElem->adcID;
            if (adcID < 0 || adcID >= descCollection->descriptors[i].numUniqueADCs) {
                return -1;
            }
            
            adcDef = &descCollection->descriptors[i].adcDefinitions[adcID];
            return adcDef->delay;
        }
        globalIdx += numSegs;
    }
    
    return -1;
}

int pulseqlib_getADCLibraryIndex(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    int globalADCIdx = 0;
    pulseqlib_TRsegment* segment;
    pulseqlib_BlockDefinition* blockDef;
    pulseqlib_BlockTableElement* blockTableElem;
    int adcID;
    int k;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return -1;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            /* Found the subsequence */
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                return -1;
            }
            
            blockDef = &descCollection->descriptors[i].blockDefinitions[segment->uniqueBlockIndices[blockIdx]];
            blockTableElem = &descCollection->descriptors[i].blockTable[blockDef->ID];
            
            /* Check if block has ADC */
            if (blockTableElem->adcID == -1) {
                return -1;
            }
            
            /* Get local ADC ID */
            adcID = blockTableElem->adcID;
            if (adcID < 0 || adcID >= descCollection->descriptors[i].numUniqueADCs) {
                return -1;
            }
            
            /* Calculate global ADC index by summing up ADC counts from previous subsequences */
            for (k = 0; k < i; ++k) {
                globalADCIdx += descCollection->descriptors[k].numUniqueADCs;
            }
            
            return globalADCIdx + adcID;
        }
        globalIdx += numSegs;
    }
    
    return -1;
}

int pulseqlib_blockHasTrigger(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    pulseqlib_TRsegment* segment;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return -1;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                return -1;
            }
            
            return segment->hasTrigger[blockIdx];
        }
        globalIdx += numSegs;
    }
    
    return -1;
}

int pulseqlib_getTriggerDelay(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    pulseqlib_TRsegment* segment;
    pulseqlib_BlockDefinition* blockDef;
    pulseqlib_BlockTableElement* blockTableElem;
    pulseqlib_TriggerEvent* triggerEvent;
    int triggerID;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return -1;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                return -1;
            }
            
            /* Check the stored hasTrigger flag first */
            if (!segment->hasTrigger[blockIdx]) {
                return -1;
            }
            
            /* 
             * Get trigger delay from the first occurrence.
             * Use the segment's startBlock to index into the blockTable.
             */
            blockTableElem = &descCollection->descriptors[i].blockTable[segment->startBlock + blockIdx];
            
            triggerID = blockTableElem->triggerID;
            if (triggerID == -1 || triggerID >= descCollection->descriptors[i].numTriggers) {
                return -1;
            }
            
            triggerEvent = &descCollection->descriptors[i].triggerEvents[triggerID];
            return (int)triggerEvent->delay;
        }
        globalIdx += numSegs;
    }
    
    return -1;
}

int pulseqlib_blockHasRotation(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    pulseqlib_TRsegment* segment;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return -1;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                return -1;
            }
            
            return segment->hasRotation[blockIdx];
        }
        globalIdx += numSegs;
    }
    
    return -1;
}

int pulseqlib_blockHasNorot(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    pulseqlib_TRsegment* segment;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return -1;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                return -1;
            }
            
            return segment->norotFlag[blockIdx];
        }
        globalIdx += numSegs;
    }
    
    return -1;
}

int pulseqlib_blockHasNopos(const pulseqlib_SequenceDescriptorCollection* descCollection, int segmentIdx, int blockIdx)
{
    int i, j;
    int globalIdx = 0;
    int numSegs;
    pulseqlib_TRsegment* segment;
    
    if (!descCollection || segmentIdx < 0 || segmentIdx >= descCollection->totalUniqueSegments) {
        return -1;
    }
    
    /* Find which subsequence and local segment index */
    for (i = 0; i < descCollection->numSubsequences; ++i) {
        numSegs = descCollection->descriptors[i].numUniqueSegments;
        if (segmentIdx < globalIdx + numSegs) {
            j = segmentIdx - globalIdx;
            segment = &descCollection->descriptors[i].segmentDefinitions[j];
            
            if (blockIdx < 0 || blockIdx >= segment->numBlocks) {
                return -1;
            }
            
            return segment->noposFlag[blockIdx];
        }
        globalIdx += numSegs;
    }
    
    return -1;
}
