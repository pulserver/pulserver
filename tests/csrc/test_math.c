#include "minunit.h"

#include <math.h>
#include <stdio.h>

#include "pulseqlib_math.h"

#define EPSILON 1e-6

MU_TEST(test_roundToRaster) {
    /* Exact multiples */
    mu_assert(fabs(pulseqlib_roundToRaster(10.0f, 2.0f) - 10.0f) < EPSILON, "10.0 rounded to raster 2.0 should be 10.0");
    mu_assert(fabs(pulseqlib_roundToRaster(0.0f, 2.0f) - 0.0f) < EPSILON, "0.0 rounded to raster 2.0 should be 0.0");
    
    /* Rounding up */
    mu_assert(fabs(pulseqlib_roundToRaster(10.1f, 2.0f) - 12.0f) < EPSILON, "10.1 rounded to raster 2.0 should be 12.0");
    mu_assert(fabs(pulseqlib_roundToRaster(11.9f, 2.0f) - 12.0f) < EPSILON, "11.9 rounded to raster 2.0 should be 12.0");
    
    /* Epsilon handling (values slightly above multiple should round up, slightly below/at should stay) */
    /* Note: The implementation uses ceil(time/raster - EPS), so 10.000000001 might stay 10 if EPS is large enough, 
       but generally we want robust rounding for floating point errors. 
       Let's test standard behavior. */
    mu_assert(fabs(pulseqlib_roundToRaster(10.00001f, 10.0f) - 20.0f) < EPSILON, "10.00001 rounded to raster 10.0 should be 20.0");
}

MU_TEST(test_generateTimeAxis) {
    float time[5];
    float raster = 10.0f;
    
    /* Mode 0: Centers (0.5, 1.5, ...) */
    pulseqlib_generateTimeAxis(time, 5, raster, 0);
    mu_assert(fabs(time[0] - 5.0f) < EPSILON, "Mode 0 sample 0 should be 5.0");
    mu_assert(fabs(time[1] - 15.0f) < EPSILON, "Mode 0 sample 1 should be 15.0");
    mu_assert(fabs(time[4] - 45.0f) < EPSILON, "Mode 0 sample 4 should be 45.0");
    
    /* Mode 1: Edges (0.0, 1.0, ...) */
    pulseqlib_generateTimeAxis(time, 5, raster, 1);
    mu_assert(fabs(time[0] - 0.0f) < EPSILON, "Mode 1 sample 0 should be 0.0");
    mu_assert(fabs(time[1] - 10.0f) < EPSILON, "Mode 1 sample 1 should be 10.0");
    mu_assert(fabs(time[4] - 40.0f) < EPSILON, "Mode 1 sample 4 should be 40.0");
    
    /* Mode 2: Oversampled (0.0, 0.5, 1.0, ...) */
    pulseqlib_generateTimeAxis(time, 5, raster, 2);
    mu_assert(fabs(time[0] - 0.0f) < EPSILON, "Mode 2 sample 0 should be 0.0");
    mu_assert(fabs(time[1] - 5.0f) < EPSILON, "Mode 2 sample 1 should be 5.0");
    mu_assert(fabs(time[2] - 10.0f) < EPSILON, "Mode 2 sample 2 should be 10.0");
    mu_assert(fabs(time[4] - 20.0f) < EPSILON, "Mode 2 sample 4 should be 20.0");
}

MU_TEST(test_interpolateLinear) {
    float in_time[] = {0.0f, 10.0f, 20.0f};
    float in_val[] = {0.0f, 100.0f, 0.0f}; /* Triangle */
    float out_time[] = {-5.0f, 0.0f, 5.0f, 10.0f, 15.0f, 20.0f, 25.0f};
    float out_val[7];
    int ret;
    
    ret = pulseqlib_interpolateLinear(out_val, 7, out_time, in_val, 3, in_time);
    mu_assert(ret == 0, "Interpolation should return success");
    
    /* Check extrapolation (clamping) */
    mu_assert(fabs(out_val[0] - 0.0f) < EPSILON, "Extrapolation left should clamp to 0.0");
    mu_assert(fabs(out_val[6] - 0.0f) < EPSILON, "Extrapolation right should clamp to 0.0");
    
    /* Check exact points */
    mu_assert(fabs(out_val[1] - 0.0f) < EPSILON, "Point at 0.0 should be 0.0");
    mu_assert(fabs(out_val[3] - 100.0f) < EPSILON, "Point at 10.0 should be 100.0");
    mu_assert(fabs(out_val[5] - 0.0f) < EPSILON, "Point at 20.0 should be 0.0");
    
    /* Check interpolation */
    mu_assert(fabs(out_val[2] - 50.0f) < EPSILON, "Point at 5.0 should be 50.0");
    mu_assert(fabs(out_val[4] - 50.0f) < EPSILON, "Point at 15.0 should be 50.0");
}

MU_TEST(test_integrateTrapezoid) {
    float time[] = {0.0f, 10.0f, 20.0f};
    float val[] = {0.0f, 10.0f, 0.0f}; /* Triangle base 20, height 10. Area = 0.5 * 20 * 10 = 100 */
    float val2[] = {5.0f, 5.0f, 5.0f}; /* Rectangle */
    float area;
    
    area = pulseqlib_integrateTrapezoid(val, time, 3);
    mu_assert(fabs(area - 100.0f) < EPSILON, "Triangle area should be 100.0");
    
    /* Rectangle */
    area = pulseqlib_integrateTrapezoid(val2, time, 3);
    mu_assert(fabs(area - 100.0f) < EPSILON, "Rectangle area should be 100.0 (20 * 5)");
}

MU_TEST_SUITE(test_math_suite) {
    MU_RUN_TEST(test_roundToRaster);
    MU_RUN_TEST(test_generateTimeAxis);
    MU_RUN_TEST(test_interpolateLinear);
    MU_RUN_TEST(test_integrateTrapezoid);
}

int test_math_main(void) {
    printf("Starting Math test suite...\n");
    MU_RUN_SUITE(test_math_suite);
    printf("Test Math suite completed.\n");
    MU_REPORT();
    return MU_EXIT_CODE;
}