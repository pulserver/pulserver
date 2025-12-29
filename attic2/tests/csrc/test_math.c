#include "minunit.h"

#include <stdlib.h>

#include "pulseqlib_methods.h"
#include "pulserverlib.h"
#include "pulserverlib_methods.h"

MU_TEST(test_first_difference_basic) {
	pulserverlib_Status status;
	const float input[] = {0.0f, 1.5f, 3.0f, 2.0f};
	float* diffs;
	int diffLength;
	int i;
	const float expected[] = {1.5f, 1.5f, -1.0f};

	diffs = NULL;
	diffLength = -1;

	status = pulserverlib_computeFirstDifference(input, 4, &diffs, &diffLength);
	mu_assert(status == PULSERVERLIB_STATUS_OK, "First difference should succeed");
	mu_assert_int_eq(3, diffLength);

	for (i = 0; i < diffLength; ++i) {
		mu_assert_double_eq(expected[i], diffs[i]);
	}

	FREE(diffs);
}

MU_TEST(test_first_difference_zero_length) {
	pulserverlib_Status status;
	float* diffs;
	int diffLength;
	float marker;

	marker = 0.0f;
	diffs = &marker;
	diffLength = 99;

	status = pulserverlib_computeFirstDifference(NULL, 0, &diffs, &diffLength);
	mu_assert(status == PULSERVERLIB_STATUS_OK, "Zero-length input should be allowed");
	mu_assert(diffs == NULL, "Output buffer should be NULL for zero-length input");
	mu_assert_int_eq(0, diffLength);
}

MU_TEST(test_first_difference_single_sample) {
	pulserverlib_Status status;
	const float input[] = {7.0f};
	float* diffs;
	int diffLength;
	float marker;

	marker = 0.0f;
	diffs = &marker;
	diffLength = -5;

	status = pulserverlib_computeFirstDifference(input, 1, &diffs, &diffLength);
	mu_assert(status == PULSERVERLIB_STATUS_OK, "Single sample should return OK");
	mu_assert(diffs == NULL, "Output buffer should be NULL for single sample");
	mu_assert_int_eq(0, diffLength);
}

MU_TEST(test_first_difference_invalid_array) {
	pulserverlib_Status status;
	float* diffs;
	int diffLength;
	float marker;

	marker = 0.0f;
	diffs = &marker;
	diffLength = 10;

	status = pulserverlib_computeFirstDifference(NULL, 3, &diffs, &diffLength);
	mu_assert(status == PULSERVERLIB_STATUS_INVALID_ARGUMENT, "NULL array with positive length should fail");
	mu_assert(diffs == NULL, "Output buffer should be cleared on failure");
	mu_assert_int_eq(0, diffLength);
}

MU_TEST(test_first_difference_negative_length) {
	pulserverlib_Status status;
	float* diffs;
	int diffLength;
	float marker;

	marker = 0.0f;
	diffs = &marker;
	diffLength = 2;

	status = pulserverlib_computeFirstDifference((const float*)&marker, -1, &diffs, &diffLength);
	mu_assert(status == PULSERVERLIB_STATUS_INVALID_ARGUMENT, "Negative length should fail");
	mu_assert(diffs == NULL, "Output buffer should be cleared on failure");
	mu_assert_int_eq(0, diffLength);
}

MU_TEST(test_grad_library_max_basic) {
	pulseqlib_Opts opts;
	pulseqlib_SeqFile* seq;
	float (*entries)[7];
	float result;

	seq = (pulseqlib_SeqFile*)ALLOC(sizeof(pulseqlib_SeqFile));
	mu_assert(seq != NULL, "Failed to allocate sequence object");
	pulseqlib_optsInit(&opts, 3.0f, 40.0f, 150.0f, 1.0f, 10.0f, 0.1f, 10.0f);
	pulseqlib_seqFileInit(seq, &opts);

	entries = (float (*)[7])ALLOC(sizeof(float) * 7U * 3U);
	mu_assert(entries != NULL, "Failed to allocate gradient library");
	seq->gradLibrary = entries;
	seq->gradLibrarySize = 3;
	seq->isGradLibraryParsed = 1;

	entries[0][1] = 10.0f;
	entries[1][1] = 25.0f;
	entries[2][1] = 5.0f;

	result = pulseqlib_getGradLibraryMaxAmplitude(seq);
	mu_assert_double_eq(25.0f, result);

	pulseqlib_seqFileFree(seq);
}

MU_TEST(test_grad_library_max_zero_entries) {
	pulseqlib_Opts opts;
	pulseqlib_SeqFile* seq;
	float result;

	seq = (pulseqlib_SeqFile*)ALLOC(sizeof(pulseqlib_SeqFile));
	mu_assert(seq != NULL, "Failed to allocate sequence object");
	pulseqlib_optsInit(&opts, 3.0f, 40.0f, 150.0f, 1.0f, 10.0f, 0.1f, 10.0f);
	pulseqlib_seqFileInit(seq, &opts);

	result = pulseqlib_getGradLibraryMaxAmplitude(seq);
	mu_assert_double_eq(0.0f, result);

	pulseqlib_seqFileFree(seq);
}

MU_TEST(test_grad_library_max_non_positive) {
	pulseqlib_Opts opts;
	pulseqlib_SeqFile* seq;
	float (*entries)[7];
	float result;

	seq = (pulseqlib_SeqFile*)ALLOC(sizeof(pulseqlib_SeqFile));
	mu_assert(seq != NULL, "Failed to allocate sequence object");
	pulseqlib_optsInit(&opts, 3.0f, 40.0f, 150.0f, 1.0f, 10.0f, 0.1f, 10.0f);
	pulseqlib_seqFileInit(seq, &opts);

	entries = (float (*)[7])ALLOC(sizeof(float) * 7U * 2U);
	mu_assert(entries != NULL, "Failed to allocate gradient library");
	seq->gradLibrary = entries;
	seq->gradLibrarySize = 2;
	seq->isGradLibraryParsed = 1;

	entries[0][1] = -4.0f;
	entries[1][1] = -0.5f;

	result = pulseqlib_getGradLibraryMaxAmplitude(seq);
	mu_assert_double_eq(0.0f, result);

	pulseqlib_seqFileFree(seq);
}

MU_TEST(test_max_grad_check_pass) {
	pulseqlib_Opts opts;
	pulseqlib_SeqFile* seq;
	float (*entries)[7];
	float result;

	seq = (pulseqlib_SeqFile*)ALLOC(sizeof(pulseqlib_SeqFile));
	mu_assert(seq != NULL, "Failed to allocate sequence object");
	pulseqlib_optsInit(&opts, 3.0f, 40.0f, 150.0f, 1.0f, 10.0f, 0.1f, 10.0f);
	opts.max_grad = 30.0f;
	pulseqlib_seqFileInit(seq, &opts);

	entries = (float (*)[7])ALLOC(sizeof(float) * 7U * 2U);
	mu_assert(entries != NULL, "Failed to allocate gradient library");
	seq->gradLibrary = entries;
	seq->gradLibrarySize = 2;
	seq->isGradLibraryParsed = 1;

	entries[0][1] = 12.0f;
	entries[1][1] = 14.0f;

	result = pulserverlib_checkMaxGradientMagnitude(seq);
	mu_assert_double_eq(14.0f, result);

	pulseqlib_seqFileFree(seq);
}

MU_TEST(test_max_grad_check_fail) {
	pulseqlib_Opts opts;
	pulseqlib_SeqFile* seq;
	float (*entries)[7];
	float result;

	seq = (pulseqlib_SeqFile*)ALLOC(sizeof(pulseqlib_SeqFile));
	mu_assert(seq != NULL, "Failed to allocate sequence object");
	pulseqlib_optsInit(&opts, 3.0f, 40.0f, 150.0f, 1.0f, 10.0f, 0.1f, 10.0f);
	opts.max_grad = 24.0f;
	pulseqlib_seqFileInit(seq, &opts);

	entries = (float (*)[7])ALLOC(sizeof(float) * 7U * 2U);
	mu_assert(entries != NULL, "Failed to allocate gradient library");
	seq->gradLibrary = entries;
	seq->gradLibrarySize = 2;
	seq->isGradLibraryParsed = 1;

	entries[0][1] = 12.0f;
	entries[1][1] = 16.0f;

	result = pulserverlib_checkMaxGradientMagnitude(seq);
	mu_assert_double_eq(-1.0f, result);

	pulseqlib_seqFileFree(seq);
}

MU_TEST(test_max_slew_check_pass) {
	pulseqlib_Opts opts;
	pulseqlib_SeqFile* seq;
	float waveX[] = {0.0f, 20.0f, 40.0f};
	float waveY[] = {0.0f, -10.0f, -10.0f};
	float waveZ[] = {0.0f};
	float* slewX;
	float* slewY;
	float* slewZ;
	int lenX;
	int lenY;
	int lenZ;
	pulserverlib_Status status;

	seq = (pulseqlib_SeqFile*)ALLOC(sizeof(pulseqlib_SeqFile));
	mu_assert(seq != NULL, "Failed to allocate sequence object");
	pulseqlib_optsInit(&opts, 3.0f, 40.0f, 150.0f, 1.0f, 10.0f, 10.0f, 10.0f);
	opts.max_slew = 5.0f;
	opts.grad_raster_time = 10.0f;
	pulseqlib_seqFileInit(seq, &opts);

	status = pulserverlib_checkMaxSlewRate(seq, waveX, 3, waveY, 3, waveZ, 1, &slewX, &lenX, &slewY, &lenY, &slewZ, &lenZ);
	mu_assert(status == PULSERVERLIB_STATUS_OK, "Slew rate check should pass");
	mu_assert_int_eq(2, lenX);
	mu_assert_int_eq(2, lenY);
	mu_assert_int_eq(0, lenZ);
	if (slewX) {
		mu_assert_double_eq(2.0f, slewX[0]);
		mu_assert_double_eq(2.0f, slewX[1]);
	}
	if (slewY) {
		mu_assert_double_eq(-1.0f, slewY[0]);
		mu_assert_double_eq(0.0f, slewY[1]);
	}
	mu_assert(slewZ == NULL, "Z slew should be NULL for single-sample waveform");

	if (slewX) FREE(slewX);
	if (slewY) FREE(slewY);
	if (slewZ) FREE(slewZ);
	pulseqlib_seqFileFree(seq);
}

MU_TEST(test_max_slew_check_first_sample_fail) {
	pulseqlib_Opts opts;
	pulseqlib_SeqFile* seq;
	float waveX[] = {60.0f};
	float waveY[] = {0.0f};
	float waveZ[] = {0.0f};
	float* slewX;
	float* slewY;
	float* slewZ;
	int lenX;
	int lenY;
	int lenZ;
	pulserverlib_Status status;

	seq = (pulseqlib_SeqFile*)ALLOC(sizeof(pulseqlib_SeqFile));
	mu_assert(seq != NULL, "Failed to allocate sequence object");
	pulseqlib_optsInit(&opts, 3.0f, 40.0f, 150.0f, 1.0f, 10.0f, 10.0f, 10.0f);
	opts.max_slew = 5.0f;
	opts.grad_raster_time = 10.0f;
	pulseqlib_seqFileInit(seq, &opts);

	status = pulserverlib_checkMaxSlewRate(seq, waveX, 1, waveY, 1, waveZ, 1, &slewX, &lenX, &slewY, &lenY, &slewZ, &lenZ);
	mu_assert(status == PULSERVERLIB_STATUS_SAFETY_LIMIT_EXCEEDED, "First sample over limit should fail");
	mu_assert(slewX == NULL && slewY == NULL && slewZ == NULL, "Outputs should remain NULL on failure");
	mu_assert(lenX == 0 && lenY == 0 && lenZ == 0, "Lengths should be zero on failure");

	pulseqlib_seqFileFree(seq);
}

MU_TEST(test_max_slew_check_derivative_fail) {
	pulseqlib_Opts opts;
	pulseqlib_SeqFile* seq;
	float waveX[] = {0.0f, 100.0f};
	float waveY[] = {0.0f};
	float waveZ[] = {0.0f};
	float* slewX;
	float* slewY;
	float* slewZ;
	int lenX;
	int lenY;
	int lenZ;
	pulserverlib_Status status;

	seq = (pulseqlib_SeqFile*)ALLOC(sizeof(pulseqlib_SeqFile));
	mu_assert(seq != NULL, "Failed to allocate sequence object");
	pulseqlib_optsInit(&opts, 3.0f, 40.0f, 150.0f, 1.0f, 10.0f, 10.0f, 10.0f);
	opts.max_slew = 5.0f;
	opts.grad_raster_time = 10.0f;
	pulseqlib_seqFileInit(seq, &opts);

	status = pulserverlib_checkMaxSlewRate(seq, waveX, 2, waveY, 1, waveZ, 1, &slewX, &lenX, &slewY, &lenY, &slewZ, &lenZ);
	mu_assert(status == PULSERVERLIB_STATUS_SAFETY_LIMIT_EXCEEDED, "Derivative over limit should fail");
	mu_assert(slewX == NULL && slewY == NULL && slewZ == NULL, "Outputs should remain NULL on failure");
	mu_assert(lenX == 0 && lenY == 0 && lenZ == 0, "Lengths should be zero on failure");

	pulseqlib_seqFileFree(seq);
}

MU_TEST_SUITE(test_math_suite) {
	MU_RUN_TEST(test_first_difference_basic);
	MU_RUN_TEST(test_first_difference_zero_length);
	MU_RUN_TEST(test_first_difference_single_sample);
	MU_RUN_TEST(test_first_difference_invalid_array);
	MU_RUN_TEST(test_first_difference_negative_length);
	MU_RUN_TEST(test_grad_library_max_basic);
	MU_RUN_TEST(test_grad_library_max_zero_entries);
	MU_RUN_TEST(test_grad_library_max_non_positive);
	MU_RUN_TEST(test_max_grad_check_pass);
	MU_RUN_TEST(test_max_grad_check_fail);
	MU_RUN_TEST(test_max_slew_check_pass);
	MU_RUN_TEST(test_max_slew_check_first_sample_fail);
	MU_RUN_TEST(test_max_slew_check_derivative_fail);
}

int test_math_main(void) {
	MU_RUN_SUITE(test_math_suite);
	MU_REPORT();
	return MU_EXIT_CODE;
}
