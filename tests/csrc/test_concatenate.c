#include "minunit.h"

#include <stdlib.h>

#include "pulseqlib_methods.h"
#include "pulserverlib.h"
#include "pulserverlib_methods.h"

MU_TEST(test_concatenate_basic) {
	pulserverlib_Status status;
	const float arr0[] = {1.0f, 2.0f};
	const float arr1[] = {3.0f, 4.0f, 5.0f};
	const float* arrays[2];
	int lengths[2];
	int sequence[4];
	float* result;
	int resultLength;
	const float expected[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
	int i;

	arrays[0] = arr0;
	arrays[1] = arr1;
	lengths[0] = 2;
	lengths[1] = 3;
	sequence[0] = 0;
	sequence[1] = 1;
	sequence[2] = 0;
	sequence[3] = 1;
	result = NULL;
	resultLength = -1;

	status = pulserverlib_concatenateIndexedFloatArrays(arrays, lengths, 2, sequence, 4, &result, &resultLength);
	mu_assert(status == PULSERVERLIB_STATUS_OK, "Concatenation should succeed");
	mu_assert_int_eq(10, resultLength);

	for (i = 0; i < resultLength; ++i) {
		mu_assert(result[i] == expected[i], "Unexpected value in concatenated result");
	}

	FREE(result);
}

MU_TEST(test_concatenate_empty_sequence) {
	pulserverlib_Status status;
	const float arr0[] = {42.0f};
	const float* arrays[1];
	int lengths[1];
	int sequence[1];
	float* result;
	int resultLength;
	float marker;

	arrays[0] = arr0;
	lengths[0] = 1;
	sequence[0] = 0;
	marker = 0.0f;
	result = &marker;
	resultLength = 123;

	status = pulserverlib_concatenateIndexedFloatArrays(arrays, lengths, 1, sequence, 0, &result, &resultLength);
	mu_assert(status == PULSERVERLIB_STATUS_OK, "Empty sequence should be allowed");
	mu_assert(result == NULL, "Result buffer should remain NULL when sequence is empty");
	mu_assert_int_eq(0, resultLength);
}

MU_TEST(test_concatenate_invalid_index) {
	pulserverlib_Status status;
	const float arr0[] = {1.0f};
	const float* arrays[1];
	int lengths[1];
	int sequence[1];
	float* result;
	int resultLength;
	float marker;

	arrays[0] = arr0;
	lengths[0] = 1;
	sequence[0] = 2;
	marker = 0.0f;
	result = &marker;
	resultLength = 1;

	status = pulserverlib_concatenateIndexedFloatArrays(arrays, lengths, 1, sequence, 1, &result, &resultLength);
	mu_assert(status == PULSERVERLIB_STATUS_INVALID_ARGUMENT, "Out-of-range index should fail");
	mu_assert(result == NULL, "Result buffer should be cleared on failure");
	mu_assert_int_eq(0, resultLength);
}

MU_TEST(test_concatenate_negative_length) {
	pulserverlib_Status status;
	const float arr0[] = {1.0f};
	const float arr1[] = {2.0f};
	const float* arrays[2];
	int lengths[2];
	int sequence[2];
	float* result;
	int resultLength;
	float marker;

	arrays[0] = arr0;
	arrays[1] = arr1;
	lengths[0] = 1;
	lengths[1] = -1;
	sequence[0] = 0;
	sequence[1] = 1;
	marker = 0.0f;
	result = &marker;
	resultLength = 1;

	status = pulserverlib_concatenateIndexedFloatArrays(arrays, lengths, 2, sequence, 2, &result, &resultLength);
	mu_assert(status == PULSERVERLIB_STATUS_INVALID_ARGUMENT, "Negative length should fail");
	mu_assert(result == NULL, "Result buffer should be cleared on failure");
	mu_assert_int_eq(0, resultLength);
}

MU_TEST_SUITE(test_concatenate_suite) {
	MU_RUN_TEST(test_concatenate_basic);
	MU_RUN_TEST(test_concatenate_empty_sequence);
	MU_RUN_TEST(test_concatenate_invalid_index);
	MU_RUN_TEST(test_concatenate_negative_length);
}

int test_concatenate_main(void) {
	MU_RUN_SUITE(test_concatenate_suite);
	MU_REPORT();
	return MU_EXIT_CODE;
}
