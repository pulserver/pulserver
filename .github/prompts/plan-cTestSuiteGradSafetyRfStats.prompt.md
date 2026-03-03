# Plan: C Test Suite — Grad Safety + RF Stats

Three files implement the first batch of C unit tests using the existing minunit framework: a shared helpers header, gradient safety tests (continuity + amplitude/slew limits), and RF tests (stat ground truth + consistency checks). The existing `tests/ctests/test_runner.c` is trimmed to only call the two implemented suites. All code must be **C89** (`-std=c89 -pedantic -Werror`).

## Steps

### 1. Create `tests/ctests/test_helpers.h`

- Include guard, then `minunit.h`, `pulseqlib_config.h`, `pulseqlib_types.h`, `pulseqlib_methods.h`, `<stdio.h>`, `<stdlib.h>`, `<math.h>`, `<string.h>`
- `#define TEST_DATA_DIR  TEST_ROOT_DIR "/tests/data/"` — paths composed at compile time from the existing `TEST_ROOT_DIR` definition in `tests/ctests/CMakeLists.txt`
- `#define GAMMA_HZ_PER_T  42577478.0f`
- Custom float assertion macro `mu_assert_float_near(msg, expected, actual, tol)` using absolute tolerance (since `mu_assert_double_eq` uses `MINUNIT_EPSILON = 1e-12` which is too tight for float). Pattern: increment `minunit_assert`, check `fabs(expected - actual) > tol`, snprintf details into `minunit_last_message`, set `minunit_status = 1`, return on failure
- `default_opts_init(pulseqlib_opts* opts)` — static helper initialising opts for the **MATLAB Pulseq default** rasters matching the grad/RF test data: γ = 42577478, B0 = 3.0, max_grad = 42577478×0.040 = 1,703,099 Hz/m, max_slew = 42577478×170 = 7,238,171,260 Hz/m/s, rf_raster = 1.0 µs, grad_raster = 10.0 µs, adc_raster = 0.1 µs, block_raster = 10.0 µs (matches rasters recorded in `01_ok_trap_extended_trap.seq` and `01_rfamp_ok_mrfingerprinting.seq`)
- `gre_opts_init(pulseqlib_opts* opts)` — opts for the segmentation generator sequences (GRE etc.): same γ/B0/limits, but rf_raster = 2.0, grad_raster = 20.0, adc_raster = 2.0, block_raster = 20.0 (matching `gre_2d_1sl_1avg.seq`)
- `load_seq(pulseqlib_collection** coll, const char* filename, const pulseqlib_opts* opts)` — static helper that calls `pulseqlib_diagnostic_init`, builds full path from `TEST_DATA_DIR`, calls `pulseqlib_read` with `cache_binary=0, verify_signature=0, parse_labels=0, num_averages=1`, returns the rc
- Forward-declare `int test_safety_grad_main(void);` and `int test_rf_stats_main(void);`

### 2. Create `tests/ctests/test_safety_grad.c`

Two suites, one `test_safety_grad_main()` entry.

#### Suite A — Gradient Limits (4 tests)

Each test: init special opts, load `.seq`, call `pulseqlib_check_safety` (no forbidden bands, no PNS), assert expected error code, free collection.

| Test | File | Opts override | Expected |
|------|------|--------------|----------|
| `test_grad_amplitude_violation` | `01_grad_amplitude_violation.seq` | `max_grad = 10.0f` | `PULSEQLIB_ERR_MAX_GRAD_EXCEEDED` |
| `test_slew_violation` | `02_slew_violation.seq` | `max_slew = 100.0f` | `PULSEQLIB_ERR_MAX_SLEW_EXCEEDED` |
| `test_grad_rss_violation` | `03_grad_rss_violation.seq` | `max_grad = 10.0f` | `PULSEQLIB_ERR_MAX_GRAD_EXCEEDED` |
| `test_slew_rss_violation` | `04_slew_rss_violation.seq` | `max_slew = 100.0f` | `PULSEQLIB_ERR_MAX_SLEW_EXCEEDED` |

For limit tests, the non-tested limit is set high (1e10) so only the target check fires. The `pulseqlib_check_safety` pipeline runs max_grad → continuity → max_slew in order and returns on first failure, so:
- Amplitude tests: max_grad fires at step 1 before continuity/slew
- Slew tests: max_grad passes (limit=1e10), continuity passes (traps start/end at 0, max_allowed=1e10×10e-6=1e4), then max_slew fires at step 3

#### Suite B — Gradient Continuity (17 tests)

Drawn from `generate_grad_continuity_test_cases.m`. Use `default_opts_init` which gives max_allowed = 7.24e9 × 10e-6 = 72,381 Hz/m — the test gradients use amplitudes of 100,000–200,000 Hz/m, so discontinuous jumps (e.g., 0→100,000) exceed this threshold while OK cases have zero-to-zero transitions.

Call `pulseqlib_check_safety` with `default_opts`, no PNS/bands, PNS threshold = 0:

| # | File | Expected |
|---|------|----------|
| 01 | `01_ok_trap_extended_trap.seq` | `PULSEQLIB_SUCCESS` |
| 02 | `02_fail_trap_then_startshigh.seq` | `PULSEQLIB_ERR_GRAD_DISCONTINUITY` |
| 03 | `03_fail_startshigh_first.seq` | `PULSEQLIB_ERR_GRAD_DISCONTINUITY` |
| 04 | `04_fail_delay_then_allhigh.seq` | `PULSEQLIB_ERR_GRAD_DISCONTINUITY` |
| 05 | `05_ok_extended_with_delay.seq` | `PULSEQLIB_SUCCESS` |
| 06 | `06_fail_delay_then_startshigh.seq` | `PULSEQLIB_ERR_GRAD_DISCONTINUITY` |
| 07 | `07_fail_nonconnecting.seq` | `PULSEQLIB_ERR_GRAD_DISCONTINUITY` |
| 08 | `08_ok_rot_identity.seq` | `PULSEQLIB_SUCCESS` |
| 09 | `09_fail_rot_identity.seq` | `PULSEQLIB_ERR_GRAD_DISCONTINUITY` |
| 10 | `10_fail_rot_first_block.seq` | `PULSEQLIB_ERR_GRAD_DISCONTINUITY` |
| 11 | `11_fail_rot_allhigh.seq` | `PULSEQLIB_ERR_GRAD_DISCONTINUITY` |
| 12 | `12_ok_rot_extended_delay.seq` | `PULSEQLIB_SUCCESS` |
| 13 | `13_fail_rot_delay_then_startshigh.seq` | `PULSEQLIB_ERR_GRAD_DISCONTINUITY` |
| 14 | `14_fail_rot_nonconnecting.seq` | `PULSEQLIB_ERR_GRAD_DISCONTINUITY` |
| 15 | `15_ok_rot_same_rotation.seq` | `PULSEQLIB_SUCCESS` |
| 16 | `16_fail_rot_diff_rotation_1.seq` | `PULSEQLIB_ERR_GRAD_DISCONTINUITY` |
| 17 | `17_fail_rot_diff_rotation_2.seq` | `PULSEQLIB_ERR_GRAD_DISCONTINUITY` |

To keep it DRY, define a data-driven helper: `static void run_safety_check_file(const char* filename, int expected_code)` that loads, calls `check_safety`, asserts `rc == expected_code`, and frees. Each `MU_TEST` is a one-liner calling this helper.

`test_safety_grad_main()`: resets minunit counters, runs both suites via `MU_RUN_SUITE`, calls `MU_REPORT()`, returns `MU_EXIT_CODE`.

### 3. Create `tests/ctests/test_rf_stats.c`

Two suites, one `test_rf_stats_main()` entry.

#### Suite A — RF180 Block Pulse Ground Truth (1 test)

Loads `00_basic_rfstat.seq` (single block containing only the 180° block pulse) with `default_opts_init`, calls `pulseqlib_get_rf_stats(coll, &stats, 0, 0)` (subseq 0, rf_idx 0 — the only RF definition).

Ground truth values (converted from GE RF_PULSE):

| GE Field | GE Value | GE Unit | → pulseqlib field | Expected | Tolerance | Conversion |
|----------|----------|---------|-------------------|----------|-----------|------------|
| abswidth | 1.0 | normalized | `abs_width` | 1.0 | 1e-4 | dimensionless, same |
| effwidth | 1.0 | normalized | `eff_width` | 1.0 | 1e-4 | dimensionless, same |
| dtycyc | 1.0 | normalized | `duty_cycle` | 1.0 | 1e-4 | dimensionless, same |
| maxpw | 1.0 | normalized | `max_pulse_width` | 1.0 | 1e-4 | dimensionless, same |
| maxb1 | 0.1174 G | Gauss | `base_amplitude_hz` | 500.0 | 1.0 | 0.1174 G × 1e-4 T/G × γ = 499.86 Hz ≈ 500 Hz |
| nom_fa | 180.0 | degrees | `flip_angle_deg` | π ≈ 3.14159 | 0.01 | field stores radians: 2π × 500 × 0.001 = π |
| nom_pw | 1000.0 | µs | `duration_us` | 999.0 | 2.0 | same unit; off-by-one raster (N-1 × raster) |
| isodelay | 500.0 | µs | `isodelay_us` | 499 | 2 | int truncation of (duration − center) |
| area | 1.0 | normalized | `area` | 0.001 | 1e-5 | area_pulseqlib = nom_pw_s × area_ge = 1e-3 × 1.0 |
| nom_bw | 3125.0 | Hz | `bandwidth_hz` | ≈3123 | 50.0 | FFT returns 0 for block pulse → fallback `3.12/duration_s` ≈ 3.12/0.000999 |
| num | 1 | count | `num_samples` | 1000 | 0 (exact) |

#### Suite B — RF Consistency (4 tests)

Each test: load `.seq`, call `pulseqlib_check_consistency`, assert expected error code.

| Test | File | Expected |
|------|------|----------|
| `test_rf_periodic_ok` | `01_rfamp_ok_mrfingerprinting.seq` | `PULSEQLIB_SUCCESS` (`> 0`) |
| `test_rf_periodic_fail` | `02_rfamp_fail_vfa.seq` | `PULSEQLIB_ERR_CONSISTENCY_RF_PERIODIC` |
| `test_rfshim_periodic_ok` | `03_rfshim_ok_pnpmrfingerprinting.seq` | `PULSEQLIB_SUCCESS` (`> 0`) |
| `test_rfshim_periodic_fail` | `04_rfshim_fail_gre.seq` | `PULSEQLIB_ERR_CONSISTENCY_RF_SHIM_PERIODIC` |

`test_rf_stats_main()`: resets counters, runs both suites, report, return exit code.

### 4. Update `tests/ctests/test_runner.c`

Comment out the 11 unimplemented suite calls. Keep only `test_safety_grad_main()` and `test_rf_stats_main()`.

### 5. No changes to `tests/ctests/CMakeLists.txt`

It already globs `*.c`, links `pulseqlib m`, passes `TEST_ROOT_DIR`, and compiles with `-std=c89 -pedantic -Werror`.

Note: `scripts/build_ctests.sh` references `SOURCE_DIR="tests/csrc"` but the actual CMakeLists.txt is in `tests/ctests` — either a symlink exists or the script needs updating to `tests/ctests`.

## Verification

- Build: `cd tests/ctests && cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_C_COMPILER=gcc && cmake --build build --target run_tests`
- Run: `build/bin/run_tests`
- Expected output: 26 tests total (4 limit + 17 continuity + 1 RF stats + 4 RF consistency), all passing. Each test prints `.` on pass; final line: `OVERALL: All suites PASSED.`
- Verify the build produces no warnings (enforced by `-Werror`)

## Decisions

- **`flip_angle_deg` stores radians**: the formula `2π × base_amp_hz × area_s` produces radians despite the field name — test asserts against π, not 180.0
- **Bandwidth fallback**: block pulse produces FFT BW ≤ 0, so `bandwidth_hz = 3.12 / duration_s ≈ 3123 Hz` (not the GE 3125; the library constant is 3.12, not 3.125)
- **Grad limit opts per test**: amplitude tests use `max_grad = 10`, slew tests use `max_slew = 100` (in Hz/m and Hz/m/s respectively — matching the raw .seq file gradient units); non-tested limits are set to 1e10
- **Continuity opts**: MATLAB Pulseq defaults (max_slew ≈ 7.24e9 Hz/m/s), giving `max_allowed = 72,381 Hz/m` per grad raster step — safely below the 100,000 Hz/m jumps in the fail test cases but above zero for passing cases
- **Data-driven tests**: continuity suite uses a helper that takes filename + expected code, keeping each `MU_TEST` a single call — avoids 17 near-identical functions
- **GRE not needed for this batch**: All tests load purpose-built test data from the MATLAB generators; the `gre_2d_1sl_1avg.seq` sequence is prepared in `test_helpers.h` via `gre_opts_init` for future suites (load, structure, segments, etc.)
