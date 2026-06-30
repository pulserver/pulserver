/*
 * test_protocol.c -- protocol parse / serialize / getter / setter tests.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "minunit.h"
#include "pulseqlib_protocol.h"

/* ================================================================== */
/*  Test: param_find / wire_name / get_type round-trip                */
/* ================================================================== */

MU_TEST(test_param_lookup)
{
    int id;

    id = pulseqlib_param_find("TE");
    mu_assert_int_eq(PULSEQLIB_PARAM_TE, id);
    mu_assert_string_eq("TE", pulseqlib_param_wire_name(id));
    mu_assert_int_eq(PULSEQLIB_PTYPE_FLOAT, pulseqlib_param_get_type(id));

    id = pulseqlib_param_find("nslices");
    mu_assert_int_eq(PULSEQLIB_PARAM_NSLICES, id);
    mu_assert_int_eq(PULSEQLIB_PTYPE_INT, pulseqlib_param_get_type(id));

    id = pulseqlib_param_find("FatSat");
    mu_assert_int_eq(PULSEQLIB_PARAM_FAT_SAT, id);
    mu_assert_int_eq(PULSEQLIB_PTYPE_BOOL, pulseqlib_param_get_type(id));

    /* Unknown key */
    mu_assert_int_eq(-1, pulseqlib_param_find("Bogus"));
    mu_assert(pulseqlib_param_wire_name(-1) == NULL, "wire_name(-1)");
    mu_assert(pulseqlib_param_wire_name(PULSEQLIB_PARAM_COUNT) == NULL,
              "wire_name(COUNT)");
}

/* ================================================================== */
/*  Test: user slots use Python-facing zero-based names               */
/* ================================================================== */

MU_TEST(test_user_slots)
{
    mu_assert_int_eq(PULSEQLIB_PARAM_USER1, pulseqlib_param_find("user0_value"));
    mu_assert_int_eq(PULSEQLIB_PARAM_USER2, pulseqlib_param_find("user1_value"));
    mu_assert_int_eq(PULSEQLIB_PARAM_USER17, pulseqlib_param_find("user16_value"));
}

/* ================================================================== */
/*  Test: parse a preamble                                            */
/* ================================================================== */

static const char *PREAMBLE =
    "[NimPulseqGUI Protocol]\n"
    "TE: 5.0\n"
    "TR: 500.0\n"
    "nslices: 10\n"
    "FatSat: true\n"
    "user0_value: 42.5\n"
    "[NimPulseqGUI Protocol End]\n";

MU_TEST(test_parse)
{
    pulseqlib_protocol proto;
    int rc;
    float fval;
    int ival, bval;

    rc = pulseqlib_protocol_parse(&proto, PREAMBLE);
    mu_assert(rc == 5, "expected 5 parsed params");

    mu_assert_int_eq(0, pulseqlib_protocol_get_float(&proto,
                                                     PULSEQLIB_PARAM_TE, &fval));
    mu_assert(fabsf(fval - 5.0f) < 1e-6f, "TE value");

    mu_assert_int_eq(0, pulseqlib_protocol_get_float(&proto,
                                                     PULSEQLIB_PARAM_TR, &fval));
    mu_assert(fabsf(fval - 500.0f) < 1e-3f, "TR value");

    mu_assert_int_eq(0, pulseqlib_protocol_get_int(&proto,
                                                   PULSEQLIB_PARAM_NSLICES, &ival));
    mu_assert_int_eq(10, ival);

    mu_assert_int_eq(0, pulseqlib_protocol_get_bool(&proto,
                                                    PULSEQLIB_PARAM_FAT_SAT, &bval));
    mu_assert_int_eq(1, bval);

    mu_assert_int_eq(0, pulseqlib_protocol_get_float(&proto,
                                                     PULSEQLIB_PARAM_USER1, &fval));
    mu_assert(fabsf(fval - 42.5f) < 1e-6f, "user0 value");
}

/* ================================================================== */
/*  Test: parse with comment-prefix lines                             */
/* ================================================================== */

static const char *PREAMBLE_COMMENTED =
    "# [NimPulseqGUI Protocol]\n"
    "# TE: 3.0\n"
    "# [NimPulseqGUI Protocol End]\n";

MU_TEST(test_parse_commented)
{
    pulseqlib_protocol proto;
    int rc;
    float fval;

    rc = pulseqlib_protocol_parse(&proto, PREAMBLE_COMMENTED);
    mu_assert(rc == 1, "expected 1 parsed param");

    mu_assert_int_eq(0, pulseqlib_protocol_get_float(&proto,
                                                     PULSEQLIB_PARAM_TE, &fval));
    mu_assert(fabsf(fval - 3.0f) < 1e-6f, "TE from commented preamble");
}

/* ================================================================== */
/*  Test: typed setters + getters                                     */
/* ================================================================== */

MU_TEST(test_setters)
{
    pulseqlib_protocol proto;
    float fval;
    int ival, bval;

    memset(&proto, 0, sizeof(proto));

    mu_assert_int_eq(0, pulseqlib_protocol_set_float(&proto,
                                                     PULSEQLIB_PARAM_FOV, 240.0f));
    mu_assert_int_eq(0, pulseqlib_protocol_set_int(&proto,
                                                   PULSEQLIB_PARAM_MATRIX, 256));
    mu_assert_int_eq(0, pulseqlib_protocol_set_bool(&proto,
                                                    PULSEQLIB_PARAM_SPOILER, 1));

    mu_assert_int_eq(0, pulseqlib_protocol_get_float(&proto,
                                                     PULSEQLIB_PARAM_FOV, &fval));
    mu_assert(fabsf(fval - 240.0f) < 1e-6f, "fov");

    mu_assert_int_eq(0, pulseqlib_protocol_get_int(&proto,
                                                   PULSEQLIB_PARAM_MATRIX, &ival));
    mu_assert_int_eq(256, ival);

    mu_assert_int_eq(0, pulseqlib_protocol_get_bool(&proto,
                                                    PULSEQLIB_PARAM_SPOILER, &bval));
    mu_assert_int_eq(1, bval);

    /* Type mismatch: getting float from int slot should fail */
    mu_assert_int_eq(-1, pulseqlib_protocol_get_float(&proto,
                                                      PULSEQLIB_PARAM_MATRIX, &fval));
}

/* ================================================================== */
/*  Test: serialize round-trip                                        */
/* ================================================================== */

MU_TEST(test_roundtrip)
{
    pulseqlib_protocol p1, p2;
    char buf[2048];
    int n, rc;
    float fval;
    int ival;

    memset(&p1, 0, sizeof(p1));
    pulseqlib_protocol_set_float(&p1, PULSEQLIB_PARAM_TE, 3.5f);
    pulseqlib_protocol_set_int(&p1, PULSEQLIB_PARAM_NSLICES, 20);
    pulseqlib_protocol_set_bool(&p1, PULSEQLIB_PARAM_RF_SPOILING, 1);

    n = pulseqlib_protocol_serialize(&p1, buf, sizeof(buf));
    mu_assert(n > 0, "serialize should succeed");

    rc = pulseqlib_protocol_parse(&p2, buf);
    mu_assert(rc == 3, "round-trip should parse 3 params");

    mu_assert_int_eq(0, pulseqlib_protocol_get_float(&p2,
                                                     PULSEQLIB_PARAM_TE, &fval));
    mu_assert(fabsf(fval - 3.5f) < 1e-3f, "TE round-trip");

    mu_assert_int_eq(0, pulseqlib_protocol_get_int(&p2,
                                                   PULSEQLIB_PARAM_NSLICES, &ival));
    mu_assert_int_eq(20, ival);
}

/* ================================================================== */
/*  Test: rich schema format "type|value|min|max|incr|unit"           */
/* ================================================================== */

static const char *PREAMBLE_RICH =
    "[NimPulseqGUI Protocol]\n"
    "TE: float|5.0|0.5|100.0|0.1|ms\n"
    "TR: float|500.0|10.0|10000.0|1.0|ms\n"
    "nslices: int|10|1|256|1|slices\n"
    "FatSat: bool|1\n"
    "user0_value: float|42.5|-100.0|100.0|0.5|\n"
    "[NimPulseqGUI Protocol End]\n";

MU_TEST(test_parse_rich)
{
    pulseqlib_protocol proto;
    int rc, idx;
    float fval;
    int ival, bval;
    const pulseqlib_protocol_value *pv;

    rc = pulseqlib_protocol_parse(&proto, PREAMBLE_RICH);
    mu_assert(rc == 5, "expected 5 parsed params (rich)");

    /* TE: value + schema */
    mu_assert_int_eq(0, pulseqlib_protocol_get_float(&proto,
                                                     PULSEQLIB_PARAM_TE, &fval));
    mu_assert(fabsf(fval - 5.0f) < 1e-6f, "TE value (rich)");

    idx = pulseqlib_protocol_find(&proto, PULSEQLIB_PARAM_TE);
    mu_assert(idx >= 0, "TE found");
    pv = &proto.values[idx];
    mu_assert_int_eq(1, pv->has_schema);
    mu_assert(fabsf(pv->range_min - 0.5f) < 1e-6f, "TE min");
    mu_assert(fabsf(pv->range_max - 100.0f) < 1e-6f, "TE max");
    mu_assert(fabsf(pv->range_incr - 0.1f) < 1e-6f, "TE incr");
    mu_assert_string_eq("ms", pv->unit);

    /* TR: value + schema */
    mu_assert_int_eq(0, pulseqlib_protocol_get_float(&proto,
                                                     PULSEQLIB_PARAM_TR, &fval));
    mu_assert(fabsf(fval - 500.0f) < 1e-3f, "TR value (rich)");

    idx = pulseqlib_protocol_find(&proto, PULSEQLIB_PARAM_TR);
    pv = &proto.values[idx];
    mu_assert_int_eq(1, pv->has_schema);
    mu_assert(fabsf(pv->range_max - 10000.0f) < 1e-1f, "TR max");

    /* NSlices: int with schema */
    mu_assert_int_eq(0, pulseqlib_protocol_get_int(&proto,
                                                   PULSEQLIB_PARAM_NSLICES, &ival));
    mu_assert_int_eq(10, ival);

    idx = pulseqlib_protocol_find(&proto, PULSEQLIB_PARAM_NSLICES);
    pv = &proto.values[idx];
    mu_assert_int_eq(1, pv->has_schema);
    mu_assert(fabsf(pv->range_min - 1.0f) < 1e-6f, "NSlices min");
    mu_assert(fabsf(pv->range_max - 256.0f) < 1e-6f, "NSlices max");
    mu_assert_string_eq("slices", pv->unit);

    /* FatSat: bool (no range schema, but has_schema=0 since bool has no min/max) */
    mu_assert_int_eq(0, pulseqlib_protocol_get_bool(&proto,
                                                    PULSEQLIB_PARAM_FAT_SAT, &bval));
    mu_assert_int_eq(1, bval);

    /* user0_value: float with schema, empty unit */
    mu_assert_int_eq(0, pulseqlib_protocol_get_float(&proto,
                                                     PULSEQLIB_PARAM_USER1, &fval));
    mu_assert(fabsf(fval - 42.5f) < 1e-6f, "user0 (rich)");

    idx = pulseqlib_protocol_find(&proto, PULSEQLIB_PARAM_USER1);
    pv = &proto.values[idx];
    mu_assert_int_eq(1, pv->has_schema);
    mu_assert(fabsf(pv->range_min - (-100.0f)) < 1e-4f, "user0 min");
}

/* ================================================================== */
/*  Test: rich + simple mixed (backward compat)                       */
/* ================================================================== */

static const char *PREAMBLE_MIXED =
    "[NimPulseqGUI Protocol]\n"
    "TE: float|5.0|0.5|100.0|0.1|ms\n"
    "TR: 500.0\n"
    "nslices: int|10|1|256|1|\n"
    "FatSat: true\n"
    "[NimPulseqGUI Protocol End]\n";

MU_TEST(test_parse_mixed)
{
    pulseqlib_protocol proto;
    int rc, idx;
    float fval;
    int ival, bval;

    rc = pulseqlib_protocol_parse(&proto, PREAMBLE_MIXED);
    mu_assert(rc == 4, "expected 4 from mixed format");

    /* TE has schema */
    idx = pulseqlib_protocol_find(&proto, PULSEQLIB_PARAM_TE);
    mu_assert(idx >= 0, "TE");
    mu_assert_int_eq(1, proto.values[idx].has_schema);
    mu_assert_int_eq(PULSEQLIB_MODE_TYPEIN, (int)proto.values[idx].mode);

    /* TR is simple — no schema, mode defaults to TYPEIN */
    mu_assert_int_eq(0, pulseqlib_protocol_get_float(&proto,
                                                     PULSEQLIB_PARAM_TR, &fval));
    mu_assert(fabsf(fval - 500.0f) < 1e-3f, "TR simple");
    idx = pulseqlib_protocol_find(&proto, PULSEQLIB_PARAM_TR);
    mu_assert_int_eq(0, proto.values[idx].has_schema);
    mu_assert_int_eq(PULSEQLIB_MODE_TYPEIN, (int)proto.values[idx].mode);

    /* NSlices has schema */
    idx = pulseqlib_protocol_find(&proto, PULSEQLIB_PARAM_NSLICES);
    mu_assert_int_eq(1, proto.values[idx].has_schema);
    mu_assert_int_eq(0, pulseqlib_protocol_get_int(&proto,
                                                   PULSEQLIB_PARAM_NSLICES, &ival));
    mu_assert_int_eq(10, ival);

    /* FatSat is simple */
    mu_assert_int_eq(0, pulseqlib_protocol_get_bool(&proto,
                                                    PULSEQLIB_PARAM_FAT_SAT, &bval));
    mu_assert_int_eq(1, bval);
    idx = pulseqlib_protocol_find(&proto, PULSEQLIB_PARAM_FAT_SAT);
    mu_assert_int_eq(0, proto.values[idx].has_schema);
}

/* ================================================================== */
/*  Test: dropdown wire format with mode + options                    */
/* ================================================================== */

static const char *PREAMBLE_DROPDOWN =
    "[NimPulseqGUI Protocol]\n"
    "TE: float|dropdown|12.0|5.0|80.0|1.0|ms|8.0|12.0|16.0\n"
    "TR: float|typein|500.0|10.0|10000.0|1.0|ms\n"
    "nx: int|dropdown|128|64|512|1||64|128|256\n"
    "bandwidth: float|off|125000.0|10000.0|500000.0|1000.0|Hz/px\n"
    "nslices: int|10|1|256|1|\n"
    "[NimPulseqGUI Protocol End]\n";

MU_TEST(test_parse_dropdown)
{
    pulseqlib_protocol proto;
    int rc, idx;
    const pulseqlib_protocol_value *pv;

    rc = pulseqlib_protocol_parse(&proto, PREAMBLE_DROPDOWN);
    mu_assert(rc == 5, "expected 5 parsed params (dropdown)");

    /* TE: dropdown with 3 options */
    idx = pulseqlib_protocol_find(&proto, PULSEQLIB_PARAM_TE);
    mu_assert(idx >= 0, "TE found");
    pv = &proto.values[idx];
    mu_assert_int_eq(PULSEQLIB_MODE_DROPDOWN, (int)pv->mode);
    mu_assert_int_eq(PULSEQLIB_PTYPE_FLOAT, (int)pv->type);
    mu_assert(fabsf(pv->v.f - 12.0f) < 1e-6f, "TE value");
    mu_assert_int_eq(1, pv->has_schema);
    mu_assert(fabsf(pv->range_min - 5.0f) < 1e-6f, "TE min");
    mu_assert(fabsf(pv->range_max - 80.0f) < 1e-6f, "TE max");
    mu_assert(fabsf(pv->range_incr - 1.0f) < 1e-6f, "TE incr");
    mu_assert_string_eq("ms", pv->unit);
    mu_assert_int_eq(3, pv->num_options);
    mu_assert(fabsf(pv->options[0] - 8.0f) < 1e-6f, "TE opt[0]");
    mu_assert(fabsf(pv->options[1] - 12.0f) < 1e-6f, "TE opt[1]");
    mu_assert(fabsf(pv->options[2] - 16.0f) < 1e-6f, "TE opt[2]");

    /* TR: explicit typein mode */
    idx = pulseqlib_protocol_find(&proto, PULSEQLIB_PARAM_TR);
    mu_assert(idx >= 0, "TR found");
    pv = &proto.values[idx];
    mu_assert_int_eq(PULSEQLIB_MODE_TYPEIN, (int)pv->mode);
    mu_assert(fabsf(pv->v.f - 500.0f) < 1e-3f, "TR value");
    mu_assert_int_eq(1, pv->has_schema);
    mu_assert_int_eq(0, pv->num_options);

    /* nx: int dropdown with 3 options */
    idx = pulseqlib_protocol_find(&proto, PULSEQLIB_PARAM_MATRIX);
    mu_assert(idx >= 0, "nx found");
    pv = &proto.values[idx];
    mu_assert_int_eq(PULSEQLIB_MODE_DROPDOWN, (int)pv->mode);
    mu_assert_int_eq(PULSEQLIB_PTYPE_INT, (int)pv->type);
    mu_assert_int_eq(128, pv->v.i);
    mu_assert_int_eq(3, pv->num_options);
    mu_assert(fabsf(pv->options[0] - 64.0f) < 1e-6f, "nx opt[0]");
    mu_assert(fabsf(pv->options[1] - 128.0f) < 1e-6f, "nx opt[1]");
    mu_assert(fabsf(pv->options[2] - 256.0f) < 1e-6f, "nx opt[2]");

    /* bandwidth: off mode */
    idx = pulseqlib_protocol_find(&proto, PULSEQLIB_PARAM_BANDWIDTH);
    mu_assert(idx >= 0, "bandwidth found");
    pv = &proto.values[idx];
    mu_assert_int_eq(PULSEQLIB_MODE_OFF, (int)pv->mode);
    mu_assert(fabsf(pv->v.f - 125000.0f) < 1.0f, "bandwidth value");
    mu_assert_int_eq(1, pv->has_schema);
    mu_assert_int_eq(0, pv->num_options);

    /* nslices: old format (no explicit mode) — defaults to TYPEIN */
    idx = pulseqlib_protocol_find(&proto, PULSEQLIB_PARAM_NSLICES);
    mu_assert(idx >= 0, "nslices found");
    pv = &proto.values[idx];
    mu_assert_int_eq(PULSEQLIB_MODE_TYPEIN, (int)pv->mode);
    mu_assert_int_eq(10, pv->v.i);
}

/* ================================================================== */
/*  Suite setup                                                       */
/* ================================================================== */

MU_TEST_SUITE(protocol_suite)
{
    MU_RUN_TEST(test_param_lookup);
    MU_RUN_TEST(test_user_slots);
    MU_RUN_TEST(test_parse);
    MU_RUN_TEST(test_parse_commented);
    MU_RUN_TEST(test_setters);
    MU_RUN_TEST(test_roundtrip);
    MU_RUN_TEST(test_parse_rich);
    MU_RUN_TEST(test_parse_mixed);
    MU_RUN_TEST(test_parse_dropdown);
}

int test_protocol_main(void)
{
    MU_RUN_SUITE(protocol_suite);
    MU_REPORT();
    return minunit_fail;
}
