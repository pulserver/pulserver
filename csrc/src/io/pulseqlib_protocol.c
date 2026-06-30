/**
 * @file pulseqlib_protocol.c
 * @brief Vendor-neutral MR protocol parse / serialize.
 *
 * Implements the parameter lookup table, preamble parser, and
 * serializer for the NimPulseqGUI wire format.  Pure C89, no
 * vendor dependencies.
 */

#include "pulseqlib_protocol.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Static parameter table                                            */
/* ------------------------------------------------------------------ */

static const pulseqlib_param_entry g_param_table[] = {
    /* Timing */
    {"TE", PULSEQLIB_PARAM_TE, PULSEQLIB_PTYPE_FLOAT},
    {"TR", PULSEQLIB_PARAM_TR, PULSEQLIB_PTYPE_FLOAT},
    {"prep_time", PULSEQLIB_PARAM_TI, PULSEQLIB_PTYPE_FLOAT},
    {"TE2", PULSEQLIB_PARAM_TE2, PULSEQLIB_PTYPE_FLOAT},
    {"Trecovery", PULSEQLIB_PARAM_TRECOVERY, PULSEQLIB_PTYPE_FLOAT},
    /* Spatial */
    {"fov", PULSEQLIB_PARAM_FOV, PULSEQLIB_PTYPE_FLOAT},
    {"phase_fov", PULSEQLIB_PARAM_PHASE_FOV, PULSEQLIB_PTYPE_FLOAT},
    {"slice_thickness", PULSEQLIB_PARAM_SLICE_THICKNESS, PULSEQLIB_PTYPE_FLOAT},
    {"slice_spacing", PULSEQLIB_PARAM_SLICE_SPACING, PULSEQLIB_PTYPE_FLOAT},
    {"nslices", PULSEQLIB_PARAM_NSLICES, PULSEQLIB_PTYPE_INT},
    {"nx", PULSEQLIB_PARAM_MATRIX, PULSEQLIB_PTYPE_INT},
    {"ny", PULSEQLIB_PARAM_NY, PULSEQLIB_PTYPE_INT},
    {"num_slabs", PULSEQLIB_PARAM_NUM_SLABS, PULSEQLIB_PTYPE_INT},
    {"overlap_locations", PULSEQLIB_PARAM_OVERLAP_LOCS, PULSEQLIB_PTYPE_INT},
    /* Acquisition */
    {"num_echoes", PULSEQLIB_PARAM_NECHOES, PULSEQLIB_PTYPE_INT},
    {"nex", PULSEQLIB_PARAM_NEX, PULSEQLIB_PTYPE_FLOAT},
    {"num_shots", PULSEQLIB_PARAM_NUM_SHOTS, PULSEQLIB_PTYPE_INT},
    {"etl", PULSEQLIB_PARAM_ETL, PULSEQLIB_PTYPE_INT},
    /* Contrast */
    {"flip", PULSEQLIB_PARAM_FLIP_ANGLE, PULSEQLIB_PTYPE_FLOAT},
    {"bandwidth", PULSEQLIB_PARAM_BANDWIDTH, PULSEQLIB_PTYPE_FLOAT},
    /* Enum / stringlist */
    {"sequence_type", PULSEQLIB_PARAM_SEQUENCE_TYPE, PULSEQLIB_PTYPE_STRINGLIST},
    {"imaging_mode", PULSEQLIB_PARAM_IMAGING_MODE, PULSEQLIB_PTYPE_STRINGLIST},
    {"preparation_type", PULSEQLIB_PARAM_PREP_TYPE, PULSEQLIB_PTYPE_STRINGLIST},
    {"trigger_type", PULSEQLIB_PARAM_TRIGGER_TYPE, PULSEQLIB_PTYPE_STRINGLIST},
    /* Flags */
    {"FatSat", PULSEQLIB_PARAM_FAT_SAT, PULSEQLIB_PTYPE_BOOL},
    {"Spoiler", PULSEQLIB_PARAM_SPOILER, PULSEQLIB_PTYPE_BOOL},
    {"RFSpoiling", PULSEQLIB_PARAM_RF_SPOILING, PULSEQLIB_PTYPE_BOOL},
    {"swap_phase_freq", PULSEQLIB_PARAM_SWAP_PF, PULSEQLIB_PTYPE_BOOL},
    {"enable_saturation_ui", PULSEQLIB_PARAM_ENABLE_SAT_UI, PULSEQLIB_PTYPE_BOOL},
    {"record_physio", PULSEQLIB_PARAM_RECORD_PHYSIO, PULSEQLIB_PTYPE_BOOL},
    /* Acceleration */
    {"Ry", PULSEQLIB_PARAM_RY, PULSEQLIB_PTYPE_FLOAT},
    {"Rz", PULSEQLIB_PARAM_RZ, PULSEQLIB_PTYPE_FLOAT},
    {"compressed_sensing", PULSEQLIB_PARAM_COMPRESSED_SENS, PULSEQLIB_PTYPE_FLOAT},
    {"multiband", PULSEQLIB_PARAM_MULTIBAND, PULSEQLIB_PTYPE_FLOAT},
    /* Cine / trigger */
    {"num_frames", PULSEQLIB_PARAM_NUM_FRAMES, PULSEQLIB_PTYPE_INT},
    {"delay_time", PULSEQLIB_PARAM_DELAY_TIME, PULSEQLIB_PTYPE_FLOAT},
    {"trigger_delay", PULSEQLIB_PARAM_TRIGGER_DELAY, PULSEQLIB_PTYPE_FLOAT},
    {"trigger_window", PULSEQLIB_PARAM_TRIGGER_WINDOW, PULSEQLIB_PTYPE_FLOAT},
    /* Diffusion */
    {"diffusion_bvalues", PULSEQLIB_PARAM_DIFF_BVALUES, PULSEQLIB_PTYPE_FLOAT},
    {"diffusion_directions", PULSEQLIB_PARAM_DIFF_DIRECTIONS, PULSEQLIB_PTYPE_INT},
    /* Saturation bands */
    {"sat_x", PULSEQLIB_PARAM_SAT_X, PULSEQLIB_PTYPE_INT},
    {"sat_y", PULSEQLIB_PARAM_SAT_Y, PULSEQLIB_PTYPE_INT},
    {"sat_z", PULSEQLIB_PARAM_SAT_Z, PULSEQLIB_PTYPE_INT},
    {"sat_x_loc1", PULSEQLIB_PARAM_SAT_X_LOC1, PULSEQLIB_PTYPE_FLOAT},
    {"sat_x_loc2", PULSEQLIB_PARAM_SAT_X_LOC2, PULSEQLIB_PTYPE_FLOAT},
    {"sat_y_loc1", PULSEQLIB_PARAM_SAT_Y_LOC1, PULSEQLIB_PTYPE_FLOAT},
    {"sat_y_loc2", PULSEQLIB_PARAM_SAT_Y_LOC2, PULSEQLIB_PTYPE_FLOAT},
    {"sat_z_loc1", PULSEQLIB_PARAM_SAT_Z_LOC1, PULSEQLIB_PTYPE_FLOAT},
    {"sat_z_loc2", PULSEQLIB_PARAM_SAT_Z_LOC2, PULSEQLIB_PTYPE_FLOAT},
    {"sat_x_thickness", PULSEQLIB_PARAM_SAT_X_THICK, PULSEQLIB_PTYPE_FLOAT},
    {"sat_y_thickness", PULSEQLIB_PARAM_SAT_Y_THICK, PULSEQLIB_PTYPE_FLOAT},
    {"sat_z_thickness", PULSEQLIB_PARAM_SAT_Z_THICK, PULSEQLIB_PTYPE_FLOAT},
    /* Info */
    {"TA", PULSEQLIB_PARAM_TA, PULSEQLIB_PTYPE_FLOAT},
    /* User CVs: user0..user16 map to GE opuser3..opuser19. */
    {"user0_value", PULSEQLIB_PARAM_USER1, PULSEQLIB_PTYPE_FLOAT},
    {"user1_value", PULSEQLIB_PARAM_USER2, PULSEQLIB_PTYPE_FLOAT},
    {"user2_value", PULSEQLIB_PARAM_USER3, PULSEQLIB_PTYPE_FLOAT},
    {"user3_value", PULSEQLIB_PARAM_USER4, PULSEQLIB_PTYPE_FLOAT},
    {"user4_value", PULSEQLIB_PARAM_USER5, PULSEQLIB_PTYPE_FLOAT},
    {"user5_value", PULSEQLIB_PARAM_USER6, PULSEQLIB_PTYPE_FLOAT},
    {"user6_value", PULSEQLIB_PARAM_USER7, PULSEQLIB_PTYPE_FLOAT},
    {"user7_value", PULSEQLIB_PARAM_USER8, PULSEQLIB_PTYPE_FLOAT},
    {"user8_value", PULSEQLIB_PARAM_USER9, PULSEQLIB_PTYPE_FLOAT},
    {"user9_value", PULSEQLIB_PARAM_USER10, PULSEQLIB_PTYPE_FLOAT},
    {"user10_value", PULSEQLIB_PARAM_USER11, PULSEQLIB_PTYPE_FLOAT},
    {"user11_value", PULSEQLIB_PARAM_USER12, PULSEQLIB_PTYPE_FLOAT},
    {"user12_value", PULSEQLIB_PARAM_USER13, PULSEQLIB_PTYPE_FLOAT},
    {"user13_value", PULSEQLIB_PARAM_USER14, PULSEQLIB_PTYPE_FLOAT},
    {"user14_value", PULSEQLIB_PARAM_USER15, PULSEQLIB_PTYPE_FLOAT},
    {"user15_value", PULSEQLIB_PARAM_USER16, PULSEQLIB_PTYPE_FLOAT},
    {"user16_value", PULSEQLIB_PARAM_USER17, PULSEQLIB_PTYPE_FLOAT},
    {"user17_value", PULSEQLIB_PARAM_USER18, PULSEQLIB_PTYPE_FLOAT},
    {"user18_value", PULSEQLIB_PARAM_USER19, PULSEQLIB_PTYPE_FLOAT},
    /* User CV name labels */
    {"user0_name", PULSEQLIB_PARAM_USER1_NAME, PULSEQLIB_PTYPE_DESCRIPTION},
    {"user1_name", PULSEQLIB_PARAM_USER2_NAME, PULSEQLIB_PTYPE_DESCRIPTION},
    {"user2_name", PULSEQLIB_PARAM_USER3_NAME, PULSEQLIB_PTYPE_DESCRIPTION},
    {"user3_name", PULSEQLIB_PARAM_USER4_NAME, PULSEQLIB_PTYPE_DESCRIPTION},
    {"user4_name", PULSEQLIB_PARAM_USER5_NAME, PULSEQLIB_PTYPE_DESCRIPTION},
    {"user5_name", PULSEQLIB_PARAM_USER6_NAME, PULSEQLIB_PTYPE_DESCRIPTION},
    {"user6_name", PULSEQLIB_PARAM_USER7_NAME, PULSEQLIB_PTYPE_DESCRIPTION},
    {"user7_name", PULSEQLIB_PARAM_USER8_NAME, PULSEQLIB_PTYPE_DESCRIPTION},
    {"user8_name", PULSEQLIB_PARAM_USER9_NAME, PULSEQLIB_PTYPE_DESCRIPTION},
    {"user9_name", PULSEQLIB_PARAM_USER10_NAME, PULSEQLIB_PTYPE_DESCRIPTION},
    {"user10_name", PULSEQLIB_PARAM_USER11_NAME, PULSEQLIB_PTYPE_DESCRIPTION},
    {"user11_name", PULSEQLIB_PARAM_USER12_NAME, PULSEQLIB_PTYPE_DESCRIPTION},
    {"user12_name", PULSEQLIB_PARAM_USER13_NAME, PULSEQLIB_PTYPE_DESCRIPTION},
    {"user13_name", PULSEQLIB_PARAM_USER14_NAME, PULSEQLIB_PTYPE_DESCRIPTION},
    {"user14_name", PULSEQLIB_PARAM_USER15_NAME, PULSEQLIB_PTYPE_DESCRIPTION},
    {"user15_name", PULSEQLIB_PARAM_USER16_NAME, PULSEQLIB_PTYPE_DESCRIPTION},
    {"user16_name", PULSEQLIB_PARAM_USER17_NAME, PULSEQLIB_PTYPE_DESCRIPTION},
    {"user17_name", PULSEQLIB_PARAM_USER18_NAME, PULSEQLIB_PTYPE_DESCRIPTION},
    {"user18_name", PULSEQLIB_PARAM_USER19_NAME, PULSEQLIB_PTYPE_DESCRIPTION}};

#define PARAM_TABLE_SIZE (sizeof(g_param_table) / sizeof(g_param_table[0]))

/* ------------------------------------------------------------------ */
/*  Lookup functions                                                  */
/* ------------------------------------------------------------------ */

int pulseqlib_param_find(const char *wire_name)
{
    int i;
    if (!wire_name)
        return -1;
    for (i = 0; i < (int)PARAM_TABLE_SIZE; i++)
    {
        if (strcmp(g_param_table[i].wire_name, wire_name) == 0)
        {
            return (int)g_param_table[i].id;
        }
    }
    return -1;
}

const char *pulseqlib_param_wire_name(int param_id)
{
    int i;
    if (param_id < 0 || param_id >= PULSEQLIB_PARAM_COUNT)
        return NULL;
    for (i = 0; i < (int)PARAM_TABLE_SIZE; i++)
    {
        if ((int)g_param_table[i].id == param_id)
        {
            return g_param_table[i].wire_name;
        }
    }
    return NULL;
}

int pulseqlib_param_get_type(int param_id)
{
    int i;
    if (param_id < 0 || param_id >= PULSEQLIB_PARAM_COUNT)
        return -1;
    for (i = 0; i < (int)PARAM_TABLE_SIZE; i++)
    {
        if ((int)g_param_table[i].id == param_id)
        {
            return (int)g_param_table[i].type;
        }
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/*  Protocol helpers                                                  */
/* ------------------------------------------------------------------ */

/** Add or update a value in the protocol for the given param_id. */
static int protocol_set(pulseqlib_protocol *p, int param_id,
                        const pulseqlib_protocol_value *val)
{
    int i;
    /* Check if already present */
    for (i = 0; i < p->count; i++)
    {
        if ((int)p->keys[i] == param_id)
        {
            p->values[i] = *val;
            return 0;
        }
    }
    /* Append */
    if (p->count >= PULSEQLIB_PARAM_COUNT)
        return -1;
    p->keys[p->count] = (pulseqlib_param_id)param_id;
    p->values[p->count] = *val;
    p->count++;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Parse                                                             */
/* ------------------------------------------------------------------ */

/** Trim leading whitespace in place, return pointer to first non-space. */
static const char *skip_ws(const char *s)
{
    while (*s == ' ' || *s == '\t')
        s++;
    return s;
}

/** Trim trailing whitespace/newlines from a mutable string. */
static void trim_trailing(char *s)
{
    int len = (int)strlen(s);
    while (len > 0 && (s[len - 1] == ' ' || s[len - 1] == '\t' ||
                       s[len - 1] == '\n' || s[len - 1] == '\r'))
    {
        s[--len] = '\0';
    }
}

/** Parse a pipe-delimited field from *pp, write into dst (up to dstsz-1).
 *  Advances *pp past the consumed '|'.  Returns 0 on success, -1 if no
 *  more fields remain. */
static int next_pipe_field(const char **pp, char *dst, int dstsz)
{
    const char *start = *pp;
    const char *bar;
    int len;

    if (!start || !*start)
        return -1;

    bar = strchr(start, '|');
    if (bar)
    {
        len = (int)(bar - start);
        *pp = bar + 1;
    }
    else
    {
        len = (int)strlen(start);
        *pp = start + len;
    }
    if (len >= dstsz)
        len = dstsz - 1;
    memcpy(dst, start, len);
    dst[len] = '\0';
    return 0;
}

/** Try to parse "type|value|..." rich format.  Returns 1 if rich format
 *  was detected and parsed, 0 if this is a simple value line. */
static int try_parse_rich(const char *valstr,
                          pulseqlib_protocol_value *pv)
{
    /* Rich format always starts with a type tag followed by '|' */
    const char *bar = strchr(valstr, '|');
    const char *p;
    char field[256];

    if (!bar)
        return 0; /* no pipe → simple format */

    /* Check that the prefix before '|' is a known type tag */
    {
        int pfx_len = (int)(bar - valstr);
        if (pfx_len < 3 || pfx_len > 11)
            return 0; /* bounds check */
    }

    memset(pv, 0, sizeof(*pv));

    p = valstr;

    /* Field 0: type tag */
    if (next_pipe_field(&p, field, sizeof(field)) < 0)
        return 0;

    if (strcmp(field, "float") == 0)
    {
        pv->type = PULSEQLIB_PTYPE_FLOAT;
        pv->mode = PULSEQLIB_MODE_TYPEIN; /* default for backward compat */
        /* Field 1: mode or value (backward compat) */
        if (next_pipe_field(&p, field, sizeof(field)) < 0)
            return 0;
        if (strcmp(field, "typein") == 0)
        {
            pv->mode = PULSEQLIB_MODE_TYPEIN;
            if (next_pipe_field(&p, field, sizeof(field)) < 0)
                return 0;
        }
        else if (strcmp(field, "dropdown") == 0)
        {
            pv->mode = PULSEQLIB_MODE_DROPDOWN;
            if (next_pipe_field(&p, field, sizeof(field)) < 0)
                return 0;
        }
        else if (strcmp(field, "off") == 0)
        {
            pv->mode = PULSEQLIB_MODE_OFF;
            if (next_pipe_field(&p, field, sizeof(field)) < 0)
                return 0;
        }
        /* else: field already is the value (old format, mode stays TYPEIN) */
        pv->v.f = (float)atof(field);
        /* min */
        if (next_pipe_field(&p, field, sizeof(field)) == 0 && field[0])
        {
            pv->has_schema = 1;
            pv->range_min = (float)atof(field);
        }
        /* max */
        if (next_pipe_field(&p, field, sizeof(field)) == 0 && field[0])
            pv->range_max = (float)atof(field);
        /* incr */
        if (next_pipe_field(&p, field, sizeof(field)) == 0 && field[0])
            pv->range_incr = (float)atof(field);
        /* unit */
        if (next_pipe_field(&p, field, sizeof(field)) == 0)
        {
            size_t _n = strlen(field);
            if (_n >= sizeof(pv->unit))
                _n = sizeof(pv->unit) - 1;
            memcpy(pv->unit, field, _n);
            pv->unit[_n] = '\0';
        }
        /* Trailing dropdown options */
        pv->num_options = 0;
        while (pv->num_options < PULSEQLIB_MAX_DROPDOWN_OPTIONS &&
               next_pipe_field(&p, field, sizeof(field)) == 0 && field[0])
        {
            pv->options[pv->num_options++] = (float)atof(field);
        }
        return 1;
    }
    else if (strcmp(field, "int") == 0)
    {
        pv->type = PULSEQLIB_PTYPE_INT;
        pv->mode = PULSEQLIB_MODE_TYPEIN; /* default for backward compat */
        /* Field 1: mode or value (backward compat) */
        if (next_pipe_field(&p, field, sizeof(field)) < 0)
            return 0;
        if (strcmp(field, "typein") == 0)
        {
            pv->mode = PULSEQLIB_MODE_TYPEIN;
            if (next_pipe_field(&p, field, sizeof(field)) < 0)
                return 0;
        }
        else if (strcmp(field, "dropdown") == 0)
        {
            pv->mode = PULSEQLIB_MODE_DROPDOWN;
            if (next_pipe_field(&p, field, sizeof(field)) < 0)
                return 0;
        }
        else if (strcmp(field, "off") == 0)
        {
            pv->mode = PULSEQLIB_MODE_OFF;
            if (next_pipe_field(&p, field, sizeof(field)) < 0)
                return 0;
        }
        pv->v.i = atoi(field);
        /* min */
        if (next_pipe_field(&p, field, sizeof(field)) == 0 && field[0])
        {
            pv->has_schema = 1;
            pv->range_min = (float)atoi(field);
        }
        /* max */
        if (next_pipe_field(&p, field, sizeof(field)) == 0 && field[0])
            pv->range_max = (float)atoi(field);
        /* incr */
        if (next_pipe_field(&p, field, sizeof(field)) == 0 && field[0])
            pv->range_incr = (float)atoi(field);
        /* unit */
        if (next_pipe_field(&p, field, sizeof(field)) == 0)
        {
            size_t _n = strlen(field);
            if (_n >= sizeof(pv->unit))
                _n = sizeof(pv->unit) - 1;
            memcpy(pv->unit, field, _n);
            pv->unit[_n] = '\0';
        }
        /* Trailing dropdown options */
        pv->num_options = 0;
        while (pv->num_options < PULSEQLIB_MAX_DROPDOWN_OPTIONS &&
               next_pipe_field(&p, field, sizeof(field)) == 0 && field[0])
        {
            pv->options[pv->num_options++] = (float)atoi(field);
        }
        return 1;
    }
    else if (strcmp(field, "bool") == 0)
    {
        pv->type = PULSEQLIB_PTYPE_BOOL;
        pv->mode = PULSEQLIB_MODE_TYPEIN; /* default visible; bool wire format has no mode tag */
        /* Field 1: value */
        if (next_pipe_field(&p, field, sizeof(field)) < 0)
            return 0;
        pv->v.b = (strcmp(field, "true") == 0 || strcmp(field, "1") == 0) ? 1 : 0;
        return 1;
    }
    else if (strcmp(field, "stringlist") == 0)
    {
        pv->type = PULSEQLIB_PTYPE_STRINGLIST;
        pv->mode = PULSEQLIB_MODE_DROPDOWN; /* stringlists are dropdown selectors */
        /* Field 1: selected index */
        if (next_pipe_field(&p, field, sizeof(field)) < 0)
            return 0;
        pv->v.stringlist_idx = atoi(field);
        /* Remaining fields: option strings → join with '|' into stringlist_options */
        pv->stringlist_options[0] = '\0';
        {
            int off = 0;
            int first = 1;
            while (next_pipe_field(&p, field, sizeof(field)) == 0 && field[0])
            {
                int flen = (int)strlen(field);
                if (!first && off < (int)sizeof(pv->stringlist_options) - 1)
                    pv->stringlist_options[off++] = '|';
                if (off + flen < (int)sizeof(pv->stringlist_options))
                {
                    memcpy(pv->stringlist_options + off, field, flen);
                    off += flen;
                }
                pv->stringlist_options[off] = '\0';
                first = 0;
            }
        }
        return 1;
    }
    else if (strcmp(field, "description") == 0)
    {
        pv->type = PULSEQLIB_PTYPE_DESCRIPTION;
        pv->mode = PULSEQLIB_MODE_TYPEIN; /* description is informational; mark visible */
        /* Field 1: text (rest of line after first pipe) */
        strncpy(pv->v.desc, p, PULSEQLIB_PROTOCOL_DESC_MAX - 1);
        pv->v.desc[PULSEQLIB_PROTOCOL_DESC_MAX - 1] = '\0';
        return 1;
    }

    return 0; /* unknown type tag → fall through to simple parse */
}

int pulseqlib_protocol_parse(pulseqlib_protocol *out, const char *preamble)
{
    const char *p;
    char line[512];
    int in_block = 0;
    int parsed = 0;
    int line_len;
    const char *line_start;
    const char *line_end;

    if (!out || !preamble)
        return -1;

    memset(out, 0, sizeof(*out));

    p = preamble;
    while (*p)
    {
        /* Extract one line */
        line_start = p;
        line_end = strchr(p, '\n');
        if (line_end)
        {
            line_len = (int)(line_end - line_start);
            p = line_end + 1;
        }
        else
        {
            line_len = (int)strlen(line_start);
            p = line_start + line_len;
        }
        if (line_len >= (int)sizeof(line))
            line_len = (int)sizeof(line) - 1;
        memcpy(line, line_start, line_len);
        line[line_len] = '\0';
        trim_trailing(line);

        /* Strip leading # if present, then leading whitespace */
        {
            const char *lp = skip_ws(line);
            if (*lp == '#')
            {
                lp = skip_ws(lp + 1);
                memmove(line, lp, strlen(lp) + 1);
                trim_trailing(line);
            }
        }

        /* Check for delimiters */
        if (strstr(line, "[NimPulseqGUI Protocol End]"))
        {
            break;
        }
        if (strstr(line, "[NimPulseqGUI Protocol]"))
        {
            in_block = 1;
            continue;
        }
        if (strstr(line, "[VERSION]"))
        {
            break;
        }
        if (!in_block)
            continue;

        /* Parse "key: value" */
        {
            char key[64];
            char valstr[256];
            const char *colon = strchr(line, ':');
            int key_len;
            int param_id;
            int param_type;

            if (!colon)
                continue;
            key_len = (int)(colon - line);
            if (key_len <= 0 || key_len >= (int)sizeof(key))
                continue;
            memcpy(key, line, key_len);
            key[key_len] = '\0';
            trim_trailing(key);

            /* Value is everything after ": " */
            {
                const char *vp = skip_ws(colon + 1);
                strncpy(valstr, vp, sizeof(valstr) - 1);
                valstr[sizeof(valstr) - 1] = '\0';
                trim_trailing(valstr);
            }

            /* Look up key */
            param_id = pulseqlib_param_find(key);
            if (param_id < 0)
                continue; /* unknown key, skip */

            param_type = pulseqlib_param_get_type(param_id);
            if (param_type < 0)
                continue;

            {
                pulseqlib_protocol_value pv;
                memset(&pv, 0, sizeof(pv));

                /* Try rich "type|value|min|max|incr|unit" format first */
                if (try_parse_rich(valstr, &pv))
                {
                    /* Rich format parsed — type came from wire, not table */
                }
                else
                {
                    /* Simple "key: value" format */
                    pv.type = (pulseqlib_param_type)param_type;
                    pv.mode = PULSEQLIB_MODE_TYPEIN; /* visible by default */

                    switch (param_type)
                    {
                    case PULSEQLIB_PTYPE_FLOAT:
                        pv.v.f = (float)atof(valstr);
                        break;
                    case PULSEQLIB_PTYPE_INT:
                        pv.v.i = atoi(valstr);
                        break;
                    case PULSEQLIB_PTYPE_BOOL:
                        if (strcmp(valstr, "true") == 0 || strcmp(valstr, "1") == 0)
                        {
                            pv.v.b = 1;
                        }
                        else
                        {
                            pv.v.b = 0;
                        }
                        break;
                    case PULSEQLIB_PTYPE_STRINGLIST:
                        pv.v.stringlist_idx = atoi(valstr);
                        break;
                    case PULSEQLIB_PTYPE_DESCRIPTION:
                    {
                        size_t _n = strlen(valstr);
                        if (_n >= (size_t)PULSEQLIB_PROTOCOL_DESC_MAX)
                            _n = (size_t)PULSEQLIB_PROTOCOL_DESC_MAX - 1;
                        memcpy(pv.v.desc, valstr, _n);
                        pv.v.desc[_n] = '\0';
                        break;
                    }
                    default:
                        continue;
                    }
                }

                if (protocol_set(out, param_id, &pv) == 0)
                {
                    parsed++;
                }
            }
        }
    }

    return parsed;
}

/* ------------------------------------------------------------------ */
/*  Serialize                                                         */
/* ------------------------------------------------------------------ */

/** Append to buffer; return new offset or -1 on overflow. */
static int ser_append(char *buf, int bufsz, int n, const char *s)
{
    int len = (int)strlen(s);
    if (n + len >= bufsz)
        return -1;
    memcpy(buf + n, s, len);
    buf[n + len] = '\0';
    return n + len;
}

int pulseqlib_protocol_serialize(const pulseqlib_protocol *p,
                                 char *buf, int bufsz)
{
    int n = 0;
    int i;

    if (!p || !buf || bufsz <= 0)
        return -1;

    n = ser_append(buf, bufsz, n, "[NimPulseqGUI Protocol]\n");
    if (n < 0)
        return -1;

    for (i = 0; i < p->count; i++)
    {
        const char *wn = pulseqlib_param_wire_name((int)p->keys[i]);
        char tmp[384];
        if (!wn)
            continue;

        switch (p->values[i].type)
        {
        case PULSEQLIB_PTYPE_FLOAT:
            sprintf(tmp, "%s: %g\n", wn, (double)p->values[i].v.f);
            break;
        case PULSEQLIB_PTYPE_INT:
            sprintf(tmp, "%s: %d\n", wn, p->values[i].v.i);
            break;
        case PULSEQLIB_PTYPE_BOOL:
            sprintf(tmp, "%s: %s\n", wn,
                    p->values[i].v.b ? "true" : "false");
            break;
        case PULSEQLIB_PTYPE_STRINGLIST:
            sprintf(tmp, "%s: %d\n", wn, p->values[i].v.stringlist_idx);
            break;
        case PULSEQLIB_PTYPE_DESCRIPTION:
            sprintf(tmp, "%s: %s\n", wn, p->values[i].v.desc);
            break;
        default:
            tmp[0] = '\0';
            break;
        }

        if (tmp[0])
        {
            n = ser_append(buf, bufsz, n, tmp);
            if (n < 0)
                return -1;
        }
    }

    n = ser_append(buf, bufsz, n, "[NimPulseqGUI Protocol End]\n");
    if (n < 0)
        return -1;

    return n;
}

/* ------------------------------------------------------------------ */
/*  Typed getters / setters                                           */
/* ------------------------------------------------------------------ */

int pulseqlib_protocol_find(const pulseqlib_protocol *p, int param_id)
{
    int i;
    if (!p)
        return -1;
    for (i = 0; i < p->count; i++)
    {
        if ((int)p->keys[i] == param_id)
            return i;
    }
    return -1;
}

int pulseqlib_protocol_get_float(const pulseqlib_protocol *p,
                                 int param_id, float *out)
{
    int idx = pulseqlib_protocol_find(p, param_id);
    if (idx < 0 || p->values[idx].type != PULSEQLIB_PTYPE_FLOAT)
        return -1;
    if (out)
        *out = p->values[idx].v.f;
    return 0;
}

int pulseqlib_protocol_get_int(const pulseqlib_protocol *p,
                               int param_id, int *out)
{
    int idx = pulseqlib_protocol_find(p, param_id);
    if (idx < 0 || p->values[idx].type != PULSEQLIB_PTYPE_INT)
        return -1;
    if (out)
        *out = p->values[idx].v.i;
    return 0;
}

int pulseqlib_protocol_get_bool(const pulseqlib_protocol *p,
                                int param_id, int *out)
{
    int idx = pulseqlib_protocol_find(p, param_id);
    if (idx < 0 || p->values[idx].type != PULSEQLIB_PTYPE_BOOL)
        return -1;
    if (out)
        *out = p->values[idx].v.b;
    return 0;
}

int pulseqlib_protocol_set_float(pulseqlib_protocol *p,
                                 int param_id, float value)
{
    pulseqlib_protocol_value pv;
    memset(&pv, 0, sizeof(pv));
    pv.type = PULSEQLIB_PTYPE_FLOAT;
    pv.v.f = value;
    return protocol_set(p, param_id, &pv);
}

int pulseqlib_protocol_set_int(pulseqlib_protocol *p,
                               int param_id, int value)
{
    pulseqlib_protocol_value pv;
    memset(&pv, 0, sizeof(pv));
    pv.type = PULSEQLIB_PTYPE_INT;
    pv.v.i = value;
    return protocol_set(p, param_id, &pv);
}

int pulseqlib_protocol_set_bool(pulseqlib_protocol *p,
                                int param_id, int value)
{
    pulseqlib_protocol_value pv;
    memset(&pv, 0, sizeof(pv));
    pv.type = PULSEQLIB_PTYPE_BOOL;
    pv.v.b = value ? 1 : 0;
    return protocol_set(p, param_id, &pv);
}

int pulseqlib_protocol_get_stringlist(const pulseqlib_protocol *p,
                                      int param_id, int *idx_out)
{
    int idx = pulseqlib_protocol_find(p, param_id);
    if (idx < 0 || p->values[idx].type != PULSEQLIB_PTYPE_STRINGLIST)
        return -1;
    if (idx_out)
        *idx_out = p->values[idx].v.stringlist_idx;
    return 0;
}

int pulseqlib_protocol_set_stringlist(pulseqlib_protocol *p,
                                      int param_id, int sel_idx,
                                      const char *options)
{
    pulseqlib_protocol_value pv;
    memset(&pv, 0, sizeof(pv));
    pv.type = PULSEQLIB_PTYPE_STRINGLIST;
    pv.v.stringlist_idx = sel_idx;
    if (options)
    {
        strncpy(pv.stringlist_options, options,
                sizeof(pv.stringlist_options) - 1);
    }
    return protocol_set(p, param_id, &pv);
}
