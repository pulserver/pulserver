/**
 * @file pulseg_protocol.c
 * @brief Vendor-neutral MR protocol parse / serialize.
 *
 * Implements the parameter lookup table, preamble parser, and
 * serializer for the NimPulseqGUI wire format.  Pure C89, no
 * vendor dependencies.
 */

#include "pulseg_protocol.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================== */
/*  Static parameter table                                            */
/* ================================================================== */

static const pulseg_param_entry g_param_table[] = {
    /* Timing */
    {"TE", PULSEG_PARAM_TE, PULSEG_PTYPE_FLOAT},
    {"TR", PULSEG_PARAM_TR, PULSEG_PTYPE_FLOAT},
    {"prep_time", PULSEG_PARAM_TI, PULSEG_PTYPE_FLOAT},
    {"TE2", PULSEG_PARAM_TE2, PULSEG_PTYPE_FLOAT},
    {"Trecovery", PULSEG_PARAM_TRECOVERY, PULSEG_PTYPE_FLOAT},
    /* Spatial */
    {"fov", PULSEG_PARAM_FOV, PULSEG_PTYPE_FLOAT},
    {"phase_fov", PULSEG_PARAM_PHASE_FOV, PULSEG_PTYPE_FLOAT},
    {"slice_thickness", PULSEG_PARAM_SLICE_THICKNESS, PULSEG_PTYPE_FLOAT},
    {"slice_spacing", PULSEG_PARAM_SLICE_SPACING, PULSEG_PTYPE_FLOAT},
    /* Off-isocentre translation, mm, in the LOGICAL frame (see the header). */
    {"fov_offset_x", PULSEG_PARAM_FOV_OFFSET_X, PULSEG_PTYPE_FLOAT},
    {"fov_offset_y", PULSEG_PARAM_FOV_OFFSET_Y, PULSEG_PTYPE_FLOAT},
    {"fov_offset_z", PULSEG_PARAM_FOV_OFFSET_Z, PULSEG_PTYPE_FLOAT},
    {"nslices", PULSEG_PARAM_NSLICES, PULSEG_PTYPE_INT},
    {"nx", PULSEG_PARAM_MATRIX, PULSEG_PTYPE_INT},
    {"ny", PULSEG_PARAM_NY, PULSEG_PTYPE_INT},
    {"num_slabs", PULSEG_PARAM_NUM_SLABS, PULSEG_PTYPE_INT},
    {"overlap_locations", PULSEG_PARAM_OVERLAP_LOCS, PULSEG_PTYPE_INT},
    /* Acquisition */
    {"num_echoes", PULSEG_PARAM_NECHOES, PULSEG_PTYPE_INT},
    {"nex", PULSEG_PARAM_NEX, PULSEG_PTYPE_FLOAT},
    {"num_shots", PULSEG_PARAM_NUM_SHOTS, PULSEG_PTYPE_INT},
    {"etl", PULSEG_PARAM_ETL, PULSEG_PTYPE_INT},
    /* Contrast */
    {"flip", PULSEG_PARAM_FLIP_ANGLE, PULSEG_PTYPE_FLOAT},
    {"bandwidth", PULSEG_PARAM_BANDWIDTH, PULSEG_PTYPE_FLOAT},
    /* Enum / stringlist */
    {"sequence_type", PULSEG_PARAM_SEQUENCE_TYPE, PULSEG_PTYPE_STRINGLIST},
    {"imaging_mode", PULSEG_PARAM_IMAGING_MODE, PULSEG_PTYPE_STRINGLIST},
    {"preparation_type", PULSEG_PARAM_PREP_TYPE, PULSEG_PTYPE_STRINGLIST},
    {"trigger_type", PULSEG_PARAM_TRIGGER_TYPE, PULSEG_PTYPE_STRINGLIST},
    /* Flags */
    {"FatSat", PULSEG_PARAM_FAT_SAT, PULSEG_PTYPE_BOOL},
    {"Spoiler", PULSEG_PARAM_SPOILER, PULSEG_PTYPE_BOOL},
    {"RFSpoiling", PULSEG_PARAM_RF_SPOILING, PULSEG_PTYPE_BOOL},
    {"swap_phase_freq", PULSEG_PARAM_SWAP_PF, PULSEG_PTYPE_BOOL},
    {"enable_saturation_ui", PULSEG_PARAM_ENABLE_SAT_UI, PULSEG_PTYPE_BOOL},
    {"record_physio", PULSEG_PARAM_RECORD_PHYSIO, PULSEG_PTYPE_BOOL},
    /* Acceleration */
    {"Ry", PULSEG_PARAM_RY, PULSEG_PTYPE_FLOAT},
    {"Rz", PULSEG_PARAM_RZ, PULSEG_PTYPE_FLOAT},
    {"compressed_sensing", PULSEG_PARAM_COMPRESSED_SENS, PULSEG_PTYPE_FLOAT},
    {"multiband", PULSEG_PARAM_MULTIBAND, PULSEG_PTYPE_FLOAT},
    /* Cine / trigger */
    {"num_frames", PULSEG_PARAM_NUM_FRAMES, PULSEG_PTYPE_INT},
    {"delay_time", PULSEG_PARAM_DELAY_TIME, PULSEG_PTYPE_FLOAT},
    {"trigger_delay", PULSEG_PARAM_TRIGGER_DELAY, PULSEG_PTYPE_FLOAT},
    {"trigger_window", PULSEG_PARAM_TRIGGER_WINDOW, PULSEG_PTYPE_FLOAT},
    /* Diffusion */
    {"diffusion_bvalues", PULSEG_PARAM_DIFF_BVALUES, PULSEG_PTYPE_FLOAT},
    {"diffusion_directions", PULSEG_PARAM_DIFF_DIRECTIONS, PULSEG_PTYPE_INT},
    /* Saturation bands */
    {"sat_x", PULSEG_PARAM_SAT_X, PULSEG_PTYPE_INT},
    {"sat_y", PULSEG_PARAM_SAT_Y, PULSEG_PTYPE_INT},
    {"sat_z", PULSEG_PARAM_SAT_Z, PULSEG_PTYPE_INT},
    {"sat_x_loc1", PULSEG_PARAM_SAT_X_LOC1, PULSEG_PTYPE_FLOAT},
    {"sat_x_loc2", PULSEG_PARAM_SAT_X_LOC2, PULSEG_PTYPE_FLOAT},
    {"sat_y_loc1", PULSEG_PARAM_SAT_Y_LOC1, PULSEG_PTYPE_FLOAT},
    {"sat_y_loc2", PULSEG_PARAM_SAT_Y_LOC2, PULSEG_PTYPE_FLOAT},
    {"sat_z_loc1", PULSEG_PARAM_SAT_Z_LOC1, PULSEG_PTYPE_FLOAT},
    {"sat_z_loc2", PULSEG_PARAM_SAT_Z_LOC2, PULSEG_PTYPE_FLOAT},
    {"sat_x_thickness", PULSEG_PARAM_SAT_X_THICK, PULSEG_PTYPE_FLOAT},
    {"sat_y_thickness", PULSEG_PARAM_SAT_Y_THICK, PULSEG_PTYPE_FLOAT},
    {"sat_z_thickness", PULSEG_PARAM_SAT_Z_THICK, PULSEG_PTYPE_FLOAT},
    /* Info */
    {"TA", PULSEG_PARAM_TA, PULSEG_PTYPE_FLOAT},
    /* User CVs: user0..user16 map to GE opuser3..opuser19. */
    {"user0_value", PULSEG_PARAM_USER1, PULSEG_PTYPE_FLOAT},
    {"user1_value", PULSEG_PARAM_USER2, PULSEG_PTYPE_FLOAT},
    {"user2_value", PULSEG_PARAM_USER3, PULSEG_PTYPE_FLOAT},
    {"user3_value", PULSEG_PARAM_USER4, PULSEG_PTYPE_FLOAT},
    {"user4_value", PULSEG_PARAM_USER5, PULSEG_PTYPE_FLOAT},
    {"user5_value", PULSEG_PARAM_USER6, PULSEG_PTYPE_FLOAT},
    {"user6_value", PULSEG_PARAM_USER7, PULSEG_PTYPE_FLOAT},
    {"user7_value", PULSEG_PARAM_USER8, PULSEG_PTYPE_FLOAT},
    {"user8_value", PULSEG_PARAM_USER9, PULSEG_PTYPE_FLOAT},
    {"user9_value", PULSEG_PARAM_USER10, PULSEG_PTYPE_FLOAT},
    {"user10_value", PULSEG_PARAM_USER11, PULSEG_PTYPE_FLOAT},
    {"user11_value", PULSEG_PARAM_USER12, PULSEG_PTYPE_FLOAT},
    {"user12_value", PULSEG_PARAM_USER13, PULSEG_PTYPE_FLOAT},
    {"user13_value", PULSEG_PARAM_USER14, PULSEG_PTYPE_FLOAT},
    {"user14_value", PULSEG_PARAM_USER15, PULSEG_PTYPE_FLOAT},
    {"user15_value", PULSEG_PARAM_USER16, PULSEG_PTYPE_FLOAT},
    {"user16_value", PULSEG_PARAM_USER17, PULSEG_PTYPE_FLOAT},
    {"user17_value", PULSEG_PARAM_USER18, PULSEG_PTYPE_FLOAT},
    {"user18_value", PULSEG_PARAM_USER19, PULSEG_PTYPE_FLOAT},
    /* User CV name labels */
    {"user0_name", PULSEG_PARAM_USER1_NAME, PULSEG_PTYPE_DESCRIPTION},
    {"user1_name", PULSEG_PARAM_USER2_NAME, PULSEG_PTYPE_DESCRIPTION},
    {"user2_name", PULSEG_PARAM_USER3_NAME, PULSEG_PTYPE_DESCRIPTION},
    {"user3_name", PULSEG_PARAM_USER4_NAME, PULSEG_PTYPE_DESCRIPTION},
    {"user4_name", PULSEG_PARAM_USER5_NAME, PULSEG_PTYPE_DESCRIPTION},
    {"user5_name", PULSEG_PARAM_USER6_NAME, PULSEG_PTYPE_DESCRIPTION},
    {"user6_name", PULSEG_PARAM_USER7_NAME, PULSEG_PTYPE_DESCRIPTION},
    {"user7_name", PULSEG_PARAM_USER8_NAME, PULSEG_PTYPE_DESCRIPTION},
    {"user8_name", PULSEG_PARAM_USER9_NAME, PULSEG_PTYPE_DESCRIPTION},
    {"user9_name", PULSEG_PARAM_USER10_NAME, PULSEG_PTYPE_DESCRIPTION},
    {"user10_name", PULSEG_PARAM_USER11_NAME, PULSEG_PTYPE_DESCRIPTION},
    {"user11_name", PULSEG_PARAM_USER12_NAME, PULSEG_PTYPE_DESCRIPTION},
    {"user12_name", PULSEG_PARAM_USER13_NAME, PULSEG_PTYPE_DESCRIPTION},
    {"user13_name", PULSEG_PARAM_USER14_NAME, PULSEG_PTYPE_DESCRIPTION},
    {"user14_name", PULSEG_PARAM_USER15_NAME, PULSEG_PTYPE_DESCRIPTION},
    {"user15_name", PULSEG_PARAM_USER16_NAME, PULSEG_PTYPE_DESCRIPTION},
    {"user16_name", PULSEG_PARAM_USER17_NAME, PULSEG_PTYPE_DESCRIPTION},
    {"user17_name", PULSEG_PARAM_USER18_NAME, PULSEG_PTYPE_DESCRIPTION},
    {"user18_name", PULSEG_PARAM_USER19_NAME, PULSEG_PTYPE_DESCRIPTION}};

#define PARAM_TABLE_SIZE (sizeof(g_param_table) / sizeof(g_param_table[0]))

/* ================================================================== */
/*  Lookup functions                                                  */
/* ================================================================== */

int pulseg_param_find(const char *wire_name)
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

const char *pulseg_param_wire_name(int param_id)
{
    int i;
    if (param_id < 0 || param_id >= PULSEG_PARAM_COUNT)
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

int pulseg_param_get_type(int param_id)
{
    int i;
    if (param_id < 0 || param_id >= PULSEG_PARAM_COUNT)
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

/* ================================================================== */
/*  Protocol helpers                                                  */
/* ================================================================== */

/** Add or update a value in the protocol for the given param_id. */
static int protocol_set(pulseg_protocol *p, int param_id, const pulseg_protocol_value *val)
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
    if (p->count >= PULSEG_PARAM_COUNT)
        return -1;
    p->keys[p->count] = (pulseg_param_id)param_id;
    p->values[p->count] = *val;
    p->count++;
    return 0;
}

/* ================================================================== */
/*  Parse                                                             */
/* ================================================================== */

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
    while (len > 0 &&
           (s[len - 1] == ' ' || s[len - 1] == '\t' || s[len - 1] == '\n' || s[len - 1] == '\r'))
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
static int try_parse_rich(pulseg_protocol_value *pv, const char *valstr)
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
        pv->type = PULSEG_PTYPE_FLOAT;
        pv->mode = PULSEG_MODE_TYPEIN; /* default for backward compat */
        /* Field 1: mode or value (backward compat) */
        if (next_pipe_field(&p, field, sizeof(field)) < 0)
            return 0;
        if (strcmp(field, "typein") == 0)
        {
            pv->mode = PULSEG_MODE_TYPEIN;
            if (next_pipe_field(&p, field, sizeof(field)) < 0)
                return 0;
        }
        else if (strcmp(field, "dropdown") == 0)
        {
            pv->mode = PULSEG_MODE_DROPDOWN;
            if (next_pipe_field(&p, field, sizeof(field)) < 0)
                return 0;
        }
        else if (strcmp(field, "off") == 0)
        {
            pv->mode = PULSEG_MODE_OFF;
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
        while (pv->num_options < PULSEG_MAX_DROPDOWN_OPTIONS &&
               next_pipe_field(&p, field, sizeof(field)) == 0 && field[0])
        {
            pv->options[pv->num_options++] = (float)atof(field);
        }
        return 1;
    }
    else if (strcmp(field, "int") == 0)
    {
        pv->type = PULSEG_PTYPE_INT;
        pv->mode = PULSEG_MODE_TYPEIN; /* default for backward compat */
        /* Field 1: mode or value (backward compat) */
        if (next_pipe_field(&p, field, sizeof(field)) < 0)
            return 0;
        if (strcmp(field, "typein") == 0)
        {
            pv->mode = PULSEG_MODE_TYPEIN;
            if (next_pipe_field(&p, field, sizeof(field)) < 0)
                return 0;
        }
        else if (strcmp(field, "dropdown") == 0)
        {
            pv->mode = PULSEG_MODE_DROPDOWN;
            if (next_pipe_field(&p, field, sizeof(field)) < 0)
                return 0;
        }
        else if (strcmp(field, "off") == 0)
        {
            pv->mode = PULSEG_MODE_OFF;
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
        while (pv->num_options < PULSEG_MAX_DROPDOWN_OPTIONS &&
               next_pipe_field(&p, field, sizeof(field)) == 0 && field[0])
        {
            pv->options[pv->num_options++] = (float)atoi(field);
        }
        return 1;
    }
    else if (strcmp(field, "bool") == 0)
    {
        pv->type = PULSEG_PTYPE_BOOL;
        pv->mode = PULSEG_MODE_TYPEIN; /* default visible; bool wire format has no mode tag */
        /* Field 1: value */
        if (next_pipe_field(&p, field, sizeof(field)) < 0)
            return 0;
        pv->v.b = (strcmp(field, "true") == 0 || strcmp(field, "1") == 0) ? 1 : 0;
        return 1;
    }
    else if (strcmp(field, "stringlist") == 0)
    {
        pv->type = PULSEG_PTYPE_STRINGLIST;
        pv->mode = PULSEG_MODE_DROPDOWN; /* stringlists are dropdown selectors */
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
        pv->type = PULSEG_PTYPE_DESCRIPTION;
        pv->mode = PULSEG_MODE_TYPEIN; /* description is informational; mark visible */
        /* Field 1: text (rest of line after first pipe) */
        strncpy(pv->v.desc, p, PULSEG_PROTOCOL_DESC_MAX - 1);
        pv->v.desc[PULSEG_PROTOCOL_DESC_MAX - 1] = '\0';
        return 1;
    }

    return 0; /* unknown type tag → fall through to simple parse */
}

int pulseg_protocol_parse(pulseg_protocol *out, const char *preamble)
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
            param_id = pulseg_param_find(key);
            if (param_id < 0)
                continue; /* unknown key, skip */

            param_type = pulseg_param_get_type(param_id);
            if (param_type < 0)
                continue;

            {
                pulseg_protocol_value pv;
                memset(&pv, 0, sizeof(pv));

                /* Try rich "type|value|min|max|incr|unit" format first */
                if (try_parse_rich(&pv, valstr))
                {
                    /* Rich format parsed — type came from wire, not table */
                }
                else
                {
                    /* Simple "key: value" format */
                    pv.type = (pulseg_param_type)param_type;
                    pv.mode = PULSEG_MODE_TYPEIN; /* visible by default */

                    switch (param_type)
                    {
                    case PULSEG_PTYPE_FLOAT:
                        pv.v.f = (float)atof(valstr);
                        break;
                    case PULSEG_PTYPE_INT:
                        pv.v.i = atoi(valstr);
                        break;
                    case PULSEG_PTYPE_BOOL:
                        if (strcmp(valstr, "true") == 0 || strcmp(valstr, "1") == 0)
                        {
                            pv.v.b = 1;
                        }
                        else
                        {
                            pv.v.b = 0;
                        }
                        break;
                    case PULSEG_PTYPE_STRINGLIST:
                        pv.v.stringlist_idx = atoi(valstr);
                        break;
                    case PULSEG_PTYPE_DESCRIPTION:
                    {
                        size_t _n = strlen(valstr);
                        if (_n >= (size_t)PULSEG_PROTOCOL_DESC_MAX)
                            _n = (size_t)PULSEG_PROTOCOL_DESC_MAX - 1;
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

/* ================================================================== */
/*  Serialize                                                         */
/* ================================================================== */

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

int pulseg_protocol_serialize(const pulseg_protocol *p, char *buf, int bufsz)
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
        const char *wn = pulseg_param_wire_name((int)p->keys[i]);
        char tmp[384];
        if (!wn)
            continue;

        switch (p->values[i].type)
        {
        case PULSEG_PTYPE_FLOAT:
            sprintf(tmp, "%s: %g\n", wn, (double)p->values[i].v.f);
            break;
        case PULSEG_PTYPE_INT:
            sprintf(tmp, "%s: %d\n", wn, p->values[i].v.i);
            break;
        case PULSEG_PTYPE_BOOL:
            sprintf(tmp, "%s: %s\n", wn, p->values[i].v.b ? "true" : "false");
            break;
        case PULSEG_PTYPE_STRINGLIST:
            sprintf(tmp, "%s: %d\n", wn, p->values[i].v.stringlist_idx);
            break;
        case PULSEG_PTYPE_DESCRIPTION:
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

/* ================================================================== */
/*  Typed getters / setters                                           */
/* ================================================================== */

int pulseg_protocol_find(const pulseg_protocol *p, int param_id)
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

int pulseg_protocol_get_float(const pulseg_protocol *p, float *out, int param_id)
{
    int idx = pulseg_protocol_find(p, param_id);
    if (idx < 0 || p->values[idx].type != PULSEG_PTYPE_FLOAT)
        return -1;
    if (out)
        *out = p->values[idx].v.f;
    return 0;
}

int pulseg_protocol_get_int(const pulseg_protocol *p, int *out, int param_id)
{
    int idx = pulseg_protocol_find(p, param_id);
    if (idx < 0 || p->values[idx].type != PULSEG_PTYPE_INT)
        return -1;
    if (out)
        *out = p->values[idx].v.i;
    return 0;
}

int pulseg_protocol_get_bool(const pulseg_protocol *p, int *out, int param_id)
{
    int idx = pulseg_protocol_find(p, param_id);
    if (idx < 0 || p->values[idx].type != PULSEG_PTYPE_BOOL)
        return -1;
    if (out)
        *out = p->values[idx].v.b;
    return 0;
}

int pulseg_protocol_set_float(pulseg_protocol *p, int param_id, float value)
{
    pulseg_protocol_value pv;
    memset(&pv, 0, sizeof(pv));
    pv.type = PULSEG_PTYPE_FLOAT;
    pv.v.f = value;
    return protocol_set(p, param_id, &pv);
}

int pulseg_protocol_set_int(pulseg_protocol *p, int param_id, int value)
{
    pulseg_protocol_value pv;
    memset(&pv, 0, sizeof(pv));
    pv.type = PULSEG_PTYPE_INT;
    pv.v.i = value;
    return protocol_set(p, param_id, &pv);
}

int pulseg_protocol_set_bool(pulseg_protocol *p, int param_id, int value)
{
    pulseg_protocol_value pv;
    memset(&pv, 0, sizeof(pv));
    pv.type = PULSEG_PTYPE_BOOL;
    pv.v.b = value ? 1 : 0;
    return protocol_set(p, param_id, &pv);
}

int pulseg_protocol_get_stringlist(const pulseg_protocol *p, int *idx_out, int param_id)
{
    int idx = pulseg_protocol_find(p, param_id);
    if (idx < 0 || p->values[idx].type != PULSEG_PTYPE_STRINGLIST)
        return -1;
    if (idx_out)
        *idx_out = p->values[idx].v.stringlist_idx;
    return 0;
}

int pulseg_protocol_set_stringlist(
    pulseg_protocol *p,
    int param_id,
    int sel_idx,
    const char *options)
{
    pulseg_protocol_value pv;
    memset(&pv, 0, sizeof(pv));
    pv.type = PULSEG_PTYPE_STRINGLIST;
    pv.v.stringlist_idx = sel_idx;
    if (options)
    {
        strncpy(pv.stringlist_options, options, sizeof(pv.stringlist_options) - 1);
    }
    return protocol_set(p, param_id, &pv);
}
