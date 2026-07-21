/* pulseq_names.c -- Pulseq label / soft-delay-hint name tables.
 *
 * Maps the names appearing in a .seq file's LABELSET / LABELINC and
 * soft-delay rows onto the numeric ids the raw model stores.
 */

#include <string.h>

#include "pulseq_internal.h"

/* ================================================================== */
/*  Label / flag names                                                */
/* ================================================================== */

static const pulseq_table_entry label_table[] = {
    {"SLC", PULSEQ_LABEL_SLC},
    {"SEG", PULSEQ_LABEL_SEG},
    {"REP", PULSEQ_LABEL_REP},
    {"AVG", PULSEQ_LABEL_AVG},
    {"SET", PULSEQ_LABEL_SET},
    {"ECO", PULSEQ_LABEL_ECO},
    {"PHS", PULSEQ_LABEL_PHS},
    {"LIN", PULSEQ_LABEL_LIN},
    {"PAR", PULSEQ_LABEL_PAR},
    {"ACQ", PULSEQ_LABEL_ACQ},
    {"NAV", PULSEQ_LABEL_NAV},
    {"REV", PULSEQ_LABEL_REV},
    {"SMS", PULSEQ_LABEL_SMS},
    {"REF", PULSEQ_LABEL_REF},
    {"IMA", PULSEQ_LABEL_IMA},
    {"NOISE", PULSEQ_LABEL_NOISE},
    {"PMC", PULSEQ_LABEL_PMC},
    {"NOROT", PULSEQ_LABEL_NOROT},
    {"NOPOS", PULSEQ_LABEL_NOPOS},
    {"NOSCL", PULSEQ_LABEL_NOSCL},
    {"ONCE", PULSEQ_LABEL_ONCE},
    {"TRID", PULSEQ_LABEL_TRID},
    {"OFF", PULSEQ_LABEL_OFF},
    {"MODULE", PULSEQ_LABEL_MODULE},
    {NULL, -1}};

int pulseq_label_id_for_name(const char *name)
{
    int i;

    if (!name)
        return -1;

    for (i = 0; label_table[i].name != NULL; i++)
    {
        if (strcmp(name, label_table[i].name) == 0)
            return label_table[i].value;
    }
    return -1;
}

/* ================================================================== */
/*  Soft-delay hint names                                             */
/* ================================================================== */

static const pulseq_table_entry hint_table[] = {
    {"TE", PULSEQ_HINT_TE},
    {"TR", PULSEQ_HINT_TR},
    {"TI", PULSEQ_HINT_TI},
    {"ESP", PULSEQ_HINT_ESP},
    {"RECTIME", PULSEQ_HINT_RECTIME},
    {"T2PREP", PULSEQ_HINT_T2PREP},
    {"TE2", PULSEQ_HINT_TE2},
    {NULL, -1}};

int pulseq_hint_id_for_name(const char *name)
{
    int i;

    if (!name)
        return -1;

    for (i = 0; hint_table[i].name != NULL; i++)
    {
        if (strcmp(name, hint_table[i].name) == 0)
            return hint_table[i].value;
    }
    return -1;
}
