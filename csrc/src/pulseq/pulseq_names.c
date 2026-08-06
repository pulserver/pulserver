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

/*
 * Names the file uses and Pulseq does not define.
 *
 * A research sequence may label its blocks with something outside the list
 * above, and the only thing that can say what such a label means is the file
 * itself.  So the reader keeps the name and hands out an id past the end of
 * the table: nothing switches on it -- an interpreter ignores a label it does
 * not recognize, which is all a custom label has ever meant -- but a writer
 * can ask for the name back and put it where it found it.
 *
 * The table is static and grows only by appending, so an id means the same
 * thing everywhere in the process for as long as the process runs.  It is
 * bounded, and a name arriving past the bound reads as -1, which is exactly
 * what every unknown name did before this existed.
 *
 * Not thread-safe to register from two threads at once.  The scanner never
 * registers at all (nothing in a PSD reads a label it cannot name), and the
 * host reads under the GIL.
 */
static char custom_names[PULSEQ_MAX_CUSTOM_LABELS][PULSEQ_LABEL_NAME_LENGTH];
static int num_custom_names = 0;

static int custom_label_id(const char *name)
{
    int i;

    for (i = 0; i < num_custom_names; i++)
    {
        if (strcmp(name, custom_names[i]) == 0)
            return PULSEQ_LABEL_BUILTIN_COUNT + i + 1;
    }
    return -1;
}

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
    return custom_label_id(name);
}

int pulseq_label_register_name(const char *name)
{
    int id;
    size_t len;

    if (!name)
        return -1;

    id = pulseq_label_id_for_name(name);
    if (id > 0)
        return id;

    len = strlen(name);
    if (num_custom_names >= PULSEQ_MAX_CUSTOM_LABELS || len == 0 ||
        len >= (size_t)PULSEQ_LABEL_NAME_LENGTH)
        return -1;

    memcpy(custom_names[num_custom_names], name, len + 1);
    num_custom_names++;
    return PULSEQ_LABEL_BUILTIN_COUNT + num_custom_names;
}

const char *pulseq_label_name_for_id(int id)
{
    int i;

    for (i = 0; label_table[i].name != NULL; i++)
    {
        if (label_table[i].value == id)
            return label_table[i].name;
    }
    if (id > PULSEQ_LABEL_BUILTIN_COUNT && id <= PULSEQ_LABEL_BUILTIN_COUNT + num_custom_names)
        return custom_names[id - PULSEQ_LABEL_BUILTIN_COUNT - 1];
    return NULL;
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

const char *pulseq_hint_name_for_id(int id)
{
    int i;

    for (i = 0; hint_table[i].name != NULL; i++)
    {
        if (hint_table[i].value == id)
            return hint_table[i].name;
    }
    return NULL;
}
