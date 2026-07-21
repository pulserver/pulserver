/**
 * @file pulseg_protocol.h
 * @brief Vendor-neutral MR protocol parameter table, parse, and serialize.
 *
 * Maps the standard NimPulseqGUI preamble wire format to a fixed set
 * of parameter IDs (mirroring the Python UIParam enum).  No vendor-
 * specific units, CV names, or UI concepts appear here.
 *
 * Wire format (standard NimPulseqGUI preamble):
 *   [NimPulseqGUI Protocol]
 *   TE: 5.0
 *   TR: 500.0
 *   NSlices: 10
 *   FatSat: 1
 *   [NimPulseqGUI Protocol End]
 */

#ifndef PULSEG_PROTOCOL_H
#define PULSEG_PROTOCOL_H

#ifdef __cplusplus
extern "C"
{
#endif

    /* ================================================================== */
    /*  Parameter IDs  (mirror Python UIParam enum + User1..19)           */
    /*                                                                    */
    /*  Expressed as #define + typedef rather than C enum for              */
    /*  compatibility with older EPIC compilers that reject C enums.      */
    /* ================================================================== */

    typedef int pulseg_param_id;

/* Timing */
#define PULSEG_PARAM_TE 0
#define PULSEG_PARAM_TR 1
#define PULSEG_PARAM_TI 2 /* wire: "prep_time" */
/* Spatial */
#define PULSEG_PARAM_FOV 3
#define PULSEG_PARAM_SLICE_THICKNESS 4
#define PULSEG_PARAM_NSLICES 5
#define PULSEG_PARAM_MATRIX 6  /* wire: "nx" */
#define PULSEG_PARAM_NECHOES 7 /* wire: "num_echoes" */
/* Contrast */
#define PULSEG_PARAM_FLIP_ANGLE 8 /* wire: "flip" */
#define PULSEG_PARAM_BANDWIDTH 9
/* Flags */
#define PULSEG_PARAM_FAT_SAT 10
#define PULSEG_PARAM_SPOILER 11
#define PULSEG_PARAM_RF_SPOILING 12
/* Info (read-only) */
#define PULSEG_PARAM_TA 13
/* User CV slots. Meaning of the wire-protocol slot -> vendor CV mapping
 * is vendor-defined; see the private vendor layer for specifics (e.g.
 * pulserver_ge_protocol.h in pulserver-interpreter for GE's opuser
 * mapping). */
#define PULSEG_PARAM_USER1 14
#define PULSEG_PARAM_USER2 15
#define PULSEG_PARAM_USER3 16
#define PULSEG_PARAM_USER4 17
#define PULSEG_PARAM_USER5 18
#define PULSEG_PARAM_USER6 19
#define PULSEG_PARAM_USER7 20
#define PULSEG_PARAM_USER8 21
#define PULSEG_PARAM_USER9 22
#define PULSEG_PARAM_USER10 23
#define PULSEG_PARAM_USER11 24
#define PULSEG_PARAM_USER12 25
#define PULSEG_PARAM_USER13 26
#define PULSEG_PARAM_USER14 27
#define PULSEG_PARAM_USER15 28
#define PULSEG_PARAM_USER16 29
#define PULSEG_PARAM_USER17 30
#define PULSEG_PARAM_USER18 31
#define PULSEG_PARAM_USER19 32
/* --- Extended timing --- */
#define PULSEG_PARAM_TE2 33
#define PULSEG_PARAM_TRECOVERY 34
/* --- Extended spatial --- */
#define PULSEG_PARAM_PHASE_FOV 35
#define PULSEG_PARAM_SLICE_SPACING 36
#define PULSEG_PARAM_NY 37
#define PULSEG_PARAM_NUM_SLABS 38
#define PULSEG_PARAM_OVERLAP_LOCS 39
/* --- Acquisition --- */
#define PULSEG_PARAM_NEX 40
#define PULSEG_PARAM_NUM_SHOTS 41
#define PULSEG_PARAM_ETL 42
/* --- Enum / stringlist --- */
#define PULSEG_PARAM_SEQUENCE_TYPE 43
#define PULSEG_PARAM_IMAGING_MODE 44
#define PULSEG_PARAM_PREP_TYPE 45
#define PULSEG_PARAM_TRIGGER_TYPE 46
/* --- Extended flags --- */
#define PULSEG_PARAM_SWAP_PF 47
#define PULSEG_PARAM_ENABLE_SAT_UI 48
#define PULSEG_PARAM_RECORD_PHYSIO 49
/* --- Acceleration --- */
#define PULSEG_PARAM_RY 50
#define PULSEG_PARAM_RZ 51
#define PULSEG_PARAM_COMPRESSED_SENS 52
#define PULSEG_PARAM_MULTIBAND 53
/* --- Cine / trigger --- */
#define PULSEG_PARAM_NUM_FRAMES 54
#define PULSEG_PARAM_DELAY_TIME 55
#define PULSEG_PARAM_TRIGGER_DELAY 56
#define PULSEG_PARAM_TRIGGER_WINDOW 57
/* --- Diffusion --- */
#define PULSEG_PARAM_DIFF_BVALUES 58
#define PULSEG_PARAM_DIFF_DIRECTIONS 59
/* --- Saturation bands --- */
#define PULSEG_PARAM_SAT_X 60
#define PULSEG_PARAM_SAT_Y 61
#define PULSEG_PARAM_SAT_Z 62
#define PULSEG_PARAM_SAT_X_LOC1 63
#define PULSEG_PARAM_SAT_X_LOC2 64
#define PULSEG_PARAM_SAT_Y_LOC1 65
#define PULSEG_PARAM_SAT_Y_LOC2 66
#define PULSEG_PARAM_SAT_Z_LOC1 67
#define PULSEG_PARAM_SAT_Z_LOC2 68
#define PULSEG_PARAM_SAT_X_THICK 69
#define PULSEG_PARAM_SAT_Y_THICK 70
#define PULSEG_PARAM_SAT_Z_THICK 71
/* User CV name labels (description type; mirror USER1..USER19 value slots) */
#define PULSEG_PARAM_USER1_NAME 72
#define PULSEG_PARAM_USER2_NAME 73
#define PULSEG_PARAM_USER3_NAME 74
#define PULSEG_PARAM_USER4_NAME 75
#define PULSEG_PARAM_USER5_NAME 76
#define PULSEG_PARAM_USER6_NAME 77
#define PULSEG_PARAM_USER7_NAME 78
#define PULSEG_PARAM_USER8_NAME 79
#define PULSEG_PARAM_USER9_NAME 80
#define PULSEG_PARAM_USER10_NAME 81
#define PULSEG_PARAM_USER11_NAME 82
#define PULSEG_PARAM_USER12_NAME 83
#define PULSEG_PARAM_USER13_NAME 84
#define PULSEG_PARAM_USER14_NAME 85
#define PULSEG_PARAM_USER15_NAME 86
#define PULSEG_PARAM_USER16_NAME 87
#define PULSEG_PARAM_USER17_NAME 88
#define PULSEG_PARAM_USER18_NAME 89
#define PULSEG_PARAM_USER19_NAME 90
#define PULSEG_PARAM_COUNT 91 /* sentinel */

    /* ================================================================== */
    /*  Parameter types                                                   */
    /* ================================================================== */

    typedef int pulseg_param_type;

#define PULSEG_PTYPE_FLOAT 0
#define PULSEG_PTYPE_INT 1
#define PULSEG_PTYPE_BOOL 2
#define PULSEG_PTYPE_STRINGLIST 3
#define PULSEG_PTYPE_DESCRIPTION 4

    /* ================================================================== */
    /*  Input mode (mirrors Python InputMode enum)                        */
    /* ================================================================== */

    typedef int pulseg_input_mode;

#define PULSEG_MODE_OFF 0      /* hidden from UI */
#define PULSEG_MODE_TYPEIN 1   /* type-in field */
#define PULSEG_MODE_DROPDOWN 2 /* type-in + dropdown options */

    /* ================================================================== */
    /*  Parameter table entry (wire name -> id + type)                    */
    /* ================================================================== */

    typedef struct pulseg_param_entry
    {
        const char *wire_name; /* e.g. "TE", "FlipAngle" */
        pulseg_param_id id;
        pulseg_param_type type;
    } pulseg_param_entry;

    /* ================================================================== */
    /*  Protocol value (tagged union)                                     */
    /* ================================================================== */

#define PULSEG_PROTOCOL_DESC_MAX 128
#define PULSEG_PROTOCOL_SLIST_MAX 256
#define PULSEG_MAX_DROPDOWN_OPTIONS 5

    /** @brief One protocol parameter: tagged value plus its UI schema. */
    typedef struct pulseg_protocol_value
    {
        pulseg_param_type type;
        union
        {
            float f;
            int i;
            int b; /* 0 or 1 */
            int stringlist_idx;
            char desc[PULSEG_PROTOCOL_DESC_MAX];
        } v;
        /* For stringlist: pipe-delimited options stored as raw string */
        char stringlist_options[PULSEG_PROTOCOL_SLIST_MAX];
        /* Schema metadata (populated from rich wire format) */
        int has_schema;   /* 1 if range_min/max/incr populated */
        float range_min;  /* float/int minimum */
        float range_max;  /* float/int maximum */
        float range_incr; /* step increment */
        char unit[32];    /* unit string (e.g. "ms", "mm", "deg") */
        /* Input mode + dropdown options */
        pulseg_input_mode mode;                     /* off / typein / dropdown */
        int num_options;                            /* 0..5 dropdown option count */
        float options[PULSEG_MAX_DROPDOWN_OPTIONS]; /* dropdown values */
    } pulseg_protocol_value;

    /* ================================================================== */
    /*  Protocol container (fixed-size, stack-allocatable)                */
    /* ================================================================== */

    /** @brief A whole protocol: parallel key/value arrays, stack-allocatable. */
    typedef struct pulseg_protocol
    {
        int count; /* number of populated entries */
        pulseg_param_id keys[PULSEG_PARAM_COUNT];
        pulseg_protocol_value values[PULSEG_PARAM_COUNT];
    } pulseg_protocol;

    /* Zero-initializer */
    /* clang-format off */
    /* clang-format on */

    /* ================================================================== */
    /*  Lookup functions                                                  */
    /* ================================================================== */

    /**
     * @brief Find a parameter id by its wire name (case-sensitive).
     * @return Parameter id (>= 0), or -1 if the name is unknown.
     */
    int pulseg_param_find(const char *wire_name);

    /**
     * @brief Wire name for a parameter id.
     * @return Static string, or NULL if @p param_id is out of range.
     */
    const char *pulseg_param_wire_name(int param_id);

    /**
     * @brief Declared type of a parameter id.
     * @return A PULSEG_PTYPE_* value, or -1 if @p param_id is out of range.
     */
    int pulseg_param_get_type(int param_id);

    /* ================================================================== */
    /*  Parse / serialize                                                 */
    /* ================================================================== */

    /**
     * @brief Parse a NimPulseqGUI preamble into a protocol.
     *
     * Accepts both the simple form ("key: value") and the rich schema form
     * ("key: type|value|min|max|incr|unit"); rich lines additionally populate
     * has_schema, range_* and unit. Unknown keys are skipped silently.
     *
     * @param[out] out       Caller-allocated protocol; zeroed on entry.
     * @param[in]  preamble  NUL-terminated preamble text, delimiters included.
     * @return Number of parameters parsed, or -1 on error.
     */
    int pulseg_protocol_parse(pulseg_protocol *out, const char *preamble);

    /**
     * @brief Serialize a protocol as a value-only preamble (no schema).
     *
     * This is the form the VALIDATE and GENERATE bridge commands send.
     *
     * @param[in]  p      Protocol to serialize.
     * @param[out] buf    Destination buffer.
     * @param[in]  bufsz  Capacity of @p buf in bytes.
     * @return Bytes written excluding the NUL, or -1 if @p buf is too small.
     */
    int pulseg_protocol_serialize(const pulseg_protocol *p, char *buf, int bufsz);

    /* ================================================================== */
    /*  Typed getters / setters                                           */
    /* ================================================================== */

    /**
     * @brief Locate a parameter id among the protocol's populated entries.
     * @return Index into p->keys / p->values (>= 0), or -1 if absent.
     */
    int pulseg_protocol_find(const pulseg_protocol *p, int param_id);

    /**
     * @brief Read a parameter's value, type-checked against its declared type.
     *
     * @param[in]  p         Protocol to read from.
     * @param[out] out       Receives the value; untouched on failure.
     * @param[in]  param_id  PULSEG_PARAM_* id.
     * @return PULSEG_SUCCESS, or a negative code if @p param_id is absent or
     *         holds a different type.
     */
    int pulseg_protocol_get_float(const pulseg_protocol *p, float *out, int param_id);
    /** @copydoc pulseg_protocol_get_float */
    int pulseg_protocol_get_int(const pulseg_protocol *p, int *out, int param_id);
    /** @copydoc pulseg_protocol_get_float */
    int pulseg_protocol_get_bool(const pulseg_protocol *p, int *out, int param_id);

    /**
     * @brief Set a parameter's value, appending the entry if not yet present.
     *
     * @param[in,out] p         Protocol to modify.
     * @param[in]     param_id  PULSEG_PARAM_* id.
     * @param[in]     value     New value.
     * @return PULSEG_SUCCESS, or a negative code if the protocol is full or
     *         @p param_id declares a different type.
     */
    int pulseg_protocol_set_float(pulseg_protocol *p, int param_id, float value);
    /** @copydoc pulseg_protocol_set_float */
    int pulseg_protocol_set_int(pulseg_protocol *p, int param_id, int value);
    /** @copydoc pulseg_protocol_set_float */
    int pulseg_protocol_set_bool(pulseg_protocol *p, int param_id, int value);

    /**
     * @brief Read the selected index of a stringlist (dropdown) parameter.
     * @see pulseg_protocol_get_float for the return contract.
     */
    int pulseg_protocol_get_stringlist(const pulseg_protocol *p, int *idx_out, int param_id);

    /**
     * @brief Set a stringlist parameter's selected index and its option list.
     *
     * @param[in,out] p         Protocol to modify.
     * @param[in]     param_id  PULSEG_PARAM_* id.
     * @param[in]     idx       Selected option index.
     * @param[in]     options   Pipe-delimited option string, e.g. "off|low|high".
     * @return PULSEG_SUCCESS or a negative error code.
     */
    int pulseg_protocol_set_stringlist(
        pulseg_protocol *p,
        int param_id,
        int idx,
        const char *options);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_PROTOCOL_H */
