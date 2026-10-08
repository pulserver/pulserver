/**
 * @file pulseg_protocol.h
 * @brief Vendor-neutral MR protocol parameter table, parse, and serialize.
 *
 * Maps the protocol block's wire format to a fixed set
 * of parameter IDs (mirroring the Python UIParam enum).  No vendor-
 * specific units, parameter names or UI concepts appear here.
 *
 * Wire format:
 *   [Protocol]
 *   TE: 5.0
 *   TR: 500.0
 *   NSlices: 10
 *   FatSat: 1
 *   [Protocol End]
 */

#ifndef PULSEG_PROTOCOL_H
#define PULSEG_PROTOCOL_H

#ifdef __cplusplus
extern "C"
{
#endif

    /* ================================================================== */
    /*  Parameter IDs  (mirror Python UIParam enum + User1..46)           */
    /*                                                                    */
    /*  Expressed as #define + typedef rather than C enum for              */
    /*  compatibility with sequence toolchains that reject C enums.       */
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
/* User parameter slots. Which scanner parameter holds a slot is the vendor
 * layer's choice. */
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
#define PULSEG_PARAM_USER20 33
#define PULSEG_PARAM_USER21 34
#define PULSEG_PARAM_USER22 35
#define PULSEG_PARAM_USER23 36
#define PULSEG_PARAM_USER24 37
#define PULSEG_PARAM_USER25 38
#define PULSEG_PARAM_USER26 39
#define PULSEG_PARAM_USER27 40
#define PULSEG_PARAM_USER28 41
#define PULSEG_PARAM_USER29 42
#define PULSEG_PARAM_USER30 43
#define PULSEG_PARAM_USER31 44
#define PULSEG_PARAM_USER32 45
#define PULSEG_PARAM_USER33 46
#define PULSEG_PARAM_USER34 47
#define PULSEG_PARAM_USER35 48
#define PULSEG_PARAM_USER36 49
#define PULSEG_PARAM_USER37 50
#define PULSEG_PARAM_USER38 51
#define PULSEG_PARAM_USER39 52
#define PULSEG_PARAM_USER40 53
#define PULSEG_PARAM_USER41 54
#define PULSEG_PARAM_USER42 55
#define PULSEG_PARAM_USER43 56
#define PULSEG_PARAM_USER44 57
#define PULSEG_PARAM_USER45 58
#define PULSEG_PARAM_USER46 59
/* --- Extended timing --- */
#define PULSEG_PARAM_TE2 60
#define PULSEG_PARAM_TRECOVERY 61
/* --- Extended spatial --- */
#define PULSEG_PARAM_PHASE_FOV 62
#define PULSEG_PARAM_SLICE_SPACING 63
#define PULSEG_PARAM_NY 64
#define PULSEG_PARAM_NUM_SLABS 65
#define PULSEG_PARAM_OVERLAP_LOCS 66
/* --- Acquisition --- */
#define PULSEG_PARAM_NEX 67
#define PULSEG_PARAM_NUM_SHOTS 68
#define PULSEG_PARAM_ETL 69
/* --- Enum / stringlist --- */
#define PULSEG_PARAM_SEQUENCE_TYPE 70
#define PULSEG_PARAM_IMAGING_MODE 71
#define PULSEG_PARAM_PREP_TYPE 72
#define PULSEG_PARAM_TRIGGER_TYPE 73
/* --- Extended flags --- */
#define PULSEG_PARAM_SWAP_PF 74
#define PULSEG_PARAM_ENABLE_SAT_UI 75
#define PULSEG_PARAM_RECORD_PHYSIO 76
/* --- Acceleration --- */
#define PULSEG_PARAM_RY 77
#define PULSEG_PARAM_RZ 78
#define PULSEG_PARAM_COMPRESSED_SENS 79
#define PULSEG_PARAM_MULTIBAND 80
/* --- Cine / trigger --- */
#define PULSEG_PARAM_NUM_FRAMES 81
#define PULSEG_PARAM_DELAY_TIME 82
#define PULSEG_PARAM_TRIGGER_DELAY 83
#define PULSEG_PARAM_TRIGGER_WINDOW 84
/* --- Diffusion --- */
#define PULSEG_PARAM_DIFF_BVALUES 85
#define PULSEG_PARAM_DIFF_DIRECTIONS 86
/* User parameter name labels (description type; one per USER1..USER46 slot) */
#define PULSEG_PARAM_USER1_NAME 87
#define PULSEG_PARAM_USER2_NAME 88
#define PULSEG_PARAM_USER3_NAME 89
#define PULSEG_PARAM_USER4_NAME 90
#define PULSEG_PARAM_USER5_NAME 91
#define PULSEG_PARAM_USER6_NAME 92
#define PULSEG_PARAM_USER7_NAME 93
#define PULSEG_PARAM_USER8_NAME 94
#define PULSEG_PARAM_USER9_NAME 95
#define PULSEG_PARAM_USER10_NAME 96
#define PULSEG_PARAM_USER11_NAME 97
#define PULSEG_PARAM_USER12_NAME 98
#define PULSEG_PARAM_USER13_NAME 99
#define PULSEG_PARAM_USER14_NAME 100
#define PULSEG_PARAM_USER15_NAME 101
#define PULSEG_PARAM_USER16_NAME 102
#define PULSEG_PARAM_USER17_NAME 103
#define PULSEG_PARAM_USER18_NAME 104
#define PULSEG_PARAM_USER19_NAME 105
#define PULSEG_PARAM_USER20_NAME 106
#define PULSEG_PARAM_USER21_NAME 107
#define PULSEG_PARAM_USER22_NAME 108
#define PULSEG_PARAM_USER23_NAME 109
#define PULSEG_PARAM_USER24_NAME 110
#define PULSEG_PARAM_USER25_NAME 111
#define PULSEG_PARAM_USER26_NAME 112
#define PULSEG_PARAM_USER27_NAME 113
#define PULSEG_PARAM_USER28_NAME 114
#define PULSEG_PARAM_USER29_NAME 115
#define PULSEG_PARAM_USER30_NAME 116
#define PULSEG_PARAM_USER31_NAME 117
#define PULSEG_PARAM_USER32_NAME 118
#define PULSEG_PARAM_USER33_NAME 119
#define PULSEG_PARAM_USER34_NAME 120
#define PULSEG_PARAM_USER35_NAME 121
#define PULSEG_PARAM_USER36_NAME 122
#define PULSEG_PARAM_USER37_NAME 123
#define PULSEG_PARAM_USER38_NAME 124
#define PULSEG_PARAM_USER39_NAME 125
#define PULSEG_PARAM_USER40_NAME 126
#define PULSEG_PARAM_USER41_NAME 127
#define PULSEG_PARAM_USER42_NAME 128
#define PULSEG_PARAM_USER43_NAME 129
#define PULSEG_PARAM_USER44_NAME 130
#define PULSEG_PARAM_USER45_NAME 131
#define PULSEG_PARAM_USER46_NAME 132
/* --- FOV offset ---
 * The prescribed field-of-view offset, in millimetres along the logical
 * readout, phase-encoding and slice axes.
 * The interpreter fills these from its prescription and sends them with the
 * protocol; the host applies the offset to the logical-frame design as RF and
 * ADC frequency and phase when it builds the IR, so the cache is played
 * through the prescription's rotation matrix alone. */
#define PULSEG_PARAM_FOV_OFFSET_X 133
#define PULSEG_PARAM_FOV_OFFSET_Y 134
#define PULSEG_PARAM_FOV_OFFSET_Z 135
/* --- Configuration (PULSEG_PTYPE_CONFIG; see below) --- */
/* The scan's request to be costed against the scanner's SAR burst limits
 * rather than its continuous ones.  A request, not a setting: the vendor
 * layer offers it to the scanner, which grants or refuses it. */
#define PULSEG_PARAM_ENABLE_SAR_BURST 136
/* --- FOV orientation ---
 * The prescription's rotation from the logical readout, phase-encoding and
 * slice axes to the physical x, y and z gradient axes, element (i, j) as
 * FOV_ROTATION_ij, so that physical = R logical: column j is logical axis j
 * in physical coordinates.  R is orthonormal, a reflection included, to the
 * precision the protocol carries.  The interpreter fills these from its
 * prescription and sends them with the protocol, identity when it has none.
 * The host checks the design's gradients in the physical frame this rotation
 * gives; the cache itself is played through the scanner's rotation matrix. */
#define PULSEG_PARAM_FOV_ROTATION_11 137
#define PULSEG_PARAM_FOV_ROTATION_12 138
#define PULSEG_PARAM_FOV_ROTATION_13 139
#define PULSEG_PARAM_FOV_ROTATION_21 140
#define PULSEG_PARAM_FOV_ROTATION_22 141
#define PULSEG_PARAM_FOV_ROTATION_23 142
#define PULSEG_PARAM_FOV_ROTATION_31 143
#define PULSEG_PARAM_FOV_ROTATION_32 144
#define PULSEG_PARAM_FOV_ROTATION_33 145
/* --- Gating --- */
#define PULSEG_PARAM_HEART_RATE 146
/* --- Explicit saturation bands: the mask is a configuration
 *     (PULSEG_PTYPE_CONFIG) the sequence declares, bit n - 1 set for each
 *     band n it plays; each band's normal along the physical axes, its
 *     centre's distance from the isocentre along it and its thickness, in
 *     mm --- */
#define PULSEG_PARAM_EXSAT_MASK 147
#define PULSEG_PARAM_EXSAT1_NORMAL_X 148
#define PULSEG_PARAM_EXSAT1_NORMAL_Y 149
#define PULSEG_PARAM_EXSAT1_NORMAL_Z 150
#define PULSEG_PARAM_EXSAT1_LOC 151
#define PULSEG_PARAM_EXSAT1_THICK 152
#define PULSEG_PARAM_EXSAT2_NORMAL_X 153
#define PULSEG_PARAM_EXSAT2_NORMAL_Y 154
#define PULSEG_PARAM_EXSAT2_NORMAL_Z 155
#define PULSEG_PARAM_EXSAT2_LOC 156
#define PULSEG_PARAM_EXSAT2_THICK 157
#define PULSEG_PARAM_EXSAT3_NORMAL_X 158
#define PULSEG_PARAM_EXSAT3_NORMAL_Y 159
#define PULSEG_PARAM_EXSAT3_NORMAL_Z 160
#define PULSEG_PARAM_EXSAT3_LOC 161
#define PULSEG_PARAM_EXSAT3_THICK 162
#define PULSEG_PARAM_EXSAT4_NORMAL_X 163
#define PULSEG_PARAM_EXSAT4_NORMAL_Y 164
#define PULSEG_PARAM_EXSAT4_NORMAL_Z 165
#define PULSEG_PARAM_EXSAT4_LOC 166
#define PULSEG_PARAM_EXSAT4_THICK 167
#define PULSEG_PARAM_EXSAT5_NORMAL_X 168
#define PULSEG_PARAM_EXSAT5_NORMAL_Y 169
#define PULSEG_PARAM_EXSAT5_NORMAL_Z 170
#define PULSEG_PARAM_EXSAT5_LOC 171
#define PULSEG_PARAM_EXSAT5_THICK 172
#define PULSEG_PARAM_EXSAT6_NORMAL_X 173
#define PULSEG_PARAM_EXSAT6_NORMAL_Y 174
#define PULSEG_PARAM_EXSAT6_NORMAL_Z 175
#define PULSEG_PARAM_EXSAT6_LOC 176
#define PULSEG_PARAM_EXSAT6_THICK 177
#define PULSEG_PARAM_COUNT 178 /* sentinel */

    /* ================================================================== */
    /*  Parameter types                                                   */
    /* ================================================================== */

    typedef int pulseg_param_type;

#define PULSEG_PTYPE_FLOAT 0
#define PULSEG_PTYPE_INT 1
#define PULSEG_PTYPE_BOOL 2
#define PULSEG_PTYPE_STRINGLIST 3
#define PULSEG_PTYPE_DESCRIPTION 4
/* A configuration value: an integer the sequence declares about itself, which
 * the vendor layer reads once while it is setting the scan up.  It carries no
 * schema because there is no widget to build from one, and it is not
 * serialized back, because nothing on the console side can change it. */
#define PULSEG_PTYPE_CONFIG 5

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
     * @brief Parse a protocol block into a protocol.
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
     * This is the value block `pulserver design validate` and
     * `pulserver design generate` read on standard input.
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
    /** @copydoc pulseg_protocol_get_float */
    int pulseg_protocol_get_config(const pulseg_protocol *p, int *out, int param_id);

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
