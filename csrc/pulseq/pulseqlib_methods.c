#include <ctype.h>
#include <stdio.h>
#include <string.h>

#include "pulseqlib_methods.h"

#define INIT_LIBRARY(seq, fieldPtr, sizeField, flagField) \
    do { \
        (seq)->fieldPtr = NULL; \
        (seq)->sizeField = 0; \
        (seq)->flagField = 0; \
    } while (0)

static void seqFileSetDefaults(pulseqlib_SeqFile* seq);
static void seqFileInit(pulseqlib_SeqFile* seq, const pulseqlib_Opts* opts);
static void seqFileReset(pulseqlib_SeqFile* seq);


typedef struct {
    const char *name;
    int value;
} TableEntry;


static const TableEntry label_table[] = {
    { "SLC", SLC }, 
    { "SEG", SEG }, 
    { "REP", REP }, 
    { "AVG", AVG },
    { "SET", SET }, 
    { "ECO", ECO }, 
    { "PHS", PHS }, 
    { "LIN", LIN },
    { "PAR", PAR }, 
    { "ACQ", ACQ }, 
    { "TRID", TRID },
    { "COREID", COREID },
    { "NAV", NAV },
    { "REV", REV }, 
    { "SMS", SMS }, 
    { "REF", REF }, 
    { "IMA", IMA },
    { "NOISE", NOISE }, 
    { "PMC", PMC }, 
    { "NOROT", NOROT },
    { "NOPOS", NOPOS }, 
    { "NOSCL", NOSCL }, 
    { "ONCE", ONCE },
    { NULL, -1 }
};


int label2enum(const char *label) {
    int i;
    if (!label) return -1;
    for (i = 0; label_table[i].name != NULL; i++) {
        if (strcmp(label, label_table[i].name) == 0) return label_table[i].value;
    }
    return -1;
}


static const TableEntry hint_table[] = {
    { "TE", HINT_TE }, 
    { "TR", HINT_TR },
    { "TI", HINT_TI }, 
    { "ESP", HINT_ESP },
    { "RECTIME", HINT_RECTIME },
    { "T2PREP", HINT_T2PREP }, 
    { "TE2", HINT_TE2 },
    { "TR2", HINT_TR2 },
    { NULL, -1 }
};


int hint2enum(const char *hint) {
    int i;
    if (!hint) return -1;
    for (i = 0; hint_table[i].name != NULL; i++) {
        if (strcmp(hint, hint_table[i].name) == 0) return hint_table[i].value;
    }
    return -1;
}


int initStandardLibrary(FILE* f, const long* offsets, int numSections, void** target, int* targetCount, int N) {
    char line[MAX_LINE_LENGTH];
    int maxIndex = -1;
    int sec, i, j, idx;
    char* p;
    float *array_raw;

    if (!f) return 1;
    for (sec = 0; sec < numSections; sec++) {
        if (offsets[sec] < 0) continue;  /* Skip not found */

        if (fseek(f, offsets[sec], SEEK_SET) != 0) {
            return 1;
        }

        /* Skip the section header line */
        if (!fgets(line, sizeof(line), f)) {
            return 1;
        }

        /* Read until next section or EOF */
        while (fgets(line, sizeof(line), f)) {
            p = line;
            while (*p == ' ' || *p == '\t') p++;
            if (*p == '[' || *p == 'e') break;     /* Next section */
            if (*p == '\0' || *p == '#') continue; /* Skip blank/comment */
            if (sscanf(p, "%d", &idx) == 1) {
                if (idx > maxIndex) maxIndex = idx;
            }
        }
    }

    if (maxIndex <= 0) {
        *target = NULL;
        *targetCount = 0;
        return 0; /* No entries found: treat as empty */
    }

    /* Allocate zero-filled 2D array as a single block */
    array_raw = (float*) ALLOC(sizeof(float) * N * maxIndex);
    if (!array_raw) return 1;
    for (i = 0; i < maxIndex * N; i++) {
        array_raw[i] = 0.0f;
    }
    *target = (void*)array_raw;
    *targetCount = maxIndex;
    return 0;
}


int initDefinitionsLibrary(FILE* f, long offset, pulseqlib_Definition** target, int* targetCount) {
    char line[MAX_LINE_LENGTH];
    int count = 0;
    char* p;
    char* nameToken;
    pulseqlib_Definition* defs;

    if (!f || offset < 0 || !target || !targetCount) return 1;

    if (fseek(f, offset, SEEK_SET) != 0) return 1;

    /* Skip section header line */
    if (!fgets(line, sizeof(line), f)) {
        return 1;
    }

    /* Count valid definition lines */
    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '[' || *p == 'e') break;     /* Next section */
        if (*p == '\0' || *p == '#') continue; /* Skip blank/comment */
        nameToken = strtok(p, " \t\r\n");
        if (nameToken) count++;
    }

    if (count == 0) {
        *target = NULL;
        *targetCount = 0;
        return 1;  /* No Definitions found */
    }

    defs = (pulseqlib_Definition*) ALLOC(sizeof(pulseqlib_Definition) * count);
    if (!defs) return 1;

    *target = defs;
    *targetCount = count;
    return 0;
}


int initShapesLibrary(FILE* f, long offset, pulseqlib_ShapeArbitrary** target, int* targetCount) {
    char line[MAX_LINE_LENGTH];
    int maxIndex = -1;
    int n, i, idx;
    char* p;
    pulseqlib_ShapeArbitrary* shapes = NULL;

    if (!f || !offset || !target || !targetCount) {
        return 1;  /* Invalid arguments */
    }

    if (fseek(f, offset, SEEK_SET) != 0) {
        return 1;  /* Seek failed */
    }

    /* Skip the section header line */
    if (!fgets(line, sizeof(line), f)) {
        return 1;
    }

    /* First pass: find max shape_id and count shapes */
    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '[' || *p == 'e') break;     /* Next section */
        if (*p == '\0' || *p == '#') continue; /* Skip blank/comment */
        if (strncmp(p, "shape_id", 8) == 0) {
            if (sscanf(p + 8, "%d", &idx) == 1) {
                if (idx > maxIndex) maxIndex = idx;
            }
        }
    }

    if (maxIndex <= 0) {
        *target = NULL;
        *targetCount = 0;
        return 0;
    }

    shapes = (pulseqlib_ShapeArbitrary*) ALLOC(sizeof(pulseqlib_ShapeArbitrary) * maxIndex);
    if (!shapes) return 1;
    for (i = 0; i < maxIndex; i++) {
        shapes[i].numSamples = 0;
        shapes[i].numUncompressedSamples = 0;
        shapes[i].samples = NULL;
    }

    /* Reset file pointer for second pass */
    if (fseek(f, offset, SEEK_SET) != 0) {
        FREE(shapes);
        return 1;
    }

    /* Skip the section header line */
    if (!fgets(line, sizeof(line), f)) {
        return 1;
    }

    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '[' || *p == 'e') break;     /* Next section */
        if (*p == '\0' || *p == '#') continue; /* Skip blank/comment */
        if (strncmp(p, "shape_id", 8) == 0) {
            if (sscanf(p + 8, "%d", &idx) == 1) {
                shapes[idx - 1].numSamples = 0;
                shapes[idx - 1].numUncompressedSamples = 0;
                shapes[idx - 1].samples = NULL;
            } 
        }
        else if (strncmp(p, "num_samples", 11) == 0) {
            if (sscanf(p + 11, "%d", &n) == 1) {
                shapes[idx - 1].numUncompressedSamples = n;
            }
        }
        else {
            shapes[idx - 1].numSamples++;
        }
    }

    /* Adjust numSamples to account for the last sample */
    for (i = 0; i < maxIndex; i++) {
        shapes[i].numSamples -= 1;  
    }

    /* Allocate sample arrays */
    for (i = 0; i < maxIndex; i++) {
        int j;
        n = shapes[i].numSamples;
        if (n > 0) {
            shapes[i].samples = (float*) ALLOC(sizeof(float) * n);
            if (!shapes[i].samples) {
                for (j = 0; j < i; j++) {
                    if (shapes[j].samples) FREE(shapes[j].samples);
                }
                FREE(shapes);
                return 1;
            }
        } else {
            shapes[i].samples = NULL;
        }
    }

    *target = shapes;
    *targetCount = maxIndex;
    return 0;
}


int initRfShimLibrary(FILE* f, long offset, pulseqlib_RfShimEntry** target, int* targetCount) {
    char line[MAX_LINE_LENGTH];
    int maxIndex = -1;
    char* p;
    int idx, i;
    pulseqlib_RfShimEntry* array;

    if (!f || !target || !targetCount) return 1;
    if (fseek(f, offset, SEEK_SET) != 0) return 1;

    /* Skip section header line */
    if (!fgets(line, sizeof(line), f)) {
        return 1;
    }

    /* First pass: determine max index */
    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '[' || *p == 'e') break;     /* Next section */
        if (*p == '\0' || *p == '#') continue; /* Skip blank/comment */
        if (sscanf(p, "%d", &idx) == 1) {
            if (idx > maxIndex) maxIndex = idx;
        }
    }

    if (maxIndex <= 0) {
        *target = NULL;
        *targetCount = 0;
        return 1; /* No entries found */
    }

    /* Allocate array of RfShimEntry */
    array = (pulseqlib_RfShimEntry*) ALLOC(sizeof(pulseqlib_RfShimEntry) * maxIndex);
    if (!array) return 1;

    for (i = 0; i < maxIndex; i++) {
        array[i].nChannels = 0;
    }

    *target = array;
    *targetCount = maxIndex;
    return 0;
}


typedef struct {
    int size;         /**< Number of values to scale */
    const float* values; /**< Array of scaling factors */
} Scale;


int readStandardLibrary(FILE* f, long offset, void* target, int targetCount, int N, Scale scale, int flag) {
    char line[MAX_LINE_LENGTH];
    int idx, parsed, consumed, n, offsetCol;                        
    float vals[MAX_SCALE_SIZE];
    char* scanPtr;
    char* p;
    float v;

    float *array_raw = (float*)target;
    if (!f) return 1;
    if (scale.size > MAX_SCALE_SIZE) return 1;
    if (fseek(f, offset, SEEK_SET) != 0) {
        return 1;
    }

    /* Skip section header line */
    if (!fgets(line, sizeof(line), f)) {
        return 1;
    }

    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '[' || *p == 'e') break;     /* Next section */
        if (*p == '\0' || *p == '#') continue; /* Skip blank/comment */
        if (sscanf(p, "%d", &idx) != 1) continue;
        if (idx <= 0 || idx > targetCount) continue;

        /* Move pointer past index */
        while (*p && *p != ' ' && *p != '\t') p++;
        while (*p == ' ' || *p == '\t') p++;

        parsed = 0;
        scanPtr = p;

        for (n = 0; n < scale.size; n++) {
            consumed = 0;
            if (sscanf(scanPtr, "%f%n", &v, &consumed) != 1) break;
            vals[n] = v;
            scanPtr += consumed;
            while (*scanPtr == ' ' || *scanPtr == '\t') scanPtr++;
            parsed++;
        }

        if (parsed != scale.size) continue;

        offsetCol = (flag >= 0) ? 1 : 0;
        for (n = 0; n < scale.size; n++) {
            array_raw[(idx - 1) * N + n + offsetCol] = vals[n] * scale.values[n];
        }
        if (flag >= 0) {
            array_raw[(idx - 1) * N + 0] = (float)flag;
        }
    }

    return 0;
}


void getSectionOffsets(pulseqlib_SeqFile* seq, FILE* f) {
    char line[MAX_LINE_LENGTH];
    char* p;
    long pos;

    char extName[EXT_NAME_LENGTH];
    int extId, extEnum;

    if (fseek(f, 0L, SEEK_SET) != 0) return;

    while (fgets(line, sizeof(line), f)) {
        pos = ftell(f);
        if (pos < 0) break;

        p = line;
        while (*p == ' ' || *p == '\t') p++;

        if (*p == '[') {
            if (strncmp(p, "[VERSION]", 9) == 0)
                seq->offsets.version = pos - strlen(line);
            else if (strncmp(p, "[DEFINITIONS]", 13) == 0)
                seq->offsets.definitions = pos - strlen(line);
            else if (strncmp(p, "[BLOCKS]", 8) == 0)
                seq->offsets.blocks = pos - strlen(line);
            else if (strncmp(p, "[RF]", 4) == 0)
                seq->offsets.rf = pos - strlen(line);
            else if (strncmp(p, "[GRADIENTS]", 11) == 0)
                seq->offsets.grad = pos - strlen(line);
            else if (strncmp(p, "[TRAP]", 6) == 0)
                seq->offsets.trap = pos - strlen(line);
            else if (strncmp(p, "[ADC]", 5) == 0)
                seq->offsets.adc = pos - strlen(line);
            else if (strncmp(p, "[EXTENSIONS]", 12) == 0)
                seq->offsets.extensions = pos - strlen(line);
            else if (strncmp(p, "[SHAPES]", 8) == 0)
                seq->offsets.shapes = pos - strlen(line);
            else if (strncmp(p, "[SIGNATURE]", 11) == 0)
                seq->offsets.signature = pos - strlen(line);
        }
        else if (strncmp(p, "extension", 9) == 0 && (*(p + 9) == ' ' || *(p + 9) == '\t')) {
            extId = -1;
            extEnum = EXT_UNKNOWN;
            if (sscanf(p, "extension %31s %d", extName, &extId) == 2) {
                if (strcmp(extName, "TRIGGERS") == 0) extEnum = EXT_TRIGGER;
                else if (strcmp(extName, "ROTATIONS") == 0) extEnum = EXT_ROTATION;
                else if (strcmp(extName, "LABELSET") == 0) extEnum = EXT_LABELSET;
                else if (strcmp(extName, "LABELINC") == 0) extEnum = EXT_LABELINC;
                else if (strcmp(extName, "RF_SHIMS") == 0) extEnum = EXT_RF_SHIM;
                else if (strcmp(extName, "DELAYS") == 0) extEnum = EXT_DELAY;
                switch (extEnum) {
                    case EXT_TRIGGER:
                        seq->offsets.triggers = pos - strlen(line);
                        break;
                    case EXT_ROTATION:
                        seq->offsets.rotations = pos - strlen(line);
                        break;
                    case EXT_LABELSET:
                        seq->offsets.labelset = pos - strlen(line);
                        break;
                    case EXT_LABELINC:
                        seq->offsets.labelinc = pos - strlen(line);
                        break;
                    case EXT_RF_SHIM:
                        seq->offsets.rfshim = pos - strlen(line);
                        break;
                    case EXT_DELAY:
                        seq->offsets.delays = pos - strlen(line);
                        break;
                    default:
                        break;
                }   
                if (extEnum >= 0 && extEnum < EXT_UNKNOWN) {
                    seq->extensionMap[extEnum] = extId;
                }
            }
        }
    }

    /* Set scan_cursor to EOF */
    seq->offsets.scan_cursor = ftell(f);
}


void readVersion(pulseqlib_SeqFile* seq, FILE* f) {
    char line[MAX_LINE_LENGTH];
    int major = 0, minor = 0, revision = 0;
    char key[32];
    int value;
    char* p;

    /* Check if library was already parsed */
    if (seq->isVersionParsed) return;

    /* Go to the correct section */
    if (seq->offsets.version < 0) {
        seq->isVersionParsed = 1;
        return;
    }

    if (fseek(f, seq->offsets.version, SEEK_SET) != 0) {
        return;
    }

    /* Skip section header */
    if (!fgets(line, sizeof(line), f)) {
        return;
    }

    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '\0' || *p == '#') continue;
        if (*p == '[') break;
        if (sscanf(p, "%31s %d", key, &value) == 2) {
            if (strcmp(key, "major") == 0) major = value;
            else if (strcmp(key, "minor") == 0) minor = value;
            else if (strcmp(key, "revision") == 0) revision = value;
        }
    }

    seq->versionMajor = major;
    seq->versionMinor = minor;
    seq->versionRevision = revision;
    seq->versionCombined = major * 1000000 + minor * 1000 + revision;
    seq->isVersionParsed = 1;
}


void readDefinitionsLibrary(pulseqlib_SeqFile* seq, FILE* f) {
    int ret;
    char line[MAX_LINE_LENGTH];
    int defIndex = 0;
    char* p;
    char* nameToken;
    char* token;
    char** newArray;
    int i;
    pulseqlib_Definition def;

    /* Check if library was already parsed */
    if (seq->isDefinitionsLibraryParsed) return;

    /* Go to the correct section */
    if (seq->offsets.definitions < 0) {
        seq->isDefinitionsLibraryParsed = 1;
        return;
    }

    /* Preallocate definitions array */
    ret = initDefinitionsLibrary(f, (seq->offsets).definitions, &seq->definitionsLibrary, &seq->numDefinitions);
    if (ret != 0) {
        fprintf(stderr, "Error: Failed to initialize definitionsLibrary\n");
        return;
    }

    /* Second pass — parse values */
    if (fseek(f, seq->offsets.definitions, SEEK_SET) != 0) return;

    /* Skip section header line */
    if (!fgets(line, sizeof(line), f)) {
        return;
    }

    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (isspace((unsigned char)*p)) p++;

        if (*p == '\0' || *p == '#') continue;
        if (*p == '[') break;  /* Next section begins */

        def.valueSize = 0;
        def.value = NULL;

        /* Parse name */
        nameToken = strtok(p, " \t\r\n");
        if (!nameToken) continue;
        strncpy(def.name, nameToken, DEFINITION_NAME_LENGTH - 1);
        def.name[DEFINITION_NAME_LENGTH - 1] = '\0';

        /* Parse values */
        while ((token = strtok(NULL, " \t\r\n")) != NULL) {
            newArray = (char**) ALLOC(sizeof(char*) * (def.valueSize + 1));
            for (i = 0; i < def.valueSize; i++) {
                newArray[i] = def.value[i];
            }

            newArray[def.valueSize] = (char*) ALLOC(strlen(token) + 1);
            strcpy(newArray[def.valueSize], token);
            if (def.value) FREE(def.value);
            def.value = newArray;
            def.valueSize++;
        }

        /* Assign parsed definition */
        seq->definitionsLibrary[defIndex++] = def;
    }

    seq->isDefinitionsLibraryParsed = 1;
}


void readDefinitions(pulseqlib_SeqFile* seq) {
    int i;
    char* key;
    char* value;
    float temp[3];

    /* Parse definitionsLibrary */
    for (i = 0; i < seq->numDefinitions; i++) {
        key = seq->definitionsLibrary[i].name;
        value = seq->definitionsLibrary[i].value[0];

        /* Parse required reserved definitions */
        if (strcmp(key, "GradientRasterTime") == 0) {
            seq->reservedDefinitionsLibrary.gradientRasterTime = atof(value) * 1e6; /* Convert to us */
        } else if (strcmp(key, "RadiofrequencyRasterTime") == 0) {
            seq->reservedDefinitionsLibrary.radiofrequencyRasterTime = atof(value) * 1e6; /* Convert to us */
        } else if (strcmp(key, "AdcRasterTime") == 0) {
            seq->reservedDefinitionsLibrary.adcRasterTime = atof(value) * 1e6; /* Convert to us */
        } else if (strcmp(key, "BlockDurationRaster") == 0) {
            seq->reservedDefinitionsLibrary.blockDurationRaster = atof(value) * 1e6; /* Convert to us */
        }

        /* Parse optional reserved definitions */
        else if (strcmp(key, "Name") == 0) {
            strncpy(seq->reservedDefinitionsLibrary.name, value, sizeof(seq->reservedDefinitionsLibrary.name) - 1);
            seq->reservedDefinitionsLibrary.name[sizeof(seq->reservedDefinitionsLibrary.name) - 1] = '\0';
        } else if (strcmp(key, "FOV") == 0) {
            if (sscanf(value, "%f %f %f", &temp[0], &temp[1], &temp[2]) == 3) {
                seq->reservedDefinitionsLibrary.fov[0] = temp[0] * 100.0f; /* Convert to cm */
                seq->reservedDefinitionsLibrary.fov[1] = temp[1] * 100.0f; /* Convert to cm */
                seq->reservedDefinitionsLibrary.fov[2] = temp[2] * 100.0f; /* Convert to cm */
            }
        } else if (strcmp(key, "TotalDuration") == 0) {
            seq->reservedDefinitionsLibrary.totalDuration = atof(value); /* Already in seconds */
        }
    }

    /* Check for missing required definitions */
    if (seq->reservedDefinitionsLibrary.gradientRasterTime == 0.0f ||
        seq->reservedDefinitionsLibrary.radiofrequencyRasterTime == 0.0f ||
        seq->reservedDefinitionsLibrary.adcRasterTime == 0.0f ||
        seq->reservedDefinitionsLibrary.blockDurationRaster == 0.0f) {
        fprintf(stderr, "Error: Missing required reserved definitions.\n");
    }
}


void readBlockLibrary(pulseqlib_SeqFile* seq, FILE* f) {
    int ret;
    float block_values[7] = {1, 1, 1, 1, 1, 1, 1};
    Scale blockScale;
    blockScale.size = 7;
    blockScale.values = block_values;
    const char* block_section[] = {"[BLOCKS]"};

    /* Check if library was already parsed */
    if (seq->isBlockLibraryParsed) return;

    /* Go to the correct section */
    if (seq->offsets.blocks < 0) {
        seq->isBlockLibraryParsed = 1;
        return;
    }

    /* Preallocate library */
    ret = initStandardLibrary(f,  &((seq->offsets).blocks), 1, (void**)&seq->blockLibrary, &seq->numBlocks, blockScale.size);
    if (ret != 0) {
        fprintf(stderr, "Error: Failed to initialize blockLibrary\n");
        return;
    }

    /* Parse Block library */
    ret = readStandardLibrary(f, seq->offsets.blocks, seq->blockLibrary, seq->numBlocks, blockScale.size, blockScale, -1);
    if (ret != 0) {
        fprintf(stderr, "Error: Failed to read blockLibrary from file %s\n", seq->filePath);
        return;
    }

    seq->isBlockLibraryParsed = 1;
}


void readRfLibrary(pulseqlib_SeqFile* seq, FILE* f) {
    int ret;
    float rf_values[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    Scale rfScale;
    rfScale.size = 10;
    rfScale.values = rf_values;
    const char* rf_section[] = {"[RF]"};

    /* Check if library was already parsed */
    if (seq->isRfLibraryParsed) return;

    /* Go to the correct section */
    if (seq->offsets.rf < 0) {
        seq->isRfLibraryParsed = 1;
        return;
    }

    /* Preallocate library */
    ret = initStandardLibrary(f,  &((seq->offsets).rf), 1, (void**)&seq->rfLibrary, &seq->rfLibrarySize, rfScale.size);
    if (ret != 0) {
        fprintf(stderr, "Error: Failed to initialize rfLibrary\n");
        return;
    }

    /* Parse RF library */
    ret = readStandardLibrary(f, seq->offsets.rf, seq->rfLibrary, seq->rfLibrarySize, rfScale.size, rfScale, -1);
    if (ret != 0) {
        fprintf(stderr, "Error: Failed to read rfLibrary from file %s\n", seq->filePath);
        return;
    }

    seq->isRfLibraryParsed = 1;
}


void readGradLibrary(pulseqlib_SeqFile* seq, FILE* f) {
    int ret;
    long offsets[2] = { seq->offsets.grad, seq->offsets.trap };
    int numSections = 0;
    Scale gradScale;
    gradScale.size = 6;
    gradScale.values = (float[]){ 1, 1, 1, 1, 1, 1 };
    Scale trapScale;
    trapScale.size = 5;
    trapScale.values = (float[]){ 1, 1, 1, 1, 1 };
    const char* sections[] = { "[GRADIENTS]", "[TRAP]" };

    if (seq->isGradLibraryParsed) return;

    /* Go to the correct section */
    (seq->offsets).grad = offsets[0];
    (seq->offsets).trap = offsets[1];

    /* If SeqFile does not have gradients, exit*/
    if ((seq->offsets).grad >= 0) numSections++;
    if ((seq->offsets).trap >= 0) numSections++;
    if (numSections == 0) {
        seq->isGradLibraryParsed = 1;
        return;
    }

    /* Preallocate library */
    ret = initStandardLibrary(f, offsets, 2, (void**)&seq->gradLibrary, &seq->gradLibrarySize, gradScale.size + 1);
    if (ret != 0) {
        fprintf(stderr, "Error: Failed to initialize gradLibrary\n");
        return;
    }

    /* Parse GRADIENTS library */
    if ((seq->offsets).grad >= 0){
        ret = readStandardLibrary(f, offsets[0], seq->gradLibrary, seq->gradLibrarySize, gradScale.size + 1, gradScale, 1);
        if (ret != 0) {
            fprintf(stderr, "Error: Failed to read gradLibrary ([GRADIENTS] section) from file %s\n", seq->filePath);
            return;
        }
    }

    /* Parse TRAP library */
    if ((seq->offsets).trap >= 0){
        ret = readStandardLibrary(f, offsets[1], seq->gradLibrary, seq->gradLibrarySize, gradScale.size + 1, trapScale, 0);
        if (ret != 0) {
            fprintf(stderr, "Error: Failed to read gradLibrary ([TRAP] section) from file %s\n", seq->filePath);
            return;
        }
    }

    seq->isGradLibraryParsed = 1;
}


void readAdcLibrary(pulseqlib_SeqFile* seq, FILE* f) {
    int ret;
    Scale adcScale;
    adcScale.size = 8;
    adcScale.values = (float[]){1, 1, 1, 1, 1, 1, 1, 1};
    const char* adc_section[] = {"[ADC]"};

    /* Check if library was already parsed */
    if (seq->isAdcLibraryParsed) return;

    /* Go to the correct section */
    if (seq->offsets.adc < 0) {
        seq->isAdcLibraryParsed = 1;
        return;
    }

    /* Preallocate library */
    ret = initStandardLibrary(f,  &((seq->offsets).adc), 1, (void**)&seq->adcLibrary, &seq->adcLibrarySize, adcScale.size);
    if (ret != 0) {
        fprintf(stderr, "Error: Failed to initialize adcLibrary\n");
        return;
    }

    /* Parse ADC library */
    ret = readStandardLibrary(f, seq->offsets.adc, seq->adcLibrary, seq->adcLibrarySize, adcScale.size, adcScale, -1);
    if (ret != 0) {
        fprintf(stderr, "Error: Failed to read adcLibrary from file %s\n", seq->filePath);
        return;
    }

    seq->isAdcLibraryParsed = 1;
}


void readShapesLibrary(pulseqlib_SeqFile* seq, FILE* f) {
    int ret;
    const char* shape_section[] = {"[SHAPES]"};
    char line[MAX_LINE_LENGTH];
    int shapeIndex;
    int sampleIndex;
    long pos;
    char* p;
    float val;
    
    /* Check if library was already parsed */
    if (seq->isShapesLibraryParsed) return;

    /* Go to the correct section */
    if (seq->offsets.shapes < 0) {
        seq->isShapesLibraryParsed = 1;
        return;
    }

    /* Preallocate shapes array */
    ret = initShapesLibrary(f, (seq->offsets).shapes, &seq->shapesLibrary, &seq->shapesLibrarySize);
    if (ret != 0) {
        fprintf(stderr, "Error: Failed to initialize shapesLibrary\n");
        return;
    }

    /* Second pass: Parse and fill waveform data */
    pos = seq->offsets.shapes;
    if (fseek(f, pos, SEEK_SET) != 0) return;

    /* Skip section header line */
    if (!fgets(line, sizeof(line), f)) {
        return;
    }

    /* Actual parsing */
    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;

        if (*p == '\0' || *p == '#') continue;
        if (*p == '[') break;

        /* Beginning of waveform: parse shape ID */
        if (strncmp(p, "shape_id", 8) == 0) {
            if (sscanf(p + 8, "%d", &shapeIndex) == 1) sampleIndex = 0;
        }

        /* Number of uncompressed samples: skip (already stored) */
        if (strncmp(p, "num_samples", 11) == 0) {
            continue;
        }

        /* Parse waveform sample value */
        if (shapeIndex > 0 && shapeIndex <= seq->shapesLibrarySize) {
            if (sscanf(p, "%f", &val) == 1){
                if(sampleIndex < seq->shapesLibrary[shapeIndex - 1].numSamples) {
                    seq->shapesLibrary[shapeIndex - 1].samples[sampleIndex++] = val;
                }
            }
        }
    }

    seq->isShapesLibraryParsed = 1;
}


int readLabelLibrary(FILE* f, long offset, void* target, int targetCount, int N, int* isLabelDefined) {
    char line[MAX_LINE_LENGTH];
    char* p;
    int idx, labelCode;
    float val;
    char label[LABEL_NAME_LENGTH];

    float *array_raw = (float*)target;
    if (!f || offset < 0) return 1;
    if (fseek(f, offset, SEEK_SET) != 0) return 1;

    /* Skip section header line */
    if (!fgets(line, sizeof(line), f)) {
        return 1;
    }

    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '[' || *p == 'e') break;     /* Next section */
        if (*p == '\0' || *p == '#') continue; /* Skip blank/comment */
        if (sscanf(p, "%d %f %31s", &idx, &val, label) != 3) continue;
        if (idx <= 0 || idx > targetCount) continue;
        labelCode = label2enum(label); 
        if (labelCode > 0){ /* bookkeep found labels and flags */
            isLabelDefined[labelCode - 1] = 1;
        }
        array_raw[(idx - 1) * N + 0] = val;
        array_raw[(idx - 1) * N + 1] = (float)labelCode;
    }
    
    return 0;
}


int readDelayLibrary(FILE* f, long offset, void* target, int targetCount, int N, int* isDelayDefined) {
    char line[MAX_LINE_LENGTH];
    char* p;
    int idx, hintCode;
    float numID, offsetVal, scaleVal;
    char hint[SOFT_DELAY_HINT_LENGTH];

    float *array_raw = (float*)target;
    if (!f || offset < 0) return 1;
    if (fseek(f, offset, SEEK_SET) != 0) return 1;
    
    /* Skip section header line */
    if (!fgets(line, sizeof(line), f)) {
        return 1;
    }

    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '[' || *p == 'e') break;     /* Next section */
        if (*p == '\0' || *p == '#') continue; /* Skip blank/comment */
        if (sscanf(p, "%d %f %f %f %31s", &idx, &numID, &offsetVal, &scaleVal, hint) != 5) continue;
        if (idx <= 0 || idx > targetCount) continue;
        hintCode = hint2enum(hint);
        if (hintCode > 0){ /* bookkeep found hint for UI */
            isDelayDefined[hintCode - 1] = 1;
        }
        array_raw[(idx - 1) * N + 0] = numID;
        array_raw[(idx - 1) * N + 1] = offsetVal;
        array_raw[(idx - 1) * N + 2] = scaleVal;
        array_raw[(idx - 1) * N + 3] = (float)hintCode;
    }

    return 0;
}


int readRfShimLibrary(FILE* f, long offset, pulseqlib_RfShimEntry* target, int targetCount) {
    char line[MAX_LINE_LENGTH];
    char* p;
    int idx, nCh, i, consumed;
    float val;

    if (!f || !target) return 1;
    if (fseek(f, offset, SEEK_SET) != 0) return 1;

    /* Skip section header line */
    if (!fgets(line, sizeof(line), f)) {
        return 1;
    }

    while (fgets(line, sizeof(line), f)) {
        p = line;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '[' || *p == 'e') break;     /* Next section */
        if (*p == '\0' || *p == '#') continue; /* Skip blank/comment */
        if (sscanf(p, "%d %d", &idx, &nCh) != 2) continue;
        if (idx <= 0 || idx > targetCount || nCh <= 0) continue;

        /* Skip past the index and nCh */
        while (*p && *p != ' ') p++; while (*p == ' ') p++;
        while (*p && *p != ' ') p++; while (*p == ' ') p++;

        if (nCh > MAX_RF_SHIM_CHANNELS) return 1; /* Too many channels */
        target[idx - 1].nChannels = nCh;
        for (i = 0; i < 2 * nCh; i++) {
            consumed = 0;
            if (sscanf(p, "%f%n", &val, &consumed) != 1) break;
            target[idx - 1].values[i] = val;
            p += consumed;
            while (*p == ' ' || *p == '\t') p++;
        }

    }

    return 0;
}


void readExtensionsLibrary(pulseqlib_SeqFile* seq, FILE* f) {
    int ret;
    int n;
    Scale extScale;
    extScale.size = 3;
    extScale.values = (float[]){ 1, 1, 1 };
    Scale trigScale;
    trigScale.size = 4;
    trigScale.values = (float[]){ 1, 1, 1, 1 };
    Scale rotScale;
    rotScale.size = 4;
    rotScale.values = (float[]){ 1, 1, 1, 1 };

    /* Check if library was already parsed */
    if (seq->isExtensionsLibraryParsed) return;

    /* Go to the correct section */
    if (seq->offsets.extensions < 0) {
        seq->isExtensionsLibraryParsed = 1;
        return;
    }

    /* Preallocate library */
    ret = initStandardLibrary(f, &((seq->offsets).extensions), 1, (void**)&seq->extensionsLibrary, &seq->extensionsLibrarySize, extScale.size);
    if (ret != 0) {
        fprintf(stderr, "Error: Failed to initialize extensionsLibrary\n");
        return;
    }

    if (seq->offsets.triggers >= 0){
        ret = initStandardLibrary(f, &((seq->offsets).triggers), 1, (void**)&seq->triggerLibrary, &seq->triggerLibrarySize, trigScale.size);
        if (ret != 0) {
            fprintf(stderr, "Error: Failed to initialize trigger library\n");
            return;
        }
    }

    if (seq->offsets.rotations >= 0){
        ret = initStandardLibrary(f, &((seq->offsets).rotations), 1, (void**)&seq->rotationQuaternionLibrary, &seq->rotationLibrarySize, rotScale.size);
        if (ret != 0) {
            fprintf(stderr, "Error: Failed to initialize rotations library\n");
            return;
        }
    }

    if (seq->offsets.labelset >= 0){
        ret = initStandardLibrary(f, &((seq->offsets).labelset), 1, (void**)&seq->labelsetLibrary, &seq->labelsetLibrarySize, 2);
        if (ret != 0) {
            fprintf(stderr, "Error: Failed to initialize labelset library\n");
            return;
        }
    }

    if (seq->offsets.labelinc >= 0){
        ret = initStandardLibrary(f, &((seq->offsets).labelinc), 1, (void**)&seq->labelincLibrary, &seq->labelincLibrarySize, 2);
        if (ret != 0) {
            fprintf(stderr, "Error: Failed to initialize labelinc library\n");
            return;
        }
    }

    if (seq->offsets.delays >= 0){
        ret = initStandardLibrary(f, &((seq->offsets).delays), 1, (void**)&seq->softDelayLibrary, &seq->softDelayLibrarySize, 4);
        if (ret != 0) {
            fprintf(stderr, "Error: Failed to initialize delays library\n");
            return;
        }
    }

    if (seq->offsets.rfshim >= 0){
        ret = initRfShimLibrary(f, seq->offsets.rfshim, &seq->rfShimLibrary, &seq->rfShimLibrarySize);
        if (ret != 0) {
            fprintf(stderr, "Error: Failed to initialize rf shim library\n");
            return;
        }
    }

    /* Parse Extensions library */
    ret = readStandardLibrary(f, seq->offsets.extensions, seq->extensionsLibrary, seq->extensionsLibrarySize, extScale.size, extScale, -1);
    if (ret != 0) {
        fprintf(stderr, "Error: Failed to read extensionsLibrary from file %s\n", seq->filePath);
        return;
    }

    if (seq->offsets.triggers >= 0){
        ret = readStandardLibrary(f, seq->offsets.triggers, seq->triggerLibrary, seq->triggerLibrarySize, trigScale.size, trigScale, -1);
        if (ret != 0) {
            fprintf(stderr, "Error: Failed to initialize trigger library\n");
            return;
        }
    }

    if (seq->offsets.rotations >= 0){
        ret = readStandardLibrary(f, seq->offsets.rotations, seq->rotationQuaternionLibrary, seq->rotationLibrarySize, rotScale.size, rotScale, -1);
        if (ret != 0) {
            fprintf(stderr, "Error: Failed to initialize rotations library\n");
            return;
        }
        {
            float quatNorm;
            int n;
            for(n = 1; n < seq->rotationLibrarySize; n++){
                quatNorm = sqrtf(powf(seq->rotationQuaternionLibrary[n][0], 2) + powf(seq->rotationQuaternionLibrary[n][1], 2) + 
                            powf(seq->rotationQuaternionLibrary[n][2], 2) + powf(seq->rotationQuaternionLibrary[n][3], 2));
                seq->rotationQuaternionLibrary[n][0] = seq->rotationQuaternionLibrary[n][0] / quatNorm; /* manually unroll - with so few entries, more readable than loop */
                seq->rotationQuaternionLibrary[n][1] = seq->rotationQuaternionLibrary[n][1] / quatNorm;
                seq->rotationQuaternionLibrary[n][2] = seq->rotationQuaternionLibrary[n][2] / quatNorm;
                seq->rotationQuaternionLibrary[n][3] = seq->rotationQuaternionLibrary[n][3] / quatNorm;
            }
        }
        
        /* If using matrix format, we'll convert the quaternions to matrices in pulseqlib_seqFile */
    }

    if (seq->offsets.labelset >= 0){
        ret = readLabelLibrary(f, seq->offsets.labelset, seq->labelsetLibrary, seq->labelsetLibrarySize, 2, seq->isLabelDefined);
        if (ret != 0) {
            fprintf(stderr, "Error: Failed to initialize labelset library\n");
            return;
        }
    }

    if (seq->offsets.labelinc >= 0){
        ret = readLabelLibrary(f, seq->offsets.labelinc, seq->labelincLibrary, seq->labelincLibrarySize, 2, seq->isLabelDefined);
        if (ret != 0) {
            fprintf(stderr, "Error: Failed to initialize labelinc library\n");
            return;
        }
    }

    if (seq->offsets.delays >= 0){
        ret = readDelayLibrary(f, seq->offsets.delays, seq->softDelayLibrary, seq->softDelayLibrarySize, 4, seq->isDelayDefined);
        if (ret != 0) {
            fprintf(stderr, "Error: Failed to initialize delays library\n");
            return;
        }
    }

    if (seq->offsets.rfshim >= 0){
        ret = readRfShimLibrary(f, seq->offsets.rfshim, seq->rfShimLibrary, seq->rfShimLibrarySize);
        if (ret != 0) {
            fprintf(stderr, "Error: Failed to initialize rf shim library\n");
            return;
        }
    }

    /* Prepare extensionLUT */
    for (n = 0; n < 8; n++){
        if (seq->extensionLUTSize < seq->extensionMap[n]){
            seq->extensionLUTSize = seq->extensionMap[n];
        }
    }
    if (seq->extensionLUTSize > 0){
        seq->extensionLUT = ALLOC(sizeof(int) * (seq->extensionLUTSize + 1));
        for (n = 0; n < 8; n++){
            if (seq->extensionMap[n] > 0) seq->extensionLUT[seq->extensionMap[n]] = n;
        }
    }

    seq->isExtensionsLibraryParsed = 1;
}


int decompressShape(pulseqlib_ShapeArbitrary* encoded, pulseqlib_ShapeArbitrary* result)
{
    int i, rep;
    const float *packed;
    int numPacked, numSamples;
    int countPack = 1;
    int countUnpack = 1;
    float* unpacked;
    
    /* Validate inputs */
    if (!encoded || !result) {
        return 0; /* Invalid inputs */
    }
    
    packed = encoded->samples;
    numPacked = encoded->numSamples;
    numSamples = encoded->numUncompressedSamples;
    
    /* Input shape is uncompressed - copy it */
    if (encoded->numSamples == encoded->numUncompressedSamples) {
        result->numSamples = encoded->numSamples;
        result->numUncompressedSamples = encoded->numUncompressedSamples;
        result->samples = (float*)ALLOC(sizeof(float) * encoded->numSamples);
        if (!result->samples) {
            return 0; /* Allocation failed */
        }
        memcpy(result->samples, encoded->samples, sizeof(float) * encoded->numSamples);
        return 1; /* Success */
    }

    unpacked = (float*) ALLOC(sizeof(float) * numSamples);
    if (unpacked == NULL) {
        return 0; /* Allocation failed */
    }

    while (countPack < numPacked) {
        if (packed[countPack - 1] != packed[countPack]) {
            unpacked[countUnpack - 1] = packed[countPack - 1];
            countPack++;
            countUnpack++;
        } else {
            rep = (int)(packed[countPack + 1]) + 2;
            if (fabsf(packed[countPack + 1] + 2 - (float)rep) > 1e-6f) {
                /* Malformed shape compression format */
                FREE(unpacked);
                return 0; /* Failed */
            }
            for (i = countUnpack - 1; i <= countUnpack + rep - 2; i++) {
                unpacked[i] = packed[countPack - 1];
            }
            countPack += 3;
            countUnpack += rep;
        }
    }

    if (countPack == numPacked) {
        unpacked[countUnpack - 1] = packed[countPack - 1];
    }

    /* Cumulative sum */
    for (i = 1; i < numSamples; i++) {
        unpacked[i] += unpacked[i - 1];
    }

    result->numSamples = numSamples;
    result->numUncompressedSamples = numSamples;
    result->samples = unpacked;

    return 1; /* Success */
}


/******************************************* Public methods *************************************************/
void pulseqlib_optsInit(pulseqlib_Opts* opts, float B0, float max_grad, float max_slew, float rf_raster_time, float grad_raster_time, float adc_raster_time, float block_duration_raster){
    if (!opts) return;
    opts->B0 = B0;
    opts->max_grad = max_grad;
    opts->max_slew = max_slew;
    opts->rf_raster_time = rf_raster_time;
    opts->grad_raster_time = grad_raster_time;
    opts->adc_raster_time = adc_raster_time;
    opts->block_duration_raster = block_duration_raster;
}


void pulseqlib_optsFree(pulseqlib_Opts* opts) {
    if (!opts) return;
    memset(opts, 0, sizeof(*opts));
}


static void seqFileSetDefaults(pulseqlib_SeqFile* seq) {
    int i;
    if (!seq) return;

    seq->offsets.scan_cursor = 0;
    seq->offsets.version = -1;
    seq->offsets.definitions = -1;
    seq->offsets.blocks = -1;
    seq->offsets.rf = -1;
    seq->offsets.grad = -1;
    seq->offsets.trap = -1;
    seq->offsets.adc = -1;
    seq->offsets.extensions = -1;
    seq->offsets.triggers = -1;
    seq->offsets.rfshim = -1;
    seq->offsets.labelset = -1;
    seq->offsets.labelinc = -1;
    seq->offsets.delays = -1;
    seq->offsets.rotations = -1;
    seq->offsets.shapes = -1;
    seq->offsets.signature = -1;

    seq->isVersionParsed = 0;
    seq->versionCombined = 0;
    seq->versionMajor = 0;
    seq->versionMinor = 0;
    seq->versionRevision = 0;

    INIT_LIBRARY(seq, definitionsLibrary, numDefinitions, isDefinitionsLibraryParsed);
    seq->reservedDefinitionsLibrary = (pulseqlib_ReservedDefinitions){0};

    INIT_LIBRARY(seq, blockLibrary, numBlocks, isBlockLibraryParsed);
    seq->blockIDs = NULL;
    INIT_LIBRARY(seq, rfLibrary, rfLibrarySize, isRfLibraryParsed);
    INIT_LIBRARY(seq, gradLibrary, gradLibrarySize, isGradLibraryParsed);
    INIT_LIBRARY(seq, adcLibrary, adcLibrarySize, isAdcLibraryParsed);
    INIT_LIBRARY(seq, extensionsLibrary, extensionsLibrarySize, isExtensionsLibraryParsed);
    INIT_LIBRARY(seq, triggerLibrary, triggerLibrarySize, isExtensionsLibraryParsed);
    INIT_LIBRARY(seq, rotationQuaternionLibrary, rotationLibrarySize, isExtensionsLibraryParsed);
    INIT_LIBRARY(seq, rotationMatrixLibrary, rotationLibrarySize, isExtensionsLibraryParsed);
    INIT_LIBRARY(seq, labelsetLibrary, labelsetLibrarySize, isExtensionsLibraryParsed);
    INIT_LIBRARY(seq, labelincLibrary, labelincLibrarySize, isExtensionsLibraryParsed);
    for (i = 0; i < 22; i++) {
        seq->isLabelDefined[i] = 0;
    }
    memset(&seq->labelLimits, 0, sizeof(seq->labelLimits));
    for (i = 0; i < 8; i++) {
        seq->isDelayDefined[i] = 0;
        seq->extensionMap[i] = -1;
    }
    INIT_LIBRARY(seq, softDelayLibrary, softDelayLibrarySize, isExtensionsLibraryParsed);
    INIT_LIBRARY(seq, rfShimLibrary, rfShimLibrarySize, isExtensionsLibraryParsed);
    seq->extensionLUTSize = 0;
    seq->extensionLUT = NULL;
    INIT_LIBRARY(seq, shapesLibrary, shapesLibrarySize, isShapesLibraryParsed);
}


static void seqFileInit(pulseqlib_SeqFile* seq, const pulseqlib_Opts* opts) {
    if (!seq) return;
    seq->filePath = NULL;
    seq->opts = *opts;
    seqFileSetDefaults(seq);
}


void pulseqlib_seqFileInit(pulseqlib_SeqFile* seq, const pulseqlib_Opts* opts) {
    seqFileInit(seq, opts);
}


static void seqFileReset(pulseqlib_SeqFile* seq) {
    int i, j;
    if (!seq) return;
    if (seq->isDefinitionsLibraryParsed && seq->definitionsLibrary) {
        for (i = 0; i < seq->numDefinitions; i++) {
            FREE(seq->definitionsLibrary[i].value);
        }
        FREE(seq->definitionsLibrary);
    }
    if (seq->isBlockLibraryParsed) {
        FREE(seq->blockLibrary);
        FREE(seq->blockIDs);
        seq->blockIDs = NULL;
    }
    if (seq->isRfLibraryParsed)         FREE(seq->rfLibrary);
    if (seq->isGradLibraryParsed)       FREE(seq->gradLibrary);
    if (seq->isAdcLibraryParsed)        FREE(seq->adcLibrary);
    if (seq->isExtensionsLibraryParsed) {
        FREE(seq->extensionsLibrary);
        FREE(seq->triggerLibrary);
        
        /* Free both rotation libraries to be safe */
        FREE(seq->rotationQuaternionLibrary);
        FREE(seq->rotationMatrixLibrary);
        
        FREE(seq->labelsetLibrary);
        FREE(seq->labelincLibrary);
        FREE(seq->softDelayLibrary);
        FREE(seq->rfShimLibrary);
    }
    if (seq->isShapesLibraryParsed && seq->shapesLibrary) {
        for (i = 0; i < seq->shapesLibrarySize; i++) {
            FREE(seq->shapesLibrary[i].samples);
            seq->shapesLibrary[i].samples = NULL;
            seq->shapesLibrary[i].numUncompressedSamples = 0;
            seq->shapesLibrary[i].numSamples = 0;
        }
        FREE(seq->shapesLibrary);
    }

    FREE(seq->extensionLUT);
    seq->extensionLUT = NULL;
    
    seqFileSetDefaults(seq);
}


void pulseqlib_seqFileFree(pulseqlib_SeqFile *seq) {
    if (!seq) return;
    seqFileReset(seq);
    if (seq->filePath) {
        FREE(seq->filePath);
        seq->filePath = NULL;
    }
    pulseqlib_optsFree(&seq->opts);
    FREE(seq);
}


/**
 * @brief Initializes a sequence block with default values.
 *
 * @param[out] block The pre-allocated block structure to initialize
 * @return 1 if successful, 0 if failed
 */
void pulseqlib_seqBlockInit(pulseqlib_SeqBlock* block) {
    pulseqlib_RFEvent rf;
    pulseqlib_GradEvent gx;
    pulseqlib_GradEvent gy;
    pulseqlib_GradEvent gz;
    pulseqlib_ADCEvent adc;
    pulseqlib_TriggerEvent trigger;
    pulseqlib_RotationEvent rotation;
    pulseqlib_FlagEvent flag;
    pulseqlib_LabelEvent labelset;
    pulseqlib_LabelEvent labelinc;
    pulseqlib_SoftDelayEvent delay;
    pulseqlib_RfShimmingEvent rfShimming;
    
    /* Check for null pointer */
    if (!block) return;

    /* Initialize rf Event*/
    rf.type = 0;
    gx.type = 0;
    gy.type = 0;
    gz.type = 0;
    adc.type = 0;
    trigger.type = 0;
    rotation.type = 0;
    delay.type = 0;
    rfShimming.type = 0;

    /* Initialize flag values to 0 */
    flag.trid = 0;
    flag.coreid = 0;
    flag.nav = 0;
    flag.rev = 0;
    flag.sms = 0;
    flag.ref = 0;
    flag.ima = 0;
    flag.noise = 0;
    flag.pmc = 0;
    flag.norot = 0;
    flag.nopos = 0;
    flag.noscl = 0;
    flag.once = 0;
    
    /* Initialize label values to 0 */
    labelset.slc = 0;
    labelset.seg = 0;
    labelset.rep = 0;
    labelset.avg = 0;
    labelset.set = 0;
    labelset.eco = 0;
    labelset.phs = 0;
    labelset.lin = 0;
    labelset.par = 0;
    labelset.acq = 0;
    labelinc.slc = 0;
    labelinc.seg = 0;
    labelinc.rep = 0;
    labelinc.avg = 0;
    labelinc.set = 0;
    labelinc.eco = 0;
    labelinc.phs = 0;
    labelinc.lin = 0;
    labelinc.par = 0;
    labelinc.acq = 0;

    /* Initialize the block */
    block->rf = rf;
    block->gx = gx;
    block->gy = gy;
    block->gz = gz;
    block->adc = adc;
    block->trigger = trigger;
    block->rotation = rotation;
    block->flag = flag;
    block->labelset = labelset;
    block->labelinc = labelinc;
    block->delay = delay;
    block->rfShimming = rfShimming;
}


/**
 * @brief Frees all resources associated with a SeqBlock.
 *
 * This function deallocates memory for all waveform samples and resets the block.
 *
 * @param[in,out] block The SeqBlock to be freed.
 */
void pulseqlib_seqBlockFree(pulseqlib_SeqBlock* block) {
    if (block == 0) return;

    /* RF waveforms */
    if (block->rf.type > 0){
        if (block->rf.magShape.samples) {
            FREE(block->rf.magShape.samples);
            block->rf.magShape.samples = NULL;
        }
        if (block->rf.phaseShape.samples) {
            FREE(block->rf.phaseShape.samples);
            block->rf.phaseShape.samples = NULL;
        }
        if (block->rf.timeShape.samples) {
            FREE(block->rf.timeShape.samples);
            block->rf.timeShape.samples = NULL;
        }
    }

    /* GX waveforms */
    if (block->gx.type > 1){
        if (block->gx.waveShape.samples) {
            FREE(block->gx.waveShape.samples);
            block->gx.waveShape.samples = NULL;
        }
        if (block->gx.timeShape.samples) {
            FREE(block->gx.timeShape.samples);
            block->gx.timeShape.samples = NULL;
        }
    }

    /* GY waveforms */
    if (block->gy.type > 1){
        if (block->gy.waveShape.samples) {
            FREE(block->gy.waveShape.samples);
            block->gy.waveShape.samples = NULL;
        }
        if (block->gy.timeShape.samples) {
            FREE(block->gy.timeShape.samples);
            block->gy.timeShape.samples = NULL;
        }
    }

    /* GZ waveforms */
    if (block->gz.type > 1){
        if (block->gz.waveShape.samples) {
            FREE(block->gz.waveShape.samples);
            block->gz.waveShape.samples = NULL;
        }
        if (block->gz.timeShape.samples) {
            FREE(block->gz.timeShape.samples);
            block->gz.timeShape.samples = NULL;
        }
    }

    /* ADC waveform */
    if (block->adc.type > 0){
        if (block->adc.phaseModulationShape.samples) {
            FREE(block->adc.phaseModulationShape.samples);
            block->adc.phaseModulationShape.samples = NULL;
        }
    }

    /* RF shimming arrays */
    if (block->rfShimming.type > 0){
        if (block->rfShimming.amplitudes) {
            FREE(block->rfShimming.amplitudes);
            block->rfShimming.amplitudes = NULL;
        }
        if (block->rfShimming.phases) {
            FREE(block->rfShimming.phases);
            block->rfShimming.phases = NULL;
        }
    }
}

/**
 * @brief Read SeqFile content from buffer.
 * 
 * @param[in, out] seq The SeqFile structure.
 * @param[in] f The FILE buffer.
 */
void pulseqlib_readSeqFromBuffer(pulseqlib_SeqFile* seq, FILE* f) {
    if (!seq || !f) return;

    seqFileReset(seq);

    if (seq->filePath) {
        FREE(seq->filePath);
        seq->filePath = NULL;
    }

    getSectionOffsets(seq, f);
    readVersion(seq, f);
    if (seq->versionCombined < 1005000) {
        fprintf(stderr, "Error: Unsupported sequence file version %d.%d.%d\n", seq->versionMajor, seq->versionMinor, seq->versionRevision);
        return;
    }
    readDefinitionsLibrary(seq, f); 
    readDefinitions(seq);
    readBlockLibrary(seq, f);
    readRfLibrary(seq, f);
    readGradLibrary(seq, f);
    readAdcLibrary(seq, f);
    readShapesLibrary(seq, f);
    readExtensionsLibrary(seq, f);  
    return;    
}

/**
 * @brief Read SeqFile content from file.
 * 
 * @param[in, out] seq The SeqFile structure.
 * @param[in] filePath The path to the sequence file.
 */
void pulseqlib_readSeq(pulseqlib_SeqFile* seq, const char* filePath) {
    FILE* f;
    if (!seq || !filePath) return;
    f = fopen(filePath, "r");
    if (!f) return;
    pulseqlib_readSeqFromBuffer(seq, f);
    fclose(f);
    return;
}


static int getRawBlockContentIDs(const pulseqlib_SeqFile* seq, int blockIndex, pulseqlib_RawBlock* block, int parseExtensions) {
    int nextExtID;
    int extCount;
    float* eventFloat;
    float* extData;

    if (!seq || !block || blockIndex < 0 || blockIndex >= seq->numBlocks) {
        return 0;
    }

    block->block_duration = 0;
    block->rf = -1;
    block->gx = -1;
    block->gy = -1;
    block->gz = -1;
    block->adc = -1;
    block->extCount = 0;

    if (!seq->blockLibrary || !seq->blockLibrary[blockIndex]) {
        return 0;
    }

    eventFloat = seq->blockLibrary[blockIndex];

    block->block_duration = (int)eventFloat[0];
    block->rf = (int)eventFloat[1] - 1;
    block->gx = (int)eventFloat[2] - 1;
    block->gy = (int)eventFloat[3] - 1;
    block->gz = (int)eventFloat[4] - 1;
    block->adc = (int)eventFloat[5] - 1;

    if (!parseExtensions) {
        block->extCount = 0;
        return 1;
    }

    if (!seq->isExtensionsLibraryParsed || !seq->extensionsLibrary || seq->extensionsLibrarySize <= 0) {
        return 1;
    }

    nextExtID = (int)eventFloat[6];
    extCount = 0;

    while (nextExtID > 0 && nextExtID <= seq->extensionsLibrarySize && extCount < MAX_EXTENSIONS_PER_BLOCK) {
        extData = seq->extensionsLibrary[nextExtID - 1];
        block->ext[extCount][0] = (int)extData[0];
        block->ext[extCount][1] = (int)extData[1] - 1;
        nextExtID = (int)extData[2];
        extCount += 1;
    }

    block->extCount = extCount;
    return 1;
}

static void pulseqlib_clear_block_labels(pulseqlib_BlockLabels* labels) {
    if (!labels) return;
    memset(labels, 0, sizeof(*labels));
}


static void pulseqlib_clear_block_dynamic(pulseqlib_BlockDynamic* dynamic) {
    if (!dynamic) return;
    memset(dynamic, 0, sizeof(*dynamic));
}


static void pulseqlib_apply_block_labels(const pulseqlib_BlockLabels* labels, pulseqlib_SeqBlock* block) {
    if (!labels || !block) return;
    block->labelset = labels->labelset;
    block->labelinc = labels->labelinc;
    block->flag = labels->flag;
}


static void pulseqlib_apply_block_dynamic(const pulseqlib_BlockDynamic* dynamic, pulseqlib_SeqBlock* block) {
    int i;
    if (!dynamic || !block) return;

    if (dynamic->rf.present) {
        block->rf.type = 1;
        block->rf.amplitude = dynamic->rf.amplitude;
        block->rf.freqOffset = dynamic->rf.freqOffset;
        block->rf.freqPPM = dynamic->rf.freqPPM;
        block->rf.phaseOffset = dynamic->rf.phaseOffset;
        block->rf.phasePPM = dynamic->rf.phasePPM;
    }

    if (dynamic->gx.present) {
        block->gx.amplitude = dynamic->gx.amplitude;
    }

    if (dynamic->gy.present) {
        block->gy.amplitude = dynamic->gy.amplitude;
    }

    if (dynamic->gz.present) {
        block->gz.amplitude = dynamic->gz.amplitude;
    }

    if (dynamic->adc.present) {
        block->adc.type = 1;
        block->adc.freqOffset = dynamic->adc.freqOffset;
        block->adc.freqPPM = dynamic->adc.freqPPM;
        block->adc.phaseOffset = dynamic->adc.phaseOffset;
        block->adc.phasePPM = dynamic->adc.phasePPM;
    }

    if (dynamic->rotation.present && dynamic->rotation.data && dynamic->rotation.length > 0) {
        block->rotation.type = 1;
        if (dynamic->rotation.length == 4) {
            for (i = 0; i < 4; ++i) {
                block->rotation.data.rotQuaternion[i] = dynamic->rotation.data[i];
            }
        } else if (dynamic->rotation.length == 9) {
            for (i = 0; i < 9; ++i) {
                block->rotation.data.rotMatrix[i] = dynamic->rotation.data[i];
            }
        }
    }

    if (dynamic->rfShim.present && dynamic->rfShim.entry) {
        const pulseqlib_RfShimEntry* entry = dynamic->rfShim.entry;
        int n = entry->nChannels;
        if (n > 0 && n <= MAX_RF_SHIM_CHANNELS) {
            float* amplitudes = (float*)ALLOC(sizeof(float) * n);
            float* phases = (float*)ALLOC(sizeof(float) * n);
            if (amplitudes && phases) {
                block->rfShimming.type = 1;
                block->rfShimming.nChan = n;
                block->rfShimming.amplitudes = amplitudes;
                block->rfShimming.phases = phases;
                for (i = 0; i < n; ++i) {
                    block->rfShimming.amplitudes[i] = entry->values[2 * i];
                    block->rfShimming.phases[i] = entry->values[2 * i + 1];
                }
            } else {
                if (amplitudes) FREE(amplitudes);
                if (phases) FREE(phases);
                block->rfShimming.type = 0;
                block->rfShimming.nChan = 0;
            }
        }
    }
}


static int getBlockStatic(const pulseqlib_SeqFile* seq, const pulseqlib_RawBlock* raw, pulseqlib_SeqBlock* block) {
    float* farray;
    int idx;
    int i;
    int* isRealSample = NULL;
    float* trig;
    float* delay;
    pulseqlib_ShapeArbitrary shape;

    if (!seq || !raw || !block) {
        return 0;
    }

    block->duration = raw->block_duration;

    if (raw->rf >= 0 && seq->rfLibrary && raw->rf < seq->rfLibrarySize) {
        int numRealSamples = 0;
        farray = seq->rfLibrary[raw->rf];
        block->rf.type = 1;
        block->rf.amplitude = farray[0];

        idx = (int)farray[1];
        if (idx > 0 && seq->isShapesLibraryParsed && idx <= seq->shapesLibrarySize) {
            if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape)) goto block_static_fail;
            block->rf.magShape = shape;
        } else {
            block->rf.magShape.numSamples = 0;
            block->rf.magShape.numUncompressedSamples = 0;
            block->rf.magShape.samples = NULL;
        }

        idx = (int)farray[2];
        if (idx > 0 && seq->isShapesLibraryParsed && idx <= seq->shapesLibrarySize) {
            if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape)) goto block_static_fail;
            block->rf.phaseShape = shape;
            for (i = 0; i < block->rf.phaseShape.numSamples; i++) {
                block->rf.phaseShape.samples[i] *= TWO_PI;
            }
        } else {
            block->rf.phaseShape.numSamples = 0;
            block->rf.phaseShape.numUncompressedSamples = 0;
            block->rf.phaseShape.samples = NULL;
        }

        if (DETECT_REAL_RF && block->rf.magShape.numSamples > 0 && block->rf.phaseShape.numSamples > 0) {
            isRealSample = (int*)ALLOC(block->rf.magShape.numSamples * sizeof(int));
            if (!isRealSample) goto block_static_fail;
            for (i = 0; i < block->rf.magShape.numSamples; i++) {
                isRealSample[i] = fabs(block->rf.phaseShape.samples[i]) < 1e-6 || fabs(block->rf.phaseShape.samples[i] - M_PI) < 1e-6;
            }
            for (i = 0; i < block->rf.magShape.numSamples; i++) {
                if (isRealSample[i]) {
                    numRealSamples++;
                }
            }
            if (numRealSamples == block->rf.magShape.numSamples) {
                for (i = 0; i < block->rf.magShape.numSamples; i++) {
                    if (fabs(block->rf.phaseShape.samples[i] - M_PI) < 1e-6) {
                        block->rf.magShape.samples[i] *= -1;
                    }
                }
                FREE(block->rf.phaseShape.samples);
                block->rf.phaseShape.numSamples = 0;
                block->rf.phaseShape.numUncompressedSamples = 0;
                block->rf.phaseShape.samples = NULL;
            }
            FREE(isRealSample);
            isRealSample = NULL;
        }

        idx = (int)farray[3];
        if (idx > 0 && seq->isShapesLibraryParsed && idx <= seq->shapesLibrarySize) {
            if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape)) goto block_static_fail;
            block->rf.timeShape = shape;
        } else {
            block->rf.timeShape.numSamples = 0;
            block->rf.timeShape.numUncompressedSamples = 0;
            block->rf.timeShape.samples = NULL;
        }

        block->rf.center = farray[4];
        block->rf.delay = (int)farray[5];
        block->rf.freqPPM = farray[6];
        block->rf.phasePPM = farray[7];
        block->rf.freqOffset = farray[8];
        block->rf.phaseOffset = farray[9];
    }

    if (raw->gx >= 0 && seq->gradLibrary && raw->gx < seq->gradLibrarySize) {
        farray = seq->gradLibrary[raw->gx];
        block->gx.amplitude = farray[1];
        if ((int)farray[0] == 0) {
            block->gx.type = 1;
            block->gx.trap.riseTime = (long)farray[2];
            block->gx.trap.flatTime = (long)farray[3];
            block->gx.trap.fallTime = (long)farray[4];
            block->gx.delay = (int)farray[5];
            block->gx.first = 0;
            block->gx.last = 0;
        } else if ((int)farray[0] == 1) {
            block->gx.type = 2;
            block->gx.first = farray[2];
            block->gx.last = farray[3];
            idx = (int)farray[4];
            if (idx > 0 && seq->isShapesLibraryParsed && idx <= seq->shapesLibrarySize) {
                if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape)) goto block_static_fail;
                block->gx.waveShape = shape;
            }
            idx = (int)farray[5];
            if (idx > 0 && seq->isShapesLibraryParsed && idx <= seq->shapesLibrarySize) {
                if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape)) goto block_static_fail;
                block->gx.timeShape = shape;
            } else {
                block->gx.timeShape.numSamples = 0;
                block->gx.timeShape.numUncompressedSamples = 0;
                block->gx.timeShape.samples = NULL;
            }
            block->gx.delay = (int)farray[6];
        }
    }

    if (raw->gy >= 0 && seq->gradLibrary && raw->gy < seq->gradLibrarySize) {
        farray = seq->gradLibrary[raw->gy];
        block->gy.amplitude = farray[1];
        if ((int)farray[0] == 0) {
            block->gy.type = 1;
            block->gy.trap.riseTime = (long)farray[2];
            block->gy.trap.flatTime = (long)farray[3];
            block->gy.trap.fallTime = (long)farray[4];
            block->gy.delay = (int)farray[5];
            block->gy.first = 0;
            block->gy.last = 0;
        } else if ((int)farray[0] == 1) {
            block->gy.type = 2;
            block->gy.first = farray[2];
            block->gy.last = farray[3];
            idx = (int)farray[4];
            if (idx > 0 && seq->isShapesLibraryParsed && idx <= seq->shapesLibrarySize) {
                if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape)) goto block_static_fail;
                block->gy.waveShape = shape;
            }
            idx = (int)farray[5];
            if (idx > 0 && seq->isShapesLibraryParsed && idx <= seq->shapesLibrarySize) {
                if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape)) goto block_static_fail;
                block->gy.timeShape = shape;
            } else {
                block->gy.timeShape.numSamples = 0;
                block->gy.timeShape.numUncompressedSamples = 0;
                block->gy.timeShape.samples = NULL;
            }
            block->gy.delay = (int)farray[6];
        }
    }

    if (raw->gz >= 0 && seq->gradLibrary && raw->gz < seq->gradLibrarySize) {
        farray = seq->gradLibrary[raw->gz];
        block->gz.amplitude = farray[1];
        if ((int)farray[0] == 0) {
            block->gz.type = 1;
            block->gz.trap.riseTime = (long)farray[2];
            block->gz.trap.flatTime = (long)farray[3];
            block->gz.trap.fallTime = (long)farray[4];
            block->gz.delay = (int)farray[5];
            block->gz.first = 0;
            block->gz.last = 0;
        } else if ((int)farray[0] == 1) {
            block->gz.type = 2;
            block->gz.first = farray[2];
            block->gz.last = farray[3];
            idx = (int)farray[4];
            if (idx > 0 && seq->isShapesLibraryParsed && idx <= seq->shapesLibrarySize) {
                if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape)) goto block_static_fail;
                block->gz.waveShape = shape;
            }
            idx = (int)farray[5];
            if (idx > 0 && seq->isShapesLibraryParsed && idx <= seq->shapesLibrarySize) {
                if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape)) goto block_static_fail;
                block->gz.timeShape = shape;
            } else {
                block->gz.timeShape.numSamples = 0;
                block->gz.timeShape.numUncompressedSamples = 0;
                block->gz.timeShape.samples = NULL;
            }
            block->gz.delay = (int)farray[6];
        }
    }

    if (raw->adc >= 0 && seq->adcLibrary && raw->adc < seq->adcLibrarySize) {
        farray = seq->adcLibrary[raw->adc];
        block->adc.type = 1;
        block->adc.numSamples = (int)farray[0];
        block->adc.dwellTime = (int)farray[1];
        block->adc.delay = (int)farray[2];
        block->adc.freqPPM = farray[3];
        block->adc.phasePPM = farray[4];
        block->adc.freqOffset = farray[5];
        block->adc.phaseOffset = farray[6];
        idx = (int)farray[7];
        if (idx > 0 && seq->isShapesLibraryParsed && idx <= seq->shapesLibrarySize) {
            if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape)) goto block_static_fail;
            block->adc.phaseModulationShape = shape;
        } else {
            block->adc.phaseModulationShape.numSamples = 0;
            block->adc.phaseModulationShape.numUncompressedSamples = 0;
            block->adc.phaseModulationShape.samples = NULL;
        }
    }

    if (seq->isExtensionsLibraryParsed && seq->extensionLUT) {
        for (i = 0; i < raw->extCount; ++i) {
            int extType = seq->extensionLUT[raw->ext[i][0]];
            int refIdx = raw->ext[i][1];
            switch (extType) {
                case EXT_TRIGGER:
                    if (seq->triggerLibrary) {
                        trig = seq->triggerLibrary[refIdx];
                        block->trigger.type = 1;
                        block->trigger.duration = (long)trig[3];
                        block->trigger.delay = (long)trig[2];
                        block->trigger.triggerType = (int)trig[0];
                        block->trigger.triggerChannel = (int)trig[1];
                    }
                    break;
                case EXT_DELAY:
                    if (seq->softDelayLibrary) {
                        delay = seq->softDelayLibrary[refIdx];
                        block->delay.type = 1;
                        block->delay.numID = (int)delay[0];
                        block->delay.offset = (int)delay[1];
                        block->delay.factor = (int)delay[2];
                        block->delay.hintID = (int)delay[3];
                    }
                    break;
                default:
                    break;
            }
        }
    }

    return 1;

block_static_fail:
    if (isRealSample) {
        FREE(isRealSample);
    }
    pulseqlib_seqBlockFree(block);
    pulseqlib_seqBlockInit(block);
    return 0;
}


static void getBlockDynamic(const pulseqlib_SeqFile* seq, const pulseqlib_RawBlock* raw, pulseqlib_BlockDynamic* dynamic) {
    float b0 = seq->opts.B0;
    float* farray;
    if (!dynamic) return;
    pulseqlib_clear_block_dynamic(dynamic);
    if (!seq || !raw) return;

    if (raw->rf >= 0 && seq->rfLibrary && raw->rf < seq->rfLibrarySize) {
        farray = seq->rfLibrary[raw->rf];
        dynamic->rf.present = 1;
        dynamic->rf.amplitude = farray[0];
        dynamic->rf.freqOffset = farray[8];
        dynamic->rf.freqPPM = farray[6];
        dynamic->rf.phaseOffset = farray[9];
        dynamic->rf.phasePPM = farray[7];
        dynamic->rf.totalFrequency = dynamic->rf.freqOffset + dynamic->rf.freqPPM * b0;
        dynamic->rf.totalPhase = dynamic->rf.phaseOffset + dynamic->rf.phasePPM * b0;
    }

    if (raw->gx >= 0 && seq->gradLibrary && raw->gx < seq->gradLibrarySize) {
        farray = seq->gradLibrary[raw->gx];
        dynamic->gx.present = 1;
        dynamic->gx.type = (int)farray[0];
        dynamic->gx.amplitude = farray[1];
        if (dynamic->gx.type == 1) {
            dynamic->gx.waveShapeId = (int)farray[4];
            dynamic->gx.timeShapeId = (int)farray[5];
        } else {
            dynamic->gx.waveShapeId = 0;
            dynamic->gx.timeShapeId = 0;
        }
    }

    if (raw->gy >= 0 && seq->gradLibrary && raw->gy < seq->gradLibrarySize) {
        farray = seq->gradLibrary[raw->gy];
        dynamic->gy.present = 1;
        dynamic->gy.type = (int)farray[0];
        dynamic->gy.amplitude = farray[1];
        if (dynamic->gy.type == 1) {
            dynamic->gy.waveShapeId = (int)farray[4];
            dynamic->gy.timeShapeId = (int)farray[5];
        } else {
            dynamic->gy.waveShapeId = 0;
            dynamic->gy.timeShapeId = 0;
        }
    }

    if (raw->gz >= 0 && seq->gradLibrary && raw->gz < seq->gradLibrarySize) {
        farray = seq->gradLibrary[raw->gz];
        dynamic->gz.present = 1;
        dynamic->gz.type = (int)farray[0];
        dynamic->gz.amplitude = farray[1];
        if (dynamic->gz.type == 1) {
            dynamic->gz.waveShapeId = (int)farray[4];
            dynamic->gz.timeShapeId = (int)farray[5];
        } else {
            dynamic->gz.waveShapeId = 0;
            dynamic->gz.timeShapeId = 0;
        }
    }

    if (raw->adc >= 0 && seq->adcLibrary && raw->adc < seq->adcLibrarySize) {
        farray = seq->adcLibrary[raw->adc];
        dynamic->adc.present = 1;
        dynamic->adc.freqOffset = farray[5];
        dynamic->adc.freqPPM = farray[3];
        dynamic->adc.phaseOffset = farray[6];
        dynamic->adc.phasePPM = farray[4];
        dynamic->adc.totalFrequency = dynamic->adc.freqOffset + dynamic->adc.freqPPM * b0;
        dynamic->adc.totalPhase = dynamic->adc.phaseOffset + dynamic->adc.phasePPM * b0;
    }

    if (seq->isExtensionsLibraryParsed && seq->extensionLUT) {
        int i;
        for (i = 0; i < raw->extCount; ++i) {
            int typeIdx = raw->ext[i][0];
            int refIdx = raw->ext[i][1];
            int extType;
            if (typeIdx < 0 || typeIdx >= seq->extensionLUTSize) continue;
            extType = seq->extensionLUT[typeIdx];
            if (refIdx < 0) continue;
            switch (extType) {
                case EXT_ROTATION:
                    dynamic->rotation.present = 1;
                    dynamic->rotation.index = refIdx;
#if ROTATION_FORMAT == ROTATION_FORMAT_QUATERNION
                    if (seq->rotationQuaternionLibrary && refIdx < seq->rotationLibrarySize) {
                        dynamic->rotation.data = seq->rotationQuaternionLibrary[refIdx];
                        dynamic->rotation.length = 4;
                    }
#elif ROTATION_FORMAT == ROTATION_FORMAT_MATRIX
                    if (seq->rotationMatrixLibrary && refIdx < seq->rotationLibrarySize) {
                        dynamic->rotation.data = seq->rotationMatrixLibrary[refIdx];
                        dynamic->rotation.length = 9;
                    }
#endif
                    break;
                case EXT_RF_SHIM:
                    if (seq->rfShimLibrary && refIdx < seq->rfShimLibrarySize) {
                        dynamic->rfShim.present = 1;
                        dynamic->rfShim.entry = &seq->rfShimLibrary[refIdx];
                    }
                    break;
                default:
                    break;
            }
        }
    }
}


static void getBlockLabels(const pulseqlib_SeqFile* seq, const pulseqlib_RawBlock* raw, pulseqlib_BlockLabels* labels) {
    int i;
    pulseqlib_clear_block_labels(labels);
    if (!seq || !raw || !labels) return;
    if (!seq->isExtensionsLibraryParsed || !seq->extensionLUT) return;

    for (i = 0; i < raw->extCount; ++i) {
        int typeIdx = raw->ext[i][0];
        int refIdx = raw->ext[i][1];
        int extType;
        if (typeIdx < 0 || typeIdx >= seq->extensionLUTSize) continue;
        extType = seq->extensionLUT[typeIdx];
        if (refIdx < 0) continue;

        if (extType == EXT_LABELSET && seq->labelsetLibrary && refIdx < seq->labelsetLibrarySize) {
            int labelValue = (int)seq->labelsetLibrary[refIdx][0];
            int labelID = (int)seq->labelsetLibrary[refIdx][1];
            switch (labelID) {
                case SLC: labels->labelset.slc = labelValue; break;
                case SEG: labels->labelset.seg = labelValue; break;
                case REP: labels->labelset.rep = labelValue; break;
                case AVG: labels->labelset.avg = labelValue; break;
                case SET: labels->labelset.set = labelValue; break;
                case ECO: labels->labelset.eco = labelValue; break;
                case PHS: labels->labelset.phs = labelValue; break;
                case LIN: labels->labelset.lin = labelValue; break;
                case PAR: labels->labelset.par = labelValue; break;
                case ACQ: labels->labelset.acq = labelValue; break;
                case TRID: labels->flag.trid = labelValue; break;
                case COREID: labels->flag.coreid = labelValue; break;
                case NAV: labels->flag.nav = labelValue; break;
                case REV: labels->flag.rev = labelValue; break;
                case SMS: labels->flag.sms = labelValue; break;
                case REF: labels->flag.ref = labelValue; break;
                case IMA: labels->flag.ima = labelValue; break;
                case NOISE: labels->flag.noise = labelValue; break;
                case PMC: labels->flag.pmc = labelValue; break;
                case NOROT: labels->flag.norot = labelValue; break;
                case NOPOS: labels->flag.nopos = labelValue; break;
                case NOSCL: labels->flag.noscl = labelValue; break;
                case ONCE: labels->flag.once = labelValue; break;
                default: break;
            }
        } else if (extType == EXT_LABELINC && seq->labelincLibrary && refIdx < seq->labelincLibrarySize) {
            int labelValue = (int)seq->labelincLibrary[refIdx][0];
            int labelID = (int)seq->labelincLibrary[refIdx][1];
            switch (labelID) {
                case SLC: labels->labelinc.slc = labelValue; break;
                case SEG: labels->labelinc.seg = labelValue; break;
                case REP: labels->labelinc.rep = labelValue; break;
                case AVG: labels->labelinc.avg = labelValue; break;
                case SET: labels->labelinc.set = labelValue; break;
                case ECO: labels->labelinc.eco = labelValue; break;
                case PHS: labels->labelinc.phs = labelValue; break;
                case LIN: labels->labelinc.lin = labelValue; break;
                case PAR: labels->labelinc.par = labelValue; break;
                case ACQ: labels->labelinc.acq = labelValue; break;
                default: break;
            }
        }
    }
}


void pulseqlib_getBlockStatic(const pulseqlib_SeqFile* seq, pulseqlib_SeqBlock* block, const int blockIndex) {
    pulseqlib_RawBlock raw;
    if (!seq || !block) {
        return;
    }
    if (!getRawBlockContentIDs(seq, blockIndex, &raw, 1)) {
        return;
    }
    if (!getBlockStatic(seq, &raw, block)) {
        return;
    }
}


void pulseqlib_getBlockDynamic(const pulseqlib_SeqFile* seq, pulseqlib_BlockDynamic* dynamic, const int blockIndex) {
    pulseqlib_RawBlock raw;
    if (!dynamic) {
        return;
    }
    if (!seq || !getRawBlockContentIDs(seq, blockIndex, &raw, 1)) {
        pulseqlib_clear_block_dynamic(dynamic);
        return;
    }
    getBlockDynamic(seq, &raw, dynamic);
}


void pulseqlib_getBlockDynamicWithoutExtensions(const pulseqlib_SeqFile* seq, pulseqlib_BlockDynamic* dynamic, const int blockIndex) {
    pulseqlib_RawBlock raw;
    if (!dynamic) {
        return;
    }
    if (!seq || !getRawBlockContentIDs(seq, blockIndex, &raw, 0)) {
        pulseqlib_clear_block_dynamic(dynamic);
        return;
    }
    getBlockDynamic(seq, &raw, dynamic);
}


void pulseqlib_getBlockLabels(const pulseqlib_SeqFile* seq, pulseqlib_BlockLabels* labels, const int blockIndex) {
    pulseqlib_RawBlock raw;
    if (!labels) {
        return;
    }
    if (!seq || !getRawBlockContentIDs(seq, blockIndex, &raw, 1)) {
        pulseqlib_clear_block_labels(labels);
        return;
    }
    getBlockLabels(seq, &raw, labels);
}


/**
 * @brief Retrieves a block from the sequence file.
 *
 * @param[in] seq Pointer to the SeqFile structure.
 * @param[in, out] block Pointer to a pre-allocated SeqBlock to fill.
 * @param[in] blockIndex Index of the block to retrieve.
 */
void pulseqlib_getBlock(const pulseqlib_SeqFile* seq, pulseqlib_SeqBlock* block, const int blockIndex) {
    pulseqlib_RawBlock rawBlock;
    pulseqlib_BlockDynamic dynamic;
    pulseqlib_BlockLabels labels;
    
    /* Check inputs */
    if (!seq || !block || blockIndex < 0 || blockIndex >= seq->numBlocks) {
        return; /* Invalid inputs */
    }

    if (!getRawBlockContentIDs(seq, blockIndex, &rawBlock, 1)) {
        return;
    }

    if (!getBlockStatic(seq, &rawBlock, block)) {
        return;
    }

    getBlockDynamic(seq, &rawBlock, &dynamic);
    pulseqlib_apply_block_dynamic(&dynamic, block);

    getBlockLabels(seq, &rawBlock, &labels);
    pulseqlib_apply_block_labels(&labels, block);
}

float pulseqlib_getGradLibraryMaxAmplitude(const pulseqlib_SeqFile* seq) {
    float maxAmplitude;
    int i;

    maxAmplitude = 0.0f;

    if (!seq || !seq->isGradLibraryParsed || !seq->gradLibrary || seq->gradLibrarySize <= 0) {
        return maxAmplitude;
    }

    for (i = 0; i < seq->gradLibrarySize; ++i) {
        float amplitude = seq->gradLibrary[i][1];
        if (amplitude > maxAmplitude) {
            maxAmplitude = amplitude;
        }
    }

    return maxAmplitude;
}

#define M 23   /* number of columns */

typedef struct {
    unsigned long long hash;
    int row_index;
    int label;
    char used;
} HashEntry;

static unsigned long long hash_row(const int *row)
{
    unsigned long long h = 1469598103934665603ULL;
    int i;

    for (i = 0; i < M; ++i) {
        h ^= (unsigned long long)row[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static int rows_equal(const int *a, const int *b)
{
    int i;
    for (i = 0; i < M; ++i)
        if (a[i] != b[i])
            return 0;
    return 1;
}

static size_t next_pow2(size_t x)
{
    size_t p = 1;
    while (p < x) p <<= 1;
    return p;
}

/**
 * @brief Obtain unique blocks in sequence.
 *
 * @param[in] seq Pointer to the SeqFile structure.
 * @param[in, out] uniqueBlockDefs Array of K unique block definitions, each element containing the index of first occurrence.
 * @param[in, out] uniqueBlockTable Array of N elements mapping each block to its unique definition index.
 * @param[in] index_min Minimum block index to consider (inclusive). If negative, starts from 0.
 * @param[in] index_max Maximum block index to consider (exclusive). If negative, goes to seq->numBlocks.
 * @return The number of unique blocks K.
 */
int pulseqlib_getUniqueBlocks(const pulseqlib_SeqFile* seq, int* uniqueBlockDefs, int* uniqueBlockTable, int index_min, int index_max) {
    int numBlocks;
    int startIndex, endIndex, rangeCount;
    int numUniqueBlocks;
    int (*blockDefinitions)[23];
    int* pureDelayBlock;
    int n, r, idx;
    int noRF, noGx, noGy, noGz, noADC, noExt;

    HashEntry *table;
    size_t table_size, mask;
    size_t t;
    size_t pos;
    unsigned long long h;

    /* Get number of blocks */
    if (!seq || !uniqueBlockDefs || !uniqueBlockTable) {
         return 0;
     }
    numBlocks = seq->numBlocks;
    if (numBlocks <= 0 || !seq->blockLibrary) return 0;

    /* Determine range of blocks to process */
    startIndex = (index_min < 0) ? 0 : index_min;
    endIndex   = (index_max < 0) ? numBlocks : index_max;
    if (startIndex < 0) startIndex = 0;
    if (startIndex > numBlocks) startIndex = numBlocks;
    if (endIndex < startIndex) endIndex = startIndex;
    if (endIndex > numBlocks) endIndex = numBlocks;
    rangeCount = endIndex - startIndex;

    for (n = 0; n < numBlocks; ++n) {
        uniqueBlockTable[n] = -1;
    }
    if (rangeCount <= 0) return 0;

    /* Allocate block definition matrix */
    blockDefinitions = ALLOC(rangeCount * sizeof(*blockDefinitions));
    if (!blockDefinitions) {
        return 0;
    }
    pureDelayBlock = ALLOC(rangeCount * sizeof(int));
    if (!pureDelayBlock) {
        FREE(blockDefinitions);
        return 0;
    }

    /* Build the matrix of block definition */
    for (r = 0; r < rangeCount; ++r) {
        n = startIndex + r;
        noRF = noGx = noGy = noGz = noADC = noExt = 1;

        /* 1) Duration */
        blockDefinitions[r][0] = (int)(seq->blockLibrary[n][0]);

        /* 2) RF */
        idx = (int)(seq->blockLibrary[n][1]) - 1; /* 1-based in file, 0 means none */
        if (idx >= 0 && seq->rfLibrary && idx < seq->rfLibrarySize) {
            noRF = 0;
            blockDefinitions[r][1] = (int)(seq->rfLibrary[idx][1]); /* mag_id */
            blockDefinitions[r][2] = (int)(seq->rfLibrary[idx][2]); /* phase_id */
            blockDefinitions[r][3] = (int)(seq->rfLibrary[idx][3]); /* time_id */
            blockDefinitions[r][4] = (int)(seq->rfLibrary[idx][5]); /* delay */
        } else {
            blockDefinitions[r][1] = -1;
            blockDefinitions[r][2] = -1;
            blockDefinitions[r][3] = -1;
            blockDefinitions[r][4] = -1;
        }

        /* 3) Gx */
        idx = (int)(seq->blockLibrary[n][2]) - 1; /* 1-based in file, 0 means none */
        if (idx >= 0 && seq->gradLibrary && idx < seq->gradLibrarySize) {
            noGx = 0;
            blockDefinitions[r][5] = (int)(seq->gradLibrary[idx][0]);  /* type */
            blockDefinitions[r][6] = (int)(seq->gradLibrary[idx][2]);  /* riseTime / first */
            blockDefinitions[r][7] = (int)(seq->gradLibrary[idx][3]);  /* flatTime / last */
            blockDefinitions[r][8] = (int)(seq->gradLibrary[idx][4]);  /* fallTime / wave_id */
            blockDefinitions[r][9] = (int)(seq->gradLibrary[idx][5]);  /* time_id */
            blockDefinitions[r][10] = (int)(seq->gradLibrary[idx][6]); /* delay */
        } else {
            blockDefinitions[r][5] = -1;
            blockDefinitions[r][6] = -1;
            blockDefinitions[r][7] = -1;
            blockDefinitions[r][8] = -1;
            blockDefinitions[r][9] = -1;
            blockDefinitions[r][10] = -1;
        }

        /* 4) Gy */
        idx = (int)(seq->blockLibrary[n][3]) - 1; /* 1-based in file, 0 means none */
        if (idx >= 0 && seq->gradLibrary && idx < seq->gradLibrarySize) {
            noGy = 0;
            blockDefinitions[r][11] = (int)(seq->gradLibrary[idx][0]);  /* type */
            blockDefinitions[r][12] = (int)(seq->gradLibrary[idx][2]);  /* riseTime / first */
            blockDefinitions[r][13] = (int)(seq->gradLibrary[idx][3]);  /* flatTime / last */
            blockDefinitions[r][14] = (int)(seq->gradLibrary[idx][4]);  /* fallTime / wave_id */
            blockDefinitions[r][15] = (int)(seq->gradLibrary[idx][5]);  /* time_id */
            blockDefinitions[r][16] = (int)(seq->gradLibrary[idx][6]); /* delay */
        } else {
            blockDefinitions[r][11] = -1;
            blockDefinitions[r][12] = -1;
            blockDefinitions[r][13] = -1;
            blockDefinitions[r][14] = -1;
            blockDefinitions[r][15] = -1;
            blockDefinitions[r][16] = -1;
        }

        /* 5) Gz */
        idx = (int)(seq->blockLibrary[n][4]) - 1; /* 1-based in file, 0 means none */
        if (idx >= 0 && seq->gradLibrary && idx < seq->gradLibrarySize) {
            noGz = 0;
            blockDefinitions[r][17] = (int)(seq->gradLibrary[idx][0]);  /* type */
            blockDefinitions[r][18] = (int)(seq->gradLibrary[idx][2]);  /* riseTime / first */
            blockDefinitions[r][19] = (int)(seq->gradLibrary[idx][3]);  /* flatTime / last */
            blockDefinitions[r][20] = (int)(seq->gradLibrary[idx][4]);  /* fallTime / wave_id */
            blockDefinitions[r][21] = (int)(seq->gradLibrary[idx][5]);  /* time_id */
            blockDefinitions[r][22] = (int)(seq->gradLibrary[idx][6]); /* delay */
        } else {
            blockDefinitions[r][17] = -1;
            blockDefinitions[r][18] = -1;
            blockDefinitions[r][19] = -1;
            blockDefinitions[r][20] = -1;
            blockDefinitions[r][21] = -1;
            blockDefinitions[r][22] = -1;
        }

        /* 6) ADC */
        idx = (int)(seq->blockLibrary[n][5]) - 1; /* 1-based in file, 0 means none */
        if (idx >= 0 && seq->adcLibrary && idx < seq->adcLibrarySize) {
            noADC = 0;
        } else {
            noADC = 1;
        }

        /* 7) Extensions */
        idx = (int)(seq->blockLibrary[n][6]) - 1;
        if (idx >= 0 && seq->extensionsLibrary && idx < seq->extensionsLibrarySize) {
            noExt = 0;
        } else {
            noExt = 1;
        }

        /* Check for pure delay block */
        if (noRF && noGx && noGy && noGz && noADC && noExt) {
            pureDelayBlock[r] = 1;
        } else {
            pureDelayBlock[r] = 0;
        }
    }

    /* Pure delay are all equals regarding block uniqueness */
    for (r = 0; r < rangeCount; ++r) {
        if (pureDelayBlock[r]) {
            blockDefinitions[r][0] = 0; /* Set duration to zero for uniqueness check */
        }
    }

    /* -------- HASH DEDUPLICATION -------- */
    table_size = next_pow2((size_t)rangeCount * 2);
    mask = table_size - 1;

    table = (HashEntry *)ALLOC(table_size * sizeof(HashEntry));
    if (!table) {
        FREE(blockDefinitions);
        FREE(pureDelayBlock);
        return 0;
    }

    /* Initialize hash to zero */
    for (t = 0; t < table_size; ++t) {
        table[t].used = 0;
        table[t].hash = 0;
        table[t].row_index = 0;
        table[t].label = 0;
    }

    numUniqueBlocks = 0;

    for (r = 0; r < rangeCount; ++r) {
        h = hash_row(blockDefinitions[r]);
        pos = (size_t)h & mask;
        n = startIndex + r;

        while (1) {
            if (!table[pos].used) {
                table[pos].used = 1;
                table[pos].hash = h;
                table[pos].row_index = r;
                table[pos].label = numUniqueBlocks;

                uniqueBlockDefs[numUniqueBlocks] = n;
                uniqueBlockTable[n] = numUniqueBlocks;

                numUniqueBlocks++;
                break;
            }

            if (table[pos].hash == h && rows_equal(blockDefinitions[r], blockDefinitions[table[pos].row_index])) {
                uniqueBlockTable[n] = table[pos].label;
                break;
            }

            pos = (pos + 1) & mask;
        }
    }

    FREE(table);
    FREE(blockDefinitions);
    FREE(pureDelayBlock);

    return numUniqueBlocks;
}
