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
        return 1; /* No entries found */
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
        fprintf(stderr, "Error: Failed to initialize rfLibrary\n");
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
void seqFileInit(pulseqlib_SeqFile* seq) {
    int i;
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

    INIT_LIBRARY(seq, definitionsLibrary, numDefinitions, isDefinitionsLibraryParsed);
    INIT_LIBRARY(seq, blockLibrary, numBlocks, isBlockLibraryParsed);
    seq->blockIDs = NULL; /* Initialize blockIDs to NULL */
    INIT_LIBRARY(seq, rfLibrary, rfLibrarySize, isRfLibraryParsed);
    INIT_LIBRARY(seq, gradLibrary, gradLibrarySize, isGradLibraryParsed);
    INIT_LIBRARY(seq, adcLibrary, adcLibrarySize, isAdcLibraryParsed);
    INIT_LIBRARY(seq, extensionsLibrary, extensionsLibrarySize, isExtensionsLibraryParsed);
    INIT_LIBRARY(seq, triggerLibrary, triggerLibrarySize, isExtensionsLibraryParsed);
    INIT_LIBRARY(seq, rotationQuaternionLibrary, rotationLibrarySize, isExtensionsLibraryParsed);
    INIT_LIBRARY(seq, rotationMatrixLibrary, rotationLibrarySize, isExtensionsLibraryParsed);
    INIT_LIBRARY(seq, labelsetLibrary, labelsetLibrarySize, isExtensionsLibraryParsed);
    INIT_LIBRARY(seq, labelincLibrary, labelincLibrarySize, isExtensionsLibraryParsed);
    for (i = 0; i < 22; i++){
        seq->isLabelDefined[i] = 0;
    }
    INIT_LIBRARY(seq, softDelayLibrary, softDelayLibrarySize, isExtensionsLibraryParsed);
    INIT_LIBRARY(seq, rfShimLibrary, rfShimLibrarySize, isExtensionsLibraryParsed);
    for (i = 0; i < 8; i++){
        seq->extensionMap[i] = -1;
    }
    seq->extensionLUTSize = 0;
    seq->extensionLUT = NULL;
    INIT_LIBRARY(seq, shapesLibrary, shapesLibrarySize, isShapesLibraryParsed);
}


/**
 * @brief Initialize SeqFile fields.
 * 
 * @param[in] filePath The path of .seq file on disk.  
 * @param[in, out] seq The uninitialized SeqFile structure.
 */
void pulseqlib_seqFileInit(const char* filePath, pulseqlib_SeqFile* seq) {
    seqFileInit(seq);

    /* Allocate and copy the file path */
    seq->filePath = (char*) ALLOC(strlen(filePath) + 1);
    strcpy(seq->filePath, filePath);
}


/**
 * @brief Free SeqFile structure.
 * 
 * @param[in, out] seq The SeqFile structure.
 */
void pulseqlib_seqFileFree(pulseqlib_SeqFile *seq) {
    pulseqlib_seqFileReset(seq);
    FREE(seq->filePath);
    FREE(seq);
}


/**
 * @brief Reset SeqFile fields.
 * 
 * @param[in, out] seq The SeqFile structure.
 */
void pulseqlib_seqFileReset(pulseqlib_SeqFile* seq) {
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
    
    seqFileInit(seq);
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
 * @brief Read SeqFile content
 * 
 * @param[in, out] seq The SeqFile structure.
 * @param[in] readBlocks Whether to parse the BlockLibrary or not.   
 */
void pulseqlib_readSeq(pulseqlib_SeqFile* seq, const int readBlocks) {
    FILE* f = fopen(seq->filePath, "r");
    
    if (!f) return;
    getSectionOffsets(seq, f);
    readVersion(seq, f);
    if (seq->versionCombined < 1005000) {
        fprintf(stderr, "Error: Unsupported sequence file version %d.%d.%d\n", seq->versionMajor, seq->versionMinor, seq->versionRevision);
        fclose(f);
        return;
    }
    readDefinitionsLibrary(seq, f); 
    readDefinitions(seq);
    if (readBlocks) {
        readBlockLibrary(seq, f);
    }
    readRfLibrary(seq, f);
    readGradLibrary(seq, f);
    readAdcLibrary(seq, f);
    readShapesLibrary(seq, f);
    readExtensionsLibrary(seq, f);      
    fclose(f);
    
    return;
}


/**
 * @brief Get the raw block content IDs from the sequence file.
 *
 * @param[in] seq Pointer to the SeqFile structure.
 * @param[in] blockIndex Index of the block to retrieve.
 * @param[in, out] block Pointer to the block's content IDs and extension data.
 */
void pulseqlib_getRawBlockContentIDs(const pulseqlib_SeqFile* seq, const int blockIndex, pulseqlib_RawBlock* block) {
    int i, nextExtID;
    float* eventFloat;
    float* extData;
    int extCount;

    /* Initialize */
    block->adc = 0;
    block->rf = 0;
    block->gx = 0;
    block->gy = 0;
    block->gz = 0;
    block->adc = 0;
    block->extCount = 0;

    /* Sanity check */
    if (seq == 0 || blockIndex < 0 || blockIndex >= seq->numBlocks) {
        return;
    }

    /* Access float data row and cast entries to int */
    eventFloat = seq->blockLibrary[blockIndex];

    int duration = (int)(eventFloat[0]);
    int rfID = (int)(eventFloat[1]) - 1;
    int gxID = (int)(eventFloat[2]) - 1;
    int gyID = (int)(eventFloat[3]) - 1;
    int gzID = (int)(eventFloat[4]) - 1;
    int adcID  = (int)(eventFloat[5]) - 1;
    int extID = (int)(eventFloat[6]);

    block->block_duration = duration;
    block->rf = rfID;
    block->gx = gxID;
    block->gy = gyID;
    block->gz = gzID;
    block->adc = adcID;

    /* Handle extensions if present */
    if (extID > 0 && seq->isExtensionsLibraryParsed) {
        nextExtID = extID;
        extCount = 0;

        while (nextExtID > 0 && nextExtID <= seq->extensionsLibrarySize) {
            extData = seq->extensionsLibrary[nextExtID - 1]; /* [type, ref, next_id] */
            block->ext[extCount][0] = (int)extData[0];      /* type */
            block->ext[extCount][1] = (int)extData[1] - 1;  /* ref */
            nextExtID = (int)extData[2]; /* next in chain */
            extCount += 1;
        }

        block->extCount = extCount;
    }

    return;
}

/**
 * @brief Retrieves a block from the sequence file.
 *
 * @param[in] seq Pointer to the SeqFile structure.
 * @param[in] blockIndex Index of the block to retrieve.
 * @param[in, out] block Pointer to a pre-allocated SeqBlock to fill.
 */
void pulseqlib_getBlock(const pulseqlib_SeqFile* seq, const int blockIndex, pulseqlib_SeqBlock* block) {
    float* farray;
    int idx;
    int i, labelID, labelValue, extType, extIdx;
    int numRealSamples = 0;
    float* trig;
    float* rot;
    float* delay;
    int* isRealSample;
    pulseqlib_RfShimEntry rfshim;
    pulseqlib_RawBlock rawBlock;
    pulseqlib_ShapeArbitrary shape;
    
    /* Check inputs */
    if (!seq || !block || blockIndex < 0 || blockIndex >= seq->numBlocks) {
        return; /* Invalid inputs */
    }
    
    pulseqlib_getRawBlockContentIDs(seq, blockIndex, &rawBlock);

    /* Set the duration */
    block->duration = rawBlock.block_duration;

    /* ------------------ RF Event ------------------ */
    if (rawBlock.rf >= 0) {
        farray = seq->rfLibrary[rawBlock.rf];
        block->rf.type = 1;
        block->rf.amplitude = farray[0];

        idx = (int)farray[1];
        if (idx > 0) {
            if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape)) {
                return; /* Failed to decompress shape */
            }
            block->rf.magShape = shape;
        }

        idx = (int)farray[2];
        if (idx > 0) {
            if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape)) {
                return; /* Failed to decompress shape */
            }
            block->rf.phaseShape = shape;
            for (i = 0; i < block->rf.phaseShape.numSamples; i++) {
                block->rf.phaseShape.samples[i] *= TWO_PI; /* Rescale phase shape to radians */
            }
        } else {
            block->rf.phaseShape.numSamples = 0; /* Set phase shape to 0 samples */
            block->rf.phaseShape.numUncompressedSamples = 0; /* Set phase shape to 0 samples */
            block->rf.phaseShape.samples = NULL; /* Free phase shape samples */
        }

        /* Attempt to detect real-valued RF waveform */
        if (DETECT_REAL_RF && block->rf.magShape.numSamples > 0 && block->rf.phaseShape.numSamples > 0) {
            isRealSample = (int*)ALLOC(block->rf.magShape.numSamples * sizeof(int));

            /* Check if the phase shape is real-valued */
            for (i = 0; i < block->rf.magShape.numSamples; i++) {
                isRealSample[i] = fabs(block->rf.phaseShape.samples[i]) < 1e-6 || fabs(block->rf.phaseShape.samples[i] - M_PI) < 1e-6;
            }
            for (i = 0; i < block->rf.magShape.numSamples; i++) {
                if (isRealSample[i]) {
                    numRealSamples++;
                }
            }

            /* If all samples are real, restore sign, set the phase shape to 0 samples and free it */
            if (numRealSamples == block->rf.magShape.numSamples) {

                /* Restore sign of magnitude shape */
                for (i = 0; i < block->rf.magShape.numSamples; i++) {
                    if (fabs(block->rf.phaseShape.samples[i] - M_PI) < 1e-6) {
                        block->rf.magShape.samples[i] *= -1;
                    }
                }

                /* Free phase shape */
                block->rf.phaseShape.numSamples = 0; /* Set phase shape to 0 samples */
                block->rf.phaseShape.numUncompressedSamples = 0; /* Set phase shape to 0 samples */
                FREE(block->rf.phaseShape.samples); /* Free phase shape samples */
                block->rf.phaseShape.samples = NULL; /* Free phase shape samples */
            }
            FREE(isRealSample);
        }

        idx = (int)farray[3];
        if (idx > 0) {
            if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape)) {
                return; /* Failed to decompress shape */
            }
            block->rf.timeShape = shape;
        } else {
            block->rf.timeShape.numSamples = 0; /* Set time shape to 0 samples */
            block->rf.timeShape.numUncompressedSamples = 0; /* Set time shape to 0 samples */
            block->rf.timeShape.samples = NULL; /* Free time shape samples */
        }

        block->rf.center = farray[4];
        block->rf.delay = (int)farray[5];
        block->rf.freqPPM = farray[6];
        block->rf.phasePPM = farray[7];
        block->rf.freqOffset = farray[8];
        block->rf.phaseOffset = farray[9];
    }

    /* ------------------ Gradient GX ------------------ */
    if (rawBlock.gx >= 0) {
        farray = seq->gradLibrary[rawBlock.gx];
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
            if (idx > 0) {
                if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape)) {
                    return; /* Failed to decompress shape */
                }
                block->gx.waveShape = shape;
            }

            idx = (int)farray[5];
            if (idx > 0) {
                if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape)) {
                    return; /* Failed to decompress shape */
                }
                block->gx.timeShape = shape;
            } else {
                block->gx.timeShape.numSamples = 0; /* Set time shape to 0 samples */
                block->gx.timeShape.numUncompressedSamples = 0; /* Set time shape to 0 samples */
                block->gx.timeShape.samples = NULL; /* Free time shape samples */
            }

            block->gx.delay = (int)farray[6];
        }
    }

    /* ------------------ Gradient GY ------------------ */
    if (rawBlock.gy >= 0) {
        farray = seq->gradLibrary[rawBlock.gy];
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
            if (idx > 0) {
                if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape)) {
                    return; /* Failed to decompress shape */
                }
                block->gy.waveShape = shape;
            }

            idx = (int)farray[5];
            if (idx > 0) {
                if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape)) {
                    return; /* Failed to decompress shape */
                }
                block->gy.timeShape = shape;
            } else {
                block->gy.timeShape.numSamples = 0; /* Set time shape to 0 samples */
                block->gy.timeShape.numUncompressedSamples = 0; /* Set time shape to 0 samples */
                block->gy.timeShape.samples = NULL; /* Free time shape samples */
            }

            block->gy.delay = (int)farray[6];
        }
    }

    /* ------------------ Gradient GZ ------------------ */
    if (rawBlock.gz >= 0) {
        farray = seq->gradLibrary[rawBlock.gz];
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
            if (idx > 0) {
                if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape)) {
                    return; /* Failed to decompress shape */
                }
                block->gz.waveShape = shape;
            }

            idx = (int)farray[5];
            if (idx > 0) {
                if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape)) {
                    return; /* Failed to decompress shape */
                }
                block->gz.timeShape = shape;
            } else {
                block->gz.timeShape.numSamples = 0; /* Set time shape to 0 samples */
                block->gz.timeShape.numUncompressedSamples = 0; /* Set time shape to 0 samples */
                block->gz.timeShape.samples = NULL; /* Free time shape samples */
            }

            block->gz.delay = (int)farray[6];
        }
    }

    /* ------------------ ADC Event ------------------ */
    if (rawBlock.adc >= 0) {
        farray = seq->adcLibrary[rawBlock.adc];
        block->adc.type = 1;
        block->adc.numSamples = (int)farray[0];
        block->adc.dwellTime = (int)farray[1];
        block->adc.delay = (int)farray[2];
        block->adc.freqPPM = farray[3];
        block->adc.phasePPM = farray[4];
        block->adc.freqOffset = farray[5];
        block->adc.phaseOffset = farray[6];

        idx = (int)farray[7];
        if (idx > 0) {
            if (!decompressShape(&(seq->shapesLibrary[idx - 1]), &shape)) {
                return; /* Failed to decompress shape */
            }
            block->adc.phaseModulationShape = shape;
        } else {
            block->adc.phaseModulationShape.numSamples = 0; /* Set phase modulation shape to 0 samples */
            block->adc.phaseModulationShape.numUncompressedSamples = 0; /* Set phase modulation shape to 0 samples */
            block->adc.phaseModulationShape.samples = NULL; /* Free phase modulation shape samples */
        }
    }

     /* ------------------ Extensions ------------------ */
    for (i = 0; i < rawBlock.extCount; i++) {
        extType = seq->extensionLUT[rawBlock.ext[i][0]];
        extIdx = rawBlock.ext[i][1];

        switch (extType) {
            case EXT_TRIGGER:   
                trig = seq->triggerLibrary[extIdx];
                block->trigger.type = 1;
                block->trigger.duration = (long)trig[3];
                block->trigger.delay = (long)trig[2];
                block->trigger.triggerType = (int)trig[0];
                block->trigger.triggerChannel = (int)trig[1];
                break;
            case EXT_ROTATION:
                {
                    int ridx;
                    block->rotation.type = 1;
                    
                    #if ROTATION_FORMAT == ROTATION_FORMAT_QUATERNION
                    if (seq->rotationQuaternionLibrary) {
                        rot = seq->rotationQuaternionLibrary[extIdx];
                        block->rotation.data.rotQuaternion[0] = rot[0];
                        block->rotation.data.rotQuaternion[1] = rot[1];
                        block->rotation.data.rotQuaternion[2] = rot[2];
                        block->rotation.data.rotQuaternion[3] = rot[3];
                    }
                    #elif ROTATION_FORMAT == ROTATION_FORMAT_MATRIX
                    if (seq->rotationMatrixLibrary) {
                        /* Copy the rotation matrix data */
                        for (ridx = 0; ridx < 9; ridx++) {
                            block->rotation.data.rotMatrix[ridx] = seq->rotationMatrixLibrary[extIdx][ridx];
                        }
                    }
                    #endif
                }
                break;
            case EXT_LABELSET:
                labelID = seq->labelsetLibrary[extIdx][1];
                labelValue = seq->labelsetLibrary[extIdx][0];
                
                /* Handle flag values - those that don't affect ADC labeling */
                switch (labelID) {
                    case SLC: block->labelset.slc = labelValue; break;
                    case SEG: block->labelset.seg = labelValue; break;
                    case REP: block->labelset.rep = labelValue; break;
                    case AVG: block->labelset.avg = labelValue; break;
                    case SET: block->labelset.set = labelValue; break;
                    case ECO: block->labelset.eco = labelValue; break;
                    case PHS: block->labelset.phs = labelValue; break;
                    case LIN: block->labelset.lin = labelValue; break;
                    case PAR: block->labelset.par = labelValue; break;
                    case ACQ: block->labelset.acq = labelValue; break;
                    case TRID: block->flag.trid = labelValue; break;
                    case COREID: block->flag.coreid = labelValue; break;
                    case NAV: block->flag.nav = labelValue; break;
                    case REV: block->flag.rev = labelValue; break;
                    case SMS: block->flag.sms = labelValue; break;
                    case REF: block->flag.ref = labelValue; break;
                    case IMA: block->flag.ima = labelValue; break;
                    case NOISE: block->flag.noise = labelValue; break;
                    case PMC: block->flag.pmc = labelValue; break;
                    case NOROT: block->flag.norot = labelValue; break;
                    case NOPOS: block->flag.nopos = labelValue; break;
                    case NOSCL: block->flag.noscl = labelValue; break;
                    case ONCE: block->flag.once = labelValue; break;
                    default: break;
                }
                break;
            case EXT_LABELINC:
                switch (labelID) {  
                    case SLC: block->labelset.slc = labelValue; break;
                    case SEG: block->labelinc.seg = labelValue; break;
                    case REP: block->labelinc.rep = labelValue; break;
                    case AVG: block->labelinc.avg = labelValue; break;
                    case SET: block->labelinc.set = labelValue; break;
                    case ECO: block->labelinc.eco = labelValue; break;
                    case PHS: block->labelinc.phs = labelValue; break;
                    case LIN: block->labelinc.lin = labelValue; break;
                    case PAR: block->labelinc.par = labelValue; break;
                    case ACQ: block->labelinc.acq = labelValue; break;
                    default: break;
                }
                break;
            case EXT_RF_SHIM:
                rfshim = seq->rfShimLibrary[extIdx];
                block->rfShimming.type = 1;
                block->rfShimming.nChan = rfshim.nChannels;
                block->rfShimming.amplitudes = ALLOC(sizeof(float) * rfshim.nChannels);
                block->rfShimming.phases = ALLOC(sizeof(float) * rfshim.nChannels);
                for (idx = 0; idx < rfshim.nChannels; idx++) {
                    block->rfShimming.amplitudes[idx] = rfshim.values[2 * idx];
                    block->rfShimming.phases[idx] = rfshim.values[2 * idx + 1];
                }
                break;
            case EXT_DELAY:
                delay = seq->softDelayLibrary[extIdx];
                block->delay.type = 1;
                block->delay.numID = delay[0];
                block->delay.offset = delay[1];
                block->delay.factor = delay[2];
                block->delay.hintID = delay[3];
                break;
            default:
                break;
        }
    }
}