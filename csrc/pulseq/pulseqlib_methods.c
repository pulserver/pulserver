#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "pulseqlib_methods.h"

#define SHAPE_LIBRARY_MAGIC 0x12345678
#define SHAPE_FILE_BUFFER_SIZE 16384

/* C89-compliant 4-byte swap helper */
static void swap4(void* v) {
    char* b = (char*)v;
    char t;
    t = b[0]; b[0] = b[3]; b[3] = t;
    t = b[1]; b[1] = b[2]; b[2] = t;
}

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

void readShapesLibrary(pulseqlib_SeqFile* seq, FILE* f) {
    char line[MAX_LINE_LENGTH];
    int idx;
    char* p;
    float val;
    FILE* binFile;
    int maxIndex = -1;
    int i;
    int magic;
    int numShapesHeader;
    
    if (seq->isShapesLibraryParsed) return;
    if (seq->offsets.shapes < 0) {
        /* No shapes section, nothing to do */
        seq->isShapesLibraryParsed = 1;
        return;
    }

    /* Try to open existing binary file */
    binFile = fopen(seq->shapelibPath, "rb");
    if (!binFile) {
        /* Create binary file from text */
        binFile = fopen(seq->shapelibPath, "wb");
        if (!binFile) {
            fprintf(stderr, "Error: Could not create shape library file %s\n", seq->shapelibPath);
            return;
        }

        /* Write Header */
        magic = SHAPE_LIBRARY_MAGIC;
        numShapesHeader = 0;
        fwrite(&magic, sizeof(int), 1, binFile);
        fwrite(&numShapesHeader, sizeof(int), 1, binFile);

        /* Reset to start of shapes in text file */
        if (fseek(f, seq->offsets.shapes, SEEK_SET) != 0) { fclose(binFile); return; }
        if (!fgets(line, sizeof(line), f)) { fclose(binFile); return; } /* Skip header */

        int currentID = -1;
        int numUncompressed = 0;
        int count = 0;
        int capacity = 1024;
        float* buffer = (float*)ALLOC(sizeof(float) * capacity);

        while (fgets(line, sizeof(line), f)) {
            p = line;
            while (*p == ' ' || *p == '\t') p++;
            if (*p == '[' || *p == 'e') break;
            if (*p == '\0' || *p == '#') continue;

            if (strncmp(p, "shape_id", 8) == 0) {
                /* Flush previous shape */
                if (currentID != -1) {
                    pulseqlib_ShapeArbitrary compressedShape;
                    pulseqlib_ShapeArbitrary uncompressedShape;
                    
                    compressedShape.samples = buffer;
                    compressedShape.numSamples = count;
                    compressedShape.numUncompressedSamples = numUncompressed;
                    
                    if (decompressShape(&compressedShape, &uncompressedShape)) {
                        fwrite(&uncompressedShape.numSamples, sizeof(int), 1, binFile);
                        fwrite(&uncompressedShape.numSamples, sizeof(int), 1, binFile); /* Stored count is now uncompressed count */
                        if (uncompressedShape.numSamples > 0) 
                            fwrite(uncompressedShape.samples, sizeof(float), uncompressedShape.numSamples, binFile);
                        
                        if (uncompressedShape.samples) FREE(uncompressedShape.samples);
                    } else {
                        /* Error or empty shape */
                        int zero = 0;
                        fwrite(&zero, sizeof(int), 1, binFile);
                        fwrite(&zero, sizeof(int), 1, binFile);
                    }
                    numShapesHeader++;
                }
                
                /* Start new shape */
                if (sscanf(p + 8, "%d", &currentID) == 1) {
                    /* Valid ID */
                }
                count = 0;
                numUncompressed = 0;
            }
            else if (strncmp(p, "num_samples", 11) == 0) {
                sscanf(p + 11, "%d", &numUncompressed);
            }
            else {
                /* Sample value */
                if (sscanf(p, "%f", &val) == 1) {
                    if (count >= capacity) {
                        capacity *= 2;
                        float* newBuf = (float*)ALLOC(sizeof(float) * capacity);
                        if (newBuf) {
                            memcpy(newBuf, buffer, sizeof(float) * count);
                            FREE(buffer);
                            buffer = newBuf;
                        }
                    }
                    if (buffer) buffer[count++] = val;
                }
            }
        }
        /* Flush last shape */
        if (currentID != -1) {
            pulseqlib_ShapeArbitrary compressedShape;
            pulseqlib_ShapeArbitrary uncompressedShape;
            
            compressedShape.samples = buffer;
            compressedShape.numSamples = count;
            compressedShape.numUncompressedSamples = numUncompressed;
            
            if (decompressShape(&compressedShape, &uncompressedShape)) {
                fwrite(&uncompressedShape.numSamples, sizeof(int), 1, binFile);
                fwrite(&uncompressedShape.numSamples, sizeof(int), 1, binFile);
                if (uncompressedShape.numSamples > 0) 
                    fwrite(uncompressedShape.samples, sizeof(float), uncompressedShape.numSamples, binFile);
                
                if (uncompressedShape.samples) FREE(uncompressedShape.samples);
            } else {
                int zero = 0;
                fwrite(&zero, sizeof(int), 1, binFile);
                fwrite(&zero, sizeof(int), 1, binFile);
            }
            numShapesHeader++;
        }
        FREE(buffer);
        
        /* Update header with actual count */
        fseek(binFile, sizeof(int), SEEK_SET);
        fwrite(&numShapesHeader, sizeof(int), 1, binFile);
        
        fclose(binFile);
        
        /* Re-open for reading */
        binFile = fopen(seq->shapelibPath, "rb");
    }

    if (!binFile) return;

    /* Read Header */
    if (fread(&magic, sizeof(int), 1, binFile) != 1) {
        fclose(binFile);
        return;
    }
    
    /* Check magic number */
    if (magic != SHAPE_LIBRARY_MAGIC) {
        /* Try byte swapping */
        swap4(&magic);
        if (magic == SHAPE_LIBRARY_MAGIC) {
            seq->shapesLibrary.byteSwap = 1;
        } else {
            fprintf(stderr, "Error: Invalid shape library magic number\n");
            fclose(binFile);
            return;
        }
    }
    
    if (fread(&numShapesHeader, sizeof(int), 1, binFile) != 1) {
        fclose(binFile);
        return;
    }
    if (seq->shapesLibrary.byteSwap) swap4(&numShapesHeader);
    
    seq->shapesLibrary.shapesLibrarySize = numShapesHeader;
    seq->shapesLibrary.shapeOffsets = (int*)ALLOC(sizeof(int) * numShapesHeader);
    seq->shapesLibrary.numSamples = (int*)ALLOC(sizeof(int) * numShapesHeader);
    
    /* Scan file to fill index */
    long offset = ftell(binFile);
    int nu, ns;
    
    for (i = 0; i < numShapesHeader; i++) {
        seq->shapesLibrary.shapeOffsets[i] = offset;
        
        /* Read numUncompressed and numSamples to advance offset */
        if (fread(&nu, sizeof(int), 1, binFile) != 1) break;
        if (fread(&ns, sizeof(int), 1, binFile) != 1) break;
        
        if (seq->shapesLibrary.byteSwap) {
            swap4(&nu);
            swap4(&ns);
        }
        
        seq->shapesLibrary.numSamples[i] = ns;
        
        /* Advance by samples size (float = 4 bytes) */
        offset += 2 * sizeof(int) + ns * sizeof(float);
        if (fseek(binFile, offset, SEEK_SET) != 0) break;
    }

    /* Close the file to support lazy loading / resource management */
    fclose(binFile);
    seq->shapesLibrary.file = NULL;
    seq->shapesLibrary.open = 0;
    seq->isShapesLibraryParsed = 1;
}


static int loadShape(pulseqlib_SeqFile* seq, int index, pulseqlib_ShapeArbitrary* shape) {
    int j;
    
    /* Lazy load: Open file if not open */
    if (!seq->shapesLibrary.open) {
        seq->shapesLibrary.file = fopen(seq->shapelibPath, "rb");
        if (!seq->shapesLibrary.file) return 0;
        seq->shapesLibrary.open = 1;
    }

    if (index < 0 || index >= seq->shapesLibrary.shapesLibrarySize) return 0;
    
    /* Get size from RAM lookup */
    shape->numSamples = seq->shapesLibrary.numSamples[index];
    shape->numUncompressedSamples = shape->numSamples;

    /* If no samples, we are done */
    if (shape->numSamples <= 0) {
        shape->samples = NULL;
        return 1;
    }

    /* Allocate memory for the waveform */
    shape->samples = (float*)ALLOC(sizeof(float) * shape->numSamples);
    if (!shape->samples) return 0;

    long offset = seq->shapesLibrary.shapeOffsets[index];
    /* Skip header (2 ints: numUncompressed, numSamples) */
    if (fseek(seq->shapesLibrary.file, offset + 2 * sizeof(int), SEEK_SET) != 0) {
        FREE(shape->samples);
        shape->samples = NULL;
        return 0;
    }
    
    if (fread(shape->samples, sizeof(float), shape->numSamples, seq->shapesLibrary.file) != (size_t)shape->numSamples) {
        FREE(shape->samples);
        shape->samples = NULL;
        return 0;
    }
    
    if (seq->shapesLibrary.byteSwap) {
        for (j = 0; j < shape->numSamples; j++) {
            swap4(&shape->samples[j]);
        }
    }
    return 1;
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

    seq->versionMajor = 0;
    seq->versionMinor = 0;
    seq->versionRevision = 0;
    seq->versionCombined = 0;
    seq->isVersionParsed = 0;

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
    seq->isShapesLibraryParsed = 0;
    seq->shapesLibrary.open = 0;
    seq->shapesLibrary.file = NULL;
    seq->shapesLibrary.ioBuffer = NULL;
    seq->shapesLibrary.shapeOffsets = NULL;
    seq->shapesLibrary.numSamples = NULL;
    seq->shapesLibrary.shapesLibrarySize = 0;
    seq->shapesLibrary.byteSwap = 0;
}


/**
 * @brief Initialize SeqFile fields.
 * 
 * @param[in] filePath The path of .seq file on disk.  
 * @param[in, out] seq The uninitialized SeqFile structure.
 */
void pulseqlib_seqFileInit(const char* filePath, pulseqlib_SeqFile* seq) {
    char* ext;
    seqFileInit(seq);

    /* Allocate and copy the file path */
    if (filePath) {
        seq->filePath = (char*) ALLOC(strlen(filePath) + 1);
        strcpy(seq->filePath, filePath);

        /* Create path for shape library */
        seq->shapelibPath = (char*)ALLOC(strlen(filePath) + 8); /* .seq -> .shapes + null */
        strcpy(seq->shapelibPath, filePath);
        ext = strrchr(seq->shapelibPath, '.');
        if (ext && strcmp(ext, ".seq") == 0) {
            strcpy(ext, ".shapes");
        } else {
            strcat(seq->shapelibPath, ".shapes");
        }
    } else {
        seq->filePath = NULL;
        seq->shapelibPath = NULL;
    }
}


void pulseqlib_seqFileFree(pulseqlib_SeqFile* seq) {
    if (!seq) return;
    if (seq->filePath) FREE(seq->filePath);
    if (seq->shapelibPath) FREE(seq->shapelibPath);
    
    /* Free libraries */
    if (seq->definitionsLibrary) {
        int i, j;
        for (i = 0; i < seq->numDefinitions; i++) {
            for (j = 0; j < seq->definitionsLibrary[i].valueSize; j++) {
                FREE(seq->definitionsLibrary[i].value[j]);
            }
            FREE(seq->definitionsLibrary[i].value);
        }
        FREE(seq->definitionsLibrary);
    }
    
    if (seq->blockLibrary) FREE(seq->blockLibrary);
    if (seq->rfLibrary) FREE(seq->rfLibrary);
    if (seq->gradLibrary) FREE(seq->gradLibrary);
    if (seq->adcLibrary) FREE(seq->adcLibrary);
    if (seq->extensionsLibrary) FREE(seq->extensionsLibrary);
    if (seq->triggerLibrary) FREE(seq->triggerLibrary);
    if (seq->rotationQuaternionLibrary) FREE(seq->rotationQuaternionLibrary);
    if (seq->rotationMatrixLibrary) FREE(seq->rotationMatrixLibrary);
    if (seq->labelsetLibrary) FREE(seq->labelsetLibrary);
    if (seq->labelincLibrary) FREE(seq->labelincLibrary);
    if (seq->softDelayLibrary) FREE(seq->softDelayLibrary);
    if (seq->rfShimLibrary) FREE(seq->rfShimLibrary);
    if (seq->extensionLUT) FREE(seq->extensionLUT);
    if (seq->blockIDs) FREE(seq->blockIDs);

    /* Shapes Library */
    if (seq->shapesLibrary.open && seq->shapesLibrary.file) {
        fclose(seq->shapesLibrary.file);
        seq->shapesLibrary.open = 0;
    }
    if (seq->shapesLibrary.ioBuffer) FREE(seq->shapesLibrary.ioBuffer); /* Free the I/O buffer */
    if (seq->shapesLibrary.shapeOffsets) FREE(seq->shapesLibrary.shapeOffsets);
    if (seq->shapesLibrary.numSamples) FREE(seq->shapesLibrary.numSamples);
}

void pulseqlib_seqBlockInit(pulseqlib_SeqBlock* block) {
    if (block) {
        memset(block, 0, sizeof(pulseqlib_SeqBlock));
    }
}

void pulseqlib_seqBlockFree(pulseqlib_SeqBlock* block) {
    if (!block) return;
    if (block->rf.magShape.samples) FREE(block->rf.magShape.samples);
    if (block->rf.phaseShape.samples) FREE(block->rf.phaseShape.samples);
    if (block->rf.timeShape.samples) FREE(block->rf.timeShape.samples);
    if (block->gx.waveShape.samples) FREE(block->gx.waveShape.samples);
    if (block->gx.timeShape.samples) FREE(block->gx.timeShape.samples);
    if (block->gy.waveShape.samples) FREE(block->gy.waveShape.samples);
    if (block->gy.timeShape.samples) FREE(block->gy.timeShape.samples);
    if (block->gz.waveShape.samples) FREE(block->gz.waveShape.samples);
    if (block->gz.timeShape.samples) FREE(block->gz.timeShape.samples);
    if (block->adc.phaseModulationShape.samples) FREE(block->adc.phaseModulationShape.samples);
    if (block->rfShimming.amplitudes) FREE(block->rfShimming.amplitudes);
    if (block->rfShimming.phases) FREE(block->rfShimming.phases);
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
 * @param[in] parseExtensions Flag indicating whether to parse extensions.
 * @param[in, out] block Pointer to the block's content IDs and extension data.
 */
void pulseqlib_getRawBlockContentIDs(const pulseqlib_SeqFile* seq, const int blockIndex, const int parseExtensions, pulseqlib_RawBlock* block) {
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
    if (parseExtensions && extID > 0 && seq->isExtensionsLibraryParsed) {
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
 * @param[in] parseExtensions Flag indicating whether to parse extensions.
 * @param[in, out] block Pointer to a pre-allocated SeqBlock to fill.
 */
void pulseqlib_getBlock(pulseqlib_SeqFile* seq, const int blockIndex, const int parseExtensions, pulseqlib_SeqBlock* block) {
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
    
    /* Check inputs */
    if (!seq || !block || blockIndex < 0 || blockIndex >= seq->numBlocks) {
        return; /* Invalid inputs */
    }
    
    /* Ensure file is open */
    if (!seq->shapesLibrary.open) {
        seq->shapesLibrary.file = fopen(seq->shapelibPath, "rb");
        if (seq->shapesLibrary.file) {
            seq->shapesLibrary.open = 1;
            
            /* Allocate and assign custom buffer to mimic legacy real-time behavior */
            seq->shapesLibrary.ioBuffer = (char*)ALLOC(SHAPE_FILE_BUFFER_SIZE);
            if (seq->shapesLibrary.ioBuffer) {
                setvbuf(seq->shapesLibrary.file, seq->shapesLibrary.ioBuffer, _IOFBF, SHAPE_FILE_BUFFER_SIZE);
            }
        } else {
            fprintf(stderr, "Error: Could not open shape library file %s\n", seq->shapelibPath);
            return;
        }
    }
    
    pulseqlib_getRawBlockContentIDs(seq, blockIndex, parseExtensions, &rawBlock);

    /* Set the duration */
    block->duration = rawBlock.block_duration;

    /* ------------------ RF Event ------------------ */
    if (rawBlock.rf >= 0) {
        farray = seq->rfLibrary[rawBlock.rf];
        block->rf.type = 1;
        block->rf.amplitude = farray[0];

        idx = (int)farray[1];
        if (idx > 0) {
            if (!loadShape(seq, idx - 1, &block->rf.magShape)) return;
        }

        idx = (int)farray[2];
        if (idx > 0) {
            if (!loadShape(seq, idx - 1, &block->rf.phaseShape)) return;
            for (i = 0; i < block->rf.phaseShape.numSamples; i++) {
                block->rf.phaseShape.samples[i] *= TWO_PI; /* Rescale phase shape to radians */
            }
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
            if (!loadShape(seq, idx - 1, &block->rf.timeShape)) return;
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
                if (!loadShape(seq, idx - 1, &block->gx.waveShape)) return;
            }

            idx = (int)farray[5];
            if (idx > 0) {
                if (!loadShape(seq, idx - 1, &block->gx.timeShape)) return;
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
                if (!loadShape(seq, idx - 1, &block->gy.waveShape)) return;
            }

            idx = (int)farray[5];
            if (idx > 0) {
                if (!loadShape(seq, idx - 1, &block->gy.timeShape)) return;
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
                if (!loadShape(seq, idx - 1, &block->gz.waveShape)) return;
            }

            idx = (int)farray[5];
            if (idx > 0) {
                if (!loadShape(seq, idx - 1, &block->gz.timeShape)) return;
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
            if (!loadShape(seq, idx - 1, &block->adc.phaseModulationShape)) return;
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
                block->rfShimming.amplitudes = (float*)ALLOC(sizeof(float) * rfshim.nChannels);
                block->rfShimming.phases = (float*)ALLOC(sizeof(float) * rfshim.nChannels);
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