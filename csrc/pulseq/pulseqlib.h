#ifndef PULSEQLIB_H
#define PULSEQLIB_H

#define TWO_PI 6.283185307179586476925286766558
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define DEFINITION_NAME_LENGTH 32
#define EXT_NAME_LENGTH 32
#define LABEL_NAME_LENGTH 32
#define SEQUENCE_NAME_LENGTH 256
#define SOFT_DELAY_HINT_LENGTH 32
#define MAX_EXTENSIONS_PER_BLOCK 64
#define MAX_LINE_LENGTH 256
#define MAX_SCALE_SIZE 16
#define MAX_RF_SHIM_CHANNELS 64

/********************************************************************      Gradient types     *******************************************************************************************/
#define TRAP 1
#define GRAD 2

/*********************************************************************      Extensions     *******************************************************************************************/
#define EXT_LIST      0
#define EXT_TRIGGER   1
#define EXT_ROTATION  2
#define EXT_LABELSET  3
#define EXT_LABELINC  4
#define EXT_RF_SHIM   5
#define EXT_DELAY     6
#define EXT_UNKNOWN   7 /* marks the end of the enum, should always be the last */

/*****************************************************************      Trigger types and Channels    *******************************************************************************************/
#define TRIGGER_TYPE_OUTPUT 1
#define TRIGGER_TYPE_INPUT  2

#define TRIGGER_CHANNEL_INPUT_PHYSIO_1 1
#define TRIGGER_CHANNEL_INPUT_PHYSIO_2 2
#define TRIGGER_CHANNEL_OUTPUT_OSC_0 1
#define TRIGGER_CHANNEL_OUTPUT_OSC_1 2
#define TRIGGER_CHANNEL_OUTPUT_EXT_1 3

/*********************************************************************      Time Hints     ******************************************************************************************/
#define HINT_TE 1
#define HINT_TR 2
#define HINT_TI 3
#define HINT_ESP 4
#define HINT_RECTIME 5
#define HINT_T2PREP 6
#define HINT_TE2 7

/*********************************************************************      Labels and Flags     ******************************************************************************************/
/*      Label        |     Type     | Data Mapping | Description                                                                                                                           */                    
#define SLC 1     /* | counter      |      Yes     | Slice counter (or slab counter for 3D multi-slab sequences) */
#define SEG 2     /* | counter      |      Yes     | Segment counter e.g. for segmented FLASH or EPI */
#define REP 3     /* | counter      |      Yes     | Repetition counter */
#define AVG 4     /* | counter      |      Yes     | Averaging counter */
#define SET 5     /* | counter      |      Yes     | Flexible counter without firm assignment */
#define ECO 6     /* | counter      |      Yes     | Echo counter in multi-echo sequences */
#define PHS 7     /* | counter      |      Yes     | Cardiac phase counter */
#define LIN 8     /* | counter      |      Yes     | Line counter in 2D and 3D acquisitions */
#define PAR 9     /* | counter      |      Yes     | Partition counter; it counts phase encoding steps in the 2nd (through-slab) phase encoding direction in 3D sequences */
#define ACQ 10    /* | counter      |      Yes     | Spectroscopic acquisition counter */
#define NAV 11    /* | flag         |      Yes     | Navigator data flag */
#define REV 12    /* | flag         |      Yes     | Flag indicating that the readout direction is reversed */
#define SMS 13    /* | flag         |      Yes     | Simultaneous multi-slice (SMS) acquisition */
#define REF 14    /* | flag         |      Yes     | Parallel imaging flag indicating reference / auto-calibration data */
#define IMA 15    /* | flag         |      Yes     | Parallel imaging flag indicating imaging data within the ACS region */
#define NOISE 16  /* | flag         |      Yes     | Flag for the noise adjust scan e.g for the parallel imaging acceleration */
#define PMC 17    /* | flag         |      No      | Flag for the MoCo/PMC Pulseq version marking blocks that can/should be prospectively corrected for motion */
#define NOROT 18  /* | flag         |      No      | Instructs the interpreter to ignore the rotation of the FOV specified on the UI for the given block(s) */
#define NOPOS 19  /* | flag         |      No      | Instructs the interpreter to ignore the the FOV offset specified on the UI for the given block(s) */
#define NOSCL 20  /* | flag         |      No      | Instructs the interpreter to ignore the scaling of the FOV specified on the UI for the given block(s) */
#define ONCE 21   /* | 3-state flag |      No      | A 3-state flag that instructs the interpreter to alter the sequence when executing multiple repeats as follows: blocks with ONCE==0 are executed on every repetition; ONCE==1: only on the first repetition; ONCE==2: only on the last repetition */
#define TRID 22   /* | 3-state flag |      No      | If set to 1, marks the limit (beginning or end) of repeatable module in the sequence. If set to 2, marks the limit of a TR segment. */ 

/*********************************************************************      Error Codes     ******************************************************************************************/
/* Success */
#define PULSEQLIB_OK                          1

/* Generic errors (-1 to -99) */
#define PULSEQLIB_ERR_NULL_POINTER           -1   /**< Required pointer argument is NULL */
#define PULSEQLIB_ERR_INVALID_ARGUMENT       -2   /**< Invalid argument value */
#define PULSEQLIB_ERR_ALLOC_FAILED           -3   /**< Memory allocation failed */

/* Unique block errors (-50 to -59) */
#define PULSEQLIB_ERR_INVALID_PREP_POSITION      -50  /**< Invalid preparation block position */
#define PULSEQLIB_ERR_INVALID_COOLDOWN_POSITION  -51  /**< Invalid cooldown block position */

/* TR detection errors (-100 to -199) */
#define PULSEQLIB_ERR_TR_NO_BLOCKS          -100  /**< Sequence has no blocks */
#define PULSEQLIB_ERR_TR_NO_IMAGING_REGION  -101  /**< No imaging region (prep+cooldown >= numBlocks) */
#define PULSEQLIB_ERR_TR_NO_PERIODIC_PATTERN -102 /**< No periodic pattern found in imaging region */
#define PULSEQLIB_ERR_TR_PATTERN_MISMATCH   -103  /**< Periodic pattern does not repeat consistently */
#define PULSEQLIB_ERR_TR_PREP_TOO_LONG      -104  /**< Non-degenerate prep section exceeds threshold */
#define PULSEQLIB_ERR_TR_COOLDOWN_TOO_LONG  -105  /**< Non-degenerate cooldown section exceeds threshold */

/* Segmentation errors (-200 to -299) */
#define PULSEQLIB_ERR_SEG_NONZERO_START_GRAD -200 /**< First block does not start with zero gradient */
#define PULSEQLIB_ERR_SEG_NONZERO_END_GRAD   -201 /**< Last block does not end with zero gradient */
#define PULSEQLIB_ERR_SEG_NO_SEGMENTS_FOUND  -202 /**< No segment boundaries could be identified */

/* Parsing/file errors (-10 to -19) */
#define PULSEQLIB_ERR_FILE_NOT_FOUND        -10  /**< File could not be opened */
#define PULSEQLIB_ERR_FILE_READ_FAILED      -11  /**< Error reading from file */
#define PULSEQLIB_ERR_UNSUPPORTED_VERSION   -12  /**< Sequence file version not supported */
#define PULSEQLIB_ERR_PARSE_FAILED          -13  /**< Failed to parse sequence data */

/* Code checking */
#define PULSEQLIB_SUCCEEDED(code) ((code) > 0)
#define PULSEQLIB_FAILED(code)    ((code) < 0)

typedef struct pulseqlib_Diagnostic {
    int code;                      /**< Error code (PULSEQLIB_OK or PULSEQLIB_ERR_*) */
    
    /* Location info (where the error occurred) */
    int blockIndex;                /**< Block index where error was detected (-1 if N/A) */
    int channel;                   /**< Gradient channel (0=Gx, 1=Gy, 2=Gz, -1 if N/A) */
    
    /* Pattern detection info */
    int numUniqueBlocks;           /**< Number of unique block definitions found */
    int imagingRegionLength;       /**< Length of imaging region in blocks */
    int candidatePatternLength;    /**< Best candidate pattern length found (0 if none) */
    int mismatchPosition;          /**< Position where pattern mismatch occurred (-1 if N/A) */
    
    /* Gradient info (for segmentation errors) */
    float gradientAmplitude;       /**< Gradient amplitude at error location (Hz/m) */
    float maxAllowedAmplitude;     /**< Maximum allowed amplitude for "zero" (Hz/m) */
    
} pulseqlib_Diagnostic;

#define PULSEQLIB_DIAGNOSTIC_INIT { \
    PULSEQLIB_OK, -1, -1, 0, 0, 0, -1, 0.0f, 0.0f \
}

/********************************************************* Shapes  ******************************************************/
typedef struct pulseqlib_ShapeArbitrary {
    int numUncompressedSamples; /**< @brief Number of uncompressed waveform samples */
    int numSamples; /**< @brief Number of waveform samples */
    float *samples; /**< @brief Waveform samples */
} pulseqlib_ShapeArbitrary; /* mirrors Pulseq CompressedShape */

#define PULSEQLIB_SHAPE_ARBITRARY_INIT {0, 0, NULL}

typedef struct pulseqlib_ShapeTrap {
    long riseTime; /**< @brief Ramp up time of trapezoid (us)  */
    long flatTime; /**< @brief Flat-top time of trapezoid (us)  */
    long fallTime; /**< @brief Ramp down time of trapezoid (us) */
} pulseqlib_ShapeTrap; /* no Pulseq equivalent */

#define PULSEQLIB_SHAPE_TRAP_INIT {0, 0, 0}

/********************************************************* Events  ******************************************************/
typedef struct pulseqlib_RFEvent {
    short type; /**< @brief NULL or ARBITRARY */    
    float amplitude; /**< @brief Peak magnitude of magShape (Hz) */
    pulseqlib_ShapeArbitrary magShape;   /**< @brief Arbitrary waveform, unitary peak amplitude */
    pulseqlib_ShapeArbitrary phaseShape; /**< @brief Abitrary waveform */
    pulseqlib_ShapeArbitrary timeShape;  /**< @brief Arbitrary waveform */
    float center; /**< @brief Effective RF center of the pulse shape measured from the start of the shape (us) */
    float freqPPM; /**< @brief B0-dependent frequency offset of transmitter (ppm) */
    float phasePPM; /**< @brief B0-dependent phase offset of transmitter (rad/MHz) */
    float freqOffset; /**< @brief Frequency offset of transmitter (Hz) */
	float phaseOffset; /**< @brief Phase offset of transmitter (rad) */
    int delay; /**< @brief Delay prior to the pulse (us) */
    char use;  /**< @brief Single character indicating the intended use of the pulse, e.g. e,r,etc... */
} pulseqlib_RFEvent; /* mirrors Pulseq RFEvent */


#define PULSEQLIB_RF_EVENT_INIT {0, 0.0f, PULSEQLIB_SHAPE_ARBITRARY_INIT, PULSEQLIB_SHAPE_ARBITRARY_INIT, PULSEQLIB_SHAPE_ARBITRARY_INIT, 0.0f, 0.0f, 0.0f, 0.0f, 0, '\0'}

typedef struct pulseqlib_GradEvent {
    short type; /**< @brief NULL, TRAP, or ARBITRARY */  
    float amplitude; /**< @brief Peak amplitude of the gradient (Hz/m) */
    int delay; /**< @brief Delay prior to the gradient (us) */
    pulseqlib_ShapeTrap trap; /**< @brief Trapezoid, unitary plateau amplitude */
    pulseqlib_ShapeArbitrary waveShape; /**< @brief Arbitrary waveform, unitary peak amplitude */
    pulseqlib_ShapeArbitrary timeShape; /**< @brief Arbitrary waveform */
    float first; /**< @brief Amplitude at the start of the shape for arbitrary gradient */
    float last; /**< @brief Amplitude at the end of the shape for arbitrary gradient */
} pulseqlib_GradEvent; /* mirrors Pulseq GradEvent */

#define PULSEQLIB_GRAD_EVENT_INIT {0, 0.0f, 0, PULSEQLIB_SHAPE_TRAP_INIT, PULSEQLIB_SHAPE_ARBITRARY_INIT, PULSEQLIB_SHAPE_ARBITRARY_INIT, 0.0f, 0.0f}

typedef struct pulseqlib_ADCEvent {
    short type; /**< @brief NULL or ADC */
    int numSamples; /**< @brief Number of ADC samples */
    int dwellTime; /**< @brief Dwell time of ADC readout (ns) */
    int delay; /**< @brief Delay before first sample (us) */
    float freqPPM; /**< @brief B0-dependent frequency offset of receiver (ppm) */
    float phasePPM; /**< @brief B0-dependent phase offset of receiver (rad/MHz) */
    float freqOffset; /**< @brief Frequency offset of receiver (Hz) */
	float phaseOffset; /**< @brief Phase offset of receiver (rad) */
    pulseqlib_ShapeArbitrary phaseModulationShape; /**< @brief Phase modulation shape of receiver (rad) */
} pulseqlib_ADCEvent; /* mirrors Pulseq ADCEvent */

#define PULSEQLIB_ADC_EVENT_INIT {0, 0, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, PULSEQLIB_SHAPE_ARBITRARY_INIT}

typedef struct pulseqlib_TriggerEvent {
    short type; /**< @brief OFF or ON */
    long duration; /**< @brief Duration of trigger event (us) */
    long delay; /**< @brief Delay prior to the trigger event (us) */
    int triggerType; /**< @brief Type of trigger (system dependent). 0: undefined / unused */
    int triggerChannel; /**< @brief Channel of trigger (system dependent). 0: undefined / unused */
} pulseqlib_TriggerEvent; /* mirrors Pulseq TriggerEvent */

#define PULSEQLIB_TRIGGER_EVENT_INIT {0, 0L, 0L, 0, 0}

typedef struct pulseqlib_RotationEvent {
    short type; /**< @brief NULL or DEFINED */
    union {
        float rotQuaternion[4]; /**< @brief Gradient rotation quaternion [w, x, y, z] */
        float rotMatrix[9]; /**< @brief Gradient rotation matrix (3x3, row-major) */
    } data;
} pulseqlib_RotationEvent; /* extends Pulseq RotationEvent */

#define PULSEQLIB_ROTATION_EVENT_INIT {0, {{0.0f, 0.0f, 0.0f, 0.0f}}}

typedef struct pulseqlib_LabelOrFlagEvent {
    short type; /**< @brief NULL or DEFINED */
    int slc; /**< Slice counter */
    int seg; /**< Segment counter e.g. for segmented FLASH or EPI */
    int rep; /**< Repetition counter */
    int avg; /**< Averaging counter */
    int set; /**< Flexible counter without firm assignment */
    int eco; /**< Echo counter in multi-echo sequences */
    int phs; /**< Cardiac phase counter */
    int lin; /**< Line counter in 2D and 3D acquisitions */
    int par; /**< Partition counter; it counts phase encoding steps in the 2nd (through-slab) phase encoding direction in 3D sequences */
    int acq; /**< Spectroscopic acquisition counter */
    int nav; /**< Navigator data flag */
    int rev; /**< Flag indicating that the readout direction is reversed */
    int sms; /**< Simultaneous multi-slice (SMS) acquisition */
    int ref; /**< Parallel imaging flag indicating reference / auto-calibration data */
    int ima; /**< Parallel imaging flag indicating imaging data within the ACS region */
    int noise; /**< Flag for the noise adjust scan e.g for the parallel imaging acceleration */
    int pmc; /**< Flag for the MoCo/PMC Pulseq version marking blocks that can/should be prospectively corrected for motion */
    int norot; /**< Instructs the interpreter to ignore the rotation of the FOV specified on the UI for the given block(s) */
    int nopos; /**< Instructs the interpreter to ignore the the FOV offset specified on the UI for the given block(s) */
    int noscl; /**< Instructs the interpreter to ignore the the FOV scaling specified on the UI for the given block(s) */
    int once; /**< A 3-state flag indicating whether the label is to be used once (0), multiple times (1), or not at all (2) */
    int trid; /**< If set to 1, marks the limit (beginning or end) of repeatable module in the sequence. If set to 2, marks the limit of a TR segment. */
} pulseqlib_LabelOrFlagEvent; /* no Pulseq equivalent */



typedef struct pulseqlib_LabelEvent {
    int slc; /**< Slice counter */
    int seg; /**< Segment counter e.g. for segmented FLASH or EPI */
    int rep; /**< Repetition counter */
    int avg; /**< Averaging counter */
    int set; /**< Flexible counter without firm assignment */
    int eco; /**< Echo counter in multi-echo sequences */
    int phs; /**< Cardiac phase counter */
    int lin; /**< Line counter in 2D and 3D acquisitions */
    int par; /**< Partition counter; it counts phase encoding steps in the 2nd (through-slab) phase encoding direction in 3D sequences */
    int acq; /**< Spectroscopic acquisition counter */
} pulseqlib_LabelEvent; /* no Pulseq equivalent */


typedef struct pulseqlib_FlagEvent {
    int trid; /**< If set to 1, marks the limit (beginning or end) of repeatable module in the sequence. If set to 2, marks the limit of a TR segment. */
    int nav; /**< Navigator data flag */
    int rev; /**< Flag indicating that the readout direction is reversed */
    int sms; /**< Simultaneous multi-slice (SMS) acquisition */
    int ref; /**< Parallel imaging flag indicating reference / auto-calibration data */
    int ima; /**< Parallel imaging flag indicating imaging data within the ACS region */
    int noise; /**< Flag for the noise adjust scan e.g for the parallel imaging acceleration */
    int pmc; /**< Flag for the MoCo/PMC Pulseq version marking blocks that can/should be prospectively corrected for motion */
    int norot; /**< Instructs the interpreter to ignore the rotation of the FOV specified on the UI for the given block(s) */
    int nopos; /**< Instructs the interpreter to ignore the the FOV offset specified on the UI for the given block(s) */
    int noscl; /**< Instructs the interpreter to ignore the the FOV scaling specified on the UI for the given block(s) */
    int once; /**< A 3-state flag indicating whether the label is to be used once (0), multiple times (1), or not at all (2) */
} pulseqlib_FlagEvent; /* no Pulseq equivalent */


typedef struct pulseqlib_SoftDelayEvent {
    short type; /**< @brief NULL or DEFINED */
    int numID; /**< @brief Numeric index of the soft delay to help the intepreter (together with the hint string) to identify the delay and allocate it to the UI element */
    int offset; /**< @brief Offset (positive or negative) added to the delay after the division by the factor (us) */
    int factor; /**< @brief Factor by which the value on the user interface needs to be divided for calculating the final delay applied to the sequence */
    int hintID; /**< @brief Enum hint corresponding to this soft delay, e.g. TE, to help the interpreter to identify the delay and allocate it to the UI element */
} pulseqlib_SoftDelayEvent; /* mirrors Pulseq SoftDelayEvent */


typedef struct pulseqlib_RfShimmingEvent {
    short type; /**< @brief NULL or DEFINED */
    int nChan; /**< @brief Number of RF channels */
    float* amplitudes; /**< @brief Amplitude scaling factor for each channel */
    float* phases; /**< @brief Additional phase for each channel */
} pulseqlib_RfShimmingEvent; /* mirrors Pulseq RfShimmingEvent */


/********************************************************* Event Blocks  ******************************************************/
typedef struct pulseqlib_RawBlock {
    int block_duration; /**<@brief Block duration in us */
    int rf; /**<@brief RF event ID in RF Library */
    int gx; /**<@brief gradient event ID in GRAD Library (X channel) */
    int gy; /**<@brief gradient event ID in GRAD Library (Y channel) */
    int gz; /**<@brief gradient event ID in GRAD Library (Z channel) */
    int adc; /**<@brief ADC event ID in ADC library */
    int extCount; /***<@brief Actual number of extensions in current block (from 0 to MAX_EXTENSIONS_PER_BLOCK)  */
    int ext[MAX_EXTENSIONS_PER_BLOCK][2]; /* Tuples of extension library (labelset, labelinc, rotation etc) and ID [type, ref] */
} pulseqlib_RawBlock;


typedef struct pulseqlib_SeqBlock {
    int duration; /**< @brief Duration of the block (us) */
    pulseqlib_RFEvent rf; /**< @brief RF event */
    pulseqlib_GradEvent gx; /**< @brief Gradient event on X channel */
    pulseqlib_GradEvent gy; /**< @brief Gradient event on Y channel */
    pulseqlib_GradEvent gz; /**< @brief Gradient event on Z channel */
    pulseqlib_ADCEvent adc; /**< @brief ADC event */
    pulseqlib_TriggerEvent trigger; /**< @brief Trigger event */
    pulseqlib_RotationEvent rotation; /**< @brief Rotation event */
    pulseqlib_FlagEvent flag; /**< @brief Flag event containing flag values */
    pulseqlib_LabelEvent labelset; /**< @brief Label event containing the 'SET' label values */
    pulseqlib_LabelEvent labelinc; /**< @brief Label event containing the 'INC' label values */
    pulseqlib_SoftDelayEvent delay; /**< @brief Soft delay event */
    pulseqlib_RfShimmingEvent rfShimming; /**< @brief RF shimming event */
} pulseqlib_SeqBlock; /* Mirrors Pulseq SeqBlock */


typedef struct pulseqlib_Opts {
    float gamma; /**< @brief Gyromagnetic ratio in Hz/T */
    float B0; /**< @brief Main magnetic field strength in Tesla for frequency offset calculations */
    float max_grad; /**< @brief Maximum gradient amplitude in Hz/m */
    float max_slew; /**< @brief Maximum slew rate in Hz/m/s */
    float rf_raster_time; /**< @brief RF raster time in us */
    float grad_raster_time; /**< @brief Gradient raster time in us */
    float adc_raster_time; /**< @brief ADC raster time in us */
    float block_duration_raster; /**< @brief Block duration raster time in us */
} pulseqlib_Opts;


typedef struct pulseqlib_SectionOffsets {
    long scan_cursor;
    long version;
    long definitions;
    long blocks;
    long rf;
    long grad;
    long trap;
    long adc;
    long extensions;
    long triggers;
    long rotations;
    long labelset;
    long labelinc;
    long delays;
    long rfshim;
    long shapes;
    long signature;
} pulseqlib_SectionOffsets;


typedef struct pulseqlib_Definition {
    char name[DEFINITION_NAME_LENGTH];
    int valueSize;
    char** value;
} pulseqlib_Definition;


typedef struct pulseqlib_ReservedDefinitions {
    float gradientRasterTime; /**< GradientRasterTime in us */
    float radiofrequencyRasterTime; /**< RadiofrequencyRasterTime in us */
    float adcRasterTime; /**< AdcRasterTime in us */
    float blockDurationRaster; /**< BlockDurationRaster in us */
    char name[SEQUENCE_NAME_LENGTH]; /**< Sequence Name (optional) */
    float fov[3]; /**< FOV in cm (optional) */
    float totalDuration; /**< TotalDuration in seconds (optional) */
} pulseqlib_ReservedDefinitions;


typedef struct pulseqlib_LabelLimit {
    int min; /**< Minimum value for this label type */
    int max; /**< Maximum value for this label type */
} pulseqlib_LabelLimit;


typedef struct pulseqlib_LabelLimits {
    pulseqlib_LabelLimit slc; /**< Slice label limits */
    pulseqlib_LabelLimit phs; /**< Phase label limits */
    pulseqlib_LabelLimit rep; /**< Repetition label limits */
    pulseqlib_LabelLimit avg; /**< Average label limits */
    pulseqlib_LabelLimit seg; /**< Segment label limits */
    pulseqlib_LabelLimit set; /**< Set label limits */
    pulseqlib_LabelLimit eco; /**< Echo label limits */
    pulseqlib_LabelLimit par; /**< Partition label limits */
    pulseqlib_LabelLimit lin; /**< Line label limits */
    pulseqlib_LabelLimit acq; /**< Acquisition label limits */
} pulseqlib_LabelLimits;


typedef struct pulseqlib_GlobalLabelTable {
    int slc;
    int seg;
    int rep;
    int avg;
    int set;
    int echo;
    int phs;
    int lin;
    int par;
    int acq;
} pulseqlib_GlobalLabelTable;


#define INIT_GLOBAL_LABEL_TABLE {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}


typedef struct pulseqlib_RfShimEntry {
    int nChannels; /**< Number of channels */
    float values[2 * MAX_RF_SHIM_CHANNELS]; /**< Pointer to array of size 2 * nChannels (mag1, phase1, mag2, phase2, ...) */
} pulseqlib_RfShimEntry;


typedef struct pulseqlib_BlockLabels {
    pulseqlib_LabelEvent labelset; /**< Label set values for the block */
    pulseqlib_LabelEvent labelinc; /**< Label increment values for the block */
    pulseqlib_FlagEvent flag; /**< Flag values carried by the block */
} pulseqlib_BlockLabels;


typedef struct pulseqlib_RFDynamic {
    int present; /**< Non-zero if the block contains an RF event */
    float amplitude; /**< RF amplitude (Hz) */
    float freqOffset; /**< Static frequency offset component (Hz) */
    float freqPPM; /**< Frequency offset coefficient (ppm/T) */
    float phaseOffset; /**< Static phase offset component (rad) */
    float phasePPM; /**< Phase offset coefficient (rad/T) */
    float totalFrequency; /**< Combined frequency offset given a specific B0 (Hz) */
    float totalPhase; /**< Combined phase offset given a specific B0 (rad) */
} pulseqlib_RFDynamic;


typedef struct pulseqlib_ADCDynamic {
    int present; /**< Non-zero if the block contains an ADC event */
    float freqOffset; /**< Static frequency offset component (Hz) */
    float freqPPM; /**< Frequency offset coefficient (ppm/T) */
    float phaseOffset; /**< Static phase offset component (rad) */
    float phasePPM; /**< Phase offset coefficient (rad/T) */
    float totalFrequency; /**< Combined frequency offset given a specific B0 (Hz) */
    float totalPhase; /**< Combined phase offset given a specific B0 (rad) */
} pulseqlib_ADCDynamic;


typedef struct pulseqlib_GradDynamic {
    int present; /**< Non-zero if the block contains a gradient event */
    int type; /**< Gradient type encoded in the library (TRAP/GRAD) */
    float amplitude; /**< Gradient amplitude (Hz/m) */
    int waveShapeId; /**< Identifier of the associated waveform shape (0 if trapezoid) */
    int timeShapeId; /**< Identifier of the associated time shape (0 if none) */
} pulseqlib_GradDynamic;


typedef struct pulseqlib_RotationDynamic {
    int present; /**< Non-zero if the block carries a rotation extension */
    const float* data; /**< Pointer to the rotation coefficients in the sequence library */
    int length; /**< Number of coefficients pointed to by data (4 for quaternion, 9 for matrix) */
    int index; /**< Index of the rotation entry in the library */
} pulseqlib_RotationDynamic;


typedef struct pulseqlib_RfShimDynamic {
    int present; /**< Non-zero if the block carries an RF shimming extension */
    const pulseqlib_RfShimEntry* entry; /**< Pointer to the RF shimming entry in the sequence library */
} pulseqlib_RfShimDynamic;


typedef struct pulseqlib_BlockDynamic {
    pulseqlib_RFDynamic rf; /**< RF dynamic parameters */
    pulseqlib_GradDynamic gx; /**< Gx dynamic parameters */
    pulseqlib_GradDynamic gy; /**< Gy dynamic parameters */
    pulseqlib_GradDynamic gz; /**< Gz dynamic parameters */
    pulseqlib_ADCDynamic adc; /**< ADC dynamic parameters */
    pulseqlib_RfShimDynamic rfShim; /**< RF shimming parameters */
    pulseqlib_RotationDynamic rotation; /**< Rotation parameters */
} pulseqlib_BlockDynamic;


typedef struct pulseqlib_SeqFile {
    pulseqlib_Opts opts;
    char* filePath; /**< @brief Path to the sequence (.seq) file. */
    pulseqlib_SectionOffsets offsets; /**< @brief Line position of each section. */
    int isVersionParsed; /**< @brief Flag indicating if the version was parsed successfully. */
    int versionCombined; /**< @brief Combined version number calculated as: 1000000 * versionMajor + 1000 * versionMinor + versionRevision. */
    int versionMajor; /**< @brief Major version number. */
    int versionMinor; /**< @brief Minor version number. */
    int versionRevision; /**< @brief Revision version number. */
    int isDefinitionsLibraryParsed; /**< @brief Flag indicating if the definitions library was parsed successfully. */
    int numDefinitions; /**< @brief Number of definitions parsed. */
    pulseqlib_Definition* definitionsLibrary; /**< @brief Array of parsed definitions. */
    pulseqlib_ReservedDefinitions reservedDefinitionsLibrary; /**< Parsed reserved definitions */
    int isBlockLibraryParsed; /**< @brief Flag indicating if the block library was parsed. */
    int numBlocks; /**< @brief Number of block entries. */
    float (*blockLibrary)[7]; /**< @brief Block library data with columns: duration, rf, gx, gy, gz, adc, ext. */
    int* blockIDs; /**< @brief Mapping from original blocks to unique blocks. NULL by default. */
    int isRfLibraryParsed; /**< @brief Flag indicating if the RF library was parsed. */
    int rfLibrarySize; /**< @brief Number of RF entries. */
    float (*rfLibrary)[10]; /**< @brief RF library data with columns: amp, mag_id, phase_id, time_id, center, delay, freqPPM, phasePPM, freq, phase. */
    int isGradLibraryParsed; /**< @brief Flag indicating if the gradient library was parsed. */
    int gradLibrarySize; /**< @brief Number of gradient entries. */
    float (*gradLibrary)[7]; /**< @brief Gradient library data with columns: type, amp, rise/first, flat/last, fall/shape_id, delay/time_id, unused/delay. */
    int isAdcLibraryParsed; /**< @brief Flag indicating if the ADC library was parsed. */
    int adcLibrarySize; /**< @brief Number of ADC entries. */
    float (*adcLibrary)[8]; /**< @brief ADC library data with columns: num, dwell, delay, freqPPM, phasePPM, freq, phase, phase_id. */
    int isExtensionsLibraryParsed; /**< @brief Flag indicating if the extensions library was parsed. */
    int extensionsLibrarySize; /**< @brief Number of extension entries. */
    float (*extensionsLibrary)[3]; /**< @brief Extensions library data with columns: type, ref, next_id. */
    int triggerLibrarySize; /**< @brief Number of trigger entries. */
    float (*triggerLibrary)[4]; /**< @brief Trigger library data with columns: duration, delay, type, channel. */
    int rotationLibrarySize;/**< @brief Number of rotation entries. */
    float (*rotationQuaternionLibrary)[4]; /**< @brief Rotation quaternion data with columns: RotQuat0, RotQuatX, RotQuatY, RotQuatZ. */
    float (*rotationMatrixLibrary)[9]; /**< @brief Rotation matrix data as flattened 3x3 matrices in row-major order (9 values per entry). */
    int isLabelDefined[22]; /**< For each type of Label in constants.h, flags whether it was defined or not in the given SeqFile */
    int labelsetLibrarySize; /**< @brief Number of label set entries. */
    float (*labelsetLibrary)[2]; /**< @brief Label set data with columns: set, labelstring index. */
    int labelincLibrarySize; /**< @brief Number of label increment entries. */
    float (*labelincLibrary)[2]; /**< @brief Label increment data with columns: increment, labelstring index. */
    pulseqlib_LabelLimit labelLimits; /**< @brief Min and max values for each label type in the sequence */
    int isDelayDefined[8]; /**< @brief For each type of Delay in constants.h, flags whether it was defined or not in the given SeqFile */
    int softDelayLibrarySize; /**< @brief Number of soft delay entries. */
    float (*softDelayLibrary)[4]; /**< @brief Soft delay data with columns:  numID, offset, factor. */
    int rfShimLibrarySize; /**< @brief Number of RF shim entries. */
    pulseqlib_RfShimEntry* rfShimLibrary; /**< @brief RF shim data; per-channel magnitude and phase arrays: magn_c1, phase_c1, magn_c2, phase_c2, ... */
    int extensionMap[8]; /**< @brief Maps extension types to numeric IDs. */
    int extensionLUTSize; /**< @brief Size of extension lookup table. */
    int* extensionLUT; /**< @brief Extension lookup table. */
    int isShapesLibraryParsed; /**< @brief Flag indicating if the shapes library was parsed. */
    int shapesLibrarySize; /**< @brief Number of shape entries. */
    pulseqlib_ShapeArbitrary* shapesLibrary; /**< @brief Array of arbitrary shape definitions. */
} pulseqlib_SeqFile; /* Mirrors Pulseq SeqFile */

typedef struct pulseqlib_RfDefinition {
    int ID; /**< Unique RF ID */
    int magShapeID; /**< Magnitude shape ID */
    int phaseShapeID; /**< Phase shape ID */
    int timeShapeID; /**< Time shape ID */
    int delay; /**< Delay prior to the pulse (us) */
} pulseqlib_RfDefinition;

typedef struct pulseqlib_RfTableElement {
    int ID; /**< Unique RF ID */
    float amplitude; /**< RF amplitude (Hz) */
    float freqOffset; /**< Frequency offset (Hz) */
    float phaseOffset; /**< Phase offset (rad) */
} pulseqlib_RfTableElement;

#define PULSEQLIB_RF_TABLE_ELEMENT_INIT {0, 0.0f, 0.0f, 0.0f}

typedef struct pulseqlib_GradDefinition {
    int ID; /**< Unique Grad ID */
    int type; /**< Gradient type encoded in the library (TRAP/GRAD) */
    int riseTimeOrFirst; /**< Rise time (us) for trapezoid, or first amplitude for arbitrary */
    int flatTimeOrLast; /**< Flat time (us) for trapezoid, or last amplitude for arbitrary */
    int fallTimeOrNumUncompressedSamples; /**< Fall time (us) for trapezoid, or number of uncompressed samples for arbitrary */
    int unusedOrTimeShapeID; /**< Unused Time shape ID */
    int delay; /**< Delay prior to the pulse (us) */
} pulseqlib_GradDefinition;

typedef struct pulseqlib_GradTableElement {
    int ID;
    int shotIndex; /**< Index of the shot this gradient belongs to */
    float amplitude; /**< Gradient amplitude (Hz/m) */
} pulseqlib_GradTableElement;

#define PULSEQLIB_GRAD_TABLE_ELEMENT_INIT {0, 0, 0.0f}

typedef struct pulseqlib_AdcDefinition {
    int ID; /**< Unique ADC ID */
    int numSamples; /**< Number of ADC samples */
    int dwellTime; /**< Dwell time of ADC readout (ns) */
    int delay; /**< Delay before first sample (us) */
} pulseqlib_AdcDefinition;

#define PULSEQLIB_ADC_DEFINITION_INIT {0, 0, 0, 0}

typedef struct pulseqlib_AdcTableElement {
    int ID; /**< Unique ADC ID */
    float freqOffset; /**< Frequency offset (Hz) */
    float phaseOffset; /**< Phase offset (rad) */
} pulseqlib_AdcTableElement;

#define PULSEQLIB_ADC_TABLE_ELEMENT_INIT {0, 0.0f, 0.0f}

typedef struct pulseqlib_TRdescriptor {
    int numPrepBlocks; /**< Number of preparation blocks before the main TR */
    int numCooldownBlocks; /**< Number of cooldown blocks after the main TR */
    int trSize; /**< Size of the TR in number of blocks */
    int numTRs; /**< Number of TRs in the sequence */
    int numPrepTRs; /**< Number of preparation TR before the main TR */
    int degeneratePrep; /**< Non-zero if the preparation blocks are degenerate (i.e. identical to main TR) */
    int numCooldownTRs; /**< Number of cooldown TR after the main TR */
    int degenerateCooldown; /**< Non-zero if the cooldown blocks are degenerate (i.e. identical to main TR) */
} pulseqlib_TRdescriptor;

#define PULSEQLIB_TR_DESCRIPTOR_INIT {0, 0, 0, 0, 0, 0, 0, 0}

typedef struct pulseqlib_SequenceDescriptor {
    int numBlocks; /**< Total number of blocks in the sequence */
    int* uniqueBlockTable; /**< Pointer to array mapping block index → unique block ID */
    int* isPureDelayBlock; /**< Pointer to array indicating if block index is a pure delay block (1) or not (0) */

    int numUniqueBlocks; /**< Number of unique blocks in the sequence */
    int (*uniqueBlockDefinitions)[6]; /**< (ID, duration_us, rf=unique_RF_id, gx=unique_grad_id, gy=unique_grad_id, gz=unique_grad_id) */
    
    int numUniqueRFs; /**< Number of unique RF events in the sequence */
    pulseqlib_RfDefinition* rfDefinitions; /**< (ID, rf_mag_id, rf_phase_id, rf_time_id, delay) */
    pulseqlib_RfTableElement* rfTable; /**< Pointer to array mapping unique RF ID → RF dynamic parameters */

    int numUniqueGrads; /**< Number of unique gradient events in the sequence */
    pulseqlib_GradDefinition* gradDefinitions; /**< (ID, type, rise/first ; flat/last ; fall/numUncompressedSamples, unused/time_id, delay) */
    pulseqlib_GradTableElement* gradTable; /**< Pointer to array mapping unique Grad ID → Grad dynamic parameters */

    int numUniqueADCs; /**< Number of unique ADC events in the sequence */
    pulseqlib_AdcDefinition* adcDefinitions; /**< (ID, numSamples, dwellTime, delay) */
    pulseqlib_AdcTableElement* adcTable; /**< Pointer to array mapping unique ADC ID → ADC dynamic parameters */
    pulseqlib_TRdescriptor trDescriptor; /**< TR segmentation descriptor */

} pulseqlib_SequenceDescriptor;

#define PULSEQLIB_SEQUENCE_DESCRIPTOR_INIT {0, NULL, NULL, 0, NULL, 0, NULL, NULL, 0, NULL, NULL, PULSEQLIB_TR_DESCRIPTOR_INIT}

typedef struct pulseqlib_TRsegment {
    int startBlock; /**< Starting block index of the TR segment */
    int numBlocks; /**< Number of blocks in the TR segment */
    int* uniqueBlockIndices; /**< Pointer to array of unique block indices in the segment */
} pulseqlib_TRsegment;

typedef struct pulseqlib_SegmentTableResult {
    int numUniqueSegments;       /**< Total number of unique segment definitions */
    
    /* Prep section */
    int numPrepSegments;         /**< Number of segments in prep section */
    int* prepSegmentTable;       /**< Maps prep segment index → unique segment ID */
    
    /* Main TR section */
    int numMainSegments;         /**< Number of segments in main TR */
    int* mainSegmentTable;       /**< Maps main segment index → unique segment ID */
    
    /* Cooldown section */
    int numCooldownSegments;     /**< Number of segments in cooldown section */
    int* cooldownSegmentTable;   /**< Maps cooldown segment index → unique segment ID */
} pulseqlib_SegmentTableResult;


/*************************** Actual Pulserver Objects ******************************/
typedef struct pulseqlib_RFObject {
    int num_samples; /**< Number of RF samples */
    int complex_flag; /**< True if RF phase is present */
    int default_raster_flag; /**< True if RF uses default raster time from sequence options */
    float* magnitude; /**< Pointer to RF samples array (max(abs(magnitude)) == 1.0) */
    float* phase; /**< Pointer to RF phase samples array (present if complex_flag is non-zero) */
    int* time_us; /**< Pointer to RF time samples array in microseconds (present if default_raster_flag is zero) */
    int duration_us; /**< RF duration from start in microseconds */
    int center_us; /**< RF isocenter time from start in microseconds */
    float bandwidth; /**< RF bandwidth in Hz */
    float max_amplitude; /**< Maximum RF amplitude in Hz across instances */
    float max_flip_angle; /**< Maximum flip angle in radians across instances */
} pulseqlib_RFObject;

#define PULSEQLIB_RF_OBJECT_INIT {0, 0, 0, NULL, NULL, NULL, 0, 0, 0.0f, 0.0f}

typedef struct pulseqlib_GradientObject {
    int num_samples; /**< Number of gradient samples */
    int default_raster_flag; /**< True if gradient uses default raster time from sequence options */
    int num_waveforms; /**< Number of gradient waveforms (0 for trapezoids; 1 for single-shot; N for multi-shot) */
    float** waveform; /**< Pointer to gradient samples array (max(abs(waveform)) == 1.0) */
    int* time_us; /**< Pointer to gradient time samples array in microseconds (present if default_raster_flag is zero) */
    int duration_us; /**< Gradient duration from start in microseconds */
    float* normalized_energy; /**< Gradient normalized energy in (Hz/m) s for each waveform */
    float* normalized_slew; /**< Gradient normalized slew rate in 1 / s  for each waveform */
    float max_energy; /**< Maximum energy across all waveforms (Hz/m) s */
    float max_slew; /**< Maximum slew rate across all waveforms 1 / s */
    int max_energy_index; /**< Index of the waveform with maximum energy */
    int max_slew_index; /**< Index of the waveform with maximum slew rate */
} pulseqlib_GradientObject;

#define PULSEQLIB_GRADIENT_OBJECT_INIT {0, 0, 0, NULL, NULL, 0, NULL, NULL, 0.0f, 0.0f, -1, -1}

typedef struct pulseqlib_GradientTupleObject {
    int gx_ID; /**< Gx gradient object ID */
    int gx_delay_us; /**< Gx gradient delay in microseconds */
    int gy_ID; /**< Gy gradient object ID */
    int gy_delay_us; /**< Gy gradient delay in microseconds */
    int gz_ID; /**< Gz gradient object ID */
    int gz_delay_us; /**< Gz gradient delay in microseconds */
} pulseqlib_GradientTupleObject;

typedef struct pulseqlib_AdcObject {
    int num_samples;
    int dwell_time_us; /**< Dwell time in microseconds */
} pulseqlib_AdcObject;

#define PULSEQLIB_ADC_OBJECT_INIT {0, 0}

typedef struct pulseqlib_FrequencyModulationObject {
    int num_samples;
    int default_raster_flag; /**< True if frequency modulation uses default raster time from sequence options */
    float* frequency; /**< Pointer to frequency modulation samples array in Hz */
    int* time_us; /**< Pointer to time samples array in microseconds */
} pulseqlib_FrequencyModulationObject;

#define PULSEQLIB_FREQUENCY_MODULATION_OBJECT_INIT {0, 0, NULL, NULL}

typedef struct pulseqlib_SegmentedSequenceBlock {
    int duration_us; /**< Block duration in microseconds */

    /* RF */
    int rf_ID; /**< RF object ID */
    int rf_delay_us; /**< RF delay in microseconds wrt block start */

    /* Gradients */
    int grad_ID; /**< Gradient tuple object ID */
    int grad_delay_us; /**< Gradient tuple delay in microseconds wrt block start */

    /* ADC */
    int adc_ID; /**< ADC object ID */
    int adc_delay_us; /**< ADC delay in microseconds wrt block start */
} pulseqlib_SegmentedSequenceBlock;

#define PULSEQLIB_SEGMENTEDSEQUENCEBLOCK_INIT {0, -1, 0, -1, 0, -1}

typedef struct pulseqlib_SegmentObject {
    int duration_us; /**< Duration of the segment in microseconds */
    int startBlock; /**< Starting block index of the segment */
    int numBlocks; /**< Number of blocks in the segment */
    pulseqlib_SegmentedSequenceBlock* blocks; /**< Array of blocks in the segment */
    int* rotext; /**< Array of flags indicating presence of rotation extension for a given block */
    int* norot;  /**< Array of flags indicating whether fov rotation has to be ignored for a given block */
    int* nopos;  /**< Array of flags indicating whether fov shift has to be ignored for a given block */
} pulseqlib_SegmentObject;

#define PULSEQLIB_SEGMENTOBJECT_INIT {0, 0, 0, NULL, NULL, NULL, NULL}

typedef struct pulseqlib_SegmentedSequenceTR {
    int duration_us; /**< Duration of the TR in microseconds */
    int startBlock; /**< Starting block index of the TR */
    int numBlocks; /**< Number of blocks in the TR */
    int numSegments; /**< Number of segments in the TR */
    int* segment_IDs; /**< Array of segments in the TR */
    float rf_scaling; /**< RF scaling factor for each pulse instance in the TR */
} pulseqlib_SegmentedSequenceTR;

#define PULSEQLIB_SEGMENTEDSEQUENCETR_INIT {0, 0, 0, 0, NULL, 1.0f}

typedef struct pulseqlib_SegmentedSequenceLoop {
    int numBlocks;   /**< Number of blocks in the loop */
    int (*block)[7]; /**< Array of tuples (rf_event, gx_event, gy_event, gz_event, adc_event, rot_event, once) */
    int numRFEvents; /**< Number of RF events in the loop */
    int (*rfEvents)[2]; /**< Array of tuples (rf_freq, rf_phase) */
    int (*numGxEvents)[2]; /**< Array of tuples (gx_wave_id, gx_amplitude) */
    int (*numGyEvents)[2]; /**< Array of tuples (gy_wave_id, gy_amplitude) */
    int (*numGzEvents)[2]; /**< Array of tuples (gz_wave_id, gz_amplitude) */
    int (*numAdcEvents)[3]; /**< Array of tuples (adc_freq, adc_phase, adc_rtfeedback) */
    int (*numRotMatrices)[9]; /**< Array of rotation matrices (3x3, row-major) */
} pulseqlib_SegmentedSequenceLoop;

typedef struct pulseqlib_SegmentedSubSequence {
    pulseqlib_SegmentedSequenceTR prepTR; /**< Preparation TR */
    pulseqlib_SegmentedSequenceLoop prepLoop; /**< Preparation loop */

    pulseqlib_SegmentedSequenceTR mainTR; /**< Main TR */
    pulseqlib_SegmentedSequenceLoop mainLoop; /**< Main loop */

    pulseqlib_SegmentedSequenceTR cooldownTR; /**< Cooldown TR */
    pulseqlib_SegmentedSequenceLoop cooldownLoop; /**< Cooldown loop */
} pulseqlib_SegmentedSubSequence;

typedef struct pulseqlib_SegmentedSequence {
    pulseqlib_Opts opts; /**< Sequence options */

    int num_rf_objects; /**< Number of RF objects in the sequence */
    pulseqlib_RFObject* rf_objects; /**< Array of RF objects */

    int num_gradient_objects; /**< Number of gradient objects in the sequence */
    pulseqlib_GradientObject* gradient_objects; /**< Array of gradient objects */

    int num_adc_objects; /**< Number of ADC objects in the sequence */
    pulseqlib_AdcObject* adc_objects; /**< Array of ADC objects */

    int num_segment_objects; /**< Number of segments in the sequence */
    pulseqlib_SegmentObject* segment_object; /**< Array of segments */

    int numSubSequences; /**< Number of sub-sequences in the sequence */

} pulseqlib_SegmentedSequence;

#endif /* PULSEQLIB_H */

