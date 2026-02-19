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
#define SEQUENCE_FILENAME_LENGTH 256
#define SOFT_DELAY_HINT_LENGTH 32
#define MAX_EXTENSIONS_PER_BLOCK 64
#define MAX_LINE_LENGTH 256
#define MAX_SCALE_SIZE 16
#define MAX_RF_SHIM_CHANNELS 64

/********************************************************************      Gradient axes     *******************************************************************************************/
#define PULSEQLIB_GRAD_AXIS_X 0
#define PULSEQLIB_GRAD_AXIS_Y 1
#define PULSEQLIB_GRAD_AXIS_Z 2

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
#define PULSEQLIB_ERR_INVALID_ONCE_FLAGS         -52  /**< Invalid ONCE flag configuration in preparation/cooldown blocks */

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
#define PULSEQLIB_ERR_TOO_MANY_GRAD_SHOTS    -203 /**< Number of gradient shots exceeds MAX_GRAD_SHOTS */

/* Selective excitation errors (-300 to -399) */
#define PULSEQLIB_ERR_SELEXC_GRAD_SCALING    -300 /**< Selective excitation block has varying gradient amplitude */
#define PULSEQLIB_ERR_SELEXC_ROTATION        -301 /**< Selective excitation block has rotation extension */

/* Parsing/file errors (-10 to -19) */
#define PULSEQLIB_ERR_FILE_NOT_FOUND        -10  /**< File could not be opened */
#define PULSEQLIB_ERR_FILE_READ_FAILED      -11  /**< Error reading from file */
#define PULSEQLIB_ERR_UNSUPPORTED_VERSION   -12  /**< Sequence file version not supported */
#define PULSEQLIB_ERR_PARSE_FAILED          -13  /**< Failed to parse sequence data */

/* Acoustic resonance errors (-400 to -449) */
#define PULSEQLIB_ERR_ACOUSTIC_INVALID_WINDOW    -400  /**< Invalid window size for acoustic analysis */
#define PULSEQLIB_ERR_ACOUSTIC_INVALID_RESOLUTION -401 /**< Invalid spectral resolution */
#define PULSEQLIB_ERR_ACOUSTIC_NO_WAVEFORM       -402  /**< No waveform data for acoustic analysis */
#define PULSEQLIB_ERR_ACOUSTIC_FFT_FAILED        -403  /**< FFT computation failed */
#define PULSEQLIB_ERR_ACOUSTIC_VIOLATION         -404  /**< Acoustic resonance violation detected */

/* PNS errors (-450 to -499) */
#define PULSEQLIB_ERR_PNS_INVALID_PARAMS         -450  /**< Invalid PNS parameters */
#define PULSEQLIB_ERR_PNS_INVALID_CHRONAXIE      -451  /**< Invalid chronaxie value */
#define PULSEQLIB_ERR_PNS_INVALID_RHEOBASE       -452  /**< Invalid rheobase value (GE model) */
#define PULSEQLIB_ERR_PNS_NO_WAVEFORM            -453  /**< No waveform data for PNS analysis */
#define PULSEQLIB_ERR_PNS_FFT_FAILED             -454  /**< FFT convolution failed */
#define PULSEQLIB_ERR_PNS_THRESHOLD_EXCEEDED     -455  /**< PNS threshold exceeded (>100%) */

#define PULSEQLIB_ERR_COLLECTION_EMPTY           -500  /**< No sequences in collection */
#define PULSEQLIB_ERR_COLLECTION_CHAIN_BROKEN    -501  /**< NextSequence file not found */
#define PULSEQLIB_ERR_COLLECTION_MAX_DEPTH       -503  /**< Too many chained sequences (circular?) */

#define PULSEQLIB_ERR_NOT_IMPLEMENTED      -999  /**< Functionality not yet implemented */

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

typedef struct pulseqlib_RawExtension {
    pulseqlib_LabelEvent labelset;  /**< Label set values */
    pulseqlib_LabelEvent labelinc;  /**< Label increment values */
    pulseqlib_FlagEvent flag;       /**< Flag values */
    int rotationIndex;  /**< Index into rotation library (-1 if none) */
    int rfShimIndex;    /**< Index into RF shim library (-1 if none) */
    int triggerIndex;   /**< Index into trigger library (-1 if none) */    
    int softDelayIndex; /**< Index into soft delay library (-1 if none) */
} pulseqlib_RawExtension;

#define PULSEQLIB_RAW_EXTENSION_INIT { \
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, \
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, \
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1} \
    , -1, -1, -1, -1 \
}

typedef struct pulseqlib_ExtensionBlock {
    pulseqlib_LabelEvent labelset;  /**< Label set values */
    pulseqlib_LabelEvent labelinc;  /**< Label increment values */
    pulseqlib_FlagEvent flag;       /**< Flag values */
    pulseqlib_RotationEvent rotation; /**< Rotation quaternion or matrix */
    pulseqlib_RfShimmingEvent rfShimming; /**< RF shimming amplitudes and phases */
    pulseqlib_TriggerEvent trigger; /**< Trigger event */
    pulseqlib_SoftDelayEvent softDelay; /**< Soft delay event */
} pulseqlib_ExtensionBlock;

#define PULSEQLIB_EXTENSION_BLOCK_INIT { \
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, \
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, \
    {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}, \
    PULSEQLIB_ROTATION_EVENT_INIT, \
    PULSEQLIB_RF_SHIMMING_EVENT_INIT, \
    PULSEQLIB_TRIGGER_EVENT_INIT, \
    PULSEQLIB_SOFT_DELAY_EVENT_INIT \
}

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
    char nextSequence[SEQUENCE_FILENAME_LENGTH]; /**< Next sequence filename (empty if last) */
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

typedef struct pulseqlib_SeqFileCollection {
    int numSequences;              /**< Number of sequences in the collection */
    pulseqlib_SeqFile* sequences;  /**< Array of parsed sequence files */
    char* basePath;                /**< Base path for resolving relative filenames */
} pulseqlib_SeqFileCollection;

#define PULSEQLIB_SEQ_FILE_COLLECTION_INIT {0, NULL, NULL}

typedef struct pulseqlib_SubsequenceInfo {
    int sequenceIndex;         /**< Index in the SeqFileCollection */
    int adcIDOffset;           /**< Offset to add to local ADC IDs for global uniqueness */
    int segmentIDOffset;       /**< Offset to add to local segment IDs for global uniqueness */
    int blockIndexOffset;      /**< Offset for block indices (cumulative block count) */
} pulseqlib_SubsequenceInfo;

#define PULSEQLIB_SUBSEQUENCE_INFO_INIT {0, 0, 0, 0}

/********************************************************* Interpreter-related structs  ******************************************************/
typedef struct pulseqlib_RfDefinition {
    /* Core definition fields (used for deduplication/uniqueness matching) */
    int ID;           /**< Unique RF ID (0-based index into rfDefinitions array) */
    int magShapeID;   /**< Magnitude shape ID (1-based index into shapes library, 0 if none) */
    int phaseShapeID; /**< Phase shape ID (1-based index into shapes library, 0 if none) */
    int timeShapeID;  /**< Time shape ID (1-based index into shapes library, 0 if none) */
    int delay;        /**< Delay prior to the pulse (us) */

#if VENDOR == GEHC
    /* Statistics fields for peak-normalized waveform (max|magnitude| = 1). */
    int numSamples;     /**< Number of RF samples (count) */
    float maxAmplitude; /**< Max amplitude across instances (Hz) */
    float flipAngle;    /**< Flip angle at Max amplitude across instances(radians) */
    float area;         /**< Signed integral / duration (dimensionless) */
    float abswidth;     /**< Integral of |rf| / duration (dimensionless) */
    float effwidth;     /**< Integral of rf² / duration (dimensionless) */
    float dtycyc;       /**< Fraction of duration with |rf| > 0.2236 (dimensionless) */
    float maxpw;        /**< Widest contiguous lobe / duration (dimensionless) */
    float duration_us;  /**< RF pulse duration (us) */
    int isodelay_us;    /**< Time from peak magnitude to pulse end (us) */
    float bandwidth;    /**< RF bandwidth at 50% cutoff (Hz) */
#endif

} pulseqlib_RfDefinition;

#if VENDOR == GEHC
#define PULSEQLIB_RF_DEFINITION_INIT {0, 0, 0, 0, 0, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0, 0.0f}
#else
#define PULSEQLIB_RF_DEFINITION_INIT {0, 0, 0, 0, 0}
#endif

typedef struct pulseqlib_RfTableElement {
    int ID; /**< Unique RF ID */
    float amplitude; /**< RF amplitude (Hz) */
    float freqOffset; /**< Frequency offset (Hz) */
    float phaseOffset; /**< Phase offset (rad) */
} pulseqlib_RfTableElement;

#define PULSEQLIB_RF_TABLE_ELEMENT_INIT {0, 0.0f, 0.0f, 0.0f}

#define MAX_GRAD_SHOTS 16  /**< Maximum number of shots/interleaves per gradient definition */

typedef struct pulseqlib_GradDefinition {
    /* Core definition fields (used for deduplication/uniqueness matching) */
    int ID;                                  /**< Unique Grad ID (0-based index into gradDefinitions array) */
    int type;                                /**< Gradient type: 0=TRAP, non-zero=ARBITRARY/EXTENDED */
    int riseTimeOrUnused;                    /**< TRAP: rise time (us); ARBITRARY: unused) */
    int flatTimeOrUnused;                    /**< TRAP: flat time (us); ARBITRARY: unused) */
    int fallTimeOrNumUncompressedSamples;    /**< TRAP: fall time (us); ARBITRARY: number of uncompressed samples */
    int unusedOrTimeShapeID;                 /**< TRAP: unused; ARBITRARY: time shape ID (1-based, 0 if none) */
    int delay;                               /**< Delay prior to the gradient (us) */

    /* Multi-shot fields (populated after deduplication by compute_grad_shot_indices) */
    int numShots;                            /**< Number of distinct waveform shapes (1 for trapezoids) */
    int shotShapeIDs[MAX_GRAD_SHOTS];        /**< Shape IDs per shot (1-based into shapes library, 0 if unused) */
    
    /* Statistics fields for peak-normalized waveform (max|waveform| = 1) 
     * To get physical values, multiply by amplitude from GradTableElement:
     *   physical_slew = slewRate * amplitude  -> (Hz/m)/s
     *   physical_energy = energy * amplitude² -> (Hz/m)² * s
     */
    float maxAmplitude[MAX_GRAD_SHOTS];      /**< Max amplitude across instances (Hz/m) */
    float slewRate[MAX_GRAD_SHOTS];          /**< Max |d(waveform)/dt| (1/s) */
    float energy[MAX_GRAD_SHOTS];            /**< Integral of waveform² dt (s) */
    float firstValue[MAX_GRAD_SHOTS];        /**< First sample of normalized waveform (dimensionless, range -1 to 1) */
    float lastValue[MAX_GRAD_SHOTS];         /**< Last sample of normalized waveform (dimensionless, range -1 to 1) */
} pulseqlib_GradDefinition;

#define PULSEQLIB_GRAD_DEFINITION_INIT {0, 0, 0, 0, 0, 0, 0, 1, {0}, {0.0f}, {0.0f}, {0.0f}, {0.0f}}

typedef struct pulseqlib_GradTableElement {
    int ID;
    int shotIndex; /**< Index of the shot this gradient belongs to */
    float amplitude; /**< Gradient amplitude (Hz/m) */
} pulseqlib_GradTableElement;

#define PULSEQLIB_GRAD_TABLE_ELEMENT_INIT {0, 0, 0.0f}

typedef struct pulseqlib_AdcDefinition {
    int ID;         /**< Unique ADC ID (0-based index into adcDefinitions array) */
    int numSamples; /**< Number of ADC samples (count) */
    int dwellTime;  /**< Dwell time between samples (ns) */
    int delay;      /**< Delay before first sample (us) */
} pulseqlib_AdcDefinition;

#define PULSEQLIB_ADC_DEFINITION_INIT {0, 0, 0, 0}

typedef struct pulseqlib_AdcTableElement {
    int ID; /**< Unique ADC ID */
    float freqOffset; /**< Frequency offset (Hz) */
    float phaseOffset; /**< Phase offset (rad) */
} pulseqlib_AdcTableElement;

#define PULSEQLIB_ADC_TABLE_ELEMENT_INIT {0, 0.0f, 0.0f}

typedef struct pulseqlib_BlockDefinition {
    int ID;          /**< Unique Block ID (0-based index into blockDefinitions array) */
    int duration_us; /**< Block duration (us) */
    int rfID;        /**< RF definition ID (-1 if no RF in this block) */
    int gxID;        /**< Gradient X definition ID (-1 if no Gx in this block) */
    int gyID;        /**< Gradient Y definition ID (-1 if no Gy in this block) */
    int gzID;        /**< Gradient Z definition ID (-1 if no Gz in this block) */
} pulseqlib_BlockDefinition;

#define PULSEQLIB_BLOCK_DEFINITION_INIT {0, 0, 0, 0, 0, 0}

typedef struct pulseqlib_BlockTableElement {
    int ID; /**< Unique Block ID */
    int duration_us; /**< Positive if the block is a pure delay block (no RF, Grad, ADC, or extensions) - otherwise refers to base block duration */
    int rfID; /**< RF event ID in RF table */
    int gxID; /**< Gradient X event ID in Grad table */
    int gyID; /**< Gradient Y event ID in Grad table */
    int gzID; /**< Gradient Z event ID in Grad table */
    int adcID; /**< ADC event ID in ADC table */
    int triggerID; /**< Trigger extension ID */
    int rotationID; /**< Rotation extension ID */
    
    /* Flow control fields */
    int onceFlag; /**< 3-state flag indicating whether the block is to be used once (0), multiple times (1), or not at all (2) */
    int norotFlag; /**< Ignore FOV rotation flag */
    int noposFlag; /**< Ignore FOV position flag */
    int pmcFlag; /**< Prospective motion correction flag */
    int navFlag; /**< Navigator flag */
} pulseqlib_BlockTableElement;

#define PULSEQLIB_BLOCK_TABLE_ELEMENT_INIT {0, 0, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0}

typedef struct pulseqlib_TRdescriptor {
    int numPrepBlocks; /**< Number of preparation blocks before the main TR */
    int numCooldownBlocks; /**< Number of cooldown blocks after the main TR */
    int trSize; /**< Size of the TR in number of blocks */
    int numTRs; /**< Number of TRs in the sequence */
    int numPrepTRs; /**< Number of preparation TR before the main TR */
    int degeneratePrep; /**< Non-zero if the preparation blocks are degenerate (i.e. identical to main TR) */
    int numCooldownTRs; /**< Number of cooldown TR after the main TR */
    int degenerateCooldown; /**< Non-zero if the cooldown blocks are degenerate (i.e. identical to main TR) */
    float trDuration_us; /**< Duration of the TR in microseconds */
} pulseqlib_TRdescriptor;

#define PULSEQLIB_TR_DESCRIPTOR_INIT {0, 0, 0, 0, 0, 0, 0, 0, 0.0f}

typedef struct pulseqlib_TRsegment {
    int startBlock; /**< Starting block index of the TR segment */
    int numBlocks; /**< Number of blocks in the TR segment */
    int* uniqueBlockIndices; /**< Pointer to array of unique block indices in the segment */
    int* hasTrigger;    /**< Per-block flag: 1 if any instance has trigger, 0 otherwise */
    int* hasRotation;   /**< Per-block flag: 1 if any instance has rotation, 0 otherwise */
    int* norotFlag;     /**< Per-block flag: 1 if any instance has norot, 0 otherwise */
    int* noposFlag;     /**< Per-block flag: 1 if any instance has nopos, 0 otherwise */
    int maxEnergyStartBlock; /**< startBlock of the instance with maximum total gradient energy */
} pulseqlib_TRsegment;

#define PULSEQLIB_TR_SEGMENT_INIT {0, 0, NULL, NULL, NULL, NULL, NULL, 0}

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

#define PULSEQLIB_SEGMENT_TABLE_RESULT_INIT {0, 0, NULL, 0, NULL, 0, NULL}

typedef struct pulseqlib_SequenceDescriptor {
    int numPrepBlocks; /**< Number of preparation blocks before the main sequence */
    int numCooldownBlocks; /**< Number of cooldown blocks after the main sequence */

    /* Timing rasters (in microseconds) */
    float rfRasterTime_us; /**< RF raster time in microseconds */
    float gradRasterTime_us; /**< Gradient raster time in microseconds */
    float adcRasterTime_us; /**< ADC raster time in microseconds */
    float blockDurationRaster_us; /**< Block duration raster time in microseconds */

    int numUniqueBlocks; /**< Number of unique blocks in the sequence */
    pulseqlib_BlockDefinition* blockDefinitions; /**< (ID, duration_us, rf_id, gx_id, gy_id, gz_id) */

    int numBlocks; /**< Total number of blocks in the sequence */
    pulseqlib_BlockTableElement* blockTable; /**< Pointer to array mapping unique Block ID → Block dynamic parameters */
    
    int numUniqueRFs; /**< Number of unique RF events in the sequence */
    pulseqlib_RfDefinition* rfDefinitions; /**< (ID, rf_mag_id, rf_phase_id, rf_time_id, delay) */

    int rfTableSize; /**< Size of the RF table */
    pulseqlib_RfTableElement* rfTable; /**< Pointer to array mapping unique RF ID → RF dynamic parameters */

    int numUniqueGrads; /**< Number of unique gradient events in the sequence */
    pulseqlib_GradDefinition* gradDefinitions; /**< (ID, type, rise/first ; flat/last ; fall/numUncompressedSamples, unused/time_id, delay) */
    
    int gradTableSize; /**< Size of the Grad table */
    pulseqlib_GradTableElement* gradTable; /**< Pointer to array mapping unique Grad ID → Grad dynamic parameters */

    int numUniqueADCs; /**< Number of unique ADC events in the sequence */
    pulseqlib_AdcDefinition* adcDefinitions; /**< (ID, numSamples, dwellTime, delay) */

    int adcTableSize; /**< Size of the ADC table */
    pulseqlib_AdcTableElement* adcTable; /**< Pointer to array mapping unique ADC ID → ADC dynamic parameters */

    /* Rotation matrices (converted from quaternions) */
    int numRotations; /**< Number of rotation matrices */
    float (*rotationMatrices)[9]; /**< Rotation matrices as flattened 3x3 in row-major order */

    /* Trigger events */
    int numTriggers; /**< Number of trigger events */
    pulseqlib_TriggerEvent* triggerEvents; /**< Array of trigger events */

    /* Shape library (decompressed waveforms) */
    int numShapes; /**< Number of shapes */
    pulseqlib_ShapeArbitrary* shapes; /**< Array of decompressed shape waveforms */

    /* TR descriptor (populated by pulseqlib_findTRInSequence) */
    pulseqlib_TRdescriptor trDescriptor; /**< TR structure info */

    /* Segment descriptor (populated by pulseqlib_findSegmentsInTR) */
    int numUniqueSegments; /**< Number of unique segments */
    pulseqlib_TRsegment* segmentDefinitions; /**< Array of unique segment definitions */
    pulseqlib_SegmentTableResult segmentTable; /**< Segment table mapping */
    
} pulseqlib_SequenceDescriptor;

#define PULSEQLIB_SEQUENCE_DESCRIPTOR_INIT {0, 0, 0.0f, 0.0f, 0.0f, 0.0f, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, 0, NULL, PULSEQLIB_TR_DESCRIPTOR_INIT, 0, NULL, PULSEQLIB_SEGMENT_TABLE_RESULT_INIT}

typedef struct pulseqlib_SequenceDescriptorCollection {
    int numSubsequences;                         /**< Number of subsequences */
    pulseqlib_SequenceDescriptor* descriptors;   /**< Array of sequence descriptors */
    pulseqlib_SubsequenceInfo* subsequenceInfo;  /**< Offset info for each subsequence */
    
    /* Global counts (after applying offsets) */
    int totalUniqueSegments;    /**< Total unique segments across all subsequences */
    int totalUniqueADCs;        /**< Total unique ADC definitions across all subsequences */
    int totalBlocks;            /**< Total blocks across all subsequences */
    
    /* Combined duration */
    float totalDuration_us;     /**< Total duration of composite sequence (us) */

} pulseqlib_SequenceDescriptorCollection;

#define PULSEQLIB_SEQUENCE_DESCRIPTOR_COLLECTION_INIT {0, NULL, NULL, 0, 0, 0, 0.0f}

typedef struct pulseqlib_TRGradientWaveforms {
    int numSamplesGx;    /**< Number of samples in Gx waveform */
    int numSamplesGy;    /**< Number of samples in Gy waveform */
    int numSamplesGz;    /**< Number of samples in Gz waveform */
    float* timeGx;       /**< Time array for Gx (us) */
    float* timeGy;       /**< Time array for Gy (us) */
    float* timeGz;       /**< Time array for Gz (us) */
    float* waveformGx;   /**< Amplitude array for Gx (Hz/m) */
    float* waveformGy;   /**< Amplitude array for Gy (Hz/m) */
    float* waveformGz;   /**< Amplitude array for Gz (Hz/m) */
} pulseqlib_TRGradientWaveforms;

#define PULSEQLIB_TR_GRADIENT_WAVEFORMS_INIT {0, 0, 0, NULL, NULL, NULL, NULL, NULL, NULL}

typedef struct pulseqlib_TRAcousticSpectra {
    int numWindows;        /**< Number of windows (1 if combined=true) */
    int numFreqBins;       /**< Number of frequency bins per spectrum (sliding window) */
    int combined;          /**< If true, spectra are pointwise max across windows (1D arrays) */
    float freqResolution;  /**< Frequency resolution in Hz (sliding window) */
    float* frequencies;    /**< Frequency axis in Hz (size: numFreqBins), for sliding window spectra */
    float* spectraGx;      /**< Gx spectra: if combined, (numFreqBins,); else (numWindows, numFreqBins) row-major */
    float* spectraGy;      /**< Gy spectra: if combined, (numFreqBins,); else (numWindows, numFreqBins) row-major */
    float* spectraGz;      /**< Gz spectra: if combined, (numFreqBins,); else (numWindows, numFreqBins) row-major */

     /* Maximum envelope values for sliding window analysis */
    float* maxEnvelopeGx;  /**< Max envelope per window for Gx: if combined, scalar (1,); else (numWindows,) */
    float* maxEnvelopeGy;  /**< Max envelope per window for Gy: if combined, scalar (1,); else (numWindows,) */
    float* maxEnvelopeGz;  /**< Max envelope per window for Gz: if combined, scalar (1,); else (numWindows,) */
    
    /* Peak detection arrays for sliding window analysis (for visualization) */
    int* peaksGx;          /**< Peak indicators for Gx: 1 at peak, 0 elsewhere; if combined, (numFreqBins,); else (numWindows, numFreqBins) row-major */
    int* peaksGy;          /**< Peak indicators for Gy: 1 at peak, 0 elsewhere; if combined, (numFreqBins,); else (numWindows, numFreqBins) row-major */
    int* peaksGz;          /**< Peak indicators for Gz: 1 at peak, 0 elsewhere; if combined, (numFreqBins,); else (numWindows, numFreqBins) row-major */
    
    /* Full TR spectrum (single window covering entire TR - continuous envelope) */
    int numFreqBinsFull;   /**< Number of frequency bins for full TR spectrum */
    float freqResolutionFull; /**< Frequency resolution in Hz for full TR spectrum */
    float* frequenciesFull; /**< Frequency axis in Hz (size: numFreqBinsFull), for full TR spectra */
    float* spectraGxFull;  /**< Gx full TR spectrum (numFreqBinsFull,) - continuous envelope */
    float* spectraGyFull;  /**< Gy full TR spectrum (numFreqBinsFull,) - continuous envelope */
    float* spectraGzFull;  /**< Gz full TR spectrum (numFreqBinsFull,) - continuous envelope */

    /* Maximum envelope values for full TR analysis */
    float maxEnvelopeGxFull;  /**< Maximum absolute value in Gx full TR waveform */
    float maxEnvelopeGyFull;  /**< Maximum absolute value in Gy full TR waveform */
    float maxEnvelopeGzFull;  /**< Maximum absolute value in Gz full TR waveform */
    
    /* Full sequence spectrum (N TRs - spectral line picking at harmonics of 1/TR) */
    int numTRs;            /**< Number of TR repetitions in sequence */
    float trDuration_us;   /**< TR duration in microseconds */
    float fundamentalFreq; /**< Fundamental frequency = 1/TR (Hz) */
    int numFreqBinsSeq;    /**< Number of spectral lines (harmonics) in sequence spectrum */
    float* frequenciesSeq; /**< Frequency axis in Hz (size: numFreqBinsSeq), harmonic frequencies */
    float* spectraGxSeq;   /**< Gx sequence spectrum (numFreqBinsSeq,) - magnitudes at harmonics */
    float* spectraGySeq;   /**< Gy sequence spectrum (numFreqBinsSeq,) - magnitudes at harmonics */
    float* spectraGzSeq;   /**< Gz sequence spectrum (numFreqBinsSeq,) - magnitudes at harmonics */
    
    /* Peak detection arrays for sequence spectra (for visualization) */
    int* peaksGxSeq;       /**< Peak indicators for Gx sequence: 1 at peak, 0 elsewhere (numFreqBinsSeq,) */
    int* peaksGySeq;       /**< Peak indicators for Gy sequence: 1 at peak, 0 elsewhere (numFreqBinsSeq,) */
    int* peaksGzSeq;       /**< Peak indicators for Gz sequence: 1 at peak, 0 elsewhere (numFreqBinsSeq,) */

} pulseqlib_TRAcousticSpectra;

#define PULSEQLIB_TR_ACOUSTIC_SPECTRA_INIT {0, 0, 0, 0.0f, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 0, 0.0f, NULL, NULL, NULL, NULL, 0.0f, 0.0f, 0.0f, 0, 0.0f, 0.0f, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, PULSEQLIB_ACOUSTIC_CHECK_RESULT_INIT, PULSEQLIB_ACOUSTIC_CHECK_RESULT_INIT}

typedef struct pulseqlib_ForbiddenBand {
    float freqMin_Hz;      /**< Minimum frequency of the band (Hz) */
    float freqMax_Hz;      /**< Maximum frequency of the band (Hz) */
    float maxAmplitude;    /**< Maximum allowed gradient amplitude in this band (Hz/m) */
} pulseqlib_ForbiddenBand;

#define PULSEQLIB_FORBIDDEN_BAND_INIT {0.0f, 0.0f, 0.0f}

typedef struct pulseqlib_AcousticViolation {
    int detected;              /**< Non-zero if violation detected */
    int bandIndex;             /**< Index of the violated band (-1 if none) */
    float peakFrequency_Hz;    /**< Frequency of the detected peak (Hz) */
    float maxAmplitude;        /**< Max amplitude in the waveform (Hz/m) */
    float allowedAmplitude;    /**< Allowed amplitude for this band (Hz/m) */
} pulseqlib_AcousticViolation;

#define PULSEQLIB_ACOUSTIC_VIOLATION_INIT {0, -1, 0.0f, 0.0f, 0.0f}

typedef struct pulseqlib_AcousticCheckResult {
    pulseqlib_AcousticViolation gx;  /**< Violation info for Gx */
    pulseqlib_AcousticViolation gy;  /**< Violation info for Gy */
    pulseqlib_AcousticViolation gz;  /**< Violation info for Gz */
} pulseqlib_AcousticCheckResult;

#define PULSEQLIB_ACOUSTIC_CHECK_RESULT_INIT {PULSEQLIB_ACOUSTIC_VIOLATION_INIT, PULSEQLIB_ACOUSTIC_VIOLATION_INIT, PULSEQLIB_ACOUSTIC_VIOLATION_INIT}

#if VENDOR == SIEMENS

typedef struct SafeParams {
    float a1;
    float tau1;
    float a2;
    float tau2;
    float a3;
    float tau3;
    float g_scale;
    float stim_thresh; /**< T/m/s */
    float stim_limit;  /**< T/m/s */
} SafeParams;

#define PULSEQLIB_SAFE_PARAMS_INIT {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f}

#endif

/**
 * @brief PNS parameters (vendor-specific via compile-time macro)
 * 
 * GE model (IEC 60601-2-33:2022 Eq. AA.21):
 *   f(tau) = dt/Smin * c / (c + tau)^2
 *   Requires: chronaxie_us, rheobase
 * 
 * Siemens model (SAFE, RC lowpass):
 *   f[k] = alpha * (1 - alpha)^k, alpha = dt / (tau + dt)
 *   Requires: tau_us (equivalent to chronaxie)
 */
typedef struct pulseqlib_PNSParams {
#if VENDOR == GEHC
    float chronaxie_us;  /**< Chronaxie time constant (µs) */
    float rheobase;      /**< Minimum slew rate for stimulation Smin (T/m/s) */
    float alpha;         /**< Effective coil length (m). Smin = rheobase / alpha */
#elif VENDOR == SIEMENS
    SafeParams x;    /**< SAFE model parameters for X axis */
    SafeParams y;    /**< SAFE model parameters for Y axis */
    SafeParams z;    /**< SAFE model parameters for Z axis */
#endif
} pulseqlib_PNSParams;

#if VENDOR == GEHC
#define PULSEQLIB_PNS_PARAMS_INIT {0.0f, 0.0f, 1.0f}
#elif VENDOR == SIEMENS
#define PULSEQLIB_PNS_PARAMS_INIT {PULSEQLIB_SAFE_PARAMS_INIT, PULSEQLIB_SAFE_PARAMS_INIT, PULSEQLIB_SAFE_PARAMS_INIT}
#endif

typedef struct pulseqlib_PNSResult {
    int numSamples;          /**< Number of output samples */
    float* pnsX;             /**< PNS waveform for X (% threshold), or NULL */
    float* pnsY;             /**< PNS waveform for Y (% threshold), or NULL */
    float* pnsZ;             /**< PNS waveform for Z (% threshold), or NULL */
    float* pnsTotal;         /**< Combined PNS: sqrt(X² + Y² + Z²) */
    float maxPNS;            /**< Maximum PNS value (%) */
    int maxPNS_index;        /**< Index of maximum */
    float maxPNS_time_us;    /**< Time of maximum (µs) */
} pulseqlib_PNSResult;

#define PULSEQLIB_PNS_RESULT_INIT {0, NULL, NULL, NULL, NULL, 0.0f, 0, 0.0f}

#endif /* PULSEQLIB_H */

