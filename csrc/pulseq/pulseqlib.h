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

/********************************************************* Shapes  ******************************************************/
typedef struct pulseqlib_ShapeArbitrary {
    int numUncompressedSamples; /**< @brief Number of uncompressed waveform samples */
    int numSamples; /**< @brief Number of waveform samples */
    float *samples; /**< @brief Waveform samples */
} pulseqlib_ShapeArbitrary; /* mirrors Pulseq CompressedShape */


typedef struct pulseqlib_ShapeTrap {
    long riseTime; /**< @brief Ramp up time of trapezoid (us)  */
    long flatTime; /**< @brief Flat-top time of trapezoid (us)  */
    long fallTime; /**< @brief Ramp down time of trapezoid (us) */
} pulseqlib_ShapeTrap; /* no Pulseq equivalent */


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


typedef struct pulseqlib_TriggerEvent {
    short type; /**< @brief OFF or ON */
    long duration; /**< @brief Duration of trigger event (us) */
    long delay; /**< @brief Delay prior to the trigger event (us) */
    int triggerType; /**< @brief Type of trigger (system dependent). 0: undefined / unused */
    int triggerChannel; /**< @brief Channel of trigger (system dependent). 0: undefined / unused */
} pulseqlib_TriggerEvent; /* mirrors Pulseq TriggerEvent */


typedef struct pulseqlib_RotationEvent {
    short type; /**< @brief NULL or DEFINED */
    union {
        float rotQuaternion[4]; /**< @brief Gradient rotation quaternion [w, x, y, z] */
        float rotMatrix[9]; /**< @brief Gradient rotation matrix (3x3, row-major) */
    } data;
} pulseqlib_RotationEvent; /* extends Pulseq RotationEvent */


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


/********************************************************* EveBlocknts  ******************************************************/
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

typedef struct pulseqlib_TRdescriptor {
    int trSize; /**< Size of the TR in number of blocks */
    int numTRs; /**< Number of TRs in the sequence */
    int numPrepTRs; /**< Number of preparation TR before the main TR */
    int numPrepBlocks; /**< Number of preparation blocks before the main TR */
    int degeneratePrep; /**< Non-zero if the preparation blocks are degenerate (i.e. identical to main TR) */
    int numCooldownTRs; /**< Number of cooldown TR after the main TR */
    int numCooldownBlocks; /**< Number of cooldown blocks after the main TR */
    int degenerateCooldown; /**< Non-zero if the cooldown blocks are degenerate (i.e. identical to main TR) */
} pulseqlib_TRdescriptor;

typedef struct pulseqlib_TRsegment {
    int startBlock; /**< Starting block index of the TR segment */
    int numBlocks; /**< Number of blocks in the TR segment */
    int* uniqueBlockIndices; /**< Pointer to array of unique block indices in the segment */
} pulseqlib_TRsegment;


#endif /* PULSEQLIB_H */

