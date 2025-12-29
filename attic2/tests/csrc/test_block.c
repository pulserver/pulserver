#include "minunit.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "pulseqlib.h"
#include "pulseqlib_methods.h"

/* Helper function to get a block with the new API */
static pulseqlib_SeqBlock* getBlock(pulseqlib_SeqFile* seq, int blockIndex) {
    pulseqlib_SeqBlock* block;

    /* Allocate memory for the sequence file */
    block = (pulseqlib_SeqBlock*)ALLOC(sizeof(pulseqlib_SeqBlock));
    if (!block) return NULL;

    /* Initialize the block with default values */
    pulseqlib_seqBlockInit(block);
    if (!block) {
        FREE(block);
        return NULL;
    }
    
    /* Get the block using the new API */
    pulseqlib_getBlock(seq, block, blockIndex);
    return block;
}

/* UTILS */
/*
#define LABEL_FIELD(label, idx) \
    ((idx)==0 ? (label)->slc : \
    (idx)==1 ? (label)->seg : \
    (idx)==2 ? (label)->rep : \
    (idx)==3 ? (label)->avg : \
    (idx)==4 ? (label)->set : \
    (idx)==5 ? (label)->eco : \
    (idx)==6 ? (label)->phs : \
    (idx)==7 ? (label)->lin : \
    (idx)==8 ? (label)->par : \
    (idx)==9 ? (label)->acq : 0)

#define FLAG_FIELD(flag, idx) \
    ((idx)==0 ? (flag)->trid : \
    (idx)==1 ? (flag)->nav : \
    (idx)==2 ? (flag)->rev : \
    (idx)==3 ? (flag)->sms : \
    (idx)==4 ? (flag)->ref : \
    (idx)==5 ? (flag)->ima : \
    (idx)==6 ? (flag)->noise : \
    (idx)==7 ? (flag)->pmc : \
    (idx)==8 ? (flag)->norot : \
    (idx)==9 ? (flag)->nopos : \
    (idx)==10 ? (flag)->noscl : \
    (idx)==11 ? (flag)->once : 0)
*/
static pulseqlib_SeqFile* load_seq(char* filePath) {
    char seq_path[1024];
    pulseqlib_SeqFile* seq;
    pulseqlib_Opts opts;
    
    /* Allocate memory for the sequence file */
    seq = (pulseqlib_SeqFile*)ALLOC(sizeof(pulseqlib_SeqFile));
    if (!seq) return NULL;
    
    /* Create the full path to the sequence file */
    snprintf(seq_path, sizeof(seq_path), "%s/%s", TEST_ROOT_DIR, filePath);

    /* Initialize runtime options and the sequence structure */
    pulseqlib_optsInit(&opts, 3.0f, 40.0f, 150.0f, 1.0f, 10.0f, 0.1f, 10.0f);
    pulseqlib_seqFileInit(seq, &opts);
    pulseqlib_optsFree(&opts);

    /* Read the sequence data */
    pulseqlib_readSeq(seq, seq_path);
    return seq;
}

/*
static void assert_label_event(const LabelEvent* label, int target_idx, int expected) {
    int i;
    for (i = 0; i < 10; i++) {
        if (i == target_idx) {
            mu_assert(LABEL_FIELD(label, i) == expected, "Target label value mismatch");
        } else {
            mu_assert(LABEL_FIELD(label, i) == 0, "Non-target label field should be 0");
        }
    }
}

static void assert_flag_event(const FlagEvent* flag, int target_idx, int expected) {
    int i;
    for (i = 0; i < 12; i++) {
        if (i == target_idx) {
            mu_assert(FLAG_FIELD(flag, i) == expected, "Target flag value mismatch");
        } else {
            mu_assert(FLAG_FIELD(flag, i) == 0, "Non-target flag field should be 0");
        }
    }
}
*/
/* END UTILS */


MU_TEST(test_rf) {
    int i;
    pulseqlib_SeqBlock* block;
    pulseqlib_SeqFile* seq = load_seq("expected_output/seq2.seq");

    for (i = 0; i < 2; i++){
        block = getBlock(seq, i);
        mu_assert(block != NULL, "getBlock should return a valid block");
        mu_assert(block->rf.type == 1, "Block should have RF event");
        mu_assert(block->gx.type == 0, "Block should not have Gx event");
        mu_assert(block->gy.type == 0, "Block should not have Gy event");
        mu_assert(block->gz.type == 0, "Block should not have Gz event");
        mu_assert(block->adc.type == 0, "Block should not have ADC event");
        mu_assert(block->trigger.type == 0, "Block should not have trigger event");
        mu_assert(block->rotation.type == 0, "Block should not have rotation event");
        mu_assert(block->delay.type == 0, "Block should not have delay event");
        
        mu_assert(block->rf.magShape.numSamples > 0, "RF should have magnitude shape samples");
        mu_assert(block->rf.timeShape.numSamples == 0, "RF should not have time shape samples");
        mu_assert(fabs(block->rf.phaseOffset) < 1e-6, "RF should not have phase offset");
        mu_assert(fabs(block->rf.freqOffset) < 1e-6, "RF should not have freq offset");
        mu_assert(fabs(block->rf.phasePPM) < 1e-6, "RF should not have PPM phase offset");
        mu_assert(fabs(block->rf.freqPPM) < 1e-6, "RF should not have PPM freq offset");
        mu_assert(block->rf.delay == 0, "RF should not have delay");
    }

    /* Real RF */
    block = getBlock(seq, 0);
    mu_assert(block->duration == 400, "Block 0 duration should be 400 block raster units");
    mu_assert(block->rf.phaseShape.numSamples == 0, "Block 0 RF should not have phase shape samples");
    mu_assert(fabs(block->rf.center - 2000.0) < 1e-6, "Block 0 RF should have center at 2000 us");
    mu_assert(block->rfShimming.type == 1, "Block 0 should have RF shimming event");

    mu_assert(block->rfShimming.nChan == 8, "Block 0 RF shimming should have 8 channels");
    for (i = 0; i < block->rfShimming.nChan; i++) {
        mu_assert(fabs(block->rfShimming.amplitudes[i] - 1) < 1e-6, "Block 0 RF shimming channel amplitudes should be 1");
        mu_assert(fabs(block->rfShimming.phases[i]) < 1e-6, "Block 0 RF shimming channel phases should be 0");
    }

    /* Complex RF */
    block = getBlock(seq, 1);
    mu_assert(block->duration == 1000, "Block 1 duration should be 1000 block raster units");
    mu_assert(block->rf.phaseShape.numSamples > 0, "Block 1 RF should have phase shape samples");
    mu_assert(fabs(block->rf.center - 5000.5) < 1e-6, "Block 1 RF should have center at 5000.5 us");
    mu_assert(block->rfShimming.type == 0, "Block 1 should not have RF shimming event");

    pulseqlib_seqBlockFree(block);
    pulseqlib_seqFileFree(seq);
}

MU_TEST(test_adc) {
    int i;
    pulseqlib_SeqBlock* block;
    pulseqlib_SeqFile* seq = load_seq("expected_output/seq2.seq");

    for (i = 2; i < 4; i++) {
        block = getBlock(seq, i);
        mu_assert(block != NULL, "getBlock should return a valid block");
        mu_assert(block->duration == 128, "Block duration should be 128 block raster units");
        mu_assert(block->duration == 128, "Block duration should be 128 block raster units");
        mu_assert(block->rf.type == 0, "Block should not have RF event");
        mu_assert(block->gx.type == 0, "Block should not have Gx event");
        mu_assert(block->gy.type == 0, "Block should not have Gy event");
        mu_assert(block->gz.type == 0, "Block should not have Gz event");
        mu_assert(block->adc.type == 1, "Block should have ADC event");
        mu_assert(block->trigger.type == 0, "Block should not have trigger event");
        mu_assert(block->rotation.type == 0, "Block should not have rotation event");
        mu_assert(block->delay.type == 0, "Block should not have delay event");
        mu_assert(block->rfShimming.type == 0, "Block should not have RF shimming event");

        mu_assert(block->adc.numSamples == 128, "ADC should have 128 samples");
        mu_assert(block->adc.dwellTime == 10000, "ADC should have 10us dwell time");
        mu_assert(fabs(block->adc.phaseOffset) < 1e-6, "ADC should not have phase offset");
        mu_assert(fabs(block->adc.freqOffset) < 1e-6, "ADC should not have freq offset");
        mu_assert(fabs(block->adc.phasePPM) < 1e-6, "ADC should not have PPM phase offset");
        mu_assert(fabs(block->adc.freqPPM) < 1e-6, "ADC should not have PPM freq offset");
        mu_assert(block->adc.delay == 0, "ADC should not have delay");
    }

    /* phase modulated ADC */
    block = getBlock(seq, 2);
    mu_assert(block->adc.phaseModulationShape.numSamples > 0, "Block 2 ADC should have phase modulation");

    /* standard ADC */
    block = getBlock(seq, 3); /* No phase modulation extension for block 3 */
    mu_assert(block->adc.phaseModulationShape.numSamples == 0, "Block 3 ADC should not have phase modulation");

    pulseqlib_seqBlockFree(block);
    pulseqlib_seqFileFree(seq);
}

MU_TEST(test_grad) {
    int i;
    pulseqlib_SeqBlock* block;
    pulseqlib_SeqFile* seq = load_seq("expected_output/seq2.seq");

    for (i = 36; i < 45; i++){
        block = getBlock(seq, i);
        mu_assert(block != NULL, "getBlock should return a valid block");
        mu_assert(block->rf.type == 0, "Block should not have RF event");
        mu_assert(block->adc.type == 0, "Block should not have ADC event");
        mu_assert(block->trigger.type == 0, "Block should not have trigger event");
        mu_assert(block->delay.type == 0, "Block should not have delay event");
        mu_assert(block->rfShimming.type == 0, "Block should not have RF shimming event");
    }

    /**** TRAPEZOIDS ****/
    /* Gx */
    block = getBlock(seq, 36);
    mu_assert(block->duration == 100, "Block 36 duration should be 100 block raster units");
    mu_assert(block->gx.type == 1, "Block 36 should have trapezoidal Gx event");
    mu_assert(block->gy.type == 0, "Block 36 should not have Gy event");
    mu_assert(block->gz.type == 0, "Block 36 should not have Gz event");
    mu_assert(block->rotation.type == 0, "Block 36 should not have rotation event");

    mu_assert(fabs(block->gx.amplitude - 10) < 1e-6, "Block 36 Gx amplitude should be 10 Hz/m");
    mu_assert(block->gx.delay == 0, "Block 36 Gx should not have delay");
    mu_assert(block->gx.trap.riseTime == 10, "Block 36 Gx rise time should be 10 us");
    mu_assert(block->gx.trap.flatTime == 980, "Block 36 Gx flat time should be 980 us");
    mu_assert(block->gx.trap.fallTime == 10, "Block 36 Gx fall time should be  10 us");
    mu_assert(block->gx.first == 0, "Block 36 Gx first sample should be 0");
    mu_assert(block->gx.last == 0, "Block 36 Gx last sample should be 0");

    /* Gy */
    block = getBlock(seq, 37);
    mu_assert(block->duration == 100, "Block 37 duration should be 100 block raster units");
    mu_assert(block->gx.type == 0, "Block 37 should not have Gx event");
    mu_assert(block->gy.type == 1, "Block 37 should have trapezoidal Gy event");
    mu_assert(block->gz.type == 0, "Block 37 should not have Gz event");
    mu_assert(block->rotation.type == 0, "Block 37 should not have rotation event");

    mu_assert(fabs(block->gy.amplitude - 5154.64014) < 1e-5, "Block 37 Gy amplitude should be 5154.64 Hz/m");
    mu_assert(block->gy.delay == 0, "Block 37 Gy should not have delay");
    mu_assert(block->gy.trap.riseTime == 30, "Block 37 Gy rise time should be 30 us");
    mu_assert(block->gy.trap.flatTime == 940, "Block 37 Gy flat time should be 940 us");
    mu_assert(block->gy.trap.fallTime == 30, "Block 37 Gy fall time should be  30 us");
    mu_assert(block->gy.first == 0, "Block 37 Gy first sample should be 0");
    mu_assert(block->gy.last == 0, "Block 37 Gy last sample should be 0");

    /* Gz */
    block = getBlock(seq, 38);
    mu_assert(block->duration == 100, "Block 38 duration should be 100 block raster units");
    mu_assert(block->gx.type == 0, "Block 38 should not have Gx event");
    mu_assert(block->gy.type == 0, "Block 38 should not have Gy event");
    mu_assert(block->gz.type == 1, "Block 38 should have trapezoidal Gz event");
    mu_assert(block->rotation.type == 0, "Block 38 should not have rotation event");

    mu_assert(fabs(block->gz.amplitude - 5154.64014) < 1e-5, "Block 38 Gz amplitude should be 5154.64 Hz/m");
    mu_assert(block->gz.delay == 0, "Block 38 Gz should not have delay");
    mu_assert(block->gz.trap.riseTime == 30, "Block 38 Gz rise time should be 30 us");
    mu_assert(block->gz.trap.flatTime == 940, "Block 38 Gz flat time should be 940 us");
    mu_assert(block->gz.trap.fallTime == 30, "Block 38 Gz fall time should be  30 us");
    mu_assert(block->gz.first == 0, "Block 38 Gz first sample should be 0");
    mu_assert(block->gz.last == 0, "Block 38 Gz last sample should be 0");
    /**** END TRAPEZOIDS ****/

    /**** ARBITRARY GRAD ****/
    /* Gx */
    block = getBlock(seq, 39);
    mu_assert(block->duration == 5, "Block 39 duration should be 5 block raster units");
    mu_assert(block->gx.type == 2, "Block 39 should have arbitrary Gx event");
    mu_assert(block->gy.type == 0, "Block 39 should not have Gy event");
    mu_assert(block->gz.type == 0, "Block 39 should not have Gz event");
    mu_assert(block->rotation.type == 1, "Block 39 should have rotation event");

    mu_assert(fabs(block->gx.amplitude - 1) < 1e-6, "Block 39 Gx amplitude should be 1 Hz/m");
    mu_assert(block->gx.delay == 0, "Block 39 Gx should not have delay");
    mu_assert(block->gx.waveShape.numSamples > 0, "Block 39 Gx should have magnitude shape samples");
    mu_assert(block->gx.timeShape.numSamples == 0, "Block 39 Gx should not have time shape samples");
    mu_assert(fabs(block->gx.first + 0.05) < 1e-6, "Block 39 Gx first sample should be -0.05");
    mu_assert(fabs(block->gx.last + 0.05) < 1e-6, "Block 39 Gx last sample should be -0.05");
    mu_assert(fabs(block->rotation.data.rotQuaternion[0] - 1.0) < 1e-6, "Block 39 Rotation quaternion quat0 should be 1.0");
    mu_assert(fabs(block->rotation.data.rotQuaternion[1]) < 1e-6, "Block 39 Rotation quaternion quatX should be 0.0");
    mu_assert(fabs(block->rotation.data.rotQuaternion[2]) < 1e-6, "Block 39 Rotation quaternion quatY should be 0.0");
    mu_assert(fabs(block->rotation.data.rotQuaternion[3]) < 1e-6, "Block 39 Rotation quaternion quatZ should be 0.0");

    /* Gy */
    block = getBlock(seq, 40);
    mu_assert(block->duration == 5, "Block 40 duration should be 5 block raster units");
    mu_assert(block->gx.type == 0, "Block 40 should not have Gx event");
    mu_assert(block->gy.type == 2, "Block 40 should have arbitrary Gy event");
    mu_assert(block->gz.type == 0, "Block 40 should not have Gz event");
    mu_assert(block->rotation.type == 1, "Block 40 should have rotation event");

    mu_assert(fabs(block->gy.amplitude - 1) < 1e-6, "Block 40 Gy amplitude should be 1 Hz/m");
    mu_assert(block->gy.delay == 0, "Block 40 Gy should not have delay");
    mu_assert(block->gy.waveShape.numSamples > 0, "Block 40 Gy should have magnitude shape samples");
    mu_assert(block->gy.timeShape.numSamples == 0, "Block 40 Gy should not have time shape samples");
    mu_assert(fabs(block->gy.first + 0.05) < 1e-6, "Block 40 Gy first sample should be -0.05");
    mu_assert(fabs(block->gy.last + 0.05) < 1e-6, "Block 40 Gy last sample should be -0.05");
    mu_assert(fabs(block->rotation.data.rotQuaternion[0] - 1.0) < 1e-6, "Block 40 Rotation quaternion quat0 should be 1.0");
    mu_assert(fabs(block->rotation.data.rotQuaternion[1]) < 1e-6, "Block 40 Rotation quaternion quatX should be 0.0");
    mu_assert(fabs(block->rotation.data.rotQuaternion[2]) < 1e-6, "Block 40 Rotation quaternion quatY should be 0.0");
    mu_assert(fabs(block->rotation.data.rotQuaternion[3]) < 1e-6, "Block 40 Rotation quaternion quatZ should be 0.0");

    /* Gz */
    block = getBlock(seq, 41);
    mu_assert(block->duration == 5, "Block 41 duration should be 5 block raster units");
    mu_assert(block->gx.type == 0, "Block 41 should not have Gx event");
    mu_assert(block->gy.type == 0, "Block 41 should not have Gy event");
    mu_assert(block->gz.type == 2, "Block 41 should have arbitrary Gz event");
    mu_assert(block->rotation.type == 1, "Block 41 should have rotation event");

    mu_assert(fabs(block->gz.amplitude - 1) < 1e-6, "Block 41 Gz amplitude should be 1 Hz/m");
    mu_assert(block->gz.delay == 0, "Block 41 Gz should not have delay");
    mu_assert(block->gz.waveShape.numSamples > 0, "Block 41 Gz should have magnitude shape samples");
    mu_assert(block->gz.timeShape.numSamples == 0, "Block 41 Gz should not have time shape samples");
    mu_assert(fabs(block->gz.first + 0.05) < 1e-6, "Block 41 Gz first sample should be -0.05");
    mu_assert(fabs(block->gz.last + 0.05) < 1e-6, "Block 41 Gz last sample should be -0.05");
    mu_assert(fabs(block->rotation.data.rotQuaternion[0] - 1.0) < 1e-6, "Block 41 Rotation quaternion quat0 should be 1.0");
    mu_assert(fabs(block->rotation.data.rotQuaternion[1]) < 1e-6, "Block 41 Rotation quaternion quatX should be 0.0");
    mu_assert(fabs(block->rotation.data.rotQuaternion[2]) < 1e-6, "Block 41 Rotation quaternion quatY should be 0.0");
    mu_assert(fabs(block->rotation.data.rotQuaternion[3]) < 1e-6, "Block 41 Rotation quaternion quatZ should be 0.0");
    /**** END ARBITRARY GRAD ****/

    /**** EXTENDED TRAPEZOIDS GRAD ****/
    /* Gx */
    block = getBlock(seq, 42);
    mu_assert(block->duration == 203, "Block 42 duration should be 5 block raster units");
    mu_assert(block->gx.type == 2, "Block 42 should have arbitrary Gx event");
    mu_assert(block->gy.type == 0, "Block 42 should not have Gy event");
    mu_assert(block->gz.type == 0, "Block 42 should not have Gz event");
    mu_assert(block->rotation.type == 0, "Block 42 should not have rotation event");

    mu_assert(fabs(block->gx.amplitude - 1) < 1e-6, "Block 42 Gx amplitude should be 1 Hz/m");
    mu_assert(block->gx.delay == 0, "Block 42 Gx should not have delay");
    mu_assert(block->gx.waveShape.numSamples > 0, "Block 42 Gx should have magnitude shape samples");
    mu_assert(block->gx.timeShape.numSamples > 0, "Block 42 Gx should have time shape samples");
    mu_assert(fabs(block->gx.first) < 1e-6, "Block 42 Gx first sample should be 0");
    mu_assert(fabs(block->gx.last) < 1e-6, "Block 42 Gx last sample should be 0");

    /* Gy */
    block = getBlock(seq, 43);
    mu_assert(block->duration == 203, "Block 43 duration should be 5 block raster units");
    mu_assert(block->gx.type == 0, "Block 43 should not have Gx event");
    mu_assert(block->gy.type == 2, "Block 43 should have arbitrary Gy event");
    mu_assert(block->gz.type == 0, "Block 43 should not have Gz event");
    mu_assert(block->rotation.type == 0, "Block 43 should not have rotation event");

    mu_assert(fabs(block->gy.amplitude - 1) < 1e-6, "Block 43 Gy amplitude should be 1 Hz/m");
    mu_assert(block->gy.delay == 0, "Block 43 Gy should not have delay");
    mu_assert(block->gy.waveShape.numSamples > 0, "Block 43 Gy should have magnitude shape samples");
    mu_assert(block->gy.timeShape.numSamples > 0, "Block 43 Gy should have time shape samples");
    mu_assert(fabs(block->gy.first) < 1e-6, "Block 43 Gy first sample should be 0");
    mu_assert(fabs(block->gy.last) < 1e-6, "Block 43 Gy last sample should be 0");

    /* Gz */
    block = getBlock(seq, 44);
    mu_assert(block->duration == 203, "Block 44 duration should be 5 block raster units");
    mu_assert(block->gx.type == 0, "Block 44 should not have Gx event");
    mu_assert(block->gy.type == 0, "Block 44 should not have Gy event");
    mu_assert(block->gz.type == 2, "Block 44 should have arbitrary Gz event");
    mu_assert(block->rotation.type == 0, "Block 44 should not have rotation event");

    mu_assert(fabs(block->gz.amplitude - 1) < 1e-6, "Block 44 Gz amplitude should be 1 Hz/m");
    mu_assert(block->gz.delay == 0, "Block 44 Gz should not have delay");
    mu_assert(block->gz.waveShape.numSamples > 0, "Block 44 Gz should have magnitude shape samples");
    mu_assert(block->gz.timeShape.numSamples > 0, "Block 44 Gz should have time shape samples");
    mu_assert(fabs(block->gz.first) < 1e-6, "Block 44 Gz first sample should be 0");
    mu_assert(fabs(block->gz.last) < 1e-6, "Block 44 Gz last sample should be 0");
    /**** END EXTENDED TRAPEZOIDS GRAD ****/

    pulseqlib_seqBlockFree(block);
    pulseqlib_seqFileFree(seq);
}

/*
MU_TEST(test_flags) {
    int i;
    int n = 0;
    pulseqlib_SeqBlock* block;
    pulseqlib_SeqFile* seq = load_seq("tests/expected_output/seq2.seq");

    for (i = 3; i < 25; i++) {
        block = getBlock(seq, i);
        mu_assert(block != NULL, "getBlock should return a valid block");
        mu_assert(block->duration == 128, "Block duration should be 128 block raster units");
        mu_assert(block->trigger.type == 0, "Block should not have trigger event");
        mu_assert(block->rotation.type == 0, "Block should not have rotation event");
        mu_assert(block->delay.type == 0, "Block should not have delay event");
        mu_assert(block->rfShimming.type == 0, "Block should not have RF shimming event");

        assert_flag_event(&block->flag, n, 1);
        n += 1;
    }
    pulseqlib_seqBlockFree(block);
    pulseqlib_seqFileFree(seq);
}

MU_TEST(test_labels) {
    int i;
    int n = 0;
    pulseqlib_SeqBlock* block;
    pulseqlib_SeqFile* seq = load_seq("tests/expected_output/seq2.seq");

    for (i = 25; i < 35; i++) {
        block = getBlock(seq, i);
        mu_assert(block != NULL, "getBlock should return a valid block");
        mu_assert(block->duration == 128, "Block duration should be 128 block raster units");
        mu_assert(block->trigger.type == 0, "Block should not have trigger event");
        mu_assert(block->rotation.type == 0, "Block should not have rotation event");

        assert_label_event(&block->label, n, 1);
        n += 1;
    }
    pulseqlib_seqBlockFree(block);
    pulseqlib_seqFileFree(seq);
}
*/
MU_TEST(test_trigger) {
    int i;
    pulseqlib_SeqBlock* block;
    pulseqlib_SeqFile* seq = load_seq("expected_output/seq2.seq");

    for (i = 45; i < 50; i++) {
        block = getBlock(seq, i);
        mu_assert(block != NULL, "getBlock should return a valid block");
        mu_assert(block->rf.type == 0, "Block should not have RF event");
        mu_assert(block->gx.type == 0, "Block should not have Gx event");
        mu_assert(block->gy.type == 0, "Block should not have Gy event");
        mu_assert(block->gz.type == 0, "Block should not have Gz event");
        mu_assert(block->adc.type == 0, "Block should not have ADC event");
        mu_assert(block->trigger.type == 1, "Block should have trigger event");
        mu_assert(block->rotation.type == 0, "Block should not have rotation event");
        mu_assert(block->delay.type == 0, "Block should not have delay event");
        mu_assert(block->rfShimming.type == 0, "Block should not have RF shimming event");
    }

    block = getBlock(seq, 45);
    mu_assert(block->duration == 1, "Block 45 duration should be 1 block raster units");
    mu_assert(block->trigger.duration == 10, "Block 45 trigger duration should be 10 us");
    mu_assert(block->trigger.delay == 0, "Block 45 trigger delay should be 0");
    mu_assert(block->trigger.triggerType == TRIGGER_TYPE_INPUT, "Block 45 trigger should be a cardiac trigger");
    mu_assert(block->trigger.triggerChannel == TRIGGER_CHANNEL_INPUT_PHYSIO_1, "Block 45 channel type should be physio1");

    block = getBlock(seq, 46);
    mu_assert(block->duration == 1, "Block 46 duration should be 1 block raster units");
    mu_assert(block->trigger.duration == 10, "Block 46 trigger duration should be 10 us");
    mu_assert(block->trigger.delay == 0, "Block 46 trigger delay should be 0");
    mu_assert(block->trigger.triggerType == TRIGGER_TYPE_INPUT, "Block 46 trigger should be a cardiac trigger");
    mu_assert(block->trigger.triggerChannel == TRIGGER_CHANNEL_INPUT_PHYSIO_2, "Block 46 channel type should be physio2");

    block = getBlock(seq, 47);
    mu_assert(block->duration == 400, "Block 47 duration should be 400 block raster units");
    mu_assert(block->trigger.duration == 4000, "Block 47 trigger duration should be 4000 us");
    mu_assert(block->trigger.delay == 0, "Block 47 trigger delay should be 0");
    mu_assert(block->trigger.triggerType == TRIGGER_TYPE_OUTPUT, "Block 47 trigger should be a digital output trigger");
    mu_assert(block->trigger.triggerChannel == TRIGGER_CHANNEL_OUTPUT_OSC_0, "Block 47 channel type should be osc0");

    block = getBlock(seq, 48);
    mu_assert(block->duration == 400, "Block 48 duration should be 400 block raster units");
    mu_assert(block->trigger.duration == 4000, "Block 48 trigger duration should be 4000 us");
    mu_assert(block->trigger.delay == 0, "Block 48 trigger delay should be 0");
    mu_assert(block->trigger.triggerType == TRIGGER_TYPE_OUTPUT, "Block 48 trigger should be a digital output trigger");
    mu_assert(block->trigger.triggerChannel == TRIGGER_CHANNEL_OUTPUT_OSC_1, "Block 48 channel type should be osc1");

    block = getBlock(seq, 49);
    mu_assert(block != NULL, "getBlock should return a valid block");
    mu_assert(block->duration == 400, "Block 49 duration should be 400 block raster units");
    mu_assert(block->trigger.duration == 4000, "Block 49 trigger duration should be 4000 us");
    mu_assert(block->trigger.delay == 0, "Block 49 trigger delay should be 0");
    mu_assert(block->trigger.triggerType == TRIGGER_TYPE_OUTPUT, "Block 49 trigger should be a digital output trigger");
    mu_assert(block->trigger.triggerChannel == TRIGGER_CHANNEL_OUTPUT_EXT_1, "Block 49 channel type should be ext1");

    pulseqlib_seqBlockFree(block);
    pulseqlib_seqFileFree(seq);
}

MU_TEST(test_delay) {
    int i;
    pulseqlib_SeqBlock* block;
    pulseqlib_SeqFile* seq = load_seq("expected_output/seq2.seq");

    for (i = 50; i < 58; i++) {
        block = getBlock(seq, i);
        mu_assert(block != NULL, "getBlock should return a valid block");
        mu_assert(block->duration == 1, "Block duration should be 1 block raster units");
        mu_assert(block->rf.type == 0, "Block should not have RF event");
        mu_assert(block->gx.type == 0, "Block should not have Gx event");
        mu_assert(block->gy.type == 0, "Block should not have Gy event");
        mu_assert(block->gz.type == 0, "Block should not have Gz event");
        mu_assert(block->adc.type == 0, "Block should not have ADC event");
        mu_assert(block->trigger.type == 0, "Block should not have trigger event");
        mu_assert(block->rotation.type == 0, "Block should not have rotation event");
        mu_assert(block->delay.type == 1, "Block should have delay event");
        mu_assert(block->rfShimming.type == 0, "Block should not have RF shimming event");

        mu_assert(block->delay.offset == 0, "Block delay should have offset 0");
        mu_assert(block->delay.factor == 1, "Block delay should have factor 1");
    }

    block = getBlock(seq, 50);
    mu_assert(block->delay.numID == 0, "Block 50 delay should have numID 0");
    mu_assert(block->delay.hintID == HINT_TE, "Block 50 delay type should be 'TE'");

    block = getBlock(seq, 51);
    mu_assert(block->delay.numID == 1, "Block 51 delay should have numID 1");
    mu_assert(block->delay.hintID == HINT_TR, "Block 51 delay type should be 'TR'");

    block = getBlock(seq, 52);
    mu_assert(block->delay.numID == 2, "Block 52 delay should have numID 2");
    mu_assert(block->delay.hintID == HINT_TI, "Block 52 delay type should be 'TI'");

    block = getBlock(seq, 53);
    mu_assert(block->delay.numID == 3, "Block 53 delay should have numID 3");
    mu_assert(block->delay.hintID == HINT_ESP, "Block 53 delay type should be 'ESP'");

    block = getBlock(seq, 54);
    mu_assert(block->delay.numID == 4, "Block 54 delay should have numID 4");
    mu_assert(block->delay.hintID == HINT_RECTIME, "Block 54 delay type should be 'RECTIME'");

    block = getBlock(seq, 55);
    mu_assert(block->delay.numID == 5, "Block 55 delay should have numID 5");
    mu_assert(block->delay.hintID == HINT_T2PREP, "Block 55 delay type should be 'T2PREP'");

    block = getBlock(seq, 56);
    mu_assert(block->delay.numID == 6, "Block 56 delay should have numID 6");
    mu_assert(block->delay.hintID == HINT_TE2, "Block 56 delay type should be 'TE2'");

    block = getBlock(seq, 57);
    mu_assert(block->delay.numID == 7, "Block 57 delay should have numID 7");
    mu_assert(block->delay.hintID == HINT_TR2, "Block 57 delay type should be 'TR2'");

    pulseqlib_seqBlockFree(block);
    pulseqlib_seqFileFree(seq);
}

MU_TEST(test_pure_delay) {
    pulseqlib_SeqBlock* block;
    pulseqlib_SeqFile* seq = load_seq("expected_output/seq2.seq");

    block = getBlock(seq, 58);
    mu_assert(block != NULL, "getBlock should return a valid block");
    mu_assert(block->duration == 100000, "Block 58 duration should be 100000");
    mu_assert(block->rf.type == 0, "Block 58 should not have RF event");
    mu_assert(block->gx.type == 0, "Block 58 should not have Gx event");
    mu_assert(block->gy.type == 0, "Block 58 should not have Gy event");
    mu_assert(block->gz.type == 0, "Block 58 should not have Gz event");
    mu_assert(block->adc.type == 0, "Block 58 should not have ADC event");
    mu_assert(block->trigger.type == 0, "Block 58 should not have trigger event");
    mu_assert(block->rotation.type == 0, "Block 58 should not have rotation event");
    mu_assert(block->delay.type == 0, "Block 58 should not have delay event");
    mu_assert(block->rfShimming.type == 0, "Block 58 should not have RF shimming event");

    pulseqlib_seqBlockFree(block);
    pulseqlib_seqFileFree(seq);
}


MU_TEST_SUITE(test_seqblock_suite) {
    printf("Running test_rf...\n");
    MU_RUN_TEST(test_rf);
    printf("Running test_adc...\n");
    MU_RUN_TEST(test_adc);
    printf("Running test_grad...\n");
    MU_RUN_TEST(test_grad);
    /*printf("Running test_flags...\n");
    MU_RUN_TEST(test_flags);
    printf("Running test_labels...\n");
    MU_RUN_TEST(test_labels);*/
    printf("Running test_trigger...\n");
    MU_RUN_TEST(test_trigger);
    printf("Running test_delay...\n");
    MU_RUN_TEST(test_delay);
    printf("Running test_pure_delay...\n");
    MU_RUN_TEST(test_pure_delay);
}

int test_block_main(void) {
    printf("Starting test suite...\n");
    MU_RUN_SUITE(test_seqblock_suite);
    printf("Test suite completed.\n");
    MU_REPORT();
    return MU_EXIT_CODE;
}