/* Diagnostic: dump C library GZ waveform for bSSFP & GRE max mode */
#include <stdio.h>
#include <stdlib.h>
#include "pulseqlib_methods.h"
#include "pulseqlib_types.h"

#define GAMMA 42577478.0f
#define DATA_DIR "../data/"

int main(void) {
    const char* seqs[] = {
        "bssfp_2d_1sl_1avg",
        "gre_2d_1sl_1avg",
        NULL
    };
    int s;
    for (s = 0; seqs[s]; ++s) {
        char path[512];
        pulseqlib_collection* coll = NULL;
        pulseqlib_tr_waveforms wf = PULSEQLIB_TR_WAVEFORMS_INIT;
        pulseqlib_diagnostic diag = PULSEQLIB_DIAGNOSTIC_INIT;
        pulseqlib_opts opts;
        int rc, i;

        pulseqlib_opts_init(&opts, GAMMA, 3.0f,
            GAMMA * 0.028f, GAMMA * 150.0f,
            2.0f, 20.0f, 2.0f, 20.0f);

        snprintf(path, sizeof(path), DATA_DIR "%s.seq", seqs[s]);
        rc = pulseqlib_read(&coll, &diag, path, &opts, 0, 0, 0, 1);
        if (rc < 0) { printf("load failed for %s: %d\n", seqs[s], rc); continue; }

        rc = pulseqlib_get_tr_waveforms(coll, 0, PULSEQLIB_AMP_MAX_POS, 0, 0, 0, 0, &wf, &diag);
        if (rc < 0) { printf("wf failed for %s: %d\n", seqs[s], rc); pulseqlib_collection_free(coll); continue; }

        printf("=== %s MAX_POS ===\n", seqs[s]);
        printf("  GX: %d samples\n", wf.gx.num_samples);
        printf("  GY: %d samples\n", wf.gy.num_samples);
        printf("  GZ: %d samples\n", wf.gz.num_samples);
        printf("  GZ data:\n");
        for (i = 0; i < wf.gz.num_samples; ++i) {
            printf("    %d: time=%.1f amp=%.1f\n", i, (double)wf.gz.time_us[i], (double)wf.gz.amplitude[i]);
        }

        pulseqlib_tr_waveforms_free(&wf);
        pulseqlib_collection_free(coll);
    }
    return 0;
}
