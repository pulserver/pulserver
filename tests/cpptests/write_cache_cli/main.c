/*
 * write_cache_cli/main.c
 *
 * Test-only helper for the mrdserver C++ trajectory_cache_loader test
 * (and the analogous TruthBuilder run_generators step).
 *
 * Given a path to a Pulseq .seq file, produces the matching pulseqlib
 * binary cache <base>.pge. pulseqlib_read (cache_binary=1) emits the whole
 * sectioned cache in one shot at load time: COMMON, ROTATIONS, SHAPES,
 * SCANLOOP, FREQMOD, TRAJECTORY and SEQDESC. No separate trajectory /
 * sequence-description pass is needed here any more.
 *
 * Lives under mrdserver/tests/ on purpose: examples/ is the reference
 * caller and must stay minimal; this CLI carries the full one-shot
 * recipe used by the test-truth toolchain.
 */

#include "pulseqlib_methods.h"
#include "pulseqlib_types.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(rc, diag, label)                                         \
    do                                                                 \
    {                                                                  \
        if (PULSEQLIB_FAILED(rc))                                      \
        {                                                              \
            fprintf(stderr, "[write_cache] %s failed: rc=%d (%s)\n",   \
                    (label), (rc),                                     \
                    (diag).message[0] ? (diag).message : "(no diag)"); \
            goto fail;                                                 \
        }                                                              \
    } while (0)

int main(int argc, char **argv)
{
    pulseqlib_collection *coll = NULL;
    pulseqlib_diagnostic diag = PULSEQLIB_DIAGNOSTIC_INIT;
    pulseqlib_opts opts = PULSEQLIB_OPTS_INIT;
    int rc;
    const char *seq_path;

    int num_averages = 1;
    if (argc != 2)
    {
        fprintf(stderr,
                "usage: %s <path/to/sequence.seq>\n"
                "\n"
                "Produces <base>.pge in the same directory.\n"
                "If <base>_meta.txt exists alongside the .seq and contains a\n"
                "'num_averages N' line, that NEX is honoured (matches truth).\n",
                argv[0]);
        return 2;
    }
    seq_path = argv[1];

    /* Optional: discover NEX from the companion <base>_meta.txt that
     * TruthBuilder writes. .seq files have no NEX field, so without this
     * hint we would always cache only one average and disagree with truth
     * on _Navg_ fixtures. */
    {
        const char *dot = strrchr(seq_path, '.');
        size_t base_len = dot ? (size_t)(dot - seq_path) : strlen(seq_path);
        char meta_path[1024];
        if (base_len + sizeof("_meta.txt") < sizeof(meta_path))
        {
            memcpy(meta_path, seq_path, base_len);
            memcpy(meta_path + base_len, "_meta.txt", sizeof("_meta.txt"));
            FILE *mf = fopen(meta_path, "r");
            if (mf)
            {
                char line[256];
                while (fgets(line, sizeof(line), mf))
                {
                    int nv;
                    if (sscanf(line, "num_averages %d", &nv) == 1 && nv > 0)
                    {
                        num_averages = nv;
                        break;
                    }
                }
                fclose(mf);
            }
        }
    }

    /* Vendor-neutral defaults consistent with TruthBuilder fixtures.
     * Limits are intentionally generous so any fixture passes safety;
     * we are not validating the .seq, only producing the cache.
     * vendor = GEHC: only GEHC label-parsing is currently implemented. */
    opts.vendor = PULSEQLIB_VENDOR_GEHC;
    opts.gamma_hz_per_t = 42577478.0f;
    opts.b0_t = 3.0f;
    opts.max_grad_hz_per_m = 42577478.0f * 1.0f;           /* 1000 mT/m */
    opts.max_slew_hz_per_m_per_s = 42577478.0f * 10000.0f; /* 10000 T/m/s */
    opts.rf_raster_us = 1.0f;
    opts.grad_raster_us = 10.0f;
    opts.adc_raster_us = 0.1f;
    opts.block_raster_us = 10.0f;

    /* parse_labels=1: the trajectory/seqdesc sections need per-ADC label state.
     * cache_binary=1: pulseqlib_read writes the full sectioned .pge at load. */
    rc = pulseqlib_read(&coll, &diag, seq_path, &opts,
                        1,             /* cache_binary     */
                        1,             /* verify_signature */
                        1,             /* parse_labels     */
                        num_averages); /* num_averages     */
    CHECK(rc, diag, "pulseqlib_read");

    pulseqlib_collection_free(coll);
    fprintf(stderr, "[write_cache] OK: wrote cache for %s (num_averages=%d)\n",
            seq_path, num_averages);
    return 0;

fail:
    if (coll)
        pulseqlib_collection_free(coll);
    return 1;
}
