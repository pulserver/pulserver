/**
 * @file pulseq.h
 * @brief Standalone ANSI-C reader for the Pulseq .seq sequence file format.
 *
 * This module is self-contained: it depends on no other library in this
 * repository, and may be reused on its own to read Pulseq files.  It owns
 * the raw file model (pulseq_types.h), the parser, the shape RLE codec, and
 * the label/hint name tables.
 *
 */

#ifndef PULSEQ_H
#define PULSEQ_H

#include <stdio.h>

#include "pulseq_config.h"
#include "pulseq_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* ============================================================== */
    /*  Lifecycle                                                     */
    /* ============================================================== */

    /**
     * @brief Zero-initialize a pulseq_file before reading into it.
     * @param[out] seq     File to initialize.
     * @param[in]  raster  Design-time rasters, used only for the ones the
     *                     .seq file's [DEFINITIONS] section omits; the file's
     *                     own declared rasters always win.  NULL leaves them
     *                     zero.  These are NOT system/hardware rasters -- see
     *                     pulseq_raster.
     */
    void pulseq_file_init(pulseq_file *seq, const pulseq_raster *raster);

    /** @brief Free all heap allocations inside a pulseq_file. */
    void pulseq_file_free(pulseq_file *seq);

    /** @brief Free every file in a pulseq_file_set (and the array itself). */
    void pulseq_file_set_free(pulseq_file_set *set);

    /* ============================================================== */
    /*  Reading                                                       */
    /* ============================================================== */

    /**
     * @brief Parse a .seq file from disk into a raw pulseq model.
     * @param[out] seq        Must have been pulseq_file_init'd.
     * @param[in]  file_path  Path to the .seq file.
     * @return PULSEQ_SUCCESS on success, negative PULSEQ_ERR_* on failure.
     */
    int pulseq_read(pulseq_file *seq, const char *file_path);

    /**
     * @brief Parse a .seq file already open on a FILE* (e.g. a tmpfile()
     * populated from an in-memory buffer) into a raw pulseq model.
     */
    int pulseq_read_from_buffer(pulseq_file *seq, FILE *f);

    /**
     * @brief Parse a (possibly "next sequence"-chained) .seq file into a set.
     * @param[out] set              Receives the parsed chain.
     * @param[in]  first_file_path  Path to the first .seq file.
     * @param[in]  raster           Fallback rasters, or NULL for defaults.
     */
    int pulseq_file_set_read(
        pulseq_file_set *set,
        const char *first_file_path,
        const pulseq_raster *raster);

    /** @brief Verify a .seq file's embedded [SIGNATURE] MD5 hash. */
    int pulseq_verify_signature(const char *file_path);

    /**
     * @brief Parse only the [DEFINITIONS] section of a .seq file.
     *
     * Enough to read TotalDuration, the rasters, and the "next sequence"
     * chain link without decoding any library.  Free with pulseq_file_free().
     */
    int pulseq_read_definitions_only(pulseq_file *seq, const char *file_path);

    /* ============================================================== */
    /*  Path helpers (for following a "next sequence" chain)          */
    /* ============================================================== */

    /**
     * @brief Directory part of @p file_path, without the trailing separator.
     * @return PULSEQ_ALLOC'd string (caller frees with PULSEQ_FREE), or NULL.
     */
    char *pulseq_path_dirname(const char *file_path);

    /**
     * @brief Join a base directory and a file name.
     * @return PULSEQ_ALLOC'd string (caller frees with PULSEQ_FREE), or NULL.
     */
    char *pulseq_path_join(const char *base_path, const char *filename);

    /* ============================================================== */
    /*  Block accessors                                               */
    /* ============================================================== */

    /**
     * @brief Resolve a block's raw content ids (rf/gx/gy/gz/adc/extension
     * chain head) from the BLOCKS table, without inlining event data.
     */
    int pulseq_get_raw_block_content_ids(
        const pulseq_file *seq,
        pulseq_raw_block *block,
        int block_index,
        int parse_extensions);

    /** @brief Walk a block's extension chain into a resolved index set. */
    void pulseq_get_raw_extension(
        const pulseq_file *seq,
        pulseq_raw_extension *ext,
        const pulseq_raw_block *raw);

    /* ============================================================== */
    /*  Shape codec                                                   */
    /* ============================================================== */

    /**
     * @brief Decompress a run-length-encoded SHAPES library entry.
     * @param[out] result   Receives the decompressed samples (caller frees
     *                      result->samples with PULSEQ_FREE).
     * @param[in]  encoded  Raw (possibly RLE-compressed) shape.
     * @param[in]  scale    Multiplier applied to every decompressed sample.
     * @return 1 on success, 0 on failure.
     */
    int pulseq_decompress_shape(pulseq_shape *result, const pulseq_shape *encoded, float scale);

    /* ============================================================== */
    /*  Name tables                                                   */
    /* ============================================================== */

    /**
     * @brief Map a label/flag name (e.g. "SLC", "PMC", "TRID") to the numeric
     * label id used in pulseq_file.labelset_library / .labelinc_library raw
     * rows (column 1) and decoded by pulseq_get_raw_extension().
     *
     * Needed by any external producer of a pulseq_file (e.g. the
     * ExternalSequence adapter, cxx/pulseq_adapter) that cannot re-derive
     * this id itself.
     *
     * @return The numeric id, or -1 if @p name is not a recognized label/flag.
     */
    int pulseq_label_id_for_name(const char *name);

    /**
     * @brief Map a soft-delay hint name (e.g. "TE", "TR", "TI") to the numeric
     * hint id used in pulseq_file.soft_delay_library raw rows (column 3).
     * @return The numeric id, or -1 if @p name is not a recognized hint.
     */
    int pulseq_hint_id_for_name(const char *name);

#ifdef __cplusplus
}
#endif

#endif /* PULSEQ_H */
