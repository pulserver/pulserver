/**
 * @file pulseg_cache.h
 * @brief Binary sequence cache: whole-file save/load plus per-section loads.
 *
 * The cache sits beside the .seq file and holds the PSD-side structures
 * that are expensive to recompute (dedup tables, segmentation, the
 * execution stream). Its extension is vendor-selectable via
 * pulseg_opts.cache_ext -- .pseg by default, .pge on GE. The recon side
 * reads the seqfile directly and never touches the cache.
 *
 * pulseg_save_cache() writes every section; the per-section loaders exist so
 * a consumer can pay only for what it needs (the pulse-generation pass reads
 * COMMON+SHAPES, the scan loop additionally reads INSTANCES and SCANLOOP).
 */

#ifndef PULSEG_CACHE_H
#define PULSEG_CACHE_H

#include <stdio.h>
#include "pulseg_config.h"
#include "pulseg_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* ================================================================== */
    /*  Binary cache (serialization / deserialization)                    */
    /* ================================================================== */

    /**
     * @brief Load the pulsegen-stage cache for a sequence path.
     *
     * Reads the COMMON + SHAPES sections only -- neither the per-instance
     * event tables (INSTANCES) nor the execution stream (SCANLOOP), which
     * scale with the scan length. Everything this stage resolves per
     * (segment, block position) is frozen into the segment definitions at
     * parse time. This stage does not enforce source-size matching.
     */
    int pulseg_load_geninstructions_cache(pulseg_collection **out_coll, const char *seq_path);

    /**
     * @brief Load the scan-stage cache for a sequence path.
     *
     * Reads the COMMON + INSTANCES + ROTATIONS + SHAPES + SCANLOOP sections
     * of the cache file derived from @p seq_path. This stage does not enforce
     * source-size matching.
     */
    int pulseg_load_scanloop_cache(pulseg_collection **out_coll, const char *seq_path);

    /**
     * @brief Delete the cache file associated with a sequence path.
     *
     * It is not an error if the cache file does not exist.
     */
    int pulseg_clear_cache(const char *seq_path);

    /**
     * @brief Save a loaded collection to a binary cache file.
     *
     * @param[in]  coll          Collection to save.
     * @param[in]  path          Output file path (e.g. "my_sequence.pseg").
     * @param[in]  source_size   Size (in bytes) of the original .seq buffer
     *                            used to load this collection.  Written into
     *                            the cache header for integrity validation on
     *                            reload.
     * @return PULSEG_SUCCESS on success, negative on failure.
     */
    int pulseg_save_cache(const pulseg_collection *coll, const char *path, int source_size);

    /**
     * @brief Load a collection from a binary cache file.
     *
     * The caller must allocate a pulseg_collection (e.g. via calloc)
     * before calling this function.  On success the collection is populated
     * from the cache.  On failure the collection is unchanged.
     *
     * @param[out] coll          Pre-allocated collection to populate.
     * @param[in]  path          Cache file path.
     * @param[in]  source_size   Expected size (bytes) of the original .seq
     *                            buffer.  Must match the value stored in the
     *                            cache header (guards against stale caches).
     * @return PULSEG_SUCCESS on success, negative on failure.
     */
    int pulseg_load_cache(pulseg_collection *coll, const char *path, int source_size);

    /* ================================================================== */
    /*  Freq-mod unified cache (embedded in collection cache)             */
    /* ================================================================== */

    /**
     * @brief Append an opaque vendor blob as the VENDOR section.
     *
     * Invokes @p opts->vendor_section_write_fn(opts->vendor_section_ctx, &buf,
     * &len) to obtain a caller-allocated (PULSEG_ALLOC'd) buffer, writes it
     * verbatim as a length-prefixed section, then frees the buffer. A NULL
     * @c vendor_section_write_fn is not an error -- it simply means no
     * VENDOR section is written. GE leaves this callback unset.
     *
     * @param[in] coll      Loaded collection (must already have a base cache
     *                       on disk, i.e. called after pulseg__write_cache).
     * @param[in] seq_path  Path to the .seq file.
     * @param[in] opts      Options carrying vendor_section_write_fn/ctx.
     * @return PULSEG_SUCCESS, negative error code, or PULSEG_SUCCESS with no
     *         section written when the callback is NULL.
     */
    int pulseg_write_vendor_cache_section(
        const pulseg_collection *coll,
        const char *seq_path,
        const pulseg_opts *opts);

    /**
     * @brief Read back the raw VENDOR section blob, if present.
     *
     * @param[in]  seq_path  Path to the .seq file.
     * @param[in]  cache_ext Cache file extension incl. dot, or NULL for the
     *                       public default.
     * @param[out] out_buf   Receives a PULSEG_ALLOC'd copy of the blob
     *                       (caller frees via PULSEG_FREE); NULL if absent.
     * @param[out] out_len   Receives the blob length in bytes (0 if absent).
     * @return PULSEG_SUCCESS if the cache was readable (even with no VENDOR
     *         section present, in which case *out_buf=NULL, *out_len=0);
     *         negative error code on I/O failure.
     */
    int pulseg_read_vendor_cache_section(
        const char *seq_path,
        const char *cache_ext,
        unsigned char **out_buf,
        int *out_len);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_CACHE_H */
