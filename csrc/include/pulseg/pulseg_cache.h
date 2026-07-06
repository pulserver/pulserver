/**
 * @file pulseg_cache.h
 * @brief Binary (.pge) cache read/write.
 *
 * Split out of the former pulseg_methods.h (Stage 1 layout normalization).
 * All functions use the pulseg_ prefix and are declared extern "C" when
 * compiled with a C++ compiler.
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
     * Reads the COMMON + SHAPES sections of the cache file derived from
     * @p seq_path. This stage does not enforce source-size matching.
     */
    int pulseg_load_geninstructions_cache(
        pulseg_collection **out_coll,
        const char *seq_path);

    /**
     * @brief Load the scan-stage cache for a sequence path.
     *
     * Reads the COMMON + ROTATIONS + SCANLOOP sections of the cache file
     * derived from @p seq_path. This stage does not enforce source-size
     * matching.
     */
    int pulseg_load_scanloop_cache(
        pulseg_collection **out_coll,
        const char *seq_path);

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
     * @param[in]  path          Output file path (e.g. "my_sequence.pge").
     * @param[in]  source_size   Size (in bytes) of the original .seq buffer
     *                            used to load this collection.  Written into
     *                            the cache header for integrity validation on
     *                            reload.
     * @return PULSEG_SUCCESS on success, negative on failure.
     */
    int pulseg_save_cache(
        const pulseg_collection *coll,
        const char *path,
        int source_size);

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
    int pulseg_load_cache(
        pulseg_collection *coll,
        const char *path,
        int source_size);

    /* ================================================================== */
    /*  Freq-mod unified cache (embedded in collection cache)             */
    /* ================================================================== */

    /**
     * @brief Load freq-mod data from the collection cache (FREQMOD section).
     *
     * On success, populates coll->freq_mod.  Returns an error code if the
     * freq-mod section is absent or empty.
     */
    int pulseg_load_freq_mod_cache(
        pulseg_collection *coll,
        const char *seq_path);

    /**
     * @brief Append freq-mod data to an existing collection cache.
     *
     * Opens the cache file for the given .seq path, writes freq-mod data
     * at the end, and updates the FREQMOD section index entry.
     */
    int pulseg_write_freq_mod_cache(
        const pulseg_collection *coll,
        const char *seq_path);

    /**
     * @brief Append the sequence description as the SEQDESC section to the binary cache.
     *
     * Must be called after the collection is loaded and all descriptors computed.
     *
     * @param[in] coll      Loaded collection.
     * @param[in] seq_path  Path to the .seq file (cache is .seq → .pge).
     * @return PULSEG_SUCCESS or negative error code.
     */
    int pulseg_write_sequence_description_cache(const pulseg_collection *coll,
                                                   const char *seq_path);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_CACHE_H */
