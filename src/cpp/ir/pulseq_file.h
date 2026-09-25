/**
 * @file pulseq_file.h
 * @brief Lifecycle of a pulseq_file and the accessors over it, on the host.
 *
 * Implemented in pulseq_file.cpp beside this header; the conversion passes
 * resolve blocks through these. Not part of the library a scanner links.
 */

#ifndef PULSERVER_IR_PULSEQ_FILE_H
#define PULSERVER_IR_PULSEQ_FILE_H

#include "pulseq.h"

#ifdef __cplusplus
extern "C"
{
#endif

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

    /** @brief Release everything a file holds and leave it initialised. */
    void pulseq_file_free(pulseq_file *seq);

    /**
     * @brief Resolve a block's raw content ids (rf/gx/gy/gz/adc/extension
     * chain head) from the BLOCKS table, without inlining event data.
     */
    int pulseq_get_raw_block_content_ids(
        const pulseq_file *seq,
        pulseq_raw_block *block,
        int block_index,
        int parse_extensions);

    /**
     * @brief The TRIGGERS row a block's extension chain names, counted from
     * 0; the last one when it names several, -1 for none.
     */
    int pulseq_block_trigger(const pulseq_file *seq, const pulseq_raw_block *raw);

#ifdef __cplusplus
}
#endif

#endif /* PULSERVER_IR_PULSEQ_FILE_H */
