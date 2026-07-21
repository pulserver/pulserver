/**
 * @file pulseg_bridge.h
 * @brief Out-of-process bridge to a persistent pypulseq_host child.
 *
 * Sequence *generation* is Python (pypulseq); everything downstream of a .seq
 * file is this C library. The bridge is the seam: it spawns
 * @c pypulseq_host @c --persistent, keeps it alive across a session, and
 * exchanges line-oriented commands over its stdin/stdout pipes.
 *
 * Vendor-neutral, pure C89 + POSIX. All entry points return 0 / a non-negative
 * count on success and -1 on error (errno set), NOT the @c PULSEG_ERR_* codes
 * used by the rest of the library -- failures here are process and I/O
 * failures, not sequence-model failures.
 */

#ifndef PULSEG_BRIDGE_H
#define PULSEG_BRIDGE_H

#include "pulseg_protocol.h"
#include "pulseg_types.h"

#ifdef __cplusplus
extern "C"
{
#endif

    /* ================================================================== */
    /*  Bridge handle                                                     */
    /* ================================================================== */

#define PULSEG_BRIDGE_LINE_MAX 512

    /** @brief Handle for one live pypulseq_host child process. */
    typedef struct pulseg_bridge
    {
        int pid;        /**< child PID, 0 when not running */
        int to_child;   /**< fd the host writes commands to */
        int from_child; /**< fd the host reads responses from */
    } pulseg_bridge;

    /* ================================================================== */
    /*  Lifecycle                                                         */
    /* ================================================================== */

    /**
     * @brief Spawn the bridge child, forwarding scanner limits on its command
     *        line (--gamma, --B0, --maxGrad, --maxSlew, and the rasters).
     *
     * @param[out] b            Caller-allocated handle; zeroed on entry.
     * @param[in]  exe_path     Path to the pypulseq_host executable.
     * @param[in]  script_path  Path to the Python plugin script.
     * @param[in]  opts         Scanner limits; NULL spawns with none, leaving
     *                          the plugin on its own defaults.
     * @return 0 on success, -1 on error (errno set).
     */
    int pulseg_bridge_open_with_opts(
        pulseg_bridge *b,
        const char *exe_path,
        const char *script_path,
        const pulseg_opts *opts);

    /**
     * @brief Send QUIT, close both pipes, and reap the child.
     *
     * Escalates to SIGTERM if the child has not exited after ~100 ms. Safe on
     * an already-closed or zero-initialized handle.
     *
     * @param[in,out] b  Bridge handle.
     */
    void pulseg_bridge_close(pulseg_bridge *b);

    /* ================================================================== */
    /*  Commands                                                          */
    /* ================================================================== */

    /**
     * @brief LIST_PROTOCOL: ask the plugin for its default protocol.
     *
     * @param[in,out] b    Open bridge handle.
     * @param[out]    out  Receives the default protocol.
     * @return Number of parsed parameters (>= 0), or -1 on error.
     */
    int pulseg_bridge_list_protocol(pulseg_bridge *b, pulseg_protocol *out);

    /**
     * @brief VALIDATE: ask the plugin whether a protocol is playable.
     *
     * @param[in,out] b         Open bridge handle.
     * @param[out]    duration  Reported scan duration; NULL to ignore.
     * @param[out]    info      Buffer for the plugin's message; NULL to ignore.
     * @param[in]     infosz    Capacity of @p info in bytes.
     * @param[in]     proto     Protocol to validate.
     * @return 1 if valid, 0 if invalid, -1 on communication error.
     */
    int pulseg_bridge_validate(
        pulseg_bridge *b,
        float *duration,
        char *info,
        int infosz,
        const pulseg_protocol *proto);

    /**
     * @brief GENERATE: ask the plugin to write a .seq file for a protocol.
     *
     * @param[in,out] b            Open bridge handle.
     * @param[in]     proto        Protocol to generate from.
     * @param[in]     output_path  Where the child writes the .seq file.
     * @return 0 on success, -1 on error.
     */
    int pulseg_bridge_generate(
        pulseg_bridge *b,
        const pulseg_protocol *proto,
        const char *output_path);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_BRIDGE_H */
