/**
 * @file pulseg_bridge.h
 * @brief Vendor-neutral POSIX bridge to pypulseq_host child process.
 *
 * Manages a persistent child process (pypulseq_host --persistent)
 * communicating over stdin/stdout pipes.  Pure C89 + POSIX.
 */

#ifndef PULSEG_BRIDGE_H
#define PULSEG_BRIDGE_H

#include "pulseg_protocol.h"
#include "pulseg_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Bridge handle                                                     */
/* ------------------------------------------------------------------ */

#define PULSEG_BRIDGE_LINE_MAX 512

typedef struct pulseg_bridge {
    int    pid;          /* child PID (0 = not running) */
    int    to_child;     /* fd: host writes commands here */
    int    from_child;   /* fd: host reads responses here */
    char   line_buf[PULSEG_BRIDGE_LINE_MAX];
} pulseg_bridge;

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                         */
/* ------------------------------------------------------------------ */

/**
 * Spawn the bridge child process.
 *
 * @param b           Bridge handle (caller allocates, will be zeroed).
 * @param exe_path    Path to pypulseq_host executable.
 * @param script_path Path to the Python plugin script.
 * @return 0 on success, -1 on error (errno set).
 */
int pulseg_bridge_open(pulseg_bridge* b,
                           const char* exe_path,
                           const char* script_path);

/**
 * Spawn the bridge child process, passing scanner system limits as CLI
 * flags to nimpulseqgui (--gamma, --B0, --maxGrad, --maxSlew, rasters).
 *
 * @param b           Bridge handle (caller allocates, will be zeroed).
 * @param exe_path    Path to pypulseq_host executable.
 * @param script_path Path to the Python plugin script.
 * @param opts        Scanner limits (may be NULL, falls back to plain open).
 * @return 0 on success, -1 on error (errno set).
 */
int pulseg_bridge_open_with_opts(pulseg_bridge* b,
                                     const char* exe_path,
                                     const char* script_path,
                                     const pulseg_opts* opts);

/**
 * Send QUIT and close the bridge.  Waits for child to exit.
 * Safe to call on an already-closed or zero-initialized handle.
 */
void pulseg_bridge_close(pulseg_bridge* b);

/**
 * Check whether the child process is still alive.
 * @return 1 if alive, 0 if dead or not started.
 */
int pulseg_bridge_alive(const pulseg_bridge* b);

/* ------------------------------------------------------------------ */
/*  Commands                                                          */
/* ------------------------------------------------------------------ */

/**
 * LIST_PROTOCOL: request the default protocol from the plugin.
 *
 * @param b   Open bridge handle.
 * @param out Populated on success with the default protocol.
 * @return Number of parsed params (>= 0), or -1 on error.
 */
int pulseg_bridge_list_protocol(pulseg_bridge* b,
                                    pulseg_protocol* out);

/**
 * VALIDATE: send a protocol and check validity.
 *
 * @param b        Open bridge handle.
 * @param proto    Protocol to validate.
 * @param duration If non-NULL, set to reported duration on success.
 * @param info     If non-NULL, info string buffer (caller provides).
 * @param infosz   Size of info buffer.
 * @return 1 if valid, 0 if invalid, -1 on comm error.
 */
int pulseg_bridge_validate(pulseg_bridge* b,
                               const pulseg_protocol* proto,
                               float* duration,
                               char* info, int infosz);

/**
 * GENERATE: send a protocol and request a .seq file.
 *
 * @param b           Open bridge handle.
 * @param proto       Protocol to use.
 * @param output_path Where the child should write the .seq file.
 * @return 0 on success, -1 on error.
 */
int pulseg_bridge_generate(pulseg_bridge* b,
                               const pulseg_protocol* proto,
                               const char* output_path);

#ifdef __cplusplus
}
#endif

#endif /* PULSEG_BRIDGE_H */
