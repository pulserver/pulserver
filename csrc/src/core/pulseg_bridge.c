/**
 * @file pulseg_bridge.c
 * @brief POSIX bridge to pypulseq_host child process.
 *
 * Spawns the child via fork/exec with stdin/stdout pipes.
 * Communicates using the NimPulseqGUI persistent text protocol:
 *   LIST_PROTOCOL, VALIDATE, GENERATE, QUIT.
 *
 * Pure C89 + POSIX (no vendor dependencies).
 */

#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200112L
#endif

#include "pulseg_bridge.h"

#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/types.h>

/* Avoid <sys/wait.h> and <signal.h> — the GE host32 cross-compiler
 * sysroot lacks bits/signum-generic.h (pulled in transitively by both).
 * Provide the few POSIX definitions we actually need instead. */
#ifndef WNOHANG
#define WNOHANG 1
#endif
#ifndef SIGTERM
#define SIGTERM 15
#endif
extern pid_t waitpid(pid_t, int *, int);
extern int kill(pid_t, int);

#include <sys/select.h>
#include <sys/time.h>

/* ------------------------------------------------------------------ */
/*  Read timeout defaults (overridable via env)                       */
/* ------------------------------------------------------------------ */
#ifndef PULSEG_BRIDGE_READ_TIMEOUT_DEFAULT_SEC
#define PULSEG_BRIDGE_READ_TIMEOUT_DEFAULT_SEC 60
#endif
#ifndef PULSEG_BRIDGE_READ_TIMEOUT_GENERATE_SEC
#define PULSEG_BRIDGE_READ_TIMEOUT_GENERATE_SEC 300
#endif

/* Distinct rc for timeout vs. EOF/error. */
#define PULSEG_BRIDGE_RC_TIMEOUT (-2)

#include <stdarg.h>

/* Wire-level bridge log. Writes to
 * $PULSERVER_BASE_DIR/log/pulserver<pid>.log when PULSERVER_BASE_DIR is set,
 * otherwise to /tmp/pulserver<pid>.log. Silent on fopen failure. */
static void bridge_log(const char *fmt, ...)
{
    FILE *fp;
    char path[512];
    const char *base;
    va_list ap;

    base = getenv("PULSERVER_BASE_DIR");
    if (base && base[0])
        snprintf(path, sizeof(path), "%s/log/pulserver%d.log", base, (int)getpid());
    else
        snprintf(path, sizeof(path), "/tmp/pulserver%d.log", (int)getpid());
    fp = fopen(path, "a");
    if (!fp)
        return;
    va_start(ap, fmt);
    vfprintf(fp, fmt, ap);
    va_end(ap);
    fclose(fp);
}

static int bridge_env_timeout(const char *name, int default_sec)
{
    const char *s = getenv(name);
    int v;
    if (!s || !*s)
        return default_sec;
    v = atoi(s);
    if (v <= 0)
        return default_sec;
    return v;
}

/* ------------------------------------------------------------------ */
/*  Internal I/O helpers                                               */
/* ------------------------------------------------------------------ */

/** Write a string to fd. Returns 0 on success, -1 on error. */
static int write_str(int fd, const char *s)
{
    size_t len = strlen(s);
    while (len > 0)
    {
        ssize_t n = write(fd, s, len);
        if (n < 0)
        {
            if (errno == EINTR)
                continue;
            return -1;
        }
        s += n;
        len -= (size_t)n;
    }
    return 0;
}

/**
 * Read one line from fd into buf (up to bufsz-1 chars, NUL-terminated).
 * Strips the trailing newline. Uses select() with timeout_sec deadline.
 * Returns the line length (>= 0), -1 on EOF/error, or
 * PULSEG_BRIDGE_RC_TIMEOUT on timeout.
 */
static int read_line_to(int fd, char *buf, int bufsz, int timeout_sec)
{
    int pos = 0;
    fd_set rfds;
    struct timeval tv;
    struct timeval deadline;
    struct timeval now;
    int rs;

    if (timeout_sec <= 0)
        timeout_sec = PULSEG_BRIDGE_READ_TIMEOUT_DEFAULT_SEC;
    gettimeofday(&deadline, NULL);
    deadline.tv_sec += timeout_sec;

    while (pos < bufsz - 1)
    {
        ssize_t n;

        gettimeofday(&now, NULL);
        if (now.tv_sec > deadline.tv_sec ||
            (now.tv_sec == deadline.tv_sec && now.tv_usec >= deadline.tv_usec))
        {
            buf[pos] = '\0';
            return PULSEG_BRIDGE_RC_TIMEOUT;
        }
        tv.tv_sec = deadline.tv_sec - now.tv_sec;
        tv.tv_usec = deadline.tv_usec - now.tv_usec;
        if (tv.tv_usec < 0)
        {
            tv.tv_sec -= 1;
            tv.tv_usec += 1000000;
        }

        FD_ZERO(&rfds);
        FD_SET(fd, &rfds);
        rs = select(fd + 1, &rfds, NULL, NULL, &tv);
        if (rs < 0)
        {
            if (errno == EINTR)
                continue;
            return -1;
        }
        if (rs == 0)
        {
            buf[pos] = '\0';
            return PULSEG_BRIDGE_RC_TIMEOUT;
        }

        n = read(fd, buf + pos, 1);
        if (n < 0)
        {
            if (errno == EINTR)
                continue;
            return -1;
        }
        if (n == 0)
        {
            /* EOF */
            if (pos == 0)
                return -1;
            break;
        }
        if (buf[pos] == '\n')
            break;
        pos++;
    }
    buf[pos] = '\0';
    /* Strip trailing \r */
    if (pos > 0 && buf[pos - 1] == '\r')
    {
        buf[pos - 1] = '\0';
        pos--;
    }
    return pos;
}

/* Back-compat wrapper using default timeout. */
static int read_line(int fd, char *buf, int bufsz)
{
    int t = bridge_env_timeout("PULSERVER_BRIDGE_READ_TIMEOUT_SEC",
                               PULSEG_BRIDGE_READ_TIMEOUT_DEFAULT_SEC);
    return read_line_to(fd, buf, bufsz, t);
}

/**
 * Read lines from the child until [NimPulseqGUI Protocol End] is seen.
 * Accumulates the full preamble (including delimiters) into buf.
 * Returns total bytes written, or -1 on error.
 */
static int read_preamble_block(int fd, char *buf, int bufsz)
{
    int total = 0;
    char line[PULSEG_BRIDGE_LINE_MAX];
    int in_block = 0;
    int len;
    int need;

    while (1)
    {
        len = read_line(fd, line, (int)sizeof(line));
        if (len < 0)
            return total > 0 ? total : -1;

        /* Append line + newline to buffer */
        need = len + 1; /* line + '\n' */
        if (total + need >= bufsz)
            return -1; /* buffer overflow */
        memcpy(buf + total, line, len);
        buf[total + len] = '\n';
        total += need;
        buf[total] = '\0';

        if (strstr(line, "[NimPulseqGUI Protocol]") && !in_block)
        {
            in_block = 1;
        }
        if (strstr(line, "[NimPulseqGUI Protocol End]"))
        {
            break;
        }
    }
    return total;
}

/* ------------------------------------------------------------------ */
/*  Lifecycle                                                         */
/* ------------------------------------------------------------------ */

/* Private helper: fork + exec with caller-provided argv. */
static int bridge_do_open(pulseg_bridge *b, const char *exe_path,
                          const char **argv)
{
    int to_child[2];
    int from_child[2];
    pid_t pid;

    if (pipe(to_child) < 0)
        return -1;
    if (pipe(from_child) < 0)
    {
        close(to_child[0]);
        close(to_child[1]);
        return -1;
    }

    pid = fork();
    if (pid < 0)
    {
        close(to_child[0]);
        close(to_child[1]);
        close(from_child[0]);
        close(from_child[1]);
        return -1;
    }

    if (pid == 0)
    {
        /* Child process */
        const char *log_dir;
        char stderr_path[512];
        int stderr_fd;

        close(to_child[1]);
        close(from_child[0]);

        dup2(to_child[0], STDIN_FILENO);
        dup2(from_child[1], STDOUT_FILENO);
        close(to_child[0]);
        close(from_child[1]);

        /* Capture child stderr to a persistent log file so Nim/Python
         * tracebacks aren't lost. Without this, GENERATE/LIST_PROTOCOL
         * failures are silent — the parent just times out on read_line.
         * Path: $PULSERVER_BASE_DIR/log/pypulseq_host.stderr.log,
         * fallback /tmp. */
        log_dir = getenv("PULSERVER_BASE_DIR");
        if (log_dir && log_dir[0])
            snprintf(stderr_path, sizeof(stderr_path),
                     "%s/log/pypulseq_host.stderr.log", log_dir);
        else
            snprintf(stderr_path, sizeof(stderr_path),
                     "/tmp/pypulseq_host.stderr.log");
        stderr_fd = open(stderr_path,
                         O_WRONLY | O_CREAT | O_APPEND, 0644);
        if (stderr_fd >= 0)
        {
            dup2(stderr_fd, STDERR_FILENO);
            if (stderr_fd != STDERR_FILENO)
                close(stderr_fd);
        }

        execv(exe_path, (char *const *)argv);
        /* exec failed */
        _exit(127);
    }

    /* Parent */
    close(to_child[0]);
    close(from_child[1]);

    /* Mark parent-side bridge pipes close-on-exec so they are NOT inherited
     * by downstream subprocesses (e.g. GE's pulsegen/download fork+exec).
     * Otherwise a leaked stdin/stdout pipe to the persistent bridge keeps
     * the bridge's write end open in those children and can deadlock the
     * scanner's download step. File mode never opens the bridge, which is
     * why the same .seq downloads cleanly there. */
#ifdef FD_CLOEXEC
    {
        int _fl;
        _fl = fcntl(to_child[1], F_GETFD);
        if (_fl >= 0)
            fcntl(to_child[1], F_SETFD, _fl | FD_CLOEXEC);
        _fl = fcntl(from_child[0], F_GETFD);
        if (_fl >= 0)
            fcntl(from_child[0], F_SETFD, _fl | FD_CLOEXEC);
    }
#endif

    b->pid = (int)pid;
    b->to_child = to_child[1];
    b->from_child = from_child[0];
    return 0;
}

int pulseg_bridge_open(pulseg_bridge *b,
                          const char *exe_path,
                          const char *script_path)
{
    const char *argv[6];

    if (!b || !exe_path || !script_path)
        return -1;
    memset(b, 0, sizeof(*b));

    argv[0] = exe_path;
    argv[1] = "--persistent";
    argv[2] = "--script";
    argv[3] = script_path;
    argv[4] = (const char *)NULL;

    return bridge_do_open(b, exe_path, argv);
}

int pulseg_bridge_open_with_opts(pulseg_bridge *b,
                                    const char *exe_path,
                                    const char *script_path,
                                    const pulseg_opts *opts)
{
    /* All declarations at top (C89) */
    const char *argv[24];
    int n;
    /* Per-flag string buffers (64 bytes each) */
    char buf_gamma[64];
    char buf_b0[64];
    char buf_maxgrad[64];
    char buf_maxslew[64];
    char buf_rfraster[64];
    char buf_gradraster[64];
    char buf_adcraster[64];
    char buf_blockraster[64];

    if (!b || !exe_path || !script_path)
        return -1;
    memset(b, 0, sizeof(*b));

    if (!opts)
    {
        /* No limits available: fall back to plain open */
        return pulseg_bridge_open(b, exe_path, script_path);
    }

    n = 0;
    argv[n++] = exe_path;
    argv[n++] = "--persistent";
    argv[n++] = "--script";
    argv[n++] = script_path;

    if (opts->gamma_hz_per_t > 0.0f)
    {
        sprintf(buf_gamma, "--gamma=%.8g", (double)opts->gamma_hz_per_t);
        argv[n++] = buf_gamma;
    }
    if (opts->b0_t > 0.0f)
    {
        sprintf(buf_b0, "--B0=%.8g", (double)opts->b0_t);
        argv[n++] = buf_b0;
    }
    if (opts->max_grad_hz_per_m > 0.0f)
    {
        sprintf(buf_maxgrad, "--maxGrad=%.8g", (double)opts->max_grad_hz_per_m);
        argv[n++] = buf_maxgrad;
        argv[n++] = "--gradUnit=Hz/m";
    }
    if (opts->max_slew_hz_per_m_per_s > 0.0f)
    {
        sprintf(buf_maxslew, "--maxSlew=%.8g",
                (double)opts->max_slew_hz_per_m_per_s);
        argv[n++] = buf_maxslew;
        argv[n++] = "--slewUnit=Hz/m/s";
    }
    if (opts->rf_raster_us > 0.0f)
    {
        sprintf(buf_rfraster, "--rfRasterTime=%.8g",
                (double)(opts->rf_raster_us * 1.0e-6f));
        argv[n++] = buf_rfraster;
    }
    if (opts->grad_raster_us > 0.0f)
    {
        sprintf(buf_gradraster, "--gradRasterTime=%.8g",
                (double)(opts->grad_raster_us * 1.0e-6f));
        argv[n++] = buf_gradraster;
    }
    if (opts->adc_raster_us > 0.0f)
    {
        sprintf(buf_adcraster, "--adcRasterTime=%.8g",
                (double)(opts->adc_raster_us * 1.0e-6f));
        argv[n++] = buf_adcraster;
    }
    if (opts->block_raster_us > 0.0f)
    {
        sprintf(buf_blockraster, "--blockDurationRaster=%.8g",
                (double)(opts->block_raster_us * 1.0e-6f));
        argv[n++] = buf_blockraster;
    }
    argv[n] = (const char *)NULL;

    return bridge_do_open(b, exe_path, argv);
}

void pulseg_bridge_close(pulseg_bridge *b)
{
    if (!b || b->pid <= 0)
        return;

    /* Send QUIT */
    write_str(b->to_child, "QUIT\n");

    close(b->to_child);
    close(b->from_child);
    b->to_child = -1;
    b->from_child = -1;

    /* Wait for child with timeout: try non-blocking first */
    {
        int status;
        struct timespec ts;
        pid_t w = waitpid((pid_t)b->pid, &status, WNOHANG);
        if (w == 0)
        {
            /* Child still running; give it a moment then SIGTERM */
            ts.tv_sec = 0;
            ts.tv_nsec = 100000000L; /* 100 ms */
            nanosleep(&ts, NULL);
            w = waitpid((pid_t)b->pid, &status, WNOHANG);
            if (w == 0)
            {
                kill((pid_t)b->pid, SIGTERM);
                waitpid((pid_t)b->pid, &status, 0);
            }
        }
    }

    b->pid = 0;
}

int pulseg_bridge_alive(const pulseg_bridge *b)
{
    int status;
    if (!b || b->pid <= 0)
        return 0;
    if (waitpid((pid_t)b->pid, &status, WNOHANG) == 0)
        return 1;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Commands                                                          */
/* ------------------------------------------------------------------ */

int pulseg_bridge_list_protocol(pulseg_bridge *b,
                                   pulseg_protocol *out)
{
    char preamble[4096];
    int len;
    char resp_line[PULSEG_BRIDGE_LINE_MAX];
    int rc;

    if (!b || !out || b->pid <= 0)
        return -1;

    bridge_log("[BRIDGE] >LIST_PROTOCOL\n");
    if (write_str(b->to_child, "LIST_PROTOCOL\n") < 0)
    {
        bridge_log("[BRIDGE] write LIST_PROTOCOL failed errno=%d\n", errno);
        return -1;
    }

    /* Child responds: "PROTOCOL\n" then the preamble block */
    rc = read_line(b->from_child, resp_line, (int)sizeof(resp_line));
    if (rc < 0)
    {
        bridge_log("[BRIDGE] <LIST_PROTOCOL %s\n",
                   rc == PULSEG_BRIDGE_RC_TIMEOUT ? "TIMEOUT" : "EOF/ERROR");
        return -1;
    }
    bridge_log("[BRIDGE] <%s\n", resp_line);
    if (strncmp(resp_line, "PROTOCOL", 8) != 0)
        return -1;

    len = read_preamble_block(b->from_child, preamble, (int)sizeof(preamble));
    if (len <= 0)
    {
        bridge_log("[BRIDGE] <LIST_PROTOCOL preamble read failed len=%d\n", len);
        return -1;
    }
    bridge_log("[BRIDGE] <LIST_PROTOCOL preamble %d bytes\n", len);

    return pulseg_protocol_parse(out, preamble);
}

/** Send the serialized protocol (preamble) to the child. */
static int send_protocol(pulseg_bridge *b,
                         const pulseg_protocol *proto)
{
    char buf[4096];
    int n = pulseg_protocol_serialize(proto, buf, (int)sizeof(buf));
    if (n <= 0)
        return -1;
    return write_str(b->to_child, buf);
}

int pulseg_bridge_validate(pulseg_bridge *b,
                              float *duration,
                              char *info, int infosz,
                              const pulseg_protocol *proto)
{
    char resp[PULSEG_BRIDGE_LINE_MAX];
    int rc;

    if (!b || !proto || b->pid <= 0)
        return -1;

    bridge_log("[BRIDGE] >VALIDATE\n");
    if (write_str(b->to_child, "VALIDATE\n") < 0)
    {
        bridge_log("[BRIDGE] write VALIDATE failed errno=%d\n", errno);
        return -1;
    }
    if (send_protocol(b, proto) < 0)
    {
        bridge_log("[BRIDGE] send_protocol (VALIDATE) failed\n");
        return -1;
    }

    /* Response: "VALID <duration> <info>" or "INVALID <info>" */
    rc = read_line(b->from_child, resp, (int)sizeof(resp));
    if (rc < 0)
    {
        bridge_log("[BRIDGE] <VALIDATE %s\n",
                   rc == PULSEG_BRIDGE_RC_TIMEOUT ? "TIMEOUT" : "EOF/ERROR");
        return -1;
    }
    bridge_log("[BRIDGE] <%s\n", resp);

    if (strncmp(resp, "VALID ", 6) == 0)
    {
        /* Parse "VALID 5.32 TA = 5.32 s" */
        const char *p = resp + 6;
        if (duration)
            *duration = (float)atof(p);
        /* Find start of info (after the float) */
        while (*p && *p != ' ')
            p++;
        if (*p == ' ')
            p++;
        if (info && infosz > 0)
        {
            strncpy(info, p, infosz - 1);
            info[infosz - 1] = '\0';
        }
        return 1;
    }
    if (strncmp(resp, "INVALID", 7) == 0)
    {
        const char *p = resp + 7;
        if (*p == ' ')
            p++;
        if (info && infosz > 0)
        {
            strncpy(info, p, infosz - 1);
            info[infosz - 1] = '\0';
        }
        return 0;
    }

    return -1; /* unexpected response */
}

int pulseg_bridge_generate(pulseg_bridge *b,
                              const pulseg_protocol *proto,
                              const char *output_path)
{
    char cmd[1024];
    char resp[PULSEG_BRIDGE_LINE_MAX];
    int rc;
    int gen_timeout;

    if (!b || !proto || !output_path || b->pid <= 0)
        return -1;

    /* "GENERATE <path>\n" */
    if (strlen(output_path) > sizeof(cmd) - 16)
        return -1;
    sprintf(cmd, "GENERATE %s\n", output_path);

    bridge_log("[BRIDGE] >GENERATE %s\n", output_path);
    if (write_str(b->to_child, cmd) < 0)
    {
        bridge_log("[BRIDGE] write GENERATE failed errno=%d\n", errno);
        return -1;
    }
    if (send_protocol(b, proto) < 0)
    {
        bridge_log("[BRIDGE] send_protocol (GENERATE) failed\n");
        return -1;
    }

    /* Response: "GENERATED <path>" or "ERROR <msg>".
     * Sequence generation can be slow — use a longer timeout. */
    gen_timeout = bridge_env_timeout("PULSERVER_BRIDGE_GENERATE_TIMEOUT_SEC",
                                     PULSEG_BRIDGE_READ_TIMEOUT_GENERATE_SEC);
    rc = read_line_to(b->from_child, resp, (int)sizeof(resp), gen_timeout);
    if (rc < 0)
    {
        bridge_log("[BRIDGE] <GENERATE %s (timeout=%ds)\n",
                   rc == PULSEG_BRIDGE_RC_TIMEOUT ? "TIMEOUT" : "EOF/ERROR",
                   gen_timeout);
        return -1;
    }
    bridge_log("[BRIDGE] <%s\n", resp);

    if (strncmp(resp, "GENERATED", 9) == 0)
        return 0;
    return -1;
}
