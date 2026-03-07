# bridge/ — Python & MATLAB Host Executables for pulserver

Pre-compiled Nim executables that embed CPython (or MCR) and load
user-supplied sequence plugins at runtime. Uses **stock nimpulseqgui**
as a Nimble dependency — no source modifications needed.

## Architecture

```
                   ┌──────────────────────────┐
                   │  nimpulseqgui (stock)     │
                   │  - GUI property editor    │
                   │  - makeSequenceExe()      │
                   │  - bool validator          │
                   └──────────┬───────────────┘
                              │ Nimble dependency
              ┌───────────────┴───────────────┐
              │       bridge_common.nim        │
              │  - ValidationResult (rich)     │
              │  - readProtocolFromString()    │
              │  - persistent-mode helpers     │
              │  - preamble formatting         │
              └───────┬──────────────┬────────┘
                      │              │
           ┌──────────┴──┐    ┌──────┴──────────┐
           │pypulseq_host│    │  matlab_host     │
           │ --script .py│    │  --script .m     │
           │ nimpy bridge │    │  MCR/subprocess  │
           └─────────────┘    └─────────────────┘
```

## Modes

| Flag | Description | Validator output |
|------|-------------|-----------------|
| *(none)* | Interactive GUI via `makeSequenceExe` | `bool` (stock nimpulseqgui) |
| `--no-gui` | Headless write via `makeSequenceExe` | `bool` |
| `--validate-only` | Print JSON `{valid, duration, info}`, exit | **Rich** `ValidationResult` |
| `--list-protocol` | Print default preamble, exit | N/A |
| `--persistent` | Stdin/stdout command loop for GE `popen()` | **Rich** per-command |

Headless modes (`--validate-only`, `--persistent`) are handled **by the bridge
itself**, bypassing `makeSequenceExe`, so the GE driver always gets the full
`ValidationResult{valid, duration, info}`.

## Persistent Protocol (stdin/stdout)

```
→ VALIDATE\n
→ [NimPulseqGUI Protocol]\n
→ TE: 5.0\n
→ ...\n
→ [NimPulseqGUI Protocol End]\n
← VALID 5.32 TA = 5.32 s\n

→ GENERATE /tmp/out.seq\n
→ [NimPulseqGUI Protocol]\n
→ ...\n
→ [NimPulseqGUI Protocol End]\n
← GENERATED /tmp/out.seq\n

→ LIST_PROTOCOL\n
← PROTOCOL\n
← [NimPulseqGUI Protocol]\n
← ...\n
← [NimPulseqGUI Protocol End]\n

→ QUIT\n
```

## Building

```bash
cd bridge/
nimble build
```

Compile-time configuration:

```bash
nim c -d:pythonHome=./python pypulseq_host.nim
```
