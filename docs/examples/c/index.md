# C — writing an interpreter

The five stages of an interpreter, one page each, written against the C API.

```{toctree}
:maxdepth: 1

safety_gate
stage1_setup
stage2_prescription
stage3_preparation
stage4_waveform_generation
stage5_playout
```

Each page renders a file under `examples/c/`, compiled by
`scripts/build_examples.sh` with `-std=c89 -pedantic -Wall -Werror`.

Each file has a C++ counterpart of the same name in `examples/cpp/`, and the
two produce the same output on the same input. See {doc}`../cpp/index`.

## The vendor side is stubbed

`examples/c/vendor.h` supplies the scanner half — the hardware's limits, the
console, the waveform memory, the sequencer — as no-ops that print what a real
implementation would have done. A vendor integration replaces this file.

```{literalinclude} ../../../examples/c/vendor.h
:language: c
:caption: examples/c/vendor.h
```

## Running them

`stage3_preparation` writes the cache that `stage4_waveform_generation` and
`stage5_playout` read.

```console
$ bash scripts/build_examples.sh
$ cd build/examples-build/examples

$ ./stage3_preparation scan.seq
read: 1 subsequence(s), 2 segments, 0.1 s
checks: passed
slew alone: within limits
cache: written beside the sequence

$ ./stage4_waveform_generation scan.seq
cache: definitions, shapes and execution stream loaded
plan: resident, 10 distinct waveform(s) in 1 chunk(s)
    upload: axis 0, 110 points -> hardware id 2

$ ./stage5_playout scan.seq 3
    segment 0
        3080 us  rf     43.8 Hz  g (      0.0       0.0  266667.0) Hz/m
         480 us  rf      0.0 Hz  g (      0.0       0.0 -1688890.0) Hz/m
        2200 us  rf      0.0 Hz  g ( 546773.0  552640.0       0.0) Hz/m
```

The sequence above is a spiral gradient echo. Ten distinct waveforms cover the
whole scan: the arms are one waveform turned by a rotation.
