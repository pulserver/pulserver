# C++ — writing an interpreter

The same five stages as the {doc}`C examples <../c/index>`, written against
the C++ API. Each page renders a file under `examples/cpp/`, compiled by
`scripts/build_examples.sh` with `-Wall -Wextra -Werror` under C++17.

```{toctree}
:maxdepth: 1

safety_gate
stage1_setup
stage2_prescription
stage3_preparation
stage4_waveform_generation
stage5_playout
```

Each file has a C counterpart of the same name in `examples/c/`, and the two
produce the same output on the same input.

| Page | C counterpart |
|---|---|
| {doc}`safety_gate` | {doc}`../c/safety_gate` |
| {doc}`stage1_setup` | {doc}`../c/stage1_setup` |
| {doc}`stage2_prescription` | {doc}`../c/stage2_prescription` |
| {doc}`stage3_preparation` | {doc}`../c/stage3_preparation` |
| {doc}`stage4_waveform_generation` | {doc}`../c/stage4_waveform_generation` |
| {doc}`stage5_playout` | {doc}`../c/stage5_playout` |

## The vendor side is stubbed

`examples/cpp/vendor.hpp` supplies the scanner half — the hardware's limits,
the console, the waveform memory, the sequencer — as no-ops that print what a
real implementation would have done. A vendor integration replaces this file.

```{literalinclude} ../../../examples/cpp/vendor.hpp
:language: cpp
:caption: examples/cpp/vendor.hpp
```

## Running them

`stage3_preparation` writes the cache that `stage4_waveform_generation` and
`stage5_playout` read.

```console
$ bash scripts/build_examples.sh
$ cd build/examples-build/examples
$ ./cpp_stage3_preparation scan.seq
$ ./cpp_stage4_waveform_generation scan.seq
$ ./cpp_stage5_playout scan.seq 4
```
