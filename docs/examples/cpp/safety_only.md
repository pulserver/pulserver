# Safety checking, in an interpreter you already have

You have an interpreter. It parses `.seq` files, it plays them, and it works.
What it probably does not have is a reason to trust that the file it is about
to play is inside the system's limits — the format carries no hardware
description, so nothing in it can tell you.

That check is the smallest useful thing Pulserver does, and it does not
require adopting anything else: no cache, no IR on disk, no change to how you
play the sequence.

## The whole integration

```cpp
#include <pulseg/pulseg.hpp>

bool sequence_is_playable(const char* seq_path, std::string& why_not)
{
    pulseg::Opts opts;
    opts.gamma_hz_per_t        = 42'577'478.0f;
    opts.b0_t                  = 3.0f;
    opts.max_grad_hz_per_m     = opts.gamma_hz_per_t * 0.040f;   // 40 mT/m
    opts.max_slew_hz_per_m_per_s = opts.gamma_hz_per_t * 180.0f; // 180 T/m/s
    opts.rf_raster_us = 1.0f;
    opts.grad_raster_us = 10.0f;
    opts.adc_raster_us = 0.1f;
    opts.block_raster_us = 10.0f;

    try
    {
        pulseg::Collection scan(seq_path, opts);
        scan.check_consistency();
        scan.check_safety();
        return true;
    }
    catch (const pulseg::Error& error)
    {
        why_not = error.what();
        return false;
    }
}
```

That is the integration. `Collection` parses the file (following any
`NextSequence` chain), derives the structure, and holds it in memory;
`check_consistency` verifies the sequence is internally coherent — rasters
respected, events inside their blocks, the structure actually periodic;
`check_safety` gates it against the limits in `opts`. Both throw
`pulseg::Error` carrying which check failed, on which block, and by how much.
The object frees itself.

Nothing is written to disk. If you want the parsed structure cached for a
later replay that is available, but it is a separate decision — see
{doc}`interpreter`.

## What the gate covers

| Check | Refuses |
|---|---|
| gradient amplitude | any axis, and the vector magnitude, above `max_grad` |
| slew rate | the same, above `max_slew`, including across block boundaries |
| gradient continuity | a waveform that starts or ends at an amplitude the previous or next block does not meet |
| RF consistency | a repetition whose RF differs from the others in a way the safety model cannot bound |
| ADC | windows off the dwell raster, or overlapping RF |

Two further checks are opt-in, because they need site data the file cannot
carry:

```cpp
// Peripheral nerve stimulation, against a coil's own response model.
pulseg::PnsParams pns{ /* chronaxie_us */ 360.0f,
                       /* saturation   */ 4.25e8f,
                       /* effective_len*/ 0.333f };
scan.check_safety({}, &pns, /* threshold_percent */ 80.0f);

// Mechanical resonance, against the system's forbidden bands.
std::vector<pulseg::ForbiddenBand> bands = {
    {1100.0f, 1250.0f, 0.0f},   // Hz, Hz, max amplitude Hz/m
};
scan.check_safety(bands);
```

The reasoning behind each — what is computed, and why the threshold is what
it is — is in {doc}`../../explanations/safety/index`.

## Where to put it

Before download. The point of checking at all is that the operator learns at
the console that a protocol is unplayable, rather than the system aborting at
the magnet with the patient in it. If your interpreter has a
prepare-then-play split, `check_safety` belongs at the end of prepare.

It costs roughly the parse: for a 100 000-block scan the parse dominates and
the checks are a fraction of it — {doc}`../../explanations/performance/index`
has the numbers.

## Building against it

Pulserver's C++ headers are header-only over the C library, so linking is one
static library and one include path:

```cmake
add_subdirectory(pulserver/csrc)
target_link_libraries(my_interpreter PRIVATE pulseg)
target_include_directories(my_interpreter PRIVATE pulserver/cxx/include)
```

There are no third-party dependencies — the parser, the structure detection
and the safety engine are C89 with no allocation beyond what the sequence
itself requires.
