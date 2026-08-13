# How to write a new module

The shipped classes cover the standard excitation, preparation and readout
families. When yours is not among them, you write a `SequenceModule`. When
your *loop* is not among them, you write nothing at all — that is the point of
leaving the loop in the plugin.

## Do you actually need one?

Reach for these first:

- **A different pulse in a shipped readout** — a readout takes an RF *event*,
  so hand it whichever one you built: an excitation for a gradient echo, a
  refocusing pulse for the second half of a spin echo.
- **A different k-space traversal** — build the plan yourself and scale the
  published encodes. No readout has an opinion about which shots you play.
- **A single gradient or delay** — that is a plain Pulseq event, not a module.
  `seq.add_block(event)` already works.

Write a module when you have **several blocks that always travel together and
a design worth doing once**: gradients to solve, a TE to budget, an ADC to
land on two rasters. That is the whole contract.

## Implement the contract

A subclass provides exactly one thing: `init_module`. It reads like an
ordinary PyPulseq script — build events, assign `self.seq`, add blocks — and
everything else comes from the base class. Do not override `__init__`.

Here is a spatial saturation band: a slice-selective pulse plus a spoiler,
small enough to read whole.

```python
import numpy as np
import pulserver.pypulseq as pp
from pulserver import SequenceModule


class SaturationBand(SequenceModule):
    """A spatially selective saturation band.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits.
    thickness_m : float
        Band thickness (m).
    flip_angle_deg : float, optional
        Saturation flip angle.
    spoiling_cycles : float, optional
        Dephasing after the pulse, in cycles across ``voxel_size_m``.
    voxel_size_m : float, optional
        Length the spoiling is counted over (m).
    axis : {'z', 'x', 'y'}, optional
        Band normal.

    Attributes
    ----------
    rf : RfEvent
        The saturation pulse. Set ``rf.freq_offset`` to move the band.
    gz_select : TrapEvent
        Its selection gradient.
    gspoil : GradEvent
        The spoiler.
    """

    def init_module(
        self,
        system,
        thickness_m,
        flip_angle_deg=90.0,
        *,
        spoiling_cycles=4.0,
        voxel_size_m=1e-3,
        axis="z",
    ):
        rf, gz_select, _ = pp.make_slr_pulse(
            np.deg2rad(flip_angle_deg),
            duration=2e-3,
            slice_thickness=thickness_m,
            return_gz=True,
            use="saturation",
            system=system,
        )
        gz_select.channel = axis
        gspoil, _, _ = pp.make_crusher(spoiling_cycles, voxel_size_m, axis, system=system)

        self.seq = pp.Sequence(system)
        self.seq.add_block(rf, gz_select)
        self.seq.add_block(gspoil)

        self.center = float(rf.delay) + float(rf.center)
```

That is the entire module. `rf`, `gz_select` and `gspoil` are published under
those names because those are the names the constructor gave them, so a plugin
writes:

```python
band = SaturationBand(system, 20e-3)
band.rf.freq_offset = offset_hz(position_m)
for block in band.blocks:
    seq.add_block(*block)
```

## What the base class does for you

**Publication.** Every event that reaches a block is recorded and matched
against `init_module`'s locals. One distinct object under a name publishes as
that object; several publish as a list. See {doc}`../reference/design` for the
rule and its two escape hatches, `self.publish()` and `self.register(...)`.

**Timing.** `duration` sums the blocks unless you assign one. `center` is
yours to set — only the module knows where its isodelay or echo is.

**Inspection.** `blocks`, and the sequence-level analyses (`plot`,
`plot_kspace`, `calculate_kspace`, `calculate_pns`, `check_timing`,
`test_report`) forwarded to the sequence you built in.

## Three things that will bite

**Set `use=` on every pulse.** The trajectory core needs to know which pulses
open a readout period and refuses a sequence whose pulses do not say.

**Pulseq events cannot be copied.** They are C extension types; there is no
`copy.copy`. Build a second event rather than cloning one, and use
`pp.scale_grad`, which returns a new event.

**Name your intermediates with a leading underscore.** Publication skips
those, which is how a scratch variable that happens to hold a played event
stays out of the public surface.

## Subclassing a family

To add a non-Cartesian family — a different trajectory under the same
placement — subclass `NonCartesianReadout` rather than `SequenceModule`, and
it inherits the whole repetition layout: prewinder, acquisition, rewinder,
spoiler, TE and TR budget, and the `explicit` / `angles` path. You supply a
`NonCartesianGradient`. To add RF conveniences, subclass `RfModule`.
