[![Tests](https://github.com/pulserver/pulserver/actions/workflows/test-ci.yml/badge.svg)](https://github.com/pulserver/pulserver/actions/workflows/test-ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pulserver.svg)](https://pypi.org/project/pulserver/)
[![Docs: stable](https://img.shields.io/badge/docs-stable-2b76ad)](https://pulserver.github.io/pulserver/stable/)
[![Virtual scanner: try it](https://img.shields.io/badge/virtual%20scanner-try%20it-ffbd28)](https://pulserver.github.io/MaRGE/)
[![License: MIT](https://img.shields.io/badge/license-MIT-ffbd28.svg)](https://github.com/pulserver/pulserver/blob/main/LICENSE)

<p align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pulserver/pulserver/main/docs/_static/pulserver-logo-dark.svg">
  <img src="https://raw.githubusercontent.com/pulserver/pulserver/main/docs/_static/pulserver-logo.svg" alt="pulserver" width="580">
</picture></p>

pulserver lets you run your own sequence on a clinical scanner as if it were
a product sequence: the operator edits it in the scanner UI, and the images
come back to the console. Sequences are exchanged as Pulseq, so any tool that
writes a `.seq` file (pypulseqpp, MATLAB Pulseq, BART's `seq`...) can provide
one; pypulseqpp is the one pulserver uses natively.

You write two small Python plugins, one that designs the sequence and one that
reconstructs it. pulserver does everything in between: it answers the scanner
UI, checks the sequence, converts it into the form the scanner plays, and
routes the raw data back to your reconstruction.

<p align="center"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/pulserver/pulserver/main/docs/_static/architecture-dark.svg">
  <img src="https://raw.githubusercontent.com/pulserver/pulserver/main/docs/_static/architecture.svg" alt="pulserver architecture" width="900">
</picture></p>

The scanner side is a vendor interpreter, a small program on the scanner that
calls pulserver. pulserver itself never touches the hardware.

## How it works

1. The operator opens your sequence. The scanner asks pulserver which
   parameters to show (TE, TR, flip angle, matrix...), and pulserver reads them
   from your sequence plugin.
2. Every time the operator changes a value, pulserver asks your plugin
   whether the protocol is feasible and what it achieves (the TE, the TR, the
   scan time), and sends that back to the scanner. How your plugin answers is
   up to you: design one TR, or compute the times from the block durations.
   If the value is impossible, the scanner shows your error message instead.
3. When the operator presses *Scan*, pulserver designs the whole sequence,
   runs the checks (timing, gradient and slew limits, PNS, mechanical
   resonance, acoustic noise) and converts it into the cache the interpreter
   plays, shifted to the prescribed field of view.
4. The scanner plays it and streams the raw data to pulserver.
5. pulserver fills in everything the sequence knows and the scanner does not:
   the MRD header (encoding spaces, matrix, field of view), and on every
   readout its acquisition header, flags, encoding counters from the
   sequence's labels, and trajectory. Your recon plugin receives these
   enriched readouts, sorts them into k-space and reconstructs as soon as a
   slice or volume is complete.
6. Your plugin returns images; pulserver turns them into DICOM and sends them
   back to the console.

## The sequence plugin

A sequence plugin takes a pypulseqpp sequence function, a function that returns
a `Sequence`, and tells the scanner which of its arguments the operator can
edit:

```python
# sequences/gre.py
from pypulseqpp.sequences.sequence.gre2D_sequence import gre2d
from pulserver.design import Evaluation, FloatParam, SequencePlugin, TimeParam, UIParam


class Gre(SequencePlugin):
    app = gre2d
    protocol = {
        UIParam.TE: TimeParam("te", range_min=2000, range_max=20000),  # µs on the UI, s for gre2d
        UIParam.FLIP: FloatParam("flip_angle_deg", unit="deg", range_min=1, range_max=90),
    }

    def evaluate(self, system, protocol):
        one_line = self.app(system, **protocol.arguments, ry=128, n_acs_y=0, n_dummy=0)  # one line of 128
        return Evaluation(protocol.replace({UIParam.TE: one_line.definitions["TE"][0]}))
```

- `protocol` maps each scanner parameter to an argument of your function.
  Parameters you leave out keep the function's defaults.
- `evaluate` runs on every edit, so keep it fast. Here it designs one line
  and reads back the TE; it could equally compute the TR from block durations,
  or accept the requested TR when it is feasible. To refuse a value, raise;
  the message goes to the operator.
- The whole scan is built only when the operator starts it, by calling your
  function with the final protocol.

pulserver ships a plugin for each of pypulseqpp's 28 built-in sequences
(Cartesian, radial, spiral, PROPELLER, EPI, FSE, MPRAGE, bSSFP, ZTE), so
`--plugin gre2d` works without writing anything.

## The recon plugin

The reconstruction side follows Gadgetron and ISMRMRD: the data arrive as an
MRD stream (a header, then one acquisition per readout), and a recon plugin
handles them as they arrive, much like a Gadgetron gadget chain. Any framework
can do the maths; this one uses bartorch:

```python
# recon/pics.py
import torch
import bartorch.tools as bt

from bartorch import apps, priors
from pulserver import recon


class Pics(recon.ReconPlugin):
    def recon(self, context, branch, data):
        kspace = torch.from_numpy(data.data.kspace)  # (coils, y, x)
        maps = bt.ecalib(kspace, maps=1)
        image = apps.pics(kspace, maps, regularizers=priors.Wavelet((-1, -2), 0.005))
        return recon.ReconResult(image.abs().numpy())


PLUGIN = Pics()
```

- The plugin places each readout in k-space as it arrives, by its encoding
  counters, and `recon` is called once per slice or volume, when the flag that
  closes it arrives. By default that is the end of the scan; pass
  `triggers={"imaging": AcquisitionFlag.LAST_IN_SLICE}` to reconstruct slice
  by slice.
- The flags and counters come from the labels in your sequence
  (`pp.make_label`, as in PyPulseq). Label it right and every readout lands in
  place.
- Return a `ReconResult`; pulserver writes the image header (position,
  orientation, field of view) from the sequence and the scanner.

The scanner chooses the recon by name, independently of the sequence, so one
recon can serve many sequences.

## Try it without a scanner

The virtual scanner plays your sequence on a phantom and streams the data to
your recon, through exactly the path a real scan takes:

```bash
pip install pulserver bartorch
printf '[Limits]\nB0: 3.0\n[Limits End]\n' > limits.txt
pulserver proxy --store designs --port 9002 --plugins recon &
pulserver scan --plugins sequences --plugin gre --reconstruction pics \
  --limits limits.txt --store designs --recon 127.0.0.1:9002 --output images
```

Or skip the install and try it in your browser:
<https://pulserver.github.io/MaRGE/>.

## On a real scanner

You need the vendor interpreter for your scanner, which calls
`pulserver design` for steps 1 to 3 and streams the raw data to
`pulserver proxy` for steps 4 to 6. [Running the services](https://pulserver.github.io/pulserver/stable/user-guide/running.html)
explains how to set both up.

pulserver's checks compute estimates. Passing them does not establish scanner
or patient safety: SAR and RF and gradient heating are checked by the
scanner's own routines.

## Learn more

- [Course](https://pulserver.github.io/pulserver/stable/examples/course.html):
  five lessons from a protocol to an image.
- [Documentation](https://pulserver.github.io/pulserver/stable/), for every
  option and API detail.

## How to cite

If you use pulserver in your work, please cite:

```bibtex
@inproceedings{cencini2025pulserver,
  title     = {Pulserver: an open-source Pulseq-based client-server framework for vendor agnostic, interactive {MR} sequence design},
  author    = {Cencini, Matteo and Wang, Kang and Huang, Sherry and Schulte, Rolf F. and Sprenger, Tim
               and Noll, Douglas C. and Tosetti, Michela and Nielsen, Jon-Fredrik},
  booktitle = {Proceedings of the International Society for Magnetic Resonance in Medicine},
  pages     = {1275},
  year      = {2025}
}
```

pulserver builds on Pulseq ([Layton et al., 2017](https://doi.org/10.1002/mrm.26235))
and the ISMRM raw data format ([Inati et al., 2017](https://doi.org/10.1002/mrm.26089));
please cite those too.

## License

MIT, see [LICENSE](https://github.com/pulserver/pulserver/blob/main/LICENSE).
