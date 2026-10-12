# License and notices

The pulserver package is distributed under the
[MIT licence](https://github.com/pulserver/pulserver/blob/main/LICENSE).
Modules adapted from third-party code, and code vendored unchanged, keep their
notices:

| Component | Adapted in | Licence | Notice |
| --- | --- | --- | --- |
| [gadgetron-python](https://github.com/gadgetron/gadgetron-python) | `pulserver.recon`: the MRD stream connection, readers and writers | MIT | [`LICENSES/gadgetron-python-MIT.txt`](https://github.com/pulserver/pulserver/blob/main/LICENSES/gadgetron-python-MIT.txt) |
| [python-ismrmrd-server](https://github.com/kspaceKelvin/python-ismrmrd-server) | `pulserver.recon`: the message identifiers, the stream capture, and the DICOM converter | MIT | [`LICENSES/python-ismrmrd-server-MIT.txt`](https://github.com/pulserver/pulserver/blob/main/LICENSES/python-ismrmrd-server-MIT.txt) |

pulserver's container image carries BrainWeb's normal brain, the fuzzy tissue
model of the [BrainWeb](https://brainweb.bic.mni.mcgill.ca/) Simulated Brain
Database of the McConnell Brain Imaging Centre, as
[brainweb-dl](https://github.com/paquiteau/brainweb-dl) downloads it, and the
coils' field maps and VOPs [mariepy](https://github.com/pulserver/mariepy)
solves in it, which BrainWeb's terms of use cover as they cover the model. Work
using them cites Collins et al., IEEE Trans Med Imaging 17:463, 1998, and Kwan
et al., IEEE Trans Med Imaging 18:1085, 1999.

The pulserver logo shares the bipolar-gradient mark of the pypulseqpp logo,
which derives from the MIT-licensed
[PyPulseq](https://github.com/imr-framework/pypulseq) wordmark. The SVG artwork
in this repository is maintained as pulserver source material.
