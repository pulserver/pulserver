# Related projects

| Project | Relationship |
| --- | --- |
| [pypulseqpp](https://github.com/pulserver/pypulseqpp) | Sequence design engine: the applications a scanner sequence binds, and the Pulseq reader and writer |
| [bartorch](https://github.com/mcencini/bartorch) | Reconstruction engine, imported ahead of a series by the reconstruction proxy when installed |
| [Pulseq](https://pulseq.github.io/) | Open sequence-file format the designs are written in |
| [ISMRMRD](https://ismrmrd.readthedocs.io/) | Raw-data format (MRD) the reconstruction side receives |
| [python-ismrmrd-server](https://github.com/kspaceKelvin/python-ismrmrd-server) | Reference MRD streaming server whose message protocol the reconstruction proxy speaks |
| [Gadgetron](https://gadgetron.github.io/) | Reconstruction framework whose acquisition-bucket layout `pulserver.mrd` follows |
