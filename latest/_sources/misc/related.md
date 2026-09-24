# Related projects

| Project | Relationship |
| --- | --- |
| [pypulseqpp](https://github.com/pulserver/pypulseqpp) | Sequence design engine: the sequence applications a scanner sequence binds, and the Pulseq reader and writer. |
| [bartorch](https://github.com/mcencini/bartorch) | Reconstruction engine, imported into the spare reconstruction workers when installed. |
| [Pulseq](https://pulseq.github.io/) | Open sequence-file specification of every design pulserver stores. |
| [ISMRMRD](https://ismrmrd.readthedocs.io/) | Raw-data format and library of the reconstruction side. |
| [python-ismrmrd-server](https://github.com/kspaceKelvin/python-ismrmrd-server) | Reference MRD streaming server; the reconstruction proxy speaks the same message protocol, and parts of `pulserver.recon` are adapted from it. |
| [gadgetron-python](https://github.com/gadgetron/gadgetron-python) | Python interface to Gadgetron, whose stream connection, readers and writers `pulserver.recon` adapts. |
| [Gadgetron](https://gadgetron.github.io/) | Reconstruction framework whose acquisition-bucket layout `pulserver.mrd` follows. |
