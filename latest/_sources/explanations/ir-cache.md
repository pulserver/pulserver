# The scanner IR

A Pulseq file lists every block of a scan in play order, with each event stored
once in a library and referenced by id. A scanner's pulse generator is
programmed differently: waveform memory is allocated once per distinct
waveform, and a scan is played as repetitions of a small number of instruction
sequences, with amplitudes, phases and rotations updated between repetitions.
Pulserver computes this intermediate representation (IR) of a sequence on the
host and writes it into a binary cache beside the sequence file. The
interpreter loads the cache rather than parsing the Pulseq file.

## Subsequences

{func}`~pulserver.ir.convert` reads the `NextSequence` chain starting at a
sequence file, text or binary, with `pypulseqpp.Sequence`. Each file of the
chain is a subsequence of one scan, played in order; a prescan and the imaging
sequence are typically two subsequences. Each subsequence is analysed on its
own and the results are chained into one collection.

## Passes

| Pass | Result |
| --- | --- |
| Event deduplication | A library of distinct RF, gradient and ADC definitions, and a per-block instance table recording the definition each block plays and its amplitude |
| Repetition detection | The repeating unit of each subsequence, its repetition time (TR), and the preparation and cool-down regions around it |
| Segmentation | The repeating unit divided into segments at block boundaries where every gradient waveform is zero |
| Execution stream | The order in which segments are played over the whole scan |
| Label table | The Pulseq labels of every readout, three of which fill the ADC label columns |

The limits and rasters of the scanner (`pypulseqpp.Opts`) are those under which
the scan is segmented. `label_column_map` selects the three labels the
interpreter records per readout, as indices in the order SLC, PHS, REP, AVG,
SEG, SET, ECO, PAR, LIN, ACQ.

## Cache file

The cache has the name of the first sequence file with its extension replaced:
`.pseg` by default and `.pge` on GE. It is divided into sections that a
consumer loads independently. The pulse-generation stage of a playout reads the
definitions and their waveforms; the scan loop also reads the per-block
instances, the rotations and the execution stream, whose size scales with the
scan length. Integer and float fields are 4 bytes. The byte order is recorded
in the file, and a reader on a machine of the other byte order swaps on load. A
cache whose format version differs from the reader's is rejected rather than
read in part.

A cache may be tagged with a vendor code (`PULSEG_VENDOR_*`), which selects the
reader built for that vendor. {func}`~pulserver.ir.summary` reports the
subsequences, segments and readouts of a sequence, computed from the chain or
loaded from a vendor-neutral cache.

## Language constraint

The IR is built by C++ passes in `src/cpp/ir/`, which run only on the host. The
cache reader, and the accessors a playout uses to walk the loaded collection,
are compiled into the interpreter, whose vendor toolchains accept ANSI C. They
are therefore C89, in `src/c/`, and call nothing in `src/cpp/`. The test suite
compiles `src/c/` as a scanner build does, 32-bit and vendor-tagged, and reads
back a cache written on the host.

## See also

* {doc}`../api/ir` — the conversion interface.
* [`src/c/include/pulseg/`](https://github.com/pulserver/pulserver/tree/main/src/c/include/pulseg)
  — the public C headers the interpreter includes.
* {doc}`/generated/gallery/02-scanner-ir/01_segmentation` — the segmentation of shipped sequences, executed.
