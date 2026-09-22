# The IR cache

A Pulseq file lists every block of a scan. An interpreter that plays it on
scanner hardware instead needs each distinct waveform once, the structure that
repeats, and the order in which to play the repetitions. Pulserver computes
that intermediate representation (IR) on the host and writes it into a binary
cache beside the sequence file, which the interpreter loads.

## Conversion

{func}`~pulserver.ir.convert` reads the `NextSequence` chain starting at a
sequence file, text or binary, with `pypulseqpp.Sequence`; each file of the
chain is one subsequence of the scan, played in order. The chain is then
segmented in the compiled extension:

| Pass | Result |
| --- | --- |
| Event deduplication | A library of distinct RF, gradient and ADC definitions, and per-block instances that reference a definition with their own amplitude |
| Repetition detection | The repeating unit of each subsequence (its TR) and the preparation and cool-down regions around it |
| Segmentation | The repeating unit cut into virtual segments at boundaries where every gradient is zero |
| Execution stream | The order in which the segments are played across the scan |
| Label table | The ADC labels of every readout, three of which fill the ADC label columns (`label_column_map`) |

The limits and rasters of the scanner (`pypulseqpp.Opts`) are those the scan is
segmented under. The host daemon converts every revision it generates or
imports, with the conversion options of the session.

## The cache file

The cache has the sequence file's name with the extension replaced, `.pseg` by
default and `.pge` on GE. It is written in sections a consumer loads
independently: the pulse-generation stage of a playout reads the definitions
and waveforms, and the scan loop additionally reads the per-block instances,
the rotations and the execution stream. The byte order is recorded in the file
and a reader on a machine of the other byte order swaps on load. A cache whose
format version differs from the reader's is rejected rather than partly read.

A cache may be tagged with a vendor code; a reader built for that vendor loads
it. {func}`~pulserver.ir.summary` reports the segmentation of a sequence, from
the chain or from a vendor-neutral cache.

## Why the reader is C89

The code that reads the cache and walks the loaded collection is compiled into
the scanner's interpreter, whose toolchains accept ANSI C. It is kept in
`src/c/`, complete on its own and calling nothing that runs only on the host;
the tests compile it as a scanner does, 32-bit and vendor-tagged, and read back
a cache written by the host. The passes that run only on the host, which build
the IR, are C++ in `src/cpp/ir/`.
