# Terminology and conventions

{doc}`documentation` is the generic guide, and states what belongs in each
form of documentation and how it should be written. This page holds the
conventions specific to pulserver, and governs every docstring, documentation
page, code comment and user-facing diagnostic string in this repository,
including the C and C++ doc comments under `src/`. It is binding on human
contributors and on automated agents alike; `AGENTS.md` requires compliance
with both.

The target register is reference documentation for MRI researchers and
sequence developers who run acquisitions on clinical scanners. The reader is
assumed to know MR physics, Pulseq and the MRD raw-data format. The
documentation's job is to state what an object *is*, in the field's own
vocabulary, together with the semantics a signature cannot carry: units,
frames, the layer a statement belongs to, state and invariants.

The failure mode this guide exists to prevent is the replacement of an
established MRI, Pulseq or MRD term by a paraphrase of what the thing does. A
reader then has to reconstruct the standard concept from a description of it.
Repeating the correct technical term is always preferable to varying the
wording.

## 1. Terminology is the substance

Use the established term. Do not explain around it.

| Do not write | Write |
|---|---|
| what the operator asks for | the requested protocol, the prescription |
| what the sequence will actually play | the resolved protocol |
| the numbers the scanner keeps | the scanner control variables (CVs) |
| a design written earlier | a stored design, by its identifier |
| the folder both sides look at | the design store |
| the scanner program | the interpreter |
| the host process for one sequence | the PSD host process |
| the compiled form of a sequence | the IR, the IR cache |
| the part that repeats | the repeating unit, the TR |
| pieces of a TR | segments |
| the order the pieces are played in | the execution stream |
| filling in the raw data | MRD enrichment |
| what the reconstruction runs on | the reconstruction worker |
| the reconstruction's code | the reconstruction plugin |
| the sequence's code | the scanner-sequence plugin, the sequence application |

### Distinctions that must not be blurred

**Protocol.** A *prescription* or *requested protocol* is what the operator
enters. The *resolved protocol* is what the design achieves and what the
interpreter plays. A *listing* carries entries with their UI schema; a *value
block* carries values only. A *preset* is a negative time value standing for a
design choice, not a time.

**Design.** A *sequence application* is the pypulseqpp `SequenceApp` subclass
that designs the sequence. A *scanner sequence* is the pulserver
`ScannerSequence` that binds an application to protocol entries. A *plugin* is
the file either kind is loaded from; say which kind.

**Storage.** A *design* is one generated or imported chain with its IR cache,
immutable once written. Its *identity* is the hash of what it depends on; its
*identifier* is the first 18 hexadecimal digits of the identity. The *design
store* is the directory holding every design. A *design call* is one `list`,
`validate`, `generate` or `import` of a PSD host process; no call keeps state
beyond the designs it stores.

**IR.** A *subsequence* is one file of a `NextSequence` chain. A *definition*
is a distinct event after deduplication; an *instance* is its occurrence in a
block, with its own amplitude. The *repeating unit* is the TR of a
subsequence; a *segment* is part of it bounded by zero gradient amplitude; the
*execution stream* is the order of segments over the scan. The *IR cache* is
the file; the *collection* is what a reader loads from it.

**Raw data.** A *series* is one MRD stream from the reconstruction client. An
*acquisition* is one MRD readout record; a *readout* is the ADC event of the
sequence it corresponds to. *Encoding counters* are the MRD `idx` fields;
*flags* are the MRD acquisition flags; an *encoding space* is an entry of the
header's `encoding` list. *Enrichment* replaces these with what the sequence
states.

**Layers.** The *Pulseq representation* is the content of a `.seq` file. The
*engines* are pypulseqpp (design) and the reconstruction a plugin imports.
*Orchestration* is pulserver. *Scanner execution* is what the interpreter does
on the hardware. Do not attribute a property of one layer to another: a
design is not a `.seq` file, and the IR cache is not the design of record.

### Fixed vocabulary

- **design calls**, **warm server**, **reconstruction proxy**;
  `pulserver.host` and `pulserver.vre` when the module is meant.
- **PSD host process**, not "host PSD" or "PSD process".
- **interpreter** for the scanner-side program; **reconstruction client** for
  the scanner-side sender of raw data.
- **IR** and **IR cache**; `.pseg` and `.pge` are file extensions, not names
  for the representation.
- **MRD** for the format; **ISMRMRD** for the library and the HDF5 file.

## 2. Register

Write dry, declarative technical prose.

**No personification.** A server does not decide, a design does not know, a
proxy does not wait for anything it is not blocked on, a plugin does not ask.
Objects have properties and functions have behaviour.

**No literary compression.** Do not describe an object by a relative clause
where a noun exists: not "the directory the design calls write into" but "the
design store"; not "what the design managed" but "the resolved protocol".

**No taglines.** An API summary classifies its object.

**No conversational or tutorial voice.** No "simply", "just", "note that",
"under the hood", "powerful", "seamless". No rhetorical questions.

**No history.** Describe the code as it is. No "used to", "previously", "this
replaces", no named fixed bugs, no comparison with removed designs.

**Variation is not a virtue.** Use the same term for the same concept every
time it appears.

## 3. Summary lines

The first line is one sentence, on one line, ending in a period.

- **Functions and methods** take an imperative or third-person declarative
  verb that classifies the operation: "Return…", "Read…", "Format…",
  "Segment…".
- **Classes** take a noun phrase naming what the class represents.
- **Modules and packages** take one line naming the responsibility.

Do not restate the signature, the annotations or the parameter names in prose.

## 4. Units and frames

Every documented quantity carries its unit.

| Quantity | Unit |
|---|---|
| Time entry of a protocol, on the wire and in a CV | integer µs |
| Time argument of a sequence application | s |
| Float protocol entry | the entry's `unit`; the argument is the value times `scale` |
| Scan time in a `VALIDATE` reply | s |
| Rasters passed to the IR conversion | s in `pypulseqpp.Opts`, µs in the cache |
| Gyromagnetic ratio, field strength | Hz/T, T |
| Field-of-view offset in a protocol or an import block | mm |
| Field-of-view offset passed to `ir.convert` and `ir.prescribe` | m |
| Prescription rotation (`fov_rotation_ij`) | unitless, element (i, j) of an orthonormal matrix |
| PNS limit (`pns_limit`) | fraction of the nerve model's threshold |
| Forbidden band | Hz; amplitude allowed in it in mT/m |
| SAR ratio (`vop_sar_ratio`, `vop_global_sar_ratio`) | unitless: energy over that of the reference repetition |
| Default channel weights (`vop_default_shim`) | unitless magnitude, phase in rad |
| Dwell time in an enriched acquisition (`sample_time_us`) | µs |
| k-space trajectory | 1/m, as pypulseqpp reports it |
| Grid trajectory (`ReconBuffer.grid_trajectory`) | k times the reconstructed field of view; an N-point matrix spans [-N/2, N/2) |
| Design identifier | 18 hexadecimal digits: three 24-bit integers, each exact in a float32 |

State the coordinate frame wherever a position or a k-space quantity appears.
The field-of-view offset is expressed along the logical readout, phase and
slice axes. The prescription rotation maps the logical axes to the physical x,
y and z gradient axes, physical = R logical. The trajectory is expressed along
the sequence's x, y and z gradient axes, with the sequence's block rotations
applied and no prescription rotation. The host's checks are made in the
physical frame, with the prescription rotation composed after the block
rotations.

State the precision a value is exchanged at when it matters: time entries are
rounded to the nearest microsecond, ties to even, and float entries to six
significant digits.

## 5. Safety language

Pulserver performs no safety check of its own. The timing, gradient, PNS,
mechanical-resonance and SAR checks belong to pypulseqpp and compute
estimates. The host runs all but the SAR check, under the limits the PSD
passes, before it writes an IR cache; from the SAR check it writes each
subsequence's SAR relative to a reference pulse into the cache, and the PSD
computes the SAR and the gradient heating. Passing these checks does not
establish scanner or patient safety.

- Never write "safe", "validated", "compliant" or "approved" of a sequence, a
  protocol or a design.
- A valid `VALIDATE` reply means the sequence application designed the
  prescription under the scanner limits it was given. State that, not more.

## 6. Source of truth

Existing documentation is not evidence. Before writing or changing a
substantive statement, verify it against, in order of authority:

1. the implementation being documented, including `src/cpp/` and `src/c/`;
2. the tests that protect it;
3. pypulseqpp, for sequence design and the Pulseq format;
4. the ISMRMRD specification and library, for MRD;
5. the Pulseq specification and the primary literature, cited by DOI.

Do not infer semantics from a name, a type annotation or another docstring.
If a contract cannot be established from these sources, say what is known and
leave the rest undocumented rather than guessing.

Do not print measured constants that are not guaranteed across releases or
hardware.

## 7. What to document, and what not to

The `Docstrings` section of `AGENTS.md` governs docstrings: document what the
name, signature and annotations do not already state, and write no docstring
for a private helper whose behaviour is evident. Use local comments for
implementation detail and for a choice a reader would otherwise undo.

## 8. Format

- NumPy-style docstrings throughout, rendered by `sphinx.ext.napoleon`.
- Documentation pages are Markdown with MyST roles and directives.
- C and C++ documentation comments are Doxygen-style and follow this guide's
  terminology and register rules in full.

## 9. Do not churn

Rewriting correct technical prose for stylistic preference wastes review effort
and risks introducing errors. Leave alone documentation that already uses the
conventional term, states its units and frame, and classifies its object. When
modifying code, clean up nearby documentation that clearly violates this guide;
do not broaden a focused change into a repository-wide rewrite unless that is
the task.
