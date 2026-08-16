# Test fixtures and the oracle

Everything checked in under `expected/` and `../python/fixtures/` is written
by Python in this repository — regenerate with:

```bash
python tests/utils/generate_fixtures.py       # both fixture trees
python tests/utils/make_binary_fixtures.py    # expected/binary/ pairs
```

Output is deterministic: running the generators twice leaves the tree clean.

## The two fixture trees

* `../python/fixtures/` — the zoo corpus: one small protocol per sequence-zoo
  slot, plus parser edges and the EPI NextSequence chains. Built by
  `../python/fixture_corpus.py` through `pulserver.pypulseq` itself.
* `expected/` — the synthetic C-test corpus, from `synthetic_fixtures.py`:
  the numbered gradient/RF safety cases (most deliberately *invalid*, so
  they are written as exact event-table text no designer would emit), the
  TRID label specimens, the interior-delay pair, the dedup NextSequence
  pair, the 32×32 blipped GRE written without deduplication, the navigated
  GRE, and the corrupted-signature specimen derived from the corpus
  `gre_2d.seq`. `expected/binary/` holds the paired text/binary encodings
  `test_pulseq_binary.c` compares.

## Where the references live

There are no stored truth files. The reference for the interpreter's
resolved content is computed at test time by the spec-first oracle in
`pulseg_oracle/` (identity normalization, partition validation, instance
table, label walk, composed TR waveforms) and compared against the C
interpreter in process by `../python/test_pulseg_vs_oracle.py` through the
`_pulseg_wrapper` projections. The C test lane (`../ctests/`) keeps
self-contained expectations only.
