# pulserver
This scaffold contains bridge/interface contracts and host stubs.

## Installation

Install pulserver from the repository root:

```bash
pip install .
```

Development install with test tools:

```bash
pip install .[dev,test]
```

## Included from monorepo
- `bridge/`
- `python/pulserver/core/_base.py`
- `python/pulserver/core/_params.py`
- `python/pulserver/__init__.py`
- `LICENSE.txt`

## Testing

Run Python tests:

```bash
pytest tests/python
```

Run Nim bridge tests:

```bash
bash tests/nim/run_tests.sh
```

## Bridge plugin loading

`pypulseq_host` supports either direct plugin path loading or folder-based
loading with an explicit plugin selector.

Examples:

```bash
./bridge/pypulseq_host --script bridge/tests/test_plugin.py --validate-only
./bridge/pypulseq_host --plugin-folder bridge/tests --script test_plugin --validate-only
```

## Fast Sequence Path

`pulserver.FastSequence` is a sequential-only `pypulseq.Sequence` replacement
for production bridge execution. It disables positional `set_block()` and can
skip per-block deduplication and continuity checks during build, then relies on
`write(remove_duplicates=True)` to compact once at the end.

Use `pulserver.write(seq, output=...)` to write either to disk or binary blobs.

```python
import pulserver

seq = pulserver.FastSequence()
# seq.add_block(...)

payload = pulserver.write(seq, output=None, check_timing=False)  # bytes
```

## Notes
- Intended as a standalone interface package for bridge/plugin contracts.
- MATLAB host stubs are kept under `bridge/`.
