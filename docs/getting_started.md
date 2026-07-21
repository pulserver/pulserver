# Getting started

Install the sequence-authoring extra:

```bash
python -m pip install "pulserver[sequence]"
```

Use `pulserver.pypulseq` as the PyPulseq-compatible authoring namespace:

```python
import pulserver.pypulseq as pp

seq = pp.Sequence()
seq.add_block(pp.make_delay(1e-3))
```

See the {doc}`API reference <api/index>` for the grouped public interface.
