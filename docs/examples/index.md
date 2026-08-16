# Examples

Worked code for each of the three interfaces. The Python examples build and
check sequences; the C and C++ examples build the software that plays them
and reconstructs them.

## Python

```{toctree}
:maxdepth: 1

python/first_sequence
python/protocol_ui
python/safety_report
python/new_readout
python/bridge_gui
python/reconstruction
```

## C — writing an interpreter

A `.seq` file arrives; the scanner has to play it. These build up the pieces
of an interpreter that does, using nothing but the C library.

```{toctree}
:maxdepth: 1

c/interpreter
c/replay
```

## C++ — integrating, playing, reconstructing

```{toctree}
:maxdepth: 1

cpp/safety_only
cpp/interpreter
cpp/gadgetron_client
```

{doc}`cpp/safety_only` is the shortest one and the one most people want: five
lines that gate a sequence against a system's limits, inside an interpreter
that already exists.
