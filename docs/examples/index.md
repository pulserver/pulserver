# Examples

Worked code for each of the three interfaces. The Python examples build and
check sequences; the C and C++ examples build the software that plays them
and reconstructs them.

## Python

```{toctree}
:maxdepth: 1

python/index
```

Still to be written, and listed here so the gaps are visible: a first sequence
end to end, declaring a protocol and the UI it produces, reading a safety
report, subclassing a readout family, and driving one plugin from both the GUI
and a console loop.

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
