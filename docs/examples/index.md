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

The five stages of an interpreter, one page each, written against the C API.
The scanner half is stubbed, and every page renders a file the build compiles.

```{toctree}
:maxdepth: 1

c/index
```

## C++ — writing an interpreter

The same five stages against the C++ API. Each file has a C counterpart of the
same name, and the two produce the same output on the same input.

```{toctree}
:maxdepth: 1

cpp/index
```
