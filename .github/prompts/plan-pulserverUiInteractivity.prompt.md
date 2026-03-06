# Plan: Pulserver UI Interactivity — Python/MATLAB Bridge via nimpulseqgui (v3)

**TL;DR**: Replace TCP/Docker server with persistent local process. Host executables embed CPython/MCR, delegate to user plugins. GE driver communicates via stdin/stdout pipes (plain text, no JSON, no temp files). Python plugins enforced via ABC in the `pulserver` package with auto-discovery. A self-extracting installer script packages everything for deployment.

---

## Repo Layout (complete — including existing components)

```
pulserver/                         # PUSHED
├── csrc/                          # C89 pulseqlib (parser, segmentation, safety)
├── extensions/pulseqlib/          # C++ RAII wrappers
├── python/pulserver/
│   ├── core/                      # SequenceCollection, plot, pns, validate, etc.
│   └── sequences/                 # NEW — ABC base + built-in examples
│       ├── __init__.py            #   exports PulseqSequence, UIParam, params
│       ├── _base.py               #   ABC definition
│       ├── _params.py             #   UIParam(StrEnum) + dataclass param types
│       └── gre_2d.py              #   built-in GRE (class + module-level funcs)
├── matlab/+pulserver/
│   ├── SequenceCollection.m       # existing wrapper
│   └── +sequences/gre_2d.m       # NEW — MATLAB GRE example
├── bridge/                        # NEW — Nim host executables
│   ├── pypulseq_host.nim
│   ├── matlab_host.nim
│   └── README.md
├── scripts/
│   ├── make_installer.sh          # NEW — self-extracting archive builder
│   └── ...                        # existing build scripts
├── tests/pytests/
│   └── test_plugin_contract.py    # NEW
├── external/                      # GITIGNORED — modified nimpulseq copies
│   ├── nimpulseqgui/
│   ├── nimpulseq/
│   └── PulseqSystems/
│
3p/                                # NOT PUSHED (proprietary)
├── pulserver-driver/              # GE EPIC (pipe caller, seqparams formatter)
└── uipython/                      # design drafts
```

---

## Phase 1: nimpulseqgui Modifications (in `external/`)

| Step | File | Change |
|------|------|--------|
| 1.1 | `definitions.nim` | `ValidationResult{valid,duration,info}` + converter `bool→ValidationResult` |
| 1.2 | `utils/gui/propertyedit/sequenceexe.nim` | `.valid` checks (~5 lines) |
| 1.3 | `io.nim` | Extract `readProtocolFromString` |
| 1.4 | `sequenceexe.nim` | `--protocol`, `--validate-only`, `--list-protocol`, `--persistent`, `--input -` |

---

## Phase 2: Persistent Process Protocol

Plain-text wire protocol over stdin/stdout pipes (C89-native `sprintf`/`fgets`):

| Command | Response |
|---------|----------|
| `VALIDATE\n` + preamble | `VALID 5.32 TA = 5.32 s\n` or `INVALID TE too short\n` |
| `GENERATE /tmp/out.seq\n` + preamble | `GENERATED /tmp/out.seq\n` or `ERROR msg\n` |
| `LIST_PROTOCOL\n` | `PROTOCOL\n` + preamble |
| `QUIT\n` | (process exits) |

`--persistent` in `sequenceexe.nim`: read-loop on stdin, dispatch, flush stdout, loop until QUIT/EOF.

---

## Phase 3: Python Plugin Contract (ABC in `pulserver` package)

### 3a. Typed Protocol Keys — `UIParam(StrEnum)` in `_params.py`

```python
from enum import StrEnum

class UIParam(StrEnum):
    # Timing
    TE = "TE"
    TR = "TR"
    TI = "TI"
    # Spatial
    FOV = "FOV"
    SLICE_THICKNESS = "SliceThickness"
    NSLICES = "NSlices"
    MATRIX = "Matrix"
    NECHOES = "NEchoes"
    # Contrast
    FLIP_ANGLE = "FlipAngle"
    BANDWIDTH = "Bandwidth"
    # Flags
    FAT_SAT = "FatSat"
    SPOILER = "Spoiler"
    # Description row
    TA = "TA"

    @staticmethod
    def user(n: int) -> str:
        """GE user CV slot: UIParam.user(0) → 'User0'."""
        return f"User{n}"
```

- `StrEnum` members *are* strings — zero conversion cost at bridge boundary
- Typos become `AttributeError` instead of silent wrong keys
- `UIParam.user(n)` covers GE's `opuser0..opuser47` CVs
- Extensible: plugin authors subclass or use raw strings for one-off params

### 3b. Typed Protocol Values — dataclasses in `_params.py`

```python
from dataclasses import dataclass, asdict
from enum import StrEnum

class Validate(StrEnum):
    SEARCH = "search"   # binary-search for valid min/max (nimpulseqgui default)
    CLIP   = "clip"     # clamp to [min, max]
    NONE   = "none"     # no auto-validation

@dataclass
class FloatParam:
    value: float; min: float; max: float; incr: float
    unit: str = ""; validate: Validate = Validate.SEARCH
    type: str = "float"   # wire-format tag

@dataclass
class IntParam:
    value: int; min: int; max: int; incr: int
    unit: str = ""; validate: Validate = Validate.SEARCH
    type: str = "int"

@dataclass
class BoolParam:
    value: bool
    type: str = "bool"

@dataclass
class StringListParam:
    options: list[str]; index: int
    type: str = "stringlist"

@dataclass
class Description:
    text: str
    type: str = "description"

ProtocolValue = FloatParam | IntParam | BoolParam | StringListParam | Description
Protocol = dict[UIParam | str, ProtocolValue]
```

**Bridge serialization** (at Nim↔Python boundary, not in plugin code):
```python
def param_to_dict(p: ProtocolValue) -> dict:
    return asdict(p)

def dict_to_param(d: dict) -> ProtocolValue:
    tag = d.pop("type")
    return {"float": FloatParam, "int": IntParam, "bool": BoolParam,
            "stringlist": StringListParam, "description": Description}[tag](**d)
```

### 3c. ABC in `_base.py`

```python
from abc import ABC, abstractmethod
from ._params import Protocol

class PulseqSequence(ABC):
    @abstractmethod
    def get_default_protocol(self, opts: dict) -> Protocol: ...
    @abstractmethod
    def validate_protocol(self, opts: dict, protocol: Protocol) -> dict: ...
    @abstractmethod
    def make_sequence(self, opts: dict, protocol: Protocol) -> str: ...
```

**Why ABC is a good idea here:**
- **Enforcement**: miss a method → `TypeError` at instantiation, not a cryptic runtime error from Nim
- **IDE support**: autocompletion, type hints, docstrings for plugin authors
- **Shared utilities**: base class can later provide helpers like `_make_system(opts)` → `pp.Opts`
- **Auto-discovery**: bridge finds the subclass via `inspect`, no manual wiring needed

**How module-level exposure works (each plugin):**

```python
class GRE2D(PulseqSequence):
    def get_default_protocol(self, opts): ...
    def validate_protocol(self, opts, protocol): ...
    def make_sequence(self, opts, protocol): ...

# Auto-expose for bridge fallback + direct import + testing
_instance = GRE2D()
get_default_protocol = _instance.get_default_protocol
validate_protocol = _instance.validate_protocol
make_sequence = _instance.make_sequence
```

**Bridge discovery** (in `pypulseq_host.nim`):
1. First: `inspect.getmembers(module)` → find class subclassing `PulseqSequence` → instantiate → use methods
2. Fallback: look for module-level `get_default_protocol`/`validate_protocol`/`make_sequence` functions (backward compat with plain-function scripts)

**Benefits of the 3-line footer**: `from pulserver.sequences.gre_2d import validate_protocol` works directly — plugin is testable with plain pytest without the Nim bridge.

---

## Phase 4: pypulseq_host (Nim bridge)

- Compiled once per platform, links `nimpy`, `--script <path.py>` loads plugin at runtime
- Auto-discovers `PulseqSequence` subclass or falls back to module functions
- Calls `makeSequenceExe(...)` — inherits all CLI modes including `--persistent`

---

## Phase 5: matlab_host + Phase 6: GE Integration

(Same as previous plan — subprocess start for MATLAB, MCR later. GE integration via popen + pipe protocol, all in `3p/`.)

---

## Phase 7: Self-Extracting Installer (`scripts/make_installer.sh`)

Builds a `.sh` archive containing:
- Compiled `pypulseq_host` binary (platform-specific)
- Bundled CPython + pypulseq + numpy + scipy (minimal venv)
- Plugin directory with example sequences
- Install script: extract → set `PYTHONHOME` → create symlinks → print usage

```bash
#!/bin/bash
# Self-extracting installer for pulserver bridge
INSTALL_DIR="${1:-$HOME/pulserver}"
ARCHIVE_LINE=$(awk '/^__ARCHIVE__/ {print NR + 1; exit}' "$0")
mkdir -p "$INSTALL_DIR"
tail -n +"$ARCHIVE_LINE" "$0" | tar xz -C "$INSTALL_DIR"
# ... setup symlinks, PYTHONHOME, etc.
exit 0
__ARCHIVE__
```

**Reuse in private GE driver repo**: same pattern packages `pypulseq_host` + Python env + GE driver PSD + plugin `.py` files into a single installer.

---

## Implementation Order

```
Phase 1 (nimpulseqgui mods) ─┐
                              ├→ Phase 2 (persistent protocol) → Phase 4 (pypulseq_host) → Phase 7 (installer)
Phase 3 (ABC + tests) ───────┘
                                                                  Phase 5 (matlab_host) → Phase 6 (GE integration)
```

Phases 1 and 3 are independent and can proceed in parallel. Phase 2 depends on Phase 1. Phase 4 needs both.

---

## Decisions

- **ABC for Python plugins**: `pulserver.sequences.PulseqSequence` — enforced, auto-discovered, with module-level function fallback
- **`UIParam(StrEnum)`** for protocol keys: standard params as enum members, `UIParam.user(n)` for GE user CVs, raw strings allowed for one-offs
- **Dataclass protocol values**: `FloatParam`, `IntParam`, `BoolParam`, `StringListParam`, `Description` — IDE-friendly, self-documenting, `asdict()` for free serialization at bridge boundary
- **`Validate(StrEnum)`**: `SEARCH` (binary-search min/max), `CLIP` (clamp), `NONE` — controls nimpulseqgui's `PropertyValidate` behavior per-param
- **No JSON/temp files**: plain-text preamble via pipes, C89-native
- **Persistent process via popen()**: interpreter stays warm across CVEval calls
- **Modified nimpulseqgui in `external/`**: gitignored, eventually upstream PR + submodule
- **Self-extracting installer**: in public `scripts/`, reused by private driver repo
- **All GE code in `3p/`**: never pushed
