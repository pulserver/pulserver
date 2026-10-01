"""Shared pytest fixtures."""

import os
import subprocess
from pathlib import Path

import pytest


def _cuda_available():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


# Without a GPU, Triton's kernels run on the CPU under its interpreter, which
# is chosen when triton is first imported.
if not _cuda_available():
    os.environ.setdefault("TRITON_INTERPRET", "1")


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda",
            marks=[
                pytest.mark.cuda,
                pytest.mark.skipif(not _cuda_available(), reason="no CUDA device"),
            ],
        ),
    ]
)
def device(request):
    """Device name, ``"cpu"`` or ``"cuda"``; the CUDA leg is skipped without a device."""
    return request.param


#: The vendor the C library is compiled for where a test builds it as a scanner
#: does, and the Pulseq labels filling that vendor's three ADC label columns.
VENDOR = 5
LABELS = (8, 7, 6)

_ROOT = Path(__file__).parents[1]
_C_SOURCES = _ROOT / "src" / "c"


def _build(directory, program, name, *defines):
    """``program`` linked with the C library as a scanner builds it: 32-bit, for one vendor."""
    probe = directory / "probe.c"
    probe.write_text("int main(void) { return 0; }\n")
    toolchain = subprocess.run(
        ["gcc", "-m32", str(probe), "-o", str(directory / "probe")],
        capture_output=True,
        check=False,
    )
    if toolchain.returncode != 0:
        pytest.skip("no 32-bit C toolchain")
    folders = ("pulseq", "core", "io", "structure", "cache", "playout")
    sources = [
        str(p) for folder in folders for p in sorted((_C_SOURCES / folder).glob("*.c"))
    ]
    includes = [
        f"-I{_C_SOURCES / sub}"
        for sub in ("", "include", "include/pulseg", "include/pulseq", "pulseq")
    ]
    output = directory / name
    subprocess.run(
        [
            "gcc",
            "-m32",
            "-std=c89",
            f"-DPULSEG_VENDOR={VENDOR}",
            *defines,
            *includes,
            str(_ROOT / "tests" / "native" / program),
            *sources,
            "-lm",
            "-o",
            str(output),
        ],
        check=True,
    )
    return output


@pytest.fixture(name="scanner_build", scope="session")
def scanner_build_fixture():
    """Build a native program against the C library as a scanner builds it.

    ``build(directory, program, name, *defines)``, where ``program`` is a file
    of ``tests/native/``. Skips the test where no 32-bit toolchain is present.
    """
    return _build
