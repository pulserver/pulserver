"""Parse the C and C++ headers into the XML the reference is rendered from.

Doxygen is a system package rather than a Python one, so this is the one part
of the documentation that can be missing on an otherwise complete environment.
It is skipped rather than demanded, and the caller decides what to do without
it: the pages carry their prose and say the reference is not in this build.
"""

from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ElementTree
from pathlib import Path

_REPOSITORY = Path(__file__).resolve().parents[1]

#: The headers the reference is generated from -- the public C and C++ include
#: trees, plus the reconstruction-side headers that have no include directory
#: of their own.
HEADERS = ("src/c/include", "src/cpp/include", "src/cpp/recon")

#: Obtained from the Gadgetron repository and kept as they arrived. They are a
#: dependency this project vendors, not API it publishes.
VENDORED = (
    "GadgetronClient.h",
    "NHLBICompression.h",
    "cpuisa.h",
    "gadgetron_ismrmrd_client.h",
)

#: The C include tree on its own. ``raw64.hpp`` deliberately re-instantiates
#: the C Pulseq reader inside ``pulseq::raw64`` at double precision, so in a
#: run covering both trees every ``pulseq_*`` name exists twice and Breathe
#: cannot tell which one a C page means. A second run over the C headers alone
#: gives those pages an unambiguous scope.
C_HEADERS = ("src/c/include",)

#: Where the XML lands. Generated output, so it is not tracked.
XML = Path(__file__).resolve().parent / "_doxygen"

#: Where the C-only XML lands.
XML_C = Path(__file__).resolve().parent / "_doxygen_c"


def run() -> bool:
    """Generate the Doxygen XML, or report that Doxygen is not installed.

    Two runs are produced: :data:`XML` over every header, and :data:`XML_C`
    over the C include tree alone. See :data:`C_HEADERS` for why.

    Returns
    -------
    bool
        Whether Doxygen ran. ``False`` means it is not on ``PATH``.

    Raises
    ------
    subprocess.CalledProcessError
        If Doxygen ran and failed, which a malformed comment causes.
    """
    doxygen = shutil.which("doxygen")
    if doxygen is None:
        return False

    _generate(doxygen, XML, HEADERS)
    _generate(doxygen, XML_C, C_HEADERS)
    return True


def _generate(doxygen: str, output: Path, trees: tuple[str, ...]) -> None:
    """Run Doxygen over ``trees``, writing XML into ``output``."""
    output.mkdir(parents=True, exist_ok=True)
    headers = " ".join(str(_REPOSITORY / part) for part in trees)
    # Everything else is Doxygen's default. The comments in the headers are
    # what this build is for; the HTML, the graphs and the LaTeX are not.
    configuration = "\n".join(
        [
            "PROJECT_NAME = pulserver",
            f"INPUT = {headers}",
            "EXCLUDE = "
            + " ".join(str(_REPOSITORY / "src/cpp/recon" / n) for n in VENDORED),
            f"INCLUDE_PATH = {headers} {_REPOSITORY / 'src/c/include/pulseq'}",
            "FILE_PATTERNS = *.h *.hpp",
            "RECURSIVE = YES",
            "GENERATE_HTML = NO",
            "GENERATE_LATEX = NO",
            "GENERATE_XML = YES",
            f"XML_OUTPUT = {output}",
            "QUIET = YES",
            # A parameter left undocumented is a style choice; a malformed
            # comment is a defect, so ``WARN_IF_DOC_ERROR`` stays on.
            "WARN_IF_INCOMPLETE_DOC = NO",
            "WARN_IF_UNDOCUMENTED = NO",
            # A Markdown heading inside a comment is a heading, not a
            # section of its own: Doxygen's sectioning is what Breathe
            # renders structurally, and a heading promoted into it arrives
            # as a section with no title.
            "TOC_INCLUDE_HEADINGS = 0",
            "EXTRACT_ALL = YES",
            "EXTRACT_STATIC = YES",
            "MACRO_EXPANSION = YES",
            "EXPAND_ONLY_PREDEF = YES",
            # ``extern "C"`` and the export macro are noise in a signature.
            "PREDEFINED = __cplusplus= PULSEG_API= PULSEQ_API=",
        ]
    )
    subprocess.run(
        [doxygen, "-"],
        input=configuration,
        text=True,
        check=True,
        cwd=_REPOSITORY,
    )
    _substitute_dashes(output)


def _substitute_dashes(output: Path) -> None:
    """Put the dash Doxygen marked up back as the character it stands for.

    Doxygen reads ``--`` in a comment as typography and emits an element for
    it. Rendered, that arrives as the numeric character reference itself,
    visible in the page. The XML is this build's own intermediate, so the
    substitution belongs here rather than in the headers, which say what they
    mean in plain ASCII.
    """
    for document in output.glob("*.xml"):
        text = document.read_text(encoding="utf-8")
        substituted = text.replace("<ndash/>", "–").replace("<mdash/>", "—")
        if substituted != text:
            document.write_text(substituted, encoding="utf-8")


def entities(xml: Path = XML) -> set[str]:
    """The names Doxygen found, as the reference pages spell them.

    Free functions and enumerations come back qualified by their namespace
    where they have one, classes and structs by their full nested name.

    Parameters
    ----------
    xml : Path, optional
        Directory holding a Doxygen XML run.

    Returns
    -------
    set of str
        Every documentable class, struct, free function and enumeration.
    """
    index = ElementTree.parse(xml / "index.xml").getroot()
    found: set[str] = set()
    for compound in index.findall("compound"):
        kind, name = compound.get("kind"), compound.findtext("name")
        if kind in {"class", "struct", "union"}:
            found.add(name)
        if kind not in {"file", "namespace"}:
            continue
        prefix = f"{name}::" if kind == "namespace" else ""
        for member in compound.findall("member"):
            if member.get("kind") in {"function", "enum"}:
                found.add(prefix + member.findtext("name"))
    return found
