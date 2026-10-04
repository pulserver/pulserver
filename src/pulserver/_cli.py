"""The ``pulserver`` command."""

from __future__ import annotations

import sys

_USAGE = (
    "usage: pulserver design {list,validate,generate,import,prune,push,serve} ...\n"
    "       pulserver scan (--seq FILE | --plugin NAME --plugins DIR) --limits FILE ...\n"
    "       pulserver console --plugins DIR --limits FILE --store DIR ...\n"
    "       pulserver validate SEQ [--vendor NAME] [--played FILE] ...\n"
)


def main(argv: list[str] | None = None) -> int:
    """Run ``pulserver <command> ...``; return its exit status."""
    argv = sys.argv[1:] if argv is None else argv
    from ._memory import without_huge_page_advice

    without_huge_page_advice()
    if argv and argv[0] == "design":
        from .host._command import main as design

        return design(argv[1:])
    if argv and argv[0] == "scan":
        from .virtual._command import main as scan

        return scan(argv[1:])
    if argv and argv[0] == "validate":
        from .validate._command import main as check

        return check(argv[1:])
    if argv and argv[0] == "console":
        from .virtual._console import main as console

        return console(argv[1:])
    sys.stderr.write(_USAGE)
    return 2


if __name__ == "__main__":
    sys.exit(main())
