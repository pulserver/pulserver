#!/usr/bin/env python3
"""Place a built documentation tree into the published site, one version deep.

The site holds a directory per version: ``latest`` for main, one per release
tag, and ``stable``, a copy of the newest release. The root sends a reader to
``stable``, or to ``latest`` before there is a release, and ``versions.json``
beside them is the list the version switcher reads.

    python scripts/publish_docs.py docs/build/html latest ../pages

The third argument is a checkout of the branch the site is served from. Only
the published version's directory is replaced, and a release replaces
``stable`` only when it is the newest one. The site keeps ``latest``,
``stable`` and the ``--keep`` newest releases beside them; an older release is
removed, because the branch is never pruned otherwise and every version is a
whole documentation tree.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

#: A release directory is named by its tag.
RELEASE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")

LATEST = "latest"
STABLE = "stable"

REDIRECT = '<!doctype html>\n<meta http-equiv="refresh" content="0; url={target}/">\n'


def _release_key(name: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in RELEASE.match(name).groups())


def releases(site: Path) -> list[str]:
    """Return the release directories the site holds, newest first."""
    names = [d.name for d in site.iterdir() if d.is_dir() and RELEASE.match(d.name)]
    return sorted(names, key=_release_key, reverse=True)


def catalogue(site: Path, url: str) -> list[dict[str, object]]:
    """Return the versions as the switcher reads them, the newest release preferred.

    The newest release is listed once, as stable, at the ``stable`` copy; a page
    built from it matches it by its tag, wherever that page is served from.
    """
    entries: list[dict[str, object]] = []
    for index, tag in enumerate(releases(site)):
        newest = index == 0 and (site / STABLE).is_dir()
        entry: dict[str, object] = {
            "name": f"{tag} (stable)" if newest else tag,
            "version": tag,
            "url": f"{url}/{STABLE if newest else tag}/",
        }
        if newest:
            entry["preferred"] = True
        entries.append(entry)
    if (site / LATEST).is_dir():
        entries.append({"name": LATEST, "version": LATEST, "url": f"{url}/{LATEST}/"})
    return entries


#: Releases kept beside ``latest`` and ``stable``, newest first.
KEEP = 3


def prune(site: Path, keep: int) -> list[str]:
    """Remove every release directory but the ``keep`` newest, and say which.

    ``stable`` is a copy of the newest release rather than one of these, so it
    is never what is removed; neither is ``latest``.
    """
    dropped = releases(site)[max(keep, 0) :]
    for name in dropped:
        shutil.rmtree(site / name, ignore_errors=True)
    return dropped


def publish(
    built: Path, version: str, site: Path, url: str, keep: int = KEEP
) -> list[dict[str, object]]:
    """Put ``built`` in the site as ``version``, then rewrite the root and the list."""
    if not (built / "index.html").is_file():
        raise SystemExit(f"{built} does not look like a built documentation tree")
    if version != LATEST and not RELEASE.match(version):
        raise SystemExit(f"{version!r} is neither {LATEST!r} nor a release tag")

    site.mkdir(parents=True, exist_ok=True)
    target = site / version
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(built, target)

    # A patch tagged after a later release archives itself without taking
    # stable backwards.
    if version != LATEST and releases(site)[0] == version:
        shutil.rmtree(site / STABLE, ignore_errors=True)
        shutil.copytree(target, site / STABLE)

    prune(site, keep)

    root = STABLE if (site / STABLE).is_dir() else LATEST
    (site / "index.html").write_text(REDIRECT.format(target=root), encoding="utf-8")
    entries = catalogue(site, url)
    (site / "versions.json").write_text(
        json.dumps(entries, indent=2) + "\n", encoding="utf-8"
    )
    # Pages runs the branch through Jekyll unless told not to, and Jekyll
    # drops the _static and _sources directories Sphinx writes.
    (site / ".nojekyll").touch()
    return entries


def main(argv: list[str] | None = None) -> int:
    """Publish one built tree into the site."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("built", type=Path, help="the tree Sphinx wrote")
    parser.add_argument("version", help="latest, or the release tag")
    parser.add_argument("site", type=Path, help="a checkout of the site branch")
    parser.add_argument(
        "--keep",
        type=int,
        default=KEEP,
        help=f"releases kept beside latest and stable (default {KEEP})",
    )
    parser.add_argument(
        "--url",
        default="https://pulserver.github.io/pulserver",
        help="where the site is served from",
    )
    arguments = parser.parse_args(argv)
    entries = publish(
        arguments.built,
        arguments.version,
        arguments.site,
        arguments.url,
        arguments.keep,
    )
    listed = ", ".join(str(entry["name"]) for entry in entries)
    print(f"{arguments.version} published; the switcher lists {listed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
