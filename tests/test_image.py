"""The console image's entry point: its console, with the options given after the image's name added to its own, or another pulserver command in the console's place."""

import ipaddress
import os
import subprocess
from pathlib import Path

import pytest

from pulserver.virtual._console import _parser

ENTRYPOINT = Path(__file__).parents[1] / "docker" / "entrypoint.sh"


@pytest.fixture
def run(tmp_path):
    """Run the entry point with a ``pulserver`` that prints its arguments, one per line."""
    stub = tmp_path / "pulserver"
    stub.write_text('#!/bin/sh\nfor argument in "$@"; do echo "$argument"; done\n')
    stub.chmod(0o755)
    environment = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"}

    def run(*arguments: str) -> list[str]:
        done = subprocess.run(
            ["sh", str(ENTRYPOINT), *arguments],
            capture_output=True,
            text=True,
            env=environment,
            check=True,
        )
        return done.stdout.splitlines()

    return run


def test_the_image_serves_its_console_without_arguments(run):
    command, *options = run()

    args = _parser().parse_args(options)
    assert command == "console"
    assert args.plugins == [Path("/console/user/plugins")]
    assert args.recon_plugins == [Path("/console/user/recon")]
    assert (args.fields, args.port, args.speed) == (Path("/console/fields"), 8765, 1.0)
    assert ipaddress.ip_address(args.host).is_unspecified


def test_options_after_the_image_s_name_follow_its_console_s_own(run):
    added = (
        "--spins",
        "4",
        "--diffusion",
        "--speed",
        "2",
        "--origin",
        "http://example.org",
    )
    command, *options = run(*added)

    args = _parser().parse_args(options)
    assert [command, *options] == [*run(), *added]
    assert (args.spins, args.diffusion, args.speed) == (4, True, 2.0)
    assert args.origins == [
        "https://pulserver.github.io",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://example.org",
    ]


def test_a_command_runs_in_the_console_s_place(run):
    assert run("design", "--help") == ["design", "--help"]
