"""Service logs: one file collects a process's log, its output and its children's."""

import subprocess
import sys

SCRIPT = """
import logging, multiprocessing, sys, warnings
from pulserver._logs import configure

def child():
    configure()
    logging.getLogger("plugin").info("from a spawned child")
    print("printed by a spawned child")

if __name__ == "__main__":
    configure("INFO", sys.argv[1])
    logging.getLogger("service").info("from the service")
    print("printed by the service")
    warnings.warn("warned by the service")
    process = multiprocessing.get_context("spawn").Process(target=child)
    process.start()
    process.join()
"""


def test_a_logfile_collects_log_output_warnings_and_spawned_children(tmp_path):
    script = tmp_path / "service.py"
    script.write_text(SCRIPT)
    log = tmp_path / "service.log"
    result = subprocess.run(
        [sys.executable, str(script), str(log)],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert result.stdout == result.stderr == ""
    text = log.read_text()
    for line in (
        "INFO service: from the service",
        "printed by the service",
        "warned by the service",
        "INFO plugin: from a spawned child",
        "printed by a spawned child",
    ):
        assert line in text
