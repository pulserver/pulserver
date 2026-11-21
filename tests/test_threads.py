import subprocess
import sys

from pulserver._threads import BLAS_THREAD_VARIABLES, serial_blas


def test_blas_is_given_one_thread_where_no_count_is_set():
    environ = {}
    serial_blas(environ)
    assert all(environ[name] == "1" for name in BLAS_THREAD_VARIABLES)


def test_a_blas_thread_count_set_by_the_site_is_kept():
    environ = {"OPENBLAS_NUM_THREADS": "4"}
    serial_blas(environ)
    assert environ["OPENBLAS_NUM_THREADS"] == "4"


def test_torch_keeps_its_openmp_threads():
    environ = {}
    serial_blas(environ)
    assert "OMP_NUM_THREADS" not in environ


def test_the_command_does_not_load_numpy_before_it_serialises_blas():
    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, pulserver._cli; print('numpy' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert loaded == "False"
