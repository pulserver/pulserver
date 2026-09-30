"""Slots counted across every process on the host, not within one."""

import subprocess
import sys
import textwrap

from pulserver.recon._runtime.concurrency import HostSlots

TAKE_IN_ANOTHER_PROCESS = textwrap.dedent(
    """
    import sys
    from pulserver.recon._runtime.concurrency import HostSlots

    count = int(sys.argv[2])
    slots = HostSlots([None] * count, sys.argv[1])
    taken = []
    # At most one round: a slot that can be taken twice is a slot that bounds
    # nothing, and this says so rather than taking forever.
    for _ in range(count + 1):
        slot = slots.take(wait=False)
        if slot is None:
            break
        taken.append(slot)
    print(len(taken))
    """
)


def taken_by_another_process(directory, count):
    """How many slots a second process can take of ``count``."""
    done = subprocess.run(
        [sys.executable, "-c", TAKE_IN_ANOTHER_PROCESS, str(directory), str(count)],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(done.stdout.strip())


def test_a_slot_one_process_holds_is_not_free_to_another(tmp_path):
    """The bound is the host's: two proxies must not each admit the whole of it."""
    slots = HostSlots([None, None], tmp_path)
    held = slots.take(wait=False)
    assert held is not None
    try:
        assert taken_by_another_process(tmp_path, 2) == 1
    finally:
        slots.release(held)


def test_every_slot_is_free_again_once_released(tmp_path):
    slots = HostSlots([None, None], tmp_path)
    held = [slots.take(wait=False), slots.take(wait=False)]
    assert all(slot is not None for slot in held)
    assert taken_by_another_process(tmp_path, 2) == 0
    for slot in held:
        slots.release(slot)
    assert taken_by_another_process(tmp_path, 2) == 2


def test_a_process_that_dies_frees_what_it_held(tmp_path):
    """A lock is the kernel's, so nothing has to notice that a proxy died."""
    taker = subprocess.Popen(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys, time
                from pulserver.recon._runtime.concurrency import HostSlots
                slots = HostSlots([None], sys.argv[1])
                assert slots.take(wait=False) is not None
                print("held", flush=True)
                time.sleep(60)
                """
            ),
            str(tmp_path),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert taker.stdout.readline().strip() == "held"
        assert HostSlots([None], tmp_path).take(wait=False) is None
    finally:
        taker.kill()
        taker.wait(timeout=10)
    assert HostSlots([None], tmp_path).take(wait=False) is not None


def test_a_slot_holds_the_device_of_its_index(tmp_path):
    """Two proxies must not put different series on one GPU."""
    slots = HostSlots(["cuda:0", "cuda:1"], tmp_path)
    first = slots.take(wait=False)
    assert first.device == "cuda:0"
    second = slots.take(wait=False)
    assert second.device == "cuda:1"
    slots.release(first)
    slots.release(second)


def test_waiting_returns_a_slot_another_process_frees(tmp_path):
    slots = HostSlots([None], tmp_path)
    held = slots.take(wait=False)
    assert slots.take(wait=False) is None
    slots.release(held)
    taken = slots.take(wait=True, poll=0.01)
    assert taken is not None
    slots.release(taken)


def test_the_slots_of_one_host_are_one_directory_by_default():
    """Nothing has to be configured for two proxies to share a bound."""
    from pulserver.recon._runtime.concurrency import DEFAULT_SLOT_DIRECTORY

    assert HostSlots([None]).directory == DEFAULT_SLOT_DIRECTORY
    assert DEFAULT_SLOT_DIRECTORY.is_absolute()
