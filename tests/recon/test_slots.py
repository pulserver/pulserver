"""The slots series take, and the GPU each holds."""

import pytest

from pulserver.recon._runtime import concurrency
from pulserver.recon._runtime.concurrency import Slots, gpu_devices, slot_devices


@pytest.fixture
def gpus(monkeypatch):
    """Make ``nvidia-smi`` list this many GPUs, with ample host memory."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(concurrency, "_available_ram_gb", lambda: 1000.0)

    def listed(count):
        monkeypatch.setattr(concurrency, "_listed_gpus", lambda: count)

    return listed


def test_without_a_gpu_every_slot_is_on_the_host(gpus):
    gpus(0)
    assert slot_devices(3) == (None, None, None)


def test_the_slots_take_the_gpus_in_turn(gpus):
    gpus(2)
    assert slot_devices(3) == ("cuda:0", "cuda:1", "cuda:0")
    assert slot_devices(per_gpu=2) == ("cuda:0", "cuda:1", "cuda:0", "cuda:1")


def test_host_memory_caps_the_slots_the_gpus_offer(gpus, monkeypatch):
    gpus(4)
    monkeypatch.setattr(concurrency, "_available_ram_gb", lambda: 125.0)
    assert slot_devices() == ("cuda:0", "cuda:1")


def test_cuda_visible_devices_names_the_gpus_numbered_from_zero(gpus, monkeypatch):
    gpus(8)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")
    assert gpu_devices() == ("cuda:0", "cuda:1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert gpu_devices() == ()


def test_a_slot_is_taken_until_it_is_released():
    slots = Slots(["cuda:0"])
    slot = slots.take(wait=False)
    assert slot.device == "cuda:0"
    assert slots.take(wait=False) is None
    slots.release(slot)
    assert slots.take(wait=False) is slot
