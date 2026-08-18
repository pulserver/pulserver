"""Build an EPG signal dictionary and temporal subspace from LiveSDK waveforms."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from pulserver.recon import (
    TissueProperties,
    decode_sequence_description,
    simulate_subspace,
)


def build_subspace_basis(
    waveforms: Iterable[Any],
    *,
    sequence: str,
    t1_ms: Any,
    t2_ms: Any,
    rank: int,
    repetitions: int = 1,
    echo_only: bool = False,
    device: Any = None,
):
    """Decode one LiveSDK resource set and estimate its temporal basis.

    Parameters
    ----------
    waveforms
        ISMRMRD waveforms received from LiveSDK. Unrelated physiological
        waveforms are ignored by the decoder.
    sequence
        One of ``"fse"``, ``"spgr"``, ``"ssfp-echo"``, ``"ssfp-fid"``,
        or ``"bssfp"``.
    t1_ms, t2_ms
        Scalars or arrays defining the tissue dictionary.
    rank
        Number of left singular vectors to retain.
    repetitions
        Number of times to repeat the serialized state-machine cycle.
    echo_only
        Record only ADC positions whose producer-side ``echo`` flag is true.
    device
        Torch device, for example ``"cuda"``.
    """

    resource = decode_sequence_description(waveforms)
    description = resource.subsequence(0)
    return simulate_subspace(
        description,
        sequence,
        TissueProperties(t1_ms=t1_ms, t2_ms=t2_ms),
        rank=rank,
        repetitions=repetitions,
        record="echo" if echo_only else "all",
        device=device,
    )
