"""The evaluation of a shipped scanner sequence: the chain its function designs, timed, read back and costed."""

from __future__ import annotations

from collections.abc import Callable

import pypulseqpp as pp
from pypulseqpp import sequences

from ..design import Evaluation, Protocol, RfLayout, SequencePlugin
from ..protocol import ProtocolKey, UIParam


def _first_adc_bandwidth(main: pp.Sequence) -> float:
    """Return the receiver bandwidth in Hz of the first ADC event: the inverse of its dwell time."""
    tables = main.libraries()
    played = tables.blocks[:, 4]
    return 1.0 / float(tables.adc[played[played > 0][0] - 1, 1])


#: The entries that hold the value the design achieved, and where the main
#: sequence records it. A multi-echo sequence lists every echo time; the entry
#: holds the first.
_ACHIEVED: dict[ProtocolKey, Callable[[pp.Sequence], float]] = {
    UIParam.TE: lambda main: main.definitions["TE"][0],
    UIParam.TR: lambda main: main.definitions["TR"][0],
    UIParam.BANDWIDTH: _first_adc_bandwidth,
    UIParam.SLICE_THICKNESS: lambda main: main.definitions["SliceThickness"][0],
}


def _rf_layout(main: pp.Sequence, scaled: bool) -> RfLayout:
    """Return the RF of ``main``, the flip angle scaling each excitation when ``scaled``, and no entry any other pulse."""
    instances = main.rf_instances()
    control = [
        UIParam.FLIP
        if scaled and instances.definitions[number].use == "excitation"
        else None
        for number in instances.definition
    ]
    return RfLayout.of(main, control)


def evaluation(
    plugin: SequencePlugin, system: pp.Opts, protocol: Protocol
) -> Evaluation:
    """Return the evaluation of a protocol from the chain the plugin's function designs.

    The scan time is the summed duration of the chain. An entry for the echo
    time, the repetition time, the receiver bandwidth or the slice thickness
    holds the value the design achieved: the ``TE`` and ``TR`` definitions of
    the main sequence, the inverse of the dwell time of its first ADC event
    and its ``SliceThickness`` definition. The RF layout is the whole main
    sequence, whose period is its duration, with the flip angle scaling the
    excitations where the plugin has a flip entry.
    """
    designed = plugin.app(system, **protocol.arguments)
    chain = [designed] if isinstance(designed, pp.Sequence) else list(designed)
    main = chain[-1]
    achieved = {
        key: read(main) for key, read in _ACHIEVED.items() if key in plugin.protocol
    }
    return Evaluation(
        protocol.replace(achieved),
        sequences.duration(chain),
        rf_layout=_rf_layout(main, UIParam.FLIP in plugin.protocol),
    )
