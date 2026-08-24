"""Abstract contract for pulserver sequence plugins.

Examples
--------
>>> from pulserver.design import SequencePlugin, TypeinFloatParam, UIParam
>>> class DemoSequence(SequencePlugin):
...     def get_default_protocol(self, system):
...         return {UIParam.TR: TypeinFloatParam(value=500.0, unit="ms")}
...     def validate_protocol(self, system, protocol):
...         return {"valid": True, "duration": None, "info": None}
...     def make_sequence(self, system, protocol, output_path, *, offline=False):
...         _ = (system, protocol, output_path, offline)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pypulseq as pp

from ._params import Protocol


class SequencePlugin(ABC):
    """Base class that every pulserver sequence plugin must satisfy.

    A plugin is one file declaring three things: the protocol it exposes to
    the scanner UI, whether a given protocol is feasible, and how to turn that
    protocol into a ``.seq`` file. Implement the three abstract methods below
    and the bridge discovers the subclass automatically — no registration, no
    entry point.

    The split matters at runtime: ``validate_protocol`` runs on every UI
    interaction and must stay fast and side-effect free, while
    ``make_sequence`` runs once, at download, and is where the waveforms are
    actually built.

    Examples
    --------
    A complete, minimal plugin:

    >>> import pulserver.pypulseq as pp
    >>> from pulserver.design import (
    ...     SequencePlugin, TypeinFloatParam, UIParam, params, write_sequence
    ... )
    >>> class DemoSequence(SequencePlugin):
    ...     def get_default_protocol(self, system):
    ...         return {UIParam.TR: TypeinFloatParam(value=500.0, unit="ms")}
    ...     def validate_protocol(self, system, protocol):
    ...         tr_ms = params.param_float(protocol, UIParam.TR)
    ...         return {"valid": tr_ms >= 10.0, "duration": None, "info": None}
    ...     def make_sequence(self, system, protocol, output_path, *, offline=False):
    ...         delay = pp.make_delay(1e-3)
    ...         seq = pp.Sequence(system)
    ...         seq.add_block(delay)
    ...         write_sequence(seq, output_path, offline=offline)
    >>> plugin = DemoSequence()
    >>> plugin.get_default_protocol(pp.Opts())[UIParam.TR].value
    500.0

    Expose it as an offline CLI with :func:`pulserver.run_cli`.

    See Also
    --------
    pulserver.run_cli : offline command-line entry point for a plugin.
    pulserver.params : read and write protocol values by canonical key.
    """

    @abstractmethod
    def get_default_protocol(self, system: pp.Opts) -> Protocol:
        """Return the default protocol for this sequence.

        Parameters
        ----------
        system : pypulseq.Opts
            Scanner/system limits passed in by the bridge.

        Returns
        -------
        Protocol
            Default protocol mapping keyed by ``UIParam`` or canonical key
            strings.

        Examples
        --------
        >>> from pulserver.design import TypeinFloatParam, UIParam
        >>> protocol = {UIParam.TR: TypeinFloatParam(value=500.0, unit="ms")}
        >>> protocol[UIParam.TR].value
        500.0
        """
        ...

    @abstractmethod
    def validate_protocol(self, system: pp.Opts, protocol: Protocol) -> dict:
        """Validate a protocol against hardware constraints.

        Parameters
        ----------
        system : pypulseq.Opts
            Scanner/system limits passed in by the bridge.
        protocol : Protocol
            Current protocol mapping to validate.

        Returns
        -------
        dict
            Dictionary with the keys ``valid``, ``duration``, and ``info``.

        Notes
        -----
        This method is called frequently during interactive validation and
        should avoid file I/O.
        """
        ...

    @abstractmethod
    def make_sequence(
        self,
        system: pp.Opts,
        protocol: Protocol,
        output_path: str,
        *,
        offline: bool = False,
    ) -> None:
        """Build the full sequence and write it to disk.

        Parameters
        ----------
        system : pypulseq.Opts
            Scanner/system limits passed in by the bridge.
        protocol : Protocol
            Final validated protocol mapping.
        output_path : str
            Destination path for the generated ``.seq`` file.
        offline : bool, default False
            Whether the file is going somewhere other than a scanner, which
            decides the form it is written in. Hand it to
            :func:`pulserver.write_sequence` rather than reading it: the
            default is the scanner's form, and :func:`pulserver.run_cli`
            passes ``True``.
        """
        ...
