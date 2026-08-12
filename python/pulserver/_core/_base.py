"""Abstract contract for pulserver sequence plugins.

Examples
--------
>>> from pulserver import PulseqSequence, TypeinFloatParam, UIParam
>>> class DemoSequence(PulseqSequence):
...     def get_default_protocol(self, opts):
...         return {UIParam.TR: TypeinFloatParam(value=500.0, unit="ms")}
...     def validate_protocol(self, opts, protocol):
...         return {"valid": True, "duration": None, "info": None}
...     def make_sequence(self, opts, protocol, output_path):
...         _ = (opts, protocol, output_path)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pypulseq as pp

from ._params import Protocol


class PulseqSequence(ABC):
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

    ``pulserver.Sequence`` is an alias for this class.

    Examples
    --------
    A complete, minimal plugin:

    >>> import pulserver.pypulseq as pp
    >>> from pulserver import PulseqSequence, TypeinFloatParam, UIParam, params
    >>> class DemoSequence(PulseqSequence):
    ...     def get_default_protocol(self, opts):
    ...         return {UIParam.TR: TypeinFloatParam(value=500.0, unit="ms")}
    ...     def validate_protocol(self, opts, protocol):
    ...         tr_ms = params.param_float(protocol, UIParam.TR)
    ...         return {"valid": tr_ms >= 10.0, "duration": None, "info": None}
    ...     def make_sequence(self, opts, protocol, output_path):
    ...         delay = pp.make_delay(1e-3)
    ...         seq = pp.Sequence(opts)
    ...         seq.add_block(delay)
    ...         seq.write(output_path)
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
    def get_default_protocol(self, opts: pp.Opts) -> Protocol:
        """Return the default protocol for this sequence.

        Parameters
        ----------
        opts : pypulseq.Opts
            Scanner/system limits passed in by the bridge.

        Returns
        -------
        Protocol
            Default protocol mapping keyed by ``UIParam`` or canonical key
            strings.

        Examples
        --------
        >>> from pulserver import TypeinFloatParam, UIParam
        >>> protocol = {UIParam.TR: TypeinFloatParam(value=500.0, unit="ms")}
        >>> protocol[UIParam.TR].value
        500.0
        """
        ...

    @abstractmethod
    def validate_protocol(self, opts: pp.Opts, protocol: Protocol) -> dict:
        """Validate a protocol against hardware constraints.

        Parameters
        ----------
        opts : pypulseq.Opts
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
    def make_sequence(self, opts: pp.Opts, protocol: Protocol, output_path: str) -> None:
        """Build the full sequence and write it to disk.

        Parameters
        ----------
        opts : pypulseq.Opts
            Scanner/system limits passed in by the bridge.
        protocol : Protocol
            Final validated protocol mapping.
        output_path : str
            Destination path for the generated ``.seq`` file.
        """
        ...
