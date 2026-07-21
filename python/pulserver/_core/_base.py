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

    Notes
    -----
    Subclass this and implement the three abstract methods. The Python bridge
    discovers the subclass automatically.
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
