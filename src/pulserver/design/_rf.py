"""The RF an evaluation plays and the scanner controls its amplitudes follow."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..protocol import FloatKey, UserKey

if TYPE_CHECKING:
    import pypulseqpp as pp

#: A scanner control the amplitude of an RF instance is proportional to: the
#: flip angle, or a float user entry.
RfControl = Literal[FloatKey.FLIP] | UserKey


@dataclass(frozen=True)
class RfLayout:
    """The RF definitions a plugin plays and the instances of one repetition.

    An instance plays the RF amplitude ``amplitude * peak_hz * waveform`` of its
    definition, as :meth:`pypulseqpp.Sequence.rf_instances` states. A control
    scales its instance with the protocol: a scanner that plays another value of
    the control multiplies the amplitude by the ratio of that value to the value
    in the evaluated protocol. An instance with control ``None`` plays at the
    amplitude it was evaluated at.

    Attributes
    ----------
    instances
        The definitions, numbered by first play, and the instances in play order.
    control
        One entry per instance: the control its amplitude is proportional to,
        or ``None``.
    period
        The time in seconds over which the instances repeat.

    Raises
    ------
    ValueError
        If ``control`` does not hold one entry per instance, or ``period`` is
        not positive.
    TypeError
        If a control is neither ``None``, ``FloatKey.FLIP`` nor a
        :class:`~pulserver.protocol.UserKey` member. Other keys do not scale
        RF, and a plain string is not a member.
    """

    instances: pp.RfInstances
    control: tuple[RfControl | None, ...]
    period: float

    def __post_init__(self) -> None:
        count = len(self.instances.definition)
        if len(self.control) != count:
            raise ValueError(
                f"an RF layout needs one control per instance: {len(self.control)} "
                f"controls for {count} instances"
            )
        for control in self.control:
            if (
                control is not None
                and control is not FloatKey.FLIP
                and not isinstance(control, UserKey)
            ):
                raise TypeError(
                    "an RF amplitude is proportional to the flip angle or to a user "
                    f"entry, not to {control!r}"
                )
        if not self.period > 0:
            raise ValueError(
                f"the period of an RF layout is positive, not {self.period!r}"
            )

    @classmethod
    def of(
        cls,
        repetition: pp.Sequence,
        control: RfControl | Iterable[RfControl | None] | None,
        *,
        period: float | None = None,
    ) -> RfLayout:
        """Return the layout of the RF instances of a sequence.

        Parameters
        ----------
        repetition
            The sequence whose RF instances are the layout: the whole scan, or
            one representative repetition such as a TR, a shot or a train.
        control
            The control of every instance, or ``None`` for none; or one control
            per instance, in play order. A single key is one control.
        period
            Seconds over which the instances repeat. The duration of
            ``repetition`` where omitted.
        """
        instances = repetition.rf_instances()
        if (
            control is None
            or isinstance(control, str)
            or not isinstance(control, Iterable)
        ):
            controls = (control,) * len(instances.definition)
        else:
            controls = tuple(control)
        return cls(
            instances,
            controls,
            repetition.duration()[0] if period is None else period,
        )
