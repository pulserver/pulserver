"""Vendor-neutral Pulseq system defaults."""

from __future__ import annotations

import pypulseq as _pp


class Opts(_pp.Opts):
    """PyPulseq system limits with vendor-neutral raster defaults.

    The constructor signature is upstream's, argument for argument, so an
    :class:`Opts` built by a PyPulseq script behaves identically here. Only
    the *defaults* differ, and only for the rasters and the dead/ringdown
    times.

    The raster defaults are the common multiples of the two vendors' hardware
    rasters -- 20 us for gradients and block durations (GE 4 us, Siemens
    10 us), 2 us for RF and ADC (Siemens 1 us and 100 ns, GE 2 us) -- so a
    sequence designed without explicit scanner limits plays on either without
    re-rastering. Dead and ringdown times default to zero because they are
    guidelines rather than hardware constants; set them from the system you
    are targeting when it matters.

    Amplitude, slew, gamma and B0 defaults remain PyPulseq's.

    Notes
    -----
    Coarser gradient rasters quantise ramps more coarsely, so a short blip
    designed at 20 us may come out longer than one designed at 10 us. Pass
    ``grad_raster_time`` explicitly when targeting one vendor.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> system = pp.Opts()
    >>> system.grad_raster_time, system.rf_raster_time
    (2e-05, 2e-06)

    Targeting one vendor means naming its raster and its limits:

    >>> ge = pp.Opts(
    ...     grad_raster_time=4e-6,
    ...     max_grad=50, grad_unit="mT/m",
    ...     max_slew=200, slew_unit="T/m/s",
    ... )
    >>> ge.grad_raster_time
    4e-06

    Amplitudes are held in Hz/m, so reading one back in mT/m divides by
    ``gamma``:

    >>> round(ge.max_grad / ge.gamma * 1e3, 1)
    50.0
    """

    def __init__(
        self,
        adc_dead_time: float | None = 0.0,
        adc_raster_time: float | None = 2e-6,
        block_duration_raster: float | None = 20e-6,
        gamma: float | None = None,
        grad_raster_time: float | None = 20e-6,
        grad_unit: str = "Hz/m",
        max_grad: float | None = None,
        max_slew: float | None = None,
        rf_dead_time: float | None = 0.0,
        rf_raster_time: float | None = 2e-6,
        rf_ringdown_time: float | None = 0.0,
        adc_samples_limit: int | None = None,
        adc_samples_divisor: int | None = None,
        rise_time: float | None = None,
        slew_unit: str = "Hz/m/s",
        B0: float | None = None,
    ) -> None:
        # Every raster is passed explicitly rather than left None: upstream
        # fills a None from ``pypulseq.Opts.default``, which is the Siemens
        # table, not this class's.
        super().__init__(
            adc_dead_time=adc_dead_time,
            adc_raster_time=adc_raster_time,
            block_duration_raster=block_duration_raster,
            gamma=gamma,
            grad_raster_time=grad_raster_time,
            grad_unit=grad_unit,
            max_grad=max_grad,
            max_slew=max_slew,
            rf_dead_time=rf_dead_time,
            rf_raster_time=rf_raster_time,
            rf_ringdown_time=rf_ringdown_time,
            adc_samples_limit=adc_samples_limit,
            adc_samples_divisor=adc_samples_divisor,
            rise_time=rise_time,
            slew_unit=slew_unit,
            B0=B0,
        )

    def set_as_default(self) -> None:
        """Use this object as the default for Pulserver's factories."""
        type(self).default = self

    @classmethod
    def reset_default(cls) -> None:
        """Restore Pulserver's vendor-neutral defaults."""
        cls.default = cls()


Opts.default = Opts()


def default_system(system: _pp.Opts | None) -> _pp.Opts:
    """Return ``system`` or Pulserver's current default limits."""
    return Opts.default if system is None else system


#: Fraction of the hardware limits a designed waveform is allowed to reach.
MAX_GRAD_DERATE = 0.9
MAX_SLEW_DERATE = 0.9


def apply_system_derates(
    opts: _pp.Opts,
    *,
    grad_derate: float = MAX_GRAD_DERATE,
    slew_derate: float = MAX_SLEW_DERATE,
) -> _pp.Opts:
    """Return a copy of ``opts`` whose gradient and slew limits are derated.

    The headroom belongs to the waveform being designed, not to the sequence:
    a designer that wants to stay clear of the ceiling derates the limits it
    designs against, while the sequence goes on declaring what the scanner
    actually has. Handing back a copy is what keeps those two apart -- a
    derate applied in place would travel back to every module that shares the
    system object and retroactively put already-designed events over a limit
    that moved under them.

    The base limits are carried on the copy, and a copy that already has them
    is derated from those rather than from its current values, so applying
    this twice does not compound.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits. Not modified.
    grad_derate, slew_derate : float, optional
        Fraction of the base ``max_grad`` / ``max_slew`` to allow.

    Returns
    -------
    pypulseq.Opts
        A copy of ``opts`` with the derated limits.

    See Also
    --------
    cap_system : the same copy-not-mutate contract, for an absolute ceiling.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> opts = pp.Opts(max_grad=40, grad_unit="mT/m", max_slew=150, slew_unit="T/m/s")
    >>> derated = pp.apply_system_derates(opts)
    >>> derated.max_grad == 0.9 * opts.max_grad
    True
    >>> derated is opts
    False

    The caller's own limits are untouched, and repeating does not compound:

    >>> pp.apply_system_derates(derated).max_grad == derated.max_grad
    True
    """
    import copy

    derated = copy.copy(opts)
    if not hasattr(derated, "_pulserver_base_max_grad"):
        derated._pulserver_base_max_grad = float(opts.max_grad)
    if not hasattr(derated, "_pulserver_base_max_slew"):
        derated._pulserver_base_max_slew = float(opts.max_slew)

    derated.max_grad = derated._pulserver_base_max_grad * float(grad_derate)
    derated.max_slew = derated._pulserver_base_max_slew * float(slew_derate)
    return derated


def cap_system(
    opts: _pp.Opts,
    *,
    max_grad: float | None = None,
    max_slew: float | None = None,
    grad_unit: str = "mT/m",
    slew_unit: str = "T/m/s",
) -> _pp.Opts:
    """Return a copy of ``opts`` whose gradient and slew limits are capped.

    A sequence never has to run its gradients as hard as the scanner allows.
    This lowers ``max_grad`` and/or ``max_slew`` to the smaller of the value
    the system reports and the ceiling asked for, so a plugin can hold a
    sequence below the hardware limits -- for peripheral-nerve-stimulation
    headroom, acoustic comfort, or eddy-current control -- while still
    honouring a scanner that is *less* capable than the ceiling.

    The limits are only ever lowered: a cap above what ``opts`` already allows
    leaves that axis untouched, so the same ceiling is safe across scanners of
    different gradient performance. ``None`` on either axis leaves it alone.

    A fresh :class:`Opts` is returned rather than mutating ``opts`` in place,
    because the offline CLI shares one system object across a plugin's
    ``get_default_protocol``, ``validate_protocol`` and ``make_sequence``
    calls, and an in-place cap would leak from one into the next.

    Parameters
    ----------
    opts : pypulseq.Opts
        System limits to cap. Not modified.
    max_grad : float or None, optional
        Gradient-amplitude ceiling, in ``grad_unit``. ``None`` leaves the
        gradient limit untouched. Default is ``None``.
    max_slew : float or None, optional
        Slew-rate ceiling, in ``slew_unit``. ``None`` leaves the slew limit
        untouched. Default is ``None``.
    grad_unit : str, optional
        Unit ``max_grad`` is given in, e.g. ``"mT/m"`` or ``"Hz/m"``. Default
        is ``"mT/m"``.
    slew_unit : str, optional
        Unit ``max_slew`` is given in, e.g. ``"T/m/s"`` or ``"Hz/m/s"``.
        Default is ``"T/m/s"``.

    Returns
    -------
    pypulseq.Opts
        A copy of ``opts`` with the capped limits.

    Examples
    --------
    >>> import pulserver.pypulseq as pp
    >>> from pypulseq.convert import convert
    >>> system = pp.Opts(max_grad=80, grad_unit="mT/m", max_slew=200, slew_unit="T/m/s")
    >>> capped = pp.cap_system(system, max_grad=40, max_slew=150)
    >>> capped is system
    False
    >>> round(convert(from_value=capped.max_grad, from_unit="Hz/m", to_unit="mT/m", gamma=capped.gamma))
    40
    """
    import copy

    from pypulseq.convert import convert

    capped = copy.copy(opts)
    if max_grad is not None:
        ceiling = convert(
            from_value=float(max_grad),
            from_unit=grad_unit,
            to_unit="Hz/m",
            gamma=opts.gamma,
        )
        capped.max_grad = min(float(opts.max_grad), ceiling)
    if max_slew is not None:
        ceiling = convert(
            from_value=float(max_slew),
            from_unit=slew_unit,
            to_unit="Hz/m/s",
            gamma=opts.gamma,
        )
        capped.max_slew = min(float(opts.max_slew), ceiling)
    return capped
