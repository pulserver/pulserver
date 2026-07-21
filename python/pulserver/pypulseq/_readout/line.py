"""Cartesian line (GRE-family) readout trains.

Two callable readout objects, built once and then called per shot to append a
single- or multi-echo Cartesian line readout to a sequence:

* :class:`Line2D` -- 2D line readout: one ``ky`` (``LIN``) per shot.
* :class:`Line3D` -- 3D line readout: one ``ky``/``kz`` (``LIN``/``PAR``) pair
  per shot.

Both are *train only*: the excitation lives in the caller, exactly like
:mod:`pulserver.pypulseq._readout.epi` / :mod:`pulserver.pypulseq._readout.fse`.
Unlike those two families, a line-readout shot visits a *single* k-space line
possibly several times (multi-echo, same ``ky``/``kz``, different TE/contrast)
-- it never advances the phase encode within a shot. It emits absolute
``SET LIN``/``SET PAR`` labels on the first ADC block.

A shot is three blocks:

.. code-block:: text

    (x prewind, y enc [z enc])   (x read, adc) x num_echoes   (x rewind/spoil, y enc [z enc])

The prewind/rewind block also carries the phase-encode (and, for
:class:`Line3D`, partition) gradient -- prewound before the train and
symmetrically rewound after it, on every axis, regardless of the readout
axis's spoiling below. ``num_echoes > 1`` inserts extra ``(x read, adc)``
blocks in between, either

* **bipolar** (``flyback=False``, default) -- alternating readout polarity,
  echoes abut directly with no gap; or
* **flyback** (``flyback=True``) -- every echo the same polarity, separated
  by a dedicated rewind gradient that retraces the same k-window.

Every echo, first to last, samples the identical (possibly partial-Fourier
asymmetric, see ``pf``) k-window -- only the *direction* (bipolar) or a
retrace (flyback) differs. ``ECO`` labels are the train's own responsibility
for a multi-echo shot (``num_echoes > 1``): ``SET ECO 0`` on the first echo,
``INC ECO 1`` on every following one. ``LIN``/``PAR`` are set once per shot
on the first ADC block because ``ky``/``kz`` remains fixed for the train.

Spoiling (``spoil_position``)
------------------------------
The readout-axis prewind/rewind pair is sized from the train's own k-window
(``spoil_factor * k_width``, a fraction of the readout's own bandwidth-driven
k-space width -- the same convention as
:mod:`pulserver.pypulseq._readout.fse`'s ``x_spoil_factor``), not from an
external slice thickness -- this train has no notion of "slice" vs "slab"
(see the EPI/FSE module docstrings for the same design choice).

* ``"none"`` -- fully balanced: the rewind exactly cancels every axis's
  prewind, net zero gradient moment per shot (true bSSFP / balanced
  condition).
* ``"post"`` (default) -- after the last echo, an extra ``spoil_factor *
  k_width`` of area is tacked directly onto wherever the (unmodified) natural
  echo excursion ends -- SSFP-FID / spoiled-GRE / FISP / GRASS: the extra
  dephasing happens *after* the read, and only affects the *next* shot's
  starting magnetization, not this one's own acquired data.
* ``"pre"`` -- the mirror construction, before echo 0 instead: SSFP-Echo /
  PSIF / CE-FAST. Matching ``"post"`` exactly (gre.py's own convention), the
  bridge's own area is ``spoil_factor * k_width`` alone, independent of the
  prewind's normal target -- *not* "the normal prewind target plus spoil",
  which would force the bridge to swing through/past zero to hit a
  large-magnitude, wrong-signed area (a real bug caught during development).
  Unlike ``"post"``, this necessarily shifts the k-window this shot's own
  echo(es) sample by a fixed, shot-independent amount (``spoil_factor *
  k_width`` doesn't depend on ``pe_idx``) -- a benign linear phase ramp on
  reconstruction (Fourier shift theorem), not an aliasing/positioning error,
  since it is identical every shot.

The phase-encode/partition axes are always fully rewound regardless of
``spoil_position`` -- only the readout axis is affected by spoiling.

Whenever ``spoil_position != "none"``, the spoiled prewind/rewind is a
*bridged* trapezoid -- the same construction as ``gre.py``'s
``unbalanced_line``: the adjoining echo (echo 0 for ``"pre"``, the last echo
for ``"post"``) is split (:func:`pypulseq.split_gradient_at`) to drop the
ramp it no longer needs, and the spoiler is reshaped
(:func:`pypulseq.make_extended_trapezoid_area`) to hand off directly at the
readout's own plateau amplitude -- never returning to, and re-ramping from,
zero in between. The spoiler's sign always matches the plateau amplitude it
hands off to/from, so the resulting shape is a monotonic ``(0,
spoil_plateau, readout_plateau)``-style staircase (rise to an elevated
plateau, then settle to the echo's own amplitude) rather than a reversal
through zero. A plain (0 -> area -> 0) trapezoid is used only for the
unspoiled side (both sides, when ``spoil_position == "none"``).

Partial Fourier (``pf``)
-------------------------
``pf`` is the *readout* fractional-echo factor (asymmetric echo along x, in
``(0.5, 1]``) -- exactly :mod:`pulserver.pypulseq._readout.fse`'s ``pf``.
Phase-encode partial Fourier is the caller's job (pass fewer ``pe_idx``/
``par_idx`` values across shots).

Rotation
--------
``__call__`` accepts an optional ``rotation`` (a scipy-``Rotation``-like
object, see :func:`pulserver.pypulseq.make_rotation`), attached to every block
the train emits -- same convention as :mod:`pulserver.pypulseq._readout.epi`.
"""

from __future__ import annotations

__all__ = [
    "Line2D",
    "Line3D",
    "DEFAULT_SPOIL_FACTOR",
]

import copy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pypulseq as pp

from .._system import (
    DEFAULT_BANDWIDTH_HZ_PX,
    apply_system_derates,
    ceil_to_raster,
    quantize_readout_timing,
    round_to_raster,
)
from ._base import Readout, normalize_rotation

READOUT_GRAD_MARGIN = 0.95  # headroom below max_grad for the readout plateau
DEFAULT_SPOIL_FACTOR = 1.0  # readout-axis spoiler area as a fraction of k_width
SPOIL_POSITIONS = ("none", "pre", "post")


@dataclass(frozen=True)
class _LineState:
    pe_idx: int
    par_idx: int
    adc_phase_rad: float
    rotation: object | None


def _make_axis(spec):
    """Build the ``(axis, delta_k, areas, max_area, label)`` bundle from a ``pe_spec``/``par_spec`` 4-tuple."""
    if spec is None:
        return None
    axis, fov_m, n_points, label = spec
    n_points = int(n_points)
    delta_k = 1.0 / float(fov_m)
    areas = (np.arange(n_points) - 0.5 * n_points) * delta_k
    return SimpleNamespace(
        axis=axis, n=n_points, label=label, delta_k=delta_k, areas=areas,
        max_area=float(np.max(np.abs(areas))) if n_points > 0 else 0.0,
    )


def _min_trap_duration(system, axis, area):
    """Shortest ``pp.make_trapezoid`` duration (s) for ``area`` on ``axis`` (0.0 if ``area == 0``)."""
    if area == 0.0:
        return 0.0
    trial = pp.make_trapezoid(channel=axis, area=abs(area), system=system)
    return pp.calc_duration(trial)


def _bridge(system, channel, area, grad_start, grad_end):
    """The shortest ``grad_start -> ... -> grad_end`` extended trapezoid achieving ``area``.

    Thin wrapper around :func:`pypulseq.make_extended_trapezoid_area` -- the
    same bridged-crusher construction ``gre.py``'s ``unbalanced_line`` uses:
    it searches for a slew-safe ramp/plateau solution directly (unlike a
    fixed-ramp-time trapezoid, this stays feasible even when ``grad_start``/
    ``grad_end`` and the solved plateau have opposite signs and/or a combined
    swing approaching ``2 * max_grad``, exactly the readout-vs-spoiler case
    here). Returned with ``delay=0`` (left-aligned); callers that need the
    bridge to instead *end* at a fixed block boundary (see ``spoil_position
    == "pre"``) shift it there by setting ``.delay`` afterward.
    """
    grad, _, _ = pp.make_extended_trapezoid_area(
        area=area, channel=channel, grad_start=grad_start, grad_end=grad_end, system=system
    )
    return grad


def _build_readout_x(opts, ro_axis, nx, fov_x_m, bandwidth_hz_px, oversamp, pf):
    """Compute the readout-axis geometry: the flat readout trapezoid (+ its negation) and the ADC.

    ``oversamp`` densifies the sampling (delta_k shrinks, resolution/area
    fixed); ``pf`` truncates the pre-echo side (fewer pre-echo samples),
    shifting the echo (k=0 crossing) earlier in the ADC -- identical
    semantics to :func:`pulserver.pypulseq._readout.fse._build_readout`.
    """
    if not (0.5 < pf <= 1.0):
        raise ValueError(f"pf (readout fractional echo) must be in (0.5, 1], got {pf}")
    if oversamp < 1.0:
        raise ValueError(f"oversamp must be >= 1, got {oversamp}")

    delta_k = 1.0 / (oversamp * fov_x_m)
    n_full = int(round(oversamp * nx))
    n_post = n_full // 2
    n_samp = max(n_post + 1, int(round(pf * n_full)))
    n_pre = n_samp - n_post
    k_width = n_samp * delta_k

    dwell_s, flat_time_s = quantize_readout_timing(
        nx_ro=n_samp,
        target_bw_hz_px=bandwidth_hz_px,
        grad_raster_s=opts.grad_raster_time,
        adc_raster_s=opts.adc_raster_time,
        min_flat_time_s=k_width / (READOUT_GRAD_MARGIN * opts.max_grad),
    )
    gx = pp.make_trapezoid(channel=ro_axis, flat_area=k_width, flat_time=flat_time_s, system=opts)
    gx_neg = pp.scale_grad(gx, -1.0)
    adc = pp.make_adc(num_samples=n_samp, dwell=dwell_s, delay=gx.rise_time, system=opts)
    adc_center_s = gx.rise_time + (n_pre + 0.5) * dwell_s

    return SimpleNamespace(
        gx=gx, gx_neg=gx_neg, adc=adc, k_width=k_width, delta_k=delta_k,
        n_samples=n_samp, n_pre=n_pre, n_post=n_post, dwell_s=dwell_s, adc_center_s=adc_center_s,
    )


class _LineTrain(Readout):
    """Shared Cartesian line readout train (see module docstring).

    Subclasses (:class:`Line2D`, :class:`Line3D`) configure the concurrent
    phase-encode axis (``pe_spec``) and, for 3D, a partition axis
    (``par_spec``).
    """

    def __init__(
        self,
        system,
        fov_x_m,
        nx,
        num_echoes=1,
        *,
        pe_spec=None,
        par_spec=None,
        bandwidth_hz_px=DEFAULT_BANDWIDTH_HZ_PX,
        oversamp=1.0,
        pf=1.0,
        flyback=False,
        spoil_position="post",
        spoil_factor=DEFAULT_SPOIL_FACTOR,
        esp_s=None,
        start_polarity_positive=True,
        ro_axis="x",
        derate=True,
    ):
        Readout.__init__(self, system)
        if int(num_echoes) < 1:
            raise ValueError(f"num_echoes must be >= 1, got {num_echoes}")
        if spoil_position not in SPOIL_POSITIONS:
            raise ValueError(f"spoil_position must be one of {SPOIL_POSITIONS}, got {spoil_position!r}")

        if derate:
            apply_system_derates(system)
        self._opts = system
        self.fov_x_m, self.nx = float(fov_x_m), int(nx)
        self.num_echoes = int(num_echoes)
        self.ro_axis = ro_axis
        self._flyback = bool(flyback)
        self._spoil_position = spoil_position
        self._spoil_factor = float(spoil_factor)
        self._start_pos = bool(start_polarity_positive)

        raster = system.grad_raster_time
        self._pe = _make_axis(pe_spec)
        self._par = _make_axis(par_spec)

        self._ro = _build_readout_x(system, ro_axis, self.nx, self.fov_x_m, bandwidth_hz_px, oversamp, pf)
        self.n_samples = self._ro.n_samples
        self.adc_dwell_s = self._ro.dwell_s
        self.adc_center_s = self._ro.adc_center_s

        # --- echo spacing ---------------------------------------------
        gx_dur = pp.calc_duration(self._ro.gx)
        if self._flyback and self.num_echoes > 1:
            gx_fb_natural = pp.make_trapezoid(channel=ro_axis, area=-self._ro.gx.area, system=system)
            flyback_dur = ceil_to_raster(pp.calc_duration(gx_fb_natural), raster)
            self._gx_flyback = pp.make_trapezoid(
                channel=ro_axis, area=-self._ro.gx.area, duration=flyback_dur, system=system
            )
        else:
            flyback_dur = 0.0
            self._gx_flyback = None
        esp_min = ceil_to_raster(gx_dur + flyback_dur, raster)
        if esp_s is None:
            self.esp = esp_min
        else:
            self.esp = round_to_raster(float(esp_s), raster)
            if self.esp < esp_min - 1e-9:
                raise ValueError(f"esp_s={esp_s * 1e3:.3f} ms is below the minimum feasible ESP ({esp_min * 1e3:.3f} ms)")
        self._esp_pad_s = round_to_raster(self.esp - esp_min, raster)

        # --- readout-axis k-space bookkeeping ---------------------------
        # A *plain* prewind/rewind trapezoid (spoil_position on the other
        # side, or "none") returns to zero, so its own block is followed by
        # an echo that ramps 0 -> gx.amplitude -> 0 *within its own block* --
        # its area target is therefore the *block* boundary, one ramp's worth
        # of area beyond the (ADC-sampled) flat-top window. A *bridged*
        # prewind/rewind (this side spoiled) instead targets the flat-top
        # boundary directly and hands off at gx.amplitude (nonzero) -- the
        # adjoining echo is split (via pp.split_gradient_at) to drop the
        # ramp it no longer needs, exactly mirroring gre.py's unbalanced_line
        # bridged-crusher construction (no redundant return-through-zero
        # between the echo and its spoiler).
        n_pre, n_post, dk = self._ro.n_pre, self._ro.n_post, self._ro.delta_k
        ramp_area = 0.5 * self._ro.gx.amplitude * self._ro.gx.rise_time

        def _flat_k_start(positive):
            return (-n_pre * dk) if positive else (n_post * dk)

        def _flat_k_end(positive):
            return (n_post * dk) if positive else (-n_pre * dk)

        def _echo_amp(positive):
            return self._ro.gx.amplitude if positive else self._ro.gx_neg.amplitude

        positive0 = self._start_pos
        last_positive = self._start_pos if self._flyback else ((self.num_echoes - 1) % 2 == 0) == self._start_pos

        spoil_delta = self._spoil_factor * self._ro.k_width if self._spoil_position != "none" else 0.0

        self._echo0_override = None
        self._last_echo_override = None

        # --- prewind: plain rewind-to-echo0, or bridged w/ pre-spoil ------
        if self._spoil_position == "pre":
            # Mirrors the post-spoil bridge exactly (gre.py convention): the
            # bridge's own area is spoil_delta alone, signed to match
            # grad_end so the shape is a clean (0, spoil_plateau,
            # readout_plateau) staircase -- rising to an elevated plateau
            # then settling to the echo's own amplitude, never reversing
            # through zero. Any resulting k-space offset relative to the
            # unspoiled prewind position is a fixed, TR-independent constant
            # (spoil_delta doesn't vary with pe_idx) -- a benign linear phase
            # ramp on reconstruction, not a positioning error.
            pre_grad_end = _echo_amp(positive0)
            pre_sign = 1.0 if positive0 else -1.0
            pre_bridge_area = pre_sign * spoil_delta
            self._gx_pre = _bridge(system, ro_axis, pre_bridge_area, 0.0, pre_grad_end)
            t_pre_gx = pp.calc_duration(self._gx_pre)

            gx0 = self._ro.gx if positive0 else self._ro.gx_neg
            _, gx0_tail = pp.split_gradient_at(gx0, gx0.rise_time, system=system)
            gx0_tail.delay = 0.0
            adc0 = copy.deepcopy(self._ro.adc)
            adc0.delay = 0.0
            self._echo0_override = (gx0_tail, adc0)
            t_first_echo_offset_s = (n_pre + 0.5) * self._ro.dwell_s  # no rise_time: bridge already reached plateau
        else:
            self._x_pre_area = _flat_k_start(positive0) - (ramp_area if positive0 else -ramp_area)
            t_pre_gx = _min_trap_duration(system, ro_axis, self._x_pre_area)
            t_first_echo_offset_s = self._ro.adc_center_s

        # --- postwind: plain rewind-to-zero, or bridged w/ post-spoil -----
        if self._spoil_position == "post":
            # Matches gre.py's unbalanced_line exactly: the spoiler's own
            # area is spoil_delta alone (signed to continue in the same
            # direction the last echo was already heading) -- positioning is
            # already done (the prewind, unaffected by spoiling), so nothing
            # here needs to "return to" any particular position first.
            post_grad_start = _echo_amp(last_positive)
            post_sign = 1.0 if last_positive else -1.0
            post_bridge_area = post_sign * spoil_delta
            self._gx_post = _bridge(system, ro_axis, post_bridge_area, post_grad_start, 0.0)
            t_post_gx = pp.calc_duration(self._gx_post)

            gx_last = self._ro.gx if last_positive else self._ro.gx_neg
            gx_last_head, _ = pp.split_gradient_at(gx_last, gx_last.rise_time + gx_last.flat_time, system=system)
            self._last_echo_override = gx_last_head
        else:
            kx_after_last = _flat_k_end(last_positive) + (ramp_area if last_positive else -ramp_area)
            self._x_post_area = -kx_after_last
            t_post_gx = _min_trap_duration(system, ro_axis, self._x_post_area)

        # --- fixed pre-/post-block durations (worst-case PE/PAR area) ---
        t_pe = _min_trap_duration(system, self._pe.axis, self._pe.max_area) if self._pe is not None else 0.0
        t_par = _min_trap_duration(system, self._par.axis, self._par.max_area) if self._par is not None else 0.0
        self.t_prephase_s = ceil_to_raster(max(t_pre_gx, t_pe, t_par), raster)
        self.t_postphase_s = ceil_to_raster(max(t_post_gx, t_pe, t_par), raster)
        self.t_first_echo_s = self.t_prephase_s + t_first_echo_offset_s

        # A bridged prewind (spoil_position == "pre") must end *exactly* at
        # the prewind block's boundary -- echo 0's split remainder resumes at
        # gx.amplitude from t=0 of the *next* block, so any extra time from a
        # PE/PAR area needing longer than the bridge's own natural duration
        # has to lead the bridge (right-aligned), never trail it. A bridged
        # postwind has no such constraint -- it already ends at 0, so extra
        # trailing time is implicitly zero on the readout axis (harmless).
        if self._spoil_position == "pre":
            self._gx_pre.delay = round_to_raster(self.t_prephase_s - t_pre_gx, raster)
        else:
            self._gx_pre = pp.make_trapezoid(channel=ro_axis, area=self._x_pre_area, duration=self.t_prephase_s, system=system)
        if self._spoil_position != "post":
            self._gx_post = pp.make_trapezoid(channel=ro_axis, area=self._x_post_area, duration=self.t_postphase_s, system=system)

        self.duration = self.t_prephase_s + max(self.num_echoes - 1, 0) * self.esp + gx_dur + self.t_postphase_s

    # ------------------------------------------------------------------
    def _build_blocks(self):
        state = self._require_state()
        return self._collect_blocks(
            lambda seq: self._run(
                seq,
                state.pe_idx,
                state.par_idx,
                state.adc_phase_rad,
                state.rotation,
            )
        )

    def _run(self, seq, pe_idx, par_idx, rf_phase_rad, rotation):
        if seq is None:
            import pulserver.pypulseq as _ps

            seq = _ps.Sequence(self._opts)

        pre = [self._gx_pre]
        post = [self._gx_post]
        if self._pe is not None:
            area = self._pe.areas[int(pe_idx)]
            pre.append(pp.make_trapezoid(channel=self._pe.axis, area=area, duration=self.t_prephase_s, system=self._opts))
            post.append(pp.make_trapezoid(channel=self._pe.axis, area=-area, duration=self.t_postphase_s, system=self._opts))
        if self._par is not None:
            area = self._par.areas[int(par_idx)]
            pre.append(pp.make_trapezoid(channel=self._par.axis, area=area, duration=self.t_prephase_s, system=self._opts))
            post.append(pp.make_trapezoid(channel=self._par.axis, area=-area, duration=self.t_postphase_s, system=self._opts))
        if rotation is not None:
            pre.append(rotation)
            post.append(rotation)

        seq.add_block(*pre)

        esp_pad_delay = pp.make_delay(self._esp_pad_s) if self._esp_pad_s > 0.0 else None

        for i in range(self.num_echoes):
            positive = self._start_pos if self._flyback else ((i % 2 == 0) == self._start_pos)
            if i == 0 and self._echo0_override is not None:
                gx, adc = self._echo0_override
                adc = copy.deepcopy(adc)
            elif i == self.num_echoes - 1 and self._last_echo_override is not None:
                gx = self._last_echo_override
                adc = copy.deepcopy(self._ro.adc)
            else:
                gx = self._ro.gx if positive else self._ro.gx_neg
                adc = copy.deepcopy(self._ro.adc)
            adc.phase_offset = rf_phase_rad

            labels = []
            if i == 0:
                if self._pe is not None:
                    labels.append(pp.make_label(type="SET", label=self._pe.label, value=int(pe_idx)))
                if self._par is not None:
                    labels.append(pp.make_label(type="SET", label=self._par.label, value=int(par_idx)))
            if self.num_echoes > 1:
                labels.append(pp.make_label(type="SET", label="ECO", value=0) if i == 0 else pp.make_label(type="INC", label="ECO", value=1))

            events = [gx, adc, *labels]
            if rotation is not None:
                events.append(rotation)
            seq.add_block(*events)

            if i < self.num_echoes - 1:
                if self._flyback:
                    fb_events = [self._gx_flyback] + ([rotation] if rotation is not None else [])
                    seq.add_block(*fb_events)
                if esp_pad_delay is not None:
                    seq.add_block(esp_pad_delay)

        seq.add_block(*post)
        return seq


class Line2D(_LineTrain):
    """2D Cartesian line readout: one ``ky`` per shot, optionally multi-echo.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits (derated in place unless ``derate=False``).
    fov : tuple[float, float]
        In-plane field of view ``(fov_x_m, fov_y_m)`` in metres.
    matrix : tuple[int, int]
        Matrix size ``(nx, ny)``.
    num_echoes : int
        Number of readouts (echoes/contrasts) per shot, all at the same
        ``ky``. ``1`` is the plain single-echo GRE line.
    bandwidth_hz_px, oversamp, pf : float
        Readout receiver bandwidth (Hz/px), frequency oversampling (>= 1) and
        readout fractional-echo factor (in ``(0.5, 1]``).
    flyback : bool
        ``False`` (default) alternates readout polarity between echoes
        (bipolar, no gap); ``True`` keeps every echo the same polarity,
        separated by a dedicated flyback rewind.
    spoil_position : str
        ``"none"`` (fully balanced), ``"post"`` (default, SSFP-FID/spoiled-
        GRE), or ``"pre"`` (SSFP-Echo/PSIF) -- see module docstring.
    spoil_factor : float
        Readout-axis spoiler area as a fraction of the train's own k-space
        width (ignored when ``spoil_position == "none"``).
    esp_s : float, optional
        Echo spacing (s); ``None`` uses the minimum feasible ESP, else >= it.
    start_polarity_positive : bool
        Polarity of the first echo's readout gradient.
    ro_axis, pe_axis : str
        Readout / phase-encode gradient channels.
    derate : bool
        Apply the standard system derates (default True).

    Attributes
    ----------
    duration : float
        One shot's whole train duration (s), exact (at ``pe_idx``-independent
        worst-case pre-/post-block timing).
    esp : float
        Echo spacing (s), constant across the train.
    t_prephase_s, t_postphase_s : float
        Pre-/post-block durations (s), fixed regardless of ``pe_idx``.
    t_first_echo_s : float
        Time from shot start (the prewind block) to the first echo's k=0
        sample (s) -- add ``(rf_duration - rf_center_s)`` for the full
        excitation-to-echo TE budget.
    n_samples, adc_dwell_s, adc_center_s
        Readout sample count, ADC dwell (s), and per-echo block-relative
        offset (s) to the k=0 sample.
    """

    def __init__(self, system, fov, matrix, num_echoes=1, *, pe_axis="y", **kw):
        super().__init__(
            system, fov[0], matrix[0], num_echoes,
            pe_spec=(pe_axis, fov[1], matrix[1], "LIN"), **kw,
        )
        self.pe_axis = pe_axis
        self.fov_y_m, self.ny = float(fov[1]), int(matrix[1])

    def set_state(self, lin_idx=0, adc_phase_rad=0.0, rotation=None):
        """Set the complete dynamic state for one 2D line readout."""
        self._replace_state(
            _LineState(
                pe_idx=int(lin_idx),
                par_idx=0,
                adc_phase_rad=float(adc_phase_rad),
                rotation=normalize_rotation(rotation),
            )
        )
        return self

    def __call__(self, seq=None, pe_idx=0, rf_phase_rad=0.0, rotation=None):
        """Append one shot. ``pe_idx`` is the absolute ``LIN`` index -- the
        the readout emits a ``SET LIN`` label with this value.
        ``rf_phase_rad`` is set as every echo ADC's ``phase_offset`` (RF
        spoiling). ``rotation`` is an optional scipy ``Rotation``-like object
        (see :func:`pulserver.pypulseq.make_rotation`) applied to every block.
        Returns the sequence."""
        self.set_state(lin_idx=pe_idx, adc_phase_rad=rf_phase_rad, rotation=rotation)
        return self._add_or_get(seq)


class Line3D(_LineTrain):
    """3D Cartesian line readout: one ``ky``/``kz`` pair per shot, optionally multi-echo.

    ``fov = (fov_x_m, fov_y_m, fov_z_m)`` and ``matrix = (nx, ny, nz)``; see
    :class:`Line2D` for the shared parameters.
    """

    def __init__(self, system, fov, matrix, num_echoes=1, *, pe_axis="y", par_axis="z", **kw):
        super().__init__(
            system, fov[0], matrix[0], num_echoes,
            pe_spec=(pe_axis, fov[1], matrix[1], "LIN"),
            par_spec=(par_axis, fov[2], matrix[2], "PAR"), **kw,
        )
        self.pe_axis, self.par_axis = pe_axis, par_axis
        self.fov_y_m, self.ny = float(fov[1]), int(matrix[1])
        self.fov_z_m, self.nz = float(fov[2]), int(matrix[2])

    def set_state(self, lin_idx=0, par_idx=0, adc_phase_rad=0.0, rotation=None):
        """Set the complete dynamic state for one 3D line readout."""
        self._replace_state(
            _LineState(
                pe_idx=int(lin_idx),
                par_idx=int(par_idx),
                adc_phase_rad=float(adc_phase_rad),
                rotation=normalize_rotation(rotation),
            )
        )
        return self

    def __call__(self, seq=None, pe_idx=0, par_idx=0, rf_phase_rad=0.0, rotation=None):
        """Append one shot. ``pe_idx``/``par_idx`` are the absolute ``LIN``/
        ``PAR`` indices; the readout emits their absolute labels. See
        :meth:`Line2D.__call__`."""
        self.set_state(
            lin_idx=pe_idx,
            par_idx=par_idx,
            adc_phase_rad=rf_phase_rad,
            rotation=rotation,
        )
        return self._add_or_get(seq)
