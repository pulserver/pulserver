"""Fast/turbo spin-echo (FSE/CPMG) readout trains.

Three stateful readout modules, built once and then re-indexed per shot to emit a
CPMG refocusing/readout train to a sequence:

* :class:`MultiEchoSE` -- 1D multi-echo spin echo (no phase encoding; each echo
  is a different TE, ``ECO`` echo/contrast label incremented).
* :class:`Fse2D` -- 2D FSE (per-echo ``ky``, ``LIN`` label; per-slice frequency
  offset).
* :class:`Fse3D` -- 3D FSE (per-echo ``ky`` and ``kz``, ``LIN`` + ``PAR``).

All three are *train only*: the excitation (90) and the outer shot/slice/TR
loop live in the caller. Each also takes the refocusing pulse as a mandatory
``(refoc_rf, refoc_gz)`` pair built by the caller (see
:func:`build_refocusing_pulse`) -- the train itself has no notion of "slice"
vs. "slab", it only fuses whatever gradient waveform ``refoc_gz`` carries.
:class:`MultiEchoSE`/:class:`Fse2D` are typically given a slice-selective
pulse and :class:`Fse3D` a slab-selective one, but that choice belongs to the
caller. The object exposes the timing the caller needs (:attr:`esp`,
:attr:`duration`, :attr:`t_first_echo`, :attr:`t_exc_center_to_train_start`,
:attr:`refoc_center_s`) to align its 90 and pad TR.

Gradient surgery
----------------
Adjacent gradients on the same axis are *fused* into single extended
trapezoids (via :func:`pypulseq.make_extended_trapezoid`) so ramps are shared
and the gradient never returns to zero needlessly -- exactly the construction
of the pypulseq TSE demo (after Kelley/Zaitsev's ``writeTSE``). The repeating
unit, built from component spoiler/readout trapezoids, is

    z:  [GS4 flat refoc lobe] [GS5 ref->0]           0        [GS7 0->ref]
    x:            0           [GR5 0->ro ] [GR6 readout+ADC] [GR7 ro->0 ]
    y:            0           [phase-enc]      0            [ unwind  ]
        --------- 180 -------  ----------- readout -----------

``GS5``/``GS7`` fuse the z crusher into ``refoc_gz``'s ramp; ``GR5``/``GR7``
fuse the x spoiler into the readout ramp. A one-off lead-in prephaser (x) with
a 0->ref z crusher precedes the first 180; a ref->0 lobe closes out. The
phase-encode prewinder/rewinder and the lead-in prephaser are each built at
their natural (shortest) duration and then right/left-aligned
(:func:`pypulseq.align`) into the fixed-duration segment around them, rather
than stretched to fill it.

Design notes
------------
* z-only crushers, balanced around every 180 (same spoiler area both sides), so
  the desired spin-echo pathway is preserved and the net moment per unit on
  every axis returns to zero.
* the once-per-shot prephaser positions ``kx=0`` on the echo sample; the x
  spoilers carry the small fractional-echo (``pf < 1``) rebalance so every echo
  starts the readout at the same k.
* ``pf`` is the *readout* fractional-echo factor (asymmetric echo along x);
  phase-encode partial Fourier is the caller's job (pass fewer indices).
* per-echo RF ``freq_offset`` (slice/slab select) is phase-compensated
  (``phase_offset -= 2*pi*freq_offset*refoc_center_s``), per the pypulseq
  ``write_tse``/``write_haste`` reference scripts.
"""

from __future__ import annotations

__all__ = [
    "CPMG_PHASE_OFFSET_RAD",
    "DEFAULT_REFOCUS_FLIP_DEG",
    "Fse2D",
    "Fse3D",
    "MultiEchoSE",
    "build_refocus_flip_schedule",
    "build_refocusing_pulse",
    "build_z_crusher",
]

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pypulseq as pp

from .._gradients import CRUSHER_CYCLES_Z, make_crusher
from .._schedules import make_traps_schedule
from .._system import (
    DEFAULT_BANDWIDTH_HZ_PX,
    apply_system_derates,
    ceil_to_raster,
    copy_event,
    scale_grad,
    quantize_readout_timing,
    round_to_raster,
)
from ..._core._module import _CHANNEL_KEY
from ._base import Readout, adopt_waveform, rescale_trap

RF_REFOCUS_TIME_S = 3.0e-3
RF_REFOCUS_APODIZATION = 0.5
RF_REFOCUS_TIME_BW_PRODUCT = 4.0
CPMG_PHASE_OFFSET_RAD = np.pi / 2  # Hennig CPMG condition: 90 deg from the excitation phase
DEFAULT_REFOCUS_FLIP_DEG = 180.0
READOUT_GRAD_MARGIN = 0.95  # headroom below max_grad for the readout plateau
DEFAULT_X_SPOIL_FACTOR = 1.0  # x crusher area as a fraction of the readout area
DEFAULT_Z_SPOIL_FACTOR = 0.5  # z crusher area beyond the slice-select rephasing

@dataclass(frozen=True)
class _FseState:
    lin_idx: object | None
    par_idx: object | None
    freq_offset_hz: float


def _snapshot_index(value):
    if value is None or np.ndim(value) == 0:
        return value
    snapshot = np.array(value, copy=True)
    snapshot.setflags(write=False)
    return snapshot


def build_z_crusher(opts: pp.Opts, thickness_m: float, cycles: float = CRUSHER_CYCLES_Z):
    """Standalone z-axis crusher trapezoid (helper; the trains fuse their own)."""
    return make_crusher(opts, "z", dephasing_cycles=cycles, voxel_size=thickness_m)


def build_refocusing_pulse(
    opts: pp.Opts,
    thickness_m: float,
    *,
    flip_angle_deg: float = DEFAULT_REFOCUS_FLIP_DEG,
    duration_s: float = RF_REFOCUS_TIME_S,
    apodization: float = RF_REFOCUS_APODIZATION,
    time_bw_product: float = RF_REFOCUS_TIME_BW_PRODUCT,
    phase_offset_rad: float = CPMG_PHASE_OFFSET_RAD,
):
    """Build a slice- or slab-selective ``(rf, gz)`` refocusing pulse.

    Pass the result as the mandatory ``refoc_rf``/``refoc_gz`` pair to
    :class:`MultiEchoSE`, :class:`Fse2D` or :class:`Fse3D` -- ``thickness_m``
    is the slice thickness for the former two, the slab thickness for the
    latter. Only ``gz.amplitude`` is used by the train (the selection lobe is
    replayed as a fused flat-only lobe); per-echo flip variation is an
    envelope scaling.
    """
    rf, gz, _ = pp.make_sinc_pulse(
        flip_angle=np.deg2rad(flip_angle_deg),
        duration=duration_s,
        slice_thickness=thickness_m,
        apodization=apodization,
        time_bw_product=time_bw_product,
        system=opts,
        use="refocusing",
        return_gz=True,
        delay=opts.rf_dead_time,
    )
    rf.phase_offset = phase_offset_rad
    return rf, gz


def build_refocus_flip_schedule(etl: int, refocus_flip_deg: float, variable: bool) -> list[float]:
    """Per-echo refocusing flip angle (deg), length ``etl`` -- a caller helper.

    ``variable=False`` gives a constant angle; ``variable=True`` gives the
    TRAPS 'opt' ramp decaying from near 180 deg toward ``refocus_flip_deg``.
    Divide by the base pulse's nominal flip to get the ``refoc_flip_scale``
    array the readout objects expect.
    """
    schedule = make_traps_schedule(etl, np.deg2rad(refocus_flip_deg), variable=variable)
    return np.rad2deg(schedule).tolist()


class _FseTrain(Readout):
    """Shared fused CPMG refocusing/readout train (see module docstring).

    Subclasses configure the concurrent phase-encode axis (``pe_spec``) and, for
    3D, a z partition axis (``par_spec``) folded into the crusher lobes.
    """

    def __init__(
        self,
        system,
        fov_x_m,
        nx,
        etl,
        refoc_rf,
        refoc_gz,
        *,
        pe_spec=None,
        par_spec=None,
        multi_echo=False,
        bandwidth_hz_px=DEFAULT_BANDWIDTH_HZ_PX,
        oversamp=1.0,
        pf=1.0,
        esp_s=None,
        refoc_flip_scale=None,
        refoc_phase_rad=None,
        crusher_cycles=CRUSHER_CYCLES_Z,
        spoil_voxel_size=None,
        x_spoil_factor=DEFAULT_X_SPOIL_FACTOR,
        z_spoil_factor=DEFAULT_Z_SPOIL_FACTOR,
        ro_axis="x",
        derate=True,
        **declared,
    ):
        Readout.__init__(self, system, **declared)
        if etl < 1:
            raise ValueError(f"etl must be >= 1, got {etl}")
        if refoc_rf is None or refoc_gz is None:
            raise ValueError(
                "refoc_rf and refoc_gz are mandatory -- build them with build_refocusing_pulse(...)"
            )

        if derate:
            apply_system_derates(system)
        self._opts = system
        self.etl = int(etl)
        self.fov_x_m, self.nx = float(fov_x_m), int(nx)
        self.ro_axis = ro_axis
        self._multi_echo = multi_echo
        raster = system.grad_raster_time
        self._dG = ceil_to_raster(system.max_grad / system.max_slew, raster)  # full-scale ramp time

        # --- refocusing pulse + fused flat GS4 lobe ------------------------
        self._rf = refoc_rf
        self._flip_scale = _as_schedule(refoc_flip_scale, self.etl, 1.0, "refoc_flip_scale")
        self._refoc_signals: dict[int, np.ndarray] = {}  # echo index -> scaled RF samples
        self._partition_cache: dict[int, tuple] = {}  # PAR index -> its two z lobes
        self._phase = _as_schedule(refoc_phase_rad, self.etl, CPMG_PHASE_OFFSET_RAD, "refoc_phase_rad")
        self.refoc_center_s = pp.calc_rf_center(self._rf)[0]
        self._ref_amp = refoc_gz.amplitude
        # Derived from the actual refoc_rf block, not RF_REFOCUS_TIME_S -- a
        # caller-supplied pulse (e.g. a short nonselective hard pulse) need
        # not match build_refocusing_pulse's default duration.
        self._t_refwd = pp.calc_duration(refoc_rf)
        gs_ref = pp.make_trapezoid(channel="z", amplitude=self._ref_amp, flat_time=self._t_refwd, rise_time=self._dG, system=system)
        self._gs4 = pp.make_extended_trapezoid(
            channel="z", times=np.array([0.0, self._t_refwd]), amplitudes=np.array([self._ref_amp, self._ref_amp]), system=system
        )
        spoil_voxel_size = self.fov_x_m / self.nx if spoil_voxel_size is None else float(spoil_voxel_size)
        if spoil_voxel_size <= 0 or crusher_cycles < 0:
            raise ValueError("spoil_voxel_size must be > 0 and spoil_cycles must be >= 0")
        spoil_area = float(crusher_cycles) / spoil_voxel_size
        self._Cz = 0.5 * gs_ref.area + spoil_area

        # --- x readout geometry + flat lobe (GR6) -------------------------
        self._ro = _build_readout(system, ro_axis, self.nx, self.fov_x_m, bandwidth_hz_px, oversamp, pf)
        self.n_samples = self._ro.n_samples
        self.echo_sample = self._ro.echo_sample
        self.adc_dwell_s = self._ro.dwell_s
        # GR5/GR7 net areas: a5 = Sx + dx_reb, a7 = Sx - dx_reb. The 180 flips k
        # between echoes, so for every readout to start at the same k the areas
        # must satisfy a5 - a7 = ro_amp*(readout_time - 2*adc_echo_offset) (the
        # GR6 sweep minus twice the echo lead-in). The once-per-shot prephaser
        # then lands kx=0 on the echo sample; it works out to half the GR6 sweep.
        ro_amp, readout_time_s = self._ro.ro_amp, self._ro.readout_time_s
        self._Sx = spoil_area
        self._dx_reb = 0.5 * ro_amp * (readout_time_s - 2.0 * self._ro.adc_echo_offset_s)
        self._prephase_area = self._Sx + 0.5 * ro_amp * readout_time_s

        # --- phase-encode / partition specs -------------------------------
        self._pe = self._make_axis(pe_spec)
        self._par = self._make_axis(par_spec)

        # --- ESP / tau timing ---------------------------------------------
        refoc_center_s = self.refoc_center_s
        adc_echo = self._ro.adc_echo_offset_s
        half_pre_fixed = (self._t_refwd - refoc_center_s) + adc_echo
        half_post_fixed = refoc_center_s + (readout_time_s - adc_echo)
        floor = self._segment_floor(system)
        esp_min = ceil_to_raster(2.0 * (max(half_pre_fixed, half_post_fixed) + floor), raster)
        if esp_s is None:
            self.esp = esp_min
        else:
            self.esp = round_to_raster(float(esp_s), raster)
            if self.esp < esp_min - 1e-9:
                raise ValueError(
                    f"esp_s={esp_s * 1e3:.3f} ms is below the minimum feasible ESP ({esp_min * 1e3:.3f} ms)"
                )
        half_esp = 0.5 * self.esp
        self._t_sp_pre = round_to_raster(half_esp - half_pre_fixed, raster)
        self._t_sp_post = round_to_raster(half_esp - half_post_fixed, raster)

        # --- canonical per-echo PE/PAR templates (see _pe_template) -------
        self._pe_worst = self._pe.max_area if self._pe is not None else 0.0
        self._pe_pre_tmpl = (
            _pe_template(system, self._pe.axis, self._pe_worst, self._t_sp_pre, self._dG)
            if self._pe is not None
            else None
        )
        self._pe_rew_tmpl = (
            _pe_template(system, self._pe.axis, self._pe_worst, self._t_sp_post, self._dG)
            if self._pe is not None
            else None
        )
        self._par_worst = self._par.max_area if self._par is not None else 0.0
        self._par_pre_tmpl = (
            _pe_template(system, "z", self._par_worst, self._t_sp_pre, self._dG) if self._par is not None else None
        )
        self._par_rew_tmpl = (
            _pe_template(system, "z", self._par_worst, self._t_sp_post, self._dG) if self._par is not None else None
        )

        # --- fused fixed x/z lobes (GR5/GR7/GS5/GS7) + lead-in/close-out ---
        self._gr5 = _fuse_lobe(system, ro_axis, self._Sx + self._dx_reb - 0.5 * self._dG * ro_amp,
                               0.0, ro_amp, self._t_sp_pre, self._dG)
        self._gr7 = _fuse_lobe(system, ro_axis, self._Sx - self._dx_reb - 0.5 * self._dG * ro_amp,
                               ro_amp, 0.0, self._t_sp_post, self._dG)
        self._gs5 = _fuse_lobe(system, "z", self._Cz, self._ref_amp, 0.0, self._t_sp_pre, self._dG)
        self._gs7 = _fuse_lobe(system, "z", self._Cz, 0.0, self._ref_amp, self._t_sp_post, self._dG)

        # calculate_kspace() flips the running z-moment sign at each refocusing
        # RF's true center (pp.calc_rf_center() + rf.delay), not at the
        # midpoint of GS4's flat window. Every steady-state GS7-then-GS4-then-
        # GS5 triad still telescopes to zero net z-moment at the following
        # flip regardless of where within GS4 that center sits (their combined
        # area is flip-position-independent) -- but the lead-in has no
        # preceding GS7 to pair with, so it alone must seed the telescoping
        # with the area a steady-state triad would have carried into flip 0.
        # Without this, even and odd echoes settle on two different residual
        # z baselines (harmless for MultiEchoSE/Fse2D's plain crusher, but a
        # real per-echo kz error once Fse3D folds PAR onto this axis).
        true_flip_time_s = self._rf.delay + refoc_center_s
        lead_cz = self._Cz + self._ref_amp * (0.5 * self._t_refwd - true_flip_time_s)
        self._t_lead = ceil_to_raster(
            max(self._t_sp_pre, _lobe_dur(system, self._dG, self._prephase_area),
                _lobe_dur(system, self._dG, lead_cz)), raster)
        gr_pre_natural = pp.make_trapezoid(channel=ro_axis, area=self._prephase_area, rise_time=self._dG, system=system)
        gs_up_fixed = _fuse_lobe(system, "z", lead_cz, 0.0, self._ref_amp, self._t_lead, self._dG)
        self._gr_pre, self._gs_up = pp.align(right=[gr_pre_natural, gs_up_fixed])
        self._gs_down = _fuse_lobe(system, "z", self._Cz, self._ref_amp, 0.0, self._t_sp_post, self._dG)

        # --- exposed timing + exact duration ------------------------------
        self.t_first_echo = self._t_lead + refoc_center_s + half_esp
        self.t_exc_center_to_train_start = half_esp - (self._t_lead + refoc_center_s)
        self.duration = self._t_lead + self.etl * self.esp + pp.calc_duration(self._gs_down)

    # ------------------------------------------------------------------
    def _build_blocks(self):
        state = self._require_state()
        # Nothing an FSE shot varies is structural: the train length, the
        # crushers and the readout are fixed by the design, and the encode
        # moves amplitudes, a slice offset and two counters.
        return self._retuned_blocks(
            (),
            lambda record: self._collect_blocks(
                lambda seq: self._run(seq, state.lin_idx, state.par_idx, state.freq_offset_hz, record)
            ),
            lambda record: self._retune(record, self._require_state()),
        )

    def waveform_inventory(self):
        """Every partition's z lobes, so a TR template can move the samples.

        A partition encode rides the slice crushers, and two gradients summed on
        one axis are an arbitrary waveform that moves with the encode -- the one
        thing this train cannot express by rescaling a trapezoid. It depends on
        ``kz`` alone, though, and the partition count is fixed at construction,
        so the complete candidate set is ``nz`` waveforms per lobe and it can be
        stated here rather than discovered per shot.

        This is what lets an FSE 3D train be templated at all. Without it the
        writer has no registered shape for a later partition to point at, and
        the choice would be between compressing samples inside ``add_block`` and
        emitting partition 0's shape under partition 3's amplitude.
        """
        if self._par is None:
            return None
        # Taken first: the layout record is built by rendering, and it is what
        # names the two lobes whose samples move.
        blocks = self.blocks
        record = self._layout[2] if self._layout else {}
        pre_events, rew_events = record.get("gs5") or (), record.get("gs7") or ()
        if not pre_events and not rew_events:
            return None

        # Solved once per partition, which is what _partition_cache is for; the
        # waveform *object* is shared with the shot that adopts it, so a block
        # naming its samples resolves by identity and never compares arrays.
        lobes = [self._partition_lobes(kz) for kz in range(len(self._par.areas))]
        sources: dict[int, tuple] = {}
        for event in pre_events:
            sources[id(event)] = tuple(pair[0] for pair in lobes)
        for event in rew_events:
            sources[id(event)] = tuple(pair[1] for pair in lobes)

        inventory = []
        for block in blocks:
            entry: dict[str, tuple] = {}
            for event in block:
                candidates = sources.get(id(event))
                if candidates is None:
                    continue
                entry[_CHANNEL_KEY[event.channel]] = (
                    tuple(source.waveform for source in candidates),
                    tuple(source.tt for source in candidates),
                )
            inventory.append(entry)
        return tuple(inventory)

    def _retune(self, record, state):
        """Rewrite this shot's numbers on the events the layout already owns."""
        freq_offset = state.freq_offset_hz
        for i_echo, rf in enumerate(record["refoc"]):
            rf.freq_offset = freq_offset
            # Spelled exactly as _scaled_refoc spells it: the multiplication
            # order decides the last bit, and the file is compared on it.
            rf.phase_offset = self._phase[i_echo] - 2.0 * np.pi * freq_offset * self.refoc_center_s

        if self._pe is not None:
            lin = self._norm(state.lin_idx, self._pe.label)
            for i_echo, (pre, rew, label) in enumerate(
                zip(record["pe_pre"], record["pe_rew"], record["pe_label"], strict=True)
            ):
                ky = int(lin[i_echo])
                area = self._pe.areas[ky]
                rescale_trap(pre, self._pe_pre_tmpl, area, self._pe_worst)
                rescale_trap(rew, self._pe_rew_tmpl, -area, self._pe_worst)
                label.value = ky

        if self._par is not None:
            par = self._norm(state.par_idx, self._par.label)
            for i_echo, (gs5, gs7, label) in enumerate(
                zip(record["gs5"], record["gs7"], record["par_label"], strict=True)
            ):
                kz = int(par[i_echo])
                source_gs5, source_gs7 = self._partition_lobes(kz)
                adopt_waveform(gs5, source_gs5)
                adopt_waveform(gs7, source_gs7)
                label.value = kz

    def _partition_lobes(self, kz: int):
        """The z lobes for one partition index, solved once per index.

        A partition encode rides the slice crushers rather than a lobe of its
        own, and two gradients summed on one axis are an arbitrary waveform --
        the one thing an FSE shot cannot express by rescaling. It depends on
        ``kz`` alone, though, and a 3D acquisition revisits every partition
        once per phase encode, so ``nz`` of them cover the whole scan instead
        of one per echo of every shot.
        """
        cached = self._partition_cache.get(kz)
        if cached is None:
            area = self._par.areas[kz]
            cached = (
                pp.add_gradients(
                    [self._gs5, _scaled_pe_grad(self._par_pre_tmpl, area, self._par_worst)], system=self._opts
                ),
                pp.add_gradients(
                    [self._gs7, _scaled_pe_grad(self._par_rew_tmpl, -area, self._par_worst)], system=self._opts
                ),
            )
            self._partition_cache[kz] = cached
        return cached

    def _make_axis(self, spec):
        if spec is None:
            return None
        axis, fov_m, n_points, label = spec
        delta_k = 1.0 / float(fov_m)
        areas = (np.arange(n_points) - 0.5 * n_points) * delta_k
        return SimpleNamespace(
            axis=axis,
            n=int(n_points),
            label=label,
            areas=areas,
            max_area=float(np.max(np.abs(areas))) if n_points > 0 else 0.0,
        )

    def _segment_floor(self, system):
        half_ramp = 0.5 * self._dG * self._ro.ro_amp
        floor = max(
            _lobe_dur(system, self._dG, self._Sx + abs(self._dx_reb) - half_ramp),
            _lobe_dur(system, self._dG, self._Cz),
        )
        if self._pe is not None and self._pe.max_area > 0:
            floor = max(floor, _lobe_dur(system, self._dG, self._pe.max_area))
        if self._par is not None and self._par.max_area > 0:
            floor = max(floor, _lobe_dur(system, self._dG, self._Cz + self._par.max_area))
        return floor

    def _scaled_refoc(self, echo_idx, freq_offset):
        rf = copy_event(self._rf)
        # One array per echo of the train, not per shot: the flip schedule is
        # fixed, so the writer can register each refocusing waveform's shape
        # once and reuse it for every shot that replays this echo index.
        scaled = self._refoc_signals.get(echo_idx)
        if scaled is None:
            scaled = self._rf.signal * self._flip_scale[echo_idx]
            self._refoc_signals[echo_idx] = scaled
        rf.signal = scaled
        rf.freq_offset = freq_offset
        rf.phase_offset = self._phase[echo_idx] - 2.0 * np.pi * freq_offset * self.refoc_center_s
        return rf

    def _norm(self, idx, label):
        arr = np.atleast_1d(np.asarray(idx))
        if arr.size == 1:
            arr = np.full(self.etl, int(arr[0]))
        if arr.size != self.etl:
            raise ValueError(f"{label} index must be a scalar or length-{self.etl} array")
        return arr.astype(int)

    def _run(self, seq, lin_idx, par_idx, freq_offset, record=None):
        """Emit one shot's blocks.

        ``record``, when given, is filled with the events whose payload depends
        on the shot -- the refocusing pulses, the encoding lobes and the
        encoding labels -- which is what :meth:`_retune` moves for every later
        shot instead of rebuilding this.
        """
        if record is None:
            record = {}
        record.setdefault("refoc", [])
        for name in ("pe_pre", "pe_rew", "pe_label", "gs5", "gs7", "par_label"):
            record.setdefault(name, [])
        lin = self._norm(lin_idx, self._pe.label) if self._pe is not None else None
        par = self._norm(par_idx, self._par.label) if self._par is not None else None

        seq.add_block(self._gr_pre, self._gs_up)
        for i_echo in range(self.etl):
            refoc = self._scaled_refoc(i_echo, freq_offset)
            record["refoc"].append(refoc)
            seq.add_block(self._gs4, refoc)

            gs5, gs7 = self._gs5, self._gs7
            pre_extra, rew_extra, labels = [], [], []
            if self._pe is not None:
                ky = int(lin[i_echo])
                area = self._pe.areas[ky]
                pre_extra.append(_scaled_pe_grad(self._pe_pre_tmpl, area, self._pe_worst))
                rew_extra.append(_scaled_pe_grad(self._pe_rew_tmpl, -area, self._pe_worst))
                labels.append(pp.make_label(type="SET", label=self._pe.label, value=ky))
                record["pe_label"].append(labels[-1])
            if self._par is not None:
                kz = int(par[i_echo])
                # Copies, because without a pe encode there is no align() below
                # to make one, and the layout must own an event it can retune
                # without writing into the memoised lobes themselves.
                gs5, gs7 = (copy_event(lobe) for lobe in self._partition_lobes(kz))
                labels.append(pp.make_label(type="SET", label=self._par.label, value=kz))
                record["par_label"].append(labels[-1])
            if self._multi_echo and i_echo > 0:
                labels.append(pp.make_label(type="INC", label="ECO", value=1))

            # Prewinder/rewinder built at natural duration, then aligned to
            # abut the readout rather than stretched to fill the segment.
            pre_block = pp.align(right=[self._gr5, gs5, *pre_extra]) if pre_extra else [self._gr5, gs5]
            post_block = pp.align(left=[self._gr7, gs7, *rew_extra]) if rew_extra else [self._gr7, gs7]
            record["gs5"].append(pre_block[1])
            record["gs7"].append(post_block[1])
            if pre_extra:
                record["pe_pre"].append(pre_block[2])
                record["pe_rew"].append(post_block[2])

            adc = copy_event(self._ro.adc)
            adc.phase_offset = self._phase[i_echo]
            seq.add_block(*pre_block)
            seq.add_block(self._ro.gr6, adc, *labels)
            seq.add_block(*post_block)
        seq.add_block(self._gs_down)
        return seq


class MultiEchoSE(_FseTrain):
    """1D multi-echo spin echo: no phase encoding, ``ECO`` incremented per echo.

    Each echo is a different TE at the same (readout-only) k-space line. The
    caller sets ``ECO = 0`` (a ``SET ECO`` label) before each shot; this object
    increments from the second echo on, labelling echoes ``0..etl-1``. See
    :class:`Fse2D` for the shared parameters (``refoc_rf``/``refoc_gz`` are
    mandatory here too -- build them with :func:`build_refocusing_pulse`).
    """

    def __init__(self, system, fov_x_m, nx, etl, refoc_rf, refoc_gz, **kw):
        super().__init__(system, fov_x_m, nx, etl, refoc_rf, refoc_gz, multi_echo=True, **kw)

    def _set_state(self, freq_offset_hz=0.0):
        """Set the slice frequency offset for one multi-echo train."""
        self._replace_state(_FseState(None, None, float(freq_offset_hz)))
        return self


class Fse2D(_FseTrain):
    """2D fast/turbo spin echo: per-echo ``ky`` (``LIN``), per-slice offset.

    Parameters
    ----------
    system : pypulseq.Opts
        System limits (derated in place unless ``derate=False``).
    fov : tuple[float, float]
        In-plane field of view ``(fov_x_m, fov_y_m)`` in metres.
    matrix : tuple[int, int]
        Matrix size ``(nx, ny)``.
    etl : int
        Echo train length -- refocusing pulses (and encoded lines) per shot.
    refoc_rf, refoc_gz
        Mandatory slice-selective refocusing pulse and its slice-select
        gradient, e.g. from ``build_refocusing_pulse(system, slice_thickness_m)``.
    bandwidth_hz_px, oversamp, pf : float
        Readout receiver bandwidth (Hz/px), frequency oversampling (>= 1) and
        readout fractional-echo factor (in ``(0.5, 1]``).
    esp_s : float, optional
        Echo spacing (s); ``None`` uses the minimum feasible ESP, else >= it.
    refoc_flip_scale, refoc_phase_rad : scalar or length-``etl`` array, optional
        Per-echo envelope scaling (default 1.0) and refocusing/receiver phase
        (default CPMG ``pi/2``).
    crusher_cycles, x_spoil_factor, z_spoil_factor : float
        Spoiler strengths.
    ro_axis, pe_axis : str
        Readout / phase-encode gradient channels.
    derate : bool
        Apply the standard system derates (default True).

    Attributes
    ----------
    duration : float
        One shot's whole ``etl``-echo train duration (s), exact.
    esp : float
        Echo spacing (s).
    refoc_center_s : float
        Refocusing-pulse isodelay (block start -> RF center), s.
    t_first_echo : float
        Time from train start to the first echo center (s).
    t_exc_center_to_train_start : float
        Gap the caller leaves between its 90 *center* and the train start so the
        first 180 lands at ``esp/2`` after the 90 (first echo at TE = ``esp``).
    n_samples, echo_sample, adc_dwell_s
        Readout sample count, 0-based echo sample index, ADC dwell (s).
    """

    def __init__(self, system, fov, matrix, etl, refoc_rf, refoc_gz, *, pe_axis="y", **kw):
        super().__init__(system, fov[0], matrix[0], etl, refoc_rf, refoc_gz,
                         pe_spec=(pe_axis, fov[1], matrix[1], "LIN"), **kw)
        self.pe_axis = pe_axis
        self.fov_y_m, self.ny = float(fov[1]), int(matrix[1])

    def _set_state(self, lin_idx=0, freq_offset_hz=0.0):
        """Set line indices and slice frequency offset for one FSE shot."""
        self._replace_state(
            _FseState(
                lin_idx=_snapshot_index(lin_idx),
                par_idx=None,
                freq_offset_hz=float(freq_offset_hz),
            )
        )
        return self


class Fse3D(_FseTrain):
    """3D fast/turbo spin echo: per-echo ``ky`` and ``kz`` (``LIN`` + ``PAR``).

    ``fov = (fov_x_m, fov_y_m, fov_z_m)`` and ``matrix = (nx, ny, nz)``.
    ``refoc_rf``/``refoc_gz`` (mandatory, e.g. from
    ``build_refocusing_pulse(system, slab_thickness_m)``) size the
    slab-selective refocusing pulse and the z crusher; the kz partition encode
    is folded into the z crusher lobes with :func:`pypulseq.add_gradients`
    (``GS5 += kz``, ``GS7 -= kz``) so the net z-moment per unit stays zero. A
    single slab is excited, so there is no per-slice frequency offset. See
    :class:`Fse2D` for the shared parameters.
    """

    def __init__(self, system, fov, matrix, etl, refoc_rf, refoc_gz, *, pe_axis="y", par_axis="z", **kw):
        super().__init__(system, fov[0], matrix[0], etl, refoc_rf, refoc_gz,
                         pe_spec=(pe_axis, fov[1], matrix[1], "LIN"),
                         par_spec=(par_axis, fov[2], matrix[2], "PAR"), **kw)
        self.pe_axis, self.par_axis = pe_axis, par_axis
        self.fov_y_m, self.ny = float(fov[1]), int(matrix[1])
        self.fov_z_m, self.nz = float(fov[2]), int(matrix[2])

    def _set_state(self, lin_idx=0, par_idx=0):
        """Set line and partition index schedules for one FSE shot."""
        self._replace_state(
            _FseState(
                lin_idx=_snapshot_index(lin_idx),
                par_idx=_snapshot_index(par_idx),
                freq_offset_hz=0.0,
            )
        )
        return self


# ----------------------------------------------------------------------
# Private module-level helpers (only used internally by _FseTrain).
# ----------------------------------------------------------------------

def _fuse_lobe(system, channel, spoiler_area, g_start, g_end, duration_s, ramp_s):
    """A ``g_start -> plateau -> g_end`` extended trapezoid fusing a spoiler lobe.

    The plateau is solved by :func:`pypulseq.make_trapezoid` (a spoiler with the
    given ``spoiler_area`` over ``duration_s``); the result is reshaped to the
    requested endpoints with :func:`pypulseq.make_extended_trapezoid`. The lobe's
    net area is ``spoiler_area + 0.5 * ramp_s * (g_start + g_end)``.
    """
    spr = pp.make_trapezoid(channel=channel, area=spoiler_area, duration=duration_s, rise_time=ramp_s, system=system)
    t0 = spr.rise_time
    t1 = t0 + spr.flat_time
    t2 = t1 + spr.fall_time
    return pp.make_extended_trapezoid(
        channel=channel,
        times=np.array([0.0, t0, t1, t2]),
        amplitudes=np.array([g_start, spr.amplitude, spr.amplitude, g_end]),
        system=system,
    )


def _lobe_dur(system, dG, area):
    """Shortest ``duration`` a ``_fuse_lobe`` spoiler of ``area`` needs with fixed
    ``dG`` ramps: ``2*dG`` (triangle) unless the area forces a plateau."""
    need = dG + abs(area) / system.max_grad  # plateau at max_grad
    return ceil_to_raster(max(2.0 * dG, need), system.grad_raster_time)


def _pe_template(system, axis, worst_mag, duration_s, rise_time):
    """One canonical PE/PAR trapezoid at the worst-case ``|area|`` (see line.py's ``_trap_template``).

    Every echo's smaller-magnitude area on ``axis`` is realised by scaling
    this one shape instead of a fresh ``pp.make_trapezoid`` call per echo --
    the previous per-echo ``area`` + ``rise_time`` (no ``duration``) call
    silently dropped ``rise_time`` (pypulseq falls back to its "shortest
    duration for area" solver whenever neither ``duration`` nor ``flat_time``
    is also given), so every distinct PE/PAR value got its own ramp/flat
    split -- both a correctness bug (the intended fixed-``dG`` ramp was never
    applied) and a base-block-dedup blowup (one gradient shape per distinct
    echo value instead of one, scaled).
    """
    if worst_mag <= 0.0:
        return None
    return pp.make_trapezoid(channel=axis, area=worst_mag, duration=duration_s, rise_time=rise_time, system=system)


def _scaled_pe_grad(template, area, worst_mag):
    if template is None:
        return None
    return scale_grad(template, area / worst_mag)


def _build_readout(opts, ro_axis, nx, fov_x_m, bandwidth_hz_px, oversamp, pf):
    """Compute the x readout geometry, the flat readout lobe (GR6) and the ADC.

    ``oversamp`` densifies the sampling (Δk shrinks, resolution/area fixed);
    ``pf`` truncates the pre-echo side (fewer pre-echo samples), shifting the
    echo (k=0 crossing) earlier in the ADC.
    """
    if not (0.5 < pf <= 1.0):
        raise ValueError(f"pf (readout fractional echo) must be in (0.5, 1], got {pf}")
    if oversamp < 1.0:
        raise ValueError(f"oversamp must be >= 1, got {oversamp}")

    delta_k = 1.0 / (oversamp * fov_x_m)
    n_full = round(oversamp * nx)
    n_post = n_full // 2
    n_samp = max(n_post + 1, round(pf * n_full))
    n_pre = n_samp - n_post
    echo_sample = n_pre
    k_width = n_samp * delta_k

    dwell_s, sampling_time_s = quantize_readout_timing(
        nx_ro=n_samp,
        target_bw_hz_px=bandwidth_hz_px,
        grad_raster_s=opts.grad_raster_time,
        adc_raster_s=opts.adc_raster_time,
        min_flat_time_s=k_width / (READOUT_GRAD_MARGIN * opts.max_grad),
    )
    readout_time_s = sampling_time_s + 2.0 * opts.adc_dead_time
    ro_amp = k_width / sampling_time_s

    gr6 = pp.make_extended_trapezoid(
        channel=ro_axis, times=np.array([0.0, readout_time_s]), amplitudes=np.array([ro_amp, ro_amp]), system=opts
    )
    adc = pp.make_adc(num_samples=n_samp, dwell=dwell_s, delay=opts.adc_dead_time, system=opts)
    adc_echo_offset_s = opts.adc_dead_time + (echo_sample + 0.5) * dwell_s

    return SimpleNamespace(
        gr6=gr6, adc=adc, ro_amp=ro_amp, delta_k=delta_k, k_width=k_width,
        n_pre=n_pre, n_post=n_post, n_samples=n_samp, echo_sample=echo_sample,
        dwell_s=dwell_s, readout_time_s=readout_time_s, adc_echo_offset_s=adc_echo_offset_s,
    )


def _as_schedule(value, etl, default, name):
    """Broadcast a scalar or validate an array-like to length ``etl``."""
    if value is None:
        value = default
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.size == 1:
        arr = np.full(etl, float(arr[0]))
    if arr.size != etl:
        raise ValueError(f"{name} must be a scalar or length-{etl} array, got length {arr.size}")
    return arr
