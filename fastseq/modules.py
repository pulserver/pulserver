"""
"""

__all__ = []

import abc
import copy
import itertools
import math
import operator
import sys
import warnings

from fractions import Fraction
from types import SimpleNamespace

import numpy as np

import pypulseq as pp

from pypulseq.compress_shape import compress_shape
from pypulseq.utils.seq_plot import SeqPlot

# Reached both ways: `import fastseq.modules`, where these are siblings inside a
# package, and `cd fastseq; import modules`, where they are top-level modules on
# their own. The relative form raises when there is no package to be relative
# to, which is what picks between them.
try:
    from .sequence import LABEL_ID, Sequence, rf_shim_row
except ImportError:
    from sequence import LABEL_ID, Sequence, rf_shim_row

try:
    #: The loop and the deduplication kernels in C++. Optional on purpose:
    #: everything here works without it, a little more slowly, and a source
    #: checkout stays importable with nothing but NumPy and PyPulseq. Three
    #: places notice -- `_rounded`, `_unique_rows`, and `PeriodicSequence`,
    #: which binds `add_block` to the compiled writer when there is one.
    try:
        from . import _fastseq_wrapper as _accel
    except ImportError:
        import _fastseq_wrapper as _accel
except ImportError:  # pragma: no cover - depends on whether the wheel was built
    _accel = None


def effective_bandwidth(
    requested_rbw: float,
    n_samples: int,
    grad_raster_time: float,
    adc_raster_time: float,
) -> float:
    """
    """
    ratio = (
        Fraction(str(grad_raster_time))
        / Fraction(str(adc_raster_time))
    )

    # If grad / adc = p / q, validity requires p | (N * k),
    # where dwell = k * adc_raster_time.
    p = ratio.numerator
    k_step = p // math.gcd(p, n_samples)

    requested_k = 1.0 / (requested_rbw * adc_raster_time)
    j = requested_k / k_step

    # Check the valid dwell immediately below and above the request.
    candidates = {
        max(1, math.floor(j)) * k_step,
        max(1, math.ceil(j)) * k_step,
    }

    return min(
        (1.0 / (k * adc_raster_time) for k in candidates),
        key=lambda rbw: abs(rbw - requested_rbw),
    )

class SeqPlotMixin:
    def plot(
        self,
        label: str = str(),
        show_blocks: bool = False,
        save: bool = False,
        time_range=(0, np.inf),
        time_disp: str = 's',
        grad_disp: str = 'kHz/m',
        plot_now: bool = True,
        clear: bool = True,
        overlay: SeqPlot = None,
        stacked: bool = False,
        show_guides: bool = False,
    ) -> SeqPlot:
        return self.seq.plot(
            label, 
            show_blocks, 
            save, 
            time_range,
            time_disp,
            grad_disp,
            plot_now,
            clear,
            overlay,
            stacked,
            show_guides,
        )

#: Everything registration caches on an event. All of it is meaningful only in
#: the sequence it was measured against, so all of it has to go when an event
#: crosses into another one. `shape_id` is Pulseq's own ADC cache and is read
#: back by `register_adc_event`, which is why leaving it behind would not merely
#: be untidy: the new sequence would be told a shape it does not have.
_CACHED_IDS = ('id', 'shape_IDs', 'shape_id', '_ps_code', '_ps_home', '_ps_shape')


def _fresh(event: SimpleNamespace) -> SimpleNamespace:
    """A copy of `event` with any cached library IDs stripped.

    An ID only means something in the sequence it was registered against, so an
    event handed over from another module has to be registered from scratch.
    Copying keeps the donor module reusable.
    """
    clone = copy.copy(event)
    for attr in _CACHED_IDS:
        if hasattr(clone, attr):
            delattr(clone, attr)
    return clone


def _get_block(seq: pp.Sequence, block_index): # standard get_block does not assign id nor or shape_IDs
    raw_block = seq.get_raw_block_content_IDs(block_index)
    block = seq.get_block(block_index)
    
    if raw_block.rf > 0:
        rf_id = raw_block.rf
        rf_shape_IDs = seq.rf_library.data[rf_id][1:4]
        block.rf.id = rf_id
        block.rf.shape_IDs = rf_shape_IDs
    
    if raw_block.gx > 0 and block.gx.type == 'grad':
        gx_id = raw_block.gx
        gx_shape_IDs = seq.grad_library.data[gx_id][3:5]
        block.gx.id = gx_id
        block.gx.shape_IDs = gx_shape_IDs
        
    if raw_block.gy > 0 and block.gy.type == 'grad':
        gy_id = raw_block.gy
        gy_shape_IDs = seq.grad_library.data[gy_id][3:5]
        block.gy.id = gy_id
        block.gy.shape_IDs = gy_shape_IDs
    
    if raw_block.gz > 0 and block.gz.type == 'grad':
        gz_id = raw_block.gz
        gz_shape_IDs = seq.grad_library.data[gz_id][3:5]
        block.gz.id = gz_id
        block.gz.shape_IDs = gz_shape_IDs

    if raw_block.adc > 0:
        adc_id = raw_block.adc
        block.adc.id = adc_id
        block.adc.shape_IDs = seq.adc_library.data[adc_id][7:8]

    return block


class _ModuleSequence(Sequence):
    """A module's own Pulseq sequence: it registers whatever is added to it.

    Registering is not something a module should have to ask for. An event that
    reaches a block is an event the module plays, and playing it is what makes
    its shape worth compressing and its peak worth measuring -- so the sequence
    does it on the way past, once per object however many blocks it appears in.
    That is also what lets `publish` tell what a constructor uses from what it
    merely built along the way.

    A module gets one of these by assigning `self.seq = pp.Sequence(...)`; the
    setter rebrands it. Everything else about it is an ordinary `pp.Sequence`.
    """

    def add_block(self, *events):
        module = self._module
        if module._init_frame is None:
            module._find_init_frame()
        played = module._played
        for event in events:
            if id(event) not in played:
                played.add(id(event))
                _register_event(self, event)
        super().add_block(*events)


class SequenceModule(SeqPlotMixin, abc.ABC):
    """One period's worth of blocks, built once and replayed by a scan loop.

    A module owns a private `pp.Sequence`. Everything it holds -- shape IDs,
    event IDs, peak amplitudes -- is measured there, once, so that the loop that
    plays it thousands of times never has to look at a sample again.

    Subclasses implement `init_module`, not `__init__`, and it reads like an
    ordinary Pulseq script::

        class Readout(SequenceModule):
            def init_module(self, system, num_shots=1):
                self.num_shots = num_shots
                gx_read = [pp.make_trapezoid(...) for _ in range(num_shots)]
                adc = pp.make_adc(...)
                self.seq = pp.Sequence(system=system)
                for shot in range(num_shots):
                    self.seq.add_block(gx_read[shot], adc)

    `__init__` is not to be overridden: it sets up what every module needs
    before `init_module` runs and checks what every module owes afterwards.
    Events used by its blocks are published automatically under the constructor's
    own local variable names, so a readout that calls something `gx_read` is a
    readout whose user writes `readout.gx_read` -- or `readout.events.gx_read`,
    which is the same object and is what an event has to be reached by if its
    name happens to be one the module itself answers to.

    Registration also gives every published RF pulse and arbitrary gradient an
    `amplitude`, which is what a scan loop scales instead of rebuilding samples;
    see `_register_event`.

    Two counts describe how the module is meant to be played:

    `num_shots` is how many periods `seq` lays out. A trajectory that is a
    rotation or a scaling of a single base shot lays out one; a trajectory whose
    shots are not related by anything the interpreter can apply at runtime lays
    out every one of them, which is what registers every one of their waveforms.
    (A module that instead keeps a bank of waveforms and lays out a single
    period leaves this at one -- registering is what makes a waveform reachable,
    and a block only says when something is played.)

    `num_blocks` is the length of one period. It is inferred as the total block
    count divided by `num_shots`. Only that many blocks reach the template,
    because the shots differ from it in values the loop writes anyway -- but
    every shot's waveform was registered on the way past, so the loop can hand
    any of them to `add_block`. A multishot constructor must assign its input to
    `self.num_shots`; single-shot modules inherit the default of one.

    `duration` and `center` are the period's length and the middle of its
    readout. The first defaults to the sum of the period's block durations; the
    second is the module's to set, since only it knows where its echo is.
    """

    def __init__(self, *args, **kwargs):
        self._seq = None
        self._played = set()
        self._init_frame = None
        self._explicit_events = set()
        self.events = SimpleNamespace()
        self.center = 0.0
        self.duration = None      # None -> the period's own block durations
        self.num_blocks = None    # None -> however many blocks one shot took
        self.num_shots = 1        # complete shot periods laid out in self.seq

        self.init_module(*args, **kwargs)

        self._finalize()

    @abc.abstractmethod
    def init_module(self, *args, **kwargs):
        """Build the module: the part a subclass writes.

        Assign `self.seq`, add blocks to it, and set `center` if needed. A
        multishot module must also set `self.num_shots`; events are discovered
        automatically.
        """

    def _find_init_frame(self) -> None:
        """Keep hold of the running `init_module` frame, found from `add_block`.

        A frame someone still holds a reference to keeps its locals after it
        returns -- this is the same guarantee that lets a traceback be inspected
        once the stack has unwound -- so one reference taken at the first
        `add_block` is enough to read the constructor's final namespace in
        `_finalize`. That reads exactly what a closing `self.publish()` would
        have, and costs one stack walk per module rather than a dictionary copy
        per block.

        Every automatically published event has to reach a block, so the module
        sequence is the natural place to catch the constructor still running.
        Events a helper function builds are its locals and not `init_module`'s;
        those are what `publish` and `register` are still there for.
        """
        init = self.init_module
        target_code = getattr(getattr(init, '__func__', init), '__code__', None)
        frame = sys._getframe(2)
        while frame is not None:
            if frame.f_code is target_code and frame.f_locals.get('self') is self:
                self._init_frame = frame
                return
            frame = frame.f_back

    def _finalize(self) -> None:
        """Check what `init_module` owed, and fill in what it can be spared."""
        name = type(self).__name__
        if self._seq is None:
            raise TypeError(f'{name}.init_module never assigned self.seq')

        blocks = len(self._seq.block_events)
        if blocks == 0:
            raise TypeError(f'{name}.init_module added no blocks to self.seq')

        if self._init_frame is not None:
            self._publish_locals(self._init_frame.f_locals)
        self._init_frame = None   # the constructor and its locals are free to go

        if not vars(self.events):
            # Nothing this module plays was held in a local variable of
            # `init_module` -- built in a helper, or passed to `add_block`
            # unnamed -- so there was no name to publish it under. Saying so
            # beats handing back a module whose events are simply missing.
            warnings.warn(
                f'{name} published no events: nothing it added to self.seq was '
                f'named by a local variable of {name}.init_module. Call '
                'self.publish() from wherever the events were built, or name '
                'them with self.register(name=event).',
                stacklevel=3,
            )

        if self.num_blocks is not None:
            self.num_blocks = self._positive_count('num_blocks', self.num_blocks)
        self.num_shots = self._positive_count('num_shots', self.num_shots)

        # An explicit base-period length is enough information on its own: the
        # number of laid-out shots is simply however many such periods fit.
        if self.num_shots == 1 and self.num_blocks is not None and self.num_blocks != blocks:
            self.num_shots, remainder = divmod(blocks, self.num_blocks)
            if remainder:
                raise ValueError(
                    f'{name} lays out {blocks} blocks, which is not a whole number '
                    f'of {self.num_blocks}-block shots'
                )

        if self.num_blocks is None:
            self.num_blocks, remainder = divmod(blocks, self.num_shots)
            if remainder:
                raise ValueError(
                    f'{name} lays out {blocks} blocks, which cannot be divided '
                    f'into {self.num_shots} equal shots'
                )
        if self.num_blocks * self.num_shots != blocks:
            raise ValueError(
                f'{name} lays out {blocks} blocks, which is not {self.num_shots} '
                f'shots of {self.num_blocks}'
            )

        if self.duration is None:
            spans = list(self._seq.block_durations.values())
            self.duration = float(sum(spans[: self.num_blocks]))

    def _positive_count(self, field: str, value) -> int:
        """Return an integer count, rejecting floats, booleans, and zero."""
        try:
            count = operator.index(value)
        except TypeError:
            raise TypeError(
                f'{type(self).__name__}.{field} must be an integer, not '
                f'{type(value).__name__}'
            ) from None
        if isinstance(value, (bool, np.bool_)) or count < 1:
            raise ValueError(f'{type(self).__name__}.{field} must be positive, got {value!r}')
        return count

    def __getattr__(self, name: str):
        """Fall through to `self.events`, so `readout.gx_read` is `readout.events.gx_read`.

        Python only calls this when ordinary lookup has already failed, so a
        module's own attributes always win and nothing an event is called can
        shadow `duration`, `center` or `seq`. `_shadowed` warns when a
        constructor picks such a name, since the event is then reachable only
        through `events`.
        """
        if name.startswith('_'):
            # Never route dunder or private lookups -- `copy`, `pickle` and
            # `inspect` probe for those, and answering would be answering for
            # the module, not for an event.
            raise AttributeError(name)
        try:
            events = object.__getattribute__(self, 'events')
        except AttributeError:
            raise AttributeError(name) from None
        try:
            return getattr(events, name)
        except AttributeError:
            raise AttributeError(
                f'{type(self).__name__!r} module has no attribute or event {name!r}'
            ) from None

    def __dir__(self):
        """Attributes plus published events, so completion offers both."""
        return sorted(set(super().__dir__()) | set(vars(self.events)))

    def _shadowed(self, name: str) -> bool:
        """Whether a published name is one the module itself already answers to."""
        if name in vars(self) or hasattr(type(self), name):
            warnings.warn(
                f'{type(self).__name__} publishes an event called {name!r}, which is '
                f'also a module attribute; reach it as .events.{name}',
                stacklevel=4,
            )
            return True
        return False

    @property
    def seq(self) -> Sequence:
        """The module's Pulseq sequence. Assign one to start building."""
        return self._seq

    @seq.setter
    def seq(self, sequence) -> None:
        """Take a sequence to build in, whichever kind was handed over.

        A constructor that writes `self.seq = pp.Sequence(system=system)` -- the
        line any Pulseq script would open with -- gets that sequence wrapped, so
        it can carry a rotation or a shim without the author having to know that
        upstream cannot. One built as a `fastseq.Sequence` already can, and is
        taken as it stands.
        """
        if isinstance(sequence, pp.Sequence):
            sequence = Sequence(seq=sequence)
        elif not isinstance(sequence, Sequence):
            raise TypeError(
                f"a module's seq must be a pp.Sequence or a fastseq Sequence, "
                f'not {type(sequence).__name__}'
            )
        sequence.__class__ = _ModuleSequence
        sequence._module = self
        self._seq = sequence

    def publish(self, **named) -> None:
        """Publish this module's events under the names they have in the caller's code.

        This happens automatically when `init_module` returns. Calling
        `publish()` remains supported when publication is wanted before then,
        or when explicit keyword aliases are convenient. Only what actually
        reached a block is published, so intermediates stay private.

        An event carrying Pulseq's optional `name` is published under that
        instead, for something built where it had no useful variable name.
        Anything passed by keyword is published as given, and wins.
        """
        self._publish_locals(sys._getframe(1).f_locals)
        self.register(**named)

    def _publish_locals(self, namespace) -> None:
        """Publish played events found in one constructor namespace."""
        for name, value in namespace.items():
            if name.startswith('_') or value is self:
                continue
            items = value if isinstance(value, (list, tuple)) else (value,)
            if not any(id(item) in self._played for item in items):
                continue
            # A bank may hold shots no block laid out; they are part of the
            # module all the same, and the loop is entitled to reach them.
            for item in items:
                if id(item) not in self._played:
                    self._played.add(id(item))
                    _register_event(self._seq, item)
            public_name = getattr(value, 'name', name)
            if public_name not in self._explicit_events:
                self._shadowed(public_name)
                setattr(self.events, public_name, value)

    def register(self, **events) -> None:
        """Publish events by name, for what `publish` cannot see or should not guess."""
        for name, event in events.items():
            for item in (event if isinstance(event, (list, tuple)) else (event,)):
                if id(item) not in self._played:
                    self._played.add(id(item))
                    _register_event(self._seq, item)
            self._shadowed(name)
            setattr(self.events, name, event)
            self._explicit_events.add(name)

    @property
    def blocks(self):
        num_blocks = len(self._seq.block_events)
        return [_get_block(self._seq, n) for n in range(1, num_blocks+1)]


class InversionPrep(SequenceModule): 
    """
    """
    def init_module(
        self,
        system: pp.Opts,
        duration_s: float = 10.0e-3,
        spoiling_area: float = 2000.0,
        labels : tuple | None = None, # the inversion is where a shot is counted
    ):
        if labels is None:
            labels = []

        rf = pp.make_adiabatic_pulse(
            pulse_type='hypsec',
            system=system,
            duration=duration_s,
        )
        gz_crusher = pp.make_trapezoid(
            channel='z',
            system=system,
            area=spoiling_area,
        )
        # A slot per label, so the loop has somewhere to put one. An iteration
        # can leave it out, or say something different with it, but it cannot
        # add a slot the module never built.
        prep_labels = [pp.make_label(type='SET', label=lbl, value=0) for lbl in labels]

        self.seq = pp.Sequence(system=system)

        # Create Inversion Preparation module
        self.seq.add_block(rf, *prep_labels)
        self.seq.add_block(gz_crusher)

        # Assign objects
        self.center = rf.center


class NonSelectiveExcitation(SequenceModule):
    def init_module(
        self, 
        system: pp.Opts,
        flip_angle_deg = 10.0,
        duration_s: float = 1e-3,
    ):
        rf = pp.make_block_pulse(
            flip_angle=np.deg2rad(flip_angle_deg),
            system=system,
            duration=duration_s,
        )
        
        self.seq = pp.Sequence(system=system)

        # Create Excitation module
        self.seq.add_block(rf)

        # Assign objects
        self.center = rf.center

class StackOfStarsReadout(SequenceModule):
    """
    """
    def init_module(
        self,
        system: pp.Opts,
        excitation: SimpleNamespace,
        FOV_m: tuple[float] | float,
        matrix: tuple[int] | int, # used to compute area of gradients, not number of steps
        osf: float = 1.0,
        readout_bandwidth_hz: float = 250e3,
        spoiling_area: float = 0.0, # would be 0 for bSSFP
        te: float | None = None, 
        tr: float | None = None, # # set delay between excitation and echo, plus the final delay to get target spgr train spacing
        labels : tuple | None = None, # a 3D mprage has these as encoding dims;
        trigger: SimpleNamespace | None = None, # could have a TTL digital output or a physio trigger?
        num_shots: int = 1,
    ):
        self.num_shots = num_shots

        if labels is None:
            labels = []
        
        # Calculate effective number of samples
        num_samples = int(np.ceil(osf * matrix[0]))
        
        # Calculate effective readout bandwidth
        readout_bandwidth_hz = effective_bandwidth(
            readout_bandwidth_hz,
            num_samples,
            system.grad_raster_time,
            system.adc_raster_time,
        )
        
        # Calculate readout duration and area
        dwell_time = 1.0 / readout_bandwidth_hz
        readout_duration = num_samples * dwell_time
        readout_area_step = 1 / FOV_m[0]
        readout_area = num_samples * readout_area_step
        
        # Calculate readout gradient
        gx_read = pp.make_trapezoid(
            channel='x', 
            system=system, 
            flat_area=readout_area,
            flat_time=readout_duration,
        )        
        gz_flat = pp.scale_grad(gx_read, scale=0.0)
        gz_flat.channel='z'
        
        # Calculate prephasor. A spoke runs from -k to +k through the centre and
        # comes back, so the same lobe serves as prephasor and as rewinder.
        gx_phase = pp.make_trapezoid(
            channel='x',
            system=system,
            area=-0.5 * gx_read.area,
        )
        
        # Compute phase encoding gradient
        pez_area_step = 1 / FOV_m[1]
        pez_area = -matrix[1] // 2 * pez_area_step
        gz_phase = pp.make_trapezoid(
            channel='z',
            system=system,
            area=pez_area
        )
        
        # Align phasors to readout boundary
        gx_pre, gz_pre = pp.align(right=[
            copy.deepcopy(gx_phase), 
            copy.deepcopy(gz_phase)]
        )
        gx_rew, gz_rew = pp.align(left=[
            copy.deepcopy(gx_phase), 
            copy.deepcopy(pp.scale_grad(gz_phase, scale=-1.0))]
        )
        
        # Create adc
        adc = pp.make_adc(
            system=system, 
            num_samples=num_samples,
            dwell=dwell_time,
            delay=gx_read.rise_time,
        )
        
        # Shift in time
        gx_pre_duration = pp.calc_duration(gx_pre)
        gx_read.delay += gx_pre_duration
        gz_flat.delay += gx_pre_duration
        gx_read_duration = pp.calc_duration(gx_read)
        gx_rew.delay += gx_read_duration
        gz_rew.delay += gx_read_duration
        adc.delay += gx_pre_duration
        
        # Combine in single grad
        gx_read = pp.add_gradients(system=system, grads=[gx_pre, gx_read, gx_rew])
        gz_phase = pp.add_gradients(system=system, grads=[gz_pre, gz_flat, gz_rew])
        
        # For proof of principle of arbitrary grads, convert gx to uniform raster
        gx_read = pp.make_extended_trapezoid(
            channel='x',
            system=system,
            amplitudes=gx_read.waveform,
            times=gx_read.tt,
            convert_to_arbitrary=True,
        )
        gy_read = pp.scale_grad(gx_read, scale=0.0)
        gy_read.channel = 'y'
        
        # Again, for demonstration purpose create explicit spokes instead of
        # using the rotation extension: every angle gets its own pair of
        # waveforms, and the module lays out all `num_shots` TRs.
        #
        # Rotated by hand rather than through `pp.rotate`, which drops
        # zero-amplitude gradients -- the first spoke would then have no y
        # gradient at all, and the template is built from the first spoke, so
        # the loop would have nowhere to put the y waveform of every other one.
        gx_read_base = copy.deepcopy(gx_read)

        angular_increment = 180.0 / num_shots
        angles_rad = np.deg2rad(angular_increment) * np.arange(num_shots)

        gx_read = []
        gy_read = []
        for angle in angles_rad:
            _gx_read = pp.scale_grad(gx_read_base, float(np.cos(angle)))
            _gy_read = pp.scale_grad(gx_read_base, float(np.sin(angle)))
            _gy_read.channel = 'y'
            gx_read.append(_gx_read)
            gy_read.append(_gy_read)

        # Create spoiler gradient
        gz_spoil = pp.make_trapezoid(channel='z', system=system, area=spoiling_area)

        rf = _fresh(excitation.events.rf)
        adc_labels = [pp.make_label(type='SET', label=lbl, value=0) for lbl in labels]

        self.seq = pp.Sequence(system=system)

        # Every shot laid out in full, which is what registers every shot's
        # waveform. Only one shot's worth of blocks reaches the template -- what
        # separates spoke n from spoke 0 is an amplitude, two endpoints and a
        # shape ID, and those are exactly what the loop writes.
        for n in range(num_shots):
            self.seq.add_block(rf)
            self.seq.add_block(gx_read[n], gy_read[n], gz_phase, adc, *adc_labels)
            self.seq.add_block(gz_spoil)

        self.center = pp.calc_duration(rf) + gx_read[0].delay + 0.5 * readout_duration
        
        
        
class LineReadout3D(SequenceModule):
    """
    """
    def init_module(
        self,
        system: pp.Opts,
        excitation: SimpleNamespace,
        FOV_m: tuple[float] | float,
        matrix: tuple[int] | int, # used to compute area of gradients, not number of steps
        partial_echo: float = 1.0, # partial fourier factor along readout dir
        osf: float = 1.0,
        readout_bandwidth_hz: float = 250e3,
        spoiling_area: float = 0.0, # would be 0 for bSSFP
        te: float | None = None, 
        tr: float | None = None, # # set delay between excitation and echo, plus the final delay to get target spgr train spacing
        spoiling_position: str ='post', # default
        labels : tuple | None = None, # a 3D mprage has these as encoding dims;
        trigger: SimpleNamespace | None = None, # could have a TTL digital output or a physio trigger?
    ):
        spoiled = spoiling_area > 0.0
        if labels is None:
            labels = []
        
        # Calculate effective number of samples
        num_samples_full = int(np.ceil(osf * matrix[0]))
        num_samples = int(np.ceil(osf * partial_echo * matrix[0]))
        
        # Calculate effective readout bandwidth
        readout_bandwidth_hz = effective_bandwidth(
            readout_bandwidth_hz,
            num_samples,
            system.grad_raster_time,
            system.adc_raster_time,
        )
        
        # Calculate effective partial echo factor
        partial_echo = num_samples_full / num_samples
        
        # Calculate readout duration and area
        dwell_time = 1.0 / readout_bandwidth_hz
        readout_duration = num_samples * dwell_time
        readout_area_step = 1 / FOV_m[0]
        readout_area = num_samples * readout_area_step
        
        # Calculate readout gradient
        gx_read_full = pp.make_trapezoid(
            channel='x', 
            system=system, 
            flat_area=readout_area,
            flat_time=readout_duration,
        )
        
        gx_read = pp.make_trapezoid(
            channel='x', 
            system=system, 
            flat_area=partial_echo * readout_area,
            flat_time=readout_duration,
        )
        gx_read_rise_time = gx_read.rise_time
        
        # Calculate prephasor area
        gx_pre_area = 0.5 * gx_read_full.area - gx_read.area
        gx_rew_area = -0.5 * gx_read_full.area
        
        # Calculate readout phasors and spoilers
        if spoiled:
            if spoiling_position == 'pre':
                gx_read_rise_time = 0.0
                spoiling_area += gx_pre_area
                gx_pre_area = 0.0
                if spoiling_area > 0.0:
                    gx_pre, _, _ = pp.make_extended_trapezoid_area(
                        channel='x',
                        system=system,
                        area=spoiling_area,
                        grad_start=0.0, 
                        grad_end=gx_read.amplitude,
                    )
                else:
                    raise ValueError(f'Target spoil area must be larger than {0.5 * gx_read_full.area}')
                # Build bridged spoiler
                amplitudes = (gx_read.amplitude, gx_read.amplitude, 0.0)
                times = np.cumsum((0.0, gx_read.flat_time, gx_read.fall_time))
                gx_read = pp.make_extended_trapezoid(
                    channel='x',
                    system=system,
                    amplitudes=amplitudes,
                    times=times,
                )
            elif spoiling_position == 'post':
                spoiling_area += gx_rew_area
                gx_rew_area = 0.0
                if spoiling_area > 0.0:
                    gx_rew, _, _ = pp.make_extended_trapezoid_area(
                        channel='x',
                        system=system,
                        area=spoiling_area,
                        grad_start=gx_read.amplitude, 
                        grad_end=0.0,
                    )
                elif spoiling_area < 0.0:
                    raise ValueError(f'Target spoil area must be larger than {0.5 * gx_read_full.area}')

                # Build bridged spoiler
                amplitudes = (0.0, gx_read.amplitude, gx_read.amplitude)
                times = np.cumsum((0.0, gx_read.rise_time, gx_read.flat_time))
                gx_read = pp.make_extended_trapezoid(
                    channel='x',
                    system=system,
                    amplitudes=amplitudes,
                    times=times,
                )
            else:
                raise ValueError(f'spoiling_position not recognized - must be either "pre" or "post (got {spoiling_position})')
                
        # Calculate readout phasors
        if gx_pre_area != 0.0:
            gx_pre = pp.make_trapezoid(
                channel='x',
                system=system,
                area=gx_pre_area,
            )
        if gx_rew_area != 0.0:
            gx_rew = pp.make_trapezoid(
                channel='x',
                system=system,
                area=gx_rew_area,
            )
        # Special case: enforce same object for pre/rewinder
        # if readout is balanced and symmetric
        if not spoiled and partial_echo == 1.0:
            gx_rew = gx_pre
            
        # Calculate phase encoding gradients
        pey_area_step = 1 / FOV_m[1]
        pey_area = -matrix[1] // 2 * pey_area_step
        gy_phase = pp.make_trapezoid(
            channel='y',
            system=system,
            area=pey_area
        ) 
        pez_area_step = 1 / FOV_m[2]
        pez_area = -matrix[2] // 2 * pez_area_step
        gz_phase = pp.make_trapezoid(
            channel='z',
            system=system,
            area=pez_area
        )
        
        # Calculate timings
        excitation_reference = excitation.events.rf.center
        if hasattr(excitation.events, 'gz_slab'):
            readout_reference = pp.calc_duration(
                excitation.events.rf,
                excitation.events.gz_slab
            ) + pp.calc_duration(
                gx_pre, gy_phase, gz_phase
            ) + gx_read_rise_time + (partial_echo-0.5) * dwell_time * num_samples_full
        else:
            readout_reference = pp.calc_duration(
                excitation.events.rf,
            ) + pp.calc_duration(
                gx_pre, gy_phase, gz_phase
            ) + gx_read_rise_time + (partial_echo-0.5) * dwell_time * num_samples_full
            
        te_min = readout_reference - excitation_reference
        if te is not None:
            te_delay = te - te_min
            if te_delay < 0:
                raise ValueError('requested TE is shorter than min TE')
            wait_te = pp.make_delay(te_delay)
            readout_reference += te_delay
            
        # Assemble the remaining events. The excitation module keeps its own
        # copies, so take a fresh one to register against this module.
        rf = _fresh(excitation.events.rf)
        gz_slab = None
        if hasattr(excitation.events, 'gz_slab'):
            gz_slab = _fresh(excitation.events.gz_slab)
        once_label = pp.make_label(type='SET', label='ONCE', value=0)
        adc_labels = [pp.make_label(type='SET', label=lbl, value=0) for lbl in labels]
        adc = pp.make_adc(
            system=system,
            num_samples=num_samples,
            dwell=dwell_time,
            delay=gx_read_rise_time,
        )

        # Align first, register second: aligning shifts delays, and a delay is
        # part of what a registered event is.
        if trigger is not None:
            trigger, gx_pre, gy_pre, gz_pre = pp.align(
                left=trigger,
                right=[gx_pre, gy_phase, gz_phase]
            )
        else:
            gx_pre, gy_pre, gz_pre = pp.align(
                right=[gx_pre, gy_phase, gz_phase]
            )
        gx_rew, gy_rew, gz_rew = pp.align(
            left=[gx_rew, pp.scale_grad(gy_phase, -1.0), pp.scale_grad(gz_phase, -1.0)]
        )

        # Initialize readout object
        self.seq = pp.Sequence(system=system)

        # Build readout
        if gz_slab is not None:
            self.seq.add_block(rf, gz_slab, once_label)
        else:
            self.seq.add_block(rf, once_label)
        if te is not None:
            self.seq.add_block(wait_te)
        if trigger is not None:
            self.seq.add_block(gx_pre, gy_pre, gz_pre, trigger)
        else:
            self.seq.add_block(gx_pre, gy_pre, gz_pre)
        self.seq.add_block(gx_read, adc, *adc_labels)
        self.seq.add_block(gx_rew, gy_rew, gz_rew)
        tr_min = self.seq.duration()[0]
        if tr is not None:
            tr_delay = tr - tr_min
            if tr_delay < 0:
                raise ValueError('requested TR is shorter than min TR')
            wait_tr = pp.make_delay(tr_delay)
            self.seq.add_block(wait_tr)

        # Assign objects. No remove_duplicates() here: the IDs cached on the
        # events index this sequence's libraries, and deduplication renumbers
        # them. Duplicates are collapsed once, at write time.
        self.center = readout_reference.item()

#: Extension type name -> its column in the per-block extension count table.
#: These columns are Pulserver's own, fixed ordering; the numeric IDs Pulseq
#: writes into a ``.seq`` file are assigned per sequence and are recovered from
#: ``Sequence.extension_numeric_idx`` at the end of the constructor.
EXTENSION_MAP = {
    'TRIGGERS': 0,
    'LABELSET': 1,
    'LABELINC': 2,
    'RF_SHIMS': 3,
    'ROTATIONS': 4,
    'DELAYS': 5,
    }

#: Where each extension type's own specifications live on a Pulseq sequence.
#: Not every PyPulseq has every one of them -- rotations and RF shims are not in
#: 1.5.0 -- so the attribute is looked up rather than assumed.
_EXTENSION_LIBRARY = {
    'TRIGGERS': 'trigger_library',
    'LABELSET': 'label_set_library',
    'LABELINC': 'label_inc_library',
    'RF_SHIMS': 'rf_shim_library',
    'ROTATIONS': 'rotation_library',
    'DELAYS': 'soft_delay_library',
}

#: The reverse of `EXTENSION_MAP`, for the messages and lookups that start from
#: a column.
_EXTENSION_NAME = {column: name for name, column in EXTENSION_MAP.items()}

#: `event.type` -> the extension column that event belongs in. This is also the
#: set of event types that carry no waveform and so need no registration.
_EXTENSION_COLUMN = {
    'trigger': EXTENSION_MAP['TRIGGERS'],
    'output': EXTENSION_MAP['TRIGGERS'],
    'labelset': EXTENSION_MAP['LABELSET'],
    'labelinc': EXTENSION_MAP['LABELINC'],
    'rf_shim': EXTENSION_MAP['RF_SHIMS'],
    'rot3D': EXTENSION_MAP['ROTATIONS'],
    'soft_delay': EXTENSION_MAP['DELAYS'],
}

#: How wide each extension type's specification row is, by column.
#: A soft delay's row ends in a string, so it is the one extension whose table
#: cannot be a float array; `None` says so.
_EXTENSION_WIDTH = {
    EXTENSION_MAP['TRIGGERS']: 4,     # type, channel, delay, duration
    EXTENSION_MAP['LABELSET']: 2,     # value, label
    EXTENSION_MAP['LABELINC']: 2,     # value, label
    EXTENSION_MAP['RF_SHIMS']: None,  # not in this PyPulseq
    EXTENSION_MAP['ROTATIONS']: None,
    EXTENSION_MAP['DELAYS']: None,    # numID, offset, factor, hint
}

#: Trigger rows hold a coded type and channel. The channel names of the two
#: kinds do not overlap, so one table answers for both.
_TRIGGER_TYPE = {'output': 1, 'trigger': 2}
_TRIGGER_CHANNEL = {'osc0': 1, 'osc1': 2, 'ext1': 3, 'physio1': 1, 'physio2': 2}

# Columns of a `block_events` row. Column 0 is Pulseq's retired delay slot.
_BLOCK_RF = 1
_BLOCK_GRAD = (2, 3, 4)
_BLOCK_ADC = 5
_BLOCK_EXT = 6

# Where the shape IDs sit inside each library's data tuple.
_RF_SHAPE_COLUMNS = (1, 2, 3)     # magnitude, phase, time
_GRAD_SHAPE_COLUMNS = (3, 4)      # waveform, time -- arbitrary gradients only
_ADC_SHAPE_COLUMNS = (7,)         # phase modulation

# Row widths. Trapezoids and arbitrary waveforms share Pulseq's gradient
# library and are told apart only by how wide their row is.
_RF_WIDTH = 10                    # amp, 3 shapes, center, delay, 2 ppm, 2 offsets
_TRAP_WIDTH = 5                   # amp, rise, flat, fall, delay
_ARB_WIDTH = 6                    # amp, first, last, 2 shapes, delay
_ADC_WIDTH = 9                    # n, dwell, delay, 2 ppm, 2 offsets, shape, dead

#: Kinds in the `(kind, row)` gradient index.
_TRAP, _ARB = 0, 1

# -----------------------------------------------------------------------------
# What a loop iteration may vary
# -----------------------------------------------------------------------------
#
# A block slot's *timing* belongs to the module that built it: the template
# seeded it and `add_block` never writes it back, so handing in an event with a
# different rise time or dwell changes nothing. Only these columns move.
#
#   RF     amplitude, freq_ppm, phase_ppm, freq_offset, phase_offset
#          fixed: the three shape IDs, center, delay, use
#   TRAP   amplitude
#          fixed: rise_time, flat_time, fall_time, delay
#   ARB    amplitude, first, last, amplitude shape ID
#          fixed: time shape ID, delay
#   ADC    freq_ppm, phase_ppm, freq_offset, phase_offset, phase-modulation shape ID
#          fixed: num_samples, dwell, delay, dead_time
#
# The two shape IDs are the surprise. To the interpreter every waveform sharing
# an arbitrary gradient slot's `(time_shape_id, delay)` is interchangeable -- it
# swaps them with `setwave` -- so the shape is a runtime choice, not a structural
# one. Most waveforms still resolve to one shape plus an amplitude and a
# rotation; varying it is what makes a multishot module work, and
# `PeriodicSequence.register_waveform` is the way to add one late.
#
# The ADC's phase-modulation shape varies on the same terms and for the same
# reason: a non-Cartesian shot whose gradient waveform is its own usually wants
# its own demodulation too, so the two travel together, one per shot. A module
# that lays out `num_shots` periods registers a phase modulation per shot
# exactly as it registers a waveform per shot, and the loop picks a shot by
# handing in that shot's pair.
_ARB_AMP_SHAPE = 3
_ADC_PHASE_SHAPE = 7

# -----------------------------------------------------------------------------
# Registration
# -----------------------------------------------------------------------------
#
# Registering an event compresses its waveform once, measures its peak once, and
# caches both on the event. The loop then never touches a sample: it moves an
# amplitude, two endpoints and a shape ID.
#
# Registration also stamps `_ps_code`, and that code does three jobs at once. It
# says what kind of event this is, so `add_block` dispatches on an integer
# instead of a string. It says which of the five event columns of a block the
# event belongs in, so no channel lookup is needed. And its absence says the
# event was never registered -- which is how a hand-rolled trapezoid, whose
# timing would otherwise be silently ignored in favour of the slot's, is caught.
#
#   code  0   not registered
#         1   trapezoid on x      -> column 1     column = code
#         2   trapezoid on y      -> column 2
#         3   trapezoid on z      -> column 3
#         4   RF                  -> column 0
#         5   arbitrary on x      -> column 1     column = code - 4
#         6   arbitrary on y      -> column 2
#         7   arbitrary on z      -> column 3
#         8   ADC                 -> column 4
#         9   delay               -> block duration, no column
#        10   extension, TRIGGERS -> its own chained slot   column = code - 10
#        11   extension, LABELSET
#        12   extension, LABELINC
#        13   extension, RF_SHIMS
#        14   extension, ROTATIONS
#        15   extension, DELAYS
#
# The order is not arbitrary: `add_block` splits on `code < 4` first, so the
# three trapezoids of a Cartesian block take the shortest path through it. The
# extensions come last and are numbered by their `EXTENSION_MAP` column, so the
# dispatch that identifies one also says which table it writes into.
_PS_TRAP = {'x': 1, 'y': 2, 'z': 3}
_PS_RF = 4
_PS_ARB = {'x': 5, 'y': 6, 'z': 7}
_PS_ADC = 8
_PS_DELAY = 9
_PS_EXT = 10
_PS_EXT_TRIGGERS = _PS_EXT + EXTENSION_MAP['TRIGGERS']
_PS_EXT_RF_SHIMS = _PS_EXT + EXTENSION_MAP['RF_SHIMS']
_PS_EXT_ROTATIONS = _PS_EXT + EXTENSION_MAP['ROTATIONS']
_PS_EXT_LAST = _PS_EXT + max(EXTENSION_MAP.values())

#: The extension types whose specification an iteration may rewrite -- which is
#: all of them but the soft delay, whose row is a name the operator reads and a
#: number nothing about a loop can change.
_EXTENSION_VARIES = frozenset(
    {
        EXTENSION_MAP['TRIGGERS'],
        EXTENSION_MAP['LABELSET'],
        EXTENSION_MAP['LABELINC'],
        EXTENSION_MAP['RF_SHIMS'],
        EXTENSION_MAP['ROTATIONS'],
    }
)

#: Codes are stamped as `base + tag`, where each PeriodicSequence owns a
#: distinct multiple of this stride. One subtraction in the loop therefore
#: answers both "what kind of event is this" and "is it mine", and an event
#: belonging to another sequence lands outside the dispatch rather than writing
#: into a row it has no business in. The stride has to clear every code above.
_TAG_STRIDE = 32

#: Block columns in slot/presence order: rf, gx, gy, gz, adc.
_SLOT_COLUMNS = 5



def _signed_peak(waveform: np.ndarray) -> float:
    """Pulseq's arbitrary-gradient amplitude: the peak, signed by the first non-zero sample.

    The sign belongs in the amplitude because the stored shape is
    ``waveform / amplitude``. Carrying it there is what makes a sign flip a pure
    scaling -- the shape of ``-g`` is the shape of ``g`` -- so a loop can negate
    a phase encode without registering a second waveform.
    """
    amplitude = np.max(np.abs(waveform))
    if amplitude > 0:
        nonzero = np.nonzero(waveform)[0]
        if nonzero.size:
            amplitude *= np.sign(waveform[nonzero[0]])
    return float(amplitude)


def _register_event(seq: pp.Sequence, event):
    """Register one event against `seq`, caching on it everything a loop reads.

    Registration is also where an RF pulse and an arbitrary gradient acquire an
    `amplitude`. PyPulseq does not put one there -- it keeps `signal` and
    `waveform` at their physical scale, and only a trapezoid carries an explicit
    peak -- but the .seq format does: what it stores is a *normalised* shape and
    the amplitude that scales it, which is exactly the split a scan loop wants.
    So the amplitude is added here, and the samples are left alone rather than
    replaced by it: `signal` and `waveform` are what `register_rf_event` and
    `register_grad_event` read when a module hands an event to another module
    (see `_fresh`), and what any borrowed PyPulseq code expects to find.

    The consequence to know about is that the two are only equal at
    registration. `scale_waveform` moves the amplitude and leaves the samples
    where they are, because the shape of a scaled waveform is the shape it
    already had -- so `amplitude` is what the sequence plays, and the samples
    are what it was built from.
    """
    if isinstance(event, (list, tuple)):
        return [_register_event(seq, item) for item in event]

    kind = getattr(event, 'type', None)
    if kind == 'rf':
        event.amplitude = float(np.max(np.abs(event.signal)))
        event.id, event.shape_IDs = seq.register_rf_event(event)
        event._ps_code = _PS_RF
        event._ps_home = (event.id, tuple(event.shape_IDs))
    elif kind == 'grad':
        # Set before registering: `register_grad_event` measures its own peak,
        # but the loop reads this one, and the two must agree.
        event.amplitude = _signed_peak(event.waveform)
        event.id, event.shape_IDs = seq.register_grad_event(event)
        event._ps_code = _PS_ARB[event.channel]
        event._ps_home = (event.id, tuple(event.shape_IDs))
    elif kind == 'trap':
        event.id = seq.register_grad_event(event)
        event._ps_code = _PS_TRAP[event.channel]
        event._ps_home = (event.id, ())
    elif kind == 'adc':
        # The second return is the phase-modulation shape. `shape_id` is where
        # Pulseq caches it and reads it back from; `shape_IDs` is the spelling
        # the RF and gradient events use, and having both means one rule --
        # "shape IDs live in shape_IDs" -- covers every registered event.
        event.id, shape_id = seq.register_adc_event(event)
        event.shape_id = shape_id
        event.shape_IDs = [shape_id]
        event._ps_code = _PS_ADC
        event._ps_home = (event.id, (shape_id,))
    elif kind == 'delay':
        event._ps_code = _PS_DELAY
    elif kind in _EXTENSION_COLUMN:
        event._ps_code = _PS_EXT + _EXTENSION_COLUMN[kind]
    return event


def scale_waveform(event: SimpleNamespace, scale: float) -> SimpleNamespace:
    """Scale a registered event in place, without touching its samples.

    `pp.scale_grad` builds a new sample array on every call, which a scan loop
    cannot afford and does not need: the interpreter reads an amplitude, two
    endpoints and a shape, and a scaled waveform has the shape it already had.
    So the three numbers move and the samples stay put.

    In place, and returning the event, so a loop can keep one scratch copy per
    slot rather than allocating a fresh one per shot.
    """
    kind = event.type
    if kind == 'trap':
        event.amplitude *= scale
        event.flat_area *= scale
    elif kind == 'grad':
        event.amplitude *= scale
        event.first *= scale
        event.last *= scale
    elif kind == 'rf':
        event.amplitude *= scale
    else:
        raise ValueError(f'{kind!r} events have no amplitude to scale')
    if hasattr(event, 'area'):
        event.area *= scale
    return event


# =============================================================================
# Shape rebasing
# =============================================================================
#
# A module builds its events against its own private `pp.Sequence`, so every ID
# it caches -- `event.id`, `event.shape_IDs`, and the IDs inside its block table
# -- is only meaningful there. Two modules both own shape 1, RF 1, gradient 1.
#
# Rebasing replays a module's *already compressed* libraries into the parent and
# returns the old-ID -> new-ID map for each one. Nothing is decompressed,
# recompressed or revalidated: composing a module is a pure integer remap of a
# handful of small tables, no matter how long its waveforms are.


def _lookup_table(mapping: dict[int, int]) -> np.ndarray:
    """`mapping` as an array, so a whole ID column can be remapped at once."""
    table = np.zeros(max(mapping) + 1, dtype=np.int32)
    for old_id, new_id in mapping.items():
        table[old_id] = new_id
    return table


def _merge_shapes(dst, src) -> dict[int, int]:
    # 0 ("no shape") and -1 ("half raster" flag) are sentinels, not IDs.
    mapping = {0: 0, -1: -1}
    for old_id, data in src.data.items():
        mapping[old_id], _ = dst.find_or_insert(data)
    return mapping


def _merge_shaped(dst, src, shape_map, columns, shapeless_type=None) -> dict[int, int]:
    """Merge a library whose rows reference shapes at `columns`.

    Rows whose type equals `shapeless_type` are copied verbatim -- that is how
    trapezoids, which share the gradient library with arbitrary waveforms but
    carry no shape at all, are left alone.
    """
    mapping = {0: 0}
    for old_id, data in src.data.items():
        data_type = src.type.get(old_id, str())
        if data_type != shapeless_type:
            data = list(data)
            for column in columns:
                data[column] = shape_map[data[column]]
        mapping[old_id], _ = dst.find_or_insert(tuple(data), data_type)
    return mapping


def _merge_plain(dst, src) -> dict[int, int]:
    """Merge a library whose rows hold values only."""
    mapping = {0: 0}
    for old_id, data in src.data.items():
        mapping[old_id], _ = dst.find_or_insert(tuple(data), src.type.get(old_id, str()))
    return mapping


def _merge_soft_delays(dst_seq: pp.Sequence, src_seq: pp.Sequence) -> dict[int, int]:
    """Merge a module's soft delays, keeping one numeric ID per hint.

    A soft delay is named twice over: by a hint the operator reads on the
    console, and by a numeric ID that orders the delays in the interface.
    Modules assign their IDs independently -- each one starts counting from
    zero -- so composing two of them has to renumber. What a hint means is what
    has to survive; the number it happened to get in the module it was built in
    does not.
    """
    mapping = {0: 0}
    for old_id, data in src_seq.soft_delay_library.data.items():
        _, offset, factor, hint = data
        assigned = dst_seq.soft_delay_hints.get(hint)
        if assigned is None:
            assigned = max([-1, *dst_seq.soft_delay_hints.values()]) + 1
            dst_seq.soft_delay_hints[hint] = assigned
        mapping[old_id], _ = dst_seq.soft_delay_library.find_or_insert(
            (assigned, offset, factor, hint)
        )
    return mapping


def _merge_extensions(dst_seq: pp.Sequence, src_seq: pp.Sequence, ref_maps) -> dict[int, int]:
    """Merge the extension chain table, remapping both type IDs and refs.

    Numeric extension type IDs are assigned per sequence, so the module's IDs
    are translated back through their names. Chains are built tail first, so a
    row's `next` always has a lower ID than the row itself: walking the source
    in ascending ID order guarantees the target is already mapped.
    """
    src_names = dict(zip(src_seq.extension_numeric_idx, src_seq.extension_string_idx))
    mapping = {0: 0}
    for old_id in sorted(src_seq.extensions_library.data):
        type_id, ref_id, next_id = src_seq.extensions_library.data[old_id]
        name = src_names[type_id]
        if name not in ref_maps:
            raise ValueError(f'Unsupported block extension {name!r}')
        data = (
            dst_seq.get_extension_type_ID(name),
            ref_maps[name][ref_id],
            mapping[next_id],
        )
        mapping[old_id], _ = dst_seq.extensions_library.find_or_insert(data)
    return mapping


def _rebase_module(parent: pp.Sequence, module: SequenceModule, tag: int) -> SimpleNamespace:
    """Re-register one module's libraries inside `parent`.

    Returns the binding: the per-library ID maps, plus the module's block table
    with every ID column already translated into the parent's numbering. The
    module itself is left untouched, so the same instance can be composed any
    number of times and into more than one sequence.
    """
    source = module._seq
    shape_map = _merge_shapes(parent.shape_library, source.shape_library)
    maps = {
        'shape': shape_map,
        'rf': _merge_shaped(
            parent.rf_library, source.rf_library, shape_map, _RF_SHAPE_COLUMNS,
        ),
        'grad': _merge_shaped(
            parent.grad_library, source.grad_library, shape_map, _GRAD_SHAPE_COLUMNS,
            shapeless_type='t',
        ),
        'adc': _merge_shaped(
            parent.adc_library, source.adc_library, shape_map, _ADC_SHAPE_COLUMNS,
        ),
        'TRIGGERS': _merge_plain(parent.trigger_library, source.trigger_library),
        'LABELSET': _merge_plain(parent.label_set_library, source.label_set_library),
        'LABELINC': _merge_plain(parent.label_inc_library, source.label_inc_library),
        'DELAYS': _merge_soft_delays(parent, source),
        'RF_SHIMS': _merge_plain(parent.rf_shim_library, source.rf_shim_library),
        'ROTATIONS': _merge_plain(parent.rotation_library, source.rotation_library),
    }
    maps['extension'] = _merge_extensions(parent, source, maps)

    # Translate the module's block table column by column.
    rows = np.stack(list(source.block_events.values())).astype(np.int32)
    rows[:, _BLOCK_RF] = _lookup_table(maps['rf'])[rows[:, _BLOCK_RF]]
    grad_table = _lookup_table(maps['grad'])
    for column in _BLOCK_GRAD:
        rows[:, column] = grad_table[rows[:, column]]
    rows[:, _BLOCK_ADC] = _lookup_table(maps['adc'])[rows[:, _BLOCK_ADC]]
    rows[:, _BLOCK_EXT] = _lookup_table(maps['extension'])[rows[:, _BLOCK_EXT]]

    durations = np.fromiter(source.block_durations.values(), dtype=float)

    # The module's own events must address the parent from now on, so that
    # `readout.events.gx_read` can be handed straight to the loop.
    _rebase_events(module, maps, tag)

    num_blocks = module.num_blocks
    if num_blocks is None:
        num_blocks = len(durations)
    elif num_blocks > len(durations):
        raise ValueError(
            f'{type(module).__name__} declares a base period of {num_blocks} blocks '
            f'but holds only {len(durations)}'
        )

    return SimpleNamespace(
        module=module,
        maps=maps,
        block_events=rows,
        block_durations=durations,
        num_blocks=num_blocks,
    )


def _rebase_events(module: SequenceModule, maps, tag: int) -> None:
    """Point a module's published events at the parent's libraries, in place.

    A module keeps its events so the user can hand them straight to the loop,
    which means they have to address the sequence the loop is filling rather
    than the private one they were built in. Rebasing them in place is what lets
    `readout.events.gx_read[7]` work with no further ceremony.

    The translation always starts from `_ps_home` -- the IDs the module itself
    gave the event -- and never from whatever is on the event now. So composing
    the same module into a second sequence rebases it again rather than
    rebasing a rebasing, and a module stays reusable; what it cannot do is be
    filled by two sequences at once, since it points at the newer one.

    The arbitrary gradient and the ADC get `_ps_shape` here and not before,
    because the amplitude shape and the phase-modulation shape are the shape IDs
    the loop writes, and only now do they mean anything outside the module.
    """
    shape_map = maps['shape']
    library_of = {'rf': maps['rf'], 'grad': maps['grad'], 'trap': maps['grad'], 'adc': maps['adc']}

    pending = list(vars(module.events).values())
    while pending:
        event = pending.pop()
        if isinstance(event, (list, tuple)):
            pending.extend(event)
            continue
        home = getattr(event, '_ps_home', None)
        if home is None:
            continue
        home_id, home_shapes = home
        event.id = library_of[event.type][home_id]
        if home_shapes:
            event.shape_IDs = [shape_map[shape_id] for shape_id in home_shapes]
        if event.type == 'grad':
            event._ps_shape = event.shape_IDs[0]
        elif event.type == 'adc':
            # Pulseq reads `shape_id` back when re-registering, so it has to
            # move with the rest rather than keep pointing at the module.
            event.shape_id = event._ps_shape = event.shape_IDs[0]
        # Re-stamp with this sequence's tag. The module's own code addressed the
        # module's libraries; from here the event addresses the parent's, and
        # the tag is what says so at the cost of one subtraction in the loop.
        event._ps_code = (event._ps_code % _TAG_STRIDE) + tag


def _append_binding(seq: pp.Sequence, binding: SimpleNamespace) -> None:
    """Append a rebased module's *base period* into `seq`'s block table.

    `set_block` is deliberately bypassed: the module already validated these
    blocks against the same `Opts`, and every ID in them is a parent ID by now.

    Only `num_blocks` rows are appended. A multishot module has already handed
    the parent every shot's waveform -- that happened when its libraries were
    merged -- so the template needs one shot's worth of blocks and no more; the
    others differ from it only in values the loop writes anyway.
    """
    base = seq.next_free_block_ID
    for offset in range(binding.num_blocks):
        seq.block_events[base + offset] = binding.block_events[offset].copy()
        seq.block_durations[base + offset] = binding.block_durations[offset]
    seq.next_free_block_ID = base + binding.num_blocks


def _dense_rows(library, template_ids: np.ndarray, width: int, loop_size: int) -> np.ndarray:
    """One row per *occurrence* of an event, seeded from the row it came from.

    The template is deduplicated -- a thousand identical readouts share one
    gradient row -- and the loop needs the opposite: a row it can write into
    without disturbing the other 999. `template_ids` lists, in block order, the
    library row each occurrence in one period starts life as.

    Seeding beats zeroing on both counts. The static half of each row is already
    right, so `add_block` writes only what an iteration actually varies; and an
    iteration that skips a block still plays exactly what the module designed.

    The table is NumPy, but the loop reaches it through a `memoryview` (see
    `_flat`): one scalar store costs NumPy ~110 ns and a memoryview ~24 ns, and
    since the view is over the same buffer, that is a fourfold saving on the
    hottest operation in the system for no memory at all.
    """
    period = np.zeros((template_ids.size, width), dtype=float)
    for n, template_id in enumerate(template_ids):
        period[n] = library.data[template_id]
    if _accel is not None:
        return _accel.tile_rows(period, loop_size)
    return np.tile(period.ravel(), loop_size).reshape(-1, width)


def _flat(table: np.ndarray) -> memoryview:
    """A flat, scalar-addressable view of `table`, sharing its buffer.

    NumPy's scalar indexing pays for a boxed result and a full index machinery
    that a single store does not need. A memoryview of the same memory does not,
    so the loop writes through this and everything else reads the array.
    """
    return memoryview(table.reshape(-1))


def _offsets(present: np.ndarray, width: int, loop_size: int) -> np.ndarray:
    """Where each block's row starts in its dense table, one-based; 0 if it has none.

    Occurrences are numbered in block order and every period repeats the pattern
    one whole table-length further along, so a block's row is a rank times a row
    width -- which is why `add_block` can find it with one array read and no
    lookup at all.

    Only the period is reasoned about elementwise; the scan-length part is one
    broadcast into the destination width. Tiling the period and then repeating a
    per-period shift beside it would build four scan-length temporaries in the
    64 bits `arange` and `cumsum` default to, to fill a table that is declared
    `int32` -- and on a scan of two million blocks that arithmetic is entirely
    memory, so the width it is carried in is most of the cost.
    """
    rank = np.cumsum(present) - 1
    period = np.where(present, rank * width + 1, 0)
    stride = int(present.sum()) * width

    offsets = np.empty((loop_size, period.size), dtype=np.int32)
    offsets[:] = np.arange(loop_size, dtype=np.int64).reshape(-1, 1) * stride
    offsets += period.astype(np.int32)
    # A block with no such event holds zero rather than an offset, and zero has
    # to survive the shift -- so it is stamped back in afterwards, by column,
    # which is a period's worth of decisions rather than a scan's.
    offsets[:, period == 0] = 0
    return offsets.reshape(-1)


# =============================================================================
# Deduplication
# =============================================================================
#
# The constructor gave every occurrence of every event its own row, which is
# what let the loop write into one without disturbing the rest. Once the values
# are final that generosity has to be paid back: the rows that turned out
# identical -- the thousand readouts that really were the same readout --
# collapse into one, and the rows nothing points at go with them.

_FNV_PRIME = np.uint64(1099511628211)

#: How precisely Pulseq compares two events before calling them the same. A
#: positive entry is significant digits, a non-positive one decimal places --
#: and in every case it is the resolution the .seq file writes that column at.
#: Rounding to it therefore costs nothing and merges rows the file could not
#: have told apart, which is the difference between one gradient row and six.
_RF_DIGITS = (6, 0, 0, 0, 6, 6, 6, 6, 6, 6)
_GRAD_DIGITS = (6, -6, -6, -6, -6, -6)
_ADC_DIGITS = (0, -9, -6, 6, 6, 6, 6, 0, 6)   # column 7 is a shape ID: round to it exactly


def _rounded(table: np.ndarray, digits=()) -> np.ndarray:
    """`table` at the precision that decides identity, column by column.

    The compiled kernel, when there is one, does the same arithmetic in one
    row-major pass instead of NumPy's copy plus three strided passes and a
    temporary per column. `_rounded_py` below is what it has to agree with, bit
    for bit, and what runs when the extension was not built.
    """
    if _accel is not None and table.ndim == 2 and table.dtype == np.float64:
        return _accel.round_rows(table, [int(digit) for digit in digits])
    return _rounded_py(table, digits)


def _rounded_py(table: np.ndarray, digits=()) -> np.ndarray:
    """`_rounded` in NumPy alone -- the reference the kernel is checked against."""
    out = table.copy()
    for column, digit in enumerate(digits[: table.shape[1]]):
        values = out[:, column]
        if digit > 0:
            scale = 10.0 ** (digit - np.ceil(np.log10(np.abs(values) + 1e-12)))
        else:
            scale = 10.0 ** (-digit)
        out[:, column] = np.round(values * scale) / scale
    # -0.0 and 0.0 are the same number and differ only in their bits, which is
    # exactly what the hash looks at. Adding zero folds one onto the other --
    # it is the identity everywhere else -- so a phase encode scaled by minus
    # nothing stops being a row of its own.
    out += 0.0
    return out


def _unique_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The distinct rows of `rows`, and which of them each row is.

    Rows come back in order of first appearance, which is the order plain Pulseq
    would have assigned had it registered them one block at a time.

    The compiled kernel answers this with an open-addressed table, which sees
    each row once; `_unique_rows_py` below hashes and then sorts the hashes,
    which is the best NumPy can do and is where most of the time in a large
    scan's deduplication goes. They agree exactly -- both compare candidate rows
    bit for bit before calling them equal.
    """
    if _accel is not None and rows.ndim == 2 and rows.dtype.itemsize == 8:
        return _accel.unique_rows(np.ascontiguousarray(rows))
    return _unique_rows_py(rows)


def _unique_rows_py(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`_unique_rows` in NumPy alone -- the reference the kernel is checked against.

    `np.unique(axis=0)` sorts whole rows against each other and costs about five
    seconds on the trapezoid table of a large 3D scan. Hashing each row down to
    sixty-four bits and sorting *that* costs about six hundred milliseconds for
    the same answer, because the sort compares eight bytes instead of forty.

    The hash is not trusted. Every row is checked against its group's
    representative bit for bit, and a genuine collision falls back to a full
    lexicographic sort. Exactness is not negotiable here: these numbers are the
    .seq file.
    """
    count = rows.shape[0]
    if count == 0:
        return rows[:0], np.zeros(0, dtype=np.int64)

    bits = np.ascontiguousarray(rows).view(np.uint64).reshape(count, -1)
    digest = np.zeros(count, dtype=np.uint64)
    for column in range(bits.shape[1]):
        digest *= _FNV_PRIME
        digest ^= bits[:, column]

    _, first, code = np.unique(digest, return_index=True, return_inverse=True)
    if not (bits == bits[first][code]).all():
        return _unique_rows_sorted(rows)

    # Renumber by first appearance rather than by hash.
    order = np.argsort(first, kind='stable')
    rank = np.empty(order.size, dtype=np.int64)
    rank[order] = np.arange(order.size)
    return rows[first[order]], rank[code]


def _unique_rows_sorted(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`_unique_rows` the slow, unconditionally correct way."""
    order = np.lexsort(rows.T[::-1])
    ordered = rows[order]
    starts = np.empty(rows.shape[0], dtype=bool)
    starts[0] = True
    np.any(ordered[1:] != ordered[:-1], axis=1, out=starts[1:])
    group = np.cumsum(starts) - 1
    code = np.empty(rows.shape[0], dtype=np.int64)
    code[order] = group
    first = order[starts]
    appearance = np.argsort(first, kind='stable')
    rank = np.empty(appearance.size, dtype=np.int64)
    rank[appearance] = np.arange(appearance.size)
    return rows[first[appearance]], rank[code]


def _collapse(table: np.ndarray, column: np.ndarray, digits, extra: np.ndarray = None):
    """Deduplicate the rows a block column actually points at, and renumber it.

    `column` holds, per block, the one-based row of `table` that block plays, or
    zero for a block that plays nothing here. Rows no column names are dropped
    on the way through -- pruning is not a separate pass, it is what happens to
    a row nobody asked about.

    `extra` is keyed on alongside the row without being part of it: an RF
    pulse's `use` decides identity but is stored beside the data, not in it.
    """
    played = column > 0
    unique, code = _collapse_gathered(table, column[played] - 1, digits, extra)
    renumbered = np.zeros(column.size, dtype=np.int32)
    renumbered[played] = code + 1
    if extra is None:
        return unique, renumbered, None
    return unique[:, : table.shape[1]], renumbered, unique[:, table.shape[1] :]


def _collapse_gathered(table: np.ndarray, take: np.ndarray, digits, extra=None):
    """Rows `take` of `table`, rounded, deduplicated -- in one pass.

    The rounding decides identity and the deduplication acts on it, so neither
    is any use without the other and nothing needs the gathered table in
    between. The compiled kernel therefore never builds it: on a large 3D scan
    that table is four million rows of six doubles, and what comes out of it is
    a couple of thousand.

    `_collapse_gathered_py` below is the NumPy version it has to agree with row
    for row, and what runs when the extension was not built.
    """
    if _accel is not None and table.ndim == 2 and table.dtype == np.float64:
        return _accel.collapse_rows(
            table, np.ascontiguousarray(take, dtype=np.int64),
            [int(digit) for digit in digits],
            None if extra is None else np.ascontiguousarray(extra, dtype=np.float64),
        )
    return _collapse_gathered_py(table, take, digits, extra)


def _collapse_gathered_py(table: np.ndarray, take: np.ndarray, digits, extra=None):
    """`_collapse_gathered` in NumPy alone -- the reference the kernel is checked against."""
    rows = _rounded(table[take], digits)
    keyed = rows if extra is None else np.column_stack([rows, extra[take]])
    return _unique_rows(keyed)


def _block_columns_py(slots, on, rf_width, adc_width, trap_width, arb_width):
    """`block_columns` in NumPy alone -- the reference the kernel is checked against."""
    played = np.frombuffer(on, dtype=np.uint8).reshape(-1, _SLOT_COLUMNS)

    rf_column = ((slots[:, 0] - 1) // rf_width + 1).astype(np.int32) * played[:, 0]
    adc_column = ((slots[:, 4] - 1) // adc_width + 1).astype(np.int32) * played[:, 4]

    offsets = np.ascontiguousarray(slots[:, 1:4]).ravel()
    here = offsets != 0
    sounding = here & np.ascontiguousarray(played[:, 1:4]).ravel().astype(bool)
    ranks = np.cumsum(here, dtype=np.int32)
    grad_column = ranks * sounding

    picked = offsets[sounding]
    arbitrary = picked < 0                           # _TRAP is +, _ARB is -
    index = np.empty((picked.size, 2), dtype=np.int32)
    index[:, 0] = arbitrary
    magnitude = np.abs(picked, out=picked)
    magnitude -= 1
    column = index[:, 1]
    np.floor_divide(magnitude, trap_width, out=column)
    if arbitrary.any():
        column[arbitrary] = magnitude[arbitrary] // arb_width
    return rf_column, grad_column, adc_column, index


def _factorize(names: list) -> tuple[list, np.ndarray]:
    """The distinct strings in sorted order, and each entry's index into them.

    What `np.unique(..., return_inverse=True)` answers, without turning a list
    of a million short strings into a fixed-width unicode array and sorting it:
    that is sixty-four bytes per entry and an O(n log n) pass to tell half a
    dozen values apart, where a dictionary sees each string once. The order is
    the same -- Python and NumPy both sort strings by code point.
    """
    index = {}
    for name in names:
        if name not in index:
            index[name] = None
    distinct = sorted(index)
    index = {name: number for number, name in enumerate(distinct)}
    code = np.fromiter(map(index.__getitem__, names), dtype=np.int64, count=len(names))
    return distinct, code


def _first_appearance(code: np.ndarray, count: int) -> np.ndarray:
    """Where each of `count` codes first turns up in `code`.

    `_unique_rows` hands back codes numbered in order of first appearance, so
    this is what `np.unique(code, return_index=True)[1]` would say -- but that
    sorts a column as long as the scan to find out, and scattering the indices
    in reverse leaves the smallest one standing for nothing.
    """
    first = np.empty(count, dtype=np.int64)
    first[code[::-1]] = np.arange(code.size - 1, -1, -1, dtype=np.int64)
    return first


def _collapse_chains(chains: np.ndarray, heads: np.ndarray, refs: dict):
    """Deduplicate the extension chain table, tail first.

    A chain is only the same as another if what it points at is the same too, so
    a row cannot be judged before its successor has been. `next` always points
    forward here, so the table resolves in a handful of sweeps: everything that
    terminates, then everything that points at what just resolved, and so on --
    as deep as the longest chain, which is two or three.

    `refs` maps each extension type's numeric ID to the renumbering its own
    specification library just went through.
    """
    if chains is None or chains.size == 0:
        return None, np.zeros(heads.size, dtype=np.int32)

    types, ref, following = chains[:, 0], chains[:, 1], chains[:, 2]

    # Rewrite the references into the specification libraries first: they sit
    # below the chains and have already been collapsed.
    if len(refs) == 1:
        # One type, so every row is that type: a gather rather than a mask, a
        # gather under it and a scatter back.
        (mapping,) = refs.values()
        new_ref = mapping[ref]
    else:
        new_ref = np.zeros(ref.size, dtype=np.int64)
        for type_id, mapping in refs.items():
            mine = types == type_id
            new_ref[mine] = mapping[ref[mine]]

    new_id = np.zeros(chains.shape[0] + 1, dtype=np.int64)   # 1-based, 0 stays 0
    resolved = np.zeros(chains.shape[0], dtype=bool)
    unique = []
    issued = 0

    while not resolved.all():
        tail_done = np.ones(chains.shape[0], dtype=bool)
        chained = following > 0
        tail_done[chained] = resolved[following[chained] - 1]
        ready = tail_done & ~resolved
        if not ready.any():
            raise ValueError('extension chains contain a cycle')

        keys = np.column_stack([types[ready], new_ref[ready], new_id[following[ready]]])
        rows, code = _unique_rows(keys)
        new_id[1:][ready] = issued + code + 1
        unique.append(rows)
        issued += rows.shape[0]
        resolved |= ready

    return np.concatenate(unique), new_id[heads].astype(np.int32)


#: Where each extension type's specifications live on a Pulseq sequence.
def _by_id(groups) -> list:
    """Interleave separately deduplicated tables back into one ID-ordered library.

    Trapezoids and arbitrary waveforms collapse apart but share a numbering, so
    the two results have to be laid back out in the order the IDs say.
    """
    ordered = [None] * sum(table.shape[0] for table, _, _ in groups)
    for table, ids, kind in groups:
        for row, new_id in zip(table, ids):
            ordered[new_id - 1] = (row, kind)
    return ordered


def _fill_library(library, table, types=None) -> None:
    """Populate a Pulseq event library from a table of already-unique rows.

    `tolist` unboxes the whole table in one loop inside numpy; iterating the
    array and calling `tuple` on each row costs nearly twice as much, because
    every element becomes a numpy scalar on the way. Soft delays arrive as
    tuples already -- one of their fields is a string, so they were never an
    array to begin with.
    """
    rows = table if isinstance(table, list) else list(map(tuple, table.tolist()))
    library.data = dict(enumerate(rows, start=1))
    library.keymap = dict(zip(rows, range(1, len(rows) + 1)))
    library.next_free_ID = len(rows) + 1
    if types is not None:
        library.type = dict(enumerate(types, start=1))


def _no_slot(cursor: int, at: int, what: str) -> None:
    """A block was handed an event the template it was built from has no room for."""
    if at == 0:
        raise ValueError(
            f'block {cursor} carries no {what} in the template it was built from; '
            'an iteration can switch a template event off, but not add one'
        )
    raise ValueError(
        f'block {cursor} was built to carry {"an arbitrary" if at > 0 else "a trapezoidal"} '
        f'gradient on that axis, and was handed a {"trapezoidal" if at > 0 else "arbitrary"} one'
    )


def _stale_copy(event, cursor: int) -> None:
    """An event copied out of a module before the sequence was composed.

    Composing is what gives a waveform an amplitude shape -- or an ADC a
    phase-modulation shape -- that the sequence can name; a copy taken before it
    happened carries the module's own numbering, which means nothing here. Copy
    after, not before, which is the same rule the scratch gradients of any scan
    loop already follow.
    """
    what = (
        'an ADC' if event.type == 'adc'
        else f'an arbitrary g{event.channel}'
    )
    shape = 'phase-modulation' if event.type == 'adc' else 'amplitude'
    raise ValueError(
        f'block {cursor} was handed {what} that has no {shape} shape in this '
        'sequence. Copies of a module event have to be taken after the '
        'PeriodicSequence is built, or registered with register_waveform if the '
        'shape itself is new.'
    )


def _no_extension(cursor: int, event, allocated: int) -> None:
    """A block was handed more extensions of a type than the template built it with."""
    name = next(
        key for key, value in EXTENSION_MAP.items()
        if value == _EXTENSION_COLUMN[event.type]
    )
    if allocated == 0:
        raise ValueError(
            f'block {cursor} carries no {name} extension in the template it was '
            'built from; an iteration can switch a template extension off, or '
            'change what it says, but not add one'
        )
    raise ValueError(
        f'block {cursor} was built to carry {allocated} {name} extension(s) and '
        'was handed more than that'
    )


def _wrong_shim(cursor: int, handed: int, allocated: int) -> None:
    """A shim vector of the wrong length for the slot it was handed to."""
    raise ValueError(
        f'block {cursor} was handed an RF shim for {handed} channels, and the '
        f'template it was built from has room for {allocated}. How many channels '
        'a block shims is part of what the module designed, not of what an '
        'iteration chooses.'
    )


def _adopt(event, cursor: int, tag: int) -> int:
    """Stamp an event that reached the loop without a code this sequence issued.

    Delays, labels and triggers hold no waveform and name nothing in a library,
    so there is nothing to register -- they are stamped where they are first
    seen and cost nothing thereafter.

    Anything with a waveform is refused, for one of two reasons. An event that
    was never registered has its timing on itself rather than in the slot, so
    accepting it would play a shape and a duration nobody chose while discarding
    the ones they did. An event registered against a *different* sequence is
    worse: its IDs are real, and they point into the wrong library.
    """
    kind = getattr(event, 'type', None)
    if kind == 'delay':
        event._ps_code = _PS_DELAY + tag
        return _PS_DELAY
    if kind in _EXTENSION_COLUMN:
        code = _PS_EXT + _EXTENSION_COLUMN[kind]
        event._ps_code = code + tag
        return code
    if kind in ('rf', 'grad', 'trap', 'adc'):
        if hasattr(event, '_ps_code'):
            raise ValueError(
                f'block {cursor} was handed a {kind!r} event belonging to a '
                'different sequence. A module points at whichever PeriodicSequence '
                'composed it last, so take its events from the sequence you are '
                'filling, and build a second module rather than sharing one.'
            )
        raise ValueError(
            f'block {cursor} was handed an unregistered {kind!r} event. Only the '
            'amplitude and the offsets of a registered event vary per iteration; '
            'its timing belongs to the module that built the block, so this one '
            'would have been played with timing you did not choose. Build it in a '
            'SequenceModule (or hand it to PeriodicSequence) and scale that.'
        )
    raise ValueError(f'block {cursor}: unsupported event type {kind!r}')




class PeriodicSequence(SeqPlotMixin):
    """A scan built as one period played `loop_size` times, and then flattened.

    The blocks and modules handed to the constructor make up a single period.
    They are composed once, into `self._seq`, and everything the loop can vary --
    event rows, extension specifications -- is then unrolled to `loop_size`
    periods, so every occurrence in the scan owns the row it writes into and no
    iteration has to look anything up.

    `add_block` fills those rows in order, one call per block. When the last one
    is filled, `seq` collapses the whole thing back down into an ordinary
    `pp.Sequence`: identical rows merge, unplayed rows vanish, and what comes out
    is a sequence PyPulseq cannot tell from one written a block at a time. That
    is the trade the class exists to make -- the expensive part of building a
    scan is paid once, at the end, instead of a million times on the way.
    """

    #: Issues each sequence its own stamp; see `_TAG_STRIDE`.
    _tags = itertools.count(_TAG_STRIDE, _TAG_STRIDE)

    def __init__(
        self,
        system: pp.Opts,
        loop_size: int,
        *blocks_or_modules,
    ):
        self.system = system
        self.loop_size = loop_size
        self._tag = next(PeriodicSequence._tags)
        self.bindings = {}
        self.trigger_library = None
        self.label_set_library = None
        self.label_inc_library = None
        self.rf_shim_library = None
        self.rotation_library = None
        self.soft_delay_library = None
        self._output = None

        seq = Sequence(system=system)

        # Pass 1 -- every shape anything here could ever play, registered once,
        # so that from now on a shape ID means the same thing everywhere.
        #
        # A module is merged by identity, so the thousand-readout train of an
        # MPRAGE costs one merge; and a module is merged *whole*, so a multishot
        # readout contributes all of its shots even though only one of them will
        # reach the template. That is the point: the loop may hand any shot to
        # `add_block`, and the shape it names has to already exist.
        for item in blocks_or_modules:
            if isinstance(item, SequenceModule) and id(item) not in self.bindings:
                self.bindings[id(item)] = _rebase_module(seq, item, self._tag)
            elif not isinstance(item, SequenceModule):
                # A loose event is registered straight into the parent, so its
                # IDs are parent IDs already and there is nothing to rebase --
                # but `_ps_shape` is what the loop writes, and only rebasing
                # normally sets it, so it is stamped here instead.
                _register_event(seq, item)
                if hasattr(item, '_ps_code'):
                    item._ps_code = (item._ps_code % _TAG_STRIDE) + self._tag
                    if getattr(item, 'shape_IDs', None) is not None and item.type != 'rf':
                        item._ps_shape = item.shape_IDs[0]

        # Pass 2 -- the template: each module's base period, in order. What the
        # shots have in common is here; what distinguishes them is a value.
        for item in blocks_or_modules:
            if isinstance(item, SequenceModule):
                _append_binding(seq, self.bindings[id(item)])
            else:
                seq.add_block(item)

        self._seq = seq
        self.tr = seq.duration()[0]
        self._writer = None
        self._cursor = 0
        self._views = None
        self._views_at = -1

        self._build_extension_libraries()
        self._build_event_libraries()

        if _accel is not None:
            # The compiled writer takes raw pointers into the buffers the two
            # builds above just allocated, so it is made after them and nothing
            # from here on may reallocate one. `add_block` is bound on the
            # instance, which shadows the method below without hiding it: the
            # Python one stays reachable as `_add_block_py`, and is what the
            # tests compare against.
            self._writer = _accel.BlockWriter(self)
            # `bind` rather than the writer's own method: the same code, reached
            # through CPython's vectorcall instead of pybind11's dispatcher,
            # which is about two hundred nanoseconds a block that a method
            # taking `*args` and returning nothing has no use for.
            self.add_block = self._writer.bind()

    @property
    def cursor(self) -> int:
        """How many blocks the loop has filled.

        Kept by whichever `add_block` is in use, and read from there, so that
        the compiled writer needs no reference back to this object. It used to
        keep one and write the count into this instance after every block --
        which cost a dictionary store per block, and, because a pybind11 class
        has no `tp_traverse`, made the pair of them a cycle Python could not
        collect: every sequence built this way stayed in memory for the life of
        the process, dense tables and all.
        """
        writer = self._writer
        return self._cursor if writer is None else writer.cursor

    @cursor.setter
    def cursor(self, value: int) -> None:
        writer = self._writer
        if writer is None:
            self._cursor = value
        else:
            writer.cursor = value

    def _build_event_libraries(self) -> None:
        """Unroll the period's events into one row per occurrence per period.

        The template shares one row between every block that plays the same
        event; the loop needs a row per block instance, so every occurrence gets
        its own, repeated `loop_size` times, and the block table is rewritten to
        point at them. Nothing is deduplicated again here -- that happens once,
        at write time, by which point the rows hold their final values.

        Shapes are the exception: they stay in `self._seq.shape_library`, shared
        and deduplicated, because a loop scales waveforms rather than replacing
        them.
        """
        loop_size = self.loop_size
        seq = self._seq

        period = np.stack(list(seq.block_events.values())).astype(np.int64)
        blocks = period.shape[0]
        self.num_period_blocks = blocks
        self.num_blocks = blocks * loop_size

        # -- RF -------------------------------------------------------------
        rf_ids = period[:, _BLOCK_RF]
        rf_here = rf_ids > 0
        self.rf_library = _dense_rows(seq.rf_library, rf_ids[rf_here], _RF_WIDTH, loop_size)
        # One period's worth, not the scan's. What an RF pulse is *for* is a
        # property of the template -- an excitation stays an excitation however
        # many times it is played -- so the list is the period's and the codes
        # deduplication needs are tiled from it, rather than a million short
        # strings being sorted into a handful of distinct ones.
        self.rf_use_period = [seq.rf_library.type[rf_id] for rf_id in rf_ids[rf_here]]

        # -- Gradients ------------------------------------------------------
        # Pulseq keeps trapezoids and arbitrary waveforms in one ragged library.
        # Split them into a dense table each; a slot then holds a *signed*
        # offset -- positive into the trapezoid table, negative into the
        # arbitrary one -- so one read answers both where the row is and which
        # kind it is, and disagreeing with it costs a comparison the loop was
        # making anyway.
        grad_ids = period[:, _BLOCK_GRAD].ravel()  # block-major: gx, gy, gz
        widths = np.zeros(max(seq.grad_library.data, default=0) + 1, dtype=np.int64)
        for grad_id, data in seq.grad_library.data.items():
            widths[grad_id] = len(data)
        width_of = widths[grad_ids]
        is_trap = width_of == _TRAP_WIDTH
        is_arb = width_of == _ARB_WIDTH

        self.trap_library = _dense_rows(seq.grad_library, grad_ids[is_trap], _TRAP_WIDTH, loop_size)
        self.arb_library = _dense_rows(seq.grad_library, grad_ids[is_arb], _ARB_WIDTH, loop_size)

        # -- ADC ------------------------------------------------------------
        adc_ids = period[:, _BLOCK_ADC]
        adc_here = adc_ids > 0
        self.adc_library = _dense_rows(seq.adc_library, adc_ids[adc_here], _ADC_WIDTH, loop_size)

        # -- Slots ----------------------------------------------------------
        # Five columns per block -- rf, gx, gy, gz, adc -- holding where that
        # block's row starts. This replaces the block table in the hot path
        # entirely: `add_block` reads one integer and stores a few floats.
        self.block_slot_offsets = np.zeros((self.num_blocks, _SLOT_COLUMNS), dtype=np.int32)
        self.block_slot_offsets[:, 0] = _offsets(rf_here, _RF_WIDTH, loop_size)
        self.block_slot_offsets[:, 1:4] = (
            _offsets(is_trap, _TRAP_WIDTH, loop_size)
            - _offsets(is_arb, _ARB_WIDTH, loop_size)
        ).reshape(-1, 3)
        self.block_slot_offsets[:, 4] = _offsets(adc_here, _ADC_WIDTH, loop_size)

        # Whether each of those five columns is actually played. Everything
        # starts off and `add_block` switches on what it is handed, so omitting
        # an event is not a special case: it is the absence of one.
        self._on = bytearray(_SLOT_COLUMNS * self.num_blocks)

        # Duration lives outside the block table: it is a float, and Pulseq's
        # column 0 is the retired delay-event slot, not a duration.
        self.block_durations = np.tile(
            np.fromiter(seq.block_durations.values(), dtype=float), loop_size,
        )

        # The loop's own handles on all of the above.
        self._rf = _flat(self.rf_library)
        self._trap = _flat(self.trap_library)
        self._arb = _flat(self.arb_library)
        self._adc = _flat(self.adc_library)
        self._slots = _flat(self.block_slot_offsets)
        self._durations = _flat(self.block_durations)

    @property
    def block_events(self) -> np.ndarray:
        """The Pulseq block table: what the loop has actually asked to be played.

        A column is zero either because the template never had that event or
        because this iteration left it out. The row behind it still exists --
        that is what makes leaving it out free -- and pruning drops the ones
        nothing points at.

        Assembled here rather than kept, because the table is what a *reader*
        wants: everything on the way to a file works from the columns it is made
        of, and gathering seven of them into one seven-wide table only to slice
        them apart again is two strided passes over the length of the scan.
        """
        rf_column, grad_column, adc_column, _ = self._rebuild_views()
        events = np.zeros((self.num_blocks, 7), dtype=np.int32)
        events[:, _BLOCK_RF] = rf_column
        events[:, _BLOCK_GRAD] = grad_column.reshape(-1, 3)
        events[:, _BLOCK_ADC] = adc_column
        events[:, _BLOCK_EXT] = self._extension_tally(played=True)
        return events

    @property
    def block_slots(self) -> np.ndarray:
        """The same table as `block_events`, but as *allocated* rather than as played.

        This is the sequence at its widest: every row every block could use. It
        is what the template promised; `block_events` is what the loop took up.

        Derived on every ask rather than cached beside the played columns,
        because nothing in the path from a loop to a file wants it -- it answers
        "what did the template promise", which is a question about the design,
        not about the scan -- and on a scan of two million blocks it is sixty
        megabytes to build and sixty more to keep.
        """
        slots = self.block_slot_offsets
        allocated = np.zeros((self.num_blocks, 7), dtype=np.int32)
        allocated[:, _BLOCK_RF] = (slots[:, 0] - 1) // _RF_WIDTH + 1
        allocated[:, _BLOCK_ADC] = (slots[:, 4] - 1) // _ADC_WIDTH + 1
        here = slots[:, 1:4].ravel() != 0
        ranks = np.cumsum(here, dtype=np.int32)
        ranks *= here
        allocated[:, _BLOCK_GRAD] = ranks.reshape(-1, 3)
        allocated[:, _BLOCK_EXT] = self._extension_tally()
        return allocated

    @property
    def grad_library(self) -> np.ndarray:
        """The `(kind, row)` each played gradient slot names, in block order.

        One row per slot the loop switched on -- not per slot the template
        allocated -- because that is the order deduplication reads them in.
        """
        return self._rebuild_views()[3]

    def _rebuild_views(self):
        """The played block columns, and the gradient index, from the slots and switches.

        None of them is maintained during the loop -- they are a function of what
        has been added, so keeping them current would be work done twice. They
        are rebuilt when asked for, and only if the loop has moved since.

        Columns rather than a block table: the RF, gradient and ADC columns are
        each dense and separate here, and only a caller that wants the Pulseq
        table pays to gather them into one.
        """
        if self._views is not None and self._views_at == self.cursor:
            return self._views

        if _accel is not None:
            views = _accel.block_columns(
                self.block_slot_offsets, self._on,
                _RF_WIDTH, _ADC_WIDTH, _TRAP_WIDTH, _ARB_WIDTH,
            )
        else:
            views = _block_columns_py(
                self.block_slot_offsets, self._on,
                _RF_WIDTH, _ADC_WIDTH, _TRAP_WIDTH, _ARB_WIDTH,
            )

        self._views = views
        self._views_at = self.cursor
        return self._views

    def register_waveform(self, event: SimpleNamespace) -> int:
        """Give an arbitrary gradient, or an ADC, a shape this sequence can address.

        The escape hatch, for a waveform or a phase modulation that was not
        known when the modules were built. A multishot module does not need it
        -- its shots were all registered when the sequence was composed -- and
        neither does a rescaled waveform, which keeps the shape it already had.

        The interpreter treats every waveform sharing a slot's
        `(time_shape_id, delay)` as interchangeable -- it swaps them with
        `setwave` -- so the amplitude shape is the one shape ID a gradient slot
        may vary; it is capped at 16 per `(time_shape_id, delay)` pair, which
        the caller is left to respect until pruning can count them. An ADC's
        phase-modulation shape varies on the same terms.

        The compressed shape is deduplicated against the sequence's own shape
        library and stamped on the event, which leaves it looking exactly like
        one a module produced -- so `add_block` needs no special case for it.
        """
        if event.type == 'grad':
            amplitude = _signed_peak(event.waveform)
            waveform = event.waveform / amplitude if amplitude != 0 else event.waveform
            event.amplitude = amplitude
            code = _PS_ARB[event.channel]
        elif event.type == 'adc':
            # The phase modulation is stored as it stands: it is a phase, so
            # there is no amplitude to factor out of it.
            waveform = np.asarray(event.phase_modulation).ravel()
            code = _PS_ADC
        else:
            raise ValueError(
                f'only arbitrary gradients and ADCs carry a variable shape, not {event.type!r}'
            )

        shape = compress_shape(waveform)
        data = np.concatenate(([shape.num_samples], shape.data))
        event._ps_shape, _ = self._seq.shape_library.find_or_insert(data)
        event._ps_code = code + self._tag
        if event.type == 'adc':
            event.shape_id = event._ps_shape
        return event._ps_shape

    def _add_block_py(self, *args):
        """Write one block of the unrolled scan, then advance the cursor.

        Every block is visited exactly once, in order, so the block at `cursor`
        already owns its rows: there is nothing to look up, nothing to allocate,
        and nothing to deduplicate. What is left is a handful of scalar stores,
        and this method is written to be little else -- it runs once per block of
        a scan that may hold millions.

        Two things follow from the preallocation. What the template carries but
        `args` leaves out is *switched off* -- the row stays allocated but
        unreferenced, for pruning to drop later, which is how a dummy shot skips
        the ADC of an otherwise identical imaging block. And an event the
        template never had cannot be conjured, because no row exists to hold it.

        Extensions obey both rules and nothing else. A block's labels, triggers
        and rotations are a *set*, not a sequence -- Pulseq stores them as a
        linked list built in library-ID order, so which one comes first is not
        something anybody chose -- so the n-th extension of a given type handed
        in fills the n-th slot of that type, and the whole specification row is
        written rather than patched. That is what makes `lin_label.value = y`
        followed by handing the label in do what it reads as, and what makes
        leaving it out mean the block carries no label at all.

        Only the variable fields are read. Timing belongs to the module that
        built the block, so it is not enough to hand in an event with a
        different rise time and expect the block to change -- which is exactly
        why unregistered events are refused rather than quietly retimed.
        """
        i = self.cursor
        if i >= self.num_blocks:
            raise IndexError(
                f'sequence holds {self.num_blocks} blocks '
                f'({self.num_period_blocks} per period x {self.loop_size} periods); '
                'add_block was called once too often'
            )

        base = _SLOT_COLUMNS * i
        slots = self._slots
        on = self._on
        tag = self._tag

        # How many extensions of each type this block has taken so far, one
        # eight-bit field per type packed into a single integer. A block's
        # extensions fill its slots of that type in the order they are handed
        # in, and this is the whole of what remembering that costs.
        seen = 0

        for arg in args:
            # Subtracting the tag is the ownership check: an event this sequence
            # stamped lands in 1..15, anything else outside it. Adopting is what
            # happens to the outsiders that can be adopted -- delays and
            # extensions, which hold no waveform and name nothing -- and it
            # happens once per object, after which they take the fast path.
            code = getattr(arg, '_ps_code', 0) - tag
            if code < 1 or code > _PS_EXT_LAST:
                code = _adopt(arg, i, tag)

            if code < 4:
                # Trapezoid: the commonest event by some way, and the cheapest.
                # Its code is already the block column it belongs in.
                at = slots[base + code]
                if at <= 0:
                    _no_slot(i, at, f'trapezoidal g{arg.channel}')
                self._trap[at - 1] = arg.amplitude
                on[base + code] = 1

            elif code < 8:
                if code == _PS_RF:
                    at = slots[base]
                    if at == 0:
                        _no_slot(i, at, 'RF pulse')
                    rf = self._rf
                    at -= 1
                    rf[at] = arg.amplitude
                    rf[at + 6] = arg.freq_ppm
                    rf[at + 7] = arg.phase_ppm
                    rf[at + 8] = arg.freq_offset
                    rf[at + 9] = arg.phase_offset
                    on[base] = 1
                else:
                    # Arbitrary waveform. The shape goes in unconditionally:
                    # every shot of a multishot module has its own, and a slot
                    # holds whichever one it was last handed.
                    column = code - 4
                    at = slots[base + column]
                    if at >= 0:
                        _no_slot(i, at, f'arbitrary g{arg.channel}')
                    arb = self._arb
                    at = -at - 1
                    arb[at] = arg.amplitude
                    arb[at + 1] = arg.first
                    arb[at + 2] = arg.last
                    try:
                        arb[at + 3] = arg._ps_shape
                    except AttributeError:
                        _stale_copy(arg, i)
                    on[base + column] = 1

            elif code == _PS_ADC:
                at = slots[base + 4]
                if at == 0:
                    _no_slot(i, at, 'ADC')
                adc = self._adc
                at -= 1
                adc[at + 3] = arg.freq_ppm
                adc[at + 4] = arg.phase_ppm
                adc[at + 5] = arg.freq_offset
                adc[at + 6] = arg.phase_offset
                # The phase-modulation shape, on the same terms as the
                # arbitrary gradient's: unconditionally, because a multishot
                # module has one per shot and the slot holds whichever it was
                # last handed. Zero is a real value here -- it is what an ADC
                # with no phase modulation carries.
                try:
                    adc[at + 7] = arg._ps_shape
                except AttributeError:
                    _stale_copy(arg, i)
                on[base + 4] = 1

            elif code == _PS_DELAY:
                # A bare delay only ever sets how long the block lasts.
                self._durations[i] = arg.delay

            else:
                # An extension. Its code is its `EXTENSION_MAP` column, which
                # `_ext_column` turns into the column this sequence actually
                # keeps a table for.
                column = self._ext_column[code - _PS_EXT]
                if column < 0:
                    _no_extension(i, arg, 0)

                shift = column << 3
                k = (seen >> shift) & 255
                seen += 1 << shift

                at = self._ext_stride * i + column + column
                slot = self._slots_ext
                if k >= slot[at + 1]:
                    _no_extension(i, arg, slot[at + 1])
                row = slot[at] + k
                self._ext_switch[column][row] = 1

                table = self._ext_tables[column]
                if table is not None:
                    at = row * self._ext_widths[column]
                    if code < _PS_EXT_RF_SHIMS:
                        if code > _PS_EXT_TRIGGERS:
                            # A label: its value is what a scan loop moves, and
                            # its name goes in beside it because which of a
                            # block's label slots this one landed in is not
                            # fixed.
                            table[at] = arg.value
                            table[at + 1] = LABEL_ID[arg.label]
                        else:
                            table[at] = _TRIGGER_TYPE[arg.type]
                            table[at + 1] = _TRIGGER_CHANNEL[arg.channel]
                            table[at + 2] = arg.delay
                            table[at + 3] = arg.duration
                    elif code == _PS_EXT_ROTATIONS:
                        # Four numbers, scalar first. A `Rotation` is converted
                        # here if that is what was handed in, but converting one
                        # costs more than the rest of this method put together,
                        # so a loop that varies the rotation per block should
                        # hand in the quaternion itself.
                        quaternion = arg.rot_quaternion
                        as_quat = getattr(quaternion, 'as_quat', None)
                        if as_quat is not None:
                            quaternion = as_quat(canonical=True, scalar_first=True)
                        table[at] = quaternion[0]
                        table[at + 1] = quaternion[1]
                        table[at + 2] = quaternion[2]
                        table[at + 3] = quaternion[3]
                    else:
                        values = rf_shim_row(arg)
                        width = self._ext_widths[column]
                        if len(values) != width:
                            _wrong_shim(i, len(values) // 2, width // 2)
                        for offset in range(width):
                            table[at + offset] = values[offset]

        # Last, not first: a block that was refused half-way through is a block
        # the caller can fix and hand back, rather than one silently skipped.
        self.cursor = i + 1

    #: The loop, when the compiled writer is not there to shadow it. It stays on
    #: the class either way, under both names, because it is the specification:
    #: `tests/python/test_fastseq_writer.py` runs every scan twice and compares.
    add_block = _add_block_py

    def deduplicate_events(self) -> SimpleNamespace:
        """Collapse the dense per-occurrence tables into real event libraries.

        This is the other half of the bargain the constructor made. One row per
        occurrence is what let the loop write without looking anything up; now
        that the values are final, the rows that turned out identical become one
        row, and the rows nothing points at disappear -- an ADC the dummy shots
        switched off was never referenced, so nothing has to remember to prune
        it.

        Bottom up, in the sense that a level is renumbered before anything above
        it is rewritten: the events first, then the shapes they name (which is
        also when a shape no surviving event mentions is dropped), then each
        extension's own specifications, then the chains that string them
        together, and last the block table that points at all of it.
        """
        if self.cursor != self.num_blocks:
            raise ValueError(
                f'the loop has filled {self.cursor} of {self.num_blocks} blocks; '
                'deduplicating now would freeze whatever the untouched blocks '
                'happen to hold'
            )
        played_rf, played_grad, played_adc, pointed = self._rebuild_views()

        # -- RF -------------------------------------------------------------
        # `use` decides identity but is stored beside the data, so it is keyed
        # on and then split back off.
        uses, period_code = _factorize(self.rf_use_period)
        use_code = np.empty((self.loop_size, period_code.size))
        use_code[:] = period_code
        rf, rf_column, rf_uses = _collapse(
            self.rf_library, played_rf, _RF_DIGITS, extra=use_code.reshape(-1),
        )
        rf_type = [str(uses[int(code)]) for code in rf_uses[:, 0]] if rf.size else []

        # -- Gradients ------------------------------------------------------
        # Trapezoids and arbitrary waveforms deduplicate apart -- they cannot
        # possibly be equal, and their rows are not even the same width -- but
        # share one library and one numbering, as Pulseq expects. The numbering
        # follows first appearance across both, so the result reads like a
        # sequence somebody wrote block by block.
        played = played_grad > 0
        index = pointed                  # already one row per played slot, in order
        is_arb = index[:, 0] == _ARB
        # Almost every scan is all trapezoids or all waveforms, and when it is,
        # splitting costs two boolean gathers, two more to say where each half
        # came from, and a merge -- five passes over every gradient slot in the
        # scan, to arrive back at the order it started in.
        arbitrary = int(is_arb.sum())
        one_kind = is_arb.size and arbitrary in (0, is_arb.size)

        empty_arb = (np.zeros((0, _ARB_WIDTH)), np.zeros(0, dtype=np.int64))
        empty_trap = (np.zeros((0, _TRAP_WIDTH)), np.zeros(0, dtype=np.int64))
        if one_kind and arbitrary:
            (arb, arb_code), (trap, trap_code) = (
                _collapse_gathered(self.arb_library, index[:, 1], _GRAD_DIGITS), empty_trap)
            appearance = _first_appearance(arb_code, arb.shape[0])
        elif one_kind:
            (arb, arb_code), (trap, trap_code) = (
                empty_arb, _collapse_gathered(self.trap_library, index[:, 1], _GRAD_DIGITS))
            appearance = _first_appearance(trap_code, trap.shape[0])
        else:
            arb, arb_code = _collapse_gathered(
                self.arb_library, index[is_arb, 1], _GRAD_DIGITS)
            trap, trap_code = _collapse_gathered(
                self.trap_library, index[~is_arb, 1], _GRAD_DIGITS)
            appearance = np.concatenate([
                np.flatnonzero(is_arb)[_first_appearance(arb_code, arb.shape[0])],
                np.flatnonzero(~is_arb)[_first_appearance(trap_code, trap.shape[0])],
            ])

        grad_id = np.empty(appearance.size, dtype=np.int64)
        grad_id[np.argsort(appearance, kind='stable')] = np.arange(appearance.size) + 1
        arb_id, trap_id = grad_id[: arb.shape[0]], grad_id[arb.shape[0] :]

        if one_kind:
            renumbered = arb_id[arb_code] if arbitrary else trap_id[trap_code]
        else:
            renumbered = np.empty(index.shape[0], dtype=np.int64)
            renumbered[is_arb] = arb_id[arb_code]
            renumbered[~is_arb] = trap_id[trap_code]
        new_grad_column = np.zeros(played_grad.size, dtype=np.int32)
        new_grad_column[played] = renumbered

        # -- ADC ------------------------------------------------------------
        adc, adc_column, _ = _collapse(self.adc_library, played_adc, _ADC_DIGITS)

        # -- Shapes ---------------------------------------------------------
        shape_map, shapes = self._collapse_shapes(rf, arb, adc)

        # -- Extension specifications ---------------------------------------
        specifications, ref_maps = self._collapse_extension_specs()

        # -- Extension chains -----------------------------------------------
        chains, ext_column = self._collapse_extension_chains(ref_maps)

        # -- Block table ----------------------------------------------------
        table = np.zeros((self.num_blocks, 7), dtype=np.int32)
        table[:, _BLOCK_RF] = rf_column
        table[:, _BLOCK_GRAD] = new_grad_column.reshape(-1, 3)
        table[:, _BLOCK_ADC] = adc_column
        table[:, _BLOCK_EXT] = ext_column

        return SimpleNamespace(
            shapes=shapes,
            shape_map=shape_map,
            rf=rf,
            rf_type=rf_type,
            grad=_by_id(((arb, arb_id, 'g'), (trap, trap_id, 't'))),
            adc=adc,
            specifications=specifications,
            chains=chains,
            block_events=table,
            block_durations=self.block_durations,
        )

    def _collapse_shapes(self, rf: np.ndarray, arb: np.ndarray, adc: np.ndarray):
        """Renumber the shape library over the shapes the survivors still name.

        Shapes are the one library that was already made unique on the way in --
        rebasing merges every module's shapes through `find_or_insert` -- so
        there is nothing to deduplicate here. What is left is pruning: a
        multishot module registers a waveform and a phase modulation per shot,
        and a scan that never played one has no reason to carry it.

        Shape IDs live in three places -- three columns of an RF row, two of an
        arbitrary gradient row, one of an ADC row -- and all of them are
        rewritten here, in place, on the already-deduplicated tables. Zero means
        "no shape" and minus one is Pulseq's half-raster flag; neither is an ID
        and neither moves.
        """
        wanted = np.concatenate([
            rf[:, _RF_SHAPE_COLUMNS].ravel() if rf.size else np.zeros(0),
            arb[:, _GRAD_SHAPE_COLUMNS].ravel() if arb.size else np.zeros(0),
            adc[:, _ADC_SHAPE_COLUMNS].ravel() if adc.size else np.zeros(0),
        ]).astype(np.int64)

        shape_map = {0: 0, -1: -1}
        shapes = []
        for shape_id in wanted:
            if shape_id not in shape_map:
                shapes.append(self._seq.shape_library.data[int(shape_id)])
                shape_map[int(shape_id)] = len(shapes)

        lookup = np.zeros(max(shape_map, default=0) + 2, dtype=np.int64)
        for old, new in shape_map.items():
            lookup[old] = new                       # index -1 is the last slot
        if rf.size:
            rf[:, _RF_SHAPE_COLUMNS] = lookup[rf[:, _RF_SHAPE_COLUMNS].astype(np.int64)]
        if arb.size:
            arb[:, _GRAD_SHAPE_COLUMNS] = lookup[arb[:, _GRAD_SHAPE_COLUMNS].astype(np.int64)]
        if adc.size:
            adc[:, _ADC_SHAPE_COLUMNS] = lookup[adc[:, _ADC_SHAPE_COLUMNS].astype(np.int64)]
        return shape_map, shapes

    def _extension_instances(self):
        """Every extension instance of the scan, as `(block, column, row, alive)`.

        Instances are enumerated block by block and, within a block, type by
        type, which is the order `_ext_slots` hands their rows out in. `row` is
        the instance's row in its own type's dense table -- what the loop wrote
        into -- and `alive` says whether the loop switched it on at all.
        """
        n_ext = len(self._ext_present)
        if n_ext == 0:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, empty, np.zeros(0, dtype=bool)

        starts = self._ext_slots[:, 0::2].astype(np.int64).ravel()
        counts = self._ext_slots[:, 1::2].astype(np.int64).ravel()

        # Only the cells that hold anything. Most blocks of most scans carry no
        # extension at all, and every array below is as long as the cells it is
        # built from -- so enumerating from the occupied ones costs a pass to
        # find them and saves several over the length of the scan.
        occupied = np.flatnonzero(counts)
        held = counts[occupied]
        cell = np.repeat(occupied, held)
        block, column = np.divmod(cell, n_ext)
        within = np.arange(cell.size, dtype=np.int64) - np.repeat(
            np.cumsum(held) - held, held)
        row = starts[cell] + within

        if n_ext == 1:
            # One type of extension, which is what a scan that only labels its
            # readouts has. Every instance is that type, so there is nothing to
            # separate and the switches can be read straight through.
            switch = np.frombuffer(self._ext_switch[0], dtype=np.uint8)
            return block, column, row, switch[row] != 0

        alive = np.zeros(cell.size, dtype=bool)
        for j in range(n_ext):
            mine = column == j
            switch = np.frombuffer(self._ext_switch[j], dtype=np.uint8)
            alive[mine] = switch[row[mine]] != 0
        return block, column, row, alive

    def _extension_tally(self, played: bool = False):
        """Per block: how many extension instances it was built with, or played.

        A running sum over each type's switches answers the second question in
        one pass per type, which is what keeps the block-table views from
        costing an enumeration of every instance in the scan.

        The two are asked for separately because they are wanted separately:
        writing a file needs only what was played, and computing the other
        alongside it would be a scan-length column built for nobody.
        """
        blocks = self.num_blocks
        if not self._ext_present:
            return np.zeros(blocks, dtype=np.int32)

        if not played:
            return self._ext_slots[:, 1::2].sum(axis=1).astype(np.int32)

        total = np.zeros(blocks, dtype=np.int32)
        for j in range(len(self._ext_present)):
            start = self._ext_slots[:, 2 * j].astype(np.int64)
            count = self._ext_slots[:, 2 * j + 1].astype(np.int64)
            switch = np.frombuffer(self._ext_switch[j], dtype=np.uint8)
            running = np.concatenate(([0], np.cumsum(switch, dtype=np.int32)))
            total += running[start + count] - running[start]
        return total

    def _collapse_extension_specs(self):
        """Deduplicate each extension type's own specification library.

        These are the leaves: a trigger, a label and a value, a rotation matrix,
        a soft delay's hint. They know nothing about blocks, so they collapse on
        their own -- over the instances the loop switched on, which is also how
        an instance nobody played stops costing anything -- and what comes back
        is the renumbering the chains above them need.
        """
        specifications = {}
        ref_maps = {}
        for j, column in enumerate(self._ext_present):
            name = _EXTENSION_NAME[column]
            switch = np.frombuffer(self._ext_switch[j], dtype=np.uint8)
            played = np.nonzero(switch)[0]
            if column == EXTENSION_MAP['DELAYS']:
                # A soft delay's row ends in a string, so it is made unique by
                # what it says rather than by rounding it to the precision the
                # file writes.
                unique, seen = [], {}
                mapping = np.zeros(switch.size + 1, dtype=np.int64)
                for row in played:
                    data = tuple(self._soft_delay_rows[row])
                    if data not in seen:
                        unique.append(data)
                        seen[data] = len(unique)
                    mapping[row + 1] = seen[data]
                specifications[name] = unique
            else:
                table = getattr(self, _EXTENSION_LIBRARY[name])
                unique, code = _unique_rows(_rounded(table[played]))
                mapping = np.zeros(table.shape[0] + 1, dtype=np.int64)
                mapping[played + 1] = code + 1
                specifications[name] = unique
            ref_maps[self._ext_native[column]] = mapping
        return specifications, ref_maps

    def _collapse_extension_chains(self, ref_maps):
        """String the surviving instances of each block into one extension chain.

        Pulseq stores a block's extensions as a linked list whose head is its
        last entry, so the chain is built tail first and a row's `next` is
        always the row before it -- which is also what `_collapse_chains` needs
        in order to resolve a chain before anything that points at it.

        The chain is assembled here and not in the constructor because until the
        loop has run there is no telling which instances a block carries: a
        dummy shot that was handed no label ends up with a head of zero, and the
        rows it did not use were never referenced by anything.
        """
        block, column, row, alive = self._extension_instances()
        heads = np.zeros(self.num_blocks, dtype=np.int64)
        if not alive.any():
            return None, heads.astype(np.int32)

        keep = np.nonzero(alive)[0]
        block, column, row = block[keep], column[keep], row[keep]
        native = np.array(
            [self._ext_native[c] for c in self._ext_present], dtype=np.int64
        )

        count = keep.size
        following = np.zeros(count, dtype=np.int64)
        same = block[1:] == block[:-1]
        following[1:][same] = np.arange(1, count, dtype=np.int64)[same]

        last = np.ones(count, dtype=bool)
        last[:-1] = ~same
        heads[block[last]] = np.arange(1, count + 1, dtype=np.int64)[last]

        chains = np.column_stack([native[column], row + 1, following])
        return _collapse_chains(chains, heads, ref_maps)

    def build(self) -> Sequence:
        """A fresh Pulseq sequence holding the whole scan.

        Deduplication has already made the libraries small; what is left is
        transcription. The block table is the one thing that stays large -- it
        is one row per block by definition -- and Pulseq wants it as a dict,
        which is the cost this whole design exists to defer rather than avoid.
        """
        compact = self.deduplicate_events()
        seq = Sequence(system=self.system)
        seq.definitions = dict(self._seq.definitions)

        for new_id, data in enumerate(compact.shapes, start=1):
            seq.shape_library.insert(new_id, data)
        _fill_library(seq.rf_library, compact.rf, compact.rf_type)
        for new_id, (row, kind) in enumerate(compact.grad, start=1):
            seq.grad_library.insert(new_id, tuple(row), kind)
        _fill_library(seq.adc_library, compact.adc)

        # Keep each extension's numeric ID, because the chains name it.
        native_of = dict(zip(self._seq.extension_string_idx, self._seq.extension_numeric_idx))
        for name, table in compact.specifications.items():
            if len(table) == 0:
                continue
            seq.set_extension_string_ID(name, native_of[name])
            _fill_library(getattr(seq, _EXTENSION_LIBRARY[name]), table)
            if name == 'DELAYS':
                # The hint is what names a soft delay on the console, and the
                # sequence keeps its own index of them.
                seq.soft_delay_hints = {row[3]: int(row[0]) for row in table}
        if compact.chains is not None:
            # Handed over rather than filled in, for the same reason the block
            # table is: half a million chain rows on a 3D scan, and writing a
            # file wants them back as columns. `defer_chains` builds the library
            # on the first ask, which is what decoding a block does and what
            # writing one does not.
            seq.defer_chains(compact.chains)

        # Handed over as the two arrays deduplication produced rather than as
        # the dicts Pulseq keeps, because writing a file wants the columns back
        # and building a million-row dict only to take it apart again is the
        # single largest cost left in this method. `defer_blocks` builds them on
        # the first ask, and writing never asks.
        seq.defer_blocks(compact.block_events, compact.block_durations)
        return seq

    @property
    def seq(self) -> Sequence:
        """The whole scan, as an ordinary Pulseq sequence.

        This is the conversion the design has been deferring: the dense
        per-occurrence tables collapse into real event libraries, and what comes
        back is a `pp.Sequence` on the same `Opts` the period was built against,
        with no trace of how it was assembled. Everything PyPulseq can do to a
        sequence -- `write`, `plot`, `calculate_kspace`, `test_report` -- it can
        do to this.

        Built on first ask and kept, because building it is the expensive half
        of the whole exercise and the loop is over by the time anyone can ask:
        `deduplicate_events` refuses to run until every block has been filled,
        and once it has, `add_block` cannot be called again.
        """
        if self._output is None:
            self._output = self.build()
        return self._output

    def write(self, path=None, **kwargs):
        """Write the scan as a .seq file, or return its bytes when `path` is None.

        This is the direct route it looks like: `seq` no longer builds anything
        much -- deduplication produces the tables, and what it hands to the
        sequence it hands over rather than transcribing -- so writing goes from
        the deduplicated arrays to the file with a few microseconds of Pulseq
        bookkeeping in between. The sequence stays available afterwards, which
        is what `plot`, `calculate_kspace` and `get_block` need; they are the
        only things that pay to have the libraries built as dictionaries.
        """
        return self.seq.write(path, **kwargs)

    def write_binary(self, path=None, **kwargs):
        """Write the scan in Pulseq's binary format, or return its bytes.

        Same route as `write`, into the other container. See
        `fastseq.binary.write_binary`.
        """
        return self.seq.write_binary(path, **kwargs)

    def rebased_events(self, module: SequenceModule) -> SimpleNamespace:
        """`module.events`, addressing this sequence's libraries.

        Composing the module already rebased them in place, so this is just
        `module.events` -- and the point of saying so is that the loop may take
        its events straight off the module, before or after this sequence was
        built, with no bookkeeping of its own.
        """
        if id(module) not in self.bindings:
            raise KeyError(f'{type(module).__name__} is not part of this sequence')
        return module.events

    def _build_extension_libraries(self) -> None:
        """Unroll the period's extensions into one row per instance per period.

        Same bargain as the event libraries: an extension the template carries
        once is given a row per block instance of the whole scan, so the loop can
        put a different label value in every TR without disturbing its
        neighbours, and pay for it once at write time.

        What comes out of here is three things per extension type the period
        actually uses. A dense table of specification rows, seeded from the
        template. A switch per instance, since an extension the loop does not
        hand in is one the block does not carry. And `_ext_slots`, which is what
        turns "the second label of block n" into a row number with two array
        reads and no lookup at all.
        """
        seq = self._seq
        loop_size = self.loop_size

        column_of = {}
        for native_id, name in zip(seq.extension_numeric_idx, seq.extension_string_idx):
            if name not in EXTENSION_MAP:
                raise ValueError(f'Unsupported block extension {name!r}')
            column_of[native_id] = EXTENSION_MAP[name]
        self._ext_native = {column: native_id for native_id, column in column_of.items()}

        block_rows = np.stack(list(seq.block_events.values())).astype(np.int64)
        heads = block_rows[:, _BLOCK_EXT]
        period_blocks = heads.size

        # Per block, how many extensions of each type, and which library row each
        # instance points at. Pulseq stores a block's extensions as a reversed
        # linked list ordered by library ID, so what comes out of the walk is not
        # the order they were written in -- which does not matter, because a
        # block's extensions of one type are a set, not a sequence, and the loop
        # writes every column of the row it fills.
        counts = np.zeros((period_blocks, len(EXTENSION_MAP)), dtype=np.int64)
        refs = [[[] for _ in EXTENSION_MAP] for _ in range(period_blocks)]
        for n, ext_id in enumerate(heads):
            while ext_id > 0:
                type_id, ref_id, ext_id = seq.extensions_library.data[ext_id]
                column = column_of[type_id]
                counts[n, column] += 1
                refs[n][column].append(ref_id)

        self.extension_counts = counts
        if counts.max(initial=0) > 255:
            raise ValueError(
                'a block carries more than 255 extensions of one type, which is '
                'more than add_block can count through in one block'
            )

        # Only the types this period actually uses get a table, a switch and a
        # column of `_ext_slots`. A Cartesian scan labels and nothing else, so
        # this is normally one column rather than six.
        self._ext_present = [
            column for column in range(len(EXTENSION_MAP)) if counts[:, column].any()
        ]
        self._ext_column = tuple(
            self._ext_present.index(column) if column in self._ext_present else -1
            for column in range(len(EXTENSION_MAP))
        )

        # Seed each preallocated row with the value the period already carries,
        # so an instance the loop switches on but never writes plays what the
        # module designed. Rows are laid out type by type, and within a type in
        # block order, which is the numbering `_ext_slots` hands out.
        self._ext_tables = []
        self._ext_widths = []
        self._ext_switch = []
        self._soft_delay_rows = None
        delays = EXTENSION_MAP['DELAYS']
        for column in self._ext_present:
            name = _EXTENSION_NAME[column]
            source = getattr(seq, _EXTENSION_LIBRARY[name], None)
            if source is None:
                raise ValueError(f'this PyPulseq has no library for {name!r} extensions')
            rows = [source.data[ref] for block in refs for ref in block[column]]
            if column == delays:
                # A soft delay's row ends in the hint the operator reads, so it
                # is the one specification that is not a row of numbers. It is
                # also one nothing varies about, so a list is enough.
                self._soft_delay_rows = rows * loop_size
                self._ext_tables.append(None)
                self._ext_widths.append(0)
            else:
                table = np.tile(np.array(rows, dtype=float), (loop_size, 1))
                setattr(self, _EXTENSION_LIBRARY[name], table)
                varies = column in _EXTENSION_VARIES
                self._ext_tables.append(_flat(table) if varies else None)
                self._ext_widths.append(table.shape[1])
            self._ext_switch.append(bytearray(len(rows) * loop_size))

        # Where each block's instances of each type begin, and how many there
        # are. Interleaved so that one cache line answers both questions.
        n_ext = len(self._ext_present)
        per_block = counts[:, self._ext_present] if n_ext else np.zeros((period_blocks, 0), int)
        totals = per_block.sum(axis=0)
        starts = np.cumsum(per_block, axis=0) - per_block
        slots = np.zeros((self.loop_size * period_blocks, 2 * n_ext), dtype=np.int32)
        if n_ext:
            # The period broadcast into the scan's shape, rather than tiled and
            # then added to a per-period offset repeated beside it: the same
            # answer without three scan-length temporaries in the 64 bits
            # `arange` and `cumsum` default to, for a table declared `int32`.
            unrolled = np.empty((loop_size, period_blocks, n_ext), dtype=np.int32)
            unrolled[:] = starts
            unrolled += np.arange(loop_size, dtype=np.int64)[:, None, None] * totals
            slots[:, 0::2] = unrolled.reshape(-1, n_ext)
            unrolled[:] = per_block
            slots[:, 1::2] = unrolled.reshape(-1, n_ext)
        self._ext_slots = slots
        self._ext_stride = 2 * n_ext
        self._slots_ext = _flat(slots)

# %% example
def make_mprage(path=None, binary=True):
    """A 512x1024x512 MPRAGE, designed and serialized.

    Ends by writing rather than by handing back a sequence, so that timing this
    function times the whole job: composing the period, two million blocks
    through the loop, deduplicating them, and the Pulseq file itself.
    `path` may be a filename, or None to serialize and drop the bytes, which is
    the same work without the disk in it.

    Writes Pulseq's native *binary* format by default, which is what the C
    reader wants: the same scan parses in about a tenth of the time, because
    every section carries its own entry count and nothing has to be tokenized.
    Pass ``binary=False`` for the text `.seq`. A path is given the matching
    suffix either way, so `make_mprage('mprage')` writes `mprage.bin` here and
    `mprage.seq` there.

    The `PeriodicSequence` comes back either way. Nothing has been spent on
    making it comfortable to inspect -- its libraries are still the arrays
    deduplication produced -- and asking it for `seq.plot()` or
    `seq.calculate_kspace()` is what pays for that, once.
    """
    # Hardcoded params
    txrx_raster = 2e-6
    grad_raster = 4e-6
    
    flip_angle_deg = 10.0
    excitation_duration_s = 1e-3
    FOV_m = (225e-3, 225e-3, 225e-3)
    Nx, Ny, Nz = 512, 1024, 512
    spoiling_area = 3e3
    adc_labels = ('LIN', 'PAR')
    TR_s = 7.0
    rf_spoiling_inc = 117.0
    
    system = pp.Opts(
        rf_raster_time=txrx_raster, 
        adc_raster_time=txrx_raster, 
        grad_raster_time=grad_raster, 
        block_duration_raster=grad_raster
    )
    excitation = NonSelectiveExcitation(
        system=system, 
        flip_angle_deg=flip_angle_deg, 
        duration_s=excitation_duration_s
    )
    # The inversion is asked for a label slot, because the loop marks the first
    # shot with ONCE and a block can only carry an extension its template built
    # room for.
    inversion = InversionPrep(system=system, labels=('ONCE',))
    spgr_read = LineReadout3D(
        system=system, 
        excitation=excitation, 
        FOV_m=FOV_m, 
        matrix=(Nx, Ny, Nz), 
        spoiling_area=spoiling_area, 
        labels=adc_labels,
    )
    spgr_train = Ny * [spgr_read]
    
    # Extract from modules the events, so they have correct registration. A
    # module answers to its published events directly, so `inversion.rf` and
    # `inversion.events.rf` are the same object.
    rf_inv = inversion.rf
    gz_crusher = inversion.gz_crusher
    rf_exc = spgr_read.rf
    gx_pre, gy_phase, gz_phase = spgr_read.gx_pre, spgr_read.gy_pre, spgr_read.gz_pre
    gx_read, adc = spgr_read.gx_read, spgr_read.adc
    gx_rew = spgr_read.gx_rew
    
    # Generate delay to target MPRAGE tr
    tr_min = inversion.duration + Ny * spgr_read.duration
    tr_delay = TR_s - tr_min
    wait_tr = pp.make_delay(tr_delay)
    
    # Generate scalings.
    #
    # `tolist` and the two `float`s are not cosmetic: what the loop assigns is
    # read back out of a SimpleNamespace and stored into a plain float buffer, so
    # a numpy scalar has to be unboxed on the way. Multiplying two numpy scalars
    # costs about a hundred nanoseconds against fourteen for two floats, and this
    # happens four times per readout -- a tenth of a microsecond per block on a
    # scan of two million of them.
    pey_scale = ((2 * np.arange(Ny) - Ny) / Ny).tolist()  # Inner readout phase encode already accounts for 1/2
    pez_scale = ((2 * np.arange(Nz) - Nz) / Nz).tolist()  # Inner readout phase encode already accounts for 1/2
    gy_amplitude = float(gy_phase.amplitude)
    gz_amplitude = float(gz_phase.amplitude)
    #: `np.deg2rad` on one scalar costs about six hundred nanoseconds -- it is a
    #: ufunc call, not arithmetic -- and the RF spoiling phase is converted twice
    #: per readout.
    deg2rad = np.pi / 180.0

    # Initialize sequence
    mprage = PeriodicSequence(system, Nz+1, inversion, *spgr_train, wait_tr)
    rf_phase = 0.0
    rf_inc = 0.0
    
    gy_pre = copy.deepcopy(gy_phase)
    gy_rew = copy.deepcopy(gy_phase)
    gz_pre = copy.deepcopy(gz_phase)
    gz_rew = copy.deepcopy(gz_phase)
    
    # Dummy shots
    once_label = pp.make_label(type='SET', label='ONCE', value=1)
    gz_pre.amplitude = 0.0
    gz_rew.amplitude = 0.0
    mprage.add_block(rf_inv, once_label)
    mprage.add_block(gz_crusher)
    for y in range(Ny):
        rf_exc.phase_offset = rf_phase * deg2rad
        gy_pre.amplitude = 0.0
        gy_rew.amplitude = 0.0
        mprage.add_block(rf_exc)
        mprage.add_block(gx_pre, gy_pre, gz_pre)
        mprage.add_block(gx_read)
        mprage.add_block(gx_rew, gy_rew, gz_rew)
        rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
        rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
    mprage.add_block(wait_tr)
        
    # Imaging
    once_label.value = 0
    lin_label = pp.make_label(type='SET', label='LIN', value=0)
    par_label = pp.make_label(type='SET', label='PAR', value=0)
        
    for z in range(Nz):
        par_label.value = z
        gz_pre.amplitude = pez_scale[z] * gz_amplitude
        gz_rew.amplitude = -pez_scale[z] * gz_amplitude
        if z == 1:
            mprage.add_block(rf_inv, once_label)
        else:
            mprage.add_block(rf_inv)
        mprage.add_block(gz_crusher)
        for y in range(Ny):
            phase = rf_phase * deg2rad
            rf_exc.phase_offset = phase
            adc.phase_offset = phase
            lin_label.value = y
            gy_pre.amplitude = pey_scale[y] * gy_amplitude
            gy_rew.amplitude = -pey_scale[y] * gy_amplitude
            mprage.add_block(rf_exc)
            mprage.add_block(gx_pre, gy_pre, gz_pre)
            mprage.add_block(gx_read, adc, lin_label, par_label)
            mprage.add_block(gx_rew, gy_rew, gz_rew)
            rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]
        mprage.add_block(wait_tr)

    # The loop is over, so the scan is decided: libraries deduplicated, unplayed
    # rows dropped, one block table. That is the expensive half, and it happens
    # here, on the way to the file -- the tables go to `write` as they come out
    # of it, and nothing is built into dictionaries that only a reader would
    # want. What comes back is a sequence PyPulseq cannot tell from one written
    # a block at a time, and `plot`, `calculate_kspace` and `get_block` all work
    # on it; they are simply what pays for the dictionaries, and only if asked.
    if binary:
        mprage.write_binary(path)
    else:
        mprage.write(path)

    return mprage


# %% example: a trajectory that is *not* one base shot plus a rotation
def make_stack_of_stars():
    """A stack of stars with one explicitly built waveform per spoke.

    A radial stack is of course a single spoke plus a rotation, and that is how
    it should be played. It is built the other way round here on purpose, to
    stand in for the case that has no shortcut: a trajectory whose shots are not
    related by anything the interpreter can apply at runtime.

    The cost of doing it the dumb way is one registered waveform per spoke and
    nothing else -- the module is still three blocks long, and the loop picks a
    spoke by handing one in. (A radial stack then deduplicates back down to a
    single shape at write time, because a rotation really is a scaling of one
    waveform. A trajectory that genuinely differs would keep its shapes, and
    that is the only difference.)
    """
    txrx_raster = 2e-6
    grad_raster = 4e-6

    flip_angle_deg = 10.0
    excitation_duration_s = 1e-3
    FOV_m = (225e-3, 225e-3)
    Nx, Nz, num_spokes = 256, 64, 233
    spoiling_area = 3e3
    rf_spoiling_inc = 117.0

    system = pp.Opts(
        rf_raster_time=txrx_raster,
        adc_raster_time=txrx_raster,
        grad_raster_time=grad_raster,
        block_duration_raster=grad_raster,
    )
    excitation = NonSelectiveExcitation(
        system=system,
        flip_angle_deg=flip_angle_deg,
        duration_s=excitation_duration_s,
    )
    readout = StackOfStarsReadout(
        system=system,
        excitation=excitation,
        FOV_m=FOV_m,
        matrix=(Nx, Nz),
        spoiling_area=spoiling_area,
        labels=('LIN', 'PAR'),
        num_shots=num_spokes,
    )

    # The events, exactly as the module published them. `gx_read` and `gy_read`
    # are lists, one entry per spoke, each carrying its own registered shape.
    rf = readout.rf
    gx_read, gy_read = readout.gx_read, readout.gy_read
    gz_phase, adc = readout.gz_phase, readout.adc
    gz_spoil = readout.gz_spoil
    lin_label, par_label = readout.adc_labels

    pez_scale = (2 * np.arange(Nz) - Nz) / Nz

    # One period per TR: the loop is as long as the scan.
    sos = PeriodicSequence(system, Nz * num_spokes, readout)
    rf_phase = 0.0
    rf_inc = 0.0

    # After the sequence, not before: composing is what tells this waveform
    # which amplitude shape it plays.
    gz_encode = copy.deepcopy(gz_phase)

    for z in range(Nz):
        par_label.value = z
        # Scaling an arbitrary gradient is three numbers, not a new waveform:
        # the samples are the shape, and the shape does not change.
        gz_encode.amplitude = pez_scale[z] * gz_phase.amplitude
        gz_encode.first = pez_scale[z] * gz_phase.first
        gz_encode.last = pez_scale[z] * gz_phase.last
        for s in range(num_spokes):
            rf.phase_offset = np.deg2rad(rf_phase)
            adc.phase_offset = np.deg2rad(rf_phase)
            lin_label.value = s
            sos.add_block(rf)
            # The spoke is chosen here, and choosing it costs one shape ID.
            sos.add_block(gx_read[s], gy_read[s], gz_encode, adc, lin_label, par_label)
            sos.add_block(gz_spoil)
            rf_inc = divmod(rf_inc + rf_spoiling_inc, 360.0)[1]
            rf_phase = divmod(rf_phase + rf_inc, 360.0)[1]

    return sos
