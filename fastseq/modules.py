"""
"""

__all__ = []

import math

from fractions import Fraction
from types import SimpleNamespace
from typing import Literal

import numpy as np

import pypulseq as pp

from pypulseq.utils.seq_plot import SeqPlot

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
        rf_plot: Literal['auto', 'abs', 'real', 'imag'] = 'auto',
    ) -> SeqPlot:
        self._seq.plot(
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
        
    return block


class SequenceModule(SeqPlotMixin):
    """
    """
    def __init__(self):
        self._seq = None
        self.center = 0.0
        self.duration = 0.0
        self.events = SimpleNamespace()
            
    @property
    def blocks(self):
        num_blocks = len(self._seq.block_events)
        return [_get_block(self._seq, n) for n in range(1, num_blocks+1)]
        
        
class InversionPrep(SequenceModule): 
    """
    """
    def __init__(
        self, 
        system: pp.Opts,
        duration_s: float = 10.0e-3,
        spoiling_area: float = 2000.0,
    ):
        super().__init__()
        rf = pp.make_adiabatic_pulse(
            pulse_type='hypsec',
            duration=duration_s,
        )
        gz_crusher = pp.make_trapezoid(
            channel='z', 
            system=system, 
            area=spoiling_area,
        )
        seq = pp.Sequence(system=system)
        
        # Register events
        rf.id, rf.shape_IDs = seq.register_rf_event(rf)
        
        # Create Inversion Preparation module
        seq.add_block(rf)
        seq.add_block(gz_crusher)
        
        # Assign objects
        self._seq = seq
        self.duration = seq.duration()[0]
        self.center = rf.center
        self.events.rf = rf
        self.events.gz_crusher = gz_crusher


class NonSelectiveExcitation(SequenceModule):
    def __init__(
        self, 
        system: pp.Opts,
        flip_angle_deg = 10.0,
        duration_s: float = 1e-3,
    ):
        super().__init__()
        rf = pp.make_block_pulse(
            flip_angle=np.deg2rad(flip_angle_deg),
            duration=duration_s,
        )
        
        seq = pp.Sequence(system=system)
        
        # Register event
        rf.id, rf.shape_IDs = seq.register_rf_event(rf)
        
        # Create Excitation module
        seq.add_block(rf)
        
        # Assign objects
        self._seq = seq
        self.duration = seq.duration()[0]
        self.center = rf.center
        self.events.rf = rf
    
        
class LineReadout3D(SequenceModule):
    """
    """
    def __init__(
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
        super().__init__()
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
            area=pey_area
        ) 
        pez_area_step = 1 / FOV_m[2]
        pez_area = -matrix[2] // 2 * pez_area_step
        gz_phase = pp.make_trapezoid(
            channel='z',
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
            
        # Initialize readout object
        seq = pp.Sequence(system=system)
        
        # Register shaped objects
        delattr(excitation.events.rf, 'id')
        delattr(excitation.events.rf, 'shape_IDs')
        excitation.events.rf.id, excitation.events.rf.shape_IDs = seq.register_rf_event(excitation.events.rf)
        if hasattr(excitation.events, 'gz_slab') and excitation.events.gz_slab.type == 'grad':
            delattr(excitation.events.gz_slab, 'id')
            delattr(excitation.events.gz_slab, 'shape_IDs')
            excitation.events.gz_slab.id, excitation.events.gz_slab.shape_IDs = seq.register_grad_event(excitation.events.gz_slab)
        if gx_pre.type == 'grad':
            gx_pre.id, gx_pre.shape_IDs = seq.register_grad_event(gx_pre)
        if gx_read.type == 'grad':
            gx_read.id, gx_read.shape_IDs = seq.register_grad_event(gx_read)
        if gx_rew.type == 'grad':
            gx_rew.id, gx_rew.shape_IDs = seq.register_grad_event(gx_rew)
        
        # Build readout
        once_label = pp.make_label(type='SET', label='ONCE', value=0)
        if hasattr(excitation.events, 'gz_slab'):
            seq.add_block(
                excitation.events.rf, 
                excitation.events.gz_slab, 
                once_label
            )
        else:
            seq.add_block(
                excitation.events.rf, 
                once_label
            )
        if te is not None:
            seq.add_block(wait_te)
        if trigger is not None:
            trigger, gx_pre, gy_pre, gz_pre = pp.align(
                left=trigger, 
                right=[gx_pre, gy_phase, gz_phase]
            )
            seq.add_block(
                gx_pre, gy_pre, gz_pre, trigger,
            )
        else:
            gx_pre, gy_pre, gz_pre = pp.align(
                right=[gx_pre, gy_phase, gz_phase]
            )
            seq.add_block(
                gx_pre, gy_pre, gz_pre,
            )
        adc_labels = [pp.make_label(type='SET', label=lbl, value=0) for lbl in labels]
        adc = pp.make_adc(
            system=system, 
            num_samples=num_samples,
            dwell=dwell_time,
            delay=gx_read_rise_time,
        )
        seq.add_block(
            gx_read,
            adc,
            *adc_labels,
        )
        gx_rew, gy_rew, gz_rew = pp.align(
            left=[gx_rew, pp.scale_grad(gy_phase, -1.0), pp.scale_grad(gz_phase, -1.0)]
        )
        seq.add_block(
            gx_rew, gy_rew, gz_rew,
        )
        tr_min = seq.duration()[0]
        if tr is not None:
            tr_delay = tr - tr_min
            if tr_delay < 0:
                raise ValueError('requested TR is shorter than min TR')
            wait_tr = pp.make_delay(tr_delay)
            seq.add_block(wait_tr)
            
        # Assign objects
        self._seq = seq.remove_duplicates()
        self.duration = seq.duration()[0]
        self.center = readout_reference.item()
        self.events.once_label = once_label
        self.events.rf = excitation.events.rf
        if hasattr(excitation.events, 'gz_slab'):
            self.events.gz_slab = excitation.events.gz_slab
        if te is not None:
            self.events.wait_te = wait_te
        self.events.gx_pre = gx_pre
        self.events.gy_pre = gy_pre
        self.events.gy_pre = gz_pre
        if trigger is not None:
            self.events.trigger = trigger
        self.events.gx_read = gx_read
        self.events.adc = adc
        if len(adc_labels):
            self.events.adc_labels = adc_labels
        self.events.gx_rew = gx_rew
        self.events.gy_rew = gy_rew
        self.events.gz_rew = gz_rew
                
EXTENSION_MAP = {
    'TRIGGERS': 0,
    'LABELSET': 1,
    'LABELINC': 2,
    'RF_SHIMS': 3,
    'ROTATIONS': 4,
    }


def _build_extension_table(A: np.ndarray) -> np.ndarray:
    _, M = A.shape
    counts = A.ravel(order="C")
    K = int(counts.sum())

    if K == 0:
        return None
    
    B = np.zeros((K, 3), dtype=np.int64)

    # Flattened A-cell index for every individual event.
    cell_id = np.repeat(
        np.arange(A.size, dtype=np.intp),
        counts,
    )

    row_id, col_id = np.divmod(cell_id, M)

    # Position of each event within its A[n, m] entry: 1, ..., A[n, m].
    cell_start = np.cumsum(counts, dtype=np.int64) - counts
    within_cell = (
        np.arange(K, dtype=np.int64)
        - cell_start[cell_id]
        + 1
    )

    # Number of preceding events of the same type in earlier timepoints.
    previous_of_type = (
        np.cumsum(A, axis=0, dtype=np.int64) - A
    ).ravel(order="C")

    # 1-based event-type/column ID.
    B[:, 0] = col_id + 1

    # 1-based index in the corresponding event-type library.
    B[:, 1] = previous_of_type[cell_id] + within_cell

    # 1-based pointer to the next B row belonging to the same A row.
    # Zero terminates the chain.
    B[:-1, 2] = np.where(
        row_id[:-1] == row_id[1:],
        np.arange(2, K + 1),
        0,
    )

    return B


def _rebase_shapes(seq: pp.Sequence, module: SequenceModule):
    _seq = pp.Sequence(system=module._seq.system)
    
    # Unpack input module in its blocks
    blocks = module.blocks
    
    # Enfoce module event names
    events = {}
    event_id_to_name = {}
    for name, event in module.events.__dict__.items():
        if isinstance(event, SimpleNamespace):
            if event.type == 'rf' or event.type == 'grad':
                event.name = name
                event_id_to_name[event.id] = name
                delattr(event, 'id')
                delattr(event, 'shape_IDs')
        elif isinstance(event, list):
            for n in range(len(event)):
                if isinstance(event[n], SimpleNamespace) :
                    if event[n].type == 'rf' or event[n].type == 'grad':
                        event[n].name = name
                        event_id_to_name[event[n].id] = name
                        delattr(event[n], 'id')
                        delattr(event[n], 'shape_IDs')
        events[name] = event
        
    # Iterate over blocks and update shape ID
    for block in blocks:
        current_events = list(pp.block_to_events.block_to_events(block))
        for n in range(len(current_events)):
            if isinstance(current_events[n], SimpleNamespace):
                if current_events[n].type == 'rf':
                    print(current_events[n])
                    event_name = event_id_to_name[current_events[n].id]
                    new_id, new_shape_IDs = seq.register_rf_event(current_events[n])
                    current_events[n].id = new_id
                    current_events[n].shape_IDs = new_shape_IDs
                    events[event_name].id = new_id
                    events[event_name].shape_IDs = new_shape_IDs
                if current_events[n].type == 'grad':
                    event_name = event_id_to_name[current_events[n].id]
                    new_id, new_shape_IDs = seq.register_grad_event(current_events[n])
                    current_events[n].id = new_id
                    current_events[n].shape_IDs = new_shape_IDs
                    events[event_name].id = new_id
                    events[event_name].shape_IDs = new_shape_IDs
            elif isinstance(current_events[n], list):
                for m in range(len(current_events[n])):
                    if isinstance(current_events[n][m], SimpleNamespace):
                        if current_events[n][m].type == 'rf':
                           event_name = event_id_to_name[current_events[n][m].id]
                           new_id, new_shape_IDs = seq.register_rf_event(current_events[n][m])
                           current_events[n][m].id = new_id
                           current_events[n][m].shape_IDs = new_shape_IDs
                           events[event_name][m].id = new_id
                           events[event_name][m].shape_IDs = new_shape_IDs
                        if current_events[n][m].type == 'grad':
                            event_name = event_id_to_name[current_events[n][m].id]
                            new_id, new_shape_IDs = seq.register_grad_event(current_events[n][m])
                            current_events[n][m].id = new_id
                            current_events[n][m].shape_IDs = new_shape_IDs
                            events[event_name][m].id = new_id
                            events[event_name][m].shape_IDs = new_shape_IDs
                            
        # transform duration in delay
        current_events[0] = pp.make_delay(current_events[0])
        _seq.add_block(*current_events)
    
    # Update
    module._seq = _seq
    module.events = SimpleNamespace(**events)
    
    return module


class PeriodicSequence(SeqPlotMixin):
    def __init__(
        self,
        system: pp.Opts,
        loop_size: int,
        *blocks_or_modules,
    ):
        self.trigger_library = None
        self.label_set_library = None
        self.label_inc_library = None
        self.rf_shim_library = None
        self.rotation_library = None
        seq = pp.Sequence(system=system)
        
        # Register shapes
        itemIDs = [id(item) for item in blocks_or_modules]
        _, unique_blocks_or_modules = np.unique(itemIDs, return_index=True)
        unique_blocks_or_modules = unique_blocks_or_modules.tolist()
        sorted(unique_blocks_or_modules)
        for index in unique_blocks_or_modules:
            item = blocks_or_modules[index]
            if isinstance(item, SimpleNamespace):
                if item.type == 'rf':
                    item.id, item.shape_IDs = seq.register_rf_event(item)
                if item.type == 'grad':
                    item.id, item.shape_IDs = seq.register_grad_event(item)
            else:
                item = _rebase_shapes(seq, item)

        # Build template
        for item in blocks_or_modules:
            if isinstance(item, SimpleNamespace):
                seq.add_block(item)
            else:
                for block in item.blocks:
                    seq.add_block(block)
    
        self._seq = seq
        self.tr = seq.duration()[0]
        self.cursor = 1 # pypulseq is 1-based
        
        if len(self._seq.soft_delay_library.data) > 0:
            raise ValueError ('Soft delays are unsupported at the moment')
        
        # Build structured sequence
        block_events = np.stack(list(seq.block_events.values()), axis=1)
        
        # Get extension column
        extensions = block_events[-1]

        # (triggers/digitalout; labelset; labelinc; rfshim; rotations)
        extensions_per_tr = np.zeros((extensions.size, 5), dtype=int)
        extensions_map = {
            native_id: EXTENSION_MAP[name]
            for native_id, name in zip(
                seq.extension_numeric_idx,
                seq.extension_string_idx,
            )
        }
        
        # Fixed numerical ID -> native extension_numeric_idx
        native_extensions_map = {
            fixed_id: native_id
            for native_id, fixed_id in extensions_map.items()
        }
        
        # Loop over extensions and count extension type for each row
        for n, ext_id in enumerate(extensions):
            while ext_id > 0:
                ext_entry = seq.extensions_library.data[ext_id]
                ext_type = extensions_map[ext_entry[0]]
                extensions_per_tr[n, ext_type] += 1
                ext_id = ext_entry[-1]
        
        # Allocate each individual extension library type
        num_trigger_per_tr = sum(extensions_per_tr, 0)
        if num_trigger_per_tr:
            self.trigger_library = np.zeros((loop_size * num_trigger_per_tr, 4))    
        num_label_set_per_tr = sum(extensions_per_tr, 1)
        if num_label_set_per_tr:
            self.label_set_library = np.zeros((loop_size * num_label_set_per_tr, 2))
        num_label_inc_per_tr = sum(extensions_per_tr, 2)
        if num_label_inc_per_tr:
            self.label_inc_library = np.zeros((loop_size * num_label_inc_per_tr, 2))
        num_rf_shim_per_tr = sum(extensions_per_tr, 3)
        if num_rf_shim_per_tr:
            num_channels = self.rf_shim_library.data[1][0]
            self.rf_shim_library = np.zeros((loop_size * num_rf_shim_per_tr, 2 * num_channels + 1))
            self.rf_shim_library[:, 0] = num_channels
        num_rotation_per_tr = sum(extensions_per_tr, 4)
        if num_rotation_per_tr:
            self.rotation_library = np.zeros((loop_size * num_rotation_per_tr, 4))
            
        # Now create the extensions library
        extensions_in_sequence = np.tile(extensions_per_tr, [loop_size, 1])
        self.extension_library = _build_extension_table(extensions_in_sequence)
        
        # Restore native extension type identifier
        for n in range(self.extension_library.shape[0]):
            self.extension_library[n, 0] = native_extensions_map[self.extension_library[n, 0]]
        
                    
# %% example
def make_mprage():
    system = pp.Opts(rf_raster_time=2e-6, adc_raster_time=2e-6, grad_raster_time=4e-6, block_duration_raster=4e-6)
    excitation = NonSelectiveExcitation(system=system, flip_angle_deg=10.0, duration_s=1e-3)
    
    inversion = InversionPrep(system=system)
    spgr_read = LineReadout3D(system=system, excitation=excitation, FOV_m=(225e-3,225e-3,225e-3), matrix=(512,512,512), spoiling_area=3e3, labels=('LIN', 'PAR'))
    spgr_train = 1024 * [spgr_read]
    mprage = PeriodicSequence(system, 512, inversion, *spgr_train)
    
    return mprage
