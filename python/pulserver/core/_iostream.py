"""Serialization of Sequence to binary stream."""

__all__ = ["write_to_stream"]

import io
import hashlib

from warnings import warn

import numpy as np
import pypulseq as pp

from pypulseq.supported_labels_rf_use import get_supported_rf_uses, get_supported_labels
from pypulseq.utils.tracing import format_trace, trace_enabled

def write_to_stream(
    seq: pp.Sequence,
    create_signature: bool = True,
    remove_duplicates: bool = False,
    check_timing: bool = False,
) -> bytes | tuple[bytes, str]:
    """
    Write the sequence data to a binary stream using the open file format for MR sequences.

    See also `pypulseq.Sequence.write_seq.write()`.

    Parameters
    ----------
    seq : pp.Sequence
        Sequence object to be serielized
    create_signature : bool, default=True
        Boolean flag to indicate if the file has to be signed.
    remove_duplicates : bool, default=True
        Remove duplicate events from the sequence before writing
    v141_compat: bool, default=False
        Write the sequence in v1.4.1 compatible file format.

    Returns
    -------
    stream : bytes
        Bytes string corresponding to object.
    signature : str
        If create_signature is True, it returns the written .seq file's signature as a string.
        Note that, if remove_duplicates is True, signature belongs to the
        deduplicated sequences signature, 
        and not the Sequence that is stored in the Sequence object.
    
    """
    stream = io.StringIO()
    
    # Check if there are any timing errors in the sequence
    if check_timing:
        is_ok, error_report = seq.check_timing()
        if not is_ok:
            warn(f'write(): {len(error_report)} timing errors found in the sequence', stacklevel=2)

    # Calculate sequence duration and stored it in the TotalDuration definition
    seq.set_definition('TotalDuration', sum(seq.block_durations.values()))

    # Check whether all gradients in the last block are ramped down properly
    last_block_id = next(reversed(seq.block_events))
    last_block = seq.get_block(last_block_id)
    for channel, event in zip(('x', 'y', 'z'), (last_block.gx, last_block.gy, last_block.gz), strict=False):
        if (
            event is not None
            and event.type == 'grad'
            and abs(event.last) > seq.system.max_slew * seq.system.grad_raster_time
        ):
            warn_msg = f'write(): Gradient on channel {channel} in last sequence block does not ramp down to 0'

            if trace_enabled():
                trace = seq.block_trace.get(last_block_id, None)

                if hasattr(trace, 'block'):
                    warn_msg += '\nLast block defined here:\n' + format_trace(trace.block)
                if hasattr(trace, 'g' + channel):
                    warn_msg += f'\n`g{channel}` defined here:\n' + format_trace(getattr(trace, 'g' + channel))

            warn(warn_msg, stacklevel=2)

    # Write the sequence
    stream, signature = _write_to_stream(seq, stream, create_signature, remove_duplicates)

    # Return the sequence md5 signature if requested
    if signature is not None:
        seq.signature_type = 'md5'
        seq.signature_file = 'text'
        seq.signature_value = signature
        return stream, signature
    else:
        return stream
        
# %% Internal helpers
def _write_to_stream(
    seq: pp.Sequence, 
    stream: io.StringIO, 
    create_signature: bool, 
    remove_duplicates: bool,
) -> bytes | tuple[bytes, str]:
    """Serialization routine mirroring Sequence.write()."""
    # If removing duplicates, make a copy of the sequence with the duplicate
    # events removed.
    if remove_duplicates:
        seq = seq.remove_duplicates()

    # Re-define stream as pseudo output_file
    output_file = stream
    
    # Re-define seq as self
    self = seq

    # Write to stream
    output_file.write('# Pulseq sequence file\n')
    output_file.write('# Created by PyPulseq\n\n')

    output_file.write('[VERSION]\n')
    output_file.write(f'major {self.version_major}\n')
    output_file.write(f'minor {self.version_minor}\n')
    output_file.write(f'revision {self.version_revision}\n')
    output_file.write('\n')

    if len(self.definitions) != 0:
        output_file.write('[DEFINITIONS]\n')
        keys = sorted(self.definitions.keys())
        values = [self.definitions[k] for k in keys]
        for block_counter in range(len(keys)):
            output_file.write(f'{keys[block_counter]} ')
            if isinstance(values[block_counter], str):
                output_file.write(values[block_counter] + ' ')
            elif isinstance(values[block_counter], (int, float)):
                output_file.write(f'{values[block_counter]:0.9g} ')
            elif isinstance(values[block_counter], (list, tuple, np.ndarray)):  # e.g. [FOV_x, FOV_y, FOV_z]
                for i in range(len(values[block_counter])):
                    if isinstance(values[block_counter][i], (int, float)):
                        output_file.write(f'{values[block_counter][i]:0.9g} ')
                    else:
                        output_file.write(f'{values[block_counter][i]} ')
            else:
                raise RuntimeError('Unsupported definition')
            output_file.write('\n')
        output_file.write('\n')

    output_file.write('# Format of blocks:\n')
    output_file.write('# NUM DUR RF  GX  GY  GZ  ADC  EXT\n')
    output_file.write('[BLOCKS]\n')
    id_format_width = '{:' + str(len(str(len(self.block_events)))) + 'd}'
    id_format_str = id_format_width + ' {:3d} {:3d} {:3d} {:3d} {:3d} {:2d} {:2d}\n'
    for block_counter in self.block_events:
        block_duration = self.block_durations[block_counter] / self.block_duration_raster
        block_duration_rounded = round(block_duration)

        if abs(block_duration_rounded - block_duration) >= 1e-6:
            raise ValueError('Inconsistent block duration after rounding')

        s = id_format_str.format(
            *(
                block_counter,
                block_duration_rounded,
                *self.block_events[block_counter][1:],
            )
        )
        output_file.write(s)
    output_file.write('\n')

    if len(self.rf_library.data) != 0:
        output_file.write('# Format of RF events:\n')
        output_file.write('# id ampl. mag_id phase_id time_shape_id center delay freqPPm phasePPM freq phase use\n')
        output_file.write('# ..   Hz      ..       ..            ..     us    us     ppm  rad/MHz   Hz   rad  ..\n')
        output_file.write(f'# Field "use" is the initial of: {" ".join(get_supported_rf_uses()).strip()}\n')
        output_file.write('[RF]\n')
        id_format_str = (
            '{:.0f} {:12g} {:.0f} {:.0f} {:.0f} {:g} {:g} {:g} {:g} {:g} {:g} {:s}\n'  # Refer lines 20-21
        )
        for k in self.rf_library.data:
            lib_data1 = self.rf_library.data[k][0:4]
            lib_data2 = self.rf_library.data[k][6:10]
            center = self.rf_library.data[k][4] * 1e6  # us
            delay = round(self.rf_library.data[k][5] / self.rf_raster_time) * self.rf_raster_time * 1e6
            s = id_format_str.format(k, *lib_data1, center, delay, *lib_data2, self.rf_library.type[k])
            output_file.write(s)
        output_file.write('\n')

    grad_lib_values = np.array(list(self.grad_library.type.values()))
    arb_grad_mask = grad_lib_values == 'g' if self.grad_library.type else False
    trap_grad_mask = grad_lib_values == 't' if self.grad_library.type else False

    if np.any(arb_grad_mask):
        output_file.write('# Format of arbitrary gradients:\n')
        output_file.write(
            '#   time_shape_id of 0 means default timing (stepping with grad_raster starting at 1/2 of grad_raster)\n'
        )
        output_file.write('# id amplitude first last amp_shape_id time_shape_id delay\n')
        output_file.write('# ..      Hz/m  Hz/m Hz/m        ..         ..          us\n')
        output_file.write('[GRADIENTS]\n')
        id_format_str = '{:.0f} {:12g} {:12g} {:12g} {:.0f} {:.0f} {:.0f}\n'  # Refer lines 20-21
        keys = np.array(list(self.grad_library.data.keys()))
        for k in keys[arb_grad_mask]:
            s = id_format_str.format(
                k,
                *self.grad_library.data[k][:5],
                round(self.grad_library.data[k][5] * 1e6),
            )
            output_file.write(s)
        output_file.write('\n')

    if np.any(trap_grad_mask):
        output_file.write('# Format of trapezoid gradients:\n')
        output_file.write('# id amplitude rise flat fall delay\n')
        output_file.write('# ..      Hz/m   us   us   us    us\n')
        output_file.write('[TRAP]\n')
        keys = np.array(list(self.grad_library.data.keys()))
        id_format_str = '{:2.0f} {:12g} {:3.0f} {:4.0f} {:3.0f} {:3.0f}\n'
        for k in keys[trap_grad_mask]:
            data = np.copy(self.grad_library.data[k])  # Make a copy to leave the original untouched
            data[1:] = np.round(1e6 * data[1:])
            """
            Python & Numpy always round to nearest even value - inconsistent with MATLAB Pulseq's .seq files.
            [1] https://stackoverflow.com/questions/29671945/format-string-rounding-inconsistent
            [2] https://stackoverflow.com/questions/50374779/how-to-avoid-incorrect-rounding-with-numpy-round
            """
            s = id_format_str.format(k, *data)
            output_file.write(s)
        output_file.write('\n')

    if len(self.adc_library.data) != 0:
        output_file.write('# Format of ADC events:\n')
        output_file.write('# id num dwell delay freqPPM phasePPM freq phase phase_id\n')
        output_file.write('# ..  ..    ns    us     ppm  rad/MHz   Hz   rad       ..\n')
        output_file.write('[ADC]\n')
        id_format_str = '{:.0f} {:.0f} {:.0f} {:.0f} {:g} {:g} {:g} {:g} {:.0f}\n'  # Refer lines 20-21
        for k in self.adc_library.data:
            data = np.multiply(self.adc_library.data[k][0:8], [1, 1e9, 1e6, 1, 1, 1, 1, 1])
            s = id_format_str.format(k, *data)
            output_file.write(s)
        output_file.write('\n')

    if len(self.extensions_library.data) != 0:
        output_file.write('# Format of extension lists:\n')
        output_file.write('# id type ref next_id\n')
        output_file.write('# next_id of 0 terminates the list\n')
        output_file.write('# Extension list is followed by extension specifications\n')
        output_file.write('[EXTENSIONS]\n')
        id_format_str = '{:.0f} {:.0f} {:.0f} {:.0f}\n'  # Refer lines 20-21
        for k in self.extensions_library.data:
            s = id_format_str.format(k, *np.round(self.extensions_library.data[k]))
            output_file.write(s)
        output_file.write('\n')

    if len(self.trigger_library.data) != 0:
        output_file.write('# Extension specification for digital output and input triggers:\n')
        output_file.write('# id type channel delay (us) duration (us)\n')
        output_file.write(f'extension TRIGGERS {self.get_extension_type_ID("TRIGGERS")}\n')
        id_format_str = '{:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'  # Refer lines 20-21
        for k in self.trigger_library.data:
            s = id_format_str.format(k, *np.round(self.trigger_library.data[k] * np.array([1, 1, 1e6, 1e6])))
            output_file.write(s)
        output_file.write('\n')

    if len(self.label_set_library.data) != 0:
        labels = get_supported_labels()

        output_file.write('# Extension specification for setting labels:\n')
        output_file.write('# id set labelstring\n')
        tid = self.get_extension_type_ID('LABELSET')
        output_file.write(f'extension LABELSET {tid}\n')
        id_format_str = '{:.0f} {:.0f} {}\n'  # Refer lines 20-21
        for k in self.label_set_library.data:
            value = self.label_set_library.data[k][0]
            label_id = labels[int(self.label_set_library.data[k][1]) - 1]  # label_id is +1 in add_block()
            s = id_format_str.format(k, value, label_id)
            output_file.write(s)
        output_file.write('\n')

    if len(self.label_inc_library.data) != 0:
        labels = get_supported_labels()

        output_file.write('# Extension specification for setting labels:\n')
        output_file.write('# id set labelstring\n')
        tid = self.get_extension_type_ID('LABELINC')
        output_file.write(f'extension LABELINC {tid}\n')
        id_format_str = '{:.0f} {:.0f} {}\n'  # See comment at the beginning of this method definition
        for k in self.label_inc_library.data:
            value = self.label_inc_library.data[k][0]
            label_id = labels[self.label_inc_library.data[k][1] - 1]  # label_id is +1 in add_block()
            s = id_format_str.format(k, value, label_id)
            output_file.write(s)
        output_file.write('\n')

    if len(self.soft_delay_library.data) != 0:
        output_file.write('# Extension specification for soft delays:\n')
        output_file.write('# id num offset factor hint\n')
        output_file.write('# ..  ..     us     ..   ..\n')

        tid = self.get_extension_type_ID('DELAYS')
        output_file.write(f'extension DELAYS {tid}\n')
        id_format_str = '{:.0f} {:.0f} {:.0f} {:.0f} {}\n'

        for k in self.soft_delay_library.data:
            data = self.soft_delay_library.data[k]
            s = id_format_str.format(k, data[0], np.round(data[1] * 1e6), data[2], data[3])
            output_file.write(s)
        output_file.write('\n')

    if len(self.rf_shim_library.data) != 0:
        output_file.write('# Extension specification for RF shimming:\n')
        output_file.write('# id num_chan factor magn_c1 phase_c1 magn_c2 phase_c2 ...\n')
        output_file.write(f'extension RF_SHIMS {self.get_extension_type_ID("RF_SHIMS")}\n')

        for k in self.rf_shim_library.data:
            shim_vector_length = len(self.rf_shim_library.data[k])
            id_format_str = '{:d} {:d}' + ''.join(' {:g}' for _ in range(shim_vector_length)) + '\n'
            s = id_format_str.format(k, int(0.5 * shim_vector_length), *self.rf_shim_library.data[k])
            output_file.write(s)
        output_file.write('\n')

    if len(self.rotation_library.data) != 0:
        output_file.write('# Extension specification for rotation events:\n')
        output_file.write('# id RotQuat0 RotQuatX RotQuatY RotQuatZ\n')
        output_file.write(f'extension ROTATIONS {self.get_extension_type_ID("ROTATIONS")}\n')
        id_format_str = '{:.0f} {:12g} {:12g} {:12g} {:12g}\n'  # Refer lines 20-21
        for k in self.rotation_library.data:
            s = id_format_str.format(k, *self.rotation_library.data[k])
            output_file.write(s)
        output_file.write('\n')

    if len(self.shape_library.data) != 0:
        output_file.write('# Sequence Shapes\n')
        output_file.write('[SHAPES]\n\n')
        for k in self.shape_library.data:
            shape_data = self.shape_library.data[k]
            s = 'shape_id {:.0f}\n'.format(k)
            output_file.write(s)
            s = 'num_samples {:.0f}\n'.format(shape_data[0])
            output_file.write(s)
            s = ('{:.9g}\n' * len(shape_data[1:])).format(*shape_data[1:])
            output_file.write(s)
            output_file.write('\n')

    if create_signature:  # Sign the file
        # Calculate digest
        buffer = output_file.getvalue()
        md5 = hashlib.md5(buffer.encode('utf-8')).hexdigest()

        # Write signature
        output_file.write('\n[SIGNATURE]\n')
        output_file.write(
            '# This is the hash of the Pulseq file, calculated right before the [SIGNATURE] section was added\n'
        )
        output_file.write(
            '# It can be reproduced/verified with md5sum if the file trimmed to the position right above [SIGNATURE]\n'
        )
        output_file.write(
            '# The new line character preceding [SIGNATURE] BELONGS to the signature (and needs to be stripped away for '
            'recalculating/verification)\n'
        )
        output_file.write('Type md5\n')
        output_file.write(f'Hash {md5}\n')

        return output_file.getvalue().encode('utf-8'), md5
    
    return output_file.getvalue().encode('utf-8')