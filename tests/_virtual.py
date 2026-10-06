"""The prescriptions, phantoms and designed timing the virtual scanner is tested with."""

import numpy as np
from pypulseqpp.sequences.preparation.fatsat import FAT_SHIFT_PPM
from scipy.spatial.transform import Rotation

from pulserver import virtual

#: A field-of-view offset along the logical axes, in metres.
OFFSET = np.array([0.02, -0.012, 0.0])
#: A frequency offset of every spin from the centre frequency, in Hz.
OFF_RESONANCE_HZ = 60.0
#: A prescription turned 20 degrees in plane about z, then tilted 30 degrees about x.
OBLIQUE = Rotation.from_euler("zx", [20.0, 30.0], degrees=True).as_matrix()
#: The oblique prescription with its phase-encoding axis reversed: a reflection.
REFLECTED = OBLIQUE @ np.diag([1.0, -1.0, 1.0])
#: The prescriptions a scan is repeated under, with their test ids.
ORIENTATIONS = {"axial": np.eye(3), "oblique": OBLIQUE, "reflected": REFLECTED}


def phantom(position=(0.0, 0.0, 0.0), rotation=None, coils=2):
    """An object without a mirror symmetry, with its origin at ``position`` and its axes turned by ``rotation``."""
    return virtual.Phantom(
        [
            virtual.Ellipse((0.0, 0.0, 0.0), (0.08, 0.06), 0.3),
            virtual.Ellipse((0.03, 0.02, 0.0), (0.02, 0.015), 0.0, 0.5),
        ],
        coils=coils,
        rotation=rotation,
        position=position,
    )


def posed(rotation, coils=2):
    """The phantom posed where a prescription of ``OFFSET`` and ``rotation`` places the field of view."""
    return phantom(rotation @ OFFSET, rotation, coils)


def water_and_fat(coils=2):
    """A water ellipse with a fat one inside it, at the main fat resonance."""
    return virtual.Phantom(
        [
            virtual.Ellipse((0.0, 0.0, 0.0), (0.08, 0.06), 0.3),
            virtual.Ellipse((0.03, 0.02, 0.0), (0.03, 0.02), 0.0, 1.0, FAT_SHIFT_PPM),
        ],
        coils=coils,
    )


def precession(sequence):
    """Seconds of free precession each ADC sample of a file accrues, as its design times its RF pulses.

    Counted from the centre of the last excitation, and negated about the
    centre of each refocusing pulse after it; NaN before the first excitation.
    """
    _, _, excitations, refocusings, sampled = sequence.calculate_kspace()
    origin = np.full(sampled.shape, np.nan)
    events = sorted(
        [(t, True) for t in excitations] + [(t, False) for t in refocusings]
    )
    for time, excites in events:
        after = sampled > time
        origin[after] = time if excites else 2.0 * time - origin[after]
    return sampled - origin


def synthetic_sensitivities(model, channels):
    """Smooth sensitivities in place of BART's ``model``, as ``_coils._sampled`` returns them: each channel stronger to one side and turning in phase along x."""
    from pulserver.virtual import _coils

    index = (np.arange(_coils._SAMPLES) - _coils._SAMPLES // 2) / _coils._SAMPLES
    z = index[:, None, None] if model == "HEAD_3D_64CH" else np.zeros((1, 1, 1))
    y, x = index[None, :, None], index[None, None, :]
    maps = []
    for channel in range(channels):
        angle = 2.0 * np.pi * channel / channels
        magnitude = 1.5 + np.cos(angle) * x + np.sin(angle) * y + 0.2 * z
        maps.append(magnitude * np.exp(1j * (angle + 3.0 * x)))
    return np.asarray(maps, dtype=np.complex64)


#: Channels of each coil's field maps :func:`write_fields` writes.
FIELD_CHANNELS = {"body": 2, "head8": 8, "head32": 32, "head48": 48}
#: Voxels along x, y and z of those maps, their pitch in m, and the voxel
#: centred on the isocentre.
FIELD_SHAPE, FIELD_RESOLUTION, FIELD_ISOCENTRE = (7, 8, 5), 0.02, (3, 4, 2)


def write_fields(directory, field_t=3.0):
    """Write each coil's field maps as mariepy's ``maps.write`` does, and VOPs for the two that transmit.

    Each channel's ``plus`` and ``minus`` vary smoothly in magnitude and turn
    in phase along x, from a phase of their own; the voxels of one edge of the
    box lie outside the body.
    """
    import json

    import pypulseqpp as pp

    generator = np.random.default_rng(3)
    grid = np.indices(FIELD_SHAPE, dtype=float)
    mask = np.ones(FIELD_SHAPE, dtype=bool)
    mask[0, 0, :] = False
    for name, channels in FIELD_CHANNELS.items():
        components = {}
        for component in ("plus", "minus"):
            slope = generator.uniform(-1.0, 1.0, (channels, 3))
            phase = generator.uniform(-np.pi, np.pi, (channels, 1, 1, 1))
            magnitude = 1.0 + 0.05 * np.einsum("ca,axyz->cxyz", slope, grid)
            field = 1e-6 * magnitude * np.exp(1j * (phase + 0.3 * grid[0]))
            components[component] = np.where(mask, field, 0.0).astype(np.complex64)
        metadata = {
            "coil": name,
            "channels": [f"{name}-{index + 1}" for index in range(channels)],
            "frequency_hz": pp.Opts().gamma * field_t,
            "drive_unit": "1 A",
            "origin": [-FIELD_RESOLUTION * index for index in FIELD_ISOCENTRE],
            "resolution": FIELD_RESOLUTION,
            "frame": "the physical frame",
            "bodies": ["a box"],
            "mariepy_version": "0",
            "data_licence": "none",
        }
        np.savez_compressed(
            directory / f"{name}.npz",
            **components,
            mask=mask,
            metadata=np.array(json.dumps(metadata)),
        )
        if name in ("body", "head8"):
            vectors = generator.standard_normal((2, 4, channels, channels))
            points = vectors[0] + 1j * vectors[1]
            vops = points @ np.conj(points).transpose(0, 2, 1)
            np.savez_compressed(
                directory / f"{name}_vops.npz",
                vops=vops,
                global_matrix=vops.mean(axis=0, keepdims=True),
            )
    return directory
