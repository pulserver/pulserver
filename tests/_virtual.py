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
