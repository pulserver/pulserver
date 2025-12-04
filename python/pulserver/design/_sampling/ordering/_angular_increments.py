"""Angular increment generators for MR trajectory ordering. 

This module provides various angular increment schemes commonly used
in MRI acquisitions, including golden angle variants and linear spacing. 

References
----------
..  [1] Winkelmann S, et al. "An optimal radial profile order based on the
       Golden Ratio for time-resolved MRI." IEEE TMI 2007;26(1):68-76.
..  [2] Wundrak S, et al. "Golden ratio sparse MRI using tiny golden angles."
       MRM 2016;75(6):2372-2378.
..  [3] Chan RW, et al. "Temporal stability of adaptive 3D radial MRI using
       multidimensional golden means." MRM 2009;61(2):354-363.
"""

__all__ = [
    "AngularIncrement",
    "LinearIncrement",
    "GoldenAngle",
    "TinyGoldenAngle",
    "GoldenMeans2D",
    "GoldenMeans3D",
    "RationalGoldenAngle",
    "GOLDEN_RATIO",
    "GOLDEN_ANGLE",
]

from abc import ABC, abstractmethod
import math

import numpy as np
from numpy.typing import NDArray


GOLDEN_RATIO: float = (1 + math.sqrt(5)) / 2  # φ ≈ 1.618033988749895
GOLDEN_ANGLE: float = math.pi / GOLDEN_RATIO  # π/φ ≈ 111.246° in radians


class AngularIncrement(ABC):
    """
    Abstract base class for angular increment generators. 

    Subclasses must implement `get_angles` which returns the angular
    positions for a given number of spokes/projections.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return a descriptive name for this increment scheme."""
        pass

    @abstractmethod
    def get_angles(self, n: int, start: float = 0.0) -> NDArray[np.floating]:
        """
        Generate angular positions for n spokes. 

        Parameters
        ----------
        n : int
            Number of angular positions to generate. 
        start : float
            Starting angle in radians. Default is 0. 

        Returns
        -------
        angles : NDArray[np.floating]
            Array of n angular positions in radians.
        """
        pass

    def get_increment(self) -> float | NDArray[np.floating]:
        """
        Return the angular increment(s) between consecutive spokes.

        Returns
        -------
        increment : float | NDArray
            Constant increment (float) or array of increments. 
            For variable increment schemes, returns the increment array.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support get_increment().  "
            "Use get_angles() instead."
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class LinearIncrement(AngularIncrement):
    """
    Uniform linear angular spacing.

    Divides the angular range evenly among n spokes. 

    Parameters
    ----------
    angular_range : float
        Total angular range to cover in radians.  Default is π (half-circle,
        appropriate for center-out spokes that cover both directions). 

    Examples
    --------
    >>> inc = LinearIncrement(angular_range=np.pi)
    >>> angles = inc.get_angles(4)  # [0, π/4, π/2, 3π/4]
    """

    def __init__(self, angular_range: float = math.pi):
        self._angular_range = angular_range

    @property
    def name(self) -> str:
        return "linear"

    @property
    def angular_range(self) -> float:
        """Return the total angular range."""
        return self._angular_range

    def get_angles(self, n: int, start: float = 0.0) -> NDArray[np.floating]:
        if n <= 0:
            return np.array([], dtype=np.float64)
        if n == 1:
            return np. array([start], dtype=np.float64)
        increment = self._angular_range / n
        return start + np.arange(n, dtype=np.float64) * increment

    def get_increment(self) -> float:
        raise ValueError(
            "LinearIncrement increment depends on n.  Use get_angles() directly "
            "or compute as angular_range / n."
        )

    def __repr__(self) -> str:
        return f"LinearIncrement(angular_range={self._angular_range})"


class GoldenAngle(AngularIncrement):
    """
    Standard golden angle increment (~111.246°). 

    The golden angle provides optimal uniform coverage properties and
    allows flexible retrospective binning of data.

    Parameters
    ----------
    full_circle : bool
        If True, use 2π/φ (~222.5°) for full spoke acquisitions.
        If False (default), use π/φ (~111. 246°) for half-spoke/center-out. 

    References
    ----------
    .. [1] Winkelmann S, et al. IEEE TMI 2007;26(1):68-76. 

    Examples
    --------
    >>> inc = GoldenAngle()
    >>> angles = inc.get_angles(5)  # 5 spokes with golden angle spacing
    """

    def __init__(self, full_circle: bool = False):
        self._full_circle = full_circle
        self._increment = (2 * math.pi / GOLDEN_RATIO) if full_circle else GOLDEN_ANGLE

    @property
    def name(self) -> str:
        return "golden_angle" + ("_full" if self._full_circle else "")

    @property
    def full_circle(self) -> bool:
        """Return whether using full circle (2π) or half circle (π) base."""
        return self._full_circle

    def get_angles(self, n: int, start: float = 0.0) -> NDArray[np. floating]:
        if n <= 0:
            return np. array([], dtype=np.float64)
        indices = np.arange(n, dtype=np.float64)
        return start + indices * self._increment

    def get_increment(self) -> float:
        return self._increment

    def __repr__(self) -> str:
        return f"GoldenAngle(full_circle={self._full_circle})"


class TinyGoldenAngle(AngularIncrement):
    """
    Tiny golden angle for smoother angular progression.

    Uses smaller golden angle increments based on continued fraction
    approximations, providing smoother transitions between adjacent
    spokes while maintaining good coverage properties.

    Parameters
    ----------
    order : int
        Order of the tiny golden angle (1, 2, 3, ...). 
        Higher orders give smaller increments. 
        Order 1 is the standard golden angle. 
    full_circle : bool
        If True, base on 2π; if False (default), base on π.

    References
    ----------
    ..  [1] Wundrak S, et al. MRM 2016;75(6):2372-2378. 

    Examples
    --------
    >>> inc = TinyGoldenAngle(order=7)  # τ_7 ≈ 23.6°
    >>> angles = inc.get_angles(100)
    """

    def __init__(self, order: int = 7, full_circle: bool = False):
        if order < 1:
            raise ValueError(f"Order must be >= 1, got {order}")
        self._order = order
        self._full_circle = full_circle
        self._increment = self._compute_tiny_golden_angle(order, full_circle)

    @property
    def name(self) -> str:
        return f"tiny_golden_angle_{self._order}"

    @property
    def order(self) -> int:
        """Return the tiny golden angle order."""
        return self._order

    @staticmethod
    def _compute_tiny_golden_angle(order: int, full_circle: bool) -> float:
        """
        Compute tiny golden angle of given order.

        The tiny golden angle τ_n is computed using Fibonacci numbers:
        τ_n = π / (F_n + F_{n-2}/F_n * F_{n-1})

        Simplified: τ_n = π * F_{n-1} / F_{n+1} where F_n is Fibonacci. 
        """
        # Generate Fibonacci numbers up to order + 2
        fib = [1, 1]
        for _ in range(order + 1):
            fib.append(fib[-1] + fib[-2])

        # τ_order = π / (φ^order) ≈ π * F_{order-1} / F_{order+1}
        base = 2 * math.pi if full_circle else math.pi
        return base * fib[order - 1] / fib[order + 1]

    def get_angles(self, n: int, start: float = 0.0) -> NDArray[np. floating]:
        if n <= 0:
            return np. array([], dtype=np.float64)
        indices = np.arange(n, dtype=np. float64)
        return start + indices * self._increment

    def get_increment(self) -> float:
        return self._increment

    def __repr__(self) -> str:
        return f"TinyGoldenAngle(order={self._order}, full_circle={self._full_circle})"


class GoldenMeans2D(AngularIncrement):
    """
    2D golden means for stack-of-stars or similar trajectories.

    Uses the 2D golden mean to provide optimal coverage in 2D angular
    space (e.g., azimuthal angle for stack-of-stars). 

    The 2D golden mean is the real root of x³ = x + 1, approximately 1.32472.

    Parameters
    ----------
    full_circle : bool
        If True, base on 2π; if False (default), base on π. 

    References
    ----------
    .. [1] Chan RW, et al. MRM 2009;61(2):354-363. 
    """

    # 2D golden mean: real root of x³ = x + 1
    GOLDEN_MEAN_2D: float = 1.3247179572447458

    def __init__(self, full_circle: bool = False):
        self._full_circle = full_circle
        base = 2 * math.pi if full_circle else math. pi
        self._increment = base / self.GOLDEN_MEAN_2D

    @property
    def name(self) -> str:
        return "golden_means_2d"

    def get_angles(self, n: int, start: float = 0.0) -> NDArray[np. floating]:
        if n <= 0:
            return np. array([], dtype=np.float64)
        indices = np.arange(n, dtype=np. float64)
        return start + indices * self._increment

    def get_increment(self) -> float:
        return self._increment

    def __repr__(self) -> str:
        return f"GoldenMeans2D(full_circle={self._full_circle})"


class GoldenMeans3D(AngularIncrement):
    """
    3D golden means for koosh-ball or 3D radial trajectories.

    Provides two angular increments for 3D coverage (azimuthal and polar).
    The primary angle returned by get_angles is the azimuthal increment.

    The 3D golden means are based on the real root of x⁴ = x + 1. 

    Parameters
    ----------
    full_circle : bool
        If True, base on 2π; if False (default), base on π.

    References
    ----------
    .. [1] Chan RW, et al. MRM 2009;61(2):354-363. 
    """

    # 3D golden mean: real root of x⁴ = x + 1
    GOLDEN_MEAN_3D: float = 1.2207440846057596

    def __init__(self, full_circle: bool = False):
        self._full_circle = full_circle
        base = 2 * math.pi if full_circle else math. pi
        self._increment_azimuthal = base / self. GOLDEN_MEAN_3D
        self._increment_polar = base / (self.GOLDEN_MEAN_3D ** 2)

    @property
    def name(self) -> str:
        return "golden_means_3d"

    @property
    def increment_polar(self) -> float:
        """Return the polar angle increment."""
        return self._increment_polar

    def get_angles(self, n: int, start: float = 0.0) -> NDArray[np.floating]:
        """Get azimuthal angles.  Use get_angles_3d for both angles."""
        if n <= 0:
            return np.array([], dtype=np.float64)
        indices = np. arange(n, dtype=np.float64)
        return start + indices * self._increment_azimuthal

    def get_angles_3d(
        self, n: int, start_azimuthal: float = 0.0, start_polar: float = 0.0
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Get both azimuthal and polar angles.

        Returns
        -------
        azimuthal : NDArray
            Azimuthal angles in radians.
        polar : NDArray
            Polar angles in radians.
        """
        if n <= 0:
            empty = np.array([], dtype=np.float64)
            return empty, empty
        indices = np.arange(n, dtype=np.float64)
        azimuthal = start_azimuthal + indices * self._increment_azimuthal
        polar = start_polar + indices * self._increment_polar
        return azimuthal, polar

    def get_increment(self) -> float:
        """Return azimuthal increment.  Use increment_polar for polar."""
        return self._increment_azimuthal

    def __repr__(self) -> str:
        return f"GoldenMeans3D(full_circle={self._full_circle})"


class RationalGoldenAngle(AngularIncrement):
    """
    Rational approximation of golden angle using Fibonacci numbers.

    Provides exact uniform coverage after a specific number of spokes,
    useful when a fixed number of spokes is known a priori.

    The angle is π * F_n / F_{n+2} where F_n is the nth Fibonacci number. 

    Parameters
    ----------
    fibonacci_index : int
        Index into Fibonacci sequence (>= 2).  Common values:
        - 5: 5/13 * π ≈ 69.2°, uniform after 13 spokes
        - 6: 8/21 * π ≈ 68.6°, uniform after 21 spokes
        - 7: 13/34 * π ≈ 68.8°, uniform after 34 spokes
    full_circle : bool
        If True, use 2π base; if False (default), use π base. 

    Examples
    --------
    >>> inc = RationalGoldenAngle(fibonacci_index=7)
    >>> angles = inc.get_angles(34)  # Exactly uniform coverage
    """

    # Pre-computed Fibonacci numbers
    _FIBONACCI: tuple[int, ... ] = (
        1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987
    )

    def __init__(self, fibonacci_index: int = 7, full_circle: bool = False):
        if fibonacci_index < 2:
            raise ValueError(f"fibonacci_index must be >= 2, got {fibonacci_index}")
        if fibonacci_index >= len(self._FIBONACCI) - 2:
            raise ValueError(
                f"fibonacci_index must be < {len(self._FIBONACCI) - 2}, "
                f"got {fibonacci_index}"
            )

        self._fibonacci_index = fibonacci_index
        self._full_circle = full_circle

        f_n = self._FIBONACCI[fibonacci_index]
        f_n2 = self._FIBONACCI[fibonacci_index + 2]
        base = 2 * math.pi if full_circle else math. pi
        self._increment = base * f_n / f_n2
        self._uniform_at = f_n2

    @property
    def name(self) -> str:
        return f"rational_golden_{self._fibonacci_index}"

    @property
    def fibonacci_index(self) -> int:
        """Return the Fibonacci index used."""
        return self._fibonacci_index

    @property
    def uniform_at(self) -> int:
        """Return the number of spokes for exact uniform coverage."""
        return self._uniform_at

    def get_angles(self, n: int, start: float = 0.0) -> NDArray[np.floating]:
        if n <= 0:
            return np.array([], dtype=np.float64)
        indices = np. arange(n, dtype=np.float64)
        return start + indices * self._increment

    def get_increment(self) -> float:
        return self._increment

    def __repr__(self) -> str:
        return (
            f"RationalGoldenAngle(fibonacci_index={self._fibonacci_index}, "
            f"full_circle={self._full_circle})"
        )