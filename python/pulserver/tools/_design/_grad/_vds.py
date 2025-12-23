__all__ = [
    'fov_constant',
    'fov_dual_density',
    'fov_triple_density',
    'fov_variable',
    'make_spiral_grad',
]

import inspect
from typing import Callable

import numpy as np


def make_spiral_grad():
    """
    Placeholder for spiral gradient design wrapper around `calc_vds_raw`.

    Implement this to call `calc_vds_raw` with appropriate arguments
    (slewmax, gradmax, Tgsample, Tdsample, Ninterleaves, fov, etc.)
    and return gradients in the desired units.
    """
    raise NotImplementedError


def fov_constant(F_const_m: float) -> np.ndarray:
    """
    Generate FOV coefficients for a constant-density spiral:
    FOV(kr) = F_const (constant) for all k-space radii.

    The input FOV is specified in meters; it is converted to centimeters
    internally because the C code expects FOV in cm.

    Parameters
    ----------
    F_const_m : float
        Constant field of view in meters.

    Returns
    -------
    fov : numpy.ndarray
        1D array of coefficients [c0], such that:
        FOV_cm(kr) = c0  (in cm), for all kr in 1/cm.

    Notes
    -----
    The C code expects a polynomial FOV(kr) of the form:
        FOV_cm(kr) = sum_{n=0}^{N} fov[n] * kr^n
    where kr is in 1/cm and FOV in cm. For a constant FOV, only the
    c0 term is nonzero.

    Examples
    --------
    >>> fov = fov_constant(0.24)  # 0.24 m = 24 cm
    >>> fov
    array([24.])
    """
    F_const_cm = float(F_const_m) * 100.0
    return np.array([F_const_cm], dtype=float)


def fov_variable(
    rmax: float,
    F_inner_m: float,
    undersampling_factor_outer: float,
    degree: int = 6,
    num_samples: int = 200,
    F_target_m: Callable | None = None,
) -> np.ndarray:
    """
    Fit a polynomial FOV(kr) to a target FOV profile, with kr in 1/cm.

    The target FOV profile is specified in meters and is converted
    internally to centimeters. The resulting coefficients are in cm,
    suitable for use in the C code, which expects:
        FOV_cm(kr) = sum_{n=0}^degree c_n * kr^n
    with kr in 1/cm and FOV in cm.

    If `F_target_m` is None, a default linear profile in meters is used
    based on an undersampling factor:

        - F_inner_m: center FOV in meters.
        - undersampling_factor_outer: ratio of outer FOV to inner FOV.
          For example, undersampling_factor_outer = 2.0 gives
          F_outer_m = 2 * F_inner_m.

        Default target when F_target_m is None:
            F_outer_m = undersampling_factor_outer * F_inner_m
            FOV_m(kr) = F_inner_m + (F_outer_m - F_inner_m) * (kr / rmax)

    If `F_target_m` is provided, it may have one of two signatures:

        1. F_target_m(kr) -> FOV_m
        2. F_target_m(kr, F_inner_m, undersampling_factor_outer) -> FOV_m

    allowing the user to optionally make use of the provided initial FOV
    and outer undersampling factor.

    Parameters
    ----------
    rmax : float
        Maximum k-space radius in 1/cm (same as krmax in the C code).
    F_inner_m : float
        Inner FOV in meters used for the default target shape.
    undersampling_factor_outer : float
        F_outer_m / F_inner_m for the default target FOV. Must be >= 1.0.
    degree : int, optional
        Degree of the polynomial, by default 6.
    num_samples : int, optional
        Number of sample points used in [0, rmax] for the least-squares
        fit, by default 200.
    F_target_m : callable or None, optional
        Target FOV function in meters. If None, the default linear profile
        in kr is used. If not None, may be:

            F_target_m(kr)
        or
            F_target_m(kr, F_inner_m, undersampling_factor_outer)

        where kr is a numpy array of radii in 1/cm.

    Returns
    -------
    fov : numpy.ndarray
        1D array of coefficients [c0, c1, ..., c_degree] in centimeters,
        such that:
            FOV_cm(kr) ≈ sum_{n=0}^degree c_n * kr^n.

    Examples
    --------
    Default linear FOV in kr:

    >>> rmax = 5.0           # 1/cm
    >>> F_inner_m = 0.16     # 16 cm
    >>> u_outer = 2.0        # outer FOV is 2x inner
    >>> fov = fov_variable(rmax, F_inner_m, u_outer, degree=4)
    >>> fov.shape
    (5,)

    Custom target that uses the provided parameters:

    >>> def F_quad_m(kr, F_inner_m, u_outer):
    ...     F_outer_m = u_outer * F_inner_m
    ...     return F_inner_m + (F_outer_m - F_inner_m) * (kr / rmax)**2
    ...
    >>> fov2 = fov_variable(rmax, F_inner_m, u_outer,
    ...                     degree=6, F_target_m=F_quad_m)
    """
    if undersampling_factor_outer < 1.0:
        raise ValueError('undersampling_factor_outer must be >= 1.0.')

    # Sample kr in 1/cm
    kr = np.linspace(0.0, rmax, num_samples)

    if F_target_m is None:
        # Default: linear FOV in meters from F_inner_m to F_outer_m as a function of kr
        F_outer_m = undersampling_factor_outer * F_inner_m

        def F_target_m_default(kr_arr: np.ndarray) -> np.ndarray:
            return F_inner_m + (F_outer_m - F_inner_m) * (kr_arr / rmax)

        F_m = F_target_m_default(kr)
    else:
        # Allow F_target_m to optionally use F_inner_m and undersampling_factor_outer
        sig = inspect.signature(F_target_m)
        n_params = len(sig.parameters)

        if n_params == 1:
            F_m = F_target_m(kr)
        elif n_params >= 3:
            F_m = F_target_m(kr, F_inner_m, undersampling_factor_outer)
        else:
            raise TypeError(
                'F_target_m must accept either (kr) or '
                '(kr, F_inner_m, undersampling_factor_outer).'
            )

    # Convert meters -> centimeters
    F_cm = F_m * 100.0

    # Fit directly in kr (1/cm), which matches C usage
    p = np.polyfit(kr, F_cm, degree)
    fov = p[::-1]  # ascending powers: fov[n] is coeff of kr^n
    return fov


def fov_dual_density(
    rmax: float,
    r1: float,
    F_inner_m: float,
    F_outer_m: float,
    transition_width: float | None = None,
    degree: int = 10,
    num_samples: int = 400,
) -> np.ndarray:
    """
    Generate FOV coefficients for a dual-density spiral.

    The design approximates:
        FOV_m(kr) ≈ F_inner_m  (high density) for small kr
        FOV_m(kr) ≈ F_outer_m  (low density)  for large kr
    with a smooth cosine transition around radius r1.

    Input FOV values are in meters; they are converted internally to
    centimeters. The resulting polynomial coefficients are in cm.

    Parameters
    ----------
    rmax : float
        Maximum k-space radius in 1/cm (same as krmax in the C code).
    r1 : float
        Center of the transition region in 1/cm.
    F_inner_m : float
        Inner-region FOV in meters (smaller value => higher sampling density).
    F_outer_m : float
        Outer-region FOV in meters (larger value => lower sampling density).
    transition_width : float or None, optional
        Width of the transition band in 1/cm. If None, defaults to 0.2 * rmax.
    degree : int, optional
        Degree of the polynomial fit, by default 10. (Values around 8–14
        are typical for reasonably sharp transitions.)
    num_samples : int, optional
        Number of sample radii in [0, rmax] for fitting, by default 400.

    Returns
    -------
    fov : numpy.ndarray
        1D array of coefficients [c0, c1, ..., c_degree] in centimeters,
        suitable for use as `fov[]` in the C code.

    Notes
    -----
    The target FOV profile in meters is constructed as:
        F_inner_m                for kr <= r1 - w/2
        cosine-tapered blend     for r1 - w/2 < kr < r1 + w/2
        F_outer_m                for kr >= r1 + w/2
    where w is `transition_width`. This profile is converted to cm and
    polynomial-fitted over [0, rmax].

    Examples
    --------
    >>> rmax = 5.0        # 1/cm
    >>> r1 = 2.0          # 1/cm
    >>> F_inner_m = 0.16  # 16 cm
    >>> F_outer_m = 0.32  # 32 cm
    >>> fov = fov_dual_density(rmax, r1, F_inner_m, F_outer_m,
    ...                        transition_width=1.0, degree=13)
    >>> len(fov)
    14
    """
    if transition_width is None:
        transition_width = 0.2 * rmax

    kr = np.linspace(0.0, rmax, num_samples)

    w = transition_width
    ra = r1 - w / 2.0
    rb = r1 + w / 2.0

    F_m = np.empty_like(kr)

    for i, kri in enumerate(kr):
        if kri <= ra:
            F_m[i] = F_inner_m
        elif kri >= rb:
            F_m[i] = F_outer_m
        else:
            t = (kri - ra) / (rb - ra)  # 0 -> 1
            wcos = 0.5 * (1.0 - np.cos(np.pi * t))  # smooth step
            F_m[i] = F_inner_m + (F_outer_m - F_inner_m) * wcos

    # Convert to cm
    F_cm = F_m * 100.0

    p = np.polyfit(kr, F_cm, degree)
    fov = p[::-1]
    return fov


def fov_triple_density(
    rmax: float,
    r1: float,
    r2: float,
    F1_m: float,
    F2_m: float,
    F3_m: float,
    trans_width1: float | None = None,
    trans_width2: float | None = None,
    degree: int = 12,
    num_samples: int = 500,
) -> np.ndarray:
    """
    Generate FOV coefficients for a triple-density spiral.

    The design approximates three plateau regions in FOV(kr) (in meters):
        F1_m  for center region (high density)
        F2_m  for middle region
        F3_m  for outer region (lowest density)
    with smooth cosine transitions between them around radii r1 and r2.

    Input FOV values are in meters; they are converted to centimeters
    internally. The resulting polynomial coefficients are in cm.

    Parameters
    ----------
    rmax : float
        Maximum k-space radius in 1/cm.
    r1 : float
        Center radius of the first transition (between F1_m and F2_m), in 1/cm.
    r2 : float
        Center radius of the second transition (between F2_m and F3_m), in 1/cm.
        Must satisfy 0 < r1 < r2 <= rmax.
    F1_m : float
        Inner-region FOV in meters.
    F2_m : float
        Middle-region FOV in meters.
    F3_m : float
        Outer-region FOV in meters.
    trans_width1 : float or None, optional
        Width of the first transition band in 1/cm. If None, a heuristic
        value based on (r2 - 0) is used.
    trans_width2 : float or None, optional
        Width of the second transition band in 1/cm. If None, a heuristic
        value based on (rmax - r1) is used.
    degree : int, optional
        Degree of the polynomial fit, by default 12.
    num_samples : int, optional
        Number of sample radii in [0, rmax] for fitting, by default 500.

    Returns
    -------
    fov : numpy.ndarray
        1D array of coefficients [c0, c1, ..., c_degree] in centimeters,
        suitable for use as `fov[]` in the C code.

    Notes
    -----
    The target FOV profile in meters is:
        F1_m                        for kr <= r1 - w1/2
        cosine-taper F1_m -> F2_m   for r1 - w1/2 < kr < r1 + w1/2
        F2_m                        for r1 + w1/2 <= kr <= r2 - w2/2
        cosine-taper F2_m -> F3_m   for r2 - w2/2 < kr < r2 + w2/2
        F3_m                        for kr >= r2 + w2/2

    This profile is converted to cm and polynomial-fitted over [0, rmax].

    Examples
    --------
    >>> rmax = 5.0
    >>> r1, r2 = 1.5, 3.5
    >>> F1_m, F2_m, F3_m = 0.12, 0.20, 0.32  # 12 cm, 20 cm, 32 cm
    >>> fov = fov_triple_density(rmax, r1, r2, F1_m, F2_m, F3_m,
    ...                          trans_width1=0.5, trans_width2=0.7,
    ...                          degree=13)
    >>> len(fov)
    14
    """
    if trans_width1 is None:
        trans_width1 = 0.15 * (r2 - 0.0)
    if trans_width2 is None:
        trans_width2 = 0.15 * (rmax - r1)

    w1 = trans_width1
    w2 = trans_width2

    kr = np.linspace(0.0, rmax, num_samples)
    F_m = np.empty_like(kr)

    r1a = r1 - w1 / 2.0  # start  F1 -> F2 transition
    r1b = r1 + w1 / 2.0  # end    F1 -> F2 transition
    r2a = r2 - w2 / 2.0  # start  F2 -> F3 transition
    r2b = r2 + w2 / 2.0  # end    F2 -> F3 transition

    for i, kri in enumerate(kr):
        if kri <= r1a:
            F_m[i] = F1_m
        elif kri < r1b:
            # Transition F1 -> F2
            t = (kri - r1a) / (r1b - r1a)
            wcos = 0.5 * (1.0 - np.cos(np.pi * t))
            F_m[i] = F1_m + (F2_m - F1_m) * wcos
        elif kri <= r2a:
            F_m[i] = F2_m
        elif kri < r2b:
            # Transition F2 -> F3
            t = (kri - r2a) / (r2b - r2a)
            wcos = 0.5 * (1.0 - np.cos(np.pi * t))
            F_m[i] = F2_m + (F3_m - F2_m) * wcos
        else:
            F_m[i] = F3_m

    # Convert to cm
    F_cm = F_m * 100.0

    p = np.polyfit(kr, F_cm, degree)
    fov = p[::-1]
    return fov
