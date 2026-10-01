"""Runs of repetitions carried on a torch device in the isochromat engine's place: the slots a device holds and the Triton kernels that carry and spread them."""

from __future__ import annotations

import functools

import numpy as np

try:
    import torch
    import triton
    import triton.language as tl
except ImportError:  # without the gpu extra, which the device names
    torch = triton = tl = None

#: Repetitions of a tile, as the engine carries them.
TILE = 16

#: Slots the carry takes per program, and per step of the spreading's matrix
#: products.
SLOTS = 128
CHUNK = 32

#: Share of a run's slots dropped at which the rest are compacted.
COMPACT_AT = 0.25

#: Grid points of a turned window each program of its spreading sums onto,
#: and repetitions of a tile its slots are sorted for at once.
POINTS = 32
SORTED = 4


if triton is not None:

    @triton.jit
    def _phase(
        kinds_ptr,
        at_ptr,
        values_ptr,
        tables_ptr,
        index_ptr,
        coordinate_ptr,
        angle_ptr,
        s,
        r,
        i,
        mask,
        n,
    ):
        """Return the phase-encoding phase of set ``s`` at repetition ``r`` of slots ``i``: each axis's tabulated or computed phase, multiplied."""
        real = tables_ptr.dtype.element_ty
        re = tl.full(i.shape, 1.0, real)
        im = tl.zeros(i.shape, real)
        for axis in tl.static_range(3):
            kind = tl.load(kinds_ptr + s * 3 + axis)
            if kind == 1:
                at = tl.load(at_ptr + s * 3 + axis)
                values = tl.load(values_ptr + axis)
                k = tl.load(index_ptr + axis * n + i, mask=mask, other=0).to(tl.int64)
                pr = tl.load(tables_ptr + at + k * 16 + r, mask=mask, other=1.0)
                pi = tl.load(
                    tables_ptr + at + values * 16 + k * 16 + r, mask=mask, other=0.0
                )
                re, im = re * pr - im * pi, re * pi + im * pr
            elif kind == 2:
                angle = tl.load(angle_ptr + (s * 3 + axis) * 16 + r)
                x = tl.load(coordinate_ptr + axis * n + i, mask=mask, other=0.0)
                turn = angle * x.to(tl.float64)
                pr = tl.cos(turn).to(real)
                pi = tl.sin(turn).to(real)
                re, im = re * pr - im * pi, re * pi + im * pr
        return re, im

    @triton.jit
    def _point(p, w0, w1, w2, w3, e: tl.constexpr):
        """Return value ``e`` of the cubic of weights ``w0`` to ``w3`` through the four table points from ``p``, 16 values each."""
        return (
            w0 * tl.load(p + e)
            + w1 * tl.load(p + 16 + e)
            + w2 * tl.load(p + 32 + e)
            + w3 * tl.load(p + 48 + e)
        )

    @triton.jit
    def _cycle(phase):
        """Return the cosine and sine of ``phase`` cycles, given in double precision, in the precision of the run."""
        turn = (phase - tl.floor(phase + 0.5)) * 6.283185307179586
        return tl.cos(turn), tl.sin(turn)

    @triton.jit
    def _pulse(
        mx,
        my,
        mz,
        i,
        mask,
        n,
        r,
        field_ptr,
        place_ptr,
        class_ptr,
        row_ptr,
        weight_ptr,
        turn_ptr,
        density_ptr,
        maps_ptr,
        at_ptr,
        first_ptr,
        columns_ptr,
        relax_ptr,
        delta_ptr,
        spacing,
        before,
        after,
        ROWS: tl.constexpr,
    ):
        """Return the magnetisation of slots ``i`` carried through the first block of repetition ``r``, its pulse's map read off the tables as the engine reads it."""
        real = mx.dtype
        dx = tl.load(delta_ptr + r * 3)
        dy = tl.load(delta_ptr + r * 3 + 1)
        dz = tl.load(delta_ptr + r * 3 + 2)
        nu = tl.load(field_ptr + i, mask=mask, other=0.0)
        nu += dx * tl.load(place_ptr + i, mask=mask, other=0.0)
        nu += dy * tl.load(place_ptr + n + i, mask=mask, other=0.0)
        nu += dz * tl.load(place_ptr + 2 * n + i, mask=mask, other=0.0)
        at = nu / spacing
        point = tl.floor(at)
        u = (at - point).to(real)
        k = tl.load(class_ptr + i, mask=mask, other=0).to(tl.int64)
        columns = tl.load(columns_ptr + k, mask=mask, other=0)
        column = point.to(tl.int64) - 1 - tl.load(first_ptr + k, mask=mask, other=0)
        row = tl.load(row_ptr + i, mask=mask, other=0).to(tl.int64)
        base = tl.load(at_ptr + k, mask=mask, other=0) + 16 * (row * columns + column)
        base = tl.where(mask, base, 0)
        w0 = -u * (u - 1.0) * (u - 2.0) / 6.0
        w1 = (u + 1.0) * (u - 1.0) * (u - 2.0) / 2.0
        w2 = -(u + 1.0) * u * (u - 2.0) / 2.0
        w3 = (u + 1.0) * u * (u - 1.0) / 6.0
        t0 = tl.zeros(u.shape, real)
        t1 = tl.zeros(u.shape, real)
        t2 = tl.zeros(u.shape, real)
        t3 = tl.zeros(u.shape, real)
        t4 = tl.zeros(u.shape, real)
        t5 = tl.zeros(u.shape, real)
        t6 = tl.zeros(u.shape, real)
        t7 = tl.zeros(u.shape, real)
        t8 = tl.zeros(u.shape, real)
        t9 = tl.zeros(u.shape, real)
        t10 = tl.zeros(u.shape, real)
        t11 = tl.zeros(u.shape, real)
        for j in tl.range(0, ROWS, loop_unroll_factor=1):
            weight = tl.load(weight_ptr + j * n + i, mask=mask, other=0.0)
            p = maps_ptr + base + 16 * j * columns
            t0 += weight * _point(p, w0, w1, w2, w3, 0)
            t1 += weight * _point(p, w0, w1, w2, w3, 1)
            t2 += weight * _point(p, w0, w1, w2, w3, 2)
            t3 += weight * _point(p, w0, w1, w2, w3, 3)
            t4 += weight * _point(p, w0, w1, w2, w3, 4)
            t5 += weight * _point(p, w0, w1, w2, w3, 5)
            t6 += weight * _point(p, w0, w1, w2, w3, 6)
            t7 += weight * _point(p, w0, w1, w2, w3, 7)
            t8 += weight * _point(p, w0, w1, w2, w3, 8)
            t9 += weight * _point(p, w0, w1, w2, w3, 9)
            t10 += weight * _point(p, w0, w1, w2, w3, 10)
            t11 += weight * _point(p, w0, w1, w2, w3, 11)
        # The precession at the field before the pulse times the drive's
        # turn's conjugate, and after it times the turn.
        cb, sb = _cycle(-nu * before)
        ca, sa = _cycle(-nu * after)
        cb, sb, ca, sa = cb.to(real), sb.to(real), ca.to(real), sa.to(real)
        tr = tl.load(turn_ptr + i, mask=mask, other=0.0)
        ti = tl.load(turn_ptr + n + i, mask=mask, other=0.0)
        br = cb * tr + sb * ti
        bi = sb * tr - cb * ti
        ar = ca * tr - sa * ti
        ai = ca * ti + sa * tr
        density = tl.load(density_ptr + i, mask=mask, other=0.0)
        e2b = tl.load(relax_ptr + 4 * k, mask=mask, other=0.0)
        e1b = tl.load(relax_ptr + 4 * k + 1, mask=mask, other=0.0)
        e2a = tl.load(relax_ptr + 4 * k + 2, mask=mask, other=0.0)
        e1a = tl.load(relax_ptr + 4 * k + 3, mask=mask, other=0.0)
        x = e2b * mx
        y = e2b * my
        z = e1b * mz + (1.0 - e1b) * density
        # The map's rows, each turned by the precession before the pulse.
        cu = (br * t0 + bi * t1) * x + (br * t1 - bi * t0) * y + t2 * z + t9 * density
        cv = (br * t3 + bi * t4) * x + (br * t4 - bi * t3) * y + t5 * z + t10 * density
        w = (br * t6 + bi * t7) * x + (br * t7 - bi * t6) * y + t8 * z + t11 * density
        return (
            e2a * (ar * cu - ai * cv),
            e2a * (ai * cu + ar * cv),
            e1a * w + (1.0 - e1a) * density,
        )

    @triton.jit
    def _carry(
        m_ptr,
        a_ptr,
        b_ptr,
        u_ptr,
        v_ptr,
        limit_ptr,
        turns_ptr,
        kinds_ptr,
        at_ptr,
        values_ptr,
        tables_ptr,
        index_ptr,
        coordinate_ptr,
        angle_ptr,
        e_ptr,
        dropped_ptr,
        field_ptr,
        place_ptr,
        class_ptr,
        row_ptr,
        weight_ptr,
        turn_ptr,
        density_ptr,
        maps_ptr,
        table_at_ptr,
        first_ptr,
        columns_ptr,
        relax_ptr,
        pulse_delta_ptr,
        spacing,
        before,
        after,
        n,
        count,
        WINDOWS: tl.constexpr,
        OFFSETS: tl.constexpr,
        NETTED: tl.constexpr,
        DROP: tl.constexpr,
        PULSED: tl.constexpr,
        ROWS: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Carry BLOCK slots through the tile's ``count`` repetitions, writing each window's phase-encoded coefficients.

        Where PULSED, the magnetisation is first carried through the first
        block by its map read off the pulse's tables (:func:`_pulse`).
        Repetition r's coefficient of window w is then u . m + v, times set
        w's phase-encoding phase, written to e[w][r][Re, Im][slot]. The
        magnetisation then becomes A m + b, turned by the repetition's turn,
        times each slot's net-area phase where NETTED; without OFFSETS, A m
        alone. With DROP, a transient at or below its limit is zeroed and
        counted.
        """
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = i < n
        mx = tl.load(m_ptr + i, mask=mask, other=0.0)
        my = tl.load(m_ptr + n + i, mask=mask, other=0.0)
        mz = tl.load(m_ptr + 2 * n + i, mask=mask, other=0.0)
        a0 = tl.load(a_ptr + i, mask=mask, other=0.0)
        a1 = tl.load(a_ptr + n + i, mask=mask, other=0.0)
        a2 = tl.load(a_ptr + 2 * n + i, mask=mask, other=0.0)
        a3 = tl.load(a_ptr + 3 * n + i, mask=mask, other=0.0)
        a4 = tl.load(a_ptr + 4 * n + i, mask=mask, other=0.0)
        a5 = tl.load(a_ptr + 5 * n + i, mask=mask, other=0.0)
        a6 = tl.load(a_ptr + 6 * n + i, mask=mask, other=0.0)
        a7 = tl.load(a_ptr + 7 * n + i, mask=mask, other=0.0)
        a8 = tl.load(a_ptr + 8 * n + i, mask=mask, other=0.0)
        if OFFSETS:
            b0 = tl.load(b_ptr + i, mask=mask, other=0.0)
            b1 = tl.load(b_ptr + n + i, mask=mask, other=0.0)
            b2 = tl.load(b_ptr + 2 * n + i, mask=mask, other=0.0)
        for r in range(count):
            if PULSED:
                mx, my, mz = _pulse(
                    mx,
                    my,
                    mz,
                    i,
                    mask,
                    n,
                    r,
                    field_ptr,
                    place_ptr,
                    class_ptr,
                    row_ptr,
                    weight_ptr,
                    turn_ptr,
                    density_ptr,
                    maps_ptr,
                    table_at_ptr,
                    first_ptr,
                    columns_ptr,
                    relax_ptr,
                    pulse_delta_ptr,
                    spacing,
                    before,
                    after,
                    ROWS,
                )
            for w in tl.static_range(WINDOWS):
                u = u_ptr + w * 6 * n + i
                q_re = tl.load(u, mask=mask, other=0.0) * mx
                q_re += tl.load(u + n, mask=mask, other=0.0) * my
                q_re += tl.load(u + 2 * n, mask=mask, other=0.0) * mz
                q_im = tl.load(u + 3 * n, mask=mask, other=0.0) * mx
                q_im += tl.load(u + 4 * n, mask=mask, other=0.0) * my
                q_im += tl.load(u + 5 * n, mask=mask, other=0.0) * mz
                if OFFSETS:
                    q_re += tl.load(v_ptr + w * 2 * n + i, mask=mask, other=0.0)
                    q_im += tl.load(v_ptr + (w * 2 + 1) * n + i, mask=mask, other=0.0)
                p_re, p_im = _phase(
                    kinds_ptr,
                    at_ptr,
                    values_ptr,
                    tables_ptr,
                    index_ptr,
                    coordinate_ptr,
                    angle_ptr,
                    w,
                    r,
                    i,
                    mask,
                    n,
                )
                e = e_ptr + ((w * 16 + r) * 2) * n + i
                tl.store(e, q_re * p_re - q_im * p_im, mask=mask)
                tl.store(e + n, q_re * p_im + q_im * p_re, mask=mask)
            nx = a0 * mx + a1 * my + a2 * mz
            ny = a3 * mx + a4 * my + a5 * mz
            nz = a6 * mx + a7 * my + a8 * mz
            if OFFSETS:
                bx = nx + b0
                by = ny + b1
                nz = nz + b2
                c = tl.load(turns_ptr + r)
                sn = tl.load(turns_ptr + 16 + r)
                if NETTED:
                    g_re, g_im = _phase(
                        kinds_ptr,
                        at_ptr,
                        values_ptr,
                        tables_ptr,
                        index_ptr,
                        coordinate_ptr,
                        angle_ptr,
                        WINDOWS,
                        r,
                        i,
                        mask,
                        n,
                    )
                    c, sn = c * g_re - sn * g_im, sn * g_re + c * g_im
                nx = c * bx - sn * by
                ny = sn * bx + c * by
            mx = nx
            my = ny
            mz = nz
        if DROP:
            limit = tl.load(limit_ptr + i, mask=mask, other=0.0)
            size = mx * mx + my * my + mz * mz
            gone = (size > 0.0) & (size <= limit) & mask
            mx = tl.where(gone, 0.0, mx)
            my = tl.where(gone, 0.0, my)
            mz = tl.where(gone, 0.0, mz)
            tl.atomic_add(dropped_ptr, tl.sum(gone.to(tl.int32), axis=0))
        tl.store(m_ptr + i, mx, mask=mask)
        tl.store(m_ptr + n + i, my, mask=mask)
        tl.store(m_ptr + 2 * n + i, mz, mask=mask)

    @triton.jit
    def _spread(
        e_ptr,
        order_ptr,
        first_ptr,
        weight_ptr,
        factor_ptr,
        grid_ptr,
        n,
        count,
        coils,
        cells,
        taps,
        w,
        region,
        ORDERED: tl.constexpr,
        DOT: tl.constexpr,
        COILS: tl.constexpr,
        K: tl.constexpr,
    ):
        """Sum onto grid point g of T2 class d, for COILS coils, the coefficients of the slots that spread onto it.

        The slots of class d whose first grid point is s lie at
        order[first[d cells + s]] to before order[first[d cells + s + 1]],
        consecutively where ORDERED. Each adds its coefficient at every
        repetition times its tap's weight and each coil's factor, as COILS x K
        by K x 16 matrix products where DOT. The sums are written to the
        grid at [region + ((d coils + coil) (cells + taps) + g)][Re, Im]
        [repetition].
        """
        real = grid_ptr.dtype.element_ty
        g = tl.program_id(0)
        d = tl.program_id(1)
        coil = tl.program_id(2) * COILS + tl.arange(0, COILS)
        cmask = coil < coils
        rep = tl.arange(0, 16)
        rmask = rep < count
        acc_re = tl.zeros((COILS, 16), real)
        acc_im = tl.zeros((COILS, 16), real)
        for j in range(taps):
            s = g - j
            if (s >= 0) & (s < cells):
                lo = tl.load(first_ptr + d * cells + s)
                hi = tl.load(first_ptr + d * cells + s + 1)
                for k0 in range(lo, hi, K):
                    kk = k0 + tl.arange(0, K)
                    km = kk < hi
                    if ORDERED:
                        slot = kk.to(tl.int64)
                    else:
                        slot = tl.load(order_ptr + kk, mask=km, other=0).to(tl.int64)
                    weight = tl.load(
                        weight_ptr + (w * taps + j) * n + slot, mask=km, other=0.0
                    )
                    fmask = km[:, None] & cmask[None, :]
                    f = (
                        factor_ptr
                        + ((w * n + slot[:, None]) * coils + coil[None, :]) * 2
                    )
                    f_re = tl.load(f, mask=fmask, other=0.0) * weight[:, None]
                    f_im = tl.load(f + 1, mask=fmask, other=0.0) * weight[:, None]
                    emask = km[:, None] & rmask[None, :]
                    e = e_ptr + ((w * 16 + rep[None, :]) * 2) * n + slot[:, None]
                    e_re = tl.load(e, mask=emask, other=0.0)
                    e_im = tl.load(e + n, mask=emask, other=0.0)
                    if DOT:
                        t_re = tl.trans(f_re)
                        t_im = tl.trans(f_im)
                        acc_re += tl.dot(t_re, e_re, input_precision="ieee")
                        acc_re -= tl.dot(t_im, e_im, input_precision="ieee")
                        acc_im += tl.dot(t_re, e_im, input_precision="ieee")
                        acc_im += tl.dot(t_im, e_re, input_precision="ieee")
                    else:
                        acc_re += tl.sum(
                            f_re[:, :, None] * e_re[:, None, :]
                            - f_im[:, :, None] * e_im[:, None, :],
                            axis=0,
                        )
                        acc_im += tl.sum(
                            f_re[:, :, None] * e_im[:, None, :]
                            + f_im[:, :, None] * e_re[:, None, :],
                            axis=0,
                        )
        at = (
            region
            + ((d * coils + coil[:, None]) * (cells + taps) + g) * 32
            + rep[None, :]
        )
        tl.store(grid_ptr + at, acc_re, mask=cmask[:, None])
        tl.store(grid_ptr + at + 16, acc_im, mask=cmask[:, None])

    @triton.jit
    def _spread_turned(
        e_ptr,
        order_ptr,
        first_ptr,
        offset_ptr,
        lo_ptr,
        hi_ptr,
        polynomial_ptr,
        factor_ptr,
        grid_ptr,
        n,
        coils,
        points,
        w,
        r0,
        region,
        blocks,
        ranges,
        DOT: tl.constexpr,
        COILS: tl.constexpr,
        K: tl.constexpr,
        TAPS: tl.constexpr,
        POWERS: tl.constexpr,
        POINTS: tl.constexpr,
    ):
        """Sum onto POINTS grid points of turned window w, from b POINTS on, for T2 class d, repetition r0 + rr and COILS coils, the coefficients of the slots that spread onto them.

        At repetition r0 + rr the slots, in order of class and first grid
        point, are order[rr n + k], and those of class d that spread onto the
        points lie at k from lo to before hi, [rr ranges + d blocks + b];
        each slot's first grid point and its place in its interval are
        first[rr n + slot] and offset[rr n + slot]. Its weight onto the j-th
        point from its first is polynomial j of the kernel at its place,
        [power][tap]. Each adds its coefficient times its weights and each
        coil's factor, as POINTS x K by K x COILS matrix products where DOT.
        The sums are written to the grid at [region + ((d 16 + r) points +
        g) coils][coil][Re, Im].
        """
        real = grid_ptr.dtype.element_ty
        b = tl.program_id(0)
        d = tl.program_id(1)
        coil_blocks = tl.cdiv(coils, COILS)
        rr = tl.program_id(2) // coil_blocks
        r = r0 + rr
        coil = (tl.program_id(2) % coil_blocks) * COILS + tl.arange(0, COILS)
        cmask = coil < coils
        g = b * POINTS + tl.arange(0, POINTS)
        lo = tl.load(lo_ptr + rr * ranges + d * blocks + b)
        hi = tl.load(hi_ptr + rr * ranges + d * blocks + b)
        acc_re = tl.zeros((POINTS, COILS), real)
        acc_im = tl.zeros((POINTS, COILS), real)
        for k0 in range(lo, hi, K):
            kk = k0 + tl.arange(0, K)
            km = kk < hi
            slot = tl.load(order_ptr + rr * n + kk, mask=km, other=0).to(tl.int64)
            first = tl.load(first_ptr + rr * n + slot, mask=km, other=0)
            place = tl.load(offset_ptr + rr * n + slot, mask=km, other=0.0)
            e = e_ptr + ((w * 16 + r) * 2) * n + slot
            e_re = tl.load(e, mask=km, other=0.0)
            e_im = tl.load(e + n, mask=km, other=0.0)
            tap = g[:, None] - first[None, :]
            weight = tl.zeros((POINTS, K), real)
            for j in tl.static_range(TAPS):
                value = tl.full((K,), 0.0, real) + tl.load(
                    polynomial_ptr + (POWERS - 1) * TAPS + j
                )
                for power in tl.static_range(POWERS - 1):
                    value = value * place + tl.load(
                        polynomial_ptr + (POWERS - 2 - power) * TAPS + j
                    )
                weight += tl.where(tap == j, value[None, :], 0.0)
            m_re = weight * e_re[None, :]
            m_im = weight * e_im[None, :]
            fmask = km[:, None] & cmask[None, :]
            f = factor_ptr + ((w * n + slot[:, None]) * coils + coil[None, :]) * 2
            f_re = tl.load(f, mask=fmask, other=0.0)
            f_im = tl.load(f + 1, mask=fmask, other=0.0)
            if DOT:
                acc_re += tl.dot(m_re, f_re, input_precision="ieee")
                acc_re -= tl.dot(m_im, f_im, input_precision="ieee")
                acc_im += tl.dot(m_re, f_im, input_precision="ieee")
                acc_im += tl.dot(m_im, f_re, input_precision="ieee")
            else:
                acc_re += tl.sum(
                    m_re[:, :, None] * f_re[None, :, :]
                    - m_im[:, :, None] * f_im[None, :, :],
                    axis=1,
                )
                acc_im += tl.sum(
                    m_re[:, :, None] * f_im[None, :, :]
                    + m_im[:, :, None] * f_re[None, :, :],
                    axis=1,
                )
        at = (
            region
            + ((d * 16 + r) * points + g[:, None]) * 2 * coils
            + coil[None, :] * 2
        )
        omask = (g[:, None] < points) & cmask[None, :]
        tl.store(grid_ptr + at, acc_re, mask=omask)
        tl.store(grid_ptr + at + 1, acc_im, mask=omask)


@functools.cache
def _tuned_spread():
    """Return the spreading autotuned on a CUDA device over coils, slots per product and warps."""
    # Wider products than 32 x 32 spill kilobytes of registers on sm_89.
    configs = [
        triton.Config({"COILS": coils, "K": k}, num_warps=warps)
        for coils in (16, 32)
        for k in (16, 32)
        for warps in (4, 8)
    ]
    return triton.autotune(configs=configs, key=["coils", "taps", "ORDERED"])(_spread)


@functools.cache
def _tuned_spread_turned():
    """Return a turned window's spreading autotuned on a CUDA device over coils, slots per product and warps."""
    configs = [
        triton.Config({"COILS": coils, "K": k}, num_warps=warps)
        for coils in (16, 32)
        for k in (16, 32)
        for warps in (4, 8)
    ]
    return triton.autotune(configs=configs, key=["coils", "TAPS", "POWERS"])(
        _spread_turned
    )


class Run:
    """The slots of one run of repetitions on a device, carried and spread as the engine carries them.

    Each per-slot array is a row per value over the slots, in the engine's
    order until the run is compacted; :attr:`ids` maps the slots held to the
    engine's. Per window, the slots are ordered by T2 class and first grid
    point, with where each class's slots of each first grid point start.
    """

    def __init__(self, run: dict, device: torch.device):
        self.device = device
        self.real = torch.float32 if run["single"] else torch.float64
        n = int(run["slots"])
        self.offsets = bool(run["offsets"])
        self.limits = bool(run["limits"])
        self.cells = [int(c) for c in run["cells"]]
        self.region = [int(r) for r in run["region"]]
        self.windows = len(self.cells)
        self.turned = [bool(t) for t in np.asarray(run["turned"])]
        self.coils = int(run["coils"])
        self.taps = int(run["taps"])
        self.classes = int(run["classes"])
        pulse = run.get("pulse")
        self.pulsed = pulse is not None
        self.rows = {
            name: torch.tensor(np.ascontiguousarray(row)).to(device)
            for name, row in _slot_rows(run).items()
            if row.size
        }
        self.tables, self.pulse_rows, self.timing = _pulse_reading(pulse, device)
        # What a kernel is handed for a row the run does not hold.
        self._none = torch.zeros(1, dtype=self.real, device=device)
        # (windows, slots, coils, Re/Im), as the engine lays the factors out per slot.
        self.factor = torch.tensor(
            np.ascontiguousarray(np.asarray(run["factor"]).transpose(1, 0, 2, 3))
        ).to(device)
        self.lattice = torch.tensor(np.asarray(run["lattice"], dtype=np.int64)).to(
            device
        )
        self.ids = torch.arange(n, device=device)
        self.slots = n
        self.dropped = 0
        self._counter = torch.zeros(1, dtype=torch.int32, device=device)
        self._group()

    def _group(self) -> None:
        """Order each window's slots by T2 class and first grid point, and find where each group starts."""
        n = self.slots
        self.e = torch.zeros(
            (max(self.windows, 1), TILE, 2, n), dtype=self.real, device=self.device
        )
        self.orders, self.firsts, self.ordered = [], [], []
        decay = self.rows["decay"][0]
        for w, cells in enumerate(self.cells):
            if self.turned[w]:
                # Ordered anew at each repetition.
                self.orders.append(None)
                self.firsts.append(None)
                self.ordered.append(False)
                continue
            key = decay * max(cells, 1) + (self.rows["start"][w] if cells else 0)
            order = torch.argsort(key, stable=True)
            counts = torch.bincount(key, minlength=self.classes * max(cells, 1))
            first = torch.zeros(
                counts.numel() + 1, dtype=torch.int64, device=self.device
            )
            first[1:] = torch.cumsum(counts, 0)
            self.orders.append(order.to(torch.int32))
            self.firsts.append(first)
            self.ordered.append(
                bool(torch.equal(order, torch.arange(n, device=self.device)))
            )

    def row(self, name: str) -> torch.Tensor:
        return self.rows.get(name, self._none)

    def carry(self, tile: dict, clock, lap) -> int:
        """Carry the slots through ``tile`` and write the grids the engine reads into its ``grid``; return the transients dropped.

        ``clock()`` and ``lap(stage, began)`` time the stages, as the device
        profiles them.
        """
        began = clock()
        n = self.slots
        count = int(tile["count"])
        turns = np.concatenate([tile["turn_cos"], tile["turn_sin"]])
        kinds = np.asarray(tile["encoding"], dtype=np.int32)
        at = np.zeros(kinds.shape, dtype=np.int64)
        parts = []
        offset = 0
        for place, table in enumerate(tile["tables"]):
            if table is not None:
                at.flat[place] = offset
                parts += [np.ravel(table[0]), np.ravel(table[1])]
                offset += 2 * table[0].size
        tables = np.concatenate(parts) if parts else np.zeros(1)

        def put(values, dtype=None):
            return torch.tensor(np.ascontiguousarray(values), dtype=dtype).to(
                self.device
            )

        grid = torch.zeros(len(tile["grid"]), dtype=self.real, device=self.device)
        pulse_delta = (
            put(np.ravel(tile["pulse_delta"]), torch.float64)
            if self.pulsed
            else self._none
        )
        began = lap("upload", began)
        if n:
            self._counter.zero_()
            row = self.row
            table = self.tables.get
            _carry[(triton.cdiv(n, SLOTS),)](
                row("m"),
                row("a"),
                row("b"),
                row("u"),
                row("v"),
                row("limit"),
                put(turns, self.real),
                put(kinds),
                put(at),
                self.lattice,
                put(tables, self.real),
                row("index"),
                row("coordinate"),
                put(tile["angle"], torch.float64),
                self.e,
                self._counter,
                row("field"),
                row("place"),
                row("pulse_class"),
                row("row"),
                row("row_weight"),
                row("drive_turn"),
                row("density"),
                table("maps", self._none),
                table("at", self._none),
                table("first", self._none),
                table("columns", self._none),
                table("relax", self._none),
                pulse_delta,
                *self.timing,
                n,
                count,
                WINDOWS=self.windows,
                OFFSETS=self.offsets,
                NETTED=bool(tile["netted"]),
                DROP=bool(tile["drop"]) and self.limits,
                PULSED=self.pulsed,
                ROWS=self.pulse_rows,
                BLOCK=SLOTS,
            )
            began = lap("carry", began)
            for w in range(self.windows):
                if self.turned[w]:
                    self._spread_turned(w, tile, grid)
                else:
                    self._spread(w, count, grid)
            began = lap("spread", began)
        _download(grid, tile["grid"])
        lap("download", began)
        dropped = int(self._counter.item()) if n else 0
        self.dropped += dropped
        if self.dropped > COMPACT_AT * self.slots:
            self._compact()
        return dropped

    def _spread(self, w: int, count: int, grid: torch.Tensor) -> None:
        cells = self.cells[w]
        if cells == 0:
            # One sample: each coil's sum over the slots, without a kernel.
            f = torch.complex(self.factor[w, :, :, 0], self.factor[w, :, :, 1])
            e = torch.complex(self.e[w, :count, 0], self.e[w, :count, 1])
            sums = torch.zeros((self.coils, TILE), dtype=f.dtype, device=self.device)
            sums[:, :count] = (e @ f).T
            part = grid[self.region[w] : self.region[w + 1]].view(self.coils, 2, TILE)
            part[:, 0] = sums.real
            part[:, 1] = sums.imag
            return
        arguments = (
            self.e,
            self.orders[w],
            self.firsts[w],
            self.row("weight"),
            self.factor,
            grid,
            self.slots,
            count,
            self.coils,
            cells,
            self.taps,
            w,
            self.region[w],
        )
        dot = self.coils >= 16
        if self.device.type == "cuda" and dot:
            _tuned_spread()[
                lambda meta: (
                    cells + self.taps,
                    self.classes,
                    triton.cdiv(self.coils, meta["COILS"]),
                )
            ](*arguments, ORDERED=self.ordered[w], DOT=True)
            return
        coils = 16 if dot else triton.next_power_of_2(self.coils)
        _spread[(cells + self.taps, self.classes, triton.cdiv(self.coils, coils))](
            *arguments, ORDERED=self.ordered[w], DOT=dot, COILS=coils, K=CHUNK
        )

    def _spread_turned(self, w: int, tile: dict, grid: torch.Tensor) -> None:
        """Spread turned window ``w`` a repetition at a time, its slots in order of T2 class and the first grid point they reach."""
        n, count = self.slots, int(tile["count"])
        cells, taps = self.cells[w], self.taps
        points = cells + taps
        delta = torch.tensor(np.asarray(tile["delta"][w]), device=self.device)
        place = self.rows["place"]
        decay = self.rows["decay"][0].to(torch.int32) * cells
        blocks = triton.cdiv(points, POINTS)
        g0 = torch.arange(blocks, device=self.device, dtype=torch.int32) * POINTS
        base = torch.arange(self.classes, device=self.device, dtype=torch.int32)
        bounds = [
            (base[:, None] * cells + edge.clamp(0, cells)[None, :]).reshape(1, -1)
            for edge in (g0 - taps + 1, g0 + POINTS)
        ]
        polynomial = torch.tensor(np.ascontiguousarray(tile["polynomials"][w])).to(
            self.device
        )
        constants = {"TAPS": taps, "POWERS": polynomial.shape[0], "POINTS": POINTS}
        dot = self.coils >= 16
        for r0 in range(0, count, SORTED):
            rows = min(SORTED, count - r0)
            step = delta[:, r0 : r0 + rows, None]
            # Where each slot is spread at each repetition, as Nufft::locate()
            # finds it.
            u = self.rows["origin"][w] + step[0] * place[0]
            u += step[1] * place[1]
            u += step[2] * place[2]
            y = cells * (u - torch.round(u)) - 0.5 * taps + 0.5
            del u
            first = torch.round(y)
            offset = (y - first).to(self.real)
            del y
            first = torch.where(first < 0, first + cells, first).to(torch.int32)
            keys, order = torch.sort(decay + first, dim=1)
            lo, hi = (
                torch.searchsorted(keys, edge.expand(rows, -1).contiguous()).to(
                    torch.int32
                )
                for edge in bounds
            )
            del keys
            arguments = (
                self.e,
                order.to(torch.int32),
                first,
                offset,
                lo,
                hi,
                polynomial,
                self.factor,
                grid,
                n,
                self.coils,
                points,
                w,
                r0,
                self.region[w],
                blocks,
                self.classes * blocks,
            )
            del order
            if self.device.type == "cuda" and dot:
                _tuned_spread_turned()[
                    lambda meta, rows=rows: (
                        blocks,
                        self.classes,
                        rows * triton.cdiv(self.coils, meta["COILS"]),
                    )
                ](*arguments, DOT=True, **constants)
                continue
            coils = 16 if dot else triton.next_power_of_2(self.coils)
            _spread_turned[
                (blocks, self.classes, rows * triton.cdiv(self.coils, coils))
            ](*arguments, DOT=dot, COILS=coils, K=16, **constants)

    def _compact(self) -> None:
        """Keep only the slots whose magnetisation is not zero, in their order."""
        kept = torch.nonzero(self.rows["m"].abs().sum(0) > 0).ravel()
        self.rows = {name: row[:, kept].contiguous() for name, row in self.rows.items()}
        self.factor = self.factor[:, kept].contiguous()
        self.ids = self.ids[kept]
        self.slots = int(kept.numel())
        self.dropped = 0
        self._group()

    def write(self, state: dict) -> None:
        """Write the magnetisation into the engine's ``state["m"]``, zero for the slots dropped."""
        m = self.rows["m"]
        n = int(state["slots"])
        if self.slots != n:
            m = torch.zeros((3, n), dtype=m.dtype, device=m.device).index_copy_(
                1, self.ids, m
            )
        torch.from_numpy(state["m"]).copy_(m)

    def load(self, state: dict) -> None:
        """Take the magnetisation in the engine's ``state["m"]`` as the slots'."""
        m = torch.from_numpy(state["m"]).to(self.device)
        self.rows["m"].copy_(m if self.slots == int(state["slots"]) else m[:, self.ids])

    def nbytes(self) -> int:
        """Bytes the run holds on the device."""
        held = [
            *self.rows.values(),
            *self.tables.values(),
            self.factor,
            self.e,
            *self.orders,
            *self.firsts,
        ]
        return sum(t.numel() * t.element_size() for t in held)


def _slot_rows(run: dict) -> dict:
    """Return each slot's values the carry and the spreading take, a row per value over the slots, in the engine's order; empty rows for values the run has none of."""
    n = int(run["slots"])
    pack = np.asarray(run["pack"])
    values = pack.transpose(1, 0, 2).reshape(pack.shape[1], -1)[:, :n]
    none = values[0:0]
    windows = len(run["cells"])
    offsets = bool(run["offsets"])
    u_at, u_width = int(run["u_at"]), int(run["u_width"])
    limit_at = int(run["limit_at"])
    rows = {
        "m": values[0:3],
        "a": values[3:12],
        "b": values[12:15] if offsets else none,
        "u": _rows(values, u_at, u_width, 0, 6, windows),
        "v": _rows(values, u_at, u_width, 6, 8, windows) if offsets else none,
        "limit": values[limit_at : limit_at + 1] if run["limits"] else none,
        "weight": np.asarray(run["weight"]).transpose(1, 2, 0).reshape(-1, n),
        "decay": np.asarray(run["decay"], dtype=np.int64)[None],
        "start": np.asarray(run["start"], dtype=np.int64).T,
        "index": np.zeros((3, n), dtype=np.int32),
        "coordinate": np.zeros((3, n), dtype=values.dtype),
    }
    for axis in range(3):
        if run["index"][axis] is not None:
            rows["index"][axis] = run["index"][axis]
        if run["coordinate"][axis] is not None:
            rows["coordinate"][axis] = run["coordinate"][axis]
    turned = bool(np.any(run["turned"]))
    pulse = run.get("pulse")
    if turned:
        rows["origin"] = np.asarray(run["origin"]).T
    if turned or pulse is not None:
        rows["place"] = np.asarray(run["place"]).T
    if pulse is not None:
        rows.update(_pulse_rows(pulse))
    return rows


def _pulse_reading(
    pulse: dict | None, device: torch.device
) -> tuple[dict, int, tuple[float, float, float]]:
    """Return what reading the first block's pulse off tables takes besides each slot's values: the tables on ``device``, the drive's rows each slot's map is interpolated over, and the tables' spacing and the times from the block's start to the pulse's centre and from it to the block's end; no tables, one row and no times where the pulse is not read so."""
    if pulse is None:
        return {}, 1, (1.0, 0.0, 0.0)
    tables = {
        name: torch.tensor(np.ascontiguousarray(values)).to(device)
        for name, values in _pulse_tables(pulse).items()
    }
    timing = (float(pulse["spacing"]), float(pulse["before"]), float(pulse["after"]))
    return tables, int(np.shape(pulse["row_weight"])[1]), timing


def _pulse_rows(pulse: dict) -> dict:
    """Return each slot's values reading the first block's pulse off tables takes, a row per value over the slots."""
    return {
        "field": np.asarray(pulse["field"])[None],
        "pulse_class": np.asarray(pulse["pulse_class"], dtype=np.int32)[None],
        "row": np.asarray(pulse["row"], dtype=np.int32)[None],
        "row_weight": np.asarray(pulse["row_weight"]).T,
        "drive_turn": np.asarray(pulse["drive_turn"]).T,
        "density": np.asarray(pulse["density"])[None],
    }


def _pulse_tables(pulse: dict) -> dict:
    """Return the pulse's tables and, per class, where each starts, its first column, its columns and its decays.

    Raises
    ------
    ValueError
        If a table point holds other than 16 values, as :func:`_point` reads
        them.
    """
    if int(pulse["table_values"]) != 16:
        raise ValueError(
            f"a pulse table's points hold 16 values, not {pulse['table_values']}"
        )
    return {
        "maps": np.asarray(pulse["maps"]),
        "at": np.asarray(pulse["table_at"], dtype=np.int64),
        "first": np.asarray(pulse["table_first"], dtype=np.int64),
        "columns": np.asarray(pulse["table_columns"], dtype=np.int64),
        "relax": np.ravel(pulse["relax"]),
    }


def _rows(
    values: np.ndarray, u_at: int, u_width: int, low: int, high: int, windows: int
) -> np.ndarray:
    """Values ``low`` to before ``high`` of each window's part of a slot, stacked: (windows * (high - low), slots)."""
    parts = [
        values[u_at + w * u_width + low : u_at + w * u_width + high]
        for w in range(windows)
    ]
    return np.concatenate(parts) if parts else values[0:0]


def _download(grid: torch.Tensor, into: np.ndarray) -> None:
    into[:] = grid.cpu().numpy()


def bytes_for(run: dict) -> int:
    """Bytes a run takes on the device, from what the engine hands, and those a tile of a turned window takes."""
    n = int(run["slots"])
    windows = len(run["cells"])
    itemsize = 4 if run["single"] else 8
    per_slot = (
        3
        + 9
        + 3
        + windows * 8
        + 1
        + windows * int(run["taps"])
        + windows * int(run["coils"]) * 2
    ) * itemsize
    per_slot += (
        max(windows, 1) * TILE * 2 * itemsize + 3 * (4 + itemsize) + 8 * (2 + windows)
    )
    if np.any(np.asarray(run["turned"])):
        # Places and origins, and per repetition sorted at once the
        # positions, grid points, places, keys and orders of the slots, and
        # what sorting them takes besides.
        per_slot += 8 * (3 + windows) + SORTED * (
            8 + 8 + 8 + itemsize + 4 + 4 + 8 + 4 + 24
        )
    pulse = run.get("pulse")
    if pulse is None:
        return n * per_slot
    # Field and place, class and row, the rows' weights, the drive's turn and
    # the density; and the tables.
    rows = np.shape(pulse["row_weight"])[1]
    per_slot += 8 * 4 + 4 * 2 + (rows + 3) * itemsize
    return n * per_slot + np.size(pulse["maps"]) * itemsize
