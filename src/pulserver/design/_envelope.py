"""The gradient envelope of a sequence's repetition: the repetition played at every extreme of its gradients."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pypulseqpp as pp

#: Sides of the polygon that bounds a plane of gradient variation: its
#: vertices lie at most ``1 / cos(pi / SIDES)`` of the plane's extent past
#: what any copy plays.
SIDES = 12

#: Varying blocks back from a block that are searched for one varying with
#: it, so that the two play their extremes together.
_REACH = 8

#: Duration (s) of a repetition below which the nerve response one copy
#: carries into the next is bounded too.
_CARRIED = 0.1

#: Repetitions a short repetition's envelope holds each vertex for.
_RUN = 4

#: Relative residual under which one block's variation is taken to follow
#: another's.
_FOLLOWS = 1e-6

_AXES = ("gx", "gy", "gz")
_CHANNELS = "xyz"


def repetition_envelope(sequence: pp.Sequence) -> pp.Sequence:
    """Return the repetition of ``sequence`` played at every extreme of its gradients over the scan.

    The repetition is :meth:`pypulseqpp.Sequence.repetition`'s, the scan's
    outer loop, and its copies, the last cut short included, play the same
    block definitions (:meth:`pypulseqpp.Sequence.block_definitions`): a
    block's gradients differ from copy to copy by their amplitudes and by the
    block's rotation alone, and are, per copy, one vector on the axes for
    each gradient shape the block plays. Blocks whose vectors vary along one
    line, or in one plane, of the same copy-wise parameters -- an encode and
    its rewinder, the two halves of a gradient split across blocks -- are
    bounded together by one polytope: the interval the copies span on a line,
    or a regular :data:`SIDES`-gon circumscribing them in a plane, turned onto
    the polytope before it so that neighbouring blocks reach their extremes
    in the same sense. The envelope plays the repetition once per vertex,
    each such block at its polytope's vertex with its rotation applied to its
    gradients, every other block as the first copy plays it, a wait as the
    shortest any copy plays; a repetition shorter than :data:`_CARRIED` is
    also played into every other vertex and into itself.

    The slew rate on any axis under any rotation of the whole, the slew rate
    summed over the axes and the nerve response, a linear filter of the slew
    rate, are convex in the vectors, so over blocks bounded together each is
    largest at a vertex: the envelope reads at least what the scan does, but
    for a response carried between neighbouring blocks that vary
    independently and in other senses than the envelope pairs them.

    A sequence that does not repeat, whose copies differ in a block
    definition, or with a block whose vectors vary in more than a plane -- a
    blade turned while it is phase encoded, a spoke turned in 3D -- is
    returned unchanged: it is its own envelope. The definitions and the
    system are those of ``sequence``.
    """
    definitions = np.asarray(sequence.block_definitions())
    total = definitions.size
    size, start = sequence.repetition()
    if size == 0 or size >= total or start != 1:
        return sequence
    copies = -(-total // size)
    padded = np.full(copies * size, -1, dtype=np.int64)
    padded[:total] = definitions
    stacked = padded.reshape(copies, size)
    played = stacked >= 0
    if not np.all((stacked == stacked[0]) | ~played):
        return sequence
    vectors = np.zeros((copies * size, 3, 3))
    vectors[:total] = _vectors(sequence)
    vectors = vectors.reshape(copies, size, 9)
    vertices = _vertices(vectors, played)
    if vertices is None:
        return sequence
    return _played(sequence, size, vectors, played, vertices)


def _vectors(sequence: pp.Sequence) -> np.ndarray:
    """Return, per block, its rotation times its gradient amplitudes, ``(N, 3, 3)`` in Hz/m.

    Column ``j`` is the vector the shape on logical axis ``j`` plays on the
    axes the block's rotation turns it onto; zero where the block plays no
    gradient there.
    """
    tables = sequence.libraries()
    count = 1 + max(
        int(tables.trapezoid_ids.max(initial=0)),
        int(tables.arbitrary_gradient_ids.max(initial=0)),
    )
    amplitude = np.zeros(count)
    amplitude[tables.trapezoid_ids] = tables.trapezoids[:, 0]
    # An arbitrary gradient's samples are its shape, normalised to a peak of
    # one, times its amplitude.
    amplitude[tables.arbitrary_gradient_ids] = tables.arbitrary_gradients[:, 0]
    ids = np.asarray(tables.blocks)[:, 1:4]
    amplitudes = amplitude[ids] * (ids > 0)
    rotations = np.concatenate(
        [np.eye(3)[None], _matrices(np.asarray(tables.rotations).reshape(-1, 4))]
    )
    turned = rotations[np.asarray(sequence.block_rotations())]
    return turned * amplitudes[:, None, :]


def _matrices(quaternions: np.ndarray) -> np.ndarray:
    """Return the rotation matrix of each unit quaternion, scalar first."""
    w, x, y, z = quaternions.T
    return np.stack(
        [
            np.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1
            ),
            np.stack(
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1
            ),
            np.stack(
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1
            ),
        ],
        -2,
    )


def _varying(vectors: np.ndarray, played: np.ndarray) -> list[int]:
    """Return the blocks of the repetition whose vectors differ between the copies that play them."""
    high = np.where(played[..., None], vectors, -np.inf).max(axis=0)
    low = np.where(played[..., None], vectors, np.inf).min(axis=0)
    scale = np.abs(vectors).max(initial=0.0)
    return [int(p) for p in np.flatnonzero(np.any(high - low > 1e-9 * scale, axis=1))]


def _vertices(vectors: np.ndarray, played: np.ndarray) -> dict[int, np.ndarray] | None:
    """Return, per varying block, the ``(n, 9)`` vectors it plays at each vertex the envelope plays.

    Each varying block is reduced to its coordinates along its principal
    directions of variation, over the copies that play it, and joins the
    polytope of a recent block whose coordinates its own follow linearly;
    ``None`` where a block varies in more than a plane. Each polytope is
    turned onto the one before it, so that the envelope plays neighbouring
    blocks at the extremes the copies hold together; see :func:`_aligned`.
    """
    reduced = {}
    for position in _varying(vectors, played):
        rows = np.flatnonzero(played[:, position])
        cloud = vectors[rows, position]
        centre = cloud.mean(axis=0)
        _, values, basis = np.linalg.svd(cloud - centre, full_matrices=False)
        directions = int(np.count_nonzero(values > 1e-9 * values[0]))
        if directions > 2:
            return None
        basis = basis[:directions]
        along = (cloud - centre) @ basis.T
        reduced[position] = (rows, centre, along, basis, np.linalg.qr(along))
    # Each polytope: the block whose coordinates are its parameters, and per
    # member the map from those parameters to the member's vectors.
    polytopes: list[tuple[int, dict[int, np.ndarray]]] = []
    for position, (rows, _, along, basis, _) in reduced.items():
        for reference, members in reversed(polytopes[-_REACH:]):
            mapped = _follows(reduced[reference], rows, along)
            if mapped is not None:
                members[position] = mapped @ basis
                break
        else:
            polytopes.append((position, {position: basis}))
    plane = any(reduced[reference][2].shape[1] == 2 for reference, _ in polytopes)
    normals = _normals(plane)
    found = {}
    previous = None
    for reference, members in polytopes:
        rows, _, along, _, _ = reduced[reference]
        frame = _aligned(along, rows, previous, plane)
        previous = (rows, along @ frame)
        corners = _corners(along, frame, normals)
        for position, mapping in members.items():
            found[position] = reduced[position][1] + corners @ mapping
    return found


def _follows(reference, rows: np.ndarray, along: np.ndarray) -> np.ndarray | None:
    """Return the map from a reference block's coordinates to ``along``, or None where they do not follow them.

    Only copies that play both blocks are compared; a block played by a copy
    its reference is not played by does not follow it.
    """
    reference_rows, _, reference_along, _, (orthonormal, triangle) = reference
    if reference_along.shape[1] < along.shape[1]:
        return None
    if reference_rows.size != rows.size or np.any(reference_rows != rows):
        where = np.searchsorted(reference_rows, rows)
        if np.any(where >= reference_rows.size) or np.any(
            reference_rows[where] != rows
        ):
            return None
        orthonormal, triangle = np.linalg.qr(reference_along[where])
    projected = orthonormal.T @ along
    if np.linalg.norm(along - orthonormal @ projected) > _FOLLOWS * np.linalg.norm(
        along
    ):
        return None
    return np.linalg.solve(triangle, projected)


def _normals(plane: bool) -> np.ndarray:
    """Return the directions, in the frame every polytope is turned onto, whose extremes the envelope plays in turn.

    :data:`SIDES` around a plane, or the two senses of a line where no block
    varies in more than one.
    """
    if not plane:
        return np.array([[1.0], [-1.0]])
    angles = 2 * np.pi * np.arange(SIDES) / SIDES
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _aligned(along, rows, previous, plane: bool) -> np.ndarray:
    """Return the map from a polytope's coordinates onto the common frame, ``(d, 1)`` or ``(d, 2)``.

    The map, a rotation or reflection of a plane's coordinates and a
    direction for a line's, best matches the coordinates the polytope before
    it takes over the copies both play; a polytope with none before it keeps
    its own. Neighbouring blocks whose copies play nearby views -- an encode
    and the rewinder or encode next to it -- are thereby played at their
    extremes in the same sense.
    """
    width = 2 if plane else 1
    own = np.eye(along.shape[1], width)
    if previous is None:
        return own
    earlier_rows, earlier = previous
    _, mine, theirs = np.intersect1d(rows, earlier_rows, return_indices=True)
    product = along[mine].T @ earlier[theirs]
    if not np.any(product):
        return own
    if along.shape[1] == 1:
        return product / np.linalg.norm(product)
    left, _, right = np.linalg.svd(product)
    return left @ right


def _corners(along: np.ndarray, frame: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Return, per normal of the common frame, the vertex of the polytope bounding coordinates ``along`` that reaches furthest along it.

    A line's polytope is the interval the copies span; a plane's, the
    regular :data:`SIDES`-gon whose sides are normal to ``normals``,
    circumscribing them.
    """
    if along.shape[1] == 1:
        # A plane's vertex k lies between its normals k and k + 1.
        between = (
            normals + np.roll(normals, -1, axis=0) if normals.shape[1] == 2 else normals
        )
        ends = np.array([along.min(), along.max()])
        return ends[((frame @ between.T)[0] > 0).astype(int)][:, None]
    reach = (along @ (frame @ normals.T)).max(axis=0)
    # Each vertex is where the sides along two neighbouring normals meet; in
    # the common frame, then back in the polytope's own coordinates.
    corners = np.linalg.solve(
        np.stack([normals, np.roll(normals, -1, axis=0)], axis=1),
        np.stack([reach, np.roll(reach, -1)], axis=1)[..., None],
    )[..., 0]
    return corners @ frame.T


def _played(
    sequence: pp.Sequence,
    size: int,
    vectors: np.ndarray,
    played: np.ndarray,
    vertices: dict[int, np.ndarray],
) -> pp.Sequence:
    """Return the repetition played at its vertices, in the order :func:`_order` gives."""
    envelope = pp.Sequence(sequence.system)
    for key, value in sequence.definitions.items():
        envelope.set_definition(key=key, value=value)
    first = [sequence.get_block(position + 1) for position in range(size)]
    # A wait between copies plays as the shortest any copy plays, which
    # carries the most nerve response from one copy into the next.
    total = played.size
    durations = np.full(total, np.inf)
    durations[: len(sequence.block_durations)] = sequence.libraries().block_durations
    shortest = durations.reshape(played.shape).min(axis=0)
    for position, block in enumerate(first):
        if shortest[position] < block.block_duration:
            first[position] = SimpleNamespace(**vars(block))
            first[position].block_duration = float(shortest[position])
    # A sum of turned shapes is measured here, not designed: it is held to no
    # limit when it is made.
    unlimited = copy.copy(sequence.system)
    unlimited.max_grad = unlimited.max_slew = np.inf
    shapes = {
        position: _shapes(sequence, size, vectors, played, position)
        for position in vertices
    }
    placed = {
        position: [
            _block(unlimited, first[position], shapes[position], corner.reshape(3, 3))
            for corner in corners
        ]
        for position, corners in vertices.items()
    }
    count = max((len(corners) for corners in vertices.values()), default=1)
    period = float(np.sum(shortest))
    for index in _order(count, size, played.shape[0] * size, period):
        for position, block in enumerate(first):
            corners = placed.get(position)
            envelope.add_block(
                block if corners is None else corners[index % len(corners)]
            )
    return envelope


def _order(count: int, size: int, total: int, period: float) -> list[int]:
    """Return the vertices the repetitions of the envelope play, in turn.

    Each vertex once, for a repetition lasting at least :data:`_CARRIED`
    seconds. A shorter one plays every ordered pair of the ``count`` vertices
    instead, a de Bruijn sequence closed on its first, each vertex held for
    up to :data:`_RUN` repetitions, while that plays at most half the scan's
    ``total`` blocks: a copy then runs into every other, and into copies like
    itself, as the copies of a scan that plays one view several times in a
    row, at each slice of a packet, do.
    """
    pairs = count * count + 1
    run = min(_RUN, total // (2 * pairs * size))
    if period >= _CARRIED or run == 0:
        return list(range(count))
    order: list[int] = []
    word = [0, 0, 0]

    def extend(t: int, p: int) -> None:
        # The Lyndon words of length dividing two, concatenated in order.
        if t > 2:
            if 2 % p == 0:
                order.extend(word[1 : p + 1])
            return
        word[t] = word[t - p]
        extend(t + 1, p)
        for symbol in range(word[t - p] + 1, count):
            word[t] = symbol
            extend(t + 1, t)

    extend(1, 1)
    return [vertex for vertex in (*order, order[0]) for _ in range(run)]


def _shapes(
    sequence: pp.Sequence,
    size: int,
    vectors: np.ndarray,
    played: np.ndarray,
    position: int,
) -> dict[int, tuple[object, float]]:
    """Return, per logical axis a block plays a gradient on, the event of its largest copy and that event's amplitude."""
    amplitudes = np.linalg.norm(vectors[:, position].reshape(-1, 3, 3), axis=1)
    amplitudes[~played[:, position]] = 0.0
    shapes = {}
    for axis in range(3):
        largest = int(np.argmax(amplitudes[:, axis]))
        if amplitudes[largest, axis] == 0.0:
            continue
        event = getattr(sequence.get_block(largest * size + position + 1), _AXES[axis])
        shapes[axis] = (event, float(event.amplitude))
    return shapes


def _block(
    system: pp.Opts,
    block: SimpleNamespace,
    shapes: dict[int, tuple[object, float]],
    vector: np.ndarray,
) -> SimpleNamespace:
    """Return ``block`` playing each shape at the column of ``vector`` for its axis, the rotation applied."""
    least = 1e-12 * np.abs(vector).max(initial=0.0)
    rows = vector.tolist()
    turned = SimpleNamespace(**vars(block))
    turned.rotation = None
    for axis, row in enumerate(rows):
        parts = []
        for source, (event, amplitude) in shapes.items():
            value = row[source]
            if abs(value) > least:
                part = pp.scale_grad(event, value / amplitude)
                if source != axis:
                    part.channel = _CHANNELS[axis]
                parts.append(part)
        if len(parts) > 1:
            parts = [pp.add_gradients(parts, system=system)]
        setattr(turned, _AXES[axis], parts[0] if parts else None)
    return turned
