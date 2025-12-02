"""Automatic segment definition subroutines."""

__all__ = ["find_segment_definitions", "find_segments", "split_rotated_segments"]

import numpy as np
import numba as nb


def find_segment_definitions(arr):
    # here, we make the (reasonable?) assumption
    # that all the "prescan", dummy / syncronization / calibration
    # scans are performed at the beginning of the sequence,
    # then we have a long periodic main loop. We want to find the beginning
    # of the main loop
    main_loop, rem = _find_periodic_pattern(arr.astype(int))

    # hopefully rem is empty; if it is not, append at the beginning
    if len(rem) > 0 and len(main_loop) == 0:
        main_loop = [rem.tolist()]
    elif len(rem) > 0:
        main_loop = [rem.tolist() + main_loop.tolist()]
    else:
        main_loop = [main_loop.tolist()]

    return main_loop


def find_segments(array, subarray):
    arr = nb.typed.List(array)
    subarr = nb.typed.List(subarray)
    return list(_find_segments(arr, subarr))


@nb.njit(cache=True, fastmath=True)
def _find_segments(array, subarray):
    n = len(array)
    m = len(subarray)
    result = np.zeros(n, dtype=np.bool_)

    for i in range(n - m + 1):
        match = True
        for j in range(m):
            if array[i + j] != subarray[j]:
                match = False
                break
        if match:
            for k in range(m):
                result[i + k] = True

    return result


def split_rotated_segments(input):
    output = []
    for segment in input:
        tmp = _split_signed_blocks(segment)
        for item in tmp:
            output.append(item)
    return output


# %% local utils
def _principal_period(s):
    i = (s + s).find(s, 1, -1)
    return None if i == -1 else s[:i]


def _find_periodic_pattern(arr):
    numel = len(arr)
    loop = _principal_period(arr.tobytes())

    for start in range(numel):
        loop = _principal_period(arr[start:].tobytes())
        if loop is not None:
            break

    if loop is not None:
        loop = np.frombuffer(loop, dtype=int)
        remainder = np.frombuffer(arr[:start], dtype=int)
    else:
        loop = np.frombuffer(arr, dtype=int)
        remainder = np.asarray([], dtype=int)

    return loop, remainder


# Here we make the (I think) reasonable assumption that rotated blocks with different angles
# (e.g., two consecutive (spiral arm, spiral rewinder) pairs with different angles)
# are separated by non rotated blocks (e.g., RF pulses or z-spoilers)
def _split_signed_blocks(arr):
    if not arr:
        return []

    # Convert the input list to a numpy array
    arr = np.array(arr)

    # Create a sign array where positive numbers are marked as 1, negative as -1
    sign = np.sign(arr)
    sign[sign == 0] = 1

    # Find the indices where the sign changes
    change_indices = np.where(np.diff(sign) != 0)[0] + 1

    # Split the array at the change indices
    split_blocks = np.split(arr, change_indices)

    # Convert the numpy arrays in the list to python lists
    result = [list(abs(block)) for block in split_blocks]

    # Get sizes of each block
    osize = np.asarray([len(block) for block in result])
    max_size = max(osize)
    tmp = [np.pad(result[n], (0, max_size - osize[n])) for n in range(len(result))]
    tmp = np.stack(tmp, axis=0)
    tmp, idx = np.unique(tmp, return_index=True, axis=0)
    tmp = tmp[np.argsort(idx)]
    osize = osize[np.sort(idx)]
    result = [tmp[n][: osize[n]].tolist() for n in range(len(osize))]

    return result
