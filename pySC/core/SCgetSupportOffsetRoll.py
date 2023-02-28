import numpy as np
from pySC.constants import SUPPORT_TYPES
from typing import Tuple
from numpy import ndarray
from pySC.classes import SimulatedComissioning


def SCgetSupportOffset(SC: SimulatedComissioning, s: ndarray) -> ndarray:  # Just as reference, not used
    offsets, rolls = support_offset_and_roll(SC, s)
    return offsets


def SCgetSupportRoll(SC: SimulatedComissioning, s: ndarray) -> ndarray:  # Just as reference, not used
    offsets, rolls = support_offset_and_roll(SC, s)
    return rolls


def support_offset_and_roll(SC: SimulatedComissioning, s_locations: ndarray) -> Tuple[ndarray, ndarray]:
    lengths = np.array([SC.RING[i].Length for i in range(len(SC.RING))])
    ring_length = np.sum(lengths)
    s0 = np.cumsum(lengths)
    sposMID = s0 - lengths / 2
    off0 = np.zeros((3, len(s0)))
    roll0 = np.zeros((3, len(s0)))

    for suport_type in SUPPORT_TYPES:  # Order has to be Section, Plinth, Girder
        if suport_type in SC.ORD:
            ord1 = SC.ORD[suport_type][0, :]  # Beginning ordinates
            ord2 = SC.ORD[suport_type][1, :]  # End ordinates
            s1 = sposMID[ord1]
            s2 = sposMID[ord2]
            tmpoff1 = np.zeros((3, len(ord1)))
            tmpoff2 = np.zeros((3, len(ord2)))
            for i in range(len(ord1)):
                tmpoff1[:, i] = off0[:, ord1[i]] + getattr(SC.RING[ord1[i]], f"{suport_type}Offset")
                tmpoff2[:, i] = off0[:, ord2[i]] + getattr(SC.RING[ord2[i]], f"{suport_type}Offset")
            for i in range(3):
                off0[i, :] = _interpolation(off0[i, :], s0, ring_length, s1, ord1, tmpoff1[i, :], s2, ord2, tmpoff2[i, :])

    for support_type in SUPPORT_TYPES:  # Order has to be Section, Plinth, Girder
        for ords in SC.ORD[support_type].T:
            roll_start0 = getattr(SC.RING[ords[0]], f"{support_type}Roll")[0]
            struct_length = s0[ords[1]] - s0[ords[0]]
            mask = np.zeros(len(s0), dtype=bool)
            mask[ords[0]:ords[1]] = True
            offset1 = off0[1, ords[1]] - off0[1, ords[0]]
            offset2 = off0[0, ords[1]] - off0[0, ords[0]]
            if ords[0] > ords[1]:
                struct_length = ring_length + struct_length
                mask[ords[0]] = False
                mask = ~mask
            else:
                mask[ords[1]] = True
            roll0[0, mask] += roll_start0
            roll0[1, mask] = offset1 / struct_length
            roll0[2, mask] = offset2 / struct_length

    if not np.array_equal(s_locations, s0):
        b = np.unique(s0, return_index=True)[1]
        off, roll = np.empty((3, len(s_locations))), np.empty((3, len(s_locations)))
        for i in range(3):
            off[i, :] = np.interp(s_locations, s0[b], off0[i, b])
            roll[i, :] = np.interp(s_locations, s0[b], roll0[i, b])
        return off, roll
    return off0, roll0


def _interpolation(off, s, C, s1, ord1, f1, s2, ord2, f2):
    for n in range(len(s1)):
        if s1[n] == s2[n]:  # Sampling points have same s-position
            if f1[n] != f2[n]:
                raise ValueError('Something went wrong.')
            ind = np.arange(ord1[n], ord2[n] + 1)
            off[ind] = f1[n]
        elif s1[n] < s2[n]:  # Standard interpolation
            ind = np.arange(ord1[n], ord2[n] + 1)  # last point included
            off[ind] = np.interp(s[ind], np.array([s1[n], s2[n]]), np.array([f1[n], f2[n]]))
        else:  # Injection is within sampling points
            ind1 = np.arange(ord2[n] + 1)
            ind2 = np.arange(ord1[n], len(off) + 1)
            off[ind1] = np.interp(C + s[ind1], np.array([s1[n], s2[n] + C]), np.array([f1[n], f2[n]]))
            off[ord1[n]:] = np.interp(s[ord1[n]:], np.array([s1[n], s2[n] + C]), np.array([f1[n], f2[n]]))
    return off
