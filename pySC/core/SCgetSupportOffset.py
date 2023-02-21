import numpy as np
from pySC.constants import SUPPORT_TYPES

def SCgetSupportOffset(SC, s):
    lengths = np.array([SC.RING[i].Length for i in range(len(SC.RING))])
    C = np.sum(lengths)
    s0 = np.cumsum(lengths)
    sposMID = s0 - lengths / 2
    off0 = np.zeros((3, len(s0)))
    for suport_type in SUPPORT_TYPES:
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
            off0[0, :] = limp(off0[0, :], s0, C, s1, ord1, tmpoff1[0, :], s2, ord2, tmpoff2[0, :])
            off0[1, :] = limp(off0[1, :], s0, C, s1, ord1, tmpoff1[1, :], s2, ord2, tmpoff2[1, :])
            off0[2, :] = limp(off0[2, :], s0, C, s1, ord1, tmpoff1[2, :], s2, ord2, tmpoff2[2, :])
    if not np.array_equal(s, s0):
        b = np.unique(s0, return_index=True)[1]
        off = np.empty((3, len(s)))
        off[0, :] = np.interp(s, s0[b], off0[0, b])
        off[1, :] = np.interp(s, s0[b], off0[1, b])
        off[2, :] = np.interp(s, s0[b], off0[2, b])
        return off
    return off0


def limp(off, s, C, s1, ord1, f1, s2, ord2, f2):
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
            off[ind2] = np.interp(s[ind2], np.array([s1[n], s2[n] + C]), np.array([f1[n], f2[n]]))
    return off
# End
