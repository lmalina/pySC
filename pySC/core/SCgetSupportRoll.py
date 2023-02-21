import numpy as np
from pySC.constants import SUPPORT_TYPES
from pySC import findspos
from pySC.core.SCgetSupportOffset import SCgetSupportOffset


def SCgetSupportRoll(SC, s):
    lengths = np.array([SC.RING[i].Length for i in range(len(SC.RING))])
    C = np.sum(lengths)
    s0 = np.cumsum(lengths)
    roll0 = np.zeros((3, len(s0)))
    off0 = SCgetSupportOffset(SC, s0)
    for type in SUPPORT_TYPES:
        for nEl in range(SC.ORD[type].shape[1]):
            ords = SC.ORD[type][:, nEl]
            if ords[1] - ords[0] > 0:
                roll0[0, ords[0]:ords[1]] = roll0[0, ords[0]:ords[1]] + getattr(SC.RING[ords[0]], f"{type}Roll")[0]
                roll0[1, ords[0]:ords[1]] = (off0[1, ords[1]] - off0[1, ords[0]]) / (s0[ords[1]] - s0[ords[0]])
                roll0[2, ords[0]:ords[1]] = (off0[0, ords[1]] - off0[0, ords[0]]) / (s0[ords[1]] - s0[ords[0]])
            else:
                roll0[0, ords[0]:] = roll0[0, ords[0]:] + getattr(SC.RING[ords[0]], f"{type}Roll")[0]
                roll0[0, :ords[1]] = roll0[0, :ords[1]] + getattr(SC.RING[ords[0]], f"{type}Roll")[0]
                roll0[1, ords[0]:] = (off0[1, ords[1]] - off0[1, ords[0]]) / (C - s0[ords[0]] - s0[ords[1]])
                roll0[1, :ords[1]] = (off0[1, ords[1]] - off0[1, ords[0]]) / (C - s0[ords[0]] - s0[ords[1]])
                roll0[2, ords[0]:] = (off0[0, ords[1]] - off0[0, ords[0]]) / (C - s0[ords[0]] - s0[ords[1]])
                roll0[2, :ords[1]] = (off0[0, ords[1]] - off0[0, ords[0]]) / (C - s0[ords[0]] - s0[ords[1]])
    _, b = np.unique(s0, return_index=True)
    roll = np.empty((3, len(s)))
    roll[0, :] = np.interp(s, s0[b], roll0[0, b])
    roll[1, :] = np.interp(s, s0[b], roll0[1, b])
    roll[2, :] = np.interp(s, s0[b], roll0[2, b])
    return roll
