import numpy as np
from pySC.constants import SUPPORT_TYPES
from pySC.core.SCgetSupportOffset import SCgetSupportOffset


def SCgetSupportRoll(SC, s):
    lengths = np.array([SC.RING[i].Length for i in range(len(SC.RING))])
    C = np.sum(lengths)
    s0 = np.cumsum(lengths)
    roll0 = np.zeros((3, len(s0)))
    off0 = SCgetSupportOffset(SC, s0)

    for type in SUPPORT_TYPES:  # TODO order dependent
        for ords in SC.ORD[type].T:
            roll_start0 = getattr(SC.RING[ords[0]], f"{type}Roll")[0]
            struct_length = s0[ords[1]] - s0[ords[0]]
            offset1 = off0[1, ords[1]] - off0[1, ords[0]]
            offset2 = off0[0, ords[1]] - off0[0, ords[0]]
            if ords[0] > ords[1]:
                struct_length = C - struct_length
                roll0[0, ords[0]:] += roll_start0
                roll0[0, :ords[1]+1] += roll_start0
                roll0[1, ords[0]:] = -offset1 / struct_length
                roll0[1, :ords[1]+1] = -offset1 / struct_length
                roll0[2, ords[0]:] = -offset2 / struct_length
                roll0[2, :ords[1]+1] = -offset2 / struct_length
            else:
                roll0[0, ords[0]:ords[1]+1] += roll_start0
                roll0[1, ords[0]:ords[1]+1] = offset1 / struct_length  # TODO this is overwriting
                roll0[2, ords[0]:ords[1]+1] = offset2 / struct_length

    _, b = np.unique(s0, return_index=True)
    roll = np.empty((3, len(s)))
    roll[0, :] = np.interp(s, s0[b], roll0[0, b])
    roll[1, :] = np.interp(s, s0[b], roll0[1, b])
    roll[2, :] = np.interp(s, s0[b], roll0[2, b])
    return roll
