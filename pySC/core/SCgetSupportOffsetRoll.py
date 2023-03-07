from numpy import ndarray
from pySC.classes import SimulatedComissioning

def SCgetSupportOffset(SC: SimulatedComissioning, s: ndarray) -> ndarray:  # Just as reference, not used
    offsets, rolls = SC.support_offset_and_roll(s)
    return offsets


def SCgetSupportRoll(SC: SimulatedComissioning, s: ndarray) -> ndarray:  # Just as reference, not used
    offsets, rolls = SC.support_offset_and_roll(s)
    return rolls
