from at import Lattice
from numpy import ndarray

from pySC.classes import SimulatedComissioning


def SCinit(RING: Lattice) -> SimulatedComissioning:
    return SimulatedComissioning(RING)


def SCregisterBPMs(SC: SimulatedComissioning, BPMords: ndarray, **kwargs) -> SimulatedComissioning:
    SC.register_bpms(ords=BPMords, **kwargs)
    return SC


def SCregisterCAVs(SC: SimulatedComissioning, CAVords: ndarray, **kwargs) -> SimulatedComissioning:
    SC.register_cavities(ords=CAVords, **kwargs)
    return SC


def SCregisterMagnets(SC: SimulatedComissioning, MAGords: ndarray, **kwargs) -> SimulatedComissioning:
    SC.register_magnets(ords=MAGords, **kwargs)
    return SC


def SCregisterSupport(SC: SimulatedComissioning, support_ords: ndarray, support_type: str,  **kwargs) -> SimulatedComissioning:
    SC.register_supports(support_ords=support_ords, support_type=support_type, **kwargs)
    return SC


def SCapplyErrors(SC: SimulatedComissioning, nsigmas: float = 2) -> SimulatedComissioning:
    SC.apply_errors(nsigmas=nsigmas)
    return SC


def SCupdateCAVs(SC: SimulatedComissioning, ords: ndarray = None) -> SimulatedComissioning:
    SC.update_cavities(ords=ords)
    return SC


def SCupdateMagnets(SC: SimulatedComissioning, ords: ndarray = None) -> SimulatedComissioning:
    SC.update_magnets(ords=ords)
    return SC


def SCupdateSupport(SC: SimulatedComissioning, BPMstructOffset: bool = True, MAGstructOffset: bool = True) -> SimulatedComissioning:
    SC.update_supports(offset_bpms=BPMstructOffset, offset_magnets=MAGstructOffset)
    return SC

