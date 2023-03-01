import numpy as np
from at import Lattice
from numpy import ndarray
from pySC.at_wrapper import findspos
from pySC.classes import SimulatedComissioning
from pySC.constants import RF_PROPERTIES
from pySC.core.SCgetSupportOffsetRoll import support_offset_and_roll
from pySC.core.SCgetTransformation import SCgetTransformation


def SCinit(RING: Lattice) -> SimulatedComissioning:
    return SimulatedComissioning(RING)


def SCregisterBPMs(SC: SimulatedComissioning, BPMords: ndarray, **kwargs) -> SimulatedComissioning:
    SC.register_bpms(BPMords=BPMords, **kwargs)
    return SC


def SCregisterCAVs(SC: SimulatedComissioning, CAVords: ndarray, **kwargs) -> SimulatedComissioning:
    SC.register_cavities(CAVords=CAVords, **kwargs)
    return SC


def SCregisterMagnets(SC: SimulatedComissioning, MAGords: ndarray, **kwargs) -> SimulatedComissioning:
    SC.register_magnets(MAGords=MAGords, **kwargs)
    return SC


def SCregisterSupport(SC: SimulatedComissioning, support_ords: ndarray, support_type: str,  **kwargs) -> SimulatedComissioning:
    SC.register_supports(support_ords=support_ords, support_type=support_type, **kwargs)
    return SC


def SCupdateCAVs(SC: SimulatedComissioning, ords: ndarray = None) -> SimulatedComissioning:
    for ord in (SC.ORD.RF if ords is None else ords):
        for field in RF_PROPERTIES:
            setattr(SC.RING[ord], field,
                    getattr(SC.RING[ord], f"{field}SetPoint")
                    * (1 + getattr(SC.RING[ord], f"{field}CalError"))
                    + getattr(SC.RING[ord], f"{field}Offset"))
    return SC


def SCupdateMagnets(SC: SimulatedComissioning, ords: ndarray = None) -> SimulatedComissioning:
    for ord in (SC.ORD.Magnet if ords is None else ords):
        SC = _updateMagnets(SC, ord, ord)
        if hasattr(SC.RING[ord], 'MasterOf'):
            for childOrd in SC.RING[ord].MasterOf:
                SC = _updateMagnets(SC, ord, childOrd)
    return SC


def SCupdateSupport(SC: SimulatedComissioning, BPMstructOffset: bool = True, MAGstructOffset: bool = True) -> SimulatedComissioning:
    if MAGstructOffset:
        if len(SC.ORD.Magnet):
            s = findspos(SC.RING, SC.ORD.Magnet)
            offsets, rolls = support_offset_and_roll(SC, s)
            for i, ord in enumerate(SC.ORD.Magnet):
                setattr(SC.RING[ord], "SupportOffset", offsets[:, i])
                setattr(SC.RING[ord], "SupportRoll", rolls[:, i])
                magLength = SC.RING[ord].Length
                magTheta = SC.RING[ord].BendingAngle if hasattr(SC.RING[ord], 'BendingAngle') else 0
                magnet_offsets = SC.RING[ord].SupportOffset + SC.RING[ord].MagnetOffset
                magnet_rolls = np.roll(SC.RING[ord].MagnetRoll + SC.RING[ord].SupportRoll, -1)  # z,x,y -> x,y,z
                SC.RING[ord].T1, SC.RING[ord].T2, SC.RING[ord].R1, SC.RING[ord].R2 = SCgetTransformation(
                    magnet_offsets, magnet_rolls, magTheta, magLength)
                if hasattr(SC.RING[ord], 'MasterOf'):
                    for childOrd in SC.RING[ord].MasterOf:
                        for field in ("T1", "T2", "R1", "R2"):
                            setattr(SC.RING[childOrd], field, getattr(SC.RING[ord], field))

        else:
            print('SC: No magnets have been registered!')
    if BPMstructOffset:
        if len(SC.ORD.BPM):
            s = findspos(SC.RING, SC.ORD.BPM)
            offsets, rolls = support_offset_and_roll(SC, s)
            for i, ord in enumerate(SC.ORD.BPM):
                setattr(SC.RING[ord], "SupportOffset", offsets[0:2, i])  # Longitudinal BPM offsets not implemented
                setattr(SC.RING[ord], "SupportRoll", np.array([rolls[0, i]]))  # BPM pitch and yaw angles not  implemented
        else:
            print('SC: No BPMs have been registered!')
    return SC


def _updateMagnets(SC, source, target):  # TODO simplify AB calculated in place
    SC.RING[target].PolynomB = SC.RING[source].SetPointB * _add_padded(np.ones(len(SC.RING[source].SetPointB)),
                                                                       SC.RING[source].CalErrorB)
    SC.RING[target].PolynomA = SC.RING[source].SetPointA * _add_padded(np.ones(len(SC.RING[source].SetPointA)),
                                                                       SC.RING[source].CalErrorA)
    sysPolynomB = []
    sysPolynomA = []
    if hasattr(SC.RING[target], 'SysPolBFromB'):
        for n in range(len(SC.RING[target].SysPolBFromB)):
            if SC.RING[target].SysPolBFromB[n] is not None:
                sysPolynomB.append(SC.RING[target].PolynomB[n] * SC.RING[target].SysPolBFromB[n])
    if hasattr(SC.RING[target], 'SysPolBFromA'):
        for n in range(len(SC.RING[target].SysPolBFromA)):
            if SC.RING[target].SysPolBFromA[n] is not None:
                sysPolynomB.append(SC.RING[target].PolynomA[n] * SC.RING[target].SysPolBFromA[n])
    if hasattr(SC.RING[target], 'SysPolAFromB'):
        for n in range(len(SC.RING[target].SysPolAFromB)):
            if SC.RING[target].SysPolAFromB[n] is not None:
                sysPolynomA.append(SC.RING[target].PolynomB[n] * SC.RING[target].SysPolAFromB[n])
    if hasattr(SC.RING[target], 'SysPolAFromA'):
        for n in range(len(SC.RING[target].SysPolAFromA)):
            if SC.RING[target].SysPolAFromA[n] is not None:
                sysPolynomA.append(SC.RING[target].PolynomA[n] * SC.RING[target].SysPolAFromA[n])
    if len(sysPolynomA) > 0:
        for n in range(len(sysPolynomA) - 1):
            sysPolynomA[n + 1] = _add_padded(sysPolynomA[n + 1], sysPolynomA[n])
        SC.RING[target].PolynomA = _add_padded(SC.RING[target].PolynomA, sysPolynomA[-1])
    if len(sysPolynomB) > 0:
        for n in range(len(sysPolynomB) - 1):
            sysPolynomB[n + 1] = _add_padded(sysPolynomB[n + 1], sysPolynomB[n])
        SC.RING[target].PolynomB = _add_padded(SC.RING[target].PolynomB, sysPolynomB[-1])
    if hasattr(SC.RING[target], 'PolynomBOffset'):
        SC.RING[target].PolynomB = _add_padded(SC.RING[target].PolynomB, SC.RING[target].PolynomBOffset)
        SC.RING[target].PolynomA = _add_padded(SC.RING[target].PolynomA, SC.RING[target].PolynomAOffset)
    if hasattr(SC.RING[source], 'BendingAngleError'):
        SC.RING[target].PolynomB[0] = SC.RING[target].PolynomB[0] + SC.RING[source].BendingAngleError * SC.RING[
            target].BendingAngle / SC.RING[target].Length
    if hasattr(SC.RING[source], 'BendingAngle'):
        if hasattr(SC.RING[source], 'CombinedFunction') and SC.RING[source].CombinedFunction == 1:
            alpha_act = SC.RING[source].SetPointB[1] * (1 + SC.RING[source].CalErrorB[1]) / SC.RING[source].NomPolynomB[
                1]
            effBendingAngle = alpha_act * SC.RING[target].BendingAngle
            SC.RING[target].PolynomB[0] = SC.RING[target].PolynomB[0] + (
                    effBendingAngle - SC.RING[target].BendingAngle) / SC.RING[target].Length
    if SC.RING[source].PassMethod == 'CorrectorPass':
        SC.RING[target].KickAngle[0] = SC.RING[target].PolynomB[0]
        SC.RING[target].KickAngle[1] = SC.RING[target].PolynomA[0]
    SC.RING[target].MaxOrder = len(SC.RING[target].PolynomB) - 1
    return SC


def _add_padded(v1, v2):
    if v1.ndim != v2.ndim:
        raise ValueError(f'Unmatched number of dimensions {v1.ndim} and {v2.ndim}.')
    max_dims = np.array([max(d1, d2) for d1, d2 in zip(v1.shape, v2.shape)])
    if np.sum(max_dims > 1) > 1:
        raise ValueError(f'Wrong or mismatching dimensions {v1.shape} and {v2.shape}.')
    vsum = np.zeros(np.prod(max_dims))
    vsum[:np.max(v1.shape)] += v1
    vsum[:np.max(v2.shape)] += v2
    return vsum
