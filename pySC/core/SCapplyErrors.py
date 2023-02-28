import re
import numpy as np
from pySC.constants import SUPPORT_TYPES
from pySC.core.SCregisterUpdate import SCupdateCAVs, SCupdateMagnets, SCupdateSupport
from pySC.core.SCrandnc import SCrandnc
from pySC.core.SCscaleCircumference import SCscaleCircumference
from pySC.at_wrapper import findspos

from pySC.classes import SimulatedComissioning


def SCapplyErrors(SC: SimulatedComissioning, nsigmas: float = 2) -> SimulatedComissioning:
    SC = _apply_cavity_error(SC, nsigmas)
    SC = _apply_injection_error(SC, nsigmas)
    SC = _apply_bpm_errors(SC, nsigmas)
    SC = _apply_circumference_error(SC, nsigmas)
    SC = _apply_support_alignment_error(SC, nsigmas)
    SC = _apply_magnet_error(SC, nsigmas)
    SC = SCupdateSupport(SC)
    if len(SC.ORD.Magnet):
        SC = SCupdateMagnets(SC)
    if len(SC.ORD.Cavity) and len(SC.SIG.RF):
        SC = SCupdateCAVs(SC)
    return SC


def _apply_cavity_error(SC, nsigmas):
    for ord in _intersect(SC.ORD.Cavity, SC.SIG.RF.keys()):
        for field in SC.SIG.RF[ord]:
            setattr(SC.RING[ord], field, _randn_cutoff(SC.SIG.RF[ord][field], nsigmas))
    return SC


def _apply_injection_error(SC, nsigmas):
    if 'staticInjectionZ' in SC.SIG.keys():
        SC.INJ.Z0 = SC.INJ.Z0ideal + SC.SIG.staticInjectionZ * SCrandnc(nsigmas, (6,))
        print('Static injection error applied.')
    if 'randomInjectionZ' in SC.SIG.keys():
        SC.INJ.randomInjectionZ = SC.SIG.randomInjectionZ[:]
        print('Random injection error applied.')
    return SC


def _apply_bpm_errors(SC, nsigmas):
    for ord in _intersect(SC.ORD.BPM, SC.SIG.BPM.keys()):
        for field in SC.SIG.BPM[ord]:
            if re.search('Noise', field):
                setattr(SC.RING[ord], field, SC.SIG.BPM[ord][field])
            else:
                setattr(SC.RING[ord], field, _randn_cutoff(SC.SIG.BPM[ord][field], nsigmas))
    return SC


def _apply_circumference_error(SC, nsigmas):
    if 'Circumference' in SC.SIG.keys():
        circScaling = 1 + SC.SIG.Circumference * SCrandnc(nsigmas, (1, 1))
        SC.RING = SCscaleCircumference(SC.RING, circScaling, 'rel')
        print('Circumference error applied.')
    return SC


def _apply_support_alignment_error(SC, nsigmas):
    for support_type in SUPPORT_TYPES:
        for ordPair in SC.ORD[support_type].T:
            if ordPair[0] not in SC.SIG.Support.keys():
                continue
            for field, value in SC.SIG.Support[ordPair[0]].items():
                if support_type not in field:
                    continue
                setattr(SC.RING[ordPair[0]], field, _randn_cutoff(value, nsigmas))
                setattr(SC.RING[ordPair[1]], field, _randn_cutoff(value, nsigmas) if field in SC.SIG.Support[ordPair[1]].keys()
                        else getattr(SC.RING[ordPair[0]], field))

            struct_length = np.diff(findspos(SC.RING, ordPair))
            if ordPair[0] > ordPair[1]:
                struct_length = findspos(SC.RING, len(SC.RING))[0] - struct_length

            rolls0 = getattr(SC.RING[ordPair[0]], f"{support_type}Roll")
            offsets0 = getattr(SC.RING[ordPair[0]], f"{support_type}Offset")
            rolls1 = getattr(SC.RING[ordPair[1]], f"{support_type}Roll")  # TODO Not implemented?
            offsets1 = getattr(SC.RING[ordPair[1]], f"{support_type}Offset")

            if rolls0[1] != 0:
                if f"{support_type}Offset" in SC.SIG.Support[ordPair[1]].keys():
                    raise Exception(f'Pitch angle errors can not be given explicitly if {support_type} start and endpoints '
                                    f'each have offset uncertainties.')
                offsets0[1] -= rolls0[1] * struct_length / 2
                offsets1[1] += rolls0[1] * struct_length / 2

            else:
                rolls0[1] = (offsets1[1] - offsets0[1]) / struct_length
            if rolls0[2] != 0:
                if f"{support_type}Offset" in SC.SIG.Support[ordPair[1]].keys():
                    raise Exception(f'Yaw angle errors can not be given explicitly if {support_type} start and endpoints '
                                    f'each have offset uncertainties.')
                offsets0[0] -= rolls0[2] * struct_length / 2
                offsets1[0] += rolls0[2] * struct_length / 2
            else:
                rolls0[2] = (offsets1[0] - offsets0[0]) / struct_length
            setattr(SC.RING[ordPair[0]], f"{support_type}Roll", rolls0)
            setattr(SC.RING[ordPair[0]], f"{support_type}Offset", offsets0)
            #setattr(SC.RING[ordPair[1]], f"{support_type}Roll", rolls1)
            setattr(SC.RING[ordPair[1]], f"{support_type}Offset", offsets1)
    return SC


def _apply_magnet_error(SC, nsigmas):
    for ord in _intersect(SC.ORD.Magnet, SC.SIG.Mag.keys()):
        for field in SC.SIG.Mag[ord]:
            setattr(SC.RING[ord], 'BendingAngleError' if field == 'BendingAngle' else field,
                    _randn_cutoff(SC.SIG.Mag[ord][field], nsigmas))
    return SC


def _intersect(primary, secondary):
    return [elem for elem in primary if elem in secondary]


def _randn_cutoff(field, default_cut_off):
    if isinstance(field, list):
        return field[0] * SCrandnc(field[1], np.shape(field[0]))
    return field * SCrandnc(default_cut_off, np.shape(field))
