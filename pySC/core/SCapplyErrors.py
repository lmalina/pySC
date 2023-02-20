import re
import numpy as np
from pySC.constants import SUPPORT_TYPES
from pySC.core.SCupdateSupport import SCupdateSupport
from pySC.core.SCupdateCAVs import SCupdateCAVs
from pySC.core.SCupdateMagnets import SCupdateMagnets
from pySC.core.SCrandnc import SCrandnc
from pySC.core.SCscaleCircumference import SCscaleCircumference
import at


def SCapplyErrors(SC, nSig: float = 2):
    SC = applyCavityError(SC, nSig)
    SC = applyInjectionError(SC, nSig)
    SC = applyBPMerrors(SC, nSig)
    SC = applyCircumferenceError(SC, nSig)
    SC = applySupportAlignmentError(SC, nSig)
    SC = applyMagnetError(SC, nSig)
    SC = SCupdateSupport(SC)
    if SC.ORD.Magnet:
        SC = SCupdateMagnets(SC)
    if SC.ORD.Cavity and SC.SIG.RF:
        SC = SCupdateCAVs(SC)
    return SC


def applyCavityError(SC, nSig):
    for ord in SC.ORD.Cavity:
        if not SC.SIG.RF[ord]:
            continue
        for field in SC.SIG.RF[ord]:
            if isinstance(SC.SIG.RF[ord][field], list):
                setattr(SC.RING[ord], field, SC.SIG.RF[ord][field][0] * SCrandnc(SC.SIG.RF[ord][field][1]))
            else:
                setattr(SC.RING[ord], field, SC.SIG.RF[ord][field] * SCrandnc(nSig))
    return SC


def applyInjectionError(SC, nSig):
    if 'staticInjectionZ' in SC.SIG.keys():
        SC.INJ.Z0 = SC.INJ.Z0ideal + SC.SIG.staticInjectionZ * SCrandnc(nSig, (6,))
        print('Static injection error applied.')
    if 'randomInjectionZ' in SC.SIG.keys():
        SC.INJ.randomInjectionZ = SC.SIG.randomInjectionZ[:]
        print('Random injection error applied.')
    return SC


def applyBPMerrors(SC, nSig):
    for ord in SC.ORD.BPM:
        if ord not in SC.SIG.BPM.keys():
            continue
        for field in SC.SIG.BPM[ord]:
            if re.search('Noise', field):
                setattr(SC.RING[ord], field, SC.SIG.BPM[ord][field])
            else:
                if isinstance(SC.SIG.BPM[ord][field], list):
                    setattr(SC.RING[ord], field, SC.SIG.BPM[ord][field][0] *
                            SCrandnc(SC.SIG.BPM[ord][field][1], np.shape(SC.SIG.BPM[ord][field][0])))
                else:
                    setattr(SC.RING[ord], field,
                            SC.SIG.BPM[ord][field] * SCrandnc(nSig, np.shape(SC.SIG.BPM[ord][field])))
    return SC


def applyCircumferenceError(SC, nSig):
    if 'Circumference' in SC.SIG.keys():
        circScaling = 1 + SC.SIG.Circumference * SCrandnc(nSig, (1, 1))
        SC.RING = SCscaleCircumference(SC.RING, circScaling, 'rel')
        print('Circumference error applied.')
    return SC


def rand_support(field, nSig):
    if isinstance(field, list):
        return field[0] * SCrandnc(field[1], np.shape(field[0]))
    return field * SCrandnc(nSig, np.shape(field))


def applySupportAlignmentError(SC, nSig):
    for support_type in SUPPORT_TYPES:
        for ordPair in SC.ORD[support_type].T:
            if ordPair[0] not in SC.SIG.Support.keys():
                continue
            for field, value in SC.SIG.Support[ordPair[0]].items():
                if support_type not in field:
                    continue
                setattr(SC.RING[ordPair[0]], field, rand_support(value, nSig))
                if len(SC.SIG.Support) >= ordPair[1] and field in SC.SIG.Support[ordPair[1]].keys():  # TODO also rather strange condition
                    setattr(SC.RING[ordPair[1]], field, rand_support(value, nSig))
                else:
                    setattr(SC.RING[ordPair[1]], field, getattr(SC.RING[ordPair[0]], field))

            if ordPair[1] - ordPair[0] >= 0:
                structLength = np.abs(np.diff(at.get_s_pos(SC.RING, ordPair)))
            else:
                structLength = at.get_s_pos(SC.RING, ordPair[1]) + np.diff(
                    at.get_s_pos(SC.RING, np.array([ordPair[0], len(SC.RING) + 1])))
                # TODO strange. I would expect something like
                #  lentgth = np.diff(at.get_s_pos(SC.RING, ordPair))
                #  if ordPair[0] > ordPair[1]:
                #      length = C-length


            rolls0 = getattr(SC.RING[ordPair[0]], f"{support_type}Roll")
            offsets0 = getattr(SC.RING[ordPair[0]], f"{support_type}Offset")
            rolls1 = getattr(SC.RING[ordPair[1]], f"{support_type}Roll")  # Not implemented?
            offsets1 = getattr(SC.RING[ordPair[1]], f"{support_type}Offset")

            if rolls0[1] != 0:
                if len(SC.SIG.Support) >= ordPair[1] and f"{support_type}Offset" in SC.SIG.Support[ordPair[1]].keys():
                    raise Exception(f'Pitch angle errors can not be given explicitly if {support_type} start and endpoints '
                                    f'each have offset uncertainties.')
                offsets0[1] -= rolls0[1] * structLength / 2
                offsets1[1] += rolls0[1] * structLength / 2

            else:
                rolls0[1] = (offsets1[1] - offsets0[1]) / structLength
            if rolls0[2] != 0:
                if len(SC.SIG.Support) >= ordPair[1] and f"{support_type}Offset" in SC.SIG.Support[ordPair[1]].keys():
                    raise Exception(f'Yaw angle errors can not be given explicitly if {support_type} start and endpoints '
                                    f'each have offset uncertainties.')
                offsets0[0] -= rolls0[2] * structLength / 2
                offsets1[0] += rolls0[2] * structLength / 2
            else:
                rolls0[2] = (offsets1[0] - offsets0[0]) / structLength
            setattr(SC.RING[ordPair[0]], f"{support_type}Roll", rolls0)
            setattr(SC.RING[ordPair[0]], f"{support_type}Offset", offsets0)
            #setattr(SC.RING[ordPair[1]], f"{support_type}Roll", rolls1)
            setattr(SC.RING[ordPair[1]], f"{support_type}Offset", offsets1)
    return SC


def applyMagnetError(SC, nSig):
    for ord in SC.ORD.Magnet:
        if ord not in SC.SIG.Mag.keys():
            continue
        for field in SC.SIG.Mag[ord]:
            if isinstance(SC.SIG.Mag[ord][field], list):
                cut_off = SC.SIG.Mag[ord][field][1]
                sig = SC.SIG.Mag[ord][field][0]
            else:
                cut_off = nSig
                sig = SC.SIG.Mag[ord][field]
            if field == 'BendingAngle':
                setattr(SC.RING[ord], 'BendingAngleError', sig * SCrandnc(cut_off, (1, 1)))
            else:
                setattr(SC.RING[ord], field, sig * SCrandnc(cut_off, np.shape(sig)))
    return SC
