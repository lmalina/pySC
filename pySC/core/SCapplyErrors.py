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
    if not SC.SIG:
        print('No uncertanties provided.')
        return SC
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
    if 'staticInjectionZ' in SC.SIG:
        SC.INJ['Z0'] = SC.INJ['Z0ideal'] + SC.SIG['staticInjectionZ'][:] * SCrandnc(nSig, (6, 1))
        print('Static injection error applied.')
    if 'randomInjectionZ' in SC.SIG:
        SC.INJ.randomInjectionZ = SC.SIG.randomInjectionZ[:]
        print('Random injection error applied.')
    return SC


def applyBPMerrors(SC, nSig):
    if 'BPM' not in SC.SIG:
        return
    for ord in SC.ORD.BPM:
        if not SC.SIG.BPM[ord]:
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
    if 'Circumference' in SC.SIG:
        circScaling = 1 + SC.SIG['Circumference'] * SCrandnc(nSig, (1, 1))
        SC.RING = SCscaleCircumference(SC.RING, circScaling, 'rel')
        print('Circumference error applied.')
    return SC


def applySupportAlignmentError(SC, nSig):
    for type in SUPPORT_TYPES:
        if type not in SC.ORD:
            continue
        for ordPair in SC.ORD[type].T:
            if not SC.SIG['Support'][ordPair[0]]:
                continue
            for field in SC.SIG['Support'][ordPair[0]]:
                if type not in field:
                    continue
                if isinstance(SC.SIG['Support'][ordPair[0]][field], list):
                    setattr(SC.RING[ordPair[0]], field, SC.SIG['Support'][ordPair[0]][field][0] *
                            SCrandnc(SC.SIG['Support'][ordPair[0]][field][1],
                                     np.shape(SC.SIG['Support'][ordPair[0]][field][0])))
                else:
                    setattr(SC.RING[ordPair[0]], field, SC.SIG['Support'][ordPair[0]][field] *
                            SCrandnc(nSig, np.shape(SC.SIG['Support'][ordPair[0]][field])))
                if len(SC.SIG['Support']) >= ordPair[1] and field in SC.SIG['Support'][ordPair[1]]:
                    if isinstance(SC.SIG['Support'][ordPair[1]][field], list):
                        setattr(SC.RING[ordPair[1]], field, SC.SIG['Support'][ordPair[1]][field][0] *
                                SCrandnc(SC.SIG['Support'][ordPair[1]][field][1],
                                         np.shape(SC.SIG['Support'][ordPair[1]][field][0])))
                    else:
                        setattr(SC.RING[ordPair[1]], field, SC.SIG['Support'][ordPair[1]][field] *
                                SCrandnc(nSig, np.shape(SC.SIG['Support'][ordPair[1]][field])))
                else:
                    setattr(SC.RING[ordPair[1]], field, getattr(SC.RING[ordPair[0]], field))
            if ordPair[1] - ordPair[0] >= 0:
                structLength = np.abs(np.diff(at.get_s_pos(SC.RING, ordPair)))
            else:
                structLength = at.get_s_pos(SC.RING, ordPair[1]) + np.diff(
                    at.get_s_pos(SC.RING, np.array([ordPair[0], len(SC.RING) + 1])))
            if getattr(SC.RING[ordPair[0]], f"{type}Roll")[1] != 0:
                if (len(SC.SIG['Support']) >= ordPair[1] and
                        SC.SIG['Support'][ordPair[1]] != [] and
                        f"{type}Offset" in SC.SIG['Support'][ordPair[1]]):
                    raise Exception(f'Pitch angle errors can not be given explicitly if {type} start and endpoints '
                                    f'each have offset uncertainties.')

                setattr(SC.RING[ordPair[0]], f"{type}Offset"[1], getattr(SC.RING[ordPair[0]], f"{type}Offset")[1] -
                        getattr(SC.RING[ordPair[0]], f"{type}Roll")[1] * structLength / 2)
                setattr(SC.RING[ordPair[1]], f"{type}Offset"[1], getattr(SC.RING[ordPair[1]], f"{type}Offset")[1] +
                        getattr(SC.RING[ordPair[0]], f"{type}Roll")[1] * structLength / 2)
            else:
                setattr(SC.RING[ordPair[0]], f"{type}Roll"[1], (getattr(SC.RING[ordPair[1]], f"{type}Offset"[1]) -
                                                                getattr(SC.RING[ordPair[0]],
                                                                        f"{type}Offset"[1])) / structLength)
            if getattr(SC.RING[ordPair[0]], f"{type}Roll")[2] != 0:
                if len(SC.SIG['Support']) >= ordPair[1] and SC.SIG['Support'][ordPair[1]] != [] and type + 'Offset' in \
                        SC.SIG['Support'][ordPair[1]]:
                    raise Exception(f'Yaw angle errors can not be given explicitly if {type} start and endpoints '
                                    f'each have offset uncertainties.')
                SC.RING[ordPair[0]][type + 'Offset'][0] = SC.RING[ordPair[0]][type + 'Offset'][0] - \
                                                          SC.RING[ordPair[0]][type + 'Roll'][2] * structLength / 2
                SC.RING[ordPair[1]][type + 'Offset'][0] = SC.RING[ordPair[1]][type + 'Offset'][0] + \
                                                          SC.RING[ordPair[0]][type + 'Roll'][2] * structLength / 2
            else:
                SC.RING[ordPair[0]][type + 'Roll'][2] = (SC.RING[ordPair[1]][type + 'Offset'][0] -
                                                         SC.RING[ordPair[0]][type + 'Offset'][0]) / structLength
    return SC


def applyMagnetError(SC, nSig):
    if 'Mag' not in SC.SIG:
        return
    for ord in SC.ORD['Magnet']:
        if not SC.SIG['Mag'][ord]:
            continue
        for field in SC.SIG['Mag'][ord]:
            if isinstance(SC.SIG['Mag'][ord][field], list):
                nSig = SC.SIG['Mag'][ord][field][1]
                sig = SC.SIG['Mag'][ord][field][0]
            else:
                nSig = nSig
                sig = SC.SIG['Mag'][ord][field]
            if field == 'BendingAngle':
                setattr(SC.RING[ord], 'BendingAngleError', sig * SCrandnc(nSig, (1, 1)))
            else:
                setattr(SC.RING[ord], field, sig * SCrandnc(nSig, np.shape(sig)))
    return SC
