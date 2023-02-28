import numpy as np
from pySC.constants import SUPPORT_TYPES, RF_PROPERTIES
from pySC.core.SCfeedbackRun import SCfeedbackRun
from pySC.core.SCgetModelRM import SCgetModelRM
from pySC.core.SCgetPinv import SCgetPinv
from pySC.core.SCscaleCircumference import SCscaleCircumference
from pySC.core.SCregisterUpdate import SCupdateMagnets, SCupdateSupport


def SCrampUpErrors(SC, nStepsRamp=10, eps=1e-5, target=0, alpha=10, maxsteps=30, verbose=0):
    errFieldsMag = ['CalErrorB', 'CalErrorA', 'PolynomAOffset', 'PolynomBOffset', 'MagnetOffset', 'MagnetRoll']
    errFieldsSup = ['Roll', 'Offset']
    errFieldsBPM = ['Noise', 'NoiseCO', 'Offset', 'SupportOffset', 'Roll', 'SupportRoll', 'CalError']
    errFieldsRF = ['Offset', 'CalError']
    SC0 = SC
    M = SCgetModelRM(SC, SC.ORD.BPM, SC.ORD.CM, nTurns=SC.INJ.nTurns, trackMode=SC.INJ.trackMode)
    Mplus = SCgetPinv(M, alpha=alpha)
    for scale in np.linspace(1 / nStepsRamp, 1, nStepsRamp):
        if verbose:
            print('Ramping up errors with scaling factor %.2f.' % scale)
        SC = scaleSupport(SC, SC0, errFieldsSup, scale)
        SC = scaleMagnets(SC, SC0, errFieldsMag, scale)
        SC = scaleBPMs(SC, SC0, errFieldsBPM, scale)
        SC = scaleRF(SC, SC0, errFieldsRF, scale)
        SC = scaleInjection(SC, SC0, scale)
        SC = scaleCircumference(SC, SC0, scale)
        try:
            SC = SCfeedbackRun(SC, Mplus, target=target, maxsteps=maxsteps, eps=eps, verbose=verbose, plotFunctionFlag=True)
        except RuntimeError:
            if 2 * nStepsRamp > 100:
                raise Exception(f'Ramping up failed at scaling {scale:.2f} with {nStepsRamp} ramping steps. '
                                f'Try different feedback parameters.')
            else:
                print(f'Feedback did not succeed at scaling {scale:.2f}. Trying with {2 * nStepsRamp} ramping steps.')
                SC = SCrampUpErrors(SC0, nStepsRamp=2 * nStepsRamp, eps=eps, target=target, alpha=alpha,
                                    maxsteps=maxsteps, verbose=verbose)

    return SC


def scaleSupport(SC, SC0, fields, scale):
    for type in SUPPORT_TYPES:
        if type in SC.ORD:
            for ordPair in SC.ORD[type]:
                for field in fields:
                    for se in range(2):
                        if field in SC.RING[ordPair[se]][type]:
                            SC.RING[ordPair[se]][type][field] = scale * SC0.RING[ordPair[se]][type][field]
    return SC


def scaleMagnets(SC, SC0, fields, scale):
    for ord in SC.ORD.Magnet:
        for field in fields:
            if field in SC.RING[ord]:
                SC.RING[ord][field] = scale * SC0.RING[ord][field]
    SC = SCupdateSupport(SC)
    SC = SCupdateMagnets(SC)
    return SC


def scaleBPMs(SC, SC0, fields, scale):
    for ord in SC.ORD.BPM:
        for field in fields:
            if field in SC.RING[ord]:
                SC.RING[ord][field] = scale * SC0.RING[ord][field]
    return SC


def scaleRF(SC, SC0, fields, scale):
    for type in RF_PROPERTIES:
        for field in fields:
            for ord in SC.ORD.Cavity:
                SC.RING[ord][type + field] = scale * SC0.RING[ord][type + field]
    return SC


def scaleInjection(SC, SC0, scale):
    SC.INJ.Z0 = SC0.INJ.Z0ideal + scale * (SC0.INJ.Z0 - SC0.INJ.Z0ideal)
    SC.INJ.randomInjectionZ = scale * SC0.INJ.randomInjectionZ
    SC.INJ.beamSize = scale * SC0.INJ.beamSize
    return SC


def scaleCircumference(SC, SC0, scale):
    D = 0
    D0 = 0
    for ord in range(len(SC0.RING)):
        D += SC0.RING[ord].Length
        D0 += SC0.IDEALRING[ord].Length
    SC.RING = SCscaleCircumference(SC.RING, scale * (D - D0) + D0, 'abs')
    return SC
