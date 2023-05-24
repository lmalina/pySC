import numpy as np

from pySC.utils.at_wrapper import atlinopt, atmatchchromdelta
from pySC.core.classes import DotDict
from pySC.correction.SCfeedback import SCfeedbackRun
from pySC.lattice_properties.SCgetDispersion import SCgetDispersion
from pySC.lattice_properties.SCgetModelRM import SCgetModelRM
from pySC.utils.sc_tools import SCgetPinv
from pySC.lattice_properties.SCgetRespMat import SCgetRespMat
from pySC.core.lattice_setting import SCsetMags2SetPoints


def SClocoLib(funName, *varargin):  # TODO don't work on this, not realy needed and not present in pyAT
    eval(funName + '(*varargin)')


def setupLOCOmodel(SC, *varargin):
    RINGdata, Init, LOCOflags = DotDict(), DotDict(), DotDict()
    RINGdata.CavityFrequency = SC.IDEALRING[SC.ORD.RF].Frequency
    RINGdata.CavityHarmNumber = SC.IDEALRING[SC.ORD.RF].HarmNumber
    RINGdata.Lattice = SC.IDEALRING
    Init.SC = SC
    LOCOflags.HorizontalDispersionWeight = 10  # Hor. dispersion VS. ORM elements
    LOCOflags.VerticalDispersionWeight = 10  # Ver. dispersion VS. ORM elements
    LOCOflags.Dispersion = 'No'  # Include dispersion
    LOCOflags.FitHCMEnergyShift = 'No'  # Fit HCM energy Shift
    LOCOflags.FitVCMEnergyShift = 'No'  # Fit VCM energy Shift
    LOCOflags.SVmethod = 1E-3  # Cut off
    LOCOflags.AutoCorrectDelta = 'No'  # NO!
    LOCOflags.Normalize = 'Yes'  # Normalization flag
    LOCOflags.Linear = 'Yes'  # Response matrix calculator
    LOCOflags.Coupling = 'No'  # Include off-diagonal ORM elements
    LOCOflags.Dispersion = 'No'  # Include dispersion
    for i in range(0, len(varargin), 2):
        LOCOflags[varargin[i]] = varargin[i + 1]
    return RINGdata, LOCOflags, Init


def getBPMCMstructure(SC, CMsteps, *varargin):
    BPMData, CMData = DotDict(), DotDict()
    BPMData.FamName = 'BPM'
    BPMData.BPMIndex = SC.ORD.BPM
    BPMData.HBPMIndex = np.arange(len(SC.ORD.BPM))
    BPMData.VBPMIndex = np.arange(len(SC.ORD.BPM))
    BPMData.FitGains = 'No'
    CMData.FamName = 'CM'
    CMData.HCMIndex = SC.ORD.CM[0]
    CMData.VCMIndex = SC.ORD.CM[1]
    CMData.FitKicks = 'No'
    for i in range(len(varargin)):
        if varargin[i][0] == 'CMords':
            CMData.HCMIndex = varargin[i][1]
            CMData.VCMIndex = varargin[i][2]
        elif varargin[i][0] == 'BPMords':
            BPMData.BPMIndex = np.unique(np.concatenate((varargin[i][1], varargin[i][2])))
            BPMData.HBPMIndex = np.where(np.in1d(BPMData.BPMIndex, varargin[i][1]))[0]
            BPMData.VBPMIndex = np.where(np.in1d(BPMData.BPMIndex, varargin[i][2]))[0]
        elif varargin[i][0] == 'BPM':
            BPMData[varargin[i][1]] = varargin[i][2]
        elif varargin[i][0] == 'CM':
            CMData[varargin[i][1]] = varargin[i][2]
        else:
            raise Exception('Unsuported type.')
    if isinstance(CMsteps, (int, float)) and len(CMsteps) == 1:
        CMsteps = [np.ones(len(CMData.HCMIndex)) * CMsteps, np.ones(len(CMData.VCMIndex)) * CMsteps]
    CMData.CMsteps = CMsteps
    BPMData.HBPMGain = np.ones(len(BPMData.HBPMIndex))
    BPMData.VBPMGain = np.ones(len(BPMData.VBPMIndex))
    BPMData.HBPMCoupling = np.zeros(len(BPMData.HBPMIndex))
    BPMData.VBPMCoupling = np.zeros(len(BPMData.VBPMIndex))
    CMData.HCMCoupling = np.zeros(len(CMData.HCMIndex))
    CMData.VCMCoupling = np.zeros(len(CMData.VCMIndex))
    CMData.HCMKicks = 1E3 * 2 * CMData.CMsteps[0] * np.ones(len(CMData.HCMIndex))  # [mrad]
    CMData.VCMKicks = 1E3 * 2 * CMData.CMsteps[1] * np.ones(len(CMData.VCMIndex))  # [mrad]
    return BPMData, CMData


def getMeasurement(SC, CMstep, deltaRF, BPMords, CMords, *varargin):
    LocoMeasData = DotDict()
    LocoMeasData.RF = SC.RING[SC.ORD.RF[0]].Frequency
    LocoMeasData.DeltaRF = deltaRF
    LocoMeasData.BPMSTD = 1E-3 * np.ones(2 * len(BPMords))  # [mm]
    RM, Err, CMsteps = SCgetRespMat(SC, CMstep, BPMords, CMords, *varargin)
    CMsteps = [np.max(np.abs(CMsteps[0]), axis=1), np.max(np.abs(CMsteps[1]), axis=1)]
    LocoMeasData.M = 2 * 1000 * np.concatenate((CMsteps[0], CMsteps[1])) * RM
    LocoMeasData.Eta = LocoMeasData.DeltaRF * 1000 * SCgetDispersion(SC, LocoMeasData.DeltaRF, 'BPMords', BPMords,
                                                                     'nSteps', 3)
    return LocoMeasData, CMsteps


def setupFitparameters(SC, RING0, RINGdata, DeltaRF, *varargin):
    FitParameters = DotDict()
    FitParameters.FitRFFrequency = 'Yes'
    FitParameters.DeltaRF = DeltaRF
    nGroup = 1
    for nFam in range(len(varargin)):
        nElem = 1
        for ord in varargin[nFam][0]:
            if varargin[nFam][1] == 'normal':
                FitParameters.Params[nGroup][0][nElem].FieldName = 'PolynomB'
                FitParameters.Params[nGroup][0][nElem].SCFieldName = 'SetPointB'
            elif varargin[nFam][1] == 'skew':
                FitParameters.Params[nGroup][0][nElem].FieldName = 'PolynomA'
                FitParameters.Params[nGroup][0][nElem].SCFieldName = 'SetPointA'
            else:
                raise Exception('Unsoported type.')
            FitParameters.Params[nGroup][0][nElem].ElemIndex = ord
            FitParameters.Params[nGroup][0][nElem].FieldIndex = {1, 2}
            FitParameters.Params[nGroup][0][nElem].Function = lambda x: x
            FitParameters.Params[nGroup][0][nElem].Args = {}
            FitParameters.Values[nGroup][0] = RINGdata.Lattice[ord][FitParameters.Params[nGroup][0][nElem].FieldName][2]
            FitParameters.IdealValues[nGroup][0] = SC.IDEALRING[ord][FitParameters.Params[nGroup][0][nElem].FieldName][
                2]
            FitParameters.OrigValues[nGroup][0] = RING0[ord][FitParameters.Params[nGroup][0][nElem].SCFieldName][2]
            FitParameters.Deltas[nGroup][0] = varargin[nFam][4]
            if varargin[nFam][3] == 'family':
                nElem = nElem + 1
                if ord == varargin[nFam][0][-1]:
                    nGroup = nGroup + 1
            else:
                nGroup = nGroup + 1
    return FitParameters


def applyLatticeCorrection(SC, FitParameters, dipCompensation=True, damping=1):
    for nGroup in range(len(FitParameters.Params)):
        for nElem in range(len(FitParameters.Params[nGroup])):
            ord = FitParameters.Params[nGroup][nElem].ElemIndex
            field = FitParameters.Params[nGroup][nElem].SCFieldName
            setpoint = FitParameters.OrigValues[nGroup] + damping * (
                    FitParameters.IdealValues[nGroup] - FitParameters.Values[nGroup])
            if field == 'SetPointB':  # Normal quadrupole
                SC = SCsetMags2SetPoints(SC, ord, False, 1, setpoint, dipCompensation=dipCompensation)
            elif field == 'SetPointA':  # Skew quadrupole
                SC = SCsetMags2SetPoints(SC, ord, True, 1, setpoint)
    SC = SC.update_magnets(SC.ORD.Magnet)
    return SC


def applyDiagnosticCorrection(SC, CMstep, CMData, BPMData, CMcalOffsets=[], meanToZero=0, outlierRemovalAt=[]):
    if not isinstance(CMstep, list) and not len(CMstep) == 1:
        raise ValueError(
            'CM steps must be defined as single value or cell array matching the number of used HCM and VCM.')
    if not isinstance(CMstep, list):
        CMstep = [np.repeat(CMstep, len(CMData.HCMIndex)), np.repeat(CMstep, len(CMData.VCMIndex))]
    fields = ['H', 'V']
    SCfields = ['CalErrorB', 'CalErrorA']
    for nDim in range(2):
        if CMData.FitKicks == 'Yes':
            fitCalCM = CMData[fields[nDim] + 'CMKicks'] / CMstep[nDim] / 1000 / 2
            if not CMcalOffsets == []:
                fitCalCM = fitCalCM - CMcalOffsets[nDim]
            if not outlierRemovalAt == []:
                fitCalCM[np.abs(1 - fitCalCM) >= outlierRemovalAt] = 1
            if meanToZero == 1:
                fitCalCM = fitCalCM + np.mean(1 - fitCalCM)
            i = 0
            for ord in CMData[fields[nDim] + 'CMIndex']:
                SC.RING[ord][SCfields[nDim]][0] = SC.RING[ord][SCfields[nDim]][0] + (1 - fitCalCM[i])
                i = i + 1
        if BPMData.FitGains == 'Yes':
            fitCalBPM = BPMData[fields[nDim] + 'BPMGain']
            if not outlierRemovalAt == []:
                fitCalBPM[np.abs(1 - fitCalBPM) >= outlierRemovalAt] = 1
            if meanToZero == 1:
                fitCalBPM = fitCalBPM + np.mean(1 - fitCalBPM)
            i = 0
            for ord in BPMData.BPMIndex[BPMData[fields[nDim] + 'BPMIndex']]:
                SC.RING[ord].CalError[nDim] = SC.RING[ord].CalError[nDim] + (1 - fitCalBPM[i])
                i = i + 1
    if BPMData.FitCoupling == 'Yes':
        fitRollBPM = (BPMData.VBPMCoupling - BPMData.HBPMCoupling) / 2
        i = 0
        for ord in BPMData.BPMIndex:
            SC.RING[ord].Roll = SC.RING[ord].Roll - fitRollBPM[i]
            i = i + 1
    return SC


def applyOrbitCorrection(SC, Minv=[], alpha=50, CMords=None, BPMords=None, verbose=0):
    if CMords is None:
        CMords = SC.ORD.CM
    if BPMords is None:
        BPMords = SC.ORD.BPM
    if not Minv:
        M = SCgetModelRM(SC, BPMords, CMords, trackMode='ORB')
        if np.any(np.isnan(M)):
            raise ValueError('NaN in model response, aborting.')
        Minv = SCgetPinv(M, alpha=alpha)
    CUR, ERROR = SCfeedbackRun(SC, Minv, target=0, maxsteps=30, BPMords=BPMords, CMords=CMords, verbose=verbose)
    if ERROR:
        print('Feedback crashed.')
    else:
        SC = CUR
    return SC


def matchChromaticity(SC, sFamOrds, chromTarget):
    """
    This function will be removed in a future release. Use 'fitChromaticity' within this library instead
    """
    print('This function will be removed in a future release. Use ''fitChromaticity'' within this library instead')
    tmp, _ = atmatchchromdelta(SC.RING, chromTarget, sFamOrds)
    for nFam in range(len(sFamOrds)):
        factor = tmp[sFamOrds[nFam][0]].PolynomB[2] / SC.RING[sFamOrds[nFam][0]].PolynomB[2]
        for ord in sFamOrds[nFam]:
            SC.RING[ord].SetPointB[2] = SC.RING[ord].SetPointB[2] * factor
    SC = SC.update_magnets(SC.ORD.Magnet)
    return SC


def fitChromaticity(SC, sOrds, targetChrom=[], verbose=0, InitStepSize=[2, 2], TolX=1E-4, TolFun=1E-3,
                    sepTunesWithOrds=[], sepTunesDeltaK=[]):
    if len(targetChrom) == 0:
        _, _, targetChrom = atlinopt(SC.IDEALRING, 0, [])
    if np.any(np.isnan(targetChrom)):
        print('Target chromaticity must not contain NaN. Aborting.')
        return SC
    SC0 = SC
    if len(sepTunesWithOrds) > 0 and len(sepTunesDeltaK) > 0:
        for nFam in range(len(sepTunesWithOrds)):
            SC = SCsetMags2SetPoints(SC, sepTunesWithOrds[nFam], False, 1, sepTunesDeltaK[nFam], method='add')  # TODO quads here?
    if verbose:
        _, _, xi0 = atlinopt(SC.RING, 0, [])
        print('Fitting chromaticities from [%.3f,%.3f] to [%.3f,%.3f].' % (
            xi0[0], xi0[1], targetChrom[0], targetChrom[1]))
    SP0 = np.zeros((len(sOrds), len(sOrds[0])))  # TODO can the lengts vary
    for nFam in range(len(sOrds)):
        for n in range(len(sOrds[nFam])):
            SP0[nFam][n] = SC.RING[sOrds[nFam][n]].SetPointB[2]
    fun = lambda x: fitFunction(SC, sOrds, x, SP0, targetChrom)
    sol = fminsearch(fun, InitStepSize, optimset('TolX', TolX, 'TolFun', TolFun))
    SC = applySetpoints(SC0, sOrds, sol, SP0)
    if verbose:
        _, _, xi1 = atlinopt(SC.RING, 0, [])
        print('        Final chromaticity: [%.3f,%.3f]\n          Setpoints change: [%.2f,%.2f]' % (
            xi1[0], xi1[1], sol[0], sol[1]))
    return SC


def fitTune(SC, qOrds, targetTune=[], verbose=0, TolX=1E-4, TolFun=1E-3, InitStepSize=[.01, .01], FitInteger=1):
    if len(targetTune) == 0:
        if FitInteger:
            ld, _, _ = atlinopt(SC.IDEALRING, 0, range(len(SC.IDEALRING) + 1))
            targetTune = ld[-1].mu / 2 / np.pi
        else:
            _, targetTune, _ = atlinopt(SC.IDEALRING, 0)
    if verbose:
        nu0 = getLatProps(SC, FitInteger)
        print('Fitting tunes from [%.4f,%.4f] to [%.4f,%.4f].' % (nu0[0], nu0[1], targetTune[0], targetTune[1]))
    SP0 = np.zeros((len(qOrds), len(qOrds[0])))  # TODO can the lengts vary
    for nFam in range(len(qOrds)):
        for n in range(len(qOrds[nFam])):
            SP0[nFam][n] = SC.RING[qOrds[nFam][n]].SetPointB[1]
    fun = lambda x: fitFunction(SC, qOrds, x, SP0, targetTune, FitInteger)
    sol = fminsearch(fun, InitStepSize, optimset('TolX', TolX, 'TolFun', TolFun))
    SC = applySetpoints(SC, qOrds, sol, SP0)
    if verbose:
        nu1 = getLatProps(SC, FitInteger)
        print('       Final tune: [%.4f,%.4f]\n  Setpoints change: [%.3f,%.3f]' % (nu1[0], nu1[1], sol[0], sol[1]))
    return SC


def fitFunction(SC, qOrds, setpoints, SP0, target, FitInteger):
    SC = applySetpoints(SC, qOrds, setpoints, SP0)
    nu = getLatProps(SC, FitInteger)
    out = np.sqrt(np.mean((nu - target) ** 2))
    return out


def getLatProps(SC, FitInteger):
    if FitInteger:
        ld, _, _ = atlinopt(SC.RING, 0, range(len(SC.RING) + 1))
        nu = ld[-1].mu / 2 / np.pi
    else:
        _, nu, _ = atlinopt(SC.RING, 0)
    return nu


def applySetpoints(SC, ords, setpoints, SP0):
    for nFam in range(len(ords)):
        SC = SCsetMags2SetPoints(SC, ords[nFam], False, 1, setpoints[nFam] + SP0[nFam], method='abs', dipCompensation=True)
    return SC
