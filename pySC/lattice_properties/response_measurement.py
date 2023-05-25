import numpy as np

from pySC.core.beam import SCgetBPMreading
from pySC.core.lattice_setting import SCsetCMs2SetPoints, SCgetCMSetPoints, SCsetCavs2SetPoints
from pySC.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)

def SCgetRespMat(SC, Amp, BPMords, CMords, mode='fixedKick', nSteps=2, fit='linear'):
    if (not isinstance(Amp, list) and not len(Amp) == 1) or (
            isinstance(Amp, list) and (len(Amp[0]) != len(CMords[0]) and len(Amp[1]) != len(CMords[1]))):
        raise ValueError('RM amplitude must be defined as single value or cell array matching the number of used HCM and VCM.')
    if not isinstance(Amp, list):
        Amp = [np.ones(len(CMords[0])) * Amp, np.ones(len(CMords[1])) * Amp]
    LOGGER.debug('Calculate {:d}-turn trajectory response matrix for {:d} BPMs and {:d}|{:d} CMs with mode ''{}'' and amplitude {:.0e}|{:.0e} using {:d} steps ...'.format(SC.INJ.nTurns, len(BPMords), len(CMords[0]), len(CMords[1]), mode, np.mean(Amp[0]), np.mean(Amp[1]), nSteps))
    RM = np.nan * np.zeros((2 * SC.INJ.nTurns * len(BPMords), len(CMords[0]) + len(CMords[1])))
    Err = np.nan * np.zeros((2 * SC.INJ.nTurns * len(BPMords), len(CMords[0]) + len(CMords[1])))
    CMsteps = [np.zeros((nSteps, len(CMords[0]))), np.zeros((nSteps, len(CMords[1])))]
    Bref = np.reshape(SCgetBPMreading(SC, BPMords=BPMords), [], 1)
    if SC.INJ.trackMode == 'ORB' and np.any(np.isnan(Bref)):
        raise ValueError('No closed orbit found.')
    i = 0
    for nDim in range(2):
        cmstart = SCgetCMSetPoints(SC, CMords[nDim], nDim)
        for nCM in range(len(CMords[nDim])):
            MaxStep, dB = getKickAmplitude(SC, Bref, BPMords, CMords[nDim][nCM], Amp[nDim][nCM], nDim, SC.INJ.nTurns,
                                           nSteps, mode)
            CMstepVec = np.linspace(-MaxStep, MaxStep, nSteps)
            if nSteps != 2:
                realCMsetPoint = cmstart[nCM] + CMstepVec
                dB = np.vstack((np.zeros((nSteps - 1, len(Bref))), dB.T))
                for nStep in range(nSteps):
                    if CMstepVec[nStep] != 0 and CMstepVec[nStep] != MaxStep:
                        SC, realCMsetPoint[nStep] = SCsetCMs2SetPoints(SC, CMords[nDim][nCM], cmstart[nCM] + CMstepVec[nStep], skewness=nDim)
                        dB[nStep, :] = np.reshape(SCgetBPMreading(SC, BPMords=BPMords), [], 1) - Bref
                dCM = realCMsetPoint - cmstart[nCM]
            else:
                dCM = MaxStep
            CMsteps[nDim][:, nCM] = dCM
            if nSteps == 2:
                RM[:, i] = dB / dCM
            else:
                for nBPM in range(dB.shape[1]):
                    x = dCM[~np.isnan(dB[:, nBPM])]
                    y = dB[~np.isnan(dB[:, nBPM]), nBPM]
                    if fit == 'linear':
                        RM[nBPM, i] = np.linalg.lstsq(x[:, None], y, rcond=None)[0]
                    elif fit == 'quadratic':
                        tmp = np.polyfit(x, y, 2)
                        RM[nBPM, i] = tmp[1]
                    Err[nBPM, i] = np.sqrt(
                        np.mean((RM[nBPM, i] * dCM[~np.isnan(dB[:, nBPM])] - dB[~np.isnan(dB[:, nBPM]), nBPM]).T ** 2))
            i = i + 1
            SC, _ = SCsetCMs2SetPoints(SC, CMords[nDim][nCM], cmstart[nCM], skewness=nDim)
    RM[np.isnan(RM)] = 0
    LOGGER.debug(' done.')
    return RM, Err, CMsteps


def SCgetDispersion(SC,RFstep,BPMords=None,CAVords=None,nSteps=2):
    if BPMords is None:
        BPMords = SC.ORD.BPM
    if CAVords is None:
        CAVords = SC.ORD.RF
    RFsteps = np.zeros((len(CAVords),nSteps))
    for nCav in range(len(CAVords)):
        RFsteps[nCav,:] = SC.RING[CAVords[nCav]].FrequencySetPoint + np.linspace(-RFstep,RFstep,nSteps)
    Bref = np.reshape(SCgetBPMreading(SC,BPMords=BPMords),[],1)
    if nSteps==2:
        SC = SCsetCavs2SetPoints(SC,CAVords,'Frequency',RFstep,'add')
        B = np.reshape(SCgetBPMreading(SC,BPMords=BPMords),[],1)
        eta = (B-Bref)/RFstep
    else:
        dB = np.zeros((nSteps,*np.shape(Bref)))
        for nStep in range(nSteps):
            SC = SCsetCavs2SetPoints(SC,CAVords,'Frequency',RFsteps[:,nStep],'abs')
            dB[nStep,:] = np.reshape(SCgetBPMreading(SC,BPMords=BPMords),[],1) - Bref
        eta = np.linalg.lstsq(np.linspace(-RFstep,RFstep,nSteps),dB)[0]
    return eta


def getKickAmplitude(SC, Bref, BPMords, CMord, Amp, skewness: bool, nTurns, nSteps, mode):
    cmstart = SCgetCMSetPoints(SC, CMord, skewness)
    MaxStep = Amp
    if mode == 'fixedKick':
        for n in range(20):
            SC, realCMsetPoint = SCsetCMs2SetPoints(SC, CMord, cmstart + MaxStep, skewness)
            if realCMsetPoint != (cmstart + MaxStep):
                LOGGER.debug('CM  clipped. Using different CM direction.')
                MaxStep = - MaxStep
                SC, _ = SCsetCMs2SetPoints(SC, CMord, cmstart + MaxStep, skewness)
            B = np.reshape(SCgetBPMreading(SC, BPMords=BPMords), [], 1)
            maxpos = min([np.where(np.isnan(B))[0][0] - 1, nTurns * len(BPMords)])
            maxposRef = min([np.where(np.isnan(Bref))[0][0] - 1, nTurns * len(BPMords)])
            if not (maxpos < maxposRef):
                dB = B - Bref
                break
            else:
                MaxStep = 0.9 * MaxStep
                LOGGER.debug(f'Insufficient beam reach ({maxpos:d}/{maxposRef:d}). CMstep reduced to {1E6 * MaxStep:.1f}urad.')
    elif mode == 'fixedOffset':
        for n in range(4):
            SC, realCMsetPoint = SCsetCMs2SetPoints(SC, CMord, cmstart + MaxStep, skewness)
            if realCMsetPoint != (cmstart + MaxStep):
                LOGGER.debug('CM  clipped. Using different CM direction.')
                MaxStep = - MaxStep
                SC, _ = SCsetCMs2SetPoints(SC, CMord, cmstart + MaxStep, skewness)
            B = np.reshape(SCgetBPMreading(SC, BPMords=BPMords), [], 1)
            maxpos = min([np.where(np.isnan(B))[0][0] - 1, nTurns * len(BPMords)])
            maxposRef = min([np.where(np.isnan(Bref))[0][0] - 1, nTurns * len(BPMords)])
            if maxpos < maxposRef:
                MaxStep = 0.5 * MaxStep
                LOGGER.debug(f'Insufficient beam reach ({maxpos:d}/{maxposRef:d}). CMstep reduced to {1E6 * MaxStep:.1f}urad.')
                continue
            MaxStep = MaxStep * Amp / np.max(np.abs(B - Bref))
        dB = np.reshape(SCgetBPMreading(SC, BPMords=BPMords), [], 1) - Bref
    return MaxStep, dB
# End
