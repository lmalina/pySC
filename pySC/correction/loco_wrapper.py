import numpy as np

from pySC.core.classes import DotDict
from pySC.lattice_properties.response_measurement import response_matrix, dispersion
from pySC.core.lattice_setting import set_magnet_setpoints
from pySC.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)


def loco_model(SC, **kwargs):
    flags = DotDict(
        dict(Dispersion=False, HorizontalDispersionWeight=10, VerticalDispersionWeight=10,  # weights vs ORM elements
             FitHCMEnergyShift=False, FitVCMEnergyShift=False, AutoCorrectDelta=False, SVmethod=1E-3,  # SVD cot-off
             Normalize=True, Coupling=False, Linear=True  # Response matrix calculator
             ))
    flags.update(**kwargs)
    ring_data, init = DotDict(), DotDict()
    ring_data.CavityFrequency = SC.IDEALRING[SC.ORD.RF].Frequency
    ring_data.CavityHarmNumber = SC.IDEALRING[SC.ORD.RF].HarmNumber
    ring_data.Lattice = SC.IDEALRING
    init.SC = SC
    return ring_data, flags, init


def loco_bpm_structure(SC, bpmords=None, **kwargs):
    if bpmords is None:
        bpmords = SC.ORD.BPM
    n_bpms = bpmords.shape[0]
    bpm_data = DotDict(dict(FamName='BPM', BPMIndex=bpmords.copy(), FitGains=False,
                            HBPMIndex=np.arange(n_bpms, dtype=int), VBPMIndex=np.arange(n_bpms, dtype=int),
                            HBPMGain=np.ones(n_bpms), VBPMGain=np.ones(n_bpms),
                            HBPMCoupling=np.zeros(n_bpms), VBPMCoupling=np.zeros(n_bpms)))
    bpm_data.update(**kwargs)
    return bpm_data


def loco_cm_structure(SC, cmords=None, cm_steps=None, **kwargs):
    if cmords is None:
        cmords = SC.ORD.CM
    n_hcms, n_vcms = cmords[0].shape[0], cmords[1].shape[0]
    cm_data = DotDict(dict(FamName='CM', HCMIndex=cmords[0].copy(), VCMIndex=cmords[1].copy(), FitKicks=False,
                           HCMCoupling=np.zeros(n_hcms), VCMCoupling=np.zeros(n_vcms),
                           HCMsteps=np.ones(n_hcms), VCMsteps=np.ones(n_vcms),
                           HCMKicks=2e3 * np.ones(n_hcms), VCMKicks=2e3 * np.ones(n_vcms)  # [mrad] TODO why 2?
                           ))
    if isinstance(cm_steps, (int, float)) or len(cm_steps) == 1:
        cm_data.HCMsteps *= cm_steps
        cm_data.VCMsteps *= cm_steps
    else:
        cm_data.HCMsteps, cm_data.VCMsteps = cm_steps[0].copy(), cm_steps[1].copy()
    cm_data.HCMKicks *= cm_data.HCMsteps
    cm_data.VCMKicks *= cm_data.VCMsteps
    cm_data.update(**kwargs)
    return cm_data


def loco_measurement(SC, CMstep, deltaRF, BPMords, CMords, **kwargs):
    loco_meas_data = DotDict()
    loco_meas_data.RF = SC.RING[SC.ORD.RF[0]].Frequency
    loco_meas_data.DeltaRF = deltaRF
    loco_meas_data.BPMSTD = 1E-3 * np.ones(2 * len(BPMords))  # [mm]
    RM, Err, CMsteps = response_matrix(SC, CMstep, BPMords, CMords, *kwargs)
    CMsteps = [np.max(np.abs(CMsteps[0]), axis=1), np.max(np.abs(CMsteps[1]), axis=1)]
    loco_meas_data.M = 2 * 1000 * np.concatenate((CMsteps[0], CMsteps[1])) * RM
    loco_meas_data.Eta = deltaRF * 1000 * dispersion(SC, rf_step=deltaRF, bpm_ords=BPMords, n_steps=3)
    return loco_meas_data, CMsteps


def loco_fit_parameters(SC, RING0, RINGdata, DeltaRF, *args):
    fit_parameters = DotDict()
    fit_parameters.FitRFFrequency = True
    fit_parameters.DeltaRF = DeltaRF
    n_group = 0
    for nFam in range(len(args)):
        n_elem = 0
        for ord in args[nFam][0]:
            (fname, sc_fname) = ('PolynomA', 'SetPointA') if args[nFam][1] else ('PolynomB', 'SetPointB')
            fit_parameters.Params[n_group][0][n_elem].FieldName = fname
            fit_parameters.Params[n_group][0][n_elem].SCFieldName = sc_fname
            fit_parameters.Params[n_group][0][n_elem].ElemIndex = ord
            fit_parameters.Params[n_group][0][n_elem].FieldIndex = np.arange(2)
            fit_parameters.Params[n_group][0][n_elem].Function = lambda x: x
            fit_parameters.Params[n_group][0][n_elem].Args = []
            fit_parameters.Values[n_group][0] = RINGdata.Lattice[ord][fit_parameters.Params[n_group][0][n_elem].FieldName][2]
            fit_parameters.IdealValues[n_group][0] = SC.IDEALRING[ord][fit_parameters.Params[n_group][0][n_elem].FieldName][2]
            fit_parameters.OrigValues[n_group][0] = RING0[ord][fit_parameters.Params[n_group][0][n_elem].SCFieldName][2]
            fit_parameters.Deltas[n_group][0] = args[nFam][4]
            if args[nFam][3] == 'family':
                n_elem = n_elem + 1
                if ord == args[nFam][0][-1]:
                    n_group = n_group + 1
            else:
                n_group = n_group + 1
    return fit_parameters


def apply_lattice_correction(SC, fit_parameters, dipole_compensation=True, damping=1):
    for n_group in range(len(fit_parameters.Params)):
        for n_elem in range(len(fit_parameters.Params[n_group])):
            ord = fit_parameters.Params[n_group][n_elem].ElemIndex
            field = fit_parameters.Params[n_group][n_elem].SCFieldName
            setpoint = fit_parameters.OrigValues[n_group] + damping * (
                    fit_parameters.IdealValues[n_group] - fit_parameters.Values[n_group])
            if field == 'SetPointB':  # Normal quadrupole
                SC = set_magnet_setpoints(SC, ord, False, 1, setpoint, dipole_compensation=dipole_compensation)
            elif field == 'SetPointA':  # Skew quadrupole
                SC = set_magnet_setpoints(SC, ord, True, 1, setpoint)
    SC = SC.update_magnets(SC.ORD.Magnet)
    return SC


def applyDiagnosticCorrection(SC, cm_step, CMData, BPMData, CMcalOffsets=None, meanToZero=False, outlierRemovalAt=None):
    if not isinstance(cm_step, list) and not len(cm_step) == 1:
        raise ValueError(
            'CM steps must be defined as single value or cell array matching the number of used HCM and VCM.')
    if not isinstance(cm_step, list):
        new_cm_step = [np.repeat(cm_step, len(CMData.HCMIndex)), np.repeat(cm_step, len(CMData.VCMIndex))]
    else:
        new_cm_step = cm_step
    fields = ['H', 'V']
    sc_fields = ['CalErrorB', 'CalErrorA']
    for nDim in range(2):
        if CMData.FitKicks:
            fit_cal_cm = CMData[fields[nDim] + 'CMKicks'] / new_cm_step[nDim] / 1000 / 2
            if CMcalOffsets is not None:
                fit_cal_cm = fit_cal_cm - CMcalOffsets[nDim]
            if outlierRemovalAt is not None:
                fit_cal_cm[np.abs(1 - fit_cal_cm) >= outlierRemovalAt] = 1
            if meanToZero:
                fit_cal_cm = fit_cal_cm + np.mean(1 - fit_cal_cm)
            i = 0
            for ord in CMData[fields[nDim] + 'CMIndex']:
                SC.RING[ord][sc_fields[nDim]][0] = SC.RING[ord][sc_fields[nDim]][0] + (1 - fit_cal_cm[i])
                i = i + 1
        if BPMData.FitGains:
            fit_cal_bpm = BPMData[fields[nDim] + 'BPMGain']
            if not outlierRemovalAt == []:
                fit_cal_bpm[np.abs(1 - fit_cal_bpm) >= outlierRemovalAt] = 1
            if meanToZero:
                fit_cal_bpm = fit_cal_bpm + np.mean(1 - fit_cal_bpm)
            i = 0
            for ord in BPMData.BPMIndex[BPMData[fields[nDim] + 'BPMIndex']]:
                SC.RING[ord].CalError[nDim] = SC.RING[ord].CalError[nDim] + (1 - fit_cal_bpm[i])
                i = i + 1

    if BPMData.FitCoupling:
        fit_roll_bpm = (BPMData.VBPMCoupling - BPMData.HBPMCoupling) / 2
        for i, ord in enumerate(BPMData.BPMIndex):
            SC.RING[ord].Roll = SC.RING[ord].Roll - fit_roll_bpm[i]
    return SC
