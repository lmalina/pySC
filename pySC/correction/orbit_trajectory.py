import numpy as np

from pySC.core.beam import bpm_reading
from pySC.core.lattice_setting import set_cavity_setpoints, set_cm_setpoints
from pySC.utils import logging_tools
from pySC.utils.sc_tools import SCrandnc

LOGGER = logging_tools.get_logger(__name__)


def SCfeedbackFirstTurn(SC, Mplus, reference=None, CMords=None, BPMords=None, maxsteps=100, nRepro=3, wiggle_after=20, wiggle_steps=32, wiggle_range=(500E-6, 1000E-6)):
    LOGGER.debug('SCfeedbackFirstTurn: Start')
    BPMords, CMords, reference = _check_ords(SC, Mplus, reference, BPMords, CMords)
    bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords)  # Inject...

    for n in range(maxsteps):
        if transmission_history[-1] == 0:
            raise RuntimeError('SCfeedbackFirstTurn: FAIL (no BPM reading to begin with)')
        
        # Set BPM readings
        measurement = bpm_readings[:, :].reshape(reference.shape)
        measurement[np.isnan(measurement)] = 0

        # Correction step
        dphi = np.dot(Mplus, (measurement - reference))
        lastCMh = _get_last_cm(transmission_history[-1]-1, 1, BPMords, CMords[0])[1][0]
        lastCMv = _get_last_cm(transmission_history[-1]-1, 1, BPMords, CMords[1])[1][0]         
        SC, _ = set_cm_setpoints(SC, CMords[0][:lastCMh + 1], -dphi[:lastCMh + 1], skewness=False, method='add')
        SC, _ = set_cm_setpoints(SC, CMords[1][:lastCMv + 1], -dphi[len(CMords[0]):len(CMords[0]) + lastCMv + 1], skewness=True, method='add')
        bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(
            SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)  # Inject...
 
        # Check stopping criteria
        if _is_repro(transmission_history, nRepro) and transmission_history[-1] == bpm_readings.shape[1]:   # last three the same and full transmission
            LOGGER.debug('SCfeedbackFirstTurn: Success')
            return SC
        if _is_repro(transmission_history, wiggle_after):
            SC = _wiggling(SC, BPMords, CMords, transmission_history[-1] + 1, angle_range=wiggle_range, num_angle_steps=wiggle_steps, nRepro=nRepro)

    raise RuntimeError('SCfeedbackFirstTurn: FAIL (maxsteps reached)')


def SCfeedbackStitch(SC, Mplus, reference=None, CMords=None, BPMords=None, nBPMs=4, maxsteps=30, nRepro=3, wiggle_steps=32, wiggle_range=(500E-6, 1000E-6)):
    LOGGER.debug('SCfeedbackStitch: Start')
    if SC.INJ.nTurns != 2:
        raise ValueError("Stitching works only with two turns.")
    BPMords, CMords, reference = _check_ords(SC, Mplus, reference, BPMords, CMords)
    bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords)  # Inject...
    transmission_limit = len(BPMords)+nBPMs
    if transmission_history[-1] < len(BPMords):
        raise RuntimeError("Stitching works only with full 1st turn transmission.")
    
    # Check if minimum transmission for algorithm to work is reached
    if transmission_history[-1] < transmission_limit:
        SC = _wiggling(SC, BPMords, CMords, transmission_limit, angle_range=wiggle_range, num_angle_steps=wiggle_steps, nRepro=nRepro)
        bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(
            SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)
        if transmission_history[-1] < transmission_limit:
            raise RuntimeError("Not enough transmission for stitching to work.")

    # Prepare reference
    reference = reference.reshape(2, len(bpm_readings[0]))
    reference[:, len(BPMords):] = 0
    reference =reference.reshape(Mplus.shape[1])
    
    # Main loop
    for steps in range(maxsteps):
        # Set BPM readings
        delta_b = np.zeros((2,len(BPMords)))
        delta_b[0][:nBPMs] = bpm_readings[0][len(BPMords):len(BPMords)+nBPMs] - bpm_readings[0][:nBPMs]
        delta_b[1][:nBPMs] = bpm_readings[1][len(BPMords):len(BPMords)+nBPMs] - bpm_readings[1][:nBPMs]
        measurement = np.concatenate((bpm_readings[:, :len(BPMords)], delta_b), axis=1).ravel()
        measurement[np.isnan(measurement)] = 0

        # Correction step
        dphi = np.dot(Mplus, (measurement - reference))
        SC, _ = set_cm_setpoints(SC, CMords[0], -dphi[:len(CMords[0])], skewness=False, method="add")
        SC, _ = set_cm_setpoints(SC, CMords[1], -dphi[len(CMords[0]):], skewness=True, method="add")
        bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(
            SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)

        # Check stopping criteria
        if transmission_history[-1] < transmission_history[-2]:
            RuntimeError('SCfeedbackStitch: FAIL Setback')
        if _is_repro(transmission_history, nRepro) and transmission_history[-1] == bpm_readings.shape[1]:
            LOGGER.debug('SCfeedbackStitch: Success')
            return SC
    raise RuntimeError('SCfeedbackStitch: FAIL Reached maxsteps')



def SCfeedbackBalance(SC, Mplus, reference=None, CMords=None, BPMords=None, eps=1e-4, maxsteps=10, nRepro=3):
    LOGGER.debug('SCfeedbackBalance: Start')
    if SC.INJ.nTurns != 2:
        raise ValueError("Balancing works only with two turns.")
    BPMords, CMords, reference = _check_ords(SC, Mplus, reference, BPMords, CMords)
    bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords)  # Inject...
    if transmission_history[-1] < bpm_readings.shape[1]:
        raise ValueError("Balancing works only with full 2 turn transmission.")

    # Prepare reference
    reference = reference.reshape(2, len(bpm_readings[0]))
    reference[:, len(BPMords):] = 0
    reference = reference.reshape(Mplus.shape[1])

    # Main loop
    for steps in range(maxsteps):
        # Set BPM readings
        delta_b = [bpm_readings[0][len(BPMords):] - bpm_readings[0][:len(BPMords)], bpm_readings[1][len(BPMords):] - bpm_readings[1][:len(BPMords)]]
        measurement = np.concatenate((bpm_readings[:, :len(BPMords)], delta_b), axis=1).ravel()
 
        # Correction step
        dphi = np.dot(Mplus, (measurement - reference))
        SC, _ = set_cm_setpoints(SC, CMords[0], -dphi[:len(CMords[0])], skewness=False, method="add")
        SC, _ = set_cm_setpoints(SC, CMords[1], -dphi[len(CMords[0]):], skewness=True, method="add")
        bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(
            SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)

        # Check stopping criteria
        if transmission_history[-1] < bpm_readings.shape[1]:
            raise RuntimeError('SCfeedbackBalance: FAIL (lost transmission)')
        if _is_stable_or_converged(nRepro, eps, rms_orbit_history):
            LOGGER.debug(f'SCfeedbackBalance: Success (converged after {steps} steps)')
            return SC

    raise RuntimeError('SCfeedbackBalance: FAIL (maxsteps reached, unstable)')


def SCfeedbackRun(SC, Mplus, reference=None, CMords=None, BPMords=None, eps=1e-4, target=0, maxsteps=30, nRepro=3, scaleDisp=0):
    LOGGER.debug('SCfeedbackRun: Start')
    BPMords, CMords, reference = _check_ords(SC, Mplus, reference, BPMords, CMords)
    bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords)  # Inject ...

    # Main loop
    for steps in range(maxsteps):
        # Set BPM readings
        measurement = bpm_readings[:, :].reshape(reference.shape)

        # Correction step
        dphi = np.dot(Mplus, (measurement - reference))
        if scaleDisp != 0:   # TODO this is weight
            SC = set_cavity_setpoints(SC, SC.ORD.RF, "Frequency", -scaleDisp * dphi[-1], method="add")
            dphi = dphi[:-1]  # TODO the last setpoint is cavity frequency
        SC, _ = set_cm_setpoints(SC, CMords[0], -dphi[:len(CMords[0])], skewness=False, method="add")
        SC, _ = set_cm_setpoints(SC, CMords[1], -dphi[len(CMords[0]):], skewness=True, method="add")
        bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(
            SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)  # Inject ...

        # Check stopping criteria
        if np.any(np.isnan(bpm_readings[0, :])):
            raise RuntimeError('SCfeedbackRun: FAIL (lost transmission)')
        if max(rms_orbit_history[-1]) < target and _is_stable_or_converged(min(nRepro, maxsteps), eps, rms_orbit_history):
            LOGGER.debug(f"SCfeedbackRun: Success (target reached after {steps:d} steps)")
            return SC
        if _is_stable_or_converged(nRepro, eps, rms_orbit_history):
            LOGGER.debug(f"SCfeedbackRun: Success (converged after {steps:d} steps)")
            return SC
    if _is_stable_or_converged(min(nRepro, maxsteps), eps, rms_orbit_history) or maxsteps == 1:
        LOGGER.debug("SCfeedbackRun: Success (maxsteps reached)")
        return SC
    raise RuntimeError("SCfeedbackRun: FAIL (maxsteps reached, unstable)")


def SCpseudoBBA(SC, BPMords, MagOrds, postBBAoffset, sigma=2):
    # TODO this looks fishy ... assumes BPMs attached to quads?
    #  at the same time two separate 2D arrays?
    if len(postBBAoffset) == 1:
        postBBAoffset = np.tile(postBBAoffset, (2,np.size(BPMords, axis=1)))
    for nBPM in range(np.size(BPMords, axis=1)):
        for nDim in range(2):
            SC.RING[BPMords[nDim][nBPM]].Offset[nDim] = (SC.RING[MagOrds[nDim][nBPM]].MagnetOffset[nDim]
                                                         + SC.RING[MagOrds[nDim][nBPM]].SupportOffset[nDim]
                                                         - SC.RING[BPMords[nDim][nBPM]].SupportOffset[nDim]
                                                         + postBBAoffset[nDim][nBPM] * SCrandnc(sigma))
    return SC


def _check_ords(SC, Mplus, reference, BPMords, CMords):
    if CMords is None:
        CMords = SC.ORD.CM.copy()
    if BPMords is None:
        BPMords = SC.ORD.BPM.copy()
    if reference is None:
        reference = np.zeros((Mplus.shape[1], 1))
    return BPMords, CMords, reference


def _bpm_reading_and_logging(SC, BPMords, ind_history=None, orb_history=None):
    bpm_readings = bpm_reading(SC, bpm_ords=BPMords)
    bpms_reached = ~np.isnan(bpm_readings[0])
    if ind_history is None or orb_history is None:
        return bpm_readings, [np.sum(bpms_reached)], [np.sqrt(np.mean(np.square(bpm_readings[:, bpms_reached]), axis=1))]
    ind_history.append(np.sum(bpms_reached))
    orb_history.append(np.sqrt(np.mean(np.square(bpm_readings[:, bpms_reached]), axis=1)))
    return bpm_readings, ind_history, orb_history  # assumes no bad BPMs


def _get_last_cm(lastBPMidx, n, BPMords, CMords):  # Refactored
    # Returns last CM ords and index of last CMs
    if lastBPMidx >= len(BPMords):  # If there is no beam loss in the first turn
        return CMords[-n:], range(len(CMords)-n, len(CMords))  # ... just return the last n CMs
    if len(CMords[np.where(CMords <= BPMords[lastBPMidx])]) == 0:
        raise RuntimeError('No CM upstream of BPMs')
    return CMords[np.where(CMords <= BPMords[lastBPMidx])[0][-n:]], np.where(CMords <= BPMords[lastBPMidx])[0][-n:]


def _is_repro(hist, n):
    if len(hist) < n:
        return False
    return np.max(np.abs(np.array(hist[-n:]) - hist[-1])) == 0


def _golden_donut_diffs(r0, r1, n_points):
    r02, r12 = r0 ** 2, r1 ** 2
    ints = np.arange(n_points)
    phi = 2 * np.pi / ((1 + np.sqrt(5)) / 2)  # golden ratio
    pos_2d = np.sqrt(r02 + (r12 - r02) * ints / (n_points - 1)) * np.vstack((np.cos(ints * phi), np.sin(ints * phi)))
    return np.diff(np.hstack((np.zeros((2, 1)), pos_2d)), axis=1)


def _is_stable_or_converged(n, eps, hist):  # TODO rethink
    if len(hist) < n:
        return False
    return (np.var(hist[-n:]) / np.std(hist[-n:])) < eps


def _wiggling(SC, BPMords, CMords, transmission_limit, angle_range=(50E-6, 200E-6), num_angle_steps=32, nums_correctors=range(1, 9), nRepro=3):
    LOGGER.debug('Wiggling')
    dpts = _golden_donut_diffs(angle_range[0], angle_range[1], num_angle_steps)
    bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=None, orb_history=None)  # Inject...

    for nWiggleCM in nums_correctors:
        LOGGER.debug(f'Number of magnets used for wiggling: {nWiggleCM}. \n')
        tmpCMordsH = _get_last_cm(transmission_history[-1] - 1, nWiggleCM, BPMords, CMords[0])[0]  # Last CMs in horz
        tmpCMordsV = _get_last_cm(transmission_history[-1] - 1, nWiggleCM, BPMords, CMords[1])[0]  # Last CMs in vert

        for i in range(dpts.shape[1]):
            SC, _ = set_cm_setpoints(SC, tmpCMordsH, np.array([dpts[0, i]]), skewness=False, method='add')
            SC, _ = set_cm_setpoints(SC, tmpCMordsV, np.array([dpts[1, i]]), skewness=True, method='add')
            bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(
                SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)
            if transmission_history[-1] >= transmission_limit:
                for _ in range(2):
                    bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(
                        SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)
                if _is_repro(transmission_history, nRepro):
                    LOGGER.debug('...completed')
                    return SC  # Continue with feedback

    if not transmission_history[-1] >= transmission_limit:
        raise RuntimeError('Wiggling failed')
    return SC
