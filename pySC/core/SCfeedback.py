import numpy as np

from pySC.core.SCgetBPMreading import SCgetBPMreading
from pySC.core.SCsetpoints import SCsetCavs2SetPoints, SCsetCMs2SetPoints
from pySC.utils import logging_tools


LOGGER = logging_tools.get_logger(__name__)


def SCfeedbackFirstTurn(SC, Mplus, R0=None, CMords=None, BPMords=None, maxsteps=100, wiggle_after=20, wiggle_steps=32, wiggle_range=np.array([500E-6, 1000E-6])):
    # Initialize and get reference
    LOGGER.debug('SCfeedbackFirstTurn: Start')
    BPMords, CMords, R0 = _check_ords(SC, Mplus, R0, BPMords, CMords)
    B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords)  # Inject...

    # Main loop
    for n in range(maxsteps):
        if transmission_history[-1] == 0:
            raise RuntimeError('SCfeedbackFirstTurn: FAIL (no BPM reading to begin with)')
        
        SC = _correction_step_firstturn(SC, transmission_history[-1]-1, BPMords, CMords, B, R0, Mplus)     
        B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)  # Inject...
 
        # Check stopping criteria
        if _is_repro(transmission_history, 5) and transmission_history[-1] == B.shape[1]:   # last five the same and full transmission
            LOGGER.debug('SCfeedbackFirstTurn: Success')
            return SC
        if _is_repro(transmission_history, wiggle_after):
            SC = _wiggling(SC,BPMords,CMords,transmission_history[-1]+1,wiggle_range=wiggle_range,wiggle_steps=wiggle_steps) # TODO: find more elegant solution for parsing fine tuning parameters. Maybe something like 'SC.config.feedbackFirstTurn.wiggle'?

    raise RuntimeError('SCfeedbackFirstTurn: FAIL (maxsteps reached)')


def SCfeedbackStitch(SC, Mplus, R0=None, CMords=None, BPMords=None, nBPMs=4, maxsteps=30, nRepro=3, wiggle_steps=32, wiggle_range=np.array([500E-6, 1000E-6])):
    # Initialize and get reference
    LOGGER.debug('SCfeedbackStitch: Start')
    BPMords, CMords, R0 = _check_ords(SC, Mplus, R0, BPMords, CMords)
    B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords)  # Inject...
    transmission_limit = len(BPMords)+nBPMs

    # Check state
    if SC.INJ.nTurns != 2:
        raise ValueError("Stitching works only with two turns.") # TODO: ValueError vs RuntimeError -> which one for which?
    if transmission_history[-1] < len(BPMords):
        raise ValueError("Stitching works only with full 1st turn transmission.")
    
    # Check if minimum transmission for algorithm to work is reached
    if not transmission_history[-1] >= transmission_limit:
        SC = _wiggling(SC, BPMords, CMords, transmission_limit, wiggle_range=wiggle_range, wiggle_steps=wiggle_steps)
        B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)
        if not transmission_history[-1] >= transmission_limit:
            raise ValueError("Not enough transmission for stitching to work.")

    # Prepare reference
    R0= R0.reshape(2,len(B[0]))
    R0[:,len(BPMords):] = 0
    R0=R0.reshape(Mplus.shape[1])
    
    # Main loop
    for steps in range(maxsteps):
        # Set BPM readings
        delta_b = np.zeros((2,len(BPMords)))
        delta_b[0][:nBPMs] = B[0][len(BPMords):len(BPMords)+nBPMs] - B[0][:nBPMs]
        delta_b[1][:nBPMs] = B[1][len(BPMords):len(BPMords)+nBPMs] - B[1][:nBPMs]
        R=np.concatenate((B[:, :len(BPMords)], delta_b), axis=1).ravel()
        R[np.isnan(R)] = 0

        # Correction step
        dphi = np.split(Mplus @ (R - R0),2)  # TODO what if unequal lengths?
        SC, _ = SCsetCMs2SetPoints(SC, CMords[0], -dphi[0], skewness=False, method='add')
        SC, _ = SCsetCMs2SetPoints(SC, CMords[1], -dphi[1], skewness=True, method='add')      
        B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)

        # Check stopping criteria
        if transmission_history[-1] < transmission_history[-2]:
            RuntimeError('SCfeedbackStitch: FAIL Setback')
        if _is_repro(transmission_history, nRepro) and transmission_history[-1] == B.shape[1]: # TODO remove
            LOGGER.debug('SCfeedbackStitch: Success')
            return SC
    raise RuntimeError('SCfeedbackStitch: FAIL Reached maxsteps')



def SCfeedbackBalance(SC, Mplus, R0=None, CMords=None, BPMords=None, eps=1e-4, maxsteps=10):
    # Initialize and get reference
    LOGGER.debug('SCfeedbackBalance: Start')
    BPMords, CMords, R0 = _check_ords(SC, Mplus, R0, BPMords, CMords)
    B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords)  # Inject...

    # Check state
    if SC.INJ.nTurns != 2:
        raise ValueError("Balancing works only with two turns.")
    if transmission_history[-1] < 2*len(BPMords):
        raise ValueError("Balancing works only with full 2 turn transmission.")

   # Prepare reference
    R0= R0.reshape(2,len(B[0]))
    R0[:,len(BPMords):] = 0
    R0=R0.reshape(Mplus.shape[1])

    # Main loop
    for steps in range(maxsteps):
        # Set BPM readings
        delta_b = [B[0][len(BPMords):] - B[0][:len(BPMords)], B[1][len(BPMords):] - B[1][:len(BPMords)]]
        R=np.concatenate((B[:, :len(BPMords)], delta_b), axis=1).ravel()
 
        # Correction step
        dphi = np.split(np.dot(Mplus,(R - R0)),2)  # TODO what if unequal lengths?
        SC, _ = SCsetCMs2SetPoints(SC, CMords[0], -dphi[0], skewness=False, method='add')
        SC, _ = SCsetCMs2SetPoints(SC, CMords[1], -dphi[1], skewness=True, method='add')  
        B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)

        # Check stopping criteria
        if len(transmission_history)>1 and transmission_history[-1] < transmission_history[-2]:
            raise RuntimeError('SCfeedbackBalance: FAIL (setback)')
        if transmission_history[-1] < B.shape[1]:
            raise RuntimeError('SCfeedbackBalance: FAIL (lost transmission)')
        if _is_stable_or_converged(3, eps, rms_orbit_history):
            LOGGER.debug(f'SCfeedbackBalance: Success (converged after {steps} steps)')
            return SC
    if _is_stable_or_converged(min(10, maxsteps), eps, rms_orbit_history):
        LOGGER.debug('SCfeedbackBalance: Success (maxsteps reached)')
        return SC
    raise RuntimeError('SCfeedbackBalance: FAIL (maxsteps reached, unstable)')


def SCfeedbackRun(SC, Mplus, R0=None, CMords=None, BPMords=None, eps=1e-4, target=0, maxsteps=30, scaleDisp=0):
    # Initialize and get reference
    LOGGER.debug('SCfeedbackRun: Start')
    BPMords, CMords, R0 = _check_ords(SC, Mplus, R0, BPMords, CMords)
    B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords)  # Inject ...

    # Main loop
    for steps in range(maxsteps):
        # Set BPM readings
        R = B[:, :].reshape(R0.shape)
        
        # Correction step
        dphi = np.dot(Mplus, (R - R0))
        if scaleDisp != 0:   # TODO this is weight
            SC = SCsetCavs2SetPoints(SC, SC.ORD.RF, "Frequency", -scaleDisp * dphi[-1], method="add")
            dphi = dphi[:-1]  # TODO the last setpoint is cavity frequency
        SC, _ = SCsetCMs2SetPoints(SC, CMords[0], -dphi[:len(CMords[0])], skewness=False, method="add")
        SC, _ = SCsetCMs2SetPoints(SC, CMords[1], -dphi[len(CMords[0]):], skewness=True, method="add")
        B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)  # Inject ...

        # Check stopping criteria
        if np.any(np.isnan(B[0, :])):
            raise RuntimeError('SCfeedbackRun: FAIL (lost transmission)')
        if max(rms_orbit_history[-1]) < target and _is_stable_or_converged(min(10, maxsteps), eps, rms_orbit_history):
            LOGGER.debug("SCfeedbackRun: Success (target reached)")
            return SC
        if _is_stable_or_converged(3, eps, rms_orbit_history):
            LOGGER.debug(f"SCfeedbackRun: Success (converged after {steps:d} steps)")
            return SC
        if _is_stable_or_converged(min(10, maxsteps), eps, rms_orbit_history) or maxsteps == 1:
            LOGGER.debug("SCfeedbackRun: Success (maxsteps reached)")
            return SC
    raise RuntimeError("SCfeedbackRun: FAIL (maxsteps reached, unstable)")


def _check_ords(SC, Mplus, R0, BPMords, CMords):
    if CMords is None:
        CMords = SC.ORD.CM.copy()
    if BPMords is None:
        BPMords = SC.ORD.BPM.copy()
    if R0 is None:
        R0 = np.zeros((Mplus.shape[1], 1))
    return BPMords, CMords, R0


def _bpm_reading_and_logging(SC, BPMords, ind_history=None, orb_history=None):
    bpm_readings = SCgetBPMreading(SC, BPMords=BPMords)
    bpms_reached = ~np.isnan(bpm_readings[0])
    if ind_history is None or orb_history is None:
        return bpm_readings, [np.sum(bpms_reached)], [np.sqrt(np.mean(np.square(bpm_readings[:, bpms_reached]), axis=1))]
    ind_history.append(np.sum(bpms_reached))
    orb_history.append(np.sqrt(np.mean(np.square(bpm_readings[:, bpms_reached]), axis=1)))
    return bpm_readings, ind_history, orb_history  # assumes no bad BPMs


def _correction_step_firstturn(SC, bpm_ind, BPMords, CMords, B, R0, Mplus):
    dR = B[:, :].reshape(R0.shape) - R0  # TODO perhaps check the dimension, potentially more turns in
    dR[np.isnan(dR)] = 0

    dphi = np.split(Mplus @ dR,2)  # TODO same as somewhere above

    lastCMh = _get_last_cm(bpm_ind, 1, BPMords, CMords[0])[1][0]
    lastCMv = _get_last_cm(bpm_ind, 1, BPMords, CMords[1])[1][0]
        
    SC, _ = SCsetCMs2SetPoints(SC, CMords[0][:lastCMh+1], -dphi[0][:lastCMh+1], skewness=False, method='add')
    SC, _ = SCsetCMs2SetPoints(SC, CMords[1][:lastCMv+1], -dphi[1][:lastCMv+1], skewness=True, method='add')
    return SC

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


def _is_stable_or_converged(n, eps, hist):  # Balance and Run  # TODO rethink
    if len(hist) < n:
        return False
    return (np.var(hist[-n:]) / np.std(hist[-n:])) < eps

def _wiggling(SC,BPMords,CMords,transmission_limit,wiggle_range=[50E-6,200E-6],wiggle_steps=32,nCM_for_wiggling=range(1, 9)):
    LOGGER.debug('Wiggling')
    dpts = _golden_donut_diffs(wiggle_range[0], wiggle_range[1], wiggle_steps)
    B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=None, orb_history=None)  # Inject...

    for nWiggleCM in nCM_for_wiggling:
        LOGGER.debug(f'Number of magnets used for wiggling: {nWiggleCM}. \n')
        tmpCMordsH = _get_last_cm(transmission_history[-1] - 1, nWiggleCM, BPMords, CMords[0])[0]  # Last CMs in horz
        tmpCMordsV = _get_last_cm(transmission_history[-1] - 1, nWiggleCM, BPMords, CMords[1])[0]  # Last CMs in vert

        for i in range(dpts.shape[1]):
            SC, _ = SCsetCMs2SetPoints(SC, tmpCMordsH, np.array([dpts[0, i]]), skewness=False, method='add')
            SC, _ = SCsetCMs2SetPoints(SC, tmpCMordsV, np.array([dpts[1, i]]), skewness=True, method='add')
            B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)
            if transmission_history[-1] >= transmission_limit:
                for _ in range(2):
                    B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)
                if _is_repro(transmission_history, 3):
                    LOGGER.debug('...completed')
                    return SC  # Continue with feedback
                
    if not transmission_history[-1] >= transmission_limit:
        raise RuntimeError('Wiggling failed')
    return SC