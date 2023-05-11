import numpy as np

from pySC.core.SCgetBPMreading import SCgetBPMreading
from pySC.core.SCsetpoints import SCsetCavs2SetPoints, SCsetCMs2SetPoints
from pySC.utils import logging_tools


LOGGER = logging_tools.get_logger(__name__)


def SCfeedbackFirstTurn(SC, Mplus, R0=None, CMords=None, BPMords=None, maxsteps=100, wiggleAfter=20, wiggleSteps=64, wiggleRange=np.array([50E-6, 200E-6]),):
    LOGGER.debug('SCfeedbackFirstTurn: Start')
    BPMords, CMords, R0 = _check_ords(SC, Mplus, R0, BPMords, CMords)
    nWiggleCM = 1
    transmission_history, rms_orbit_history = None, None
    for n in range(maxsteps):
        B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)  # Inject...
        if transmission_history[-1] == 0:
            raise RuntimeError('SCfeedbackFirstTurn: FAIL (no BPM reading to begin with)')
        SC = _correction_step_firstturn(SC, transmission_history[-1]-1, BPMords, CMords, B, R0, Mplus)
        if _is_repro(transmission_history, 5) and transmission_history[-1] == B.shape[1]:   # last five the same and full transmission
            LOGGER.debug('SCfeedbackFirstTurn: Success')
            return SC
        if _is_repro(transmission_history, wiggleAfter):
            print('SCfeedbackFirstTurn: Wiggling')
            CMidxsH = _get_last_cm_inds(transmission_history[-1] - 1, nWiggleCM, BPMords, CMords[0])  # Last CMs in horz
            CMidxsV = _get_last_cm_inds(transmission_history[-1] - 1, nWiggleCM, BPMords, CMords[1])  # Last CMs in vert
            CMordsH = CMords[0][CMidxsH]
            CMordsV = CMords[1][CMidxsV]
            dpts = _golden_donut_diffs(wiggleRange[0], wiggleRange[1], wiggleSteps)
            for i in range(dpts.shape[1]):
                SC, _ = SCsetCMs2SetPoints(SC, CMordsH, dpts[0, i], skewness=False, method='add')
                SC, _ = SCsetCMs2SetPoints(SC, CMordsV, dpts[1, i], skewness=True, method='add')
                W, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)
                if transmission_history[-1] != transmission_history[-2]:
                    for _ in range(3):
                        B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)
                    if _is_repro(transmission_history, 3):
                        transmission_history[0:3] = -1  # void last hist  # TODO list -> should  be last 3
                        nWiggleCM = 1  # Reset Wiggler CM number
                        break  # Continue with feedback
            for _ in range(3):
                B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)
                SC = _correction_step_firstturn(SC, transmission_history[-1]-1, BPMords, CMords, B, R0, Mplus)
            nWiggleCM = nWiggleCM + 1
    raise RuntimeError('SCfeedbackFirstTurn: FAIL (maxsteps reached)')


def SCfeedbackStitch(SC, Mplus, R0=None, CMords=None, BPMords=None, nBPMs=4, maxsteps=30, nRepro=3):
    LOGGER.debug('SCfeedbackStitch: Start')
    BPMords, CMords, R0 = _check_ords(SC, Mplus, R0, BPMords, CMords)
    # assumes 2 turns
    if SC.INJ.nTurns != 2:
        raise ValueError("Stitching works only with two turns.")
    B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords)
    transmission_limit = int(B.shape[1] / 2 + nBPMs)
    if not transmission_history[-1] >= transmission_limit:
        LOGGER.debug('SCfeedbackStitch: Wiggling')
        dpts = _golden_donut_diffs(50E-6, 200E-6, 32)
        for nWiggleCM in range(1, 9):  # TODO basicalyy the sme as First turn
            LOGGER.debug(f'SCfeedbackStitch: Number of magnets used for wiggling: {nWiggleCM}. \n')
            CMords2 = _get_last_cm_ords(B, nWiggleCM, BPMords, CMords)
            for i in range(len(dpts[0])):
                SC, _ = SCsetCMs2SetPoints(SC, CMords2, dpts[0, i], skewness=False, method='add')
                SC, _ = SCsetCMs2SetPoints(SC, CMords2, dpts[1, i], skewness=True, method='add')
                W, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)
                if transmission_history[-1] >= transmission_limit:
                    for _ in range(3):
                        W, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)
                    if _is_repro(transmission_history, 3):
                        transmission_history[0:3] = -1  # void last hist  # TODO list -> should  be last 3
                        break
    B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)
    if not transmission_history[-1] >= transmission_limit:
        raise RuntimeError('SCfeedbackStitch: FAIL Wiggling failed')

    for steps in range(maxsteps):
        B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)
        lBPM = len(B[0])
        delta_b = [B[0][lBPM // 2:] - B[0][:lBPM // 2], B[1][lBPM // 2:] - B[1][:lBPM // 2]]
        delta_b[0][nBPMs:] = 0
        delta_b[1][nBPMs:] = 0
        #delta_b = np.squeeze(np.diff(B.reshape(2, 2, lBPM // 2), axis=1))
        #delta_b[:, nBPMs:] = 0
        R=np.concatenate((B[:, :lBPM//2], delta_b), axis=1).ravel()
        R0= R0.reshape(2,lBPM)
        R0[:,lBPM//2:] = 0
        R0=R0.reshape(Mplus.shape[1])

        R[np.isnan(R)] = 0
        dphi = np.dot(Mplus, (R-R0))
        SC, _ = SCsetCMs2SetPoints(SC, CMords[0], -dphi[:len(CMords[0])], skewness=False, method='add')
        SC, _ = SCsetCMs2SetPoints(SC, CMords[1], -dphi[len(CMords[0]):], skewness=True, method='add')
        if transmission_history[-1] < transmission_history[-2]:
            RuntimeError('SCfeedbackStitch: FAIL Setback')
        if _is_repro(transmission_history, nRepro) and transmission_history[-1] == B.shape[1]: # TODO remove
            LOGGER.debug('SCfeedbackStitch: Success')
            return SC
    raise RuntimeError('SCfeedbackStitch: FAIL Reached maxsteps')


def SCfeedbackRun(SC, Mplus, R0=None, CMords=None, BPMords=None, eps=1e-5, target=0, maxsteps=30, scaleDisp=0):
    LOGGER.debug('SCfeedbackRun: Start')
    BPMords, CMords, R0 = _check_ords(SC, Mplus, R0, BPMords, CMords)

    B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords)  # Inject ...
    for steps in range(maxsteps):
        R = B[:, :].reshape(R0.shape)
        R[np.isnan(R)] = 0
        dphi = np.dot(Mplus, (R - R0))
        if scaleDisp != 0:   # TODO this is weight
            SC = SCsetCavs2SetPoints(SC, SC.ORD.RF, "Frequency", -scaleDisp * dphi[-1], method="add")
            dphi = dphi[:-1]  # TODO the last setpoint is cavity frequency
        SC, _ = SCsetCMs2SetPoints(SC, CMords[0], -dphi[:len(CMords[0])], skewness=False, method="add")
        SC, _ = SCsetCMs2SetPoints(SC, CMords[1], -dphi[len(CMords[0]):], skewness=True, method="add")
        B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)  # Inject ...

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


def SCfeedbackBalance(SC, Mplus, R0=None, CMords=None, BPMords=None, eps=1e-5, maxsteps=10):
    LOGGER.debug('SCfeedbackBalance: Start')
    BPMords, CMords, R0 = _check_ords(SC, Mplus, R0, BPMords, CMords)
    transmission_history = None
    rms_orbit_history = None
    for steps in range(maxsteps):
        B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history)
        lBPM = len(B[0])
        # delta_b = np.squeeze(np.diff(B.reshape(2, lBPM // 2, 2), axis=2))
        delta_b = [B[0][lBPM // 2:] - B[0][:lBPM // 2], B[1][lBPM // 2:] - B[1][:lBPM // 2]]

        R=np.concatenate((B[:, :lBPM//2], delta_b), axis=1).ravel()
        R0= R0.reshape(2,lBPM)
        R0[:,lBPM//2:] = 0
        R0=R0.reshape(Mplus.shape[1])

        R[np.isnan(R)] = 0
        dphi = np.dot(Mplus, (R-R0))

        SC, _ = SCsetCMs2SetPoints(SC, CMords[0], -dphi[:len(CMords[0])], skewness=False, method='add')
        SC, _ = SCsetCMs2SetPoints(SC, CMords[1], -dphi[len(CMords[0]):], skewness=True, method='add')
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
    dphi = Mplus @ dR
    lastCMh = _get_last_cm_inds(bpm_ind, 1, BPMords, CMords[0])[0]
    lastCMv = _get_last_cm_inds(bpm_ind, 1, BPMords, CMords[1])[0]
    SC, _ = SCsetCMs2SetPoints(SC, CMords[0][:lastCMh+1], -dphi[:lastCMh+1], skewness=False, method='add')
    SC, _ = SCsetCMs2SetPoints(SC, CMords[1][:lastCMv+1], -dphi[len(CMords[0]):len(CMords[0]) + lastCMv+1], skewness=True, method='add')
    return SC


def _get_last_cm_inds(lastBPMidx, n, BPMords, CMords):  # firstturn
    return np.where(CMords <= BPMords[lastBPMidx])[0][-n:]


def _get_last_cm_ords(lastBPMidx, n, BPMords, CMords):  # Stitch
    dualCMords = np.intersect1d(CMords[0], CMords[1])  # Generate a list of CMs that can act in both planes.
    if lastBPMidx > len(BPMords):  # If there is no beam loss in the first turn
        return dualCMords[-n:]  # ... just return the last n CMs
    return dualCMords[np.where(dualCMords <= BPMords[lastBPMidx])[0][-n:]]


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
