import numpy as np

from pySC.core.SCgetBPMreading import SCgetBPMreading
from pySC.core.SCsetpoints import SCsetCavs2SetPoints, SCsetCMs2SetPoints
from pySC.utils import logging_tools


LOGGER = logging_tools.get_logger(__name__)


def SCfeedbackFirstTurn(SC, Mplus, R0=None, CMords=None, BPMords=None, maxsteps=100, wiggleAfter=20, wiggleSteps=64, wiggleRange=np.array([50E-6, 200E-6]),):
    BPMords, CMords, R0 = _check_ords(SC, Mplus, R0, BPMords, CMords, "FirstTurn")
    nWiggleCM = 1
    transmission_history, rms_orbit_history = [],[]
    for n in range(maxsteps):
        B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history, do_plot=True)  # Inject...
        if transmission_history[-1] == 0:
            raise RuntimeError('SCfeedbackFirstTurn: FAIL (no BPM reading to begin with)')
        SC = _correction_step_firstturn(SC, transmission_history[-1]-1, BPMords, CMords, B, R0, Mplus)
        if _is_repro(transmission_history, 5) and transmission_history[-1] == B.shape[1]:   # last five the same and full transmission
            LOGGER.debug('SCfeedbackFirstTurn: Success')
            return SC
        elif _is_repro(transmission_history, wiggleAfter):
            print('SCfeedbackFirstTurn: Wiggling')
            CMidxsH = _get_last_cm_inds(transmission_history[-1] - 1, nWiggleCM, BPMords, CMords[0])  # Last CMs in horz
            CMidxsV = _get_last_cm_inds(transmission_history[-1] - 1, nWiggleCM, BPMords, CMords[1])  # Last CMs in vert
            CMordsH = CMords[0][CMidxsH]
            CMordsV = CMords[1][CMidxsV]
            pts = np.array([[0, 0], [0, 0]])
            pts = np.append(pts, _golden_donut(wiggleRange[0], wiggleRange[1], wiggleSteps), axis=1)
            dpts = np.diff(pts, 1, 1)
            for i in range(dpts.shape[1]):
                SC, _ = SCsetCMs2SetPoints(SC, CMordsH, dpts[0, i], skewness=False, method='add')
                SC, _ = SCsetCMs2SetPoints(SC, CMordsV, dpts[1, i], skewness=True, method='add')
                W, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history, do_plot=True)
                if transmission_history[-1] != transmission_history[-2]:
                    for _ in range(3):
                        B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history, do_plot=True)
                    if _is_repro(transmission_history, 3):
                        transmission_history[0:3] = -1  # void last hist  # TODO list -> should  be last 3
                        nWiggleCM = 1  # Reset Wiggler CM number
                        break  # Continue with feedback
            for _ in range(3):
                B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history, do_plot=True)
                SC = _correction_step_firstturn(SC, transmission_history[-1]-1, BPMords, CMords, B, R0, Mplus)
            nWiggleCM = nWiggleCM + 1
    raise RuntimeError('SCfeedbackFirstTurn: FAIL (maxsteps reached)')


def SCfeedbackStitch(SC, Mplus, R0=None, CMords=None, BPMords=None, nBPMs=4, maxsteps=30, nRepro=3):
    BPMords, CMords, R0 = _check_ords(SC, Mplus, R0, BPMords, CMords, "Stitch")
    # assumes 2 turns
    if SC.INJ.nTurns != 2:
        raise ValueError("Stitching works only with two turns.")
    B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, do_plot=True)
    transmission_limit = int(B.shape[1] / 2 + nBPMs)
    if not transmission_history[-1] >= transmission_limit:
        LOGGER.debug('SCfeedbackStitch: Wiggling')
        pts = np.hstack((np.aray([[0], [0]]), _golden_donut(50E-6, 200E-6, 32)))
        dpts = np.diff(pts, axis=1)
        for nWiggleCM in range(1, 9):  # TODO basicalyy the sme as First turn
            LOGGER.debug(f'SCfeedbackStitch: Number of magnets used for wiggling: {nWiggleCM}. \n')
            CMords2 = _get_last_cm_ords(B, nWiggleCM, BPMords, CMords)
            for i in range(len(dpts[0])):
                for ord in CMords2:
                    SC, _ = SCsetCMs2SetPoints(SC, ord, dpts[0][i], skewness=False, method='add')
                    SC, _ = SCsetCMs2SetPoints(SC, ord, dpts[1][i], skewness=True, method='add')
                W, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history, do_plot=True)
                if transmission_history[-1] >= transmission_limit:
                    for _ in range(3):
                        W, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history, do_plot=True)
                    if _is_repro(transmission_history, 3):
                        transmission_history[0:3] = -1  # void last hist  # TODO list -> should  be last 3
                        break
    B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history, do_plot=True)
    if not transmission_history[-1] >= transmission_limit:
        raise RuntimeError('SCfeedbackStitch: FAIL Wiggling failed')

    for steps in range(maxsteps):
        B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, ind_history=transmission_history, orb_history=rms_orbit_history, do_plot=True)

        lBPM = len(B[0])
        delta_b = np.squeeze(np.diff(B.reshape(2, lBPM // 2, 2), axis=2))
        delta_b[:, nBPMs:] = 0
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


def SCfeedbackRun(SC, Mplus, R0=None, CMords=None, BPMords=None, eps=1e-5, target=0, maxsteps=30, scaleDisp=0, weight=None, do_plot=False):
    if weight is None:
        weight = np.ones((Mplus.shape[1], 1))
    BPMords, CMords, R0 = _check_ords(SC, Mplus, R0, BPMords, CMords, "Run")
    B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords,
                                                                          do_plot=True)  # Inject ...
    for steps in range(maxsteps):
        B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords,ind_history=transmission_history, orb_history=rms_orbit_history, do_plot=True) # Inject ...

        if np.any(np.isnan(B[0, :])):
            raise RuntimeError('SCfeedbackRun: FAIL (lost transmission)')
        if max(rms_orbit_history[-1]) < target and _is_stable_or_converged(min(10, maxsteps), eps, rms_orbit_history):
            LOGGER.debug("SCfeedbackRun: Success (target reached)")
            return SC
        if _is_stable_or_converged(3, eps, rms_orbit_history):
            LOGGER.debug(f"SCfeedbackRun: Success (converged after {steps:d} steps)")
            return SC

        R = B[:,:].reshape(R0.shape)
        R[np.isnan(R)] = 0
        dphi = np.dot(Mplus, ((R - R0) * weight))
        if scaleDisp != 0:   # TODO this is weight
            SC = SCsetCavs2SetPoints(SC, SC.ORD.RF, "Frequency", -scaleDisp * dphi[-1], method="add")
            dphi = dphi[:-1]  # TODO the last setpoint is cavity frequency
        SC, _ = SCsetCMs2SetPoints(SC, CMords[0], -dphi[:len(CMords[0])], skewness=False, method="add")
        SC, _ = SCsetCMs2SetPoints(SC, CMords[1], -dphi[len(CMords[0]):], skewness=True, method="add")

    if _is_stable_or_converged(min(10, maxsteps), eps, rms_orbit_history) or maxsteps == 1:
        LOGGER.debug("SCfeedbackRun: Success (maxsteps reached)")
        return SC
    raise RuntimeError("SCfeedbackRun: FAIL (maxsteps reached, unstable)")


def SCfeedbackBalance(SC, Mplus, R0=None, CMords=None, BPMords=None, eps=1e-5, maxsteps=10):
    BPMords, CMords, R0 = _check_ords(SC, Mplus, R0, BPMords, CMords, "Balance")
    B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, do_plot=True)
    for steps in range(maxsteps):
        B, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, BPMords=BPMords, do_plot=True)
        lBPM = B.shape[1]
        delta_b = np.diff(B.reshape(2, lBPM // 2, 2), axis=2)

        Bx1 = B[0, 0:lBPM // 2]
        By1 = B[1, 0:lBPM // 2]
        Bx2 = B[0, lBPM // 2:]
        By2 = B[1, lBPM // 2:]
        DELTABx = Bx2 - Bx1
        DELTABy = By2 - By1
        R = np.vstack((Bx1 - R0[0, :], DELTABx, By1 - R0[1, :], DELTABy)).T
        R[np.isnan(R)] = 0
        dphi = Mplus @ R
        SC, _ = SCsetCMs2SetPoints(SC, CMords[0], -dphi[:len(CMords[0])], skewness=False, method='add')
        SC, _ = SCsetCMs2SetPoints(SC, CMords[1], -dphi[len(CMords[0]):], skewness=True, method='add')
        if transmission_history[-1] < transmission_history[-2]:
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


def _check_ords(SC, Mplus, R0, BPMords, CMords, start_str):
    if CMords is None:
        CMords = SC.ORD.CM[:]
    if BPMords is None:
        BPMords = SC.ORD.BPM[:]
    if R0 is None:
        R0 = np.zeros((Mplus.shape[1], 1))
    LOGGER.debug(f'SCfeedback{start_str}: Start')
    return BPMords, CMords, R0


def _bpm_reading_and_logging(SC, BPMords, ind_history=None, orb_history=None, do_plot=False):
    bpm_readings = SCgetBPMreading(SC, BPMords=BPMords, do_plot=do_plot)
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


def _is_repro(hist, N):
    if len(hist) >= N:
        return np.max(np.abs(np.array(hist[-N:])-hist[-1])) == 0
    return False


def _golden_donut(r0, r1, Npts):
    out = np.zeros((2, Npts))  # initialize output array
    phi = 2 * np.pi / ((1 + np.sqrt(5)) / 2)  # golden ratio
    theta = 0
    for n in range(Npts):
        out[:, n] = np.sqrt((r1 ** 2 - r0 ** 2) * n / (Npts - 1) + r0 ** 2) * np.array([np.cos(theta), np.sin(theta)])
        theta = theta + phi
    return out


def _is_stable_or_converged(n, eps, hist):  # Balance and Run  # TODO rethink
    cv = np.var(hist[-n:]) / np.std(hist[-n:])
    return cv < eps
