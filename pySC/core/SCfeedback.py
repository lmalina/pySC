import numpy as np

from pySC.core.SCgetBPMreading import SCgetBPMreading
from pySC.core.SCsetpoints import SCsetCavs2SetPoints, SCsetCMs2SetPoints
from pySC.utils import logging_tools


LOGGER = logging_tools.get_logger(__name__)


def SCfeedbackFirstTurn(SC, Mplus, R0=None, maxsteps=100, wiggleAfter=20, wiggleSteps=64, wiggleRange=np.array([50E-6, 200E-6]),
                        CMords=None, BPMords=None):
    if R0 is None:
        R0 = np.zeros((Mplus.shape[1], 1))
    BPMords, CMords = _check_ords(SC, BPMords, CMords, "FirstTurn")
    BPMhist = -1 * np.ones((1, 100))
    nWiggleCM = 1
    for n in range(maxsteps):
        B = SCgetBPMreading(SC, BPMords=BPMords)  # Inject...
        if np.all(np.isnan(B)):
            raise RuntimeError('SCfeedbackFirstTurn: FAIL (no BPM reading to begin with)')
        SC, BPMhist = _correction_step_firstturn(SC, BPMhist, BPMords, CMords, B, R0, Mplus)
        if _is_repro(BPMhist, 5) and _is_transmit(BPMhist):
            LOGGER.debug('SCfeedbackFirstTurn: Success')
            return SC
        elif _is_repro(BPMhist, wiggleAfter):
            LOGGER.debug('SCfeedbackFirstTurn: Wiggling')
            CMidxsH = _get_last_cms_dim_firstturn(B, nWiggleCM, BPMords, CMords[0])  # Last CMs in horz
            CMidxsV = _get_last_cms_dim_firstturn(B, nWiggleCM, BPMords, CMords[1])  # Last CMs in vert
            CMordsH = CMords[0][CMidxsH]
            CMordsV = CMords[1][CMidxsV]
            pts = np.array([[0, 0], [0, 0]])
            pts = np.append(pts, _golden_donut(wiggleRange[0], wiggleRange[1], wiggleSteps), axis=1)
            dpts = np.diff(pts, 1, 1)
            for i in range(dpts.shape[1]):
                SPH = dpts[0, i] * np.ones((len(CMordsH), 1))  # Horizontal setpoint change
                SPV = dpts[1, i] * np.ones((len(CMordsV), 1))  # Vertical setpoint change
                SC, _ = SCsetCMs2SetPoints(SC, CMordsH, SPH, skewness=False, method='add')
                SC, _ = SCsetCMs2SetPoints(SC, CMordsV, SPV, skewness=True, method='add')
                W = SCgetBPMreading(SC, BPMords=BPMords)
                BPMhist = _log_last_bpm(BPMhist, W)
                if _is_new(BPMhist):
                    BPMhist = _log_last_bpm(BPMhist, SCgetBPMreading(SC, BPMords=BPMords))
                    BPMhist = _log_last_bpm(BPMhist, SCgetBPMreading(SC, BPMords=BPMords))
                    BPMhist = _log_last_bpm(BPMhist, SCgetBPMreading(SC, BPMords=BPMords))
                    if _is_repro(BPMhist, 3):
                        BPMhist[0:3] = -1  # void last hist
                        nWiggleCM = 0  # Reset Wiggler CM number
                        break  # Continue with feedback
            B = SCgetBPMreading(SC, BPMords=BPMords)
            SC, BPMhist = _correction_step_firstturn(SC, BPMhist, BPMords, CMords, B, R0, Mplus)
            B = SCgetBPMreading(SC, BPMords=BPMords)
            SC, BPMhist = _correction_step_firstturn(SC, BPMhist, BPMords, CMords, B, R0, Mplus)
            B = SCgetBPMreading(SC, BPMords=BPMords)
            SC, BPMhist = _correction_step_firstturn(SC, BPMhist, BPMords, CMords, B, R0, Mplus)
            nWiggleCM = nWiggleCM + 1
    raise RuntimeError('SCfeedbackFirstTurn: FAIL (maxsteps reached)')


def SCfeedbackStitch(SC, Mplus, R0=np.zeros((2, 1)), nBPMs=4, maxsteps=30, nRepro=3, CMords=None, BPMords=None):
    BPMords, CMords = _check_ords(SC, BPMords, CMords, "Stitch")
    BPMhist = -1 * np.ones((1, 100))
    B = SCgetBPMreading(SC, BPMords=BPMords)
    if not _is_signal_stitch(B, nBPMs):
        LOGGER.debug('SCfeedbackStitch: Wiggling')
        pts = np.hstack((np.aray([[0], [0]]), _golden_donut(50E-6, 200E-6, 32)))
        dpts = np.diff(pts, axis=1)
        for nWiggleCM in range(1, 9):
            LOGGER.debug(f'SCfeedbackStitch: Number of magnets used for wiggling: {nWiggleCM}. \n')
            CMords = _get_last_cmords_stitch(B, nWiggleCM, BPMords, CMords)
            for i in range(len(dpts[0])):
                for ord in CMords:
                    SC, _ = SCsetCMs2SetPoints(SC, ord, dpts[0][i], skewness=False, method='add')
                    SC, _ = SCsetCMs2SetPoints(SC, ord, dpts[1][i], skewness=True, method='add')
                W = SCgetBPMreading(SC, BPMords=BPMords)
                BPMhist = _log_last_bpm(BPMhist, W)
                if _is_signal_stitch(W, nBPMs):
                    BPMhist = _log_last_bpm(BPMhist, SCgetBPMreading(SC, BPMords=BPMords))
                    BPMhist = _log_last_bpm(BPMhist, SCgetBPMreading(SC, BPMords=BPMords))
                    BPMhist = _log_last_bpm(BPMhist, SCgetBPMreading(SC, BPMords=BPMords))
                    if _is_repro(BPMhist, 3):
                        BPMhist[0:3] = -1  # void last hist
                        break
    B = SCgetBPMreading(SC, BPMords=BPMords)
    if not _is_signal_stitch(B, nBPMs):
        raise RuntimeError('SCfeedbackStitch: FAIL Wiggling failed')

    for steps in range(maxsteps):
        B = SCgetBPMreading(SC, BPMords=BPMords)  # Inject...
        BPMhist = _log_last_bpm(BPMhist, B)
        lBPM = len(B[0])
        Bx1 = B[0][0:lBPM / 2]
        By1 = B[1][0:lBPM / 2]
        Bx2 = B[0][(lBPM / 2 + 1):]
        By2 = B[1][(lBPM / 2 + 1):]
        DELTABx = Bx2 - Bx1
        DELTABy = By2 - By1
        DELTABx[(nBPMs + 1):] = 0
        DELTABy[(nBPMs + 1):] = 0
        R = [Bx1 - R0[0], DELTABx, By1 - R0[1], DELTABy]
        R[np.isnan(R)] = 0
        dphi = Mplus * R
        SC, _ = SCsetCMs2SetPoints(SC, CMords[0], -dphi[:len(CMords[0])], skewness=False, method='add')
        SC, _ = SCsetCMs2SetPoints(SC, CMords[1], -dphi[len(CMords[0]):], skewness=True, method='add')
        if _is_setback_stitch(BPMhist):
            RuntimeError('SCfeedbackStitch: FAIL Setback')
        if _is_repro(BPMhist, nRepro) and _is_transmit(BPMhist):
            LOGGER.debug('SCfeedbackStitch: Success')
            return SC
    raise RuntimeError('SCfeedbackStitch: FAIL Reached maxsteps')


def SCfeedbackRun(SC, Mplus, R0=None, eps=1e-5, target=0, maxsteps=30, scaleDisp=0, CMords=None, BPMords=None,
                  weight=None, do_plot=False):
    if R0 is None:
        R0 = np.zeros((Mplus.shape[1], 1))
    if weight is None:
        weight = np.ones((Mplus.shape[1], 1))
    BPMords, CMords = _check_ords(SC, BPMords, CMords, "Run")
    BPMhist = np.nan * np.ones((1, 100))
    for steps in range(maxsteps):
        B = SCgetBPMreading(SC, BPMords=BPMords, do_plot=do_plot)  # Inject ...
        R = np.array([B[0, :], B[1, :]])
        R[np.isnan(R)] = 0
        dphi = Mplus @ ((R - R0) * weight)
        if scaleDisp != 0:
            SC = SCsetCavs2SetPoints(SC, SC.ORD.RF, "Frequency", -scaleDisp * dphi[-1], method="add")
            dphi = dphi[:-1]
        SC, _ = SCsetCMs2SetPoints(SC, CMords[0], -dphi[:len(CMords[0])], skewness=False, method="add")
        SC, _ = SCsetCMs2SetPoints(SC, CMords[1], -dphi[len(CMords[0]):], skewness=True, method="add")
        BPMhist = np.roll(BPMhist, 1)
        BPMhist[0] = np.sqrt(np.mean(R ** 2, 1))
        if np.any(np.isnan(B[0, :])):
            raise RuntimeError('SCfeedbackRun: FAIL (lost transmission)')
        if BPMhist[0] < target and _is_stable_or_converged(min(10, maxsteps), eps, BPMhist):
            LOGGER.debug("SCfeedbackRun: Success (target reached)")
            return SC
        if _is_stable_or_converged(3, eps, BPMhist):
            LOGGER.debug(f"SCfeedbackRun: Success (converged after {steps:d} steps)")
            return SC

    if _is_stable_or_converged(min(10, maxsteps), eps, BPMhist) or maxsteps == 1:
        LOGGER.debug("SCfeedbackRun: Success (maxsteps reached)")
        return SC
    raise RuntimeError("SCfeedbackRun: FAIL (maxsteps reached, unstable)")


def SCfeedbackBalance(SC, Mplus, eps=1e-5, R0=np.zeros((2, 1)), maxsteps=10, CMords=None, BPMords=None):
    BPMords, CMords = _check_ords(SC, BPMords, CMords, "Balance")
    BPMindHist = -1 * np.ones(100)
    BRMShist = np.nan * np.ones(100)

    for steps in range(maxsteps):
        B = SCgetBPMreading(SC, BPMords=BPMords)  # Inject ...
        BPMindHist = _log_last_bpm(BPMindHist, B)
        lBPM = B.shape[1]
        Bx1 = B[0, 0:lBPM // 2]
        By1 = B[1, 0:lBPM // 2]
        Bx2 = B[0, lBPM // 2:]
        By2 = B[1, lBPM // 2:]
        DELTABx = Bx2 - Bx1
        DELTABy = By2 - By1
        R = np.vstack((Bx1 - R0[0, :], DELTABx, By1 - R0[1, :], DELTABy)).T
        R[np.isnan(R)] = 0
        dphi = Mplus @ R
        BRMShist = np.roll(BRMShist, 1)
        BRMShist[0] = np.sqrt(np.var(R, 1))
        SC, _ = SCsetCMs2SetPoints(SC, CMords[0], -dphi[:len(CMords[0])], skewness=False, method='add')
        SC, _ = SCsetCMs2SetPoints(SC, CMords[1], -dphi[len(CMords[0]):], skewness=True, method='add')
        if _is_setback_balance(BPMindHist):
            raise RuntimeError('SCfeedbackBalance: FAIL (setback)')
        if not _is_transmit(BPMindHist):
            raise RuntimeError('SCfeedbackBalance: FAIL (lost transmission)')
        if _is_stable_or_converged(3, eps, BRMShist):
            LOGGER.debug(f'SCfeedbackBalance: Success (converged after {steps} steps)')
            return SC

    if _is_stable_or_converged(min(10, maxsteps), eps, BRMShist):
        LOGGER.debug('SCfeedbackBalance: Success (maxsteps reached)')
        return SC
    raise RuntimeError('SCfeedbackBalance: FAIL (maxsteps reached, unstable)')


def _correction_step_firstturn(SC, BPMhist, BPMords, CMords, B, R0, Mplus):
    BPMhist = _log_last_bpm(BPMhist, B)
    R = B[:, :].reshape(R0.shape)  # TODO perhaps check the dimension, potentially more turns in
    dR = R - R0
    dR[np.isnan(dR)] = 0
    dphi = Mplus @ dR
    lastCMh = _get_last_cms_dim_firstturn(B, 1, BPMords, CMords[0])
    lastCMv = _get_last_cms_dim_firstturn(B, 1, BPMords, CMords[1])
    dphi[lastCMh + 1:len(CMords[0])] = 0
    dphi[len(CMords[0]) + lastCMv:len(CMords[0]) + len(CMords[1])] = 0
    SC, _ = SCsetCMs2SetPoints(SC, CMords[0], -dphi[len(CMords[0])], skewness=False, method='add')
    SC, _ = SCsetCMs2SetPoints(SC, CMords[1], -dphi[len(CMords[0]):len(CMords[0]) + len(CMords[1])], skewness=True, method='add')
    return SC, BPMhist


def _get_last_cms_dim_firstturn(B, n, BPMords, CMords):
    BPMinds = np.where(~np.isnan(B))[1]
    if len(BPMinds) == 0:
        lastBPMidx = len(BPMords)  # the last one
    else:
        lastBPMidx = BPMinds[-1]
    lastBPMord = BPMords[lastBPMidx]
    idxs = np.where(CMords <= lastBPMord)[0][-n:]
    return idxs


def _get_last_cmords_stitch(B, n, BPMords, CMords):
    dualCMords = np.intersect1d(CMords[0], CMords[1])  # Generate a list of CMs that can act in both planes.
    lastBPMidx = np.where(~np.isnan(B[0]))[0][-1]  # Get index of last reached BPM
    if len(lastBPMidx) == 0 or lastBPMidx > len(BPMords):  # If there is no beam loss in the first turn
        ords = dualCMords[-n:]  # ... just return the last n CMs
    else:  # in case the beam was lost in the first turn
        lastBPMord = BPMords[lastBPMidx]  # We can then find the ordinate of the last BPM.
        lastCMidx = np.where(dualCMords <= lastBPMord)[0][-1]  # Find the last CM upstream of the last BPM.
        ords = dualCMords[(lastCMidx - min(lastCMidx, n) + 1):lastCMidx]
    return ords


def _is_setback_balance(hist):
    return hist[0] < hist[1]


def _is_setback_stitch(hist):
    return hist[0] != 0 and hist[0] < hist[2] and hist[1] < hist[2]


def _is_signal_stitch(B, nBPMs):
    lastBPMidx = np.where(~np.isnan(B[0]))[0][-1]
    return lastBPMidx >= len(B[0]) / 2 + nBPMs


def _check_ords(SC, BPMords, CMords, start_str):
    if CMords is None:
        CMords = SC.ORD.CM[:]
    if BPMords is None:
        BPMords = SC.ORD.BPM[:]
    LOGGER.debug(f'SCfeedback{start_str}: Start')
    return BPMords, CMords


def _is_transmit(hist):
    return hist[0] == 0


def _is_repro(hist, N):
    return np.all(np.where(hist[0:N] == hist[0]))


def _is_new(hist):
    return hist[0] != hist[1]


def _log_last_bpm(hist, B):
    hist = np.roll(hist, 1)
    ord = _get_last_bpm_ord(B)
    hist[0] = ord if ord else 0
    return hist


def _get_last_bpm_ord(B):
    ord = np.where(np.isnan(B))[1]
    if len(ord) > 0:
        return ord[0] - 1
    return None


def _golden_donut(r0, r1, Npts):
    out = np.zeros((2, Npts))  # initialize output array
    phi = 2 * np.pi / ((1 + np.sqrt(5)) / 2)  # golden ratio
    theta = 0
    for n in range(Npts):
        out[:, n] = np.sqrt((r1 ** 2 - r0 ** 2) * n / (Npts - 1) + r0 ** 2) * np.array([np.cos(theta), np.sin(theta)])
        theta = theta + phi
    return out


def _is_stable_or_converged(n, eps, hist):  # Balance and Run
    cv = np.var(hist[:n], 1) / np.std(hist[:n])
    return cv < eps
