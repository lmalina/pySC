"""
Trajectory and Orbit
-------------

This module contains functions to correct trajectory (first turns(s)) and orbit.
"""
import numpy as np

from pySC.core.beam import bpm_reading
from pySC.utils import logging_tools
from pySC.utils.sc_tools import SCrandnc

LOGGER = logging_tools.get_logger(__name__)


def SCfeedbackFirstTurn(SC, Mplus, reference=None, CMords=None, BPMords=None,
                        maxsteps=100, nRepro=3, wiggle_after=20, wiggle_steps=32,
                        wiggle_range=(500E-6, 1000E-6)):
    """
    Achieves one-turn transmission

    Achieves a first turn in `SC.RING`.  This algorithm is based on the idea that
    repeated trajectory corrections calculated via a suitably regularized
    pseudo-inverse trajectory-response matrix `Mplus` will drive the BPM readings
    and CM settings to a fixed point.

    lim_{n->oo}  B_n = const. , with B_{n+1}  = Phi(Mplus . B_{n} ),     (1)

    where mapping Phi maps corrector kicks to BPM-readings.
    The RMS-values of both, BPM readings and CM settings, are determined by the
    regularization of Mplus.  Successively - during the course of repeated
    application of (1) - more and more transmission is achieved throughout the
    ring, more magnets are traversed near their magnetic center (which is hopefully
    at least somewhere near the BPM zero-point), resulting in decreased kicks.
    Otherwise,if trajectory correction cannot proceed further to next BPM
    the kicks of an increasing number of the last reached CMs are deterministically ``wiggled''
    until transmission to the next BPM is achieved. Then, application of (1) is resumed.

    Args:
        SC: SimulatedCommissioning class instance.
        Mplus: Pseudo-inverse trajectory-response matrix.
        reference: (None) target orbit in the format `[x_1 ... x_n y_1 ...y_n]`, where
                   [x_i,y_i]` is the target position at the i-th BPM.
        CMords: List of CM ordinates to be used for correction (SC.ORD.CM)
        BPMords: List of BPM ordinates at which the reading should be evaluated (SC.ORD.BPM)
        maxsteps: break, if this number of correction steps have been performed (default = 100)
        nRepro: (default 3)
        wiggle_after: Number of iterations without increased transmission to start wiggling. (default = 20)
        wiggle_steps: Number of wiggle steps to perform, before incresing the number. (default = 64)
        wiggle_range: Range ([min,max] in rad) within which to wiggle the CMs. (default = (500E-6, 1000E-6))

    Returns:
        SimulatedCommissioning class instance with corrected `SC.RING`

    """
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
        SC.set_cm_setpoints(CMords[0][:lastCMh + 1], -dphi[:lastCMh + 1], skewness=False, method='add')
        SC.set_cm_setpoints(CMords[1][:lastCMv + 1], -dphi[len(CMords[0]):len(CMords[0]) + lastCMv + 1], skewness=True, method='add')
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
    """
    Achieves 2-turn transmission

    The purpose of this function is to go from 1-turn transmission to 2-turn
    transmission. This is done by applying orbit correction based on the pseudo
    inverse trajectory response matrix 'Mplus' applied to the first BPMs in
    the 'SC.RING'. The reading of the BPMs in the second turn is corrected
    towards the reading of these BPMs in the first turn. This approach has been
    seen to be more stable than the direct application of the two-turn inverse
    response matrix to the two-turn BPM data.

    Args:
        SC: SimulatedCommissioning class instance.
        Mplus: Pseudo-inverse trajectory-response matrix.
        reference: (None) target orbit in the format `[x_1 ... x_n y_1 ...y_n]`, where
                   [x_i,y_i]` is the target position at the i-th BPM.
        CMords: List of CM ordinates to be used for correction (SC.ORD.CM)
        BPMords: List of BPM ordinates at which the reading should be evaluated (SC.ORD.BPM)
        maxsteps: break, if this number of correction steps have been performed (default = 100)
        nRepro: (default 3)
        wiggle_steps: Number of wiggle steps to perform, before incresing the number. (default = 64)
        wiggle_range: Range ([min,max] in rad) within which to wiggle the CMs. (default = (500E-6, 1000E-6))

    Returns:
        SimulatedCommissioning class instance with corrected `SC.RING`

    Examples:
        Calculate the 2-turn response matrix and get the pseudo inverse using a Tikhonov regularization
        parameter of 10. Switch the injection pattern to 2 turns and apply the stitching using the first
        three BPMs, for a maximum of 20 steps::

            RM2 = SCgetModelRM(SC, SC.ORD.BPM, SC.ORD.CM, nTurns=2)
            Minv2 = SCgetPinv(RM2, alpha=10)
            SC.INJ.nTurns = 2
            SC = SCfeedbackStitch(SC, Minv2, nBPMs=3, maxsteps=20)

    """
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
        SC.set_cm_setpoints(CMords[0], -dphi[:len(CMords[0])], skewness=False, method="add")
        SC.set_cm_setpoints(CMords[1], -dphi[len(CMords[0]):], skewness=True, method="add")
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
    """
    Balance two-turn BPM readings

    Generates a period-1 closed orbit, after two-turn transmission has been
    achieved. This is done by iteratively applying correction steps, calculated
    based on the pseudo-inverse two-turn trajectory response matrix `Mplus`.  The
    trajectory in the first turn is corrected towards the reference orbit `reference`,
    whereas the trajectory in the second turn is corrected towards the trajectory
    measured in the first turn; this approach seems to be more stable than the
    directly application of the two-turn TRM to the two-turn BPM readings.
    It converges to a state where BPM readings in both turns are very similar,
    indicating a period-1 closed orbit.

    Args:
        SC: SimulatedCommissioning class instance.
        Mplus: Pseudo-inverse trajectory-response matrix.
        reference: (None) target orbit in the format `[x_1 ... x_n y_1 ...y_n]`, where
                   [x_i,y_i]` is the target position at the i-th BPM.
        CMords: List of CM ordinates to be used for correction (SC.ORD.CM)
        BPMords: List of BPM ordinates at which the reading should be evaluated (SC.ORD.BPM)
        maxsteps: break, if this number of correction steps have been performed (default = 100)
        nRepro: (default 3)
        eps: break, if the coefficient of variation of the RMS BPM reading is below this value

    Returns:
        SimulatedCommissioning class instance with corrected `SC.RING`

    """
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
        SC.set_cm_setpoints(CMords[0], -dphi[:len(CMords[0])], skewness=False, method="add")
        SC.set_cm_setpoints(CMords[1], -dphi[len(CMords[0]):], skewness=True, method="add")
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
    """
    iterative orbit correction

    Iteratively applies orbit corrections using the pseudoinverse of the
    trajectory response matrix `Mplus`, until a break-condition specified by one
    of 'eps', 'target' or 'maxsteps' is met.
    The dispersion can be included, thus the rf frequency as a correction
    parameter. If the dispersion is to be included, `Mplus` has to have the size
    `(len(SC.ORD.HCM) + len(SC.ORD.VCM) + 1) x len(SC.ORD.BPM)`, otherwise the size
    `(len(SC.ORD.HCM) + len(SC.ORD.VCM)) x len(SC.ORD.BPM)`, or correspondingly if the CM
    and/or BPM ordinates for the correction is explicitly given (see options below). `SC.RING` is
    assumed to be a lattice with transmission through all considered turns.
    Raises RuntimeError if transmission is lost.

    Args:
        SC: SimulatedCommissioning class instance.
        Mplus: Pseudo-inverse trajectory-response matrix.
        reference: (None) target orbit in the format `[x_1 ... x_n y_1 ...y_n]`, where
                   [x_i,y_i]` is the target position at the i-th BPM.
        CMords: List of CM ordinates to be used for correction (SC.ORD.CM)
        BPMords: List of BPM ordinates at which the reading should be evaluated (SC.ORD.BPM)
        maxsteps: break, if this number of correction steps have been performed (default = 100)
        nRepro: (default 3)
        eps: break, if the coefficient of variation of the RMS BPM reading is below this value
        target: (default =0 ) break, if the RMS BPM reading reaches this value
        scaleDisp: (default =0 ) Scaling factor for and flag indicating if the dispersion is included in the response matrix

    Returns:
        SimulatedCommissioning class instance with corrected `SC.RING`

    Examples:

        Switch to orbit mode, get the model response matrix and dispersion. Calculate the psudo-inverse
        while scaling the dispersion by 1E7 and using a Tikhonov regularization parameter of 10.
        Finally, apply  and apply orbit feedback including dispersion::

            SC.INJ.trackMode = 'ORB'
            MCO = SCgetModelRM(SC, SC.ORD.BPM, SC.ORD.CM, trackMode='ORB')
            eta = SCgetModelDispersion(SC, SC.ORD.BPM, SC.ORD.Cavity,
                                       trackMode='ORB', Z0=np.zeros(6),
                                       nTurns=1, rfStep=1E3,
                                       useIdealRing=True)
            MinvCO = SCgetPinv(np.column_stack((MCO, 1E8 * eta)), alpha=10)
            SC = SCfeedbackRun(SC, MinvCO, target=0, maxsteps=50, scaleDisp=1E8)

    """
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
            SC.set_cavity_setpoints(SC.ORD.RF, -scaleDisp * dphi[-1], "Frequency", method="add")
            dphi = dphi[:-1]  # TODO the last setpoint is cavity frequency
        SC.set_cm_setpoints(CMords[0], -dphi[:len(CMords[0])], skewness=False, method="add")
        SC.set_cm_setpoints(CMords[1], -dphi[len(CMords[0]):], skewness=True, method="add")
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
    bpm_readings = bpm_reading(SC, bpm_ords=BPMords)[0]
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
            SC.set_cm_setpoints(tmpCMordsH, dpts[0, i], skewness=False, method='add')
            SC.set_cm_setpoints(tmpCMordsV, dpts[1, i], skewness=True, method='add')
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
