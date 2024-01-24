"""
Trajectory and Orbit
-------------

This module contains functions to correct trajectory (first turns(s)) and orbit.
In all functions, the provided response matrix needs to be matching:
    registered BPMs in SC.ORD.BPM or bpm_ords if provided
    registered CMs in SC.ORD.CM or cm_ords if provided
    assumes all BPMs are used both horizontal and vertical plane
"""
import numpy as np
from typing import Tuple

from pySC.core.beam import bpm_reading
from pySC.utils import logging_tools
from pySC.utils import sc_tools
from pySC.core.constants import SETTING_ADD

LOGGER = logging_tools.get_logger(__name__)
NREPRO: int = 5
WIGGLE_AFTER: int = 20
WIGGLE_ANGLE_STEPS: int = 32
WIGGLE_ANGLE_RANGE: Tuple[float, float] = (500E-6, 1000E-6)


def first_turn(SC, response_matrix, reference=None, cm_ords=None, bpm_ords=None, maxsteps=100, **pinv_params):
    """
    Achieves one-turn transmission

    Achieves a first turn in `SC.RING`.  This algorithm is based on the idea that
    repeated trajectory corrections calculated via a suitably regularized
    pseudo-inverse  of `response matrix` will drive the BPM readings
    and CM settings to a fixed point.

    lim_{n->oo}  B_n = const. , with B_{n+1}  = Phi(response_matrix^{-1} . B_{n} ),     (1)

    where mapping Phi maps corrector kicks to BPM-readings.
    The RMS-values of both, BPM readings and CM settings, are determined by the
    regularization of pseudo-inverted response matrix.  Successively - during the course of repeated
    application of (1) - more and more transmission is achieved throughout the
    ring, more magnets are traversed near their magnetic center (which is hopefully
    at least somewhere near the BPM zero-point), resulting in decreased kicks.
    Otherwise,if trajectory correction cannot proceed further to next BPM
    the kicks of an increasing number of the last reached CMs are deterministically ``wiggled''
    until transmission to the next BPM is achieved. Then, application of (1) is resumed.

    Args:
        SC: SimulatedCommissioning class instance.
        response_matrix: Trajectory-response matrix.
        reference: (None) target orbit in the format `[x_1 ... x_n y_1 ...y_n]`, where
                   [x_i,y_i]` is the target position at the i-th BPM.
        cm_ords: List of CM ordinates to be used for correction (SC.ORD.CM)
        bpm_ords: List of BPM ordinates at which the reading should be evaluated (SC.ORD.BPM)
        maxsteps: break, if this number of correction steps have been performed (default = 100)

    Returns:
        SimulatedCommissioning class instance with corrected `SC.RING`

    """
    LOGGER.debug('First turn threading: Start')
    bpm_ords, cm_ords, reference = _check_ords(SC, response_matrix, reference, bpm_ords, cm_ords)
    bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, bpm_ords=bpm_ords)  # Inject...
    Mplus = sc_tools.SCgetPinv(response_matrix, **pinv_params)

    for n in range(maxsteps):
        if transmission_history[-1] == 0:
            raise RuntimeError('First turn threading: FAIL (no BPM reading to begin with)')
        
        # Set BPM readings
        measurement = bpm_readings[:, :].reshape(reference.shape)
        measurement[np.isnan(measurement)] = 0

        # Correction step
        dphi = np.dot(Mplus, (measurement - reference))
        lastCMh = _get_last_cm(transmission_history[-1] - 1, 1, bpm_ords, cm_ords[0])[1][0]
        lastCMv = _get_last_cm(transmission_history[-1] - 1, 1, bpm_ords, cm_ords[1])[1][0]
        SC.set_cm_setpoints(cm_ords[0][:lastCMh + 1], -dphi[:lastCMh + 1], skewness=False, method=SETTING_ADD)
        SC.set_cm_setpoints(cm_ords[1][:lastCMv + 1], -dphi[len(cm_ords[0]):len(cm_ords[0]) + lastCMv + 1],
                            skewness=True, method=SETTING_ADD)
        bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(
            SC, bpm_ords=bpm_ords, ind_history=transmission_history, orb_history=rms_orbit_history)  # Inject...
 
        # Check stopping criteria
        if _is_repro(transmission_history, NREPRO) and transmission_history[-1] == bpm_readings.shape[1]:   # last three the same and full transmission
            LOGGER.debug('First turn threading: Success')
            return SC
        if _is_repro(transmission_history, WIGGLE_AFTER):
            SC = _wiggling(SC, bpm_ords, cm_ords, transmission_history[-1] + 1)

    raise RuntimeError('First turn threading: FAIL (maxsteps reached)')


def stitch(SC, response_matrix, reference=None, cm_ords=None, bpm_ords=None, n_bpms=4, maxsteps=30, **pinv_params):
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
        response_matrix: Trajectory-response matrix.
        reference: (None) target orbit in the format `[x_1 ... x_n y_1 ...y_n]`, where
                   [x_i,y_i]` is the target position at the i-th BPM.
        cm_ords: List of CM ordinates to be used for correction (SC.ORD.CM)
        bpm_ords: List of BPM ordinates at which the reading should be evaluated (SC.ORD.BPM)
        n_bpms: Number of BPMs to match the trajectory in first and second turn
        maxsteps: break, if this number of correction steps have been performed (default = 100)

    Returns:
        SimulatedCommissioning class instance with corrected `SC.RING`

    Examples:
        Calculate the 2-turn response matrix and get the pseudo inverse using a Tikhonov regularization
        parameter of 10. Switch the injection pattern to 2 turns and apply the stitching using the first
        three BPMs, for a maximum of 20 steps::

            RM2 = SCgetModelRM(SC, SC.ORD.BPM, SC.ORD.CM, nTurns=2)
            SC.INJ.nTurns = 2
            SC = stitch(SC, RM2, alpha=10, nBPMs=3, maxsteps=20)

    """
    LOGGER.debug('Second turn stitching: Start')
    if SC.INJ.nTurns != 2:
        raise ValueError("Stitching works only with two turns.")
    bpm_ords, cm_ords, reference = _check_ords(SC, response_matrix, reference, bpm_ords, cm_ords)
    bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, bpm_ords=bpm_ords)  # Inject...
    transmission_limit = len(bpm_ords) + n_bpms
    if transmission_history[-1] < len(bpm_ords):
        raise RuntimeError("Stitching works only with full 1st turn transmission.")
    
    # Check if minimum transmission for algorithm to work is reached
    if transmission_history[-1] < transmission_limit:
        SC = _wiggling(SC, bpm_ords, cm_ords, transmission_limit)
        bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(
            SC, bpm_ords=bpm_ords, ind_history=transmission_history, orb_history=rms_orbit_history)
        if transmission_history[-1] < transmission_limit:
            raise RuntimeError("Not enough transmission for stitching to work.")

    # Prepare reference
    reference = reference.reshape(2, len(bpm_readings[0]))
    reference[:, len(bpm_ords):] = 0
    reference = reference.reshape(response_matrix.shape[0])
    Mplus = sc_tools.SCgetPinv(response_matrix, **pinv_params)

    # Main loop
    for steps in range(maxsteps):
        # Set BPM readings
        delta_b = np.zeros((2, len(bpm_ords)))
        delta_b[0][:n_bpms] = bpm_readings[0][len(bpm_ords):len(bpm_ords) + n_bpms] - bpm_readings[0][:n_bpms]
        delta_b[1][:n_bpms] = bpm_readings[1][len(bpm_ords):len(bpm_ords) + n_bpms] - bpm_readings[1][:n_bpms]
        measurement = np.concatenate((bpm_readings[:, :len(bpm_ords)], delta_b), axis=1).ravel()
        measurement[np.isnan(measurement)] = 0

        # Correction step
        dphi = np.dot(Mplus, (measurement - reference))
        SC.set_cm_setpoints(cm_ords[0], -dphi[:len(cm_ords[0])], skewness=False, method=SETTING_ADD)
        SC.set_cm_setpoints(cm_ords[1], -dphi[len(cm_ords[0]):], skewness=True, method=SETTING_ADD)
        bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(
            SC, bpm_ords=bpm_ords, ind_history=transmission_history, orb_history=rms_orbit_history)

        # Check stopping criteria
        if transmission_history[-1] < transmission_history[-2]:
            RuntimeError('Second turn stitching: FAIL Setback')
        if _is_repro(transmission_history, NREPRO) and transmission_history[-1] == bpm_readings.shape[1]:
            LOGGER.debug('Second turn stitching: Success')
            return SC
    raise RuntimeError('Second turn stitching: FAIL Reached maxsteps')


def balance(SC, response_matrix, reference=None, cm_ords=None, bpm_ords=None, eps=1e-4, maxsteps=10, **pinv_params):
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
        response_matrix: Trajectory-response matrix.
        reference: (None) target orbit in the format `[x_1 ... x_n y_1 ...y_n]`, where
                   [x_i,y_i]` is the target position at the i-th BPM.
        cm_ords: List of CM ordinates to be used for correction (SC.ORD.CM)
        bpm_ords: List of BPM ordinates at which the reading should be evaluated (SC.ORD.BPM)
        maxsteps: break, if this number of correction steps have been performed (default = 100)
        eps: break, if the coefficient of variation of the RMS BPM reading is below this value

    Returns:
        SimulatedCommissioning class instance with corrected `SC.RING`

    """
    LOGGER.debug('Balancing two turns: Start')
    if SC.INJ.nTurns != 2:
        raise ValueError("Balancing works only with two turns.")
    bpm_ords, cm_ords, reference = _check_ords(SC, response_matrix, reference, bpm_ords, cm_ords)
    bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, bpm_ords=bpm_ords)  # Inject...
    if transmission_history[-1] < bpm_readings.shape[1]:
        raise ValueError("Balancing works only with full 2 turn transmission.")

    # Prepare reference
    reference = reference.reshape(2, len(bpm_readings[0]))
    reference[:, len(bpm_ords):] = 0
    reference = reference.reshape(response_matrix.shape[0])
    Mplus = sc_tools.SCgetPinv(response_matrix, **pinv_params)

    # Main loop
    for steps in range(maxsteps):
        # Set BPM readings
        delta_b = [bpm_readings[0][len(bpm_ords):] - bpm_readings[0][:len(bpm_ords)],
                   bpm_readings[1][len(bpm_ords):] - bpm_readings[1][:len(bpm_ords)]]
        measurement = np.concatenate((bpm_readings[:, :len(bpm_ords)], delta_b), axis=1).ravel()
 
        # Correction step
        dphi = np.dot(Mplus, (measurement - reference))
        SC.set_cm_setpoints(cm_ords[0], -dphi[:len(cm_ords[0])], skewness=False, method=SETTING_ADD)
        SC.set_cm_setpoints(cm_ords[1], -dphi[len(cm_ords[0]):], skewness=True, method=SETTING_ADD)
        bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(
            SC, bpm_ords=bpm_ords, ind_history=transmission_history, orb_history=rms_orbit_history)

        # Check stopping criteria
        if transmission_history[-1] < bpm_readings.shape[1]:
            raise RuntimeError('Balancing two turns: FAIL (lost transmission)')
        if _is_stable_or_converged(NREPRO, eps, rms_orbit_history):
            LOGGER.debug(f'Balancing two turns: Success (converged after {steps} steps)')
            return SC

    raise RuntimeError('Balancing two turns: FAIL (maxsteps reached, unstable)')


def correct(SC, response_matrix, reference=None, cm_ords=None, bpm_ords=None, eps=1e-4, target=0, maxsteps=30, scaleDisp=0, **pinv_params):
    """
    Iterative orbit/trajectory correction

    Iteratively applies orbit corrections using the pseudoinverse of the
    trajectory `response_matrix`, until a break-condition specified by one
    of 'eps', 'target' or 'maxsteps' is met.
    The dispersion can be included, thus the rf frequency as a correction
    parameter. If the dispersion is to be included, `response_matrix` has to have the size
    `(2 * len(SC.ORD.BPM)) x (len(SC.ORD.HCM) + len(SC.ORD.VCM) + 1)`, otherwise the size
    `(2 * len(SC.ORD.BPM)) x (len(SC.ORD.HCM) + len(SC.ORD.VCM))`, or correspondingly if the CM
    and/or BPM ordinates for the correction is explicitly given (see options below). `SC.RING` is
    assumed to be a lattice with transmission through all considered turns.
    Raises RuntimeError if transmission is lost.

    Args:
        SC: SimulatedCommissioning class instance.
        response_matrix: Trajectory/orbit-response matrix.
        reference: (None) target orbit in the format `[x_1 ... x_n y_1 ...y_n]`, where
                   [x_i,y_i]` is the target position at the i-th BPM.
        cm_ords: List of CM ordinates to be used for correction (SC.ORD.CM)
        bpm_ords: List of BPM ordinates at which the reading should be evaluated (SC.ORD.BPM)
        maxsteps: break, if this number of correction steps have been performed (default = 100)
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
            eta = SCgetModelDispersion(SC, SC.ORD.BPM, SC.ORD.RF,
                                       trackMode='ORB', Z0=np.zeros(6),
                                       nTurns=1, rfStep=1E3,
                                       useIdealRing=True)
            SC = correct(SC, np.column_stack((MCO, 1E8 * eta)), alpha=10, target=0, maxsteps=50, scaleDisp=1E8)

    """
    LOGGER.debug('Orbit/trajectory correction: Start')
    bpm_ords, cm_ords, reference = _check_ords(SC, response_matrix[:, :-1] if scaleDisp else response_matrix,
                                               reference, bpm_ords, cm_ords)
    bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, bpm_ords=bpm_ords)  # Inject ...
    Mplus = sc_tools.SCgetPinv(response_matrix, **pinv_params)

    # Main loop
    for steps in range(maxsteps):
        # Set BPM readings
        measurement = bpm_readings[:, :].reshape(reference.shape)

        # Correction step
        dphi = np.dot(Mplus, (measurement - reference))
        if scaleDisp != 0:   # TODO this is weight
            SC.set_cavity_setpoints(SC.ORD.RF, -scaleDisp * dphi[-1], "Frequency", method=SETTING_ADD)
            dphi = dphi[:-1]  # TODO the last setpoint is cavity frequency
        SC.set_cm_setpoints(cm_ords[0], -dphi[:len(cm_ords[0])], skewness=False, method=SETTING_ADD)
        SC.set_cm_setpoints(cm_ords[1], -dphi[len(cm_ords[0]):], skewness=True, method=SETTING_ADD)
        bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(
            SC, bpm_ords=bpm_ords, ind_history=transmission_history, orb_history=rms_orbit_history)  # Inject ...

        # Check stopping criteria
        if np.any(np.isnan(bpm_readings[0, :])):
            raise RuntimeError('Orbit/trajectory correction: FAIL (lost transmission)')
        if max(rms_orbit_history[-1]) < target and _is_stable_or_converged(min(NREPRO, maxsteps), eps, rms_orbit_history):
            LOGGER.debug(f"Orbit/trajectory correction: Success (target reached after {steps:d} steps)")
            return SC
        if _is_stable_or_converged(NREPRO, eps, rms_orbit_history):
            LOGGER.debug(f"Orbit/trajectory correction: Success (converged after {steps:d} steps)")
            return SC
    if _is_stable_or_converged(min(NREPRO, maxsteps), eps, rms_orbit_history) or maxsteps == 1:
        LOGGER.debug("Orbit/trajectory correction: Success (maxsteps reached)")
        return SC
    raise RuntimeError("Orbit/trajectory correction: FAIL (maxsteps reached, unstable)")


def _check_ords(SC, response_matrix, reference, bpm_ords, cm_ords):
    if cm_ords is None:
        cm_ords = SC.ORD.CM.copy()
    if bpm_ords is None:
        bpm_ords = SC.ORD.BPM.copy()
    if response_matrix.shape[0] != 2 * len(bpm_ords) * SC.INJ.nTurns:
        raise ValueError("Response matrix shape does not match the number of BPMs.")
    if response_matrix.shape[1] != len(cm_ords[0]) + len(cm_ords[1]):
        raise ValueError("Response matrix shape does not match the number of CMs.")
    if reference is None:
        reference = np.zeros((response_matrix.shape[0], 1))
    elif reference.shape[0] != response_matrix.shape[0]:
        raise ValueError("Reference shape does not match the shape of the response matrix.")
    return bpm_ords, cm_ords, reference


def _bpm_reading_and_logging(SC, bpm_ords, ind_history=None, orb_history=None):
    bpm_readings = bpm_reading(SC, bpm_ords=bpm_ords)[0]
    bpms_reached = ~np.isnan(bpm_readings[0])
    if ind_history is None or orb_history is None:
        return bpm_readings, [np.sum(bpms_reached)], [np.sqrt(np.mean(np.square(bpm_readings[:, bpms_reached]), axis=1))]
    ind_history.append(np.sum(bpms_reached))
    orb_history.append(np.sqrt(np.mean(np.square(bpm_readings[:, bpms_reached]), axis=1)))
    return bpm_readings, ind_history, orb_history  # assumes no bad BPMs


def _get_last_cm(lastBPMidx, n, bpm_ords, cm_ords):  # Refactored
    # Returns last CM ords and index of last CMs
    if lastBPMidx >= len(bpm_ords):  # If there is no beam loss in the first turn
        return cm_ords[-n:], range(len(cm_ords) - n, len(cm_ords))  # ... just return the last n CMs
    if len(cm_ords[np.where(cm_ords <= bpm_ords[lastBPMidx])]) == 0:
        raise RuntimeError('No CM upstream of BPMs')
    return cm_ords[np.where(cm_ords <= bpm_ords[lastBPMidx])[0][-n:]], np.where(cm_ords <= bpm_ords[lastBPMidx])[0][-n:]


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


def _wiggling(SC, bpm_ords, cm_ords, transmission_limit, nums_correctors=range(1, 9)):
    LOGGER.debug('Wiggling')
    dpts = _golden_donut_diffs(WIGGLE_ANGLE_RANGE[0], WIGGLE_ANGLE_RANGE[1], WIGGLE_ANGLE_STEPS)
    bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(SC, bpm_ords=bpm_ords, ind_history=None, orb_history=None)  # Inject...

    for nWiggleCM in nums_correctors:
        LOGGER.debug(f'Number of magnets used for wiggling: {nWiggleCM}. \n')
        tmpCMordsH = _get_last_cm(transmission_history[-1] - 1, nWiggleCM, bpm_ords, cm_ords[0])[0]  # Last CMs in horz
        tmpCMordsV = _get_last_cm(transmission_history[-1] - 1, nWiggleCM, bpm_ords, cm_ords[1])[0]  # Last CMs in vert

        for i in range(dpts.shape[1]):
            SC.set_cm_setpoints(tmpCMordsH, dpts[0, i], skewness=False, method=SETTING_ADD)
            SC.set_cm_setpoints(tmpCMordsV, dpts[1, i], skewness=True, method=SETTING_ADD)
            bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(
                SC, bpm_ords=bpm_ords, ind_history=transmission_history, orb_history=rms_orbit_history)
            if transmission_history[-1] >= transmission_limit:
                for _ in range(2):
                    bpm_readings, transmission_history, rms_orbit_history = _bpm_reading_and_logging(
                        SC, bpm_ords=bpm_ords, ind_history=transmission_history, orb_history=rms_orbit_history)
                if _is_repro(transmission_history, NREPRO):
                    LOGGER.debug('...completed')
                    return SC  # Continue with feedback

    if not transmission_history[-1] >= transmission_limit:
        raise RuntimeError('Wiggling failed')
    return SC
