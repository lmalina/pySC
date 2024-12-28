import numpy as np
import at
from pySC.core.beam import _real_bpm_reading
from pySC.lattice_properties.response_model import SCgetModelDispersion
from pySC.core.constants import NUM_TO_AB
from pySC.utils import sc_tools
import copy

LOGGER = logging_tools.get_logger(__name__)

def phase_advance_correction(ring, bpm_indices, elements_indices, dkick, cut, Px=None, Py=None, Etax=None):
    """
    Perform phase advance and horizontal dispersion correction on the given ring.
    
    Parameters:
    dkick: Change in quadrupole strength for response matrix calculation.
    cut: number of kept singular values
    Px, Py, Etax:  response matrices for horizontal and vertical phase advances and dispersion. 
                   If not provided, they will be calculated.
    
    Returns:
    corrected ring
    """
    # Initial Twiss parameters 
    _, _, twiss_err0 = at.get_optics(ring, bpm_indices)
    elemdata0, beamdata, elemdata = at.get_optics(ring, bpm_indices)
    mux0 = elemdata.mu[:, 0] / (2 * np.pi)
    muy0 = elemdata.mu[:, 1] / (2 * np.pi)
    Eta_x0 = elemdata.dispersion[:, 0]

    # Calculate Response Matrix if not provided
    if Px is None or Py is None or Etax is None:
        Px, Py, Etax = calculate_rm(dkick, ring, elements_indices, bpm_indices, mux0, muy0, Eta_x0)
    
    response_matrix = np.hstack((Px, Py, Etax))
    
    elemdata0, beamdata, elemdata = at.get_optics(ring, bpm_indices)
    mux = elemdata.mu[:, 0] / (2 * np.pi)
    muy = elemdata.mu[:, 1] / (2 * np.pi)
    Eta_xx = elemdata.dispersion[:, 0]
    
    measurement = np.concatenate((mux - mux0, muy - muy0, Eta_xx - Eta_x0), axis=0)
    
    s = np.linalg.svd(response_matrix.T, compute_uv=False)
    system_solution = np.linalg.pinv(response_matrix.T, rcond=s[cut - 1] / s[0]) @ -measurement
        ring = apply_correction(ring, system_solution, elements_indices)
    
    return ring


def calculate_rm(dkick, ring, elements_indices, bpm_indices, mux0, muy0, Eta_x0):
    """    
    Returns:
    Px, Py, Etax: Response matrices for horizontal and vertical phase advances and dispersion.
    """
    px =[]
    py =[] 
    etax = []

    for index in elements_indices:
        original_setting = ring[index].PolynomB[1]
        
        ring[index].PolynomB[1] += dkick
        _, _, elemdata = at.get_optics(ring, bpm_indices)
        
        mux = elemdata.mu[:, 0] / (2 * np.pi)
        muy = elemdata.mu[:, 1] / (2 * np.pi)
        Eta_x = elemdata.dispersion[:, 0]
        
        px.append((mux - mux0) / dkick)
        py.append((muy - muy0) / dkick)
        etax.append((Eta_x - Eta_x0) / dkick)
        
        ring[index].PolynomB[1] = original_setting

    Px = np.squeeze(np.array(px))
    Py = np.squeeze(np.array(py))
    Etax = np.squeeze(np.array(etax))
    
    return Px, Py, Etax


def apply_correction(ring, corrections, elements_indices):
    for i, index in enumerate(elements_indices):
        ring[index].PolynomB[1] += corrections[i]
    return ring

def SCgetModelPhaseAdvanceRM(SC, BPMords, ELEMords, dkick=1e-5, skewness=False, order=1, useIdealRing=True):
    """
    (to be rewritten)
    Returns:
    Px, Py, Etax: Response matrices for horizontal and vertical phase advances and dispersion.
    """
    LOGGER.info('Calculating model phase advance response matrix')
    ring = SC.IDEALRING.deepcopy() if useIdealRing else SCgetModelRING(SC)

    nBPM = len(BPMords)
    nELEM = len(ELEMords)
    RM = np.full((3 * nBPM, nELEM), np.nan) # (Px+Py+Etax, Elem)

    _, _, elemdata0 = at.get_optics(ring, BPMords)
    mux0 = elemdata0.mu[:, 0] / (2 * np.pi)
    muy0 = elemdata0.mu[:, 1] / (2 * np.pi)
    Eta_x0 = elemdata0.dispersion[:, 0]
    Ta = np.hstack((mux0, muy0, Eta_x0))

    for i, ELEMord in enumerate(ELEMords):

        PolynomNominal = getattr(ring[ELEMord], f"Polynom{NUM_TO_AB[int(skewness)]}")
        changed_polynom = copy.deepcopy(PolynomNominal[:])
        changed_polynom[order] += dkick
        setattr(ring[ELEMord], f"Polynom{NUM_TO_AB[int(skewness)]}", changed_polynom[:])

        _, _, elemdata = at.get_optics(ring, BPMords)
        mux = elemdata.mu[:, 0] / (2 * np.pi)
        muy = elemdata.mu[:, 1] / (2 * np.pi)
        Eta_x = elemdata.dispersion[:, 0]
        TdB = np.hstack((mux, muy, Eta_x))

        setattr(ring[ELEMord], f"Polynom{NUM_TO_AB[int(skewness)]}", PolynomNominal[:])
        dTdB = (TdB - Ta) / dkick
        RM[:, i] = dTdB

    return RM

def phase_advance_correction2(SC, BPMords, ELEMords, dkick=1e-5, nturns=64, skewness=False,
                              order=1, dipole_compensation=True, alpha=1e-3, RM=None):
    """
    (To be rewritten)
    Perform phase advance and horizontal dispersion correction on the given ring.

    Parameters:
    dkick: Change in quadrupole strength for response matrix calculation.
    cut: number of kept singular values
    Px, Py, Etax:  response matrices for horizontal and vertical phase advances and dispersion.
                   If not provided, they will be calculated.

    Returns:
    corrected ring
    """
    # Ideal Twiss parameters and tunes
    elemdata0, beamdata, elemdata = at.get_optics(SC.IDEALRING, BPMords)
    mux0 = elemdata.mu[:, 0] / (2 * np.pi)
    muy0 = elemdata.mu[:, 1] / (2 * np.pi)
    Eta_x0 = elemdata.dispersion[:, 0]
    Qx0 = beamdata.tune[0]
    Qy0 = beamdata.tune[1]

    # Calculate Response Matrix if not provided
    if RM is None:
        RM = SCgetModelPhaseAdvanceRM(SC, BPMords, ELEMords, dkick=dkick, skewness=skewness, order=order, useIdealRing=True)

    #track two particles, one with amplitude in x and one in y
    Z0=np.asfortranarray(np.vstack(([dkick, 0, 0, 0, 0, 0],
                                    [0, 0, dkick, 0, 0, 0])).T)
    Z=at.lattice_pass(SC.RING, Z0, nturns, refpts=SC.ORD.BPM)
    bpm_readings = Z[[0, 2], [0, 1], :, :]
    real_bpm_readings = _real_bpm_reading(SC, bpm_readings[:, np.newaxis, :, :])[0] # advanced indexing to match _real_bpm_reading input

    # get tunes and phases using NAFF algorithm
    tune_x, amp_x, phase_x = at.physics.harmonic_analysis.get_main_harmonic(real_bpm_readings[0], fmin=0.5, fmax=0.95)
    tune_y, amp_y, phase_y = at.physics.harmonic_analysis.get_main_harmonic(real_bpm_readings[1], fmin=0.05, fmax=0.5)
    mux = np.unwrap(phase_x) / (2 * np.pi)
    mux *= np.sign(mux) # sign correction needed to fit at.get_optics() phase convention
    muy = np.unwrap(phase_y) / (2 * np.pi)
    muy *= np.sign(muy) # sign correction needed to fit at.get_optics() phase convention

    # dispersion calculation
    # I advise to use useIdealRing=True here, otherwise the dispersion is quite noisy
    Eta_x = SCgetModelDispersion(SC, SC.ORD.BPM, CAVords=SC.ORD.RF, useIdealRing=True, rfStep=2e1)[:SC.ORD.BPM.size]
    Eta_x *= -SC.RING.get_rf_frequency() * SC.RING.disable_6d(copy=True).get_mcf() # need proper normalization
    # SCgetModelDispersion does not return the dispersion in [m]

    Qx = np.mean(tune_x, keepdims=True)
    Qy = np.mean(tune_y, keepdims=True)

    # measurement contains all the observables to minimize
    measurement = np.concatenate((mux - mux0, muy - muy0, Eta_x - Eta_x0, Qx - Qx0, Qy - Qy0), axis=0)
    # weights can be changed to emphasize some observables
    weights = np.hstack((np.ones(mux.size), np.ones(muy.size), np.ones(Eta_x.size), np.ones(1), np.ones(1)))

    inverse_RM = sc_tools.pinv(RM, alpha=alpha, plot=False)
    system_solution = -np.dot(inverse_RM, measurement*weights)

    return SC, system_solution
