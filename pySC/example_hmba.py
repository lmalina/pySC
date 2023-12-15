import at
import numpy as np
from at import Lattice

from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.utils.sc_tools import SCgetOrds
from pySC.utils import logging_tools
from pySC.correction.loco_modules import select_equally_spaced_elements, generating_jacobian, measure_closed_orbit_response_matrix, model_beta_beat, \
    loco_correction, set_correction, objective, get_inverse
from pySC.lattice_properties.response_model import SCgetModelRM, SCgetModelDispersion

LOGGER = logging_tools.get_logger(__name__)


def create_at_lattice() -> Lattice:
    new_ring = at.load_mat('hmba.mat')
    bpm_indexes = at.get_refpts(new_ring, at.elements.Monitor)
    for i in reversed(bpm_indexes):
        Cor = at.elements.Corrector('CXY' + str(i), length=0, kick_angle=[0, 0], PolynomA=[0, 0], PolynomB=[0, 0])
        new_ring.insert(i + 1, Cor)
    new_ring.enable_6d()
    at.set_cavity_phase(new_ring)
    at.set_rf_frequency(new_ring)
    new_ring.tapering(niter=3, quadrupole=True, sextupole=True)

    return new_ring


if __name__ == "__main__":
    ring = at.Lattice(create_at_lattice())
    LOGGER.info(f"{len(ring)=}")
    SC = SimulatedCommissioning(ring)
    SC.register_bpms(SCgetOrds(SC.RING, 'BPM'), Roll=0.0, CalError=1E-2 * np.ones(2), NoiseCO=np.array([1e-6, 1E-6]))
    SC.register_magnets(SCgetOrds(SC.RING, 'QF|QD'),  CalErrorB=np.array([0, 1E-2]))  # relative
    SC.register_magnets(SCgetOrds(SC.RING, 'CXY'), CalErrorA=np.array([1E-2, 0]), CalErrorB=np.array([1E-2, 0]))
    SC.register_magnets(SCgetOrds(SC.RING, 'BEND'))
    SC.register_magnets(SCgetOrds(SC.RING, 'SF|SD'))  # [1/m]
    SC.register_cavities(SCgetOrds(SC.RING, 'RFC'))
    SC.apply_errors()

    CorOrds = SCgetOrds(SC.RING, 'CXY')

    used_correctors1 = select_equally_spaced_elements(CorOrds, 10)
    used_correctors2 = select_equally_spaced_elements(CorOrds, 10)
    CorOrds = [used_correctors1, used_correctors2]
    used_bpm1 = select_equally_spaced_elements(SC.ORD.BPM, 10)
    used_bpm2 = select_equally_spaced_elements(SC.ORD.BPM, 10)
    used_bpms = [used_bpm1, used_bpm2]

    CAVords = SCgetOrds(SC.RING, 'RFC')
    quadsOrds = [SCgetOrds(SC.RING, 'QF'), SCgetOrds(SC.RING, 'QD')]

    CMstep = np.array([1.e-4])  # correctors change [rad]
    dk = 1.e-4  # quads change
    RFstep = 1e3

    _, _, twiss = at.get_optics(SC.IDEALRING, SC.ORD.BPM)
    orbit_response_matrix_model = SCgetModelRM(SC, SC.ORD.BPM, CorOrds, trackMode='ORB', useIdealRing=True, dkick=CMstep)
    ModelDispersion = SCgetModelDispersion(SC, SC.ORD.BPM, CAVords, trackMode='ORB', Z0=np.zeros(6), nTurns=1,
                                           rfStep=RFstep, useIdealRing=True)
    Jn = generating_jacobian(SC, orbit_response_matrix_model, CMstep, CorOrds, SC.ORD.BPM, np.concatenate(quadsOrds), dk,
                             trackMode='ORB', useIdealRing=False, skewness=False, order=1, method='add',
                             includeDispersion=False, rf_step=RFstep, cav_ords=CAVords)
    Jn = np.transpose(Jn, (0, 2, 1))
    sCut = 16
    W = 1
    from pySC.core.beam import bpm_reading
    n_samples = 3
    a = np.empty((n_samples, 2, len(SC.ORD.BPM)))
    for i in range(n_samples):
        a[i] = bpm_reading(SC)[0]
    errors = np.std(a, axis=0)

    Jt = get_inverse(Jn, sCut, W)
    _, _, twiss_err = at.get_optics(SC.RING, SC.ORD.BPM)
    orbit_response_matrix_measured = measure_closed_orbit_response_matrix(SC, SC.ORD.BPM, CorOrds, CMstep)
    numberOfIteration = 1

    for x in range(numberOfIteration):  # optics correction using QF and QD
        LOGGER.info(f'LOCO iteration {x}')

        C_measure = measure_closed_orbit_response_matrix(SC, SC.ORD.BPM, CorOrds, CMstep)
        bx_rms_err, by_rms_err = model_beta_beat(SC.RING, twiss, SC.ORD.BPM, makeplot=False)
        quads = len(np.concatenate(quadsOrds))
        cor = len(np.concatenate(CorOrds))
        bpm = len(SC.ORD.BPM) * 2

        total_length = bpm + cor + quads
        lengths = [quads, cor, bpm]
        including_fit_parameters = ['quads', 'cor', 'bpm']
        initial_guess = np.zeros(total_length)
        initial_guess[:lengths[0]] = 1e-6
        initial_guess[lengths[0]:lengths[0] + lengths[1]] = 1e-6
        initial_guess[lengths[0] + lengths[1]:] = 1e-6

        fit_parameters = loco_correction(
            lambda delta_params: objective(delta_params, np.transpose(orbit_response_matrix_model),
                                           np.transpose(orbit_response_matrix_measured), Jn, lengths,
                                           including_fit_parameters, W), initial_guess,
            np.transpose(orbit_response_matrix_model), np.transpose(orbit_response_matrix_measured), Jn, Jt, lengths,
            including_fit_parameters
            , verbose=2, max_iterations=100, eps=1e-6, method='lm', W=W)

        dg = fit_parameters[:lengths[0]]
        dx = fit_parameters[lengths[0]:lengths[0] + lengths[1]]
        dy = fit_parameters[lengths[0] + lengths[1]:]
        LOGGER.info('SVD')
        SC = set_correction(SC, dg, np.concatenate(quadsOrds))
        _, _, twiss_corr = at.get_optics(SC.RING, SC.ORD.BPM)
        bx_rms_cor, by_rms_cor = model_beta_beat(SC.RING, twiss, SC.ORD.BPM, makeplot=True)
        LOGGER.info(
            "Before LOCO correction:\n"
            f"RMS horizontal beta beating: {bx_rms_err * 100:.2f}%   RMS vertical beta beating: {by_rms_err * 100:.2f}%\n"
    
            f"After LOCO corrections\n"
            f"RMS horizontal beta beating: {bx_rms_cor * 100:.2f}%   RMS vertical beta beating: {by_rms_cor * 100:.2f}%\n"
            f"beta_x correction reduction: {(1 - bx_rms_cor / bx_rms_err) * 100:.2f}%\n"
            f"beta_y correction reduction: {(1 - by_rms_cor / by_rms_err) * 100:.2f}%\n "
        )
