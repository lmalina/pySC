import at
import numpy as np
import pytest
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.utils.sc_tools import SCgetOrds
from pySC.utils import logging_tools
from pySC.correction import loco
from pySC.lattice_properties.response_model import SCgetModelRM, SCgetModelDispersion

LOGGER = logging_tools.get_logger(__name__)


def test_loco_hmba(at_ring):
    np.random.seed(12345678)
    sc = SimulatedCommissioning(at_ring)
    sc.register_bpms(SCgetOrds(sc.RING, 'BPM'), Roll=0.0, CalError=1E-2 * np.ones(2), NoiseCO=np.array([1e-7, 1E-7]))
    sc.register_magnets(SCgetOrds(sc.RING, 'QF|QD'), CalErrorB=np.array([0, 1E-2]))  # relative
    sc.register_magnets(SCgetOrds(sc.RING, 'CXY'), CalErrorA=np.array([1E-4, 0]), CalErrorB=np.array([1E-4, 0]))
    sc.register_magnets(SCgetOrds(sc.RING, 'BEND'))
    sc.register_magnets(SCgetOrds(sc.RING, 'SF|SD'))  # [1/m]
    sc.register_cavities(SCgetOrds(sc.RING, 'RFC'))
    sc.apply_errors()
    cor_ords = SCgetOrds(sc.RING, 'CXY')

    used_correctors1 = loco.select_equally_spaced_elements(cor_ords, 10)
    used_correctors2 = loco.select_equally_spaced_elements(cor_ords, 10)
    used_cor_ords = [used_correctors1, used_correctors2]
    used_bpms_ords = loco.select_equally_spaced_elements(sc.ORD.BPM, len(sc.ORD.BPM))
    cav_ords = SCgetOrds(sc.RING, 'RFC')
    quads_ords = [SCgetOrds(sc.RING, 'QF'), SCgetOrds(sc.RING, 'QD')]

    CMstep = np.array([1.e-4])  # correctors change [rad]
    dk = 1.e-4  # quads change
    RFstep = 1e3

    _, _, twiss = at.get_optics(sc.IDEALRING, sc.ORD.BPM)
    orbit_response_matrix_model = SCgetModelRM(sc, used_bpms_ords, used_cor_ords, trackMode='ORB', useIdealRing=True, dkick=CMstep)
    model_dispersion = SCgetModelDispersion(sc, used_bpms_ords, cav_ords, trackMode='ORB', Z0=np.zeros(6), nTurns=1,
                                            rfStep=RFstep, useIdealRing=True)
    Jn = loco.calculate_jacobian(sc, orbit_response_matrix_model, CMstep, used_cor_ords, used_bpms_ords, np.concatenate(quads_ords), dk,
                            trackMode='ORB', useIdealRing=False, skewness=False, order=1, method='add',
                            includeDispersion=False, rf_step=RFstep, cav_ords=cav_ords)
    weights = np.eye(len(used_bpms_ords) * 2)
    n_singular_values = 20

    _, _, twiss_err = at.get_optics(sc.RING, sc.ORD.BPM)
    bx_rms_err, by_rms_err = loco.model_beta_beat(sc.RING, twiss, sc.ORD.BPM, plot=False)
    info_tab = 14 * " "
    LOGGER.info("RMS Beta-beating before LOCO:\n"
                f"{info_tab}{bx_rms_err * 100:04.2f}% horizontal\n{info_tab}{by_rms_err * 100:04.2f}% vertical  ")
    n_iter = 3

    for x in range(n_iter):  # optics correction using QF and QD
        LOGGER.info(f'LOCO iteration {x}')
        orbit_response_matrix_measured = loco.measure_closed_orbit_response_matrix(sc, used_bpms_ords, used_cor_ords, CMstep)
        n_quads, n_corrs, n_bpms = len(np.concatenate(quads_ords)), len(np.concatenate(used_cor_ords)), len(used_bpms_ords) * 2
        bx_rms_err, by_rms_err = loco.model_beta_beat(sc.RING, twiss, sc.ORD.BPM, plot=False)
        total_length = n_bpms + n_corrs + n_quads
        lengths = [n_quads, n_corrs, n_bpms]
        including_fit_parameters = ['quads', 'cor', 'bpm']
        initial_guess = np.zeros(total_length)

        initial_guess[:lengths[0]] = 1e-6
        initial_guess[lengths[0]:lengths[0] + lengths[1]] = 1e-6
        initial_guess[lengths[0] + lengths[1]:] = 1e-6

        # method lm (least squares)
        fit_parameters = loco.loco_correction_lm(initial_guess, orbit_response_matrix_model,
                                                 orbit_response_matrix_measured, Jn, lengths,
                                                 including_fit_parameters, bounds=(-0.03, 0.03), weights=weights, verbose=2)

        # method ng
        # fit_parameters = loco.loco_correction_ng(initial_guess, orbit_response_matrix_model,
        #                                          orbit_response_matrix_measured, Jn, lengths,
        #                                          including_fit_parameters, n_singular_values, weights=weights)

        dg = fit_parameters[:lengths[0]] if len(fit_parameters) > n_quads else fit_parameters
        sc = loco.set_correction(sc, dg, np.concatenate(quads_ords))
        bx_rms_cor, by_rms_cor = loco.model_beta_beat(sc.RING, twiss, sc.ORD.BPM, plot=False)
        LOGGER.info(f"RMS Beta-beating after {x + 1} LOCO iterations:\n"
                    f"{info_tab}{bx_rms_cor * 100:04.2f}% horizontal\n{info_tab}{by_rms_cor * 100:04.2f}% vertical  ")
        LOGGER.info(f"Correction reduction: \n"
                    f"    beta_x: {(1 - bx_rms_cor / bx_rms_err) * 100:.2f}%\n"
                    f"    beta_y: {(1 - by_rms_cor / by_rms_err) * 100:.2f}%\n")
    assert bx_rms_cor < 0.002
    assert by_rms_cor < 0.002


@pytest.fixture
def at_ring():
    ring = at.load_mat('inputs/hmba.mat')
    bpm_indexes = at.get_refpts(ring, at.elements.Monitor)
    for bpm_index in reversed(bpm_indexes):
        corrector = at.elements.Corrector(f'CXY{bpm_index}', length=0, kick_angle=[0, 0], PolynomA=[0, 0],
                                          PolynomB=[0, 0])
        ring.insert(bpm_index + 1, corrector)
    ring.enable_6d()
    at.set_cavity_phase(ring)
    at.set_rf_frequency(ring)

    ring.tapering(niter=3, quadrupole=True, sextupole=True)
    ring = at.Lattice(ring)
    return ring
