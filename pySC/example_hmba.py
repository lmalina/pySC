#!/usr/bin/env python
# coding: utf-8

# In[2]:


import at
import numpy as np
from at import Lattice

from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.utils.sc_tools import SCgetOrds
from pySC.utils import logging_tools
from pySC.correction import loco
from pySC.lattice_properties.response_model import SCgetModelRM, SCgetModelDispersion
# from pySC.core.beam import bpm_reading

LOGGER = logging_tools.get_logger(__name__)


def create_at_lattice() -> Lattice:
    new_ring = at.load_mat('hmba.mat')
    bpm_indexes = at.get_refpts(new_ring, at.elements.Monitor)
    for bpm_index in reversed(bpm_indexes):
        corrector = at.elements.Corrector(f'CXY{bpm_index}', length=0, kick_angle=[0, 0], PolynomA=[0, 0], PolynomB=[0, 0])
        new_ring.insert(bpm_index + 1, corrector)
    new_ring.enable_6d()
    at.set_cavity_phase(new_ring)
    at.set_rf_frequency(new_ring)
    new_ring.tapering(niter=3, quadrupole=True, sextupole=True)

    return new_ring


def create_sc_class():
    ring = at.Lattice(create_at_lattice())
    LOGGER.info(f"{len(ring)=}")
    SC = SimulatedCommissioning(ring)
    SC.register_bpms(SCgetOrds(SC.RING, 'BPM'), Roll=0.0, CalError=1E-2 * np.ones(2), NoiseCO=np.array([1e-60, 1E-60]))
    SC.register_magnets(SCgetOrds(SC.RING, 'QF|QD'), CalErrorB=np.array([0, 1E-2]))  # relative
    SC.register_magnets(SCgetOrds(SC.RING, 'CXY'), CalErrorA=np.array([1E-4, 0]), CalErrorB=np.array([1E-4, 0]))
    SC.register_magnets(SCgetOrds(SC.RING, 'BEND'))
    SC.register_magnets(SCgetOrds(SC.RING, 'SF|SD'))  # [1/m]
    SC.register_cavities(SCgetOrds(SC.RING, 'RFC'))
    SC.apply_errors()
    return SC


if __name__ == "__main__":
    SC = create_sc_class()
    cor_ords = SCgetOrds(SC.RING, 'CXY')

    used_correctors1 = loco.select_equally_spaced_elements(cor_ords, 10)
    used_correctors2 = loco.select_equally_spaced_elements(cor_ords, 10)
    used_cor_ords = [used_correctors1, used_correctors2]
    used_bpms_ords = loco.select_equally_spaced_elements(SC.ORD.BPM, len(SC.ORD.BPM))
    cav_ords = SCgetOrds(SC.RING, 'RFC')
    quads_ords = [SCgetOrds(SC.RING, 'QF'), SCgetOrds(SC.RING, 'QD')]

    CMstep = np.array([1.e-4])  # correctors change [rad]
    dk = 1.e-4  # quads change
    RFstep = 1e3

    _, _, twiss = at.get_optics(SC.IDEALRING, SC.ORD.BPM)
    orbit_response_matrix_model = SCgetModelRM(SC, used_bpms_ords, used_cor_ords, trackMode='ORB', useIdealRing=True, dkick=CMstep)
    model_dispersion = SCgetModelDispersion(SC, used_bpms_ords, cav_ords, trackMode='ORB', Z0=np.zeros(6), nTurns=1,
                                            rfStep=RFstep, useIdealRing=True)
    Jn = loco.calculate_jacobian(SC, orbit_response_matrix_model, CMstep, used_cor_ords, used_bpms_ords, cor_ords, np.concatenate(quads_ords), dk,
                            trackMode='ORB', useIdealRing=False, skewness=False, order=1, method='add',
                            includeDispersion=False, rf_step=RFstep, cav_ords=cav_ords)
    Jn = np.transpose(Jn, (0, 2, 1))
    #weights = 1
    weights = np.eye(len(used_bpms_ords) * 2)
    tmp = np.sum(Jn, axis=1)
    A = tmp @ weights @ tmp.T
    u, s, v = np.linalg.svd(A, full_matrices=True)
    import matplotlib.pyplot as plt

    plt.plot(np.log(s), 'd--')
    plt.title('singular value')
    plt.xlabel('singular values')
    plt.ylabel('$\log(\sigma_i)$')
    plt.show()

    n_singular_values = 20

    #Jt = loco.get_inverse(Jn, n_singular_values, weights)

    _, _, twiss_err = at.get_optics(SC.RING, SC.ORD.BPM)
    bx_rms_err, by_rms_err = loco.model_beta_beat(SC.RING, twiss, SC.ORD.BPM, plot=False)
    info_tab = 14 * " "
    LOGGER.info("RMS Beta-beating before LOCO:\n"
                f"{info_tab}{bx_rms_err * 100:04.2f}% horizontal\n{info_tab}{by_rms_err * 100:04.2f}% vertical  ")
    n_iter = 5

    for x in range(n_iter):  # optics correction using QF and QD
        LOGGER.info(f'LOCO iteration {x}')
        orbit_response_matrix_measured = loco.measure_closed_orbit_response_matrix(SC, used_bpms_ords, used_cor_ords, CMstep)
        n_quads, n_corrs, n_bpms = len(np.concatenate(quads_ords)), len(np.concatenate(used_cor_ords)), len(used_bpms_ords) * 2
        bx_rms_err, by_rms_err = loco.model_beta_beat(SC.RING, twiss, SC.ORD.BPM, plot=False)
        total_length = n_bpms + n_corrs + n_quads
        lengths = [n_quads, n_corrs, n_bpms]
        including_fit_parameters = ['quads', 'cor', 'bpm']
        initial_guess = np.zeros(total_length)

        initial_guess[:lengths[0]] = 1e-6
        initial_guess[lengths[0]:lengths[0] + lengths[1]] = 1e-6
        initial_guess[lengths[0] + lengths[1]:] = 1e-6

        # method lm (least squares)
        #fit_parameters = loco.loco_correction_lm(initial_guess, np.transpose(orbit_response_matrix_model),
        #                                         np.transpose(orbit_response_matrix_measured), Jn, lengths,
        #                                         including_fit_parameters, bounds=(-0.03, 0.03), weights=weights, verbose=2)

        # method ng
        fit_parameters = loco.loco_correction_ng(initial_guess, np.transpose(orbit_response_matrix_model),
                                                  np.transpose(orbit_response_matrix_measured), Jn, lengths,
                                                  including_fit_parameters, n_singular_values, weights=weights)

        dg = fit_parameters[:lengths[0]] if len(fit_parameters) > n_quads else fit_parameters
        SC = loco.set_correction(SC, dg, np.concatenate(quads_ords))
        bx_rms_cor, by_rms_cor = loco.model_beta_beat(SC.RING, twiss, SC.ORD.BPM, plot=False)
        LOGGER.info(f"RMS Beta-beating after {x + 1} LOCO iterations:\n"
                    f"{info_tab}{bx_rms_cor * 100:04.2f}% horizontal\n{info_tab}{by_rms_cor * 100:04.2f}% vertical  ")
        LOGGER.info(f"Correction reduction: \n"
                    f"    beta_x: {(1 - bx_rms_cor / bx_rms_err) * 100:.2f}%\n"
                    f"    beta_y: {(1 - by_rms_cor / by_rms_err) * 100:.2f}%\n")


# In[ ]:




