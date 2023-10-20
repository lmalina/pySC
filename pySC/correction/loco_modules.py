import multiprocessing
import numpy as np
from at import *
import at.plot
from pylab import *
import multiprocessing
from at import Lattice
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.core.beam import bpm_reading
from pySC.lattice_properties.response_model import SCgetModelRM, SCgetModelDispersion
from pySC.core.lattice_setting import set_magnet_setpoints, set_cm_setpoints
from pySC.utils import logging_tools
from pySC.core.constants import *
import copy
from pySC.lattice_properties.response_measurement import *

LOGGER = logging_tools.get_logger(__name__)



def generatingQuadsResponseWorker(args):
    return generatingQuadsResponseParallel(*args)

def generatingJacobian(SC, C_model, dkick, used_cor_ind, bpm_indexes, quads_ind, dk, debug=True, trackMode='ORB', useIdealRing=True, skewness=False, order=1, method='add', includeDispersion=False,  rf_step=1E3, cav_ords=None):
    pool = multiprocessing.Pool()
    quad_args = [(quad_index, SC, C_model, dkick, used_cor_ind, bpm_indexes, dk, debug, trackMode, useIdealRing, skewness, order, method,includeDispersion, rf_step, cav_ords) for quad_index in quads_ind]
    results = pool.map(generatingQuadsResponseWorker, quad_args)
    pool.close()
    pool.join()
    #dC = zip(*results)
    return results
    #list(dC)

def generatingQuadsResponseParallel(quad_index, SC, C_model, correctrs_kick, used_cor_indexes, used_bpm_indexes, dk, debug, useIdealRing, trackMode, skewness, order, method,includeDispersion, rf_step, cav_ords):
    if debug:
        print('generating response to quad of index', quad_index)

    C0 = C_model

    if includeDispersion == True:
        dispersion_model = SCgetModelDispersion(SC, used_bpm_indexes, CAVords=cav_ords,
                                                trackMode='ORB', Z0=np.zeros(6), nTurns=1,
                                                rfStep=1E3, useIdealRing=True)

        C0 = np.hstack((C0, dispersion_model.reshape(-1, 1)))
    C = quadsSensitivityMatrices(SC, correctrs_kick, used_cor_indexes, used_bpm_indexes, quad_index, dk,trackMode, useIdealRing, skewness, order, method,includeDispersion, rf_step, cav_ords)
    dC = (C - C0) / dk

    return dC


def generatingQuadsResponse(SC,C_model, correctrs_kick,used_cor_indexes, used_bpm_indexes, quads_indexes, dk, debug, useIdealRing, trackMode, includeDispersion):
    dC = []

    C0 = C_model

    if includeDispersion == True:
        dispersion_model = SCgetModelDispersion(SC, used_bpm_indexes, CAVords=SCgetOrds(SC.RING, 'RFCav'),
                                                trackMode='ORB', Z0=np.zeros(6), nTurns=1,
                                                rfStep=1E3, useIdealRing=True)
        C0 = np.hstack((C0, dispersion_model.reshape(-1, 1)))

    for i in quads_indexes:
        for quad_index in i:
            if debug == True :
               print('generating response to quad of index', quad_index)
            C1 = quadsSensitivityMatrices(SC, correctrs_kick,used_cor_indexes, used_bpm_indexes, quad_index, dk, useIdealRing,trackMode, skewness, order, method,includeDispersion)
            dC.append((C1 - C0) / dk)
    return dC

def quadsSensitivityMatrices(SC, correctors_kick,used_cor_indexes, used_bpm_indexes, quad_index, dk, useIdealRing, trackMode, skewness, order, method, includeDispersion, rf_step, cav_ords):
    #SC.IDEALRING[quad_index].PolynomB[1] += dk
    SC = set_magnet_setpoints(SC, quad_index, dk, skewness, order, method)
    print('go to model orm')
    C_measured = SCgetModelRM(SC, used_bpm_indexes, used_cor_indexes, dkick = correctors_kick, useIdealRing = useIdealRing, trackMode = trackMode)

    #SC.IDEALRING[quad_index].PolynomB[1] -= dk

    qx = C_measured

    if includeDispersion == True:
        dispersion_model = SCgetModelDispersion(SC, used_bpm_indexes, CAVords=cav_ords , trackMode='ORB', Z0=np.zeros(6), nTurns=1,
                                                rfStep=rf_step, useIdealRing=True)

        qx =     np.hstack((qx, dispersion_model.reshape(-1, 1)))

    SC = set_magnet_setpoints(SC, quad_index, -dk, skewness, order, method)

    return qx

def SCgetMeasurRM(SC, BPMords, CMords, trackMode='ORB', dkick=1e-5):
    LOGGER.info('Calculating Measure response matrix')

    nTurns = 1
    nBPM = len(BPMords)
    nCM = len(CMords[0]) + len(CMords[1])
    RM = np.full((2 * nBPM * nTurns, nCM), np.nan)

    #print('ta', Ta)
    SC.INJ.trackMode = trackMode
    orbits = bpm_reading(SC, bpm_ords=BPMords)
    closed_orbitx0 = orbits[0][0, :]
    closed_orbity0 = orbits[0][1, :]
    cnt = 0
    for nDim in range(2):
        for CMord in CMords[nDim]:
            if SC.RING[CMord].PassMethod == 'CorrectorPass':
                KickNominal = SC.RING[CMord].KickAngle[nDim]
                SC.RING[CMord].KickAngle[nDim] = KickNominal + dkick
                SC.INJ.trackMode = trackMode
                orbits = bpm_reading(SC, bpm_ords=BPMords)
                closed_orbitx1 = orbits[0][0, :]
                closed_orbity1 = orbits[0][1, :]
                SC.RING[CMord].KickAngle[nDim] = KickNominal
            else:
                PolynomNominal = getattr(SC.RING[CMord], f"Polynom{NUM_TO_AB[nDim]}") # note my change ideal ring
                delta = dkick / SC.RING[CMord].Length
                changed_polynom = copy.deepcopy(PolynomNominal[:])

                changed_polynom[0] += (-1) ** (nDim+1) * delta
                setattr(SC.RING[CMord], f"Polynom{NUM_TO_AB[nDim]}", changed_polynom[:])

                SC.INJ.trackMode = trackMode
                orbits = bpm_reading(SC, bpm_ords=BPMords)
                closed_orbitx1 = orbits[0][0, :]
                closed_orbity1 = orbits[0][1, :]
                setattr(SC.RING[CMord], f"Polynom{NUM_TO_AB[nDim]}", PolynomNominal[:])

            orbitx = (closed_orbitx1 - closed_orbitx0) / dkick
            orbity = (closed_orbity1 - closed_orbity0) / dkick

            RM[:, cnt] = np.concatenate([np.transpose(orbitx), np.transpose(orbity)])

            cnt = cnt + 1
    return RM



def loco_correction(objective_function, initial_guess0, C_model, C_measure, J, Jt, lengths, method='lm', eps=1.e-2, max_iterations=None, verbose=2):
    import numpy as np
    from scipy.optimize import least_squares
    from sklearn.metrics import r2_score, mean_squared_error

    if method == 'lm':
        result = least_squares(objective_function, initial_guess0, method=method, verbose=verbose)
        params_to_check = calculate_parameters(result.x, C_model, C_measure, J, lengths)
        return result.x, params_to_check
    else:
        if method == 'ng':
            Iter = 0

            while True:
                Iter += 1

                if max_iterations is not None and Iter > max_iterations:
                    break

                len_quads = lengths[0]
                len_corr = lengths[1]
                len_bpm = lengths[2]

                delta_g = initial_guess0[:len_quads]
                delta_x = initial_guess0[len_quads:len_quads + len_corr]
                delta_y = initial_guess0[len_quads + len_corr:]

                B = np.sum([J[k] * delta_g[k] for k in range(len(delta_g))], axis=0)
                Co = C_model * delta_x
                G = C_model * delta_y[:, np.newaxis]

                model = C_model + B + Co + G
                r = C_measure - model

                t2 = np.zeros([len(initial_guess0), 1])
                for i in range(len(initial_guess0)):
                    t2[i] = np.sum(np.dot(J[i], r.T))

                t3 = (np.dot(Jt, t2)).reshape(-1)
                initial_guess1 = initial_guess0 + t3
                t4 = abs(initial_guess1 - initial_guess0)

                if max(t4) <= eps:
                    break
                initial_guess0 = initial_guess1

            delta_g = initial_guess0[:len_quads]
            delta_x = initial_guess0[len_quads:len_quads + len_corr]
            delta_y = initial_guess0[len_quads + len_corr:]

            D = C_measure - C_model
            B = np.sum([J[k] * delta_g[k] for k in range(len(delta_g))], axis=0)
            Co = C_model * delta_x
            G = C_model * delta_y[:, np.newaxis]
            model = C_model + B + Co + G
            residuals = C_measure - model

            #r_squared = r2_score(C_measure, model)
            rmse = sqrt(mean_squared_error(C_measure, model))

            params_to_check = {
               # 'r_squared': r_squared,
                'rmse': rmse,
            }


            return initial_guess0, params_to_check



def objective(delta_params, C_model, C_measure, J, lengths):

    len_quads = lengths[0]
    len_corr = lengths[1]
    len_bpm = lengths[2]

    delta_g = delta_params[:len_quads]
    delta_x = delta_params[len_quads:len_quads + len_corr]
    delta_y = delta_params[len_quads + len_corr:]

    D = C_measure - C_model
    B  = np.sum([J[k] * delta_g[k] for k in range(len(delta_g))], axis=0)
    Co = C_model * delta_x
    G = C_model * delta_y[:, np.newaxis]

    # Define the objective function to be minimized
    residuals = np.square(D - B - Co - G)
    return residuals.ravel()


import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt

def calculate_parameters(parameters, C_model, C_measure, J, lengths):
    len_quads = lengths[0]
    len_corr = lengths[1]
    len_bpm = lengths[2]

    delta_g = parameters[:len_quads]
    delta_x = parameters[len_quads:len_quads + len_corr]
    delta_y = parameters[len_quads + len_corr:]

    D = C_measure - C_model
    B = np.sum([J[k] * delta_g[k] for k in range(len(delta_g))], axis=0)
    Co = C_model * delta_x
    G = C_model * delta_y[:, np.newaxis]
    model = C_model + np.sum([J[k] * delta_g[k] for k in range(len(delta_g))], axis=0) + Co + G
    residuals = C_measure- model

    """
    plt.figure(figsize=(8, 6))
    plt.scatter(X, residuals, marker='o', color='blue', label='Residuals')
    plt.axhline(0, color='red', linestyle='--', label='Zero Residual Line')
    plt.xlabel('X')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    """
    # Calculate R-squared
    r_squared = r2_score(C_measure, model)

    # Calculate RMSE
    rmse = sqrt(mean_squared_error(C_measure, model))

    params_to_check = {
        'r_squared': r_squared,
        'rmse': rmse,
    }

    return params_to_check


def setCorrection(SC, r, elem_ind, Individuals=True, skewness=False, order=1, method='add', elements ='quadrupole'):
    if elements == 'quadrupole':
        if Individuals:
            for i in range(len(elem_ind)):
                SC = set_magnet_setpoints(SC, elem_ind[i], -r[i], skewness, order, method)
        else:
            for quadFam in range(len(elem_ind)):
                for quad in quadFam :
                   SC = set_magnet_setpoints(SC, quad, -r[quadFam], skewness, order, method)
    else:
        for i in range(len(elem_ind)):
            SC = set_cm_setpoints(SC, elem_ind[i], r[i], skewness, order, method)
    return SC