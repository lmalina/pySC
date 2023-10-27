from at import *
from math import sqrt
import multiprocessing
from pySC.lattice_properties.response_model import SCgetModelRM, SCgetModelDispersion
from pySC.core.constants import NUM_TO_AB
import copy
from pySC.core.beam import bpm_reading
import numpy as np
from scipy.optimize import least_squares
from sklearn.metrics import r2_score, mean_squared_error
from lmfit import Parameters, Model
import lmfit

def generatingQuadsResponseWorker(args):
    return generatingQuadsResponseParallel(*args)


def generatingJacobian(SC, C_model, dkick, used_cor_ind, bpm_indexes, quads_ind, dk, debug=True, trackMode='ORB',
                       useIdealRing=True, skewness=False, order=1, method='add', includeDispersion=False, rf_step=1E3,
                       cav_ords=None, full_jacobian=True):
    pool = multiprocessing.Pool()
    quad_args = [(quad_index, SC, C_model, dkick, used_cor_ind, bpm_indexes, dk, debug, trackMode, useIdealRing,
                  skewness, order, method, includeDispersion, rf_step, cav_ords) for quad_index in quads_ind]
    results = pool.map(generatingQuadsResponseWorker, quad_args)
    pool.close()
    pool.join()
    if full_jacobian:  # # Construct the complete Jacobian matrix for the LOCO

        length_quads = len(quads_ind)
        length_corrections = len(np.concatenate(used_cor_ind))
        length_bpm = len(bpm_indexes) * 2

        j2 = np.zeros((length_corrections, C_model.shape[0],
                       C_model.shape[1]))
        for i in range(length_corrections):
            j2[i] = C_model
        j3 = np.zeros((length_bpm, C_model.shape[0], C_model.shape[1]))
        for i in range(length_bpm):
            j3[i] = C_model
        J = np.zeros((length_quads + length_corrections + length_bpm, C_model.shape[0], C_model.shape[1]))
        J[:length_quads] = results
        J[length_quads:length_quads + length_corrections] = j2
        J[length_quads + length_corrections:] = j3

        results = J

    return results


def generatingQuadsResponseParallel(quad_index, SC, C_model, correctrs_kick, used_cor_indexes, used_bpm_indexes, dk,
                                    debug, useIdealRing, trackMode, skewness, order, method, includeDispersion, rf_step,
                                    cav_ords):
    if debug:
        print('generating response to quad of index', quad_index)

    C0 = C_model

    if includeDispersion == True:
        dispersion_model = SCgetModelDispersion(SC, used_bpm_indexes, CAVords=cav_ords,
                                                trackMode='ORB', Z0=np.zeros(6), nTurns=1,
                                                rfStep=1E3, useIdealRing=True)

        C0 = np.hstack((C0, dispersion_model.reshape(-1, 1)))
    C = quadsSensitivityMatrices(SC, correctrs_kick, used_cor_indexes, used_bpm_indexes, quad_index, dk, trackMode,
                                 useIdealRing, skewness, order, method, includeDispersion, rf_step, cav_ords)
    dC = (C - C0) / dk

    return dC


def quadsSensitivityMatrices(SC, correctors_kick, used_cor_indexes, used_bpm_indexes, quad_index, dk, useIdealRing,
                             trackMode, skewness, order, method, includeDispersion, rf_step, cav_ords):
    # SC.IDEALRING[quad_index].PolynomB[1] += dk
    SC.set_magnet_setpoints(quad_index, dk, skewness, order, method)
    print('go to model orm')
    C_measured = SCgetModelRM(SC, used_bpm_indexes, used_cor_indexes, dkick=correctors_kick, useIdealRing=useIdealRing,
                              trackMode=trackMode)

    # SC.IDEALRING[quad_index].PolynomB[1] -= dk

    qx = C_measured

    if includeDispersion == True:
        dispersion_model = SCgetModelDispersion(SC, used_bpm_indexes, CAVords=cav_ords, trackMode='ORB', Z0=np.zeros(6),
                                                nTurns=1,
                                                rfStep=rf_step, useIdealRing=True)

        qx = np.hstack((qx, dispersion_model.reshape(-1, 1)))

    SC.set_magnet_setpoints(quad_index, -dk, skewness, order, method)

    return qx


def SCgetMeasurRM(SC, BPMords, CMords, dkick=1e-5):
    print('Calculating Measure response matrix')

    nTurns = 1
    nBPM = len(BPMords)
    nCM = len(CMords[0]) + len(CMords[1])
    RM = np.full((2 * nBPM * nTurns, nCM), np.nan)

    SC.INJ.trackMode = 'ORB'
    orbits = bpm_reading(SC, bpm_ords=BPMords)
    closed_orbitx0 = orbits[0][0, :]
    closed_orbity0 = orbits[0][1, :]
    cnt = 0
    for nDim in range(2):
        for CMord in CMords[nDim]:
            if SC.RING[CMord].PassMethod == 'CorrectorPass':
                KickNominal = SC.RING[CMord].KickAngle[nDim]
                SC.RING[CMord].KickAngle[nDim] = KickNominal + dkick
                SC.INJ.trackMode = 'ORB'
                orbits = bpm_reading(SC, bpm_ords=BPMords)
                closed_orbitx1 = orbits[0][0, :]
                closed_orbity1 = orbits[0][1, :]
                SC.RING[CMord].KickAngle[nDim] = KickNominal
            else:
                PolynomNominal = getattr(SC.RING[CMord], f"Polynom{NUM_TO_AB[nDim]}")
                delta = dkick / SC.RING[CMord].Length
                changed_polynom = copy.deepcopy(PolynomNominal[:])

                changed_polynom[0] += (-1) ** (nDim+1) * delta
                setattr(SC.RING[CMord], f"Polynom{NUM_TO_AB[nDim]}", changed_polynom[:])

                SC.INJ.trackMode = 'ORB'
                orbits = bpm_reading(SC, bpm_ords=BPMords)
                closed_orbitx1 = orbits[0][0, :]
                closed_orbity1 = orbits[0][1, :]
                setattr(SC.RING[CMord], f"Polynom{NUM_TO_AB[nDim]}", PolynomNominal[:])

            orbitx = (closed_orbitx1 - closed_orbitx0) / dkick
            orbity = (closed_orbity1 - closed_orbity0) / dkick

            RM[:, cnt] = np.concatenate([np.transpose(orbitx), np.transpose(orbity)])

            cnt = cnt + 1
    return RM



def loco_correction(objective_function, initial_guess0, orbit_response_matrix_model, orbit_response_matrix_measured, J, Jt, lengths, including_fit_parameters,W, method='lm', eps=1.e-2, max_iterations=None, verbose=2):

    if method == 'lm':
        result = least_squares(objective_function, initial_guess0, method=method, verbose=verbose)
        params_to_check = calculate_parameters(result.x, orbit_response_matrix_model, orbit_response_matrix_measured, J, lengths,including_fit_parameters,W)
        return result, params_to_check
    else:
        if method == 'ng':
            Iter = 0

            while True:
                Iter += 1

                if max_iterations is not None and Iter > max_iterations:
                    break

                model = orbit_response_matrix_model
                if 'quads' in including_fit_parameters:
                    len_quads = lengths[0]
                    delta_g = initial_guess0[:len_quads]
                    B = np.sum([J[k] * delta_g[k] for k in range(len(delta_g))], axis=0)
                    model += B

                if 'cor' in including_fit_parameters:
                    len_corr = lengths[1]
                    delta_x = initial_guess0[len_quads:len_quads + len_corr]
                    Co = orbit_response_matrix_model * delta_x
                    model += Co

                if 'bpm' in including_fit_parameters:
                    len_bpm = lengths[2]
                    delta_y = initial_guess0[len_quads + len_corr:]
                    G = orbit_response_matrix_model * delta_y[:, np.newaxis]
                    model += G


                r = orbit_response_matrix_measured - model

                t2 = np.zeros([len(initial_guess0), 1])
                for i in range(len(initial_guess0)):
                    t2[i] = np.sum(np.dot(np.dot(W,J[i]), r.T))

                t3 = (np.dot(Jt, t2)).reshape(-1)
                initial_guess1 = initial_guess0 + t3
                t4 = abs(initial_guess1 - initial_guess0)

                if max(t4) <= eps:
                    break
                initial_guess0 = initial_guess1

            # params_to_check_ = calculate_parameters(initial_guess0, orbit_response_matrix_model, orbit_response_matrix_measured, J, lengths,
            #                     including_fit_parameters,W)
            """
            delta_g = initial_guess0[:len_quads]
            delta_x = initial_guess0[len_quads:len_quads + len_corr]
            delta_y = initial_guess0[len_quads + len_corr:]

            D = orbit_response_matrix_measured - orbit_response_matrix_model
            B = np.sum([J[k] * delta_g[k] for k in range(len(delta_g))], axis=0)
            Co = orbit_response_matrix_model * delta_x
            G = orbit_response_matrix_model * delta_y[:, np.newaxis]
            model = orbit_response_matrix_model + B + Co + G
            residuals = orbit_response_matrix_measured - model

            r_squared = r2_score(orbit_response_matrix_measured, model)
            rms = sqrt(mean_squared_error(orbit_response_matrix_measured, model))

            params_to_check_ = {
                #'residulas': residuals,
           'r_squared': r_squared,
            'rmse': rms,
            }

             """
            params_to_check_ = 1
            return initial_guess0, params_to_check_



def objective(delta_params, orbit_response_matrix_model, orbit_response_matrix_measured, J, lengths, including_fit_parameters,W):

    D = orbit_response_matrix_measured - orbit_response_matrix_model
    residuals = D
    if 'quads' in including_fit_parameters:
        len_quads = lengths[0]
        delta_g = delta_params[:len_quads]
        B = np.sum([J[k] * delta_g[k] for k in range(len(delta_g))], axis=0)
        residuals -= B

    if 'cor' in including_fit_parameters:
        len_corr = lengths[1]
        delta_x = delta_params[len_quads:len_quads + len_corr]
        Co = orbit_response_matrix_model * delta_x
        residuals -= Co

    if 'bpm' in including_fit_parameters:
        len_bpm = lengths[2]
        delta_y = delta_params[len_quads + len_corr:]
        G = orbit_response_matrix_model * delta_y[:, np.newaxis]
        residuals -= G

    residuals = np.dot(np.sqrt(W), residuals)

    return residuals.ravel()


def model(parameters, orbit_response_matrix_model, J, lengths, including_fit_parameters, parameter_names):

    model = orbit_response_matrix_model
    len_quads = lengths[0]
    len_corr = lengths[1]
    len_bpm = lengths[2]

    if 'quads' in including_fit_parameters:
        # len_quads = lengths[0]

        delta_g = np.array([parameters[name].value for name in parameter_names[:len_quads]])

        B = np.sum([J[k] * delta_g[k] for k in range(len(delta_g))], axis=0)
        model += B

    if 'cor' in including_fit_parameters:
        # len_corr = lengths[1]
        #delta_x = parameters[len_quads:len_quads + len_corr]
        delta_x = np.array([parameters[name].value for name in parameter_names[len_quads:len_quads + len_corr]])

        Co = orbit_response_matrix_model * delta_x
        model += Co

    if 'bpm' in including_fit_parameters:
        # len_bpm = lengths[2]
        #delta_y = parameters[len_quads + len_corr:]
        delta_y = np.array([parameters[name].value for name in parameter_names[len_quads + len_corr:]])

        G = orbit_response_matrix_model * delta_y[:, np.newaxis]
        model += G

    return model


def loco1(model, delta_params, orbit_response_matrix_measured,W, orbit_response_matrix_model, J, lengths, including_fit_parameters, parameter_names):
    model = Model(model)
    result = model.fit(orbit_response_matrix_measured, params=delta_params,x=[delta_params, orbit_response_matrix_model, J, lengths, including_fit_parameters], weights=W, method='leastsq')
    optimized_params = result.params
    report = result.fit_report()
    return optimized_params, report



def loco(model, delta_params, orbit_response_matrix_measured, W, parameters, orbit_response_matrix_model, J, lengths, including_fit_parameters, parameter_names):
    #model = Model(model)
    #model = Model(model, independent_vars=['x'])
    # Pass the required parameters to model.fit()
     #result = model.fit(orbit_response_matrix_measured, params=delta_params, weights=W, method='leastsq',  parameters=parameters, orbit_response_matrix_model= orbit_response_matrix_model, J=J, lengths=lengths, including_fit_parameters=including_fit_parameters, parameter_names=parameter_names)

    result = lmfit.minimize(model, delta_params, args=(orbit_response_matrix_model, J, lengths, including_fit_parameters, parameter_names))

    optimized_params = result.params
    report = result.fit_report()
    return optimized_params, report



def calculate_parameters(parameters, orbit_response_matrix_model, orbit_response_matrix_measured, J, lengths, including_fit_parameters,W):
    model = orbit_response_matrix_model
    len_quads = lengths[0]
    len_corr = lengths[1]
    len_bpm = lengths[2]

    if 'quads' in including_fit_parameters:
        #len_quads = lengths[0]
        delta_g = parameters[:len_quads]
        B = np.sum([J[k] * delta_g[k] for k in range(len(delta_g))], axis=0)
        model += B

    if 'cor' in including_fit_parameters:
        #len_corr = lengths[1]
        delta_x = parameters[len_quads:len_quads + len_corr]
        Co = orbit_response_matrix_model * delta_x
        model += Co

    if 'bpm' in including_fit_parameters:
        #len_bpm = lengths[2]
        delta_y = parameters[len_quads + len_corr:]
        G = orbit_response_matrix_model * delta_y[:, np.newaxis]
        model += G


    residuals = orbit_response_matrix_measured- model
    # Calculate R-squared
    r_squared = r2_score(orbit_response_matrix_measured, model)#, sample_weight = 1)

    # Calculate RMSE
    rms = sqrt(mean_squared_error(orbit_response_matrix_measured,model))#, model, sample_weight = 1)) #np.diag(W)

    params_to_check_ = {
        #'residulas': residuals,
        'r_squared': r_squared,
        'rmse': rms,
    }

    return params_to_check_


def setCorrection(SC, r, elem_ind, Individuals=True, skewness=False, order=1, method='add', dipole_compensation=True):
        if Individuals:
            for i in range(len(elem_ind)):
                field = elem_ind[i].SCFieldName
                #setpoint = fit_parameters.OrigValues[n_group] + damping * (
                #        fit_parameters.IdealValues[n_group] - fit_parameters.Values[n_group])
                if field == 'SetPointB':  # Normal quadrupole
                    SC.set_magnet_setpoints(ord, -r[i], False, 1, dipole_compensation=dipole_compensation)
                elif field == 'SetPointA':  # Skew quadrupole
                    SC.set_magnet_setpoints(ord, -r[i], True, 1)

                SC.set_magnet_setpoints(elem_ind[i], -r[i], skewness, order, method)
        else:
            for quadFam in range(len(elem_ind)):
                for quad in quadFam :
                    field = elem_ind[quad].SCFieldName
                    # setpoint = fit_parameters.OrigValues[n_group] + damping * (
                    #        fit_parameters.IdealValues[n_group] - fit_parameters.Values[n_group])
                    if field == 'SetPointB':  # Normal quadrupole
                        SC.set_magnet_setpoints(ord, -r[quad], False, 1, dipole_compensation=dipole_compensation)
                    elif field == 'SetPointA':  # Skew quadrupole
                        SC.set_magnet_setpoints(ord, -r[quad], True, 1)

                    SC.set_magnet_setpoints(elem_ind[quad], -r[quad], skewness, order, method)




        return SC