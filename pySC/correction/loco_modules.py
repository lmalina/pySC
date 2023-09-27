from at import *
import at.plot
from pylab import *
import multiprocessing
import numpy as np
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


def getBetaBeat(ring, twiss, elements_indexes, showplot , debug=False):
    _, _, twiss_error = at.get_optics(ring, elements_indexes)
    s_pos = twiss_error.s_pos
    Beta_x = twiss_error.beta[:, 0]
    Beta_y = twiss_error.beta[:, 1]
    bx = np.array((twiss_error.beta[:, 0] - twiss.beta[:, 0]) / twiss.beta[:, 0])
    by = np.array((twiss_error.beta[:, 1] - twiss.beta[:, 1]) / twiss.beta[:, 1])
    bx_rms = np.sqrt(np.mean(bx ** 2))
    by_rms = np.sqrt(np.mean(by ** 2))

    if debug:
        print("RMS beta beat, x:" + str(bx_rms * 100) + "%   y: " + str(by_rms * 100) + "%")
        #print("STD beta beat, x:" + str(bx_std * 100) + "%   y: " + str(by_std * 100) + "%")

    if showplot  == True:
        plt.rc('font', size=13)
        fig, ax = plt.subplots()
        ax.plot(s_pos, Beta_x)
        ax.set_xlabel("s_pos [m]", fontsize=14)
        ax.set_ylabel(r'$\beta_x%$', fontsize=14)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True, which='both', linestyle=':', color='gray')
        plt.title('Horizontal beta')
        plt.show()
        fig, ax = plt.subplots()
        ax.plot(s_pos, Beta_y)
        ax.set_xlabel("s_pos [m]", fontsize=14)
        ax.set_ylabel(r'$\beta_y%$', fontsize=14)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True, which='both', linestyle=':', color='gray')
        plt.title('Vertical beta')
        plt.show()

    return bx_rms*100, by_rms*100

def rms_orbits(ring, elements_indexes, showplot , debug=False):
    _, _, elemdata = at.get_optics(ring, elements_indexes)
    closed_orbitx = elemdata.closed_orbit[:, 0]
    closed_orbity = elemdata.closed_orbit[:, 2]
    s_pos = elemdata.s_pos

    if showplot  == True:
        plt.rc('font', size=13)
        fig, ax = plt.subplots()
        ax.plot(s_pos, closed_orbitx/ 1.e-06)
        ax.set_xlabel("s_pos [m]", fontsize=14)
        ax.set_ylabel(r"closed_orbit x [μm]", fontsize=14)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True, which='both', linestyle=':', color='gray')
        plt.title("Closed orbit x")
        plt.show()
        fig, ax = plt.subplots()
        ax.plot(s_pos, closed_orbity / 1.e-06)
        ax.set_xlabel("s_pos [m]", fontsize=14)
        ax.set_ylabel(r"closed_orbit y [μm]", fontsize=14)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True, which='both', linestyle=':', color='gray')
        plt.title("Closed orbit y")
        plt.show()
    rmsx =np.sqrt(np.mean(closed_orbitx ** 2))
    rmsy =np.sqrt(np.mean(closed_orbity ** 2))

    if debug:
        print("RMS orbit x:" + str(rmsx*1.e6) + "[μm]   y: " + str(rmsy*1.e6) + "[μm]")

    return rmsx, rmsy

def getDispersion(SC, modeldispersion, rf_step, bpm_ords,cav_ords,  showplot , debug=False):

    measured_dispersion =  dispersion(SC, rf_step, bpm_ords=None, cav_ords=None, n_steps=2)


    bx = np.array((measured_dispersion[:len(bpm_ords)] - modeldispersion[:len(bpm_ords)]) )#/ modeldispersion[:len(bpm_ords)])
    by = np.array((measured_dispersion[len(bpm_ords):] - modeldispersion[len(bpm_ords):]) )#/ modeldispersion[:len(bpm_ords)])
    bx_rms = np.sqrt(np.mean(bx ** 2))
    by_rms = np.sqrt(np.mean(by ** 2))

    if debug:
        print("RMS dispersion, x:" + str(bx_rms) + "  y: " + str(by_rms))

    if showplot  == True:
        plt.rc('font', size=13)
        fig, ax = plt.subplots()
        ax.plot(s_pos, Beta_x)
        ax.set_xlabel("s_pos [m]", fontsize=14)
        ax.set_ylabel(r'$\Dispersion_x%$', fontsize=14)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True, which='both', linestyle=':', color='gray')
        plt.title('Horizontal Dispersion')
        plt.show()
        fig, ax = plt.subplots()
        ax.plot(s_pos, Beta_y)
        ax.set_xlabel("s_pos [m]", fontsize=14)
        ax.set_ylabel(r'$\Dispersion_y%$', fontsize=14)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        ax.grid(True, which='both', linestyle=':', color='gray')
        plt.title('Vertical Dispersion')
        plt.show()

    return bx_rms, by_rms


def choose_spaced_indices(total_elements_ind, num_used_cor1, num_used_cor2):
    if num_used_cor1 <= 1 or num_used_cor2 <= 1:
        raise ValueError("Number of used correctors must be greater than 1")
    step1 = len(total_elements_ind[0]) // (num_used_cor1 - 1)
    indexes1 = total_elements_ind[0][::step1]
    step2 = len(total_elements_ind[1]) // (num_used_cor2 - 1)
    indexes2 = total_elements_ind[1][::step2]

    return [indexes1, indexes2]


def generatingQuadsResponse(SC,C_model, correctrs_kick,used_cor_indexes, used_bpm_indexes, quads_indexes, dk, debug, useIdealRing, trackMode, includeDispersion):
    dCx = []
    dCy = []
    dCxy = []
    dCyx = []
    Cxx0 = C_model[:len(used_bpm_indexes), :len(used_cor_indexes[0])]
    Cxy0 = C_model[len(used_bpm_indexes):, :len(used_cor_indexes[0])]
    Cyx0 = C_model[:len(used_bpm_indexes), len(used_cor_indexes[0]):]
    Cyy0 = C_model[len(used_bpm_indexes):, len(used_cor_indexes[0]):]

    if includeDispersion == True:
        dispersion_model = SCgetModelDispersion(SC, used_bpm_indexes, CAVords=SCgetOrds(SC.RING, 'RFCav'),
                                                trackMode='ORB', Z0=np.zeros(6), nTurns=1,
                                                rfStep=1E3, useIdealRing=True)

        Cxx0 = np.hstack((Cxx0, dispersion_model[:len(used_bpm_indexes)].reshape(-1, 1)))
        Cyx0 = np.hstack((Cyx0, dispersion_model[len(used_bpm_indexes):].reshape(-1, 1)))
        Cxy0 = np.hstack((Cxy0, dispersion_model[:len(used_bpm_indexes)].reshape(-1, 1)))
        Cyy0 = np.hstack((Cyy0, dispersion_model[len(used_bpm_indexes):].reshape(-1, 1)))


    for i in quads_indexes:
        for quad_index in i:
            if debug == True :
               print('generating response to quad of index', quad_index)
            C1x, C1xy, C1y, C1yx = quadsSensitivityMatrices(SC, correctrs_kick,used_cor_indexes, used_bpm_indexes, quad_index, dk, useIdealRing,trackMode, skewness, order, method,includeDispersion)
            dCx.append((C1x - Cxx0) / dk), dCy.append((C1y - Cyy0) / dk), dCxy.append((C1xy - Cxy0) / dk), dCyx.append((C1yx - Cyx0) / dk)
    return dCx, dCy, dCxy, dCyx

def quadsSensitivityMatrices(SC, correctors_kick,used_cor_indexes, used_bpm_indexes, quad_index, dk, useIdealRing, trackMode, skewness, order, method, includeDispersion, rf_step, cav_ords):
    #SC.IDEALRING[quad_index].PolynomB[1] += dk
    SC = set_magnet_setpoints(SC, quad_index, dk, skewness, order, method)
    print('go to model orm')
    C_measured = SCgetModelRM(SC, used_bpm_indexes, used_cor_indexes, dkick = correctors_kick, useIdealRing = useIdealRing, trackMode = trackMode)

    #SC.IDEALRING[quad_index].PolynomB[1] -= dk

    qxx = C_measured[:len(used_bpm_indexes), :len(used_cor_indexes[0])]
    qxy = C_measured[len(used_bpm_indexes):, :len(used_cor_indexes[0])]
    qyx = C_measured[:len(used_bpm_indexes), len(used_cor_indexes[0]):]
    qyy = C_measured[len(used_bpm_indexes):, len(used_cor_indexes[0]):]

    if includeDispersion == True:
        dispersion_model = SCgetModelDispersion(SC, used_bpm_indexes, CAVords=cav_ords , trackMode='ORB', Z0=np.zeros(6), nTurns=1,
                                                rfStep=rf_step, useIdealRing=True)

        qxx =     np.hstack((qxx, dispersion_model[:len(used_bpm_indexes)].reshape(-1, 1)))
        qyx =     np.hstack((qyx, dispersion_model[len(used_bpm_indexes):].reshape(-1, 1)))
        qxy =     np.hstack((qxy, dispersion_model[:len(used_bpm_indexes)].reshape(-1, 1)))
        qyy =     np.hstack((qyy, dispersion_model[len(used_bpm_indexes):].reshape(-1, 1)))

    SC = set_magnet_setpoints(SC, quad_index, -dk, skewness, order, method)


    return qxx, qxy, qyy, qyx


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

def defineMatrices(SC, C_model, C_measure, dCx, dCy, dCxy,dCyx,bpm_indexes, used_cor_ind, includeDispersion, rf_orders):
    Cx0 = transpose(C_model[:len(bpm_indexes), :len(used_cor_ind[0])])
    Cxy0 = transpose(C_model[len(bpm_indexes):, :len(used_cor_ind[0])])
    Cyx0 = transpose(C_model[:len(bpm_indexes), len(used_cor_ind[0]):])
    Cy0 = transpose(C_model[len(bpm_indexes):, len(used_cor_ind[0]):])

    Cx = transpose(C_measure[:len(bpm_indexes), :len(used_cor_ind[0])])
    Cxy = transpose(C_measure[len(bpm_indexes):, :len(used_cor_ind[0])])
    Cyx = transpose(C_measure[:len(bpm_indexes), len(used_cor_ind[0]):])
    Cy = transpose(C_measure[len(bpm_indexes):, len(used_cor_ind[0]):])

    if includeDispersion == True:
        dispersion_model = SCgetModelDispersion(SC, bpm_indexes, CAVords=rf_orders,
                                                trackMode='ORB', Z0=np.zeros(6), nTurns=1,
                                                rfStep=1E3, useIdealRing=True)

        Cx0 = np.hstack((Cx0, dispersion_model[:len(bpm_indexes)].reshape(-1, 1)))
        Cyx0 = np.hstack((Cyx0, dispersion_model[len(bpm_indexes):].reshape(-1, 1)))
        Cxy0 = np.hstack((Cxy0, dispersion_model[:len(bpm_indexes)].reshape(-1, 1)))
        Cy0 = np.hstack((Cy0, dispersion_model[len(bpm_indexes):].reshape(-1, 1)))


        dispersion_measure = dispersion(SC, rf_step=1E3, bpm_ords=bpm_indexes, cav_ords=rf_orders,
                                        n_steps=2)

        Cx = np.hstack((Cx, dispersion_measure[:len(bpm_indexes)].reshape(-1, 1)))
        Cyx = np.hstack((Cyx, dispersion_measure[len(bpm_indexes):].reshape(-1, 1)))
        Cxy = np.hstack((Cxy, dispersion_measure[:len(bpm_indexes)].reshape(-1, 1)))
        Cy = np.hstack((Cy, dispersion_measure[len(bpm_indexes):].reshape(-1, 1)))


    Nk = len(dCx)  # number of free parameters
    Nm = len(Cx0)  # number of measurements

    Bx = np.zeros([Nk, 1])
    By = np.zeros([Nk, 1])
    Bxy = np.zeros([Nk, 1])
    Byx = np.zeros([Nk, 1])
    B = np.zeros([4 * Nk, 1])

    Dx = (Cx[0:Nm, :] - Cx0[0:Nm, :])
    Dy = (Cy[0:Nm, :] - Cy0[0:Nm, :])
    Dxy = (Cxy[0:Nm, :] - Cxy0[0:Nm, :])
    Dyx = (Cyx[0:Nm, :] - Cyx0[0:Nm, :])

    tmp = np.sum(dCx, axis=1)          # Sum over i and j for all planes
    Ax = tmp @ tmp.T                   # Final sum over k for all planes

    tmp = np.sum(dCy, axis=1)
    Ay = tmp @ tmp.T

    tmp = np.sum(dCxy, axis=1)
    Axy = tmp @ tmp.T

    tmp = np.sum(dCyx, axis=1)
    Ayx = tmp @ tmp.T

    A = np.zeros([4 * Nk, Nk])
    A[:Nk, :] = Ax
    A[Nk:2*Nk, :] = Ay
    A[2*Nk:3*Nk, :] = Axy
    A[3*Nk:, :] = Ayx

    for i in range(Nk):
        Bx[i] = np.sum(np.dot(dCx[i], Dx.T))
        By[i] = np.sum(np.dot(dCy[i], Dy.T))
        Bxy[i] = np.sum(np.dot(dCxy[i], Dxy.T))
        Byx[i] = np.sum(np.dot(dCyx[i], Dyx.T))
        B[i] = Bx[i]
        B[i + Nk] = By[i]
        B[i + 2 * Nk] = Bxy[i]
        B[i + 3 * Nk] = Byx[i]

    return A, B


def defineMatrices_Nocoupling(C_model, C_measure, dCx, dCy, dCxy,dCyx,bpm_indexes, used_cor_ind, includeDispersion):
    Cx0 = C_model[:len(bpm_indexes), :len(used_cor_ind[0])]
    Cxy0 = C_model[len(bpm_indexes):, :len(used_cor_ind[0])]
    Cyx0 = C_model[:len(bpm_indexes), len(used_cor_ind[0]):]
    Cy0 = C_model[len(bpm_indexes):, len(used_cor_ind[0]):]

    Cx = C_measure[:len(bpm_indexes), :len(used_cor_ind[0])]
    Cxy = C_measure[len(bpm_indexes):, :len(used_cor_ind[0])]
    Cyx = C_measure[:len(bpm_indexes), len(used_cor_ind[0]):]
    Cy = C_measure[len(bpm_indexes):, len(used_cor_ind[0]):]

    if includeDispersion == True:
        dispersion_model = SCgetModelDispersion(SC, bpm_indexes, CAVords=SCgetOrds(SC.RING, 'RFCav'),
                                                trackMode='ORB', Z0=np.zeros(6), nTurns=1,
                                                rfStep=1E3, useIdealRing=True)

        Cx0 = np.hstack((Cx0, dispersion_model[:len(bpm_indexes)].reshape(-1, 1)))
        Cyx0 = np.hstack((Cyx0, dispersion_model[len(bpm_indexes):].reshape(-1, 1)))
        Cxy0 = np.hstack((Cxy0, dispersion_model[:len(bpm_indexes)].reshape(-1, 1)))
        Cy0 = np.hstack((Cy0, dispersion_model[len(bpm_indexes):].reshape(-1, 1)))

        dispersion_measure = dispersion(SC, rf_step=1E3, bpm_ords=bpm_indexes, cav_ords=SCgetOrds(SC.RING, 'RFCav'),
                                        n_steps=2)

        Cx = np.hstack((Cx, dispersion_measure[:len(bpm_indexes)].reshape(-1, 1)))
        Cyx = np.hstack((Cyx, dispersion_measure[len(bpm_indexes):].reshape(-1, 1)))
        Cxy = np.hstack((Cxy, dispersion_measure[:len(bpm_indexes)].reshape(-1, 1)))
        Cy = np.hstack((Cy, dispersion_measure[len(bpm_indexes):].reshape(-1, 1)))


    Nk = len(dCx)  # number of free parameters
    Nm = len(Cx0)  # number of measurements

    Bx = np.zeros([Nk, 1])
    By = np.zeros([Nk, 1])
    #Bxy = np.zeros([Nk, 1])
    #Byx = np.zeros([Nk, 1])
    B = np.zeros([2 * Nk, 1])

    Dx = (Cx[0:Nm, :] - Cx0[0:Nm, :])
    Dy = (Cy[0:Nm, :] - Cy0[0:Nm, :])
    #Dxy = (Cxy[0:Nm, :] - Cxy0[0:Nm, :])
    #Dyx = (Cyx[0:Nm, :] - Cyx0[0:Nm, :])

    tmp = np.sum(dCx, axis=1)          # Sum over i and j for all planes
    Ax = tmp @ tmp.T                   # Final sum over k for all planes

    tmp = np.sum(dCy, axis=1)
    Ay = tmp @ tmp.T

    tmp = np.sum(dCxy, axis=1)
    Axy = tmp @ tmp.T

    tmp = np.sum(dCyx, axis=1)
    Ayx = tmp @ tmp.T

    A = np.zeros([2 * Nk, Nk])
    A[:Nk, :] = Ax
    A[Nk:2*Nk, :] = Ay
   # A[2*Nk:3*Nk, :] = Axy
   # A[3*Nk:, :] = Ayx

    for i in range(Nk):
        Bx[i] = np.sum(np.dot(dCx[i], Dx.T))
        By[i] = np.sum(np.dot(dCy[i], Dy.T))
        #Bxy[i] = np.sum(np.dot(dCxy[i], Dxy.T))
        #Byx[i] = np.sum(np.dot(dCyx[i], Dyx.T))
        B[i] = Bx[i]
        B[i + Nk] = By[i]
        #B[i + 2 * Nk] = Bxy[i]
        #B[i + 3 * Nk] = Byx[i]

    return A, B


def defineMatrices_coupling(C_model, C_measure, dCx, dCy, dCxy,dCyx,bpm_indexes, used_cor_ind, includeDispersion):

    Cx0 = C_model[:len(bpm_indexes), :len(used_cor_ind[0])]
    Cxy0 = C_model[len(bpm_indexes):, :len(used_cor_ind[0])]
    Cyx0 = C_model[:len(bpm_indexes), len(used_cor_ind[0]):]
    Cy0 = C_model[len(bpm_indexes):, len(used_cor_ind[0]):]

    Cx = C_measure[:len(bpm_indexes), :len(used_cor_ind[0])]
    Cxy = C_measure[len(bpm_indexes):, :len(used_cor_ind[0])]
    Cyx  = C_measure[:len(bpm_indexes), len(used_cor_ind[0]):]
    Cy = C_measure[len(bpm_indexes):, len(used_cor_ind[0]):]

    if includeDispersion == True:
        dispersion_model = SCgetModelDispersion(SC, bpm_indexes, CAVords=SCgetOrds(SC.RING, 'RFCav'),
                                                trackMode='ORB', Z0=np.zeros(6), nTurns=1,
                                                rfStep=1E3, useIdealRing=True)

        Cx0 = np.hstack((Cx0, dispersion_model[:len(bpm_indexes)].reshape(-1, 1)))
        Cyx0 = np.hstack((Cyx0, dispersion_model[len(bpm_indexes):].reshape(-1, 1)))
        Cxy0 = np.hstack((Cxy0, dispersion_model[:len(bpm_indexes)].reshape(-1, 1)))
        Cy0 = np.hstack((Cy0, dispersion_model[len(bpm_indexes):].reshape(-1, 1)))

        dispersion_measure = dispersion(SC, rf_step=1E3, bpm_ords=bpm_indexes, cav_ords=SCgetOrds(SC.RING, 'RFCav'),
                                        n_steps=2)

        Cx = np.hstack((Cx, dispersion_measure[:len(bpm_indexes)].reshape(-1, 1)))
        Cyx = np.hstack((Cyx, dispersion_measure[len(bpm_indexes):].reshape(-1, 1)))
        Cxy = np.hstack((Cxy, dispersion_measure[:len(bpm_indexes)].reshape(-1, 1)))
        Cy = np.hstack((Cy, dispersion_measure[len(bpm_indexes):].reshape(-1, 1)))




    Nk = len(dCx)  # number of free parameters
    Nm = len(Cx0)  # number of measurements

    #Bx = np.zeros([Nk, 1])
    #By = np.zeros([Nk, 1])
    Bxy = np.zeros([Nk, 1])
    Byx = np.zeros([Nk, 1])
    B = np.zeros([2 * Nk, 1])

    #Dx = (Cx[0:Nm, :] - Cx0[0:Nm, :])
    #Dy = (Cy[0:Nm, :] - Cy0[0:Nm, :])
    Dxy = (Cxy[0:Nm, :] - Cxy0[0:Nm, :])
    Dyx = (Cyx[0:Nm, :] - Cyx0[0:Nm, :])

    tmp = np.sum(dCx, axis=1)          # Sum over i and j for all planes
    Ax = tmp @ tmp.T                   # Final sum over k for all planes

    tmp = np.sum(dCy, axis=1)
    Ay = tmp @ tmp.T

    tmp = np.sum(dCxy, axis=1)
    Axy = tmp @ tmp.T

    tmp = np.sum(dCyx, axis=1)
    Ayx = tmp @ tmp.T

    A = np.zeros([2 * Nk, Nk])
    A[:Nk, :] = Axy
    A[Nk:2*Nk, :] = Ayx
   # A[2*Nk:3*Nk, :] = Axy
   # A[3*Nk:, :] = Ayx

    for i in range(Nk):
        Bxy[i] = np.sum(np.dot(dCxy[i], Dx.T))
        Byx[i] = np.sum(np.dot(dCyx[i], Dy.T))
        #Bxy[i] = np.sum(np.dot(dCxy[i], Dxy.T))
        #Byx[i] = np.sum(np.dot(dCyx[i], Dyx.T))
        B[i] = Bxy[i]
        B[i + Nk] = Byx[i]
        #B[i + 2 * Nk] = Bxy[i]
        #B[i + 3 * Nk] = Byx[i]

    return A, B


def getInverse(A, B,Nk, sCut, showPlots):
    u, s, v = np.linalg.svd(A, full_matrices=True) #False

    smat = 0.0 * A
    si = s ** -1
    n_sv = sCut
    si[n_sv:] *= 0.0
    print("number of singular values {}".format(len(si)))
    smat[:Nk, :Nk] = np.diag(si)
    #print('A' + str(A.shape), 'B' + str(B.shape), 'U' + str(u.shape), 'smat' + str(smat.shape), 'v' + str(v.shape))
    Ai = np.dot(v.transpose(), np.dot(smat.transpose(), u.transpose()))
    r = (np.dot(Ai, B)).reshape(-1)
    e = np.dot(A, r).reshape(-1) - B.reshape(-1)

    if showPlots == True:
       plt.figure(figsize=(4, 4))
       plt.plot(np.log(s), 'd--')
       plt.title('singular value')
       plt.show()
       plt.figure(figsize=(4, 4))
       plt.plot(si, 'd--')
       plt.title('singular value')
       plt.show()
       plt.figure(figsize=(4, 4))
       plot(r, 'd')
       plt.xlabel('s(m)')
       plt.ylabel(r'$\frac{\Delta k}{k}%$')
       plt.title('relative quads value')
       plt.show()
       plt.figure(figsize=(4, 4))
       plt.plot(e)
       plt.title('correction error')
       plt.show()
       #plt.plot(B)
       #plt.show()

    return r


def setCorrection(SC, r, quad_ind, Individuals=True):
    if Individuals:
        for i in range(len(quad_ind)):
            SC = set_magnet_setpoints(SC, quad_ind[i], -r[i], False, 1, method="add")
    else:
        for quadFam in range(len(quad_ind)):
            for quad in quadFam :
               SC = set_magnet_setpoints(SC, quad, -r[quadFam], False, 1, method="add")

    return SC

######parral

import multiprocessing

def generatingQuadsResponseParallel(quad_index, SC, C_model, correctrs_kick, used_cor_indexes, used_bpm_indexes, dk, debug, useIdealRing, trackMode, skewness, order, method,includeDispersion, rf_step, cav_ords):
    if debug:
        print('generating response to quad of index', quad_index)

    Cx0 = C_model[:len(used_bpm_indexes), :len(used_cor_indexes[0])]
    Cxy0 = C_model[len(used_bpm_indexes):, :len(used_cor_indexes[0])]
    Cyx0 = C_model[:len(used_bpm_indexes), len(used_cor_indexes[0]):]
    Cy0 = C_model[len(used_bpm_indexes):, len(used_cor_indexes[0]):]
    if includeDispersion == True:
        dispersion_model = SCgetModelDispersion(SC, used_bpm_indexes, CAVords=cav_ords,
                                                trackMode='ORB', Z0=np.zeros(6), nTurns=1,
                                                rfStep=1E3, useIdealRing=True)

        Cx0 = np.hstack((Cx0, dispersion_model[:len(used_bpm_indexes)].reshape(-1, 1)))
        Cyx0 = np.hstack((Cyx0, dispersion_model[len(used_bpm_indexes):].reshape(-1, 1)))
        Cxy0 = np.hstack((Cxy0, dispersion_model[:len(used_bpm_indexes)].reshape(-1, 1)))
        Cy0 = np.hstack((Cy0, dispersion_model[len(used_bpm_indexes):].reshape(-1, 1)))

    C1x, C1xy, C1y, C1yx = quadsSensitivityMatrices(SC, correctrs_kick, used_cor_indexes, used_bpm_indexes, quad_index, dk,trackMode, useIdealRing, skewness, order, method,includeDispersion, rf_step, cav_ords)


    dCx = (C1x - Cx0) / dk
    dCy = (C1y - Cy0) / dk
    dCxy = (C1xy - Cxy0) / dk
    dCyx = (C1yx - Cyx0) / dk

    return dCx, dCy, dCxy, dCyx

def generatingQuadsResponseWorker(args):
    return generatingQuadsResponseParallel(*args)

def generatingJacobian(SC, C_model, dkick, used_cor_ind, bpm_indexes, quads_ind, dk, debug=True, trackMode='ORB', useIdealRing=True, skewness=False, order=1, method='add', includeDispersion=False,  rf_step=1E3, cav_ords=None):
    pool = multiprocessing.Pool()
    quad_args = [(quad_index, SC, C_model, dkick, used_cor_ind, bpm_indexes, dk, debug, trackMode, useIdealRing, skewness, order, method,includeDispersion, rf_step, cav_ords) for quad_index in quads_ind]
    results = pool.map(generatingQuadsResponseWorker, quad_args)
    pool.close()
    pool.join()
    dCx, dCy, dCxy, dCyx = zip(*results)
    return list(dCx), list(dCy), list(dCxy), list(dCyx)





