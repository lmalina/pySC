#!/usr/bin/env python
# coding: utf-8

# In[1]:


import at
import numpy as np
from at import Lattice
from pySC.utils.at_wrapper import atloco
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.correction.orbit_trajectory import SCfeedbackFirstTurn, SCfeedbackStitch, SCfeedbackRun, SCfeedbackBalance,     SCpseudoBBA
from pySC.core.beam import bpm_reading, beam_transmission
from pySC.core.constants import *
from pySC.correction.tune import tune_scan
from pySC.lattice_properties.response_model import SCgetModelRM, SCgetModelDispersion
from pySC.lattice_properties.response_measurement import *
from pySC.utils.sc_tools import SCgetOrds, SCgetPinv
from pySC.correction.loco_wrapper import (loco_model, loco_fit_parameters, apply_lattice_correction, loco_measurement,
                                          loco_bpm_structure, loco_cm_structure)
from pySC.plotting.plot_phase_space import plot_phase_space
from pySC.plotting.plot_support import plot_support
from pySC.plotting.plot_lattice import plot_lattice
from pySC.core.lattice_setting import set_magnet_setpoints, switch_cavity_and_radiation, set_cm_setpoints
from pySC.correction.rf import correct_rf_phase, correct_rf_frequency, phase_and_energy_error
from pySC.utils import logging_tools
from pySC.correction.loco_modules import *
from pySC.correction.orbit_trajectory import SCfeedbackRun

LOGGER = logging_tools.get_logger(__name__)

def create_at_lattice() -> Lattice:
    def _marker(name):
        return at.Marker(name, PassMethod='IdentityPass')
    qf = at.Quadrupole('QF', 0.5, 1.2, PassMethod='StrMPoleSymplectic4RadPass')
    qd = at.Quadrupole('QD', 0.5, -1.2, PassMethod='StrMPoleSymplectic4RadPass')
    sf = at.Sextupole('SF', 0.1, 6.0487, PassMethod='StrMPoleSymplectic4RadPass')
    sd = at.Sextupole('SD', 0.1, -9.5203, PassMethod='StrMPoleSymplectic4RadPass')
    bend = at.Bend('BEND', 1, 2 * np.pi / 40, PassMethod='BndMPoleSymplectic4RadPass')
    d2 = at.Drift('Drift', 0.25)
    d3 = at.Drift('Drift', 0.2)
    BPM= at.Monitor('BPM')

    cell = at.Lattice([d2, _marker('SectionStart'), _marker('GirderStart'), bend, d3, sf, d3, _marker('GirderEnd'),
                       _marker('GirderStart'), BPM, qf, d2, d2, bend, d3, sd, d3, qd, d2, _marker('BPM'),
                       _marker('GirderEnd'), _marker('SectionEnd')], name='Simple FODO cell', energy=2.5E9)
    new_ring = at.Lattice(cell * 20)
    rfc = at.RFCavity('RFCav', energy=2.5E9, voltage=2e6, frequency=500653404.8599995, harmonic_number=167, length=0)
    new_ring.insert(0, rfc)
    new_ring.enable_6d()
    at.set_cavity_phase(new_ring)
    at.set_rf_frequency(new_ring)
    return new_ring

if __name__ == "__main__":
    ring = at.Lattice(create_at_lattice())
    LOGGER.info(f"{len(ring)=}")
    SC = SimulatedCommissioning(ring)
    ords = SCgetOrds(SC.RING, 'BPM')
    SC.register_bpms(ords,
                     Roll=0.0)
    ords = SCgetOrds(SC.RING, 'QF')
    SC.register_magnets(ords, HCM=1E-3,
                        CalErrorB=np.array([0, 5E-3]))
    ords = SCgetOrds(SC.RING, 'QD')
    SC.register_magnets(ords, VCM=1E-3,
                        CalErrorB=np.array([0, 5E-3]))
    ords = SCgetOrds(SC.RING, 'BEND')
    SC.register_magnets(ords)
    ords = SCgetOrds(SC.RING, 'SF|SD')
    SC.register_magnets(ords,
                        SkewQuad=0.1)
    ords = SCgetOrds(SC.RING, 'RFCav')
    SC.register_cavities(ords)
    ords = np.vstack((SCgetOrds(SC.RING, 'GirderStart'), SCgetOrds(SC.RING, 'GirderEnd')))
    SC.register_supports(ords, "Girder"),
    ords = np.vstack((SCgetOrds(SC.RING, 'SectionStart'), SCgetOrds(SC.RING, 'SectionEnd')))
    SC.register_supports(ords, "Section")
    SC.INJ.beamSize = np.diag(np.array([200E-6, 100E-6, 100E-6, 50E-6, 1E-3, 1E-4]) ** 2)
    for ord in SCgetOrds(SC.RING, 'Drift'):
        SC.RING[ord].EApertures = 13E-3 * np.array([1, 1])
    for ord in SCgetOrds(SC.RING, 'QF|QD|BEND|SF|SD'):
        SC.RING[ord].EApertures = 10E-3 * np.array([1, 1])
    SC.RING[SC.ORD.Magnet[50]].EApertures = np.array([6E-3, 3E-3])
    plot_lattice(SC, s_range=np.array([0, 20]))
    SC.apply_errors()
    plot_support(SC)

CorOrds = SC.ORD.CM
CAVords = SCgetOrds(SC.RING, 'RFC')
quadsOrds = [SCgetOrds(SC.RING, 'QF'), SCgetOrds(SC.RING, 'QD')]
CAVords = SCgetOrds(SC.RING, 'RFCav')
sextOrds = SCgetOrds(SC.RING, 'SF|SD')
skewOrds =  SC.ORD.SkewQuad
CMstep =  1.e-4 #correctors change [rad]
dk = 1.e-4 #quads change
RFstep = 1e3

_, _, twiss = at.get_optics(SC.IDEALRING, SC.ORD.BPM)
print('Nominal rms orbit')
rmsx, rmsy = rms_orbits(SC.IDEALRING, SC.ORD.BPM, showplot  = False, debug=True)
print(f"Nominal Tune values : {get_tune(SC.IDEALRING, get_integer=True)}, "
      f"Nominal chromaticity: {get_chrom(SC.IDEALRING)}. ")
C_model = SCgetModelRM(SC, SC.ORD.BPM, CorOrds, trackMode='ORB', useIdealRing=True, dkick= CMstep)
ModelDispersion = SCgetModelDispersion(SC, SC.ORD.BPM, CAVords, trackMode='ORB', Z0=np.zeros(6), nTurns=1, rfStep=RFstep, useIdealRing=True)

dCx, dCy, dCxy, dCyx = generatingJacobian(SC, C_model, CMstep, CorOrds, SC.ORD.BPM, concatenate(quadsOrds), dk, debug=True, trackMode='ORB', useIdealRing=False,skewness = False, order=1, method='add', includeDispersion=False, rf_step=RFstep, cav_ords=CAVords )
dCx_skew, dCy_skew, dCxy_skew, dCyx_skew = generatingJacobian(SC, C_model, CMstep, CorOrds, SC.ORD.BPM, skewOrds, dk, debug=True, trackMode='ORB', useIdealRing=False,skewness = True, order=2, method='add', includeDispersion=False,  rf_step=RFstep, cav_ords=CAVords)
dCx_s = np.concatenate((dCx, dCx_skew), axis=0)
dCy_s = np.concatenate((dCy, dCy_skew), axis=0)
dCxy_s = np.concatenate((dCxy, dCxy_skew), axis=0)
dCyx_s = np.concatenate((dCyx, dCyx_skew), axis=0)

numberOfIteration = 1
sCut = 16
for x in range(numberOfIteration): # optics correction using QF and QD
    print('LOCO iteration ', x)

    C_measure = SCgetMeasurRM(SC, SC.ORD.BPM, CorOrds, 'ORB', CMstep)
    bx_rms_err, by_rms_err = getBetaBeat(SC.RING, twiss, SC.ORD.BPM, showplot=False, debug=False)
    dx_rms_err, dy_rms_err = getDispersion(SC,ModelDispersion, RFstep, SC.ORD.BPM,CAVords, showplot =False, debug=False)
    rmsx_err, rmsy_err = rms_orbits(SC.RING, SC.ORD.BPM, showplot = False, debug=False)
    A, B = defineMatrices(SC , C_model, C_measure, np.transpose(dCx, (0, 2, 1)), np.transpose(dCy, (0, 2, 1)), np.transpose(dCxy, (0, 2, 1)),np.transpose(dCyx, (0, 2, 1)),SC.ORD.BPM, CorOrds, False, CAVords)
    Nk = len(dCx)
    print('SVD')
    r = getInverse(A, B, Nk, sCut, showPlots=False)
    SC = setCorrection(SC,r, quadsOrds)
    bx_rms_cor, by_rms_cor = getBetaBeat(SC.RING, twiss, SC.ORD.BPM, showplot=False, debug=False)
    dx_rms_cor, dy_rms_cor = getDispersion(SC,ModelDispersion, RFstep, SC.ORD.BPM,CAVords, showplot =False, debug=False)
    rmsx_corr, rmsy_corr = rms_orbits(SC.RING, SC.ORD.BPM, showplot = False, debug=False)
    print(

        "Before LOCO correction:\n"
        f"RMS horizontal beta beating: {bx_rms_err:.2f}%   RMS vertical beta beating: {by_rms_err:.2f}%\n"
        f"RMS horizontal Dispersion: {dx_rms_err:.2f}mm   RMS vertical Dispersion: {dy_rms_err:.2f}mm\n"
        "RMS orbit x:" + str(rmsx_err) + "[μm]   y: " + str(rmsy_err) + "[μm]\n "

        f"After LOCO corrections\n"
        f"RMS horizontal beta beating: {bx_rms_cor:.2f}%   RMS vertical beta beating: {by_rms_cor:.2f}%\n"
        f"beta_x correction reduction: {(1 - bx_rms_cor / bx_rms_err) * 100:.2f}%\n"
        f"beta_y correction reduction: {(1 - by_rms_cor / by_rms_err) * 100:.2f}%\n "
        f"RMS horizontal Dispersion: {dx_rms_cor:.2f}mm   RMS vertical Dispersion: {dy_rms_cor:.2f}mm\n"
        "RMS orbit x:" + str(rmsx_corr) + "[μm]   y: " + str(rmsy_corr) + "[μm]\n "
    )
numberOfIteration = 3
sCut = 40
for x in range(numberOfIteration): #optics correction using skews
    print('LOCO iteration ', x)

    C_measure = SCgetMeasurRM(SC, SC.ORD.BPM, CorOrds, 'ORB', CMstep)
    bx_rms_err, by_rms_err = getBetaBeat(SC.RING, twiss, SC.ORD.BPM, showplot=False, debug=False)
    dx_rms_err, dy_rms_err = getDispersion(SC,ModelDispersion, RFstep, SC.ORD.BPM,CAVords, showplot =False, debug=False)
    rmsx_err, rmsy_err = rms_orbits(SC.RING, SC.ORD.BPM, showplot = False, debug=False)
    A, B = defineMatrices(SC , C_model, C_measure, np.transpose(dCx_s, (0, 2, 1)), np.transpose(dCy_s, (0, 2, 1)), np.transpose(dCxy_s, (0, 2, 1)),np.transpose(dCyx_s, (0, 2, 1)),SC.ORD.BPM, CorOrds, False, CAVords)
    Nk = len(dCx_s)
    print('SVD')
    r = getInverse(A, B, Nk, sCut, showPlots=False)
    SC = setCorrection(SC,r, np.concatenate((quadsOrds[0],quadsOrds[1], skewOrds)))
    bx_rms_cor, by_rms_cor = getBetaBeat(SC.RING, twiss, SC.ORD.BPM, showplot=False, debug=False)
    dx_rms_cor, dy_rms_cor = getDispersion(SC,ModelDispersion, RFstep, SC.ORD.BPM,CAVords, showplot =False, debug=False)
    rmsx_corr, rmsy_corr = rms_orbits(SC.RING, SC.ORD.BPM, showplot = False, debug=False)
    print(

        "Before LOCO correction:\n"
        f"RMS horizontal beta beating: {bx_rms_err:.2f}%   RMS vertical beta beating: {by_rms_err:.2f}%\n"
        f"RMS horizontal Dispersion: {dx_rms_err:.2f}mm   RMS vertical Dispersion: {dy_rms_err:.2f}mm\n"
        "RMS orbit x:" + str(rmsx_err) + "[μm]   y: " + str(rmsy_err) + "[μm]\n "

        f"After LOCO corrections\n"
        f"RMS horizontal beta beating: {bx_rms_cor:.2f}%   RMS vertical beta beating: {by_rms_cor:.2f}%\n"
        f"beta_x correction reduction: {(1 - bx_rms_cor / bx_rms_err) * 100:.2f}%\n"
        f"beta_y correction reduction: {(1 - by_rms_cor / by_rms_err) * 100:.2f}%\n "
        f"RMS horizontal Dispersion: {dx_rms_cor:.2f}mm   RMS vertical Dispersion: {dy_rms_cor:.2f}mm\n"
        "RMS orbit x:" + str(rmsx_corr) + "[μm]   y: " + str(rmsy_corr) + "[μm]\n "
    )



# In[1]:





# In[1]:





# In[ ]:





# In[ ]:





# In[4]:




