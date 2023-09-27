#!/usr/bin/env python
# coding: utf-8

# In[13]:


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
    new_ring = at.load_mat('FCCee_z_566_nosol_4_bb.mat')
    new_ring.radiation_off()
    new_ring.energy = 45.6e9
    new_ring.set_cavity(Voltage=79200000.0 ,Frequency=400786627.09839934)
    new_ring.enable_6d()
    at.set_cavity_phase(new_ring)
    at.set_rf_frequency(new_ring)
    new_ring.tapering(niter=3, quadrupole=True, sextupole=True)
    return new_ring


if __name__ == "__main__":
    ring = at.Lattice(create_at_lattice())
    nominal_tune = get_tune(ring, get_integer=True)
    nominal_crom = get_chrom(ring)
    LOGGER.info(f"{len(ring)=}")
    SC = SimulatedCommissioning(ring)

    ords = SCgetOrds(SC.RING, 'BPM')
    SC.register_bpms(ords,
                     Roll=0.0)

    sextOrds = SCgetOrds(SC.RING, 'sf|sd|sy')
    quadOrds = get_refpts(ring, at.elements.Quadrupole)
    arc_quads = [i for i in quadOrds if re.match(r'q[df][1-4]', ring[i].FamName)]
    ir_sext = [
        i
        for i in sextOrds
        if (re.match(r's[fd][1-2][.]', ring[i].FamName)
            or re.match(r'sy', ring[i].FamName))]
    arc_sext = [i for i in sextOrds if i not in ir_sext]
    ir_quads = [i for i in quadOrds if i not in arc_quads]
    ords = quadOrds
    SC.register_magnets(ords)
    ords = sextOrds
    SC.register_magnets(ords)
    ords = arc_quads
    SC.register_magnets(ords,CalErrorB=np.array([0, 1E-3]) )
    ords = arc_sext
    SC.register_magnets(ords,CalErrorB=np.array([0, 0, 1E-3]))

    SC.apply_errors()


cor_ind_x = SCgetOrds(SC.RING, 'CXY')
cor_ind_y = SCgetOrds(SC.RING, 'CXY')
Corords = np.vstack((cor_ind_x, cor_ind_y))
used_Corords = choose_spaced_indices(Corords, 20, 20)
CAVords = SCgetOrds(SC.RING, 'RFC')
sextOrds = SCgetOrds(SC.RING, 'sf|sd')
skewOrds =  SC.ORD.SkewQuad
CMstep =  1.e-10 #correctors change [rad]
dk = 1.e-6 #quads change
RFstep = 1e3
_, _, twiss = at.get_optics(SC.IDEALRING, SC.ORD.BPM)
print('Nominal rms orbit')
rmsx, rmsy = rms_orbits(SC.IDEALRING, SC.ORD.BPM, showplot  = False, debug=True)
print(f"Nominal Tune values : {get_tune(SC.IDEALRING, get_integer=True)}, "
      f"Nominal chromaticity: {get_chrom(SC.IDEALRING)}. ")
C_model = SCgetModelRM(SC, SC.ORD.BPM, used_Corords, trackMode='ORB', useIdealRing=True, dkick= CMstep)
ModelDispersion = SCgetModelDispersion(SC, SC.ORD.BPM, CAVords, trackMode='ORB', Z0=np.zeros(6), nTurns=1, rfStep=RFstep, useIdealRing=True)

dCx, dCy, dCxy, dCyx = generatingJacobian(SC, C_model, CMstep, CorOrds, SC.ORD.BPM, quadOrds, dk, debug=True, trackMode='ORB', useIdealRing=False,skewness = False, order=1, method='add', includeDispersion=False, rf_step=RFstep, cav_ords=CAVords )


numberOfIteration = 3
sCut = 1250
for x in range(numberOfIteration): # optics correction using QF and QD
    print('LOCO iteration ', x)

    C_measure = SCgetMeasurRM(SC, SC.ORD.BPM, used_Corords, 'ORB', CMstep)
    bx_rms_err, by_rms_err = getBetaBeat(SC.RING, twiss, SC.ORD.BPM, showplot=False, debug=False)
    dx_rms_err, dy_rms_err = getDispersion(SC,ModelDispersion, RFstep, SC.ORD.BPM,CAVords, showplot =False, debug=False)
    rmsx_err, rmsy_err = rms_orbits(SC.RING, SC.ORD.BPM, showplot = False, debug=False)
    A, B = defineMatrices(SC , C_model, C_measure, np.transpose(dCx, (0, 2, 1)), np.transpose(dCy, (0, 2, 1)), np.transpose(dCxy, (0, 2, 1)),np.transpose(dCyx, (0, 2, 1)),SC.ORD.BPM, used_Corords, False, CAVords)
    Nk = len(dCx)
    print('SVD')
    r = getInverse(A, B, Nk, sCut, showPlots=False)
    SC = setCorrection(SC,r, quadOrds)
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


# In[ ]:




