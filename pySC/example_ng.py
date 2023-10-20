#!/usr/bin/env python
# coding: utf-8

# In[10]:


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
    new_ring.tapering(niter=3, quadrupole=True, sextupole=True)

    return new_ring

if __name__ == "__main__":
    ring = at.Lattice(create_at_lattice())
    LOGGER.info(f"{len(ring)=}")
    SC = SimulatedCommissioning(ring)
    ords = SCgetOrds(SC.RING, 'BPM')
    SC.register_bpms(ords,
                     Roll=0.0)#, CalError=5E-2 * np.ones(2))
    ords = SCgetOrds(SC.RING, 'QF')
    SC.register_magnets(ords, HCM=1E-3,
                        CalErrorB=np.array([0, 10E-3]))
    ords = SCgetOrds(SC.RING, 'QD')
    SC.register_magnets(ords, VCM=1E-3,
                        CalErrorB=np.array([0, 10E-3]))
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
    #plot_support(SC)

CorOrds = SC.ORD.CM
CAVords = SCgetOrds(SC.RING, 'RFC')
quadsOrds = [SCgetOrds(SC.RING, 'QF'), SCgetOrds(SC.RING, 'QD')]
#quadsOrds = [SCgetOrds(SC.RING, 'QF')]
CAVords = SCgetOrds(SC.RING, 'RFCav')
sextOrds = SCgetOrds(SC.RING, 'SF|SD')
skewOrds =  SC.ORD.SkewQuad
CMstep =  1.e-10 #correctors change [rad]
dk = 1.e-4 #quads change
RFstep = 1e3

C_model = SCgetModelRM(SC, SC.ORD.BPM, CorOrds, trackMode='ORB', useIdealRing=True, dkick= CMstep)
ModelDispersion = SCgetModelDispersion(SC, SC.ORD.BPM, CAVords, trackMode='ORB', Z0=np.zeros(6), nTurns=1, rfStep=RFstep, useIdealRing=True)

dC = generatingJacobian(SC, C_model, CMstep, CorOrds, SC.ORD.BPM, np.concatenate(quadsOrds), dk, debug=True, trackMode='ORB', useIdealRing=False,skewness = False, order=1, method='add', includeDispersion=False, rf_step=RFstep, cav_ords=CAVords )
#dC_skew = generatingJacobian(SC, C_model, CMstep, CorOrds, SC.ORD.BPM, skewOrds, dk, debug=True, trackMode='ORB', useIdealRing=False,skewness = True, order=2, method='add', includeDispersion=False,  rf_step=RFstep, cav_ords=CAVords)
#dC_s = np.concatenate((dC, dC_skew), axis=0)

err =[]
for i in concatenate(quadsOrds):
    err.append(SC.RING[i].K- SC.IDEALRING[i].K)

C_measure = SCgetMeasurRM(SC, SC.ORD.BPM, CorOrds, 'ORB', CMstep)
length_quads = len(concatenate(quadsOrds))
length_corrections = len(concatenate(CorOrds))
length_bpm = len(SC.ORD.BPM) * 2

j2 = np.zeros((length_corrections, C_model.shape[0],C_model.shape[1]))  # # Construct the complete Jacobian matrix for the LOCO
for i in range(length_corrections):
    j2[i] = C_model
j3 = np.zeros((length_bpm, C_model.shape[0],C_model.shape[1]))
for i in range(length_bpm):
    j3[i] = C_model
J = np.zeros((length_quads + length_corrections + length_bpm, C_model.shape[0], C_model.shape[1]))
J[:length_quads]=dC
J[length_quads:length_quads+length_corrections]=j2
J[length_quads+length_corrections:]=j3

tmp = np.sum( J, axis=1)  # Jacobian inverse
t1 = tmp @ tmp.T
u, s, v = np.linalg.svd(t1, full_matrices=True)
plt.plot(np.log(s), 'd--')
plt.show()
smat = 0.0 * t1
si = s ** -1
n_sv = 40
si[n_sv:] *= 0.0
Nk = len(J)
smat[:Nk, :Nk] = np.diag(si)
Jt = np.dot(v.transpose(), np.dot(smat.transpose(), u.transpose()))


# In[11]:


print('LOCO correction')

initial_guess = np.zeros(length_quads + length_corrections + length_bpm)
initial_guess[0:length_quads] = 1e-6
initial_guess[length_quads:length_quads + length_corrections] = 1
initial_guess[length_quads + length_corrections:] = 1
lengths = [len(concatenate(quadsOrds)),len(concatenate(CorOrds)),  len(SC.ORD.BPM) *2]

fit_parameters, params_to_check  = loco_correction(lambda delta_params: objective(delta_params, C_model, C_measure, dC, lengths),initial_guess,C_model, C_measure, J,Jt, lengths,
                             method='ng',eps =1.e-6,max_iterations= 10000 , verbose=2)

#fit_parameters = loco_fit_parameters(SC, SC.IDEALRING, ring_data, RFstep,
#                                         [SCgetOrds(SC.RING, 'QF'), False, 'individual', 1E-3],
                                         # {Ords, normal/skew, ind/fam, deltaK}
#                                         [SCgetOrds(SC.RING, 'QD'), False, 'individual', 1E-4])
dg  = fit_parameters[0:length_quads]
dx = fit_parameters[length_quads:length_quads + length_corrections]
dy = fit_parameters[length_quads + length_corrections:]
print('Fit result:', params_to_check)
#for err, dg in zip(err, dg):
#        print(f"quads err: {err}, quads fit: {dg}")
#SC = setCorrection(SC,dg, concatenate(quadsOrds))
#SC = apply_lattice_correction(SC, fit_parameters)


# In[12]:


for err, dg in zip(err, dg):
        print(f"quads err: {err}, quads fit: {dg}")


# In[12]:





# In[12]:





# In[12]:





# In[12]:





# In[12]:





# In[12]:





# In[9]:





# In[ ]:




