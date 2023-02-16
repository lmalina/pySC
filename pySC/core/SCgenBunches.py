import numpy as np
from pySC.classes import SimulatedComissioning
from pySC.core.SCrandnc import SCrandnc


def SCgenBunches(SC: SimulatedComissioning):
    global SCinjections
    Z = np.tile(SC.INJ.randomInjectionZ * SCrandnc(2, (6, 1)) + SC.INJ.Z0, (1, SC.INJ.nParticles))  # TODO
    if SC.INJ.nParticles != 1:
        V, L = np.linalg.eig(SC.INJ.beamSize)
        particles = V * np.sqrt(L) * SCrandnc(3, (6, SC.INJ.nParticles))
        Z = Z + particles
    if 'postFun' in SC.INJ.keys() and callable(SC.INJ['postFun']):
        Z = SC.INJ['postFun'](Z)
    SCinjections = SCinjections + 1
    return Z
