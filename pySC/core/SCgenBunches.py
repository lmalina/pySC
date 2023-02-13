import numpy as np
from pySC.classes import SimulatedComissioning


def SCgenBunches(SC: SimulatedComissioning):
    global SCinjections
    Z = np.tile(SC.INJ.randomInjectionZ * np.random.randn(2, 6, 1) + SC.INJ.Z0, (1, SC.INJ.nParticles))
    if SC.INJ.nParticles != 1:
        V, L = np.linalg.eig(SC.INJ.beamSize)
        particles = V * np.sqrt(L) * np.random.randn(3, 6, SC.INJ.nParticles)
        Z = Z + particles
    if 'postFun' in SC.INJ.keys() and callable(SC.INJ['postFun']):
        Z = SC.INJ['postFun'](Z)
    SCinjections = SCinjections + 1
    return Z
