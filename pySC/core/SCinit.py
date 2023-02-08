import numpy as np

def SCinit(RING):
    global plotFunctionFlag, SCinjections
    SC = {}
    SC['RING'] = RING
    SC['IDEALRING'] = RING
    SC['INJ'] = {}
    SC['INJ']['beamLostAt'] = 1
    SC['INJ']['Z0ideal'] = np.zeros(6)
    SC['INJ']['Z0'] = SC['INJ']['Z0ideal']
    SC['INJ']['beamSize'] = np.zeros((6, 6))
    SC['INJ']['randomInjectionZ'] = np.zeros(6)
    SC['INJ']['nParticles'] = 1
    SC['INJ']['nTurns'] = 1
    SC['INJ']['nShots'] = 1
    SC['INJ']['trackMode'] = 'TBT'
    SCinjections = 0
    plotFunctionFlag = []
    return SC
