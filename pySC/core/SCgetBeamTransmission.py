import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from numpy import ndarray
from pySC.core.SCgenBunches import SCgenBunches
from pySC.at_wrapper import atpass
from pySC.classes import SimulatedComissioning


def SCgetBeamTransmission(SC: SimulatedComissioning, nParticles: int = None, nTurns: int = None, plotFlag: bool = False,
                          verbose: bool = False) -> Tuple[int, ndarray]:
    if nParticles is None:
        nParticles = SC.INJ.nParticles
    if nTurns is None:
        nTurns = SC.INJ.nTurns
    if verbose:
        print(f'Calculating maximum beam transmission for {nParticles} particles and {nTurns} turns: ')
    T = atpass(SC.RING, SCgenBunches(SC, nParticles=nParticles), nTurns, np.array([len(SC.RING)]), keep_lattice=False)
    fraction_lost = np.mean(np.isnan(T[0, :, :, :]), axis=(0, 1))
    max_turns = np.sum(fraction_lost < SC.INJ.beamLostAt)
    if plotFlag:
        fig, ax = plt.subplots()
        ax.plot(fraction_lost)
        ax.plot([0, nTurns], [SC.INJ.beamLostAt, SC.INJ.beamLostAt], 'k:')
        ax.set_xlim([0, nTurns])
        ax.set_ylim([0, 1])
        ax.set_xlabel('Number of turns')
        ax.set_ylabel('EDF of lost count')
        fig.show()
    if verbose:
        print(f'{max_turns} turns and {100 * (1 - fraction_lost[-1]):.0f}% transmission.')
    return int(max_turns), fraction_lost


