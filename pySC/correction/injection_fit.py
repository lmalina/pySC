import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from scipy.optimize import fmin
from pySC.core.beam import bpm_reading
from pySC.core.simulated_commissioning import SimulatedCommissioning
from pySC.utils import at_wrapper, logging_tools

LOGGER = logging_tools.get_logger(__name__)
PLANE_STR = ('Hor.', 'Ver.')


def fit_injection_drift(SC: SimulatedCommissioning, n_dims: ndarray = np.array([0, 1]), plot: bool = False):
    # uses SC.INJ.nShots
    s_pos = at_wrapper.findspos(SC.RING)
    bpm_inds = np.arange(2, dtype=int) + len(SC.ORD.BPM) - 1
    bref = bpm_reading(SC)[0][:, bpm_inds]
    s_bpm = np.array([s_pos[SC.ORD.BPM[-1]] - s_pos[-1], s_pos[SC.ORD.BPM[0]]])
    delta_z0 = _fit_bpm_data(s_bpm, bref)
    if np.sum(np.isnan(delta_z0)):
        raise RuntimeError("Failed ")
    _log_initial_final_states(SC, delta_z0)
    if plot:
        fig, ax = plt.subplots(nrows=2, num=343, sharex="all")
        for n_dim in n_dims:
            ax[n_dim].plot(s_bpm, 1E3 * bref[n_dim, :], 'ro', label='BPM reading')
            ax[n_dim].plot(s_bpm, 1E3 * (-delta_z0[2 * n_dim + 1] * s_bpm - delta_z0[2 * n_dim]), 'b--', label='Fitted trajectory')
            ax[n_dim].plot(s_bpm, 1E3 * (SC.INJ.Z0[2 * n_dim + 1] * s_bpm + SC.INJ.Z0[2 * n_dim]), 'k-', label='Real trajectory')
            ax[n_dim].plot(s_bpm, [0, 0], 'k--')
            ax[n_dim].set_ylabel(f'{PLANE_STR[n_dim]} Beam offset [mm]')
        ax[1].set_xlabel('s [m]')
        ax[1].legend()
        fig.show()
    return delta_z0


def fit_injection_trajectory(SC: SimulatedCommissioning, bpm_inds: ndarray = np.array([0, 1, 2]), plot: bool = False):
    # uses SC.INJ.nShots
    s_pos = at_wrapper.findspos(SC.RING)
    bref = bpm_reading(SC)[0][:, bpm_inds]
    delta_z0 = np.zeros(6)
    delta_z0[0:4] = -fmin(_merit_function, np.zeros(4), args=(SC, bref, SC.ORD.BPM[bpm_inds]))
    if np.sum(np.isnan(delta_z0)):
        raise RuntimeError("Failed ")
    _log_initial_final_states(SC, delta_z0)
    SC.INJ.Z0 += delta_z0
    if plot:
        fig, ax = plt.subplots(nrows=2, num=342, sharex="all")
        SC.INJ.Z0 += delta_z0
        bnew = bpm_reading(SC)[0][:, bpm_inds]
        SC.INJ.Z0 -= delta_z0
        s_bpm = s_pos[SC.ORD.BPM[bpm_inds]]
        for n_dim in range(2):
            ax[n_dim].plot(s_bpm, 1E3 * bref[n_dim, :], 'ro--', label="Initial")
            ax[n_dim].plot(s_bpm, 1E3 * bnew[n_dim, :], 'bx--', label='After correction')
            ax[n_dim].set_ylabel(f'{PLANE_STR[n_dim]} BPM reading [mm]')
        ax[1].set_xlabel('s [m]')
        ax[1].legend()
        fig.show()
    return delta_z0


def _fit_bpm_data(s_bpm, bref):
    sol = np.polyfit(s_bpm, bref.T, 1)  # -> [[x', y'], [x, y]]
    delta_z0 = np.zeros(6)
    delta_z0[0:4] -= np.ravel(np.roll(sol.T, 1, axis=1))
    return delta_z0


def _merit_function(x, SC, bref, ords_used):
    t = at_wrapper.atpass(SC.IDEALRING, np.concatenate((x, np.zeros(2))), 1, ords_used, keep_lattice=False)[[0, 2], 0, :, 0]
    return np.sqrt(np.mean(bref - t) ** 2)


def _log_initial_final_states(SC, delta_z0):
    initial = 1e6 * SC.INJ.Z0
    final = 1e6 * (SC.INJ.Z0 + delta_z0)
    LOGGER.debug(f"Injection trajectory corrected from \n "
                 f"x:  {initial[0]:.0f} um -> {final[0]:.0f} um \n x': {initial[1]:.0f} urad -> {final[1]:.0f} urad \n "
                 f"y:  {initial[2]:.0f} um -> {final[2]:.0f} um \n y': {initial[3]:.0f} urad -> {final[3]:.0f} urad\n")
