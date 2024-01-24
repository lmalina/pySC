import matplotlib.pyplot as plt
import numpy as np

from pySC.core.constants import SPEED_OF_LIGHT
from pySC.utils import at_wrapper
from pySC.core.beam import generate_bunches


def plot_phase_space(SC, ords=np.zeros(1, dtype=int), custom_bunch=None, nParticles=None, nTurns=None, plotCO=False):
    init_font = plt.rcParams["font.size"]
    plt.rcParams.update({'font.size': 18})
    z_in, n_particles, n_turns = _check_input(SC, custom_bunch, nParticles, nTurns)
    T = at_wrapper.atpass(SC.RING, z_in, n_turns, ords, keep_lattice=False)
    T[:, np.isnan(T[0, :])] = np.nan
    label_str = [r'$\Delta x$ [$\mu$m]', r"$\Delta x'$ [$\mu$rad]", r'$\Delta y$ [$\mu$m]', r"$\Delta y'$ [$\mu$rad]",
                 r'$\Delta S$ [m]', r'$\delta E$ $[\%]$']
    title_str = ['Horizontal', 'Vertical', 'Longitudinal']
    cav_ord = SC.ORD.RF[0]
    if SC.RING[cav_ord].PassMethod == 'RFCavityPass':
        circumference = at_wrapper.findspos(SC.RING)[-1]
        length_slippage = SPEED_OF_LIGHT * SC.RING[cav_ord].HarmNumber / SC.RING[cav_ord].Frequency - circumference
        T[5, :, :, :] = T[5, :, :, :] - length_slippage * np.arange(n_turns)[np.newaxis, np.newaxis, :]
        label_str[4] = r'$\Delta S_{act}$ [m]'
    if plotCO:
        CO = np.squeeze(at_wrapper.findorbit6(SC.RING, ords)[1])
        if np.isnan(CO[0]):
            start_point_guess = np.nanmean(T, axis=(1, 2, 3))
            CO = np.squeeze(at_wrapper.findorbit6(SC.RING, ords, guess=start_point_guess)[1])
            if np.isnan(CO[0]):
                CO = np.full(6, np.nan)
    else:
        CO = np.full(6, np.nan)
    T = T * np.array([1E6, 1E6, 1E6, 1E6, 1E2, 1])[:, np.newaxis, np.newaxis, np.newaxis]
    Z0 = SC.INJ.Z0 * np.array([1E6, 1E6, 1E6, 1E6, 1E2, 1])
    CO = CO * np.array([1E6, 1E6, 1E6, 1E6, 1E2, 1])
    T[[4, 5], :, :, :] = T[[5, 4], :, :, :]
    CO[[4, 5]] = CO[[5, 4]]
    Z0[[4, 5]] = Z0[[5, 4]]

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18.5, 10.5), dpi=100, facecolor="w")
    for nType in range(3):
        for nP in range(n_particles):
            x = T[2 * nType, nP, :, :]
            y = T[2 * nType + 1, nP, :, :]
            handle = ax[nType].scatter(x, y, 10, np.arange(n_turns), vmin=0, vmax=n_turns)
        ax[nType].plot(Z0[2 * nType], Z0[2 * nType + 1], 'ro', markersize=13, label='Injection point')
        if plotCO:
            ax[nType].plot(CO[2 * nType], CO[2 * nType + 1], 'kX', markersize=18, linewidth=4, label='Closed orbit')
        ax[nType].set_xlabel(label_str[2 * nType])
        ax[nType].set_ylabel(label_str[2 * nType + 1])
        ax[nType].set_title(f"{title_str[nType]} @ {SC.RING[ords[0]].FamName}")
    c1 = plt.colorbar(handle, ax=ax[2], orientation='vertical', label='Number of turns')
    plt.pause(1)
    ax[0].legend()
    fig.tight_layout()
    fig.show()
    plt.rcParams.update({'font.size': init_font})


def _check_input(SC, custom_bunch, n_particles, n_turns):
    if n_turns is None:
        n_turns = SC.INJ.nTurns
    if custom_bunch is not None and n_particles is not None:
        raise ValueError("Either custom bunch or number of particles can be defined, not both.")
    if custom_bunch is not None:
        if custom_bunch.shape[0] != 6:
            raise ValueError("Custom bunch array must have first dimension equal to six.")
        input_bunch = custom_bunch.copy()
        n_particles = input_bunch.shape[1]
        return input_bunch, n_particles, n_turns
    if n_particles is None:
        n_particles = SC.INJ.nParticles
    input_bunch = generate_bunches(SC, nParticles=n_particles)
    return input_bunch, n_particles, n_turns
