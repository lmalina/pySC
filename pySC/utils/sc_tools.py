import re

import numpy as np
from at import Lattice
from matplotlib import pyplot as plt
from numpy import ndarray

from pySC.utils.at_wrapper import findspos
from pySC.utils import logging_tools

LOGGER = logging_tools.get_logger(__name__)


def SCrandnc(cut_off: float = 2, shape: tuple = (1, )) -> ndarray:
    """
    Generates an array of random number(s) from normal distribution with a cut-off.

    Parameters
    ----------
    cut_off : float
        The cut-off value.
    shape : tuple
        The shape of the output array.

    Returns
    -------
    out : ndarray
        The output array.
    """
    out_shape = (1,) if np.sum(shape) < 1 else shape
    out = np.random.randn(np.prod(out_shape))
    outindex = np.abs(out) > np.abs(cut_off)
    while np.sum(outindex):
        out[outindex] = np.random.randn(np.sum(outindex))
        outindex = np.abs(out) > np.abs(cut_off)
    return out.reshape(out_shape)


def SCgetOrds(ring: Lattice, regex: str) -> ndarray:
    """
    Returns the indices of the elements in the ring whose names match the regex.

    Parameters
    ----------
    ring : Lattice
        The ring to search.
    regex : str
        The regular expression to match.

    Returns
    -------
    ndarray
        The indices of the matched elements.
    """
    indices = np.array([ind for ind, el in enumerate(ring) if re.search(regex, el.FamName)], dtype=int)
    for ind in indices:
        LOGGER.debug(f'Matched: {ring[ind].FamName}')
    return indices


def SCgetPinv(matrix: ndarray, num_removed: int = 0, alpha: float = 0, damping: float = 1, plot: bool = False) -> ndarray:
    """
    Computes the pseudo-inverse of a matrix using the Singular Value Decomposition (SVD) method.

    Parameters
    ----------
    matrix : ndarray
        The matrix to be inverted.
    num_removed : int, optional
        The number of singular values to be removed from the matrix.
    alpha : float, optional
        The regularization parameter.
    damping : float, optional
        The damping factor.
    plot : bool, optional
        If True, plots the singular values and the damping factor.

    Returns
    -------
    matrix_inv : ndarray
        The pseudo-inverse of the matrix.
    """
    u_mat, s_mat, vt_mat = np.linalg.svd(matrix, full_matrices=False)
    num_singular_values = s_mat.shape[0] - num_removed if num_removed > 0 else s_mat.shape[0]
    available = np.sum(s_mat > 0.)
    keep = min(num_singular_values, available)
    d_mat = np.zeros(s_mat.shape)
    d_mat[:available] = s_mat[:available] / (np.square(s_mat[:available]) + alpha**2) if alpha else 1/s_mat[:available]
    d_mat = damping * d_mat
    matrix_inv = np.dot(np.dot(np.transpose(vt_mat[:keep, :]), np.diag(d_mat[:keep])), np.transpose(u_mat[:, :keep]))
    if plot:
        _plot_singular_values(s_mat, d_mat)
    return matrix_inv


def SCscaleCircumference(RING, circ, mode='abs'):  # TODO
    allowed_modes = ("abs", "rel")
    if mode not in allowed_modes:
        raise ValueError(f'Unsupported circumference scaling mode: ``{mode}``. Allowed are {allowed_modes}.')
    C = findspos(RING)[-1]
    D = 0
    for ind in range(len(RING)):
        if RING[ind].PassMethod == 'DriftPass':
            D += RING[ind].Length
    if mode == 'rel':
        Dscale = 1 - (1 - circ) * C / D
    else:  # mode == 'abs'
        Dscale = 1 - (C - circ) / D
    for ind in range(len(RING)):
        if RING[ind].PassMethod == 'DriftPass':
            RING[ind].Length = RING[ind].Length * Dscale
    return RING


def update_transformation(element):
    mag_length = getattr(element, "Length", 0)
    mag_theta = getattr(element, 'BendingAngle', 0)
    offsets = element.SupportOffset + element.MagnetOffset
    rolls = np.roll(element.MagnetRoll + element.SupportRoll, -1)  # z,x,y -> x,y,z
    x_axis = np.array([1, 0, 0])
    y_axis = np.array([0, 1, 0])
    z_axis = np.array([0, 0, 1])
    r_3d = rotation(rolls)
    ld = np.dot(np.dot(r_3d, z_axis), offsets)

    T = _translation_vector(ld, r_3d, np.dot(r_3d, x_axis), np.dot(r_3d, y_axis), offsets)
    element.R1 = _r_matrix(ld, r_3d)
    element.T1 = np.dot(np.linalg.inv(element.R1), T)

    RX = rotation(rolls)
    RB = rotation([0, -mag_theta, 0])
    r_3d = np.dot(RB.T, np.dot(RX.T, RB))
    OPp = np.array([(mag_length * (np.cos(mag_theta) - 1) / mag_theta if mag_theta else 0),
                    0,
                    mag_length * (np.sin(mag_theta) / mag_theta if mag_theta else 1)])

    OpPp = OPp - np.dot(RX, OPp) - offsets
    ld = np.dot(np.dot(RB, z_axis), OpPp)

    element.T2 = _translation_vector(ld, r_3d, np.dot(RB, x_axis), np.dot(RB, y_axis), OpPp)
    element.R2 = _r_matrix(ld, r_3d)
    return element


def SCmultipolesRead(fname):  # TODO sample of the input anywhere?
    f = open(fname, 'r')
    tab = np.array(f.read().split()).astype(float)
    f.close()
    if len(tab) % 3 != 0:
        LOGGER.error('Incorrect table size.')
        return
    AB = tab.reshape((-1, 3))[:, 1:]
    idx = np.where(AB == 1)
    if len(idx[0]) != 1:
        LOGGER.warning('Nominal order could not be (uniquely) determined. Continuing with idx=1.')
        idx = 1
    order, type = idx[0][0], idx[1][0]
    if type > 2:
        LOGGER.error('Ill-defined magnet type.')
        return
    return np.roll(AB, 1, axis=1), order, type  # swapping A and B


def _plot_singular_values(s_mat, d_mat):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4), dpi=100, facecolor="w")
    ax[0].semilogy(np.diag(s_mat) / np.max(np.diag(s_mat)), 'o--')
    ax[0].set_xlabel('Number of SV')
    ax[0].set_ylabel('$\sigma/\sigma_0$')
    ax[1].plot(s_mat * d_mat, 'o--')
    ax[1].set_xlabel('Number of SV')
    ax[1].set_ylabel('$\sigma * \sigma^+$')
    ax[1].yaxis.tick_right()
    ax[1].yaxis.set_label_position("right")
    fig.show()


def rotation(rolls):
    ax, ay, az = rolls
    # The order of extrinsic rotations (fixed frame) around ZYX, i.e. Roll, Yaw, Pitch
    R0 = np.array([[np.cos(ay) * np.cos(az), -np.cos(ay) * np.sin(az), np.sin(ay)],
                   [np.cos(az) * np.sin(ax) * np.sin(ay) + np.cos(ax) * np.sin(az),
                    np.cos(ax) * np.cos(az) - np.sin(ax) * np.sin(ay) * np.sin(az), -np.cos(ay) * np.sin(ax)],
                   [-np.cos(ax) * np.cos(az) * np.sin(ay) + np.sin(ax) * np.sin(az),
                    np.cos(az) * np.sin(ax) + np.cos(ax) * np.sin(ay) * np.sin(az), np.cos(ax) * np.cos(ay)]])
    return R0


def _translation_vector(ld, r3d, xaxis_xyz, yaxis_xyz, offsets):
    tD0 = np.array([-np.dot(offsets, xaxis_xyz), 0, -np.dot(offsets, yaxis_xyz), 0, 0, 0])
    T0 = np.array([ld * r3d[2, 0] / r3d[2, 2], r3d[2, 0], ld * r3d[2, 1] / r3d[2, 2], r3d[2, 1], 0, ld / r3d[2, 2]])
    return T0 + tD0


def _r_matrix(ld, r3d):
    return np.array([
        [r3d[1, 1] / r3d[2, 2], ld * r3d[1, 1] / r3d[2, 2] ** 2, -r3d[0, 1] / r3d[2, 2], -ld * r3d[0, 1] / r3d[2, 2] ** 2, 0, 0],
        [0, r3d[0, 0], 0, r3d[1, 0], r3d[2, 0], 0],
        [-r3d[1, 0] / r3d[2, 2], -ld * r3d[1, 0] / r3d[2, 2] ** 2, r3d[0, 0] / r3d[2, 2], ld * r3d[0, 0] / r3d[2, 2] ** 2, 0, 0],
        [0, r3d[0, 1], 0, r3d[1, 1], r3d[2, 1], 0],
        [0, 0, 0, 0, 1, 0],
        [-r3d[0, 2] / r3d[2, 2], -ld * r3d[0, 2] / r3d[2, 2] ** 2, -r3d[1, 2] / r3d[2, 2], -ld * r3d[1, 2] / r3d[2, 2] ** 2, 0, 1]
    ])
