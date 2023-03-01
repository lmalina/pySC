import re

import numpy as np
from at import Lattice
from matplotlib import pyplot as plt
from numpy import ndarray


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


def SCgetOrds(ring: Lattice, regex: str, verbose: bool = False) -> ndarray:
    """
    Returns the indices of the elements in the ring whose names match the regex.

    Parameters
    ----------
    ring : Lattice
        The ring to search.
    regex : str
        The regular expression to match.
    verbose : bool, optional
        If True, prints the names of matched elements.

    Returns
    -------
    ndarray
        The indices of the matched elements.
    """
    if verbose:
        return np.array([_print_elem_get_index(ind, el) for ind, el in enumerate(ring) if re.search(regex, el.FamName)],
                        dtype=int)
    return np.array([ind for ind, el in enumerate(ring) if re.search(regex, el.FamName)], dtype=int)


def _print_elem_get_index(ind, el):
    print(f'Matched: {el.FamName}')
    return ind


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
