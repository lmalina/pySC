import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray


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
        _plot(s_mat, d_mat)
    return matrix_inv


def _plot(s_mat, d_mat):
    plt.figure(66)
    plt.subplot(1, 2, 1)
    plt.semilogy(np.diag(s_mat) / np.max(np.diag(s_mat)), 'o--')
    plt.xlabel('Number of SV')
    plt.ylabel('$\sigma/\sigma_0$')
    plt.subplot(1, 2, 2)
    plt.plot(np.diag(s_mat) * np.diag(d_mat), 'o--')
    plt.xlabel('Number of SV')
    plt.ylabel('$\sigma * \sigma^+$')
    plt.gca().yaxis.tick_right()
    plt.gca().yaxis.set_label_position("right")
    plt.gcf().set_size_inches(12, 4)
    plt.gcf().set_dpi(100)
    plt.gcf().set_facecolor('w')
    plt.show()
