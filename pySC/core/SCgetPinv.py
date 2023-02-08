import numpy as np
import matplotlib.pyplot as plt

def SCgetPinv(M, N=0, alpha=0, damping=1, plot=False):
    u_mat, s_mat, vt_mat = np.linalg.svd(M, full_matrices=False)
    num_singular_values = s_mat.shape[0] - N if N > 0 else s_mat.shape[0]
    available = np.sum(s_mat > 0.)
    keep = min(num_singular_values, available)
    d_mat = np.zeros(s_mat.shape)
    d_mat[:available] = s_mat[:available] / (np.square(s_mat[:available]) + alpha**2) if alpha else 1/s_mat[:available]
    d_mat = damping * d_mat
    minv = np.dot(np.dot(np.transpose(vt_mat[:keep, :]), np.diag(d_mat[:keep])), np.transpose(u_mat[:, :keep]))
    if plot:
        _plot(s_mat, d_mat)
    return minv


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
