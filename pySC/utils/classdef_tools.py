import numpy as np

from pySC.utils.sc_tools import SCrandnc


def update_double_ordinates(ords1, ords2):
    con = np.concatenate((ords1, ords2), axis=1)
    con = con[:, np.lexsort((con[0, :], con[1, :]))]
    return con[:, np.concatenate((np.array([1]), np.sum(np.abs(np.diff(con, axis=1)), axis=0))) != 0]


def intersect(primary, secondary):
    return [elem for elem in primary if elem in secondary]


def randn_cutoff(field, default_cut_off):
    if isinstance(field, list):
        return field[0] * SCrandnc(field[1], np.shape(field[0]))
    return field * SCrandnc(default_cut_off, np.shape(field))


def add_padded(v1, v2):
    if v1.ndim != v2.ndim:
        raise ValueError(f'Unmatched number of dimensions {v1.ndim} and {v2.ndim}.')
    max_dims = np.array([max(d1, d2) for d1, d2 in zip(v1.shape, v2.shape)])
    if np.sum(max_dims > 1) > 1:
        raise ValueError(f'Wrong or mismatching dimensions {v1.shape} and {v2.shape}.')
    vsum = np.zeros(np.prod(max_dims))
    vsum[:np.max(v1.shape)] += v1
    vsum[:np.max(v2.shape)] += v2
    return vsum


def s_interpolation(off, s, C, s1, ord1, f1, s2, ord2, f2):
    for n in range(len(s1)):
        if s1[n] == s2[n]:  # Sampling points have same s-position
            if f1[n] != f2[n]:
                raise ValueError('Something went wrong.')
            ind = np.arange(ord1[n], ord2[n] + 1)
            off[ind] = f1[n]
        elif s1[n] < s2[n]:  # Standard interpolation
            ind = np.arange(ord1[n], ord2[n] + 1)  # last point included
            off[ind] = np.interp(s[ind], np.array([s1[n], s2[n]]), np.array([f1[n], f2[n]]))
        else:  # Injection is within sampling points
            ind1 = np.arange(ord2[n] + 1)
            ind2 = np.arange(ord1[n], len(off) + 1)
            off[ind1] = np.interp(C + s[ind1], np.array([s1[n], s2[n] + C]), np.array([f1[n], f2[n]]))
            off[ord1[n]:] = np.interp(s[ord1[n]:], np.array([s1[n], s2[n] + C]), np.array([f1[n], f2[n]]))
    return off
