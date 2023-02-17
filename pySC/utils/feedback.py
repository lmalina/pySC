import numpy as np


def isTransmit(hist):
    return hist[0] == 0


def isRepro(BPMhist, N):
    return np.all(np.where(BPMhist[0:N] == BPMhist[0]))


def isNew(BPMhist):
    return BPMhist[0] != BPMhist[1]


def logLastBPM(hist, B):
    hist = np.roll(hist, 1)
    ord = getLastBPMord(B)
    hist[0] = ord if ord else 0
    return hist


def getLastBPMord(B):
    ord = np.where(np.isnan(B))[1]
    if len(ord) > 0:
        return ord[0] - 1
    return None


def goldenDonut(r0, r1, Npts):
    out = np.zeros((2, Npts))  # initialize output array
    phi = 2 * np.pi / ((1 + np.sqrt(5)) / 2)  # golden ratio
    theta = 0
    for n in range(Npts):
        out[:, n] = np.sqrt((r1 ** 2 - r0 ** 2) * n / (Npts - 1) + r0 ** 2) * np.array([np.cos(theta), np.sin(theta)])
        theta = theta + phi
    return out


def is_stable_or_converged(n, eps, BPMhist):  # Balance and Run
    CV = np.var(BPMhist[:n], 1) / np.std(BPMhist[:n])
    return CV < eps
