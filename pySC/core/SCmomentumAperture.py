import numpy as np
import matplotlib.pyplot as plt
import at
from pySC import atpass

def SCmomentumAperture(RING, REFPTS, inibounds, nturns=1000, accuracy=1e-4, stepsize=1e-3, plot=0, debug=0):
    dboundHI = np.zeros(len(REFPTS))
    dboundLO = np.zeros(len(REFPTS))
    ZCOs = at.find_orbit6(RING, REFPTS)
    if any(~np.isfinite(ZCOs.flatten())):
        dbounds = np.array([dboundHI, dboundLO]).T
        print('Closed Orbit could not be determined during MA evaluation. MA set to zero.')
        return dbounds
    for i in range(len(REFPTS)):
        ord = REFPTS[i]
        local_bounds = inibounds
        SHIFTRING = RING.rotate(-ord)  # TODO +/-
        ZCO = ZCOs[:, i]
        while not check_bounds(local_bounds, SHIFTRING, ZCO, nturns):
            local_bounds = increment_bounds(local_bounds, stepsize)
            if debug: print('ord: %d; Incremented: %+0.5e %+0.5e' % (ord, local_bounds[0], local_bounds[1]))
        while abs((local_bounds[1] - local_bounds[0]) / max(local_bounds)) > accuracy:
            local_bounds = refine_bounds(local_bounds, SHIFTRING, ZCO, nturns)
            if debug: print('ord: %d; Refined: %e %e' % (ord, local_bounds[0], local_bounds[1]))
        dboundHI[i] = local_bounds[0]
        dboundLO[i] = local_bounds[1]
        if debug: print('ord: %d; Found: %+0.5e %+0.5e' % (ord, local_bounds[0], local_bounds[1]))
    dbounds = np.array([dboundHI, dboundLO]).T
    if plot:
        spos = at.get_s_pos(RING, REFPTS)
        plt.figure(81222)
        plt.clf()
        plt.hold(True)
        plt.plot(spos, dboundHI, 'kx-')
        plt.plot(spos, dboundLO, 'rx-')
        plt.xlabel('s [m]')
        plt.ylabel('MA')
        plt.show()
    return dbounds


def refine_bounds(local_bounds, RING, ZCO, nturns):
    dmean = np.mean(local_bounds)
    Z0 = ZCO
    Z0[4] = Z0[4] + dmean
    ROUT = atpass(RING, Z0, 1, nturns, [])
    if np.isnan(ROUT[0]):  # Particle dead :(
        local_bounds[1] = dmean  # Set abs-upper bound to midpoint
    else:  # Particle alive :)
        local_bounds[0] = dmean  # Set abs-lower bound to midpoint
    return local_bounds


def check_bounds(local_bounds, RING, ZCO, nturns):
    Z = np.array([ZCO, ZCO])
    Z[4, :] = Z[4, :] + local_bounds[:]
    ROUT = atpass(RING, Z, 1, nturns, [])
    if np.isnan(ROUT[0, 0]) and not np.isnan(ROUT[0, 1]):
        print('Closer-to-momentum particle is unstable. This shouldnt be!')
    return not np.isnan(ROUT[0, 0]) and np.isnan(ROUT[0, 1])


def increment_bounds(local_bounds, stepsize):
    local_bounds = local_bounds + np.array([-1, 1]) * mysign(local_bounds) * stepsize
    return local_bounds


def mysign(v):
    s = 1 - 2 * (v < 0)
    return s


def scale_bounds(local_bounds, alpha):
    lower = np.mean(local_bounds) - (np.mean(local_bounds) - local_bounds[0]) * alpha
    upper = np.mean(local_bounds) - (np.mean(local_bounds) - local_bounds[1]) * alpha
    if np.sign(lower) != np.sign(local_bounds[0]):
        lower = 0.0
    if np.sign(upper) != np.sign(local_bounds[1]):
        upper = 0.0
    out = [lower, upper]
    return out
