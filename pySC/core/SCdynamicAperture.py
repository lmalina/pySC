import matplotlib.pyplot as plt
import numpy as np

from pySC.at_wrapper import atpass, findorbit6, findorbit4


def SCdynamicAperture(RING, dE, bounds=np.array([0, 1e-3]), nturns=1000, thetas=np.linspace(0, 2 * np.pi, 16), accuracy=1e-6,
                      launchOnOrbit=False, centerOnOrbit=True, useOrbit6=False, auto=0, plot=False, verbose=False):
    inibounds = bounds
    if auto > 0:
        _, thetas = _autothetas(RING, dE, auto)
    sidx = np.argsort(np.abs(inibounds))  # Sort bounds w.r.t absolute value
    inibounds = inibounds[sidx]
    ZCO = np.zeros(6)
    if launchOnOrbit:
        if useOrbit6:
            ZCO = findorbit6(RING)
        else:
            tmp = findorbit4(RING, 0)
            if ~np.isnan(tmp[0]):
                ZCO[0:4] = tmp
    ZCO[5] = ZCO[5] + dE
    RMAXs = np.full(len(thetas), np.nan)  # Initialize output array
    DA = np.nan
    for cntt in range(len(thetas)):  # Loop over angles
        theta = thetas[cntt]
        limits = inibounds
        _fatpass(RING, np.full(6, np.nan), 1, 1, [1])  # Fake Track to initialize lattice
        scales = 0
        while scales < 16:
            if _check_bounds(RING, ZCO, nturns, theta, limits):
                break
            limits = _scale_bounds(limits, 10)
            scales = scales + 1
            if verbose:
                print('Scaled: %e %e' % (limits[0], limits[1]))
        while np.abs(limits[1] - limits[0]) > accuracy:
            limits = _refine_bounds(RING, ZCO, nturns, theta, limits)
            if verbose:
                print('Refined: %e %e' % (limits[0], limits[1]))
        RMAXs[cntt] = np.mean(limits)  # Store mean of final boundaries
    if plot:
        plt.figure(6232)
        plt.scatter(np.cos(thetas) * RMAXs, np.sin(thetas) * RMAXs)
        plt.show()
    dthetas = np.diff(thetas)
    r0 = RMAXs[0:(len(RMAXs) - 1)]
    r1 = RMAXs[1:len(RMAXs)]
    DA = np.sum(np.sin(dthetas) * r0 * r1 / 2.)
    if centerOnOrbit:
        if useOrbit6:
            tmp = findorbit6(RING)
        else:
            tmp = findorbit4(RING, 0)
        if not np.isnan(tmp[0]):
            x, y = pol2cart(RMAXs, thetas)
            x = x - tmp[0]
            y = y - tmp[2]
            RMAXs, thetas = cart2pol(x, y)
            RMAXs = RMAXs.T
    return DA, RMAXs, thetas


def _check_bounds(RING, ZCO, nturns, theta, boundsIn):
    Zmin = ZCO[:]
    Zmin[0] = boundsIn[0] * np.cos(theta)
    Zmin[2] = boundsIn[0] * np.sin(theta)
    Zmax = ZCO[:]
    Zmax[0] = boundsIn[1] * np.cos(theta)
    Zmax[2] = boundsIn[1] * np.sin(theta)
    ROUT = _fatpass(RING, [Zmin, Zmax], 0, nturns, [1])  # Track
    RLAST = ROUT[:, len(ROUT) - 1:len(ROUT)]  # Get positions after last turn
    return ~np.isnan(RLAST[0, 0]) and np.isnan(RLAST[0, 1])


def _refine_bounds(RING, ZCO, nturns, theta, boundsIn):
    rmin = boundsIn[0]
    rmax = boundsIn[1]
    rmid = np.mean(boundsIn)
    Z = ZCO[:]
    Z[0] = rmid * np.cos(theta)
    Z[2] = rmid * np.sin(theta)
    ROUT = _fatpass(RING, Z, 0, nturns, [1])  # Track
    RLAST = ROUT[:, len(ROUT) - 1]  # Get positions after last turn
    return [rmin, rmid] if np.isnan(RLAST[0]) else [rmid, rmax]  # Midpoint is outside or inside DA


def _scale_bounds(limits, alpha):
    lower = np.mean(limits) - (np.mean(limits) - limits[0]) * alpha
    upper = np.mean(limits) - (np.mean(limits) - limits[1]) * alpha
    if np.sign(lower) != np.sign(limits[0]):
        lower = 0.0
    if np.sign(upper) != np.sign(limits[1]):
        upper = 0.0
    out = [lower, upper]
    return out


def _fatpass(*args):
    return atpass(*args)


def _autothetas(RING, dE, nt):
    tin = np.linspace(0, 2 * np.pi * 3 / 4, 4)
    [DA, rs, ts] = SCdynamicAperture(RING, dE, thetas=tin, auto=0)
    a = (rs[0] + rs[2]) / 2
    b = (rs[1] + rs[3]) / 2
    e = np.sqrt(1 - b ** 2 / a ** 2)
    mu = np.arccosh(1 / e)
    nu = np.linspace(0, 2 * np.pi, nt)
    rout = np.abs(np.cosh(mu + 1j * nu))
    tout = np.angle(np.cosh(mu + 1j * nu))
    return rout, tout

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y
