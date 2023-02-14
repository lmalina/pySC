import matplotlib.pyplot as plt
import numpy as np


def SCdynamicAperture(RING,dE,bounds=[0,1e-3],nturns=1000,nsteps=0,thetas=np.linspace(0,2*np.pi,16),accuracy=1e-6,launchOnOrbit=0,centerOnOrbit=1,useOrbit6=0,auto=0,plot=0,verbose=0):
    inibounds = bounds
    if nsteps!=0:
        print('nsteps no longer supported; continuing with binary search.')
    if auto>0:
        [~,thetas] = autothetas(RING,dE,auto)
    sidx = np.argsort(np.abs(inibounds)) # Sort bounds w.r.t absolute value
    inibounds = inibounds[sidx]
    ZCO = np.zeros(6)
    if launchOnOrbit:
        if useOrbit6:
            ZCO = findorbit6(RING)
        else:
            tmp = findorbit4(RING,0)
            if ~np.isnan(tmp[0]):
                ZCO[0:4] = tmp
    ZCO[5] = ZCO[5] + dE
    RMAXs = np.full(len(thetas),np.nan) # Initialize output array
    DA = np.nan
    for cntt in range(len(thetas)): # Loop over angles
        theta=thetas[cntt]
        limits = inibounds
        fatpass(RING,np.full(6,np.nan),1,1,[1]) # Fake Track to initialize lattice
        scales=0
        while scales<16:
            if check_bounds(RING,ZCO,nturns,theta,limits): break
            limits = scale_bounds(limits,10)
            scales = scales + 1
            if verbose:
                print('Scaled: %e %e' % (limits[0],limits[1]))
        while np.abs(limits[1]-limits[0]) > accuracy:
            limits = refine_bounds(RING,ZCO,nturns,theta,limits)
            if verbose:
                print('Refined: %e %e' % (limits[0],limits[1]))
        RMAXs[cntt]=np.mean(limits) # Store mean of final boundaries
    if plot:
        plt.figure(6232)
        plt.scatter(np.cos(thetas)*RMAXs,np.sin(thetas)*RMAXs)
        plt.show()
    dthetas = np.diff(thetas)
    r0 = RMAXs[0:(len(RMAXs)-1)]
    r1 = RMAXs[1:len(RMAXs)]
    DA = np.sum(np.sin(dthetas) * r0 * r1 / 2.)
    if centerOnOrbit:
        if useOrbit6:
            tmp = findorbit6(RING)
        else:
            tmp = findorbit4(RING,0)
        if ~np.isnan(tmp[0]):
            [x,y] = pol2cart(thetas,RMAXs)
            x = x - tmp[0]
            y = y - tmp[2]
            [thetas,RMAXs] = cart2pol(x,y)
            RMAXs = RMAXs.T
    return DA,RMAXs,thetas

def check_bounds(RING,ZCO,nturns,theta,boundsIn):
    rmin = boundsIn[0]
    rmax = boundsIn[1]
    Zmin = ZCO
    Zmin[0] = rmin * np.cos(theta)
    Zmin[2] = rmin * np.sin(theta)
    Zmax = ZCO
    Zmax[0] = rmax * np.cos(theta)
    Zmax[2] = rmax * np.sin(theta)
    ROUT = fatpass(RING,[Zmin,Zmax],0,nturns,[1]) # Track
    RLAST = ROUT[:,len(ROUT)-1:len(ROUT)] # Get positions after last turn
    if (~np.isnan(RLAST[0,0]) and np.isnan(RLAST[0,1])):
        res = True
    else:
        res = False
    return res

def refine_bounds(RING,ZCO,nturns,theta,boundsIn):
    rmin = boundsIn[0]
    rmax = boundsIn[1]
    rmid = np.mean(boundsIn)
    Z = ZCO
    Z[0] = rmid * np.cos(theta)
    Z[2] = rmid * np.sin(theta)
    ROUT = fatpass(RING,Z,0,nturns,[1]) # Track
    RLAST = ROUT[:,len(ROUT)-1] # Get positions after last turn
    if np.isnan(RLAST[0]): # Midpoint is outside DA
        bounds = [rmin,rmid]
    else: # Midpoint is within DA
        bounds = [rmid,rmax]
    return bounds

def scale_bounds(limits,alpha):
    lower = np.mean(limits)-(np.mean(limits)-limits[0]) * alpha
    upper = np.mean(limits)-(np.mean(limits)-limits[1]) * alpha
    if np.sign(lower) != np.sign(limits[0]):
        lower = 0.0
    if np.sign(upper) != np.sign(limits[1]):
        upper = 0.0
    out = [lower,upper]
    return out

def fatpass(varargin):
    r = atpass(varargin)
    return r

def autothetas(RING,dE,nt):
    tin = np.linspace(0,2*np.pi*3/4,4)
    [DA,rs,ts] = SCdynamicAperture(RING,dE,thetas=tin,auto=0)
    a=(rs[0]+rs[2])/2
    b=(rs[1]+rs[3])/2
    e=np.sqrt(1-b**2/a**2)
    mu=np.arccosh(1/e)
    nu=np.linspace(0,2*np.pi,nt)
    rout = np.abs(np.cosh(mu+1j*nu))
    tout = np.angle(np.cosh(mu+1j*nu))
    return rout,tout

# End

# Test

# RING = atring('LHCB1',1e-3,1e-3)
# [DA,RMAXs,thetas] = SCdynamicAperture(RING,0,plot=1)

# End