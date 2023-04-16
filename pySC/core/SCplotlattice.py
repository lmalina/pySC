import at
import at.plot
import matplotlib.pyplot as plt
import numpy as np
from plot_synopt import plot_synopt, HCORRECTOR, VCORRECTOR, SkewQUAD
from plot_apertures import plot_apertures

def SCplotLattice(SC,
                  transferLine: bool = False,
                  sRange: list = [],
                  oList: list = [],
                  nSectors: int = 1,
                  plotIdealRing: bool = False,
                  plotMagNames: bool = False,
                  fontSize: int = 16):
    """
    py:func:'SCplotLattice' display lattice optics and correctors locations

    Parameters:
        :param SC:

    Keyword Args:
        transferLine: If true the function 'twissline' is used to calculate the lattice functions
        sRange: Array ['sMin','sMax'] defining the plot range [m].
        oList: If `'sRange'` is empty, `'oList'`  can be used to specify a list of ordinates at which the lattice should
         be plotted
        nSectors: If `'oList'` is empty, `'nSectors'` can be used to plot only the first fraction of the lattice
        plotIdealRing:  Specify if 'SC.IDEALRING' should be used to plot twiss functions, otherwise 'SC.RING'.
        plotMagNames: Specify if magnet names should be printed next to the magnets. Note: since Matlab is not able to
        find the best placement of text annotations automatically, it is likely required that the corresponding lines in
         the code are adjusted to the users discretion
        fontSize: Figure font size.

    Returns:


    Examples:
    Plots the complete lattice for a ring
    ------------------------------------------------------------------
    SCplotLattice(SC);
    -----------------------------------------------------------------
    Plots the lattice from ordinate 30 to 130 for a transfer line
    ------------------------------------------------------------------
    SCplotLattice(SC,transferLine=true,oList=range(30,120));
    ------------------------------------------------------------------

    Plots the lattice of one arc for a twelve-fold symmetric ring lattice
    ------------------------------------------------------------------
    SCplotLattice(SC,nSectors=12);
    ------------------------------------------------------------------
    """
    if plotIdealRing:
        ring = SC.IDEALRING
    else:
        ring = SC.RING

    # define figure
    fig, (axtop, axcent, axbottom) = plt.subplots(3, 1, figsize=(8, 8))
    axtopr = axtop.twinx()

    axtop.set_alpha(0.0)
    axtopr.set_alpha(0.0)
    axcent.set_alpha(0.0)
    axbottom.set_alpha(0.0)

    # Get s - positions along the lattice
    allind = np.array(range(len(ring)+1))
    sPos = at.get_s_pos(ring, allind)
    C = sPos[-1]

    # define s range based on input. sRange has priority, then olist, then nSectors,
    # Check if oList is given explicitly
    # if nSectors is provided, plot only one cell
    if not sRange:

        if nSectors != 1:
            sRange = [0, C/nSectors]

        elif oList != []:
            sRange = [sPos[min(oList)], sPos[max(oList)]]

        else: # use whole lattice
            sRange = ring.s_range

    # assign s_range in lattice, to be used by plot_beta and plot_synopt
    ring.s_range = sRange

    # get input twiss for transferline mode
    if transferLine:
        if hasattr(ring[0], 'TD'):
            inputtwiss = ring[0].TD  # input optics of TL plot
        else:
            inputtwiss = np.recarray(1,
                                 dtype=[('alpha', '<f8', (2,)),
                                         ('beta', '<f8', (2,)),
                                         ('mu', '<f8', (3,)),
                                         ('R', '<f8', (3, 6, 6)),
                                         ('A', '<f8', (6, 6)),
                                         ('dispersion', '<f8', (4,)),
                                         ('closed_orbit', '<f8', (6,)),
                                         ('M', '<f8', (6, 6)),
                                         ('s_pos', '<f8')])

    if transferLine:

        ring.plot_beta(axes=(axtop, axtopr), s_range=sRange, twiss_in=inputtwiss)
        plot_synopt(ring, axes=axtop)
    else:
        ring.plot_beta(axes=(axtop, axtopr), s_range=sRange)
        axsyn = plot_synopt(ring, axes=axtop)  # pending solution from LF and SW

    axtopleg = axtop.get_legend()
    axtopleg.set_ncols(3)

    axtop.set_xlim(sRange)
    axtopr.set_xlim(sRange)

    # plot apertures
    at.plot.generic.baseplot(ring, plot_apertures, axes=(axcent, axcent.twinx()), s_range=sRange)
    axsyncent=plot_synopt(ring, axes=axcent)
    axsyncent.legend(loc='upper center', ncol=6) #, ncol=len(axsynbottom.get_children()))
    # axsyncent.set_ylim([0, 0.3])
    axsyncent.set_xlim(sRange)
    axcent.set_xlim(sRange)


    # plot BPMs and correctors
    hcor = HCORRECTOR
    hcor['refpts'] = sc.ORD.CM
    vcor = VCORRECTOR
    vcor['refpts'] = sc.ORD.CM
    scor = SkewQUAD
    scor['refpts'] = sc.ORD.SkewQuad

    axsynbottom=plot_synopt(ring, axes=axbottom,
                dipole=None, quadrupole=None, sextupole=None, multipole=None,
                # monitor=None,
                hcorrector=hcor, vcorrector=vcor, skewquadrupole=scor)
    axsynbottom.legend(loc='upper center', ncol=4)
    axsynbottom.set_ylim([-5, 5])
    axsynbottom.set_xlim(sRange)
    axbottom.set_xlim(sRange)
    pass

class ord:
    CM=[]
    BPM=[]
    SkewQuad=[]

class SC:
    IDEALRING=None
    RING=None
    ORD = ord()


if __name__=='__main__':

    file='/machfs/liuzzo/EBS/beamdyn/matlab/optics/sr/S28F_all_BM_27Mar2022/betamodel.mat'
    lattice_variable_name = 'betamodel'
    # file = '../scfodo.mat'
    # lattice_variable_name = 'r'

    sc = SC()

    sc.RING = at.load_lattice(file, mat_key=lattice_variable_name)
    sc.IDEALRING = at.load_lattice(file, mat_key=lattice_variable_name)
    sc.ORD.BPM = sc.IDEALRING.get_refpts(at.Monitor)
    sc.ORD.CM = sc.IDEALRING.get_refpts('S[HFDJI]*')
    sc.ORD.SkewQuad = sc.IDEALRING.get_refpts(at.Quadrupole)

    # test simple input
    # SCplotLattice(sc)

    # # test sRange (OK)
    # SCplotLattice(sc, sRange=[0, 56])

    # # test transferline mode
    # sc.RING.disable_6d()
    # opt_in, _, _ = at.linopt4(sc.RING, 0)
    # opt_in.beta=np.array([1.0, 1.0])  # change input beta to see different optics
    # sc.RING[0].TD = opt_in
    # SCplotLattice(sc, transferLine=True, sRange=[0, 26.76])

    # # test oList
    # SCplotLattice(sc, oList=[1230, 2345, 2780, 3456])

    # test nSectors
    SCplotLattice(sc, nSectors=16)

    # # test nSectors+transferline
    # SCplotLattice(sc, transferLine=True, nSectors=16)

    # # test oList+transferline
    # SCplotLattice(sc, transferLine=True, oList=[0, 2345, 2780, 3456])

    """ errors with this output
     File "<input>", line 1, in <module>
  File "/machfs/liuzzo/EBS/emittance_at_Pinholes/ebs_py38/lib/python3.8/site-packages/at/plot/specific.py", line 68, in plot_beta
    return baseplot(ring, pldata_beta_disp, **kwargs)
  File "/machfs/liuzzo/EBS/emittance_at_Pinholes/ebs_py38/lib/python3.8/site-packages/at/plot/generic.py", line 106, in baseplot
    pout = plot_function(rg, rg.i_range, *args, **kwargs)
  File "/machfs/liuzzo/EBS/emittance_at_Pinholes/ebs_py38/lib/python3.8/site-packages/at/plot/specific.py", line 19, in pldata_beta_disp
    data0, _, data = get_optics(ring, refpts=refpts, get_chrom=True, **kwargs)
  File "/machfs/liuzzo/EBS/emittance_at_Pinholes/ebs_py38/lib/python3.8/site-packages/at/physics/linear.py", line 843, in get_optics
    return method(ring, refpts=refpts, dp=dp, **kwargs)
  File "/machfs/liuzzo/EBS/emittance_at_Pinholes/ebs_py38/lib/python3.8/site-packages/at/physics/linear.py", line 739, in linopt6
    return _linopt(ring, _analyze6, *args, **kwargs)
  File "/machfs/liuzzo/EBS/emittance_at_Pinholes/ebs_py38/lib/python3.8/site-packages/at/physics/linear.py", line 340, in _linopt
    orbit, sigma, d0 = build_sigma(twiss_in, orbit)
  File "/machfs/liuzzo/EBS/emittance_at_Pinholes/ebs_py38/lib/python3.8/site-packages/at/physics/linear.py", line 262, in build_sigma
    sigm = twin['R'][0, ...]+10.0*twin['R'][1, ...]
IndexError: index 1 is out of bounds for axis 0 with size 1
    """
    plt.show()

    pass