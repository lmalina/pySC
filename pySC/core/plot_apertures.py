import numpy as np
from at.lattice import Lattice, Refpts


def plot_apertures(ring: Lattice, refpts: Refpts, **kwargs):
    """Generates data for plotting beta functions and dispersion"""

    if refpts[-1]==len(ring):
        refpts = refpts[:-2]
        
    # compute linear optics at the required locations
    data = get_apertures(ring, refpts=refpts, **kwargs)
    ea=np.array(data['EAp'])
    ra = np.array(data['RAp'])

    # Extract the plot data
    s_pos = ring.get_s_pos(refpts=refpts)
    elApa = ea[:, 0] * 1e3
    elApb = ea[:, 1] * 1e3

    recApxmin = ra[:, 0] * 1e3
    recApxmax = ra[:, 1] * 1e3
    recApymin = ra[:, 2] * 1e3
    recApymax = ra[:, 3] * 1e3

    # Left axis definition
    left = (r'Elliptic [mm]', s_pos, [elApa, elApb],
            [r'$a_{ellip}$', r'$b_{ellip}$'])
    # Right axis definition
    right = ('Rectangular [mm]', s_pos, [recApxmin, recApxmax, recApymin, recApymax], ['$x_{min}$', '$x_{max}$', '$y_{min}$', '$y_{max}$'])
    return 'Apertures', left, right


def get_apertures(ring, refpts):

    data = {'EAp': [], 'RAp': []}

    for ind, el in enumerate(ring[refpts]):
        try:
            if hasattr(el, 'EApertures'):
                data['EAp'].append(el.EApertures)
            else:
                data['EAp'].append([np.nan, np.nan])

            if hasattr(el, 'RApertures'):
                data['RAp'].append(el.RApertures)
            else:
                data['RAp'].append([np.nan, np.nan, np.nan, np.nan])
        except IndexError:
            print(f'no index {ind} in ring of length {len(ring)}')

    return data