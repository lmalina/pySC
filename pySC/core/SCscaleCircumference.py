import at


def SCscaleCircumference(RING, circ, mode='abs'):
    allowed_modes = ("abs", "rel")
    if mode not in allowed_modes:
        raise ValueError(f'Unsupported circumference scaling mode: ``{mode}``. Allowed are {allowed_modes}.')
    C = at.get_s_pos(RING)[-1]
    D = 0
    for ord in range(len(RING)):
        if RING[ord].PassMethod == 'DriftPass':
            D += RING[ord].Length
    if mode == 'rel':
        Dscale = 1 - (1 - circ) * C / D
    else:  # mode == 'abs'
        Dscale = 1 - (C - circ) / D
    for ord in range(len(RING)):
        if RING[ord].PassMethod == 'DriftPass':
            RING[ord].Length = RING[ord].Length * Dscale
    return RING
