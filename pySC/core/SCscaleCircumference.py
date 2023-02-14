import at


def SCscaleCircumference(RING, circ, mode='abs'):
    C = at.get_s_pos(RING, len(RING) + 1)
    D = 0
    for ord in range(len(RING)):
        if RING[ord].PassMethod == 'DriftPass':
            D = D + RING[ord].Length
    if mode == 'rel':
        Dscale = 1 - (1 - circ) * C / D
    elif mode == 'abs':
        Dscale = 1 - (C - circ) / D
    else:
        raise ValueError('Unsupported circumference scaling mode: ''%s''', mode)
    for ord in range(len(RING)):
        if RING[ord].PassMethod == 'DriftPass':
            RING[ord].Length = RING[ord].Length * Dscale
    return RING
