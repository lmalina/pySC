def SCgetOrds(RING,rx,verbose=0):
    ords = []
    if isinstance(rx,str):
        for r in rx:
            ords.append(SCgetOrds(RING,r,verbose=verbose))
        return ords
    for ord in range(len(RING)):
        if rx in RING[ord]['FamName']:
            ords.append(ord)
            if verbose:
                print('Matched: %s' % RING[ord]['FamName'])
    return ords
# End
 
