import numpy as np
from pySC.at_wrapper import findspos
from pySC.core.SCgetSupportOffset import SCgetSupportOffset
from pySC.core.SCgetSupportRoll import SCgetSupportRoll
from pySC.core.SCgetTransformation import SCgetTransformation


def SCupdateSupport(SC, BPMstructOffset=True, MAGstructOffset=True):
    if MAGstructOffset:
        if len(SC.ORD.Magnet):
            s = findspos(SC.RING, SC.ORD.Magnet)
            offsets = SCgetSupportOffset(SC, s)
            rolls = SCgetSupportRoll(SC, s)
            for i, ord in enumerate(SC.ORD.Magnet):
                setattr(SC.RING[ord], "SupportOffset", offsets[:, i])
                setattr(SC.RING[ord], "SupportRoll", rolls[:, i])
                magLength = SC.RING[ord].Length
                magTheta = SC.RING[ord].BendingAngle if hasattr(SC.RING[ord], 'BendingAngle') else 0

                dx = SC.RING[ord].SupportOffset[0] + SC.RING[ord].MagnetOffset[0]
                dy = SC.RING[ord].SupportOffset[1] + SC.RING[ord].MagnetOffset[1]
                dz = SC.RING[ord].SupportOffset[2] + SC.RING[ord].MagnetOffset[2]
                az = SC.RING[ord].MagnetRoll[0] + SC.RING[ord].SupportRoll[0]
                ax = SC.RING[ord].MagnetRoll[1] + SC.RING[ord].SupportRoll[1]
                ay = SC.RING[ord].MagnetRoll[2] + SC.RING[ord].SupportRoll[2]
                [T1, T2, R1, R2] = SCgetTransformation(dx, dy, dz, ax, ay, az, magTheta, magLength)
                SC.RING[ord].T1 = T1
                SC.RING[ord].T2 = T2
                SC.RING[ord].R1 = R1
                SC.RING[ord].R2 = R2
                if hasattr(SC.RING[ord], 'MasterOf'):
                    for childOrd in SC.RING[ord].MasterOf:
                        for field in ("T1", "T2", "R1", "R2"):
                            setattr(SC.RING[childOrd], field, getattr(SC.RING[ord], field))

        else:
            print('SC: No magnets have been registered!')
    if BPMstructOffset:
        if len(SC.ORD.BPM):
            s = findspos(SC.RING, SC.ORD.BPM)
            offsets = SCgetSupportOffset(SC, s)
            rolls = SCgetSupportRoll(SC, s)
            for i, ord in enumerate(SC.ORD.BPM):
                setattr(SC.RING[ord], "SupportOffset", offsets[0:2, i])  # TODO Longitudinal BPM offsets not yet implemented
                setattr(SC.RING[ord], "SupportRoll", np.array([rolls[0, i]]))  # TODO BPM pitch and yaw angles not yet implemented
        else:
            print('SC: No BPMs have been registered!')
    return SC
