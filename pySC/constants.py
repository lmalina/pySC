from typing import Dict, Tuple
RF_PROPERTIES: Tuple[str, str, str] = ('Voltage', 'Frequency', 'TimeLag')
SUPPORT_TYPES:  Tuple[str, str, str] = ('Section', 'Plinth', 'Girder')  #  DO NOT change the order important for offset and roll calculation
TYPE_TO_AB: Dict[str, str] = dict(NORMAL="B", SKEW="A")
NUM_TO_AB: Tuple[str, str] = ("B", "A")
AB_TO_TYPE: Dict[str, str] = dict(B="NORMAL", A="SKEW")
AB_TO_NUM: Dict[str, int] = dict(B=0, A=1)
