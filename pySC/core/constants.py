from typing import Dict, Tuple
GENERIC_FIELDS: Tuple[str, str, str] = ("Setpoint", "Offset", "CalError")
RF_PROPERTIES: Tuple[str, str, str] = ('Voltage', 'Frequency', 'TimeLag')
RF_ERROR_FIELDS: Tuple[str] = tuple(f"{prop}{field}" for prop in RF_PROPERTIES for field in GENERIC_FIELDS)
SUPPORT_TYPES:  Tuple[str, str, str] = ('Section', 'Plinth', 'Girder')  #  DO NOT change the order important for offset and roll calculation
SUPPORT_ERROR_FIELDS: Tuple[str, str] = ("Offset", "Roll")
BPM_ERROR_FIELDS: Tuple[str, str, str, str, str, str] = ("Noise", "NoiseCO", "Offset", "Roll", "CalError", "SumError")
MAGNET_TYPE_FIELDS: Tuple[str, str, str, str, str] = ('HCM', 'VCM', 'CF', 'SkewQuad', 'MasterOf')
MAGNET_ERROR_FIELDS: Tuple[str, str, str, str, str] = ("MagnetOffset", "MagnetRoll", "CalErrorA", "CalErrorB", "BendingAngle")  # TODO BendingAngleError?
AB: Tuple[str, str] = ("A", "B")
TYPE_TO_AB: Dict[str, str] = dict(NORMAL="B", SKEW="A")
NUM_TO_AB: Tuple[str, str] = ("B", "A")
AB_TO_TYPE: Dict[str, str] = dict(B="NORMAL", A="SKEW")
AB_TO_NUM: Dict[str, int] = dict(B=0, A=1)

TRACK_TBT: str = "TBT"
TRACK_ORB: str = "ORB"
TRACK_PORB: str = "PORB"
TRACKING_MODES: Tuple[str, str, str] = (TRACK_TBT, TRACK_ORB, TRACK_PORB)

SETTING_ABS: str = "abs"
SETTING_REL: str = "rel"
SETTING_ADD: str = "add"
SETTING_METHODS: Tuple[str, str, str] = (SETTING_ABS, SETTING_REL, SETTING_ADD)
SPEED_OF_LIGHT = 299792458
SETPOINT: str = "SetPoint"
