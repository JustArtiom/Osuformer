from enum import IntEnum


class SpecialToken(IntEnum):
    PAD = 0
    SOS_SEQ = 1
    MAP_START = 2
    HISTORY_END = 3
    SOS = 4
    EOS = 5
    TIME_ABS_NULL = 6
    REL_OVERFLOW = 7


SPECIAL_COUNT = len(SpecialToken)
