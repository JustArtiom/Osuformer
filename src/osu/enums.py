from enum import Enum, IntEnum, IntFlag


class GameMode(IntEnum):
    STANDARD = 0
    TAIKO = 1
    CATCH = 2
    MANIA = 3


class Mods(IntFlag):
    NoMod = 0
    NoFail = 1
    Easy = 2
    TouchDevice = 4
    Hidden = 8
    HardRock = 16
    SuddenDeath = 32
    DoubleTime = 64
    Relax = 128
    HalfTime = 256
    NightCore = 512
    Flashlight = 1024
    AutoPlay = 2048
    SpunOut = 4096
    AutoPilot = 8192
    Perfect = 16384
    Key4 = 32768
    Key5 = 65536
    Key6 = 131072
    Key7 = 262144
    Key8 = 524288
    FadeIn = 1048576
    Random = 2097152
    Cinema = 4194304
    Target = 8388608
    Key9 = 16777216
    KeyCoop = 33554432
    Key1 = 67108864
    Key3 = 134217728
    Key2 = 268435456
    ScoreV2 = 536870912
    Mirror = 1073741824


class CurveType(Enum):
    LINEAR = "L"
    BEZIER = "B"
    CATMULL = "C"
    PERFECT = "P"


class SampleSet(IntEnum):
    DEFAULT = 0
    NORMAL = 1
    SOFT = 2
    DRUM = 3


class HitSound(IntFlag):
    NONE = 0
    NORMAL = 1
    WHISTLE = 2
    FINISH = 4
    CLAP = 8


class Effects(IntFlag):
    NONE = 0
    KIAI = 1
    OMIT_FIRST_BAR_LINE = 8
