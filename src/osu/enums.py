from enum import Enum, auto

class Mods(Enum):
  NoMod = "NoMod"
  Easy = "Easy"
  HardRock = "HardRock"
  SuddenDeath = "SuddenDeath"
  DoubleTime = "DoubleTime"
  NightCore = "NightCore"
  HalfTime = "HalfTime"
  Hidden = "Hidden"
  Flashlight = "Flashlight"
  Relax = "Relax"
  AutoPlay = "AutoPlay"
  SpunOut = "SpunOut"
  AutoPilot = "AutoPilot"

class HitResult(Enum):
  Great = auto()
  Ok = auto()
  Meh = auto()
  Miss = auto()

class CurveType(Enum):
  LINEAR = "L"
  BEZIER = "B"
  CATMULL = "C"
  PERFECT = "P"
