from enum import Enum

class EventType(Enum):
  HitObject = 1
  TimingPoint = 2

class HitObjectType(Enum):
  Circle = 1
  Slider = 2
  Spinner = 3