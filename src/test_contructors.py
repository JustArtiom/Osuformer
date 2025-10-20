from osu import (
  General, 
  TimingPoint,
  Circle, 
  Slider, 
  Spinner, 
  SliderCurve,
  SliderObjectParams, 
  SpinnerObjectParams,
  HitSample
)

print("\n[General]")
general = """AudioFilename: audio.mp3
AudioLeadIn: 0
PreviewTime: 99664
Countdown: 0
SampleSet: Normal
StackLeniency: 0.7
Mode: 0
LetterboxInBreaks: 0
UseSkinSprites: 1
SkinPreference:Default
WidescreenStoryboard: 1
SamplesMatchPlaybackRate: 1"""
g = General(raw=general)
print(str(g))

print("\n[Difficulty]")
difficulty = """HPDrainRate: 5
CircleSize: 4
OverallDifficulty: 7
ApproachRate: 9
SliderMultiplier: 1.4
SliderTickRate: 1"""
d = General(raw=difficulty)
print(str(d))


print("\n[TimingPoint]")
timing_point = "-28,461.538461538462,4,1,0,100,1,0"
t = TimingPoint(raw=timing_point)
t2 = TimingPoint(
  time=t.time,
  beat_length=t.beat_length,
  meter=t.meter,
  sample_set=t.sample_set,
  sample_index=t.sample_index,
  volume=t.volume,
  uninherited=t.uninherited,
  effects=t.effects,
)
print(str(t))
print(str(t2))

timing_point = "34125,-100,4,1,0,87,0,0"
t = TimingPoint(raw=timing_point)
t2 = TimingPoint(
  time=t.time,
  beat_length=t.beat_length,
  meter=t.meter,
  sample_set=t.sample_set,
  sample_index=t.sample_index,
  volume=t.volume,
  uninherited=t.uninherited,
  effects=t.effects,
)
print(str(t))
print(str(t2))

print("\n[HitObjects]")
circle = "255,184,3664,69,4,3:1:0:0:"
h = Circle(raw=circle)
h2 = Circle(
  x=h.x,
  y=h.y,
  time=h.time,
  type=h.type,
  hit_sound=HitSample(
    normal_set=h.hit_sample.normal_set,
    addition_set=h.hit_sample.addition_set,
    index=h.hit_sample.index,
    volume=h.hit_sample.volume
  ),
)
print(str(h))

slider = "333,159,57202,6,0,P|256:40|175:157,1,335.999989746094,4|4,0:0|0:2,3:0:0:0:"
h = Slider(raw=slider)
h2 = Slider(
  x=h.x,
  y=h.y,
  time=h.time,
  type=h.type,
  hit_sound=h.hit_sound,
  object_params=SliderObjectParams(
    curves=[
      SliderCurve(
        curve_type=curve.curve_type,
        curve_points=curve.curve_points
      ) for curve in h.object_params.curves
    ],
    slides=h.object_params.slides,
    length=h.object_params.length,
    edge_sounds=h.object_params.edge_sounds,
    edge_sets=h.object_params.edge_sets,
  ),
  hit_sample=HitSample(
    normal_set=h.hit_sample.normal_set,
    addition_set=h.hit_sample.addition_set,
    index=h.hit_sample.index,
    volume=h.hit_sample.volume
  ),
)
print(str(h))
print(str(h2))

spinner = "256,192,126433,12,4,129202,3:2:0:0:"
h = Spinner(raw=spinner)
h2 = Spinner(
  x=h.x,
  y=h.y,
  time=h.time,
  type=h.type,
  hit_sound=h.hit_sound,
  object_params=SpinnerObjectParams(
    end_time=h.object_params.end_time
  ),
  hit_sample=HitSample(
    normal_set=h.hit_sample.normal_set,
    addition_set=h.hit_sample.addition_set,
    index=h.hit_sample.index,
    volume=h.hit_sample.volume
  ),
)
print(str(h))
print(str(h2))