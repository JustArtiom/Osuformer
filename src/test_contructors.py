from osu import General, TimingPoint, Circle, Slider, Spinner

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
print(g.__dict__)
print(str(g))

timing_point = "-28,461.538461538462,4,1,0,100,1,0"
t = TimingPoint(raw=timing_point)
print(t.__dict__)
print(str(t))

timing_point = "34125,-100,4,1,0,87,0,0"
t = TimingPoint(raw=timing_point)
print(t.__dict__)
print(str(t))

circle = "255,184,3664,69,4,3:1:0:0:"
h = Circle(raw=circle)
print(h.__dict__)
print(str(h))

slider = "333,159,57202,6,0,P|256:40|175:157,1,335.999989746094,4|4,0:0|0:2,3:0:0:0:"
h = Slider(raw=slider)
print(h.__dict__)
print(h.object_params.__dict__)
print(str(h))

spinner = "256,192,126433,12,4,129202,3:2:0:0:"
h = Spinner(raw=spinner)
print(h.__dict__)
print(str(h))