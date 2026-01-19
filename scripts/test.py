from src.osu import Slider

raw = "26,340,109,6,0,P|84:343|159:311,1,135.000005149842,6|2,1:2|0:0,0:0:0:0:"
slider = Slider(raw=raw)
print(slider.__dict__)
print(slider.object_params.__dict__)
print(slider.object_params.curves[0].__dict__)