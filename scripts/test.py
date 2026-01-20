from src.osu import Beatmap, TimingPoint, Difficulty
from src.config import TokenizerConfig
from src.tokenizer import Tokenizer



beatmap = Beatmap(difficulty=Difficulty(slider_multiplier=2.0))
tp = TimingPoint(time=0, beat_length=-50, uninherited=0)
beatmap.timing_points.append(tp)

tokenizer = Tokenizer(config=TokenizerConfig)
tokens = tokenizer.encode(beatmap)
readable = [tokenizer.id_to_token[t] for t in tokens]

print(readable)