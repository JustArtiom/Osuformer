from src.osu import Beatmap, TimingPoint, Difficulty
from src.config import TokenizerConfig
from src.tokenizer import Tokenizer



beatmap = Beatmap.get_mode("test.osu")
print(beatmap)