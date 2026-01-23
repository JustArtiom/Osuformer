from src.tokenizer import Tokenizer
from src.osu import Beatmap
from src.config import config_options, load_config
from src.utils import difflog, similarity

import click
from src.utils import read_file
from src.osu import Beatmap

@click.command()
@click.argument("path")
@click.option("--decode/--no-decode", default=True, help="Decode tokens back to beatmap")
@click.option("--diff-log/--no-diff-log", "show_diff", default=True, help="Path to diff log file")
@click.option("--limit", type=int, default=-1, help="Limit number of maps to process")
@config_options
def main(path, config, decode, show_diff, limit):
  raw_map = read_file(path)
  beatmap = Beatmap(raw=raw_map)
  tokenizer = Tokenizer(config.tokenizer)

  tokens, _ = tokenizer.encode(beatmap)
  tokens = tokens[:limit] if limit > 0 else tokens
  print("Tokens:")
  print([tokenizer.id_to_token[t] for t in tokens])
  compare_to = None
  decoded_score = None

  if decode:
    decoded_beatmap = tokenizer.decode(tokens)
    decoded_beatmap.general = beatmap.general
    decoded_beatmap.editor = beatmap.editor
    decoded_beatmap.events = beatmap.events
    decoded_beatmap.colours = beatmap.colours
    decoded_beatmap.metadata = beatmap.metadata
    decoded_map_str = str(decoded_beatmap)
    decoded_score = similarity(raw_map, decoded_map_str)

    if show_diff:
      compare_to = decoded_map_str

  if compare_to:
    print(difflog(raw_map, compare_to))
    
  if decode:
    print("For map:", path)
    print(f"Similarity: {decoded_score:.2f}%")

if __name__ == "__main__":
  main()