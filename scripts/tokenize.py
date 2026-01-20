from src.tokenizer import Tokenizer
from src.osu import Beatmap
from src.config import config_options, load_config

import click
from src.utils import read_file
from src.osu import Beatmap

@click.command()
@click.argument("path")
@click.option("--diff/--no-diff", "show_diff", default=True, help="Show diff between raw and parsed map")
@config_options
def main(path, show_diff, config_path, size):
  config = load_config(config_path, size)
  raw_map = read_file(path)
  beatmap = Beatmap(raw=raw_map)
  tokenizer = Tokenizer(config.tokenizer)

  tokens = tokenizer.encode(beatmap)
  print([tokenizer.id_to_token[t] for t in tokens])

if __name__ == "__main__":
  main()