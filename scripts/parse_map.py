import click
from src.utils import read_file, difflog, similarity
from src.osu import Beatmap

@click.command()
@click.argument("path")
@click.option("--diff/--no-diff", "show_diff", default=True, help="Show diff between raw and parsed map")
def main(path, show_diff):
  raw_map = read_file(path)
  beatmap = Beatmap(raw=raw_map)
  score = similarity(raw_map, str(beatmap))

  if show_diff:
    print(difflog(raw_map, str(beatmap)))
  print("For map:", path)
  print(f"Similarity: {score:.2f}%")


if __name__ == "__main__":
  main()