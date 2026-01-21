import click
from pathlib import Path
from src.utils import read_file, difflog, similarity
from src.osu import Beatmap

@click.command()
@click.argument("path")
@click.option("--diff-log/--no-diff-log", "show_diff", default=True, help="Show diff between raw and parsed map")
def main(path, show_diff):
  raw_map = read_file(path)
  path = Path(path)
  beatmap = Beatmap(file_path=str(path))
  score = similarity(raw_map, str(beatmap))

  if show_diff:
    print(difflog(raw_map, str(beatmap)))
  print("For map:", path)
  print(f"Similarity: {score:.2f}%")
  print(f"Map Style Classes: {[str(m) for m in beatmap.get_style_classes(parent_path=str(path.parent))]}")


if __name__ == "__main__":
  main()