import click
from src.utils import read_file, diff, similarity
from src.osu import Beatmap

@click.command()
@click.argument("path")
def main(path):
    raw_map = read_file(path)
    beatmap = Beatmap(raw=raw_map)
    score = similarity(raw_map, str(beatmap))
    
    print(diff(raw_map, str(beatmap)))
    print(f"Accuracy: {score:.2f}%")


if __name__ == "__main__":
  main()