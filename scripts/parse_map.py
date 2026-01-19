import click
from src.osu import Beatmap

@click.command
@click.argument("path", )
def main(path):
  beatmap = Beatmap(file_path=path)
  print(beatmap)

if __name__ == "__main__":
  main()