import click
from .config import config_options, ExperimentConfig
from .osu import MapStyle

@click.command()
@click.option("--audio", type=str, required=True, help="Path to input audio file")
@click.option("--output", type=str, help="Path to output generated file")
@click.option("--bpm", type=str, required=True, help="Beats per minute for the generated file")
@click.option("--sr", type=int, default=4, help="Osu difficulty star rating target")
@click.option("--model", "checkpoint_path", type=str, required=True, help="Path to model checkpoint")
@click.option("--styles", type=str, help=f"Comma-separated list of styles to use for generation, {', '.join([s.value for s in MapStyle])}")
@click.option("--temperature", type=float, default=1.0, help="Sampling temperature for generation")
@config_options
def main(config: ExperimentConfig):
  pass

if __name__ == "__main__":
  main()