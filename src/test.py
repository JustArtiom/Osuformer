from .config import config_options, load_config
from .constraints import build_dsl_tokens
import click 

@click.command()
@config_options
def main(config_path, size):
  config = load_config(config_path, size)
  tokens = build_dsl_tokens(config.tokenizer)
  print("DSL Tokens:")
  print(tokens)
  print("size:", len(tokens))
  print(config)

if __name__ == "__main__":
  main()