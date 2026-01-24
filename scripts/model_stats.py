import torch
from typing import Any, Dict, Tuple
from src.config import ExperimentConfig, config_options
from src.model import build_model
from src.constraints import build_dsl_tokens
import click

def human_readable(num: float) -> str:
    for unit in ("", "K", "M", "B", "T"):
        if abs(num) < 1000.0:
            return f"{num:.2f}{unit}" if unit else f"{int(num)}"
        num /= 1000.0
    return f"{num:.2f}P"


def count_params(module: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return int(total), int(trainable)


def collect_param_stats(model: torch.nn.Module) -> Dict[str, Any]:
    total, trainable = count_params(model)
    non_trainable = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
        "total_human": human_readable(total),
        "trainable_human": human_readable(trainable),
    }

@click.command()
@config_options
def main(config: ExperimentConfig):
  vocab, _ = build_dsl_tokens(config.tokenizer)
  model = build_model(config, vocab_size=len(vocab))
  stats = collect_param_stats(model)

  print("=== Model Parameter Stats ===")
  print(f"Total params    : {stats['total']:,} ({stats['total_human']})")
  print(f"Trainable params: {stats['trainable']:,} ({stats['trainable_human']})")
  print(f"Frozen params   : {stats['non_trainable']:,}")
  print(f"Vocab size      : {len(vocab):,}")

  print("\n=== Top-level Module Breakdown ===")
  rows = []
  for name, child in model.named_children():
    total, trainable = count_params(child)
    rows.append((name, total, trainable))
  rows.sort(key=lambda r: r[1], reverse=True)
  for name, total, trainable in rows:
    print(f"{name:16s} total={total:>12,} trainable={trainable:>12,}")

  print("\n=== Model Hyperparameters ===\n")
  for section, values in config.model.__dict__.items():
    print(f"[{section}]")
    for key, value in values.__dict__.items():
      print(f"  {key}: {value}")

  print("\n=== Training Hyperparameters ===")
  for key, value in config.training.__dict__.items():
      print(f"{key}: {value}")

if __name__ == "__main__":
    main()