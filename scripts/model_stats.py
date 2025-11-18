from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import ConformerSeq2Seq
from src.utils.config import load_config


def human_readable(num: float) -> str:
    for unit in ("", "K", "M", "B", "T"):
        if abs(num) < 1000.0:
            return f"{num:.2f}{unit}" if unit else f"{int(num)}"
        num /= 1000.0
    return f"{num:.2f}P"


def collect_param_stats(model: torch.nn.Module) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
        "total_human": human_readable(total),
        "trainable_human": human_readable(trainable),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print model + training hyperparameters and parameter counts.")
    parser.add_argument("--config", default="config.yaml", help="Path to config file (default: config.yaml)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model = ConformerSeq2Seq(config)
    stats = collect_param_stats(model)

    print("=== Model Parameter Stats ===")
    print(f"Total params    : {stats['total']:,} ({stats['total_human']})")
    print(f"Trainable params: {stats['trainable']:,} ({stats['trainable_human']})")
    print(f"Frozen params   : {stats['non_trainable']:,}")

    print("\n=== Model Hyperparameters ===")
    for section, values in config.get("model", {}).items():
        print(f"[{section}]")
        for key, value in values.items():
            print(f"  {key}: {value}")

    print("\n=== Training Hyperparameters ===")
    for key, value in config.get("training", {}).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

