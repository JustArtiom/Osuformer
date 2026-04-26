from __future__ import annotations

from pathlib import Path

import click
import torch


@click.command()
@click.option("--checkpoint", "src", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help="Source .pt to strip.")
@click.option("--out", "dst", required=True, type=click.Path(dir_okay=False, path_type=Path), help="Destination .pt; place at runs/<new>/latest.pt for fresh fine-tune.")
@click.option("--reset-step", default=True, type=bool, help="Set step=0 so LR scheduler warms up from scratch.")
def main(src: Path, dst: Path, reset_step: bool) -> None:
    print(f"loading {src}...")
    payload = torch.load(src, map_location="cpu", weights_only=False)
    if "model" not in payload:
        raise SystemExit(f"checkpoint missing 'model' key: {src}")
    original_step = int(payload.get("step", 0))
    original_metric = payload.get("metric")
    stripped = {
        "step": 0 if reset_step else original_step,
        "model": payload["model"],
        "optimizer": None,
        "scheduler": None,
        "metric": None,
    }
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stripped, dst)
    print(f"original step={original_step}  metric={original_metric}")
    print(f"wrote {dst}  step={stripped['step']}  optimizer/scheduler=None")


if __name__ == "__main__":
    main()
