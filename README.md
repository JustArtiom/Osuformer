# Osuformer - Beatmap Generator Transformer For Osu

> [!WARNING]
> This is the development branch and it might have bugs or even unfinished code. Please check the releases or main branch for more stable version

# Installation

> [!WARNING]
> This code was only tested on python version `3.13.5`

```
pip install -r requirements.txt
```

# Commands

### Run tests
```
python -m pytest
```

### Update requirements.txt
```
python -m scripts.requirements
```

### Parse osu map
```
python -m scripts.parse_map <path/to/map.osu>
```

### Tokenize osu map
```
python -m scripts.tokenize <path/to/map.ous>
```

# Training

Training is a three-step process: prepare the dataset, build a cache, then run the trainer.

### 1. Prepare the dataset

Place your `.osu` beatmap sets inside the `dataset/` directory (one folder per beatmapset, each containing the audio file and `.osu` difficulty files).

To reduce disk usage you can strip unnecessary files first:

```bash
find dataset -type f \( -name "*.jpg" -o -name "*.png" -o -name "*.gif" \
  -o -name "*.mp4" -o -name "*.avi" -o -name "*.flv" \
  -o -name "*.osb" -o -name "*.osk" -o -name "*.osr" \) -delete

find dataset -type d \( -iname "sb" -o -iname "storyboard" \
  -o -iname "skin" -o -iname "skins" -o -iname "effects" \
  -o -iname "particles" -o -iname "bg" -o -iname "video" \
  \) -exec rm -rf {} +
```

### 2. Build the cache

The cache pre-processes audio into mel spectrograms and tokenizes the beatmaps so that the training loop doesn't repeat that work every epoch.

```
python cache.py <cache_name> [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--limit N` | `-1` (all) | Cap the number of maps to process |
| `--workers N` | config | Number of parallel workers |
| `--overwrite / --no-overwrite` | `--overwrite` | Re-build the cache if it already exists |
| `--config PATH` | `configs/default.yaml` | Path to config file |
| `--size NAME` | — | Model size preset (`sm`, `md`, `lg`) |

**Example:**

```bash
python cache.py my_cache --workers 8
```

### 3. Train

```
python train.py --cache <cache_name> [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--cache NAME` | required | Name of the cache built in step 2 |
| `--batch-size N` | config | Training batch size |
| `--lr FLOAT` | config | Initial learning rate |
| `--epochs N` | config | Maximum number of epochs |
| `--workers N` | config | DataLoader workers |
| `--use-ram / --no-use-ram` | config | Load entire cache into RAM |
| `--ckp NAME` | auto-numbered | Checkpoint directory name inside `checkpoints/` |
| `--config PATH` | `configs/default.yaml` | Path to config file |
| `--size NAME` | — | Model size preset (`sm`, `md`, `lg`) |

**Example — single GPU:**

```bash
python train.py --cache my_cache --size sm --batch-size 32 --ckp run1
```

**Example — multi-GPU (torchrun):**

```bash
torchrun --nproc_per_node=4 train.py --cache my_cache --size md --batch-size 16
```

Checkpoints are saved to `checkpoints/<name>/`. Each run produces:
- `best.pt` — best validation loss
- `latest.pt` — most recent epoch (used for resuming)
- `analytics/` — per-epoch loss curves

### Config & model sizes

The default config is `config/default.yaml`. Size presets in `config/size/` override the model architecture:

| Size | d_model | Enc layers | Dec layers |
|---|---|---|---|
| `sm` | 512 | 6 | 6 |
| `md` | 768 | 12 | 10 |
| `lg` | 768 | 14 | 10 |

# Generation

```
python generate.py --audio <path> --bpm <bpm> --offset <ms> --model <checkpoint> [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--audio PATH` | required | Path to the input audio file |
| `--bpm VALUE` | required | Song BPM |
| `--offset MS` | required | Timing offset in milliseconds |
| `--model PATH` | required | Path to a checkpoint file (e.g. `checkpoints/run1/best.pt`) |
| `--sr N` | `4` | Target star rating |
| `--styles LIST` | — | Comma-separated map styles (see below) |
| `--temperature FLOAT` | `1.0` | Sampling temperature (`0` = greedy) |
| `--max-len N` | `4096` | Maximum tokens to generate |
| `--start-ms MS` | `0` | Trim audio start |
| `--end-ms MS` | — | Trim audio end |
| `--strict / --no-strict` | `--strict` | Enforce checkpoint/config compatibility |
| `--config PATH` | `configs/default.yaml` | Path to config file |
| `--size NAME` | — | Model size preset |

**Available styles:** `DEDICATED`, `STREAM`, `DEATHSTREAM`, `BURST`, `SHORT_JUMPS`, `MID_JUMPS`, `LONG_JUMPS`, `DOUBLES`, `TRIPLES`, `QUADS`

**Example:**

```bash
python generate.py \
  --audio song.mp3 \
  --bpm 180 \
  --offset 432 \
  --model checkpoints/run1/best.pt \
  --sr 5 \
  --styles STREAM,BURST \
  --temperature 0.9
```

The model must have been trained with the same `--config` / `--size` flags, otherwise pass `--no-strict`.

# Nice to know

### Dataset cleanup

I managed to get x3.5 times less by cleaning up with these commands

```
find dataset -type f \( -name "*.jpg" -o -name "*.png" \
  -o -name "*.gif" -o -name "*.mp4" -o -name "*.avi" -o -name "*.flv" \
  -o -name "*.osb" -o -name "*.osk" -o -name "*.osr" \) -delete

find dataset -type d \( -iname "sb" -o -iname "storyboard" \
  -o -iname "skin" -o -iname "skins" -o -iname "effects" \
  -o -iname "particles" -o -iname "bg" -o -iname "video"\
  \) -exec rm -rf {} +
```

