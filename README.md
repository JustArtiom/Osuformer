# osu!BMG - Beatmap Generator For Osu

> [!WARNING]
> This is the development branch and it might have bugs or even unfinished code. Please check the releases or main branch for more stable version

osu!BMG is a python project that lets you train and use models different models that help for creating/generating osu! beatmaps.

Py: 3.9+

## Getting Started

1. **Prepare spectrogram cache**
   ```bash
   python3 prep_dataset.py
   ```
   This converts each audio file under `dataset/raw` into a cached mel-spectrogram (`*.mel.npz`) so training can stream data without repeatedly encoding audio.

2. **Train the model**
   ```bash
   python3 train.py --config config.yaml
   ```
   Training uses the Conformer + seq2seq architecture defined in `src/models/` and writes checkpoints + metrics under `checkpoints/<session-id>/`. Install `torch`, `librosa`, and optionally `matplotlib` to enable curve plotting.

3. **Generate a beatmap**
   ```bash
   python3 generate.py \
     --audio dataset/raw/test/audio.mp3 \
     --checkpoint checkpoints/<session-id>/best.pt \
     --template dataset/raw/test/template.osu
   ```
   Provide BPM/offset overrides via `--bpm` / `--offset` if you do not supply a template `.osu`. The script emits a ready-to-open `.osu` file (default suffix `.generated.osu`) containing the predicted circle patterns.

The most important configuration knobs (tick density, beats per sample, batch size, etc.) live in `config.yaml`. Update it to match your dataset before training.
