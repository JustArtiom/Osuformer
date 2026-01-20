import click
import librosa

@click.command()
@click.argument("path")
def main(path):
  y, sr = librosa.load(path)
  onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=32)
  tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=10)
  offset = librosa.frames_to_time(beats[0], sr=sr, hop_length=32)
  print("Audio file:", path)
  print("Tempo:", tempo)
  print("Beat frames:", beats)
  print("First beat offset:", offset)
if __name__ == "__main__":
  main()