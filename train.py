import tensorflow as tf
import numpy as np
from utils import args
from utils.dataset import Dataset

config = args.train()

np.random.seed(config.seed)
tf.random.set_seed(config.seed)

if config.device == "cuda" and tf.config.list_physical_devices("GPU"):
    print("Using GPU for training...")
    device = "/GPU:0"
else:
    print("Using CPU for training...")
    device = "/CPU:0"

dataset = Dataset(config)
osu_maps, audio_maps = dataset.load()