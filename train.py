import tensorflow as tf
import numpy as np
from utils import args
from utils.dataset import Dataset

config = args.train()
print(config)
dataset = Dataset(config)

np.random.seed(config.seed)
tf.random.set_seed(config.seed)

if config.device == "cuda" and tf.config.list_physical_devices("GPU"):
    print("Using GPU for training...")
    device = "/GPU:0"
else:
    print("Using CPU for training...")
    device = "/CPU:0"

dataset.read_dataset_info()
dataset.filter_dataset_info()
dataset.parse_dataset_info()
print(dataset.dataset_info_df)