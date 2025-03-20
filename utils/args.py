from argparse import ArgumentParser
from dataclasses import dataclass
from dotenv import load_dotenv
import os

if not load_dotenv(override=True): 
    print("[ENV] WARN: No envronment variables file has been found")

@dataclass
class TrainArgs:
    dataset_path: str
    out_path: str
    epochs: int
    batch_size: int
    learning_rate: float
    device: str
    model_path: str
    log_level: str
    seed: int
    sample_rate: int
    n_mels: int
    frame_ms: int
    sr_min: float
    sr_max: float
    ur_min: float


def train() -> TrainArgs:
    parser = ArgumentParser(
        prog="osu_bmg",
        description="Osu Beatmap Generator AI Training Script",
        epilog="© 2024 Artiom Cebotari. All rights reserved"
    )

    parser.add_argument(
        '-p', '--dataset-path',
        help="Path to the dataset folder",
        default=(os.getenv("DATASET_PATH") or "dataset"),
        type=str
    )

    parser.add_argument(
        '-op', '--out-path',
        help="The path where the files are going to be emitted",
        default=(os.getenv("TRAIN_OUTPUT_PATH") or "models"),
        type=str
    )

    parser.add_argument(
        '-e', '--epochs',
        help="Number of training epochs",
        default=int(os.getenv("EPOCS") or 50),
        type=int
    )

    parser.add_argument(
        '-bs', '--batch-size',
        help="Batch size for training",
        default=int(os.getenv("BATCH_SIZE") or 32),
        type=int
    )

    parser.add_argument(
        '-lr', '--learning-rate',
        help="Learning rate for optimization",
        default=float(os.getenv("LEARNING_RATE") or 0.001),
        type=float
    )

    parser.add_argument(
        '-d', '--device',
        help="Device to use for training (cpu or cuda)",
        choices=['cpu', 'cuda'],
        default=(os.getenv("DEVICE") or 'cuda')
    )

    parser.add_argument(
        '-mp', '--model-path',
        help="Path to a pre-trained model to continue training",
        default=(os.getenv("MODEL_PATH") or None),
        type=str
    )

    parser.add_argument(
        '-ll', '--log-level',
        help="Logging level (debug, info, warning, error, critical)",
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        default=(os.getenv("LOG_LEVEL") or "info")
    )

    parser.add_argument(
        '-s', '--seed',
        help="Random seed for reproducibility",
        default=int(os.getenv("SEED") or 42),
        type=int
    )

    # Audio Processing Arguments
    parser.add_argument(
        '-sr', '--sample-rate',
        help="Audio sample rate (Hz)",
        default=int(os.getenv("SAMPLE_RATE") or 22050),
        type=int
    )

    parser.add_argument(
        '-nm', '--n-mels',
        help="Number of Mel filters used in feature extraction",
        default=int(os.getenv("N_MELS") or 128),
        type=int
    )

    parser.add_argument(
        '-fm', '--frame-ms',
        help="Duration of each training sample in milliseconds",
        default=int(os.getenv("FRAME_MS") or 1000),
        type=int
    )

    parser.add_argument(
        '-min', '--minimum',
        help="Minimum star rating for maps to be filtered out",
        default=float(os.getenv("DATASET_STAR_RATING_MIN") or 0),
        type=float
    )

    parser.add_argument(
        '-max', '--maximum',
        help="Maximum star rating for maps to be filtered out",
        default=float(os.getenv("DATASET_STAR_RATING_MAX") or 999),
        type=float
    )


    parser.add_argument(
        '-minur', '--minimum-user-rating',
        help="The minimum user rating feedback for a map diffuclty",
        default=float(os.getenv("DATASET_USER_RATING_MIN") or 0),
        type=float
    )

    parsed_args, _ = parser.parse_known_args()

    print(parsed_args)

    return TrainArgs(
        dataset_path=parsed_args.dataset_path,
        out_path=parsed_args.out_path,
        epochs=parsed_args.epochs,
        batch_size=parsed_args.batch_size,
        learning_rate=parsed_args.learning_rate,
        device=parsed_args.device,
        model_path=parsed_args.model_path,
        log_level=parsed_args.log_level,
        seed=parsed_args.seed,
        sample_rate=parsed_args.sample_rate,
        n_mels=parsed_args.n_mels,
        frame_ms=parsed_args.frame_ms,
        sr_min=parsed_args.minimum,
        sr_max=parsed_args.maximum,
        ur_min=parsed_args.minimum_user_rating,
    )

@dataclass
class GenerateTraindataArgs:
    songs_path: str
    api_key: str
    out_path: str
    min: float
    max: float
    min_ur: float


def generate_dataset() -> GenerateTraindataArgs:
    parser = ArgumentParser(
        prog="osu_bmg",
        description="Osu dataset generator. Takes in a osu songs folder and outputs a dataset",
        epilog="© 2024 Artiom Cebotari. All rights reserved"
    )

    parser.add_argument(
        '-sp', '--songs-path',
        help="Path to the osu songs folder",
        default=os.getenv("OSU_SONGS_PATH"),
        type=str
    )

    parser.add_argument(
        '-t', "--token",
        help="Your osu v1 api token which you can get from https://osu.ppy.sh/p/api",
        default=os.getenv("OSU_API_KEY"),
        type=str
    )

    parser.add_argument(
        '-o', '--out',
        help="Output path for the dataset",
        default=(os.getenv("DATASET_OUTPUT_PATH") or "dataset"),
        type=str
    )

    parser.add_argument(
        '-min', '--minimum',
        help="Minimum star rating for maps to be filtered out",
        default=float(os.getenv("DATASET_STAR_RATING_MIN") or 0),
        type=float
    )

    parser.add_argument(
        '-max', '--maximum',
        help="Maximum star rating for maps to be filtered out",
        default=float(os.getenv("DATASET_STAR_RATING_MAX") or 999),
        type=float
    )

    parser.add_argument(
        '-minur', '--minimum-user-rating',
        help="The minimum user rating feedback for a map diffuclty",
        default=float(os.getenv("DATASET_USER_RATING_MIN") or 0),
        type=float
    )

    parsed_args, _ = parser.parse_known_args()

    return GenerateTraindataArgs(
        songs_path=parsed_args.songs_path,
        api_key=parsed_args.token,
        out_path=parsed_args.out,
        min=parsed_args.minimum,
        max=parsed_args.maximum,
        min_ur=parsed_args.minimum_user_rating
    )