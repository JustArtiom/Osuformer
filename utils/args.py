from argparse import ArgumentParser
from dataclasses import dataclass


@dataclass
class Args:
    train_path: str
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


def parse() -> Args:
    parser = ArgumentParser(
        prog="osu_bmg",
        description="Osu Beatmap Generator AI Training Script",
        epilog="© 2024 Artiom Cebotari. All rights reserved"
    )

    parser.add_argument(
        '-tp', '--train-path',
        help="Path to the training folder",
        default="train",
        type=str
    )

    parser.add_argument(
        '-op', '--out-path',
        help="The path where the files are going to be emitted",
        default="out",
        type=str
    )

    parser.add_argument(
        '-e', '--epochs',
        help="Number of training epochs",
        default=50,
        type=int
    )

    parser.add_argument(
        '-bs', '--batch-size',
        help="Batch size for training",
        default=32,
        type=int
    )

    parser.add_argument(
        '-lr', '--learning-rate',
        help="Learning rate for optimization",
        default=0.001,
        type=float
    )

    parser.add_argument(
        '-d', '--device',
        help="Device to use for training (cpu or cuda)",
        choices=['cpu', 'cuda'],
        default='cuda'
    )

    parser.add_argument(
        '-mp', '--model-path',
        help="Path to a pre-trained model to continue training",
        default=None,
        type=str
    )

    parser.add_argument(
        '-ll', '--log-level',
        help="Logging level (debug, info, warning, error, critical)",
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        default="info"
    )

    parser.add_argument(
        '-s', '--seed',
        help="Random seed for reproducibility",
        default=42,
        type=int
    )

    # Audio Processing Arguments
    parser.add_argument(
        '-sr', '--sample-rate',
        help="Audio sample rate (Hz)",
        default=22050,
        type=int
    )

    parser.add_argument(
        '-nm', '--n-mels',
        help="Number of Mel filters used in feature extraction",
        default=128,
        type=int
    )

    parser.add_argument(
        '-fm', '--frame-ms',
        help="Duration of each training sample in milliseconds",
        default=1000,
        type=int
    )

    parsed_args, _ = parser.parse_known_args()

    return Args(
        train_path=parsed_args.train_path,
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
        frame_ms=parsed_args.frame_ms
    )


if __name__ == "__main__":
    args = parse()
    print(args)
