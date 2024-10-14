from argparse import ArgumentParser
from dataclasses import dataclass


@dataclass
class Args:
    train_path: str
    out_path: str


def parse() -> Args:
    parser = ArgumentParser(
        prog="osu_bmg",
        description="Osu Beatmap generator AI",
        epilog="© 2024 Artiom Cebotari. All rights reserved"
    )

    parser.add_argument(
        '-tp', '--train-path',
        help="Path to the training folder",
        default="./train",
        type=str,
    )

    parser.add_argument(
        '-op', '--out-path',
        help="The path where the files are going to be emitted",
        default="./out",
        type=str
    )

    parsed_args, _ = parser.parse_known_args()

    return Args(
        train_path=parsed_args.train_path,
        out_path=parsed_args.out_path
    )
