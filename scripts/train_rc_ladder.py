from __future__ import annotations

import argparse
import os

from ml_circuits.config import PATHS
from ml_circuits.training.train_rc_ladder import run_rc_ladder_experiment


def resolve_data_path(path: str) -> str:
    if os.path.exists(path):
        return path
    config_path = PATHS["rc_ladder_dataset_jsonl"]
    if path == "dataset_rc_ladder.jsonl" and os.path.exists(config_path):
        return config_path
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train RC ladder pure ML experiments.")
    parser.add_argument("--data", default="dataset_rc_ladder.jsonl")
    parser.add_argument(
        "--mode",
        choices=["strict", "fewshot", "both"],
        default="both",
    )
    args = parser.parse_args()

    data_path = resolve_data_path(args.data)

    if args.mode in ("strict", "both"):
        run_rc_ladder_experiment(
            data_path=data_path,
            mode="strict",
            seeds=[40, 41, 42],
        )

    if args.mode in ("fewshot", "both"):
        run_rc_ladder_experiment(
            data_path=data_path,
            mode="fewshot",
            n4_train_fraction=0.04,
            seeds=[40, 41, 42],
        )


if __name__ == "__main__":
    main()
