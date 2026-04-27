from __future__ import annotations

import argparse
import json
import os

import pandas as pd
import torch

from ml_circuits.config import PATHS
from ml_circuits.features.rc_ladder_features import load_jsonl, parse_sections_cell
from ml_circuits.inverse.rc_ladder_inverse import (
    build_inverse_results_table,
    build_param_table_from_main_dataset,
    run_full_inverse_evaluation,
)
from ml_circuits.models.rc_ladder_nn import RCLadderPureMLModel


def resolve_main_dataset_path(path: str) -> str:
    if os.path.exists(path):
        return path
    config_path = PATHS["rc_ladder_dataset_jsonl"]
    if path == "dataset_rc_ladder.jsonl" and os.path.exists(config_path):
        return config_path
    return path


def load_best_model_and_stats(artifact_dir: str, device: torch.device):
    metrics_json_path = os.path.join(artifact_dir, "metrics_best_seed.json")
    with open(metrics_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    config = payload["config"]
    best_seed = int(payload["best_seed"])
    model_path = os.path.join(artifact_dir, f"best_model_seed_{best_seed}.pt")

    model = RCLadderPureMLModel(
        seq_in_dim=7,
        global_dim=13,
        sec_hidden=config["sec_hidden"],
        global_hidden=config["global_hidden"],
        state_dim=config["state_dim"],
        head_dim=config["head_dim"],
        phase_head_dim=config["phase_head_dim"],
        dropout=config["dropout"],
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    return model, payload["stats"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RC ladder inverse evaluation.")
    parser.add_argument("--artifact-dir", default="artifacts_rc_ladder_pure_ml_fewshot_n4_4pct")
    parser.add_argument("--eval123-params", default="rc_ladder_param_sets_eval123.csv")
    parser.add_argument("--eval123-data", default="dataset_rc_ladder_eval123.jsonl")
    parser.add_argument("--main-data", default="dataset_rc_ladder.jsonl")
    parser.add_argument("--out-dir", default="artifacts_inverse_rc_ladder_eval123_plus_n4")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model, best_stats = load_best_model_and_stats(args.artifact_dir, device=device)

    df_eval123_params = pd.read_csv(args.eval123_params)
    df_eval123_params["sections"] = df_eval123_params["sections"].apply(parse_sections_cell)

    df_eval123_dataset = load_jsonl(args.eval123_data)
    df_main_dataset = load_jsonl(resolve_main_dataset_path(args.main_data))
    df_main_params = build_param_table_from_main_dataset(df_main_dataset)

    inverse_eval = run_full_inverse_evaluation(
        model=best_model,
        stats=best_stats,
        df_eval123_params=df_eval123_params,
        df_eval123_dataset=df_eval123_dataset,
        df_main_params=df_main_params,
        df_main_dataset=df_main_dataset,
        out_dir=args.out_dir,
        n_examples_per_group=3,
        n_freq_per_example=10,
        max_sections=4,
        adam_steps=3000,
        lbfgs_steps=120,
        lr_adam=0.03,
        lr_lbfgs=0.5,
        use_lbfgs=True,
        use_multistart=True,
        noise_scales=(0.05, 0.10, 0.20),
        print_every=1000,
        seed=42,
    )
    results_table = build_inverse_results_table(
        inverse_eval,
        save_path=os.path.join(args.out_dir, "results_table.csv"),
    )

    print(results_table)


if __name__ == "__main__":
    main()
