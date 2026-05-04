from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml_circuits.evaluation.rc_ladder_metrics import (
    combined_score_group_weighted,
    compute_loss,
    evaluate_metrics,
    evaluate_predictions_df_group_weighted,
    format_metrics,
    predict_loader_to_df,
)
from ml_circuits.features.rc_ladder_features import (
    RCLadderPureMLDataset,
    fit_feature_stats,
    json_default,
    load_jsonl,
    make_artifact_dir,
    sanity_checks,
    set_seed,
)
from ml_circuits.models.rc_ladder_nn import RCLadderPureMLModel


def split_strict_blackbox(
    df: pd.DataFrame,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Формирует train/val/test-разбиение для режима строгой экстраполяции.
    """
    rng = np.random.default_rng(seed)

    df_trainval = df[df["n_sections"].isin([1, 2, 3])].copy()
    df_test = df[df["n_sections"] == 4].copy()

    train_ids: set = set()
    val_ids: set = set()

    print("=== SPLIT: STRICT BLACK-BOX EXTRAPOLATION ===")
    print("train = n_sections in {1,2,3}")
    print("val   = n_sections in {1,2,3}")
    print("test  = only n_sections = 4")
    print()

    print("=== GROUP SPLIT WITH SEPARATE PROCESSING BY n_sections ===")
    for n in [1, 2, 3]:
        sub = df_trainval[df_trainval["n_sections"] == n]
        unique_ids = sub["param_set_id"].drop_duplicates().to_list()
        rng.shuffle(unique_ids)

        n_val_ids = int(round(len(unique_ids) * val_fraction))
        n_val_ids = min(max(1, n_val_ids), len(unique_ids) - 1)
        cur_val_ids = set(unique_ids[:n_val_ids])
        cur_train_ids = set(unique_ids[n_val_ids:])

        train_ids |= cur_train_ids
        val_ids |= cur_val_ids

        print(
            f"  n={n}: total_groups={len(unique_ids)}, "
            f"train_groups={len(cur_train_ids)}, val_groups={len(cur_val_ids)}"
        )

    overlap = train_ids & val_ids
    if overlap:
        raise RuntimeError(f"Leakage detected: {len(overlap)} overlapping param_set_id")

    train_df = df_trainval[df_trainval["param_set_id"].isin(train_ids)].copy()
    val_df = df_trainval[df_trainval["param_set_id"].isin(val_ids)].copy()

    print()
    print(f"train rows: {len(train_df)}")
    print(f"val rows:   {len(val_df)}")
    print(f"test rows:  {len(df_test)}")
    print("test = only n_sections=4, train/val = only n_sections in {1,2,3}")

    def counts_by_n(x: pd.DataFrame) -> Dict[int, int]:
        return x.groupby("n_sections")["param_set_id"].nunique().sort_index().to_dict()

    print(f"train groups by n_sections: {counts_by_n(train_df)}")
    print(f"val groups by n_sections:   {counts_by_n(val_df)}")
    print(f"test groups by n_sections:  {counts_by_n(df_test)}")
    print()

    return train_df, val_df, df_test


def split_with_fewshot_n4(
    df: pd.DataFrame,
    val_fraction: float = 0.15,
    n4_train_fraction: float = 0.05,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Формирует train/val/test-разбиение для few-shot-режима.
    """
    if not (0.0 < n4_train_fraction < 1.0):
        raise ValueError("n4_train_fraction must be in (0, 1)")

    rng = np.random.default_rng(seed)

    df_123 = df[df["n_sections"].isin([1, 2, 3])].copy()
    df_4 = df[df["n_sections"] == 4].copy()

    train_ids_123: set = set()
    val_ids_123: set = set()

    print("=== SPLIT: FEW-SHOT N4 ===")
    print("train = n_sections in {1,2,3} + small share of n=4 groups")
    print("val   = only n_sections in {1,2,3}")
    print("test  = remaining n=4 groups")
    print()

    print("=== GROUP SPLIT WITH SEPARATE PROCESSING BY n_sections ===")
    for n in [1, 2, 3]:
        sub = df_123[df_123["n_sections"] == n]
        unique_ids = sub["param_set_id"].drop_duplicates().to_list()
        rng.shuffle(unique_ids)

        n_val_ids = int(round(len(unique_ids) * val_fraction))
        n_val_ids = min(max(1, n_val_ids), len(unique_ids) - 1)
        cur_val_ids = set(unique_ids[:n_val_ids])
        cur_train_ids = set(unique_ids[n_val_ids:])

        train_ids_123 |= cur_train_ids
        val_ids_123 |= cur_val_ids

        print(
            f"  n={n}: total_groups={len(unique_ids)}, "
            f"train_groups={len(cur_train_ids)}, val_groups={len(cur_val_ids)}"
        )

    overlap_123 = train_ids_123 & val_ids_123
    if overlap_123:
        raise RuntimeError(
            f"Leakage detected in n=1..3 split: {len(overlap_123)} overlapping param_set_id"
        )

    n4_unique_ids = df_4["param_set_id"].drop_duplicates().to_list()
    rng.shuffle(n4_unique_ids)

    n4_train_groups = int(round(len(n4_unique_ids) * n4_train_fraction))
    n4_train_groups = min(max(1, n4_train_groups), len(n4_unique_ids) - 1)
    n4_train_ids = set(n4_unique_ids[:n4_train_groups])
    n4_test_ids = set(n4_unique_ids[n4_train_groups:])

    overlap_4 = n4_train_ids & n4_test_ids
    if overlap_4:
        raise RuntimeError(
            f"Leakage detected in n=4 split: {len(overlap_4)} overlapping param_set_id"
        )

    train_df_123 = df_123[df_123["param_set_id"].isin(train_ids_123)].copy()
    val_df = df_123[df_123["param_set_id"].isin(val_ids_123)].copy()

    train_df_4 = df_4[df_4["param_set_id"].isin(n4_train_ids)].copy()
    test_df = df_4[df_4["param_set_id"].isin(n4_test_ids)].copy()

    train_df = pd.concat([train_df_123, train_df_4], axis=0, ignore_index=True)

    print()
    print(f"n4 few-shot groups added to train: {len(n4_train_ids)} / {len(n4_unique_ids)}")
    print(f"n4 train fraction actual: {len(n4_train_ids) / max(len(n4_unique_ids), 1):.4f}")
    print()

    print(f"train rows: {len(train_df)}")
    print(f"val rows:   {len(val_df)}")
    print(f"test rows:  {len(test_df)}")

    def counts_by_n(x: pd.DataFrame) -> Dict[int, int]:
        return x.groupby("n_sections")["param_set_id"].nunique().sort_index().to_dict()

    print(f"train groups by n_sections: {counts_by_n(train_df)}")
    print(f"val groups by n_sections:   {counts_by_n(val_df)}")
    print(f"test groups by n_sections:  {counts_by_n(test_df)}")
    print()

    return train_df, val_df, test_df


def split_experiment(
    df: pd.DataFrame,
    mode: str,
    val_fraction: float = 0.15,
    n4_train_fraction: float | None = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Выбирает схему разбиения данных в соответствии с режимом эксперимента.
    """
    if mode == "strict":
        return split_strict_blackbox(
            df=df,
            val_fraction=val_fraction,
            seed=seed,
        )

    if mode == "fewshot":
        if n4_train_fraction is None:
            raise ValueError("n4_train_fraction must be set for fewshot mode")
        return split_with_fewshot_n4(
            df=df,
            val_fraction=val_fraction,
            n4_train_fraction=n4_train_fraction,
            seed=seed,
        )

    raise ValueError(f"Unknown mode: {mode}")


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    phase_weight_min: float,
    phase_weight_power: float,
    phase_loss_scale: float,
    angular_loss_scale: float,
) -> float:
    """
    Выполняет одну эпоху обучения или валидации.
    """
    train_mode = optimizer is not None
    model.train(train_mode)
    losses = []

    for batch in loader:
        seq_feat = batch["seq_feat"].to(device)
        seq_len = batch["seq_len"].to(device)
        global_feat = batch["global_feat"].to(device)
        y = batch["y"].to(device)

        if train_mode:
            pred = model(seq_feat=seq_feat, seq_len=seq_len, global_feat=global_feat)
            loss = compute_loss(
                pred,
                y_true=y,
                phase_weight_min=phase_weight_min,
                phase_weight_power=phase_weight_power,
                phase_loss_scale=phase_loss_scale,
                angular_loss_scale=angular_loss_scale,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        else:
            with torch.no_grad():
                pred = model(
                    seq_feat=seq_feat,
                    seq_len=seq_len,
                    global_feat=global_feat,
                )
            loss = compute_loss(
                pred,
                y_true=y,
                phase_weight_min=phase_weight_min,
                phase_weight_power=phase_weight_power,
                phase_loss_scale=phase_loss_scale,
                angular_loss_scale=angular_loss_scale,
            )

        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses))


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float = 8e-4,
    weight_decay: float = 1e-5,
    epochs: int = 150,
    patience: int = 20,
    phase_weight_min: float = 0.02,
    phase_weight_power: float = 0.7,
    phase_loss_scale: float = 0.22,
    angular_loss_scale: float = 0.10,
) -> Tuple[nn.Module, Dict[str, List[float]], int]:
    """
    Обучает модель с использованием ранней остановки по значению val_loss.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        train_loss = run_epoch(
            model, train_loader, optimizer, device,
            phase_weight_min, phase_weight_power, phase_loss_scale, angular_loss_scale,
        )
        val_loss = run_epoch(
            model, val_loader, None, device,
            phase_weight_min, phase_weight_power, phase_loss_scale, angular_loss_scale,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch={best_epoch}, best val={best_val:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_epoch


def train_one_seed(
    df: pd.DataFrame,
    args,
    seed: int,
    device: torch.device,
) -> Dict[str, object]:
    """
    Выполняет полный цикл эксперимента для одного значения seed.
    """
    print(f"\n{'=' * 18} SEED {seed} {'=' * 18}")
    set_seed(seed)

    train_df, val_df, test_df = split_experiment(
        df=df,
        mode=args.mode,
        val_fraction=args.val_fraction,
        n4_train_fraction=args.n4_train_fraction,
        seed=seed,
    )

    stats = fit_feature_stats(train_df)

    train_ds = RCLadderPureMLDataset(train_df, stats=stats, max_sections=args.max_sections)
    val_ds = RCLadderPureMLDataset(val_df, stats=stats, max_sections=args.max_sections)
    test_ds = RCLadderPureMLDataset(test_df, stats=stats, max_sections=args.max_sections)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = RCLadderPureMLModel(
        seq_in_dim=7,
        global_dim=13,
        sec_hidden=args.sec_hidden,
        global_hidden=args.global_hidden,
        state_dim=args.state_dim,
        head_dim=args.head_dim,
        phase_head_dim=args.phase_head_dim,
        dropout=args.dropout,
    ).to(device)

    model, history, best_epoch = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
        phase_weight_min=args.phase_weight_min,
        phase_weight_power=args.phase_weight_power,
        phase_loss_scale=args.phase_loss_scale,
        angular_loss_scale=args.angular_loss_scale,
    )

    print(f"Best epoch restored: {best_epoch}")
    print("=== METRICS ===")
    train_metrics = evaluate_metrics(model, train_loader, device)
    val_metrics = evaluate_metrics(model, val_loader, device)

    format_metrics("train", train_metrics)
    format_metrics("val", val_metrics)

    predictions_val_df = predict_loader_to_df(model, val_loader, device)
    val_metrics_group_weighted = evaluate_predictions_df_group_weighted(predictions_val_df)

    return {
        "seed": seed,
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "stats": stats,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "val_metrics_group_weighted": val_metrics_group_weighted,
        "pred_val_df": predictions_val_df,
        "test_loader": test_loader,
        "selection_score": combined_score_group_weighted(val_metrics_group_weighted),
    }


def run_rc_ladder_experiment(
    data_path: str,
    mode: str = "strict",
    n4_train_fraction: float | None = None,
    seeds: List[int] | None = None,
    batch_size: int = 256,
    sec_hidden: int = 96,
    global_hidden: int = 64,
    state_dim: int = 128,
    head_dim: int = 128,
    phase_head_dim: int = 64,
    dropout: float = 0.10,
    lr: float = 8e-4,
    weight_decay: float = 1e-5,
    epochs: int = 150,
    patience: int = 20,
    val_fraction: float = 0.15,
    max_sections: int = 4,
    phase_weight_min: float = 0.02,
    phase_weight_power: float = 0.70,
    phase_loss_scale: float = 0.22,
    angular_loss_scale: float = 0.10,
) -> Dict[str, object]:
    seeds = seeds or [42]

    args = SimpleNamespace(
        data=data_path,
        mode=mode,
        n4_train_fraction=n4_train_fraction,
        batch_size=batch_size,
        sec_hidden=sec_hidden,
        global_hidden=global_hidden,
        state_dim=state_dim,
        head_dim=head_dim,
        phase_head_dim=phase_head_dim,
        dropout=dropout,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        patience=patience,
        val_fraction=val_fraction,
        max_sections=max_sections,
        phase_weight_min=phase_weight_min,
        phase_weight_power=phase_weight_power,
        phase_loss_scale=phase_loss_scale,
        angular_loss_scale=angular_loss_scale,
        seeds=seeds,
    )

    if args.mode == "fewshot":
        if args.n4_train_fraction is None:
            raise ValueError("n4_train_fraction must be set in fewshot mode")
        if not (0.0 < args.n4_train_fraction < 1.0):
            raise ValueError("n4_train_fraction must be in (0, 1)")
    else:
        args.n4_train_fraction = None

    artifact_dir = make_artifact_dir(args.mode, args.n4_train_fraction)
    os.makedirs(artifact_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"mode: {args.mode}")
    if args.mode == "fewshot":
        print(f"n4_train_fraction: {args.n4_train_fraction}")

    df = load_jsonl(args.data)
    sanity_checks(df)

    results = []
    for seed in args.seeds:
        result = train_one_seed(df=df, args=args, seed=seed, device=device)
        results.append(result)

    leaderboard_rows = []
    for result in results:
        val_metrics = result["val_metrics"]
        val_metrics_gw = result["val_metrics_group_weighted"]

        leaderboard_rows.append(
            {
                "seed": result["seed"],
                # score оставлен как backward-compatible alias для validation selection score.
                "score": result["selection_score"],
                "selection_score": result["selection_score"],
                "selection_metric": "validation_group_weighted",
                "val_dB_MAE": val_metrics["dB_MAE"],
                "val_phi_MAE_filtered_deg": val_metrics["phi_MAE_filtered_deg"],
                "val_phi_MAE_deg": val_metrics["phi_MAE_deg"],
                "val_logH_MAE": val_metrics["logH_MAE"],
                "val_dB_MAE_group_weighted": val_metrics_gw["dB_MAE_group_weighted"],
                "val_dB_RMSE_group_weighted": val_metrics_gw["dB_RMSE_group_weighted"],
                "val_phi_MAE_filtered_deg_group_weighted": val_metrics_gw["phi_MAE_filtered_deg_group_weighted"],
                "val_phi_MAE_deg_group_weighted": val_metrics_gw["phi_MAE_deg_group_weighted"],
                "val_logH_MAE_group_weighted": val_metrics_gw["logH_MAE_group_weighted"],
            }
        )

    leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values(
        by=[
            "selection_score",
            "val_dB_MAE_group_weighted",
            "val_phi_MAE_filtered_deg_group_weighted",
        ]
    ).reset_index(drop=True)

    print("=== LEADERBOARD BY VALIDATION SCORE (lower is better) ===")
    print(leaderboard_df.to_string(index=False))

    best_seed = int(leaderboard_df.iloc[0]["seed"])
    best_result = next(result for result in results if result["seed"] == best_seed)

    test_name = "test_extrapolation_n4" if args.mode == "strict" else "test_n4_remaining"
    print("=== FINAL TEST METRICS FOR VALIDATION-SELECTED SEED ===")
    test_metrics = evaluate_metrics(best_result["model"], best_result["test_loader"], device)
    format_metrics(test_name, test_metrics)
    predictions_test_df = predict_loader_to_df(
        best_result["model"],
        best_result["test_loader"],
        device,
    )
    test_metrics_group_weighted = evaluate_predictions_df_group_weighted(predictions_test_df)

    test_columns = {
        "test_dB_MAE": test_metrics["dB_MAE"],
        "test_phi_MAE_filtered_deg": test_metrics["phi_MAE_filtered_deg"],
        "test_phi_MAE_deg": test_metrics["phi_MAE_deg"],
        "test_logH_MAE": test_metrics["logH_MAE"],
        "test_dB_MAE_group_weighted": test_metrics_group_weighted["dB_MAE_group_weighted"],
        "test_dB_RMSE_group_weighted": test_metrics_group_weighted["dB_RMSE_group_weighted"],
        "test_phi_MAE_filtered_deg_group_weighted": test_metrics_group_weighted["phi_MAE_filtered_deg_group_weighted"],
        "test_phi_MAE_deg_group_weighted": test_metrics_group_weighted["phi_MAE_deg_group_weighted"],
        "test_logH_MAE_group_weighted": test_metrics_group_weighted["logH_MAE_group_weighted"],
    }
    for column in test_columns:
        leaderboard_df[column] = np.nan
    selected_mask = leaderboard_df["seed"] == best_seed
    for column, value in test_columns.items():
        leaderboard_df.loc[selected_mask, column] = value

    leaderboard_rows = leaderboard_df.to_dict(orient="records")
    best_result["test_metrics"] = test_metrics
    best_result["test_metrics_group_weighted"] = test_metrics_group_weighted
    best_result["pred_test_df"] = predictions_test_df

    model_path = os.path.join(artifact_dir, f"best_model_seed_{best_seed}.pt")
    torch.save(best_result["model"].state_dict(), model_path)

    predictions_test_path = os.path.join(artifact_dir, f"predictions_test_n4_seed_{best_seed}.csv")
    best_result["pred_test_df"].to_csv(predictions_test_path, index=False)

    leaderboard_path = os.path.join(artifact_dir, "leaderboard_seeds.csv")
    leaderboard_df.to_csv(leaderboard_path, index=False)

    metrics_path = os.path.join(artifact_dir, "metrics_best_seed.json")
    test_key = "test_extrapolation_n4" if args.mode == "strict" else "test_n4_remaining"

    payload = {
        "best_seed": best_seed,
        "best_seed_selection_rule": (
            "min validation group-weighted score; test metrics are reported only "
            "for the selected seed"
        ),
        "leaderboard": leaderboard_rows,
        "best_epoch": best_result["best_epoch"],
        "train": best_result["train_metrics"],
        "val": best_result["val_metrics"],
        "val_group_weighted": best_result["val_metrics_group_weighted"],
        "history": best_result["history"],
        "config": vars(args),
        "stats": best_result["stats"],
    }
    payload[test_key] = best_result["test_metrics"]
    payload[test_key + "_group_weighted"] = best_result["test_metrics_group_weighted"]

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )

    print(f"Saved best model to: {model_path}")
    print(f"Saved best predictions to: {predictions_test_path}")
    print(f"Saved leaderboard to: {leaderboard_path}")
    print(f"Saved best metrics to: {metrics_path}")

    return {
        "artifact_dir": artifact_dir,
        "leaderboard_df": leaderboard_df,
        "best_result": best_result,
        "metrics_path": metrics_path,
        "predictions_test_path": predictions_test_path,
        "model_path": model_path,
    }
