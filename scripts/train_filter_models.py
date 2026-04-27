from __future__ import annotations

from pathlib import Path
import random
import warnings

import numpy as np
import pandas as pd

from ml_circuits.config import FILTER_TRAINING_CONFIG as CONFIG
from ml_circuits.constants import AMP_GATE, EPS_LOG, SEED
from ml_circuits.evaluation.oracle_checks import run_analytical_oracle_check
from ml_circuits.evaluation.reports import (
    collect_model_metrics_table,
    make_thesis_metrics_table,
    run_report_block,
)
from ml_circuits.features.filter_features import FEATS_ORDER1, FEATS_RLC
from ml_circuits.features.targets import TARGETS_ORDER1, TARGETS_RLC
from ml_circuits.models.model_io import save_models
from ml_circuits.training.train_filters import (
    ORDER1_TOPOS,
    RLC_TOPOS,
    split_order1,
    split_rlc,
    train_one_topology,
    train_sklearn_baseline_one_topology,
)


warnings.filterwarnings("ignore")

np.set_printoptions(precision=6, suppress=True, linewidth=200)
pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)

random.seed(SEED)
np.random.seed(SEED)


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Базовые проверки согласованности dataset_ac.csv.
    """
    print(df.head())
    print("\n--- df.info() ---")
    df.info()

    print("\n--- Распределение по топологиям ---")
    print(df["topology"].value_counts(dropna=False))

    required_cols = [
        "param_set_id", "topology",
        "R", "C", "L", "Rload",
        "A_in", "f",
        "H_mag", "log_H_mag", "A_out",
        "phi_sin", "phi_cos",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    present_topos = set(df["topology"].unique())
    cfg_topos = set(CONFIG["topologies"])
    missing_topos = cfg_topos - present_topos
    if missing_topos:
        raise ValueError(f"Topologies from CONFIG not found in dataset: {sorted(missing_topos)}")

    if not df["param_set_id"].notna().all():
        raise ValueError("param_set_id contains NaN")

    if not (df["param_set_id"].astype(str).str.len() > 0).all():
        raise ValueError("param_set_id contains empty strings")

    if not (df["A_in"] > 0).all():
        raise ValueError("A_in must be > 0 everywhere")

    if not (df["f"] > 0).all():
        raise ValueError("f must be > 0 everywhere")

    n_groups_by_topo = df.groupby("topology")["param_set_id"].nunique().sort_values()
    print("\n--- Число уникальных param_set_id по топологиям ---")
    print(n_groups_by_topo)

    min_groups = 10
    if not (n_groups_by_topo >= min_groups).all():
        raise ValueError(f"Some topologies have fewer than {min_groups} groups")


def validate_targets_consistency(df: pd.DataFrame) -> None:
    """
    Проверки согласованности таргетов.
    """
    ratio = df["A_out"].to_numpy(float) / df["A_in"].to_numpy(float)
    H_mag = df["H_mag"].to_numpy(float)

    m1 = np.isfinite(ratio) & np.isfinite(H_mag) & (ratio > 0) & (H_mag > 0)

    rel_err = np.full_like(ratio, np.nan, dtype=float)
    rel_err[m1] = np.abs(H_mag[m1] - ratio[m1]) / np.maximum(ratio[m1], 1e-30)

    print("H_mag vs A_out/A_in:")
    print("  valid points:", int(m1.sum()), "/", len(df))
    print("  median rel err:", float(np.nanmedian(rel_err)))
    print("  99% rel err   :", float(np.nanquantile(rel_err, 0.99)))

    if not (np.nanmedian(rel_err) < 1e-9):
        raise ValueError("Median relative error too large for H_mag vs A_out/A_in")
    if not (np.nanquantile(rel_err, 0.99) < 1e-6):
        raise ValueError("99% relative error too large for H_mag vs A_out/A_in")

    log_H_mag = df["log_H_mag"].to_numpy(float)
    m2 = np.isfinite(log_H_mag) & np.isfinite(ratio) & (ratio > 0)

    log_ratio = np.full_like(ratio, np.nan, dtype=float)
    log_ratio[m2] = np.log(np.clip(ratio[m2], EPS_LOG, None))
    abs_err_log = np.full_like(ratio, np.nan, dtype=float)
    abs_err_log[m2] = np.abs(log_H_mag[m2] - log_ratio[m2])

    print("\nlog_H_mag vs log(max(A_out/A_in, EPS_LOG)):")
    print("  valid points:", int(m2.sum()), "/", len(df))
    print("  median abs err:", float(np.nanmedian(abs_err_log)))
    print("  99% abs err   :", float(np.nanquantile(abs_err_log, 0.99)))

    if not (np.nanmedian(abs_err_log) < 1e-9):
        raise ValueError("Median absolute error too large for log_H_mag consistency")
    if not (np.nanquantile(abs_err_log, 0.99) < 1e-6):
        raise ValueError("99% absolute error too large for log_H_mag consistency")

    phi_sin = df["phi_sin"].to_numpy(float)
    phi_cos = df["phi_cos"].to_numpy(float)

    m3 = np.isfinite(phi_sin) & np.isfinite(phi_cos)
    r2 = np.full_like(phi_sin, np.nan, dtype=float)
    r2[m3] = phi_sin[m3] ** 2 + phi_cos[m3] ** 2

    print("\nПроверка phi_sin^2 + phi_cos^2 (значение должно быть близко к 1):")
    print("  valid points:", int(m3.sum()), "/", len(df))
    print("  median:", float(np.nanmedian(r2)))
    print("  1%   :", float(np.nanquantile(r2, 0.01)))
    print("  99%  :", float(np.nanquantile(r2, 0.99)))
    print("  max  :", float(np.nanmax(r2)))

    if not (abs(float(np.nanmedian(r2)) - 1.0) < 1e-6):
        raise ValueError("Median sin^2+cos^2 deviates too much from 1")
    if not (float(np.nanquantile(r2, 0.99)) < 1.0001):
        raise ValueError("Upper tail too large for sin^2+cos^2")
    if not (float(np.nanquantile(r2, 0.01)) > 0.9999):
        raise ValueError("Lower tail too small for sin^2+cos^2")


def print_phase_gate_summary(df: pd.DataFrame) -> None:
    """
    Проверка точек с малой амплитудой для фазового обучения.
    """
    print(f"amp_gate = {AMP_GATE}")

    for topo, g in df.groupby("topology"):
        n_total = len(g)
        n_phase_bad = int((g["H_mag"] < AMP_GATE).sum())
        n_phase_ok = n_total - n_phase_bad

        print(
            f"{topo:10s} | "
            f"phase ok: {n_phase_ok:7d} / {n_total:7d} "
            f"({100.0 * n_phase_ok / n_total:6.2f}%) | "
            f"excluded: {n_phase_bad:7d}"
        )

    print("\nRLC_NOTCH near-zero amplitude examples:")
    print(
        df[
            (df["topology"] == "RLC_NOTCH")
            & (df["H_mag"] < AMP_GATE)
        ][
            ["param_set_id", "f", "f0", "Q_eff", "H_mag", "phi_sin", "phi_cos"]
        ]
        .sort_values("H_mag")
        .head(10)
    )


def main(run_plots: bool = False) -> None:
    dataset_path = Path(CONFIG["paths"]["dataset_csv"])
    df = pd.read_csv(dataset_path)

    validate_dataset(df)
    validate_targets_consistency(df)
    run_analytical_oracle_check(df)
    print_phase_gate_summary(df)

    models_order1, datasets_order1 = {}, {}
    for topo in ORDER1_TOPOS:
        model, pack = train_one_topology(
            df=df,
            topo=topo,
            split_fn=split_order1,
            FEATS=FEATS_ORDER1,
            TARGETS=TARGETS_ORDER1,
            amp_gate=AMP_GATE,
        )
        models_order1[topo] = model
        datasets_order1[topo] = pack
    save_models(models_order1, CONFIG["paths"]["models_dir"])

    models_rlc, datasets_rlc = {}, {}
    for topo in RLC_TOPOS:
        model, pack = train_one_topology(
            df=df,
            topo=topo,
            split_fn=split_rlc,
            FEATS=FEATS_RLC,
            TARGETS=TARGETS_RLC,
            amp_gate=AMP_GATE,
        )
        models_rlc[topo] = model
        datasets_rlc[topo] = pack
    save_models(models_rlc, CONFIG["paths"]["models_dir"])

    baseline_models_order1 = {}
    baseline_datasets_order1 = {}
    for topo in ORDER1_TOPOS:
        model, pack = train_sklearn_baseline_one_topology(
            df=df,
            topo=topo,
            split_fn=split_order1,
            FEATS=FEATS_ORDER1,
            TARGETS=TARGETS_ORDER1,
            amp_gate=AMP_GATE,
        )
        baseline_models_order1[topo] = model
        baseline_datasets_order1[topo] = pack

    baseline_models_rlc = {}
    baseline_datasets_rlc = {}
    for topo in RLC_TOPOS:
        model, pack = train_sklearn_baseline_one_topology(
            df=df,
            topo=topo,
            split_fn=split_rlc,
            FEATS=FEATS_RLC,
            TARGETS=TARGETS_RLC,
            amp_gate=AMP_GATE,
        )
        baseline_models_rlc[topo] = model
        baseline_datasets_rlc[topo] = pack

    metrics_compare_order1 = collect_model_metrics_table(
        order=ORDER1_TOPOS,
        main_models=models_order1,
        main_datasets=datasets_order1,
        baseline_models=baseline_models_order1,
        baseline_datasets=baseline_datasets_order1,
        amp_gate=AMP_GATE,
        group_weights=True,
    )

    metrics_compare_rlc = collect_model_metrics_table(
        order=RLC_TOPOS,
        main_models=models_rlc,
        main_datasets=datasets_rlc,
        baseline_models=baseline_models_rlc,
        baseline_datasets=baseline_datasets_rlc,
        amp_gate=AMP_GATE,
        group_weights=True,
    )

    metrics_compare_all = pd.concat(
        [metrics_compare_order1, metrics_compare_rlc],
        axis=0,
        ignore_index=True,
    )

    print(metrics_compare_all)

    metrics_dir = Path(CONFIG["paths"]["metrics_dir"])
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_compare_path = metrics_dir / "custom_vs_sklearn_hgbr_baseline.csv"
    metrics_compare_all.to_csv(metrics_compare_path, index=False)
    print(f"Saved baseline comparison: {metrics_compare_path}")

    metrics_for_thesis = make_thesis_metrics_table(metrics_compare_all)
    print(metrics_for_thesis)

    thesis_table_path = metrics_dir / "thesis_baseline_comparison.csv"
    metrics_for_thesis.to_csv(thesis_table_path, index=False)
    print(f"Saved thesis table: {thesis_table_path}")

    if run_plots:
        run_report_block(
            order=ORDER1_TOPOS,
            datasets_map=datasets_order1,
            models_map=models_order1,
            title_prefix="Order-1",
            amp_gate=AMP_GATE,
        )

        run_report_block(
            order=RLC_TOPOS,
            datasets_map=datasets_rlc,
            models_map=models_rlc,
            title_prefix="RLC",
            amp_gate=AMP_GATE,
        )


if __name__ == "__main__":
    main()
