from __future__ import annotations

import os

import pandas as pd

from ml_circuits.constants import AMP_GATE
from ml_circuits.evaluation.metrics import metrics_one
from ml_circuits.visualization.error_plots import plot_error_vs_norm_freq, plot_rlc_resonance_families
from ml_circuits.features.rc_ladder_features import EXPERIMENT_RUNS, REQUIRED_COLS


def collect_model_metrics_table(
    *,
    order: list[str],
    main_models: dict,
    main_datasets: dict,
    baseline_models: dict,
    baseline_datasets: dict,
    amp_gate: float = AMP_GATE,
    group_weights: bool = True,
) -> pd.DataFrame:
    """
    Собирает метрики основной модели и baseline в одну таблицу.
    """
    rows = []

    for topo in order:
        if topo in main_models and topo in main_datasets:
            df_te, X_te, Y_te, meta_te, feat_names = main_datasets[topo]
            m = metrics_one(
                df_test=df_te,
                model=main_models[topo],
                X_test=X_te,
                Y_test=Y_te,
                meta_test=meta_te,
                amp_gate=amp_gate,
                use_group_weights=group_weights,
            )
            rows.append({
                "topology": topo,
                "model": "physics-informed GBDT",
                **m,
            })

        if topo in baseline_models and topo in baseline_datasets:
            df_te, X_te, Y_te, meta_te, feat_names = baseline_datasets[topo]
            m = metrics_one(
                df_test=df_te,
                model=baseline_models[topo],
                X_test=X_te,
                Y_test=Y_te,
                meta_test=meta_te,
                amp_gate=amp_gate,
                use_group_weights=group_weights,
            )
            rows.append({
                "topology": topo,
                "model": "sklearn_HGBR_baseline",
                **m,
            })

    out = pd.DataFrame(rows)

    metric_cols = [
        "A_MAE", "A_RMSE",
        "logH_MAE", "logH_RMSE",
        "dB_MAE", "dB_RMSE",
        "phi_MAE_deg", "phi_MAE_filtered_deg",
        "n",
    ]

    cols = ["topology", "model"] + [c for c in metric_cols if c in out.columns]
    return out[cols].sort_values(["topology", "model"]).reset_index(drop=True)


def make_thesis_metrics_table(metrics_compare_all: pd.DataFrame) -> pd.DataFrame:
    """
    Компактная таблица метрик для ВКР.
    """
    metrics_for_thesis = metrics_compare_all[
        [
            "topology",
            "model",
            "dB_MAE",
            "dB_RMSE",
            "phi_MAE_filtered_deg",
            "A_MAE",
            "n",
        ]
    ].copy()

    metrics_for_thesis["model"] = metrics_for_thesis["model"].replace({
        "custom_GBDT": "Основная модель",
        "sklearn_HGBR_baseline": "Sklearn HGBR baseline",
    })

    metrics_for_thesis = metrics_for_thesis.rename(columns={
        "topology": "Топология",
        "model": "Модель",
        "dB_MAE": "MAE АЧХ, дБ",
        "dB_RMSE": "RMSE АЧХ, дБ",
        "phi_MAE_filtered_deg": "MAE ФЧХ, град.",
        "A_MAE": "MAE амплитуды",
        "n": "Число точек",
    })

    return metrics_for_thesis


def print_metrics_one(d: dict, topo_label: str):
    """
    Печатает компактную строку с основными метриками для одной топологии.
    """
    print(
        f"{topo_label}: n={d['n']}, "
        f"A_MAE={d['A_MAE']:.4g}, A_RMSE={d['A_RMSE']:.4g}, "
        f"logH_MAE={d['logH_MAE']:.4g}, logH_RMSE={d['logH_RMSE']:.4g}, "
        f"dB_MAE={d['dB_MAE']:.4g}, dB_RMSE={d['dB_RMSE']:.4g}, "
        f"phi_MAE={d['phi_MAE_deg']:.3f}°, "
        f"phi_MAE_filtered={d['phi_MAE_filtered_deg']:.3f}°"
    )


def run_report_block(order, datasets_map, models_map, title_prefix, amp_gate=AMP_GATE):
    """
    Итоговый отчёт по топологиям.
    """
    present = [t for t in order if (t in datasets_map) and (t in models_map)]

    print("\n=====================")
    print(f"{title_prefix} metrics")
    print("=====================")

    for topo in present:
        df_te, X_te, Y_te, meta_te, feat_names = datasets_map[topo]
        model = models_map[topo]

        m = metrics_one(
            df_test=df_te,
            model=model,
            X_test=X_te,
            Y_test=Y_te,
            meta_test=meta_te,
            amp_gate=amp_gate,
            use_group_weights=True,
        )
        print_metrics_one(m, topo)

    print("\n=====================")
    print(f"{title_prefix} plots")
    print("=====================")

    for topo in present:
        df_te, X_te, Y_te, meta_te, feat_names = datasets_map[topo]
        model = models_map[topo]

        print(f"\n--- {title_prefix}: {topo} ---")

        plot_error_vs_norm_freq(
            df_test=df_te,
            X_test=X_te,
            model=model,
            meta_test=meta_te,
            topo=topo,
        )

        if topo in ("RLC_BP", "RLC_NOTCH"):
            plot_rlc_resonance_families(
                df_test=df_te,
                X_test=X_te,
                model=model,
                meta_test=meta_te,
                topo=topo,
                n_curves=1,
            )


def summarize_all_rc_ladder_runs() -> pd.DataFrame:
    rows = []

    for run in EXPERIMENT_RUNS:
        path = run["path"]
        mode = run["mode"]

        if not os.path.exists(path):
            print(f"[WARN] Файл не найден: {path}")
            continue

        df = pd.read_csv(path)

        if df.empty:
            print(f"[WARN] Пустой файл: {path}")
            continue

        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            print(f"[WARN] В файле {path} отсутствуют столбцы: {missing}")
            continue

        best_row = df.sort_values(
            by=[
                "score",
                "test_dB_MAE_group_weighted",
                "test_phi_MAE_filtered_deg_group_weighted",
            ]
        ).iloc[0]

        rows.append(
            {
                "mode": mode,
                "n_seeds": len(df),
                "best_seed": int(best_row["seed"]),
                "best_score": float(best_row["score"]),
                "best_test_dB_MAE": float(best_row["test_dB_MAE"]),
                "best_test_phi_MAE_filtered_deg": float(best_row["test_phi_MAE_filtered_deg"]),
                "best_test_phi_MAE_deg": float(best_row["test_phi_MAE_deg"]),
                "best_test_logH_MAE": float(best_row["test_logH_MAE"]),
                "mean_score": float(df["score"].mean()),
                "std_score": float(df["score"].std(ddof=0)),
                "mean_test_dB_MAE": float(df["test_dB_MAE"].mean()),
                "std_test_dB_MAE": float(df["test_dB_MAE"].std(ddof=0)),
                "mean_test_phi_MAE_filtered_deg": float(df["test_phi_MAE_filtered_deg"].mean()),
                "std_test_phi_MAE_filtered_deg": float(df["test_phi_MAE_filtered_deg"].std(ddof=0)),
                "mean_test_phi_MAE_deg": float(df["test_phi_MAE_deg"].mean()),
                "std_test_phi_MAE_deg": float(df["test_phi_MAE_deg"].std(ddof=0)),
                "mean_test_logH_MAE": float(df["test_logH_MAE"].mean()),
                "std_test_logH_MAE": float(df["test_logH_MAE"].std(ddof=0)),
                "best_test_dB_MAE_group_weighted": float(best_row["test_dB_MAE_group_weighted"]),
                "best_test_dB_RMSE_group_weighted": float(best_row["test_dB_RMSE_group_weighted"]),
                "best_test_phi_MAE_filtered_deg_group_weighted": float(best_row["test_phi_MAE_filtered_deg_group_weighted"]),
                "best_test_phi_MAE_deg_group_weighted": float(best_row["test_phi_MAE_deg_group_weighted"]),
                "best_test_logH_MAE_group_weighted": float(best_row["test_logH_MAE_group_weighted"]),
                "mean_test_dB_MAE_group_weighted": float(df["test_dB_MAE_group_weighted"].mean()),
                "std_test_dB_MAE_group_weighted": float(df["test_dB_MAE_group_weighted"].std(ddof=0)),
                "mean_test_phi_MAE_filtered_deg_group_weighted": float(df["test_phi_MAE_filtered_deg_group_weighted"].mean()),
                "std_test_phi_MAE_filtered_deg_group_weighted": float(df["test_phi_MAE_filtered_deg_group_weighted"].std(ddof=0)),
                "mean_test_logH_MAE_group_weighted": float(df["test_logH_MAE_group_weighted"].mean()),
                "std_test_logH_MAE_group_weighted": float(df["test_logH_MAE_group_weighted"].std(ddof=0)),
            }
        )

    out_df = pd.DataFrame(rows)

    if out_df.empty:
        print("Результаты не найдены.")
        return out_df

    out_df = out_df.sort_values(by=["mean_score", "best_score"]).reset_index(drop=True)
    return out_df


def summarize_all_runs() -> pd.DataFrame:
    return summarize_all_rc_ladder_runs()
