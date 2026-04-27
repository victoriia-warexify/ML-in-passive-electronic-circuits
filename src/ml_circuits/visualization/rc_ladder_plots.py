from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_circuits.features.rc_ladder_features import wrap_angle_deg_np


def ensure_dir(path: str) -> None:
    """
    Создаёт каталог, если он отсутствует.
    """
    os.makedirs(path, exist_ok=True)


def load_predictions(predictions_csv_path: str) -> pd.DataFrame:
    """
    Загружает таблицу предсказанных и истинных характеристик.
    """
    df = pd.read_csv(predictions_csv_path)

    required_cols = {
        "param_set_id",
        "frequency_hz",
        "n_sections",
        "H_true",
        "H_pred",
        "dB_true",
        "dB_pred",
        "phi_true_deg",
        "phi_pred_deg",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in predictions CSV: {sorted(missing)}")

    return df


def load_metrics(metrics_json_path: str) -> Dict[str, object]:
    """
    Загружает JSON-файл с метриками и конфигурацией лучшего запуска.
    """
    import json

    with open(metrics_json_path, "r", encoding="utf-8") as file:
        return json.load(file)


def resolve_predictions_csv_path(artifact_dir: str, explicit_path: str | None) -> str:
    """
    Определяет путь к CSV-файлу с предсказаниями модели.
    """
    if explicit_path is not None:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"Predictions CSV not found: {explicit_path}")
        return explicit_path

    if not os.path.isdir(artifact_dir):
        raise FileNotFoundError(f"Artifact directory not found: {artifact_dir}")

    candidate_names = sorted(
        file_name
        for file_name in os.listdir(artifact_dir)
        if file_name.startswith("predictions_test_n4_seed_") and file_name.endswith(".csv")
    )
    if not candidate_names:
        raise FileNotFoundError(
            f"No predictions_test_n4_seed_*.csv found in artifact directory: {artifact_dir}"
        )

    return os.path.join(artifact_dir, candidate_names[0])


def resolve_metrics_json_path(artifact_dir: str, explicit_path: str | None) -> str:
    """
    Определяет путь к JSON-файлу с метриками лучшего запуска.
    """
    if explicit_path is not None:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"Metrics JSON not found: {explicit_path}")
        return explicit_path

    default_path = os.path.join(artifact_dir, "metrics_best_seed.json")
    if not os.path.exists(default_path):
        raise FileNotFoundError(f"Metrics JSON not found: {default_path}")
    return default_path


def build_group_summary(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Формирует сводную таблицу ошибок по группам param_set_id.
    """
    work_df = predictions_df.copy()
    work_df["dB_abs_err"] = np.abs(work_df["dB_pred"] - work_df["dB_true"])
    work_df["phi_abs_err_deg"] = np.abs(
        wrap_angle_deg_np(
            work_df["phi_pred_deg"].to_numpy(dtype=float)
            - work_df["phi_true_deg"].to_numpy(dtype=float)
        )
    )

    summary_df = (
        work_df.groupby("param_set_id", as_index=False)
        .agg(
            n_sections=("n_sections", "first"),
            n_points=("frequency_hz", "size"),
            dB_abs_err_median=("dB_abs_err", "median"),
            phi_abs_err_deg_median=("phi_abs_err_deg", "median"),
        )
        .sort_values(["dB_abs_err_median", "phi_abs_err_deg_median", "param_set_id"])
        .reset_index(drop=True)
    )
    return summary_df


def choose_representative_groups(
    predictions_df: pd.DataFrame,
    num_groups: int,
) -> List[str]:
    """
    Выбирает несколько характерных групп param_set_id для построения частотных характеристик.
    """
    summary_df = build_group_summary(predictions_df)

    if summary_df.empty:
        return []

    if len(summary_df) <= num_groups:
        return summary_df["param_set_id"].tolist()

    indices = np.linspace(0, len(summary_df) - 1, num_groups, dtype=int)
    selected = summary_df.iloc[indices]["param_set_id"].tolist()
    return selected


def plot_bode_examples(
    predictions_df: pd.DataFrame,
    output_dir: str,
    num_groups: int = 3,
) -> None:
    """
    Строит несколько характерных Bode-графиков для отдельных RC-цепочек.
    """
    selected_group_ids = choose_representative_groups(predictions_df, num_groups=num_groups)

    for plot_idx, group_id in enumerate(selected_group_ids, start=1):
        group_df = predictions_df[predictions_df["param_set_id"] == group_id].copy()
        group_df = group_df.sort_values("frequency_hz").reset_index(drop=True)

        fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)

        axes[0].plot(group_df["frequency_hz"], group_df["dB_true"], label="Истинная АЧХ")
        axes[0].plot(group_df["frequency_hz"], group_df["dB_pred"], label="Предсказанная АЧХ")
        axes[0].set_xscale("log")
        axes[0].set_ylabel("|H|, dB")
        axes[0].grid(True, which="both", alpha=0.3)
        axes[0].legend()
        axes[0].set_title(
            f"RC-цепочка: частотный отклик, param_set_id={group_id}, "
            f"n_sections={int(group_df['n_sections'].iloc[0])}"
        )

        axes[1].plot(group_df["frequency_hz"], group_df["phi_true_deg"], label="Истинная ФЧХ")
        axes[1].plot(group_df["frequency_hz"], group_df["phi_pred_deg"], label="Предсказанная ФЧХ")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Частота, Гц")
        axes[1].set_ylabel("Фаза, град")
        axes[1].grid(True, which="both", alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        output_path = os.path.join(output_dir, f"bode_example_{plot_idx:02d}_{group_id}.png")
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_error_vs_frequency(
    predictions_df: pd.DataFrame,
    output_dir: str,
    num_log_bins: int = 40,
) -> None:
    """
    Строит зависимости медианной абсолютной ошибки модели от частоты.
    """
    work_df = predictions_df.copy()

    work_df["dB_abs_err"] = np.abs(work_df["dB_pred"] - work_df["dB_true"])
    work_df["phi_abs_err_deg"] = np.abs(
        wrap_angle_deg_np(
            work_df["phi_pred_deg"].to_numpy(dtype=float)
            - work_df["phi_true_deg"].to_numpy(dtype=float)
        )
    )

    freq_values = work_df["frequency_hz"].to_numpy(dtype=float)
    positive_freq = freq_values[freq_values > 0.0]
    if positive_freq.size == 0:
        raise ValueError("All frequency values must be positive.")

    f_min = float(np.min(positive_freq))
    f_max = float(np.max(positive_freq))

    if np.isclose(f_min, f_max):
        raise ValueError("Frequency range is too narrow to build a log-binned plot.")

    bin_edges = np.logspace(np.log10(f_min), np.log10(f_max), num_log_bins + 1)

    work_df["freq_bin"] = pd.cut(
        work_df["frequency_hz"],
        bins=bin_edges,
        include_lowest=True,
        duplicates="drop",
    )

    freq_summary_df = (
        work_df.groupby("freq_bin", observed=False)
        .agg(
            dB_abs_err_median=("dB_abs_err", "median"),
            phi_abs_err_deg_median=("phi_abs_err_deg", "median"),
            freq_left=("frequency_hz", "min"),
            freq_right=("frequency_hz", "max"),
            n_points=("frequency_hz", "size"),
        )
        .reset_index(drop=True)
    )

    freq_summary_df = freq_summary_df[freq_summary_df["n_points"] > 0].copy()

    freq_summary_df["freq_center"] = np.sqrt(
        freq_summary_df["freq_left"] * freq_summary_df["freq_right"]
    )

    fig, axes = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True)

    axes[0].plot(
        freq_summary_df["freq_center"],
        freq_summary_df["dB_abs_err_median"],
    )
    axes[0].set_xscale("log")
    axes[0].set_ylabel("Медианная |ΔdB|")
    axes[0].set_title("Зависимость ошибки по модулю от частоты")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].plot(
        freq_summary_df["freq_center"],
        freq_summary_df["phi_abs_err_deg_median"],
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Частота, Гц")
    axes[1].set_ylabel("Медианная |Δφ|, град")
    axes[1].set_title("Зависимость ошибки по фазе от частоты")
    axes[1].grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    output_path = os.path.join(output_dir, "error_vs_frequency.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_group_summary(predictions_df: pd.DataFrame, output_dir: str) -> None:
    """
    Сохраняет сводную таблицу ошибок по группам param_set_id в CSV-файл.
    """
    summary_df = build_group_summary(predictions_df)
    output_path = os.path.join(output_dir, "group_error_summary.csv")
    summary_df.to_csv(output_path, index=False)
