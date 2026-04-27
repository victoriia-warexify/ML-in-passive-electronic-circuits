# src/ml_circuits/features/rc_ladder_features.py

from __future__ import annotations

import ast
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ml_circuits.constants import EPS, PHASE_FILTER_MAG, TWO_PI


DEFAULT_ARTIFACT_DIR = "artifacts_rc_ladder_pure_ml_strict_blackbox"
DEFAULT_PLOTS_DIRNAME = "plots_rc_ladder"

EXPERIMENT_RUNS = [
    {
        "mode": "strict_extrapolation",
        "path": "artifacts_rc_ladder_pure_ml_strict_blackbox/leaderboard_seeds.csv",
    },
    {
        "mode": "fewshot_4pct",
        "path": "artifacts_rc_ladder_pure_ml_fewshot_n4_4pct/leaderboard_seeds.csv",
    },
]

REQUIRED_COLS = [
    "seed",
    "score",
    "test_dB_MAE",
    "test_phi_MAE_filtered_deg",
    "test_phi_MAE_deg",
    "test_logH_MAE",
    "test_dB_MAE_group_weighted",
    "test_dB_RMSE_group_weighted",
    "test_phi_MAE_filtered_deg_group_weighted",
    "test_phi_MAE_deg_group_weighted",
    "test_logH_MAE_group_weighted",
]

RUN_CONFIGS = [
    {
        "mode_name": "strict_extrapolation",
        "mode": "strict",
        "n4_train_fraction": None,
        "artifact_dir": "artifacts_rc_ladder_pure_ml_strict_blackbox",
    },
    {
        "mode_name": "fewshot_4pct",
        "mode": "fewshot",
        "n4_train_fraction": 0.04,
        "artifact_dir": "artifacts_rc_ladder_pure_ml_fewshot_n4_4pct",
    },
]


def make_artifact_dir(mode: str, n4_train_fraction: float | None = None) -> str:
    """
    Формирует имя каталога для сохранения артефактов эксперимента.
    """
    if mode == "strict":
        return "artifacts_rc_ladder_pure_ml_strict_blackbox"
    if mode == "fewshot":
        if n4_train_fraction is None:
            raise ValueError("n4_train_fraction must be set for fewshot mode")
        frac_pct = int(round(n4_train_fraction * 100))
        return f"artifacts_rc_ladder_pure_ml_fewshot_n4_{frac_pct}pct"
    raise ValueError(f"Unknown mode: {mode}")


def set_seed(seed: int = 42) -> None:
    """
    Устанавливает seed для генераторов случайных чисел Python, NumPy и PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def wrap_angle_rad_np(x: np.ndarray) -> np.ndarray:
    """
    Нормирует углы в радианах к диапазону [-π, π).
    """
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def wrap_angle_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Нормирует углы в радианах к диапазону [-π, π) в формате PyTorch.
    """
    return torch.atan2(torch.sin(x), torch.cos(x))


def wrap_angle_deg_np(angle_deg: np.ndarray) -> np.ndarray:
    """
    Нормирует углы в градусах к диапазону [-180, 180).
    """
    return (angle_deg + 180.0) % 360.0 - 180.0


def json_default(obj: object) -> object:
    """
    Преобразует объекты NumPy к типам, совместимым с JSON-сериализацией.
    """
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def zscore(x: float, mean: float, std: float) -> float:
    """
    Выполняет z-нормировку скалярной величины.
    """
    return (x - mean) / std


def safe_log(x: float) -> float:
    """
    Возвращает натуральный логарифм аргумента с нижним ограничением EPS.
    """
    return math.log(max(float(x), EPS))


def parse_sections_cell(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except (ValueError, SyntaxError):
            return json.loads(x)
    raise TypeError(f"Unsupported sections type: {type(x)}")


def load_jsonl(path: str) -> pd.DataFrame:
    """
    Загружает датасет в формате JSONL.
    """
    return pd.read_json(path, lines=True)


def sanity_checks(df: pd.DataFrame) -> None:
    """
    Выполняет базовые проверки внутренней согласованности датасета.
    """
    print("=== SANITY CHECKS ===")
    print(f"samples: {len(df)}")

    counts = df.groupby("n_sections")["param_set_id"].nunique().sort_index()
    print("unique param_set_id by n_sections:")
    for n, c in counts.items():
        print(f"  n={n}: {c}")

    h_mag = df["H_mag"].to_numpy(dtype=float)
    a_out = df["A_out"].to_numpy(dtype=float)
    a_in = df["A_in"].to_numpy(dtype=float)
    ratio = a_out / np.maximum(a_in, EPS)

    rel_err = np.abs(h_mag - ratio) / np.maximum(np.abs(ratio), EPS)
    print(f"H_mag vs A_out/A_in median rel err: {np.median(rel_err):.3e}")

    log_expected = np.log(np.clip(ratio, EPS, None))
    log_err = np.abs(df["log_H_mag"].to_numpy(dtype=float) - log_expected)
    print(f"log_H_mag vs log(max(A_out/A_in, EPS)) median abs err: {np.median(log_err):.3e}")

    sc_dev = np.abs(
        df["phi_sin"].to_numpy(dtype=float) ** 2
        + df["phi_cos"].to_numpy(dtype=float) ** 2
        - 1.0
    )
    print(f"sin^2+cos^2 median abs deviation from 1: {np.median(sc_dev):.3e}")
    print()


def fit_feature_stats(df: pd.DataFrame) -> Dict[str, float]:
    """
    Оценивает статистики нормировки признаков только по обучающей выборке.
    """
    logR_vals = []
    logC_vals = []
    logTau_vals = []
    logWTau_vals = []
    logf_vals = []
    logRload_vals = []

    mean_logR_vals = []
    mean_logC_vals = []
    mean_logTau_vals = []
    std_logR_vals = []
    std_logC_vals = []
    min_logTau_vals = []
    max_logTau_vals = []

    mean_logWTau_vals = []
    std_logWTau_vals = []
    min_logWTau_vals = []
    max_logWTau_vals = []

    for row in df.itertuples(index=False):
        cur_logR = []
        cur_logC = []
        cur_logTau = []
        cur_logWTau = []
        f_hz = float(row.frequency_hz)

        for sec in row.sections:
            R = float(sec["R"])
            C = float(sec["C"])
            tau = R * C
            w_tau = TWO_PI * f_hz * tau

            lR = safe_log(R)
            lC = safe_log(C)
            lTau = safe_log(tau)
            lWTau = safe_log(w_tau)

            logR_vals.append(lR)
            logC_vals.append(lC)
            logTau_vals.append(lTau)
            logWTau_vals.append(lWTau)

            cur_logR.append(lR)
            cur_logC.append(lC)
            cur_logTau.append(lTau)
            cur_logWTau.append(lWTau)

        logf_vals.append(safe_log(row.frequency_hz))
        logRload_vals.append(safe_log(row.Rload))

        cur_logR_np = np.asarray(cur_logR, dtype=float)
        cur_logC_np = np.asarray(cur_logC, dtype=float)
        cur_logTau_np = np.asarray(cur_logTau, dtype=float)
        cur_logWTau_np = np.asarray(cur_logWTau, dtype=float)

        mean_logR_vals.append(float(np.mean(cur_logR_np)))
        mean_logC_vals.append(float(np.mean(cur_logC_np)))
        mean_logTau_vals.append(float(np.mean(cur_logTau_np)))
        std_logR_vals.append(float(np.std(cur_logR_np)))
        std_logC_vals.append(float(np.std(cur_logC_np)))
        min_logTau_vals.append(float(np.min(cur_logTau_np)))
        max_logTau_vals.append(float(np.max(cur_logTau_np)))

        mean_logWTau_vals.append(float(np.mean(cur_logWTau_np)))
        std_logWTau_vals.append(float(np.std(cur_logWTau_np)))
        min_logWTau_vals.append(float(np.min(cur_logWTau_np)))
        max_logWTau_vals.append(float(np.max(cur_logWTau_np)))

    def pack(arr: List[float], prefix: str) -> Dict[str, float]:
        x = np.asarray(arr, dtype=float)
        return {
            f"{prefix}_mean": float(np.mean(x)),
            f"{prefix}_std": max(float(np.std(x)), 1e-6),
        }

    stats = {}
    stats.update(pack(logR_vals, "logR"))
    stats.update(pack(logC_vals, "logC"))
    stats.update(pack(logTau_vals, "logTau"))
    stats.update(pack(logWTau_vals, "logWTau"))
    stats.update(pack(logf_vals, "logf"))
    stats.update(pack(logRload_vals, "logRload"))

    stats.update(pack(mean_logR_vals, "mean_logR"))
    stats.update(pack(mean_logC_vals, "mean_logC"))
    stats.update(pack(mean_logTau_vals, "mean_logTau"))
    stats.update(pack(std_logR_vals, "std_logR"))
    stats.update(pack(std_logC_vals, "std_logC"))
    stats.update(pack(min_logTau_vals, "min_logTau"))
    stats.update(pack(max_logTau_vals, "max_logTau"))

    stats.update(pack(mean_logWTau_vals, "mean_logWTau"))
    stats.update(pack(std_logWTau_vals, "std_logWTau"))
    stats.update(pack(min_logWTau_vals, "min_logWTau"))
    stats.update(pack(max_logWTau_vals, "max_logWTau"))

    return stats


@dataclass
class Sample:
    """
    Представление одного объекта датасета после преобразования исходной записи.
    """
    seq_feat: np.ndarray
    seq_len: int
    global_feat: np.ndarray
    a_in: float
    y: np.ndarray
    y_aux: np.ndarray
    param_set_id: str
    frequency_hz: float
    n_sections: int


class RCLadderPureMLDataset(Dataset):
    """
    Формирует PyTorch-датасет для black-box-модели распределённой RC-цепочки.
    """
    def __init__(self, df: pd.DataFrame, stats: Dict[str, float], max_sections: int = 4):
        self.df = df.reset_index(drop=True)
        self.stats = stats
        self.max_sections = max_sections
        self.samples: List[Sample] = []

        for row in self.df.itertuples(index=False):
            sections = parse_sections_cell(row.sections)
            n = int(row.n_sections)
            if not (1 <= n <= max_sections):
                raise ValueError(f"Unexpected n_sections={n}")

            f_hz = float(row.frequency_hz)
            logf = safe_log(f_hz)
            logRload = safe_log(row.Rload)
            z_logf = zscore(logf, stats["logf_mean"], stats["logf_std"])
            z_logRload = zscore(logRload, stats["logRload_mean"], stats["logRload_std"])

            cur_logR = []
            cur_logC = []
            cur_logTau = []
            cur_logWTau = []

            for sec in sections:
                R = float(sec["R"])
                C = float(sec["C"])
                tau = R * C
                w_tau = TWO_PI * f_hz * tau

                cur_logR.append(safe_log(R))
                cur_logC.append(safe_log(C))
                cur_logTau.append(safe_log(tau))
                cur_logWTau.append(safe_log(w_tau))

            cur_logR_np = np.asarray(cur_logR, dtype=float)
            cur_logC_np = np.asarray(cur_logC, dtype=float)
            cur_logTau_np = np.asarray(cur_logTau, dtype=float)
            cur_logWTau_np = np.asarray(cur_logWTau, dtype=float)

            mean_logR = float(np.mean(cur_logR_np))
            mean_logC = float(np.mean(cur_logC_np))
            mean_logTau = float(np.mean(cur_logTau_np))
            std_logR = float(np.std(cur_logR_np))
            std_logC = float(np.std(cur_logC_np))
            min_logTau = float(np.min(cur_logTau_np))
            max_logTau = float(np.max(cur_logTau_np))

            mean_logWTau = float(np.mean(cur_logWTau_np))
            std_logWTau = float(np.std(cur_logWTau_np))
            min_logWTau = float(np.min(cur_logWTau_np))
            max_logWTau = float(np.max(cur_logWTau_np))

            global_feat = np.array(
                [
                    z_logf,
                    z_logRload,
                    zscore(mean_logR, stats["mean_logR_mean"], stats["mean_logR_std"]),
                    zscore(mean_logC, stats["mean_logC_mean"], stats["mean_logC_std"]),
                    zscore(mean_logTau, stats["mean_logTau_mean"], stats["mean_logTau_std"]),
                    zscore(std_logR, stats["std_logR_mean"], stats["std_logR_std"]),
                    zscore(std_logC, stats["std_logC_mean"], stats["std_logC_std"]),
                    zscore(min_logTau, stats["min_logTau_mean"], stats["min_logTau_std"]),
                    zscore(max_logTau, stats["max_logTau_mean"], stats["max_logTau_std"]),
                    zscore(mean_logWTau, stats["mean_logWTau_mean"], stats["mean_logWTau_std"]),
                    zscore(std_logWTau, stats["std_logWTau_mean"], stats["std_logWTau_std"]),
                    zscore(min_logWTau, stats["min_logWTau_mean"], stats["min_logWTau_std"]),
                    zscore(max_logWTau, stats["max_logWTau_mean"], stats["max_logWTau_std"]),
                ],
                dtype=np.float32,
            )

            seq_feat = np.zeros((max_sections, 7), dtype=np.float32)

            for i, sec in enumerate(sections):
                R = float(sec["R"])
                C = float(sec["C"])
                tau = R * C
                w_tau = TWO_PI * f_hz * tau
                lR = safe_log(R)
                lC = safe_log(C)
                lWTau = safe_log(w_tau)

                is_first = 1.0 if i == 0 else 0.0
                is_last = 1.0 if i == n - 1 else 0.0

                seq_feat[i, 0] = zscore(lR, stats["logR_mean"], stats["logR_std"])
                seq_feat[i, 1] = zscore(lC, stats["logC_mean"], stats["logC_std"])
                seq_feat[i, 2] = zscore(lWTau, stats["logWTau_mean"], stats["logWTau_std"])
                seq_feat[i, 3] = is_first
                seq_feat[i, 4] = is_last
                seq_feat[i, 5] = z_logf
                seq_feat[i, 6] = z_logRload

            logH_per_section = float(row.log_H_mag) / max(n, 1)
            H_mag = float(row.H_mag)

            y = np.array(
                [logH_per_section, float(row.phi_sin), float(row.phi_cos), H_mag],
                dtype=np.float32,
            )
            y_aux = np.array(
                [float(row.H_mag), float(row.A_out), float(row.phi_rad), float(row.log_H_mag)],
                dtype=np.float32,
            )

            self.samples.append(
                Sample(
                    seq_feat=seq_feat,
                    seq_len=n,
                    global_feat=global_feat,
                    a_in=np.float32(float(row.A_in)),
                    y=y,
                    y_aux=y_aux,
                    param_set_id=str(row.param_set_id),
                    frequency_hz=f_hz,
                    n_sections=n,
                )
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        return {
            "seq_feat": torch.from_numpy(s.seq_feat),
            "seq_len": torch.tensor(s.seq_len, dtype=torch.long),
            "global_feat": torch.from_numpy(s.global_feat),
            "a_in": torch.tensor(s.a_in, dtype=torch.float32),
            "y": torch.from_numpy(s.y),
            "y_aux": torch.from_numpy(s.y_aux),
            "param_set_id": s.param_set_id,
            "frequency_hz": s.frequency_hz,
            "n_sections": s.n_sections,
        }
