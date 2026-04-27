# src/ml_circuits/features/rc_ladder_features.py

"""
Формирование признаков для RC ladder.

В отличие от обычных фильтров, RC ladder имеет переменное число секций.
Поэтому признаки делятся на:
    - признаки отдельных секций;
    - глобальные признаки частотной точки;
    - маску активных секций.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ml_circuits.constants import EPS_F, EPS_LOG, EPS_NORM, TWO_PI


def parse_sections_cell(sections: Any) -> list[dict[str, float]]:
    """
    Приводит поле sections к списку словарей {"R": ..., "C": ...}.

    При чтении из JSONL поле обычно уже является списком.
    При чтении из CSV оно может быть строкой JSON.
    """
    if isinstance(sections, str):
        sections = json.loads(sections)

    if not isinstance(sections, list):
        raise ValueError("sections must be a list or JSON string")

    parsed = []

    for sec in sections:
        if not isinstance(sec, dict):
            raise ValueError("Each section must be a dictionary")

        parsed.append(
            {
                "R": float(sec["R"]),
                "C": float(sec["C"]),
            }
        )

    return parsed


def safe_log(x: np.ndarray | float, eps: float = EPS_LOG) -> np.ndarray | float:
    """
    Безопасный натуральный логарифм положительной величины.
    """
    return np.log(np.clip(x, eps, None))


def rc_ladder_section_arrays(
    sections: list[dict[str, float]],
    max_sections: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Преобразует список секций в массивы R, C и маску активных секций.

    Returns
    -------
    R_arr:
        Массив сопротивлений длины max_sections.
    C_arr:
        Массив ёмкостей длины max_sections.
    mask:
        Маска активных секций длины max_sections.
    """
    if len(sections) > max_sections:
        raise ValueError(
            f"len(sections)={len(sections)} exceeds max_sections={max_sections}"
        )

    R_arr = np.ones(max_sections, dtype=float)
    C_arr = np.ones(max_sections, dtype=float)
    mask = np.zeros(max_sections, dtype=float)

    for i, sec in enumerate(sections):
        R = float(sec["R"])
        C = float(sec["C"])

        if not np.isfinite(R) or not np.isfinite(C) or R <= 0.0 or C <= 0.0:
            raise ValueError("Each RC ladder section must have positive finite R and C")

        R_arr[i] = R
        C_arr[i] = C
        mask[i] = 1.0

    return R_arr, C_arr, mask


def rc_ladder_tau_features(
    R_arr: np.ndarray,
    C_arr: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """
    Вычисляет агрегированные признаки временных постоянных RC-секций.
    """
    active = mask > 0.5

    if not np.any(active):
        raise ValueError("At least one section must be active")

    tau = R_arr[active] * C_arr[active]

    log_tau = np.log(np.clip(tau, EPS_NORM, None))

    return {
        "tau_mean_log": float(np.mean(log_tau)),
        "tau_std_log": float(np.std(log_tau)),
        "tau_min_log": float(np.min(log_tau)),
        "tau_max_log": float(np.max(log_tau)),
        "tau_sum_log": float(np.log(np.sum(tau))),
    }


def rc_ladder_ref_frequency_from_sections(sections: list[dict[str, float]]) -> float:
    """
    Оценивает характерную частоту RC ladder по среднему значению RC.
    """
    taus = []

    for sec in sections:
        R = float(sec["R"])
        C = float(sec["C"])
        taus.append(R * C)

    tau_ref = float(np.exp(np.mean(np.log(np.clip(taus, EPS_NORM, None)))))

    return 1.0 / (TWO_PI * tau_ref)


def make_rc_ladder_global_features(row: pd.Series | dict) -> tuple[np.ndarray, list[str]]:
    """
    Формирует глобальные признаки одной частотной точки RC ladder.

    Эти признаки не зависят от конкретного номера секции.
    """
    if isinstance(row, pd.Series):
        row = row.to_dict()

    sections = parse_sections_cell(row["sections"])
    n_sections = int(row["n_sections"])
    Rload = float(row["Rload"])
    f = float(row["frequency_hz"])

    f_ref = rc_ladder_ref_frequency_from_sections(sections)
    x = f / max(f_ref, EPS_F)

    feats = np.array(
        [
            safe_log(f),
            safe_log(f_ref),
            safe_log(x),
            np.log1p(x),
            x / (1.0 + x),
            safe_log(Rload),
            float(n_sections),
        ],
        dtype=float,
    )

    names = [
        "log_f",
        "log_f_ref",
        "log_f_over_f_ref",
        "log1p_f_over_f_ref",
        "f_over_f_ref_norm",
        "log_Rload",
        "n_sections",
    ]

    return feats, names


def make_rc_ladder_section_features(
    row: pd.Series | dict,
    max_sections: int = 4,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Формирует признаки секций RC ladder.

    Returns
    -------
    section_features:
        Массив формы (max_sections, n_section_features).
    mask:
        Маска активных секций формы (max_sections,).
    names:
        Имена признаков одной секции.
    """
    if isinstance(row, pd.Series):
        row = row.to_dict()

    sections = parse_sections_cell(row["sections"])
    R_arr, C_arr, mask = rc_ladder_section_arrays(
        sections=sections,
        max_sections=max_sections,
    )

    tau = R_arr * C_arr

    section_features = np.column_stack(
        [
            safe_log(R_arr),
            safe_log(C_arr),
            safe_log(tau),
        ]
    )

    names = [
        "log_R",
        "log_C",
        "log_tau",
    ]

    return section_features.astype(float), mask.astype(float), names


def make_rc_ladder_features(
    row: pd.Series | dict,
    max_sections: int = 4,
) -> dict[str, np.ndarray | list[str]]:
    """
    Формирует полный набор признаков для одной строки RC ladder.

    Возвращает словарь, который удобно использовать в Dataset-классе.
    """
    global_features, global_names = make_rc_ladder_global_features(row)
    section_features, section_mask, section_names = make_rc_ladder_section_features(
        row=row,
        max_sections=max_sections,
    )

    return {
        "global_features": global_features,
        "global_feature_names": global_names,
        "section_features": section_features,
        "section_feature_names": section_names,
        "section_mask": section_mask,
    }


def make_rc_ladder_targets(row: pd.Series | dict) -> tuple[np.ndarray, list[str]]:
    """
    Формирует целевые переменные для одной строки RC ladder.

    Target:
        logH, sin(phi), cos(phi)
    """
    if isinstance(row, pd.Series):
        row = row.to_dict()

    H_mag = float(row["H_mag"])
    phi_sin = float(row["phi_sin"])
    phi_cos = float(row["phi_cos"])

    y = np.array(
        [
            float(np.log(np.clip(H_mag, EPS_LOG, None))),
            phi_sin,
            phi_cos,
        ],
        dtype=float,
    )

    names = [
        "logH",
        "phi_sin",
        "phi_cos",
    ]

    return y, names