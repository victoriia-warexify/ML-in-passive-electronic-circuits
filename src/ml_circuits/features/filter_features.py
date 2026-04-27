# src/ml_circuits/features/filter_features.py

"""
Формирование признаков для моделей обычных пассивных фильтров.

Модуль содержит функции построения признаков для топологий:
    RC_LP, RC_HP, RL_LP, RL_HP, RLC_BP, RLC_NOTCH.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_circuits.constants import EPS_F, EPS_NORM, OPEN_R, R_FLOOR_OHM, TWO_PI


def require_cols(df: pd.DataFrame, cols: list[str]) -> None:
    """
    Проверяет наличие обязательных колонок в DataFrame.
    """
    missing = [col for col in cols if col not in df.columns]

    if missing:
        raise ValueError(f"Отсутствуют обязательные колонки: {missing}")


def get_rload_eff(df: pd.DataFrame) -> np.ndarray:
    """
    Возвращает эффективное сопротивление нагрузки.

    Если Rload отсутствует, NaN или неположительно, используется OPEN_R,
    что соответствует практически разомкнутой нагрузке.
    """
    if "Rload" not in df.columns:
        return np.full(len(df), OPEN_R, dtype=float)

    Rload = df["Rload"].to_numpy(dtype=float)

    return np.where(
        np.isfinite(Rload) & (Rload > 0.0),
        Rload,
        OPEN_R,
    )


def parallel_r(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    """
    Эквивалентное сопротивление параллельного соединения a || b.
    """
    a_safe = np.maximum(a, R_FLOOR_OHM)
    b_safe = np.maximum(b, R_FLOOR_OHM)

    return 1.0 / (1.0 / a_safe + 1.0 / b_safe)


def _safe_log(x: np.ndarray | float, eps: float = EPS_NORM) -> np.ndarray | float:
    """
    Безопасный натуральный логарифм положительной величины.
    """
    return np.log(np.maximum(x, eps))


def make_features_rc_hp(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Формирует признаки для RC-фильтра верхних частот.

    Топология:
        Vs -> C -> out,
        out -> R -> GND.
    """
    require_cols(df, ["f", "R", "C"])

    f = df["f"].to_numpy(dtype=float)
    R = df["R"].to_numpy(dtype=float)
    C = df["C"].to_numpy(dtype=float)

    w = TWO_PI * f
    fc = 1.0 / (TWO_PI * R * C)
    x = f / np.maximum(fc, EPS_F)

    X = np.column_stack(
        [
            _safe_log(f),
            _safe_log(R),
            _safe_log(C),
            _safe_log(fc),
            _safe_log(x),
            np.log1p(x),
            x / (1.0 + x),
        ]
    )

    names = [
        "log_f",
        "log_R",
        "log_C",
        "log_fc",
        "log_f_over_fc",
        "log1p_f_over_fc",
        "f_over_fc_norm",
    ]

    return X, names


def make_features_rc_lp(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Формирует признаки для RC-фильтра нижних частот с возможной нагрузкой.
    """
    require_cols(df, ["f", "R", "C"])

    f = df["f"].to_numpy(dtype=float)
    R = df["R"].to_numpy(dtype=float)
    C = df["C"].to_numpy(dtype=float)
    Rload = get_rload_eff(df)

    R_eq = parallel_r(R, Rload)

    w = TWO_PI * f
    fc_unloaded = 1.0 / (TWO_PI * R * C)
    fc_loaded = 1.0 / (TWO_PI * R_eq * C)

    x_unloaded = f / np.maximum(fc_unloaded, EPS_F)
    x_loaded = f / np.maximum(fc_loaded, EPS_F)
    load_ratio = Rload / np.maximum(R, R_FLOOR_OHM)

    X = np.column_stack(
        [
            _safe_log(f),
            _safe_log(R),
            _safe_log(C),
            _safe_log(Rload),
            _safe_log(R_eq),
            _safe_log(fc_unloaded),
            _safe_log(fc_loaded),
            _safe_log(x_unloaded),
            _safe_log(x_loaded),
            np.log1p(x_loaded),
            x_loaded / (1.0 + x_loaded),
            _safe_log(load_ratio),
        ]
    )

    names = [
        "log_f",
        "log_R",
        "log_C",
        "log_Rload",
        "log_R_eq",
        "log_fc_unloaded",
        "log_fc_loaded",
        "log_f_over_fc_unloaded",
        "log_f_over_fc_loaded",
        "log1p_f_over_fc_loaded",
        "f_over_fc_loaded_norm",
        "log_Rload_over_R",
    ]

    return X, names


def make_features_rl_lp(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Формирует признаки для RL-фильтра нижних частот.

    Топология:
        Vs -> L -> out,
        out -> R -> GND.
    """
    require_cols(df, ["f", "R", "L"])

    f = df["f"].to_numpy(dtype=float)
    R = df["R"].to_numpy(dtype=float)
    L = df["L"].to_numpy(dtype=float)

    fc = R / (TWO_PI * L)
    x = f / np.maximum(fc, EPS_F)

    X = np.column_stack(
        [
            _safe_log(f),
            _safe_log(R),
            _safe_log(L),
            _safe_log(fc),
            _safe_log(x),
            np.log1p(x),
            x / (1.0 + x),
        ]
    )

    names = [
        "log_f",
        "log_R",
        "log_L",
        "log_fc",
        "log_f_over_fc",
        "log1p_f_over_fc",
        "f_over_fc_norm",
    ]

    return X, names


def make_features_rl_hp(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Формирует признаки для RL-фильтра верхних частот с возможной нагрузкой.
    """
    require_cols(df, ["f", "R", "L"])

    f = df["f"].to_numpy(dtype=float)
    R = df["R"].to_numpy(dtype=float)
    L = df["L"].to_numpy(dtype=float)
    Rload = get_rload_eff(df)

    R_eq = parallel_r(R, Rload)

    fc_unloaded = R / (TWO_PI * L)
    fc_loaded = R_eq / (TWO_PI * L)

    x_unloaded = f / np.maximum(fc_unloaded, EPS_F)
    x_loaded = f / np.maximum(fc_loaded, EPS_F)
    load_ratio = Rload / np.maximum(R, R_FLOOR_OHM)

    X = np.column_stack(
        [
            _safe_log(f),
            _safe_log(R),
            _safe_log(L),
            _safe_log(Rload),
            _safe_log(R_eq),
            _safe_log(fc_unloaded),
            _safe_log(fc_loaded),
            _safe_log(x_unloaded),
            _safe_log(x_loaded),
            np.log1p(x_loaded),
            x_loaded / (1.0 + x_loaded),
            _safe_log(load_ratio),
        ]
    )

    names = [
        "log_f",
        "log_R",
        "log_L",
        "log_Rload",
        "log_R_eq",
        "log_fc_unloaded",
        "log_fc_loaded",
        "log_f_over_fc_unloaded",
        "log_f_over_fc_loaded",
        "log1p_f_over_fc_loaded",
        "f_over_fc_loaded_norm",
        "log_Rload_over_R",
    ]

    return X, names


def make_features_rlc_bp(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Формирует признаки для RLC band-pass фильтра.
    """
    require_cols(df, ["f", "R", "L", "C", "Rload"])

    f = df["f"].to_numpy(dtype=float)
    R = df["R"].to_numpy(dtype=float)
    L = df["L"].to_numpy(dtype=float)
    C = df["C"].to_numpy(dtype=float)
    Rload = get_rload_eff(df)

    R_eq = parallel_r(R, Rload)

    f0 = 1.0 / (TWO_PI * np.sqrt(L * C))
    rho = np.sqrt(L / C)
    Q_eff = R_eq / np.maximum(rho, EPS_NORM)

    x = f / np.maximum(f0, EPS_F)
    log_x = _safe_log(x)

    detuning = x - 1.0 / np.maximum(x, EPS_NORM)
    abs_detuning = np.abs(detuning)

    load_ratio = Rload / np.maximum(R, R_FLOOR_OHM)

    X = np.column_stack(
        [
            _safe_log(f),
            _safe_log(R),
            _safe_log(L),
            _safe_log(C),
            _safe_log(Rload),
            _safe_log(R_eq),
            _safe_log(f0),
            _safe_log(Q_eff),
            _safe_log(rho),
            log_x,
            np.abs(log_x),
            detuning,
            abs_detuning,
            Q_eff * detuning,
            Q_eff * abs_detuning,
            _safe_log(load_ratio),
        ]
    )

    names = [
        "log_f",
        "log_R",
        "log_L",
        "log_C",
        "log_Rload",
        "log_R_eq",
        "log_f0",
        "log_Q_eff",
        "log_rho",
        "log_f_over_f0",
        "abs_log_f_over_f0",
        "detuning_x_minus_invx",
        "abs_detuning",
        "Q_times_detuning",
        "Q_times_abs_detuning",
        "log_Rload_over_R",
    ]

    return X, names


def make_features_rlc_notch(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Формирует признаки для RLC notch / band-stop фильтра.
    """
    require_cols(df, ["f", "R", "L", "C", "Rload"])

    f = df["f"].to_numpy(dtype=float)
    R = df["R"].to_numpy(dtype=float)
    L = df["L"].to_numpy(dtype=float)
    C = df["C"].to_numpy(dtype=float)
    Rload = get_rload_eff(df)

    R_eq = parallel_r(R, Rload)

    f0 = 1.0 / (TWO_PI * np.sqrt(L * C))
    rho = np.sqrt(L / C)
    Q_eff = R_eq / np.maximum(rho, EPS_NORM)

    x = f / np.maximum(f0, EPS_F)
    log_x = _safe_log(x)

    detuning = x - 1.0 / np.maximum(x, EPS_NORM)
    abs_detuning = np.abs(detuning)

    load_ratio = Rload / np.maximum(R, R_FLOOR_OHM)

    X = np.column_stack(
        [
            _safe_log(f),
            _safe_log(R),
            _safe_log(L),
            _safe_log(C),
            _safe_log(Rload),
            _safe_log(R_eq),
            _safe_log(f0),
            _safe_log(Q_eff),
            _safe_log(rho),
            log_x,
            np.abs(log_x),
            detuning,
            abs_detuning,
            Q_eff * detuning,
            Q_eff * abs_detuning,
            _safe_log(load_ratio),
        ]
    )

    names = [
        "log_f",
        "log_R",
        "log_L",
        "log_C",
        "log_Rload",
        "log_R_eq",
        "log_f0",
        "log_Q_eff",
        "log_rho",
        "log_f_over_f0",
        "abs_log_f_over_f0",
        "detuning_x_minus_invx",
        "abs_detuning",
        "Q_times_detuning",
        "Q_times_abs_detuning",
        "log_Rload_over_R",
    ]

    return X, names


FEATS_ORDER1 = {
    "RC_HP": make_features_rc_hp,
    "RC_LP": make_features_rc_lp,
    "RL_HP": make_features_rl_hp,
    "RL_LP": make_features_rl_lp,
}


FEATS_RLC = {
    "RLC_BP": make_features_rlc_bp,
    "RLC_NOTCH": make_features_rlc_notch,
}