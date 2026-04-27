# src/ml_circuits/data_generation/validation.py

"""
Проверка корректности сгенерированных датасетов.

Модуль содержит базовые sanity-checks для dataset_ac.csv
и dataset_rc_ladder.jsonl:
    - отсутствие NaN/inf в основных числовых колонках;
    - согласованность H_mag с A_out / A_in;
    - согласованность log_H_mag с ln(max(H_mag, EPS_LOG));
    - корректность фазового представления sin(phi), cos(phi);
    - проверка пассивности |H(jω)| <= 1;
    - проверка отсутствия дублей по схеме и частоте.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_circuits.constants import EPS_DIV, EPS_LOG


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """
    Проверяет наличие обязательных колонок в DataFrame.
    """
    missing = [col for col in columns if col not in df.columns]

    if missing:
        raise ValueError(f"В датасете отсутствуют обязательные колонки: {missing}")


def _check_finite(df: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    """
    Проверяет, что указанные числовые колонки не содержат NaN/inf.

    Returns
    -------
    dict[str, int]
        Число неконечных значений по каждой колонке.
    """
    bad_counts: dict[str, int] = {}

    for col in columns:
        values = df[col].to_numpy(dtype=float)
        bad_counts[col] = int((~np.isfinite(values)).sum())

    total_bad = sum(bad_counts.values())

    if total_bad > 0:
        raise ValueError(f"Обнаружены NaN/inf в числовых колонках: {bad_counts}")

    return bad_counts


def validate_log_h_mag(
    df: pd.DataFrame,
    eps_log: float = EPS_LOG,
    atol: float = 1e-10,
) -> dict[str, float]:
    """
    Проверяет согласованность log_H_mag с H_mag и A_out / A_in.

    В проекте используется определение:
        log_H_mag = ln(max(|H(jω)|, EPS_LOG))

    где |H(jω)| = A_out / A_in.
    """
    _require_columns(df, ["A_in", "A_out", "H_mag", "log_H_mag"])

    A_in = df["A_in"].to_numpy(dtype=float)
    A_out = df["A_out"].to_numpy(dtype=float)
    H_mag = df["H_mag"].to_numpy(dtype=float)
    log_H_mag = df["log_H_mag"].to_numpy(dtype=float)

    ratio = A_out / np.clip(A_in, EPS_DIV, None)

    h_err = np.abs(H_mag - ratio)
    log_expected_from_h = np.log(np.clip(H_mag, eps_log, None))
    log_expected_from_ratio = np.log(np.clip(ratio, eps_log, None))

    log_err_from_h = np.abs(log_H_mag - log_expected_from_h)
    log_err_from_ratio = np.abs(log_H_mag - log_expected_from_ratio)

    max_h_err = float(np.nanmax(h_err))
    max_log_err_from_h = float(np.nanmax(log_err_from_h))
    max_log_err_from_ratio = float(np.nanmax(log_err_from_ratio))

    if max_h_err > atol:
        raise ValueError(
            f"H_mag не согласован с A_out / A_in: max_abs_err={max_h_err:.3e}"
        )

    if max_log_err_from_h > atol:
        raise ValueError(
            "log_H_mag не согласован с ln(max(H_mag, EPS_LOG)): "
            f"max_abs_err={max_log_err_from_h:.3e}"
        )

    if max_log_err_from_ratio > atol:
        raise ValueError(
            "log_H_mag не согласован с ln(max(A_out / A_in, EPS_LOG)): "
            f"max_abs_err={max_log_err_from_ratio:.3e}"
        )

    return {
        "max_H_mag_vs_ratio_abs_err": max_h_err,
        "max_log_H_mag_vs_H_mag_abs_err": max_log_err_from_h,
        "max_log_H_mag_vs_ratio_abs_err": max_log_err_from_ratio,
    }


def validate_phase_sin_cos(
    df: pd.DataFrame,
    atol: float = 1e-10,
) -> dict[str, float]:
    """
    Проверяет корректность фазового представления через sin(phi), cos(phi).

    Для каждой строки должно выполняться:
        sin(phi)^2 + cos(phi)^2 ≈ 1.
    """
    _require_columns(df, ["phi_sin", "phi_cos"])

    phi_sin = df["phi_sin"].to_numpy(dtype=float)
    phi_cos = df["phi_cos"].to_numpy(dtype=float)

    circle = phi_sin**2 + phi_cos**2
    err = np.abs(circle - 1.0)

    max_err = float(np.nanmax(err))
    mean_err = float(np.nanmean(err))

    if max_err > atol:
        raise ValueError(
            "Фазовые компоненты не лежат на единичной окружности: "
            f"max_abs_err={max_err:.3e}"
        )

    return {
        "max_sin2_plus_cos2_minus_1_abs_err": max_err,
        "mean_sin2_plus_cos2_minus_1_abs_err": mean_err,
    }


def validate_passivity(
    df: pd.DataFrame,
    tol: float = 1e-9,
) -> dict[str, float]:
    """
    Проверяет пассивность схем по условию |H(jω)| <= 1.

    Для пассивных цепей без усилителей модуль передаточной функции
    не должен превышать единицу с учётом численной погрешности.
    """
    _require_columns(df, ["H_mag"])

    H_mag = df["H_mag"].to_numpy(dtype=float)

    max_H = float(np.nanmax(H_mag))
    n_violations = int(np.sum(H_mag > 1.0 + tol))

    if n_violations > 0:
        raise ValueError(
            f"Нарушена пассивность: {n_violations} точек имеют |H| > 1 + tol. "
            f"max_H_mag={max_H:.6g}"
        )

    return {
        "max_H_mag": max_H,
        "n_passivity_violations": n_violations,
    }


def _validate_no_duplicate_frequency_points(
    df: pd.DataFrame,
    freq_col: str,
) -> dict[str, int]:
    """
    Проверяет отсутствие дублей частотных точек внутри одной схемы.
    """
    _require_columns(df, ["param_set_id", freq_col])

    n_duplicates = int(
        df.duplicated(subset=["param_set_id", freq_col]).sum()
    )

    if n_duplicates > 0:
        raise ValueError(
            "Найдены дубли частотных точек для одной и той же схемы: "
            f"n_duplicates={n_duplicates}"
        )

    return {
        "n_duplicate_param_set_frequency_rows": n_duplicates,
    }


def validate_ac_dataset(
    df: pd.DataFrame,
    passive_tol: float = 1e-9,
    consistency_atol: float = 1e-10,
) -> dict[str, Any]:
    """
    Выполняет проверку датасета обычных фильтров dataset_ac.csv.

    Ожидаемые топологии:
        RC_LP, RC_HP, RL_LP, RL_HP, RLC_BP, RLC_NOTCH.
    """
    required_columns = [
        "topology",
        "param_set_id",
        "A_in",
        "f",
        "R",
        "C",
        "L",
        "Rload",
        "H_mag",
        "log_H_mag",
        "A_out",
        "phi_rad",
        "phi_sin",
        "phi_cos",
    ]

    _require_columns(df, required_columns)

    numeric_columns = [
        "A_in",
        "f",
        "H_mag",
        "log_H_mag",
        "A_out",
        "phi_rad",
        "phi_sin",
        "phi_cos",
    ]

    result: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_param_sets": int(df["param_set_id"].nunique()),
        "topologies": sorted(df["topology"].unique().tolist()),
    }

    result["finite"] = _check_finite(df, numeric_columns)
    result["log_h_mag"] = validate_log_h_mag(
        df,
        eps_log=EPS_LOG,
        atol=consistency_atol,
    )
    result["phase_sin_cos"] = validate_phase_sin_cos(
        df,
        atol=consistency_atol,
    )
    result["passivity"] = validate_passivity(
        df,
        tol=passive_tol,
    )
    result["duplicates"] = _validate_no_duplicate_frequency_points(
        df,
        freq_col="f",
    )

    return result


def validate_rc_ladder_dataset(
    df: pd.DataFrame,
    passive_tol: float = 1e-9,
    consistency_atol: float = 1e-10,
) -> dict[str, Any]:
    """
    Выполняет проверку датасета RC ladder dataset_rc_ladder.jsonl.

    В датасете каждая строка соответствует одной частотной точке
    для конкретной RC-цепочки.
    """
    required_columns = [
        "topology",
        "param_set_id",
        "n_sections",
        "sections",
        "Rload",
        "A_in",
        "frequency_hz",
        "H_mag",
        "log_H_mag",
        "A_out",
        "phi_rad",
        "phi_sin",
        "phi_cos",
    ]

    _require_columns(df, required_columns)

    numeric_columns = [
        "n_sections",
        "Rload",
        "A_in",
        "frequency_hz",
        "H_mag",
        "log_H_mag",
        "A_out",
        "phi_rad",
        "phi_sin",
        "phi_cos",
    ]

    result: dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_param_sets": int(df["param_set_id"].nunique()),
        "topologies": sorted(df["topology"].unique().tolist()),
        "n_sections_values": sorted(
            int(x) for x in df["n_sections"].dropna().unique().tolist()
        ),
    }

    result["finite"] = _check_finite(df, numeric_columns)
    result["log_h_mag"] = validate_log_h_mag(
        df,
        eps_log=EPS_LOG,
        atol=consistency_atol,
    )
    result["phase_sin_cos"] = validate_phase_sin_cos(
        df,
        atol=consistency_atol,
    )
    result["passivity"] = validate_passivity(
        df,
        tol=passive_tol,
    )
    result["duplicates"] = _validate_no_duplicate_frequency_points(
        df,
        freq_col="frequency_hz",
    )

    if set(result["topologies"]) != {"RC_LADDER"}:
        raise ValueError(
            f"RC ladder dataset должен содержать только topology='RC_LADDER', "
            f"получено: {result['topologies']}"
        )

    if min(result["n_sections_values"]) < 1:
        raise ValueError("n_sections должен быть >= 1")

    return result