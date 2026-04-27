# src/ml_circuits/features/filter_features.py

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_circuits.constants import EPS_F, EPS_LOG, EPS_NORM, OPEN_R, R_FLOOR_OHM
from ml_circuits.training.losses import safe_log


def _require_cols(df: pd.DataFrame, cols: list[str], fn_name: str):
    """Проверяет наличие необходимых столбцов в DataFrame."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{fn_name}: missing columns: {missing}")


def get_rload_eff(Rload: np.ndarray, open_r: float = OPEN_R):
    """
    Преобразует столбец Rload в форму, удобную для построения признаков.
    """
    Rload = np.asarray(Rload, dtype=float)
    has = (np.isfinite(Rload) & (Rload > 0.0)).astype(float)
    eff = np.where(has > 0.5, Rload, float(open_r))
    return has, eff


def make_features_rc_hp(df: pd.DataFrame):
    """
    Формирует признаки для топологии RC_HP
    (высокочастотный фильтр первого порядка без нагрузки).
    """
    if not set(df["topology"].unique()).issubset({"RC_HP"}):
        raise ValueError("make_features_rc_hp() expects only RC_HP")
    _require_cols(df, ["f", "R", "C"], "make_features_rc_hp")

    f = df["f"].to_numpy(dtype=float)
    R = df["R"].to_numpy(dtype=float)
    C = df["C"].to_numpy(dtype=float)

    f_safe = np.clip(f, EPS_F, None)

    log_R = safe_log(R)
    log_C = safe_log(C)

    tau = np.clip(R * C, EPS_NORM, None)
    fc = 1.0 / (2.0 * np.pi * tau)
    log10_f_norm = np.log10(f_safe / np.clip(fc, EPS_F, None))

    num_cols = ["log10_f_norm", "log_R", "log_C"]
    X = np.vstack([log10_f_norm, log_R, log_C]).T.astype(float, copy=False)

    if X.shape[1] != len(num_cols):
        raise ValueError("Feature matrix width does not match num_cols")
    return X, num_cols


def make_features_rc_lp(df: pd.DataFrame):
    """
    Формирует признаки для топологии RC_LP
    (низкочастотный фильтр первого порядка с возможной нагрузкой).
    """
    if not set(df["topology"].unique()).issubset({"RC_LP"}):
        raise ValueError("make_features_rc_lp() expects only RC_LP")
    _require_cols(df, ["f", "R", "C", "Rload"], "make_features_rc_lp")

    f = df["f"].to_numpy(dtype=float)
    R = df["R"].to_numpy(dtype=float)
    C = df["C"].to_numpy(dtype=float)
    Rload = df["Rload"].to_numpy(dtype=float)

    f_safe = np.clip(f, EPS_F, None)

    has_Rload, Rload_eff = get_rload_eff(Rload)

    log_R = safe_log(R)
    log_C = safe_log(C)
    log_Rload_eff = safe_log(Rload_eff)

    m_rc = (
        np.isfinite(f) & (f > 0.0) &
        np.isfinite(R) & (R > 0.0) &
        np.isfinite(C) & (C > 0.0)
    )
    m_loaded = m_rc & (has_Rload > 0.5)
    m_open = m_rc & (has_Rload <= 0.5)

    Rpar = np.full_like(R, np.nan, dtype=float)
    if np.any(m_rc):
        R_safe = np.clip(R[m_rc], R_FLOOR_OHM, None)
        Rl_safe = np.clip(Rload_eff[m_rc], R_FLOOR_OHM, None)
        Rpar[m_rc] = 1.0 / (1.0 / R_safe + 1.0 / Rl_safe)

    fc_loaded = np.full_like(R, np.nan, dtype=float)
    if np.any(m_rc):
        tau_eff = np.clip(Rpar[m_rc] * C[m_rc], EPS_NORM, None)
        fc_loaded[m_rc] = 1.0 / (2.0 * np.pi * tau_eff)

    log10_f_norm_loaded = np.full_like(f_safe, np.nan, dtype=float)
    if np.any(m_rc):
        log10_f_norm_loaded[m_rc] = np.log10(f_safe[m_rc] / np.clip(fc_loaded[m_rc], EPS_F, None))

    log_R_over_Rload_eff = np.full_like(log_R, np.nan, dtype=float)
    m_rr = m_loaded & np.isfinite(log_R) & np.isfinite(log_Rload_eff)
    log_R_over_Rload_eff[m_rr] = log_R[m_rr] - log_Rload_eff[m_rr]

    dc_gain = np.full_like(R, np.nan, dtype=float)
    if np.any(m_loaded):
        R_safe = np.clip(R[m_loaded], R_FLOOR_OHM, None)
        Rl_safe = np.clip(Rload_eff[m_loaded], R_FLOOR_OHM, None)
        dc_gain[m_loaded] = Rl_safe / (R_safe + Rl_safe)
    if np.any(m_open):
        dc_gain[m_open] = 1.0

    log_dc_gain = safe_log(dc_gain)
    if np.any(m_open):
        log_dc_gain[m_open] = 0.0

    w = 2.0 * np.pi * f_safe
    Zc_mag = 1.0 / np.clip(w * C, EPS_NORM, None)
    log_Zc_over_Rpar = safe_log(Zc_mag) - safe_log(Rpar)

    num_cols = [
        "log10_f_norm_loaded",
        "log_R", "log_C",
        "log_Rload_eff",
        "log_R_over_Rload_eff",
        "log_dc_gain",
        "log_Zc_over_Rpar",
        "has_Rload",
    ]
    X = np.vstack([
        log10_f_norm_loaded,
        log_R, log_C,
        log_Rload_eff,
        log_R_over_Rload_eff,
        log_dc_gain,
        log_Zc_over_Rpar,
        has_Rload,
    ]).T.astype(float, copy=False)

    if X.shape[1] != len(num_cols):
        raise ValueError("Feature matrix width does not match num_cols")
    return X, num_cols


def make_features_rl_hp(df: pd.DataFrame):
    """
    Формирует признаки для топологии RL_HP
    (высокочастотный фильтр первого порядка с возможной нагрузкой).
    """
    if not set(df["topology"].unique()).issubset({"RL_HP"}):
        raise ValueError("make_features_rl_hp() expects only RL_HP")
    _require_cols(df, ["f", "R", "L", "Rload"], "make_features_rl_hp")

    f = df["f"].to_numpy(dtype=float)
    R = df["R"].to_numpy(dtype=float)
    L = df["L"].to_numpy(dtype=float)
    Rload = df["Rload"].to_numpy(dtype=float)

    f_safe = np.clip(f, EPS_F, None)

    has_Rload, Rload_eff = get_rload_eff(Rload)

    log_R = safe_log(R)
    log_L = safe_log(L)
    log_Rload_eff = safe_log(Rload_eff)

    m_rl = (
        np.isfinite(f) & (f > 0.0) &
        np.isfinite(R) & (R > 0.0) &
        np.isfinite(L) & (L > 0.0)
    )
    m_loaded = m_rl & (has_Rload > 0.5)
    m_open = m_rl & (has_Rload <= 0.5)

    Req = np.full_like(R, np.nan, dtype=float)
    if np.any(m_rl):
        R_safe = np.clip(R[m_rl], R_FLOOR_OHM, None)
        Rl_safe = np.clip(Rload_eff[m_rl], R_FLOOR_OHM, None)
        Req[m_rl] = 1.0 / (1.0 / R_safe + 1.0 / Rl_safe)

    tau_eff = np.full_like(R, np.nan, dtype=float)
    if np.any(m_rl):
        tau_eff[m_rl] = L[m_rl] / np.clip(Req[m_rl], EPS_NORM, None)
    log_tau_rl_eff = safe_log(tau_eff)

    fc_req = np.full_like(R, np.nan, dtype=float)
    if np.any(m_rl):
        fc_req[m_rl] = Req[m_rl] / (2.0 * np.pi * np.clip(L[m_rl], EPS_NORM, None))
    log10_f_norm_req = np.full_like(f_safe, np.nan, dtype=float)
    if np.any(m_rl):
        log10_f_norm_req[m_rl] = np.log10(f_safe[m_rl] / np.clip(fc_req[m_rl], EPS_F, None))

    fc_r = np.full_like(R, np.nan, dtype=float)
    if np.any(m_rl):
        fc_r[m_rl] = R[m_rl] / (2.0 * np.pi * np.clip(L[m_rl], EPS_NORM, None))
    log10_f_norm_r = np.full_like(f_safe, np.nan, dtype=float)
    if np.any(m_rl):
        log10_f_norm_r[m_rl] = np.log10(f_safe[m_rl] / np.clip(fc_r[m_rl], EPS_F, None))

    m_rr = np.isfinite(R) & (R > 0.0) & np.isfinite(Rload_eff) & (Rload_eff > 0.0)
    log_R_over_Rload_eff = np.full_like(log_R, np.nan, dtype=float)
    if np.any(m_rr):
        log_R_over_Rload_eff[m_rr] = log_R[m_rr] - log_Rload_eff[m_rr]

    hf_gain = np.full_like(R, np.nan, dtype=float)
    if np.any(m_loaded):
        R_safe = np.clip(R[m_loaded], R_FLOOR_OHM, None)
        Rl_safe = np.clip(Rload_eff[m_loaded], R_FLOOR_OHM, None)
        hf_gain[m_loaded] = Rl_safe / (R_safe + Rl_safe)
    if np.any(m_open):
        hf_gain[m_open] = 1.0

    log_hf_gain = safe_log(hf_gain)
    if np.any(m_open):
        log_hf_gain[m_open] = 0.0

    num_cols = [
        "log10_f_norm_req",
        "log10_f_norm_r",
        "log_R", "log_L", "log_Rload_eff",
        "log_tau_rl_eff",
        "log_R_over_Rload_eff",
        "log_hf_gain",
        "has_Rload",
    ]

    X = np.vstack([
        log10_f_norm_req,
        log10_f_norm_r,
        log_R, log_L, log_Rload_eff,
        log_tau_rl_eff,
        log_R_over_Rload_eff,
        log_hf_gain,
        has_Rload,
    ]).T.astype(float, copy=False)

    if X.shape[1] != len(num_cols):
        raise ValueError("Feature matrix width does not match num_cols")
    return X, num_cols


def make_features_rl_lp(df: pd.DataFrame):
    """
    Формирует признаки для топологии RL_LP
    (низкочастотный фильтр первого порядка без нагрузки).
    """
    if not set(df["topology"].unique()).issubset({"RL_LP"}):
        raise ValueError("make_features_rl_lp() expects only RL_LP")
    _require_cols(df, ["f", "R", "L"], "make_features_rl_lp")

    f = df["f"].to_numpy(dtype=float)
    R = df["R"].to_numpy(dtype=float)
    L = df["L"].to_numpy(dtype=float)

    f_safe = np.clip(f, EPS_F, None)

    log_R = safe_log(R)
    log_L = safe_log(L)

    m_rl = (
        np.isfinite(f) & (f > 0.0) &
        np.isfinite(R) & (R > 0.0) &
        np.isfinite(L) & (L > 0.0)
    )

    tau = np.full_like(R, np.nan, dtype=float)
    if np.any(m_rl):
        tau[m_rl] = L[m_rl] / np.clip(R[m_rl], R_FLOOR_OHM, None)
    log_tau_rl = safe_log(tau)

    fc = np.full_like(R, np.nan, dtype=float)
    if np.any(m_rl):
        fc[m_rl] = R[m_rl] / (2.0 * np.pi * np.clip(L[m_rl], EPS_NORM, None))

    log10_f_norm = np.full_like(f_safe, np.nan, dtype=float)
    if np.any(m_rl):
        log10_f_norm[m_rl] = np.log10(f_safe[m_rl] / np.clip(fc[m_rl], EPS_F, None))

    num_cols = ["log10_f_norm", "log_R", "log_L", "log_tau_rl"]
    X = np.vstack([log10_f_norm, log_R, log_L, log_tau_rl]).T.astype(float, copy=False)

    if X.shape[1] != len(num_cols):
        raise ValueError("Feature matrix width does not match num_cols")
    return X, num_cols


def _r_parallel(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ Эквивалент параллельного соединения сопротивлений a || b """
    a = np.clip(a, R_FLOOR_OHM, None)
    b = np.clip(b, R_FLOOR_OHM, None)
    return 1.0 / (1.0 / a + 1.0 / b)


def make_features_rlc_bp(df: pd.DataFrame):
    """
    Компактные физически-информированные признаки для новой RLC_BP.
    """
    if not set(df["topology"].unique()).issubset({"RLC_BP"}):
        raise ValueError("make_features_rlc_bp() expects only RLC_BP")

    _require_cols(df, ["f", "R", "L", "C", "Rload"], "make_features_rlc_bp")

    f = df["f"].to_numpy(float)
    R = df["R"].to_numpy(float)
    L = df["L"].to_numpy(float)
    C = df["C"].to_numpy(float)
    Rload = df["Rload"].to_numpy(float)

    f_safe = np.clip(f, EPS_F, None)
    R_safe = np.clip(R, R_FLOOR_OHM, None)
    L_safe = np.clip(L, EPS_NORM, None)
    C_safe = np.clip(C, EPS_NORM, None)

    _, Rload_eff = get_rload_eff(Rload)
    Rload_eff_safe = np.clip(Rload_eff, R_FLOOR_OHM, None)

    if "f0" in df.columns:
        f0 = df["f0"].to_numpy(float)
    else:
        f0 = 1.0 / (2.0 * np.pi * np.sqrt(np.clip(L_safe * C_safe, EPS_NORM, None)))

    R_eq = _r_parallel(R_safe, Rload_eff_safe)

    if "Q_eff" in df.columns:
        Q_eff = df["Q_eff"].to_numpy(float)
    else:
        Q_eff = np.sqrt(np.clip(L_safe / C_safe, EPS_NORM, None)) / np.clip(R_eq, R_FLOOR_OHM, None)

    if "f1" in df.columns:
        f1 = df["f1"].to_numpy(float)
    else:
        alpha = np.sqrt(1.0 + 4.0 * Q_eff * Q_eff)
        f1 = f0 * (alpha - 1.0) / (2.0 * Q_eff)

    if "f2" in df.columns:
        f2 = df["f2"].to_numpy(float)
    else:
        alpha = np.sqrt(1.0 + 4.0 * Q_eff * Q_eff)
        f2 = f0 * (alpha + 1.0) / (2.0 * Q_eff)

    f0_safe = np.clip(f0, EPS_F, None)
    f1_safe = np.clip(f1, EPS_F, None)
    f2_safe = np.clip(f2, EPS_F, None)

    log10_f_norm = np.log10(f_safe / f0_safe)
    log10_f_over_f1 = np.log10(f_safe / f1_safe)
    log10_f_over_f2 = np.log10(f_safe / f2_safe)

    log_Q_eff = safe_log(np.clip(Q_eff, EPS_NORM, None))

    log_R_over_Rload = safe_log(R_safe) - safe_log(Rload_eff_safe)

    w = 2.0 * np.pi * f_safe
    X_lc = w * L_safe - 1.0 / np.clip(w * C_safe, EPS_NORM, None)
    abs_X_lc = np.abs(X_lc)

    log_abs_X_lc_over_R_eq = np.log(
        np.maximum(abs_X_lc / np.clip(R_eq, R_FLOOR_OHM, None), EPS_LOG)
    )

    num_cols = [
        "log10_f_norm",
        "log10_f_over_f1",
        "log10_f_over_f2",
        "log_Q_eff",
        "log_R_over_Rload",
        "log_abs_X_lc_over_R_eq",
    ]

    X = np.vstack([
        log10_f_norm,
        log10_f_over_f1,
        log10_f_over_f2,
        log_Q_eff,
        log_R_over_Rload,
        log_abs_X_lc_over_R_eq,
    ]).T.astype(float, copy=False)

    assert X.shape[1] == len(num_cols)

    if not np.isfinite(X).all():
        bad = np.where(~np.isfinite(X))
        raise ValueError(
            f"Non-finite features in make_features_rlc_bp: "
            f"rows={bad[0][:10]}, cols={bad[1][:10]}"
        )

    return X, num_cols


def make_features_rlc_notch(df: pd.DataFrame):
    """
    Компактные физически-информированные признаки для RLC_NOTCH.
    """
    if not set(df["topology"].unique()).issubset({"RLC_NOTCH"}):
        raise ValueError("make_features_rlc_notch() expects only RLC_NOTCH")

    _require_cols(df, ["f", "R", "L", "C", "Rload"], "make_features_rlc_notch")

    f = df["f"].to_numpy(float)
    R = df["R"].to_numpy(float)
    L = df["L"].to_numpy(float)
    C = df["C"].to_numpy(float)
    Rload = df["Rload"].to_numpy(float)

    f_safe = np.clip(f, EPS_F, None)
    R_safe = np.clip(R, R_FLOOR_OHM, None)
    L_safe = np.clip(L, EPS_NORM, None)
    C_safe = np.clip(C, EPS_NORM, None)

    _, Rload_eff = get_rload_eff(Rload)
    Rload_eff_safe = np.clip(Rload_eff, R_FLOOR_OHM, None)

    if "f0" in df.columns:
        f0 = df["f0"].to_numpy(float)
    else:
        f0 = 1.0 / (
            2.0 * np.pi * np.sqrt(np.clip(L_safe * C_safe, EPS_NORM, None))
        )

    R_eq = _r_parallel(R_safe, Rload_eff_safe)
    R_eq_safe = np.clip(R_eq, R_FLOOR_OHM, None)

    if "Q_eff" in df.columns:
        Q_eff = df["Q_eff"].to_numpy(float)
    else:
        Q_eff = np.sqrt(
            np.clip(L_safe / C_safe, EPS_NORM, None)
        ) / R_eq_safe

    if "f1" in df.columns:
        f1 = df["f1"].to_numpy(float)
    else:
        alpha = np.sqrt(1.0 + 4.0 * Q_eff * Q_eff)
        f1 = f0 * (alpha - 1.0) / (2.0 * Q_eff)

    if "f2" in df.columns:
        f2 = df["f2"].to_numpy(float)
    else:
        alpha = np.sqrt(1.0 + 4.0 * Q_eff * Q_eff)
        f2 = f0 * (alpha + 1.0) / (2.0 * Q_eff)

    f0_safe = np.clip(f0, EPS_F, None)
    f1_safe = np.clip(f1, EPS_F, None)
    f2_safe = np.clip(f2, EPS_F, None)

    log10_f_norm = np.log10(f_safe / f0_safe)
    log10_f_over_f1 = np.log10(f_safe / f1_safe)
    log10_f_over_f2 = np.log10(f_safe / f2_safe)

    log_Q_eff = safe_log(np.clip(Q_eff, EPS_NORM, None))

    log_R_over_Rload = safe_log(R_safe) - safe_log(Rload_eff_safe)

    pass_gain = Rload_eff_safe / (R_safe + Rload_eff_safe)
    log_pass_gain = safe_log(pass_gain)

    w = 2.0 * np.pi * f_safe
    X_lc = w * L_safe - 1.0 / np.clip(w * C_safe, EPS_NORM, None)
    abs_X_lc = np.abs(X_lc)

    log_abs_X_lc_over_R_eq = np.log(
        np.maximum(abs_X_lc / R_eq_safe, EPS_LOG)
    )

    num_cols = [
        "log10_f_norm",
        "log10_f_over_f1",
        "log10_f_over_f2",
        "log_Q_eff",
        "log_pass_gain",
        "log_R_over_Rload",
        "log_abs_X_lc_over_R_eq",
    ]

    X = np.vstack([
        log10_f_norm,
        log10_f_over_f1,
        log10_f_over_f2,
        log_Q_eff,
        log_pass_gain,
        log_R_over_Rload,
        log_abs_X_lc_over_R_eq,
    ]).T.astype(float, copy=False)

    assert X.shape[1] == len(num_cols)

    if not np.isfinite(X).all():
        bad = np.where(~np.isfinite(X))
        raise ValueError(
            f"Non-finite features in make_features_rlc_notch: "
            f"rows={bad[0][:10]}, cols={bad[1][:10]}"
        )

    return X, num_cols


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
