# src/ml_circuits/features/targets.py

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_circuits.constants import EPS_DIV, EPS_LOG, EPS_NORM, R_FLOOR_OHM
from ml_circuits.features.filter_features import _require_cols, get_rload_eff
from ml_circuits.training.losses import safe_log


def make_targets_logmag_phi(df: pd.DataFrame):
    """
    Формирует базовые таргеты для обучения.
    """
    _require_cols(df, ["A_out", "A_in", "phi_sin", "phi_cos"], "make_targets_logmag_phi")

    A_out = df["A_out"].to_numpy(float)
    A_in = df["A_in"].to_numpy(float)

    logH = np.log(np.clip(A_out / np.clip(A_in, EPS_DIV, None), EPS_LOG, None))

    s = df["phi_sin"].to_numpy(float)
    c = df["phi_cos"].to_numpy(float)
    r = np.sqrt(s * s + c * c)
    m = np.isfinite(r) & (r > EPS_NORM)

    s_norm = np.full_like(s, np.nan)
    c_norm = np.full_like(c, np.nan)
    s_norm[m] = s[m] / r[m]
    c_norm[m] = c[m] / r[m]

    Y = np.vstack([logH, s_norm, c_norm]).T.astype(float, copy=False)
    meta = {}
    return Y, meta


def make_targets_rl_hp_residual(df: pd.DataFrame):
    """
    Формирует residual-таргет для RL_HP.
    """
    _require_cols(df, ["A_out", "A_in", "phi_sin", "phi_cos", "R", "Rload"], "make_targets_rl_hp_residual")

    A_out = df["A_out"].to_numpy(float)
    A_in = df["A_in"].to_numpy(float)
    logH = np.log(np.clip(A_out / np.clip(A_in, EPS_DIV, None), EPS_LOG, None))

    s = df["phi_sin"].to_numpy(float)
    c = df["phi_cos"].to_numpy(float)
    r = np.sqrt(s * s + c * c)
    m = np.isfinite(r) & (r > EPS_NORM)

    s_norm = np.full_like(s, np.nan)
    c_norm = np.full_like(c, np.nan)
    s_norm[m] = s[m] / r[m]
    c_norm[m] = c[m] / r[m]

    R = df["R"].to_numpy(float)
    Rload = df["Rload"].to_numpy(float)
    has, Rload_eff = get_rload_eff(Rload)

    hf_gain = np.ones_like(R, dtype=float)
    m_loaded = (has > 0.5) & np.isfinite(R) & (R > 0.0)
    if np.any(m_loaded):
        R_safe = np.clip(R[m_loaded], R_FLOOR_OHM, None)
        Rl_safe = np.clip(Rload_eff[m_loaded], R_FLOOR_OHM, None)
        hf_gain[m_loaded] = Rl_safe / (R_safe + Rl_safe)

    log_hf_gain = safe_log(hf_gain)

    m_open = has <= 0.5
    if np.any(m_open):
        log_hf_gain[m_open] = 0.0

    logH_res = logH - log_hf_gain

    Y = np.vstack([logH_res, s_norm, c_norm]).T.astype(float, copy=False)
    meta = {"log_hf_gain": log_hf_gain}
    return Y, meta


def make_targets_rc_lp_residual(df: pd.DataFrame):
    """
    Формирует residual-таргет для RC_LP.
    """
    _require_cols(df, ["A_out", "A_in", "phi_sin", "phi_cos", "R", "Rload"], "make_targets_rc_lp_residual")

    A_out = df["A_out"].to_numpy(float)
    A_in = df["A_in"].to_numpy(float)
    logH = np.log(np.clip(A_out / np.clip(A_in, EPS_DIV, None), EPS_LOG, None))

    s = df["phi_sin"].to_numpy(float)
    c = df["phi_cos"].to_numpy(float)
    r = np.sqrt(s * s + c * c)
    m = np.isfinite(r) & (r > EPS_NORM)

    s_norm = np.full_like(s, np.nan)
    c_norm = np.full_like(c, np.nan)
    s_norm[m] = s[m] / r[m]
    c_norm[m] = c[m] / r[m]

    R = df["R"].to_numpy(float)
    Rload = df["Rload"].to_numpy(float)
    has, Rload_eff = get_rload_eff(Rload)

    dc_gain = np.ones_like(R, dtype=float)
    m_loaded = (has > 0.5) & np.isfinite(R) & (R > 0.0)
    if np.any(m_loaded):
        R_safe = np.clip(R[m_loaded], R_FLOOR_OHM, None)
        Rl_safe = np.clip(Rload_eff[m_loaded], R_FLOOR_OHM, None)
        dc_gain[m_loaded] = Rl_safe / (R_safe + Rl_safe)

    log_dc_gain = safe_log(dc_gain)

    m_open = has <= 0.5
    if np.any(m_open):
        log_dc_gain[m_open] = 0.0

    logH_res = logH - log_dc_gain

    Y = np.vstack([logH_res, s_norm, c_norm]).T.astype(float, copy=False)
    meta = {"log_dc_gain": log_dc_gain}
    return Y, meta


TARGETS_ORDER1 = {
    "RC_HP": make_targets_logmag_phi,
    "RC_LP": make_targets_rc_lp_residual,
    "RL_HP": make_targets_rl_hp_residual,
    "RL_LP": make_targets_logmag_phi,
}


TARGETS_RLC = {
    "RLC_BP": make_targets_logmag_phi,
    "RLC_NOTCH": make_targets_logmag_phi,
}
