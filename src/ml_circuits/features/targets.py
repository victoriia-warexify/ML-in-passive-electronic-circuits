# src/ml_circuits/features/targets.py

"""
Формирование целевых переменных для обучения моделей фильтров.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml_circuits.constants import EPS_DIV, EPS_LOG


def make_targets_logmag_phi(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """
    Формирует основной вектор целевых переменных.

    Target:
        logH    = ln(max(A_out / A_in, EPS_LOG));
        phi_sin = sin(phi);
        phi_cos = cos(phi).
    """
    A_out = df["A_out"].to_numpy(dtype=float)
    A_in = df["A_in"].to_numpy(dtype=float)

    H_mag = A_out / np.clip(A_in, EPS_DIV, None)
    logH = np.log(np.clip(H_mag, EPS_LOG, None))

    phi_sin = df["phi_sin"].to_numpy(dtype=float)
    phi_cos = df["phi_cos"].to_numpy(dtype=float)

    Y = np.column_stack([logH, phi_sin, phi_cos])
    names = ["logH", "phi_sin", "phi_cos"]

    return Y, names


TARGETS_ORDER1 = {
    "RC_HP": make_targets_logmag_phi,
    "RC_LP": make_targets_logmag_phi,
    "RL_HP": make_targets_logmag_phi,
    "RL_LP": make_targets_logmag_phi,
}


TARGETS_RLC = {
    "RLC_BP": make_targets_logmag_phi,
    "RLC_NOTCH": make_targets_logmag_phi,
}