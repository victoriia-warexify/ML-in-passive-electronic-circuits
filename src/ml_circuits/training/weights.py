from __future__ import annotations

import numpy as np
import pandas as pd

from ml_circuits.constants import AMP_GATE


def make_group_weights(df: pd.DataFrame, group_col: str = "param_set_id") -> np.ndarray:
    """
    Формирует веса строк таким образом, чтобы каждая группа вносила одинаковый вклад
    в функцию потерь или в итоговую метрику.
    """
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found in df")
    if df[group_col].isna().any():
        raise ValueError(f"{group_col} contains NaN")

    counts = df[group_col].value_counts()
    w = 1.0 / df[group_col].map(counts).to_numpy(float)
    return w


def make_output_weights_for_logmag_phase(
    df_part: pd.DataFrame,
    base_w: np.ndarray,
    *,
    amp_gate: float = AMP_GATE,
) -> np.ndarray:
    """
    Формирует разные веса для трёх выходов модели:

        output 0: log|H|  — обучается на всех точках;
        output 1: sin(phi) — обучается только при H_mag >= amp_gate;
        output 2: cos(phi) — обучается только при H_mag >= amp_gate.
    """
    if "H_mag" not in df_part.columns:
        raise KeyError("Column 'H_mag' is required for phase masking")

    base_w = np.asarray(base_w, dtype=float)

    if base_w.ndim != 1:
        raise ValueError("base_w must be 1D")

    if base_w.shape[0] != len(df_part):
        raise ValueError(
            f"base_w length mismatch: got {base_w.shape[0]}, expected {len(df_part)}"
        )

    if np.any(~np.isfinite(base_w)) or np.any(base_w < 0.0):
        raise ValueError("base_w must be finite and non-negative")

    H = df_part["H_mag"].to_numpy(float)

    phase_ok = (
        np.isfinite(H)
        & (H >= float(amp_gate))
    )

    w_out = np.zeros((len(df_part), 3), dtype=float)
    w_out[:, 0] = base_w
    w_out[phase_ok, 1] = base_w[phase_ok]
    w_out[phase_ok, 2] = base_w[phase_ok]

    if w_out[:, 0].sum() <= 0:
        raise ValueError("Zero total weight for log|H| output")

    if w_out[:, 1].sum() <= 0 or w_out[:, 2].sum() <= 0:
        raise ValueError(
            "Zero total weight for phase outputs. "
            "amp_gate is probably too high for this subset."
        )

    return w_out
