from __future__ import annotations

import numpy as np
import pandas as pd

from ml_circuits.circuits.analytical import analytical_H_for_df
from ml_circuits.constants import EPS_LOG


def run_analytical_oracle_check(df: pd.DataFrame, *, eps_log: float = EPS_LOG):
    """
    Сравнивает dataset_ac.csv с аналитическими формулами.
    Это oracle/sanity-check, а не ML baseline.
    """
    H_oracle = analytical_H_for_df(df)

    H_mag_oracle = np.abs(H_oracle)
    log_H_oracle = np.log(np.clip(H_mag_oracle, eps_log, None))

    H_mag_true = df["H_mag"].to_numpy(float)
    log_H_true = df["log_H_mag"].to_numpy(float)

    err_H = H_mag_oracle - H_mag_true
    err_logH = log_H_oracle - log_H_true

    abs_err_H = np.abs(err_H)
    abs_err_logH = np.abs(err_logH)

    print("=== Analytical oracle / sanity-check ===")
    print("max |H_mag_oracle - H_mag|:", np.nanmax(abs_err_H))

    m_log = np.isfinite(abs_err_logH)
    if np.any(m_log):
        print("median |logH_oracle - log_H_mag|:", np.nanmedian(abs_err_logH[m_log]))
        print("99% |logH_oracle - log_H_mag|    :", np.nanquantile(abs_err_logH[m_log], 0.99))

    m_log_safe = m_log & np.isfinite(H_mag_true) & (H_mag_true > 1e-10)
    if np.any(m_log_safe):
        print(
            "max |logH_oracle - log_H_mag| for H_mag > 1e-10:",
            np.nanmax(abs_err_logH[m_log_safe]),
        )

    print("\n--- By topology: max |H_mag_oracle - H_mag| ---")
    tmp = df[["topology"]].copy()
    tmp["abs_err_H"] = abs_err_H
    print(tmp.groupby("topology")["abs_err_H"].max())

    max_abs_H_err = np.nanmax(abs_err_H)
    if max_abs_H_err > 1e-9:
        raise ValueError(
            f"Analytical oracle check failed: max |H error| = {max_abs_H_err}"
        )

    return {
        "H_mag_oracle": H_mag_oracle,
        "log_H_oracle": log_H_oracle,
    }
