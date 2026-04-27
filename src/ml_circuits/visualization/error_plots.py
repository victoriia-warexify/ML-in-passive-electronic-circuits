from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ml_circuits.constants import EPS_F, EPS_NORM


def predict_full_outputs(
    df_test: pd.DataFrame,
    X_test: np.ndarray,
    model,
    meta_test: dict | None = None,
) -> dict:
    """
    Восстанавливает физические выходные величины модели на тестовой выборке.
    """
    X_test = np.asarray(X_test, dtype=float)
    pred = np.asarray(model.predict(X_test), dtype=float)

    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError("model.predict(X_test) must have shape (n_samples, 3)")

    logH_pred = pred[:, 0].copy()
    sin_pred = pred[:, 1]
    cos_pred = pred[:, 2]

    if meta_test is not None:
        if "log_hf_gain" in meta_test:
            g = np.asarray(meta_test["log_hf_gain"], dtype=float)
            if g.shape[0] != X_test.shape[0]:
                raise ValueError("meta_test['log_hf_gain'] must have shape (n_samples,)")
            logH_pred = logH_pred + g

        if "log_dc_gain" in meta_test:
            g = np.asarray(meta_test["log_dc_gain"], dtype=float)
            if g.shape[0] != X_test.shape[0]:
                raise ValueError("meta_test['log_dc_gain'] must have shape (n_samples,)")
            logH_pred = logH_pred + g

    H_mag_pred = np.exp(logH_pred)

    A_in = df_test["A_in"].to_numpy(float)
    if A_in.shape[0] != X_test.shape[0]:
        raise ValueError("df_test['A_in'] must have length n_samples")

    A_out_pred = H_mag_pred * A_in

    r = np.sqrt(sin_pred * sin_pred + cos_pred * cos_pred)
    r = np.where(np.isfinite(r) & (r > EPS_NORM), r, np.nan)
    sin_pred_n = sin_pred / r
    cos_pred_n = cos_pred / r

    phi_pred = np.full_like(logH_pred, np.nan, dtype=float)
    m_phi = np.isfinite(sin_pred_n) & np.isfinite(cos_pred_n)
    phi_pred[m_phi] = np.arctan2(sin_pred_n[m_phi], cos_pred_n[m_phi])

    return {
        "logH_pred": logH_pred,
        "H_mag_pred": H_mag_pred,
        "A_out_pred": A_out_pred,
        "phi_pred": phi_pred,
    }


def _get_norm_frequency(df_plot: pd.DataFrame, topo: str) -> np.ndarray:
    """
    Возвращает log10(f / f*), где f* — характерная частота схемы.
    """
    f = df_plot["f"].to_numpy(float)

    if topo in ("RC_HP", "RC_LP", "RL_LP", "RL_HP"):
        if "fc_loaded" in df_plot.columns and df_plot["fc_loaded"].notna().any():
            f_star = df_plot["fc_loaded"].to_numpy(float)
        else:
            if topo in ("RC_HP", "RC_LP"):
                R = df_plot["R"].to_numpy(float)
                C = df_plot["C"].to_numpy(float)
                f_star = 1.0 / (2.0 * np.pi * np.clip(R * C, EPS_NORM, None))
            else:
                R = df_plot["R"].to_numpy(float)
                L = df_plot["L"].to_numpy(float)
                f_star = R / (2.0 * np.pi * np.clip(L, EPS_NORM, None))

    elif topo in ("RLC_BP", "RLC_NOTCH"):
        if "f0" in df_plot.columns and df_plot["f0"].notna().any():
            f_star = df_plot["f0"].to_numpy(float)
        else:
            L = df_plot["L"].to_numpy(float)
            C = df_plot["C"].to_numpy(float)
            f_star = 1.0 / (2.0 * np.pi * np.sqrt(np.clip(L * C, EPS_NORM, None)))

    else:
        raise ValueError(f"Unsupported topology: {topo}")

    return np.log10(np.clip(f, EPS_F, None) / np.clip(f_star, EPS_F, None))


def plot_error_vs_norm_freq(
    df_test: pd.DataFrame,
    X_test: np.ndarray,
    model,
    *,
    meta_test: dict | None = None,
    topo: str,
    n_bins: int = 30,
):
    """
    Строит график средней абсолютной ошибки в dB как функции нормированной частоты.
    """
    pred = predict_full_outputs(df_test, X_test, model, meta_test=meta_test)

    x = _get_norm_frequency(df_test, topo=topo)
    logH_true = df_test["log_H_mag"].to_numpy(float)

    err_db = np.abs((pred["logH_pred"] - logH_true) * (20.0 / np.log(10.0)))

    m = np.isfinite(x) & np.isfinite(err_db)
    if not np.any(m):
        raise ValueError("No valid points for plotting error vs normalized frequency")

    x = x[m]
    err_db = err_db[m]

    bins = np.linspace(np.min(x), np.max(x), n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    values = np.full(n_bins, np.nan, dtype=float)

    for i in range(n_bins):
        mb = (x >= bins[i]) & (x < bins[i + 1] if i < n_bins - 1 else x <= bins[i + 1])
        if np.any(mb):
            values[i] = np.mean(err_db[mb])

    plt.figure(figsize=(7, 4.5))
    plt.plot(centers, values, marker="o")
    plt.xlabel(r"$\log_{10}(f / f_*)$")
    plt.ylabel("Средняя абсолютная ошибка, dB")
    plt.title(f"{topo}: ошибка как функция нормированной частоты")
    plt.grid(True)
    plt.show()


def plot_rlc_resonance_families(
    df_test: pd.DataFrame,
    X_test: np.ndarray,
    model,
    *,
    meta_test: dict | None = None,
    topo: str,
    n_curves: int = 3,
):
    """
    Строит семейства истинных и предсказанных резонансных кривых для RLC-топологий.
    """
    if topo not in ("RLC_BP", "RLC_NOTCH"):
        raise ValueError("plot_rlc_resonance_families supports only RLC_BP and RLC_NOTCH")

    pred = predict_full_outputs(df_test, X_test, model, meta_test=meta_test)

    d = df_test.copy()
    d = d.assign(
        logH_pred=pred["logH_pred"],
        H_mag_pred=pred["H_mag_pred"],
    )

    if "Q_eff" in d.columns and d["Q_eff"].notna().any():
        group_q = (
            d.groupby("param_set_id")["Q_eff"]
            .median()
            .dropna()
        )

        q_med = group_q.median()

        unique_ids = (
            (group_q - q_med)
            .abs()
            .sort_values()
            .index
            .tolist()[:n_curves]
        )
    else:
        unique_ids = d["param_set_id"].drop_duplicates().tolist()[:n_curves]

    for pid in unique_ids:
        sub = d[d["param_set_id"] == pid].copy()
        if len(sub) < 5:
            continue

        L = sub["L"].to_numpy(float)
        C = sub["C"].to_numpy(float)
        f = sub["f"].to_numpy(float)

        f0 = 1.0 / (2.0 * np.pi * np.sqrt(np.clip(L * C, EPS_NORM, None)))
        x = np.log10(np.clip(f, EPS_F, None) / np.clip(f0, EPS_F, None))

        y_true = 20.0 * sub["log_H_mag"].to_numpy(float) / np.log(10.0)
        y_pred = 20.0 * sub["logH_pred"].to_numpy(float) / np.log(10.0)

        order = np.argsort(x)
        x = x[order]
        y_true = y_true[order]
        y_pred = y_pred[order]

        plt.figure(figsize=(7, 4.5))
        plt.plot(x, y_true, label="Истинная кривая")
        plt.plot(x, y_pred, label="Предсказанная кривая")
        plt.xlabel(r"$\log_{10}(f / f_0)$")
        plt.ylabel(r"$20 \log_{10}|H|$, dB")
        plt.title(f"{topo}: резонансная кривая, param_set_id={pid}")
        plt.grid(True)
        plt.legend()
        plt.show()
