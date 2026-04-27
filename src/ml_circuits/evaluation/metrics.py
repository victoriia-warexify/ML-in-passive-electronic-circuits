from __future__ import annotations

import numpy as np

from ml_circuits.constants import AMP_GATE, EPS_NORM
from ml_circuits.training.losses import phase_mae
from ml_circuits.training.weights import make_group_weights


def _wmean_abs(err: np.ndarray, w: np.ndarray) -> float:
    """Вычисляет взвешенное среднее абсолютной ошибки |err|."""
    m = np.isfinite(err) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return float("nan")
    ww = w[m]
    return float(np.sum(ww * np.abs(err[m])) / np.sum(ww))


def _wrmse(err: np.ndarray, w: np.ndarray) -> float:
    """Вычисляет взвешенную RMSE по ошибке err."""
    m = np.isfinite(err) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return float("nan")
    ww = w[m]
    return float(np.sqrt(np.sum(ww * (err[m] ** 2)) / np.sum(ww)))


def phase_diff(phi_true: np.ndarray, phi_pred: np.ndarray) -> np.ndarray:
    """Вычисляет разность фаз с учётом цикличности в диапазоне [-π, π)."""
    d = np.asarray(phi_pred, float) - np.asarray(phi_true, float)
    return (d + np.pi) % (2.0 * np.pi) - np.pi


def evaluate_logmag_phi_model(
    model,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    A_out_true: np.ndarray,
    A_in: np.ndarray,
    *,
    meta_test: dict | None = None,
    eps_norm: float = EPS_NORM,
    phase_filter_Hthr: float = 0.05,
    w_eval: np.ndarray | None = None,
    verbose: bool = False,
):
    """
    Оценивает модель по трём выходам: log|H|, sin(phi), cos(phi).
    """
    X_test = np.asarray(X_test, dtype=float)
    Y_test = np.asarray(Y_test, dtype=float)
    A_out_true = np.asarray(A_out_true, dtype=float)
    A_in = np.asarray(A_in, dtype=float)

    if X_test.ndim != 2:
        raise ValueError("X_test must have shape (n_samples, n_features)")
    if Y_test.ndim != 2 or Y_test.shape[1] != 3:
        raise ValueError("Y_test must have shape (n_samples, 3)")
    if Y_test.shape[0] != X_test.shape[0]:
        raise ValueError("X_test and Y_test must contain the same number of samples")
    if A_out_true.ndim != 1 or A_out_true.shape[0] != X_test.shape[0]:
        raise ValueError("A_out_true must have shape (n_samples,)")
    if A_in.ndim != 1 or A_in.shape[0] != X_test.shape[0]:
        raise ValueError("A_in must have shape (n_samples,)")

    pred = np.asarray(model.predict(X_test), dtype=float)
    if pred.shape != Y_test.shape:
        raise ValueError("model.predict(X_test) must have shape equal to Y_test.shape")

    logH_true = np.asarray(Y_test[:, 0], dtype=float)
    sin_true = np.asarray(Y_test[:, 1], dtype=float)
    cos_true = np.asarray(Y_test[:, 2], dtype=float)

    logH_pred = pred[:, 0]
    sin_pred = pred[:, 1]
    cos_pred = pred[:, 2]

    if meta_test is not None:
        if "log_hf_gain" in meta_test:
            g = np.asarray(meta_test["log_hf_gain"], float)
            if g.shape[0] != X_test.shape[0]:
                raise ValueError("meta_test['log_hf_gain'] must have shape (n_samples,)")
            logH_true = logH_true + g
            logH_pred = logH_pred + g
        if "log_dc_gain" in meta_test:
            g = np.asarray(meta_test["log_dc_gain"], float)
            if g.shape[0] != X_test.shape[0]:
                raise ValueError("meta_test['log_dc_gain'] must have shape (n_samples,)")
            logH_true = logH_true + g
            logH_pred = logH_pred + g

    m_phi_true = np.isfinite(sin_true) & np.isfinite(cos_true)
    phi_true = np.full_like(logH_true, np.nan, dtype=float)
    phi_true[m_phi_true] = np.arctan2(sin_true[m_phi_true], cos_true[m_phi_true])

    r = np.sqrt(sin_pred * sin_pred + cos_pred * cos_pred)
    r = np.where(np.isfinite(r) & (r > eps_norm), r, np.nan)
    sin_pred_n = sin_pred / r
    cos_pred_n = cos_pred / r

    m_phi_pred = np.isfinite(sin_pred_n) & np.isfinite(cos_pred_n)
    phi_pred = np.full_like(logH_pred, np.nan, dtype=float)
    phi_pred[m_phi_pred] = np.arctan2(sin_pred_n[m_phi_pred], cos_pred_n[m_phi_pred])

    H_mag_pred = np.exp(logH_pred)
    A_pred = H_mag_pred * A_in

    if w_eval is not None:
        w_eval = np.asarray(w_eval, float)
        if w_eval.ndim != 1 or w_eval.shape[0] != X_test.shape[0]:
            raise ValueError("w_eval must have shape (n_samples,)")

    err_logH = logH_pred - logH_true
    if w_eval is None:
        m_logH = np.isfinite(err_logH)
        logH_mae = float(np.mean(np.abs(err_logH[m_logH]))) if np.any(m_logH) else float("nan")
        logH_rmse = float(np.sqrt(np.mean(err_logH[m_logH] ** 2))) if np.any(m_logH) else float("nan")
    else:
        logH_mae = _wmean_abs(err_logH, w_eval)
        logH_rmse = _wrmse(err_logH, w_eval)

    db_factor = 20.0 / np.log(10.0)
    err_db = err_logH * db_factor
    if w_eval is None:
        m_db = np.isfinite(err_db)
        db_mae = float(np.mean(np.abs(err_db[m_db]))) if np.any(m_db) else float("nan")
        db_rmse = float(np.sqrt(np.mean(err_db[m_db] ** 2))) if np.any(m_db) else float("nan")
    else:
        db_mae = _wmean_abs(err_db, w_eval)
        db_rmse = _wrmse(err_db, w_eval)

    err_A = A_pred - A_out_true
    if w_eval is None:
        m_A = np.isfinite(err_A)
        A_mae = float(np.mean(np.abs(err_A[m_A]))) if np.any(m_A) else float("nan")
        A_rmse = float(np.sqrt(np.mean(err_A[m_A] ** 2))) if np.any(m_A) else float("nan")
    else:
        A_mae = _wmean_abs(err_A, w_eval)
        A_rmse = _wrmse(err_A, w_eval)

    m_phi = np.isfinite(phi_true) & np.isfinite(phi_pred)
    if np.any(m_phi):
        if w_eval is None:
            phi_mae_rad = phase_mae(phi_true[m_phi], phi_pred[m_phi])
        else:
            ww = w_eval[m_phi]
            sw = float(np.sum(ww))
            if sw > 0:
                phi_err = phase_diff(phi_true[m_phi], phi_pred[m_phi])
                phi_mae_rad = float(np.sum(ww * np.abs(phi_err)) / sw)
            else:
                phi_mae_rad = float("nan")
    else:
        phi_mae_rad = float("nan")
    phi_mae_deg = float(phi_mae_rad * 180.0 / np.pi) if np.isfinite(phi_mae_rad) else float("nan")

    H_true = np.exp(logH_true)
    m_f = m_phi & np.isfinite(H_true) & (H_true > phase_filter_Hthr)
    if np.any(m_f):
        if w_eval is None:
            phi_mae_f_rad = phase_mae(phi_true[m_f], phi_pred[m_f])
        else:
            ww = w_eval[m_f]
            sw = float(np.sum(ww))
            if sw > 0:
                phi_err_f = phase_diff(phi_true[m_f], phi_pred[m_f])
                phi_mae_f_rad = float(np.sum(ww * np.abs(phi_err_f)) / sw)
            else:
                phi_mae_f_rad = float("nan")
    else:
        phi_mae_f_rad = float("nan")
    phi_mae_f_deg = float(phi_mae_f_rad * 180.0 / np.pi) if np.isfinite(phi_mae_f_rad) else float("nan")

    if verbose:
        print("\n=== Custom GBDT (log|H| + sin/cos) ===")
        print(f"logH_MAE             : {logH_mae:.6f}")
        print(f"logH_RMSE            : {logH_rmse:.6f}")
        print(f"dB_MAE               : {db_mae:.6f}")
        print(f"dB_RMSE              : {db_rmse:.6f}")
        print(f"A_MAE                : {A_mae:.6f}")
        print(f"A_RMSE               : {A_rmse:.6f}")
        print(f"phi_MAE(deg)         : {phi_mae_deg:.6f}")
        print(f"phi_MAE_filtered(deg): {phi_mae_f_deg:.6f}")

    return {
        "logH_MAE": logH_mae,
        "logH_RMSE": logH_rmse,
        "dB_MAE": db_mae,
        "dB_RMSE": db_rmse,
        "A_MAE": A_mae,
        "A_RMSE": A_rmse,
        "phi_MAE_deg": phi_mae_deg,
        "phi_MAE_filtered_deg": phi_mae_f_deg,
    }


def metrics_one(df_test, model, X_test, Y_test, meta_test=None, amp_gate=AMP_GATE, use_group_weights=False):
    """
    Вычисляет набор тестовых метрик для одной модели.
    """
    w_te = None
    if use_group_weights:
        w_te = make_group_weights(df_test, group_col="param_set_id")

    kwargs = dict(
        model=model,
        X_test=X_test,
        Y_test=Y_test,
        A_out_true=df_test["A_out"].to_numpy(float),
        A_in=df_test["A_in"].to_numpy(float),
        meta_test=meta_test,
        phase_filter_Hthr=amp_gate,
    )
    if w_te is not None:
        kwargs["w_eval"] = w_te

    d = evaluate_logmag_phi_model(**kwargs)
    d["n"] = int(len(df_test))
    return d
