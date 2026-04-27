from __future__ import annotations

from typing import Optional

import numpy as np

from ml_circuits.constants import EPS_LOG


def wrap_to_pi(phi: np.ndarray) -> np.ndarray:
    """Нормирует углы к диапазону (-pi, pi]."""
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    phi = np.where(phi <= -np.pi, phi + 2 * np.pi, phi)
    return phi


def phase_mae(y_true: np.ndarray, y_pred: np.ndarray, *, units: str = "rad") -> float:
    """Вычисляет среднюю абсолютную ошибку фазы с учётом цикличности по модулю 2π."""
    if units not in {"rad", "deg"}:
        raise ValueError("units must be 'rad' or 'deg'")
    d = wrap_to_pi(np.asarray(y_pred, float) - np.asarray(y_true, float))
    d = d[np.isfinite(d)]
    if d.size == 0:
        return float("nan")
    mae = float(np.mean(np.abs(d)))
    if units == "deg":
        mae *= 180.0 / np.pi
    return mae


def safe_log(x: np.ndarray, eps: float = EPS_LOG) -> np.ndarray:
    """Возвращает ln(max(x, eps)) для конечных положительных x; для остальных значений — NaN."""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan, dtype=float)
    m = np.isfinite(x) & (x > 0.0)
    out[m] = np.log(np.maximum(x[m], eps))
    return out


def fit_median_imputer(X: np.ndarray) -> np.ndarray:
    """Оценивает медианы по столбцам, игнорируя NaN и бесконечные значения."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    X = np.where(np.isfinite(X), X, np.nan)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return med.astype(float, copy=False)


def transform_median_imputer(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    """Заменяет NaN и бесконечные значения медианами соответствующих столбцов."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    med = np.asarray(med, dtype=float)
    if med.ndim != 1 or med.shape[0] != X.shape[1]:
        raise ValueError(f"Median shape {med.shape} does not match X columns {X.shape[1]}")

    X2 = X.copy()
    bad = ~np.isfinite(X2)
    if np.any(bad):
        X2[bad] = np.broadcast_to(med, X2.shape)[bad]
    return X2


def weighted_mse(y_true: np.ndarray, y_pred: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    """
    Вычисляет взвешенную среднеквадратичную ошибку с обработкой NaN и бесконечных значений.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    e2 = (y_true - y_pred) ** 2

    if w is None:
        m = np.isfinite(e2)
        return float(np.mean(e2[m])) if np.any(m) else float("nan")

    w = np.asarray(w, dtype=float)
    m = np.isfinite(e2) & np.isfinite(w) & (w >= 0.0)
    if not np.any(m):
        return float("nan")

    w2 = w[m]
    s = float(np.sum(w2))
    if s <= 0.0:
        return float("nan")

    return float(np.sum(w2 * e2[m]) / s)
