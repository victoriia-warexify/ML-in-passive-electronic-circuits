from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor

from ml_circuits.config import BASELINE_HGBR_PARAMS


class PerOutputSklearnRegressor:
    """
    Простая обертка над sklearn-регрессором для многовыходной задачи.
    """

    def __init__(self, base_estimator, n_outputs: int = 3):
        self.base_estimator = base_estimator
        self.n_outputs = int(n_outputs)
        self.models_ = []

    def fit(self, X, Y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features)")
        if Y.ndim != 2 or Y.shape[1] != self.n_outputs:
            raise ValueError(f"Y must have shape (n_samples, {self.n_outputs})")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must contain the same number of samples")

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=float)
            if sample_weight.ndim == 1:
                if sample_weight.shape[0] != X.shape[0]:
                    raise ValueError("sample_weight must have length n_samples")
            elif sample_weight.ndim == 2:
                if sample_weight.shape != Y.shape:
                    raise ValueError("2D sample_weight must have shape equal to Y")
            else:
                raise ValueError("sample_weight must be 1D or 2D")

        self.models_ = []

        for j in range(self.n_outputs):
            yj = Y[:, j]

            if sample_weight is None:
                wj = None
                m = np.isfinite(yj)
            else:
                if sample_weight.ndim == 1:
                    wj = sample_weight
                else:
                    wj = sample_weight[:, j]

                m = np.isfinite(yj) & np.isfinite(wj) & (wj > 0.0)

            if not np.any(m):
                raise ValueError(f"No valid training samples for output {j}")

            est = clone(self.base_estimator)

            if sample_weight is None:
                est.fit(X[m], yj[m])
            else:
                est.fit(X[m], yj[m], sample_weight=wj[m])

            self.models_.append(est)

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if not self.models_:
            raise RuntimeError("Model is not fitted")

        preds = [m.predict(X) for m in self.models_]
        return np.vstack(preds).T.astype(float, copy=False)


def make_sklearn_hgbr_baseline(seed: int, seed_offset: int = 0):
    """
    Создает простой sklearn-baseline.
    """
    params = dict(BASELINE_HGBR_PARAMS)
    params["random_state"] = int(seed + seed_offset)

    base = HistGradientBoostingRegressor(**params)
    return PerOutputSklearnRegressor(base_estimator=base, n_outputs=3)
