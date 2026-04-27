from __future__ import annotations

import numpy as np

from ml_circuits.training.losses import weighted_mse


class RegressionTree:
    def __init__(
        self,
        max_depth=6,
        min_samples_leaf=5,
        max_features=None,
        random_state=42,
        n_thresholds=64,
    ):
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)
        self._root = None
        self.n_thresholds = int(n_thresholds)

    class Node:
        __slots__ = ("is_leaf", "value", "feat", "thr", "left", "right")

        def __init__(self, is_leaf, value=None, feat=None, thr=None, left=None, right=None):
            self.is_leaf = is_leaf
            self.value = value
            self.feat = feat
            self.thr = thr
            self.left = left
            self.right = right

    @staticmethod
    def _wmean(y: np.ndarray, w: np.ndarray) -> float:
        """Вычисляет взвешенное среднее, используемое как значение в листовом узле."""
        sw = float(w.sum())
        if (not np.isfinite(sw)) or (sw <= 0):
            return float(np.mean(y))
        return float(np.dot(w, y) / sw)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None):
        """Обучает регрессионное дерево по данным (X, y) с опциональными весами sample_weight."""
        X = np.asarray(X, float)
        y = np.asarray(y, float)

        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must have shape (n_samples,)")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        n, p = X.shape

        if sample_weight is None:
            w = np.ones(n, dtype=float)
        else:
            w = np.asarray(sample_weight, float)

            if w.ndim != 1:
                raise ValueError("sample_weight must have shape (n_samples,)")
            if w.shape[0] != n:
                raise ValueError("sample_weight must have shape (n_samples,)")
            if np.any(~np.isfinite(w)) or np.any(w < 0):
                raise ValueError("sample_weight must be finite and non-negative")
            if float(w.sum()) <= 0:
                raise ValueError("sample_weight sum must be > 0")

        self._root = self._build(X, y, w, depth=0, p=p)
        return self

    def _best_split(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, feat_indices: np.ndarray):
        """
        Выполняет поиск лучшего разбиения по критерию уменьшения взвешенной суммы квадратов ошибок.
        """
        n = X.shape[0]
        if n <= 2 * self.min_samples_leaf:
            return None

        w_sum = float(w.sum())
        if (not np.isfinite(w_sum)) or (w_sum <= 0):
            return None

        wy_sum = float(np.dot(w, y))
        wy2_sum = float(np.dot(w, y * y))
        sse_parent = wy2_sum - (wy_sum * wy_sum) / w_sum

        best = None

        lo = self.min_samples_leaf
        hi = n - self.min_samples_leaf
        if hi <= lo:
            return None

        n_thr = min(self.n_thresholds, hi - lo)
        if n_thr < 2:
            return None

        for j in feat_indices:
            x = X[:, j]
            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = y[order]
            w_sorted = w[order]

            pw = np.cumsum(w_sorted)
            pwy = np.cumsum(w_sorted * y_sorted)
            pwy2 = np.cumsum(w_sorted * y_sorted * y_sorted)

            idxs = np.linspace(lo, hi - 1, n_thr, dtype=int)
            idxs = np.unique(idxs)

            for i in idxs:
                if x_sorted[i] == x_sorted[i - 1]:
                    continue

                wL = float(pw[i - 1])
                wR = w_sum - wL
                if wL <= 0 or wR <= 0:
                    continue

                wyL = float(pwy[i - 1])
                wyR = wy_sum - wyL

                wy2L = float(pwy2[i - 1])
                wy2R = wy2_sum - wy2L

                sseL = wy2L - (wyL * wyL) / wL
                sseR = wy2R - (wyR * wyR) / wR
                gain = sse_parent - (sseL + sseR)

                if (best is None) or (gain > best[0]):
                    thr = 0.5 * (x_sorted[i] + x_sorted[i - 1])
                    best = (float(gain), int(j), float(thr))

        return best

    def _build(self, X: np.ndarray, y: np.ndarray, w: np.ndarray, depth: int, p: int):
        """Рекурсивно строит дерево до достижения max_depth или до невозможности корректного разбиения."""
        if depth >= self.max_depth or X.shape[0] <= 2 * self.min_samples_leaf:
            return self.Node(True, value=self._wmean(y, w))

        if self.max_features is None:
            feat_indices = np.arange(p)
        else:
            k = min(int(self.max_features), p)
            k = max(1, k)
            feat_indices = self.rng.choice(p, size=k, replace=False)

        best = self._best_split(X, y, w, feat_indices)

        if best is None or best[0] <= 1e-12:
            return self.Node(True, value=self._wmean(y, w))

        _, feat, thr = best
        left_mask = X[:, feat] <= thr
        right_mask = ~left_mask

        if left_mask.sum() < self.min_samples_leaf or right_mask.sum() < self.min_samples_leaf:
            return self.Node(True, value=self._wmean(y, w))

        left = self._build(X[left_mask], y[left_mask], w[left_mask], depth + 1, p)
        right = self._build(X[right_mask], y[right_mask], w[right_mask], depth + 1, p)
        return self.Node(False, feat=feat, thr=thr, left=left, right=right)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Выполняет предсказание путём прохода по дереву до листового узла для каждого объекта."""
        X = np.asarray(X, float)
        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features)")

        out = np.empty(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            node = self._root
            while not node.is_leaf:
                node = node.left if X[i, node.feat] <= node.thr else node.right
            out[i] = node.value
        return out


class GBDTRegressor:
    def __init__(
        self,
        n_estimators=800,
        learning_rate=0.03,
        max_depth=8,
        min_samples_leaf=10,
        subsample=0.8,
        max_features=None,
        random_state=42,
        early_stopping_rounds=50,
        verbose=100,
        n_thresholds=64,
        min_delta=0.0,
    ):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)

        self.early_stopping_rounds = int(early_stopping_rounds)
        self.verbose = verbose

        self.n_thresholds = int(n_thresholds)
        self.min_delta = float(min_delta)

        self.init_ = 0.0
        self.trees_ = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val=None,
        y_val=None,
        w: np.ndarray | None = None,
        w_val: np.ndarray | None = None,
    ):
        """
        Обучает модель градиентного бустинга над регрессионными деревьями.
        """
        X = np.asarray(X, float)
        y = np.asarray(y, float)

        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features)")
        if y.ndim != 1:
            raise ValueError("y must have shape (n_samples,)")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must contain the same number of samples")

        n = X.shape[0]

        if w is not None:
            w = np.asarray(w, float)
            if w.ndim != 1 or w.shape[0] != n:
                raise ValueError("w must have shape (n_samples,)")
            if np.any(~np.isfinite(w)) or np.any(w < 0) or float(w.sum()) <= 0:
                raise ValueError("w must be finite, non-negative, and have positive sum")
            self.init_ = float(np.dot(w, y) / float(w.sum()))
        else:
            self.init_ = float(y.mean())

        pred = np.full(n, self.init_, dtype=float)

        use_val = (X_val is not None) and (y_val is not None)
        if use_val:
            X_val = np.asarray(X_val, float)
            y_val = np.asarray(y_val, float)

            if X_val.ndim != 2:
                raise ValueError("X_val must have shape (n_val_samples, n_features)")
            if y_val.ndim != 1:
                raise ValueError("y_val must have shape (n_val_samples,)")
            if X_val.shape[0] != y_val.shape[0]:
                raise ValueError("X_val and y_val must contain the same number of samples")

            pred_val = np.full(X_val.shape[0], self.init_, dtype=float)

            if w_val is not None:
                w_val = np.asarray(w_val, float)
                if w_val.ndim != 1 or w_val.shape[0] != X_val.shape[0]:
                    raise ValueError("w_val must have shape (n_val_samples,)")
                if np.any(~np.isfinite(w_val)) or np.any(w_val < 0) or float(w_val.sum()) <= 0:
                    raise ValueError("w_val must be finite, non-negative, and have positive sum")

        self.trees_ = []

        best_val = np.inf
        best_iter = -1
        best_trees = None
        no_improve = 0
        min_delta = self.min_delta

        for m in range(self.n_estimators):
            r = y - pred

            if self.subsample < 1.0:
                k = int(np.ceil(n * self.subsample))
                idx = self.rng.choice(n, size=k, replace=False)
                X_m = X[idx]
                r_m = r[idx]
                w_m = w[idx] if w is not None else None
            else:
                X_m = X
                r_m = r
                w_m = w

            tree = RegressionTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=int(self.rng.integers(0, 1_000_000_000)),
                n_thresholds=self.n_thresholds,
            ).fit(X_m, r_m, sample_weight=w_m)

            pred += self.learning_rate * tree.predict(X)
            self.trees_.append(tree)

            if use_val:
                pred_val += self.learning_rate * tree.predict(X_val)
                val_loss = weighted_mse(y_val, pred_val, w=w_val)

                if val_loss < best_val - min_delta:
                    best_val = val_loss
                    best_iter = m
                    best_trees = list(self.trees_)
                    no_improve = 0
                else:
                    no_improve += 1

                if self.verbose and (m % self.verbose == 0 or m == self.n_estimators - 1):
                    print(f"iter={m:4d} val_mse={val_loss:.6g} best={best_val:.6g} best_iter={best_iter}")

                if no_improve >= self.early_stopping_rounds:
                    if self.verbose:
                        print(f"Early stopping at iter={m}, best_iter={best_iter}, best_val={best_val:.6g}")
                    self.trees_ = best_trees if best_trees is not None else self.trees_
                    break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Возвращает предсказание в виде суммы начальной константы и вкладов всех деревьев."""
        X = np.asarray(X, float)
        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features)")

        pred = np.full(X.shape[0], self.init_, dtype=float)
        for tree in self.trees_:
            pred += self.learning_rate * tree.predict(X)
        return pred


class MultiOutputGBDT:
    """
    Многовыходная модель: независимый GBDT для каждого выхода.
    """

    def __init__(self, n_outputs: int, **gbdt_kwargs):
        self.models_ = [GBDTRegressor(**gbdt_kwargs) for _ in range(n_outputs)]

    @staticmethod
    def _weight_for_output(w, j: int, n: int):
        if w is None:
            return None

        w = np.asarray(w, dtype=float)

        if w.ndim == 1:
            if w.shape[0] != n:
                raise ValueError(
                    f"1D weights length mismatch: got {w.shape[0]}, expected {n}"
                )
            return w

        if w.ndim == 2:
            if w.shape[0] != n:
                raise ValueError(
                    f"2D weights rows mismatch: got {w.shape[0]}, expected {n}"
                )
            if j >= w.shape[1]:
                raise ValueError(
                    f"Output index {j} is out of bounds for weights with shape {w.shape}"
                )
            return w[:, j]

        raise ValueError("w must be None, 1D, or 2D")

    def fit(self, X, Y, X_val=None, Y_val=None, w=None, w_val=None):
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)

        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features)")
        if Y.ndim != 2:
            raise ValueError("Y must have shape (n_samples, n_outputs)")
        if X.shape[0] != Y.shape[0]:
            raise ValueError("X and Y must contain the same number of samples")
        if Y.shape[1] != len(self.models_):
            raise ValueError("n_outputs does not match Y.shape[1]")

        use_val = (X_val is not None) and (Y_val is not None)

        if use_val:
            X_val = np.asarray(X_val, float)
            Y_val = np.asarray(Y_val, float)

            if X_val.ndim != 2:
                raise ValueError("X_val must have shape (n_val_samples, n_features)")
            if Y_val.ndim != 2:
                raise ValueError("Y_val must have shape (n_val_samples, n_outputs)")
            if X_val.shape[0] != Y_val.shape[0]:
                raise ValueError("X_val and Y_val must contain the same number of samples")
            if Y_val.shape[1] != Y.shape[1]:
                raise ValueError("Y_val.shape[1] must match Y.shape[1]")

        n = X.shape[0]

        for j, model_j in enumerate(self.models_):
            w_j = self._weight_for_output(w, j, n)

            if w_j is None:
                train_mask = np.isfinite(Y[:, j])
                w_train_j = None
            else:
                w_j = np.asarray(w_j, dtype=float)
                train_mask = (
                    np.isfinite(Y[:, j])
                    & np.isfinite(w_j)
                    & (w_j > 0.0)
                )
                w_train_j = w_j[train_mask]

            if train_mask.sum() == 0:
                raise ValueError(f"No valid training samples for output {j}")

            X_train_j = X[train_mask]
            y_train_j = Y[train_mask, j]

            if use_val:
                w_val_j = self._weight_for_output(w_val, j, X_val.shape[0])

                if w_val_j is None:
                    val_mask = np.isfinite(Y_val[:, j])
                    w_valid_j = None
                else:
                    w_val_j = np.asarray(w_val_j, dtype=float)
                    val_mask = (
                        np.isfinite(Y_val[:, j])
                        & np.isfinite(w_val_j)
                        & (w_val_j > 0.0)
                    )
                    w_valid_j = w_val_j[val_mask]

                if val_mask.sum() == 0:
                    raise ValueError(f"No valid validation samples for output {j}")

                model_j.fit(
                    X_train_j,
                    y_train_j,
                    X_val=X_val[val_mask],
                    y_val=Y_val[val_mask, j],
                    w=w_train_j,
                    w_val=w_valid_j,
                )
            else:
                model_j.fit(
                    X_train_j,
                    y_train_j,
                    w=w_train_j,
                )

        return self

    def predict(self, X):
        X = np.asarray(X, float)

        if X.ndim != 2:
            raise ValueError("X must have shape (n_samples, n_features)")

        preds = [m.predict(X) for m in self.models_]
        return np.vstack(preds).T
