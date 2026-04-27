from __future__ import annotations

import numpy as np
import pandas as pd


def train_test_split_groups(groups, test_size=0.2, seed=42):
    """
    Выполняет разбиение наблюдений на обучающую и тестовую выборки по группам.

    Разбиение осуществляется по уникальным значениям groups
    (param_set_id), а не по отдельным строкам.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)

    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1)")

    if uniq.size < 2:
        idx = np.ones_like(groups, dtype=bool)
        return idx, ~idx

    rng.shuffle(uniq)
    n_test = int(np.ceil(uniq.size * test_size))
    n_test = min(max(n_test, 1), uniq.size - 1)

    test_g = set(uniq[:n_test])
    test_mask = np.isin(groups, list(test_g))
    train_mask = ~test_mask
    return train_mask, test_mask


def split_train_val_test_by_group(
    df: pd.DataFrame,
    group_col: str = "param_set_id",
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
):
    """
    Выполняет разбиение DataFrame на обучающую, валидационную и тестовую выборки по группам.
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be in (0, 1)")
    if not (0.0 < val_size < 1.0):
        raise ValueError("val_size must be in (0, 1)")
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found in df")
    if df[group_col].isna().any():
        raise ValueError(f"{group_col} contains NaN")

    groups = df[group_col].to_numpy()

    trainval_mask, test_mask = train_test_split_groups(groups, test_size=test_size, seed=seed)
    df_trainval = df.loc[trainval_mask].copy()
    df_test = df.loc[test_mask].copy()

    groups_tv = df_trainval[group_col].to_numpy()
    train_mask, val_mask = train_test_split_groups(groups_tv, test_size=val_size, seed=seed + 1)
    df_train = df_trainval.loc[train_mask].copy()
    df_val = df_trainval.loc[val_mask].copy()

    g_train = set(df_train[group_col].unique())
    g_val = set(df_val[group_col].unique())
    g_test = set(df_test[group_col].unique())
    if not g_train.isdisjoint(g_val):
        raise ValueError("Train and validation groups overlap")
    if not g_train.isdisjoint(g_test):
        raise ValueError("Train and test groups overlap")
    if not g_val.isdisjoint(g_test):
        raise ValueError("Validation and test groups overlap")

    return df_train, df_val, df_test
