from __future__ import annotations

import numpy as np

from ml_circuits.config import FILTER_TRAINING_CONFIG as CONFIG, SEED_BY_TOPO
from ml_circuits.features.filter_features import FEATS_ORDER1, FEATS_RLC
from ml_circuits.features.targets import TARGETS_ORDER1, TARGETS_RLC
from ml_circuits.models.gbdt import MultiOutputGBDT
from ml_circuits.models.sklearn_baselines import make_sklearn_hgbr_baseline
from ml_circuits.training.losses import fit_median_imputer, transform_median_imputer
from ml_circuits.training.splits import split_train_val_test_by_group
from ml_circuits.training.weights import make_group_weights, make_output_weights_for_logmag_phase


_ALLOWED_GBDT_KEYS = {
    "n_estimators",
    "learning_rate",
    "max_depth",
    "min_samples_leaf",
    "subsample",
    "max_features",
    "random_state",
    "early_stopping_rounds",
    "verbose",
    "n_thresholds",
    "min_delta",
}


def make_model(topo: str):
    """Создаёт модель MultiOutputGBDT для заданной топологии и проверяет корректность гиперпараметров."""
    if topo not in CONFIG["gbdt_by_topology"]:
        raise KeyError(f"No GBDT params for topology '{topo}' in CONFIG['gbdt_by_topology']")

    gbdt_kwargs = dict(CONFIG["gbdt_by_topology"][topo])
    gbdt_kwargs["random_state"] = int(CONFIG["seed"])

    extra = set(gbdt_kwargs) - _ALLOWED_GBDT_KEYS
    if extra:
        raise ValueError(f"Unknown GBDT keys for {topo}: {sorted(extra)}")

    return MultiOutputGBDT(n_outputs=3, **gbdt_kwargs)


def build_XY(df_train, df_val, df_test, topo: str, FEATS: dict, TARGETS: dict):
    """
    Формирует матрицы признаков X и таргеты Y для train, val и test.
    """
    Y_train, meta_train = TARGETS[topo](df_train)
    Y_val, meta_val = TARGETS[topo](df_val)
    Y_test, meta_test = TARGETS[topo](df_test)

    X_train_raw, feat_names = FEATS[topo](df_train)
    X_val_raw, _ = FEATS[topo](df_val)
    X_test_raw, _ = FEATS[topo](df_test)

    med = fit_median_imputer(X_train_raw)
    X_train = transform_median_imputer(X_train_raw, med)
    X_val = transform_median_imputer(X_val_raw, med)
    X_test = transform_median_imputer(X_test_raw, med)

    return X_train, Y_train, meta_train, X_val, Y_val, meta_val, X_test, Y_test, meta_test, feat_names


def train_one_topology(df, topo, split_fn, FEATS, TARGETS, amp_gate: float = 0.05):
    """
    Выполняет полный цикл подготовки данных и обучения модели для одной топологии.
    """
    df_topo = df[df["topology"] == topo].copy()

    df_tr, df_val, df_te = split_fn(df_topo, topo)

    X_tr, Y_tr, meta_tr, X_val, Y_val, meta_val, X_te, Y_te, meta_te, feat_names = build_XY(
        df_tr,
        df_val,
        df_te,
        topo,
        FEATS=FEATS,
        TARGETS=TARGETS,
    )

    w_train_base = make_group_weights(df_tr, group_col="param_set_id")
    w_val_base = make_group_weights(df_val, group_col="param_set_id")

    w_train = make_output_weights_for_logmag_phase(
        df_tr,
        w_train_base,
        amp_gate=amp_gate,
    )

    w_val = make_output_weights_for_logmag_phase(
        df_val,
        w_val_base,
        amp_gate=amp_gate,
    )

    print(f"\n--- Training {topo} ---")
    print(
        f"[{topo}] phase train points kept: "
        f"{np.count_nonzero(w_train[:, 1] > 0)} / {len(w_train)} "
        f"({100.0 * np.count_nonzero(w_train[:, 1] > 0) / len(w_train):.2f}%)"
    )
    print(
        f"[{topo}] phase val points kept: "
        f"{np.count_nonzero(w_val[:, 1] > 0)} / {len(w_val)} "
        f"({100.0 * np.count_nonzero(w_val[:, 1] > 0) / len(w_val):.2f}%)"
    )

    model = make_model(topo)

    model.fit(
        X_tr,
        Y_tr,
        X_val=X_val,
        Y_val=Y_val,
        w=w_train,
        w_val=w_val,
    )

    dataset_pack = (df_te, X_te, Y_te, meta_te, feat_names)
    return model, dataset_pack


def split_order1(df_topo, topo):
    """Выполняет group split по param_set_id для топологий первого порядка."""
    return split_train_val_test_by_group(
        df_topo,
        group_col="param_set_id",
        test_size=CONFIG["splits"]["test_size"],
        val_size=CONFIG["splits"]["val_size"],
        seed=CONFIG["seed"] + SEED_BY_TOPO[topo],
    )


def split_rlc(df_topo, topo):
    """
    Разбиение для RLC по param_set_id.
    """
    return split_train_val_test_by_group(
        df_topo,
        group_col="param_set_id",
        test_size=CONFIG["splits"]["test_size"],
        val_size=CONFIG["splits"]["val_size"],
        seed=CONFIG["seed"] + SEED_BY_TOPO[topo],
    )


def train_sklearn_baseline_one_topology(
    df,
    topo: str,
    split_fn,
    FEATS: dict,
    TARGETS: dict,
    amp_gate: float = 0.05,
):
    """
    Обучает sklearn-baseline для одной топологии.
    """
    df_topo = df[df["topology"] == topo].copy()

    df_tr, df_val, df_te = split_fn(df_topo, topo)

    X_tr, Y_tr, meta_tr, X_val, Y_val, meta_val, X_te, Y_te, meta_te, feat_names = build_XY(
        df_tr,
        df_val,
        df_te,
        topo,
        FEATS=FEATS,
        TARGETS=TARGETS,
    )

    X_fit = X_tr
    Y_fit = Y_tr

    df_fit = df_tr.copy()

    w_fit_base = make_group_weights(df_fit, group_col="param_set_id")
    w_fit = make_output_weights_for_logmag_phase(
        df_fit,
        w_fit_base,
        amp_gate=amp_gate,
    )

    print(f"\n--- Training sklearn HGBR baseline: {topo} ---")
    print(
        f"[{topo}] baseline phase points kept: "
        f"{np.count_nonzero(w_fit[:, 1] > 0)} / {len(w_fit)} "
        f"({100.0 * np.count_nonzero(w_fit[:, 1] > 0) / len(w_fit):.2f}%)"
    )

    model = make_sklearn_hgbr_baseline(seed=CONFIG["seed"], seed_offset=SEED_BY_TOPO[topo])
    model.fit(X_fit, Y_fit, sample_weight=w_fit)

    dataset_pack = (df_te, X_te, Y_te, meta_te, feat_names)
    return model, dataset_pack


ORDER1_TOPOS = ["RC_HP", "RC_LP", "RL_HP", "RL_LP"]
RLC_TOPOS = ["RLC_BP", "RLC_NOTCH"]
