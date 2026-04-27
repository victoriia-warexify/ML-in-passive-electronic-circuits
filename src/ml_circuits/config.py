# src/ml_circuits/config.py

"""
Конфигурация экспериментов проекта.

В этом файле хранятся настройки генерации данных, обучения моделей,
список топологий, диапазоны параметров компонентов, пути к данным
и параметры запуска экспериментов.

Базовые численные константы, такие как EPS_LOG, AMP_GATE и TWO_PI,
хранятся отдельно в constants.py.
"""

from __future__ import annotations

from ml_circuits.constants import SEED, EPS


# ============================================================
# Пути к данным и результатам
# ============================================================

PATHS: dict[str, str] = {
    "dataset_csv": "data/generated/dataset_ac.csv",
    "filter_param_sets_csv": "data/generated/filter_param_sets.csv",
    "rc_ladder_param_sets_csv": "data/generated/rc_ladder_param_sets.csv",
    "rc_ladder_dataset_jsonl": "data/generated/dataset_rc_ladder.jsonl",

    "artifacts_dir": "artifacts",
    "models_dir": "artifacts/filter_models",
    "plots_dir": "artifacts/plots",
    "metrics_dir": "artifacts/metrics",
    "rc_ladder_artifacts_dir": "artifacts/rc_ladder",
    "inverse_artifacts_dir": "artifacts/inverse",
}


# ============================================================
# Диапазоны параметров компонентов для обычных фильтров
# ============================================================

R_RANGE_OHM: tuple[float, float] = (1e2, 1e5)
C_RANGE_F: tuple[float, float] = (1e-8, 1e-4)
L_RANGE_H: tuple[float, float] = (1e-4, 1e-1)
RLOAD_RANGE_OHM: tuple[float, float] = (2e2, 1e5)


# ============================================================
# Отдельные диапазоны для RLC-топологий
# ============================================================

RLC_R_RANGE_OHM: tuple[float, float] = (1.0, 1e5)
RLC_RLOAD_RANGE_OHM: tuple[float, float] = (2.0, 1e5)

# Для L и C в RLC используются те же диапазоны, что и для обычных схем.
RLC_L_RANGE_H: tuple[float, float] = L_RANGE_H
RLC_C_RANGE_F: tuple[float, float] = C_RANGE_F

# Диапазоны добротности Q_eff, по которым балансируется генерация RLC-схем.
RLC_Q_BINS: tuple[tuple[float, float], ...] = (
    (0.2, 0.7),
    (0.7, 2.0),
    (2.0, 10.0),
    (10.0, 30.0),
)

# Ограничение на отношение R / Rload при генерации RLC-схем.
RLC_R_OVER_RLOAD_RANGE: tuple[float, float] = (0.05, 20.0)


# ============================================================
# Параметры генерации dataset_ac.csv
# ============================================================

A_IN_DEFAULT: float = 1.0

F_MIN_DEFAULT: float = 5.0
F_MAX_DEFAULT: float = 500.0

N_PARAM_SETS_PER_TOPOLOGY: int = 400

# Множители для дополнительных частотных точек около характерной частоты:
# fc для RC/RL-фильтров первого порядка и f0 для RLC-фильтров.
CHAR_FREQ_MULTIPLIERS: tuple[float, ...] = (
    0.6,
    0.75,
    0.85,
    0.92,
    0.97,
    1.0,
    1.03,
    1.08,
    1.15,
    1.3,
    1.7,
)


# ============================================================
# Списки топологий
# ============================================================

ORDER1_TOPOS: list[str] = ["RC_LP", "RC_HP", "RL_LP", "RL_HP"]

RLC_TOPOS: list[str] = ["RLC_BP", "RLC_NOTCH"]

ALL_FILTER_TOPOS: list[str] = ORDER1_TOPOS + RLC_TOPOS

# ============================================================
# Разбиение train/validation/test для обычных фильтров
# ============================================================

SPLIT_CONFIG: dict[str, float] = {
    # Доля групп param_set_id, выделяемая в тестовую выборку.
    "test_size": 0.15,

    # Доля групп от train+val, выделяемая в валидационную выборку.
    "val_size": 0.15,
}


# Разные seed по топологиям для воспроизводимого group split.
SEED_BY_TOPO: dict[str, int] = {
    "RC_HP": 42,
    "RC_LP": 43,
    "RL_HP": 44,
    "RL_LP": 45,
    "RLC_BP": 52,
    "RLC_NOTCH": 53,
}


# ============================================================
# Гиперпараметры custom GBDT по топологиям
# ============================================================

GBDT_BY_TOPOLOGY: dict[str, dict] = {
    # Фильтры первого порядка с более простой зависимостью.
    "RC_HP": dict(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=12,
        subsample=0.8,
        max_features=None,
        n_thresholds=128,
        min_delta=1e-9,
        verbose=200,
        early_stopping_rounds=60,
    ),
    "RL_LP": dict(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=12,
        subsample=0.8,
        max_features=None,
        n_thresholds=128,
        min_delta=1e-9,
        verbose=200,
        early_stopping_rounds=60,
    ),

    # Фильтры первого порядка с заметным влиянием нагрузки.
    "RC_LP": dict(
        n_estimators=1200,
        learning_rate=0.02,
        max_depth=8,
        min_samples_leaf=8,
        subsample=0.8,
        max_features=None,
        n_thresholds=64,
        min_delta=1e-10,
        verbose=300,
        early_stopping_rounds=160,
    ),
    "RL_HP": dict(
        n_estimators=1500,
        learning_rate=0.02,
        max_depth=8,
        min_samples_leaf=8,
        subsample=0.8,
        max_features=None,
        n_thresholds=64,
        min_delta=1e-10,
        verbose=300,
        early_stopping_rounds=160,
    ),

    # Фильтры второго порядка.
    "RLC_BP": dict(
        n_estimators=1200,
        learning_rate=0.04,
        max_depth=7,
        min_samples_leaf=18,
        subsample=0.8,
        max_features=None,
        n_thresholds=32,
        min_delta=1e-8,
        verbose=300,
        early_stopping_rounds=120,
    ),
    "RLC_NOTCH": dict(
        n_estimators=1400,
        learning_rate=0.04,
        max_depth=7,
        min_samples_leaf=20,
        subsample=0.8,
        max_features=None,
        n_thresholds=64,
        min_delta=1e-8,
        verbose=300,
        early_stopping_rounds=150,
    ),
}


# ============================================================
# Параметры sklearn baseline
# ============================================================

BASELINE_HGBR_PARAMS: dict = dict(
    max_iter=350,
    learning_rate=0.05,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=1e-6,
    random_state=SEED,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=30,
    tol=1e-7,
)


# ============================================================
# Общая конфигурация обучения обычных фильтров
# ============================================================

FILTER_TRAINING_CONFIG: dict = {
    "seed": SEED,
    "paths": PATHS,
    "splits": SPLIT_CONFIG,
    "topologies": ALL_FILTER_TOPOS,
    "gbdt_by_topology": GBDT_BY_TOPOLOGY,
    "metrics": {
        "phase_units": "deg",
        "eps": EPS,
    },
}


# ============================================================
# RC ladder: параметры запусков
# ============================================================

RC_LADDER_DEFAULT_ARTIFACT_DIR: str = "artifacts_rc_ladder_pure_ml_strict_blackbox"
RC_LADDER_DEFAULT_PLOTS_DIRNAME: str = "plots_rc_ladder"

RC_LADDER_SEEDS: list[int] = [40, 41, 42]

# Основные экспериментальные режимы:
# 1) strict extrapolation: train на n=1,2,3; test на n=4;
# 2) few-shot 4%: малая доля n=4 добавляется в train.
RC_LADDER_RUN_CONFIGS: list[dict] = [
    {
        "mode_name": "strict_extrapolation",
        "mode": "strict",
        "n4_train_fraction": None,
        "artifact_dir": "artifacts_rc_ladder_pure_ml_strict_blackbox",
    },
    {
        "mode_name": "fewshot_4pct",
        "mode": "fewshot",
        "n4_train_fraction": 0.04,
        "artifact_dir": "artifacts_rc_ladder_pure_ml_fewshot_n4_4pct",
    },
]


# Пути к leaderboard-файлам, которые используются для итоговой сводной таблицы.
RC_LADDER_EXPERIMENT_RUNS: list[dict] = [
    {
        "mode": "strict_extrapolation",
        "path": "artifacts_rc_ladder_pure_ml_strict_blackbox/leaderboard_seeds.csv",
    },
    {
        "mode": "fewshot_4pct",
        "path": "artifacts_rc_ladder_pure_ml_fewshot_n4_4pct/leaderboard_seeds.csv",
    },
]


# Минимальный набор столбцов, ожидаемый в leaderboard-файлах RC ladder.
RC_LADDER_REQUIRED_COLS: list[str] = [
    "seed",
    "score",

    # Обычные row-wise метрики.
    "test_dB_MAE",
    "test_phi_MAE_filtered_deg",
    "test_phi_MAE_deg",
    "test_logH_MAE",

    # Group-weighted метрики.
    # Они являются основными для ВКР, потому что каждая схема param_set_id
    # имеет одинаковый вклад независимо от числа частотных точек.
    "test_dB_MAE_group_weighted",
    "test_dB_RMSE_group_weighted",
    "test_phi_MAE_filtered_deg_group_weighted",
    "test_phi_MAE_deg_group_weighted",
    "test_logH_MAE_group_weighted",
]