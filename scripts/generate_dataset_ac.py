from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml_circuits.config import PATHS
from ml_circuits.constants import SEED
from ml_circuits.data_generation.generate_ac_dataset import generate_dataset_from_params_ac
from ml_circuits.data_generation.sampling_filters import sample_param_sets


def main() -> None:
    # 1) Формирование таблицы параметров для всех топологий
    df_params = sample_param_sets(
        n_param_sets_per_topology=400,
        f_min=5.0, f_max=500.0,
        A_in=1.0,
        seed=SEED,
    )

    param_sets_path = Path(PATHS["filter_param_sets_csv"])
    param_sets_path.parent.mkdir(parents=True, exist_ok=True)
    df_params.to_csv(param_sets_path, index=False)
    print(f"Сохранена таблица параметров: {param_sets_path}")

    # 2) Генерация датасета в частотной области (AC)
    dataset_path = Path(PATHS["dataset_csv"])
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df_ac = generate_dataset_from_params_ac(
        df_params,
        out_csv=str(dataset_path),
    )
    print(f"Сформирован AC-датасет: {dataset_path}")

    # ===================================================================
    # Проверка итогового AC-датасета
    # ===================================================================

    print("--- Размер датасета ---")
    print(df_ac.shape)

    print("\n--- Количество строк по топологиям ---")
    print(df_ac["topology"].value_counts())

    print("\n--- Количество уникальных схем по топологиям ---")
    print(df_ac.groupby("topology")["param_set_id"].nunique())

    print("\n--- Дубликаты param_set_id + f ---")
    n_dup = df_ac.duplicated(["param_set_id", "f"]).sum()
    print(n_dup)

    if n_dup > 0:
        raise ValueError("Found duplicated (param_set_id, f) pairs")

    print("\n--- Проверка finite-значений ---")
    check_cols = [
        "f",
        "H_mag",
        "log_H_mag",
        "A_out",
        "phi_sin",
        "phi_cos",
    ]

    bad_counts = (~np.isfinite(df_ac[check_cols])).sum()
    print(bad_counts)

    if bad_counts.sum() > 0:
        raise ValueError("Found non-finite values in AC dataset")

    print("\n--- Проверка пассивности H_mag <= 1 ---")
    hmax = df_ac["H_mag"].max()
    n_gt_1 = (df_ac["H_mag"] > 1.0 + 1e-9).sum()

    print("max H_mag:", hmax)
    print("count H_mag > 1 + 1e-9:", n_gt_1)

    if n_gt_1 > 0:
        raise ValueError("Found significant passivity violations")

    print("\n--- Проверка sin/cos фазы ---")
    phase_norm_err = np.abs(
        df_ac["phi_sin"] ** 2 + df_ac["phi_cos"] ** 2 - 1.0
    ).max()

    print("max |sin^2 + cos^2 - 1|:", phase_norm_err)

    if phase_norm_err > 1e-9:
        raise ValueError("Bad phase sin/cos normalization")

    print("\n--- Проверка RLC q_bin в итоговом датасете ---")

    df_ac_rlc_params = (
        df_ac[df_ac["topology"].isin(["RLC_BP", "RLC_NOTCH"])]
        .drop_duplicates("param_set_id")
        .copy()
    )

    print(
        pd.crosstab(
            df_ac_rlc_params["topology"],
            df_ac_rlc_params["q_bin"],
            margins=True,
        )
    )

    print("\n--- Q_eff по topology и q_bin ---")
    print(
        df_ac_rlc_params
        .groupby(["topology", "q_bin"])["Q_eff"]
        .agg(["count", "min", "median", "max"])
    )


if __name__ == "__main__":
    main()
