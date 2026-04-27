# src/ml_circuits/data_generation/generate_ac_dataset.py

"""
Генерация dataset_ac.csv для типовых пассивных фильтров.

Модуль строит схемы по таблице параметров filter_param_sets.csv,
выполняет AC-анализ методом МНА и формирует датасет частотных характеристик.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml_circuits.config import CHAR_FREQ_MULTIPLIERS
from ml_circuits.constants import EPS_LOG
from ml_circuits.circuits import (
    ac_out_amp_phase,
    build_circuit_for_topology,
    is_valid_rload,
    rlc_band_edges,
)
from ml_circuits.data_generation.sampling_filters import (
    f_key,
    seed_from_param_set_id,
)


def _make_frequency_grid_for_param_set(
    row: pd.Series,
    f_min: float,
    f_max: float,
    n_random: int,
) -> np.ndarray:
    """
    Формирует частотную сетку для одного набора параметров.

    Сетка включает случайные логарифмически равномерные точки
    и дополнительные точки около характерной частоты схемы.
    """
    param_set_id = str(row["param_set_id"])
    rng = np.random.default_rng(seed_from_param_set_id(param_set_id))

    random_freqs = np.exp(
        rng.uniform(
            np.log(f_min),
            np.log(f_max),
            size=n_random,
        )
    )

    extra_freqs: list[float] = []

    topo = str(row["topology"])

    if topo in {"RC_LP", "RC_HP", "RL_LP", "RL_HP"}:
        fc = float(row.get("fc", np.nan))

        if np.isfinite(fc) and fc > 0:
            for mult in CHAR_FREQ_MULTIPLIERS:
                f = fc * float(mult)

                if f_min <= f <= f_max:
                    extra_freqs.append(float(f))

    elif topo in {"RLC_BP", "RLC_NOTCH"}:
        f0 = float(row.get("f0", np.nan))
        Q_eff = float(row.get("Q_eff", np.nan))

        if np.isfinite(f0) and f0 > 0:
            for mult in CHAR_FREQ_MULTIPLIERS:
                f = f0 * float(mult)

                if f_min <= f <= f_max:
                    extra_freqs.append(float(f))

        if np.isfinite(f0) and np.isfinite(Q_eff) and f0 > 0 and Q_eff > 0:
            f1, f2 = rlc_band_edges(f0=f0, Q=Q_eff)

            for f_edge in (f1, f2):
                if np.isfinite(f_edge) and f_min <= f_edge <= f_max:
                    extra_freqs.append(float(f_edge))

    freqs = np.concatenate(
        [
            random_freqs.astype(float),
            np.asarray(extra_freqs, dtype=float),
        ]
    )

    freqs = freqs[np.isfinite(freqs)]
    freqs = freqs[(freqs >= f_min) & (freqs <= f_max)]
    freqs = np.unique(np.round(freqs, decimals=9))
    freqs.sort()

    return freqs.astype(float)


def generate_dataset_from_params_ac(
    df_params: pd.DataFrame,
    f_min: float = 5.0,
    f_max: float = 500.0,
    n_random_freqs: int = 64,
) -> pd.DataFrame:
    """
    Генерирует датасет частотных характеристик для типовых фильтров.

    Parameters
    ----------
    df_params:
        Таблица параметров схем, полученная функцией sample_param_sets.
    f_min, f_max:
        Нижняя и верхняя границы частотного диапазона, Гц.
    n_random_freqs:
        Число случайных частотных точек на один набор параметров.

    Returns
    -------
    pd.DataFrame
        Таблица dataset_ac.csv.
    """
    required_columns = [
        "topology",
        "param_set_id",
        "A_in",
        "R",
        "C",
        "L",
        "Rload",
    ]

    missing = [col for col in required_columns if col not in df_params.columns]
    if missing:
        raise ValueError(f"В таблице параметров отсутствуют колонки: {missing}")

    rows: list[dict[str, Any]] = []

    for _, row in df_params.iterrows():
        topo = str(row["topology"])
        param_set_id = str(row["param_set_id"])

        A_in = float(row["A_in"])
        R = float(row["R"])
        C = float(row["C"])
        L = float(row["L"])
        Rload = float(row["Rload"])
        rload_ok = is_valid_rload(Rload)

        ckt, out_node = build_circuit_for_topology(
            topo=topo,
            R=R,
            C=C,
            L=L,
            Rload=Rload,
            A_in=A_in,
            rload_ok=rload_ok,
        )

        freqs = _make_frequency_grid_for_param_set(
            row=row,
            f_min=f_min,
            f_max=f_max,
            n_random=n_random_freqs,
        )

        for f in freqs:
            H_mag, A_out, phi_rad = ac_out_amp_phase(
                ckt=ckt,
                f=float(f),
                out_node=out_node,
                A_in=A_in,
            )

            rec = row.to_dict()

            rec.update(
                {
                    "topology": topo,
                    "param_set_id": param_set_id,
                    "f": float(f),
                    "f_key": f_key(float(f)),
                    "H_mag": float(H_mag),
                    "log_H_mag": float(np.log(np.clip(H_mag, EPS_LOG, None))),
                    "A_out": float(A_out),
                    "phi_rad": float(phi_rad),
                    "phi_sin": float(np.sin(phi_rad)),
                    "phi_cos": float(np.cos(phi_rad)),
                }
            )

            rows.append(rec)

    df = pd.DataFrame(rows)

    print(
        f"[generate_dataset_from_params_ac] generated rows={len(df)} "
        f"param_sets={df['param_set_id'].nunique() if len(df) else 0}"
    )

    return df