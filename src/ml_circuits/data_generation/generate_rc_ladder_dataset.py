# src/ml_circuits/data_generation/generate_rc_ladder_dataset.py

"""
Генерация dataset_rc_ladder.jsonl для RC ladder.

Модуль строит RC-цепочки по таблице параметров rc_ladder_param_sets.csv,
выполняет AC-анализ методом МНА и формирует датасет частотных характеристик.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml_circuits.constants import EPS_LOG
from ml_circuits.circuits import ac_out_amp_phase, gen_rc_ladder
from ml_circuits.data_generation.sampling_rc_ladder import make_rc_ladder_freq_grid
from ml_circuits.io_utils import ensure_parent_dir


def _parse_sections_if_needed(sections: Any) -> list[dict[str, float]]:
    """
    Приводит описание секций RC ladder к списку словарей.

    При чтении из CSV поле sections может быть строкой JSON.
    """
    if isinstance(sections, str):
        sections = json.loads(sections)

    if not isinstance(sections, list):
        raise ValueError("sections must be a list or JSON string")

    parsed: list[dict[str, float]] = []

    for sec in sections:
        if not isinstance(sec, dict):
            raise ValueError("Each RC ladder section must be a dictionary")

        parsed.append(
            {
                "R": float(sec["R"]),
                "C": float(sec["C"]),
            }
        )

    return parsed


def generate_rc_ladder_dataset_jsonl(
    df_params: pd.DataFrame,
    out_path: str | Path | None = None,
    n_random_freqs: int = 32,
    n_local_freqs: int = 13,
    f_min_guard: float = 1e-2,
    f_max_guard: float = 1e7,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Генерирует датасет частотных характеристик для RC ladder.

    Parameters
    ----------
    df_params:
        Таблица параметров, полученная sample_rc_ladder_param_sets.
    out_path:
        Путь для сохранения JSONL-файла. Если None, файл не сохраняется.
    n_random_freqs:
        Число случайных частотных точек на одну схему.
    n_local_freqs:
        Число локальных точек около характерной частоты схемы.
    f_min_guard, f_max_guard:
        Глобальные границы допустимого диапазона частот.
    seed:
        Seed генератора случайных чисел.

    Returns
    -------
    pd.DataFrame
        Таблица dataset_rc_ladder.jsonl.
    """
    required_columns = [
        "param_set_id",
        "topology",
        "n_sections",
        "sections",
        "Rload",
        "A_in",
    ]

    missing = [col for col in required_columns if col not in df_params.columns]
    if missing:
        raise ValueError(f"В таблице параметров отсутствуют колонки: {missing}")

    rows: list[dict[str, Any]] = []

    base_rng = np.random.default_rng(seed)

    for _, row in df_params.iterrows():
        param_set_id = str(row["param_set_id"])
        topology = str(row["topology"])

        if topology != "RC_LADDER":
            raise ValueError(f"Expected topology='RC_LADDER', got {topology}")

        n_sections = int(row["n_sections"])
        sections = _parse_sections_if_needed(row["sections"])
        Rload = float(row["Rload"])
        A_in = float(row["A_in"])

        if n_sections != len(sections):
            raise ValueError(
                f"n_sections={n_sections} не совпадает с len(sections)={len(sections)} "
                f"для param_set_id={param_set_id}"
            )

        # Для каждой схемы создаётся отдельный seed, чтобы частотная сетка
        # была воспроизводимой и не зависела от порядка строк.
        local_seed = int(base_rng.integers(0, 2**32 - 1))
        rng = np.random.default_rng(local_seed)

        freqs = make_rc_ladder_freq_grid(
            sections=sections,
            rng=rng,
            n_random=n_random_freqs,
            n_local=n_local_freqs,
            f_min_guard=f_min_guard,
            f_max_guard=f_max_guard,
        )

        ckt, out_node = gen_rc_ladder(
            sections=sections,
            A_in=A_in,
            Rload=Rload,
        )

        for f in freqs:
            H_mag, A_out, phi_rad = ac_out_amp_phase(
                ckt=ckt,
                f=float(f),
                out_node=out_node,
                A_in=A_in,
            )

            rec = {
                "param_set_id": param_set_id,
                "topology": "RC_LADDER",
                "n_sections": n_sections,
                "sections": sections,
                "Rload": Rload,
                "A_in": A_in,
                "frequency_hz": float(f),
                "H_mag": float(H_mag),
                "log_H_mag": float(np.log(np.clip(H_mag, EPS_LOG, None))),
                "A_out": float(A_out),
                "phi_rad": float(phi_rad),
                "phi_sin": float(np.sin(phi_rad)),
                "phi_cos": float(np.cos(phi_rad)),
            }

            rows.append(rec)

    df = pd.DataFrame(rows)

    if out_path is not None:
        out_path = ensure_parent_dir(out_path)
        df.to_json(
            out_path,
            orient="records",
            lines=True,
            force_ascii=False,
        )

    print(
        f"[generate_rc_ladder_dataset_jsonl] generated rows={len(df)} "
        f"param_sets={df['param_set_id'].nunique() if len(df) else 0}"
    )

    return df