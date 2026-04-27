# src/ml_circuits/data_generation/sampling_rc_ladder.py

"""
Генерация параметров для RC ladder.

Модуль формирует таблицу параметров rc_ladder_param_sets.csv.
Каждая строка таблицы соответствует одной RC-цепочке с n секциями.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from ml_circuits.config import (
    A_IN_DEFAULT,
    C_RANGE_F,
    R_RANGE_OHM,
    RLOAD_RANGE_OHM,
)
from ml_circuits.constants import TWO_PI


def make_param_set_id_rc_ladder(payload: dict[str, Any]) -> str:
    """
    Формирует устойчивый идентификатор набора параметров RC ladder.
    """
    payload_for_hash = {}

    for key, value in payload.items():
        if isinstance(value, float):
            payload_for_hash[key] = f"{value:.12g}"
        elif key == "sections":
            payload_for_hash[key] = [
                {
                    "R": f"{float(sec['R']):.12g}",
                    "C": f"{float(sec['C']):.12g}",
                }
                for sec in value
            ]
        else:
            payload_for_hash[key] = value

    raw = json.dumps(
        payload_for_hash,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]

    return f"RC_LADDER_{digest}"


def rc_ladder_ref_freq(sections: list[dict[str, float]]) -> float:
    """
    Оценивает характерную частоту RC ladder по среднему времени RC-секций.

    Значение используется только для построения частотной сетки.
    """
    if len(sections) == 0:
        raise ValueError("sections must contain at least one RC section")

    taus = []

    for sec in sections:
        R = float(sec["R"])
        C = float(sec["C"])

        if not np.isfinite(R) or not np.isfinite(C) or R <= 0.0 or C <= 0.0:
            raise ValueError("Each RC ladder section must have positive finite R and C")

        taus.append(R * C)

    tau_ref = float(np.exp(np.mean(np.log(taus))))

    return 1.0 / (TWO_PI * tau_ref)


def make_rc_ladder_freq_grid(
    sections: list[dict[str, float]],
    rng: np.random.Generator,
    n_random: int = 32,
    n_local: int = 13,
    f_min_guard: float = 1e-2,
    f_max_guard: float = 1e7,
) -> np.ndarray:
    """
    Формирует частотную сетку для одной RC ladder-схемы.

    Сетка включает:
        1. случайные логарифмически равномерные точки;
        2. локальные точки около характерной частоты цепочки.
    """
    if n_random < 1:
        raise ValueError("n_random must be >= 1")

    if n_local < 1:
        raise ValueError("n_local must be >= 1")

    f_ref = rc_ladder_ref_freq(sections)

    f_low = max(f_min_guard, f_ref / 100.0)
    f_high = min(f_max_guard, f_ref * 100.0)

    if not np.isfinite(f_low) or not np.isfinite(f_high) or f_low <= 0.0 or f_low >= f_high:
        raise ValueError("Invalid frequency range for RC ladder grid")

    random_freqs = np.exp(
        rng.uniform(
            np.log(f_low),
            np.log(f_high),
            size=n_random,
        )
    )

    local_multipliers = np.logspace(-1.0, 1.0, n_local)
    local_freqs = f_ref * local_multipliers

    freqs = np.concatenate([random_freqs, local_freqs])
    freqs = freqs[np.isfinite(freqs)]
    freqs = freqs[(freqs >= f_min_guard) & (freqs <= f_max_guard)]
    freqs = np.unique(np.round(freqs, decimals=9))
    freqs.sort()

    return freqs.astype(float)


def sample_rc_ladder_param_sets(
    n_per_section_count: int = 600,
    n_sections_values: tuple[int, ...] = (1, 2, 3, 4),
    R_range: tuple[float, float] = R_RANGE_OHM,
    C_range: tuple[float, float] = C_RANGE_F,
    Rload_range: tuple[float, float] = RLOAD_RANGE_OHM,
    A_in: float = A_IN_DEFAULT,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Генерирует таблицу параметров RC ladder.

    Parameters
    ----------
    n_per_section_count:
        Число разных схем для каждого значения n_sections.
    n_sections_values:
        Набор рассматриваемых длин RC ladder.
    R_range, C_range, Rload_range:
        Диапазоны параметров компонентов.
    A_in:
        Амплитуда входного сигнала.
    seed:
        Seed генератора случайных чисел.

    Returns
    -------
    pd.DataFrame
        Таблица параметров. Колонка sections хранит список секций
        вида [{"R": ..., "C": ...}, ...].
    """
    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    skipped = 0

    for n_sections in n_sections_values:
        if n_sections < 1:
            raise ValueError("n_sections must be >= 1")

        n_ok = 0

        while n_ok < n_per_section_count:
            sections = []

            for _ in range(n_sections):
                R = float(np.exp(rng.uniform(np.log(R_range[0]), np.log(R_range[1]))))
                C = float(np.exp(rng.uniform(np.log(C_range[0]), np.log(C_range[1]))))

                sections.append(
                    {
                        "R": R,
                        "C": C,
                    }
                )

            Rload = float(
                np.exp(
                    rng.uniform(
                        np.log(Rload_range[0]),
                        np.log(Rload_range[1]),
                    )
                )
            )

            payload = {
                "topology": "RC_LADDER",
                "n_sections": int(n_sections),
                "sections": sections,
                "Rload": Rload,
                "A_in": float(A_in),
            }

            param_set_id = make_param_set_id_rc_ladder(payload)

            if param_set_id in used_ids:
                skipped += 1
                continue

            used_ids.add(param_set_id)

            rows.append(
                {
                    "param_set_id": param_set_id,
                    **payload,
                }
            )

            n_ok += 1

    df = pd.DataFrame(rows)

    print(
        f"[sample_rc_ladder_param_sets] sampled={len(df)} "
        f"skipped={skipped}"
    )

    return df