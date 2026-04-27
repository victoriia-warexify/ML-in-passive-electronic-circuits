# src/ml_circuits/data_generation/sampling_rc_ladder.py

"""
Генерация параметров для RC ladder.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np
import pandas as pd

from ml_circuits.config import C_RANGE_F, R_RANGE_OHM


def make_param_set_id_rc_ladder(
    topo: str,
    sections: list[dict],
    Rload: float | None,
    A_in: float,
) -> str:
    """
    Формирует идентификатор набора параметров RC-цепочки.

    Идентификатор определяется топологией, параметрами всех секций,
    нагрузкой и амплитудой входного сигнала.
    """
    if Rload is None or not np.isfinite(Rload):
        rload_value = None
    else:
        rload_value = float(Rload)

    payload = {
        "topology": topo,
        "sections": [
            {"R": float(s["R"]), "C": float(s["C"])}
            for s in sections
        ],
        "Rload": rload_value,
        "A_in": float(A_in),
    }

    key = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def sample_rc_ladder_param_sets(
    n_param_sets_per_n: int = 400,
    section_counts: tuple[int, ...] = (1, 2, 3, 4),
    A_in: float = 1.0,
    seed: int | None = None,
    correlated: bool = True,
    sigma_logR: float = 0.20,
    sigma_logC: float = 0.20,
    rload_range_ohm: tuple[float, float] = (1e3, 1e6),
) -> pd.DataFrame:
    """
    Генерирует наборы параметров для распределённых RC-цепочек.

    Для каждого числа секций из section_counts формируется
    n_param_sets_per_n конфигураций.

    Возвращает DataFrame со столбцами:
        param_set_id, topology, n_sections, sections, Rload, A_in

    Здесь sections — список словарей вида:
        [{"R": ..., "C": ...}, {"R": ..., "C": ...}, ...]
    где каждый элемент задаёт параметры одного звена цепочки.

    Если correlated=True, параметры секций генерируются как небольшие
    отклонения относительно общих базовых значений R_base и C_base.
    Если correlated=False, параметры секций генерируются независимо.
    """
    rng = np.random.default_rng(seed)

    def log_uniform(a: float, b: float) -> float:
        """Выбор значения по логарифмически равномерному распределению на [a, b]."""
        return float(10.0 ** rng.uniform(np.log10(a), np.log10(b)))

    rows: list[dict] = []

    for n_sections in section_counts:
        for _ in range(n_param_sets_per_n):
            sections: list[dict] = []

            if correlated:
                # Базовые значения с последующими умеренными случайными отклонениями по секциям
                R_base = log_uniform(*R_RANGE_OHM)
                C_base = log_uniform(*C_RANGE_F)

                for _ in range(n_sections):
                    R_i = float(np.clip(
                        R_base * 10.0 ** rng.normal(0.0, sigma_logR),
                        R_RANGE_OHM[0],
                        R_RANGE_OHM[1],
                    ))
                    C_i = float(np.clip(
                        C_base * 10.0 ** rng.normal(0.0, sigma_logC),
                        C_RANGE_F[0],
                        C_RANGE_F[1],
                    ))
                    sections.append({"R": R_i, "C": C_i})
            else:
                # Независимая генерация параметров каждой секции
                for _ in range(n_sections):
                    sections.append({
                        "R": log_uniform(*R_RANGE_OHM),
                        "C": log_uniform(*C_RANGE_F),
                    })

            Rload = log_uniform(*rload_range_ohm)

            param_set_id = make_param_set_id_rc_ladder(
                topo="RC_LADDER",
                sections=sections,
                Rload=Rload,
                A_in=A_in,
            )

            rows.append({
                "param_set_id": param_set_id,
                "topology": "RC_LADDER",
                "n_sections": int(n_sections),
                "sections": sections,
                "Rload": float(Rload),
                "A_in": float(A_in),
            })

    df = pd.DataFrame(rows)

    if df["param_set_id"].nunique() != len(df):
        raise RuntimeError("Duplicate param_set_id detected in RC_LADDER parameter sets")

    print(f"[sample_rc_ladder_param_sets] sampled={len(df)}")
    return df


def _rc_ladder_ref_freq(sections: list[dict]) -> float:
    """
    Вычисляет эвристическую опорную частоту для построения частотной сетки RC-цепочки.

    В качестве опорных значений используются медианы сопротивлений и ёмкостей
    по всем секциям цепочки.
    """
    Rs = np.array([float(s["R"]) for s in sections], dtype=float)
    Cs = np.array([float(s["C"]) for s in sections], dtype=float)

    R_ref = float(np.median(Rs))
    C_ref = float(np.median(Cs))

    return 1.0 / (2.0 * math.pi * R_ref * C_ref)


def make_rc_ladder_freq_grid(
    sections: list[dict],
    *,
    f_min_guard: float = 1e-2,
    f_max_guard: float = 1e8,
    dec_span: float = 3.0,
    n_base: int = 48,
    jitter_logf: float = 0.015,
    seed: int | None = None,
) -> np.ndarray:
    """
    Строит частотную сетку для RC-цепочки вокруг опорной частоты.

    Сетка включает:
    - базовую логарифмическую сетку относительно опорной частоты;
    - набор дополнительных опорных частот в характерной области;
    - слабое логарифмическое случайное возмущение для уменьшения регулярности сетки.

    Возвращает отсортированный массив частот после удаления дублей
    и ограничения защитным диапазоном.
    """
    rng = np.random.default_rng(seed)
    f_ref = _rc_ladder_ref_freq(sections)

    # Базовая логарифмическая сетка относительно опорной частоты
    log_base = np.linspace(-dec_span, dec_span, n_base)
    freqs = f_ref * (10.0 ** log_base)

    # Дополнительные опорные частоты в широкой характерной области
    anchors = f_ref * np.array([
        0.03, 0.05, 0.07, 0.1, 0.15, 0.22, 0.33, 0.47,
        0.68, 1.0, 1.5, 2.2, 3.3, 4.7, 6.8, 10.0, 15.0, 22.0, 33.0
    ], dtype=float)
    focus = f_ref * np.array([
        0.55, 0.70, 0.82, 0.90, 0.96, 1.00, 1.04, 1.12, 1.25, 1.45, 1.80
    ], dtype=float)

    # Слабое логарифмическое возмущение базовой сетки
    jitter = 10.0 ** rng.normal(0.0, jitter_logf, size=freqs.shape)
    freqs = freqs * jitter

    freqs = np.concatenate([freqs, anchors, focus])
    freqs = freqs[np.isfinite(freqs) & (freqs > 0.0)]
    freqs = freqs[(freqs >= f_min_guard) & (freqs <= f_max_guard)]
    freqs = np.unique(np.round(freqs, 10))
    freqs.sort()

    return freqs


rc_ladder_ref_freq = _rc_ladder_ref_freq
