# src/ml_circuits/data_generation/sampling_filters.py

"""
Генерация наборов параметров для пассивных фильтров.

Модуль формирует таблицу filter_param_sets.csv для топологий:
    RC_LP, RC_HP, RL_LP, RL_HP, RLC_BP, RLC_NOTCH.

Для RLC-топологий параметры подбираются так, чтобы получить заданные
диапазоны резонансной частоты и эффективной добротности Q_eff.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np
import pandas as pd

from ml_circuits.config import (
    C_RANGE_F,
    CHAR_FREQ_MULTIPLIERS,
    L_RANGE_H,
    R_RANGE_OHM,
    RLC_C_RANGE_F,
    RLC_L_RANGE_H,
    RLC_Q_BINS,
    RLC_R_OVER_RLOAD_RANGE,
    RLC_R_RANGE_OHM,
    RLC_RLOAD_RANGE_OHM,
    RLOAD_RANGE_OHM,
)
from ml_circuits.constants import R_FLOOR_OHM


def f_key(f: float, ndigits: int = 10) -> float:
    """
    Нормализует значение частоты для использования в качестве ключа.

    Округление устраняет микроскопические различия, возникающие из-за
    арифметики чисел с плавающей точкой.
    """
    return float(np.round(float(f), ndigits))

def make_param_set_id(topo: str, R, C, L, Rload, A_in) -> str:
    """
    Формирует идентификатор набора параметров схемы.

    Идентификатор зависит от топологии и параметров схемы
    (R, C, L, Rload, A_in), но не зависит от частоты. Это позволяет
    объединять все частотные точки одной и той же конфигурации
    в одну группу по param_set_id.
    """
    def _fmt(x) -> str:
        # Единое строковое представление отсутствующего или неопределённого параметра
        if x is None:
            return "None"
        x = float(x)
        if np.isnan(x):
            return "NaN"
        # Формат с 12 значащими цифрами обеспечивает устойчивое и воспроизводимое представление параметров
        return f"{x:.12g}"

    key = "|".join([topo, _fmt(R), _fmt(C), _fmt(L), _fmt(Rload), _fmt(A_in)])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def seed_from_param_set_id(param_set_id: str, base_seed: int) -> int:
    """
    Формирует воспроизводимое целое значение seed по идентификатору набора параметров.

    Используется для детерминированной локальной рандомизации, зависящей от
    конкретной схемы (param_set_id) и общего базового seed.
    """
    h = hashlib.blake2b((str(base_seed) + "|" + param_set_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, byteorder="little", signed=False) % (2**32)


def log_uniform(rng: np.random.Generator, a: float, b: float) -> float:
    """
    Лог-равномерная выборка из [a, b].
    """
    a = float(a)
    b = float(b)

    if not (np.isfinite(a) and np.isfinite(b) and a > 0.0 and b > a):
        raise ValueError(f"Bad log-uniform bounds: a={a}, b={b}")

    return float(10.0 ** rng.uniform(np.log10(a), np.log10(b)))


def balanced_counts(total: int, n_bins: int) -> list[int]:
    """
    Делит total объектов почти поровну между n_bins диапазонами.
    Например, 400 и 4 -> [100, 100, 100, 100].
    """
    base = int(total) // int(n_bins)
    rem = int(total) % int(n_bins)

    return [
        base + (1 if i < rem else 0)
        for i in range(n_bins)
    ]


def r_parallel(a: float | np.ndarray, b: float | np.ndarray) -> float | np.ndarray:
    """
    Эквивалентное сопротивление параллельного соединения a || b.
    """
    return 1.0 / (1.0 / np.maximum(a, R_FLOOR_OHM) + 1.0 / np.maximum(b, R_FLOOR_OHM))


def rlc_feasible_f0_range(
    L_range: tuple[float, float] = RLC_L_RANGE_H,
    C_range: tuple[float, float] = RLC_C_RANGE_F,
) -> tuple[float, float]:
    """
    Возвращает диапазон резонансных частот f0, достижимый
    при заданных диапазонах L и C.
    """
    L_min, L_max = L_range
    C_min, C_max = C_range

    f0_min = 1.0 / (2.0 * math.pi * math.sqrt(L_max * C_max))
    f0_max = 1.0 / (2.0 * math.pi * math.sqrt(L_min * C_min))

    return float(f0_min), float(f0_max)


def rlc_rho_bounds_for_f0(
    f0: float,
    L_range: tuple[float, float] = RLC_L_RANGE_H,
    C_range: tuple[float, float] = RLC_C_RANGE_F,
) -> tuple[float, float] | None:
    """
    Возвращает допустимый диапазон rho = sqrt(L / C) для заданной f0.

    При фиксированной f0:
        L = rho / omega0
        C = 1 / (rho * omega0)

    где omega0 = 2πf0.
    """
    if not np.isfinite(f0) or f0 <= 0.0:
        return None

    w0 = 2.0 * math.pi * f0

    L_min, L_max = L_range
    C_min, C_max = C_range

    rho_min_from_L = L_min * w0
    rho_max_from_L = L_max * w0

    rho_min_from_C = 1.0 / (C_max * w0)
    rho_max_from_C = 1.0 / (C_min * w0)

    rho_min = max(rho_min_from_L, rho_min_from_C)
    rho_max = min(rho_max_from_L, rho_max_from_C)

    if rho_min > rho_max:
        return None

    return float(rho_min), float(rho_max)


def rlc_req_bounds_for_ratio(
    R_range: tuple[float, float] = RLC_R_RANGE_OHM,
    Rload_range: tuple[float, float] = RLC_RLOAD_RANGE_OHM,
    ratio_range: tuple[float, float] = RLC_R_OVER_RLOAD_RANGE,
) -> tuple[float, float]:
    """
    Оценивает достижимый диапазон R_eq = R || Rload
    с учётом диапазонов R, Rload и ограничения на R / Rload.
    """
    R_min, R_max = R_range
    Rl_min, Rl_max = Rload_range
    ratio_min, ratio_max = ratio_range

    candidates = []

    for Rload in (Rl_min, Rl_max):
        for ratio in (ratio_min, ratio_max):
            R = ratio * Rload

            if R_min <= R <= R_max:
                candidates.append(r_parallel(R, Rload))

    for R in (R_min, R_max):
        for ratio in (ratio_min, ratio_max):
            Rload = R / ratio

            if Rl_min <= Rload <= Rl_max:
                candidates.append(r_parallel(R, Rload))

    if not candidates:
        # Грубая оценка, если угловые точки не попали в ограничения.
        candidates = [
            r_parallel(R_min, Rl_min),
            r_parallel(R_min, Rl_max),
            r_parallel(R_max, Rl_min),
            r_parallel(R_max, Rl_max),
        ]

    return float(min(candidates)), float(max(candidates))


def split_req_by_r_ratio(
    rng: np.random.Generator,
    R_eq_target: float,
    R_range: tuple[float, float] = RLC_R_RANGE_OHM,
    Rload_range: tuple[float, float] = RLC_RLOAD_RANGE_OHM,
    ratio_range: tuple[float, float] = RLC_R_OVER_RLOAD_RANGE,
    max_tries: int = 200,
) -> tuple[float, float] | None:
    """
    Подбирает R и Rload так, чтобы их параллельное соединение
    было близко к заданному R_eq_target.
    """
    if not np.isfinite(R_eq_target) or R_eq_target <= 0.0:
        return None

    R_min, R_max = R_range
    Rl_min, Rl_max = Rload_range
    ratio_min, ratio_max = ratio_range

    for _ in range(max_tries):
        ratio = log_uniform(rng, ratio_min, ratio_max)

        # R = ratio * Rload
        # R_eq = R || Rload = ratio / (ratio + 1) * Rload
        Rload = R_eq_target * (ratio + 1.0) / ratio
        R = ratio * Rload

        if R_min <= R <= R_max and Rl_min <= Rload <= Rl_max:
            return float(R), float(Rload)

    return None


def sample_one_rlc_from_target_q(
    rng: np.random.Generator,
    f0: float,
    Q_eff: float,
    max_tries: int = 300,
) -> dict[str, float] | None:
    """
    Сэмплирует один набор параметров RLC-схемы для заданных f0 и Q_eff.

    Для используемой топологии эффективная добротность задаётся приближённо:
        Q_eff = R_eq / sqrt(L / C),
    где R_eq = R || Rload.
    """
    rho_bounds = rlc_rho_bounds_for_f0(f0)

    if rho_bounds is None:
        return None

    rho_min, rho_max = rho_bounds

    for _ in range(max_tries):
        rho = log_uniform(rng, rho_min, rho_max)

        # rho = sqrt(L / C), omega0 = 1 / sqrt(LC)
        w0 = 2.0 * math.pi * f0
        L = rho / w0
        C = 1.0 / (rho * w0)

        if not (RLC_L_RANGE_H[0] <= L <= RLC_L_RANGE_H[1]):
            continue

        if not (RLC_C_RANGE_F[0] <= C <= RLC_C_RANGE_F[1]):
            continue

        R_eq_target = Q_eff * rho
        split = split_req_by_r_ratio(rng, R_eq_target)

        if split is None:
            continue

        R, Rload = split
        R_eq = r_parallel(R, Rload)
        Q_actual = R_eq / rho

        return {
            "R": float(R),
            "Rload": float(Rload),
            "L": float(L),
            "C": float(C),
            "f0": float(f0),
            "Q_eff": float(Q_actual),
            "R_eq": float(R_eq),
        }

    return None


def _sample_order1_param_set(
    rng: np.random.Generator,
    topology: str,
    f_min: float,
    f_max: float,
    A_in: float,
    max_resample: int = 200,
) -> dict[str, Any] | None:
    """
    Сэмплирует один набор параметров для RC/RL-фильтра первого порядка.
    """
    if topology in {"RC_LP", "RC_HP"}:
        for _ in range(max_resample):
            fc = log_uniform(rng, f_min, f_max)
            R = log_uniform(rng, *R_RANGE_OHM)

            if topology == "RC_LP":
                Rload = log_uniform(rng, *RLOAD_RANGE_OHM)
                R_eq = r_parallel(R, Rload)
            else:
                Rload = np.nan
                R_eq = R

            C = 1.0 / (2.0 * math.pi * R_eq * fc)

            if C_RANGE_F[0] <= C <= C_RANGE_F[1]:
                payload = {
                    "A_in": A_in,
                    "R": float(R),
                    "C": float(C),
                    "L": np.nan,
                    "Rload": float(Rload) if np.isfinite(Rload) else np.nan,
                    "fc": float(fc),
                    "f0": np.nan,
                    "Q_eff": np.nan,
                    "R_eq": float(R_eq),
                }
                payload["param_set_id"] = make_param_set_id(topology, payload)
                payload["topology"] = topology
                return payload

    if topology in {"RL_LP", "RL_HP"}:
        for _ in range(max_resample):
            fc = log_uniform(rng, f_min, f_max)
            R = log_uniform(rng, *R_RANGE_OHM)

            if topology == "RL_HP":
                Rload = log_uniform(rng, *RLOAD_RANGE_OHM)
                R_eq = r_parallel(R, Rload)
            else:
                Rload = np.nan
                R_eq = R

            L = R_eq / (2.0 * math.pi * fc)

            if L_RANGE_H[0] <= L <= L_RANGE_H[1]:
                payload = {
                    "A_in": A_in,
                    "R": float(R),
                    "C": np.nan,
                    "L": float(L),
                    "Rload": float(Rload) if np.isfinite(Rload) else np.nan,
                    "fc": float(fc),
                    "f0": np.nan,
                    "Q_eff": np.nan,
                    "R_eq": float(R_eq),
                }
                payload["param_set_id"] = make_param_set_id(topology, payload)
                payload["topology"] = topology
                return payload

    return None


def _sample_rlc_param_sets_for_topology(
    rng: np.random.Generator,
    topology: str,
    n_param_sets: int,
    f_min: float,
    f_max: float,
    A_in: float,
    max_resample: int = 500,
) -> tuple[list[dict[str, Any]], int]:
    """
    Генерирует наборы параметров для одной RLC-топологии
    с балансировкой по диапазонам Q_eff.
    """
    rows: list[dict[str, Any]] = []
    skipped = 0

    counts = balanced_counts(n_param_sets, len(RLC_Q_BINS))

    f0_feasible_min, f0_feasible_max = rlc_feasible_f0_range()
    f0_low = max(f_min, f0_feasible_min)
    f0_high = min(f_max, f0_feasible_max)

    if f0_low >= f0_high:
        raise ValueError(
            "Диапазон f0 для RLC несовместим с диапазонами L и C: "
            f"[{f0_low}, {f0_high}]"
        )

    for q_bin, n_bin in zip(RLC_Q_BINS, counts):
        q_low, q_high = q_bin

        for _ in range(n_bin):
            row = None

            for _try in range(max_resample):
                f0 = log_uniform(rng, f0_low, f0_high)
                Q_target = log_uniform(rng, q_low, q_high)

                params = sample_one_rlc_from_target_q(
                    rng=rng,
                    f0=f0,
                    Q_eff=Q_target,
                )

                if params is None:
                    continue

                payload = {
                    "A_in": A_in,
                    "R": params["R"],
                    "C": params["C"],
                    "L": params["L"],
                    "Rload": params["Rload"],
                    "fc": np.nan,
                    "f0": params["f0"],
                    "Q_eff": params["Q_eff"],
                    "R_eq": params["R_eq"],
                    "q_bin_low": q_low,
                    "q_bin_high": q_high,
                }

                param_set_id = make_param_set_id(topology, payload)

                row = {
                    "topology": topology,
                    "param_set_id": param_set_id,
                    **payload,
                }
                break

            if row is None:
                skipped += 1
            else:
                rows.append(row)

    return rows, skipped


def sample_param_sets(
    n_param_sets_per_topology: int = 400,
    f_min: float = 5.0,
    f_max: float = 500.0,
    A_in: float = 1.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Генерирует таблицу параметров для всех основных топологий.

    Возвращает DataFrame, который затем используется для генерации dataset_ac.csv.
    """
    rng = np.random.default_rng(seed)

    rows: list[dict[str, Any]] = []
    skipped = 0
    used_ids: set[str] = set()

    order1_topologies = ["RC_LP", "RC_HP", "RL_LP", "RL_HP"]

    for topology in order1_topologies:
        n_ok = 0

        while n_ok < n_param_sets_per_topology:
            row = _sample_order1_param_set(
                rng=rng,
                topology=topology,
                f_min=f_min,
                f_max=f_max,
                A_in=A_in,
            )

            if row is None:
                skipped += 1
                continue

            param_set_id = row["param_set_id"]

            if param_set_id in used_ids:
                skipped += 1
                continue

            used_ids.add(param_set_id)
            rows.append(row)
            n_ok += 1

    for topology in ["RLC_BP", "RLC_NOTCH"]:
        rlc_rows, rlc_skipped = _sample_rlc_param_sets_for_topology(
            rng=rng,
            topology=topology,
            n_param_sets=n_param_sets_per_topology,
            f_min=f_min,
            f_max=f_max,
            A_in=A_in,
        )

        skipped += rlc_skipped

        for row in rlc_rows:
            param_set_id = row["param_set_id"]

            if param_set_id in used_ids:
                skipped += 1
                continue

            used_ids.add(param_set_id)
            rows.append(row)

    df = pd.DataFrame(rows)

    print(f"[sample_param_sets] sampled={len(df)} skipped={skipped}")

    return df