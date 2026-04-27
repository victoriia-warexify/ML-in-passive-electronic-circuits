# src/ml_circuits/data_generation/sampling_filters.py

"""
Генерация наборов параметров для пассивных фильтров.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import pandas as pd

from ml_circuits.config import (
    C_RANGE_F,
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


# Один идентификатор param_set_id соответствует одной конфигурации схемы и не зависит от частоты расчёта
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


def f_key(f: float, ndigits: int = 10) -> float:
    """
    Нормализует значение частоты для использования в качестве ключа.

    Округление устраняет микроскопические различия, возникающие из-за
    арифметики чисел с плавающей точкой.
    """
    return float(np.round(float(f), ndigits))


def _is_valid_rload(x: float) -> bool:
    """Проверка корректности сопротивления нагрузки."""
    return np.isfinite(x) and (x > 0.0)


def _r_parallel(r1: float, r2: float) -> float:
    """Эквивалент параллельного соединения сопротивлений r1 || r2."""
    r1 = float(r1)
    r2 = float(r2)
    if (not np.isfinite(r1)) or (not np.isfinite(r2)) or (r1 <= 0) or (r2 <= 0):
        return float("nan")
    r1 = max(r1, R_FLOOR_OHM)
    r2 = max(r2, R_FLOOR_OHM)
    return 1.0 / (1.0 / r1 + 1.0 / r2)


def _log_uniform(rng: np.random.Generator, a: float, b: float) -> float:
    """
    Лог-равномерная выборка из [a, b].
    """
    a = float(a)
    b = float(b)

    if not (np.isfinite(a) and np.isfinite(b) and a > 0.0 and b > a):
        raise ValueError(f"Bad log-uniform bounds: a={a}, b={b}")

    return float(10.0 ** rng.uniform(np.log10(a), np.log10(b)))


def _balanced_counts(total: int, n_bins: int) -> list[int]:
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


def _rlc_feasible_f0_range(f_min: float, f_max: float) -> tuple[float, float]:
    """
    Возвращает реально достижимый диапазон f0 для RLC с учётом
    диапазонов L и C.

    f0 = 1 / (2πsqrt(LC))

    При твоих диапазонах:
        L = 1e-4...1e-1 Гн
        C = 1e-8...1e-4 Ф

    минимальная f0 задаётся max(L)*max(C),
    максимальная f0 задаётся min(L)*min(C).
    """
    f0_min_components = 1.0 / (
        2.0 * math.pi * math.sqrt(RLC_L_RANGE_H[1] * RLC_C_RANGE_F[1])
    )

    f0_max_components = 1.0 / (
        2.0 * math.pi * math.sqrt(RLC_L_RANGE_H[0] * RLC_C_RANGE_F[0])
    )

    lo = max(float(f_min), f0_min_components)
    hi = min(float(f_max), f0_max_components)

    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo > 0.0):
        raise ValueError(
            "No feasible RLC f0 interval. "
            f"Requested [{f_min}, {f_max}], component-implied "
            f"[{f0_min_components:.6g}, {f0_max_components:.6g}]."
        )

    return lo, hi


def _rlc_rho_bounds_for_f0(f0: float) -> tuple[float, float]:
    """
    Для заданной f0 находит допустимый диапазон rho = sqrt(L / C).

    Используем:
        f0 = 1 / (2πsqrt(LC))
        omega0 = 2πf0
        rho = sqrt(L / C)

    Тогда:
        L = rho / omega0
        C = 1 / (rho * omega0)

    Ограничения на L и C дают ограничения на rho.
    """
    w0 = 2.0 * math.pi * float(f0)

    rho_lo = max(
        RLC_L_RANGE_H[0] * w0,
        1.0 / (RLC_C_RANGE_F[1] * w0),
    )

    rho_hi = min(
        RLC_L_RANGE_H[1] * w0,
        1.0 / (RLC_C_RANGE_F[0] * w0),
    )

    return float(rho_lo), float(rho_hi)


def _rlc_req_bounds_for_ratio(k: float) -> tuple[float, float]:
    """
    Для заданного k = R / Rload находит допустимый диапазон R_eq.

    Если:
        k = R / Rload

    и:
        R_eq = R || Rload

    то:
        R = R_eq * (1 + k)
        Rload = R_eq * (1 + k) / k

    Отсюда ограничения на R и Rload превращаются
    в ограничения на R_eq.
    """
    k = float(k)

    if not (np.isfinite(k) and k > 0.0):
        return np.nan, np.nan

    req_lo = max(
        RLC_R_RANGE_OHM[0] / (1.0 + k),
        RLC_RLOAD_RANGE_OHM[0] * k / (1.0 + k),
    )

    req_hi = min(
        RLC_R_RANGE_OHM[1] / (1.0 + k),
        RLC_RLOAD_RANGE_OHM[1] * k / (1.0 + k),
    )

    return float(req_lo), float(req_hi)


def _split_req_by_r_ratio(R_eq: float, k: float) -> tuple[float, float]:
    """
    Восстанавливает R и Rload из R_eq и k = R / Rload.

    Формулы:
        R = R_eq * (1 + k)
        Rload = R_eq * (1 + k) / k
    """
    R_eq = float(R_eq)
    k = float(k)

    if not (np.isfinite(R_eq) and R_eq > 0.0):
        raise ValueError("R_eq must be positive")

    if not (np.isfinite(k) and k > 0.0):
        raise ValueError("k must be positive")

    R = R_eq * (1.0 + k)
    Rload = R_eq * (1.0 + k) / k

    return float(R), float(Rload)


def _sample_one_rlc_from_target_q(
    rng: np.random.Generator,
    *,
    topology: str,
    f_min: float,
    f_max: float,
    A_in: float,
    q_bin: int,
    max_resample: int,
) -> dict:
    """
    Генерирует один RLC-набор не через случайные R, L, C, Rload,
    а через физически осмысленные параметры:

        f0
        Q_eff
        rho = sqrt(L / C)
        R_eq = R || Rload
        k = R / Rload

    Основные формулы:

        Q_eff = sqrt(L / C) / R_eq
        f0 = 1 / (2πsqrt(LC))

    Поэтому:
        rho = sqrt(L / C)
        R_eq = rho / Q_eff
        L = rho / omega0
        C = 1 / (rho * omega0)
    """
    if topology not in {"RLC_BP", "RLC_NOTCH"}:
        raise ValueError("RLC sampler expects RLC_BP or RLC_NOTCH")

    f0_lo, f0_hi = _rlc_feasible_f0_range(f_min, f_max)

    q_lo, q_hi = RLC_Q_BINS[int(q_bin)]

    for _try in range(max_resample):
        # 1. Целевая резонансная частота
        f0 = _log_uniform(rng, f0_lo, f0_hi)

        # 2. Целевая добротность из конкретного q_bin
        Q_eff = _log_uniform(rng, q_lo, q_hi)

        # 3. Отношение R / Rload
        k = _log_uniform(rng, *RLC_R_OVER_RLOAD_RANGE)

        # 4. Допустимый диапазон rho из ограничений на L и C
        rho_lc_lo, rho_lc_hi = _rlc_rho_bounds_for_f0(f0)

        # 5. Допустимый диапазон R_eq из ограничений на R и Rload
        req_lo, req_hi = _rlc_req_bounds_for_ratio(k)

        # 6. Так как Q_eff = rho / R_eq,
        #    то rho = Q_eff * R_eq.
        #    Пересекаем допустимые диапазоны rho.
        rho_lo = max(rho_lc_lo, Q_eff * req_lo)
        rho_hi = min(rho_lc_hi, Q_eff * req_hi)

        if not (
            np.isfinite(rho_lo)
            and np.isfinite(rho_hi)
            and rho_hi > rho_lo > 0.0
        ):
            continue

        # 7. Выбираем rho внутри физически допустимого пересечения
        rho = _log_uniform(rng, rho_lo, rho_hi)

        # 8. Из Q_eff = rho / R_eq получаем R_eq
        R_eq = rho / Q_eff

        # 9. Из R_eq и k восстанавливаем R и Rload
        R, Rload = _split_req_by_r_ratio(R_eq, k)

        # 10. Из f0 и rho восстанавливаем L и C
        w0 = 2.0 * math.pi * f0

        L = rho / w0
        C = 1.0 / (rho * w0)

        # 11. Жёсткие проверки диапазонов
        if not (
            RLC_R_RANGE_OHM[0] <= R <= RLC_R_RANGE_OHM[1]
            and RLC_RLOAD_RANGE_OHM[0] <= Rload <= RLC_RLOAD_RANGE_OHM[1]
            and RLC_L_RANGE_H[0] <= L <= RLC_L_RANGE_H[1]
            and RLC_C_RANGE_F[0] <= C <= RLC_C_RANGE_F[1]
        ):
            continue

        # 12. Финальная численная проверка формул
        f0_check = 1.0 / (2.0 * math.pi * math.sqrt(L * C))
        R_eq_check = _r_parallel(R, Rload)
        Q_check = math.sqrt(L / C) / R_eq_check

        if not (
            np.isfinite(f0_check)
            and np.isfinite(Q_check)
            and abs(math.log(f0_check / f0)) < 1e-10
            and abs(math.log(Q_check / Q_eff)) < 1e-10
        ):
            continue

        param_set_id = make_param_set_id(
            topology,
            R,
            C,
            L,
            Rload,
            A_in,
        )

        return {
            "param_set_id": param_set_id,
            "topology": topology,
            "R": float(R),
            "C": float(C),
            "L": float(L),
            "Rload": float(Rload),
            "A_in": float(A_in),

            # Диагностические величины.
            # Их полезно оставить в df_params.
            "f0_target": float(f0),
            "Q_eff_target": float(Q_eff),
            "R_eq_target": float(R_eq),
            "R_over_Rload": float(k),
            "q_bin": int(q_bin),
        }

    raise RuntimeError(
        f"Could not sample {topology} for q_bin={q_bin} "
        f"after {max_resample} tries. "
        "Check RLC ranges or reduce Q range."
    )


def sample_param_sets(
    n_param_sets_per_topology: int = 50,
    f_min: float = 5.0,
    f_max: float = 500.0,
    A_in: float = 1.0,
    max_resample: int = 5000,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Сэмплирует параметры схем.

    Для RC/RL первого порядка используется старая логика:
        выбираем fc_target и один компонент,
        второй компонент вычисляем из fc.

    Для RLC второго порядка используется новая логика:
        выбираем f0 и Q_eff,
        затем вычисляем R, Rload, L, C так,
        чтобы выбранный Q_eff действительно получился.

    Возвращает DataFrame со столбцами:
        param_set_id, topology, R, C, L, Rload, A_in,
        f0_target, Q_eff_target, R_eq_target, R_over_Rload, q_bin
    """
    rng = np.random.default_rng(seed)

    def log_uniform_pair(rng_pair) -> float:
        a, b = rng_pair
        return _log_uniform(rng, a, b)

    def sample_fc_target() -> float:
        return _log_uniform(rng, f_min, f_max)

    topo_list = [
        "RC_LP",
        "RC_HP",
        "RL_LP",
        "RL_HP",
        "RLC_BP",
        "RLC_NOTCH",
    ]

    rows: list[dict] = []
    skipped = 0
    used_ids: set[str] = set()

    for topo in topo_list:
        # ============================================================
        # RLC: новая генерация с равномерным покрытием q_bin
        # ============================================================
        if topo in {"RLC_BP", "RLC_NOTCH"}:
            counts_per_q_bin = _balanced_counts(
                n_param_sets_per_topology,
                len(RLC_Q_BINS),
            )

            for q_bin, need in enumerate(counts_per_q_bin):
                collected_bin = 0

                while collected_bin < need:
                    try:
                        row = _sample_one_rlc_from_target_q(
                            rng,
                            topology=topo,
                            f_min=f_min,
                            f_max=f_max,
                            A_in=A_in,
                            q_bin=q_bin,
                            max_resample=max_resample,
                        )
                    except RuntimeError:
                        skipped += 1
                        raise

                    if row["param_set_id"] in used_ids:
                        skipped += 1
                        continue

                    used_ids.add(row["param_set_id"])
                    rows.append(row)
                    collected_bin += 1

            continue

        # ============================================================
        # RC/RL: твоя старая логика первого порядка
        # ============================================================
        collected = 0

        while collected < n_param_sets_per_topology:
            ok = False

            # ---------------------------
            # RC, 1-й порядок
            # ---------------------------
            if topo in ("RC_LP", "RC_HP"):
                for _try in range(max_resample):
                    fc_tgt = sample_fc_target()
                    R = log_uniform_pair(R_RANGE_OHM)

                    if topo == "RC_LP":
                        Rload = log_uniform_pair(RLOAD_RANGE_OHM)
                        R_eq = _r_parallel(R, Rload)

                        if (not np.isfinite(R_eq)) or (R_eq <= 0):
                            continue

                        # Подбираем C под нужную fc_loaded.
                        C = 1.0 / (2.0 * math.pi * R_eq * fc_tgt)
                    else:
                        Rload = float("nan")
                        C = 1.0 / (2.0 * math.pi * R * fc_tgt)

                    L = float("nan")

                    if np.isfinite(C) and (C_RANGE_F[0] <= C <= C_RANGE_F[1]):
                        ok = True
                        break

            # ---------------------------
            # RL, 1-й порядок
            # ---------------------------
            elif topo in ("RL_LP", "RL_HP"):
                for _try in range(max_resample):
                    fc_tgt = sample_fc_target()
                    R = log_uniform_pair(R_RANGE_OHM)

                    if topo == "RL_HP":
                        Rload = log_uniform_pair(RLOAD_RANGE_OHM)
                        R_eq = _r_parallel(R, Rload)

                        if (not np.isfinite(R_eq)) or (R_eq <= 0):
                            continue

                        # Подбираем L под нужную fc_loaded.
                        L = R_eq / (2.0 * math.pi * fc_tgt)
                    else:
                        Rload = float("nan")
                        L = R / (2.0 * math.pi * fc_tgt)

                    C = float("nan")

                    if np.isfinite(L) and (L_RANGE_H[0] <= L <= L_RANGE_H[1]):
                        ok = True
                        break

            else:
                raise RuntimeError("Unknown topology")

            if not ok:
                skipped += 1
                continue

            param_set_id = make_param_set_id(topo, R, C, L, Rload, A_in)

            if param_set_id in used_ids:
                skipped += 1
                continue

            used_ids.add(param_set_id)

            rows.append({
                "param_set_id": param_set_id,
                "topology": topo,
                "R": float(R),
                "C": float(C),
                "L": float(L),
                "Rload": float(Rload),
                "A_in": float(A_in),

                # Для топологий первого порядка эти поля не применяются.
                "f0_target": np.nan,
                "Q_eff_target": np.nan,
                "R_eq_target": np.nan,
                "R_over_Rload": np.nan,
                "q_bin": np.nan,
            })

            collected += 1

    df_params = pd.DataFrame(rows)

    if df_params["param_set_id"].nunique() != len(df_params):
        raise RuntimeError("Duplicate param_set_id detected")

    print(f"[sample_param_sets] sampled={len(df_params)} skipped={skipped}")

    return df_params


def seed_from_param_set_id(param_set_id: str, base_seed: int) -> int:
    """
    Формирует воспроизводимое целое значение seed по идентификатору набора параметров.

    Используется для детерминированной локальной рандомизации, зависящей от
    конкретной схемы (param_set_id) и общего базового seed.
    """
    return _seed_from_param_set_id(param_set_id, base_seed)


def _seed_from_param_set_id(param_set_id: str, base_seed: int) -> int:
    """
    Формирует воспроизводимое целое значение seed по идентификатору набора параметров.

    Используется для детерминированной локальной рандомизации, зависящей от
    конкретной схемы (param_set_id) и общего базового seed.
    """
    h = hashlib.blake2b((str(base_seed) + "|" + param_set_id).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(h, byteorder="little", signed=False) % (2**32)


log_uniform = _log_uniform
balanced_counts = _balanced_counts
r_parallel = _r_parallel
