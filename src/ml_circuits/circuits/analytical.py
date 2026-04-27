# src/ml_circuits/circuits/analytical.py

"""
Аналитические формулы для передаточных функций пассивных схем.

Функции из этого модуля используются для sanity-check датасетов
и для сопоставления численного МНА-расчёта с известными формулами.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ml_circuits.constants import TWO_PI


def _parallel_z(z1, z2):
    """
    Эквивалентный импеданс параллельного соединения z1 || z2.
    """
    return 1.0 / (1.0 / z1 + 1.0 / z2)


def parallel_z(z1, z2):
    return _parallel_z(z1, z2)

def analytical_H_for_df(df_part: pd.DataFrame) -> np.ndarray:
    """
    Аналитически вычисляет комплексную передаточную функцию H(jw)
    для всех топологий из dataset_ac.csv.
    """
    n = len(df_part)
    H = np.full(n, np.nan + 1j * np.nan, dtype=complex)

    topo = df_part["topology"].to_numpy(str)
    f = df_part["f"].to_numpy(float)
    w = TWO_PI * f
    jw = 1j * w

    R = df_part["R"].to_numpy(float)
    C = df_part["C"].to_numpy(float)
    L = df_part["L"].to_numpy(float)
    Rload = df_part["Rload"].to_numpy(float)

    # RC_HP: Vs -> C -> out, R(out) -> GND
    m = topo == "RC_HP"
    if np.any(m):
        Zc = 1.0 / (jw[m] * C[m])
        H[m] = R[m] / (R[m] + Zc)

    # RC_LP: Vs -> R -> out, C(out) -> GND, optional Rload
    m = topo == "RC_LP"
    if np.any(m):
        Zc = 1.0 / (jw[m] * C[m])
        Zp = Zc.copy()

        has_load = np.isfinite(Rload[m]) & (Rload[m] > 0.0)
        if np.any(has_load):
            Zp[has_load] = parallel_z(
                Zc[has_load],
                Rload[m][has_load].astype(complex),
            )

        H[m] = Zp / (R[m] + Zp)

    # RL_LP: Vs -> L -> out, R(out) -> GND
    m = topo == "RL_LP"
    if np.any(m):
        Zl = jw[m] * L[m]
        H[m] = R[m] / (R[m] + Zl)

    # RL_HP: Vs -> R -> out, L(out) -> GND, optional Rload
    m = topo == "RL_HP"
    if np.any(m):
        Zl = jw[m] * L[m]
        Zp = Zl.copy()

        has_load = np.isfinite(Rload[m]) & (Rload[m] > 0.0)
        if np.any(has_load):
            Zp[has_load] = parallel_z(
                Zl[has_load],
                Rload[m][has_load].astype(complex),
            )

        H[m] = Zp / (R[m] + Zp)

    # RLC_BP:
    # Vs -> C -> L -> out
    # out -> R || Rload -> GND
    m = topo == "RLC_BP"
    if np.any(m):
        Zc = 1.0 / (jw[m] * C[m])
        Zl = jw[m] * L[m]
        R_eq = parallel_z(
            R[m].astype(complex),
            Rload[m].astype(complex),
        )
        H[m] = R_eq / (R_eq + Zl + Zc)

    # RLC_NOTCH:
    # Vs -> R -> out
    # out -> Rload -> GND
    # out -> L -> C -> GND
    m = topo == "RLC_NOTCH"
    if np.any(m):
        Zc = 1.0 / (jw[m] * C[m])
        Zl = jw[m] * L[m]
        Zlc = Zl + Zc

        Zp = parallel_z(
            Rload[m].astype(complex),
            Zlc,
        )

        H[m] = Zp / (R[m] + Zp)

    return H

def rlc_band_edges(f0: float, Q: float) -> tuple[float, float]:
    """
    Аналитические граничные частоты RLC-фильтра второго порядка.

    Для RLC_BP это частоты -3 dB относительно пика.
    Для RLC_NOTCH используются как характерные границы области подавления.

    f0 — резонансная частота, Hz.
    Q  — добротность.
    """
    if (not np.isfinite(f0)) or (not np.isfinite(Q)) or f0 <= 0.0 or Q <= 0.0:
        return np.nan, np.nan

    alpha = math.sqrt(1.0 + 4.0 * Q * Q)
    f1 = f0 * (alpha - 1.0) / (2.0 * Q)
    f2 = f0 * (alpha + 1.0) / (2.0 * Q)

    return f1, f2

def analytical_rc_ladder_n1_H(
    R: float,
    C: float,
    Rload: float,
    f: float,
) -> complex:
    """
    Аналитическая передаточная функция RC ladder при одной секции.

    При n_sections = 1 схема совпадает с нагруженным RC low-pass:
        Vs -> R -> out,
        out -> C -> GND,
        out -> Rload -> GND.

    Эта формула полезна для проверки генератора RC ladder.
    """
    w = TWO_PI * float(f)

    Zc = 1.0 / (1j * w * C)
    Zp = parallel_z(Zc, complex(Rload))

    return Zp / (R + Zp)
