# src/ml_circuits/circuits/mna.py

"""
AC-анализ схем методом модифицированного узлового анализа.
"""

from __future__ import annotations

import numpy as np

from ml_circuits.constants import TWO_PI
from ml_circuits.circuits.elements import Circuit

def ac_solve_circuit(ckt: Circuit, f: float) -> np.ndarray:
    """
    Решает задачу гармонического установившегося режима (AC-анализ) в частотной области:
        (G + jωC) X = S,

    где X — вектор комплексных фазоров неизвестных МНА:
        X = [v(1..N), i_V1, ..., i_Vm, i_L1, ...]^T.

    Предполагается, что каждый источник напряжения VSource задаёт синусоидальное
    возбуждение с амплитудой vs.A и нулевой начальной фазой. В этом случае входной
    фазор имеет вид Vin = A_in ∠ 0, а фазовый сдвиг выходного сигнала относительно
    входного корректно определяется как arg(Vout / Vin).
    """
    # Матрицы MNA: G (проводимости), C (динамические штампы)
    G, C = ckt.build_matrices()
    # Размерности: Nv — число узловых напряжений (без земли), Ni — число токовых переменных, Ntot = Nv+Ni
    Nv, _, Ntot = ckt.variable_sizes()

    w = TWO_PI * float(f)
    # Комплексная системная матрица A(ω) = G + jωC
    A = G.astype(np.complex128) + 1j * w * C.astype(np.complex128)

    # Правая часть системы в частотной области формируется только источниками напряжения.
    S = np.zeros(Ntot, dtype=np.complex128)
    for k, vs in enumerate(ckt.vsources):
        idx_i = Nv + k                  # индекс токовой переменной k-го источника напряжения
        S[idx_i] = complex(vs.A, 0.0)   # фазор источника: амплитуда vs.A, начальная фаза 0

    # Решение комплексной линейной системы A(ω) X = S
    X = np.linalg.solve(A, S)
    return X

def ac_out_amp_phase(ckt: Circuit, f: float, out_node: int, A_in: float) -> tuple[float, float, float]:
    """
    Вычисляет характеристики выходного сигнала в частотной области.

    Возвращает:
        H_mag   — модуль передаточной функции |Vout / Vin|;
        A_out   — амплитуду выходного напряжения |Vout|;
        phi_out — фазовый сдвиг выходного сигнала относительно входного,
                равный arg(Vout / Vin), в радианах.
    """

    if out_node <= 0 or out_node > ckt.N:
        raise ValueError(f"out_node должен быть в диапазоне [1, {ckt.N}]")

    if not np.isfinite(A_in) or A_in == 0:
        raise ValueError("A_in должен быть ненулевым конечным числом")
    
    X = ac_solve_circuit(ckt, f)

    # Узловые напряжения: первые Nv элементов соответствуют узлам 1..N
    Vout = X[out_node - 1]
    # Входной фазор: амплитуда A_in, начальная фаза 0
    Vin = complex(A_in, 0.0)

    H = Vout / Vin
    # Модуль передаточной функции и амплитуда выхода
    H_mag = float(np.abs(H))
    A_out = float(np.abs(Vout))
    # Фазовый сдвиг выходного сигнала относительно входного
    phi_out = float(np.angle(H))

    return H_mag, A_out, phi_out