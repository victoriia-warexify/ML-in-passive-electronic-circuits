# src/ml_circuits/circuits/elements.py

"""
Элементы электрических схем и класс Circuit.

Модуль содержит внутреннее представление пассивных RLC-схем,
используемое для построения матриц модифицированного узлового анализа.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

@dataclass
class Resistor:
    n1: int; n2: int; R: float  # Омы

@dataclass
class Capacitor:
    n1: int; n2: int; C: float  # Фарады

@dataclass
class Inductor:
    n1: int; n2: int; L: float  # Генри

@dataclass
class VSource: # Идеальный источник напряжения между узлом n_plus и землёй, амплитуда A
    n_plus: int; A: float

class Circuit:
    # --- Инициализация ---
    def __init__(self, N_nodes: int):
        if N_nodes < 1:
            raise ValueError("N_nodes >= 1")
        self.N: int = N_nodes
        self.resistors: List[Resistor] = []
        self.capacitors: List[Capacitor] = []
        self.inductors: List[Inductor] = []
        self.vsources: List[VSource] = []
    
    # --- Методы добавления ---
    def add_resistor(self, n1:int, n2:int, R:float)->None:
        self._check_nodes(n1,n2); self._check_pos(R,"R")
        self.resistors.append(Resistor(n1,n2,R))

    def add_capacitor(self, n1:int, n2:int, C:float)->None:
        self._check_nodes(n1,n2); self._check_pos(C,"C")
        self.capacitors.append(Capacitor(n1,n2,C))

    def add_inductor(self, n1:int, n2:int, L:float)->None:
        self._check_nodes(n1,n2); self._check_pos(L,"L")
        self.inductors.append(Inductor(n1,n2,L))

    def add_vsource(self, n_plus: int, A: float) -> None:
        self.vsources.append(VSource(n_plus=n_plus, A=A))
    
    # --- Проверка ---
    # номер узла находится в допустимом диапазоне [0, ... , N]
    def _check_node(self, n:int)->None:
        if not (0 <= n <= self.N):
            raise ValueError(f"Узел {n} вне диапазона [0..{self.N}]")
    
    # проверка пары узлов, между которыми должен быть подключен элемент
    def _check_nodes(self, n1:int, n2:int)->None:
        self._check_node(n1); self._check_node(n2) # оба узла существуют
        if n1==n2:                                 # узлы не совпадают
            raise ValueError("n1 и n2 совпадают") 

    @staticmethod
    def _check_pos(x: float, name: str) -> None:
        if not (x>0): raise ValueError(f"{name} должно быть > 0")

    # --- Подсчёт числа переменных ---
    def variable_sizes(self) -> Tuple[int, int, int]:
        """
        Возвращает (Nv, Ni, Ntot):
          Nv = N — число узловых напряжений (узлы 1..N)
          Ni = n_Vs + n_L — число токовых переменных (все источники напряжения + все индуктивности)
          Ntot = Nv + Ni — общий размер вектора x = [v; i]
        """
        Nv = self.N
        Ni = len(self.vsources) + len(self.inductors)
        return Nv, Ni, Nv + Ni

    # --- Сборка матриц системы ДАУ МНА ---
    def build_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Формирует матрицы МНА (G, C) для линейной RLC-схемы.

        Матрицы задают левую часть уравнений схемы в форме
            C * x'(t) + G * x(t) = s(t),
        где
            x = [v(1..N), i_vs(1..n_Vs), i_L(1..n_L)].

        В текущем проекте матрицы используются только для AC-анализа:
            (G + jωC) X = S.

        Правая часть S, соответствующая независимым источникам напряжения,
        формируется отдельно в AC-решателе и не входит в build_matrices().

        Штампы:
        - резистор R(n1, n2): вклад в G;
        - конденсатор C(n1, n2): вклад в C;
        - источник напряжения: связи между узловыми напряжениями и током источника в G;
        - индуктивность L(n1, n2): связи в G и диагональный элемент -L в C
            для уравнения ветви v_{n1} - v_{n2} - L di_L/dt = 0.
        """
        Nv, _, Ntot = self.variable_sizes()
        G = np.zeros((Ntot, Ntot), dtype=float)
        C = np.zeros((Ntot, Ntot), dtype=float)

        def add_G_conductance(n1:int, n2:int, g:float):
            if n1 != 0:
                G[n1-1, n1-1] += g
            if n2 != 0:
                G[n2-1, n2-1] += g
            if n1 != 0 and n2 != 0:
                G[n1-1, n2-1] -= g
                G[n2-1, n1-1] -= g

        def add_C_capacitance(n1:int, n2:int, c:float):
            if n1 != 0:
                C[n1-1, n1-1] += c
            if n2 != 0:
                C[n2-1, n2-1] += c
            if n1 != 0 and n2 != 0:
                C[n1-1, n2-1] -= c
                C[n2-1, n1-1] -= c

        # 1) Резисторы -> G_vv
        for r in self.resistors:
            add_G_conductance(r.n1, r.n2, 1.0 / r.R)

        # 2) Конденсаторы -> C_vv
        for c in self.capacitors:
            add_C_capacitance(c.n1, c.n2, c.C)

        # 3) Источники напряжения: связи v <-> i_vs
        for k, vs in enumerate(self.vsources):
            idx_i = Nv + k
            n = vs.n_plus
            if n != 0:
                G[n - 1, idx_i] += 1.0
                G[idx_i, n - 1] += 1.0
            # Правая часть формируется в AC-решателе

        # 4) Индуктивности: связи v <-> i_L, динамика C[ii]-=L
        n_vs = len(self.vsources)
        for k, ind in enumerate(self.inductors):
            idx_i = Nv + n_vs + k
            n1, n2 = ind.n1, ind.n2

            if n1 != 0:
                G[n1 - 1, idx_i] += 1.0
                G[idx_i, n1 - 1] += 1.0
            if n2 != 0:
                G[n2 - 1, idx_i] -= 1.0
                G[idx_i, n2 - 1] -= 1.0

            C[idx_i, idx_i] -= ind.L 


        return G, C
