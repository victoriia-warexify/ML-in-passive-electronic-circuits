# src/ml_circuits/circuits/topologies.py

"""
Генераторы топологий пассивных электронных схем.

Каждая функция возвращает:
    ckt      — объект Circuit;
    out_node — номер выходного узла.
"""

from __future__ import annotations

import numpy as np

from ml_circuits.circuits.elements import Circuit

def is_valid_rload(x: float) -> bool:
    """Проверка корректности сопротивления нагрузки."""
    return np.isfinite(x) and (x > 0.0)

# Генераторы типовых топологий пассивных схем
def gen_rc_lp(R, C, A_in, Rload=None):
    """
    Формирует RC-фильтр нижних частот.

    Схема:
        источник -> последовательный резистор R -> выходной узел,
        с выходного узла конденсатор C подключён к земле.
        При необходимости к выходному узлу добавляется резистивная нагрузка Rload.

    Возвращает:
        ckt      — объект схемы Circuit;
        out_node — номер выходного узла.
    """
    ckt = Circuit(2)
    ckt.add_vsource(1, A_in)
    ckt.add_resistor(1, 2, R)
    ckt.add_capacitor(2, 0, C)
    if Rload is not None:
        ckt.add_resistor(2, 0, Rload)
    return ckt, 2

def gen_rc_hp(R, C, A_in):
    """
    Формирует RC-фильтр верхних частот.

    Схема:
        источник -> последовательный конденсатор C -> выходной узел,
        с выходного узла резистор R подключён к земле.

    Возвращает:
        ckt      — объект схемы Circuit;
        out_node — номер выходного узла.
    """
    ckt = Circuit(2)
    ckt.add_vsource(1, A_in)
    ckt.add_capacitor(1, 2, C)
    ckt.add_resistor(2, 0, R)
    return ckt, 2

def gen_rl_lp(R, L, A_in):
    """
    Формирует RL-фильтр нижних частот.

    Схема:
        источник -> последовательная индуктивность L -> выходной узел,
        с выходного узла резистор R подключён к земле.

    Возвращает:
        ckt      — объект схемы Circuit;
        out_node — номер выходного узла.
    """
    ckt = Circuit(2)
    ckt.add_vsource(1, A_in)
    ckt.add_inductor(1, 2, L)
    ckt.add_resistor(2, 0, R)
    return ckt, 2

def gen_rl_hp(R, L, A_in, Rload=None):
    """
    Формирует RL-фильтр верхних частот.

    Схема:
        источник -> последовательный резистор R -> выходной узел,
        с выходного узла индуктивность L подключена к земле.
        При необходимости к выходному узлу добавляется резистивная нагрузка Rload.

    Возвращает:
        ckt      — объект схемы Circuit;
        out_node — номер выходного узла.
    """
    ckt = Circuit(2)
    ckt.add_vsource(1, A_in)
    ckt.add_resistor(1, 2, R)
    ckt.add_inductor(2, 0, L)
    if Rload is not None:
        ckt.add_resistor(2, 0, Rload)
    return ckt, 2

def gen_rlc_bp(R, L, C, A_in, Rload):
    """
    Генерирует пассивный RLC band-pass фильтр второго порядка.

    Топология:
        Vs(1) -> C -> (2) -> L -> (3)
        R(3) -> GND
        Rload(3) -> GND
        out = v3

    То есть конденсатор и индуктивность соединены последовательно
    между источником и выходным узлом, а выход нагружен параллельным
    соединением R и Rload.

    Физический смысл:
        - на низких частотах конденсатор имеет большой импеданс,
          поэтому сигнал почти не проходит;
        - на высоких частотах индуктивность имеет большой импеданс,
          поэтому сигнал снова подавляется;
        - около f0 реактивности L и C компенсируются, последовательная
          LC-ветвь имеет минимальный импеданс, и передача максимальна.

    Параметры:
        R: Сопротивление нагрузки на выходном узле, Ом.
        L: Индуктивность, Гн.
        C: Ёмкость, Ф.
        A_in: Амплитуда входного синусоидального источника.
        Rload: Дополнительная нагрузка на выходном узле, Ом.

    Возвращает:
        ckt: Объект Circuit.
        out_node: Номер выходного узла, равен 3.
    """
    ckt = Circuit(3)
    ckt.add_vsource(1, A_in)
    ckt.add_capacitor(1, 2, C)
    ckt.add_inductor(2, 3, L)
    ckt.add_resistor(3, 0, R)
    ckt.add_resistor(3, 0, Rload)
    return ckt, 3

def gen_rlc_notch(R, L, C, A_in, Rload):
    """
    Генерирует пассивный RLC notch / band-stop фильтр второго порядка.

    Топология:
        Vs(1) -> R -> (2)
        (2) -> L -> (3) -> C -> GND
        Rload(2) -> GND
        out = v2

    То есть R является последовательным входным сопротивлением,
    а последовательная LC-ветвь подключена от выходного узла к земле.
    Нагрузка Rload также подключена от выходного узла к земле.

    Физический смысл:
        - вне резонанса LC-ветвь имеет большой импеданс, поэтому выход
          примерно определяется делителем R и Rload;
        - на резонансной частоте импедансы L и C компенсируются,
          Z_LC стремится к нулю, LC-ветвь шунтирует выходной узел
          на землю, и на АЧХ появляется провал.

    Параметры:
        R: Последовательное входное сопротивление, Ом.
        L: Индуктивность последовательной LC-ветви, Гн.
        C: Ёмкость последовательной LC-ветви, Ф.
        A_in: Амплитуда входного синусоидального источника.
        Rload: Нагрузка на выходном узле, Ом.

    Возвращает:
        ckt: Объект Circuit.
        out_node: Номер выходного узла, равен 2.
    """
    ckt = Circuit(3)
    ckt.add_vsource(1, A_in)
    ckt.add_resistor(1, 2, R)
    ckt.add_inductor(2, 3, L)
    ckt.add_capacitor(3, 0, C)
    ckt.add_resistor(2, 0, Rload)
    return ckt, 2

def gen_rc_ladder(sections: list[dict], A_in: float, Rload: float | None = None):
    """
    Формирует распределённую RC-цепочку (RC ladder) из заданного числа секций.

    Каждая секция содержит:
        - последовательный резистор R_i,
        - конденсатор C_i, подключённый от соответствующего узла к земле.

    Схема строится как последовательная цепочка звеньев, начиная от входного узла
    источника напряжения. При необходимости на последнем узле добавляется
    резистивная нагрузка Rload.

    Параметры:
        sections — список словарей вида {"R": ..., "C": ...}, задающих параметры секций;
        A_in     — амплитуда входного сигнала;
        Rload    — сопротивление нагрузки на последнем узле, если оно используется.

    Возвращает:
        ckt      — объект схемы Circuit;
        out_node — номер выходного узла, соответствующего последней секции.
    """
    n_sections = len(sections)
    if n_sections < 1:
        raise ValueError("RC_LADDER requires at least 1 section")

    ckt = Circuit(n_sections + 1)
    ckt.add_vsource(1, A_in)

    for i, sec in enumerate(sections, start=1):
        left_node = i
        right_node = i + 1

        R_i = float(sec["R"])
        C_i = float(sec["C"])

        ckt.add_resistor(left_node, right_node, R_i)
        ckt.add_capacitor(right_node, 0, C_i)

    if Rload is not None and is_valid_rload(Rload):
        ckt.add_resistor(n_sections + 1, 0, float(Rload))

    out_node = n_sections + 1
    return ckt, out_node

def build_circuit_for_topology(topo, R, C, L, Rload, A_in, rload_ok) -> tuple[Circuit, int]:
    """
    Строит объект Circuit по названию топологии и параметрам схемы.

    Используется при генерации dataset_ac.csv.
    """
    if topo == "RC_LP":
        return gen_rc_lp(
            R=R,
            C=C,
            A_in=A_in,
            Rload=(Rload if rload_ok else None),
        )

    if topo == "RC_HP":
        return gen_rc_hp(
            R=R,
            C=C,
            A_in=A_in,
        )

    if topo == "RL_LP":
        return gen_rl_lp(
            R=R,
            L=L,
            A_in=A_in,
        )

    if topo == "RL_HP":
        return gen_rl_hp(
            R=R,
            L=L,
            A_in=A_in,
            Rload=(Rload if rload_ok else None),
        )

    if topo == "RLC_BP":
        return gen_rlc_bp(
            R=R,
            L=L,
            C=C,
            A_in=A_in,
            Rload=Rload,
        )

    if topo == "RLC_NOTCH":
        return gen_rlc_notch(
            R=R,
            L=L,
            C=C,
            A_in=A_in,
            Rload=Rload,
        )

    raise ValueError(f"Unknown topology: {topo}")