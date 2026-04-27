# src/ml_circuits/data_generation/generate_ac_dataset.py

"""
Генерация dataset_ac.csv для типовых пассивных фильтров.
"""

from __future__ import annotations

from collections import Counter
import math

import numpy as np
import pandas as pd

from ml_circuits.config import CHAR_FREQ_MULTIPLIERS as _CHAR_FREQ_MULTIPLIERS
from ml_circuits.constants import EPS_LOG
from ml_circuits.circuits import (
    ac_out_amp_phase,
    build_circuit_for_topology,
    rlc_band_edges,
)
from ml_circuits.data_generation.sampling_filters import (
    _is_valid_rload,
    _r_parallel,
    _seed_from_param_set_id,
    f_key,
)


# Относительные множители для построения дополнительных частотных точек
# в окрестности характерной частоты схемы (fc для фильтров первого порядка, f0 для RLC-цепей)
CHAR_FREQ_MULTIPLIERS = np.array(_CHAR_FREQ_MULTIPLIERS, dtype=float)


def generate_dataset_from_params_ac(
    df_params: pd.DataFrame,
    out_csv: str = "dataset_ac.csv",
    # guard-границы: страховка от экстремальных частот
    f_min_guard: float = 0.1,
    f_max_guard: float = 1e6,
    # относительные сетки (в декадах)
    dec_1st: float = 3.0,
    n_norm_1st: int = 40,
    dec_rlc: float = 2.5,
    n_norm_rlc: int = 24,
    # focus вокруг unloaded при сильном сдвиге
    use_unloaded_focus: bool = True,
    unloaded_ratio_thr: float = 1.5,
    # jitter для "дискретных" топологий
    jitter_logf_1st: float = 0.02,
    jitter_only_topos: tuple[str, ...] = ("RC_HP", "RL_LP"),
    jitter_seed: int = 123,
    # округление частоты перед AC-расчётом
    f_ndigits: int = 10,
) -> pd.DataFrame:
    """
    Генерация AC-датасета для наборов параметров df_params.

    Для каждой схемы строится частотная сетка вокруг характерной частоты:
    - для фильтров 1-го порядка: вокруг fc_loaded;
    - для RLC-фильтров: вокруг резонансной частоты f0.

    Для RLC дополнительно сохраняются диагностические величины:
    f0, Q_eff, f1, f2, R_eq, q_bin, f0_target, Q_eff_target,
    R_eq_target, R_over_Rload.
    """

    err_counter = Counter()
    err_examples: list[dict] = []
    MAX_EXAMPLES = 10

    rows: list[dict] = []
    skipped = 0

    def get_optional_float(row_obj, col: str, default=np.nan) -> float:
        if col not in df_params.columns:
            return default

        value = row_obj[col]

        try:
            value = float(value)
        except Exception:
            return default

        return value if np.isfinite(value) else default

    for _, row in df_params.iterrows():
        topo = str(row["topology"])

        R = float(row["R"])
        C = float(row["C"])
        L = float(row["L"])
        Rload = float(row["Rload"])
        A_in = float(row["A_in"])
        param_set_id = row["param_set_id"]

        q_bin_param = get_optional_float(row, "q_bin")
        f0_target_param = get_optional_float(row, "f0_target")
        Q_eff_target_param = get_optional_float(row, "Q_eff_target")
        R_eq_target_param = get_optional_float(row, "R_eq_target")
        R_over_Rload_param = get_optional_float(row, "R_over_Rload")

        rload_ok = _is_valid_rload(Rload)

        fc_unloaded = None
        fc_loaded = None
        f0 = None
        Q_eff = np.nan
        f1 = np.nan
        f2 = np.nan
        R_eq_diag = np.nan

        # ------------------------------------------------------------
        # 1. Вычисляем характерные частоты
        # ------------------------------------------------------------

        if topo == "RC_LP":
            if R <= 0 or C <= 0:
                skipped += 1
                continue

            fc_unloaded = 1.0 / (2.0 * math.pi * R * C)

            if rload_ok:
                R_eq = _r_parallel(R, Rload)
                fc_loaded = (
                    1.0 / (2.0 * math.pi * R_eq * C)
                    if np.isfinite(R_eq) and R_eq > 0.0
                    else fc_unloaded
                )
            else:
                fc_loaded = fc_unloaded

        elif topo == "RC_HP":
            if R <= 0 or C <= 0:
                skipped += 1
                continue

            fc_unloaded = 1.0 / (2.0 * math.pi * R * C)
            fc_loaded = fc_unloaded

        elif topo == "RL_LP":
            if R <= 0 or L <= 0:
                skipped += 1
                continue

            fc_unloaded = R / (2.0 * math.pi * L)
            fc_loaded = fc_unloaded

        elif topo == "RL_HP":
            if R <= 0 or L <= 0:
                skipped += 1
                continue

            fc_unloaded = R / (2.0 * math.pi * L)

            if rload_ok:
                R_eq = _r_parallel(R, Rload)
                fc_loaded = (
                    R_eq / (2.0 * math.pi * L)
                    if np.isfinite(R_eq) and R_eq > 0.0
                    else fc_unloaded
                )
            else:
                fc_loaded = fc_unloaded

        elif topo in ("RLC_BP", "RLC_NOTCH"):
            if R <= 0 or L <= 0 or C <= 0:
                skipped += 1
                continue

            if not rload_ok:
                skipped += 1
                continue

            f0 = 1.0 / (2.0 * math.pi * math.sqrt(L * C))

            R_eq = _r_parallel(R, Rload)
            R_eq_diag = R_eq

            if (not np.isfinite(R_eq)) or (R_eq <= 0.0):
                skipped += 1
                continue

            Q_eff = math.sqrt(L / C) / R_eq

            if (not np.isfinite(Q_eff)) or (Q_eff <= 0.0):
                skipped += 1
                continue

            f1, f2 = rlc_band_edges(f0, Q_eff)

            if (
                (not np.isfinite(f1))
                or (not np.isfinite(f2))
                or f1 <= 0.0
                or f2 <= 0.0
            ):
                skipped += 1
                continue

        else:
            skipped += 1
            continue

        # ------------------------------------------------------------
        # 2. Проверяем валидность характерной частоты
        # ------------------------------------------------------------

        if topo in ("RLC_BP", "RLC_NOTCH"):
            if (f0 is None) or (not np.isfinite(f0)) or (f0 <= 0.0):
                skipped += 1
                continue
        else:
            if (fc_loaded is None) or (not np.isfinite(fc_loaded)) or (fc_loaded <= 0.0):
                skipped += 1
                continue

        # ------------------------------------------------------------
        # 3. Базовая относительная сетка
        # ------------------------------------------------------------

        if topo in ("RLC_BP", "RLC_NOTCH"):
            log_f_norm = np.linspace(-dec_rlc, dec_rlc, n_norm_rlc)
            freqs_norm = f0 * (10.0 ** log_f_norm)
        else:
            log_f_norm = np.linspace(-dec_1st, dec_1st, n_norm_1st)

            if topo in jitter_only_topos and jitter_logf_1st > 0:
                seed_local = _seed_from_param_set_id(param_set_id, jitter_seed)
                rng_local = np.random.default_rng(seed_local)

                j = rng_local.normal(
                    loc=0.0,
                    scale=float(jitter_logf_1st),
                    size=log_f_norm.shape[0],
                )

                log_f_norm = np.clip(log_f_norm + j, -dec_1st, dec_1st)
                log_f_norm.sort()

            freqs_norm = fc_loaded * (10.0 ** log_f_norm)

        # ------------------------------------------------------------
        # 4. Уплотнение около fc для фильтров 1-го порядка
        # ------------------------------------------------------------

        freqs_cut = np.array([], dtype=float)

        if topo in ("RC_LP", "RC_HP", "RL_LP", "RL_HP"):
            if fc_loaded is not None and np.isfinite(fc_loaded) and fc_loaded > 0.0:
                freqs_cut = fc_loaded * CHAR_FREQ_MULTIPLIERS

        # ------------------------------------------------------------
        # 5. Дополнительный focus около fc_unloaded
        # ------------------------------------------------------------

        freqs_unloaded_focus = np.array([], dtype=float)

        if topo in ("RC_LP", "RL_HP") and use_unloaded_focus:
            if (
                fc_unloaded is not None
                and fc_loaded is not None
                and np.isfinite(fc_unloaded)
                and np.isfinite(fc_loaded)
                and fc_unloaded > 0.0
                and fc_loaded > 0.0
            ):
                ratio = max(fc_unloaded, fc_loaded) / min(fc_unloaded, fc_loaded)

                if ratio >= unloaded_ratio_thr:
                    log_u = np.linspace(-1.0, 1.0, 24)
                    freqs_unloaded_focus = fc_unloaded * (10.0 ** log_u)

        # ------------------------------------------------------------
        # 6. RLC-сетка
        # ------------------------------------------------------------

        freqs_rlc_exact = np.array([], dtype=float)
        freqs_rlc_edges = np.array([], dtype=float)
        freqs_rlc_focus = np.array([], dtype=float)
        freqs_rlc_notch_ultra = np.array([], dtype=float)
        freqs_rlc_notch_inner = np.array([], dtype=float)
        freqs_rlc_bp_tail = np.array([], dtype=float)

        if topo in ("RLC_BP", "RLC_NOTCH"):
            freqs_rlc_exact = np.array([f0], dtype=float)

            if np.isfinite(f1) and np.isfinite(f2) and f1 > 0.0 and f2 > 0.0:
                freqs_rlc_edges = np.array([f1, f2], dtype=float)

            Q_use = max(float(Q_eff), 0.2)

            if topo == "RLC_NOTCH":
                k_focus = 1.2
                min_half_dec = 0.08
                half_dec = max(k_focus / Q_use, min_half_dec)
                n_focus = int(np.clip(36 + 2.0 * Q_use, 44, 110))
            else:
                k_focus = 0.9
                min_half_dec = 0.06
                half_dec = max(k_focus / Q_use, min_half_dec)
                n_focus = int(np.clip(32 + 2.0 * Q_use, 40, 100))

            log_focus = np.linspace(-half_dec, half_dec, n_focus)
            freqs_rlc_focus = f0 * (10.0 ** log_focus)

            if topo == "RLC_NOTCH":
                ultra_half_dec = max(0.30 / Q_use, 0.012)
                n_ultra = int(np.clip(32 + 1.5 * Q_use, 40, 100))

                log_ultra = np.linspace(
                    -ultra_half_dec,
                    ultra_half_dec,
                    n_ultra,
                )

                freqs_rlc_notch_ultra = f0 * (10.0 ** log_ultra)

                rel_span = max(0.8 / Q_use, 0.003)
                rel_span = min(rel_span, 0.25)

                rel_grid = np.linspace(
                    -rel_span,
                    rel_span,
                    int(np.clip(35 + 1.2 * Q_use, 45, 100)),
                )

                rel_mult = 1.0 + rel_grid
                rel_mult = rel_mult[rel_mult > 0.0]

                freqs_rlc_notch_inner = f0 * rel_mult

            if topo == "RLC_BP":
                log_mid = np.linspace(0.8, 1.6, 10)

                freqs_mid = np.concatenate([
                    f0 * (10.0 ** (+log_mid)),
                    f0 * (10.0 ** (-log_mid)),
                ])

                base_lo, base_hi = 1.5, 2.5
                extra = min(0.4, 0.10 * max(Q_use - 1.0, 0.0))

                lo = base_lo
                hi = base_hi + extra

                n_tail = 7
                log_tail = np.linspace(lo, hi, n_tail)

                freqs_far = np.concatenate([
                    f0 * (10.0 ** (+log_tail)),
                    f0 * (10.0 ** (-log_tail)),
                ])

                freqs_rlc_bp_tail = np.unique(np.concatenate([
                    freqs_mid,
                    freqs_far,
                ])).astype(float)

        # ------------------------------------------------------------
        # 7. Объединяем частоты
        # ------------------------------------------------------------

        freqs_local = np.unique(np.concatenate([
            freqs_norm,
            freqs_cut,
            freqs_unloaded_focus,
            freqs_rlc_exact,
            freqs_rlc_edges,
            freqs_rlc_focus,
            freqs_rlc_notch_ultra,
            freqs_rlc_notch_inner,
            freqs_rlc_bp_tail,
        ]))

        freqs_local.sort()

        freqs_local = freqs_local[
            np.isfinite(freqs_local)
            & (freqs_local >= f_min_guard)
            & (freqs_local <= f_max_guard)
        ]

        if freqs_local.size == 0:
            skipped += 1
            continue

        freqs_local = np.array(
            [f_key(float(f), f_ndigits) for f in freqs_local],
            dtype=float,
        )

        freqs_local = freqs_local[
            np.isfinite(freqs_local)
            & (freqs_local >= f_min_guard)
            & (freqs_local <= f_max_guard)
        ]

        freqs_local = np.unique(freqs_local)
        freqs_local.sort()

        if freqs_local.size == 0:
            skipped += 1
            continue

        # ------------------------------------------------------------
        # 8. AC-расчёт
        # ------------------------------------------------------------

        for f in freqs_local:
            try:
                ckt, out_node = build_circuit_for_topology(
                    topo=topo,
                    R=R,
                    C=C,
                    L=L,
                    Rload=Rload,
                    A_in=A_in,
                    rload_ok=rload_ok,
                )

                H_mag, A_out, phi_out = ac_out_amp_phase(
                    ckt,
                    f=f,
                    out_node=int(out_node),
                    A_in=A_in,
                )

                if (
                    (not np.isfinite(H_mag))
                    or (H_mag < 0.0)
                    or (not np.isfinite(A_out))
                    or (not np.isfinite(phi_out))
                ):
                    skipped += 1
                    continue

                log_H_mag = float(np.log(np.clip(H_mag, EPS_LOG, None)))

                rows.append({
                    "param_set_id": param_set_id,
                    "topology": topo,

                    "R": float(R),
                    "C": float(C),
                    "L": float(L),
                    "Rload": float(Rload) if rload_ok else np.nan,
                    "A_in": float(A_in),

                    "fc_unloaded": (
                        float(fc_unloaded)
                        if fc_unloaded is not None and np.isfinite(fc_unloaded)
                        else np.nan
                    ),
                    "fc_loaded": (
                        float(fc_loaded)
                        if fc_loaded is not None and np.isfinite(fc_loaded)
                        else np.nan
                    ),
                    "f0": (
                        float(f0)
                        if f0 is not None and np.isfinite(f0)
                        else np.nan
                    ),
                    "Q_eff": (
                        float(Q_eff)
                        if np.isfinite(Q_eff)
                        else np.nan
                    ),
                    "f1": (
                        float(f1)
                        if np.isfinite(f1)
                        else np.nan
                    ),
                    "f2": (
                        float(f2)
                        if np.isfinite(f2)
                        else np.nan
                    ),
                    "R_eq": (
                        float(R_eq_diag)
                        if np.isfinite(R_eq_diag)
                        else np.nan
                    ),

                    "q_bin": (
                        int(q_bin_param)
                        if np.isfinite(q_bin_param)
                        else np.nan
                    ),
                    "f0_target": (
                        float(f0_target_param)
                        if np.isfinite(f0_target_param)
                        else np.nan
                    ),
                    "Q_eff_target": (
                        float(Q_eff_target_param)
                        if np.isfinite(Q_eff_target_param)
                        else np.nan
                    ),
                    "R_eq_target": (
                        float(R_eq_target_param)
                        if np.isfinite(R_eq_target_param)
                        else np.nan
                    ),
                    "R_over_Rload": (
                        float(R_over_Rload_param)
                        if np.isfinite(R_over_Rload_param)
                        else np.nan
                    ),

                    "f": float(f),
                    "H_mag": float(H_mag),
                    "log_H_mag": log_H_mag,
                    "A_out": float(A_out),
                    "phi_sin": float(np.sin(phi_out)),
                    "phi_cos": float(np.cos(phi_out)),
                })

            except Exception as e:
                skipped += 1
                err_counter[type(e).__name__] += 1

                if len(err_examples) < MAX_EXAMPLES:
                    err_examples.append({
                        "topology": topo,
                        "param_set_id": param_set_id,
                        "R": float(R),
                        "C": float(C),
                        "L": float(L),
                        "Rload": float(Rload),
                        "A_in": float(A_in),
                        "f": float(f),
                        "error_type": type(e).__name__,
                        "error_msg": str(e)[:200],
                    })

                continue

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise RuntimeError(
            "generate_dataset_from_params_ac produced an empty DataFrame. "
            "Check parameter ranges, frequency guards, and AC solver errors."
        )

    df.to_csv(out_csv, index=False)

    print(f"[AC] saved {out_csv} rows={len(df)} skipped={skipped}")

    if skipped > 0:
        print("[AC] error types:", dict(err_counter))

        if err_examples:
            print("[AC] first error examples:")
            for ex in err_examples:
                print(ex)

    return df
