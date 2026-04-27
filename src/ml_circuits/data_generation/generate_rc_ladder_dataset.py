# src/ml_circuits/data_generation/generate_rc_ladder_dataset.py

"""
Генерация dataset_rc_ladder.jsonl для RC ladder.
"""

from __future__ import annotations

from collections import Counter
import json

import numpy as np
import pandas as pd

from ml_circuits.constants import EPS_LOG
from ml_circuits.circuits import ac_out_amp_phase, gen_rc_ladder
from ml_circuits.data_generation.sampling_filters import _seed_from_param_set_id
from ml_circuits.data_generation.sampling_rc_ladder import make_rc_ladder_freq_grid


def generate_rc_ladder_dataset_jsonl(
    df_params: pd.DataFrame,
    out_jsonl: str = "dataset_rc_ladder.jsonl",
    *,
    f_min_guard: float = 1e-2,
    f_max_guard: float = 1e8,
    dec_span: float = 3.0,
    n_base: int = 48,
    jitter_logf: float = 0.015,
    base_seed: int = 12345,
) -> list[dict]:
    """
    Генерирует датасет для RC_LADDER в формате JSONL.

    Каждая строка JSONL соответствует одной частотной точке одной схемы.
    Для каждой конфигурации строится индивидуальная частотная сетка,
    после чего выполняется AC-расчёт и сохраняются вычисленные признаки.

    Возвращает список словарей, записанных в JSONL-файл.
    """
    required_cols = {"param_set_id", "topology", "n_sections", "sections", "Rload", "A_in"}
    missing_cols = required_cols - set(df_params.columns)
    if missing_cols:
        raise ValueError(
            f"Во входном DataFrame отсутствуют обязательные столбцы: {sorted(missing_cols)}"
        )

    records: list[dict] = []
    err_counter = Counter()
    err_examples: list[dict] = []
    MAX_EXAMPLES = 10
    skipped = 0

    with open(out_jsonl, "w", encoding="utf-8") as f_out:
        for row in df_params.itertuples(index=False):
            if row.topology != "RC_LADDER":
                continue

            param_seed = _seed_from_param_set_id(row.param_set_id, base_seed)

            freqs = make_rc_ladder_freq_grid(
                row.sections,
                f_min_guard=f_min_guard,
                f_max_guard=f_max_guard,
                dec_span=dec_span,
                n_base=n_base,
                jitter_logf=jitter_logf,
                seed=param_seed,
            )

            for f in freqs:
                try:
                    ckt, out_node = gen_rc_ladder(
                        sections=row.sections,
                        A_in=float(row.A_in),
                        Rload=float(row.Rload),
                    )

                    H_mag, A_out, phi_out = ac_out_amp_phase(
                        ckt=ckt,
                        f=float(f),
                        out_node=int(out_node),
                        A_in=float(row.A_in),
                    )

                    if (
                        (not np.isfinite(H_mag)) or
                        (H_mag < 0) or
                        (not np.isfinite(A_out)) or
                        (not np.isfinite(phi_out))
                    ):
                        skipped += 1
                        continue

                    rec = {
                        "param_set_id": row.param_set_id,
                        "topology": "RC_LADDER",
                        "n_sections": int(row.n_sections),
                        "sections": [
                            {"R": float(sec["R"]), "C": float(sec["C"])}
                            for sec in row.sections
                        ],
                        "Rload": float(row.Rload),
                        "A_in": float(row.A_in),
                        "frequency_hz": float(f),
                        "H_mag": float(H_mag),
                        "log_H_mag": float(np.log(np.clip(H_mag, EPS_LOG, None))),
                        "A_out": float(A_out),
                        "phi_rad": float(phi_out),
                        "phi_sin": float(np.sin(phi_out)),
                        "phi_cos": float(np.cos(phi_out)),
                    }

                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    records.append(rec)

                except (ValueError, np.linalg.LinAlgError, TypeError, KeyError) as e:
                    skipped += 1
                    err_counter[type(e).__name__] += 1

                    if len(err_examples) < MAX_EXAMPLES:
                        err_examples.append({
                            "param_set_id": row.param_set_id,
                            "n_sections": int(row.n_sections),
                            "Rload": float(row.Rload),
                            "A_in": float(row.A_in),
                            "frequency_hz": float(f),
                            "error_type": type(e).__name__,
                            "error_msg": str(e)[:200],
                        })
                    continue

    print(f"[generate_rc_ladder_dataset_jsonl] wrote {len(records)} rows to {out_jsonl}")
    if skipped > 0:
        print("[generate_rc_ladder_dataset_jsonl] skipped:", skipped)
        print("[generate_rc_ladder_dataset_jsonl] error types:", dict(err_counter))
        if err_examples:
            print("[generate_rc_ladder_dataset_jsonl] first error examples:")
            for ex in err_examples:
                print(ex)

    return records
