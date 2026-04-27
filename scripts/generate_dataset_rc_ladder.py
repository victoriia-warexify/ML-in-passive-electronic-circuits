from __future__ import annotations

from pathlib import Path

from ml_circuits.config import PATHS
from ml_circuits.constants import SEED
from ml_circuits.data_generation.generate_rc_ladder_dataset import generate_rc_ladder_dataset_jsonl
from ml_circuits.data_generation.sampling_rc_ladder import sample_rc_ladder_param_sets


def main() -> None:
    # 1) Формирование таблицы параметров для распределённой RC-цепочки
    df_rc_ladder_params = sample_rc_ladder_param_sets(
        n_param_sets_per_n=400,
        section_counts=(1, 2, 3, 4),
        A_in=1.0,
        seed=SEED,
        correlated=True,
        sigma_logR=0.20,
        sigma_logC=0.20,
    )

    param_sets_path = Path(PATHS["rc_ladder_param_sets_csv"])
    param_sets_path.parent.mkdir(parents=True, exist_ok=True)
    df_rc_ladder_params.to_csv(param_sets_path, index=False)
    print(f"Сохранена таблица параметров: {param_sets_path}")

    # 2) Генерация JSONL-датасета в частотной области (AC) для распределённой RC-цепочки
    dataset_path = Path(PATHS["rc_ladder_dataset_jsonl"])
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    rc_ladder_records = generate_rc_ladder_dataset_jsonl(
        df_rc_ladder_params,
        out_jsonl=str(dataset_path),
        f_min_guard=1e-2,
        f_max_guard=1e8,
        dec_span=3.0,
        n_base=48,
        jitter_logf=0.015,
        base_seed=12345,
    )

    print(f"Всего частотных точек: {len(rc_ladder_records)}")


if __name__ == "__main__":
    main()
