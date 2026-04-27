from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml_circuits.config import (  # noqa: E402
    AC_DATASET_GENERATION_CONFIG,
    PATHS,
    RC_LADDER_DATASET_GENERATION_CONFIG,
    RC_LADDER_RUN_CONFIGS,
    RC_LADDER_SEEDS,
)
from ml_circuits.constants import SEED  # noqa: E402


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_output_dirs() -> None:
    for key in (
        "artifacts_dir",
        "models_dir",
        "plots_dir",
        "metrics_dir",
        "rc_ladder_artifacts_dir",
        "inverse_artifacts_dir",
    ):
        project_path(PATHS[key]).mkdir(parents=True, exist_ok=True)

    for key in (
        "dataset_csv",
        "filter_param_sets_csv",
        "rc_ladder_param_sets_csv",
        "rc_ladder_dataset_jsonl",
    ):
        project_path(PATHS[key]).parent.mkdir(parents=True, exist_ok=True)


def require_file(path: Path, stage: str, generate_flag: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{stage}: не найден входной файл {path}. "
            f"Сначала запустите `python main.py {generate_flag}` или полный пайплайн `python main.py --all`."
        )


def generate_ac_dataset(skip_existing: bool = False) -> Path:
    dataset_path = project_path(PATHS["dataset_csv"])
    param_sets_path = project_path(PATHS["filter_param_sets_csv"])

    if skip_existing and dataset_path.exists():
        print(f"[generate:ac] Датасет уже существует, пропускаю: {dataset_path}")
        return dataset_path

    try:
        from ml_circuits.data_generation.generate_ac_dataset import generate_dataset_from_params_ac
        from ml_circuits.data_generation.sampling_filters import sample_param_sets
    except ImportError as exc:
        raise ImportError(
            "Не удалось импортировать функции AC-генерации. "
            "Нужны sample_param_sets и generate_dataset_from_params_ac."
        ) from exc

    print("[generate:ac] Начало генерации AC-датасета")
    print(f"[generate:ac] Таблица параметров: {param_sets_path}")
    print(f"[generate:ac] Датасет: {dataset_path}")

    df_params = sample_param_sets(
        seed=SEED,
        **AC_DATASET_GENERATION_CONFIG,
    )
    df_params.to_csv(param_sets_path, index=False)
    print(f"[generate:ac] Сохранена таблица параметров: {param_sets_path}")

    df_ac = generate_dataset_from_params_ac(
        df_params,
        out_csv=str(dataset_path),
    )
    print(f"[generate:ac] Сформирован AC-датасет: {dataset_path} rows={len(df_ac)}")
    return dataset_path


def generate_rc_ladder_dataset(skip_existing: bool = False) -> Path:
    dataset_path = project_path(PATHS["rc_ladder_dataset_jsonl"])
    param_sets_path = project_path(PATHS["rc_ladder_param_sets_csv"])

    if skip_existing and dataset_path.exists():
        print(f"[generate:rc-ladder] Датасет уже существует, пропускаю: {dataset_path}")
        return dataset_path

    try:
        from ml_circuits.data_generation.generate_rc_ladder_dataset import (
            generate_rc_ladder_dataset_jsonl,
        )
        from ml_circuits.data_generation.sampling_rc_ladder import sample_rc_ladder_param_sets
    except ImportError as exc:
        raise ImportError(
            "Не удалось импортировать функции RC-ladder генерации. "
            "Нужны sample_rc_ladder_param_sets и generate_rc_ladder_dataset_jsonl."
        ) from exc

    print("[generate:rc-ladder] Начало генерации RC-ladder датасета")
    print(f"[generate:rc-ladder] Таблица параметров: {param_sets_path}")
    print(f"[generate:rc-ladder] Датасет: {dataset_path}")

    cfg = dict(RC_LADDER_DATASET_GENERATION_CONFIG)
    dataset_kwargs = {
        "f_min_guard": cfg.pop("f_min_guard"),
        "f_max_guard": cfg.pop("f_max_guard"),
        "dec_span": cfg.pop("dec_span"),
        "n_base": cfg.pop("n_base"),
        "jitter_logf": cfg.pop("jitter_logf"),
        "base_seed": cfg.pop("base_seed"),
    }

    df_params = sample_rc_ladder_param_sets(
        seed=SEED,
        **cfg,
    )
    df_params.to_csv(param_sets_path, index=False)
    print(f"[generate:rc-ladder] Сохранена таблица параметров: {param_sets_path}")

    records = generate_rc_ladder_dataset_jsonl(
        df_params,
        out_jsonl=str(dataset_path),
        **dataset_kwargs,
    )
    print(f"[generate:rc-ladder] Сформирован RC-ladder датасет: {dataset_path} rows={len(records)}")
    return dataset_path


def run_generation(skip_existing: bool = False) -> None:
    generate_ac_dataset(skip_existing=skip_existing)
    generate_rc_ladder_dataset(skip_existing=skip_existing)


def run_filter_training(run_plots: bool = False) -> None:
    dataset_path = project_path(PATHS["dataset_csv"])
    require_file(dataset_path, "train", "--generate")

    try:
        from scripts.train_filter_models import main as train_filter_models_main
    except ImportError as exc:
        raise ImportError(
            "Не удалось импортировать scripts.train_filter_models.main. "
            "Добавьте wrapper для обучения основных моделей или проверьте PYTHONPATH."
        ) from exc

    print("[train] Начало обучения моделей основных топологий")
    print(f"[train] Входной датасет: {dataset_path}")
    print(f"[train] Модели: {project_path(PATHS['models_dir'])}")
    print(f"[train] Метрики: {project_path(PATHS['metrics_dir'])}")
    train_filter_models_main(run_plots=run_plots)
    print("[train] Обучение основных моделей завершено")


def run_rc_ladder_stage(run_plots: bool = False) -> None:
    dataset_path = project_path(PATHS["rc_ladder_dataset_jsonl"])
    require_file(dataset_path, "rc-ladder", "--generate")

    try:
        from ml_circuits.training.train_rc_ladder import run_rc_ladder_experiment
    except ImportError as exc:
        raise ImportError(
            "Не удалось импортировать run_rc_ladder_experiment. "
            "Добавьте wrapper для RC-ladder обучения или проверьте модуль training.train_rc_ladder."
        ) from exc

    print("[rc-ladder] Начало RC-ladder этапа")
    print(f"[rc-ladder] Входной датасет: {dataset_path}")

    for cfg in RC_LADDER_RUN_CONFIGS:
        print(f"[rc-ladder] Режим: {cfg['mode_name']} -> {cfg['artifact_dir']}")
        run_rc_ladder_experiment(
            data_path=str(dataset_path),
            mode=cfg["mode"],
            n4_train_fraction=cfg["n4_train_fraction"],
            seeds=RC_LADDER_SEEDS,
        )

    if run_plots:
        try:
            from scripts.make_reports import main as make_reports_main
        except ImportError as exc:
            raise ImportError(
                "Не удалось импортировать scripts.make_reports.main для построения RC-ladder графиков."
            ) from exc
        print("[rc-ladder] Построение графиков и сводных отчётов")
        make_reports_main()

    print("[rc-ladder] RC-ladder этап завершён")


def run_plots_only() -> None:
    plot_cache_dir = project_path(PATHS["plots_dir"]) / ".cache"
    plot_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache_dir / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache_dir / "xdg"))

    try:
        from scripts.make_reports import main as make_reports_main
    except ImportError as exc:
        raise ImportError(
            "Не удалось импортировать scripts.make_reports.main для построения графиков. "
            "Добавьте plot-only wrapper или проверьте модуль scripts.make_reports."
        ) from exc

    print("[plots] Начало построения графиков из существующих артефактов")
    print("[plots] Генерация данных и обучение не запускаются")
    make_reports_main()
    print("[plots] Построение графиков завершено")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Запуск полного ML-пайплайна: генерация данных, обучение фильтров и RC-ladder.",
    )
    parser.add_argument("--all", action="store_true", help="Выполнить полный пайплайн.")
    parser.add_argument("--generate", action="store_true", help="Только генерация AC и RC-ladder датасетов.")
    parser.add_argument("--train", action="store_true", help="Только обучение основных моделей фильтров.")
    parser.add_argument("--rc-ladder", action="store_true", help="Только RC-ladder обучение/эксперимент.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Не перегенерировать датасет, если выходной файл уже существует.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Построить/сохранить графики там, где эта логика уже есть в проекте.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.chdir(PROJECT_ROOT)
    ensure_output_dirs()

    selected = args.all or args.generate or args.train or args.rc_ladder
    run_all = args.all or not selected

    plots_only = args.plots and not (args.all or args.generate or args.train or args.rc_ladder)

    if plots_only:
        run_plots_only()
        print("[main] Пайплайн завершён")
        return

    if run_all or args.generate:
        run_generation(skip_existing=args.skip_existing)

    if run_all or args.train:
        run_filter_training(run_plots=args.plots)

    if run_all or args.rc_ladder:
        run_rc_ladder_stage(run_plots=args.plots)

    print("[main] Пайплайн завершён")


if __name__ == "__main__":
    main()
