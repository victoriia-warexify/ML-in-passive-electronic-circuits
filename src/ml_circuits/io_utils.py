# src/ml_circuits/io_utils.py

"""
Вспомогательные функции для чтения и записи файлов.

Этот модуль содержит только операции ввода-вывода:
создание директорий, чтение и сохранение CSV, JSON, JSONL,
а также служебные функции для сериализации результатов экспериментов.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    """
    Создаёт директорию, если она ещё не существует.

    Parameters
    ----------
    path:
        Путь к директории.

    Returns
    -------
    Path
        Объект Path для созданной или уже существующей директории.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(path: str | Path) -> Path:
    """
    Создаёт родительскую директорию для файла, если она ещё не существует.

    Parameters
    ----------
    path:
        Путь к файлу.

    Returns
    -------
    Path
        Объект Path для исходного файла.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_csv(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """
    Загружает CSV-файл в DataFrame.
    """
    return pd.read_csv(path, **kwargs)


def save_csv(
    df: pd.DataFrame,
    path: str | Path,
    index: bool = False,
    **kwargs: Any,
) -> None:
    """
    Сохраняет DataFrame в CSV-файл.
    """
    path = ensure_parent_dir(path)
    df.to_csv(path, index=index, **kwargs)


def load_json(path: str | Path) -> Any:
    """
    Загружает JSON-файл.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """
    Сохраняет объект в JSON-файл.
    """
    path = ensure_parent_dir(path)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            obj,
            f,
            ensure_ascii=False,
            indent=indent,
            default=json_default,
        )


def load_jsonl(path: str | Path) -> pd.DataFrame:
    """
    Загружает JSONL-файл в DataFrame.

    JSONL используется для датасета RC ladder, где каждая строка файла
    является отдельной JSON-записью.
    """
    return pd.read_json(path, lines=True)


def save_jsonl(df: pd.DataFrame, path: str | Path) -> None:
    """
    Сохраняет DataFrame в JSONL-файл.
    """
    path = ensure_parent_dir(path)
    df.to_json(path, orient="records", lines=True, force_ascii=False)


def json_default(obj: Any) -> Any:
    """
    Преобразует NumPy- и Path-объекты в типы, совместимые с JSON.

    Используется при сохранении метрик и конфигураций экспериментов.
    """
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return float(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.bool_):
        return bool(obj)

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_metrics(path: str | Path) -> dict[str, Any]:
    """
    Загружает JSON-файл с метриками эксперимента.
    """
    metrics = load_json(path)

    if not isinstance(metrics, dict):
        raise ValueError(f"Expected metrics JSON to contain a dictionary, got {type(metrics)}")

    return metrics


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    """
    Сохраняет словарь метрик в JSON-файл.
    """
    save_json(metrics, path)


def load_predictions(path: str | Path) -> pd.DataFrame:
    """
    Загружает CSV-файл с предсказаниями модели.
    """
    return load_csv(path)


def save_predictions(predictions: pd.DataFrame, path: str | Path) -> None:
    """
    Сохраняет таблицу предсказаний модели в CSV-файл.
    """
    save_csv(predictions, path, index=False)


def resolve_predictions_csv_path(artifact_dir: str | Path) -> Path:
    """
    Возвращает стандартный путь к CSV-файлу с предсказаниями
    внутри директории артефактов эксперимента.
    """
    return Path(artifact_dir) / "predictions_test.csv"


def resolve_metrics_json_path(artifact_dir: str | Path) -> Path:
    """
    Возвращает стандартный путь к JSON-файлу с метриками
    внутри директории артефактов эксперимента.
    """
    return Path(artifact_dir) / "metrics.json"


def make_artifact_dir(base_dir: str | Path, name: str) -> Path:
    """
    Создаёт директорию для артефактов конкретного эксперимента.

    Например:
    artifacts/rc_ladder/strict
    artifacts/rc_ladder/fewshot_4pct
    """
    artifact_dir = Path(base_dir) / name
    ensure_dir(artifact_dir)
    return artifact_dir