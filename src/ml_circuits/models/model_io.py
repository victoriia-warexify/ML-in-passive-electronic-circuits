from __future__ import annotations

import os

import joblib


def save_models(models: dict, models_dir: str):
    """
    Сохраняет набор обученных моделей в формате joblib.
    """
    os.makedirs(models_dir, exist_ok=True)

    for topo, model in models.items():
        path = os.path.join(models_dir, f"{topo}_model.joblib")
        joblib.dump(model, path)
        print(f"Saved: {path}")


def load_models(topologies: list[str], models_dir: str) -> dict:
    """
    Загружает набор моделей из каталога models_dir.
    """
    loaded = {}

    for topo in topologies:
        path = os.path.join(models_dir, f"{topo}_model.joblib")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        loaded[topo] = joblib.load(path)
        print(f"Loaded: {path}")

    return loaded
