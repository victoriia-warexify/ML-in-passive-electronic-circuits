from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml_circuits.constants import EPS, PHASE_FILTER_MAG
from ml_circuits.features.rc_ladder_features import wrap_angle_deg_np, wrap_angle_rad_np, wrap_angle_torch


def compute_loss(
    pred: Dict[str, torch.Tensor],
    y_true: torch.Tensor,
    phase_weight_power: float = 0.7,
    phase_loss_scale: float = 0.22,
    angular_loss_scale: float = 0.10,
) -> torch.Tensor:
    """
    Вычисляет многокомпонентную функцию потерь для обучения модели.
    """
    logH_per_section_true = y_true[:, 0]
    sin_true = y_true[:, 1]
    cos_true = y_true[:, 2]
    H_mag_true = y_true[:, 3]

    loss_logH = nn.functional.smooth_l1_loss(pred["logH_per_section"], logH_per_section_true)

    phase_weight = torch.where(
        H_mag_true >= PHASE_FILTER_MAG,
        torch.clamp(H_mag_true, min=PHASE_FILTER_MAG, max=1.0) ** phase_weight_power,
        torch.zeros_like(H_mag_true),
    )

    loss_sin = nn.functional.smooth_l1_loss(pred["sin_phi"], sin_true, reduction="none")
    loss_cos = nn.functional.smooth_l1_loss(pred["cos_phi"], cos_true, reduction="none")
    loss_sin = (phase_weight * loss_sin).mean()
    loss_cos = (phase_weight * loss_cos).mean()

    phi_true = torch.atan2(sin_true, cos_true)
    phi_pred = pred["phi"]
    dphi = wrap_angle_torch(phi_pred - phi_true)

    loss_phi = nn.functional.smooth_l1_loss(
        torch.abs(dphi),
        torch.zeros_like(dphi),
        reduction="none",
    )
    loss_phi = (phase_weight * loss_phi).mean()

    return loss_logH + phase_loss_scale * (loss_sin + loss_cos) + angular_loss_scale * loss_phi


def evaluate_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """
    Вычисляет итоговые метрики качества модели на заданном наборе данных.
    """
    model.eval()

    all_logH_true = []
    all_sin_true = []
    all_cos_true = []
    all_H_true = []
    all_A_true = []
    all_phi_true = []
    all_Ain_true = []

    all_logH_pred = []
    all_sin_pred = []
    all_cos_pred = []
    all_H_pred = []
    all_phi_pred = []

    with torch.no_grad():
        for batch in loader:
            seq_feat = batch["seq_feat"].to(device)
            seq_len = batch["seq_len"].to(device)
            global_feat = batch["global_feat"].to(device)
            y = batch["y"].to(device)
            y_aux = batch["y_aux"].to(device)
            a_in = batch["a_in"].to(device)

            pred = model(seq_feat=seq_feat, seq_len=seq_len, global_feat=global_feat)

            all_logH_true.append(y_aux[:, 3].cpu().numpy())
            all_sin_true.append(y[:, 1].cpu().numpy())
            all_cos_true.append(y[:, 2].cpu().numpy())
            all_H_true.append(y_aux[:, 0].cpu().numpy())
            all_A_true.append(y_aux[:, 1].cpu().numpy())
            all_phi_true.append(y_aux[:, 2].cpu().numpy())
            all_Ain_true.append(a_in.cpu().numpy())

            all_logH_pred.append(pred["logH"].cpu().numpy())
            all_sin_pred.append(pred["sin_phi"].cpu().numpy())
            all_cos_pred.append(pred["cos_phi"].cpu().numpy())
            all_H_pred.append(pred["H_mag"].cpu().numpy())
            all_phi_pred.append(pred["phi"].cpu().numpy())

    logH_true = np.concatenate(all_logH_true)
    sin_true = np.concatenate(all_sin_true)
    cos_true = np.concatenate(all_cos_true)
    H_true = np.concatenate(all_H_true)
    A_true = np.concatenate(all_A_true)
    phi_true = np.concatenate(all_phi_true)
    Ain_true = np.concatenate(all_Ain_true)

    logH_pred = np.concatenate(all_logH_pred)
    sin_pred = np.concatenate(all_sin_pred)
    cos_pred = np.concatenate(all_cos_pred)
    H_pred = np.concatenate(all_H_pred)
    phi_pred = np.concatenate(all_phi_pred)

    dB_true = 20.0 * np.log10(np.maximum(H_true, EPS))
    dB_pred = 20.0 * np.log10(np.maximum(H_pred, EPS))
    A_pred = H_pred * Ain_true

    phase_err = np.abs(wrap_angle_rad_np(phi_pred - phi_true))
    phase_err_deg = np.degrees(phase_err)

    phase_mask = H_true > PHASE_FILTER_MAG

    if np.any(phase_mask):
        phase_mae_filtered_deg = float(np.mean(phase_err_deg[phase_mask]))
    else:
        phase_mae_filtered_deg = float("nan")

    return {
        "n": int(len(logH_true)),
        "logH_MAE": float(np.mean(np.abs(logH_pred - logH_true))),
        "logH_RMSE": float(np.sqrt(np.mean((logH_pred - logH_true) ** 2))),
        "dB_MAE": float(np.mean(np.abs(dB_pred - dB_true))),
        "dB_RMSE": float(np.sqrt(np.mean((dB_pred - dB_true) ** 2))),
        "A_MAE": float(np.mean(np.abs(A_pred - A_true))),
        "A_RMSE": float(np.sqrt(np.mean((A_pred - A_true) ** 2))),
        "phi_MAE_deg": float(np.mean(phase_err_deg)),
        "phi_MAE_filtered_deg": phase_mae_filtered_deg,
        "sin_MAE": float(np.mean(np.abs(sin_pred - sin_true))),
        "cos_MAE": float(np.mean(np.abs(cos_pred - cos_true))),
    }


def format_metrics(name: str, metrics: Dict[str, float]) -> None:
    """
    Печатает набор метрик в удобном для чтения виде.
    """
    print(f"[{name}]")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print()


def predict_loader_to_df(model: nn.Module, loader: DataLoader, device: torch.device) -> pd.DataFrame:
    """
    Формирует таблицу с истинными и предсказанными величинами.
    """
    model.eval()
    rows = []

    with torch.no_grad():
        for batch in loader:
            seq_feat = batch["seq_feat"].to(device)
            seq_len = batch["seq_len"].to(device)
            global_feat = batch["global_feat"].to(device)
            pred = model(seq_feat=seq_feat, seq_len=seq_len, global_feat=global_feat)

            logH_pred = pred["logH"].cpu().numpy()
            logH_pred_per_section = pred["logH_per_section"].cpu().numpy()
            sin_pred = pred["sin_phi"].cpu().numpy()
            cos_pred = pred["cos_phi"].cpu().numpy()
            phi_pred = pred["phi"].cpu().numpy()
            H_pred = pred["H_mag"].cpu().numpy()

            y = batch["y"].cpu().numpy()
            y_aux = batch["y_aux"].cpu().numpy()
            a_in = batch["a_in"].cpu().numpy()
            param_ids = batch["param_set_id"]
            freqs = batch["frequency_hz"]
            nsecs = batch["n_sections"]

            for i in range(len(param_ids)):
                rows.append(
                    {
                        "param_set_id": param_ids[i],
                        "frequency_hz": float(freqs[i]),
                        "n_sections": int(nsecs[i]),
                        "A_in": float(a_in[i]),
                        "logH_true": float(y_aux[i, 3]),
                        "logH_pred": float(logH_pred[i]),
                        "logH_pred_per_section": float(logH_pred_per_section[i]),
                        "sin_true": float(y[i, 1]),
                        "cos_true": float(y[i, 2]),
                        "sin_pred": float(sin_pred[i]),
                        "cos_pred": float(cos_pred[i]),
                        "H_true": float(y_aux[i, 0]),
                        "H_pred": float(H_pred[i]),
                        "A_out_true": float(y_aux[i, 1]),
                        "A_out_pred": float(H_pred[i] * a_in[i]),
                        "phi_true_rad": float(y_aux[i, 2]),
                        "phi_true_deg": float(np.degrees(y_aux[i, 2])),
                        "phi_pred_rad": float(phi_pred[i]),
                        "phi_pred_deg": float(np.degrees(phi_pred[i])),
                        "dB_true": float(20.0 * np.log10(max(y_aux[i, 0], EPS))),
                        "dB_pred": float(20.0 * np.log10(max(H_pred[i], EPS))),
                    }
                )

    return pd.DataFrame(rows)


def evaluate_predictions_df_group_weighted(
    pred_df: pd.DataFrame,
    phase_filter_mag: float = PHASE_FILTER_MAG,
) -> Dict[str, float]:
    """
    Считает метрики с одинаковым весом для каждой схемы param_set_id.
    """
    d = pred_df.copy()

    group_size = d.groupby("param_set_id")["param_set_id"].transform("size")
    w = 1.0 / group_size.to_numpy(dtype=float)

    def wmean_abs(err, mask=None):
        err = np.asarray(err, dtype=float)
        ww = w.copy()

        m = np.isfinite(err) & np.isfinite(ww) & (ww > 0)
        if mask is not None:
            m &= mask

        if not np.any(m):
            return float("nan")

        return float(np.sum(ww[m] * np.abs(err[m])) / np.sum(ww[m]))

    def wrmse(err, mask=None):
        err = np.asarray(err, dtype=float)
        ww = w.copy()

        m = np.isfinite(err) & np.isfinite(ww) & (ww > 0)
        if mask is not None:
            m &= mask

        if not np.any(m):
            return float("nan")

        return float(np.sqrt(np.sum(ww[m] * err[m] ** 2) / np.sum(ww[m])))

    logH_err = d["logH_pred"].to_numpy(float) - d["logH_true"].to_numpy(float)
    dB_err = d["dB_pred"].to_numpy(float) - d["dB_true"].to_numpy(float)
    A_err = d["A_out_pred"].to_numpy(float) - d["A_out_true"].to_numpy(float)

    phi_err_deg = wrap_angle_deg_np(
        d["phi_pred_deg"].to_numpy(float) - d["phi_true_deg"].to_numpy(float)
    )

    H_true = d["H_true"].to_numpy(float)
    phase_mask = np.isfinite(H_true) & (H_true > phase_filter_mag)

    return {
        "n": int(len(d)),
        "n_groups": int(d["param_set_id"].nunique()),
        "logH_MAE_group_weighted": wmean_abs(logH_err),
        "logH_RMSE_group_weighted": wrmse(logH_err),
        "dB_MAE_group_weighted": wmean_abs(dB_err),
        "dB_RMSE_group_weighted": wrmse(dB_err),
        "A_MAE_group_weighted": wmean_abs(A_err),
        "A_RMSE_group_weighted": wrmse(A_err),
        "phi_MAE_deg_group_weighted": wmean_abs(phi_err_deg),
        "phi_MAE_filtered_deg_group_weighted": wmean_abs(
            phi_err_deg,
            mask=phase_mask,
        ),
    }


def combined_score(metrics: Dict[str, float]) -> float:
    """
    Вычисляет эвристический комбинированный score для сравнения запусков.
    """
    phase_term = metrics["phi_MAE_filtered_deg"]
    if np.isnan(phase_term):
        phase_term = metrics["phi_MAE_deg"]
    return 0.7 * metrics["dB_MAE"] + 0.3 * phase_term


def combined_score_group_weighted(metrics: Dict[str, float]) -> float:
    """
    Комбинированный score на основе group-weighted метрик.
    """
    phase_term = metrics["phi_MAE_filtered_deg_group_weighted"]

    if np.isnan(phase_term):
        phase_term = metrics["phi_MAE_deg_group_weighted"]

    return (
        0.7 * metrics["dB_MAE_group_weighted"]
        + 0.3 * phase_term
    )
