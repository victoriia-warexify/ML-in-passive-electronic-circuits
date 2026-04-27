# src/ml_circuits/inverse/rc_ladder_inverse.py

from __future__ import annotations

import math
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from ml_circuits.constants import EPS, TWO_PI
from ml_circuits.features.rc_ladder_features import load_jsonl, parse_sections_cell


def _to_device_const(
    x: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.tensor(float(x), dtype=dtype, device=device)


def _zscore_torch(x: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    std_safe = max(float(std), 1e-6)
    return (x - float(mean)) / std_safe


def _wrap_angle_torch(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


def _wrap_angle_deg_np(x_deg: np.ndarray) -> np.ndarray:
    x_rad = np.radians(x_deg)
    return np.degrees(np.arctan2(np.sin(x_rad), np.cos(x_rad)))


def _check_required_stats_keys(stats: Dict[str, float]) -> None:
    required_keys = [
        "logR_mean", "logR_std",
        "logC_mean", "logC_std",
        "logTau_mean", "logTau_std",
        "logWTau_mean", "logWTau_std",
        "logf_mean", "logf_std",
        "logRload_mean", "logRload_std",
        "mean_logR_mean", "mean_logR_std",
        "mean_logC_mean", "mean_logC_std",
        "mean_logTau_mean", "mean_logTau_std",
        "std_logR_mean", "std_logR_std",
        "std_logC_mean", "std_logC_std",
        "min_logTau_mean", "min_logTau_std",
        "max_logTau_mean", "max_logTau_std",
        "mean_logWTau_mean", "mean_logWTau_std",
        "std_logWTau_mean", "std_logWTau_std",
        "min_logWTau_mean", "min_logWTau_std",
        "max_logWTau_mean", "max_logWTau_std",
    ]
    missing = [k for k in required_keys if k not in stats]
    if missing:
        raise KeyError(f"Missing required stats keys: {missing}")


def _model_predict_outputs(model, seq_feat, seq_len, global_feat):
    """
    Унифицирует выход модели.

    Поддерживаемые варианты:
    1) dict с ключами: H_mag, phi, logH
    2) tensor [B, 3] = [logH, sin(phi), cos(phi)]
    """
    pred = model(seq_feat, seq_len, global_feat)

    if isinstance(pred, dict):
        if not all(k in pred for k in ["H_mag", "phi", "logH"]):
            raise KeyError("Model dict output must contain keys: H_mag, phi, logH")
        return pred["H_mag"], pred["phi"], pred["logH"]

    if torch.is_tensor(pred):
        if pred.ndim != 2 or pred.shape[1] < 3:
            raise ValueError(
                f"Tensor model output must have shape [batch, >=3], got {tuple(pred.shape)}"
            )
        pred_logH = pred[:, 0]
        pred_sin = pred[:, 1]
        pred_cos = pred[:, 2]
        pred_phi = torch.atan2(pred_sin, pred_cos)
        pred_H = torch.exp(pred_logH)
        return pred_H, pred_phi, pred_logH

    raise TypeError(f"Unsupported model output type: {type(pred)}")


def _build_features_from_log_params_single_freq(
    logR: torch.Tensor,
    logC: torch.Tensor,
    n_sections: int,
    frequency_hz: float,
    Rload: float,
    stats: Dict[str, float],
    max_sections: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if logR.ndim != 1 or logC.ndim != 1:
        raise ValueError("logR and logC must be 1D tensors.")
    if len(logR) != n_sections or len(logC) != n_sections:
        raise ValueError("Length of logR/logC must match n_sections.")
    if not (1 <= n_sections <= max_sections):
        raise ValueError(f"n_sections must be in [1, {max_sections}]")

    _check_required_stats_keys(stats)

    freq_t = _to_device_const(frequency_hz, device)
    logf = torch.log(torch.clamp(freq_t, min=EPS))

    Rload_t = _to_device_const(Rload, device)
    logRload = torch.log(torch.clamp(Rload_t, min=EPS))

    logTau = logR + logC
    logWTau = torch.log(_to_device_const(TWO_PI, device)) + logf + logTau

    z_logf = _zscore_torch(logf, stats["logf_mean"], stats["logf_std"])
    z_logRload = _zscore_torch(logRload, stats["logRload_mean"], stats["logRload_std"])

    mean_logR = torch.mean(logR)
    mean_logC = torch.mean(logC)
    mean_logTau = torch.mean(logTau)
    std_logR = torch.std(logR, unbiased=False)
    std_logC = torch.std(logC, unbiased=False)
    min_logTau = torch.min(logTau)
    max_logTau = torch.max(logTau)

    mean_logWTau = torch.mean(logWTau)
    std_logWTau = torch.std(logWTau, unbiased=False)
    min_logWTau = torch.min(logWTau)
    max_logWTau = torch.max(logWTau)

    global_feat = torch.stack(
        [
            z_logf,
            z_logRload,
            _zscore_torch(mean_logR, stats["mean_logR_mean"], stats["mean_logR_std"]),
            _zscore_torch(mean_logC, stats["mean_logC_mean"], stats["mean_logC_std"]),
            _zscore_torch(mean_logTau, stats["mean_logTau_mean"], stats["mean_logTau_std"]),
            _zscore_torch(std_logR, stats["std_logR_mean"], stats["std_logR_std"]),
            _zscore_torch(std_logC, stats["std_logC_mean"], stats["std_logC_std"]),
            _zscore_torch(min_logTau, stats["min_logTau_mean"], stats["min_logTau_std"]),
            _zscore_torch(max_logTau, stats["max_logTau_mean"], stats["max_logTau_std"]),
            _zscore_torch(mean_logWTau, stats["mean_logWTau_mean"], stats["mean_logWTau_std"]),
            _zscore_torch(std_logWTau, stats["std_logWTau_mean"], stats["std_logWTau_std"]),
            _zscore_torch(min_logWTau, stats["min_logWTau_mean"], stats["min_logWTau_std"]),
            _zscore_torch(max_logWTau, stats["max_logWTau_mean"], stats["max_logWTau_std"]),
        ],
        dim=0,
    ).to(torch.float32)

    seq_feat = torch.zeros((max_sections, 7), dtype=torch.float32, device=device)

    for i in range(n_sections):
        seq_feat[i, 0] = _zscore_torch(logR[i], stats["logR_mean"], stats["logR_std"])
        seq_feat[i, 1] = _zscore_torch(logC[i], stats["logC_mean"], stats["logC_std"])
        seq_feat[i, 2] = _zscore_torch(logWTau[i], stats["logWTau_mean"], stats["logWTau_std"])
        seq_feat[i, 3] = 1.0 if i == 0 else 0.0
        seq_feat[i, 4] = 1.0 if i == n_sections - 1 else 0.0
        seq_feat[i, 5] = z_logf
        seq_feat[i, 6] = z_logRload

    return {
        "seq_feat": seq_feat.unsqueeze(0),
        "global_feat": global_feat.unsqueeze(0),
        "seq_len": torch.tensor([n_sections], dtype=torch.long, device=device),
    }


def _build_features_from_log_params_multifreq(
    logR: torch.Tensor,
    logC: torch.Tensor,
    n_sections: int,
    frequency_list_hz: List[float],
    Rload: float,
    stats: Dict[str, float],
    max_sections: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    items = []
    for frequency_hz in frequency_list_hz:
        items.append(
            _build_features_from_log_params_single_freq(
                logR=logR,
                logC=logC,
                n_sections=n_sections,
                frequency_hz=float(frequency_hz),
                Rload=Rload,
                stats=stats,
                max_sections=max_sections,
                device=device,
            )
        )

    return {
        "seq_feat": torch.cat([x["seq_feat"] for x in items], dim=0),
        "global_feat": torch.cat([x["global_feat"] for x in items], dim=0),
        "seq_len": torch.cat([x["seq_len"] for x in items], dim=0),
    }


def _evaluate_model_multifreq_from_log_params(
    model: nn.Module,
    stats: Dict[str, float],
    logR: torch.Tensor,
    logC: torch.Tensor,
    n_sections: int,
    frequency_list_hz: List[float],
    Rload: float,
    max_sections: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    feats = _build_features_from_log_params_multifreq(
        logR=logR,
        logC=logC,
        n_sections=n_sections,
        frequency_list_hz=frequency_list_hz,
        Rload=Rload,
        stats=stats,
        max_sections=max_sections,
        device=device,
    )

    with torch.no_grad():
        pred_H, pred_phi, pred_logH = _model_predict_outputs(
            model=model,
            seq_feat=feats["seq_feat"],
            seq_len=feats["seq_len"],
            global_feat=feats["global_feat"],
        )

    pred_H = pred_H.detach().cpu().numpy()
    pred_phi_rad = pred_phi.detach().cpu().numpy()
    pred_phi_deg = np.degrees(pred_phi_rad)
    pred_logH = pred_logH.detach().cpu().numpy()

    return {
        "pred_H_list": pred_H,
        "pred_phi_rad_list": pred_phi_rad,
        "pred_phi_deg_list": pred_phi_deg,
        "pred_logH_list": pred_logH,
    }


def evaluate_true_rc_through_model(
    model,
    stats,
    row_param,
    df_dataset,
    max_sections: int = 4,
    n_freq: int = 20,
):
    device = next(model.parameters()).device
    model.eval()

    param_set_id = row_param["param_set_id"]
    n_sections = int(row_param["n_sections"])
    Rload = float(row_param["Rload"])
    sections_true = row_param["sections"]

    R_true = np.array([float(sec["R"]) for sec in sections_true], dtype=np.float32)
    C_true = np.array([float(sec["C"]) for sec in sections_true], dtype=np.float32)

    logR_true = torch.tensor(np.log(R_true), dtype=torch.float32, device=device)
    logC_true = torch.tensor(np.log(C_true), dtype=torch.float32, device=device)

    targets = choose_multifreq_targets_from_dataset(
        df_dataset=df_dataset,
        param_set_id=param_set_id,
        n_freq=n_freq,
    )

    frequency_list_hz = targets["frequency_list_hz"]
    target_H_mag_list = np.array(targets["target_H_mag_list"], dtype=np.float32)
    target_phi_rad_list = np.array(targets["target_phi_list"], dtype=np.float32)
    target_phi_deg_list = np.degrees(target_phi_rad_list)

    pred_pack = _evaluate_model_multifreq_from_log_params(
        model=model,
        stats=stats,
        logR=logR_true,
        logC=logC_true,
        n_sections=n_sections,
        frequency_list_hz=frequency_list_hz,
        Rload=Rload,
        max_sections=max_sections,
        device=device,
    )

    pred_H_mag_list = pred_pack["pred_H_list"]
    pred_phi_rad_list = pred_pack["pred_phi_rad_list"]
    pred_phi_deg_list = pred_pack["pred_phi_deg_list"]
    pred_logH_list = pred_pack["pred_logH_list"]

    phi_err_deg = np.degrees(
        np.arctan2(
            np.sin(pred_phi_rad_list - target_phi_rad_list),
            np.cos(pred_phi_rad_list - target_phi_rad_list),
        )
    )

    result_df = pd.DataFrame(
        {
            "frequency_hz": frequency_list_hz,
            "target_H_mag": target_H_mag_list,
            "pred_H_mag_on_true_RC": pred_H_mag_list,
            "abs_err_H": np.abs(pred_H_mag_list - target_H_mag_list),
            "target_phi_rad": target_phi_rad_list,
            "pred_phi_rad_on_true_RC": pred_phi_rad_list,
            "target_phi_deg": target_phi_deg_list,
            "pred_phi_deg_on_true_RC": pred_phi_deg_list,
            "phi_err_deg_wrapped": phi_err_deg,
            "abs_phi_err_deg": np.abs(phi_err_deg),
            "pred_logH_on_true_RC": pred_logH_list,
        }
    )

    summary = {
        "param_set_id": param_set_id,
        "n_sections": n_sections,
        "Rload": Rload,
        "R_true": R_true,
        "C_true": C_true,
        "frequency_list_hz": frequency_list_hz,
        "H_MAE_on_true_RC": float(np.mean(np.abs(pred_H_mag_list - target_H_mag_list))),
        "phi_MAE_deg_on_true_RC": float(np.mean(np.abs(phi_err_deg))),
        "result_df": result_df,
    }

    return summary


def optimize_rc_parameters_multifreq(
    model,
    stats,
    n_sections,
    frequency_list_hz,
    Rload,
    target_H_mag_list,
    target_phi_list,
    phi_unit="deg",
    max_sections=4,
    device="cpu",
    eps=1e-12,
    adam_steps=1500,
    lbfgs_steps=100,
    lr_adam=0.03,
    lr_lbfgs=0.5,
    use_lbfgs=True,
    seed=42,
    print_every=200,
    use_multistart=True,
    noise_scales=(0.05, 0.10, 0.20),
    weight_logH=1.0,
    weight_H=0.30,
    weight_phi=0.10,
    phase_H_threshold=0.05,
    phase_H_power=0.5,
    smoothness_reg=1e-3,
    center_reg=1e-4,
    R_min=1.0,
    R_max=1e7,
    C_min=1e-12,
    C_max=1e-2,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device)

    model.eval()

    frequency_list_hz = np.asarray(frequency_list_hz, dtype=np.float64)
    target_H_t = torch.tensor(target_H_mag_list, dtype=torch.float32, device=device)

    target_phi_t = torch.tensor(target_phi_list, dtype=torch.float32, device=device)
    if phi_unit.lower() in ("deg", "degree", "degrees"):
        target_phi_t = target_phi_t * math.pi / 180.0

    target_logH_t = torch.log(torch.clamp(target_H_t, min=eps))

    def _safe_stat(name, default=0.0):
        return float(stats[name]) if name in stats else float(default)

    prior_logR_mean = _safe_stat("logR_mean", 0.0)
    prior_logR_std = max(_safe_stat("logR_std", 1.0), 1e-8)
    prior_logC_mean = _safe_stat("logC_mean", 0.0)
    prior_logC_std = max(_safe_stat("logC_std", 1.0), 1e-8)

    def _build_freq_weights(freq_list_hz):
        f = np.asarray(freq_list_hz, dtype=np.float64)
        if len(f) == 1:
            return torch.ones(1, dtype=torch.float32, device=device)
        logf = np.log(np.clip(f, 1e-30, None))
        gaps = np.zeros_like(logf)
        gaps[1:-1] = 0.5 * (logf[2:] - logf[:-2])
        gaps[0] = logf[1] - logf[0]
        gaps[-1] = logf[-1] - logf[-2]
        gaps = np.maximum(gaps, 1e-9)
        w = gaps / gaps.sum()
        return torch.tensor(w, dtype=torch.float32, device=device)

    freq_weights_t = _build_freq_weights(frequency_list_hz)

    def _clip_physical(logR, logC):
        logR = torch.clamp(logR, min=math.log(R_min), max=math.log(R_max))
        logC = torch.clamp(logC, min=math.log(C_min), max=math.log(C_max))
        return logR, logC

    def predict_multifreq(logR_param, logC_param):
        logR_current, logC_current = _clip_physical(logR_param, logC_param)

        feats = _build_features_from_log_params_multifreq(
            logR=logR_current,
            logC=logC_current,
            n_sections=n_sections,
            frequency_list_hz=frequency_list_hz.tolist(),
            Rload=Rload,
            stats=stats,
            max_sections=max_sections,
            device=device,
        )

        pred_H_t, pred_phi_t, pred_logH_t = _model_predict_outputs(
            model=model,
            seq_feat=feats["seq_feat"],
            seq_len=feats["seq_len"],
            global_feat=feats["global_feat"],
        )

        return pred_H_t, pred_logH_t, pred_phi_t, logR_current, logC_current

    def compute_inverse_loss(pred_H_t, pred_logH_t, pred_phi_t, logR_param, logC_param):
        loss_logH_per = nn.functional.smooth_l1_loss(
            pred_logH_t, target_logH_t, reduction="none"
        )
        loss_H_per = nn.functional.smooth_l1_loss(
            pred_H_t, target_H_t, reduction="none"
        )

        dphi = _wrap_angle_torch(pred_phi_t - target_phi_t)

        phase_point_weight = torch.where(
            target_H_t >= phase_H_threshold,
            torch.clamp(target_H_t, min=phase_H_threshold) ** phase_H_power,
            torch.zeros_like(target_H_t),
        )
        loss_phi_per = phase_point_weight * (dphi ** 2)

        loss_data = (
            weight_logH * torch.sum(freq_weights_t * loss_logH_per)
            + weight_H * torch.sum(freq_weights_t * loss_H_per)
            + weight_phi * torch.sum(freq_weights_t * loss_phi_per)
        )

        reg = torch.tensor(0.0, dtype=torch.float32, device=device)

        if n_sections > 1:
            reg = reg + smoothness_reg * torch.mean((logR_param[1:] - logR_param[:-1]) ** 2)
            reg = reg + smoothness_reg * torch.mean((logC_param[1:] - logC_param[:-1]) ** 2)

        reg = reg + center_reg * torch.mean(((logR_param - prior_logR_mean) / prior_logR_std) ** 2)
        reg = reg + center_reg * torch.mean(((logC_param - prior_logC_mean) / prior_logC_std) ** 2)

        return loss_data + reg

    def build_initializations():
        starts = []

        base_logR = torch.full((n_sections,), prior_logR_mean, dtype=torch.float32, device=device)
        base_logC = torch.full((n_sections,), prior_logC_mean, dtype=torch.float32, device=device)

        trend = torch.linspace(-1.0, 1.0, n_sections, device=device)

        starts.append(("stats_mean", base_logR.clone(), base_logC.clone()))
        starts.append(("R_up", base_logR + 0.5 * prior_logR_std * trend, base_logC.clone()))
        starts.append(("R_down", base_logR - 0.5 * prior_logR_std * trend, base_logC.clone()))
        starts.append(("C_up", base_logR.clone(), base_logC + 0.5 * prior_logC_std * trend))
        starts.append(("C_down", base_logR.clone(), base_logC - 0.5 * prior_logC_std * trend))

        if use_multistart:
            g = torch.Generator(device=device)
            g.manual_seed(seed)
            idx = 0
            for ns in noise_scales:
                for _ in range(3):
                    init_logR = base_logR + torch.randn(n_sections, generator=g, device=device) * (ns * prior_logR_std)
                    init_logC = base_logC + torch.randn(n_sections, generator=g, device=device) * (ns * prior_logC_std)
                    starts.append((f"noise_{idx}", init_logR, init_logC))
                    idx += 1

        return starts

    starts = build_initializations()

    best_global = None
    all_restart_logs = []

    for restart_id, (restart_tag, init_logR, init_logC) in enumerate(starts, start=1):
        logR_param = nn.Parameter(init_logR.clone())
        logC_param = nn.Parameter(init_logC.clone())

        adam = optim.Adam([logR_param, logC_param], lr=lr_adam)

        best_local = {
            "loss": float("inf"),
            "step": None,
            "logR": None,
            "logC": None,
            "pred_H": None,
            "pred_phi": None,
            "pred_logH": None,
        }

        for step in range(1, adam_steps + 1):
            adam.zero_grad()

            pred_H_t, pred_logH_t, pred_phi_t, logR_current, logC_current = predict_multifreq(
                logR_param, logC_param
            )

            loss = compute_inverse_loss(
                pred_H_t=pred_H_t,
                pred_logH_t=pred_logH_t,
                pred_phi_t=pred_phi_t,
                logR_param=logR_param,
                logC_param=logC_param,
            )
            loss.backward()
            adam.step()

            with torch.no_grad():
                curr_loss = float(loss.item())

                if curr_loss < best_local["loss"]:
                    best_local["loss"] = curr_loss
                    best_local["step"] = step
                    best_local["logR"] = logR_current.detach().clone()
                    best_local["logC"] = logC_current.detach().clone()
                    best_local["pred_H"] = pred_H_t.detach().clone()
                    best_local["pred_phi"] = pred_phi_t.detach().clone()
                    best_local["pred_logH"] = pred_logH_t.detach().clone()

                if print_every and (step == 1 or step % print_every == 0 or step == adam_steps):
                    print(f"[restart {restart_id}/{len(starts)} | {restart_tag}] step {step:04d} | loss={curr_loss:.6f}")
                    for f_hz, hp, pp, ht, pt in zip(
                        frequency_list_hz,
                        pred_H_t.detach().cpu().numpy(),
                        pred_phi_t.detach().cpu().numpy() * 180.0 / math.pi,
                        target_H_t.detach().cpu().numpy(),
                        target_phi_t.detach().cpu().numpy() * 180.0 / math.pi,
                    ):
                        print(
                            f"    f={f_hz:.6g} Hz | "
                            f"H_pred={hp:.6f} | phi_pred_deg={pp:.3f} | "
                            f"H_target={ht:.6f} | phi_target_deg={pt:.3f}"
                        )

        if use_lbfgs:
            logR_param = nn.Parameter(best_local["logR"].clone())
            logC_param = nn.Parameter(best_local["logC"].clone())

            lbfgs = optim.LBFGS(
                [logR_param, logC_param],
                lr=lr_lbfgs,
                max_iter=lbfgs_steps,
                history_size=50,
                line_search_fn="strong_wolfe",
            )

            def closure():
                lbfgs.zero_grad()
                pred_H_t, pred_logH_t, pred_phi_t, _, _ = predict_multifreq(logR_param, logC_param)
                loss = compute_inverse_loss(
                    pred_H_t=pred_H_t,
                    pred_logH_t=pred_logH_t,
                    pred_phi_t=pred_phi_t,
                    logR_param=logR_param,
                    logC_param=logC_param,
                )
                loss.backward()
                return loss

            lbfgs.step(closure)

            with torch.no_grad():
                pred_H_t, pred_logH_t, pred_phi_t, logR_current, logC_current = predict_multifreq(
                    logR_param, logC_param
                )
                loss = compute_inverse_loss(
                    pred_H_t=pred_H_t,
                    pred_logH_t=pred_logH_t,
                    pred_phi_t=pred_phi_t,
                    logR_param=logR_param,
                    logC_param=logC_param,
                )
                curr_loss = float(loss.item())

                if curr_loss < best_local["loss"]:
                    best_local["loss"] = curr_loss
                    best_local["step"] = adam_steps + lbfgs_steps
                    best_local["logR"] = logR_current.detach().clone()
                    best_local["logC"] = logC_current.detach().clone()
                    best_local["pred_H"] = pred_H_t.detach().clone()
                    best_local["pred_phi"] = pred_phi_t.detach().clone()
                    best_local["pred_logH"] = pred_logH_t.detach().clone()

        all_restart_logs.append(
            {
                "restart_id": restart_id,
                "restart_tag": restart_tag,
                "best_loss": best_local["loss"],
                "best_step": best_local["step"],
            }
        )

        if (best_global is None) or (best_local["loss"] < best_global["best_loss"]):
            best_global = {
                "best_restart_id": restart_id,
                "best_restart_tag": restart_tag,
                "best_loss": best_local["loss"],
                "best_step": best_local["step"],
                "logR": best_local["logR"].clone(),
                "logC": best_local["logC"].clone(),
                "pred_H": best_local["pred_H"].clone(),
                "pred_phi": best_local["pred_phi"].clone(),
                "pred_logH": best_local["pred_logH"].clone(),
            }

    with torch.no_grad():
        pred_phi_deg = best_global["pred_phi"].cpu().numpy() * 180.0 / math.pi
        target_phi_deg = target_phi_t.cpu().numpy() * 180.0 / math.pi

        comparison_df = pd.DataFrame(
            {
                "frequency_hz": frequency_list_hz.copy(),
                "target_H_mag": target_H_t.cpu().numpy(),
                "pred_H_inverse": best_global["pred_H"].cpu().numpy(),
                "target_phi_deg": target_phi_deg,
                "pred_phi_deg_inverse": pred_phi_deg,
            }
        )

        result = {
            "best_restart_id": best_global["best_restart_id"],
            "best_restart_tag": best_global["best_restart_tag"],
            "best_step": best_global["best_step"],
            "best_loss": float(best_global["best_loss"]),
            "R_opt": torch.exp(best_global["logR"]).cpu().numpy(),
            "C_opt": torch.exp(best_global["logC"]).cpu().numpy(),
            "logR_opt": best_global["logR"].cpu().numpy(),
            "logC_opt": best_global["logC"].cpu().numpy(),
            "pred_H_list": best_global["pred_H"].cpu().numpy(),
            "pred_logH_list": best_global["pred_logH"].cpu().numpy(),
            "pred_phi_deg_list": pred_phi_deg,
            "target_H_list": target_H_t.cpu().numpy(),
            "target_phi_deg_list": target_phi_deg,
            "frequency_list_hz": frequency_list_hz.copy(),
            "restart_summary": all_restart_logs,
            "comparison_df": comparison_df,
        }

    print(f"best_restart_id: {result['best_restart_id']}")
    print(f"best_restart_tag: {result['best_restart_tag']}")
    print(f"best_step: {result['best_step']}")
    print(f"best_loss: {result['best_loss']}")
    print(f"R_opt: {result['R_opt']}")
    print(f"C_opt: {result['C_opt']}")
    print(f"pred_H_list: {result['pred_H_list']}")
    print(f"pred_phi_deg_list: {result['pred_phi_deg_list']}")

    return result


def build_param_table_from_main_dataset(df_main: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["param_set_id", "n_sections", "sections", "Rload", "A_in"]
    missing = [c for c in required_cols if c not in df_main.columns]
    if missing:
        raise KeyError(f"Missing columns in main dataset: {missing}")

    df_params = (
        df_main[required_cols]
        .drop_duplicates(subset=["param_set_id"])
        .copy()
        .reset_index(drop=True)
    )
    df_params["sections"] = df_params["sections"].apply(parse_sections_cell)
    return df_params


def choose_multifreq_targets_from_dataset(
    df_dataset: pd.DataFrame,
    param_set_id: str,
    n_freq: int = 6,
) -> Dict[str, List[float]]:
    sub = (
        df_dataset[df_dataset["param_set_id"] == param_set_id]
        .sort_values("frequency_hz")
        .reset_index(drop=True)
    )

    if len(sub) == 0:
        raise ValueError(f"No rows found for param_set_id={param_set_id}")

    idx = np.linspace(0, len(sub) - 1, num=min(n_freq, len(sub)), dtype=int)
    idx = np.unique(idx)
    sub_pick = sub.iloc[idx].copy().reset_index(drop=True)

    phi_rad_list = sub_pick["phi_rad"].astype(float).tolist()
    phi_deg_list = np.rad2deg(np.array(phi_rad_list, dtype=float)).tolist()

    return {
        "frequency_list_hz": sub_pick["frequency_hz"].astype(float).tolist(),
        "target_H_mag_list": sub_pick["H_mag"].astype(float).tolist(),
        "target_phi_list": phi_rad_list,
        "target_phi_deg_list": phi_deg_list,
    }


def select_examples_for_inverse_eval(
    df_eval123_params: pd.DataFrame,
    df_main_params: pd.DataFrame,
    n_examples_per_group: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    selected_parts = []

    for n_sections in [1, 2, 3]:
        sub = df_eval123_params[df_eval123_params["n_sections"] == n_sections].copy()
        if len(sub) == 0:
            continue
        if len(sub) > n_examples_per_group:
            sub = sub.sample(n=n_examples_per_group, random_state=random_state)
        sub = sub.copy()
        sub["source_name"] = "eval123"
        selected_parts.append(sub)

    sub4 = df_main_params[df_main_params["n_sections"] == 4].copy()
    if len(sub4) == 0:
        raise RuntimeError("No n=4 examples found in main dataset.")
    if len(sub4) > n_examples_per_group:
        sub4 = sub4.sample(n=n_examples_per_group, random_state=random_state)
    sub4 = sub4.copy()
    sub4["source_name"] = "main_n4"
    selected_parts.append(sub4)

    selected_df = pd.concat(selected_parts, axis=0, ignore_index=True)
    return selected_df.sort_values(["n_sections", "param_set_id"]).reset_index(drop=True)


def run_full_inverse_evaluation(
    model,
    stats: Dict[str, float],
    df_eval123_params: pd.DataFrame,
    df_eval123_dataset: pd.DataFrame,
    df_main_params: pd.DataFrame,
    df_main_dataset: pd.DataFrame,
    out_dir: str = "artifacts_inverse_rc_ladder",
    n_examples_per_group: int = 3,
    n_freq_per_example: int = 6,
    max_sections: int = 4,
    adam_steps: int = 1500,
    lbfgs_steps: int = 100,
    lr_adam: float = 0.03,
    lr_lbfgs: float = 0.5,
    use_lbfgs: bool = True,
    use_multistart: bool = True,
    noise_scales=(0.05, 0.10, 0.20),
    print_every: int = 200,
    seed: int = 42,
):
    os.makedirs(out_dir, exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    selected_examples = select_examples_for_inverse_eval(
        df_eval123_params=df_eval123_params,
        df_main_params=df_main_params,
        n_examples_per_group=n_examples_per_group,
        random_state=seed,
    )

    summary_rows = []
    freq_rows = []
    detailed_results = []

    for example_idx, row in enumerate(selected_examples.to_dict(orient="records")):
        param_set_id = row["param_set_id"]
        n_sections = int(row["n_sections"])
        sections_true = row["sections"]
        Rload_true = float(row["Rload"])
        source_name = row["source_name"]

        df_source = df_eval123_dataset if source_name == "eval123" else df_main_dataset

        targets = choose_multifreq_targets_from_dataset(
            df_dataset=df_source,
            param_set_id=param_set_id,
            n_freq=n_freq_per_example,
        )

        frequency_list_hz = targets["frequency_list_hz"]
        target_H_mag_list = targets["target_H_mag_list"]
        target_phi_list = targets["target_phi_list"]
        target_phi_deg_list = targets["target_phi_deg_list"]

        true_R = [float(sec["R"]) for sec in sections_true]
        true_C = [float(sec["C"]) for sec in sections_true]

        print("=" * 100)
        print(f"param_set_id={param_set_id} | source={source_name} | n_sections={n_sections}")

        inverse_result = optimize_rc_parameters_multifreq(
            model=model,
            stats=stats,
            n_sections=n_sections,
            frequency_list_hz=frequency_list_hz,
            Rload=Rload_true,
            target_H_mag_list=target_H_mag_list,
            target_phi_list=target_phi_list,
            phi_unit="rad",
            max_sections=max_sections,
            adam_steps=adam_steps,
            lbfgs_steps=lbfgs_steps,
            lr_adam=lr_adam,
            lr_lbfgs=lr_lbfgs,
            use_lbfgs=use_lbfgs,
            seed=seed + example_idx,
            print_every=print_every,
            use_multistart=use_multistart,
            noise_scales=noise_scales,
        )

        true_rc_check = evaluate_true_rc_through_model(
            model=model,
            stats=stats,
            row_param=row,
            df_dataset=df_source,
            max_sections=max_sections,
            n_freq=n_freq_per_example,
        )

        R_true = np.array(true_R, dtype=float)
        C_true = np.array(true_C, dtype=float)
        R_opt = np.array(inverse_result["R_opt"], dtype=float)
        C_opt = np.array(inverse_result["C_opt"], dtype=float)

        tau_true = R_true * C_true
        tau_opt = R_opt * C_opt

        logR_mae = float(np.mean(np.abs(np.log(R_opt) - np.log(R_true))))
        logC_mae = float(np.mean(np.abs(np.log(C_opt) - np.log(C_true))))
        logTau_mae = float(np.mean(np.abs(np.log(tau_opt) - np.log(tau_true))))

        R_rel_mae = float(np.mean(np.abs(R_opt - R_true) / np.maximum(R_true, 1e-30)))
        C_rel_mae = float(np.mean(np.abs(C_opt - C_true) / np.maximum(C_true, 1e-30)))
        tau_rel_mae = float(np.mean(np.abs(tau_opt - tau_true) / np.maximum(tau_true, 1e-30)))

        pred_H = np.array(inverse_result["pred_H_list"], dtype=float)
        pred_phi_deg = np.array(inverse_result["pred_phi_deg_list"], dtype=float)

        target_H = np.array(target_H_mag_list, dtype=float)
        target_phi_deg = np.array(target_phi_deg_list, dtype=float)

        H_mae_inverse = float(np.mean(np.abs(pred_H - target_H)))
        logH_mae_inverse = float(
            np.mean(
                np.abs(
                    np.log(np.maximum(pred_H, 1e-30)) -
                    np.log(np.maximum(target_H, 1e-30))
                )
            )
        )

        phi_err_inverse = _wrap_angle_deg_np(pred_phi_deg - target_phi_deg)
        phi_mae_deg_inverse = float(np.mean(np.abs(phi_err_inverse)))

        H_mae_trueRC = float(true_rc_check["H_MAE_on_true_RC"])
        phi_mae_deg_trueRC = float(true_rc_check["phi_MAE_deg_on_true_RC"])

        summary_rows.append({
            "param_set_id": param_set_id,
            "source_name": source_name,
            "n_sections": n_sections,
            "best_restart_tag": inverse_result["best_restart_tag"],
            "best_restart_id": inverse_result["best_restart_id"],
            "best_step": inverse_result["best_step"],
            "best_loss": inverse_result["best_loss"],
            "logR_mae": logR_mae,
            "logC_mae": logC_mae,
            "logTau_mae": logTau_mae,
            "R_rel_mae": R_rel_mae,
            "C_rel_mae": C_rel_mae,
            "tau_rel_mae": tau_rel_mae,
            "H_MAE_trueRC": H_mae_trueRC,
            "phi_MAE_deg_trueRC": phi_mae_deg_trueRC,
            "H_MAE_inverse": H_mae_inverse,
            "logH_MAE_inverse": logH_mae_inverse,
            "phi_MAE_deg_inverse": phi_mae_deg_inverse,
            "R_true": true_R,
            "C_true": true_C,
            "R_opt": R_opt.tolist(),
            "C_opt": C_opt.tolist(),
        })

        for k, f_hz in enumerate(frequency_list_hz):
            freq_rows.append({
                "param_set_id": param_set_id,
                "source_name": source_name,
                "n_sections": n_sections,
                "frequency_hz": float(f_hz),
                "target_H": float(target_H[k]),
                "pred_H_inverse": float(pred_H[k]),
                "target_phi_deg": float(target_phi_deg[k]),
                "pred_phi_deg_inverse": float(pred_phi_deg[k]),
                "abs_err_H_inverse": float(abs(pred_H[k] - target_H[k])),
                "abs_err_phi_deg_inverse": float(abs(phi_err_inverse[k])),
            })

        comparison_df = inverse_result["comparison_df"].copy()
        true_df = true_rc_check["result_df"].copy()

        comparison_df = comparison_df.merge(
            true_df[[
                "frequency_hz",
                "pred_H_mag_on_true_RC",
                "pred_phi_deg_on_true_RC",
            ]],
            on="frequency_hz",
            how="left",
        )
        comparison_df = comparison_df.rename(columns={
            "pred_H_mag_on_true_RC": "pred_H_trueRC",
            "pred_phi_deg_on_true_RC": "pred_phi_deg_trueRC",
        })

        detailed_results.append({
            "param_set_id": param_set_id,
            "source_name": source_name,
            "n_sections": n_sections,
            "sections_true": sections_true,
            "R_true": R_true,
            "C_true": C_true,
            "R_opt": R_opt,
            "C_opt": C_opt,
            "tau_true": tau_true,
            "tau_opt": tau_opt,
            "frequency_list_hz": np.array(frequency_list_hz, dtype=float),
            "target_H": target_H,
            "pred_H": pred_H,
            "target_phi_deg": target_phi_deg,
            "pred_phi_deg": pred_phi_deg,
            "true_rc_check": true_rc_check,
            "inverse_result": inverse_result,
            "comparison_df": comparison_df,
        })

    summary_df = pd.DataFrame(summary_rows)
    freq_df = pd.DataFrame(freq_rows)

    summary_path = os.path.join(out_dir, "inverse_summary_examples.csv")
    freq_path = os.path.join(out_dir, "inverse_frequency_points.csv")

    summary_df.to_csv(summary_path, index=False)
    freq_df.to_csv(freq_path, index=False)

    print(f"Saved summary table to: {summary_path}")
    print(f"Saved frequency table to: {freq_path}")

    return {
        "summary_df": summary_df,
        "freq_df": freq_df,
        "summary_path": summary_path,
        "freq_path": freq_path,
        "plots_dir": plots_dir,
        "detailed_results": detailed_results,
        "selected_examples": selected_examples,
    }


def build_inverse_results_table(
    inverse_eval: Dict[str, object],
    save_path: str = None,
) -> pd.DataFrame:
    """
    Строит итоговую таблицу по результатам inverse evaluation.

    Параметры
    ----------
    inverse_eval : dict
        Результат, который возвращает run_full_inverse_evaluation(...).
    save_path : str, optional
        Путь для сохранения csv. Если None, файл не сохраняется.

    Возвращает
    ----------
    pd.DataFrame
        Итоговая таблица с одной строкой на один пример.
    """
    rows = []

    for item in inverse_eval["detailed_results"]:
        param_set_id = item["param_set_id"]
        source_name = item["source_name"]
        n_sections = int(item["n_sections"])

        R_true = np.asarray(item["R_true"], dtype=float)
        C_true = np.asarray(item["C_true"], dtype=float)
        R_opt = np.asarray(item["R_opt"], dtype=float)
        C_opt = np.asarray(item["C_opt"], dtype=float)

        tau_true = np.asarray(item["tau_true"], dtype=float)
        tau_opt = np.asarray(item["tau_opt"], dtype=float)

        target_H = np.asarray(item["target_H"], dtype=float)
        pred_H = np.asarray(item["pred_H"], dtype=float)

        target_phi_deg = np.asarray(item["target_phi_deg"], dtype=float)
        pred_phi_deg = np.asarray(item["pred_phi_deg"], dtype=float)

        phi_err_deg = _wrap_angle_deg_np(pred_phi_deg - target_phi_deg)

        logR_abs_err = np.abs(np.log(np.maximum(R_opt, EPS)) - np.log(np.maximum(R_true, EPS)))
        logC_abs_err = np.abs(np.log(np.maximum(C_opt, EPS)) - np.log(np.maximum(C_true, EPS)))
        logTau_abs_err = np.abs(np.log(np.maximum(tau_opt, EPS)) - np.log(np.maximum(tau_true, EPS)))

        R_rel_err = np.abs(R_opt - R_true) / np.maximum(R_true, EPS)
        C_rel_err = np.abs(C_opt - C_true) / np.maximum(C_true, EPS)
        tau_rel_err = np.abs(tau_opt - tau_true) / np.maximum(tau_true, EPS)

        H_abs_err = np.abs(pred_H - target_H)
        logH_abs_err = np.abs(
            np.log(np.maximum(pred_H, EPS)) - np.log(np.maximum(target_H, EPS))
        )
        phi_abs_err_deg = np.abs(phi_err_deg)

        inverse_result = item["inverse_result"]
        true_rc_check = item["true_rc_check"]

        row = {
            "param_set_id": param_set_id,
            "source_name": source_name,
            "n_sections": n_sections,

            "best_restart_id": inverse_result.get("best_restart_id"),
            "best_restart_tag": inverse_result.get("best_restart_tag"),
            "best_step": inverse_result.get("best_step"),
            "best_loss": inverse_result.get("best_loss"),

            "logR_MAE": float(np.mean(logR_abs_err)),
            "logC_MAE": float(np.mean(logC_abs_err)),
            "logTau_MAE": float(np.mean(logTau_abs_err)),

            "R_rel_MAE": float(np.mean(R_rel_err)),
            "C_rel_MAE": float(np.mean(C_rel_err)),
            "tau_rel_MAE": float(np.mean(tau_rel_err)),

            "R_rel_max": float(np.max(R_rel_err)),
            "C_rel_max": float(np.max(C_rel_err)),
            "tau_rel_max": float(np.max(tau_rel_err)),

            "H_MAE_inverse": float(np.mean(H_abs_err)),
            "H_maxAE_inverse": float(np.max(H_abs_err)),
            "logH_MAE_inverse": float(np.mean(logH_abs_err)),
            "phi_MAE_deg_inverse": float(np.mean(phi_abs_err_deg)),
            "phi_maxAE_deg_inverse": float(np.max(phi_abs_err_deg)),

            "H_MAE_trueRC": float(true_rc_check["H_MAE_on_true_RC"]),
            "phi_MAE_deg_trueRC": float(true_rc_check["phi_MAE_deg_on_true_RC"]),

            "R_true": R_true.tolist(),
            "C_true": C_true.tolist(),
            "R_opt": R_opt.tolist(),
            "C_opt": C_opt.tolist(),
            "tau_true": tau_true.tolist(),
            "tau_opt": tau_opt.tolist(),
        }

        rows.append(row)

    result_df = pd.DataFrame(rows)

    if save_path is not None:
        result_df.to_csv(save_path, index=False)
        print(f"Saved inverse results table to: {save_path}")

    return result_df
