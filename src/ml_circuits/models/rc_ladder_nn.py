from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class SectionEncoder(nn.Module):
    """
    Кодировщик признаков отдельной секции RC-цепочки.
    """

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualSectionUpdater(nn.Module):
    """
    Residual-блок обновления скрытого состояния при последовательном обходе секций.
    """

    def __init__(self, sec_dim: int, global_dim: int, state_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(sec_dim + global_dim + state_dim, state_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(state_dim, state_dim),
        )

    def forward(
        self,
        sec_emb: torch.Tensor,
        global_emb: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([sec_emb, global_emb, state], dim=-1)
        delta = self.net(x)
        return state + delta


class RCLadderPureMLModel(nn.Module):
    """
    Нейросетевая black-box-модель для предсказания частотного отклика RC-цепочки.
    """
    def __init__(
        self,
        seq_in_dim: int = 7,
        global_dim: int = 13,
        sec_hidden: int = 96,
        global_hidden: int = 64,
        state_dim: int = 128,
        head_dim: int = 128,
        phase_head_dim: int = 64,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.state_dim = state_dim

        self.section_encoder = SectionEncoder(seq_in_dim, sec_hidden, dropout)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, global_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(global_hidden, global_hidden),
            nn.SiLU(),
        )
        self.updater = ResidualSectionUpdater(sec_hidden, global_hidden, state_dim, dropout)

        self.logH_head = nn.Sequential(
            nn.Linear(state_dim + global_hidden, head_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, head_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )

        self.phase_head = nn.Sequential(
            nn.Linear(state_dim + global_hidden, phase_head_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(phase_head_dim, phase_head_dim),
            nn.SiLU(),
            nn.Dropout(dropout * 0.25),
            nn.Linear(phase_head_dim, 2),
        )

    def forward(
        self,
        seq_feat: torch.Tensor,
        seq_len: torch.Tensor,
        global_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, max_len, _ = seq_feat.shape
        sec_emb = self.section_encoder(seq_feat)
        global_emb = self.global_encoder(global_feat)

        state = torch.zeros(batch_size, self.state_dim, device=seq_feat.device)
        for t in range(max_len):
            cur_sec = sec_emb[:, t, :]
            new_state = self.updater(cur_sec, global_emb, state)
            active = (t < seq_len).float().unsqueeze(1)
            state = active * new_state + (1.0 - active) * state

        final_repr = torch.cat([state, global_emb], dim=-1)

        logH_per_section = self.logH_head(final_repr).squeeze(1)
        logH = logH_per_section * seq_len.float()
        H_mag = torch.exp(logH)

        phase_raw = self.phase_head(final_repr)
        raw_sin = phase_raw[:, 0]
        raw_cos = phase_raw[:, 1]
        norm = torch.sqrt(raw_sin**2 + raw_cos**2 + 1e-12)
        sin_phi = raw_sin / norm
        cos_phi = raw_cos / norm
        phi = torch.atan2(sin_phi, cos_phi)

        return {
            "logH_per_section": logH_per_section,
            "logH": logH,
            "sin_phi": sin_phi,
            "cos_phi": cos_phi,
            "phi": phi,
            "H_mag": H_mag,
        }
