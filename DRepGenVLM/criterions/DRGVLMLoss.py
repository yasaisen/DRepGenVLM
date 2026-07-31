"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2026, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2607081515
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


from ..configs.DRGVLM_baseConfig import DRGVLM_baseConfig
from ..common.utils import log_print, _debug_print


class DRGVLMLoss(nn.Module):
    def __init__(self,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.label_smoothing = float(label_smoothing)

    @staticmethod
    def _finite_summary(tensor: torch.Tensor) -> str:
        flat = tensor.detach().reshape(-1)
        finite = torch.isfinite(flat)
        bad_count = int((~finite).sum().item())
        finite_values = flat[finite]
        if finite_values.numel() == 0:
            return f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, bad_count={bad_count}, all_nonfinite=True"
        finite_values = finite_values.float()
        return (
            f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, bad_count={bad_count}, "
            f"finite_min={float(finite_values.min().item()):.6g}, "
            f"finite_max={float(finite_values.max().item()):.6g}"
        )

    @staticmethod
    def _is_finite(tensor: torch.Tensor) -> bool:
        return bool(torch.isfinite(tensor).all().detach().cpu().item())

    def _assert_finite(self, name: str, tensor: torch.Tensor, context: str):
        if not self._is_finite(tensor):
            raise RuntimeError(
                f"Non-finite tensor in DRGVLMLoss: {name}. {context}. {self._finite_summary(tensor)}"
            )

    def forward(self,
        logits: torch.Tensor,   # (B, S, V)  – raw unnormalised logits
        labels: torch.Tensor,   # (B, S)     – -100 for ignored positions
        context: str = "",
    ) -> Dict[str, torch.Tensor]:
        if logits.ndim != 3:
            raise ValueError(f"logits must be 3-D (B, S, V), got shape={tuple(logits.shape)}.")
        if labels.ndim != 2:
            raise ValueError(f"labels must be 2-D (B, S), got shape={tuple(labels.shape)}.")
        if logits.shape[:2] != labels.shape:
            raise ValueError(
                f"logits/labels batch×seq mismatch: logits={tuple(logits.shape[:2])}, labels={tuple(labels.shape)}."
            )

        # Standard causal-LM shift: predict token[i+1] from token[i]
        shift_logits = logits[:, :-1, :].contiguous()   # (B, S-1, V)
        shift_labels = labels[:, 1:].contiguous()        # (B, S-1)

        vocab_size = shift_logits.shape[-1]
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size).to(torch.float32),
            shift_labels.view(-1),
            ignore_index=-100,
            label_smoothing=self.label_smoothing,
        )
        self._assert_finite("ce_loss", ce_loss, context)

        return {"total_loss": ce_loss}

    # ------------------------------------------------------------------
    # from_config
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls,
        cfg: DRGVLM_baseConfig,
    ) -> "DRGVLMLoss":
        criterion = cls(
            label_smoothing=getattr(cfg, "label_smoothing", 0.0),
        )
        return criterion














