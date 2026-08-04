"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2026, yasaisen (clover)

 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.

 last modified in 2607081524
"""


import glob
import hashlib
import math
import os
import json
import random
import shutil
import tempfile
import uuid
from contextlib import nullcontext
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..common.metricsTracker import MetricsTracker
from ..common.metricsTracker_v2 import TrainingMonitor
from ..evaluator.DRGVLMEvaluator import DRGVLMEvaluator
from .building_scheduler import build_cosine_annealing_with_warmup_scheduler
from ..configs.DRGVLM_baseConfig import DRGVLM_baseConfig
from ..common.utils import log_print, _debug_print


class DRGVLM_PPTrainer:
    def __init__(self,
        model: nn.Module,
        device: str = None,
        num_epochs: int = None,
        trainer_mode: str = None,
        num_batches_per_epoch: int = None,
        save_path: str = None,
        weight_filename: str = None,
        save_freq: int = 5,
        plot_freq: int = 5,
        gradient_clip_norm: float = None,
        early_stop_patience: int = None,
        amp: bool = False,
        accumulation_steps: int = None,
        max_new_tokens: int = 256,
        checkpoint_every_n_optimizer_steps: Optional[int] = 100,
    ):
        raw_device = device or "cuda:0"
        if raw_device == "cuda":
            raw_device = "cuda:0"
        self.device = raw_device

        self.num_epochs = num_epochs
        self.trainer_mode = trainer_mode
        self.num_batches_per_epoch = num_batches_per_epoch
        self.save_path = save_path
        self.weight_filename = weight_filename if weight_filename is not None else "best_model.pth"
        self.save_freq = save_freq
        self.plot_freq = plot_freq
        self.early_stop_patience = early_stop_patience
        self.max_new_tokens = max_new_tokens

        self.model = model
        self.optimizer = None
        self.scheduler = None
        self.metrics = None
        self.evaluator = None
        self.gradient_clip_norm = gradient_clip_norm

        self.amp = amp
        self.accumulation_steps = max(1, int(accumulation_steps)) if accumulation_steps is not None else 1
        self.checkpoint_every_n_optimizer_steps = (
            max(0, int(checkpoint_every_n_optimizer_steps))
            if checkpoint_every_n_optimizer_steps is not None
            else 0
        )

        self.checkpoint_epoch_idx = None
        self.global_step = 0
        self.optimizer_step = 0
        self.best_val_loss = float("inf")
        self.best_metric_values: Dict[str, Any] = {}
        self.patience_counter = 0
        self.resume_num_batches_per_epoch = None
        self.resume_checkpoint_path: Optional[str] = None
        self.resume_checkpoint_root: Optional[str] = None
        self.resume_signature_static: Dict[str, Any] = {}
        self.current_resume_signature: Dict[str, Any] = {}
        self._loaded_resume_signature: Optional[Dict[str, Any]] = None
        self._pending_train_generator_state = None
        self._pending_train_progress: Optional[Dict[str, Any]] = None
        self._active_train_generator = None
        self._active_train_progress: Optional[Dict[str, Any]] = None
        self._snapshot_cache_key = None
        self._snapshot_cache_path = None

        log_print(f"home device = {self.device}")

    # ------------------------------------------------------------------
    def get_model_raw(self) -> nn.Module:
        return self.model

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------
    def init_optimizer(self,
        lr_dict: Dict[str, float],
        total_steps: int,
        warmup_steps: int,
        weight_decay: float = 1e-2,
        betas: Tuple[float, float] = (0.9, 0.999),
    ):
        log_print("Loading Optimizer...")
        model_raw = self.get_model_raw()
        trainable_params = [p for p in model_raw.parameters() if p.requires_grad]
        if len(trainable_params) == 0:
            raise RuntimeError("No trainable parameters found. Check LoRA configuration.")

        no_decay = ["bias", "LayerNorm.weight", "BatchNorm.weight", "norm.weight"]
        module_list = [name for name in lr_dict if name != "general" and lr_dict[name] is not None]

        param_dicts = [
            {
                "params": [
                    p for n, p in model_raw.named_parameters()
                    if not any(m in n for m in module_list)
                    and not any(nd in n for nd in no_decay)
                    and p.requires_grad
                ],
                "lr": lr_dict["general"],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p for n, p in model_raw.named_parameters()
                    if not any(m in n for m in module_list)
                    and any(nd in n for nd in no_decay)
                    and p.requires_grad
                ],
                "lr": lr_dict["general"],
                "weight_decay": 0.0,
            },
        ]
        for module_name in module_list:
            param_dicts += [
                {
                    "params": [
                        p for n, p in model_raw.named_parameters()
                        if module_name in n and not any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "lr": lr_dict[module_name],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [
                        p for n, p in model_raw.named_parameters()
                        if module_name in n and any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "lr": lr_dict[module_name],
                    "weight_decay": 0.0,
                },
            ]

        self.optimizer = optim.AdamW(params=param_dicts, lr=lr_dict["general"], betas=betas)

        total_steps = max(1, int(total_steps))
        warmup_steps = int(warmup_steps)
        if warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be non-negative, got {warmup_steps}."
            )

        self.scheduler = build_cosine_annealing_with_warmup_scheduler(
            optimizer=self.optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )
        log_print("...Done\n")

    # ------------------------------------------------------------------
    def init_metrics(self,
        monitoring_every_n_steps: int = 500,
        is_custom_criterion: bool = False,
    ):
        log_print("Loading MetricsTracker...")
        self.metrics = MetricsTracker(save_path=self.save_path)
        self.monitor = TrainingMonitor(
            writer=self.metrics.writer,
            model=self.get_model_raw(),
            optimizer=self.optimizer,
            check_every_n_steps=monitoring_every_n_steps,
            is_custom_criterion=is_custom_criterion,
        )
        self.global_step = 0
        log_print("...Done\n")

    def init_evaluator(self, evaluator: DRGVLMEvaluator):
        log_print("Loading Evaluator...")
        self.evaluator = evaluator
        log_print("...Done\n")

    # ------------------------------------------------------------------
    # Finite-check helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _is_finite_tensor(t: torch.Tensor) -> bool:
        return bool(torch.isfinite(t).all().detach().cpu().item())

    @staticmethod
    def _tensor_finite_summary(t: torch.Tensor) -> str:
        flat = t.detach().reshape(-1)
        finite = torch.isfinite(flat)
        bad = int((~finite).sum().item())
        fv = flat[finite]
        if fv.numel() == 0:
            return f"shape={tuple(t.shape)}, dtype={t.dtype}, bad={bad}, all_nonfinite=True"
        fv = fv.float()
        return (
            f"shape={tuple(t.shape)}, dtype={t.dtype}, bad={bad}, "
            f"finite_min={float(fv.min()):.6g}, finite_max={float(fv.max()):.6g}"
        )

    def _raise_nonfinite(self, name: str, t: torch.Tensor, context: str):
        raise RuntimeError(
            f"Non-finite tensor: {name}. {context}. {self._tensor_finite_summary(t)}"
        )

    def _assert_loss_finite(self, loss_dict: Dict[str, torch.Tensor], context: str):
        for k, v in loss_dict.items():
            if torch.is_tensor(v) and not self._is_finite_tensor(v):
                self._raise_nonfinite(f"loss_dict[{k}]", v, context)

    def _assert_trainable_params_finite(self, context: str):
        for name, p in self.get_model_raw().named_parameters():
            if p.requires_grad and not self._is_finite_tensor(p):
                self._raise_nonfinite(f"param[{name}]", p, context)

    def _assert_trainable_grads_finite(self, context: str):
        for name, p in self.get_model_raw().named_parameters():
            if p.requires_grad and p.grad is not None and not self._is_finite_tensor(p.grad):
                self._raise_nonfinite(f"grad[{name}]", p.grad, context)

    def _assert_trainable_grads_present(self, context: str):
        """Fail before optimizer/scheduler steps when autograd missed LoRA params."""
        trainable = [
            (name, p)
            for name, p in self.get_model_raw().named_parameters()
            if p.requires_grad
        ]
        if not trainable:
            raise RuntimeError(f"No trainable parameters found. {context}")

        missing = [name for name, p in trainable if p.grad is None]
        if missing:
            raise RuntimeError(
                "Missing gradients for trainable parameters. "
                f"{context}. missing={len(missing)}/{len(trainable)}, "
                f"examples={missing[:12]}"
            )

        nonzero = [
            name
            for name, p in trainable
            if bool(torch.count_nonzero(p.grad).detach().cpu().item())
        ]
        if not nonzero:
            raise RuntimeError(
                "All trainable gradients are exactly zero. "
                f"{context}. trainable={len(trainable)}"
            )

    def _assert_optimizer_state_finite(self, context: str):
        if self.optimizer is None:
            return
        p2n = {p: n for n, p in self.get_model_raw().named_parameters() if p.requires_grad}
        for p, state in self.optimizer.state.items():
            pn = p2n.get(p, f"<unknown:{id(p)}>")
            for sn, sv in state.items():
                if torch.is_tensor(sv) and not self._is_finite_tensor(sv):
                    self._raise_nonfinite(f"opt_state[{pn}][{sn}]", sv, context)

    # ------------------------------------------------------------------
    # Batch helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _as_case_list(batch_cases) -> List[Any]:
        if isinstance(batch_cases, list):
            return batch_cases
        if isinstance(batch_cases, tuple):
            return list(batch_cases)
        return [batch_cases]

    @staticmethod
    def _mean_float_dict(buf: List[Dict[str, float]]) -> Dict[str, float]:
        if not buf:
            return {}
        return {k: float(np.mean([d[k] for d in buf])) for k in buf[0]}

    @staticmethod
    def _accumulation_block_size(
        batch_idx: int,
        num_batches: int,
        accumulation_steps: int,
    ) -> int:
        """Return the actual size of the accumulation block containing batch_idx."""
        accumulation_steps = max(1, int(accumulation_steps))
        block_start = (int(batch_idx) // accumulation_steps) * accumulation_steps
        return max(1, min(accumulation_steps, int(num_batches) - block_start))

    @staticmethod
    def _snapshot_train_batch_sampler_cache(
        dataloader: DataLoader,
    ) -> Optional[List[List[int]]]:
        """Copy a sampler's cached epoch plan when it exposes one."""
        batch_sampler = getattr(dataloader, "batch_sampler", None)
        cached_batches = getattr(batch_sampler, "_cached_batches", None)
        if cached_batches is None:
            return None
        return [list(batch) for batch in cached_batches]

    @staticmethod
    def _restore_train_batch_sampler_cache(
        dataloader: DataLoader,
        cached_batches: Optional[List[List[int]]],
    ):
        """Restore the cached epoch plan used by MaxROIBatchSampler."""
        if cached_batches is None:
            return
        batch_sampler = getattr(dataloader, "batch_sampler", None)
        if not hasattr(batch_sampler, "_cached_batches"):
            raise RuntimeError(
                "In-epoch checkpoint contains a cached batch plan, but the "
                "current DataLoader batch sampler cannot restore it."
            )
        batch_sampler._cached_batches = [
            list(batch)
            for batch in cached_batches
        ]

    # ------------------------------------------------------------------
    # Train one batch (gradient accumulation aware)
    # ------------------------------------------------------------------
    def _backward_train_batch(self,
        batch_cases,
        accumulation_denom: int,
    ) -> Tuple[Dict[str, float], int]:
        case_list = self._as_case_list(batch_cases)
        case_count = max(1, len(case_list))
        accumulation_denom = max(1, int(accumulation_denom))
        loss_dict_buffer: List[Dict[str, float]] = []
        total_dx_count = 0

        for case in case_list:
            active_dx_items = self.get_model_raw()._active_dxitems(case)
            n_dx = len(active_dx_items)
            if n_dx == 0:
                raise RuntimeError(
                    f"Case has no active DxItem ROI groups: "
                    f"case_id={case.case_id}"
                )
            total_dx_count += n_dx
            roi_assignment_count = sum(
                len(self.get_model_raw()._get_dxitem_rois(case, dx_item))
                for dx_item in active_dx_items
            )
            case_context_str = (
                f"case_id={case.case_id}, "
                f"unique_rois={len(case.rois)}, "
                f"roi_assignments={roi_assignment_count}, DxItems={n_dx}"
            )

            # Shared-vision caching performs a no_grad VLM forward followed by a
            # trainable decoder forward.  Disable the autocast weight cache on
            # this path so detached FP32->BF16 LoRA casts from the no_grad pass
            # can never be reused by a trainable pair forward.
            uses_shared_vision_cache = bool(
                getattr(self.get_model_raw(), "use_shared_vision_cache", False)
            )
            pair_loss_dict_buffer: List[Dict[str, float]] = []
            pair_loss_scale = accumulation_denom * case_count * n_dx
            dxitem_groups = (
                self.get_model_raw()._group_dxitems_by_roi_signature(
                    case,
                    dx_items=active_dx_items,
                )
            )
            for group_dx_items in dxitem_groups:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=self.amp,
                    cache_enabled=not uses_shared_vision_cache,
                ):
                    case_context_map = (
                        self.get_model_raw().prepare_case_loss_context(
                            case,
                            dx_items=group_dx_items,
                        )
                    )

                for DxItem in group_dx_items:
                    pair_context = f"{case_context_str}, DxItem={DxItem}"
                    dxitem_context = (
                        case_context_map.get(DxItem)
                        if case_context_map is not None
                        else None
                    )
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=self.amp,
                        cache_enabled=not uses_shared_vision_cache,
                    ):
                        loss_dict = (
                            self.get_model_raw().calculate_dxitem_loss(
                                case=case,
                                DxItem=DxItem,
                                case_context=dxitem_context,
                            )
                        )

                    self._assert_loss_finite(loss_dict, pair_context)
                    scaled_loss = loss_dict["total_loss"] / pair_loss_scale
                    scaled_loss.backward()
                    pair_loss_dict_buffer.append({
                        k: float(v.detach().item())
                        for k, v in loss_dict.items()
                    })
                    del scaled_loss, loss_dict
                del case_context_map

            loss_dict_buffer.append(
                self._mean_float_dict(pair_loss_dict_buffer)
            )

        return self._mean_float_dict(loss_dict_buffer), total_dx_count

    # ------------------------------------------------------------------
    # epoch_train
    # ------------------------------------------------------------------
    def epoch_train(self,
        dataloader: DataLoader,
        epoch_idx: int,
        resume_batch_idx: int = 0,
        resume_epoch_losses: Optional[List[float]] = None,
    ) -> float:
        self.get_model_raw().train()
        self.monitor.reset_phase(
            phase="train",
            global_step=self.global_step,
        )
        resume_batch_idx = int(resume_batch_idx)
        epoch_losses = [
            float(loss)
            for loss in (resume_epoch_losses or [])
        ]
        if len(epoch_losses) != resume_batch_idx:
            raise RuntimeError(
                "In-epoch checkpoint loss history does not match its next "
                f"batch index: losses={len(epoch_losses)}, "
                f"next_batch_idx={resume_batch_idx}."
            )

        epoch_start_train_generator_state = None
        if self._active_train_generator is not None:
            epoch_start_train_generator_state = (
                self._active_train_generator.get_state().clone()
            )
        num_batches = len(dataloader)
        epoch_batch_sampler_cache = self._snapshot_train_batch_sampler_cache(
            dataloader
        )
        if resume_batch_idx < 0 or resume_batch_idx > num_batches:
            raise RuntimeError(
                "In-epoch checkpoint next_batch_idx must identify an "
                f"unfinished batch, got {resume_batch_idx} for "
                f"{num_batches} batches."
            )
        if num_batches == 0:
            if resume_batch_idx != 0:
                raise RuntimeError(
                    "Cannot resume a nonzero batch index from an empty "
                    "DataLoader."
                )
            self._active_train_progress = None
            return float(np.mean(epoch_losses)) if epoch_losses else 0.0
        if resume_batch_idx == num_batches:
            raise RuntimeError(
                "In-epoch checkpoint marks every batch complete. Use the "
                "corresponding end-of-epoch checkpoint instead."
            )
        if resume_batch_idx % self.accumulation_steps != 0:
            raise RuntimeError(
                "In-epoch checkpoint must resume at an accumulation boundary, "
                f"got next_batch_idx={resume_batch_idx} with "
                f"accumulation_steps={self.accumulation_steps}."
            )

        pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx} [Train]")
        if resume_batch_idx:
            log_print(
                "Resuming incomplete epoch "
                f"{epoch_idx} from batch {resume_batch_idx}/{num_batches}."
            )

        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(pbar):
            if batch_idx < resume_batch_idx:
                continue

            is_last_batch = (batch_idx + 1 == num_batches)
            is_accum_step = ((batch_idx + 1) % self.accumulation_steps == 0) or is_last_batch

            accumulation_denom = self._accumulation_block_size(
                batch_idx=batch_idx,
                num_batches=num_batches,
                accumulation_steps=self.accumulation_steps,
            )

            with nullcontext():
                loss_dict, total_dx_count = self._backward_train_batch(
                    batch_cases=batch,
                    accumulation_denom=accumulation_denom,
                )

            batch_case_count = len(self._as_case_list(batch))
            self.global_step += 1
            batch_roi_count = sum(
                sum(
                    len(rois)
                    for rois in getattr(case, "DxItem_rois", {}).values()
                )
                for case in self._as_case_list(batch)
            )
            self.monitor.log_always_on(
                self.global_step,
                batch_size=batch_case_count,
                dx_count=total_dx_count,
                roi_count=batch_roi_count,
            )
            self.monitor.log_periodic(self.global_step)
            epoch_losses.append(float(loss_dict["total_loss"]))

            optimizer_step_lr = None
            if is_accum_step:
                step_context = (
                    f"epoch={epoch_idx}, batch_idx={batch_idx}, "
                    f"global_step={self.global_step}, cases={batch_case_count}, dx_count={total_dx_count}"
                )
                self._assert_trainable_grads_present(context=f"before grad clip, {step_context}")
                self._assert_trainable_grads_finite(context=f"before grad clip, {step_context}")
                if self.gradient_clip_norm is not None and self.gradient_clip_norm > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.get_model_raw().parameters() if p.requires_grad],
                        max_norm=self.gradient_clip_norm,
                        error_if_nonfinite=True,
                    )
                    if torch.is_tensor(grad_norm) and not self._is_finite_tensor(grad_norm):
                        self._raise_nonfinite("clip_grad_norm", grad_norm, step_context)
                    self._assert_trainable_grads_finite(context=f"after grad clip, {step_context}")

                optimizer_step_lr = float(self.optimizer.param_groups[0]["lr"])
                self.optimizer.step()
                self._assert_trainable_params_finite(context=f"after opt step, {step_context}")
                self._assert_optimizer_state_finite(context=f"after opt step, {step_context}")
                self.scheduler.step()
                self.optimizer_step += 1
                self.optimizer.zero_grad(set_to_none=True)

            current_lr = self.optimizer.param_groups[0]["lr"]
            # Record every DataLoader batch so epoch loss is not biased toward
            # only the final microbatch in each accumulation block.  LR is
            # emitted only when an optimizer update actually occurred.
            self.metrics.update(
                loss_dict=loss_dict,
                lr=optimizer_step_lr,
                stage="train",
                rank=0,
                global_step=self.global_step,
                optimizer_step=self.optimizer_step,
            )

            pbar.set_postfix({
                "Loss": f"{loss_dict['total_loss']:.4f}",
                "LR": f"{current_lr:.2e}",
                "Cases": batch_case_count,
                "DxPairs": total_dx_count,
            })

            should_save_in_epoch = (
                is_accum_step
                and not is_last_batch
                and self.checkpoint_every_n_optimizer_steps > 0
                and (
                    self.optimizer_step
                    % self.checkpoint_every_n_optimizer_steps
                    == 0
                )
            )
            if should_save_in_epoch:
                if epoch_start_train_generator_state is None:
                    raise RuntimeError(
                        "In-epoch checkpointing requires an explicit train "
                        "DataLoader generator for exact continuation."
                    )
                self._active_train_progress = {
                    "epoch_idx": int(epoch_idx),
                    "next_batch_idx": int(batch_idx + 1),
                    "epoch_losses": list(epoch_losses),
                    "epoch_start_train_generator_state": (
                        epoch_start_train_generator_state.clone()
                        if epoch_start_train_generator_state is not None
                        else None
                    ),
                    "epoch_batch_sampler_cache": (
                        [list(batch) for batch in epoch_batch_sampler_cache]
                        if epoch_batch_sampler_cache is not None
                        else None
                    ),
                }
                self.save_checkpoint(
                    epoch_idx=epoch_idx,
                    weight_filename="latest_model.pth",
                )
                self._active_train_progress = None

        self._active_train_progress = None
        return float(np.mean(epoch_losses)) if epoch_losses else 0.0

    # ------------------------------------------------------------------
    # epoch_validEval
    # ------------------------------------------------------------------
    @torch.no_grad()
    def epoch_validEval(self,
        dataloader: DataLoader,
        epoch_idx: int,
    ) -> Tuple[float, Dict[str, Any]]:
        do_eval = (self.evaluator is not None)

        self.get_model_raw().eval()
        self.monitor.reset_phase(
            phase="valid",
            global_step=self.global_step,
        )
        val_losses: List[float] = []
        val_loss_dict_buffer: List[Dict[str, float]] = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx} [Valid]")

        for batch in pbar:
            batch_cases = self._as_case_list(batch)
            case_contexts = {
                case.global_idx: self.get_model_raw().prepare_case_loss_context(
                    case
                )
                for case in batch_cases
            }
            # --- loss ---
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
                loss_dict, _ = self.get_model_raw().calculate_loss(
                    batch_cases=batch,
                    case_contexts=case_contexts,
                )

            val_losses.append(float(loss_dict["total_loss"].detach().item()))
            val_loss_dict_buffer.append({k: float(v.detach().item()) for k, v in loss_dict.items()})

            # --- generation for evaluation ---
            if do_eval:
                output_case_dict = self.get_model_raw().generate_outputs(
                    batch_cases=batch,
                    max_new_tokens=self.max_new_tokens,
                    case_contexts=case_contexts,
                )
                self.evaluator.update(
                    batch_cases=self._as_case_list(batch),
                    output_case_dict=output_case_dict,
                )

        evaluate_result: Dict[str, Any] = {}
        if do_eval:
            evaluate_result = self.evaluator.evaluate(
                epoch_idx=epoch_idx,
                save_path=self.save_path,
            )

        if val_loss_dict_buffer:
            avg_val_loss_dict = {
                k: float(np.mean([d[k] for d in val_loss_dict_buffer]))
                for k in val_loss_dict_buffer[0]
            }
            self.metrics.update(
                loss_dict=avg_val_loss_dict,
                stage="val",
                rank=0,
                global_step=self.global_step,
            )

        return float(np.mean(val_losses)) if val_losses else 0.0, evaluate_result

    # ------------------------------------------------------------------
    # train / valid
    # ------------------------------------------------------------------
    def _record_validation_result(self, val_loss: float) -> bool:
        """Update best-loss and patience state before any checkpoint is saved."""
        if val_loss < self.best_val_loss:
            self.best_val_loss = float(val_loss)
            self.patience_counter = 0
            return True
        self.patience_counter += 1
        return False

    @staticmethod
    def _checkpoint_artifact_paths(
        checkpoint_dir: str,
        weight_filename: str,
        reference_name: Optional[str] = None,
    ) -> Dict[str, str]:
        stem = os.path.splitext(weight_filename)[0]
        if reference_name is None:
            reference_name = (
                "best"
                if stem == "best_model"
                else "latest"
                if stem == "latest_model"
                else stem
            )
        return {
            "reference_path": os.path.abspath(os.path.join(
                checkpoint_dir,
                "checkpoint_refs",
                f"{reference_name}.json",
            )),
            "trainer_path": os.path.abspath(os.path.join(
                checkpoint_dir,
                f"[trainer]{weight_filename}",
            )),
            "lora_path": os.path.abspath(os.path.join(
                checkpoint_dir,
                weight_filename.replace(".pth", "_lora"),
            )),
        }

    def _write_resume_source_manifest(self):
        """Retain references to historical best artifacts in a new output dir."""
        if self.resume_checkpoint_path is None:
            return

        source_checkpoint = os.path.abspath(self.resume_checkpoint_path)
        source_dir = (
            self.resume_checkpoint_root
            if self.resume_checkpoint_root is not None
            else os.path.dirname(source_checkpoint)
        )
        destination_dir = os.path.abspath(self.save_path)
        if source_dir == destination_dir:
            log_print(
                "Resume uses the existing checkpoint directory; historical best "
                "artifacts are preserved in place."
            )
            return

        os.makedirs(destination_dir, exist_ok=True)
        manifest: Dict[str, Any] = {
            "resume_checkpoint_path": source_checkpoint,
            "source_checkpoint_directory": source_dir,
            "historical_best_val_loss": float(self.best_val_loss),
            "historical_best_model": self._checkpoint_artifact_paths(
                checkpoint_dir=source_dir,
                weight_filename=self.weight_filename,
                reference_name="best",
            ),
            "historical_metric_checkpoints": {
                filename: {
                    **self._checkpoint_artifact_paths(source_dir, filename),
                    "state": state,
                }
                for filename, state in self.best_metric_values.items()
            },
        }

        upstream_manifest_path = os.path.join(
            source_dir,
            "resume_source_checkpoints.json",
        )
        if os.path.isfile(upstream_manifest_path):
            with open(upstream_manifest_path, "r", encoding="utf-8") as file:
                manifest["upstream_resume_source"] = json.load(file)

        manifest_path = os.path.join(
            destination_dir,
            "resume_source_checkpoints.json",
        )
        self._atomic_json_dump(manifest, manifest_path)
        log_print(
            "Resume output directory differs from its checkpoint source. "
            f"Historical best references saved to: {manifest_path}"
        )

    def _prepare_run_checkpoint_artifacts(self, checkpoint_epoch_idx: int):
        """Create only a truthful initial artifact; never rewrite best on resume."""
        if checkpoint_epoch_idx < 0:
            self.save_checkpoint(
                epoch_idx=-1,
                weight_filename="initial_model.pth",
            )
            return

        self._write_resume_source_manifest()
        log_print(
            "Resume detected: no startup checkpoint was written, so existing "
            "best/latest artifacts remain unchanged until an epoch completes."
        )

    @staticmethod
    def _signature_differences(
        expected: Any,
        current: Any,
        prefix: str = "",
    ) -> List[str]:
        differences: List[str] = []
        if isinstance(expected, dict) and isinstance(current, dict):
            for key in sorted(set(expected) | set(current)):
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                if key not in expected:
                    differences.append(
                        f"{child_prefix}: checkpoint=<missing>, "
                        f"current={current[key]!r}"
                    )
                elif key not in current:
                    differences.append(
                        f"{child_prefix}: checkpoint={expected[key]!r}, "
                        "current=<missing>"
                    )
                else:
                    differences.extend(
                        DRGVLM_PPTrainer._signature_differences(
                            expected[key],
                            current[key],
                            child_prefix,
                        )
                    )
            return differences
        if expected != current:
            differences.append(
                f"{prefix}: checkpoint={expected!r}, current={current!r}"
            )
        return differences

    @classmethod
    def _build_static_resume_signature(
        cls,
        cfg: DRGVLM_baseConfig,
    ) -> Dict[str, Any]:
        def metadata_signature(path: Optional[str]) -> Optional[Dict[str, Any]]:
            if not path:
                return None
            absolute_path = os.path.abspath(path)
            return {
                "sha256": cls._sha256_file(absolute_path),
                "size_bytes": os.path.getsize(absolute_path),
            }

        max_rois_per_dxitem = getattr(
            cfg,
            "max_rois_per_dxitem",
            None,
        )
        if max_rois_per_dxitem is None:
            max_rois_per_dxitem = getattr(
                cfg,
                "max_rois_per_case",
                None,
            )

        return {
            "signature_schema_version": 2,
            "metadata": {
                "train": metadata_signature(
                    getattr(cfg, "train_metadata_path", None)
                ),
                "valid": metadata_signature(
                    getattr(cfg, "valid_metadata_path", None)
                ),
            },
            "dataset": {
                "batch_size": getattr(cfg, "batch_size", None),
                "dataloader_seed": getattr(cfg, "dataloader_seed", 42),
                "input_img": getattr(cfg, "input_img", True),
                "input_loc": getattr(cfg, "input_loc", True),
                "level_key": getattr(cfg, "level_key", "main_info"),
                "max_rois_per_dxitem": max_rois_per_dxitem,
                "roi_sampling_mode": getattr(
                    cfg,
                    "roi_sampling_mode",
                    "all",
                ),
                "use_max_roi_sampler": getattr(
                    cfg,
                    "use_max_roi_sampler",
                    False,
                ),
                "max_rois_per_batch": getattr(
                    cfg,
                    "max_rois_per_batch",
                    None,
                ),
                "DxItem_list": list(getattr(cfg, "DxItem_list", []) or []),
            },
            "optimization": {
                "accumulation_steps": int(
                    getattr(cfg, "accumulation_steps", 1)
                ),
                "total_steps": int(getattr(cfg, "total_steps", 0)),
                "warmup_steps": int(getattr(cfg, "warmup_steps", 0)),
            },
            "model": {
                "model_name": getattr(cfg, "model_name", None),
                "training_mode": getattr(cfg, "training_mode", None),
                "pp_num_gpus": getattr(cfg, "pp_num_gpus", None),
                "pp_vision_split_index": getattr(
                    cfg,
                    "pp_vision_split_index",
                    14,
                ),
                "lora_r": getattr(cfg, "lora_r", None),
                "lora_alpha": getattr(cfg, "lora_alpha", None),
                "lora_dropout": getattr(cfg, "lora_dropout", None),
                "lora_target_modules": list(
                    getattr(cfg, "lora_target_modules", []) or []
                ),
                "use_vision_lora": getattr(
                    cfg,
                    "use_vision_lora",
                    False,
                ),
            },
        }

    def _prepare_resume_runtime(self, train_dataloader: DataLoader):
        self._active_train_generator = getattr(
            train_dataloader,
            "generator",
            None,
        )
        self.current_resume_signature = {
            **self.resume_signature_static,
            "runtime": {
                "dataset_length": len(train_dataloader.dataset),
                "dataloader_batches": len(train_dataloader),
                "sampler_type": type(train_dataloader.sampler).__name__,
                "batch_sampler_type": type(
                    train_dataloader.batch_sampler
                ).__name__,
            },
        }

        is_continuation = (
            self.checkpoint_epoch_idx is not None
            and self.checkpoint_epoch_idx >= 0
            and self.trainer_mode != "reTrain"
        )
        if not is_continuation:
            return

        if self._loaded_resume_signature:
            differences = self._signature_differences(
                self._loaded_resume_signature,
                self.current_resume_signature,
            )
            if differences:
                raise RuntimeError(
                    "keepTrain resume signature mismatch; refusing to mix a "
                    "checkpoint with changed data/training/model topology:\n- "
                    + "\n- ".join(differences[:40])
                )
        else:
            log_print(
                "[WARN] Legacy checkpoint has no resume signature. Only the "
                "available num_batches_per_epoch compatibility check can run."
            )

        if (
            self.resume_num_batches_per_epoch is not None
            and int(self.resume_num_batches_per_epoch)
            != len(train_dataloader)
        ):
            raise RuntimeError(
                "keepTrain DataLoader length mismatch: checkpoint="
                f"{self.resume_num_batches_per_epoch}, "
                f"current={len(train_dataloader)}."
            )

        if self._pending_train_generator_state is not None:
            if self._active_train_generator is None:
                raise RuntimeError(
                    "Checkpoint contains a train DataLoader generator state, "
                    "but the current DataLoader has no explicit generator."
                )
            self._active_train_generator.set_state(
                self._pending_train_generator_state
            )
            log_print(
                "Restored train DataLoader generator state for sampler/worker "
                "seed continuation."
            )

    def train(self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ):
        log_print(f"Training started, total epochs {self.num_epochs}")
        self._prepare_resume_runtime(train_dataloader)
        checkpoint_epoch_idx = self.checkpoint_epoch_idx if self.checkpoint_epoch_idx is not None else -1
        self._prepare_run_checkpoint_artifacts(
            checkpoint_epoch_idx=checkpoint_epoch_idx,
        )

        resume_train_progress = self._pending_train_progress
        if resume_train_progress is not None:
            resume_epoch_idx = int(resume_train_progress["epoch_idx"])
            if resume_epoch_idx != checkpoint_epoch_idx:
                raise RuntimeError(
                    "In-epoch checkpoint epoch does not match its checkpoint "
                    f"metadata: progress={resume_epoch_idx}, "
                    f"checkpoint={checkpoint_epoch_idx}."
                )
            if resume_epoch_idx >= self.num_epochs:
                raise RuntimeError(
                    "In-epoch checkpoint refers to an epoch outside the current "
                    f"training run: epoch={resume_epoch_idx}, "
                    f"num_epochs={self.num_epochs}."
                )
            first_epoch_idx = resume_epoch_idx
        else:
            first_epoch_idx = checkpoint_epoch_idx + 1

        for epoch_idx in range(first_epoch_idx, self.num_epochs):
            is_resumed_incomplete_epoch = (
                resume_train_progress is not None
                and epoch_idx == int(resume_train_progress["epoch_idx"])
            )
            if is_resumed_incomplete_epoch:
                self._restore_train_batch_sampler_cache(
                    train_dataloader,
                    resume_train_progress["epoch_batch_sampler_cache"],
                )
            train_loss = self.epoch_train(
                train_dataloader,
                epoch_idx=epoch_idx,
                resume_batch_idx=(
                    int(resume_train_progress["next_batch_idx"])
                    if is_resumed_incomplete_epoch
                    else 0
                ),
                resume_epoch_losses=(
                    resume_train_progress["epoch_losses"]
                    if is_resumed_incomplete_epoch
                    else None
                ),
            )
            if is_resumed_incomplete_epoch:
                self._pending_train_progress = None
                resume_train_progress = None
            val_loss, metric_dict = 0.0, {}
            if val_dataloader is not None:
                val_loss, metric_dict = self.epoch_validEval(val_dataloader, epoch_idx=epoch_idx)
            log_print(f"Epoch ({epoch_idx}/{self.num_epochs}): Train={train_loss:.4f}, Val={val_loss:.4f}")

            if val_dataloader is not None and val_loss == 0.0:
                self.save_checkpoint(
                    epoch_idx=epoch_idx,
                    weight_filename=f"error_quick_checkpoint_epoch_{epoch_idx}.pth",
                )
                raise ValueError(f"Val loss=0.0 at epoch {epoch_idx}, quick checkpoint saved.")

            new_best = False
            if val_dataloader is not None:
                new_best = self._record_validation_result(val_loss=val_loss)

            self._update_metric_checkpoints(
                epoch_idx=epoch_idx,
                val_loss=val_loss,
                metric_dict=metric_dict,
            )

            if new_best:
                self.save_checkpoint(epoch_idx=epoch_idx)
                log_print(f"New best saved. Val Loss: {self.best_val_loss:.6f}")

            self.metrics.epoch_summary(metric_dict=metric_dict, global_step=self.global_step)

            if epoch_idx % self.save_freq == 0:
                self.save_checkpoint(
                    epoch_idx=epoch_idx,
                    weight_filename=f"checkpoint_epoch_{epoch_idx}.pth",
                )
            self.save_checkpoint(epoch_idx=epoch_idx, weight_filename="latest_model.pth")

            if self.early_stop_patience is not None and self.patience_counter >= self.early_stop_patience:
                log_print("Early stopping triggered.")
                break

        self.metrics.close()

    def valid(self, val_dataloader: DataLoader):
        log_print("Validation started")
        val_loss, metric_dict = self.epoch_validEval(val_dataloader, epoch_idx=self.checkpoint_epoch_idx)
        log_print(f"Val Loss = {val_loss:.4f}")
        self.metrics.epoch_summary(metric_dict=metric_dict, global_step=self.global_step)
        self.metrics.close()

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    @staticmethod
    def _atomic_json_dump(payload: Dict[str, Any], path: str):
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        fd, temporary_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            dir=directory,
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(
                    payload,
                    file,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, path)
        except Exception:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
            raise

    @staticmethod
    def _sha256_file(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _artifact_hashes(cls, snapshot_path: str) -> Dict[str, str]:
        hashes: Dict[str, str] = {}
        for root, _, filenames in os.walk(snapshot_path):
            for filename in sorted(filenames):
                absolute_path = os.path.join(root, filename)
                relative_path = os.path.relpath(
                    absolute_path,
                    snapshot_path,
                )
                if relative_path == "manifest.json":
                    continue
                hashes[relative_path] = cls._sha256_file(absolute_path)
        return hashes

    @staticmethod
    def _capture_rng_state() -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": None,
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_rng_state(state: Optional[Dict[str, Any]]):
        if not isinstance(state, dict):
            log_print(
                "[WARN] Continuation checkpoint has no RNG state; exact "
                "sample/augmentation replay cannot be guaranteed."
            )
            return
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        saved_cuda = state.get("torch_cuda")
        if saved_cuda is not None:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "Checkpoint contains CUDA RNG state but CUDA is unavailable."
                )
            current_count = torch.cuda.device_count()
            if len(saved_cuda) != current_count:
                raise RuntimeError(
                    "CUDA topology changed during keepTrain resume: "
                    f"checkpoint GPUs={len(saved_cuda)}, current GPUs={current_count}."
                )
            torch.cuda.set_rng_state_all(saved_cuda)

    def _reference_name(self, weight_filename: str) -> str:
        stem = os.path.splitext(os.path.basename(weight_filename))[0]
        if weight_filename == self.weight_filename or stem == "best_model":
            return "best"
        if stem == "latest_model":
            return "latest"
        return stem

    def _checkpoint_state_cache_key(self, epoch_idx: int) -> str:
        metric_losses = None
        if self.metrics is not None:
            metric_losses = getattr(
                getattr(self.metrics, "train_loss_logger", None),
                "losses",
                None,
            )
        serializable = {
            "epoch_idx": int(epoch_idx),
            "global_step": int(getattr(self, "global_step", 0)),
            "optimizer_step": int(getattr(self, "optimizer_step", 0)),
            "best_val_loss": float(
                getattr(self, "best_val_loss", float("inf"))
            ),
            "best_metric_values": getattr(self, "best_metric_values", {}),
            "patience_counter": int(getattr(self, "patience_counter", 0)),
            "metric_loss_count": (
                len(metric_losses)
                if isinstance(metric_losses, list)
                else None
            ),
            "train_progress": (
                {
                    "epoch_idx": int(self._active_train_progress["epoch_idx"]),
                    "next_batch_idx": int(
                        self._active_train_progress["next_batch_idx"]
                    ),
                    "epoch_loss_count": len(
                        self._active_train_progress["epoch_losses"]
                    ),
                }
                if self._active_train_progress is not None
                else None
            ),
        }
        encoded = json.dumps(
            serializable,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _checkpoint_payload(self, epoch_idx: int) -> Dict[str, Any]:
        train_generator_state = None
        if self._active_train_generator is not None:
            train_generator_state = self._active_train_generator.get_state()
        return {
            "checkpoint_schema_version": 2,
            "epoch_idx": epoch_idx,
            "global_step": int(getattr(self, "global_step", 0)),
            "optimizer_step": int(getattr(self, "optimizer_step", 0)),
            "best_val_loss": float(
                getattr(self, "best_val_loss", float("inf"))
            ),
            "best_metric_values": getattr(self, "best_metric_values", {}),
            "patience_counter": int(getattr(self, "patience_counter", 0)),
            "num_batches_per_epoch": self.num_batches_per_epoch,
            "optimizer_state_dict": (
                self.optimizer.state_dict()
                if self.optimizer is not None
                else None
            ),
            "scheduler_state_dict": (
                self.scheduler.state_dict()
                if self.scheduler is not None
                else None
            ),
            "metrics": (
                self.metrics.train_loss_logger.losses
                if self.metrics is not None
                else None
            ),
            "rng_state": self._capture_rng_state(),
            "train_dataloader_generator_state": train_generator_state,
            "train_progress": self._active_train_progress,
            "resume_signature": self.current_resume_signature,
        }

    def _create_or_reuse_snapshot(self, epoch_idx: int) -> str:
        cache_key = self._checkpoint_state_cache_key(epoch_idx)
        if (
            cache_key == self._snapshot_cache_key
            and self._snapshot_cache_path is not None
            and os.path.isdir(self._snapshot_cache_path)
        ):
            return self._snapshot_cache_path

        artifact_root = os.path.join(
            os.path.abspath(self.save_path),
            "checkpoint_artifacts",
        )
        os.makedirs(artifact_root, exist_ok=True)
        snapshot_id = (
            f"epoch_{int(epoch_idx):06d}_"
            f"step_{int(getattr(self, 'global_step', 0)):012d}_"
            f"opt_{int(getattr(self, 'optimizer_step', 0)):012d}_"
            f"{uuid.uuid4().hex[:8]}"
        )
        temporary_path = tempfile.mkdtemp(
            prefix=f".{snapshot_id}.",
            dir=artifact_root,
        )
        final_path = os.path.join(artifact_root, snapshot_id)
        try:
            trainer_path = os.path.join(temporary_path, "trainer.pth")
            torch.save(self._checkpoint_payload(epoch_idx), trainer_path)
            adapter_path = os.path.join(temporary_path, "adapter")
            os.makedirs(adapter_path, exist_ok=True)
            self.get_model_raw().save_lora_weights(path=adapter_path)

            hashes = self._artifact_hashes(temporary_path)
            manifest = {
                "snapshot_schema_version": 1,
                "snapshot_id": snapshot_id,
                "created_at": datetime.now().astimezone().isoformat(),
                "epoch_idx": int(epoch_idx),
                "global_step": int(getattr(self, "global_step", 0)),
                "optimizer_step": int(getattr(self, "optimizer_step", 0)),
                "files": {
                    relative_path: {
                        "sha256": sha256,
                        "size_bytes": os.path.getsize(
                            os.path.join(temporary_path, relative_path)
                        ),
                    }
                    for relative_path, sha256 in hashes.items()
                },
            }
            self._atomic_json_dump(
                manifest,
                os.path.join(temporary_path, "manifest.json"),
            )
            os.replace(temporary_path, final_path)
        except Exception:
            if os.path.isdir(temporary_path):
                shutil.rmtree(temporary_path)
            raise

        self._snapshot_cache_key = cache_key
        self._snapshot_cache_path = final_path
        log_print(f"Immutable checkpoint snapshot saved to: {final_path}")
        return final_path

    def _write_checkpoint_reference(
        self,
        snapshot_path: str,
        weight_filename: str,
    ) -> str:
        reference_name = self._reference_name(weight_filename)
        reference_path = os.path.join(
            os.path.abspath(self.save_path),
            "checkpoint_refs",
            f"{reference_name}.json",
        )
        relative_snapshot = os.path.relpath(
            snapshot_path,
            os.path.abspath(self.save_path),
        )
        payload = {
            "checkpoint_reference_schema_version": 1,
            "reference_name": reference_name,
            "updated_at": datetime.now().astimezone().isoformat(),
            "requested_weight_filename": weight_filename,
            "snapshot": relative_snapshot,
            "trainer": os.path.join(relative_snapshot, "trainer.pth"),
            "adapter": os.path.join(relative_snapshot, "adapter"),
            "manifest": os.path.join(relative_snapshot, "manifest.json"),
        }
        self._atomic_json_dump(payload, reference_path)
        return reference_path

    def save_checkpoint(self,
        epoch_idx: int,
        weight_filename: str = None,
    ):
        checkpoint_context = (
            f"before save, epoch_idx={epoch_idx}, "
            f"global_step={getattr(self, 'global_step', None)}"
        )
        self._assert_trainable_params_finite(context=checkpoint_context)
        self._assert_optimizer_state_finite(context=checkpoint_context)

        fn = weight_filename if weight_filename is not None else self.weight_filename
        snapshot_path = self._create_or_reuse_snapshot(epoch_idx=epoch_idx)
        reference_path = self._write_checkpoint_reference(
            snapshot_path=snapshot_path,
            weight_filename=fn,
        )
        log_print(
            f"Checkpoint reference '{self._reference_name(fn)}' updated "
            f"atomically: {reference_path}"
        )
        return reference_path

    def _verify_snapshot(self, snapshot_path: str):
        manifest_path = os.path.join(snapshot_path, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(
                f"Snapshot manifest not found: {manifest_path}"
            )
        with open(manifest_path, "r", encoding="utf-8") as file:
            manifest = json.load(file)
        for relative_path, record in manifest.get("files", {}).items():
            absolute_path = os.path.join(snapshot_path, relative_path)
            if not os.path.isfile(absolute_path):
                raise FileNotFoundError(
                    f"Checkpoint artifact is missing: {absolute_path}"
                )
            actual_hash = self._sha256_file(absolute_path)
            expected_hash = record.get("sha256")
            if actual_hash != expected_hash:
                raise RuntimeError(
                    "Checkpoint integrity verification failed for "
                    f"{absolute_path}: expected={expected_hash}, "
                    f"actual={actual_hash}"
                )

    def _resolve_checkpoint_artifacts(
        self,
        path: str,
    ) -> Tuple[str, str, str]:
        requested_path = os.path.abspath(path)

        if os.path.isfile(requested_path) and requested_path.endswith(".json"):
            with open(requested_path, "r", encoding="utf-8") as file:
                reference = json.load(file)
            if "snapshot" not in reference:
                raise ValueError(
                    f"Not a DRGVLM checkpoint reference: {requested_path}"
                )
            reference_dir = os.path.dirname(requested_path)
            checkpoint_root = (
                os.path.dirname(reference_dir)
                if os.path.basename(reference_dir) == "checkpoint_refs"
                else reference_dir
            )
            snapshot_path = os.path.join(
                checkpoint_root,
                reference["snapshot"],
            )
            self._verify_snapshot(snapshot_path)
            return (
                os.path.join(snapshot_path, "trainer.pth"),
                os.path.join(snapshot_path, "adapter"),
                checkpoint_root,
            )

        if os.path.isdir(requested_path):
            if os.path.isfile(os.path.join(requested_path, "manifest.json")):
                self._verify_snapshot(requested_path)
                checkpoint_root = os.path.dirname(
                    os.path.dirname(requested_path)
                )
                return (
                    os.path.join(requested_path, "trainer.pth"),
                    os.path.join(requested_path, "adapter"),
                    checkpoint_root,
                )
            raise ValueError(
                "Checkpoint directory must be an immutable snapshot directory "
                f"containing manifest.json, got: {requested_path}"
            )

        weight_filename = os.path.basename(requested_path)
        checkpoint_root = os.path.dirname(requested_path)
        reference_path = os.path.join(
            checkpoint_root,
            "checkpoint_refs",
            f"{self._reference_name(weight_filename)}.json",
        )
        if os.path.isfile(reference_path):
            return self._resolve_checkpoint_artifacts(reference_path)

        # Backward-compatible reader for pre-schema-v2 flat checkpoints.
        trainer_path = os.path.join(
            checkpoint_root,
            f"[trainer]{weight_filename}",
        )
        lora_dir = os.path.join(
            checkpoint_root,
            weight_filename.replace(".pth", "_lora"),
        )
        if not os.path.isfile(trainer_path):
            raise FileNotFoundError(
                "Neither a new checkpoint reference nor a legacy trainer "
                f"checkpoint was found for: {requested_path}"
            )
        if not os.path.isdir(lora_dir):
            raise FileNotFoundError(
                f"LoRA directory not found for legacy checkpoint: {lora_dir}"
            )
        return trainer_path, lora_dir, checkpoint_root

    # ------------------------------------------------------------------
    # load_checkpoint
    # ------------------------------------------------------------------
    def load_checkpoint(self, path: str):
        log_print(f"Loading checkpoint from {path}")
        self.resume_checkpoint_path = os.path.abspath(path)
        trainer_path, lora_dir, checkpoint_root = (
            self._resolve_checkpoint_artifacts(path)
        )
        self.resume_checkpoint_root = checkpoint_root
        checkpoint = torch.load(
            trainer_path,
            map_location=self.device,
            weights_only=False,
        )
        self.get_model_raw().load_lora_weights(path=lora_dir)

        loaded_epoch_idx = int(checkpoint.get("epoch_idx", -1))
        self.checkpoint_epoch_idx = loaded_epoch_idx if self.trainer_mode != "reTrain" else -1
        train_progress = checkpoint.get("train_progress")

        if self.trainer_mode == "reTrain":
            # Optimizer and scheduler were freshly initialized in from_config;
            # retain those states and use only the loaded LoRA weights.
            self.global_step = 0
            self.optimizer_step = 0
            self.best_val_loss = float("inf")
            self.best_metric_values = {}
            self.patience_counter = 0
            log_print(
                "reTrain mode: loaded LoRA weights only; optimizer, scheduler, "
                "epoch, steps, best metrics, and patience were reset."
            )
        else:
            if self.optimizer is not None:
                optimizer_state = checkpoint.get("optimizer_state_dict")
                if optimizer_state is None:
                    raise RuntimeError(
                        "Continuation checkpoint has no optimizer_state_dict. "
                        "Use --retrain if only LoRA weights should be reused."
                    )
                self.optimizer.load_state_dict(optimizer_state)
            if self.scheduler is not None:
                scheduler_state = checkpoint.get("scheduler_state_dict")
                if scheduler_state is None:
                    raise RuntimeError(
                        "Continuation checkpoint has no scheduler_state_dict. "
                        "Use --retrain if the schedule should restart."
                    )
                self.scheduler.load_state_dict(scheduler_state)
            self.global_step = int(checkpoint.get("global_step", 0))
            self.optimizer_step = int(checkpoint.get(
                "optimizer_step",
                self.global_step // max(1, self.accumulation_steps),
            ))
            self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
            self.best_metric_values = checkpoint.get("best_metric_values", {}) or {}
            self.patience_counter = int(checkpoint.get("patience_counter", 0))
            self.resume_num_batches_per_epoch = checkpoint.get(
                "num_batches_per_epoch", None
            )
            self._loaded_resume_signature = checkpoint.get(
                "resume_signature"
            )
            self._pending_train_generator_state = checkpoint.get(
                "train_dataloader_generator_state"
            )
            self._pending_train_progress = None
            if train_progress is not None:
                if not isinstance(train_progress, dict):
                    raise RuntimeError(
                        "Checkpoint train_progress must be a dictionary."
                    )
                required_progress_keys = {
                    "epoch_idx",
                    "next_batch_idx",
                    "epoch_losses",
                    "epoch_start_train_generator_state",
                    "epoch_batch_sampler_cache",
                }
                missing_progress_keys = sorted(
                    required_progress_keys - set(train_progress)
                )
                if missing_progress_keys:
                    raise RuntimeError(
                        "In-epoch checkpoint is missing train_progress fields: "
                        + ", ".join(missing_progress_keys)
                    )
                if train_progress["epoch_start_train_generator_state"] is None:
                    raise RuntimeError(
                        "In-epoch checkpoint has no epoch-start DataLoader "
                        "generator state, so exact continuation is unavailable."
                    )
                self._pending_train_progress = {
                    "epoch_idx": int(train_progress["epoch_idx"]),
                    "next_batch_idx": int(train_progress["next_batch_idx"]),
                    "epoch_losses": [
                        float(loss)
                        for loss in train_progress["epoch_losses"]
                    ],
                    "epoch_start_train_generator_state": train_progress[
                        "epoch_start_train_generator_state"
                    ],
                    "epoch_batch_sampler_cache": train_progress[
                        "epoch_batch_sampler_cache"
                    ],
                }
                self._pending_train_generator_state = self._pending_train_progress[
                    "epoch_start_train_generator_state"
                ]
            self._restore_rng_state(checkpoint.get("rng_state"))
            self._migrate_clinical_metric_state(
                checkpoint_dir=checkpoint_root,
            )
            log_print(
                "Continuation mode: restored LoRA, optimizer, scheduler, epoch, "
                "steps, best metrics, and patience state."
            )

        log_print(f"Checkpoint loaded from {path}; epoch={loaded_epoch_idx}")
        return self.checkpoint_epoch_idx

    # ------------------------------------------------------------------
    # Metric checkpoints
    # ------------------------------------------------------------------
    @staticmethod
    def _finite_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    def _migrate_clinical_metric_state(self, checkpoint_dir: str):
        """Migrate legacy clinical scores without mismatching saved adapters."""
        filename = "best_clinical_composite.pth"
        previous = self.best_metric_values.get(filename)
        if not isinstance(previous, dict):
            return
        if self.evaluator is None:
            raise RuntimeError(
                "Cannot migrate clinical metric state without an evaluator."
            )

        current_version = int(
            self.evaluator.CLINICAL_METRIC_SCHEMA_VERSION
        )
        previous_version = previous.get("metric_schema_version")
        if previous_version is not None and int(previous_version) == current_version:
            return

        eval_paths = sorted(glob.glob(os.path.join(
            checkpoint_dir,
            "eval_results*.json",
        )))
        if not eval_paths:
            raise RuntimeError(
                "The checkpoint contains a legacy clinical best score, but no "
                "saved eval_results JSON files are available for schema migration."
            )

        recomputed = []
        for eval_path in eval_paths:
            with open(eval_path, "r", encoding="utf-8") as file:
                payload = json.load(file)
            metrics = self.evaluator.recompute_saved_cases(
                stored_cases=payload.get("cases", []),
            )
            score = self._finite_float(metrics.get("clinical_macro_score"))
            if score is None:
                continue
            recomputed.append({
                "score": score,
                "epoch_idx": int(payload["epoch_idx"]),
                "metrics": metrics,
                "eval_path": os.path.abspath(eval_path),
            })

        if not recomputed:
            raise RuntimeError(
                "Unable to recompute any finite clinical score from saved "
                "evaluation results."
            )
        migrated_best = max(recomputed, key=lambda item: item["score"])
        previous_epoch = int(previous.get("epoch_idx", -1))
        if migrated_best["epoch_idx"] != previous_epoch:
            raise RuntimeError(
                "Clinical metric schema migration changed the historical best "
                f"epoch from {previous_epoch} to {migrated_best['epoch_idx']}. "
                "The existing best_clinical_composite adapter cannot be relabeled "
                "safely; restore or create an adapter for the recomputed epoch."
            )

        previous["legacy_score"] = previous.get("score")
        previous["score"] = float(migrated_best["score"])
        previous["metric_schema_version"] = current_version
        previous["metrics"] = {
            key: float(value)
            for key, value in migrated_best["metrics"].items()
            if self._finite_float(value) is not None
        }
        previous["metric_migration_eval_path"] = migrated_best["eval_path"]
        log_print(
            "Migrated best clinical metric state to schema "
            f"v{current_version}: epoch={previous_epoch}, "
            f"score={previous['score']:.6f}."
        )

    def _metric_checkpoint_scores(self,
        metric_dict: Dict[str, Any],
    ) -> Dict[str, float]:
        if not metric_dict:
            return {}
        scores: Dict[str, float] = {}
        metric_to_filename = {
            "macro_exact_match": "best_macro_exact_match.pth",
            "macro_bleu": "best_macro_bleu.pth",
            "macro_token_f1": "best_macro_token_f1.pth",
            "macro_rouge_l": "best_macro_rouge_l.pth",
        }
        for metric_name, fn in metric_to_filename.items():
            score = self._finite_float(metric_dict.get(metric_name))
            if score is not None:
                scores[fn] = score

        text_composite = self._finite_float(metric_dict.get("text_composite"))
        if text_composite is not None:
            scores["best_text_composite.pth"] = text_composite

        clinical_composite = self._finite_float(
            metric_dict.get("clinical_macro_score")
        )
        if clinical_composite is not None:
            scores["best_clinical_composite.pth"] = clinical_composite

        return scores

    def _update_metric_checkpoints(self,
        epoch_idx: int,
        val_loss: float,
        metric_dict: Dict[str, Any],
    ):
        scores = self._metric_checkpoint_scores(metric_dict=metric_dict)
        if not scores:
            return

        val_loss_value = self._finite_float(val_loss)
        improved = []
        for fn, score in scores.items():
            prev = self.best_metric_values.get(fn, {})
            prev_score = self._finite_float(prev.get("score")) if isinstance(prev, dict) else None

            metric_schema_version = None
            if fn == "best_clinical_composite.pth":
                raw_version = metric_dict.get("clinical_metric_schema_version")
                if raw_version is None:
                    raise RuntimeError(
                        "clinical_macro_score is missing its metric schema version."
                    )
                metric_schema_version = int(raw_version)
                if prev_score is not None:
                    previous_version = prev.get("metric_schema_version")
                    if (
                        previous_version is None
                        or int(previous_version) != metric_schema_version
                    ):
                        raise RuntimeError(
                            "Refusing to compare clinical checkpoint scores from "
                            "different metric schema versions."
                        )

            if prev_score is not None and score <= prev_score:
                continue
            new_state = {
                "score": float(score),
                "epoch_idx": int(epoch_idx),
                "global_step": int(getattr(self, "global_step", 0)),
                "val_loss": float(val_loss_value) if val_loss_value is not None else None,
                "metrics": {k: float(v) for k, v in metric_dict.items() if self._finite_float(v) is not None},
            }
            if metric_schema_version is not None:
                new_state["metric_schema_version"] = metric_schema_version
            self.best_metric_values[fn] = new_state
            improved.append(fn)

        for fn in improved:
            score = self.best_metric_values[fn]["score"]
            self.save_checkpoint(epoch_idx=epoch_idx, weight_filename=fn)
            log_print(
                f"New best metric checkpoint: {fn}, score={score:.6f}, epoch={epoch_idx}"
            )

        if improved:
            manifest_path = os.path.join(self.save_path, "best_metric_checkpoints.json")
            self._atomic_json_dump(
                self.best_metric_values,
                manifest_path,
            )

    # ------------------------------------------------------------------
    # from_config
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls,
        cfg: DRGVLM_baseConfig,
        model: nn.Module,
        checkpoint_path: str = None,
    ) -> "DRGVLM_PPTrainer":
        print()

        trainer = cls(
            model=model,
            device=cfg.device,
            num_epochs=cfg.num_epochs,
            trainer_mode=cfg.trainer_mode,
            num_batches_per_epoch=cfg.num_batchs_per_epoch,
            save_path=cfg.save_path,
            weight_filename=cfg.weight_filename,
            save_freq=cfg.save_freq,
            plot_freq=cfg.plot_freq,
            gradient_clip_norm=cfg.gradient_clip_norm,
            early_stop_patience=cfg.early_stop_patience,
            amp=cfg.amp,
            accumulation_steps=cfg.accumulation_steps,
            max_new_tokens=getattr(cfg, "max_new_tokens", 256),
            checkpoint_every_n_optimizer_steps=getattr(
                cfg,
                "checkpoint_every_n_optimizer_steps",
                100,
            ),
        )

        cfg.total_steps = math.ceil(
            cfg.num_batchs_per_epoch / max(1, cfg.accumulation_steps)
        ) * cfg.num_epochs
        automatic_warmup_steps = int(min(
            cfg.warmup_ratio * cfg.total_steps,
            cfg.max_warmup_steps,
        ))
        if cfg.warmup_steps is None:
            cfg.warmup_steps = automatic_warmup_steps
        else:
            cfg.warmup_steps = int(cfg.warmup_steps)
            if cfg.warmup_steps < 0:
                raise ValueError(
                    "Explicit warmup_steps must be non-negative, got "
                    f"{cfg.warmup_steps}."
                )
            warning_reasons = []
            if cfg.warmup_steps > automatic_warmup_steps:
                warning_reasons.append(
                    f"automatic recommendation {automatic_warmup_steps}"
                )
            if cfg.warmup_steps > cfg.max_warmup_steps:
                warning_reasons.append(
                    f"max_warmup_steps {cfg.max_warmup_steps}"
                )
            if cfg.warmup_steps > cfg.total_steps:
                warning_reasons.append(f"total_steps {cfg.total_steps}")
            if warning_reasons:
                log_print(
                    "[WARN] Explicit warmup_steps="
                    f"{cfg.warmup_steps} exceeds "
                    + ", ".join(warning_reasons)
                    + "; preserving the explicit value."
                )
        log_print(_debug_print("Total Steps", cfg.total_steps))
        log_print(_debug_print("Warmup Steps", cfg.warmup_steps))

        trainer.init_optimizer(
            lr_dict=cfg.learning_rate_dict,
            total_steps=cfg.total_steps,
            warmup_steps=cfg.warmup_steps,
            weight_decay=cfg.weight_decay,
        )
        trainer.init_metrics()
        evaluator = DRGVLMEvaluator.from_config(cfg=cfg)
        trainer.init_evaluator(evaluator=evaluator)
        trainer.resume_signature_static = (
            trainer._build_static_resume_signature(cfg)
        )

        if checkpoint_path:
            trainer.load_checkpoint(path=checkpoint_path)

        return trainer



