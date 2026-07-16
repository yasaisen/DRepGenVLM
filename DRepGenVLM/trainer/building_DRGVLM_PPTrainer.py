"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2026, yasaisen (clover)

 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.

 last modified in 2607081524

 Pipeline Parallelism (PP) Trainer for DRGVLM.

 Design principles
 -----------------
 * Single-process, multi-GPU: the VLM backbone is split across N GPUs via a
   manual device_map built in modelBuilder.py.  No torch.distributed / DDP.

 * Trainable parameters are only the LoRA adapter weights.  They are on
   self.device ("cuda:0") after peft wrapping.

 * Loss is computed per (case, DxItem) pair; each loss.backward() is called
   immediately (no graph accumulation across pairs) to keep peak memory low.

 * Validation runs generate_outputs() for every case every epoch, then calls
   DRGVLMEvaluator to compute text-generation metrics (BLEU, token-F1, etc.).
"""


import math
import os
import json
from contextlib import nullcontext
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
    """DRGVLM trainer using Pipeline Parallelism (single process, N GPUs).

    The VLM backbone is spread across N GPUs via device_map.
    Only LoRA adapter parameters are trainable; they reside on self.device.
    """

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

        self.checkpoint_epoch_idx = None
        self.global_step = 0
        self.optimizer_step = 0
        self.best_val_loss = float("inf")
        self.best_metric_values: Dict[str, Any] = {}
        self.patience_counter = 0
        self.resume_num_batches_per_epoch = None

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
        warmup_steps = int(max(0, min(warmup_steps, total_steps)))

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

    # ------------------------------------------------------------------
    # Train one batch (gradient accumulation aware)
    # ------------------------------------------------------------------
    def _backward_train_batch(self,
        batch_cases,
        accumulation_denom: int,
    ) -> Tuple[Dict[str, float], int]:
        case_list = self._as_case_list(batch_cases)
        case_count = max(1, len(case_list))
        loss_scale = max(1, int(accumulation_denom)) * case_count
        loss_dict_buffer: List[Dict[str, float]] = []
        total_dx_count = 0

        for case in case_list:
            n_dx = len(getattr(case, "DxItem_targets", {}))
            total_dx_count += n_dx
            context = f"case_id={case.case_id}, rois={len(case.rois)}, DxItems={n_dx}"

            # Shared-vision caching performs a no_grad VLM forward followed by a
            # trainable decoder forward in this same autocast scope.  The default
            # autocast weight cache can otherwise reuse detached FP32->BF16 LoRA
            # casts from the no_grad pass and silently leave every adapter grad
            # as None.  Keep autocast itself enabled, but disable that cache for
            # this mixed no_grad/trainable path.
            uses_shared_vision_cache = bool(
                getattr(self.get_model_raw(), "use_shared_vision_cache", False)
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=self.amp,
                cache_enabled=not uses_shared_vision_cache,
            ):
                loss_dict, _ = self.get_model_raw().calculate_loss(batch_cases=[case])

            self._assert_loss_finite(loss_dict, context)
            scaled_loss = loss_dict["total_loss"] / loss_scale
            scaled_loss.backward()

            loss_dict_buffer.append({k: float(v.detach().item()) for k, v in loss_dict.items()})

        return self._mean_float_dict(loss_dict_buffer), total_dx_count

    # ------------------------------------------------------------------
    # epoch_train
    # ------------------------------------------------------------------
    def epoch_train(self,
        dataloader: DataLoader,
        epoch_idx: int,
    ) -> float:
        self.get_model_raw().train()
        epoch_losses: List[float] = []
        num_batches = len(dataloader)
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx} [Train]")

        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(pbar):
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
            self.monitor.log_always_on(self.global_step, batch_size=batch_case_count)
            self.monitor.log_periodic(self.global_step)
            self.global_step += 1
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
        val_losses: List[float] = []
        val_loss_dict_buffer: List[Dict[str, float]] = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch_idx} [Valid]")

        for batch in pbar:
            # --- loss ---
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.amp):
                loss_dict, _ = self.get_model_raw().calculate_loss(batch_cases=batch)

            val_losses.append(float(loss_dict["total_loss"].detach().item()))
            val_loss_dict_buffer.append({k: float(v.detach().item()) for k, v in loss_dict.items()})

            # --- generation for evaluation ---
            if do_eval:
                output_case_dict = self.get_model_raw().generate_outputs(
                    batch_cases=batch,
                    max_new_tokens=self.max_new_tokens,
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
    def train(self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
    ):
        log_print(f"Training started, total epochs {self.num_epochs}")
        checkpoint_epoch_idx = self.checkpoint_epoch_idx if self.checkpoint_epoch_idx is not None else -1
        self.save_checkpoint(epoch_idx=checkpoint_epoch_idx)
        self.save_checkpoint(epoch_idx=checkpoint_epoch_idx, weight_filename="latest_model.pth")

        for epoch_idx in range(checkpoint_epoch_idx + 1, self.num_epochs):
            train_loss = self.epoch_train(train_dataloader, epoch_idx=epoch_idx)
            val_loss, metric_dict = 0.0, {}
            if val_dataloader is not None:
                val_loss, metric_dict = self.epoch_validEval(val_dataloader, epoch_idx=epoch_idx)
            log_print(f"Epoch ({epoch_idx}/{self.num_epochs}): Train={train_loss:.4f}, Val={val_loss:.4f}")

            new_best = val_loss < self.best_val_loss
            if new_best:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch_idx=epoch_idx)
                log_print(f"New best saved. Val Loss: {self.best_val_loss:.6f}")
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self._update_metric_checkpoints(
                epoch_idx=epoch_idx,
                val_loss=val_loss,
                metric_dict=metric_dict,
            )

            if val_dataloader is not None and val_loss == 0.0:
                self.save_checkpoint(
                    epoch_idx=epoch_idx,
                    weight_filename=f"error_quick_checkpoint_epoch_{epoch_idx}.pth",
                )
                raise ValueError(f"Val loss=0.0 at epoch {epoch_idx}, quick checkpoint saved.")

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

        checkpoint = {
            "epoch_idx": epoch_idx,
            "global_step": int(getattr(self, "global_step", 0)),
            "optimizer_step": int(getattr(self, "optimizer_step", 0)),
            "best_val_loss": float(getattr(self, "best_val_loss", float("inf"))),
            "best_metric_values": getattr(self, "best_metric_values", {}),
            "patience_counter": int(getattr(self, "patience_counter", 0)),
            "num_batches_per_epoch": self.num_batches_per_epoch,
            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer is not None else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "metrics": self.metrics.train_loss_logger.losses if self.metrics is not None else None,
        }

        fn = weight_filename if weight_filename is not None else self.weight_filename
        trainer_fn = f"[trainer]{fn}"
        lora_dir = fn.replace(".pth", "_lora")

        trainer_path = os.path.join(self.save_path, trainer_fn)
        torch.save(checkpoint, trainer_path)
        log_print(f"Trainer checkpoint saved to: {trainer_path}")

        lora_path = os.path.join(self.save_path, lora_dir)
        os.makedirs(lora_path, exist_ok=True)
        self.get_model_raw().save_lora_weights(path=lora_path)

    # ------------------------------------------------------------------
    # load_checkpoint
    # ------------------------------------------------------------------
    def load_checkpoint(self, path: str):
        log_print(f"Loading checkpoint from {path}")
        weight_filename = os.path.basename(path)
        trainer_path = os.path.join(os.path.dirname(path), f"[trainer]{weight_filename}")
        lora_dir = os.path.join(
            os.path.dirname(path),
            weight_filename.replace(".pth", "_lora"),
        )

        if not os.path.exists(trainer_path):
            raise FileNotFoundError(f"Trainer checkpoint not found: {trainer_path}")

        checkpoint = torch.load(trainer_path, map_location=self.device)

        if os.path.isdir(lora_dir):
            self.get_model_raw().load_lora_weights(path=lora_dir)
        else:
            raise FileNotFoundError(
                f"LoRA directory not found for checkpoint: {lora_dir}. "
                "Refusing to continue with randomly initialized adapters."
            )

        loaded_epoch_idx = int(checkpoint.get("epoch_idx", -1))
        self.checkpoint_epoch_idx = loaded_epoch_idx if self.trainer_mode != "reTrain" else -1

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
            if prev_score is not None and score <= prev_score:
                continue
            self.best_metric_values[fn] = {
                "score": float(score),
                "epoch_idx": int(epoch_idx),
                "global_step": int(getattr(self, "global_step", 0)),
                "val_loss": float(val_loss_value) if val_loss_value is not None else None,
                "metrics": {k: float(v) for k, v in metric_dict.items() if self._finite_float(v) is not None},
            }
            improved.append(fn)

        for fn in improved:
            score = self.best_metric_values[fn]["score"]
            self.save_checkpoint(epoch_idx=epoch_idx, weight_filename=fn)
            log_print(
                f"New best metric checkpoint: {fn}, score={score:.6f}, epoch={epoch_idx}"
            )

        if improved:
            manifest_path = os.path.join(self.save_path, "best_metric_checkpoints.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.best_metric_values, f, ensure_ascii=False, indent=2, sort_keys=True)

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
        )

        cfg.total_steps = math.ceil(
            cfg.num_batchs_per_epoch / max(1, cfg.accumulation_steps)
        ) * cfg.num_epochs
        cfg.warmup_steps = int(
            min(cfg.warmup_ratio * cfg.total_steps, cfg.max_warmup_steps)
            if cfg.warmup_steps is None or cfg.warmup_steps > (cfg.warmup_ratio * cfg.total_steps)
            else cfg.warmup_steps
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

        if checkpoint_path:
            trainer.load_checkpoint(path=checkpoint_path)

        return trainer















