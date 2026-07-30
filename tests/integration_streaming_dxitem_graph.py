"""Real-model diagnostic for aggregate versus streaming DxItem backward.

This is intentionally not part of the default unit-test suite.  It loads the
local MedGemma checkpoint and one validation case, then compares all trainable
LoRA gradients and CUDA peak allocation between:

1. aggregate: retain every DxItem graph and backward the mean once;
2. streaming: backward each scaled DxItem loss immediately.
"""

import gc
import json
import random
import sys
from copy import copy
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from DRepGenVLM.configs.DRGVLM_baseConfig import DRGVLM_baseConfig
from DRepGenVLM.datasets.multiROI2DxResultDataset import (
    multiROI2DxResultDataset,
)
from DRepGenVLM.models.modeling_medGemmaLoRA import DownstreamRepGenVLM
from DRepGenVLM.trainer.building_DRGVLM_PPTrainer import DRGVLM_PPTrainer


CONFIG_PATH = (
    PROJECT_ROOT
    / "DRepGenVLM"
    / "projects"
    / "MG15_basic_overfitTesting_config.json"
)
SEED = 1729
MAX_TEST_ROIS = 1
MAX_TEST_DX_ITEMS = 3


def _reset_rng():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def _reset_cuda_peaks():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baselines = []
    for device_idx in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_idx)
        baselines.append(torch.cuda.memory_allocated(device_idx))
    return baselines


def _memory_result(baselines):
    torch.cuda.synchronize()
    return [
        {
            "device": device_idx,
            "baseline_bytes": int(baselines[device_idx]),
            "peak_bytes": int(torch.cuda.max_memory_allocated(device_idx)),
            "peak_delta_bytes": int(
                torch.cuda.max_memory_allocated(device_idx)
                - baselines[device_idx]
            ),
        }
        for device_idx in range(torch.cuda.device_count())
    ]


def _gradient_snapshot(model):
    snapshot = {}
    missing = []
    nonfinite = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
            continue
        gradient = parameter.grad.detach().float().cpu().clone()
        if not torch.isfinite(gradient).all():
            nonfinite.append(name)
        snapshot[name] = gradient
    return snapshot, missing, nonfinite


def _aggregate_backward(model, case, amp, require_cached_input_grad):
    model.zero_grad(set_to_none=True)
    model.require_cached_input_grad = bool(require_cached_input_grad)
    _reset_rng()
    baselines = _reset_cuda_peaks()
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=amp,
        cache_enabled=False,
    ):
        loss_dict, _ = model.calculate_loss([case])
    loss = loss_dict["total_loss"]
    loss.backward()
    gradients, missing, nonfinite = _gradient_snapshot(model)
    result = {
        "loss": float(loss.detach()),
        "missing_gradients": missing,
        "nonfinite_gradients": nonfinite,
        "memory": _memory_result(baselines),
    }
    del loss, loss_dict
    return result, gradients


def _streaming_backward(model, case, amp, require_cached_input_grad):
    model.zero_grad(set_to_none=True)
    model.require_cached_input_grad = bool(require_cached_input_grad)
    _reset_rng()
    baselines = _reset_cuda_peaks()
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=amp,
        cache_enabled=False,
    ):
        case_contexts = model.prepare_case_loss_context(case)

    pair_losses = []
    active_dx_items = model._active_dxitems(case)
    dx_count = len(active_dx_items)
    for dx_item in active_dx_items:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=amp,
            cache_enabled=False,
        ):
            loss_dict = model.calculate_dxitem_loss(
                case=case,
                DxItem=dx_item,
                case_context=(
                    case_contexts.get(dx_item)
                    if case_contexts is not None
                    else None
                ),
            )
        pair_loss = loss_dict["total_loss"]
        pair_losses.append(float(pair_loss.detach()))
        (pair_loss / dx_count).backward()
        del pair_loss, loss_dict

    gradients, missing, nonfinite = _gradient_snapshot(model)
    result = {
        "loss": float(np.mean(pair_losses)),
        "pair_losses": pair_losses,
        "missing_gradients": missing,
        "nonfinite_gradients": nonfinite,
        "memory": _memory_result(baselines),
    }
    del case_contexts
    return result, gradients


def _trainer_streaming_backward(model, case, amp):
    model.zero_grad(set_to_none=True)
    model.require_cached_input_grad = False
    _reset_rng()
    baselines = _reset_cuda_peaks()
    trainer = DRGVLM_PPTrainer(
        model=model,
        device="cuda:0",
        num_epochs=1,
        save_path=str(PROJECT_ROOT),
        accumulation_steps=1,
        amp=amp,
    )
    loss_dict, dx_count = trainer._backward_train_batch(
        batch_cases=[case],
        accumulation_denom=1,
    )
    gradients, missing, nonfinite = _gradient_snapshot(model)
    result = {
        "loss": float(loss_dict["total_loss"]),
        "dx_count": int(dx_count),
        "missing_gradients": missing,
        "nonfinite_gradients": nonfinite,
        "memory": _memory_result(baselines),
    }
    return result, gradients


def _compare_gradients(reference, candidate):
    names = sorted(set(reference) | set(candidate))
    missing_in_reference = sorted(set(names) - set(reference))
    missing_in_candidate = sorted(set(names) - set(candidate))
    max_abs_difference = 0.0
    max_relative_difference = 0.0
    worst_parameter = None
    reference_squared = 0.0
    candidate_squared = 0.0
    difference_squared = 0.0
    dot_product = 0.0
    for name in names:
        if name not in reference or name not in candidate:
            continue
        reference_gradient = reference[name].double()
        candidate_gradient = candidate[name].double()
        difference = reference_gradient - candidate_gradient
        absolute = difference.abs()
        parameter_max_abs = float(absolute.max()) if absolute.numel() else 0.0
        denominator = reference_gradient.abs().clamp_min(1e-8)
        parameter_max_relative = (
            float((absolute / denominator).max())
            if absolute.numel()
            else 0.0
        )
        if parameter_max_abs > max_abs_difference:
            max_abs_difference = parameter_max_abs
            worst_parameter = name
        max_relative_difference = max(
            max_relative_difference,
            parameter_max_relative,
        )
        reference_squared += float(torch.sum(reference_gradient.square()))
        candidate_squared += float(torch.sum(candidate_gradient.square()))
        difference_squared += float(torch.sum(difference.square()))
        dot_product += float(torch.sum(reference_gradient * candidate_gradient))

    reference_l2 = reference_squared ** 0.5
    candidate_l2 = candidate_squared ** 0.5
    difference_l2 = difference_squared ** 0.5
    cosine_similarity = (
        dot_product / (reference_l2 * candidate_l2)
        if reference_l2 > 0.0 and candidate_l2 > 0.0
        else None
    )
    return {
        "tensor_count": len(names),
        "missing_in_reference": missing_in_reference,
        "missing_in_candidate": missing_in_candidate,
        "max_abs_difference": max_abs_difference,
        "max_elementwise_relative_difference": max_relative_difference,
        "worst_parameter": worst_parameter,
        "reference_l2": reference_l2,
        "candidate_l2": candidate_l2,
        "difference_l2": difference_l2,
        "relative_l2_difference": (
            difference_l2 / reference_l2
            if reference_l2 > 0.0
            else None
        ),
        "cosine_similarity": cosine_similarity,
    }


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this integration diagnostic.")

    cfg = DRGVLM_baseConfig.load(str(CONFIG_PATH))
    dataset = multiROI2DxResultDataset.from_config(cfg=cfg, split="valid")
    full_case = dataset[0]
    case = copy(full_case)
    selected_dx_items = list(case.DxItem_rois)[:MAX_TEST_DX_ITEMS]
    case.DxItem_rois = {
        dx_item: case.DxItem_rois[dx_item][:MAX_TEST_ROIS]
        for dx_item in selected_dx_items
    }
    case.rois = list({
        roi.global_idx: roi
        for rois in case.DxItem_rois.values()
        for roi in rois
    }.values())
    model = DownstreamRepGenVLM.from_config(cfg=cfg)
    model.train()

    aggregate_result, aggregate_gradients = _aggregate_backward(
        model=model,
        case=case,
        amp=bool(cfg.amp),
        require_cached_input_grad=True,
    )
    streaming_leaf_result, streaming_leaf_gradients = _streaming_backward(
        model=model,
        case=case,
        amp=bool(cfg.amp),
        require_cached_input_grad=True,
    )
    streaming_no_leaf_result, streaming_no_leaf_gradients = _streaming_backward(
        model=model,
        case=case,
        amp=bool(cfg.amp),
        require_cached_input_grad=False,
    )
    single_dx_case = copy(case)
    single_dx_item = next(iter(case.DxItem_rois))
    single_dx_case.DxItem_rois = {
        single_dx_item: case.DxItem_rois[single_dx_item],
    }
    single_dx_case.rois = list(case.DxItem_rois[single_dx_item])
    single_leaf_result, single_leaf_gradients = _streaming_backward(
        model=model,
        case=single_dx_case,
        amp=bool(cfg.amp),
        require_cached_input_grad=True,
    )
    single_no_leaf_result, single_no_leaf_gradients = _streaming_backward(
        model=model,
        case=single_dx_case,
        amp=bool(cfg.amp),
        require_cached_input_grad=False,
    )
    single_leaf_repeat_result, single_leaf_repeat_gradients = (
        _streaming_backward(
            model=model,
            case=single_dx_case,
            amp=bool(cfg.amp),
            require_cached_input_grad=True,
        )
    )
    single_no_leaf_repeat_result, single_no_leaf_repeat_gradients = (
        _streaming_backward(
            model=model,
            case=single_dx_case,
            amp=bool(cfg.amp),
            require_cached_input_grad=False,
        )
    )
    full_case_streaming_result, _ = _trainer_streaming_backward(
        model=model,
        case=full_case,
        amp=bool(cfg.amp),
    )

    result = {
        "case_id": case.case_id,
        "roi_count": len(case.rois),
        "dx_items": list(case.DxItem_rois),
        "aggregate": aggregate_result,
        "streaming_with_input_leaf": streaming_leaf_result,
        "streaming_without_input_leaf": streaming_no_leaf_result,
        "aggregate_vs_streaming_leaf": _compare_gradients(
            aggregate_gradients,
            streaming_leaf_gradients,
        ),
        "streaming_leaf_vs_no_leaf": _compare_gradients(
            streaming_leaf_gradients,
            streaming_no_leaf_gradients,
        ),
        "single_dx_with_input_leaf": single_leaf_result,
        "single_dx_without_input_leaf": single_no_leaf_result,
        "single_dx_leaf_vs_no_leaf": _compare_gradients(
            single_leaf_gradients,
            single_no_leaf_gradients,
        ),
        "single_dx_leaf_repeat": single_leaf_repeat_result,
        "single_dx_leaf_vs_leaf_repeat": _compare_gradients(
            single_leaf_gradients,
            single_leaf_repeat_gradients,
        ),
        "single_dx_no_leaf_repeat": single_no_leaf_repeat_result,
        "single_dx_no_leaf_vs_no_leaf_repeat": _compare_gradients(
            single_no_leaf_gradients,
            single_no_leaf_repeat_gradients,
        ),
        "full_case_streaming_without_input_leaf": {
            "roi_count": len(full_case.rois),
            "dx_items": list(full_case.DxItem_rois),
            **full_case_streaming_result,
        },
    }
    print("STREAMING_DXITEM_DIAGNOSTIC=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
