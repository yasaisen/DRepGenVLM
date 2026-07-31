"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2026, yasaisen (clover)

 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.

 last modified in 2607081524
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
import json
import os
import tempfile
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import torch

from ..common.utils import log_print, set_seed
from ..configs.DRGVLM_baseConfig import DRGVLM_baseConfig
from ..datasets.multiROI2DxResultDataset import multiROI2DxResultDataset


GENERATED_REPORT_KEY = "generated_report"
SUPPORTED_ROI_SAMPLING_MODES = ("all", "random_k", "tail_k", "head_k")


@dataclass(frozen=True)
class PipelineArtifacts:
    config_path: str
    checkpoint_path: str
    trainer_checkpoint_path: str
    adapter_path: str
    checkpoint_root: str
    checkpoint_epoch: Optional[int]


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object at {path}, got {type(data).__name__}."
        )
    return data


def _atomic_dump_json(data: Mapping[str, Any], path: str) -> None:
    output_path = os.path.abspath(path)
    output_dir = os.path.dirname(output_path)
    if not os.path.isdir(output_dir):
        raise FileNotFoundError(
            f"Output directory does not exist: {output_dir}"
        )

    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(output_path)}.",
        suffix=".tmp",
        dir=output_dir,
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, output_path)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_snapshot(snapshot_path: str) -> Dict[str, Any]:
    manifest_path = os.path.join(snapshot_path, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Checkpoint snapshot manifest not found: {manifest_path}"
        )
    manifest = _load_json(manifest_path)
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError(
            f"Checkpoint snapshot has no file manifest: {manifest_path}"
        )
    for relative_path, record in files.items():
        if not isinstance(relative_path, str) or not isinstance(record, Mapping):
            raise ValueError(
                f"Malformed checkpoint file record in {manifest_path}."
            )
        absolute_path = os.path.join(snapshot_path, relative_path)
        if not os.path.isfile(absolute_path):
            raise FileNotFoundError(
                f"Checkpoint artifact is missing: {absolute_path}"
            )
        expected_hash = record.get("sha256")
        if not isinstance(expected_hash, str) or not expected_hash:
            raise ValueError(
                f"Checkpoint artifact has no sha256: {relative_path}"
            )
        actual_hash = _sha256_file(absolute_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                "Checkpoint integrity verification failed for "
                f"{absolute_path}: expected={expected_hash}, "
                f"actual={actual_hash}"
            )
    return manifest


def _checkpoint_reference_name(weight_filename: str) -> str:
    stem = os.path.splitext(os.path.basename(weight_filename))[0]
    if stem == "best_model":
        return "best"
    if stem == "latest_model":
        return "latest"
    return stem


def _resolve_checkpoint_request(
    requested_path: str,
) -> Tuple[str, str, str, Optional[int], str]:
    requested_path = os.path.abspath(requested_path)

    if os.path.isfile(requested_path) and requested_path.endswith(".json"):
        reference = _load_json(requested_path)
        snapshot_relative = reference.get("snapshot")
        if not isinstance(snapshot_relative, str) or not snapshot_relative:
            raise ValueError(
                f"Not a DRGVLM checkpoint reference: {requested_path}"
            )
        reference_dir = os.path.dirname(requested_path)
        checkpoint_root = (
            os.path.dirname(reference_dir)
            if os.path.basename(reference_dir) == "checkpoint_refs"
            else reference_dir
        )
        snapshot_path = os.path.abspath(
            os.path.join(checkpoint_root, snapshot_relative)
        )
        manifest = _verify_snapshot(snapshot_path)
        trainer_path = os.path.join(snapshot_path, "trainer.pth")
        adapter_path = os.path.join(snapshot_path, "adapter")
        if not os.path.isfile(trainer_path):
            raise FileNotFoundError(
                f"Trainer checkpoint not found: {trainer_path}"
            )
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(
                f"LoRA adapter directory not found: {adapter_path}"
            )
        epoch = manifest.get("epoch_idx", reference.get("epoch_idx"))
        return (
            trainer_path,
            adapter_path,
            checkpoint_root,
            int(epoch) if epoch is not None else None,
            requested_path,
        )

    if os.path.isdir(requested_path):
        if not os.path.isfile(os.path.join(requested_path, "manifest.json")):
            raise ValueError(
                "Checkpoint directory must be an immutable snapshot "
                f"containing manifest.json, got: {requested_path}"
            )
        manifest = _verify_snapshot(requested_path)
        trainer_path = os.path.join(requested_path, "trainer.pth")
        adapter_path = os.path.join(requested_path, "adapter")
        if not os.path.isfile(trainer_path):
            raise FileNotFoundError(
                f"Trainer checkpoint not found: {trainer_path}"
            )
        if not os.path.isdir(adapter_path):
            raise FileNotFoundError(
                f"LoRA adapter directory not found: {adapter_path}"
            )
        checkpoint_root = os.path.dirname(os.path.dirname(requested_path))
        epoch = manifest.get("epoch_idx")
        return (
            trainer_path,
            adapter_path,
            checkpoint_root,
            int(epoch) if epoch is not None else None,
            requested_path,
        )

    weight_filename = os.path.basename(requested_path)
    checkpoint_root = os.path.dirname(requested_path)
    reference_path = os.path.join(
        checkpoint_root,
        "checkpoint_refs",
        f"{_checkpoint_reference_name(weight_filename)}.json",
    )
    if os.path.isfile(reference_path):
        return _resolve_checkpoint_request(reference_path)

    trainer_path = os.path.join(
        checkpoint_root,
        f"[trainer]{weight_filename}",
    )
    adapter_path = os.path.join(
        checkpoint_root,
        weight_filename.replace(".pth", "_lora"),
    )
    if not os.path.isfile(trainer_path):
        raise FileNotFoundError(
            "Neither a checkpoint reference nor a legacy trainer checkpoint "
            f"was found for: {requested_path}"
        )
    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(
            f"LoRA adapter directory not found: {adapter_path}"
        )
    return (
        trainer_path,
        adapter_path,
        checkpoint_root,
        None,
        requested_path,
    )


def resolve_checkpoint_artifacts(
    checkpoint_dir: str,
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> PipelineArtifacts:
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(
            f"Checkpoint directory not found: {checkpoint_dir}"
        )

    resolved_config_path = os.path.abspath(
        config_path or os.path.join(checkpoint_dir, "config.json")
    )
    if not os.path.isfile(resolved_config_path):
        raise FileNotFoundError(
            f"DRGVLM config not found: {resolved_config_path}"
        )

    if checkpoint_path is None:
        best_reference = os.path.join(
            checkpoint_dir,
            "checkpoint_refs",
            "best.json",
        )
        checkpoint_path = (
            best_reference
            if os.path.isfile(best_reference)
            else os.path.join(checkpoint_dir, "best_model.pth")
        )
    (
        trainer_path,
        adapter_path,
        checkpoint_root,
        checkpoint_epoch,
        resolved_checkpoint_path,
    ) = _resolve_checkpoint_request(checkpoint_path)
    return PipelineArtifacts(
        config_path=resolved_config_path,
        checkpoint_path=resolved_checkpoint_path,
        trainer_checkpoint_path=trainer_path,
        adapter_path=adapter_path,
        checkpoint_root=checkpoint_root,
        checkpoint_epoch=checkpoint_epoch,
    )


def _case_active_dxitems(
    case: Mapping[str, Any],
    declared_dxitems: Sequence[str],
) -> List[str]:
    active: Set[str] = set()
    for block in case.get("tissue_blocks", []):
        if not isinstance(block, Mapping):
            continue
        for stain in block.get("stains", []):
            if not isinstance(stain, Mapping):
                continue
            for roi in stain.get("roi_list", []):
                if not isinstance(roi, Mapping):
                    continue
                dx_pair = roi.get("DxPair")
                if isinstance(dx_pair, Mapping):
                    active.update(str(dx_item) for dx_item in dx_pair)
    return [
        dx_item
        for dx_item in declared_dxitems
        if dx_item in active
    ]


def _validate_case_identities(metadata: Mapping[str, Any]) -> None:
    seen_sample_indices = set()
    for case_index, case in enumerate(metadata.get("case_list", [])):
        if not isinstance(case, Mapping):
            raise ValueError(
                f"case_list[{case_index}] must be an object."
            )
        if "sample_idx" not in case:
            raise ValueError(
                f"case_list[{case_index}] is missing sample_idx."
            )
        sample_idx = case["sample_idx"]
        try:
            duplicate = sample_idx in seen_sample_indices
        except TypeError as error:
            raise ValueError(
                f"case_list[{case_index}].sample_idx must be hashable."
            ) from error
        if duplicate:
            raise ValueError(
                f"Metadata contains duplicate sample_idx={sample_idx!r}."
            )
        seen_sample_indices.add(sample_idx)
        if "case_id" not in case:
            raise ValueError(
                f"case sample_idx={sample_idx!r} is missing case_id."
            )


def _validate_generated_report_collisions(
    metadata: Mapping[str, Any],
    overwrite: bool,
) -> None:
    if overwrite:
        return
    for case in metadata.get("case_list", []):
        existing = case.get(GENERATED_REPORT_KEY)
        if existing:
            raise FileExistsError(
                "Case "
                f"sample_idx={case.get('sample_idx')!r} already contains a "
                f"non-empty {GENERATED_REPORT_KEY}. Pass --overwrite to "
                "replace generated reports."
            )


def apply_generated_reports(
    metadata: Dict[str, Any],
    predictions: Mapping[Any, Mapping[str, Mapping[str, Any]]],
    overwrite: bool = False,
) -> int:
    declared_dxitems = metadata.get("DxItem_list")
    if not isinstance(declared_dxitems, list):
        raise ValueError("Metadata is missing top-level DxItem_list.")
    _validate_case_identities(metadata)
    _validate_generated_report_collisions(metadata, overwrite=overwrite)

    applied_sample_indices = set()
    generated_count = 0
    for case in metadata.get("case_list", []):
        sample_idx = case["sample_idx"]
        expected_dxitems = _case_active_dxitems(
            case,
            declared_dxitems=declared_dxitems,
        )
        case_predictions = predictions.get(sample_idx)
        if not isinstance(case_predictions, Mapping):
            raise ValueError(
                f"Missing predictions for sample_idx={sample_idx!r}."
            )
        unexpected = set(case_predictions) - set(expected_dxitems)
        missing = set(expected_dxitems) - set(case_predictions)
        if unexpected or missing:
            raise ValueError(
                "Prediction/DxPair mismatch for "
                f"sample_idx={sample_idx!r}: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}."
            )

        generated_report: Dict[str, str] = {}
        for dx_item in expected_dxitems:
            prediction = case_predictions[dx_item]
            if not isinstance(prediction, Mapping):
                raise ValueError(
                    f"Prediction for sample_idx={sample_idx!r}, "
                    f"DxItem={dx_item} must be an object."
                )
            pred_txt = prediction.get("pred_txt")
            if not isinstance(pred_txt, str) or not pred_txt.strip():
                raise ValueError(
                    f"Empty prediction for sample_idx={sample_idx!r}, "
                    f"DxItem={dx_item}."
                )
            generated_report[dx_item] = pred_txt.strip()
            generated_count += 1

        case[GENERATED_REPORT_KEY] = generated_report
        applied_sample_indices.add(sample_idx)

    unexpected_samples = set(predictions) - applied_sample_indices
    if unexpected_samples:
        raise ValueError(
            "Predictions contain unknown sample_idx values: "
            f"{sorted(unexpected_samples, key=str)}."
        )
    return generated_count


class DRGVLMInferencePipeline:
    def __init__(
        self,
        checkpoint_dir: str,
        input_metadata_path: str,
        output_metadata_path: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        overwrite: bool = False,
        image_path: Optional[str] = None,
        weight_path: Optional[str] = None,
        pp_num_gpus: Optional[int] = None,
        device: Optional[str] = None,
        max_rois_per_dxitem: Optional[int] = None,
        roi_sampling_mode: Optional[str] = None,
        valid_sampling_seed: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        eval_prompt_batch_size: Optional[int] = None,
        amp: Optional[bool] = None,
    ):
        self.checkpoint_dir = os.path.abspath(checkpoint_dir)
        self.input_metadata_path = os.path.abspath(input_metadata_path)
        self.output_metadata_path = (
            os.path.abspath(output_metadata_path)
            if output_metadata_path
            else None
        )
        self.overwrite = bool(overwrite)
        self.image_path = image_path
        self.weight_path = weight_path
        self.pp_num_gpus = pp_num_gpus
        self.device = device
        self.max_rois_per_dxitem = max_rois_per_dxitem
        self.roi_sampling_mode = roi_sampling_mode
        self.valid_sampling_seed = valid_sampling_seed
        self.max_new_tokens = max_new_tokens
        self.eval_prompt_batch_size = eval_prompt_batch_size
        self.amp = amp

        self.artifacts = resolve_checkpoint_artifacts(
            checkpoint_dir=self.checkpoint_dir,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
        )
        self.metadata: Optional[Dict[str, Any]] = None
        self.cfg: Optional[DRGVLM_baseConfig] = None
        self.dataset: Optional[multiROI2DxResultDataset] = None

    def _load_and_localize_config(self) -> DRGVLM_baseConfig:
        cfg = DRGVLM_baseConfig.load(self.artifacts.config_path)
        local_root_pairs = (
            ("image_path", "localGPU_image_path"),
            ("weight_path", "localGPU_weight_path"),
            ("metadata_path", "localGPU_metadata_path"),
            ("root_path", "localGPU_root_path"),
        )
        for target_attr, local_attr in local_root_pairs:
            local_value = getattr(cfg, local_attr, None)
            if local_value and os.path.exists(local_value):
                setattr(cfg, target_attr, local_value)

        if self.image_path is not None:
            cfg.image_path = os.path.abspath(self.image_path)
        if self.weight_path is not None:
            cfg.weight_path = os.path.abspath(self.weight_path)
        if self.pp_num_gpus is not None:
            if int(self.pp_num_gpus) <= 0:
                raise ValueError("--pp-num-gpus must be positive.")
            cfg.pp_num_gpus = int(self.pp_num_gpus)
        if self.device is not None:
            cfg.device = self.device

        saved_roi_limit = getattr(cfg, "max_rois_per_dxitem", None)
        if saved_roi_limit is None:
            saved_roi_limit = getattr(cfg, "max_rois_per_case", None)
        cfg.max_rois_per_dxitem = (
            self.max_rois_per_dxitem
            if self.max_rois_per_dxitem is not None
            else saved_roi_limit
        )
        if self.roi_sampling_mode is not None:
            cfg.roi_sampling_mode = self.roi_sampling_mode
        elif getattr(cfg, "roi_sampling_mode", None) is None:
            cfg.roi_sampling_mode = "all"
        if cfg.roi_sampling_mode not in SUPPORTED_ROI_SAMPLING_MODES:
            raise ValueError(
                f"Unsupported roi_sampling_mode={cfg.roi_sampling_mode!r}; "
                f"expected one of {SUPPORTED_ROI_SAMPLING_MODES}."
            )
        if self.valid_sampling_seed is not None:
            cfg.valid_sampling_seed = int(self.valid_sampling_seed)
        elif getattr(cfg, "valid_sampling_seed", None) is None:
            cfg.valid_sampling_seed = 42

        if self.max_new_tokens is not None:
            if int(self.max_new_tokens) <= 0:
                raise ValueError("--max-new-tokens must be positive.")
            cfg.max_new_tokens = int(self.max_new_tokens)
        elif getattr(cfg, "max_new_tokens", None) is None:
            cfg.max_new_tokens = 256
        if self.eval_prompt_batch_size is not None:
            if int(self.eval_prompt_batch_size) <= 0:
                raise ValueError(
                    "--eval-prompt-batch-size must be positive."
                )
            cfg.eval_prompt_batch_size = int(self.eval_prompt_batch_size)
        elif getattr(cfg, "eval_prompt_batch_size", None) is None:
            cfg.eval_prompt_batch_size = 6
        if self.amp is not None:
            cfg.amp = bool(self.amp)
        elif not hasattr(cfg, "amp"):
            cfg.amp = True
        return cfg

    def preflight(self) -> Dict[str, Any]:
        if not os.path.isfile(self.input_metadata_path):
            raise FileNotFoundError(
                f"Input metadata not found: {self.input_metadata_path}"
            )
        if (
            self.output_metadata_path
            and self.output_metadata_path == self.input_metadata_path
        ):
            raise ValueError(
                "Input and output metadata paths must be different."
            )
        if (
            self.output_metadata_path
            and os.path.exists(self.output_metadata_path)
            and not self.overwrite
        ):
            raise FileExistsError(
                f"Output metadata already exists: "
                f"{self.output_metadata_path}. Pass --overwrite to replace it."
            )
        if self.output_metadata_path:
            output_dir = os.path.dirname(self.output_metadata_path)
            if not os.path.isdir(output_dir):
                raise FileNotFoundError(
                    f"Output directory does not exist: {output_dir}"
                )

        self.metadata = _load_json(self.input_metadata_path)
        _validate_case_identities(self.metadata)
        _validate_generated_report_collisions(
            self.metadata,
            overwrite=self.overwrite,
        )
        self.cfg = self._load_and_localize_config()

        metadata_dxitems = self.metadata.get("DxItem_list")
        if not isinstance(metadata_dxitems, list):
            raise ValueError("Metadata is missing top-level DxItem_list.")
        config_dxitems = getattr(self.cfg, "DxItem_list", None)
        if config_dxitems:
            if list(config_dxitems) != metadata_dxitems:
                raise ValueError(
                    "Config/metadata DxItem_list mismatch: "
                    f"config={list(config_dxitems)}, "
                    f"metadata={metadata_dxitems}"
                )
        else:
            self.cfg.DxItem_list = list(metadata_dxitems)

        required_runtime_paths = {
            "weight_path": getattr(self.cfg, "weight_path", None),
        }
        if bool(getattr(self.cfg, "input_img", True)):
            required_runtime_paths["image_path"] = getattr(
                self.cfg,
                "image_path",
                None,
            )
        missing_runtime_paths = {
            name: path
            for name, path in required_runtime_paths.items()
            if not isinstance(path, str) or not os.path.exists(path)
        }
        if missing_runtime_paths:
            raise FileNotFoundError(
                f"Missing localized runtime paths: {missing_runtime_paths}"
            )

        self.dataset = multiROI2DxResultDataset(
            image_path=self.cfg.image_path,
            metadata_path=self.input_metadata_path,
            split="valid",
            input_img=bool(getattr(self.cfg, "input_img", True)),
            input_loc=bool(getattr(self.cfg, "input_loc", True)),
            level_key=getattr(self.cfg, "level_key", "main_info"),
            max_rois_per_dxitem=getattr(
                self.cfg,
                "max_rois_per_dxitem",
                None,
            ),
            roi_sampling_mode=self.cfg.roi_sampling_mode,
            valid_sampling_seed=int(self.cfg.valid_sampling_seed),
            require_targets=False,
        )
        self.cfg.DxItem_list = list(self.dataset.DxItem_list)

        active_pair_count = sum(
            len(_case_active_dxitems(case, metadata_dxitems))
            for case in self.metadata["case_list"]
        )
        raw_roi_assignment_count = sum(
            self.dataset.get_raw_roi_count(index)
            for index in range(len(self.dataset))
        )
        effective_roi_assignment_count = sum(
            self.dataset.get_effective_roi_count(index)
            for index in range(len(self.dataset))
        )
        return {
            "case_count": len(self.dataset),
            "DxItem_list": list(self.dataset.DxItem_list),
            "active_dxitem_pair_count": active_pair_count,
            "raw_roi_assignment_count": raw_roi_assignment_count,
            "effective_roi_assignment_count": (
                effective_roi_assignment_count
            ),
            "config_path": self.artifacts.config_path,
            "checkpoint_path": self.artifacts.checkpoint_path,
            "trainer_checkpoint_path": (
                self.artifacts.trainer_checkpoint_path
            ),
            "adapter_path": self.artifacts.adapter_path,
            "checkpoint_epoch": self.artifacts.checkpoint_epoch,
            "sampling": {
                "max_rois_per_dxitem": getattr(
                    self.cfg,
                    "max_rois_per_dxitem",
                    None,
                ),
                "roi_sampling_mode": self.cfg.roi_sampling_mode,
                "valid_sampling_seed": int(
                    self.cfg.valid_sampling_seed
                ),
            },
            "generation": {
                "max_new_tokens": int(self.cfg.max_new_tokens),
                "eval_prompt_batch_size": int(
                    self.cfg.eval_prompt_batch_size
                ),
                "do_sample": False,
            },
            "pp_num_gpus": getattr(self.cfg, "pp_num_gpus", None),
            "device": getattr(self.cfg, "device", None),
            "structured_report_required": False,
        }

    def _build_model(self):
        if self.cfg is None:
            raise RuntimeError("Call preflight() before building the model.")
        device_type = torch.device(self.cfg.device).type
        if device_type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "The localized config requests CUDA, but CUDA is unavailable."
            )
        requested_gpus = getattr(self.cfg, "pp_num_gpus", None)
        if (
            device_type == "cuda"
            and requested_gpus is not None
            and int(requested_gpus) > torch.cuda.device_count()
        ):
            raise RuntimeError(
                f"Config requests {requested_gpus} GPUs, but only "
                f"{torch.cuda.device_count()} are visible."
            )

        from ..models.modeling_medGemmaLoRA import DownstreamRepGenVLM

        model = DownstreamRepGenVLM.from_config(
            cfg=self.cfg,
            load_criterion=False,
        )
        model.load_lora_weights(path=self.artifacts.adapter_path)
        model.eval()
        return model

    def _predict(self) -> Dict[Any, Dict[str, Dict[str, str]]]:
        if self.cfg is None or self.dataset is None:
            raise RuntimeError("Call preflight() before prediction.")
        model = self._build_model()
        predictions: Dict[Any, Dict[str, Dict[str, str]]] = {}
        device_type = torch.device(self.cfg.device).type
        use_autocast = bool(self.cfg.amp) and device_type == "cuda"

        with torch.inference_mode():
            for dataset_idx in range(len(self.dataset)):
                case = self.dataset[dataset_idx]
                if case.global_idx in predictions:
                    raise ValueError(
                        f"Duplicate generated sample_idx={case.global_idx!r}."
                    )
                log_print(
                    f"Generating case {dataset_idx + 1}/{len(self.dataset)}: "
                    f"sample_idx={case.global_idx!r}, "
                    f"case_id={case.case_id!r}, "
                    f"DxItems={list(case.DxItem_rois)}"
                )
                autocast_context = (
                    torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                    )
                    if use_autocast
                    else nullcontext()
                )
                with autocast_context:
                    case_context = model.prepare_case_loss_context(case)
                    output = model.generate_outputs(
                        batch_cases=[case],
                        max_new_tokens=int(self.cfg.max_new_tokens),
                        case_contexts={
                            case.global_idx: case_context,
                        },
                    )
                case_output = output.get(case.global_idx)
                if not isinstance(case_output, dict):
                    raise RuntimeError(
                        "Model returned no case output for "
                        f"sample_idx={case.global_idx!r}."
                    )
                predictions[case.global_idx] = case_output
                del case_context, output, case
        return predictions

    def run(self) -> Dict[str, Any]:
        preflight_summary = self.preflight()
        if self.output_metadata_path is None:
            raise ValueError(
                "output_metadata_path is required unless only preflight() "
                "is used."
            )
        set_seed(int(self.cfg.valid_sampling_seed))
        predictions = self._predict()
        generated_count = apply_generated_reports(
            metadata=self.metadata,
            predictions=predictions,
            overwrite=self.overwrite,
        )
        expected_count = int(
            preflight_summary["active_dxitem_pair_count"]
        )
        if generated_count != expected_count:
            raise RuntimeError(
                "Generated report count mismatch: "
                f"generated={generated_count}, expected={expected_count}."
            )
        _atomic_dump_json(self.metadata, self.output_metadata_path)
        return {
            **preflight_summary,
            "generated_report_count": generated_count,
            "output_metadata_path": self.output_metadata_path,
        }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate case-level DRGVLM reports for ROI DxPair metadata."
        )
    )
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--input-metadata", required=True)
    parser.add_argument("--output-metadata", default=None)
    parser.add_argument("--config-path", default=None)
    parser.add_argument(
        "--checkpoint-path",
        default=None,
        help=(
            "Checkpoint reference, immutable snapshot, or legacy checkpoint "
            "request. Defaults to checkpoint_refs/best.json."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--weight-path", default=None)
    parser.add_argument("--pp-num-gpus", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-rois-per-dxitem", type=int, default=None)
    parser.add_argument(
        "--roi-sampling-mode",
        choices=SUPPORTED_ROI_SAMPLING_MODES,
        default=None,
    )
    parser.add_argument("--valid-sampling-seed", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--eval-prompt-batch-size", type=int, default=None)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if not args.preflight_only and args.output_metadata is None:
        raise ValueError(
            "--output-metadata is required unless --preflight-only is set."
        )
    pipeline = DRGVLMInferencePipeline(
        checkpoint_dir=args.checkpoint_dir,
        input_metadata_path=args.input_metadata,
        output_metadata_path=args.output_metadata,
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        overwrite=args.overwrite,
        image_path=args.image_path,
        weight_path=args.weight_path,
        pp_num_gpus=args.pp_num_gpus,
        device=args.device,
        max_rois_per_dxitem=args.max_rois_per_dxitem,
        roi_sampling_mode=args.roi_sampling_mode,
        valid_sampling_seed=args.valid_sampling_seed,
        max_new_tokens=args.max_new_tokens,
        eval_prompt_batch_size=args.eval_prompt_batch_size,
        amp=args.amp,
    )
    summary = (
        pipeline.preflight()
        if args.preflight_only
        else pipeline.run()
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
