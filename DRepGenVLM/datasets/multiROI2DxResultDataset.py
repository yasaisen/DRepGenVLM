"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2026, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2607081520
"""


from typing import Any, List, Dict, Tuple, Optional
import os, json, random
from torch.utils.data import Dataset
from dataclasses import dataclass
from PIL import Image
import torchvision.transforms.functional as TF
from torchvision import transforms


from ..configs.DRGVLM_baseConfig import DRGVLM_baseConfig
from ..common.utils import log_print


@dataclass
class ROI:
    global_idx: int
    image: Optional[Image.Image]
    mpp: Optional[float]
    cxcywh: Optional[Tuple[float, float, float, float]]
    roi_wh: Tuple[float, float]


@dataclass(frozen=True)
class ROIRecord:
    """Lightweight ROI metadata used for sampling before image I/O."""
    global_idx: int
    roi_path: Optional[str]
    mpp: Optional[float]
    cxcywh: Optional[Tuple[float, float, float, float]]
    roi_wh: Tuple[float, float]


@dataclass
class Case:
    global_idx: int
    case_id: str
    # Unique union of the sampled ROIs.  This is retained for diagnostics and
    # resource accounting only; model inputs must use DxItem_rois.
    rois: List[ROI]
    # Active forward units.  A DxItem is active only when at least one ROI has
    # declared it in that ROI's DxPair keys.
    DxItem_rois: Dict[str, List[ROI]]
    # All report targets are retained even when a DxItem has no assigned ROI.
    # Some clinical references (for example Nottingham grade components) need
    # these authoritative values without requiring a separate forward.
    DxItem_targets: Dict[str, str]  # DxItem -> DxResultTxt
    DxItem_target_classes: Dict[str, str]  # DxItem -> authoritative DxResultCls


class RandomDiscreteRotation:
    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle, expand=True)


class multiROI2DxResultDataset(Dataset):
    def __init__(self,
        image_path: str, 
        metadata_path: str,
        split: str,
        input_img: bool = True,
        input_loc: bool = True,
        level_key: str = "main_info",
        max_rois_per_dxitem: Optional[int] = None,
        roi_sampling_mode: str = "all",
        valid_sampling_seed: int = 42,
        require_targets: bool = True,
        max_rois_per_case: Optional[int] = None,
    ):
        self.image_path = image_path
        self.metadata_path = metadata_path
        self.split = split
        self.input_img = input_img
        self.input_loc = input_loc
        self.level_key = level_key
        self.require_targets = bool(require_targets)
        if (
            max_rois_per_dxitem is not None
            and max_rois_per_case is not None
            and int(max_rois_per_dxitem) != int(max_rois_per_case)
        ):
            raise ValueError(
                "Conflicting ROI limits: max_rois_per_dxitem="
                f"{max_rois_per_dxitem} and legacy max_rois_per_case="
                f"{max_rois_per_case}."
            )
        if max_rois_per_dxitem is None:
            max_rois_per_dxitem = max_rois_per_case
        self.max_rois_per_dxitem = (
            int(max_rois_per_dxitem)
            if max_rois_per_dxitem is not None
            else None
        )
        self.roi_sampling_mode = roi_sampling_mode
        self.valid_sampling_seed = int(valid_sampling_seed)

        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)

        self.DxItem_list = metadata["DxItem_list"]
        self.case_list = metadata["case_list"]
        self._validate_metadata_relationships()
        self.invalid_reference_records = self._find_invalid_references()
        for record in self.invalid_reference_records:
            log_print(
                "[MetadataValidation][WARN] "
                f"split={self.split}, case_id={record['case_id']}, "
                f"DxItem={record['DxItem']}, value={record['value']!r}, "
                f"reason={record['reason']}"
            )

        if self.split == "train":
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                RandomDiscreteRotation([0, 90, 180, 270]),
            ])
        else:
            self.transform = transforms.Compose([])

    def __len__(self):
        return len(self.case_list)

    def _metadata_error(
        self,
        sample: Dict[str, Any],
        message: str,
    ) -> ValueError:
        return ValueError(
            "[MetadataValidation] "
            f"split={self.split}, sample_idx={sample.get('sample_idx')}, "
            f"case_id={sample.get('case_id')!r}: {message}"
        )

    def _validate_metadata_relationships(self):
        """Validate the DxItem universe, report targets, and ROI assignments."""
        if (
            not isinstance(self.DxItem_list, list)
            or not self.DxItem_list
            or not all(
                isinstance(dx_item, str) and dx_item
                for dx_item in self.DxItem_list
            )
        ):
            raise ValueError(
                "[MetadataValidation] DxItem_list must be a non-empty list "
                "of non-empty strings."
            )
        if len(set(self.DxItem_list)) != len(self.DxItem_list):
            raise ValueError(
                "[MetadataValidation] DxItem_list contains duplicate entries."
            )
        if not isinstance(self.case_list, list):
            raise ValueError("[MetadataValidation] case_list must be a list.")

        declared_dx_items = set(self.DxItem_list)
        for sample in self.case_list:
            if not isinstance(sample, dict):
                raise ValueError(
                    "[MetadataValidation] Every case_list entry must be an object."
                )
            structured_report = sample.get("structured_report")
            if structured_report is None and not self.require_targets:
                dx_samples = {}
            elif not isinstance(structured_report, dict):
                raise self._metadata_error(
                    sample,
                    "structured_report must be an object.",
                )
            else:
                dx_samples = structured_report.get("DxItems")
                if dx_samples is None and not self.require_targets:
                    dx_samples = {}
                elif not isinstance(dx_samples, dict):
                    raise self._metadata_error(
                        sample,
                        "structured_report.DxItems must be an object.",
                    )
            undeclared_targets = set(dx_samples) - declared_dx_items
            if undeclared_targets:
                raise self._metadata_error(
                    sample,
                    "structured_report.DxItems contains undeclared DxItems: "
                    f"{sorted(undeclared_targets)}.",
                )
            for dx_item, dx_sample in dx_samples.items():
                if not isinstance(dx_sample, dict):
                    raise self._metadata_error(
                        sample,
                        "structured_report.DxItems."
                        f"{dx_item} must be an object.",
                    )
                if self.require_targets:
                    missing_fields = [
                        field
                        for field in ("DxResultTxt", "DxResultCls")
                        if field not in dx_sample
                    ]
                    if missing_fields:
                        raise self._metadata_error(
                            sample,
                            "structured_report.DxItems."
                            f"{dx_item} is missing fields "
                            f"{missing_fields}.",
                        )

            active_dx_items = set()
            tissue_blocks = sample.get("tissue_blocks")
            if not isinstance(tissue_blocks, list):
                raise self._metadata_error(
                    sample,
                    "tissue_blocks must be a list.",
                )
            for block in tissue_blocks:
                stains = block.get("stains") if isinstance(block, dict) else None
                if not isinstance(stains, list):
                    raise self._metadata_error(
                        sample,
                        "Every tissue block must contain a stains list.",
                    )
                for stain in stains:
                    if not isinstance(stain, dict) or "roi_list" not in stain:
                        continue
                    roi_samples = stain.get("roi_list")
                    if not isinstance(roi_samples, list):
                        raise self._metadata_error(
                            sample,
                            "Every roi_list must be a list.",
                        )
                    for roi_sample in roi_samples:
                        if not isinstance(roi_sample, dict):
                            raise self._metadata_error(
                                sample,
                                "Every ROI must be an object.",
                            )
                        dx_pair = roi_sample.get("DxPair")
                        if dx_pair is None:
                            continue
                        if not isinstance(dx_pair, dict):
                            raise self._metadata_error(
                                sample,
                                "ROI "
                                f"global_idx={roi_sample.get('global_idx')} has "
                                f"non-object DxPair={dx_pair!r}.",
                            )
                        for dx_item in dx_pair:
                            if dx_item not in declared_dx_items:
                                raise self._metadata_error(
                                    sample,
                                    "ROI "
                                    f"global_idx={roi_sample.get('global_idx')} "
                                    f"references undeclared DxItem={dx_item!r}.",
                                )
                            if (
                                self.require_targets
                                and dx_item not in dx_samples
                            ):
                                raise self._metadata_error(
                                    sample,
                                    "ROI "
                                    f"global_idx={roi_sample.get('global_idx')} "
                                    f"references DxItem={dx_item!r}, but no "
                                    "structured_report target exists.",
                                )
                            active_dx_items.add(dx_item)

            if not active_dx_items:
                raise self._metadata_error(
                    sample,
                    "case has no active DxItem because no ROI DxPair contains "
                    "a valid DxItem key.",
                )

    def _find_invalid_references(self) -> List[Dict[str, Any]]:
        """Report obviously malformed targets without silently dropping samples."""
        invalid: List[Dict[str, Any]] = []
        for sample in self.case_list:
            structured_report = sample.get("structured_report")
            dx_items = (
                structured_report.get("DxItems", {})
                if isinstance(structured_report, dict)
                else {}
            )
            if not isinstance(dx_items, dict):
                dx_items = {}
            for dx_item, dx_sample in dx_items.items():
                if dx_item not in self.DxItem_list:
                    continue
                value = dx_sample.get("DxResultTxt", None)
                reason = None
                if not isinstance(value, str) or not value.strip():
                    reason = "empty or non-string reference"
                elif dx_item == "Histologic_Type":
                    compact = "".join(ch for ch in value.strip().lower() if ch.isalpha())
                    if len(compact) < 3:
                        reason = "histologic type is too short to be a valid label"
                if reason is not None:
                    invalid.append({
                        "sample_idx": sample.get("sample_idx"),
                        "case_id": str(sample.get("case_id", "")),
                        "DxItem": dx_item,
                        "value": value,
                        "reason": reason,
                    })
        return invalid

    def get_raw_roi_count(self,
        idx: int,
    ) -> int:
        """Return raw ROI-forward assignments, counting multi-DxItem ROIs once per key."""
        sample = self.case_list[idx]
        count = 0
        for block in sample["tissue_blocks"]:
            for stain in block["stains"]:
                if "roi_list" not in stain:
                    continue
                for roi_sample in stain["roi_list"]:
                    dx_pair = roi_sample.get("DxPair")
                    if isinstance(dx_pair, dict):
                        count += len(dx_pair)
        return count

    def get_effective_roi_count(self,
        idx: int,
    ) -> int:
        sample = self.case_list[idx]
        raw_counts = {dx_item: 0 for dx_item in self.DxItem_list}
        for block in sample["tissue_blocks"]:
            for stain in block["stains"]:
                if "roi_list" not in stain:
                    continue
                for roi_sample in stain["roi_list"]:
                    dx_pair = roi_sample.get("DxPair")
                    if not isinstance(dx_pair, dict):
                        continue
                    for dx_item in dx_pair:
                        raw_counts[dx_item] += 1

        if (
            self.max_rois_per_dxitem is None
            or self.max_rois_per_dxitem <= 0
            or self.roi_sampling_mode == "all"
        ):
            return sum(raw_counts.values())
        if self.roi_sampling_mode in {"random_k", "tail_k", "head_k"}:
            return sum(
                min(count, self.max_rois_per_dxitem)
                for count in raw_counts.values()
            )
        raise ValueError(
            f"Unsupported roi_sampling_mode='{self.roi_sampling_mode}'. "
            "Expected one of: all, random_k, tail_k, head_k."
        )

    def _sample_rois(self,
        roi_list: List[Any],
        rng=None,
    ) -> List[Any]:
        if self.max_rois_per_dxitem is None or self.max_rois_per_dxitem <= 0:
            return roi_list
        if len(roi_list) <= self.max_rois_per_dxitem:
            return roi_list

        mode = self.roi_sampling_mode
        if mode == "random_k":
            rng = random if rng is None else rng
            indices = sorted(
                rng.sample(
                    range(len(roi_list)),
                    self.max_rois_per_dxitem,
                )
            )
            return [roi_list[i] for i in indices]
        if mode == "tail_k":
            return roi_list[-self.max_rois_per_dxitem:]
        if mode == "head_k":
            return roi_list[:self.max_rois_per_dxitem]
        if mode == "all":
            return roi_list
        raise ValueError(
            f"Unsupported roi_sampling_mode='{mode}'. "
            "Expected one of: all, random_k, tail_k, head_k."
        )

    def __getitem__(self,
        idx: int,
    ) -> Case:
        sample = self.case_list[idx]

        # Group lightweight records by ROI.DxPair keys before sampling so every
        # DxItem receives an independent ROI cap.
        roi_records_by_dxitem: Dict[str, List[ROIRecord]] = {
            dx_item: []
            for dx_item in self.DxItem_list
        }
        for block in sample["tissue_blocks"]:
            for stain in block["stains"]:
                if "roi_list" not in stain:
                    continue
                for roi_sample in stain["roi_list"]:
                    level_info = roi_sample.get(self.level_key, {})
                    roi_path = level_info.get("roi_path", None)

                    mpp_raw = level_info.get("mpp", None)
                    if mpp_raw is not None and self.input_loc:
                        # mpp can be a list [x_mpp, y_mpp] or a float
                        if isinstance(mpp_raw, (list, tuple)):
                            mpp = float(mpp_raw[0])
                        else:
                            mpp = float(mpp_raw)
                    else:
                        mpp = None

                    cxcywh = tuple(level_info["cxcywh"]) if (self.input_loc and "cxcywh" in level_info and level_info["cxcywh"] is not None) else None
                    roi_wh = tuple(level_info["roi_wh"]) if ("roi_wh" in level_info and level_info["roi_wh"] is not None) else (0.0, 0.0)

                    record = ROIRecord(
                        global_idx=roi_sample["global_idx"],
                        roi_path=roi_path,
                        mpp=mpp,
                        cxcywh=cxcywh,
                        roi_wh=roi_wh,
                    )
                    dx_pair = roi_sample.get("DxPair")
                    if not isinstance(dx_pair, dict):
                        continue
                    for dx_item in dx_pair:
                        roi_records_by_dxitem[dx_item].append(record)

        sampled_records_by_dxitem: Dict[str, List[ROIRecord]] = {}
        for dx_item_idx, dx_item in enumerate(self.DxItem_list):
            records = roi_records_by_dxitem[dx_item]
            if not records:
                continue
            # Validation uses stable, independent per-(case, DxItem) RNGs so
            # each task's selected subset remains comparable across epochs.
            sampling_rng = None
            if self.split == "valid" and self.roi_sampling_mode == "random_k":
                sampling_rng = random.Random(
                    self.valid_sampling_seed
                    + int(idx) * max(1, len(self.DxItem_list))
                    + dx_item_idx
                )
            sampled_records_by_dxitem[dx_item] = self._sample_rois(
                records,
                rng=sampling_rng,
            )

        # Open and transform every selected physical ROI only once, even when
        # its DxPair assigns it to multiple DxItems.
        unique_records: Dict[int, ROIRecord] = {}
        for records in sampled_records_by_dxitem.values():
            for record in records:
                existing = unique_records.get(record.global_idx)
                if existing is not None and existing != record:
                    raise self._metadata_error(
                        sample,
                        "duplicate ROI global_idx="
                        f"{record.global_idx} has inconsistent level metadata.",
                    )
                unique_records.setdefault(record.global_idx, record)

        roi_by_global_idx: Dict[int, ROI] = {}
        roi_list: List[ROI] = []
        for record in unique_records.values():
            if self.input_img and record.roi_path is not None:
                with Image.open(os.path.join(self.image_path, record.roi_path)) as pil_image:
                    image = self.transform(pil_image.convert("RGB"))
            else:
                image = None
            roi = ROI(
                global_idx=record.global_idx,
                image=image,
                mpp=record.mpp,
                cxcywh=record.cxcywh,
                roi_wh=record.roi_wh,
            )
            roi_list.append(roi)
            roi_by_global_idx[record.global_idx] = roi

        DxItem_rois = {
            dx_item: [
                roi_by_global_idx[record.global_idx]
                for record in records
            ]
            for dx_item, records in sampled_records_by_dxitem.items()
        }

        # Keep the free-text target for SFT/text metrics and the class target
        # separately for clinical metrics.  In particular, the evaluator must
        # not re-infer a reference Nottingham grade from DxResultTxt.
        DxItem_targets = {}
        DxItem_target_classes = {}
        structured_report = sample.get("structured_report")
        report_dx_items = (
            structured_report.get("DxItems", {})
            if isinstance(structured_report, dict)
            else {}
        )
        if not isinstance(report_dx_items, dict):
            report_dx_items = {}
        for DxItem in self.DxItem_list:
            if DxItem not in report_dx_items:
                continue
            DxSample = report_dx_items[DxItem]
            if not isinstance(DxSample, dict):
                if self.require_targets:
                    raise self._metadata_error(
                        sample,
                        f"structured_report.DxItems.{DxItem} must be an object.",
                    )
                continue
            if "DxResultTxt" in DxSample:
                DxItem_targets[DxItem] = DxSample["DxResultTxt"]
            elif self.require_targets:
                raise self._metadata_error(
                    sample,
                    "structured_report.DxItems."
                    f"{DxItem}.DxResultTxt is missing.",
                )
            if "DxResultCls" in DxSample:
                DxItem_target_classes[DxItem] = DxSample["DxResultCls"]
            elif self.require_targets:
                raise self._metadata_error(
                    sample,
                    "structured_report.DxItems."
                    f"{DxItem}.DxResultCls is missing.",
                )

        case = Case(
            global_idx=sample["sample_idx"],
            case_id=str(sample["case_id"]),
            rois=roi_list,
            DxItem_rois=DxItem_rois,
            DxItem_targets=DxItem_targets,
            DxItem_target_classes=DxItem_target_classes,
        )
        return case

    def collate_cases(self,
        batch: List,
    ):
        return list(batch)

    @classmethod
    def from_config(cls,
        cfg: DRGVLM_baseConfig,
        split: str,
    ):
        assert split in ["train", "valid"], f"split must be 'train' or 'valid', got {split}"

        if split == "train":
            metadata_path = cfg.train_metadata_path
        else:
            metadata_path = cfg.valid_metadata_path

        new_roi_limit = getattr(cfg, "max_rois_per_dxitem", None)
        legacy_roi_limit = getattr(cfg, "max_rois_per_case", None)
        if (
            new_roi_limit is not None
            and legacy_roi_limit is not None
            and int(new_roi_limit) != int(legacy_roi_limit)
        ):
            raise ValueError(
                "Conflicting config values for max_rois_per_dxitem and legacy "
                "max_rois_per_case."
            )
        roi_limit = (
            new_roi_limit
            if new_roi_limit is not None
            else legacy_roi_limit
        )

        dataset = cls(
            image_path=cfg.image_path,
            metadata_path=metadata_path,
            split=split,
            input_img=getattr(cfg, "input_img", True),
            input_loc=getattr(cfg, "input_loc", True),
            level_key=getattr(cfg, "level_key", "main_info"),
            max_rois_per_dxitem=roi_limit,
            roi_sampling_mode=getattr(cfg, "roi_sampling_mode", "all"),
            valid_sampling_seed=getattr(cfg, "valid_sampling_seed", 42),
        )

        # Expose DxItem_list to config so model/trainer can access it
        cfg.DxItem_list = dataset.DxItem_list

        return dataset



















