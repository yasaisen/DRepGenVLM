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
    rois: List[ROI]
    DxItem_targets: Dict[str, str]  # DxItem -> DxResultTxt (case-level ground truth)
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
        max_rois_per_case: Optional[int] = None,
        roi_sampling_mode: str = "all",
        valid_sampling_seed: int = 42,
    ):
        self.image_path = image_path
        self.metadata_path = metadata_path
        self.split = split
        self.input_img = input_img
        self.input_loc = input_loc
        self.level_key = level_key
        self.max_rois_per_case = int(max_rois_per_case) if max_rois_per_case is not None else None
        self.roi_sampling_mode = roi_sampling_mode
        self.valid_sampling_seed = int(valid_sampling_seed)

        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)

        self.DxItem_list = metadata["DxItem_list"]
        self.case_list = metadata["case_list"]
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

    def _find_invalid_references(self) -> List[Dict[str, Any]]:
        """Report obviously malformed targets without silently dropping samples."""
        invalid: List[Dict[str, Any]] = []
        for sample in self.case_list:
            dx_items = sample.get("structured_report", {}).get("DxItems", {})
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
        sample = self.case_list[idx]
        count = 0
        for block in sample["tissue_blocks"]:
            for stain in block["stains"]:
                if "roi_list" not in stain:
                    continue
                count += len(stain["roi_list"])
        return count

    def get_effective_roi_count(self,
        idx: int,
    ) -> int:
        raw_count = self.get_raw_roi_count(idx=idx)
        if self.max_rois_per_case is None or self.max_rois_per_case <= 0:
            return raw_count
        if self.roi_sampling_mode == "all":
            return raw_count
        if self.roi_sampling_mode in {"random_k", "tail_k", "head_k"}:
            return min(raw_count, self.max_rois_per_case)
        raise ValueError(
            f"Unsupported roi_sampling_mode='{self.roi_sampling_mode}'. "
            "Expected one of: all, random_k, tail_k, head_k."
        )

    def _sample_rois(self,
        roi_list: List[Any],
        rng=None,
    ) -> List[Any]:
        if self.max_rois_per_case is None or self.max_rois_per_case <= 0:
            return roi_list
        if len(roi_list) <= self.max_rois_per_case:
            return roi_list

        mode = self.roi_sampling_mode
        if mode == "random_k":
            rng = random if rng is None else rng
            indices = sorted(rng.sample(range(len(roi_list)), self.max_rois_per_case))
            return [roi_list[i] for i in indices]
        if mode == "tail_k":
            return roi_list[-self.max_rois_per_case:]
        if mode == "head_k":
            return roi_list[:self.max_rois_per_case]
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

        # Collect lightweight metadata first so unselected images are never opened.
        roi_records: List[ROIRecord] = []
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

                    roi_records.append(ROIRecord(
                        global_idx=roi_sample["global_idx"],
                        roi_path=roi_path,
                        mpp=mpp,
                        cxcywh=cxcywh,
                        roi_wh=roi_wh,
                    ))

        # Validation uses the same sampling method as training, but a stable
        # per-case RNG keeps the selected subset comparable across epochs.
        sampling_rng = None
        if self.split == "valid" and self.roi_sampling_mode == "random_k":
            sampling_rng = random.Random(self.valid_sampling_seed + int(idx))
        roi_records = self._sample_rois(roi_records, rng=sampling_rng)

        roi_list: List[ROI] = []
        for record in roi_records:
            if self.input_img and record.roi_path is not None:
                with Image.open(os.path.join(self.image_path, record.roi_path)) as pil_image:
                    image = self.transform(pil_image.convert("RGB"))
            else:
                image = None
            roi_list.append(ROI(
                global_idx=record.global_idx,
                image=image,
                mpp=record.mpp,
                cxcywh=record.cxcywh,
                roi_wh=record.roi_wh,
            ))

        # Keep the free-text target for SFT/text metrics and the class target
        # separately for clinical metrics.  In particular, the evaluator must
        # not re-infer a reference Nottingham grade from DxResultTxt.
        DxItem_targets = {}
        DxItem_target_classes = {}
        for DxItem, DxSample in sample["structured_report"]["DxItems"].items():
            if DxItem not in self.DxItem_list:
                continue
            DxItem_targets[DxItem] = DxSample["DxResultTxt"]
            DxItem_target_classes[DxItem] = DxSample["DxResultCls"]

        case = Case(
            global_idx=sample["sample_idx"],
            case_id=str(sample["case_id"]),
            rois=roi_list,
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

        dataset = cls(
            image_path=cfg.image_path,
            metadata_path=metadata_path,
            split=split,
            input_img=getattr(cfg, "input_img", True),
            input_loc=getattr(cfg, "input_loc", True),
            level_key=getattr(cfg, "level_key", "main_info"),
            max_rois_per_case=getattr(cfg, "max_rois_per_case", None),
            roi_sampling_mode=getattr(cfg, "roi_sampling_mode", "all"),
            valid_sampling_seed=getattr(cfg, "valid_sampling_seed", 42),
        )

        # Expose DxItem_list to config so model/trainer can access it
        cfg.DxItem_list = dataset.DxItem_list

        return dataset
























