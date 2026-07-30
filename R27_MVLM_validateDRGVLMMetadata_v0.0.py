#!/usr/bin/env python3
"""
Preflight validator for DRGVLM train/validation metadata.

This script intentionally depends only on the Python standard library so it can
run on a login node before importing torch, transformers, or the Dataset class.
It never modifies metadata or image files.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


REQUIRED_GRADE_ITEMS = (
    "Histologic_Grade",
    "Tubular_formation",
    "Nuclear_pleomorphism",
    "Mitotic_count",
)
GRADE_CLASS_TO_SCORE = {
    "Grade I": 1,
    "Grade II": 2,
    "Grade III": 3,
}
COMPONENT_CLASS_TO_SCORE = {
    "Score 1": 1,
    "Score 2": 2,
    "Score 3": 3,
}


def _grade_from_total(total: int) -> Optional[int]:
    if 3 <= total <= 5:
        return 1
    if 6 <= total <= 7:
        return 2
    if 8 <= total <= 9:
        return 3
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_numeric_sequence(value: Any, length: int) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == length
        and all(_is_number(item) for item in value)
    )


class ValidationReport:
    def __init__(self) -> None:
        self.errors: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []
        self.stats: Dict[str, Any] = {}

    def add(
        self,
        severity: str,
        code: str,
        message: str,
        *,
        split: Optional[str] = None,
        case_id: Optional[str] = None,
        sample_idx: Any = None,
        location: Optional[str] = None,
    ) -> None:
        record = {
            "code": code,
            "message": message,
        }
        for key, value in (
            ("split", split),
            ("case_id", case_id),
            ("sample_idx", sample_idx),
            ("location", location),
        ):
            if value is not None:
                record[key] = value
        target = self.errors if severity == "error" else self.warnings
        target.append(record)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _validate_grade_classes(
    report: ValidationReport,
    dx_items: Dict[str, Any],
    *,
    split: str,
    case_id: str,
    sample_idx: Any,
) -> None:
    missing = [name for name in REQUIRED_GRADE_ITEMS if name not in dx_items]
    if missing:
        report.add(
            "error",
            "GRADE_ITEMS_MISSING",
            f"Required grade DxItems are missing: {missing}",
            split=split,
            case_id=case_id,
            sample_idx=sample_idx,
            location="structured_report.DxItems",
        )
        return

    grade_class = dx_items["Histologic_Grade"].get("DxResultCls")
    if grade_class not in GRADE_CLASS_TO_SCORE:
        report.add(
            "error",
            "INVALID_GRADE_CLASS",
            f"Expected one of {sorted(GRADE_CLASS_TO_SCORE)}, got {grade_class!r}",
            split=split,
            case_id=case_id,
            sample_idx=sample_idx,
            location="structured_report.DxItems.Histologic_Grade.DxResultCls",
        )
        return

    component_names = REQUIRED_GRADE_ITEMS[1:]
    component_scores: List[int] = []
    for name in component_names:
        value = dx_items[name].get("DxResultCls")
        if value not in COMPONENT_CLASS_TO_SCORE:
            report.add(
                "error",
                "INVALID_GRADE_COMPONENT_CLASS",
                (
                    f"{name} expected one of "
                    f"{sorted(COMPONENT_CLASS_TO_SCORE)}, got {value!r}"
                ),
                split=split,
                case_id=case_id,
                sample_idx=sample_idx,
                location=f"structured_report.DxItems.{name}.DxResultCls",
            )
        else:
            component_scores.append(COMPONENT_CLASS_TO_SCORE[value])

    if len(component_scores) == len(component_names):
        total = sum(component_scores)
        expected_grade = _grade_from_total(total)
        actual_grade = GRADE_CLASS_TO_SCORE[grade_class]
        if expected_grade != actual_grade:
            report.add(
                "error",
                "GRADE_COMPONENT_CONFLICT",
                (
                    f"{grade_class} conflicts with component total {total} "
                    f"(derived grade={expected_grade})"
                ),
                split=split,
                case_id=case_id,
                sample_idx=sample_idx,
                location="structured_report.DxItems",
            )


def _validate_roi_level(
    report: ValidationReport,
    level_info: Any,
    *,
    split: str,
    case_id: str,
    sample_idx: Any,
    location: str,
    image_root: Optional[Path],
    check_images: bool,
) -> None:
    if not isinstance(level_info, dict):
        report.add(
            "error",
            "INVALID_ROI_LEVEL",
            "ROI level information must be an object",
            split=split,
            case_id=case_id,
            sample_idx=sample_idx,
            location=location,
        )
        return

    roi_path = level_info.get("roi_path")
    if roi_path is not None and not isinstance(roi_path, str):
        report.add(
            "error",
            "INVALID_ROI_PATH",
            f"roi_path must be a string or null, got {type(roi_path).__name__}",
            split=split,
            case_id=case_id,
            sample_idx=sample_idx,
            location=f"{location}.roi_path",
        )
    elif check_images and roi_path:
        if image_root is None:
            report.add(
                "error",
                "IMAGE_ROOT_REQUIRED",
                "--check-images requires --image-root or config image_path",
                split=split,
                case_id=case_id,
                sample_idx=sample_idx,
                location=f"{location}.roi_path",
            )
        elif not (image_root / roi_path).is_file():
            report.add(
                "error",
                "IMAGE_NOT_FOUND",
                f"Image does not exist: {image_root / roi_path}",
                split=split,
                case_id=case_id,
                sample_idx=sample_idx,
                location=f"{location}.roi_path",
            )

    mpp = level_info.get("mpp")
    valid_mpp = (
        mpp is None
        or (_is_number(mpp) and mpp > 0)
        or (
            isinstance(mpp, (list, tuple))
            and len(mpp) in (1, 2)
            and all(_is_number(value) and value > 0 for value in mpp)
        )
    )
    if not valid_mpp:
        report.add(
            "error",
            "INVALID_MPP",
            f"mpp must contain positive numeric value(s), got {mpp!r}",
            split=split,
            case_id=case_id,
            sample_idx=sample_idx,
            location=f"{location}.mpp",
        )

    cxcywh = level_info.get("cxcywh")
    if cxcywh is not None:
        if not _is_numeric_sequence(cxcywh, 4):
            report.add(
                "error",
                "INVALID_CXCYWH",
                f"cxcywh must be four numbers or null, got {cxcywh!r}",
                split=split,
                case_id=case_id,
                sample_idx=sample_idx,
                location=f"{location}.cxcywh",
            )
        elif not all(0.0 <= float(value) <= 1.0 for value in cxcywh):
            report.add(
                "warning",
                "CXCYWH_OUTSIDE_NORMALIZED_RANGE",
                f"cxcywh contains value outside [0, 1]: {cxcywh!r}",
                split=split,
                case_id=case_id,
                sample_idx=sample_idx,
                location=f"{location}.cxcywh",
            )

    roi_wh = level_info.get("roi_wh")
    if (
        roi_wh is not None
        and (
            not _is_numeric_sequence(roi_wh, 2)
            or any(float(value) <= 0 for value in roi_wh)
        )
    ):
        report.add(
            "error",
            "INVALID_ROI_WH",
            f"roi_wh must be two positive numbers, got {roi_wh!r}",
            split=split,
            case_id=case_id,
            sample_idx=sample_idx,
            location=f"{location}.roi_wh",
        )


def validate_metadata(
    path: Path,
    *,
    split: str,
    level_key: str,
    image_root: Optional[Path],
    check_images: bool,
    report: ValidationReport,
) -> Tuple[Set[str], Set[str]]:
    try:
        metadata = _load_json(path)
    except Exception as exc:
        report.add(
            "error",
            "METADATA_READ_FAILED",
            f"Cannot read JSON {path}: {exc}",
            split=split,
        )
        return set(), set()

    if not isinstance(metadata, dict):
        report.add(
            "error",
            "INVALID_ROOT",
            "Metadata root must be an object",
            split=split,
        )
        return set(), set()

    declared_dx_items = metadata.get("DxItem_list")
    if (
        not isinstance(declared_dx_items, list)
        or not declared_dx_items
        or not all(isinstance(item, str) and item for item in declared_dx_items)
    ):
        report.add(
            "error",
            "INVALID_DXITEM_LIST",
            "DxItem_list must be a non-empty list of strings",
            split=split,
        )
        declared_dx_items = []
    elif len(set(declared_dx_items)) != len(declared_dx_items):
        report.add(
            "error",
            "DUPLICATE_DXITEM",
            "DxItem_list contains duplicate entries",
            split=split,
        )

    case_list = metadata.get("case_list")
    if not isinstance(case_list, list):
        report.add(
            "error",
            "INVALID_CASE_LIST",
            "case_list must be a list",
            split=split,
        )
        return set(), set()

    sample_indices: Set[str] = set()
    case_ids: Set[str] = set()
    roi_indices: Set[str] = set()
    dx_class_counts: Dict[str, Counter] = {
        item: Counter() for item in declared_dx_items
    }
    roi_count = 0

    for case_position, sample in enumerate(case_list):
        if not isinstance(sample, dict):
            report.add(
                "error",
                "INVALID_CASE",
                "Each case must be an object",
                split=split,
                location=f"case_list[{case_position}]",
            )
            continue

        sample_idx = sample.get("sample_idx")
        case_id_value = sample.get("case_id")
        case_id = str(case_id_value) if case_id_value is not None else ""
        if not isinstance(sample_idx, int):
            report.add(
                "error",
                "INVALID_SAMPLE_IDX",
                f"sample_idx must be int, got {sample_idx!r}",
                split=split,
                case_id=case_id,
                location=f"case_list[{case_position}].sample_idx",
            )
        sample_idx_key = str(sample_idx)
        if sample_idx_key in sample_indices:
            report.add(
                "error",
                "DUPLICATE_SAMPLE_IDX",
                f"Duplicate sample_idx={sample_idx!r}",
                split=split,
                case_id=case_id,
                sample_idx=sample_idx,
            )
        sample_indices.add(sample_idx_key)

        if not case_id.strip():
            report.add(
                "error",
                "EMPTY_CASE_ID",
                "case_id must be non-empty",
                split=split,
                sample_idx=sample_idx,
            )
        elif case_id in case_ids:
            report.add(
                "error",
                "DUPLICATE_CASE_ID",
                f"Duplicate case_id={case_id!r}",
                split=split,
                case_id=case_id,
                sample_idx=sample_idx,
            )
        case_ids.add(case_id)

        dx_items = sample.get("structured_report", {}).get("DxItems")
        if not isinstance(dx_items, dict):
            report.add(
                "error",
                "INVALID_DXITEMS",
                "structured_report.DxItems must be an object",
                split=split,
                case_id=case_id,
                sample_idx=sample_idx,
            )
            dx_items = {}

        for dx_item, dx_sample in dx_items.items():
            if dx_item not in declared_dx_items:
                report.add(
                    "error",
                    "UNDECLARED_DXITEM_TARGET",
                    f"Target DxItem {dx_item!r} is absent from DxItem_list",
                    split=split,
                    case_id=case_id,
                    sample_idx=sample_idx,
                    location=f"structured_report.DxItems.{dx_item}",
                )
                continue
            if not isinstance(dx_sample, dict):
                report.add(
                    "error",
                    "INVALID_DXITEM_TARGET",
                    f"Target DxItem {dx_item!r} must be an object",
                    split=split,
                    case_id=case_id,
                    sample_idx=sample_idx,
                    location=f"structured_report.DxItems.{dx_item}",
                )
                continue
            for field in ("DxResultTxt", "DxResultCls"):
                value = dx_sample.get(field)
                if not isinstance(value, str) or not value.strip():
                    report.add(
                        "error",
                        "INVALID_DX_RESULT",
                        f"{field} must be a non-empty string, got {value!r}",
                        split=split,
                        case_id=case_id,
                        sample_idx=sample_idx,
                        location=(
                            f"structured_report.DxItems.{dx_item}.{field}"
                        ),
                    )
            cls_value = dx_sample.get("DxResultCls")
            if isinstance(cls_value, str):
                dx_class_counts[dx_item][cls_value] += 1

        if "Histologic_Grade" in dx_items:
            _validate_grade_classes(
                report,
                dx_items,
                split=split,
                case_id=case_id,
                sample_idx=sample_idx,
            )

        tissue_blocks = sample.get("tissue_blocks")
        if not isinstance(tissue_blocks, list):
            report.add(
                "error",
                "INVALID_TISSUE_BLOCKS",
                "tissue_blocks must be a list",
                split=split,
                case_id=case_id,
                sample_idx=sample_idx,
            )
            continue

        case_roi_count = 0
        active_dx_items: Set[str] = set()
        for block_idx, block in enumerate(tissue_blocks):
            stains = block.get("stains") if isinstance(block, dict) else None
            if not isinstance(stains, list):
                report.add(
                    "error",
                    "INVALID_STAINS",
                    "Each tissue block must contain a stains list",
                    split=split,
                    case_id=case_id,
                    sample_idx=sample_idx,
                    location=f"tissue_blocks[{block_idx}].stains",
                )
                continue
            for stain_idx, stain in enumerate(stains):
                if not isinstance(stain, dict) or "roi_list" not in stain:
                    continue
                roi_list = stain.get("roi_list")
                if not isinstance(roi_list, list):
                    report.add(
                        "error",
                        "INVALID_ROI_LIST",
                        "roi_list must be a list",
                        split=split,
                        case_id=case_id,
                        sample_idx=sample_idx,
                        location=(
                            f"tissue_blocks[{block_idx}].stains"
                            f"[{stain_idx}].roi_list"
                        ),
                    )
                    continue
                for roi_idx, roi in enumerate(roi_list):
                    roi_location = (
                        f"tissue_blocks[{block_idx}].stains[{stain_idx}]"
                        f".roi_list[{roi_idx}]"
                    )
                    if not isinstance(roi, dict):
                        report.add(
                            "error",
                            "INVALID_ROI",
                            "ROI must be an object",
                            split=split,
                            case_id=case_id,
                            sample_idx=sample_idx,
                            location=roi_location,
                        )
                        continue
                    global_idx = roi.get("global_idx")
                    if not isinstance(global_idx, int):
                        report.add(
                            "error",
                            "INVALID_ROI_GLOBAL_IDX",
                            f"ROI global_idx must be int, got {global_idx!r}",
                            split=split,
                            case_id=case_id,
                            sample_idx=sample_idx,
                            location=f"{roi_location}.global_idx",
                        )
                    global_idx_key = str(global_idx)
                    if global_idx_key in roi_indices:
                        report.add(
                            "error",
                            "DUPLICATE_ROI_GLOBAL_IDX",
                            f"Duplicate ROI global_idx={global_idx!r}",
                            split=split,
                            case_id=case_id,
                            sample_idx=sample_idx,
                            location=f"{roi_location}.global_idx",
                        )
                    roi_indices.add(global_idx_key)

                    dx_pair = roi.get("DxPair")
                    if dx_pair is not None and not isinstance(dx_pair, dict):
                        report.add(
                            "error",
                            "INVALID_DXPAIR",
                            "DxPair should be an object or null",
                            split=split,
                            case_id=case_id,
                            sample_idx=sample_idx,
                            location=f"{roi_location}.DxPair",
                        )
                    elif isinstance(dx_pair, dict):
                        for dx_item in dx_pair:
                            if dx_item not in declared_dx_items:
                                report.add(
                                    "error",
                                    "UNDECLARED_DXPAIR_ITEM",
                                    (
                                        f"DxPair references {dx_item!r}, which "
                                        "is absent from DxItem_list"
                                    ),
                                    split=split,
                                    case_id=case_id,
                                    sample_idx=sample_idx,
                                    location=f"{roi_location}.DxPair",
                                )
                                continue
                            if not isinstance(dx_items.get(dx_item), dict):
                                report.add(
                                    "error",
                                    "DXPAIR_TARGET_MISSING",
                                    (
                                        f"DxPair references {dx_item!r}, but "
                                        "structured_report.DxItems has no "
                                        "matching target"
                                    ),
                                    split=split,
                                    case_id=case_id,
                                    sample_idx=sample_idx,
                                    location=f"{roi_location}.DxPair",
                                )
                                continue
                            active_dx_items.add(dx_item)

                    if level_key not in roi:
                        report.add(
                            "error",
                            "ROI_LEVEL_MISSING",
                            f"ROI does not contain configured level {level_key!r}",
                            split=split,
                            case_id=case_id,
                            sample_idx=sample_idx,
                            location=roi_location,
                        )
                    else:
                        _validate_roi_level(
                            report,
                            roi[level_key],
                            split=split,
                            case_id=case_id,
                            sample_idx=sample_idx,
                            location=f"{roi_location}.{level_key}",
                            image_root=image_root,
                            check_images=check_images,
                        )
                    case_roi_count += 1
                    roi_count += 1

        if case_roi_count == 0:
            report.add(
                "warning",
                "CASE_WITHOUT_ROI",
                "Case has no ROI records",
                split=split,
                case_id=case_id,
                sample_idx=sample_idx,
            )
        if not active_dx_items:
            report.add(
                "error",
                "CASE_WITHOUT_ACTIVE_DXITEM",
                "No ROI DxPair assigns this case to a valid DxItem",
                split=split,
                case_id=case_id,
                sample_idx=sample_idx,
            )

    report.stats[split] = {
        "metadata_path": str(path),
        "case_count": len(case_list),
        "roi_count": roi_count,
        "DxItem_list": declared_dx_items,
        "DxResultCls_counts": {
            dx_item: dict(counter)
            for dx_item, counter in dx_class_counts.items()
        },
    }
    return case_ids, sample_indices


def _resolve_inputs(args: argparse.Namespace) -> Tuple[
    Optional[Path],
    Optional[Path],
    Optional[Path],
    str,
]:
    config: Dict[str, Any] = {}
    if args.config:
        config_path = Path(args.config).expanduser()
        config = _load_json(config_path)
        if not isinstance(config, dict):
            raise ValueError("Config root must be a JSON object")

    train_path = args.train_metadata or config.get("train_metadata_path")
    valid_path = args.valid_metadata or config.get("valid_metadata_path")
    image_root = args.image_root or config.get("image_path")
    level_key = args.level_key or config.get("level_key", "main_info")
    return (
        Path(train_path).expanduser() if train_path else None,
        Path(valid_path).expanduser() if valid_path else None,
        Path(image_root).expanduser() if image_root else None,
        str(level_key),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate DRGVLM metadata before Dataset/DataLoader/model startup."
        )
    )
    parser.add_argument(
        "--config",
        help="DRGVLM config JSON; metadata/image paths are read from it.",
    )
    parser.add_argument("--train-metadata", help="Override train metadata JSON.")
    parser.add_argument("--valid-metadata", help="Override valid metadata JSON.")
    parser.add_argument("--image-root", help="Override image root directory.")
    parser.add_argument(
        "--level-key",
        help="ROI level to validate (default: config level_key or main_info).",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Also stat every non-null roi_path (slower on network storage).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return exit status 1 if errors or warnings are found.",
    )
    parser.add_argument(
        "--report",
        help="Write the complete machine-readable JSON report to this path.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=30,
        help="Maximum number of diagnostics printed to stderr (default: 30).",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        train_path, valid_path, image_root, level_key = _resolve_inputs(args)
    except Exception as exc:
        print(f"[metadata-preflight][FATAL] {exc}", file=sys.stderr)
        return 2
    if train_path is None and valid_path is None:
        print(
            "[metadata-preflight][FATAL] Provide --config and/or at least one "
            "metadata path.",
            file=sys.stderr,
        )
        return 2

    report = ValidationReport()
    split_ids: Dict[str, Set[str]] = {}
    for split, path in (("train", train_path), ("valid", valid_path)):
        if path is None:
            continue
        case_ids, _ = validate_metadata(
            path,
            split=split,
            level_key=level_key,
            image_root=image_root,
            check_images=args.check_images,
            report=report,
        )
        split_ids[split] = case_ids

    if "train" in split_ids and "valid" in split_ids:
        overlap = sorted(split_ids["train"] & split_ids["valid"])
        if overlap:
            report.add(
                "error",
                "TRAIN_VALID_CASE_OVERLAP",
                (
                    f"{len(overlap)} case_id values occur in both splits; "
                    f"examples={overlap[:20]}"
                ),
            )

    result = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "level_key": level_key,
        "check_images": bool(args.check_images),
        "summary": {
            "error_count": len(report.errors),
            "warning_count": len(report.warnings),
            "passed": not report.errors and (not args.strict or not report.warnings),
        },
        "stats": report.stats,
        "errors": report.errors,
        "warnings": report.warnings,
    }

    if args.report:
        report_path = Path(args.report).expanduser()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, ensure_ascii=False)
            file.write("\n")

    diagnostics = report.errors + report.warnings
    for item in diagnostics[: max(0, args.max_print)]:
        severity = "ERROR" if item in report.errors else "WARN"
        context = ", ".join(
            f"{key}={item[key]!r}"
            for key in ("split", "case_id", "sample_idx", "location")
            if key in item
        )
        print(
            f"[metadata-preflight][{severity}][{item['code']}] "
            f"{item['message']}"
            + (f" ({context})" if context else ""),
            file=sys.stderr,
        )
    if len(diagnostics) > args.max_print:
        print(
            f"[metadata-preflight] {len(diagnostics) - args.max_print} "
            "additional diagnostic(s) are present in the JSON report.",
            file=sys.stderr,
        )

    summary = result["summary"]
    print(
        "[metadata-preflight] "
        f"errors={summary['error_count']}, "
        f"warnings={summary['warning_count']}, "
        f"passed={summary['passed']}"
    )
    if args.report:
        print(f"[metadata-preflight] report={args.report}")

    if report.errors:
        return 1
    if args.strict and report.warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
