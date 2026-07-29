"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2026, yasaisen (clover)

 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.

 Text-generation and clinical-structure evaluator for DRGVLM.
"""


from datetime import datetime
import json
import os
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from ..datasets.multiROI2DxResultDataset import Case
from ..configs.DRGVLM_baseConfig import DRGVLM_baseConfig
from ..common.utils import log_print


# ---------------------------------------------------------------------------
# Lightweight text metrics (standard library + numpy only)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lower-case alphanumeric word tokenisation."""
    return re.findall(r"\b\w+\b", str(text).lower())


def _normalize_text(text: str) -> str:
    return " ".join(_tokenize(text))


def _ngrams(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    ngrams: Dict[Tuple[str, ...], int] = {}
    for idx in range(len(tokens) - n + 1):
        key = tuple(tokens[idx: idx + n])
        ngrams[key] = ngrams.get(key, 0) + 1
    return ngrams


def _clip_count(candidate: Dict, reference: Dict) -> int:
    return sum(
        min(count, reference.get(ngram, 0))
        for ngram, count in candidate.items()
    )


def sentence_bleu(
    reference_tokens: List[str],
    hypothesis_tokens: List[str],
    max_n: int = 4,
    smooth: bool = True,
) -> float:
    """Effective-order sentence BLEU suitable for one- and two-token answers."""
    if not reference_tokens or not hypothesis_tokens:
        return 0.0

    brevity_penalty = 1.0
    if len(hypothesis_tokens) < len(reference_tokens):
        brevity_penalty = float(np.exp(
            1.0 - len(reference_tokens) / len(hypothesis_tokens)
        ))

    # Do not invent non-existent 2/3/4-gram orders for short hypotheses.
    effective_order = min(max(1, int(max_n)), len(hypothesis_tokens))
    log_precisions: List[float] = []
    for ngram_order in range(1, effective_order + 1):
        candidate = _ngrams(hypothesis_tokens, ngram_order)
        reference = _ngrams(reference_tokens, ngram_order)
        candidate_count = sum(candidate.values())
        clipped_count = _clip_count(candidate, reference)
        if smooth:
            precision = (clipped_count + 1) / (candidate_count + 1)
        else:
            if clipped_count == 0:
                return 0.0
            precision = clipped_count / candidate_count
        log_precisions.append(float(np.log(precision)))

    return float(brevity_penalty * np.exp(np.mean(log_precisions)))


def token_f1(
    reference_tokens: List[str],
    hypothesis_tokens: List[str],
) -> Tuple[float, float, float]:
    """Unigram bag-of-words precision, recall, and F1."""
    if not reference_tokens or not hypothesis_tokens:
        return 0.0, 0.0, 0.0

    reference_bag: Dict[str, int] = {}
    hypothesis_bag: Dict[str, int] = {}
    for token in reference_tokens:
        reference_bag[token] = reference_bag.get(token, 0) + 1
    for token in hypothesis_tokens:
        hypothesis_bag[token] = hypothesis_bag.get(token, 0) + 1

    common = sum(
        min(count, hypothesis_bag.get(token, 0))
        for token, count in reference_bag.items()
    )
    precision = common / sum(hypothesis_bag.values())
    recall = common / sum(reference_bag.values())
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return float(precision), float(recall), float(f1)


def rouge_l(
    reference_tokens: List[str],
    hypothesis_tokens: List[str],
) -> float:
    """ROUGE-L F1 based on longest-common-subsequence length."""
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    previous = [0] * (len(hypothesis_tokens) + 1)
    for reference_token in reference_tokens:
        current = [0]
        for idx, hypothesis_token in enumerate(hypothesis_tokens, start=1):
            if reference_token == hypothesis_token:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current
    lcs_length = previous[-1]
    precision = lcs_length / len(hypothesis_tokens)
    recall = lcs_length / len(reference_tokens)
    return float(
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )


def exact_match(reference: str, hypothesis: str) -> bool:
    return _normalize_text(reference) == _normalize_text(hypothesis)


def _mean_or_none(values: List[float]) -> Optional[float]:
    return float(np.mean(values)) if values else None


# ---------------------------------------------------------------------------
# Clinical parsers
# ---------------------------------------------------------------------------

_INVALID_TYPE_LABELS = {
    "",
    "a",
    "na",
    "n a",
    "none",
    "unknown",
    "not available",
}


def _parse_histologic_type(text: str) -> Dict[str, Any]:
    """Normalize histologic type without conflating in-situ and invasive disease."""
    normalized = _normalize_text(text)
    compact_alpha = "".join(ch for ch in normalized if ch.isalpha())
    valid = normalized not in _INVALID_TYPE_LABELS and len(compact_alpha) >= 3
    canonical = None
    if valid:
        has_carcinoma = "carcinoma" in normalized
        is_idc = bool(re.search(r"\bidc\b", normalized))
        is_ilc = bool(re.search(r"\bilc\b", normalized))
        has_invasive = bool(
            re.search(r"\b(?:invasive|infiltrating)\b", normalized)
            or is_idc
            or is_ilc
        )
        is_dcis = bool(
            re.search(r"\bdcis\b", normalized)
            or "ductal carcinoma in situ" in normalized
        )
        is_lcis = bool(
            re.search(r"\blcis\b", normalized)
            or "lobular carcinoma in situ" in normalized
        )
        is_ductal = bool(
            re.search(r"\b(?:duct|ductal)\b", normalized)
            or is_idc
            or "no special type" in normalized
            or re.search(r"\bnst\b", normalized)
        )
        is_lobular = bool(re.search(r"\blobular\b", normalized) or is_ilc)

        # Specific in-situ entities must be resolved before any generic
        # ductal/lobular carcinoma rules.
        if is_dcis and not has_invasive:
            canonical = "ductal carcinoma in situ"
        elif is_lcis and not has_invasive:
            canonical = "lobular carcinoma in situ"
        elif (
            has_invasive
            and "cribriform" in normalized
            and has_carcinoma
        ):
            canonical = "invasive cribriform carcinoma"
        elif has_invasive and is_ductal and (has_carcinoma or "idc" in normalized):
            if "mucin" in normalized:
                canonical = "invasive duct carcinoma with extracellular mucin"
            else:
                canonical = "invasive duct carcinoma"
        elif has_invasive and is_lobular and (has_carcinoma or "ilc" in normalized):
            canonical = "invasive lobular carcinoma"
        elif has_invasive and has_carcinoma:
            canonical = "invasive carcinoma"
        elif is_ductal and has_carcinoma:
            # Do not silently promote an ambiguous ductal carcinoma to invasive.
            canonical = "ductal carcinoma unspecified invasion"
        elif is_lobular and has_carcinoma:
            canonical = "lobular carcinoma unspecified invasion"
        else:
            canonical = normalized
    return {
        "valid": bool(valid),
        "canonical": canonical,
        "normalized": normalized,
    }


def _extract_int(pattern: str, text: str) -> Optional[int]:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return int(match.group(1)) if match is not None else None


def _parse_histologic_grade(text: str) -> Dict[str, Optional[int]]:
    normalized = str(text)
    return {
        "grade": _extract_int(r"\bgrade\s*[:=\-]?\s*([123])\b", normalized),
        "tubular_formation": _extract_int(
            r"\btubular(?:\s+formation)?\s*[:=\-]?\s*([123])\b",
            normalized,
        ),
        "nuclear_pleomorphism": _extract_int(
            r"\bnuclear\s+pleomorphism\s*[:=\-]?\s*([123])\b",
            normalized,
        ),
        "mitotic_count": _extract_int(
            r"\bmitotic(?:\s+count)?\s*[:=\-]?\s*([123])\b",
            normalized,
        ),
        "total_score": _extract_int(
            r"\btotal(?:\s+histologic)?\s+score\s*[:=\-]?\s*([3-9])(?:\s*/\s*9)?\b",
            normalized,
        ),
    }


_GRADE_CLASS_TO_INT = {
    "Grade I": 1,
    "Grade II": 2,
    "Grade III": 3,
}
_SCORE_CLASS_TO_INT = {
    "Score 1": 1,
    "Score 2": 2,
    "Score 3": 3,
}
_GRADE_REFERENCE_FIELDS = {
    "grade": ("Histologic_Grade", _GRADE_CLASS_TO_INT),
    "tubular_formation": ("Tubular_formation", _SCORE_CLASS_TO_INT),
    "nuclear_pleomorphism": ("Nuclear_pleomorphism", _SCORE_CLASS_TO_INT),
    "mitotic_count": ("Mitotic_count", _SCORE_CLASS_TO_INT),
}


def _grade_from_total_score(total_score: Optional[int]) -> Optional[int]:
    if total_score is None:
        return None
    if 3 <= total_score <= 5:
        return 1
    if 6 <= total_score <= 7:
        return 2
    if 8 <= total_score <= 9:
        return 3
    return None


def _grade_reference_from_classes(
    target_classes: Dict[str, str],
) -> Dict[str, Any]:
    """Build the Nottingham reference only from authoritative class labels."""
    reference: Dict[str, Any] = {}
    source_classes: Dict[str, str] = {}
    for output_field, (dx_item, mapping) in _GRADE_REFERENCE_FIELDS.items():
        raw_value = target_classes.get(dx_item)
        if not isinstance(raw_value, str):
            raise ValueError(
                f"Missing DxResultCls for required grade field {dx_item!r}"
            )
        class_value = raw_value.strip()
        if class_value not in mapping:
            raise ValueError(
                f"Unsupported DxResultCls for {dx_item!r}: {raw_value!r}; "
                f"expected one of {sorted(mapping)}"
            )
        source_classes[dx_item] = class_value
        reference[output_field] = mapping[class_value]

    reference["total_score"] = (
        reference["tubular_formation"]
        + reference["nuclear_pleomorphism"]
        + reference["mitotic_count"]
    )
    reference["grade_from_total"] = _grade_from_total_score(
        reference["total_score"]
    )
    reference["grade_total_consistent"] = (
        reference["grade"] == reference["grade_from_total"]
    )
    reference["source_classes"] = source_classes
    return reference


def _complete_grade_prediction(text: str) -> Dict[str, Any]:
    """Parse a prediction and derive grade from total only when grade is absent."""
    prediction: Dict[str, Any] = dict(_parse_histologic_grade(text))
    explicit_grade = prediction["grade"]
    total_derived_grade = _grade_from_total_score(prediction["total_score"])
    prediction["explicit_grade"] = explicit_grade
    prediction["grade_from_total"] = total_derived_grade
    prediction["grade_was_derived"] = (
        explicit_grade is None and total_derived_grade is not None
    )
    prediction["grade_total_conflict"] = (
        explicit_grade is not None
        and total_derived_grade is not None
        and explicit_grade != total_derived_grade
    )
    if prediction["grade_was_derived"]:
        prediction["grade"] = total_derived_grade
    return prediction


def _parse_microcalcification(text: str) -> Dict[str, Any]:
    normalized = _normalize_text(text)
    if re.search(r"\bnot\s+(?:identified|seen|found)\b", normalized):
        status = "not_identified"
    elif re.search(r"\b(?:absent|negative)\b", normalized):
        status = "absent"
    elif re.search(r"\b(?:present|identified|seen)\b", normalized):
        status = "present"
    else:
        status = "unknown"

    locations: Set[str] = set()
    if "dcis" in normalized or "ductal carcinoma in situ" in normalized:
        locations.add("dcis")
    if "invasive carcinoma" in normalized:
        locations.add("invasive_carcinoma")
    if "non neoplastic" in normalized or "nonneoplastic" in normalized:
        locations.add("non_neoplastic_tissue")
    if (
        "carcinoma" in normalized
        and "invasive carcinoma" not in normalized
        and "dcis" not in normalized
        and "ductal carcinoma in situ" not in normalized
    ):
        locations.add("carcinoma_unspecified")

    return {
        "status": status,
        "locations": sorted(locations),
    }


def _set_f1(reference: Set[str], hypothesis: Set[str]) -> float:
    if not reference and not hypothesis:
        return 1.0
    if not reference or not hypothesis:
        return 0.0
    common = len(reference & hypothesis)
    if common == 0:
        return 0.0
    precision = common / len(hypothesis)
    recall = common / len(reference)
    return float(2 * precision * recall / (precision + recall))


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class DRGVLMEvaluator:
    """Evaluate both language realization and clinical content correctness."""

    CLINICAL_METRIC_SCHEMA_VERSION = 3

    TEXT_METRIC_KEYS = (
        "exact_match",
        "bleu",
        "rouge_l",
        "token_f1",
        "token_precision",
        "token_recall",
    )

    def __init__(
        self,
        DxItem_list: List[str],
        strict_prediction_completeness: bool = True,
        reference_metadata_path: Optional[str] = None,
    ):
        self.DxItem_list = list(DxItem_list)
        self.strict_prediction_completeness = bool(
            strict_prediction_completeness
        )
        self.reference_metadata_path = reference_metadata_path
        self._reference_class_lookup: Optional[
            Dict[Tuple[str, str], Dict[str, str]]
        ] = None
        self._reset_state()

    def _reset_state(self):
        self.cases_list: List[Dict[str, Any]] = []
        self._seen_case_indices: Set[int] = set()
        self._prediction_completeness: Dict[str, Any] = {
            "strict": self.strict_prediction_completeness,
            "expected_prediction_count": 0,
            "present_prediction_count": 0,
            "missing_prediction_count": 0,
            "empty_prediction_count": 0,
            "unexpected_output_count": 0,
            "duplicate_case_count": 0,
            "diagnostics": [],
        }

    def update(
        self,
        batch_cases: List[Case],
        output_case_dict: Dict[int, Dict[str, Any]],
    ):
        expected_dx_items = set(self.DxItem_list)
        batch_case_indices = {case.global_idx for case in batch_cases}
        local_counts = {
            "expected_prediction_count": 0,
            "present_prediction_count": 0,
            "missing_prediction_count": 0,
            "empty_prediction_count": 0,
            "unexpected_output_count": 0,
            "duplicate_case_count": 0,
        }
        diagnostics: List[str] = []
        new_entries: List[Dict[str, Any]] = []
        new_case_indices: Set[int] = set()

        for global_idx, outputs in output_case_dict.items():
            if global_idx in batch_case_indices:
                continue
            count = len(outputs) if isinstance(outputs, dict) else 1
            local_counts["unexpected_output_count"] += max(1, count)
            diagnostics.append(
                f"unexpected global_idx={global_idx!r} in output_case_dict"
            )

        for case in batch_cases:
            global_idx = case.global_idx
            if (
                global_idx in self._seen_case_indices
                or global_idx in new_case_indices
            ):
                local_counts["duplicate_case_count"] += 1
                diagnostics.append(
                    f"duplicate case global_idx={global_idx!r}, "
                    f"case_id={case.case_id!r}"
                )
                continue

            case_outputs = output_case_dict.get(global_idx, {})
            if not isinstance(case_outputs, dict):
                diagnostics.append(
                    f"non-dict output for global_idx={global_idx!r}"
                )
                case_outputs = {}

            unexpected_dx_items = set(case_outputs) - expected_dx_items
            if unexpected_dx_items:
                local_counts["unexpected_output_count"] += len(
                    unexpected_dx_items
                )
                diagnostics.append(
                    f"unexpected DxItems for global_idx={global_idx!r}: "
                    f"{sorted(unexpected_dx_items)}"
                )

            case_entry: Dict[str, Any] = {
                "global_idx": global_idx,
                "case_id": str(case.case_id),
                "DxItem_dict": {},
                "DxItem_target_classes": dict(
                    getattr(case, "DxItem_target_classes", {})
                ),
            }
            for dx_item in self.DxItem_list:
                local_counts["expected_prediction_count"] += 1
                pred_info = case_outputs.get(dx_item)
                prediction_present = isinstance(pred_info, dict)
                if prediction_present:
                    local_counts["present_prediction_count"] += 1
                    pred_txt = pred_info.get("pred_txt", "") or ""
                    if not str(pred_txt).strip():
                        local_counts["empty_prediction_count"] += 1
                else:
                    local_counts["missing_prediction_count"] += 1
                    diagnostics.append(
                        f"missing prediction: global_idx={global_idx!r}, "
                        f"case_id={case.case_id!r}, DxItem={dx_item}"
                    )
                    pred_txt = ""

                gt_txt = getattr(case, "DxItem_targets", {}).get(dx_item, "") or ""
                case_entry["DxItem_dict"][dx_item] = {
                    "pred_txt": str(pred_txt),
                    "gt_txt": str(gt_txt),
                    "prediction_present": prediction_present,
                }

            new_entries.append(case_entry)
            new_case_indices.add(global_idx)

        structural_error_count = (
            local_counts["missing_prediction_count"]
            + local_counts["empty_prediction_count"]
            + local_counts["unexpected_output_count"]
            + local_counts["duplicate_case_count"]
        )
        if self.strict_prediction_completeness and structural_error_count:
            preview = "; ".join(diagnostics[:8])
            raise ValueError(
                "Incomplete or inconsistent evaluator predictions: "
                f"missing={local_counts['missing_prediction_count']}, "
                f"empty={local_counts['empty_prediction_count']}, "
                f"unexpected={local_counts['unexpected_output_count']}, "
                f"duplicate_cases={local_counts['duplicate_case_count']}. "
                f"Examples: {preview}"
            )

        for key, value in local_counts.items():
            self._prediction_completeness[key] += value
        self._prediction_completeness["diagnostics"].extend(diagnostics)
        self.cases_list.extend(new_entries)
        self._seen_case_indices.update(new_case_indices)

    @staticmethod
    def _invalid_record(
        entry: Dict[str, Any],
        dx_item: str,
        value: str,
        reason: str,
    ) -> Dict[str, Any]:
        return {
            "global_idx": entry["global_idx"],
            "case_id": entry["case_id"],
            "DxItem": dx_item,
            "value": value,
            "reason": reason,
        }

    def _evaluate(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        eval_cases: List[Dict[str, Any]] = []
        invalid_references: List[Dict[str, Any]] = []

        text_accum: Dict[str, Dict[str, List[float]]] = {
            dx_item: {metric: [] for metric in self.TEXT_METRIC_KEYS}
            for dx_item in self.DxItem_list
        }
        clinical_accum: Dict[str, List[float]] = {
            "type_accuracy": [],
            "type_parser_coverage": [],
            "grade_overall_accuracy": [],
            "grade_tubular_accuracy": [],
            "grade_pleomorphism_accuracy": [],
            "grade_mitotic_accuracy": [],
            "grade_total_accuracy": [],
            "grade_complete_accuracy": [],
            "grade_parser_coverage": [],
            "micro_status_accuracy": [],
            "micro_presence_accuracy": [],
            "micro_location_f1": [],
            "micro_parser_coverage": [],
        }
        grade_prediction_conflicts: List[Dict[str, Any]] = []

        for entry in self.cases_list:
            case_dx_dict: Dict[str, Any] = {}
            for dx_item in self.DxItem_list:
                texts = entry["DxItem_dict"][dx_item]
                pred_txt = texts["pred_txt"]
                gt_txt = texts["gt_txt"]
                prediction_present = bool(texts["prediction_present"])
                pred_tokens = _tokenize(pred_txt)
                gt_tokens = _tokenize(gt_txt)
                precision, recall, f1 = token_f1(gt_tokens, pred_tokens)
                text_metrics = {
                    "exact_match": float(exact_match(gt_txt, pred_txt)),
                    "bleu": sentence_bleu(gt_tokens, pred_tokens),
                    "rouge_l": rouge_l(gt_tokens, pred_tokens),
                    "token_f1": f1,
                    "token_precision": precision,
                    "token_recall": recall,
                }
                if not prediction_present:
                    # Lenient mode must retain missing items in every denominator.
                    text_metrics = {
                        metric_name: 0.0
                        for metric_name in self.TEXT_METRIC_KEYS
                    }
                if dx_item in text_accum:
                    for metric_name, metric_value in text_metrics.items():
                        text_accum[dx_item][metric_name].append(metric_value)

                structured: Dict[str, Any] = {}
                if dx_item == "Histologic_Type":
                    reference = _parse_histologic_type(gt_txt)
                    prediction = _parse_histologic_type(pred_txt)
                    valid_reference = bool(reference["valid"])
                    structured = {
                        "reference": reference,
                        "prediction": prediction,
                        "valid_reference": valid_reference,
                        "correct": None,
                    }
                    if valid_reference:
                        correct = float(
                            prediction["valid"]
                            and prediction["canonical"] == reference["canonical"]
                        )
                        structured["correct"] = correct
                        clinical_accum["type_accuracy"].append(correct)
                        clinical_accum["type_parser_coverage"].append(
                            float(prediction["valid"])
                        )
                    else:
                        invalid_references.append(self._invalid_record(
                            entry,
                            dx_item,
                            gt_txt,
                            "unparseable or malformed histologic type",
                        ))

                elif dx_item == "Histologic_Grade":
                    try:
                        reference = _grade_reference_from_classes(
                            entry["DxItem_target_classes"]
                        )
                    except ValueError as exc:
                        raise ValueError(
                            "Invalid authoritative grade reference: "
                            f"global_idx={entry['global_idx']!r}, "
                            f"case_id={entry['case_id']!r}: {exc}"
                        ) from exc
                    prediction = _complete_grade_prediction(pred_txt)
                    valid_reference = True
                    field_correctness: Dict[str, Optional[float]] = {}
                    field_to_accumulator = {
                        "grade": "grade_overall_accuracy",
                        "tubular_formation": "grade_tubular_accuracy",
                        "nuclear_pleomorphism": "grade_pleomorphism_accuracy",
                        "mitotic_count": "grade_mitotic_accuracy",
                        "total_score": "grade_total_accuracy",
                    }
                    for field_name, accumulator_name in field_to_accumulator.items():
                        correct = float(
                            prediction[field_name] == reference[field_name]
                        )
                        field_correctness[field_name] = correct
                        clinical_accum[accumulator_name].append(correct)
                    complete_correct = float(all(field_correctness.values()))
                    clinical_accum["grade_complete_accuracy"].append(
                        complete_correct
                    )
                    clinical_accum["grade_parser_coverage"].append(float(
                        all(
                            prediction[field_name] is not None
                            for field_name in field_to_accumulator
                        )
                    ))
                    if prediction["grade_total_conflict"]:
                        grade_prediction_conflicts.append({
                            "global_idx": entry["global_idx"],
                            "case_id": entry["case_id"],
                            "explicit_grade": prediction["explicit_grade"],
                            "grade_from_total": prediction["grade_from_total"],
                            "prediction": pred_txt,
                        })
                    structured = {
                        "reference": reference,
                        "prediction": prediction,
                        "valid_reference": valid_reference,
                        "field_correctness": field_correctness,
                        "complete_correct": complete_correct,
                    }

                elif dx_item == "Microcalcification":
                    reference = _parse_microcalcification(gt_txt)
                    prediction = _parse_microcalcification(pred_txt)
                    valid_reference = reference["status"] != "unknown"
                    structured = {
                        "reference": reference,
                        "prediction": prediction,
                        "valid_reference": valid_reference,
                        "status_correct": None,
                        "presence_correct": None,
                        "location_f1": None,
                    }
                    if valid_reference:
                        status_correct = float(
                            prediction["status"] == reference["status"]
                        )
                        pred_presence = (
                            prediction["status"] == "present"
                            if prediction["status"] != "unknown"
                            else None
                        )
                        ref_presence = reference["status"] == "present"
                        presence_correct = float(
                            pred_presence is not None
                            and pred_presence == ref_presence
                        )
                        structured["status_correct"] = status_correct
                        structured["presence_correct"] = presence_correct
                        clinical_accum["micro_status_accuracy"].append(
                            status_correct
                        )
                        clinical_accum["micro_presence_accuracy"].append(
                            presence_correct
                        )
                        clinical_accum["micro_parser_coverage"].append(float(
                            prediction["status"] != "unknown"
                        ))

                        reference_locations = set(reference["locations"])
                        if ref_presence and reference_locations:
                            location_score = _set_f1(
                                reference_locations,
                                set(prediction["locations"]),
                            )
                            structured["location_f1"] = location_score
                            clinical_accum["micro_location_f1"].append(
                                location_score
                            )
                    else:
                        invalid_references.append(self._invalid_record(
                            entry,
                            dx_item,
                            gt_txt,
                            "unable to parse microcalcification status",
                        ))

                case_dx_dict[dx_item] = {
                    "pred_txt": pred_txt,
                    "gt_txt": gt_txt,
                    "gt_cls": entry["DxItem_target_classes"].get(dx_item),
                    "prediction_present": prediction_present,
                    "text_metrics": text_metrics,
                    "structured": structured,
                }

            eval_cases.append({
                "global_idx": entry["global_idx"],
                "case_id": entry["case_id"],
                "DxItem_target_classes": dict(
                    entry["DxItem_target_classes"]
                ),
                "DxItem_dict": case_dx_dict,
            })

        per_dx_metrics: Dict[str, Dict[str, Optional[float]]] = {}
        for dx_item in self.DxItem_list:
            per_dx_metrics[dx_item] = {
                metric: _mean_or_none(values)
                for metric, values in text_accum[dx_item].items()
            }

        macro: Dict[str, Optional[float]] = {}
        for metric in self.TEXT_METRIC_KEYS:
            values = [
                per_dx_metrics[dx_item][metric]
                for dx_item in self.DxItem_list
                if per_dx_metrics[dx_item][metric] is not None
            ]
            macro[f"macro_{metric}"] = _mean_or_none(values)

        text_composite_values = [
            macro.get("macro_exact_match"),
            macro.get("macro_bleu"),
            macro.get("macro_rouge_l"),
            macro.get("macro_token_f1"),
        ]
        text_composite = _mean_or_none([
            float(value) for value in text_composite_values if value is not None
        ])

        clinical = {
            "Histologic_Type": {
                "accuracy": _mean_or_none(clinical_accum["type_accuracy"]),
                "parser_coverage": _mean_or_none(
                    clinical_accum["type_parser_coverage"]
                ),
                "valid_reference_count": len(clinical_accum["type_accuracy"]),
            },
            "Histologic_Grade": {
                "overall_accuracy": _mean_or_none(
                    clinical_accum["grade_overall_accuracy"]
                ),
                "tubular_formation_accuracy": _mean_or_none(
                    clinical_accum["grade_tubular_accuracy"]
                ),
                "nuclear_pleomorphism_accuracy": _mean_or_none(
                    clinical_accum["grade_pleomorphism_accuracy"]
                ),
                "mitotic_count_accuracy": _mean_or_none(
                    clinical_accum["grade_mitotic_accuracy"]
                ),
                "total_score_accuracy": _mean_or_none(
                    clinical_accum["grade_total_accuracy"]
                ),
                "complete_record_accuracy": _mean_or_none(
                    clinical_accum["grade_complete_accuracy"]
                ),
                "parser_coverage": _mean_or_none(
                    clinical_accum["grade_parser_coverage"]
                ),
                "valid_reference_count": len(
                    clinical_accum["grade_overall_accuracy"]
                ),
            },
            "Microcalcification": {
                "status_accuracy": _mean_or_none(
                    clinical_accum["micro_status_accuracy"]
                ),
                "presence_accuracy": _mean_or_none(
                    clinical_accum["micro_presence_accuracy"]
                ),
                "location_f1": _mean_or_none(
                    clinical_accum["micro_location_f1"]
                ),
                "parser_coverage": _mean_or_none(
                    clinical_accum["micro_parser_coverage"]
                ),
                "valid_reference_count": len(
                    clinical_accum["micro_status_accuracy"]
                ),
                "location_reference_count": len(
                    clinical_accum["micro_location_f1"]
                ),
            },
        }

        grade_component_names = (
            "overall_accuracy",
            "tubular_formation_accuracy",
            "nuclear_pleomorphism_accuracy",
            "mitotic_count_accuracy",
            "total_score_accuracy",
        )
        grade_task_score = _mean_or_none([
            float(clinical["Histologic_Grade"][name])
            for name in grade_component_names
            if clinical["Histologic_Grade"][name] is not None
        ])
        # Status remains the primary microcalcification classification because
        # absent and not_identified intentionally retain different meanings.
        micro_task_score = _mean_or_none([
            float(value)
            for value in (
                clinical["Microcalcification"]["status_accuracy"],
                clinical["Microcalcification"]["location_f1"],
            )
            if value is not None
        ])
        clinical["Histologic_Type"]["task_score"] = clinical[
            "Histologic_Type"
        ]["accuracy"]
        clinical["Histologic_Grade"]["task_score"] = grade_task_score
        clinical["Microcalcification"]["task_score"] = micro_task_score

        clinical_task_scores = {
            dx_item: clinical[dx_item]["task_score"]
            for dx_item in (
                "Histologic_Type",
                "Histologic_Grade",
                "Microcalcification",
            )
        }
        clinical_macro_score = _mean_or_none([
            float(value)
            for value in clinical_task_scores.values()
            if value is not None
        ])

        summary: Dict[str, Any] = {
            "clinical_metric_schema_version": (
                self.CLINICAL_METRIC_SCHEMA_VERSION
            ),
            "per_DxItem": per_dx_metrics,
            **macro,
            "text_composite": text_composite,
            "clinical": clinical,
            "clinical_task_scores": clinical_task_scores,
            "clinical_macro_score": clinical_macro_score,
            "prediction_completeness": dict(self._prediction_completeness),
            "invalid_reference_count": len(invalid_references),
            "invalid_references": invalid_references,
            "grade_prediction_conflict_count": len(
                grade_prediction_conflicts
            ),
            "grade_prediction_conflicts": grade_prediction_conflicts,
        }
        return eval_cases, summary

    @staticmethod
    def _flatten_metrics(
        value: Any,
        prefix: str = "",
        output: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        output = {} if output is None else output
        if isinstance(value, dict):
            for key, child in value.items():
                child_prefix = f"{prefix}/{key}" if prefix else str(key)
                DRGVLMEvaluator._flatten_metrics(child, child_prefix, output)
        elif isinstance(value, (float, int)) or value is None:
            output[prefix] = value
        # Lists contain case-level diagnostics and are intentionally not sent
        # to TensorBoard/checkpoint score selection.
        return output

    def recompute_saved_cases(
        self,
        stored_cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Recompute the current metric schema from saved case predictions."""
        self._reset_state()
        try:
            for stored_case in stored_cases:
                stored_dx = stored_case.get("DxItem_dict", {})
                targets: Dict[str, str] = {}
                stored_classes = stored_case.get("DxItem_target_classes", {})
                target_classes: Dict[str, str] = (
                    {
                        str(dx_item): str(value)
                        for dx_item, value in stored_classes.items()
                        if isinstance(value, str)
                    }
                    if isinstance(stored_classes, dict)
                    else {}
                )
                outputs: Dict[str, Dict[str, str]] = {}
                for dx_item in self.DxItem_list:
                    record = stored_dx.get(dx_item)
                    if not isinstance(record, dict):
                        continue
                    targets[dx_item] = str(record.get("gt_txt", "") or "")
                    gt_cls = record.get("gt_cls")
                    if isinstance(gt_cls, str):
                        target_classes[dx_item] = gt_cls
                    if bool(record.get("prediction_present", True)):
                        outputs[dx_item] = {
                            "pred_txt": str(record.get("pred_txt", "") or ""),
                        }

                if (
                    "Histologic_Grade" in self.DxItem_list
                    and any(
                        dx_item not in target_classes
                        for _, (dx_item, _) in _GRADE_REFERENCE_FIELDS.items()
                    )
                ):
                    target_classes.update(self._lookup_reference_classes(
                        global_idx=stored_case.get("global_idx"),
                        case_id=str(stored_case.get("case_id", "")),
                    ))

                case = SimpleNamespace(
                    global_idx=stored_case.get("global_idx"),
                    case_id=str(stored_case.get("case_id", "")),
                    DxItem_targets=targets,
                    DxItem_target_classes=target_classes,
                )
                self.update(
                    batch_cases=[case],
                    output_case_dict={case.global_idx: outputs},
                )

            _, summary = self._evaluate()
            return self._flatten_metrics(summary)
        finally:
            self._reset_state()

    def _lookup_reference_classes(
        self,
        global_idx: Any,
        case_id: str,
    ) -> Dict[str, str]:
        """Load classes for legacy schema-v1/v2 evaluation JSON files."""
        if self._reference_class_lookup is None:
            if not self.reference_metadata_path:
                raise ValueError(
                    "Legacy evaluation cases do not contain gt_cls. Set "
                    "reference_metadata_path so schema-v3 metrics can use "
                    "metadata DxResultCls instead of parsing gt_txt."
                )
            with open(
                self.reference_metadata_path,
                "r",
                encoding="utf-8",
            ) as file:
                metadata = json.load(file)
            lookup: Dict[Tuple[str, str], Dict[str, str]] = {}
            for sample in metadata.get("case_list", []):
                classes = {
                    str(dx_item): str(dx_sample["DxResultCls"])
                    for dx_item, dx_sample in sample.get(
                        "structured_report",
                        {},
                    ).get("DxItems", {}).items()
                    if isinstance(dx_sample, dict)
                    and isinstance(dx_sample.get("DxResultCls"), str)
                }
                lookup[("global_idx", str(sample.get("sample_idx")))] = classes
                lookup[("case_id", str(sample.get("case_id", "")))] = classes
            self._reference_class_lookup = lookup

        by_index = self._reference_class_lookup.get(
            ("global_idx", str(global_idx))
        )
        by_case_id = self._reference_class_lookup.get(
            ("case_id", str(case_id))
        )
        classes = by_index or by_case_id
        if classes is None:
            raise ValueError(
                "Unable to find authoritative DxResultCls in reference "
                f"metadata for global_idx={global_idx!r}, case_id={case_id!r}"
            )
        return dict(classes)

    def evaluate(
        self,
        epoch_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        epoch_str = str(epoch_idx).zfill(3)
        eval_cases, metrics_dict = self._evaluate()

        def _fmt(value: Optional[float]) -> str:
            return "n/a" if value is None else f"{value:.4f}"

        completeness = metrics_dict.get("prediction_completeness", {})
        log_print(
            f"[DRGVLMEvaluator] epoch={epoch_str}: "
            f"text_composite={_fmt(metrics_dict.get('text_composite'))}, "
            f"clinical_macro={_fmt(metrics_dict.get('clinical_macro_score'))}, "
            f"EM={_fmt(metrics_dict.get('macro_exact_match'))}, "
            f"BLEU={_fmt(metrics_dict.get('macro_bleu'))}, "
            f"ROUGE-L={_fmt(metrics_dict.get('macro_rouge_l'))}, "
            f"predictions={completeness.get('present_prediction_count', 0)}/"
            f"{completeness.get('expected_prediction_count', 0)}, "
            f"missing={completeness.get('missing_prediction_count', 0)}, "
            f"empty={completeness.get('empty_prediction_count', 0)}, "
            "grade_prediction_conflicts="
            f"{metrics_dict.get('grade_prediction_conflict_count', 0)}, "
            f"invalid_references={metrics_dict.get('invalid_reference_count', 0)}"
        )
        for invalid in metrics_dict.get("invalid_references", []):
            log_print(
                "[DRGVLMEvaluator][INVALID_REFERENCE] "
                f"case_id={invalid['case_id']}, DxItem={invalid['DxItem']}, "
                f"value={invalid['value']!r}, reason={invalid['reason']}"
            )
        for conflict in metrics_dict.get(
            "grade_prediction_conflicts",
            [],
        ):
            log_print(
                "[DRGVLMEvaluator][GRADE_PREDICTION_CONFLICT] "
                f"case_id={conflict['case_id']}, "
                f"explicit_grade={conflict['explicit_grade']}, "
                f"grade_from_total={conflict['grade_from_total']}, "
                f"prediction={conflict['prediction']!r}"
            )

        if save_path is not None:
            nowtime = datetime.now().strftime("%y%m%d%H%M")
            out_path = os.path.join(
                save_path,
                f"eval_results[{epoch_str}]_{nowtime}.json",
            )
            os.makedirs(save_path, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as file:
                json.dump({
                    "epoch_idx": epoch_idx,
                    "metrics": metrics_dict,
                    "cases": eval_cases,
                }, file, indent=2, ensure_ascii=False)
            log_print(f"Saved eval results to {out_path}")

        flat = self._flatten_metrics(metrics_dict)
        self._reset_state()
        return flat

    @classmethod
    def from_config(cls, cfg: DRGVLM_baseConfig) -> "DRGVLMEvaluator":
        return cls(
            DxItem_list=cfg.DxItem_list,
            strict_prediction_completeness=getattr(
                cfg,
                "strict_evaluator_predictions",
                True,
            ),
            reference_metadata_path=getattr(
                cfg,
                "valid_metadata_path",
                None,
            ),
        )
