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
    normalized = _normalize_text(text)
    compact_alpha = "".join(ch for ch in normalized if ch.isalpha())
    valid = normalized not in _INVALID_TYPE_LABELS and len(compact_alpha) >= 3
    canonical = None
    if valid:
        has_invasive = "invasive" in normalized or "infiltrating" in normalized
        if "cribriform" in normalized and "carcinoma" in normalized:
            canonical = "invasive cribriform carcinoma"
        elif "duct" in normalized and "carcinoma" in normalized:
            if "mucin" in normalized:
                canonical = "invasive duct carcinoma with extracellular mucin"
            else:
                canonical = "invasive duct carcinoma"
        elif has_invasive and "carcinoma" in normalized:
            canonical = "invasive carcinoma"
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

    TEXT_METRIC_KEYS = (
        "exact_match",
        "bleu",
        "rouge_l",
        "token_f1",
        "token_precision",
        "token_recall",
    )

    def __init__(self, DxItem_list: List[str]):
        self.DxItem_list = list(DxItem_list)
        self.cases_list: List[Dict[str, Any]] = []

    def update(
        self,
        batch_cases: List[Case],
        output_case_dict: Dict[int, Dict[str, Any]],
    ):
        for case in batch_cases:
            case_entry: Dict[str, Any] = {
                "global_idx": case.global_idx,
                "case_id": str(case.case_id),
                "DxItem_dict": {},
            }
            for dx_item in self.DxItem_list:
                pred_info = output_case_dict.get(case.global_idx, {}).get(dx_item)
                if pred_info is None:
                    continue
                case_entry["DxItem_dict"][dx_item] = {
                    "pred_txt": pred_info.get("pred_txt", "") or "",
                    "gt_txt": pred_info.get("gt_txt", "") or "",
                }
            self.cases_list.append(case_entry)

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

        for entry in self.cases_list:
            case_dx_dict: Dict[str, Any] = {}
            for dx_item, texts in entry["DxItem_dict"].items():
                pred_txt = texts["pred_txt"]
                gt_txt = texts["gt_txt"]
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
                    reference = _parse_histologic_grade(gt_txt)
                    prediction = _parse_histologic_grade(pred_txt)
                    valid_reference = reference["grade"] is not None
                    field_correctness: Dict[str, Optional[float]] = {}
                    field_to_accumulator = {
                        "grade": "grade_overall_accuracy",
                        "tubular_formation": "grade_tubular_accuracy",
                        "nuclear_pleomorphism": "grade_pleomorphism_accuracy",
                        "mitotic_count": "grade_mitotic_accuracy",
                        "total_score": "grade_total_accuracy",
                    }
                    if valid_reference:
                        for field_name, accumulator_name in field_to_accumulator.items():
                            if reference[field_name] is None:
                                field_correctness[field_name] = None
                                continue
                            correct = float(
                                prediction[field_name] == reference[field_name]
                            )
                            field_correctness[field_name] = correct
                            clinical_accum[accumulator_name].append(correct)
                        required_results = [
                            value for value in field_correctness.values()
                            if value is not None
                        ]
                        complete_correct = float(
                            bool(required_results) and all(required_results)
                        )
                        clinical_accum["grade_complete_accuracy"].append(
                            complete_correct
                        )
                        clinical_accum["grade_parser_coverage"].append(
                            float(prediction["grade"] is not None)
                        )
                    else:
                        complete_correct = None
                        invalid_references.append(self._invalid_record(
                            entry,
                            dx_item,
                            gt_txt,
                            "unable to parse overall Nottingham grade",
                        ))
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
                    "text_metrics": text_metrics,
                    "structured": structured,
                }

            eval_cases.append({
                "global_idx": entry["global_idx"],
                "case_id": entry["case_id"],
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
        clinical_composite_values = [
            clinical["Histologic_Type"]["accuracy"],
            clinical["Histologic_Grade"]["overall_accuracy"],
            clinical["Microcalcification"]["status_accuracy"],
            clinical["Microcalcification"]["location_f1"],
        ]
        clinical_macro_score = _mean_or_none([
            float(value)
            for value in clinical_composite_values
            if value is not None
        ])

        summary: Dict[str, Any] = {
            "per_DxItem": per_dx_metrics,
            **macro,
            "text_composite": text_composite,
            "clinical": clinical,
            "clinical_macro_score": clinical_macro_score,
            "invalid_reference_count": len(invalid_references),
            "invalid_references": invalid_references,
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

    def evaluate(
        self,
        epoch_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        epoch_str = str(epoch_idx).zfill(3)
        eval_cases, metrics_dict = self._evaluate()

        def _fmt(value: Optional[float]) -> str:
            return "n/a" if value is None else f"{value:.4f}"

        log_print(
            f"[DRGVLMEvaluator] epoch={epoch_str}: "
            f"text_composite={_fmt(metrics_dict.get('text_composite'))}, "
            f"clinical_macro={_fmt(metrics_dict.get('clinical_macro_score'))}, "
            f"EM={_fmt(metrics_dict.get('macro_exact_match'))}, "
            f"BLEU={_fmt(metrics_dict.get('macro_bleu'))}, "
            f"ROUGE-L={_fmt(metrics_dict.get('macro_rouge_l'))}, "
            f"invalid_references={metrics_dict.get('invalid_reference_count', 0)}"
        )
        for invalid in metrics_dict.get("invalid_references", []):
            log_print(
                "[DRGVLMEvaluator][INVALID_REFERENCE] "
                f"case_id={invalid['case_id']}, DxItem={invalid['DxItem']}, "
                f"value={invalid['value']!r}, reason={invalid['reason']}"
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
        self.cases_list = []
        return flat

    @classmethod
    def from_config(cls, cfg: DRGVLM_baseConfig) -> "DRGVLMEvaluator":
        return cls(DxItem_list=cfg.DxItem_list)
