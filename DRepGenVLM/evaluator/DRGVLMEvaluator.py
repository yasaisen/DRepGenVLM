"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2026, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2607081521
"""


from datetime import datetime
import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple

import numpy as np


from ..datasets.multiROI2DxResultDataset import Case
from ..configs.DRGVLM_baseConfig import DRGVLM_baseConfig
from ..common.utils import log_print, _debug_print, save_list2json


# ---------------------------------------------------------------------------
# Lightweight BLEU / token-F1 helpers (no extra deps beyond standard library)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lower-case word tokenisation."""
    return re.findall(r"\b\w+\b", text.lower())


def _ngrams(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    ng: Dict[Tuple[str, ...], int] = {}
    for i in range(len(tokens) - n + 1):
        key = tuple(tokens[i: i + n])
        ng[key] = ng.get(key, 0) + 1
    return ng


def _clip_count(candidate: Dict, reference: Dict) -> int:
    total = 0
    for ng, cnt in candidate.items():
        total += min(cnt, reference.get(ng, 0))
    return total


def sentence_bleu(
    reference_tokens: List[str],
    hypothesis_tokens: List[str],
    max_n: int = 4,
    smooth: bool = True,
) -> float:
    """Sentence-level BLEU-1..4 with brevity penalty and smoothing."""
    if len(hypothesis_tokens) == 0:
        return 0.0

    # Brevity penalty
    bp = 1.0 if len(hypothesis_tokens) >= len(reference_tokens) else (
        float(np.exp(1.0 - len(reference_tokens) / max(1, len(hypothesis_tokens))))
    )

    log_prec = []
    for n in range(1, max_n + 1):
        cand_ng = _ngrams(hypothesis_tokens, n)
        ref_ng = _ngrams(reference_tokens, n)
        total_cand = sum(cand_ng.values())
        if total_cand == 0:
            if smooth:
                log_prec.append(np.log(1e-9))
            else:
                return 0.0
            continue
        clipped = _clip_count(cand_ng, ref_ng)
        if smooth:
            log_prec.append(np.log((clipped + 1) / (total_cand + 1)))
        else:
            if clipped == 0:
                return 0.0
            log_prec.append(np.log(clipped / total_cand))

    avg_log_prec = float(np.mean(log_prec))
    return float(bp * np.exp(avg_log_prec))


def token_f1(
    reference_tokens: List[str],
    hypothesis_tokens: List[str],
) -> Tuple[float, float, float]:
    """Unigram precision, recall, F1 (bag-of-words overlap)."""
    if not hypothesis_tokens:
        return 0.0, 0.0, 0.0
    if not reference_tokens:
        return 0.0, 0.0, 0.0

    ref_bag: Dict[str, int] = {}
    for t in reference_tokens:
        ref_bag[t] = ref_bag.get(t, 0) + 1

    hyp_bag: Dict[str, int] = {}
    for t in hypothesis_tokens:
        hyp_bag[t] = hyp_bag.get(t, 0) + 1

    common = sum(min(cnt, hyp_bag.get(tok, 0)) for tok, cnt in ref_bag.items())
    precision = common / sum(hyp_bag.values()) if hyp_bag else 0.0
    recall = common / sum(ref_bag.values()) if ref_bag else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def exact_match(reference: str, hypothesis: str) -> bool:
    return reference.strip().lower() == hypothesis.strip().lower()


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class DRGVLMEvaluator:
    """Generation-quality evaluator for DRGVLM.

    Collected per update():
        output_case_dict: {case.global_idx: {DxItem: {"pred_txt": str, "gt_txt": str}}}

    Reported metrics per DxItem (and macro-averaged):
        exact_match    – 0/1 accuracy after lower-case strip
        bleu           – sentence-level BLEU-4
        token_f1       – unigram bag-of-words F1
        token_precision
        token_recall
    """

    def __init__(self,
        DxItem_list: List[str],
    ):
        self.DxItem_list = list(DxItem_list)
        self.cases_list: List[Dict[str, Any]] = []

    def update(self,
        batch_cases: List[Case],
        output_case_dict: Dict[int, Dict[str, Any]],
    ):
        for case in batch_cases:
            case_entry: Dict[str, Any] = {
                "global_idx": case.global_idx,
                "DxItem_dict": {},
            }
            for DxItem in self.DxItem_list:
                pred_info = output_case_dict.get(case.global_idx, {}).get(DxItem, None)
                if pred_info is None:
                    continue
                pred_txt = pred_info.get("pred_txt", "") or ""
                gt_txt = pred_info.get("gt_txt", "") or ""
                case_entry["DxItem_dict"][DxItem] = {
                    "pred_txt": pred_txt,
                    "gt_txt": gt_txt,
                }
            self.cases_list.append(case_entry)

    # ------------------------------------------------------------------
    def _evaluate(self) -> Tuple[List[Dict], Dict[str, Any]]:
        eval_cases: List[Dict] = []

        # Per-DxItem accumulator
        accum: Dict[str, Dict[str, List[float]]] = {
            di: {"exact_match": [], "bleu": [], "token_f1": [], "token_precision": [], "token_recall": []}
            for di in self.DxItem_list
        }

        for entry in self.cases_list:
            case_dx_dict: Dict[str, Any] = {}
            for DxItem, txts in entry["DxItem_dict"].items():
                pred_txt = txts["pred_txt"]
                gt_txt = txts["gt_txt"]

                pred_tokens = _tokenize(pred_txt)
                gt_tokens = _tokenize(gt_txt)

                em = exact_match(gt_txt, pred_txt)
                bleu = sentence_bleu(gt_tokens, pred_tokens)
                prec, rec, f1 = token_f1(gt_tokens, pred_tokens)

                metrics = {
                    "exact_match": float(em),
                    "bleu": bleu,
                    "token_f1": f1,
                    "token_precision": prec,
                    "token_recall": rec,
                }
                case_dx_dict[DxItem] = {
                    "pred_txt": pred_txt,
                    "gt_txt": gt_txt,
                    "metrics": metrics,
                }
                if DxItem in accum:
                    for k, v in metrics.items():
                        accum[DxItem][k].append(v)

            eval_cases.append({
                "global_idx": entry["global_idx"],
                "DxItem_dict": case_dx_dict,
            })

        # Aggregate per-DxItem
        per_dx_metrics: Dict[str, Dict[str, Optional[float]]] = {}
        for DxItem in self.DxItem_list:
            a = accum[DxItem]
            per_dx_metrics[DxItem] = {
                k: float(np.mean(v)) if v else None
                for k, v in a.items()
            }

        # Macro average across DxItems
        macro: Dict[str, Optional[float]] = {}
        metric_keys = ["exact_match", "bleu", "token_f1", "token_precision", "token_recall"]
        for mk in metric_keys:
            values = [per_dx_metrics[di][mk] for di in self.DxItem_list if per_dx_metrics[di][mk] is not None]
            macro[f"macro_{mk}"] = float(np.mean(values)) if values else None

        summary: Dict[str, Any] = {
            "per_DxItem": per_dx_metrics,
            **macro,
        }
        return eval_cases, summary

    # ------------------------------------------------------------------
    def evaluate(self,
        epoch_idx: int = 0,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        epoch_str = str(epoch_idx).zfill(3)
        eval_cases, metrics_dict = self._evaluate()

        log_print(f"[DRGVLMEvaluator] epoch={epoch_str} metrics: "
                  f"macro_exact_match={metrics_dict.get('macro_exact_match'):.4f}, "
                  f"macro_bleu={metrics_dict.get('macro_bleu'):.4f}, "
                  f"macro_token_f1={metrics_dict.get('macro_token_f1'):.4f}")

        if save_path is not None:
            nowtime = datetime.now().strftime("%y%m%d%H%M")
            out_path = os.path.join(save_path, f"eval_results[{epoch_str}]_{nowtime}.json")
            os.makedirs(save_path, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({
                    "epoch_idx": epoch_idx,
                    "metrics": metrics_dict,
                    "cases": eval_cases,
                }, f, indent=2, ensure_ascii=False)
            log_print(f"Saved eval results to {out_path}")

        # Flatten for trainer compatibility (expects flat dict of floats)
        flat: Dict[str, Any] = {}
        for k, v in metrics_dict.items():
            if k == "per_DxItem":
                for di, dx_metrics in v.items():
                    for mk, mv in dx_metrics.items():
                        flat[f"{di}/{mk}"] = mv
            else:
                flat[k] = v

        # Reset for next epoch
        self.cases_list = []
        return flat

    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls,
        cfg: DRGVLM_baseConfig,
    ) -> "DRGVLMEvaluator":
        return cls(DxItem_list=cfg.DxItem_list)















