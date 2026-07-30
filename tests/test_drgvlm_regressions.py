import json
import math
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn.functional as F

from DRepGenVLM.datasets.multiROI2DxResultDataset import (
    multiROI2DxResultDataset,
)
from DRepGenVLM.datasets.maxROI_sampler import MaxROIBatchSampler
from DRepGenVLM.evaluator.DRGVLMEvaluator import (
    DRGVLMEvaluator,
    _parse_histologic_type,
    _parse_microcalcification,
)
from DRepGenVLM.models.flashAttentionPatch import (
    CausalBlockAttention,
    install_large_head_flash_attention_patch,
)
from DRepGenVLM.models.modeling_medGemmaLoRA import DownstreamRepGenVLM
from DRepGenVLM.models.modelBuilder import modelBuilder
from DRepGenVLM.trainer.building_DRGVLM_PPTrainer import DRGVLM_PPTrainer


DX_ITEMS = [
    "Histologic_Type",
    "Histologic_Grade",
    "Microcalcification",
]


def _case(global_idx=1):
    return SimpleNamespace(
        global_idx=global_idx,
        case_id=f"case-{global_idx}",
        DxItem_targets={
            "Histologic_Type": "Invasive ductal carcinoma",
            "Histologic_Grade": (
                "Grade: 1; Tubular formation: 1; Nuclear pleomorphism: 1; "
                "Mitotic count: 1; Total score: 3/9"
            ),
            "Microcalcification": "Present in DCIS",
        },
        DxItem_target_classes={
            "Histologic_Type": "Invasive breast carcinoma of no special type",
            "Histologic_Grade": "Grade I",
            "Microcalcification": "Present",
            "Tubular_formation": "Score 1",
            "Nuclear_pleomorphism": "Score 1",
            "Mitotic_count": "Score 1",
        },
    )


class EvaluatorCompletenessTests(unittest.TestCase):
    def test_sparse_case_requires_only_dxitems_with_assigned_rois(self):
        case = _case()
        case.DxItem_rois = {
            "Histologic_Type": [SimpleNamespace(global_idx=10)],
        }
        evaluator = DRGVLMEvaluator(
            DxItem_list=DX_ITEMS,
            strict_prediction_completeness=True,
        )
        evaluator.update(
            [case],
            {
                case.global_idx: {
                    "Histologic_Type": {
                        "pred_txt": "Invasive ductal carcinoma",
                    },
                },
            },
        )
        _, summary = evaluator._evaluate()
        completeness = summary["prediction_completeness"]
        self.assertEqual(completeness["expected_prediction_count"], 1)
        self.assertEqual(completeness["missing_prediction_count"], 0)
        self.assertIsNone(
            summary["per_DxItem"]["Histologic_Grade"]["exact_match"]
        )

    def test_strict_mode_rejects_missing_predictions(self):
        evaluator = DRGVLMEvaluator(
            DxItem_list=DX_ITEMS,
            strict_prediction_completeness=True,
        )
        outputs = {
            1: {
                "Histologic_Type": {
                    "pred_txt": "Invasive ductal carcinoma",
                },
            },
        }
        with self.assertRaisesRegex(ValueError, "missing=2"):
            evaluator.update([_case()], outputs)
        self.assertEqual(evaluator.cases_list, [])

    def test_strict_mode_rejects_empty_predictions(self):
        evaluator = DRGVLMEvaluator(
            DxItem_list=DX_ITEMS,
            strict_prediction_completeness=True,
        )
        outputs = {
            1: {
                dx_item: {"pred_txt": "   "}
                for dx_item in DX_ITEMS
            },
        }
        with self.assertRaisesRegex(ValueError, "empty=3"):
            evaluator.update([_case()], outputs)

    def test_lenient_mode_scores_missing_predictions_as_zero(self):
        evaluator = DRGVLMEvaluator(
            DxItem_list=DX_ITEMS,
            strict_prediction_completeness=False,
        )
        evaluator.update(
            [_case()],
            {
                1: {
                    "Histologic_Type": {
                        "pred_txt": "Invasive ductal carcinoma",
                    },
                },
            },
        )
        _, summary = evaluator._evaluate()

        completeness = summary["prediction_completeness"]
        self.assertEqual(completeness["expected_prediction_count"], 3)
        self.assertEqual(completeness["present_prediction_count"], 1)
        self.assertEqual(completeness["missing_prediction_count"], 2)
        self.assertEqual(
            summary["per_DxItem"]["Histologic_Grade"]["exact_match"],
            0.0,
        )
        self.assertAlmostEqual(summary["macro_exact_match"], 1.0 / 3.0)

    def test_clinical_macro_gives_each_dx_item_equal_weight(self):
        evaluator = DRGVLMEvaluator(DxItem_list=DX_ITEMS)
        evaluator.update(
            [_case()],
            {
                1: {
                    "Histologic_Type": {
                        "pred_txt": "Invasive ductal carcinoma",
                    },
                    "Histologic_Grade": {
                        "pred_txt": (
                            "Grade: 2; Tubular formation: 2; "
                            "Nuclear pleomorphism: 2; Mitotic count: 2; "
                            "Total score: 6/9"
                        ),
                    },
                    "Microcalcification": {
                        "pred_txt": "Present in invasive carcinoma",
                    },
                },
            },
        )
        _, summary = evaluator._evaluate()

        self.assertEqual(
            summary["clinical_task_scores"]["Histologic_Type"],
            1.0,
        )
        self.assertEqual(
            summary["clinical_task_scores"]["Histologic_Grade"],
            0.0,
        )
        self.assertEqual(
            summary["clinical_task_scores"]["Microcalcification"],
            0.5,
        )
        self.assertAlmostEqual(summary["clinical_macro_score"], 0.5)

    def test_grade_reference_uses_classes_not_reference_text(self):
        case = _case()
        case.DxItem_targets["Histologic_Grade"] = (
            "Grade: 3; Tubular formation: 3; Nuclear pleomorphism: 3; "
            "Mitotic count: 3; Total score: 9/9"
        )
        evaluator = DRGVLMEvaluator(DxItem_list=DX_ITEMS)
        evaluator.update([case], {
            1: {
                "Histologic_Type": {
                    "pred_txt": "Invasive ductal carcinoma",
                },
                # No explicit grade: schema v3 derives Grade I from total 3.
                "Histologic_Grade": {
                    "pred_txt": (
                        "Tubular formation: 1; Nuclear pleomorphism: 1; "
                        "Mitotic count: 1; Total score: 3/9"
                    ),
                },
                "Microcalcification": {
                    "pred_txt": "Present in DCIS",
                },
            },
        })
        cases, summary = evaluator._evaluate()
        grade = cases[0]["DxItem_dict"]["Histologic_Grade"]["structured"]
        self.assertEqual(grade["reference"]["grade"], 1)
        self.assertTrue(grade["prediction"]["grade_was_derived"])
        self.assertEqual(
            summary["clinical"]["Histologic_Grade"][
                "complete_record_accuracy"
            ],
            1.0,
        )

    def test_grade_prediction_conflict_is_reported(self):
        evaluator = DRGVLMEvaluator(DxItem_list=DX_ITEMS)
        evaluator.update([_case()], {
            1: {
                "Histologic_Type": {
                    "pred_txt": "Invasive ductal carcinoma",
                },
                "Histologic_Grade": {
                    "pred_txt": (
                        "Grade: 3; Tubular formation: 1; "
                        "Nuclear pleomorphism: 1; Mitotic count: 1; "
                        "Total score: 3/9"
                    ),
                },
                "Microcalcification": {
                    "pred_txt": "Present in DCIS",
                },
            },
        })
        _, summary = evaluator._evaluate()
        self.assertEqual(summary["grade_prediction_conflict_count"], 1)


class ClinicalParserTests(unittest.TestCase):
    def test_in_situ_is_not_canonicalized_as_invasive(self):
        dcis = _parse_histologic_type("Ductal carcinoma in situ")
        invasive = _parse_histologic_type("Invasive ductal carcinoma")
        idc = _parse_histologic_type("IDC")
        invasive_with_dcis = _parse_histologic_type(
            "Invasive ductal carcinoma with associated DCIS"
        )
        nst = _parse_histologic_type(
            "Invasive carcinoma of no special type (ductal)"
        )

        self.assertEqual(dcis["canonical"], "ductal carcinoma in situ")
        self.assertEqual(invasive["canonical"], "invasive duct carcinoma")
        self.assertEqual(idc["canonical"], "invasive duct carcinoma")
        self.assertEqual(
            invasive_with_dcis["canonical"],
            "invasive duct carcinoma",
        )
        self.assertEqual(nst["canonical"], "invasive duct carcinoma")
        self.assertNotEqual(dcis["canonical"], invasive["canonical"])

    def test_microcalcification_statuses_remain_distinct(self):
        absent = _parse_microcalcification("Absent")
        not_identified = _parse_microcalcification("Not identified")
        self.assertEqual(absent["status"], "absent")
        self.assertEqual(not_identified["status"], "not_identified")


class PipelinePlacementTests(unittest.TestCase):
    @staticmethod
    def _config():
        return SimpleNamespace(
            text_config=SimpleNamespace(num_hidden_layers=34),
            vision_config=SimpleNamespace(num_hidden_layers=27),
        )

    def test_four_gpu_drgvlm_reserves_gpu0_for_non_transformers(self):
        with patch(
            "transformers.AutoConfig.from_pretrained",
            return_value=self._config(),
        ):
            device_map = modelBuilder._build_manual_pp_device_map(
                model_path="/unused",
                model_name="medgemma-1.5-4b-it",
                num_gpus=4,
                project_name="DRGVLM",
            )
        layer_devices = [
            device_map[f"model.language_model.layers.{idx}"]
            for idx in range(34)
        ]
        self.assertEqual(
            [layer_devices.count(f"cuda:{idx}") for idx in range(4)],
            [0, 11, 11, 12],
        )
        self.assertEqual(device_map["model.vision_tower"], "cuda:0")
        self.assertEqual(device_map["model.language_model.norm"], "cuda:0")

    def test_eight_gpu_drgvlm_splits_vision_and_reserves_gpu0_gpu1(self):
        with patch(
            "transformers.AutoConfig.from_pretrained",
            return_value=self._config(),
        ):
            device_map = modelBuilder._build_manual_pp_device_map(
                model_path="/unused",
                model_name="medgemma-1.5-4b-it",
                num_gpus=8,
                project_name="DRGVLM",
                vision_split_index=14,
            )
        layer_devices = [
            device_map[f"model.language_model.layers.{idx}"]
            for idx in range(34)
        ]
        self.assertEqual(
            [layer_devices.count(f"cuda:{idx}") for idx in range(8)],
            [0, 0, 5, 5, 6, 6, 6, 6],
        )
        self.assertEqual(
            device_map["model.vision_tower.encoder.layers.13"],
            "cuda:0",
        )
        self.assertEqual(
            device_map["model.vision_tower.encoder.layers.14"],
            "cuda:1",
        )
        self.assertEqual(device_map["lm_head"], "cuda:1")


class FrozenBaseModeTests(unittest.TestCase):
    class LoRALikeLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.base_dropout = torch.nn.Dropout(0.5)
            self.lora_dropout = torch.nn.ModuleDict({
                "default": torch.nn.Dropout(0.25),
            })

    def test_only_lora_dropout_enters_train_mode(self):
        model = DownstreamRepGenVLM(device="cpu")
        model.vlm_model = self.LoRALikeLayer()
        model.train(True)
        self.assertFalse(model.vlm_model.training)
        self.assertFalse(model.vlm_model.base_dropout.training)
        self.assertTrue(
            model.vlm_model.lora_dropout["default"].training
        )


class SharedVisionGenerationTests(unittest.TestCase):
    class FakeGenerator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []
            self.attention_masks = []

        def generate(self, **kwargs):
            inputs_embeds = kwargs["inputs_embeds"]
            self.batch_sizes.append(inputs_embeds.shape[0])
            self.attention_masks.append(
                kwargs["attention_mask"].detach().clone()
            )
            return torch.tensor(
                [[10 + row] for row in range(inputs_embeds.shape[0])],
                dtype=torch.long,
            )

    class FakeTokenizer:
        pad_token_id = 0

        @staticmethod
        def decode(token_ids, skip_special_tokens=True):
            return f"prediction-{int(token_ids[0])}"

    def test_prompt_contains_only_rois_assigned_to_requested_dxitem(self):
        model = DownstreamRepGenVLM(device="cpu")
        model.sep_str = "<sep>"
        model.boc_str = "<boc>"
        roi_0 = SimpleNamespace(
            global_idx=0,
            image=None,
            mpp=None,
            cxcywh=None,
        )
        roi_1 = SimpleNamespace(
            global_idx=1,
            image=None,
            mpp=None,
            cxcywh=None,
        )
        case = SimpleNamespace(
            case_id="case-prompts",
            DxItem_rois={
                "DxA": [roi_0, roi_1],
                "DxB": [roi_1],
            },
        )
        content_a = model._build_user_content(case, "DxA")
        content_b = model._build_user_content(case, "DxB")
        self.assertEqual(len(content_a), 3)
        self.assertEqual(len(content_b), 2)

    def test_only_identical_ordered_roi_groups_share_a_cache_group(self):
        roi_0 = SimpleNamespace(global_idx=0)
        roi_1 = SimpleNamespace(global_idx=1)
        case = SimpleNamespace(
            case_id="case-signatures",
            DxItem_rois={
                "DxA": [roi_0, roi_1],
                "DxB": [roi_0, roi_1],
                "DxC": [roi_1, roi_0],
            },
        )
        self.assertEqual(
            DownstreamRepGenVLM._group_dxitems_by_roi_signature(case),
            [["DxA", "DxB"], ["DxC"]],
        )

    def test_six_prompts_share_one_batched_generate_call(self):
        model = DownstreamRepGenVLM(device="cpu")
        model.vlm_model = self.FakeGenerator()
        model.vlm_processor = SimpleNamespace(
            tokenizer=self.FakeTokenizer(),
        )
        text_backbone = SimpleNamespace(
            embed_tokens=torch.nn.Embedding(32, 4),
        )
        object.__setattr__(model, "text_backbone", text_backbone)

        def suffix_tokens(self, case, DxItem, expected_q_start):
            suffix_len = 2 + (int(DxItem[-1]) % 2)
            return (
                torch.ones(1, suffix_len, dtype=torch.long),
                torch.ones(1, suffix_len, dtype=torch.long),
                None,
            )

        model._build_inference_suffix_tokens = types.MethodType(
            suffix_tokens,
            model,
        )
        case = SimpleNamespace(
            case_id="case-batch",
            DxItem_targets={
                f"DxItem{idx}": f"target-{idx}"
                for idx in range(6)
            },
        )
        context = {
            "vision_prefix_embeds": torch.ones(1, 3, 4),
            "q_start_in_ids": 7,
            "vision_prefix_attn_mask": torch.ones(
                1,
                3,
                dtype=torch.long,
            ),
            "vision_prefix_token_type_ids": None,
        }
        outputs = model._generate_case_with_vision_cache(
            case=case,
            dx_items=list(case.DxItem_targets),
            case_context=context,
            max_new_tokens=4,
            prompt_batch_size=6,
        )
        self.assertEqual(model.vlm_model.batch_sizes, [6])
        self.assertEqual(len(outputs), 6)
        # Left padding is before the shared prefix; every prompt ends unmasked.
        self.assertTrue(
            torch.all(model.vlm_model.attention_masks[0][:, -1] == 1)
        )


class CausalAttentionTests(unittest.TestCase):
    def test_forward_and_backward_match_sdpa(self):
        torch.manual_seed(7)
        q = torch.randn(1, 2, 5, 8, requires_grad=True)
        k = torch.randn(1, 2, 5, 8, requires_grad=True)
        v = torch.randn(1, 2, 5, 8, requires_grad=True)
        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)

        actual = CausalBlockAttention.apply(
            q,
            k,
            v,
            True,
            None,
            -1,
            -1,
            2,
        )
        expected = F.scaled_dot_product_attention(
            q_ref,
            k_ref,
            v_ref,
            is_causal=True,
        )
        self.assertTrue(torch.allclose(actual, expected, atol=2e-5, rtol=2e-5))

        grad = torch.randn_like(actual)
        actual.backward(grad)
        expected.backward(grad)
        for actual_grad, expected_grad in (
            (q.grad, q_ref.grad),
            (k.grad, k_ref.grad),
            (v.grad, v_ref.grad),
        ):
            self.assertTrue(
                torch.allclose(
                    actual_grad,
                    expected_grad,
                    atol=4e-5,
                    rtol=4e-5,
                )
            )

    def test_bottom_right_causal_alignment_for_unequal_lengths(self):
        torch.manual_seed(9)
        q = torch.randn(1, 1, 2, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)

        actual = CausalBlockAttention.apply(
            q,
            k,
            v,
            True,
            None,
            -1,
            -1,
            2,
        )
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(8)
        # Bottom-right alignment gives query rows absolute positions 2 and 3.
        mask = torch.tensor(
            [[[[True, True, True, False], [True, True, True, True]]]]
        )
        probabilities = torch.softmax(
            scores.masked_fill(~mask, float("-inf")),
            dim=-1,
        )
        expected = torch.matmul(probabilities, v)
        self.assertTrue(torch.allclose(actual, expected, atol=2e-5, rtol=2e-5))

    def test_installed_wrapper_propagates_causal_argument(self):
        def native_flash(
            q,
            k,
            v,
            softmax_scale=None,
            causal=False,
            window_size=(-1, -1),
        ):
            raise AssertionError("native function should not handle head_dim > 256")

        def native_varlen(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale=None,
            causal=False,
            window_size=(-1, -1),
        ):
            raise AssertionError("native function should not handle head_dim > 256")

        interface = types.SimpleNamespace(
            flash_attn_func=native_flash,
            flash_attn_varlen_func=native_varlen,
        )
        self.assertTrue(install_large_head_flash_attention_patch(interface))

        torch.manual_seed(11)
        q = torch.randn(1, 4, 1, 257)
        k = torch.randn(1, 4, 1, 257)
        v = torch.randn(1, 4, 1, 257)
        actual = interface.flash_attn_func(q, k, v, causal=True)
        expected = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            is_causal=True,
        ).transpose(1, 2)
        self.assertTrue(torch.allclose(actual, expected, atol=3e-5, rtol=3e-5))
        q_varlen = torch.randn(6, 1, 257)
        k_varlen = torch.randn(6, 1, 257)
        v_varlen = torch.randn(6, 1, 257)
        cu_seqlens = torch.tensor([0, 2, 6], dtype=torch.int32)
        actual_varlen = interface.flash_attn_varlen_func(
            q_varlen,
            k_varlen,
            v_varlen,
            cu_seqlens,
            cu_seqlens,
            4,
            4,
            causal=True,
        )
        expected_sequences = []
        for start, end in ((0, 2), (2, 6)):
            expected_sequences.append(
                F.scaled_dot_product_attention(
                    q_varlen[start:end].transpose(0, 1).unsqueeze(0),
                    k_varlen[start:end].transpose(0, 1).unsqueeze(0),
                    v_varlen[start:end].transpose(0, 1).unsqueeze(0),
                    is_causal=True,
                ).squeeze(0).transpose(0, 1)
            )
        expected_varlen = torch.cat(expected_sequences, dim=0)
        self.assertTrue(
            torch.allclose(
                actual_varlen,
                expected_varlen,
                atol=3e-5,
                rtol=3e-5,
            )
        )



class DxPairAwareDatasetTests(unittest.TestCase):
    @staticmethod
    def _metadata(dx_pairs=None):
        dx_pairs = (
            [
                {"DxA": None, "DxB": None},
                {"DxA": None},
                {"DxA": None},
                {"DxB": None},
            ]
            if dx_pairs is None
            else dx_pairs
        )
        rois = [
            {
                "global_idx": roi_idx,
                "DxPair": dx_pair,
                "main_info": {
                    "roi_path": None,
                    "mpp": 1.0,
                    "roi_wh": [10, 10],
                    "cxcywh": [0.5, 0.5, 0.1, 0.1],
                },
            }
            for roi_idx, dx_pair in enumerate(dx_pairs)
        ]
        return {
            "DxItem_list": ["DxA", "DxB", "DxWithoutROI"],
            "case_list": [{
                "sample_idx": 7,
                "case_id": "case-7",
                "tissue_blocks": [{
                    "stains": [{"roi_list": rois}],
                }],
                "structured_report": {
                    "DxItems": {
                        dx_item: {
                            "DxResultTxt": f"target-{dx_item}",
                            "DxResultCls": f"class-{dx_item}",
                        }
                        for dx_item in (
                            "DxA",
                            "DxB",
                            "DxWithoutROI",
                        )
                    },
                },
            }],
        }

    @staticmethod
    def _dataset(directory, metadata, **kwargs):
        metadata_path = Path(directory) / "metadata.json"
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        return multiROI2DxResultDataset(
            image_path=directory,
            metadata_path=str(metadata_path),
            split="valid",
            input_img=False,
            input_loc=False,
            max_rois_per_dxitem=2,
            roi_sampling_mode="head_k",
            **kwargs,
        )

    def test_dxpair_keys_control_per_dxitem_sampling_and_forward_membership(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self._dataset(directory, self._metadata())
            case = dataset[0]

        self.assertEqual(
            [roi.global_idx for roi in case.DxItem_rois["DxA"]],
            [0, 1],
        )
        self.assertEqual(
            [roi.global_idx for roi in case.DxItem_rois["DxB"]],
            [0, 3],
        )
        self.assertNotIn("DxWithoutROI", case.DxItem_rois)
        self.assertIn("DxWithoutROI", case.DxItem_targets)
        self.assertEqual(
            [roi.global_idx for roi in case.rois],
            [0, 1, 3],
        )
        self.assertEqual(dataset.get_raw_roi_count(0), 5)
        self.assertEqual(dataset.get_effective_roi_count(0), 4)

    def test_invalid_dxpair_relationships_fail_during_dataset_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = self._metadata(dx_pairs=[{"Undeclared": None}])
            with self.assertRaisesRegex(ValueError, "undeclared DxItem"):
                self._dataset(directory, metadata)

        with tempfile.TemporaryDirectory() as directory:
            metadata = self._metadata(dx_pairs=[None, {}, None])
            with self.assertRaisesRegex(ValueError, "no active DxItem"):
                self._dataset(directory, metadata)

    def test_legacy_roi_limit_alias_conflict_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "metadata.json"
            metadata_path.write_text(
                json.dumps(self._metadata()),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Conflicting ROI limits"):
                multiROI2DxResultDataset(
                    image_path=directory,
                    metadata_path=str(metadata_path),
                    split="valid",
                    input_img=False,
                    max_rois_per_dxitem=2,
                    max_rois_per_case=3,
                )


class ROIBatchSamplerTests(unittest.TestCase):
    def test_budget_is_applied_per_yielded_batch(self):
        roi_counts = [40, 30, 20]
        sampler = MaxROIBatchSampler(
            sampler=[0, 1, 2],
            roi_count_func=lambda index: roi_counts[index],
            batch_size=2,
            max_rois_per_batch=60,
        )
        batches = list(sampler)
        self.assertEqual(batches, [[0, 2], [1]])
        for batch in batches:
            self.assertLessEqual(sum(roi_counts[index] for index in batch), 60)

    def test_legacy_name_is_accepted_but_conflicts_are_rejected(self):
        sampler = MaxROIBatchSampler(
            sampler=[0],
            roi_count_func=lambda _: 10,
            batch_size=1,
            max_rois_per_update=20,
        )
        self.assertEqual(sampler.max_rois_per_batch, 20)
        with self.assertRaisesRegex(ValueError, "Conflicting ROI budgets"):
            MaxROIBatchSampler(
                sampler=[0],
                roi_count_func=lambda _: 10,
                batch_size=1,
                max_rois_per_batch=20,
                max_rois_per_update=30,
            )


class _StreamingPairLossModel(torch.nn.Module):
    """Small model that fails if a second pair is forwarded before backward."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.use_shared_vision_cache = False
        self.prepared_case_ids = []
        self.forwarded_pairs = []
        self.backward_pairs = []

    @staticmethod
    def _active_dxitems(case):
        return DownstreamRepGenVLM._active_dxitems(case)

    @staticmethod
    def _get_dxitem_rois(case, DxItem):
        return DownstreamRepGenVLM._get_dxitem_rois(case, DxItem)

    @staticmethod
    def _group_dxitems_by_roi_signature(case, dx_items=None):
        return DownstreamRepGenVLM._group_dxitems_by_roi_signature(
            case,
            dx_items=dx_items,
        )

    def prepare_case_loss_context(self, case, dx_items=None):
        self.prepared_case_ids.append(case.case_id)
        return None

    def calculate_dxitem_loss(self, case, DxItem, case_context=None):
        if len(self.forwarded_pairs) != len(self.backward_pairs):
            raise AssertionError(
                "The previous DxItem graph was not backpropagated before "
                "the next DxItem forward."
            )
        self.assert_context(case=case, case_context=case_context)
        pair = (case.case_id, DxItem)
        self.forwarded_pairs.append(pair)
        loss = self.weight * float(case.loss_coefficients[DxItem])
        loss.register_hook(
            lambda grad, pair=pair: self.backward_pairs.append(pair)
        )
        return {"total_loss": loss}

    @staticmethod
    def assert_context(case, case_context):
        if case_context is not None:
            raise AssertionError("Unexpected cached context for uncached model.")


class StreamingDxItemBackwardTests(unittest.TestCase):
    @staticmethod
    def _case(case_id, loss_coefficients):
        shared_roi = SimpleNamespace(global_idx=0)
        return SimpleNamespace(
            case_id=case_id,
            rois=[shared_roi],
            DxItem_rois={
                dx_item: [shared_roi]
                for dx_item in loss_coefficients
            },
            DxItem_targets={
                dx_item: f"target-{dx_item}"
                for dx_item in loss_coefficients
            },
            loss_coefficients=dict(loss_coefficients),
        )

    @staticmethod
    def _trainer(model):
        return DRGVLM_PPTrainer(
            model=model,
            device="cpu",
            num_epochs=1,
            save_path=".",
            accumulation_steps=1,
            amp=False,
        )

    def test_each_dxitem_is_backpropagated_before_the_next_forward(self):
        model = _StreamingPairLossModel()
        trainer = self._trainer(model)
        case = self._case(
            "case-a",
            {
                "Histologic_Type": 1.0,
                "Histologic_Grade": 2.0,
                "Microcalcification": 3.0,
            },
        )

        loss_dict, dx_count = trainer._backward_train_batch(
            batch_cases=[case],
            accumulation_denom=2,
        )

        self.assertEqual(dx_count, 3)
        self.assertEqual(model.prepared_case_ids, ["case-a"])
        self.assertEqual(model.forwarded_pairs, model.backward_pairs)
        self.assertAlmostEqual(loss_dict["total_loss"], 4.0)
        # sum([1, 2, 3]) / (3 DxItems * accumulation denominator 2)
        self.assertAlmostEqual(float(model.weight.grad), 1.0)

    def test_streaming_scaling_preserves_equal_case_weighting(self):
        model = _StreamingPairLossModel()
        trainer = self._trainer(model)
        cases = [
            self._case("case-a", {"only": 1.0}),
            self._case("case-b", {"first": 3.0, "second": 5.0}),
        ]

        loss_dict, dx_count = trainer._backward_train_batch(
            batch_cases=cases,
            accumulation_denom=1,
        )

        self.assertEqual(dx_count, 3)
        self.assertEqual(model.prepared_case_ids, ["case-a", "case-b"])
        # Raw logging remains mean(case mean losses):
        # mean([2*1, mean(2*3, 2*5)]) == mean([2, 8]) == 5.
        self.assertAlmostEqual(loss_dict["total_loss"], 5.0)
        # Gradient preserves equal case weighting:
        # 1/2 + mean([3, 5])/2 == 2.5.
        self.assertAlmostEqual(float(model.weight.grad), 2.5)


class _LoRALikeVLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter_scale = torch.nn.Parameter(torch.tensor(2.0))
        self.last_input_requires_grad = None

    def forward(self, inputs_embeds, **kwargs):
        self.last_input_requires_grad = inputs_embeds.requires_grad
        return SimpleNamespace(
            logits=inputs_embeds * self.adapter_scale
        )


class _SumCriterion(torch.nn.Module):
    def forward(self, logits, labels, context=""):
        return {"total_loss": logits.sum()}


class CachedInputGradientPolicyTests(unittest.TestCase):
    @staticmethod
    def _model():
        model = DownstreamRepGenVLM(device="cpu")
        model.use_shared_vision_cache = True
        model.vlm_model = _LoRALikeVLM()
        model.criterion = _SumCriterion()

        def build_cached_inputs(self, **kwargs):
            combined_embeds = torch.ones(1, 2, 3)
            attention_mask = torch.ones(1, 2, dtype=torch.long)
            labels = torch.zeros(1, 2, dtype=torch.long)
            return combined_embeds, attention_mask, None, labels

        model.build_train_inputs_with_vision_cache = types.MethodType(
            build_cached_inputs,
            model,
        )
        return model

    @staticmethod
    def _case_context():
        return {
            "vision_prefix_embeds": torch.zeros(1),
            "q_start_in_ids": 0,
            "vision_prefix_len_merged": 0,
            "vision_prefix_attn_mask": torch.zeros(1),
            "vision_prefix_token_type_ids": None,
        }

    def test_lora_like_parameter_gets_gradient_without_input_leaf(self):
        model = self._model()
        case = SimpleNamespace(
            case_id="case-a",
            DxItem_rois={
                "Histologic_Type": [SimpleNamespace(global_idx=0)],
            },
            DxItem_targets={"Histologic_Type": "target"},
        )

        loss_dict = model.calculate_dxitem_loss(
            case=case,
            DxItem="Histologic_Type",
            case_context=self._case_context(),
        )
        self.assertFalse(model.vlm_model.last_input_requires_grad)
        loss_dict["total_loss"].backward()
        self.assertIsNotNone(model.vlm_model.adapter_scale.grad)
        self.assertNotEqual(float(model.vlm_model.adapter_scale.grad), 0.0)

    def test_input_leaf_is_created_only_when_explicitly_required(self):
        model = self._model()
        model.require_cached_input_grad = True
        case = SimpleNamespace(
            case_id="case-a",
            DxItem_rois={
                "Histologic_Type": [SimpleNamespace(global_idx=0)],
            },
            DxItem_targets={"Histologic_Type": "target"},
        )

        loss_dict = model.calculate_dxitem_loss(
            case=case,
            DxItem="Histologic_Type",
            case_context=self._case_context(),
        )
        self.assertTrue(model.vlm_model.last_input_requires_grad)
        loss_dict["total_loss"].backward()
        self.assertIsNotNone(model.vlm_model.adapter_scale.grad)


class _CheckpointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter_weight = torch.nn.Parameter(torch.tensor([1.5]))

    def save_lora_weights(self, path):
        torch.save(
            {"adapter_weight": self.adapter_weight.detach().clone()},
            Path(path) / "adapter.pt",
        )

    def load_lora_weights(self, path):
        state = torch.load(
            Path(path) / "adapter.pt",
            map_location="cpu",
            weights_only=False,
        )
        self.adapter_weight.data.copy_(state["adapter_weight"])


class TrainerCheckpointStateTests(unittest.TestCase):
    def _trainer(self, save_path):
        model = torch.nn.Linear(1, 1)
        return DRGVLM_PPTrainer(
            model=model,
            device="cpu",
            num_epochs=1,
            save_path=str(save_path),
            early_stop_patience=20,
        )

    def test_validation_improvement_resets_patience_before_save(self):
        trainer = self._trainer(".")
        trainer.best_val_loss = 2.0
        trainer.patience_counter = 7

        self.assertTrue(trainer._record_validation_result(1.0))
        self.assertEqual(trainer.best_val_loss, 1.0)
        self.assertEqual(trainer.patience_counter, 0)

    def test_immutable_snapshot_is_reused_and_round_trips(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            model = _CheckpointModel()
            trainer = DRGVLM_PPTrainer(
                model=model,
                device="cpu",
                num_epochs=2,
                save_path=checkpoint_dir,
                trainer_mode="keepTrain",
            )
            trainer.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=1e-3,
            )
            trainer.scheduler = torch.optim.lr_scheduler.StepLR(
                trainer.optimizer,
                step_size=1,
            )
            best_ref = trainer.save_checkpoint(
                epoch_idx=0,
                weight_filename="best_model.pth",
            )
            latest_ref = trainer.save_checkpoint(
                epoch_idx=0,
                weight_filename="latest_model.pth",
            )
            best_payload = json.loads(Path(best_ref).read_text())
            latest_payload = json.loads(Path(latest_ref).read_text())
            self.assertEqual(
                best_payload["snapshot"],
                latest_payload["snapshot"],
            )
            snapshots = list(
                (Path(checkpoint_dir) / "checkpoint_artifacts").iterdir()
            )
            self.assertEqual(len(snapshots), 1)

            model.adapter_weight.data.fill_(9.0)
            trainer.load_checkpoint(latest_ref)
            self.assertAlmostEqual(
                float(model.adapter_weight.item()),
                1.5,
            )

    def test_resume_start_does_not_write_or_overwrite_best(self):
        trainer = self._trainer(".")
        trainer.resume_checkpoint_path = str(
            Path(".").resolve() / "latest_model.pth"
        )
        trainer.save_checkpoint = Mock()

        trainer._prepare_run_checkpoint_artifacts(checkpoint_epoch_idx=42)
        trainer.save_checkpoint.assert_not_called()

    def test_new_resume_directory_records_historical_sources(self):
        with tempfile.TemporaryDirectory() as source_dir:
            with tempfile.TemporaryDirectory() as destination_dir:
                trainer = self._trainer(destination_dir)
                trainer.resume_checkpoint_path = str(
                    Path(source_dir) / "latest_model.pth"
                )
                trainer.best_val_loss = 0.25
                trainer.best_metric_values = {
                    "best_text_composite.pth": {
                        "score": 0.9,
                        "epoch_idx": 3,
                    },
                }
                trainer.save_checkpoint = Mock()

                trainer._prepare_run_checkpoint_artifacts(
                    checkpoint_epoch_idx=4
                )

                trainer.save_checkpoint.assert_not_called()
                manifest_path = (
                    Path(destination_dir) / "resume_source_checkpoints.json"
                )
                manifest = json.loads(manifest_path.read_text())
                self.assertEqual(manifest["historical_best_val_loss"], 0.25)
                self.assertEqual(
                    manifest["historical_metric_checkpoints"][
                        "best_text_composite.pth"
                    ]["state"]["score"],
                    0.9,
                )
                self.assertFalse(
                    (Path(destination_dir) / "[trainer]best_model.pth").exists()
                )

    def test_legacy_clinical_metric_score_is_safely_migrated(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            case = _case(global_idx=0)
            stored_case = {
                "global_idx": case.global_idx,
                "case_id": case.case_id,
                "DxItem_target_classes": case.DxItem_target_classes,
                "DxItem_dict": {
                    dx_item: {
                        "pred_txt": target,
                        "gt_txt": target,
                    }
                    for dx_item, target in case.DxItem_targets.items()
                },
            }
            eval_path = Path(checkpoint_dir) / "eval_results[000]_test.json"
            eval_path.write_text(json.dumps({
                "epoch_idx": 0,
                "cases": [stored_case],
            }))

            trainer = self._trainer(checkpoint_dir)
            trainer.init_evaluator(DRGVLMEvaluator(DX_ITEMS))
            trainer.best_metric_values = {
                "best_clinical_composite.pth": {
                    "score": 0.25,
                    "epoch_idx": 0,
                },
            }
            trainer._migrate_clinical_metric_state(checkpoint_dir)

            migrated = trainer.best_metric_values[
                "best_clinical_composite.pth"
            ]
            self.assertEqual(
                migrated["metric_schema_version"],
                DRGVLMEvaluator.CLINICAL_METRIC_SCHEMA_VERSION,
            )
            self.assertEqual(migrated["legacy_score"], 0.25)
            self.assertEqual(migrated["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
