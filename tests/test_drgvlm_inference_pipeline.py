import hashlib
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock

from DRepGenVLM.datasets.multiROI2DxResultDataset import (
    multiROI2DxResultDataset,
)
from DRepGenVLM.pipeline.drgvlmInferencePipeline import (
    DRGVLMInferencePipeline,
    apply_generated_reports,
)


def _metadata():
    return {
        "schema_version": "test",
        "DxItem_list": ["DxA", "DxB", "DxWithoutROI"],
        "case_list": [{
            "sample_idx": 7,
            "case_id": "case-7",
            "memo": "must be preserved",
            "tissue_blocks": [{
                "stains": [{
                    "roi_list": [{
                        "global_idx": 10,
                        "DxPair": {
                            "DxA": None,
                            "DxB": None,
                        },
                        "main_info": {
                            "roi_path": None,
                            "mpp": 1.0,
                            "roi_wh": [10, 10],
                            "cxcywh": [0.5, 0.5, 0.1, 0.1],
                        },
                    }],
                }],
            }],
        }],
    }


class UnlabeledDatasetTests(unittest.TestCase):
    def test_inference_mode_accepts_missing_structured_report(self):
        variants = (
            ("absent", None),
            ("null", None),
            ("empty_report", {}),
            ("empty_dxitems", {"DxItems": {}}),
        )
        for variant_name, structured_report in variants:
            with self.subTest(variant=variant_name):
                metadata = _metadata()
                if variant_name != "absent":
                    metadata["case_list"][0]["structured_report"] = (
                        structured_report
                    )
                with tempfile.TemporaryDirectory() as directory:
                    metadata_path = Path(directory) / "metadata.json"
                    metadata_path.write_text(
                        json.dumps(metadata),
                        encoding="utf-8",
                    )
                    dataset = multiROI2DxResultDataset(
                        image_path=directory,
                        metadata_path=str(metadata_path),
                        split="valid",
                        input_img=False,
                        input_loc=False,
                        require_targets=False,
                    )
                    case = dataset[0]

                self.assertEqual(
                    list(case.DxItem_rois),
                    ["DxA", "DxB"],
                )
                self.assertEqual(case.DxItem_targets, {})
                self.assertEqual(case.DxItem_target_classes, {})

    def test_training_mode_still_requires_structured_report(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata_path = Path(directory) / "metadata.json"
            metadata_path.write_text(
                json.dumps(_metadata()),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "structured_report must be an object",
            ):
                multiROI2DxResultDataset(
                    image_path=directory,
                    metadata_path=str(metadata_path),
                    split="valid",
                    input_img=False,
                    input_loc=False,
                )


class GeneratedReportTests(unittest.TestCase):
    def test_applies_only_active_dxitems_and_preserves_metadata(self):
        metadata = _metadata()
        count = apply_generated_reports(
            metadata=metadata,
            predictions={
                7: {
                    "DxA": {"pred_txt": " result A "},
                    "DxB": {"pred_txt": "result B"},
                },
            },
        )

        self.assertEqual(count, 2)
        self.assertEqual(
            metadata["case_list"][0]["generated_report"],
            {
                "DxA": "result A",
                "DxB": "result B",
            },
        )
        self.assertNotIn(
            "DxWithoutROI",
            metadata["case_list"][0]["generated_report"],
        )
        self.assertEqual(
            metadata["case_list"][0]["memo"],
            "must be preserved",
        )
        self.assertNotIn("structured_report", metadata["case_list"][0])

    def test_rejects_existing_report_without_overwrite(self):
        metadata = _metadata()
        metadata["case_list"][0]["generated_report"] = {
            "DxA": "old result",
        }
        with self.assertRaisesRegex(
            FileExistsError,
            "already contains",
        ):
            apply_generated_reports(
                metadata=metadata,
                predictions={
                    7: {
                        "DxA": {"pred_txt": "new A"},
                        "DxB": {"pred_txt": "new B"},
                    },
                },
            )

    def test_rejects_missing_or_empty_predictions(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            apply_generated_reports(
                metadata=_metadata(),
                predictions={
                    7: {
                        "DxA": {"pred_txt": "result A"},
                    },
                },
            )
        with self.assertRaisesRegex(ValueError, "Empty prediction"):
            apply_generated_reports(
                metadata=_metadata(),
                predictions={
                    7: {
                        "DxA": {"pred_txt": ""},
                        "DxB": {"pred_txt": "result B"},
                    },
                },
            )


class PipelinePreflightTests(unittest.TestCase):
    @staticmethod
    def _write_checkpoint(checkpoint_dir: Path) -> None:
        snapshot = (
            checkpoint_dir
            / "checkpoint_artifacts"
            / "epoch_000001"
        )
        adapter = snapshot / "adapter"
        adapter.mkdir(parents=True)
        trainer = snapshot / "trainer.pth"
        adapter_config = adapter / "adapter_config.json"
        trainer.write_bytes(b"trainer")
        adapter_config.write_text("{}", encoding="utf-8")

        def record(path: Path):
            payload = path.read_bytes()
            return {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }

        manifest = {
            "snapshot_schema_version": 1,
            "epoch_idx": 1,
            "files": {
                "trainer.pth": record(trainer),
                "adapter/adapter_config.json": record(adapter_config),
            },
        }
        (snapshot / "manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        references = checkpoint_dir / "checkpoint_refs"
        references.mkdir()
        (references / "best.json").write_text(
            json.dumps({
                "checkpoint_reference_schema_version": 1,
                "reference_name": "best",
                "snapshot": str(snapshot.relative_to(checkpoint_dir)),
            }),
            encoding="utf-8",
        )

    def test_preflight_uses_best_and_checkpoint_sampling_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_dir = root / "checkpoint"
            checkpoint_dir.mkdir()
            self._write_checkpoint(checkpoint_dir)
            metadata_path = root / "metadata.json"
            metadata_path.write_text(
                json.dumps(_metadata()),
                encoding="utf-8",
            )
            config = {
                "image_path": str(root),
                "weight_path": str(root),
                "input_img": False,
                "input_loc": False,
                "level_key": "main_info",
                "max_rois_per_dxitem": 13,
                "roi_sampling_mode": "head_k",
                "valid_sampling_seed": 123,
                "DxItem_list": _metadata()["DxItem_list"],
                "max_new_tokens": 17,
                "eval_prompt_batch_size": 2,
                "amp": False,
                "device": "cpu",
                "pp_num_gpus": None,
            }
            (checkpoint_dir / "config.json").write_text(
                json.dumps(config),
                encoding="utf-8",
            )

            pipeline = DRGVLMInferencePipeline(
                checkpoint_dir=str(checkpoint_dir),
                input_metadata_path=str(metadata_path),
            )
            summary = pipeline.preflight()

        self.assertTrue(summary["checkpoint_path"].endswith("best.json"))
        self.assertEqual(summary["checkpoint_epoch"], 1)
        self.assertEqual(
            summary["sampling"],
            {
                "max_rois_per_dxitem": 13,
                "roi_sampling_mode": "head_k",
                "valid_sampling_seed": 123,
            },
        )
        self.assertEqual(summary["active_dxitem_pair_count"], 2)
        self.assertFalse(summary["structured_report_required"])

    def test_run_writes_generated_report_with_fake_model(self):
        class FakeModel:
            @staticmethod
            def prepare_case_loss_context(case):
                del case
                return None

            @staticmethod
            def generate_outputs(
                batch_cases,
                max_new_tokens,
                case_contexts,
            ):
                del max_new_tokens, case_contexts
                case = batch_cases[0]
                return {
                    case.global_idx: {
                        dx_item: {
                            "pred_txt": f"generated-{dx_item}",
                            "gt_txt": "",
                        }
                        for dx_item in case.DxItem_rois
                    },
                }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_dir = root / "checkpoint"
            checkpoint_dir.mkdir()
            self._write_checkpoint(checkpoint_dir)
            metadata_path = root / "metadata.json"
            output_path = root / "output.json"
            metadata_path.write_text(
                json.dumps(_metadata()),
                encoding="utf-8",
            )
            config = {
                "image_path": str(root),
                "weight_path": str(root),
                "input_img": False,
                "input_loc": False,
                "level_key": "main_info",
                "max_rois_per_dxitem": 13,
                "roi_sampling_mode": "head_k",
                "valid_sampling_seed": 123,
                "DxItem_list": _metadata()["DxItem_list"],
                "max_new_tokens": 17,
                "eval_prompt_batch_size": 2,
                "amp": False,
                "device": "cpu",
                "pp_num_gpus": None,
            }
            (checkpoint_dir / "config.json").write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            pipeline = DRGVLMInferencePipeline(
                checkpoint_dir=str(checkpoint_dir),
                input_metadata_path=str(metadata_path),
                output_metadata_path=str(output_path),
            )
            pipeline._build_model = Mock(return_value=FakeModel())
            summary = pipeline.run()
            output = json.loads(
                output_path.read_text(encoding="utf-8")
            )

        self.assertEqual(summary["generated_report_count"], 2)
        self.assertEqual(
            output["case_list"][0]["generated_report"],
            {
                "DxA": "generated-DxA",
                "DxB": "generated-DxB",
            },
        )
        self.assertNotIn("structured_report", output["case_list"][0])


if __name__ == "__main__":
    unittest.main()
