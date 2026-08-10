import os
import random
import tempfile
import unittest

import numpy as np
import torch

from DRepGenVLM.trainer.building_DRGVLM_PPTrainer import DRGVLM_PPTrainer


class _AdapterOnlyModel:
    def __init__(self):
        self.loaded_adapter_path = None

    def load_lora_weights(self, path):
        self.loaded_adapter_path = path


class CheckpointRNGStateTests(unittest.TestCase):
    def test_cuda_trainer_keeps_cpu_control_states_on_cpu(self):
        original_python_state = random.getstate()
        original_numpy_state = np.random.get_state()
        original_torch_state = torch.get_rng_state()

        try:
            torch.manual_seed(1234)
            expected_torch_state = torch.get_rng_state().clone()
            train_generator = torch.Generator().manual_seed(5678)
            expected_generator_state = train_generator.get_state().clone()

            rng_state = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": expected_torch_state,
                "torch_cuda": None,
            }
            checkpoint = {
                "epoch_idx": 3,
                "global_step": 11,
                "optimizer_step": 7,
                "best_val_loss": 0.25,
                "best_metric_values": {},
                "patience_counter": 2,
                "num_batches_per_epoch": 4,
                "optimizer_state_dict": None,
                "scheduler_state_dict": None,
                "rng_state": rng_state,
                "train_dataloader_generator_state": expected_generator_state,
                "train_progress": None,
                "resume_signature": {},
            }

            with tempfile.TemporaryDirectory() as checkpoint_dir:
                requested_path = os.path.join(checkpoint_dir, "latest_model.pth")
                trainer_path = os.path.join(
                    checkpoint_dir,
                    "[trainer]latest_model.pth",
                )
                adapter_path = os.path.join(
                    checkpoint_dir,
                    "latest_model_lora",
                )
                os.makedirs(adapter_path)
                torch.save(checkpoint, trainer_path)

                torch.manual_seed(9999)
                model = _AdapterOnlyModel()
                trainer = DRGVLM_PPTrainer(
                    model=model,
                    device="cuda:0",
                    trainer_mode="keepTrain",
                    save_path=checkpoint_dir,
                )

                loaded_epoch = trainer.load_checkpoint(requested_path)

                self.assertEqual(loaded_epoch, 3)
                self.assertEqual(model.loaded_adapter_path, adapter_path)
                self.assertEqual(
                    trainer._pending_train_generator_state.device.type,
                    "cpu",
                )
                self.assertTrue(torch.equal(
                    trainer._pending_train_generator_state,
                    expected_generator_state,
                ))
                self.assertTrue(torch.equal(
                    torch.get_rng_state(),
                    expected_torch_state,
                ))
        finally:
            random.setstate(original_python_state)
            np.random.set_state(original_numpy_state)
            torch.set_rng_state(original_torch_state)


if __name__ == "__main__":
    unittest.main()
