"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2511041710
"""

import os
import warnings

# import torch
# torch.autograd.set_detect_anomaly(True)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["NO_ALBUMENTATIONS_UPDATE"] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from DRepGenVLM.configs.configHandler import ConfigHandler
from DRepGenVLM.datasets.building_datasetHandler import datasetHandler
from DRepGenVLM.models.modeling_medGemmaLoRA import DownstreamRepGenVLM
from DRepGenVLM.trainer.building_DRGVLM_PPTrainer import DRGVLM_PPTrainer
from DRepGenVLM.common.dist_utils import init_distributed_mode, cleanup_distributed_mode
from DRepGenVLM.common.utils import set_seed, log_print, _debug_print


def main():
    cfg_handler = ConfigHandler.get_cfg()
    set_seed()
    init_distributed_mode(cfg=cfg_handler.cfg)

    data_handler = datasetHandler.from_config(
        cfg=cfg_handler.cfg, 
    )
    model = DownstreamRepGenVLM.from_config(
        cfg=cfg_handler.cfg, 
    )
    trainer = DRGVLM_PPTrainer.from_config(
        cfg=cfg_handler.cfg, 
        model=model, 
        checkpoint_path=cfg_handler.checkpoint_path,
    )
    cfg_handler.cfg.save()

    log_print("Start training...")
    trainer.train(
        train_dataloader=data_handler.train_loader,
        val_dataloader=data_handler.valid_loader,
    )
    log_print("Training completed!")

    cleanup_distributed_mode(cfg=cfg_handler.cfg)

if __name__ == "__main__":
    main()

