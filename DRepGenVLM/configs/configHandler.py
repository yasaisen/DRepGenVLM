"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2511110015
"""

import argparse
import os


from .DRGVLM_baseConfig import DRGVLM_baseConfig
from ..common.utils import log_print


class ConfigHandler:
    def __init__(self, 
        cfg, 
        checkpoint_path: str = None,
        checkpoint_type: str = None,
        use_this_dir: bool = False,
    ):
        log_print(f"Building...", head=True)
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path
        self.checkpoint_type = checkpoint_type
        self.use_this_dir = use_this_dir
        from datetime import datetime
        self.cfg.nowtime = datetime.now().strftime("%y%m%d%H%M%S")

        if getattr(self.cfg, 'root_path', None) is None:
            pwd = os.getcwd()
            self.cfg.root_path = pwd
            log_print(f"Automatically set root_path on {pwd}")

        if self.use_this_dir and self.checkpoint_type == "best":
            raise ValueError("--use-this-dir is only supported with --latest-checkpoint-path, not --best-checkpoint-path.")

        if self.use_this_dir and self.checkpoint_path is not None:
            self.cfg.save_path = os.path.dirname(self.checkpoint_path)
            os.makedirs(self.cfg.save_path, exist_ok=True)
            log_print(f"Continue updating in checkpoint directory: {self.cfg.save_path}")
        elif getattr(self.cfg, 'save_path', None) is None or self.checkpoint_path is not None:
            self.cfg.save_path = os.path.join(self.cfg.root_path, 'checkpoints', self.cfg.nowtime)
            os.makedirs(self.cfg.save_path, exist_ok=True)
            log_print(f"Automatically set output path to {self.cfg.save_path}")

        log_print(f"Loaded config:")
        self.cfg.print_config()
        log_print(f"...Done\n")

    @classmethod
    def get_cfg(cls,
    ):
        parser = argparse.ArgumentParser()
        parser.add_argument("--cfg-path", required=False, default=None)
        parser.add_argument("--best-checkpoint-path", required=False, default=None)
        parser.add_argument("--latest-checkpoint-path", required=False, default=None)
        parser.add_argument("--weight-filename", required=False, default=None)
        parser.add_argument("--retrain", required=False, default=None)
        parser.add_argument("--use-this-dir", action="store_true", default=False)
        args = parser.parse_args()
        cfg_path = args.cfg_path
        best_checkpoint_path = args.best_checkpoint_path
        latest_checkpoint_path = args.latest_checkpoint_path
        weight_filename = args.weight_filename
        use_this_dir = args.use_this_dir
        if args.retrain is not None:
            trainer_mode = "reTrain" if args.retrain else "keepTrain"
        else:
            trainer_mode = None

        if cfg_path is None and best_checkpoint_path is None and latest_checkpoint_path is None:
            raise ValueError("At least one of --cfg-path or --best-checkpoint-path or --latest-checkpoint-path must be provided.")

        if best_checkpoint_path is not None and latest_checkpoint_path is not None:
            raise ValueError("Only one of --best-checkpoint-path or --latest-checkpoint-path can be provided, not both.")

        if use_this_dir and best_checkpoint_path is not None:
            raise ValueError("--use-this-dir is only supported with --latest-checkpoint-path, not --best-checkpoint-path.")

        if use_this_dir and latest_checkpoint_path is None:
            log_print("[WARN] --use-this-dir was provided without --latest-checkpoint-path; it will be ignored.")

        if cfg_path is None and (best_checkpoint_path is not None or latest_checkpoint_path is not None):
            checkpoint_path = best_checkpoint_path if best_checkpoint_path is not None else latest_checkpoint_path
            cfg_path = os.path.join(checkpoint_path, "config.json")
            log_print(f"Inferring cfg_path from checkpoint_path: {cfg_path}")

        cfg = DRGVLM_baseConfig.load(
            path=cfg_path,
        )
        cfg.trainer_mode = trainer_mode

        if best_checkpoint_path is not None:
            checkpoint_type = "best"
            checkpoint_path = os.path.join(checkpoint_path, cfg.weight_filename if weight_filename is None else weight_filename)
        elif latest_checkpoint_path is not None:
            checkpoint_type = "latest"
            checkpoint_path = os.path.join(checkpoint_path, "latest_model.pth")
        else:
            checkpoint_type = None
            checkpoint_path = None


        cfg_handler = cls(
            cfg=cfg, 
            checkpoint_path=checkpoint_path, 
            checkpoint_type=checkpoint_type,
            use_this_dir=use_this_dir,
        )

        return cfg_handler

























