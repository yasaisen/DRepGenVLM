"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2026, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2607081514
"""


import json
from datetime import datetime
import os
import torch
from typing import Dict, Optional, List


PROJECT_MAPPING_DICT = {
    "MG15_basic_overfitTesting": {
        "model_name": "medgemma-1.5-4b-it",
        "train_ann_file": "[metadata]downstreamRepGenVLM[NAS3test]_v2.1_2607151447.json",
        "valid_ann_file": "[metadata]downstreamRepGenVLM[NAS3test]_v2.1_2607151447.json",
        "input_img": True,
        "input_loc": True,
        "level_key": "main_info",
        "batch_size": 1,
        "accumulation_steps": 8,
        "max_rois_per_case": 30,
        "roi_sampling_mode": "random_k",
        "valid_sampling_seed": 42,
        "use_max_roi_sampler": True,
        "max_rois_per_update": 60,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "use_vision_lora": False,
        "max_new_tokens": 256,
        "general_learning_rate": 2e-4,
        "gradient_clip_norm": 1.0,
        "num_epochs": 50,
        "warmup_steps": None,
        "early_stop_patience": 20,
        "training_mode": "PP",
        "pp_num_gpus": 4,
        "attn_implementation": "sdpa",
        "use_shared_vision_cache": True,
    },
    # ======================================================== #
    # Pipeline Parallelism (PP) configurations
    # Launch with:  python R27_MVLM_trainDRGVLM_v0.0.py --cfg-path <this_config>
    # (single process, multiple GPUs via device_map)
    # ======================================================== #
    "MG15_basic_PP4G": {
        "model_name": "medgemma-1.5-4b-it",
        "train_ann_file": "[metadata]downstreamRepGenVLM[NAS3train]_v2.1_2607151447",
        "valid_ann_file": "[metadata]downstreamRepGenVLM[NAS3test]_v2.1_2607151447.json",
        "input_img": True,
        "input_loc": True,
        "level_key": "main_info",
        "batch_size": 1,
        "accumulation_steps": 8,
        "max_rois_per_case": 30,
        "roi_sampling_mode": "random_k",
        "valid_sampling_seed": 42,
        "use_max_roi_sampler": True,
        "max_rois_per_update": 60,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        "use_vision_lora": False,
        "max_new_tokens": 256,
        "general_learning_rate": 2e-4,
        "gradient_clip_norm": 1.0,
        "num_epochs": 50,
        "warmup_steps": None,
        "early_stop_patience": 20,
        "training_mode": "PP",
        "pp_num_gpus": 4,
        "attn_implementation": "sdpa",
        "use_shared_vision_cache": True,
    },
    # ======================================================== #
    "NULLMODE": {
        "model_name": None,
        "train_ann_file": "",
        "valid_ann_file": "",
        "input_img": True,
        "input_loc": True,
        "level_key": "main_info",
        "batch_size": None,
        "accumulation_steps": None,
        "max_rois_per_case": None,
        "roi_sampling_mode": None,
        "valid_sampling_seed": 42,
        "use_max_roi_sampler": False,
        "max_rois_per_update": None,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "v_proj"],
        "use_vision_lora": False,
        "max_new_tokens": 256,
        "general_learning_rate": None,
        "gradient_clip_norm": None,
        "num_epochs": None,
        "warmup_steps": None,
        "early_stop_patience": None,
        "training_mode": None,
        "pp_num_gpus": None,
        "attn_implementation": "sdpa",
        "use_shared_vision_cache": False,
    },
}


class DRGVLM_baseConfig:
    def __init__(self,
        project_name: str = None,
        is_HPC: bool = False,
    ):
        self.project_name = project_name
        self.is_HPC = is_HPC
        self.factory_timestamp = datetime.now().strftime("%y%m%d%H%M")

        self.setup_path()

        if project_name is not None:
            project_dict = PROJECT_MAPPING_DICT[project_name]
        else:
            project_dict = PROJECT_MAPPING_DICT["NULLMODE"]

        self.setup_datasetHandler(project_dict=project_dict)
        self.setup_modelBuilder(project_dict=project_dict)
        self.setup_trainer(project_dict=project_dict)
        self.setup_configHandler(project_dict=project_dict)

    def setup_path(self):
        self.localGPU_image_path = "/media/yasaisen/NAS8/for_research/datasets"
        self.localGPU_metadata_path = "/media/yasaisen/NAS8/for_research/metadatas"
        self.localGPU_weight_path = "/media/yasaisen/NAS8/for_research/weights"
        self.localGPU_root_path = "/media/yasaisen/NAS8/for_research/R27_MVLM_v3.18"

        if self.is_HPC:
            self.image_path = "/work/misaka13/datasets"
            self.metadata_path = "/work/misaka13/metadatas"
            self.weight_path = "/work/misaka13/weights"
            self.root_path = "/work/misaka13/R27_MVLM_v3.18"
        else:
            self.image_path = self.localGPU_image_path
            self.metadata_path = self.localGPU_metadata_path
            self.weight_path = self.localGPU_weight_path
            self.root_path = self.localGPU_root_path

    def setup_datasetHandler(self,
        project_dict: Dict = None,
    ):
        self.train_metadata_path = os.path.join(self.metadata_path, project_dict["train_ann_file"])
        self.valid_metadata_path = os.path.join(self.metadata_path, project_dict["valid_ann_file"])

        self.input_img = project_dict.get("input_img", True)
        self.input_loc = project_dict.get("input_loc", True)
        self.level_key = project_dict.get("level_key", "main_info")

        self.batch_size = 1 if project_dict["batch_size"] is None else project_dict["batch_size"]
        self.num_workers = 4
        self.pin_memory = True
        self.drop_last = False
        self.prefetch_factor = None

        self.num_batchs_per_epoch = None  # set by datasetHandler
        self.DxItem_list = None           # set by datasetHandler from metadata

        if self.is_HPC:
            self.max_rois_per_case = project_dict["max_rois_per_case"]
            self.roi_sampling_mode = project_dict["roi_sampling_mode"]
        else:
            self.max_rois_per_case = 2
            self.roi_sampling_mode = "random_k"
        self.valid_sampling_seed = int(project_dict.get("valid_sampling_seed", 42))

        self.use_max_roi_sampler = bool(project_dict.get("use_max_roi_sampler", False))
        self.max_rois_per_update = project_dict.get("max_rois_per_update", None)

    def setup_modelBuilder(self,
        project_dict: Dict = None,
    ):
        self.model_name = project_dict["model_name"] if project_dict["model_name"] is not None else "medgemma-1.5-4b-it"
        self.checkpoint_path = None
        self.temperature = 1.0

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sep_str = "<unused0>"
        self.boc_str = "<unused1>"

        # LoRA settings
        self.lora_r = int(project_dict.get("lora_r", 16))
        self.lora_alpha = int(project_dict.get("lora_alpha", 32))
        self.lora_dropout = float(project_dict.get("lora_dropout", 0.05))
        self.lora_target_modules = list(project_dict.get("lora_target_modules", ["q_proj", "v_proj"]))
        self.use_vision_lora = bool(project_dict.get("use_vision_lora", False))

        # Generation
        self.max_new_tokens = int(project_dict.get("max_new_tokens", 256))

        # Shared vision cache skips re-running the frozen vision tower per DxItem.
        # Vision LoRA is configured independently and cannot be combined with it.
        self.use_shared_vision_cache = bool(project_dict.get("use_shared_vision_cache", True))
        if self.use_shared_vision_cache and self.use_vision_lora:
            raise ValueError(
                "use_shared_vision_cache=True is incompatible with "
                "use_vision_lora=True because the cached vision forward runs "
                "under torch.no_grad(). Disable one of these options."
            )

        # Pipeline Parallelism settings
        self.training_mode = project_dict.get("training_mode", "PP")
        self.attn_implementation = project_dict.get("attn_implementation", "sdpa")

        if self.is_HPC:
            self.pp_num_gpus = project_dict["pp_num_gpus"]
        else:
            self.pp_num_gpus = 1 if project_dict["pp_num_gpus"] is not None else None

    def setup_trainer(self,
        project_dict: Dict = None,
    ):
        self.trainer_mode = None
        self.accumulation_steps = 8 if project_dict["accumulation_steps"] is None else project_dict["accumulation_steps"]

        self.weight_filename = "best_model.pth"
        self.save_freq = 5
        self.plot_freq = 5
        self.gradient_clip_norm = 1.0 if project_dict["gradient_clip_norm"] is None else project_dict["gradient_clip_norm"]
        self.amp = True
        self.early_stop_patience = project_dict.get("early_stop_patience", None)

        self.learning_rate_dict = {
            "general": 2e-4 if project_dict["general_learning_rate"] is None else project_dict["general_learning_rate"],
        }
        self.weight_decay = 1e-2

        self.total_steps = None  # calculated by Trainer
        self.warmup_steps = project_dict.get("warmup_steps", None)
        self.warmup_ratio = 0.05
        self.max_warmup_steps = 5000

        self.num_epochs = 50 if project_dict["num_epochs"] is None else project_dict["num_epochs"]

    def setup_configHandler(self,
        project_dict: Dict = None,
    ):
        self.nowtime = None        # handled by ConfigHandler
        self.save_path = None      # handled by ConfigHandler

        self.world_size = 1
        self.distributed = False
        self.dist_url = "env://"
        self.rank = None
        self.gpu = None
        self.dist_backend = None

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def print_config(self):
        from ..common.utils import log_print
        from ..common.dist_utils import is_main_process
        log_print("Training Configuration:")
        if is_main_process():
            for key, value in self.to_dict().items():
                print(f"  {key}: {value}")

    def save(self,
        save_path: str = None,
        filename: str = "config.json",
    ):
        save_path = save_path if self.save_path is None else self.save_path
        file_path = os.path.join(save_path, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        from ..common.utils import log_print
        log_print(f"Saved config to {file_path}")

    @classmethod
    def load(cls,
        path: str,
    ):
        with open(path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

        cfg = cls.__new__(cls)
        for key, value in config_dict.items():
            setattr(cfg, key, value)
        return cfg

    @classmethod
    def setup_configs(cls, 
        save_path: str = './DRepGenVLM/projects',
    ):
        for project_name in PROJECT_MAPPING_DICT.keys():
            for is_HPC in [True, False]:
                config = cls(
                    project_name=project_name, 
                    is_HPC=is_HPC, 
                )
                str_is_HPC = '_hpc' if is_HPC else ''
                filename = f"{project_name}_config{str_is_HPC}.json"

                config.save(
                    save_path=save_path, 
                    filename=filename, 
                )










