"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2026, yasaisen (clover)

 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.

 last modified in 2604301444
"""


import os
from typing import Optional, List

import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler, SequentialSampler

# from .classLevelVisualAttrsDataset import classLevelVisualAttrsDataset
from .classLevelDynamicMPPDataset import classLevelDynamicMPPDataset
from .maxROI_sampler import MaxROIBatchSampler
from ..common.utils import log_print


class datasetHandler:
    def __init__(self,
        train_dataset: Dataset = None, 
        valid_dataset: Dataset = None, 
        train_shuffle: bool = True, 
        batch_size: int = 2, 
        num_workers: int = 4, 

        pin_memory: bool = True,
        drop_last: bool = False,
        prefetch_factor: Optional[int] = None,
        use_max_roi_sampler: bool = False,
        max_rois_per_update: Optional[int] = None,
    ):
        log_print("Loading DataLoaders...")

        self.DDP_status_detect()

        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset

        self.train_shuffle = train_shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor
        self.use_max_roi_sampler = bool(use_max_roi_sampler)
        self.max_rois_per_update = max_rois_per_update

        self.create_train_loader()
        self.create_valid_loader()

        log_print("...Done\n")

    def _is_main(self):
        return self.rank == 0

    def DDP_status_detect(self):
        self.ddp_enabled = dist.is_available() and dist.is_initialized()
        self.rank = int(os.environ.get("RANK", 0)) if self.ddp_enabled else 0
        self.world_size = int(os.environ.get("WORLD_SIZE", 1)) if self.ddp_enabled else 1

    def _build_loader(self, 
        dataset: Dataset,
        sampler: Optional[DistributedSampler],
        shuffle_flag: bool, 
        drop_last: bool, 
        batch_sampler = None,
    ):
        loader_kwargs = {
            "dataset": dataset,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": (self.num_workers > 0),
            "collate_fn": dataset.collate_cases,
        }
        if self.prefetch_factor is not None and self.num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.prefetch_factor

        if batch_sampler is not None:
            loader_kwargs["batch_sampler"] = batch_sampler
        elif sampler is not None:
            loader_kwargs["sampler"] = sampler
            loader_kwargs["shuffle"] = False
            loader_kwargs["drop_last"] = drop_last
            loader_kwargs["batch_size"] = self.batch_size
        else:
            loader_kwargs["shuffle"] = shuffle_flag
            loader_kwargs["drop_last"] = drop_last
            loader_kwargs["batch_size"] = self.batch_size

        return DataLoader(**loader_kwargs)

    def _build_train_base_sampler(self,
        dataset: Dataset,
    ):
        if self.ddp_enabled:
            return DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=self.train_shuffle,
                drop_last=self.drop_last,
            )
        if self.train_shuffle:
            return RandomSampler(dataset)
        return SequentialSampler(dataset)

    def create_train_loader(self):
        if self.train_dataset is None:
            self.train_loader = None
            log_print("No train_loader was created")
            return

        if self.use_max_roi_sampler:
            if not hasattr(self.train_dataset, "get_effective_roi_count"):
                raise AttributeError(
                    "use_max_roi_sampler=True requires the train dataset to implement "
                    "get_effective_roi_count(idx)."
                )

            base_sampler = self._build_train_base_sampler(dataset=self.train_dataset)
            train_batch_sampler = MaxROIBatchSampler(
                sampler=base_sampler,
                roi_count_func=self.train_dataset.get_effective_roi_count,
                batch_size=self.batch_size,
                max_rois_per_update=self.max_rois_per_update,
                drop_last=self.drop_last,
            )
            self.train_loader = self._build_loader(
                dataset=self.train_dataset,
                sampler=None,
                shuffle_flag=False,
                drop_last=False,
                batch_sampler=train_batch_sampler,
            )

            if self._is_main():
                log_print(f"train_dataset: {len(self.train_loader.dataset)}")
                log_print(
                    "MaxROIBatchSampler enabled: "
                    f"batch_size={self.batch_size}, "
                    f"max_rois_per_update={self.max_rois_per_update}, "
                    f"num_update_groups={len(self.train_loader)}"
                )
            return

        train_sampler = None
        if self.ddp_enabled:
            train_sampler = self._build_train_base_sampler(dataset=self.train_dataset)

        self.train_loader = self._build_loader(
            dataset=self.train_dataset,
            sampler=train_sampler,
            shuffle_flag=self.train_shuffle,
            drop_last=self.drop_last,
        )

        if self._is_main():
            log_print(f"train_dataset: {len(self.train_loader.dataset)}")

    def create_valid_loader(self):
        if self.valid_dataset is None:
            self.valid_loader = None
            log_print("No valid_dataset was created")
            return

        valid_sampler = None
        if self.ddp_enabled:
            valid_sampler = DistributedSampler(
                self.valid_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                drop_last=False,
            )

        self.valid_loader = self._build_loader(
            dataset=self.valid_dataset,
            sampler=valid_sampler,
            shuffle_flag=False,
            drop_last=False,
        )

        if self._is_main():
            log_print(f"valid_dataset: {len(self.valid_loader.dataset)}")

    @classmethod
    def from_config(cls,
        cfg,
        train_shuffle: bool = True,
    ):
        valid_dataset = classLevelDynamicMPPDataset.from_config(
            cfg=cfg,
            split='valid',
        ) if cfg.valid_metadata_path else None

        train_dataset = classLevelDynamicMPPDataset.from_config(
            cfg=cfg,
            split='train'
        ) if cfg.train_metadata_path else None


        per_device_bs = getattr(cfg, "per_device_batch_size", None)
        batch_size = per_device_bs if per_device_bs is not None else cfg.batch_size

        handler = cls(
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            train_shuffle=train_shuffle,
            batch_size=batch_size,

            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            drop_last=cfg.drop_last,
            prefetch_factor=cfg.prefetch_factor,
            use_max_roi_sampler=getattr(cfg, "use_max_roi_sampler", False),
            max_rois_per_update=getattr(cfg, "max_rois_per_update", None),
        )
        cfg.num_batchs_per_epoch = len(handler.train_loader) if handler.train_loader is not None else 0

        cfg.DxItem_list = handler.train_dataset.DxItem_list

        return handler
