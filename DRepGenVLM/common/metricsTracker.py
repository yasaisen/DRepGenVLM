"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2512070438
"""


import torch
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass


from ..common.utils import log_print, _debug_print


@dataclass
class lossLogger:
    def __init__(self, 
    ):
        self.update_step_counter = 0
        self.losses = {
            'total_loss': [], 
            'mse_loss': [], 
            'cosine_loss': [], 
            'similarity_ce_loss': [], 
        }
        self.epoch_losses = []

    def update(self, 
        loss_dict: Dict[str, float],
    ):
        for key, value in loss_dict.items():
            if key in self.losses:
                self.losses[key].append(value)
        self.update_step_counter += 1

    def epoch_summary(self
    ):
        epoch_summary_dict = {}
        for key, value in self.losses.items():
            epoch_summary_dict[key] = np.mean(value[-self.update_step_counter:])
        self.update_step_counter = 0
        self.epoch_losses.append(epoch_summary_dict['total_loss'])
        return epoch_summary_dict

class MetricsTracker:
    def __init__(self, 
        save_path: str,
    ):
        self.save_path = save_path
        self.reset()
    
    def reset(self,
    ):
        self.nowtime = datetime.now().strftime("%y%m%d%H%M")
        self.csv_log_path = os.path.join(self.save_path, f"{self.nowtime}_metrics.jsonl")
        os.makedirs(os.path.dirname(self.csv_log_path), exist_ok=True)
        self.f = open(self.csv_log_path, "w")

        self.writer = SummaryWriter(log_dir=self.save_path)
        log_print(f'TensorBoard writer created at {self.save_path}')
        log_print(f'Use "tensorboard --logdir={self.save_path} --port=6006" to visualize')

        self.train_loss_logger = lossLogger()
        self.val_loss_logger = lossLogger()

        self.learning_rates = []
        # self.global_step = 0

    def import_tensorboard_history(
        self,
        source_logdir: str,
        max_step: Optional[int] = None,
    ):
        if source_logdir is None or not os.path.isdir(source_logdir):
            log_print(f"[TensorBoardResume] source logdir not found, skip import: {source_logdir}")
            return

        try:
            from tensorboard.backend.event_processing import event_accumulator
        except Exception as e:
            log_print(f"[TensorBoardResume] tensorboard event reader unavailable, skip import: {e}")
            return

        try:
            event_acc = event_accumulator.EventAccumulator(source_logdir)
            event_acc.Reload()
        except Exception as e:
            log_print(f"[TensorBoardResume] failed to read source logdir, skip import: {e}")
            return

        tags = event_acc.Tags()
        scalar_count = 0
        hist_count = 0

        for tag in tags.get("scalars", []):
            for event in event_acc.Scalars(tag):
                if max_step is not None and int(event.step) > int(max_step):
                    continue
                self.writer.add_scalar(
                    tag,
                    event.value,
                    global_step=int(event.step),
                    walltime=event.wall_time,
                )
                scalar_count += 1

        for tag in tags.get("histograms", []):
            for event in event_acc.Histograms(tag):
                if max_step is not None and int(event.step) > int(max_step):
                    continue
                hist = event.histogram_value
                self.writer.add_histogram_raw(
                    tag=tag,
                    min=hist.min,
                    max=hist.max,
                    num=hist.num,
                    sum=hist.sum,
                    sum_squares=hist.sum_squares,
                    bucket_limits=list(hist.bucket_limit),
                    bucket_counts=list(hist.bucket),
                    global_step=int(event.step),
                    walltime=event.wall_time,
                )
                hist_count += 1

        self.writer.flush()
        log_print(
            f"[TensorBoardResume] imported {scalar_count} scalar events and "
            f"{hist_count} histogram events from {source_logdir} up to step {max_step}."
        )

    def log(self, 
        record: dict
    ):
        record.setdefault("timestamp", datetime.now().isoformat(timespec="seconds"))
        self.f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.f.flush()

    def update(self, 
        loss_dict: Dict[str, float], 
        stage: str,
        lr: float = None,
        rank: int = 0, 
        global_step: int = 0,
    ):
        if stage not in ['train', 'val']:
            raise ValueError("stage must be either 'train' or 'val'")

        if stage == 'train' and rank == 0:
            self.train_loss_logger.update(loss_dict)
            self.learning_rates.append(lr)

            for key, value in loss_dict.items():
                if isinstance(value, (torch.Tensor, float)):
                    self.writer.add_scalar(f"Train_general/step_{key}", float(value), global_step)
            self.writer.add_scalar(f"Optimization/lr", float(lr), global_step)

        elif stage == 'val' and rank == 0:
            self.val_loss_logger.update(loss_dict)

        loss_dict.update({
            "stage": stage, 
            "lr": lr, 
            "rank": rank, 
        })
        self.log(record=loss_dict)

    def epoch_summary(self, 
        metric_dict: Dict[str, float] = None, 
        global_step: int = 0,
    ):
        # global_step = self.global_step

        train_epoch_summary = self.train_loss_logger.epoch_summary()
        val_epoch_summary = self.val_loss_logger.epoch_summary()

        for key, value in train_epoch_summary.items():
            if isinstance(value, (torch.Tensor, float)):
                self.writer.add_scalar(f"Train_general/train_{key}", float(value), global_step)

        for key, value in val_epoch_summary.items():
            if isinstance(value, (torch.Tensor, float)):
                self.writer.add_scalar(f"Valid_general/val_{key}", float(value), global_step)

        if metric_dict is not None:
            for key, value in metric_dict.items():
                if isinstance(value, (torch.Tensor, float)):
                    self.writer.add_scalar(f"Valid_general/{key}", float(value), global_step)

            self.log(record=metric_dict)

        # self.global_step += 1
        return train_epoch_summary['total_loss'], val_epoch_summary['total_loss']
    
    def close(self,
    ):
        self.f.close()

    def plot_metrics(self, 
        epoch_idx: int, 
        path: str = None, 
    ):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # class_loss
        axes[0, 0].plot(self.train_loss_logger.losses['class_loss'])
        axes[0, 0].set_title('Classification Loss')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Loss')
        
        # bbox_loss
        axes[0, 1].plot(self.train_loss_logger.losses['bbox_loss'])
        axes[0, 1].set_title('Bounding Box Loss')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].set_ylabel('Loss')
        
        # giou_loss
        axes[1, 0].plot(self.train_loss_logger.losses['giou_loss'])
        axes[1, 0].set_title('GIoU Loss')
        axes[1, 0].set_xlabel('Iteration')
        axes[1, 0].set_ylabel('Loss')
        
        # learning_rates
        axes[1, 1].plot(self.learning_rates)
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Iteration')
        axes[1, 1].set_ylabel('LR')
        
        plt.tight_layout()

        save_path = path if path is not None else os.path.join(self.save_path, f'metrics_epoch_{epoch_idx}.png')
        plt.savefig(save_path)
        # plt.show()
        plt.close()

    @classmethod
    def from_config(cls, 
        cfg, 
    ):
        tracker = cls(
            save_path=cfg.save_path,
        )

        return tracker


########################################################################
class PerformanceMonitor:    
    def __init__(self):
        self.gpu_stats = []
        self.memory_stats = []
        self.training_times = []
    
    def log_gpu_memory(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            cached = torch.cuda.memory_reserved() / 1024**3      # GB
            self.memory_stats.append({
                'allocated': allocated,
                'cached': cached,
                'timestamp': torch.cuda.Event(enable_timing=True)
            })
    
    def start_timer(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if self.start_time:
            self.start_time.record()
    
    def end_timer(self):
        if torch.cuda.is_available() and self.start_time:
            end_time = torch.cuda.Event(enable_timing=True)
            end_time.record()
            torch.cuda.synchronize()
            elapsed = self.start_time.elapsed_time(end_time) / 1000  # into seconds
            self.training_times.append(elapsed)
            return elapsed
        return 0.0
    
    def generate_report(self, save_path: str = None):
        report = {
            'max_memory_allocated': max([s['allocated'] for s in self.memory_stats]) if self.memory_stats else 0,
            'max_memory_cached': max([s['cached'] for s in self.memory_stats]) if self.memory_stats else 0,
            'total_training_time': sum(self.training_times),
            'device_info': {
                'cuda_available': torch.cuda.is_available(),
                'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'
            }
        }
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
########################################################################










