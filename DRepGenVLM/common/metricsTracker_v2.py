"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2512070438
"""


import matplotlib
matplotlib.use("Agg")


import os
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import io
import torchvision.transforms.functional as F
import torch.distributed as dist


from ..common.utils import log_print, _debug_print, variable_checker


class TrainingMonitor:
    def __init__(self, 
        writer, 
        model, 
        optimizer, 
        log_every_n_steps: int = 10, 
        check_every_n_steps: int = 500, 
        is_custom_criterion: bool = False,
    ):
        self.writer = writer
        self.model = model
        self.optimizer = optimizer
        self.log_period = log_every_n_steps
        self.check_period = check_every_n_steps
        self.is_custom_criterion = is_custom_criterion
        self.phase = "idle"
        self.last_time = time.time()
        self.last_log_step = 0
        self.pending_cases = 0
        self.pending_dx_pairs = 0
        self.pending_rois = 0
        
        # # 儲存 Hook 抓到的 Attention Weights
        # self.attn_weights = {}
        # self.hook_handles = []

    # def register_attention_hooks(self, 
    #     layer_name_keyword='attention'
    # ):
    #     """
    #     自動遍歷模型，對包含 keyword 的層註冊 Hook 以抓取 Attention Map。
    #     這通常針對 Softmax 後的輸出。
    #     """
    #     def get_attn_hook(name):
    #         def hook(module, input, output):
    #             # 假設 output 是 [Batch, Heads, Q_len, K_len]
    #             # 有些模型 output 是 tuple (attn_out, attn_weights)，請根據你的模型調整
    #             if isinstance(output, tuple):
    #                 attn = output[1] # 假設第二個是 weights
    #             else:
    #                 attn = output # 假設就是 weights (如果你 hook 的是 Softmax 層)
                
    #             # 只存一份 detach 的 copy，避免記憶體洩漏
    #             if attn is not None:
    #                 self.attn_weights[name] = attn.detach()
    #         return hook

    #     print(f"正在註冊 Attention Hooks (關鍵字: {layer_name_keyword})...")
    #     for name, module in self.model.named_modules():
    #         if layer_name_keyword in name and 'dropout' not in name:
    #             # 這裡需要你確認一下你的模型 Attention Weights 是在哪一層產出的
    #             # 通常 hook 在 Softmax 層或是 Attention Block 的最後
    #             handle = module.register_forward_hook(get_attn_hook(name))
    #             self.hook_handles.append(handle)
    #     print(f"已註冊 {len(self.hook_handles)} 個 Hooks。")

    def _get_lr(self, 
    ):
        return self.optimizer.param_groups[0]['lr']

    def reset_phase(self, phase: str, global_step: int):
        """Exclude validation/checkpoint time from training throughput windows."""
        self.phase = str(phase)
        self.last_time = time.time()
        self.last_log_step = int(global_step)
        self.pending_cases = 0
        self.pending_dx_pairs = 0
        self.pending_rois = 0

    def log_always_on(self,
        global_step,
        batch_size,
        dx_count: int = 0,
        roi_count: int = 0,
    ):
        """ [常態監測] 每 N step 跑一次，計算輕量指標 """
        self.pending_cases += int(batch_size)
        self.pending_dx_pairs += int(dx_count)
        self.pending_rois += int(roi_count)
        if global_step % self.log_period != 0:
            return

        # 1. Throughput from the actual elapsed step/work delta.  A phase reset
        # at step 0 therefore cannot emit a fictitious step-0 throughput point.
        current_time = time.time()
        time_delta = current_time - self.last_time
        step_delta = int(global_step) - self.last_log_step
        phase_prefix = f"System/{self.phase}"
        if time_delta > 0 and step_delta > 0:
            self.writer.add_scalar(
                f"{phase_prefix}/Cases_per_sec",
                self.pending_cases / time_delta,
                global_step,
            )
            self.writer.add_scalar(
                f"{phase_prefix}/DxPairs_per_sec",
                self.pending_dx_pairs / time_delta,
                global_step,
            )
            self.writer.add_scalar(
                f"{phase_prefix}/ROIs_per_sec",
                self.pending_rois / time_delta,
                global_step,
            )
            self.writer.add_scalar(
                f"{phase_prefix}/Time_per_step_sec",
                time_delta / step_delta,
                global_step,
            )
        self.last_time = current_time
        self.last_log_step = int(global_step)
        self.pending_cases = 0
        self.pending_dx_pairs = 0
        self.pending_rois = 0

        # 2. Global Gradient Norm
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.writer.add_scalar('Optimization/Global_Grad_Norm', total_norm, global_step)

        # 3. Update Ratio (大略估計，取幾個代表性的層即可，避免全算太慢)
        # Ratio = (lr * grad_norm) / weight_norm
        lr = self._get_lr()
        for name, p in self.model.named_parameters():
            # 只抽查包含 'weight' 且位於特定深度的層 (例如每隔幾層抽一個)
            # if p.grad is not None and 'weight' in name and ('encoder.layers.0' in name or 'decoder.layers.0' in name or 'output' in name):
            if p.grad is not None and 'weight' in name and ('layers.0' in name or 'output' in name):
                param_norm = p.data.norm(2)
                grad_norm = p.grad.data.norm(2)
                if param_norm > 1e-6:
                    ratio = (lr * grad_norm) / param_norm
                    self.writer.add_scalar(f'UpdateRatio/{name}', ratio, global_step)
        
        if self.is_custom_criterion:
            self.writer.add_scalar(f'CriterionError/CardinalityError', float(self.model.criterion.get_cardinality_error()), global_step)
            self.writer.add_scalar(f'CriterionError/ClassError', float(self.model.criterion.get_class_error()), global_step)


        # allocated_vram = torch.cuda.memory_allocated() / (1024 ** 3)
        # reserved_vram = torch.cuda.memory_reserved() / (1024 ** 3)
        # self.writer.add_scalar('VRAM/Allocated_GB', allocated_vram, global_step=global_step)
        # self.writer.add_scalar('VRAM/Reserved_GB', reserved_vram, global_step=global_step)

        # VRAM logging: works for PP (single-process multi-GPU) and single-GPU.
        # For DDP (multi-process), each rank only writes its own GPU to avoid
        # redundant cross-process queries; rank is derived without assuming
        # LOCAL_RANK is set.
        is_distributed = dist.is_available() and dist.is_initialized()
        if torch.cuda.is_available() and is_distributed:
            # DDP path: each rank logs its own GPU only
            local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank()))
            current_vram = torch.cuda.memory_reserved(local_rank) / (1024 ** 3)
            self.writer.add_scalar(f'VRAM_By_GPU/GPU_{local_rank}', current_vram, global_step=global_step)
        elif torch.cuda.is_available():
            # PP / single-GPU path: this process owns all visible GPUs, query each directly
            num_gpus = torch.cuda.device_count()
            for gpu_id in range(num_gpus):
                current_vram = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)
                self.writer.add_scalar(f'VRAM_By_GPU/GPU_{gpu_id}', current_vram, global_step=global_step)

    def log_periodic(self, global_step):
        """ [週期性檢查] 較耗時，包含視覺化與詳細統計 """
        if global_step % self.check_period != 0:
            return

        # 1. Layer-wise Gradient Norm & Histograms (保持不變)
        layer_grads = defaultdict(list)
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                parts = name.split('.')
                group_name = '.'.join(parts[:3]) if len(parts) > 3 else name
                layer_grads[group_name].append(p.grad.data.norm(2).item())
                if 'weight' in name:

                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        log_print(_debug_print(name, p.grad))
                        raise ValueError(f"NaN or Inf detected in gradients of {name}")
                        continue

                    self.writer.add_histogram(f'GradHist/{name.replace("model.", "")}', p.grad, global_step)

        for group_name, norms in layer_grads.items():
            avg_norm = sum(norms) / len(norms)
            self.writer.add_scalar(f'LayerGradNorm/{group_name}', avg_norm, global_step)



        # === 修改後的 Attention 抓取邏輯 ===
        # 直接遍歷所有模組，找我們剛剛埋進去的 'last_attn_weights'
        for name, module in self.model.named_modules():
            if hasattr(module, 'last_attn_weights'):
                attn = module.last_attn_weights
                
                # 防呆
                if attn is None or attn.ndim < 3: 
                    continue

                # --- 偵錯區塊 (遇到 NaN 時開啟) ---
                if torch.isnan(attn).any():
                    print(f"Warning: {name} contains NaN before entropy calc.")
                    continue
                # -------------------------------

                # 1. 確保是機率分佈 (Probability Distribution)
                # 如果最小值小於 0 或 總和不等於 1 (容許誤差)，代表它是 Logits，需要 Softmax
                if attn.min() < 0 or not torch.allclose(attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1)), atol=1e-5):
                    # 假設最後一維是 sequence length (key)
                    p = torch.nn.functional.softmax(attn, dim=-1)
                else:
                    p = attn

                # 2. 計算 Entropy (更穩定的寫法)
                # 使用 clamp 確保數值永遠大於 0，避免 log(0) 或 log(負數)
                p = torch.clamp(p, min=1e-9, max=1.0)
                
                # Entropy = - sum(p * log(p))
                entropy = -torch.sum(p * torch.log(p), dim=-1)
                
                # 3. 紀錄
                avg_entropy = entropy.mean().item()
                
                # 再次檢查結果是否為 NaN (通常是因為輸入全部被 mask 掉導致)
                if not torch.isnan(torch.tensor(avg_entropy)):
                    self.writer.add_scalar(f'AttnEntropy/{name}', avg_entropy, global_step)



                # 2. 畫熱力圖 (只畫第一層和最後一層，且只畫 Head 0)
                if ('layers.0' in name or 'layers.5' in name) and global_step % (self.check_period * 2) == 0:
                    # 假設形狀是 [Batch, Heads, Q, K]，取 [0, 0]
                    # 如果形狀是 [Batch, Q, K]，取 [0]
                    if attn.ndim == 4:
                        heatmap_data = attn[0, 0].numpy() # 已經在 cpu 了
                    elif attn.ndim == 3:
                        heatmap_data = attn[0].numpy()
                    else:
                        continue

                    # 確保是正方形或長方形矩陣
                    if heatmap_data.ndim == 2:
                        fig, ax = plt.subplots(figsize=(6, 5))
                        try:
                            sns.heatmap(heatmap_data, ax=ax, cmap='viridis', vmin=0, vmax=1)
                            ax.set_title(f'{name}')
                            self.writer.add_figure(f'Visual/{name}', fig, global_step)
                        except Exception as e:
                            print(f"Plotting error: {e}")
                        finally:
                            plt.close(fig)

        # # 2. Attention Entropy & Visualization (修改這裡！)
        # for name, attn in self.attn_weights.items():
        #     # [Debug] 先檢查抓到的 Tensor 形狀到底是什麼
        #     # 正確的 Attention Weights 應該是 3D 或 4D: [Batch, Heads, Q, K] 或 [Batch, Q, K]
        #     if attn.ndim < 3:
        #         # 如果維度小於 3，代表這根本不是 Attention Map，可能是 Linear 的輸出
        #         # 默默跳過，或者 print warning (為了不刷屏，這裡選擇跳過)
        #         continue

        #     # 計算 Entropy
        #     p = attn + 1e-9
        #     # 假設最後一維是 Key Length，這才是我们要算的分布
        #     entropy = -torch.sum(p * torch.log(p), dim=-1) 
        #     avg_entropy = entropy.mean().item()
        #     self.writer.add_scalar(f'AttnEntropy/{name}', avg_entropy, global_step)

        #     # 3. Visualization: Attention Heatmap
        #     # 只畫第一層和最後一層，且只畫符合 2D 熱力圖形狀的
        #     if 'layers.0' in name or 'layers.last' in name:
        #         # 嘗試取出第一個 Batch, 第一個 Head
        #         # 根據維度動態調整
        #         if attn.ndim == 4: # [Batch, Heads, Q, K] -> 取 [0, 0]
        #             heatmap_data = attn[0, 0].cpu().numpy()
        #         elif attn.ndim == 3: # [Batch, Q, K] -> 取 [0]
        #             heatmap_data = attn[0].cpu().numpy()
        #         else:
        #             continue # 形狀太怪，不畫
                
        #         # [關鍵修正] 確保真的是 2D 矩陣，且不是 (N, 1) 這種向量
        #         if heatmap_data.ndim != 2 or heatmap_data.shape[0] < 2 or heatmap_data.shape[1] < 2:
        #             # print(f"Skip visualization for {name}: shape {heatmap_data.shape} is not a valid heatmap.")
        #             continue

        #         fig, ax = plt.subplots(figsize=(6, 5))
        #         try:
        #             sns.heatmap(heatmap_data, ax=ax, cmap='viridis', vmin=0, vmax=1)
        #             ax.set_title(f'{name}')
        #             self.writer.add_figure(f'Visual/{name}', fig, global_step)
        #         except Exception as e:
        #             print(f"Plotting failed for {name}: {e}")
        #         finally:
        #             plt.close(fig)

    # def close(self, 
    # ):
    #     for h in self.hook_handles:
    #         h.remove()










