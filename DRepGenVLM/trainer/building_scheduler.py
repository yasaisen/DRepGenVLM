"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2026, yasaisen (clover)

 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.

 last modified in 2605251551
"""


from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR


def build_cosine_annealing_with_warmup_scheduler(
    optimizer,
    total_steps: int,
    warmup_steps: int,
):
    if warmup_steps == 0:
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        return scheduler

    if warmup_steps < total_steps:
        scheduler_warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=warmup_steps,
        )
        scheduler_cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cosine],
            milestones=[warmup_steps],
        )
        return scheduler

    if warmup_steps >= total_steps:
        scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            total_iters=total_steps,
        )
        return scheduler

    raise ValueError("Invalid warmup_steps and total_steps configuration.")












