"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2026, yasaisen (clover)

 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
"""


from collections import deque
from typing import Callable, Iterable, Iterator, List, Optional

from torch.utils.data import Sampler


class MaxROIBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        sampler: Iterable[int],
        roi_count_func: Callable[[int], int],
        batch_size: int,
        max_rois_per_update: Optional[int],
        drop_last: bool = False,
    ):
        self.sampler = sampler
        self.roi_count_func = roi_count_func
        self.batch_size = int(batch_size)
        self.max_rois_per_update = (
            int(max_rois_per_update) if max_rois_per_update is not None else None
        )
        self.drop_last = drop_last

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}.")
        if self.max_rois_per_update is not None and self.max_rois_per_update <= 0:
            raise ValueError(
                f"max_rois_per_update must be positive or None, got {self.max_rois_per_update}."
            )
        self._cached_batches: Optional[List[List[int]]] = None

    def _build_batches(self,
        indices: List[int],
    ) -> List[List[int]]:
        remaining = deque(indices)
        batches: List[List[int]] = []

        while remaining:
            batch: List[int] = []
            roi_total = 0

            for _ in range(self.batch_size):
                if not remaining:
                    break

                found_pos = None
                found_roi_count = None
                for pos, idx in enumerate(remaining):
                    roi_count = int(self.roi_count_func(idx))
                    fits_roi_budget = (
                        self.max_rois_per_update is None
                        or roi_total + roi_count <= self.max_rois_per_update
                    )
                    if fits_roi_budget or len(batch) == 0:
                        found_pos = pos
                        found_roi_count = roi_count
                        break

                if found_pos is None:
                    break

                for _ in range(found_pos):
                    remaining.append(remaining.popleft())

                batch.append(remaining.popleft())
                roi_total += int(found_roi_count)

            if len(batch) == self.batch_size:
                batches.append(batch)
            elif len(batch) > 0 and not self.drop_last:
                batches.append(batch)

        return batches

    def __iter__(self) -> Iterator[List[int]]:
        if self._cached_batches is None:
            batches = self._build_batches(indices=list(self.sampler))
        else:
            batches = self._cached_batches
            self._cached_batches = None

        yield from batches

    def __len__(self) -> int:
        if self._cached_batches is None:
            try:
                self._cached_batches = self._build_batches(indices=list(self.sampler))
            except TypeError:
                return 1
        return len(self._cached_batches)

    def set_epoch(self,
        epoch: int,
    ):
        self._cached_batches = None
        if hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)
