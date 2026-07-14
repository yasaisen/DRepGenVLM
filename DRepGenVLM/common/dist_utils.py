"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2511042142
"""


import os
import torch
import torch.distributed as dist

from ..configs.DRGVLM_baseConfig import DRGVLM_baseConfig


def init_distributed_mode(
    cfg: DRGVLM_baseConfig,
):
    if cfg.distributed is False:
        print("Not using distributed mode")
        return
    
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        cfg.rank = int(os.environ["RANK"])
        cfg.world_size = int(os.environ["WORLD_SIZE"])
        cfg.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        cfg.rank = int(os.environ["SLURM_PROCID"])
        cfg.gpu = cfg.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        cfg.distributed = False
        return

    cfg.distributed = True

    torch.cuda.set_device(cfg.gpu)
    cfg.dist_backend = "nccl"
    print(
        "| distributed init (rank {}, world {}): {}".format(
            cfg.rank, cfg.world_size, cfg.dist_url
        ),
        flush=True,
    )
    import datetime
    dist.init_process_group(
        backend=cfg.dist_backend,
        init_method=cfg.dist_url,
        world_size=cfg.world_size,
        rank=cfg.rank,
        timeout=datetime.timedelta(
            days=365
        ),  # allow auto-downloading and de-compressing
    )
    dist.barrier()
    setup_for_distributed(cfg.rank == 0)

def setup_for_distributed(
    is_master: bool,
):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def cleanup_distributed_mode(
    cfg, 
):
    if cfg.distributed is False:
        print("Not using distributed mode")
        return
    
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1  # torch.cuda.device_count()??

def get_local_rank():
    return int(os.environ.get("LOCAL_RANK", 0)) # torch.cuda.current_device()??

def is_main_process():
    return get_rank() == 0

def is_local_main_process(): # One for each machine
    return get_local_rank() == 0

def barrier():
    if is_dist_avail_and_initialized():
        dist.barrier()


def average_tensor(tensor):
    if is_dist_avail_and_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= get_world_size()
    return tensor

def bcast_bool(
    flag: bool, 
    device, 
) -> bool:
    if not is_dist_avail_and_initialized():
        return flag
    t = torch.tensor([1 if flag else 0], device=device)
    torch.distributed.broadcast(t, src=0)
    return bool(t.item())










































# """
#  SPDX-License-Identifier: MIT
#  Copyright (c) 2025, yasaisen (clover)

#  This file is part of a project licensed under the MIT License.
#  See the LICENSE file in the project root for more information.

#  This file includes portions of code from Salesforce (BSD-3-Clause):
#    Copyright (c) 2022, salesforce.com, inc.
#    SPDX-License-Identifier: BSD-3-Clause
#    See: https://opensource.org/licenses/BSD-3-Clause

#  last modified in 2505192050
# """

# import datetime
# import functools
# import os

# import torch
# import torch.distributed as dist
# # import timm.models.hub as timm_hub #$$$


# def setup_for_distributed(is_master):
#     """
#     This function disables printing when not in master process
#     """
#     import builtins as __builtin__

#     builtin_print = __builtin__.print

#     def print(*args, **kwargs):
#         force = kwargs.pop("force", False)
#         if is_master or force:
#             builtin_print(*args, **kwargs)

#     __builtin__.print = print


# def is_dist_avail_and_initialized():
#     if not dist.is_available():
#         return False
#     if not dist.is_initialized():
#         return False
#     return True


# def get_world_size():
#     if not is_dist_avail_and_initialized():
#         return 1
#     return dist.get_world_size()


# def get_rank():
#     if not is_dist_avail_and_initialized():
#         return 0
#     return dist.get_rank()

# def is_main_process():
#     return get_rank() == 0


# def init_distributed_mode(cfg):
#     from ...common.utils import Config
#     args = Config(cfg)

#     if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
#         args.rank = int(os.environ["RANK"])
#         args.world_size = int(os.environ["WORLD_SIZE"])
#         args.gpu = int(os.environ["LOCAL_RANK"])
#     elif "SLURM_PROCID" in os.environ:
#         args.rank = int(os.environ["SLURM_PROCID"])
#         args.gpu = args.rank % torch.cuda.device_count()
#     else:
#         print("Not using distributed mode")
#         args.distributed = False
#         return

#     args.distributed = True

#     torch.cuda.set_device(args.gpu)
#     args.dist_backend = "nccl"
#     print(
#         "| distributed init (rank {}, world {}): {}".format(
#             args.rank, args.world_size, args.dist_url
#         ),
#         flush=True,
#     )
#     torch.distributed.init_process_group(
#         backend=args.dist_backend,
#         init_method=args.dist_url,
#         world_size=args.world_size,
#         rank=args.rank,
#         timeout=datetime.timedelta(
#             days=365
#         ),  # allow auto-downloading and de-compressing
#     )
#     torch.distributed.barrier()
#     setup_for_distributed(args.rank == 0)


# def get_dist_info():
#     if torch.__version__ < "1.0":
#         initialized = dist._initialized
#     else:
#         initialized = dist.is_initialized()
#     if initialized:
#         rank = dist.get_rank()
#         world_size = dist.get_world_size()
#     else:  # non-distributed training
#         rank = 0
#         world_size = 1
#     return rank, world_size


# def main_process(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         rank, _ = get_dist_info()
#         if rank == 0:
#             return func(*args, **kwargs)

#     return wrapper


# def download_cached_file(url, check_hash=True, progress=False):
#     """
#     Download a file from a URL and cache it locally. If the file already exists, it is not downloaded again.
#     If distributed, only the main process downloads the file, and the other processes wait for the file to be downloaded.
#     """

#     def get_cached_file_path():
#         # a hack to sync the file path across processes
#         parts = torch.hub.urlparse(url)
#         filename = os.path.basename(parts.path)
#         cached_file = os.path.join(timm_hub.get_cache_dir(), filename)

#         return cached_file

#     if is_main_process():
#         timm_hub.download_cached_file(url, check_hash, progress)

#     if is_dist_avail_and_initialized():
#         # pass
#         dist.barrier()

#     return get_cached_file_path()
