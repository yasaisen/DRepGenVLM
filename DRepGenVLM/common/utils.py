"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2601261959
"""


import json
from datetime import datetime
import os
import inspect
import torch
import numpy as np
import random
from pathlib import Path


from .dist_utils import is_main_process


def log_print(
    text: str = '', 
    head: bool = False,
    newline: bool = True,
    traceback: bool = False,
):
    if is_main_process():
        frame = inspect.currentframe().f_back
        func_name = frame.f_code.co_name
        nowtime = datetime.now().strftime('%H:%M:%S')

        cls_name = None
        if 'self' in frame.f_locals:
            cls_name = frame.f_locals['self'].__class__.__name__
        elif 'cls' in frame.f_locals:
            cls_name = frame.f_locals['cls'].__name__

        if head:
            print()
        if cls_name:
            print(f"[{nowtime}] [{cls_name}.{func_name}] {text}", end='')
        else:
            print(f"[{nowtime}] [{func_name}] {text}", end='')
        if newline:
            print()
        if traceback:
            for line in inspect.stack():
                if line.function == func_name:
                    continue
                print(f"  -> {line.filename}:{line.lineno} in {line.function} on line {line.lineno}")

def highlight(
    text: str = 'debug',
):
    return f"\033[1;31;40m{text}\033[0m"

def highlight_2(
    text: str = 'debug',
):
    return f"\033[1;37;41m{text}\033[0m"

def _debug_print(
    variable_name: str,
    variable, 
):
    
    show_variable = variable
    if isinstance(variable, torch.Tensor):
        # show_variable = variable.shape, variable.dtype, variable.device, f"{highlight_2('min')}: {variable.min().item()}, {highlight_2('max')}: {variable.max().item()}, {highlight_2('nan')}: {torch.isnan(variable).any()}, {highlight_2('inf')}: {torch.isinf(variable).any()}"
        check = f"@nan: {torch.isnan(variable).any()}, inf: {torch.isinf(variable).any()}, min: {variable.min().item()}, max: {variable.max().item()}"
        if variable.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
            check += f", std: {variable.std().item()}, mean: {variable.mean().item()}"
        
        show_variable = variable.shape, variable.dtype, variable.device, check
    if isinstance(variable, np.ndarray):
        show_variable = variable.shape, variable.dtype, variable.device, f"@nan: {np.isnan(variable).any()}, inf: {np.isinf(variable).any()}, min: {variable.min().item()}, max: {variable.max().item()}"

    return f"[{highlight(variable_name)}] [{type(variable)}] {show_variable}"

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True




def tile_division(global_pos: float, tile_size: int):
    tiles_n = int(global_pos // tile_size)
    tile_pos = float(global_pos - tiles_n * tile_size)
    return (tiles_n, tile_pos)

def variable_checker(
    variable_name: str,
    variable, 
):
    if torch.isnan(variable).any() or torch.isinf(variable).any():
        raise RuntimeError(f"[{variable_name}] @nan: {torch.isnan(variable).any()}, inf: {torch.isinf(variable).any()}")

def _print_model_summary(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
    return f"Total Trainable Parameters: {total_params}"

# def move_to_device(obj, device):
#     if isinstance(obj, torch.Tensor):
#         return obj.to(device)
#     elif isinstance(obj, dict):
#         return {k: move_to_device(v, device) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [move_to_device(v, device) for v in obj]
#     elif isinstance(obj, tuple):
#         return tuple(move_to_device(v, device) for v in obj)
#     else:
#         return obj



from collections.abc import Mapping, Sequence

def move_to_device(obj, device, non_blocking=False):
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=non_blocking)
    
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device, non_blocking) for k, v in obj.items()}
    
    elif isinstance(obj, list):
        return [move_to_device(v, device, non_blocking) for v in obj]
    
    elif isinstance(obj, tuple):
        # 處理 namedtuple: 檢查是否有 _fields 屬性
        if hasattr(obj, '_fields'):
            # 使用原本的類別建構子重新建立 namedtuple
            return type(obj)(*(move_to_device(v, device, non_blocking) for v in obj))
        else:
            return tuple(move_to_device(v, device, non_blocking) for v in obj)
    
    elif isinstance(obj, torch.nn.Module):
        return obj.to(device)
    
    # 修改這裡：使用 Mapping 來捕捉所有字典類型的物件 (包含 UserDict, BatchEncoding 等)
    elif isinstance(obj, Mapping):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    
    # 修改這裡：使用 Sequence 來捕捉 list, tuple (但排除字串 str)
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        if isinstance(obj, tuple) and hasattr(obj, '_fields'): # 處理 namedtuple
             return type(obj)(*(move_to_device(v, device) for v in obj))
        return [move_to_device(v, device) for v in obj] # 回傳 list，通常沒問題
    
    else:
        return obj










def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return data
    
def save_list2json(
    meta_list, 
    save_filename, 
):
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    file_save_path = f'{save_filename}.json'
    with open(file_save_path, "w") as file:
        json.dump(meta_list, file, indent=4, default=convert)







def get_all_file_paths(root_dir: str):
    root = Path(root_dir)
    return [str(p).replace(root_dir + '\\', '').replace('\\', '/') for p in root.rglob("*") if p.is_file()]

