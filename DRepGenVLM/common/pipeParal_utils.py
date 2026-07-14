"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2026, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2605251707
"""


import os
import torch
from transformers import AutoConfig

from ..configs.DRGVLM_baseConfig import DRGVLM_baseConfig
from ..common.utils import log_print, _debug_print


def build_manual_pp_device_map(
    model_path: str,
    model_name: str,
    num_gpus: int,
) -> dict:
    """
    Build a layer-to-device mapping dict for Pipeline Parallelism.

    Strategy
    --------
    * All non-text modules (vision tower, projector, embed_tokens) → cuda:0.
    * Text transformer layers are split with both endpoint GPUs de-loaded:
        cuda:0 carries vision/projector/embed/lm_head, and cuda:N-1 carries
        the final norm plus the last-stage backward activations.
    * lm_head stays on cuda:0 because its weight is tied to embed_tokens.

    Supported model families
    ------------------------
    - medgemma-1.5-4b-it / medgemma-4b-it / medgemma-27b-it
    - gemma-3-4b-it / gemma-3-12b-it / gemma-3-27b-it
        → Gemma3ForConditionalGeneration
    - gemma-4-31B-it
        → Gemma4ForConditionalGeneration (via AutoModelForImageTextToText)
    - Qwen3.5-27B / Qwen3.6-27B / Qwen3.6-35B-A3B
        → Qwen3-VL (via AutoModelForImageTextToText)

    Falls back to the string "auto" for unknown models.
    """
    
    log_print(f"[PP] Building manual device_map for '{model_name}' with {num_gpus} GPU(s)...")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    try:
        num_text_layers = config.text_config.num_hidden_layers
    except AttributeError:
        try:
            num_text_layers = config.num_hidden_layers
        except AttributeError:
            log_print(f"[PP] Cannot determine num_hidden_layers for '{model_name}', falling back to device_map='auto'.")
            return "auto"

    # Strategy: explicit per-module enumeration (NO catch-all)
    # ---------------------------------------------------------------
    # The catch-all approach ("model": "cuda:0" + per-layer overrides)
    # causes accelerate's dispatch_model to place AlignDevicesHook at
    # the wrong granularity: it treats the whole "model" subtree as a
    # single cuda:0 unit and does NOT insert a hidden-state transfer
    # hook at layer boundaries, so hidden_states stays on cuda:0 even
    # when the layer weights are on cuda:1.
    #
    # The correct approach is to name EVERY top-level sub-module
    # explicitly in the device_map so that accelerate can independently
    # dispatch each one and place transition hooks exactly where the
    # device assignment changes.
    # ---------------------------------------------------------------
    if model_name in [
        'medgemma-1.5-4b-it', 'medgemma-4b-it', 'medgemma-27b-it',
        'gemma-3-4b-it', 'gemma-3-12b-it', 'gemma-3-27b-it',
    ]:
        # Gemma3ForConditionalGeneration confirmed parameter layout:
        #   model.vision_tower.*               → cuda:0
        #   model.multi_modal_projector.*      → cuda:0
        #   model.language_model.embed_tokens  → cuda:0
        #   model.language_model.layers.{i}    → split across GPUs
        #   model.language_model.norm          → last GPU
        #   lm_head  (weight-tied to embed_tokens) → cuda:0
        static_cuda0_keys = [
            "model.vision_tower",
            "model.multi_modal_projector",
            "model.language_model.embed_tokens",
        ]
        text_layer_fmt = "model.language_model.layers.{}"
        text_norm_key  = "model.language_model.norm"
        lm_head_key    = "lm_head"

    elif model_name == 'gemma-4-31B-it':
        # Gemma4ForConditionalGeneration — same top-level layout as Gemma3
        static_cuda0_keys = [
            "model.vision_tower",
            "model.multi_modal_projector",
            "model.language_model.embed_tokens",
        ]
        text_layer_fmt = "model.language_model.layers.{}"
        text_norm_key  = "model.language_model.norm"
        lm_head_key    = "lm_head"

    elif model_name in ['Qwen3.5-27B', 'Qwen3.6-27B', 'Qwen3.6-35B-A3B']:
        # Qwen3-VL via AutoModelForImageTextToText
        static_cuda0_keys = [
            "visual",
            "model.embed_tokens",
        ]
        text_layer_fmt = "model.layers.{}"
        text_norm_key  = "model.norm"
        lm_head_key    = "lm_head"

    else:
        log_print(f"[PP] Unknown model family '{model_name}', falling back to device_map='auto'.")
        return "auto"

    # Build the device_map dict
    device_map: dict = {}

    # Non-text modules and embed_tokens → cuda:0 (explicit, no catch-all)
    for key in static_cuda0_keys:
        device_map[key] = "cuda:0"

    # lm_head is weight-tied to embed_tokens; MUST stay on cuda:0
    # so the shared tensor is never relocated to another GPU.
    device_map[lm_head_key] = "cuda:0"

    layer_idx = 0
    if num_gpus <= 1:
        layers_on_gpu = [num_text_layers]
    elif num_gpus == 2:
        cuda0_layers = num_text_layers // 2
        layers_on_gpu = [cuda0_layers, num_text_layers - cuda0_layers]
    else:
        endpoint_layers = num_text_layers // (2 * (num_gpus - 1))
        if num_text_layers >= num_gpus:
            endpoint_layers = max(1, endpoint_layers)

        middle_gpu_count = num_gpus - 2
        middle_layers = num_text_layers - (2 * endpoint_layers)
        base_middle_layers = middle_layers // middle_gpu_count
        remainder_middle_layers = middle_layers % middle_gpu_count

        layers_on_gpu = [endpoint_layers]
        for middle_idx in range(middle_gpu_count):
            n_layers = base_middle_layers + (1 if middle_idx < remainder_middle_layers else 0)
            layers_on_gpu.append(n_layers)
        layers_on_gpu.append(endpoint_layers)

    for gpu_idx, n_layers in enumerate(layers_on_gpu):
        for _ in range(n_layers):
            device_map[text_layer_fmt.format(layer_idx)] = f"cuda:{gpu_idx}"
            layer_idx += 1

    # norm on the last GPU.
    # Accelerate's AlignDevicesHook on lm_head (cuda:0) will automatically
    # move the norm output from the last GPU to cuda:0 before lm_head runs.
    last_gpu = f"cuda:{num_gpus - 1}"
    device_map[text_norm_key] = last_gpu

    log_print(f"[PP] Manual device_map built: {num_text_layers} text layers across {num_gpus} GPU(s).")
    log_print(f"[PP] layers_on_gpu = {layers_on_gpu}")
    log_print(f"[PP] device_map = {device_map}")
    return device_map




































