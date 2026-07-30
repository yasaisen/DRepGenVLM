"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2025, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2602011121
"""

import os
import torch

from .flashAttentionPatch import install_large_head_flash_attention_patch


# Optional compatibility fallback for FA3 interfaces that reject head_dim > 256.
# The fallback honors causal/window semantics and refuses unsupported options
# instead of silently changing attention behavior.
try:
    import flash_attn_interface

    install_large_head_flash_attention_patch(flash_attn_interface)
except ModuleNotFoundError as error:
    # SDPA configurations do not require flash_attn_interface.
    if error.name != "flash_attn_interface":
        print(
            "[FA3 Patch Warning] Failed to install large-head causal fallback: "
            f"{error}",
            flush=True,
        )
except Exception as error:
    print(
        "[FA3 Patch Warning] Failed to install large-head causal fallback: "
        f"{error}",
        flush=True,
    )


from ..common.utils import log_print, _debug_print, load_json_data, _print_model_summary


class modelBuilder:
    def __init__(self,
        weight_path: str = '../weights',
        add_pos: bool = True,
        flash_attention: bool = True,
        attn_type_list: list = None,
        freeze_tile_encoder: bool = True,
        freeze_slide_encoder: bool = True,
        focus_limit_ratio: float = None,
        lora_dict: dict = None,
    ):
        self.weight_path = weight_path
        self.weight_mapping_dict = {
            'prov-gigapath_tile': { # 1,134,953,984
                'checkpoint_path': 'tile_encoder.pth',
                'config_path': 'tile_encoder_args.json',
            }, 
            'prov-gigapath_slide': {
                'checkpoint_path': 'slide_encoder.pth',
                'config_path': 'slide_encoder_args.json',
            },
            'H-optimus-0': { # 1,134,774,272
                'checkpoint_path': 'H-optimus-0.pth',
                'config_path': 'H-optimus-0_args.json',
            },
            'vit_base': 'vit_base_patch32_clip_224.pth',  # 87,849,728
            'vit_large': 'vit_large_patch14_clip_224.pth',  # 303,966,976


            'gemma-3-4b-it': {
                'checkpoint_path': 'gemma-3-4b-it', 
            },
            'gemma-3-12b-it': {
                'checkpoint_path': 'gemma-3-12b-it', 
            },
            'gemma-3-27b-it': {
                'checkpoint_path': 'gemma-3-27b-it', 
            },

            'medgemma-4b-it': {
                'checkpoint_path': 'medgemma-4b-it', 
            },
            'medgemma-27b-it': {
                'checkpoint_path': 'medgemma-27b-it', 
            },
            'medgemma-1.5-4b-it': {
                'checkpoint_path': 'medgemma-1.5-4b-it', 
            },

            'gemma-4-31B-it': {
                'checkpoint_path': 'gemma-4-31B-it', 
            },
            'gemma-4-E2B-it': {
                'checkpoint_path': 'gemma-4-E2B-it', 
            },
            'gemma-4-E4B-it': {
                'checkpoint_path': 'gemma-4-E4B-it', 
            },

            'gpt-oss-20b': {
                'checkpoint_path': 'gpt-oss-20b', 
            },
            'gpt-oss-120b': {
                'checkpoint_path': 'gpt-oss-120b', 
            },

            'Qwen3.5-27B': {
                'checkpoint_path': 'Qwen3.5-27B', 
            },
            'Qwen3.6-27B': {
                'checkpoint_path': 'Qwen3.6-27B', 
            },
            'Qwen3.6-35B-A3B': {
                'checkpoint_path': 'Qwen3.6-35B-A3B', 
            },

            'histogpt-3b': {
                'checkpoint_path': 'histogpt-3b-6k-pruned.pth', 
                'config_path': 'biogpt-large-config.json', 
            },
            'histogpt-1b': {
                'checkpoint_path': 'histogpt-1b-6k-pruned.pth', 
            },

            'vicuna-7b': {
                'checkpoint_path': 'blip2/vicuna-7b', 
            }

        }

        self.freeze_tile_encoder = freeze_tile_encoder
        self.freeze_slide_encoder = freeze_slide_encoder
        self.add_pos = add_pos
        self.flash_attention = flash_attention
        self.attn_type_list = attn_type_list
        self.focus_limit_ratio = focus_limit_ratio
        self.lora_dict = lora_dict

    def create_registered_model(self,
        model_name: str, 
        device: str = None, 
        load_checkpoint: bool = True, 
        freeze_weight: bool = None,
    ):
        if model_name not in self.weight_mapping_dict.keys():
            raise ValueError(f'Model name {model_name} not found in self.weight_mapping_dict')
        
        if model_name == 'prov-gigapath_tile': # 1,134,953,984
            visual_encoder = self.create_tile_encoder(
                checkpoint_path=os.path.join(self.weight_path, self.weight_mapping_dict[model_name]['checkpoint_path']),
                config_path=os.path.join(self.weight_path, self.weight_mapping_dict[model_name]['config_path']),
                load_checkpoint=load_checkpoint, 
                freeze_weight=freeze_weight if freeze_weight is not None else self.freeze_tile_encoder, 
            )
        elif model_name == 'prov-gigapath_slide': # 86,330,892
            visual_encoder = self.create_slide_encoder(
                checkpoint_path=os.path.join(self.weight_path, self.weight_mapping_dict[model_name]['checkpoint_path']),
                config_path=os.path.join(self.weight_path, self.weight_mapping_dict[model_name]['config_path']),
                load_checkpoint=load_checkpoint, 
                freeze_weight=freeze_weight if freeze_weight is not None else self.freeze_slide_encoder, 
            )
        elif model_name == 'H-optimus-0': # 1,134,774,272
            visual_encoder = self.create_Hoptimus0(
                checkpoint_path=os.path.join(self.weight_path, self.weight_mapping_dict[model_name]['checkpoint_path']),
                config_path=os.path.join(self.weight_path, self.weight_mapping_dict[model_name]['config_path']),
                load_checkpoint=load_checkpoint, 
                freeze_weight=freeze_weight, 
            )
        elif model_name == 'vit_base': # 87,849,728
            import timm
            visual_encoder = timm.create_model(
                "vit_base_patch32_224_clip_laion2b", 
                pretrained=True, 
                # pretrained=False, 
                # checkpoint_path=os.path.join(self.weight_path, self.weight_mapping_dict[model_name]),
                dynamic_img_size=True
            )
            model = self.freeze_weight(model)
        elif model_name == 'vit_large': # 303,966,976
            import timm
            visual_encoder = timm.create_model(
                "vit_large_patch14_224_clip_laion2b", 
                pretrained=True, 
                # pretrained=False, 
                # checkpoint_path=os.path.join(self.weight_path, self.weight_mapping_dict[model_name]),
                dynamic_img_size=True
            )
            model = self.freeze_weight(model)

        if device is not None:
            visual_encoder.to(device)
        return visual_encoder.eval()



    @staticmethod
    def _build_manual_pp_device_map(
        config: object,
        model_name: str,
        num_gpus: int = None,
        project_name: str = None,
        vision_split_index: int = None,
    ) -> dict:
        """
        Build a layer-to-device mapping dict for Pipeline Parallelism.

        Strategy
        --------
        * All non-text modules (vision tower, projector, embed_tokens) → cuda:0.
        * Text transformer layers are split with endpoint de-loading:
          cuda:0 carries vision/projector/embed/lm_head and is always de-loaded.
          For CLEE GemmaX base-model paths without a runtime lm_head, cuda:N-1
          is no longer specially de-loaded and receives a regular layer share.
          When exactly 8 GPUs are requested, cuda:0 carries no text layers;
          all text transformer layers are balanced across cuda:1..cuda:7.
        * lm_head stays on cuda:0 because its weight is tied to embed_tokens.

        Supported model families
        ------------------------
        - medgemma-1.5-4b-it
          → loaded through Gemma3ForConditionalGeneration, then returned as Gemma3Model
        - medgemma-4b-it / medgemma-27b-it
        - gemma-3-4b-it / gemma-3-12b-it / gemma-3-27b-it
          → Gemma3ForConditionalGeneration
        - gemma-4-E4B-it
          → Gemma4Model, with audio_config disabled for CLEE
        - gemma-4-31B-it / gemma-4-E2B-it
          → Gemma4ForConditionalGeneration (via AutoModelForImageTextToText)
        - Qwen3.5-27B / Qwen3.6-27B / Qwen3.6-35B-A3B
          → Qwen3-VL (via AutoModelForImageTextToText)

        Falls back to the string "auto" for unknown models.
        """

        if num_gpus is not None and num_gpus > 1:
            log_print(f"[PP] Building manual device_map for '{model_name}' with {num_gpus} GPU(s)...")
        else:
            log_print(f"[PP] Invalid num_gpus={num_gpus} for '{model_name}', falling back to device_map='auto'.")
            return "auto"

        try:
            text_config = config.text_config if hasattr(config, "text_config") else config
            num_text_layers = text_config.num_hidden_layers
        except AttributeError:
            log_print(f"[PP] Cannot determine num_hidden_layers for '{model_name}', falling back to device_map='auto'.")
            return "auto"

        if vision_split_index is not None:
            try:
                vision_config = getattr(config, "vision_config", None)
                num_vision_layers = getattr(vision_config, "num_hidden_layers", None)
            except AttributeError:
                raise ValueError(f"Cannot determine num_hidden_layers for vision_config in '{model_name}'.")
            if not 0 <= int(vision_split_index) < int(num_vision_layers):
                raise ValueError(
                    "pp_vision_split_index must be inside the vision "
                    f"layer range [0, {num_vision_layers}), got "
                    f"{vision_split_index}."
                )


        if project_name in ["DRGVLM", "CLEE"] and model_name in ["medgemma-1.5-4b-it", "gemma-4-E4B-it"]:
            text_layer_fmt = "model.language_model.layers.{}"
            device_map: dict = {}
            if num_gpus < 8:
                if project_name == "CLEE" and model_name == "medgemma-1.5-4b-it":
                    device_map.update({
                        "model.vision_tower": "cuda:0",
                        "model.multi_modal_projector": "cuda:0",
                        "model.language_model.embed_tokens": "cuda:0",
                        "model.language_model.norm": "cuda:0",
                        "lm_head": "cuda:0",
                    })
                elif project_name == "CLEE" and model_name == "gemma-4-E4B-it":
                    device_map.update({
                        "vision_tower": "cuda:0",
                        "embed_vision": "cuda:0",
                        "language_model.embed_tokens": "cuda:0",
                        "language_model.embed_tokens_per_layer": "cuda:0",
                        "language_model.per_layer_model_projection": "cuda:0",
                    })
                elif project_name == "DRGVLM" and model_name == "medgemma-1.5-4b-it":
                    device_map.update({
                        "model.vision_tower": "cuda:0",
                        "model.multi_modal_projector": "cuda:0",
                        "model.language_model.embed_tokens": "cuda:0",
                        "model.language_model.norm": "cuda:0",
                        "lm_head": "cuda:0",
                    })
                else:
                    raise ValueError(
                        f"Unsupported project_name/model_name combination: {project_name}/{model_name}"
                    )
                text_gpu_indices = list(range(1, num_gpus))
            elif num_gpus >= 6:
                if project_name == "CLEE" and model_name == "medgemma-1.5-4b-it":
                    device_map.update({
                        "model.vision_tower": "cuda:0",
                        "model.multi_modal_projector": "cuda:0",
                        "model.language_model.embed_tokens": "cuda:0",
                        "model.language_model.norm": "cuda:0",
                        "lm_head": "cuda:0",
                    })
                    text_gpu_indices = list(range(1, num_gpus))
                elif project_name == "CLEE" and model_name == "gemma-4-E4B-it":
                    for layer_idx in range(int(num_vision_layers)):
                        device_map[f"vision_tower.encoder.layers.{layer_idx}"] = "cuda:0" if layer_idx < vision_split_index else "cuda:1"
                    device_map.update({
                        "vision_tower.patch_embedder": "cuda:1",
                        "embed_vision": "cuda:1",
                        "language_model.embed_tokens": "cuda:1",
                        "language_model.embed_tokens_per_layer": "cuda:1",
                        "language_model.per_layer_model_projection": "cuda:1",
                    })
                    text_gpu_indices = list(range(2, num_gpus))
                elif project_name == "DRGVLM" and model_name == "medgemma-1.5-4b-it":
                    for layer_idx in range(int(num_vision_layers)):
                        device_map[f"model.vision_tower.encoder.layers.{layer_idx}"] = "cuda:0" if layer_idx < vision_split_index else "cuda:1"
                    device_map.update({
                        "model.vision_tower.embeddings": "cuda:1",
                        "model.vision_tower.post_layernorm": "cuda:1",
                        "model.multi_modal_projector": "cuda:1",
                        "model.language_model.embed_tokens": "cuda:1",
                        "model.language_model.norm": "cuda:1",
                        "lm_head": "cuda:1",
                    })
                    text_gpu_indices = list(range(2, num_gpus))
                else:
                    raise ValueError(
                        f"Unsupported project_name/model_name combination: {project_name}/{model_name}"
                    )
            else:
                raise ValueError(
                    f"The {project_name} PP policy currently supports 1-8 "
                    f"GPUs, got {num_gpus}."
                )

            base_layers = num_text_layers // len(text_gpu_indices)
            remainder = num_text_layers % len(text_gpu_indices)
            layers_on_gpu = [0 for _ in range(num_gpus)]
            layer_idx = 0
            for offset, gpu_idx in enumerate(text_gpu_indices):
                # Put remainder blocks on later GPUs.  For 34 text layers this
                # gives [0, 11, 11, 12] on 4 GPUs and
                # [0, 0, 5, 5, 6, 6, 6, 6] on 8 GPUs.
                gets_remainder = (
                    remainder > 0
                    and offset >= len(text_gpu_indices) - remainder
                )
                count = base_layers + (1 if gets_remainder else 0)
                layers_on_gpu[gpu_idx] = count
                for _ in range(count):
                    device_map[text_layer_fmt.format(layer_idx)] = (
                        f"cuda:{gpu_idx}"
                    )
                    layer_idx += 1

            log_print(
                "[PP] Manual Gemma device_map built: "
                f"text_layers={num_text_layers}, "
                f"layers_on_gpu={layers_on_gpu}, "
                f"vision_split_index="
                f"{vision_split_index if num_gpus == 8 else 'n/a'}"
            )
            log_print(f"[PP] device_map = {device_map}")
            return device_map

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
        de_load_last_gpu = True
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
            if model_name == 'medgemma-1.5-4b-it':
                # This loader needs the outer lm_head only while matching the
                # checkpoint keys; create_language_model() returns .model to
                # CLEE, so the runtime PP graph has no lm_head.
                de_load_last_gpu = False

        elif model_name == 'gemma-4-E4B-it':
            # CLEE uses only image+text hidden states for Gemma4 E4B.  Load the
            # base Gemma4Model directly and disable audio_config in the loader,
            # so the PP map must use base-model keys and must not mention
            # lm_head/audio modules.
            static_cuda0_keys = [
                "vision_tower",
                "embed_vision",
                "language_model.embed_tokens",
            ]
            text_config = getattr(config, "text_config", None)
            if getattr(text_config, "hidden_size_per_layer_input", 0):
                static_cuda0_keys.extend([
                    "language_model.embed_tokens_per_layer",
                    "language_model.per_layer_model_projection",
                    "language_model.per_layer_projection_norm",
                ])
            text_layer_fmt = "language_model.layers.{}"
            text_norm_key  = "language_model.norm"
            lm_head_key    = None
            de_load_last_gpu = False

        elif model_name in ['gemma-4-31B-it', 'gemma-4-E2B-it']:
            # Gemma4ForConditionalGeneration.
            # 31B is image+text only; E2B/E4B also carry per-layer text input
            # embeddings and an audio tower. Add those only when present in
            # the config so the manual map matches each checkpoint layout.
            static_cuda0_keys = [
                "model.vision_tower",
                "model.embed_vision",
                "model.language_model.embed_tokens",
            ]
            text_config = getattr(config, "text_config", None)
            if getattr(text_config, "hidden_size_per_layer_input", 0):
                static_cuda0_keys.extend([
                    "model.language_model.embed_tokens_per_layer",
                    "model.language_model.per_layer_model_projection",
                    "model.language_model.per_layer_projection_norm",
                ])
            if getattr(config, "audio_config", None) is not None:
                static_cuda0_keys.extend([
                    "model.audio_tower",
                    "model.embed_audio",
                ])
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

        # lm_head is weight-tied to embed_tokens for ForConditionalGeneration
        # loaders, so keep it with embed_tokens.  Base GemmaX loaders omit it.
        if lm_head_key is not None:
            device_map[lm_head_key] = "cuda:0"

        # Distribute transformer layers.  cuda:0 carries non-text modules
        # and gets about half a regular layer share.  Older
        # ForConditionalGeneration paths keep the historical behavior of also
        # de-loading cuda:N-1; CLEE GemmaX base-model paths do not.
        layer_idx = 0
        if num_gpus <= 1:
            layers_on_gpu = [num_text_layers]
        elif num_gpus == 2:
            if de_load_last_gpu:
                cuda0_layers = num_text_layers // 2
            else:
                cuda0_layers = max(1, round(num_text_layers / 3)) if num_text_layers >= num_gpus else 1
            layers_on_gpu = [cuda0_layers, num_text_layers - cuda0_layers]
        elif num_gpus == 8:
            other_gpu_count = num_gpus - 1
            base_other_layers = num_text_layers // other_gpu_count
            remainder_other_layers = num_text_layers % other_gpu_count

            layers_on_gpu = [0]
            for gpu_offset in range(other_gpu_count):
                gets_remainder = gpu_offset >= other_gpu_count - remainder_other_layers
                n_layers = base_other_layers + (1 if gets_remainder else 0)
                layers_on_gpu.append(n_layers)
        else:
            if num_text_layers < num_gpus:
                layers_on_gpu = [1 if gpu_idx < num_text_layers else 0 for gpu_idx in range(num_gpus)]
            elif de_load_last_gpu:
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
            else:
                cuda0_layers = round(num_text_layers / (2 * num_gpus - 1))
                if num_text_layers >= num_gpus:
                    cuda0_layers = max(1, cuda0_layers)
                cuda0_layers = min(cuda0_layers, num_text_layers - (num_gpus - 1))

                remaining_layers = num_text_layers - cuda0_layers
                other_gpu_count = num_gpus - 1
                base_other_layers = remaining_layers // other_gpu_count
                remainder_other_layers = remaining_layers % other_gpu_count

                layers_on_gpu = [cuda0_layers]
                for gpu_offset in range(other_gpu_count):
                    # Put remainder layers on the later GPUs so cuda:N-1 is
                    # not accidentally the smallest non-home stage.
                    n_layers = base_other_layers + (1 if gpu_offset >= other_gpu_count - remainder_other_layers else 0)
                    layers_on_gpu.append(n_layers)

        for gpu_idx, n_layers in enumerate(layers_on_gpu):
            for _ in range(n_layers):
                device_map[text_layer_fmt.format(layer_idx)] = f"cuda:{gpu_idx}"
                layer_idx += 1

        # norm on the last GPU.
        # Accelerate's AlignDevicesHook on lm_head (cuda:0), when present, will
        # move the norm output from the last GPU to cuda:0 before lm_head runs.
        last_gpu = f"cuda:{num_gpus - 1}"
        device_map[text_norm_key] = last_gpu

        log_print(f"[PP] Manual device_map built: {num_text_layers} text layers across {num_gpus} GPU(s).")
        log_print(f"[PP] layers_on_gpu = {layers_on_gpu}")
        log_print(f"[PP] device_map = {device_map}")
        return device_map

    def create_language_model(self, 
        model_name: str, 
        project_name: str = None,
        freeze_weight: bool = True, 
        load_visual_processor: bool = False,
        torch_dtype = torch.bfloat16,
        lora_dict: dict = None,
        config_dict: dict = None,
        pp_num_gpus: int = None,
    ):
        if model_name not in self.weight_mapping_dict.keys():
            raise ValueError(f'Model name {model_name} not found in self.weight_mapping_dict')

        _config_dict = config_dict if config_dict is not None else {}
        model_path = os.path.join(self.weight_path, self.weight_mapping_dict[model_name]['checkpoint_path'])
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        _pp_device_map = self._build_manual_pp_device_map(
            config=config,
            model_name=model_name,
            num_gpus=pp_num_gpus,
            project_name=project_name,
            vision_split_index=_config_dict.get("pp_vision_split_index", None),
        )

        model = None
        max_seq_len = None
        if model_name in ['medgemma-1.5-4b-it', 'gemma-4-E4B-it'] and project_name == "CLEE":
            from transformers import AutoTokenizer as classTokenizer
            from transformers import AutoProcessor as classProcessor
            chat_template = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": None},
                        {"type": "text", "text": None}
                    ]
                }
            ]

            use_bidirectional_attention = _config_dict.get('use_bidirectional_attention', None)
            attn_implementation = _config_dict.get('attn_implementation', "sdpa")
            config_modifier = None
            return_model_attr = None
            if model_name == 'medgemma-1.5-4b-it':
                from transformers import Gemma3ForConditionalGeneration as classModel
                # Local MedGemma 1.5 checkpoints are keyed for the outer
                # conditional-generation wrapper.  Load through it for exact
                # checkpoint compatibility, then return only the base
                # Gemma3Model so CLEE does not retain lm_head.
                return_model_attr = "model"
            else:
                from transformers import Gemma4Model as classModel
                # Gemma4 E4B ships audio weights/config, but CLEE currently
                # consumes image+text only.  Removing audio_config prevents
                # audio_tower/embed_audio from being constructed or dispatched.
                if isinstance(use_bidirectional_attention, bool):
                    use_bidirectional_attention = 'all' if use_bidirectional_attention else None
                def config_modifier(config):
                    config.audio_config = None
                    return config

            model, processor, generate_func = self._bulid_language_model(
                model_path=model_path,
                load_visual_processor=load_visual_processor,
                torch_dtype=torch_dtype,
                classTokenizer=classTokenizer,
                classProcessor=classProcessor,
                classModel=classModel,
                set_pad_token_as_eos=True,
                use_bidirectional_attention=use_bidirectional_attention,
                attn_implementation=attn_implementation, 
                device_map=_pp_device_map,
                config_modifier=config_modifier,
                return_model_attr=return_model_attr,
            )
            hidden_size = model.config.text_config.hidden_size
            max_seq_len = model.config.text_config.max_position_embeddings

        elif model_name in ['medgemma-4b-it', 'medgemma-27b-it', 'medgemma-1.5-4b-it', 'gemma-3-4b-it', 'gemma-3-12b-it', 'gemma-3-27b-it']:
            from transformers import AutoTokenizer as classTokenizer
            # from .gemma3.processing_gemma3 import Gemma3Processor as classProcessor
            # from .gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration as classModel
            from transformers import Gemma3Processor as classProcessor
            from transformers import Gemma3ForConditionalGeneration as classModel
            chat_template = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": None},
                        {"type": "text", "text": None}
                    ]
                }
            ]
            use_bidirectional_attention = _config_dict.get('use_bidirectional_attention', None)
            attn_implementation = _config_dict.get('attn_implementation', "sdpa")
            model, processor, generate_func = self._bulid_language_model(
                model_path=model_path,
                load_visual_processor=load_visual_processor,
                torch_dtype=torch_dtype,
                classTokenizer=classTokenizer,
                classProcessor=classProcessor, 
                classModel=classModel,
                set_pad_token_as_eos=True,
                use_bidirectional_attention=use_bidirectional_attention,
                attn_implementation=attn_implementation,
                device_map=_pp_device_map,
            )
            hidden_size = model.config.text_config.hidden_size
            max_seq_len = model.config.text_config.max_position_embeddings

        elif model_name in ['Qwen3.5-27B', "gemma-4-31B-it", "Qwen3.6-35B-A3B", "Qwen3.6-27B", "gemma-4-E2B-it", 'gemma-4-E4B-it']:
            classTokenizer = None
            from transformers import AutoProcessor as classProcessor
            from transformers import AutoModelForImageTextToText as classModel
            chat_template = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": None},
                        {"type": "text", "text": None}
                    ]
                }
            ]

            use_bidirectional_attention = _config_dict.get('use_bidirectional_attention', None)
            attn_implementation = _config_dict.get('attn_implementation', "sdpa")
            if isinstance(use_bidirectional_attention, bool):
                use_bidirectional_attention = 'all' if use_bidirectional_attention else None
            model, processor, generate_func = self._bulid_language_model(
                model_path=model_path,
                load_visual_processor=load_visual_processor,
                torch_dtype=torch_dtype,
                classTokenizer=classTokenizer,
                classProcessor=classProcessor, 
                classModel=classModel,
                set_pad_token_as_eos=True,
                use_bidirectional_attention=use_bidirectional_attention,
                attn_implementation=attn_implementation,
                device_map=_pp_device_map,
            )
            hidden_size = model.config.text_config.hidden_size
            max_seq_len = model.config.text_config.max_position_embeddings

        elif model_name in ['gpt-oss-20b', 'gpt-oss-120b']:
            from transformers import AutoTokenizer as classTokenizer
            classProcessor = None
            # from .gpt_oss.modeling_gpt_oss import GptOssForCausalLM as classModel
            from transformers import GptOssForCausalLM as classModel
            chat_template = [
                {
                    "role": "user", 
                    "content": None
                },
            ]
            model, processor, generate_func = self._bulid_language_model(
                model_path=os.path.join(self.weight_path, self.weight_mapping_dict[model_name]['checkpoint_path']),
                load_visual_processor=load_visual_processor,
                torch_dtype=torch_dtype,
                classTokenizer=classTokenizer,
                classProcessor=classProcessor,  
                classModel=classModel,
            )
            hidden_size = model.config.hidden_size
            max_seq_len = model.config.max_position_embeddings

        elif model_name == 'vicuna-7b':
            from transformers import LlamaTokenizer
            from .blip2.modeling_llama import LlamaForCausalLM
            chat_template = None
            generate_func = None

            processor = LlamaTokenizer.from_pretrained(
                os.path.join(self.weight_path, self.weight_mapping_dict[model_name]['checkpoint_path']),
                use_fast=False, 
                truncation_side="left", 
            )
            model = LlamaForCausalLM.from_pretrained(
                os.path.join(self.weight_path, self.weight_mapping_dict[model_name]['checkpoint_path']), 
                torch_dtype=torch.float16, 
            )
            processor.add_special_tokens({'pad_token': '[PAD]'})
            processor.add_special_tokens({'bos_token': '</s>'})
            processor.add_special_tokens({'eos_token': '</s>'})
            processor.add_special_tokens({'unk_token': '</s>'})
            # processor.pad_token = processor.unk_token
            model.resize_token_embeddings(len(processor))

        elif model_name in ['histogpt-3b', 'histogpt-1b']:
            from transformers import BioGptConfig, BioGptTokenizer
            from ...models.histogpt.models import HistoGPTForCausalLM, PerceiverResamplerConfig
            from ...models.histogpt.helpers.inference import generate
            chat_template = None
            
            processor = BioGptTokenizer.from_pretrained("microsoft/biogpt")
            generate_func = generate

            if model_name == 'histogpt-3b':
                biogpt_config = BioGptConfig.from_pretrained(os.path.join(self.weight_path, self.weight_mapping_dict[model_name]['config_path']))
            elif model_name == 'histogpt-1b':
                biogpt_config = BioGptConfig()
            model = HistoGPTForCausalLM(biogpt_config, PerceiverResamplerConfig())
            state_dict = torch.load(
                os.path.join(self.weight_path, self.weight_mapping_dict[model_name]['checkpoint_path']), 
                map_location='cpu', 
                weights_only=False
            )
            msg = model.load_state_dict(state_dict)
            log_print(f"Loaded HistogPT model {model_name} with msg: {msg}")
        else:
            raise ValueError(f'Model name {model_name} not implemented yet')

        if freeze_weight:
            model = self.freeze_weight(model)

        if lora_dict is not None:
            model = self.add_lora_to_model(model, lora_dict)

        if hidden_size is None:
            hidden_size = model.config.hidden_size

        return model, processor, generate_func, hidden_size, max_seq_len, chat_template

    def _bulid_language_model(self, 
        model_path: str, 
        load_visual_processor: bool = False,
        torch_dtype = torch.bfloat16,
        classTokenizer = None,
        classProcessor = None,
        classModel = None,
        set_pad_token_as_eos: bool = False,
        use_bidirectional_attention: bool = None,
        attn_implementation: str = "sdpa",
        device_map = "auto",
        config_modifier = None,
        return_model_attr: str = None,
    ):
        """
        pp_device_map: explicit layer→device dict built by _build_manual_pp_device_map.
                       When None (default), falls back to device_map='auto'.
        """
        if load_visual_processor and classProcessor is None:
            log_print("[Warning] classProcessor is None, loading only tokenizer")
            load_visual_processor = False
        
        if load_visual_processor:
            processor = classProcessor.from_pretrained(model_path)
            try:
                if processor.tokenizer.pad_token is None or set_pad_token_as_eos:
                    log_print("[Warning] have no pad_token, set pad_token = eos_token")
                    processor.tokenizer.pad_token = processor.tokenizer.eos_token
                    processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
                else:
                    log_print(f"pad_token: {processor.tokenizer.pad_token}, pad_token_id: {processor.tokenizer.pad_token_id}")
                    log_print(f"eos_token: {processor.tokenizer.eos_token}, eos_token_id: {processor.tokenizer.eos_token_id}")
            except:
                log_print("[Warning] processor has no tokenizer attribute, skip setting pad_token")
                
        else:
            processor = classTokenizer.from_pretrained(model_path)
            if processor.pad_token is None or set_pad_token_as_eos:
                log_print("[Warning] have no pad_token, set pad_token = eos_token")
                processor.pad_token = processor.eos_token
                processor.pad_token_id = processor.eos_token_id
            else:
                log_print(f"pad_token: {processor.pad_token}, pad_token_id: {processor.pad_token_id}")
                log_print(f"eos_token: {processor.eos_token}, eos_token_id: {processor.eos_token_id}")

        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if use_bidirectional_attention is not None:
            config.text_config.use_bidirectional_attention = use_bidirectional_attention
            log_print(f"Setting use_bidirectional_attention to {use_bidirectional_attention} for model at {model_path}")
        if config_modifier is not None:
            config = config_modifier(config) or config
            log_print(f"Applied config_modifier to model at {model_path}")
        # log_print(config)

        model = classModel.from_pretrained(
            model_path,
            config=config, 
            dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation, 
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()

        if device_map is not None:
            log_print(f"Model loaded with manual PP device_map: {device_map}")

        if return_model_attr is not None:
            model = getattr(model, return_model_attr)

        return model, processor, getattr(model, "generate", None)

    @staticmethod
    def load_checkpoint_from_path(
        model, 
        checkpoint_path, 
        ignore_key = [], 
    ):
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            try:
                missing_keys, unexpected_keys = model.load_state_dict(state_dict["model"], strict=False)
            except:
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

            if len(missing_keys) > 0:
                for k in missing_keys:
                    for ig_key in ignore_key:
                        if ig_key not in k:
                            log_print(f"Missing {k}")
            if len(unexpected_keys) > 0:
                for k in unexpected_keys:
                    log_print(f"Unexpected {k}")

            log_print("\033[92m Successfully Loaded Pretrained model from {} \033[00m".format(checkpoint_path))
        else:
            log_print("\033[93m Pretrained weights not found at {}. Randomly initialized the model! \033[00m".format(checkpoint_path))

        return model

    @staticmethod
    def freeze_weight(model):
        log_print(_print_model_summary(model))
        for name, param in model.named_parameters():
            param.requires_grad = False
        log_print(_print_model_summary(model))
        return model

    @staticmethod
    def add_lora_to_model(
        model, 
        lora_dict: dict
    ):
        log_print("Adding LoRA...")
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=lora_dict['rank'], 
            lora_alpha=lora_dict['alpha'], 
            lora_dropout=lora_dict['dropout'],
            bias="none",
            task_type="CAUSAL_LM", 
            target_modules=lora_dict.get('target_modules', None), 
        )
        model = get_peft_model(model, peft_config)

        model.print_trainable_parameters()
        log_print(_print_model_summary(model))

        return model.train()

    @staticmethod
    def merge_lora_weights( 
        base_model, 
        adapter_save_path: str,
    ):
        log_print("Merging LoRA weights...")
        from peft import PeftModel

        model_to_merge = PeftModel.from_pretrained(base_model, adapter_save_path)
        merged_model = model_to_merge.merge_and_unload()

        return merged_model



    def create_tile_encoder(self,
        checkpoint_path: str,
        config_path: str,
        load_checkpoint: bool = True, 
        freeze_weight: bool = True,
    ):
        from .tile_encoder.modeling_vit import vit_giant_patch14_dinov2

        cfg = load_json_data(config_path)
        pretrained_cfg = cfg.get('pretrained_cfg', None)
        model_args = cfg.get('model_args', None)

        kwargs = {
            'dynamic_img_size': True, 
            'drop_rate': 0.3, # TODO tune drop_rate
        }
        if model_args:
            for k, v in model_args.items():
                kwargs.setdefault(k, v)

        model = vit_giant_patch14_dinov2(
            pretrained=False,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=None,
            cache_dir=None,
            **kwargs,
        )
        if load_checkpoint:
            model = self.load_checkpoint_from_path(
                model=model, 
                checkpoint_path=checkpoint_path
            )
        if freeze_weight:
            model = self.freeze_weight(model)

        # for name, param in model.named_parameters():
        #     # log_print(_debug_print(name, param))
        #     if '' in name:
        #         param.requires_grad = True
        log_print(_print_model_summary(model))

        return model

    def create_Hoptimus0(self,
        checkpoint_path: str,
        config_path: str,
        load_checkpoint: bool = True, 
        freeze_weight: bool = True,
    ):
        from .tile_encoder.modeling_vit import vit_giant_patch14_reg4_dinov2

        cfg = load_json_data(config_path)
        pretrained_cfg = cfg.get('pretrained_cfg', None)
        model_args = cfg.get('model_args', None)

        kwargs = {
            # 'dynamic_img_size': True, 
            # 'drop_rate': 0.3, # TODO tune drop_rate
        }
        if model_args:
            for k, v in model_args.items():
                kwargs.setdefault(k, v)

        model = vit_giant_patch14_reg4_dinov2(
            pretrained=False,
            pretrained_cfg=pretrained_cfg,
            pretrained_cfg_overlay=None,
            cache_dir=None,
            **kwargs,
        )
        if load_checkpoint:
            model = self.load_checkpoint_from_path(
                model=model, 
                checkpoint_path=checkpoint_path
            )
        if freeze_weight:
            model = self.freeze_weight(model)

        # for name, param in model.named_parameters():
        #     # log_print(_debug_print(name, param))
        #     if '' in name:
        #         param.requires_grad = True
        log_print(_print_model_summary(model))

        return model

    def create_slide_encoder(self,
        checkpoint_path: str,
        config_path: str, 
        load_checkpoint: bool = True, 
        freeze_weight: bool = True,
    ):
        from ..models.slide_encoder.modeling_longnet import LongNetViT
        from ..models.slide_encoder.config import EncoderConfig

        cfg = load_json_data(config_path)
        longnet_args = cfg.get('longnet_args', None)
        longnetvit_args = cfg.get('longnetvit_args', None)

        longnet_args["flash_attention"] = self.flash_attention

        longnet_args = EncoderConfig(**longnet_args)
        model = LongNetViT(
            longnet_args=longnet_args, 
            add_pos=self.add_pos, 
            attn_type_list=self.attn_type_list, 
            focus_limit_ratio=self.focus_limit_ratio, 
            **longnetvit_args
        )
        if load_checkpoint:
            model = self.load_checkpoint_from_path(
                model=model, 
                checkpoint_path=checkpoint_path, 
                ignore_key=[], 
            )
        if freeze_weight:
            model = self.freeze_weight(model)

        for name, param in model.named_parameters():
            # log_print(_debug_print(name, param))
            if 'msk_proj' in name:
                log_print(f"Unfreeze {name}")
                param.requires_grad = True

        log_print(_print_model_summary(model))


        if self.lora_dict is not None:
            from .lora.lora import ModelWithLoRA
            log_print("Adding LoRA...")
            model_lora = ModelWithLoRA(model)
            model_lora.add_lora(
                rank=self.lora_dict['rank'], 
                alpha=self.lora_dict['alpha'], 
                dropout=self.lora_dict['dropout'], 
                excluded_names_list=['msk_proj'],
                )
            model = model_lora.base_model
            log_print(_print_model_summary(model))

        return model
    
    @classmethod
    def from_config(cls, 
        cfg, 
    ):
        builder = cls(
            weight_path=cfg.weight_path, 
            freeze_tile_encoder=cfg.freeze_tile_encoder,
            freeze_slide_encoder=cfg.freeze_slide_encoder,

            add_pos=getattr(cfg, 'add_pos', None), 
            flash_attention=getattr(cfg, 'flash_attention', None), 
            attn_type_list=getattr(cfg, 'attn_type_list', None),
            focus_limit_ratio=getattr(cfg, 'focus_limit_ratio', None),
            lora_dict=getattr(cfg, 'lora_dict', None), 
        )

        return builder
