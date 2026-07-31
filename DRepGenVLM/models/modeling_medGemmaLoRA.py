"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2026, yasaisen (clover)
 
 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.
 
 last modified in 2607080100
"""


import os
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple

from transformers import AddedToken


from ..datasets.multiROI2DxResultDataset import ROI, Case
from ..models.modelBuilder import modelBuilder
from ..configs.DRGVLM_baseConfig import DRGVLM_baseConfig
from ..common.utils import _debug_print, _print_model_summary, log_print, move_to_device


SYSTEM_PROMPT = ""


class DownstreamRepGenVLM(nn.Module):
    def __init__(self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.use_text_gradient_checkpointing = False
        # Reentrant text gradient checkpointing may require a grad-bearing
        # inputs_embeds leaf.  The current non-checkpointed path does not.
        self.require_cached_input_grad = False
        self.use_shared_vision_cache = True  # controlled by from_config via cfg.use_shared_vision_cache
        self.use_vision_lora = False
        self.eval_prompt_batch_size = 6

    # ------------------------------------------------------------------
    # train() override – keep VLM backbone in eval when not checkpointing
    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        vlm_model = getattr(self, "vlm_model", None)
        if isinstance(vlm_model, nn.Module):
            # Keep the complete frozen VLM (including the transformer base)
            # in eval mode.  PEFT LoRA dropout is the only stochastic module
            # enabled during training; requires_grad on adapter parameters is
            # independent from Module.training.
            vlm_model.eval()
            if mode:
                for module in vlm_model.modules():
                    lora_dropout = getattr(module, "lora_dropout", None)
                    if isinstance(lora_dropout, nn.ModuleDict):
                        for dropout in lora_dropout.values():
                            dropout.train(True)
                    elif isinstance(lora_dropout, nn.Module):
                        lora_dropout.train(True)
        return self

    # ------------------------------------------------------------------
    # init_vlm_model
    # ------------------------------------------------------------------
    def init_vlm_model(self,
        vlm_processor,
        vlm_model,
        max_senLen: int,
    ):
        """Mount frozen VLM + processor; extract text backbone metadata."""
        self.vlm_processor = vlm_processor
        if not hasattr(self.vlm_processor, "tokenizer"):
            raise ValueError("vlm_processor must have a 'tokenizer' attribute.")

        self.model_type = vlm_model.config.model_type
        self.model_dtype = vlm_model.dtype
        try:
            self.hidden_size = vlm_model.config.text_config.hidden_size
        except AttributeError:
            self.hidden_size = vlm_model.config.hidden_size

        self.img_tok_id = getattr(vlm_model.config, "boi_token_id", None)
        if self.img_tok_id is None:
            self.img_tok_id = self.vlm_processor.tokenizer.convert_tokens_to_ids("<start_of_image>")
            log_print(f"Using tokenizer to get ID for '<start_of_image>': {self.img_tok_id}")
        else:
            log_print(f"Using model config's boi_token_id: {self.img_tok_id}")

        # Store VLM as a direct attribute (not in module tree) so
        # optimizer only sees LoRA params after apply_lora().
        object.__setattr__(self, "vlm_model", vlm_model)
        log_print(f"vlm_model mounted; dtype={self.model_dtype}, hidden_size={self.hidden_size}")

    # ------------------------------------------------------------------
    # _resolve_text_backbone
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_text_backbone(root: nn.Module) -> nn.Module:
        """BFS search for the text decoder (has .layers) inside a VLM or peft-wrapped model."""
        queue = [root]
        visited: set = set()
        while queue:
            module = queue.pop(0)
            mid = id(module)
            if mid in visited:
                continue
            visited.add(mid)
            layers = getattr(module, "layers", None)
            if layers is not None:
                try:
                    if len(layers) > 0 and isinstance(layers[0], nn.Module):
                        return module
                except Exception:
                    pass
            for name in ("language_model", "model", "text_model", "transformer", "base_model"):
                child = getattr(module, name, None)
                if isinstance(child, nn.Module):
                    queue.append(child)
        raise RuntimeError(
            "Unable to find text backbone (.layers) in the model. "
            "Check the VLM architecture or peft wrapping."
        )

    # ------------------------------------------------------------------
    # apply_lora
    # ------------------------------------------------------------------
    def apply_lora(self,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: List[str],
        use_vision_lora: bool = False,
    ):
        """Wrap the VLM with PEFT LoRA; only LoRA params become trainable."""
        try:
            from peft import get_peft_model, LoraConfig, TaskType
        except ImportError:
            raise ImportError("peft is required for LoRA. Install via: pip install peft")

        self.use_vision_lora = bool(use_vision_lora)
        # PEFT target-module lists match suffixes (for example q_proj), which
        # otherwise injects adapters into both Gemma's text decoder and vision
        # tower.  Keep those scopes independent from the vision-cache setting.
        vision_exclude_pattern = r"(?:.*\.)?vision_tower(?:\..*)?"
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=list(lora_target_modules),
            exclude_modules=None if self.use_vision_lora else vision_exclude_pattern,
            bias="none",
            inference_mode=False,
        )

        vlm_model = getattr(self, "vlm_model")
        peft_model = get_peft_model(vlm_model, lora_config)
        peft_model.print_trainable_parameters()

        trainable_names = []
        total_lora_params = 0
        vision_lora_params = 0
        for name, param in peft_model.named_parameters():
            if not param.requires_grad:
                continue
            trainable_names.append(name)
            param_count = int(param.numel())
            total_lora_params += param_count
            if "vision_tower" in name:
                vision_lora_params += param_count

        if total_lora_params == 0:
            raise RuntimeError(
                "LoRA injection produced no trainable parameters. "
                f"Check target_modules={lora_target_modules}."
            )
        if not self.use_vision_lora and vision_lora_params != 0:
            offending = [name for name in trainable_names if "vision_tower" in name]
            raise RuntimeError(
                "use_vision_lora=False, but trainable vision-tower adapters were found: "
                + ", ".join(offending[:8])
            )
        if self.use_vision_lora and vision_lora_params == 0:
            raise RuntimeError(
                "use_vision_lora=True, but no trainable vision-tower LoRA parameters "
                "were created. Check the configured target modules and model topology."
            )

        text_lora_params = total_lora_params - vision_lora_params
        log_print(
            "LoRA scope verified: "
            f"text_trainable={text_lora_params:,}, "
            f"vision_trainable={vision_lora_params:,}, "
            f"total_trainable={total_lora_params:,}"
        )

        # Replace raw vlm_model with the peft-wrapped version.
        # Use normal nn.Module attribute assignment so that vlm_model is registered
        # in self._modules and its parameters() traversal includes the LoRA params.
        # (object.__setattr__ is intentionally NOT used here, unlike init_vlm_model:
        #  peft has already frozen all base params; only LoRA params have requires_grad=True,
        #  so the optimizer will only pick those up.  The PP device_map is already set,
        #  and the trainer explicitly avoids calling .to() on this model.)
        self.vlm_model = peft_model

        # Resolve text backbone from peft-wrapped model.
        # Stored so that get_merged_embeds() can attach its hook, and
        # _get_embed_tokens() can retrieve the frozen token embeddings.
        text_backbone = self._resolve_text_backbone(peft_model)
        object.__setattr__(self, "text_backbone", text_backbone)
        log_print("text_backbone resolved after LoRA wrapping.")

        log_print(
            f"LoRA applied: r={lora_r}, alpha={lora_alpha}, "
            f"dropout={lora_dropout}, targets={lora_target_modules}, "
            f"use_vision_lora={self.use_vision_lora}"
        )

    # ------------------------------------------------------------------
    # init_sep
    # ------------------------------------------------------------------
    def init_sep(self,
        sep_str: str = "<unused0>",
        boc_str: str = "<unused1>",
    ):
        """Register ROI chunk separator tokens into the tokenizer."""
        self.sep_str = sep_str
        sep_token = AddedToken(self.sep_str, normalized=False, special=True)
        self.vlm_processor.tokenizer.add_special_tokens({"additional_special_tokens": [sep_token]})
        self.sep_tok_id = self.vlm_processor.tokenizer.convert_tokens_to_ids(self.sep_str)
        assert self.sep_tok_id != self.vlm_processor.tokenizer.unk_token_id, (
            f"{self.sep_str} is not in vocabulary."
        )
        log_print(f"[SEP] '{self.sep_str}' -> id={self.sep_tok_id}")

        self.boc_str = boc_str
        boc_token = AddedToken(self.boc_str, normalized=False, special=True)
        self.vlm_processor.tokenizer.add_special_tokens({"additional_special_tokens": [boc_token]})
        self.boc_tok_id = self.vlm_processor.tokenizer.convert_tokens_to_ids(self.boc_str)
        assert self.boc_tok_id != self.vlm_processor.tokenizer.unk_token_id, (
            f"{self.boc_str} is not in vocabulary."
        )
        log_print(f"[BOC] '{self.boc_str}' -> id={self.boc_tok_id}")

    # ------------------------------------------------------------------
    # init_criterion
    # ------------------------------------------------------------------
    def init_criterion(self, criterion):
        self.criterion = criterion

    # ------------------------------------------------------------------
    # Message building
    # ------------------------------------------------------------------
    @staticmethod
    def _active_dxitems(
        case: Case,
    ) -> List[str]:
        dxitem_rois = getattr(case, "DxItem_rois", None)
        if not isinstance(dxitem_rois, dict):
            raise AttributeError(
                f"case_id={case.case_id} has no DxItem_rois mapping. "
                "Load cases with the DxPair-aware dataset."
            )
        return [
            dx_item
            for dx_item, rois in dxitem_rois.items()
            if rois
        ]

    @staticmethod
    def _get_dxitem_rois(
        case: Case,
        DxItem: str,
    ) -> List[ROI]:
        dxitem_rois = getattr(case, "DxItem_rois", None)
        if not isinstance(dxitem_rois, dict) or DxItem not in dxitem_rois:
            raise KeyError(
                f"DxItem={DxItem!r} is not active for case_id={case.case_id}; "
                f"active={list(dxitem_rois or {})}"
            )
        rois = dxitem_rois[DxItem]
        if not rois:
            raise ValueError(
                f"DxItem={DxItem!r} has no assigned ROI for "
                f"case_id={case.case_id}."
            )
        return rois

    @classmethod
    def _group_dxitems_by_roi_signature(
        cls,
        case: Case,
        dx_items: Optional[List[str]] = None,
    ) -> List[List[str]]:
        """Group DxItems only when their ordered ROI inputs are identical."""
        dx_items = cls._active_dxitems(case) if dx_items is None else dx_items
        grouped: Dict[Tuple[int, ...], List[str]] = {}
        for dx_item in dx_items:
            signature = tuple(
                roi.global_idx
                for roi in cls._get_dxitem_rois(case, dx_item)
            )
            grouped.setdefault(signature, []).append(dx_item)
        return list(grouped.values())

    def _build_user_content(self,
        case: Case,
        DxItem: str,
    ) -> List[Dict]:
        """Build the user-turn content list for one (case, DxItem) pair.

        Layout:  [img, text_sep, img, text_sep, ..., DxItem question]
        """
        content = []
        for roi in self._get_dxitem_rois(case, DxItem):
            if roi.image is not None:
                content.append({"type": "image"})

            text = ""
            if roi.image is None:
                text += self.boc_str
            if roi.mpp is not None:
                text += f"MPP: {roi.mpp:.6f}, "
            if roi.cxcywh is not None:
                cx, cy, width, height = roi.cxcywh
                text += (
                    "cxcywh: "
                    f"({cx:.6f}, {cy:.6f}, {width:.6f}, {height:.6f}), "
                )
            text += self.sep_str
            content.append({"type": "text", "text": text})

        # Append the diagnostic question
        question_text = f"Based on the histological images above, what is the {DxItem}?"
        content.append({"type": "text", "text": question_text})
        return content

    def _build_train_messages(self,
        case: Case,
        DxItem: str,
    ) -> List[Dict]:
        """Full conversation for SFT: user + assistant turns."""
        user_content = self._build_user_content(case=case, DxItem=DxItem)
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": case.DxItem_targets[DxItem]},
        ]
        return messages

    def _build_inference_messages(self,
        case: Case,
        DxItem: str,
    ) -> List[Dict]:
        """Inference prompt: user turn only; model generates the answer."""
        user_content = self._build_user_content(case=case, DxItem=DxItem)
        messages = [
            {"role": "user", "content": user_content},
        ]
        return messages

    # ------------------------------------------------------------------
    # Tokenisation helpers
    # ------------------------------------------------------------------
    def _apply_chat_template(self,
        messages: List[Dict],
        add_generation_prompt: bool = False,
    ) -> str:
        common_kwargs = dict(
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if self.model_type in ["gemma4"]:
            common_kwargs["enable_thinking"] = False
        return self.vlm_processor.apply_chat_template(messages, **common_kwargs)

    def _encode_inputs(self,
        text_prompt: str,
        images: Optional[List],
    ) -> Dict[str, torch.Tensor]:
        """Run processor and move tensors to the VLM's first device."""
        raw_inputs = self.vlm_processor(
            text=text_prompt,
            images=images if images else None,
            return_tensors="pt",
        )
        inputs = {}
        for k, v in raw_inputs.items():
            if torch.is_tensor(v):
                v = v.to(self.device)
                if torch.is_floating_point(v):
                    v = v.to(self.model_dtype)
            inputs[k] = v
        return inputs

    # ------------------------------------------------------------------
    # build_train_inputs
    # ------------------------------------------------------------------
    def build_train_inputs(self,
        case: Case,
        DxItem: str,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Build tokenised inputs + labels for one (case, DxItem) pair.

        Returns:
            inputs : dict ready for vlm_model forward
            labels : (1, S) long tensor; -100 on user/image positions
        """
        messages = self._build_train_messages(case=case, DxItem=DxItem)
        full_prompt = self._apply_chat_template(messages, add_generation_prompt=False)

        # Also build the prompt-only version to find where assistant starts
        prompt_only_messages = self._build_inference_messages(case=case, DxItem=DxItem)
        prompt_only = self._apply_chat_template(prompt_only_messages, add_generation_prompt=True)

        images = [
            roi.image
            for roi in self._get_dxitem_rois(case, DxItem)
            if roi.image is not None
        ]

        inputs = self._encode_inputs(text_prompt=full_prompt, images=images)
        prompt_inputs = self._encode_inputs(text_prompt=prompt_only, images=images)

        input_ids = inputs["input_ids"]          # (1, S_full)
        prompt_len = prompt_inputs["input_ids"].shape[1]  # length of user+generation_prompt

        # Build labels: -100 everywhere except the assistant answer tokens
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100

        # Also mask any padding
        if "attention_mask" in inputs:
            pad_mask = inputs["attention_mask"] == 0
            labels[pad_mask] = -100

        return inputs, labels

    # ------------------------------------------------------------------
    # Shared vision cache helpers
    # ------------------------------------------------------------------
    def _get_embed_tokens(self) -> nn.Module:
        """Return the frozen token-embedding module from the text backbone."""
        text_backbone = getattr(self, "text_backbone", None)
        if text_backbone is None:
            raise RuntimeError(
                "_get_embed_tokens: text_backbone not initialised. "
                "Call apply_lora() before training."
            )
        embed_tokens = getattr(text_backbone, "embed_tokens", None)
        if embed_tokens is None:
            raise RuntimeError("embed_tokens not found in text_backbone.")
        return embed_tokens

    @torch.no_grad()
    def get_merged_embeds(self,
        inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Run a no_grad VLM forward and capture merged multimodal embeddings.

        A pre-hook on text_backbone.layers[0] captures the hidden_states tensor
        just before the first transformer layer, i.e. after the vision tower and
        projector have expanded image tokens into the text sequence.

        Returns:
            merged_embeds: (1, S_merged, H) on self.device
        """
        text_backbone = getattr(self, "text_backbone", None)
        if text_backbone is None:
            raise RuntimeError("get_merged_embeds: text_backbone not initialised.")

        captured: Dict[str, torch.Tensor] = {}

        def _hook(module, args, kwargs):
            x = args[0] if args else kwargs.get("hidden_states")
            if x is not None:
                captured["embeds"] = x.detach().clone()

        first_layer = text_backbone.layers[0]
        handle = first_layer.register_forward_pre_hook(_hook, with_kwargs=True)
        try:
            vlm = getattr(self, "vlm_model")
            vlm(**inputs, output_hidden_states=False, use_cache=False)
        finally:
            handle.remove()

        if "embeds" not in captured:
            raise RuntimeError(
                "get_merged_embeds: hook did not capture any embeddings. "
                "Verify that text_backbone.layers[0] is the first text decoder layer."
            )
        return captured["embeds"].to(device=self.device, dtype=self.model_dtype)

    def _get_vision_prefix_embeds(self,
        case: Case,
        DxItem: str,
    ) -> Tuple[
        torch.Tensor,
        int,
        int,
        torch.Tensor,
        Optional[torch.Tensor],
    ]:
        """One no_grad VLM forward to obtain the shared vision prefix embeddings.

        Strategy:
          1. Build the inference prompt for one DxItem's assigned ROI sequence.
          2. Run get_merged_embeds() to capture the full merged sequence.
          3. Use the position of the last sep token (<unused0>) in input_ids to
             locate where the question begins in token space.
          4. Since all tokens after the question are pure text (no image expansion),
             the merged split position is:
               vision_prefix_len_merged = S_merged - (S_ids - q_start_in_ids)

        Returns:
            vision_prefix_embeds    : (1, V, H) – vision part of merged embeddings
            q_start_in_ids          : int – last-sep + 1 in token space (shared across DxItems)
            vision_prefix_len_merged: int – corresponding position in merged space
            vision_prefix_attn_mask : (1, V) – cached attention-mask prefix
            vision_prefix_token_type_ids: optional (1, V) Gemma multimodal token types
        """
        messages = self._build_inference_messages(case=case, DxItem=DxItem)
        prompt = self._apply_chat_template(messages, add_generation_prompt=True)
        images = [
            roi.image
            for roi in self._get_dxitem_rois(case, DxItem)
            if roi.image is not None
        ]
        ref_inputs = self._encode_inputs(text_prompt=prompt, images=images)

        # Capture merged embeddings (no_grad)
        merged_embeds = self.get_merged_embeds(ref_inputs)  # (1, S_merged, H)
        ref_ids = ref_inputs["input_ids"]                    # (1, S_ref_ids)

        # Locate question start via last sep token position.
        # <unused0> terminates every ROI text block; the question follows immediately.
        sep_positions = (ref_ids[0] == self.sep_tok_id).nonzero(as_tuple=True)[0].tolist()
        if not sep_positions:
            raise RuntimeError(
                f"[VisionCache] No sep token (id={self.sep_tok_id}) found in reference "
                f"input_ids for case_id={case.case_id}. "
                "Cannot determine vision/text split position."
            )
        q_start_in_ids = sep_positions[-1] + 1  # token position right after last sep

        # Compute merged-space split.
        # Tokens from q_start_in_ids to end of ref_ids are pure text (no image
        # expansion), so they occupy exactly (S_ref_ids - q_start_in_ids) positions
        # at the tail of merged_embeds.
        tail_text_len = ref_ids.shape[1] - q_start_in_ids
        vision_prefix_len_merged = merged_embeds.shape[1] - tail_text_len

        if vision_prefix_len_merged <= 0:
            raise RuntimeError(
                f"[VisionCache] vision_prefix_len_merged={vision_prefix_len_merged} <= 0 "
                f"for case_id={case.case_id}. "
                f"S_merged={merged_embeds.shape[1]}, "
                f"S_ref_ids={ref_ids.shape[1]}, "
                f"q_start_in_ids={q_start_in_ids}."
            )

        vision_prefix_embeds = merged_embeds[:, :vision_prefix_len_merged, :].clone()

        ref_attn_mask = ref_inputs.get("attention_mask")
        if ref_attn_mask is None:
            ref_attn_mask = torch.ones_like(ref_ids, dtype=torch.long)
        if ref_attn_mask.shape[1] != merged_embeds.shape[1]:
            raise RuntimeError(
                "[VisionCache] attention_mask and merged embeddings have different "
                f"sequence lengths ({ref_attn_mask.shape[1]} vs "
                f"{merged_embeds.shape[1]}). Disable use_shared_vision_cache for "
                "this processor/model combination."
            )
        vision_prefix_attn_mask = ref_attn_mask[
            :, :vision_prefix_len_merged
        ].to(device=vision_prefix_embeds.device)

        ref_token_type_ids = ref_inputs.get("token_type_ids")
        vision_prefix_token_type_ids = None
        if ref_token_type_ids is not None:
            if ref_token_type_ids.shape[1] != merged_embeds.shape[1]:
                raise RuntimeError(
                    "[VisionCache] token_type_ids and merged embeddings have different "
                    f"sequence lengths ({ref_token_type_ids.shape[1]} vs "
                    f"{merged_embeds.shape[1]})."
                )
            vision_prefix_token_type_ids = ref_token_type_ids[
                :, :vision_prefix_len_merged
            ].to(device=vision_prefix_embeds.device)

        return (
            vision_prefix_embeds,
            q_start_in_ids,
            vision_prefix_len_merged,
            vision_prefix_attn_mask,
            vision_prefix_token_type_ids,
        )

    def build_train_inputs_with_vision_cache(self,
        case,
        DxItem: str,
        vision_prefix_embeds: torch.Tensor,
        q_start_in_ids: int,
        vision_prefix_len_merged: int,
        vision_prefix_attn_mask: torch.Tensor,
        vision_prefix_token_type_ids: Optional[torch.Tensor],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
    ]:
        """Build cached multimodal inputs while preserving Gemma mask semantics.

        The cached vision_prefix_embeds (no grad) is concatenated with a text-only
        suffix embedding (question + template + answer tokens).  The suffix is
        obtained by running embed_tokens on the token IDs from q_start_in_ids
        onwards, avoiding a second costly vision-tower forward.

        Labels are -100 for the vision prefix and the question/template tokens;
        real token IDs are placed at the answer positions.

        Returns:
            combined_embeds : (1, S, H)
            attn_mask       : (1, S)
            token_type_ids  : optional (1, S) Gemma multimodal token types
            labels          : (1, S)  – -100 on non-answer positions
        """
        images = [
            roi.image
            for roi in self._get_dxitem_rois(case, DxItem)
            if roi.image is not None
        ]

        # Full training prompt (user + assistant turn)
        full_messages = self._build_train_messages(case=case, DxItem=DxItem)
        full_prompt = self._apply_chat_template(full_messages, add_generation_prompt=False)
        full_inputs = self._encode_inputs(text_prompt=full_prompt, images=images)
        full_ids = full_inputs["input_ids"]  # (1, S_full)

        # Inference prompt to locate the start of the answer in full_ids.
        # prompt_len = length of user turn + generation-prompt tokens.
        infer_messages = self._build_inference_messages(case=case, DxItem=DxItem)
        infer_prompt = self._apply_chat_template(infer_messages, add_generation_prompt=True)
        infer_inputs = self._encode_inputs(text_prompt=infer_prompt, images=images)
        prompt_len = infer_inputs["input_ids"].shape[1]

        # Suffix token IDs: from question start to end of answer + EOS.
        # q_start_in_ids is shared only by DxItems with this exact ROI sequence.
        suffix_ids = full_ids[:, q_start_in_ids:]  # (1, S_suffix)

        # Embed suffix using the frozen embed_tokens (no_grad; not in LoRA targets).
        embed_tokens = self._get_embed_tokens()
        with torch.no_grad():
            suffix_embeds = embed_tokens(
                suffix_ids.to(embed_tokens.weight.device)
            ).to(device=vision_prefix_embeds.device, dtype=vision_prefix_embeds.dtype)

        # Concatenate: [vision_prefix | text_suffix]
        combined_embeds = torch.cat([vision_prefix_embeds, suffix_embeds], dim=1)
        S_combined = combined_embeds.shape[1]

        full_attn_mask = full_inputs.get("attention_mask")
        if full_attn_mask is None:
            full_attn_mask = torch.ones_like(full_ids, dtype=torch.long)
        if full_attn_mask.shape[1] != full_ids.shape[1]:
            raise RuntimeError(
                "[VisionCache] full attention_mask length does not match input_ids: "
                f"{full_attn_mask.shape[1]} vs {full_ids.shape[1]}."
            )
        suffix_attn_mask = full_attn_mask[:, q_start_in_ids:].to(
            device=combined_embeds.device
        )
        attn_mask = torch.cat(
            [vision_prefix_attn_mask.to(combined_embeds.device), suffix_attn_mask],
            dim=1,
        )

        full_token_type_ids = full_inputs.get("token_type_ids")
        if (vision_prefix_token_type_ids is None) != (full_token_type_ids is None):
            raise RuntimeError(
                "[VisionCache] token_type_ids are present in only one of the "
                "reference/full prompts; cached forward would not be equivalent."
            )
        combined_token_type_ids = None
        if full_token_type_ids is not None:
            if full_token_type_ids.shape[1] != full_ids.shape[1]:
                raise RuntimeError(
                    "[VisionCache] full token_type_ids length does not match input_ids: "
                    f"{full_token_type_ids.shape[1]} vs {full_ids.shape[1]}."
                )
            suffix_token_type_ids = full_token_type_ids[:, q_start_in_ids:].to(
                device=combined_embeds.device
            )
            combined_token_type_ids = torch.cat(
                [
                    vision_prefix_token_type_ids.to(combined_embeds.device),
                    suffix_token_type_ids,
                ],
                dim=1,
            )

        if attn_mask.shape[1] != S_combined:
            raise RuntimeError(
                "[VisionCache] rebuilt attention_mask length does not match "
                f"combined embeddings: {attn_mask.shape[1]} vs {S_combined}."
            )
        if (
            combined_token_type_ids is not None
            and combined_token_type_ids.shape[1] != S_combined
        ):
            raise RuntimeError(
                "[VisionCache] rebuilt token_type_ids length does not match "
                f"combined embeddings: {combined_token_type_ids.shape[1]} vs {S_combined}."
            )

        # Labels:
        #   - vision prefix tokens  → -100
        #   - question + template   → -100  (suffix tokens before the answer)
        #   - answer + EOS          → actual token IDs
        #
        #   Within suffix, answer starts at (prompt_len - q_start_in_ids).
        #   In combined space it starts at vision_prefix_len_merged + that offset.
        answer_start_in_suffix = prompt_len - q_start_in_ids
        answer_start_in_combined = vision_prefix_len_merged + answer_start_in_suffix

        labels = torch.full((1, S_combined), -100, dtype=torch.long, device=combined_embeds.device)
        if answer_start_in_combined < S_combined:
            answer_ids = suffix_ids[:, answer_start_in_suffix:].to(labels.device)
            labels[:, answer_start_in_combined:] = answer_ids
        else:
            log_print(
                f"[VisionCache][WARN] answer_start_in_combined={answer_start_in_combined} "
                f">= S_combined={S_combined} for case_id={case.case_id}, DxItem={DxItem}. "
                "No answer tokens found in labels – loss will be zero for this pair."
            )
        labels[attn_mask == 0] = -100

        return combined_embeds, attn_mask, combined_token_type_ids, labels

    # ------------------------------------------------------------------
    # Pair-wise loss helpers
    # ------------------------------------------------------------------
    def prepare_case_loss_context(
        self,
        case: Case,
        dx_items: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Dict[str, Any]]]:
        """Prepare immutable vision contexts for active DxItems.

        Contexts are shared only between DxItems whose ordered sampled ROI
        sequences are exactly identical.  Different ROI groups must never reuse
        a multimodal prefix.
        """
        if not self.use_shared_vision_cache:
            return None

        dx_items = self._active_dxitems(case) if dx_items is None else dx_items
        contexts: Dict[str, Dict[str, Any]] = {}
        for group_dx_items in self._group_dxitems_by_roi_signature(
            case,
            dx_items=dx_items,
        ):
            ref_dx_item = group_dx_items[0]
            (
                vision_prefix_embeds,
                q_start_in_ids,
                vision_prefix_len_merged,
                vision_prefix_attn_mask,
                vision_prefix_token_type_ids,
            ) = self._get_vision_prefix_embeds(
                case=case,
                DxItem=ref_dx_item,
            )
            context = {
                "vision_prefix_embeds": vision_prefix_embeds,
                "q_start_in_ids": q_start_in_ids,
                "vision_prefix_len_merged": vision_prefix_len_merged,
                "vision_prefix_attn_mask": vision_prefix_attn_mask,
                "vision_prefix_token_type_ids": (
                    vision_prefix_token_type_ids
                ),
            }
            for dx_item in group_dx_items:
                contexts[dx_item] = context
        return contexts

    def calculate_dxitem_loss(
        self,
        case: Case,
        DxItem: str,
        case_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward and compute loss for exactly one (case, DxItem) pair."""
        if DxItem not in case.DxItem_targets:
            raise KeyError(
                f"Unknown DxItem={DxItem!r} for case_id={case.case_id}; "
                f"available={list(case.DxItem_targets)}"
            )

        vlm = getattr(self, "vlm_model")
        if self.use_shared_vision_cache:
            if case_context is None:
                raise ValueError(
                    "case_context is required when use_shared_vision_cache=True. "
                    "Call prepare_case_loss_context(case) before forwarding "
                    "the active DxItem."
                )
            log_print(
                f"case_id={case.case_id}, DxItem={DxItem}, "
                f"rois={len(self._get_dxitem_rois(case, DxItem))}, "
                "context_len="
                f"{case_context['vision_prefix_attn_mask'].shape}, "
            )
            combined_embeds, attn_mask, token_type_ids, labels = (
                self.build_train_inputs_with_vision_cache(
                    case=case,
                    DxItem=DxItem,
                    vision_prefix_embeds=case_context["vision_prefix_embeds"],
                    q_start_in_ids=case_context["q_start_in_ids"],
                    vision_prefix_len_merged=case_context["vision_prefix_len_merged"],
                    vision_prefix_attn_mask=case_context["vision_prefix_attn_mask"],
                    vision_prefix_token_type_ids=case_context[
                        "vision_prefix_token_type_ids"
                    ],
                )
            )
            # Text LoRA parameters create their own autograd path even though the
            # cached inputs are detached.  A dummy input leaf is needed only for
            # model topologies such as reentrant gradient checkpointing that
            # explicitly require a grad-bearing input.
            # if self.require_cached_input_grad:
            #     combined_embeds = combined_embeds.requires_grad_(True)
            forward_kwargs = {
                "inputs_embeds": combined_embeds,
                "attention_mask": attn_mask,
                "use_cache": False,
            }
            if token_type_ids is not None:
                forward_kwargs["token_type_ids"] = token_type_ids
            outputs = vlm(**forward_kwargs)
        else:
            inputs, labels = self.build_train_inputs(case=case, DxItem=DxItem)
            outputs = vlm(
                **inputs,
                output_hidden_states=False,
                use_cache=False,
            )

        context = f"case_id={case.case_id}, DxItem={DxItem}"
        return self.criterion(
            logits=outputs.logits,
            labels=labels,
            context=context,
        )

    # ------------------------------------------------------------------
    # calculate_loss  (aggregate/no-backward callers such as validation)
    # ------------------------------------------------------------------
    def calculate_loss(self,
        batch_cases: List[Case],
        case_contexts: Optional[
            Dict[int, Optional[Dict[str, Dict[str, Any]]]]
        ] = None,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, Dict[str, Any]]]:
        """Return the mean loss for callers that do not stream backward.

        Training uses prepare_case_loss_context() + calculate_dxitem_loss()
        directly so each pair's graph can be released immediately.  Validation
        runs under torch.no_grad(), where aggregating scalar losses is harmless.
        """
        batch_cases = self._normalize_batch(batch_cases)
        all_losses: List[torch.Tensor] = []
        output_case_dict: Dict[int, Dict[str, Any]] = {}

        for case in batch_cases:
            active_dx_items = self._active_dxitems(case)
            log_print(
                f"case_id={case.case_id}, unique_rois={len(case.rois)}, "
                f"active_DxItems={active_dx_items}"
            )
            output_case_dict[case.global_idx] = {}
            case_context_map = (
                case_contexts.get(case.global_idx)
                if case_contexts is not None
                else self.prepare_case_loss_context(case)
            )

            for DxItem in active_dx_items:
                dxitem_context = (
                    case_context_map.get(DxItem)
                    if case_context_map is not None
                    else None
                )
                loss_dict = self.calculate_dxitem_loss(
                    case=case,
                    DxItem=DxItem,
                    case_context=dxitem_context,
                )
                all_losses.append(loss_dict["total_loss"])
                output_case_dict[case.global_idx][DxItem] = {
                    "pred_txt": None,
                    "gt_txt": case.DxItem_targets[DxItem],
                }

        if len(all_losses) == 0:
            raise RuntimeError("Empty batch; unable to compute loss.")

        total_loss = torch.stack(all_losses).mean()
        return {"total_loss": total_loss}, output_case_dict

    # ------------------------------------------------------------------
    # generate_outputs  (inference / validation)
    # ------------------------------------------------------------------
    def _build_inference_suffix_tokens(
        self,
        case: Case,
        DxItem: str,
        expected_q_start: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Tokenize only to recover the post-vision question suffix.

        The processor is called one prompt at a time, so pixel_values are never
        replicated six-fold.  Pixel tensors are discarded without entering the
        VLM; the already cached merged vision prefix is used for generation.
        """
        messages = self._build_inference_messages(case=case, DxItem=DxItem)
        prompt = self._apply_chat_template(
            messages,
            add_generation_prompt=True,
        )
        images = [
            roi.image
            for roi in self._get_dxitem_rois(case, DxItem)
            if roi.image is not None
        ]
        raw_inputs = self.vlm_processor(
            text=prompt,
            images=images if images else None,
            return_tensors="pt",
        )
        input_ids = raw_inputs["input_ids"]
        sep_positions = (
            input_ids[0] == self.sep_tok_id
        ).nonzero(as_tuple=True)[0]
        if sep_positions.numel() == 0:
            raise RuntimeError(
                f"No sep token found for case_id={case.case_id}, "
                f"DxItem={DxItem}."
            )
        q_start = int(sep_positions[-1].item()) + 1
        if q_start != int(expected_q_start):
            raise RuntimeError(
                "The shared multimodal prefix changed between DxItems: "
                f"expected q_start={expected_q_start}, got {q_start} for "
                f"case_id={case.case_id}, DxItem={DxItem}."
            )

        suffix_ids = input_ids[:, q_start:].clone()
        raw_attention_mask = raw_inputs.get("attention_mask")
        if raw_attention_mask is None:
            suffix_attention_mask = torch.ones_like(suffix_ids)
        else:
            suffix_attention_mask = raw_attention_mask[
                :,
                q_start:,
            ].clone()
        raw_token_type_ids = raw_inputs.get("token_type_ids")
        suffix_token_type_ids = (
            raw_token_type_ids[:, q_start:].clone()
            if raw_token_type_ids is not None
            else None
        )
        return (
            suffix_ids,
            suffix_attention_mask,
            suffix_token_type_ids,
        )

    def _generate_case_with_vision_cache(
        self,
        case: Case,
        dx_items: List[str],
        case_context: Dict[str, Any],
        max_new_tokens: int,
        prompt_batch_size: int,
    ) -> Dict[str, Dict[str, str]]:
        results: Dict[str, Dict[str, str]] = {}
        vlm = getattr(self, "vlm_model")
        embed_tokens = self._get_embed_tokens()

        prefix_embeds = case_context["vision_prefix_embeds"]
        prefix_attention = case_context["vision_prefix_attn_mask"]
        prefix_token_types = case_context["vision_prefix_token_type_ids"]
        q_start = int(case_context["q_start_in_ids"])
        prefix_length = int(prefix_embeds.shape[1])
        hidden_size = int(prefix_embeds.shape[2])

        for chunk_start in range(0, len(dx_items), prompt_batch_size):
            chunk_items = dx_items[
                chunk_start:chunk_start + prompt_batch_size
            ]
            suffix_records = [
                self._build_inference_suffix_tokens(
                    case=case,
                    DxItem=dx_item,
                    expected_q_start=q_start,
                )
                for dx_item in chunk_items
            ]
            suffix_lengths = [
                int(record[0].shape[1])
                for record in suffix_records
            ]
            max_sequence_length = prefix_length + max(suffix_lengths)
            batch_size = len(chunk_items)
            combined_embeds = torch.zeros(
                (
                    batch_size,
                    max_sequence_length,
                    hidden_size,
                ),
                dtype=prefix_embeds.dtype,
                device=prefix_embeds.device,
            )
            attention_mask = torch.zeros(
                (batch_size, max_sequence_length),
                dtype=prefix_attention.dtype,
                device=prefix_embeds.device,
            )
            token_type_ids = (
                torch.zeros(
                    (batch_size, max_sequence_length),
                    dtype=prefix_token_types.dtype,
                    device=prefix_embeds.device,
                )
                if prefix_token_types is not None
                else None
            )

            for row_idx, (
                suffix_ids,
                suffix_attention,
                suffix_token_types,
            ) in enumerate(suffix_records):
                suffix_length = suffix_lengths[row_idx]
                # Decoder-only batched generation requires left padding.  Place
                # padding before the shared prefix so prefix and question remain
                # contiguous and every row ends at the same final prompt index.
                left_padding = (
                    max_sequence_length - prefix_length - suffix_length
                )
                prefix_end = left_padding + prefix_length
                combined_embeds[
                    row_idx,
                    left_padding:prefix_end,
                ] = prefix_embeds[0]
                attention_mask[
                    row_idx,
                    left_padding:prefix_end,
                ] = prefix_attention[0]

                with torch.no_grad():
                    suffix_embeds = embed_tokens(
                        suffix_ids.to(embed_tokens.weight.device)
                    ).to(
                        device=prefix_embeds.device,
                        dtype=prefix_embeds.dtype,
                    )
                combined_embeds[
                    row_idx,
                    prefix_end:,
                ] = suffix_embeds[0]
                attention_mask[
                    row_idx,
                    prefix_end:,
                ] = suffix_attention[0].to(
                    prefix_embeds.device
                )

                if token_type_ids is not None:
                    if suffix_token_types is None:
                        raise RuntimeError(
                            "token_type_ids disappeared between the cached "
                            "prefix and inference suffix."
                        )
                    token_type_ids[
                        row_idx,
                        left_padding:prefix_end,
                    ] = prefix_token_types[0]
                    token_type_ids[
                        row_idx,
                        prefix_end:,
                    ] = suffix_token_types[0].to(
                        prefix_embeds.device
                    )
                elif suffix_token_types is not None:
                    raise RuntimeError(
                        "token_type_ids appeared only in the inference suffix."
                    )

            generation_kwargs = {
                "inputs_embeds": combined_embeds,
                "attention_mask": attention_mask,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "top_p": None,
                "top_k": None,
                "pad_token_id": (
                    self.vlm_processor.tokenizer.pad_token_id
                ),
                "use_cache": True,
            }
            if token_type_ids is not None:
                generation_kwargs["token_type_ids"] = token_type_ids
            generated = vlm.generate(**generation_kwargs)

            # transformers initializes an empty input_ids sequence when
            # inputs_embeds is supplied to a decoder-only model, so `generated`
            # contains only newly generated token IDs.
            for row_idx, dx_item in enumerate(chunk_items):
                pred_txt = self.vlm_processor.tokenizer.decode(
                    generated[row_idx],
                    skip_special_tokens=True,
                ).strip()
                results[dx_item] = {
                    "pred_txt": pred_txt,
                    "gt_txt": getattr(
                        case,
                        "DxItem_targets",
                        {},
                    ).get(dx_item, ""),
                }

        return results

    def _generate_case_uncached(
        self,
        case: Case,
        max_new_tokens: int,
    ) -> Dict[str, Dict[str, str]]:
        results: Dict[str, Dict[str, str]] = {}
        vlm = getattr(self, "vlm_model")
        for DxItem in self._active_dxitems(case):
            messages = self._build_inference_messages(
                case=case,
                DxItem=DxItem,
            )
            prompt = self._apply_chat_template(
                messages,
                add_generation_prompt=True,
            )
            images = [
                roi.image
                for roi in self._get_dxitem_rois(case, DxItem)
                if roi.image is not None
            ]
            inputs = self._encode_inputs(
                text_prompt=prompt,
                images=images,
            )
            generated = vlm.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_p=None,
                top_k=None,
                pad_token_id=(
                    self.vlm_processor.tokenizer.pad_token_id
                ),
                use_cache=True,
            )
            prompt_len = inputs["input_ids"].shape[1]
            pred_txt = self.vlm_processor.tokenizer.decode(
                generated[0, prompt_len:],
                skip_special_tokens=True,
            ).strip()
            results[DxItem] = {
                "pred_txt": pred_txt,
                "gt_txt": getattr(
                    case,
                    "DxItem_targets",
                    {},
                ).get(DxItem, ""),
            }
        return results

    @torch.no_grad()
    def generate_outputs(self,
        batch_cases: List[Case],
        max_new_tokens: int = 256,
        case_contexts: Optional[
            Dict[int, Optional[Dict[str, Dict[str, Any]]]]
        ] = None,
    ) -> Dict[int, Dict[str, Any]]:
        """Generate active DxItems, batching only identical ROI prefixes.

        Returns:
            output_case_dict: {case.global_idx: {DxItem: {"pred_txt": str, "gt_txt": str}}}
        """
        batch_cases = self._normalize_batch(batch_cases)
        output_case_dict: Dict[int, Dict[str, Any]] = {}
        for case in batch_cases:
            active_dx_items = self._active_dxitems(case)
            case_context_map = (
                case_contexts.get(case.global_idx)
                if case_contexts is not None
                else self.prepare_case_loss_context(case)
            )
            case_results: Dict[str, Dict[str, str]] = {}
            cache_failed = False
            if self.use_shared_vision_cache and case_context_map is not None:
                for group_dx_items in self._group_dxitems_by_roi_signature(
                    case,
                    dx_items=active_dx_items,
                ):
                    group_context = case_context_map[group_dx_items[0]]
                    try:
                        group_results = self._generate_case_with_vision_cache(
                            case=case,
                            dx_items=group_dx_items,
                            case_context=group_context,
                            max_new_tokens=max_new_tokens,
                            prompt_batch_size=max(
                                1,
                                int(self.eval_prompt_batch_size),
                            ),
                        )
                    except (RuntimeError, ValueError, TypeError) as exc:
                        log_print(
                            "[VisionCache][WARN] Batched cached generation "
                            f"failed for case_id={case.case_id}, "
                            f"DxItems={group_dx_items}: {exc}. Retrying one "
                            "prompt at a time."
                        )
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        try:
                            group_results = (
                                self._generate_case_with_vision_cache(
                                    case=case,
                                    dx_items=group_dx_items,
                                    case_context=group_context,
                                    max_new_tokens=max_new_tokens,
                                    prompt_batch_size=1,
                                )
                            )
                        except (RuntimeError, ValueError, TypeError) as retry_exc:
                            log_print(
                                "[VisionCache][WARN] Sequential cached "
                                "generation also failed for "
                                f"case_id={case.case_id}, "
                                f"DxItems={group_dx_items}: {retry_exc}. "
                                "Falling back to full-VLM generation."
                            )
                            cache_failed = True
                            break
                    case_results.update(group_results)

            if (
                cache_failed
                or not self.use_shared_vision_cache
                or case_context_map is None
            ):
                case_results = self._generate_case_uncached(
                    case=case,
                    max_new_tokens=max_new_tokens,
                )
            output_case_dict[case.global_idx] = case_results

        return output_case_dict

    # ------------------------------------------------------------------
    # Save / Load LoRA weights
    # ------------------------------------------------------------------
    def save_lora_weights(self, path: str):
        """Save only the LoRA adapter weights."""
        vlm = getattr(self, "vlm_model")
        if hasattr(vlm, "save_pretrained"):
            vlm.save_pretrained(path)
            log_print(f"LoRA weights saved to {path}")
        else:
            raise RuntimeError("vlm_model does not support save_pretrained; LoRA may not be applied.")

    def load_lora_weights(self, path: str):
        """Load LoRA adapter weights from a peft save_pretrained directory."""
        try:
            from peft import PeftModel
        except ImportError:
            raise ImportError("peft is required. Install via: pip install peft")

        vlm = getattr(self, "vlm_model")
        if hasattr(vlm, "load_adapter"):
            load_result = vlm.load_adapter(
                path,
                adapter_name="default",
                is_trainable=True,
            )
            missing_keys = list(getattr(load_result, "missing_keys", []) or [])
            unexpected_keys = list(getattr(load_result, "unexpected_keys", []) or [])
            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    "LoRA checkpoint topology does not match the current model. "
                    f"missing_keys={missing_keys[:12]}, "
                    f"unexpected_keys={unexpected_keys[:12]}. "
                    "This commonly occurs when loading a former vision+text adapter "
                    "after switching to text-only LoRA."
                )
            if hasattr(vlm, "set_adapter"):
                vlm.set_adapter("default")
            log_print(f"LoRA weights loaded from {path}")
        else:
            raise RuntimeError("vlm_model does not support load_adapter.")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_batch(batch: Any) -> List[Any]:
        if batch is None:
            return []
        if isinstance(batch, list):
            return batch
        if isinstance(batch, tuple):
            return list(batch)
        return [batch]

    # ------------------------------------------------------------------
    # from_config
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls,
        cfg: DRGVLM_baseConfig,
        load_criterion: bool = True,
    ) -> "DownstreamRepGenVLM":
        log_print(f"Loading Model...", head=True)
        log_print(f"model_name: {cfg.model_name}")

        use_shared_vision_cache = bool(
            getattr(cfg, "use_shared_vision_cache", True)
        )
        use_vision_lora = bool(getattr(cfg, "use_vision_lora", False))
        if use_shared_vision_cache and use_vision_lora:
            raise ValueError(
                "use_shared_vision_cache=True is incompatible with "
                "use_vision_lora=True. The cached vision prefix is computed under "
                "torch.no_grad(), so vision adapters would not receive gradients."
            )

        builder = modelBuilder(weight_path=cfg.weight_path)
        vlm_model, vlm_processor, _, _, vlm_max_senLen, _ = builder.create_language_model(
            model_name=cfg.model_name,
            project_name="DRGVLM",
            freeze_weight=True,
            load_visual_processor=True,
            torch_dtype=torch.bfloat16,
            config_dict={
                "use_bidirectional_attention": False,  # SFT requires causal (left-to-right) attention
                "attn_implementation": getattr(cfg, "attn_implementation", None),
                "pp_vision_split_index": getattr(cfg, "pp_vision_split_index", None),
            },
            pp_num_gpus=getattr(cfg, "pp_num_gpus", None) if getattr(cfg, "training_mode", None) == "PP" else None,
        )

        model = cls(device=cfg.device)
        model.init_vlm_model(
            vlm_processor=vlm_processor,
            vlm_model=vlm_model,
            max_senLen=vlm_max_senLen,
        )
        model.apply_lora(
            lora_r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            lora_target_modules=cfg.lora_target_modules,
            use_vision_lora=use_vision_lora,
        )
        model.init_sep(
            sep_str=getattr(cfg, "sep_str", "<unused0>"),
            boc_str=getattr(cfg, "boc_str", "<unused1>"),
        )

        model.use_shared_vision_cache = use_shared_vision_cache
        model.eval_prompt_batch_size = max(
            1,
            int(getattr(cfg, "eval_prompt_batch_size", 6)),
        )
        log_print(f"use_shared_vision_cache = {model.use_shared_vision_cache}")
        log_print(
            f"eval_prompt_batch_size = {model.eval_prompt_batch_size}"
        )
        log_print(f"use_vision_lora = {model.use_vision_lora}")

        if load_criterion:
            log_print("Loading Criterion...")
            from ..criterions.DRGVLMLoss import DRGVLMLoss
            criterion = DRGVLMLoss.from_config(cfg=cfg)
            model.init_criterion(criterion)
            log_print("...Done\n")

        log_print("...Done\n")
        return model

