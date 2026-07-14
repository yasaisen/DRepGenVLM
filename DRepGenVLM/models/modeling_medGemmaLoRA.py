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
    """medgemma-1.5-4b-it + LoRA SFT model for diagnostic report generation.

    Design:
    - The VLM backbone (medgemma) is frozen during init, then LoRA adapters are
      injected via peft.  Only LoRA parameters are trainable.
    - Each forward pass handles one (Case, DxItem) pair:
        user turn  : all ROI images + DxItem name as the question
        assistant  : DxResultTxt as the answer target
    - calculate_loss iterates over all DxItems for every case in the batch.
    - generate_outputs uses model.generate() with greedy / beam search.
    """

    def __init__(self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.use_text_gradient_checkpointing = False
        self.use_shared_vision_cache = True  # controlled by from_config via cfg.use_shared_vision_cache

    # ------------------------------------------------------------------
    # train() override – keep VLM backbone in eval when not checkpointing
    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        vlm_model = getattr(self, "vlm_model", None)
        if isinstance(vlm_model, nn.Module):
            # LoRA wrappers keep trainable params; frozen base stays eval
            if mode:
                vlm_model.train()  # peft model.train() enables LoRA dropout etc.
            else:
                vlm_model.eval()
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
    ):
        """Wrap the VLM with PEFT LoRA; only LoRA params become trainable."""
        try:
            from peft import get_peft_model, LoraConfig, TaskType
        except ImportError:
            raise ImportError("peft is required for LoRA. Install via: pip install peft")

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_r),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            target_modules=list(lora_target_modules),
            bias="none",
            inference_mode=False,
        )

        vlm_model = getattr(self, "vlm_model")
        peft_model = get_peft_model(vlm_model, lora_config)
        peft_model.print_trainable_parameters()

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
            f"dropout={lora_dropout}, targets={lora_target_modules}"
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
    def _build_user_content(self,
        case: Case,
        DxItem: str,
    ) -> List[Dict]:
        """Build the user-turn content list for one (case, DxItem) pair.

        Layout:  [img, text_sep, img, text_sep, ..., DxItem question]
        """
        content = []
        for roi in case.rois:
            if roi.image is not None:
                content.append({"type": "image"})

            text = ""
            if roi.image is None:
                text += self.boc_str
            if roi.mpp is not None:
                text += f"MPP: {roi.mpp:.6f}, "
            if roi.cxcywh is not None:
                text += f"cxcywh: {roi.cxcywh}, "
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

        images = [roi.image for roi in case.rois if roi.image is not None]

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
        case,
    ) -> Tuple[torch.Tensor, int, int]:
        """One no_grad VLM forward to obtain the shared vision prefix embeddings.

        Strategy:
          1. Build the inference prompt for an arbitrary DxItem (all DxItems share
             the same ROI images + sep tokens before the question).
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
        """
        ref_DxItem = next(iter(case.DxItem_targets))
        messages = self._build_inference_messages(case=case, DxItem=ref_DxItem)
        prompt = self._apply_chat_template(messages, add_generation_prompt=True)
        images = [roi.image for roi in case.rois if roi.image is not None]
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
        return vision_prefix_embeds, q_start_in_ids, vision_prefix_len_merged

    def build_train_inputs_with_vision_cache(self,
        case,
        DxItem: str,
        vision_prefix_embeds: torch.Tensor,
        q_start_in_ids: int,
        vision_prefix_len_merged: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build (inputs_embeds, attention_mask, labels) reusing the cached vision prefix.

        The cached vision_prefix_embeds (no grad) is concatenated with a text-only
        suffix embedding (question + template + answer tokens).  The suffix is
        obtained by running embed_tokens on the token IDs from q_start_in_ids
        onwards, avoiding a second costly vision-tower forward.

        Labels are -100 for the vision prefix and the question/template tokens;
        real token IDs are placed at the answer positions.

        Returns:
            combined_embeds : (1, S, H)
            attn_mask       : (1, S)  – all ones
            labels          : (1, S)  – -100 on non-answer positions
        """
        images = [roi.image for roi in case.rois if roi.image is not None]

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
        # q_start_in_ids is shared across all DxItems (vision content is identical).
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

        # Attention mask: all tokens attend (no padding)
        attn_mask = torch.ones(1, S_combined, dtype=torch.long, device=combined_embeds.device)

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

        return combined_embeds, attn_mask, labels

    # ------------------------------------------------------------------
    # calculate_loss  (training)
    # ------------------------------------------------------------------
    def calculate_loss(self,
        batch_cases: List[Case],
    ) -> Tuple[Dict[str, torch.Tensor], Dict[int, Dict[str, Any]]]:
        """Forward + loss for a batch; iterates cases × DxItems.

        When use_shared_vision_cache=True (default):
          - One no_grad VLM forward per case captures the merged vision prefix.
          - Each DxItem then only runs the text decoder (with LoRA) forward.
          - Vision tower is run once regardless of DxItem count – no redundant
            image encoding, and no vision-tower activations stored for backward.

        When use_shared_vision_cache=False:
          - Original full forward per (case, DxItem).
          - Use this when the vision tower itself has LoRA adapters.
        """
        batch_cases = self._normalize_batch(batch_cases)
        all_losses: List[torch.Tensor] = []
        output_case_dict: Dict[int, Dict[str, Any]] = {}

        for case in batch_cases:
            log_print(
                f"case_id={case.case_id}, rois={len(case.rois)}, "
                f"DxItems={list(case.DxItem_targets.keys())}"
            )
            output_case_dict[case.global_idx] = {}

            # ---- Shared vision cache path ----
            if self.use_shared_vision_cache:
                vision_prefix_embeds, q_start_in_ids, vision_prefix_len_merged = (
                    self._get_vision_prefix_embeds(case)
                )
                for DxItem in case.DxItem_targets:
                    combined_embeds, attn_mask, labels = (
                        self.build_train_inputs_with_vision_cache(
                            case=case,
                            DxItem=DxItem,
                            vision_prefix_embeds=vision_prefix_embeds,
                            q_start_in_ids=q_start_in_ids,
                            vision_prefix_len_merged=vision_prefix_len_merged,
                        )
                    )
                    # combined_embeds has no grad_fn (vision prefix and suffix were
                    # both produced under no_grad).  Marking it as a requires_grad
                    # leaf ensures PyTorch builds a full autograd graph through the
                    # LoRA params during forward, so backward() can reach them.
                    # The dummy gradient accumulated in combined_embeds.grad is
                    # discarded when the local variable goes out of scope.
                    combined_embeds = combined_embeds.requires_grad_(True)
                    vlm = getattr(self, "vlm_model")
                    outputs = vlm(
                        inputs_embeds=combined_embeds,
                        attention_mask=attn_mask,
                        use_cache=False,
                    )
                    logits = outputs.logits
                    context = f"case_id={case.case_id}, DxItem={DxItem}"
                    loss_dict = self.criterion(logits=logits, labels=labels, context=context)
                    all_losses.append(loss_dict["total_loss"])
                    output_case_dict[case.global_idx][DxItem] = {
                        "pred_txt": None,
                        "gt_txt": case.DxItem_targets[DxItem],
                    }

            # ---- Original full forward path (vision tower has LoRA) ----
            else:
                for DxItem in case.DxItem_targets:
                    inputs, labels = self.build_train_inputs(case=case, DxItem=DxItem)
                    vlm = getattr(self, "vlm_model")
                    outputs = vlm(**inputs, output_hidden_states=False, use_cache=False)
                    logits = outputs.logits
                    context = f"case_id={case.case_id}, DxItem={DxItem}"
                    loss_dict = self.criterion(logits=logits, labels=labels, context=context)
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
    @torch.no_grad()
    def generate_outputs(self,
        batch_cases: List[Case],
        max_new_tokens: int = 256,
    ) -> Dict[int, Dict[str, Any]]:
        """Run model.generate() for each (case, DxItem); return pred texts.

        Returns:
            output_case_dict: {case.global_idx: {DxItem: {"pred_txt": str, "gt_txt": str}}}
        """
        batch_cases = self._normalize_batch(batch_cases)
        output_case_dict: Dict[int, Dict[str, Any]] = {}
        vlm = getattr(self, "vlm_model")

        for case in batch_cases:
            output_case_dict[case.global_idx] = {}
            for DxItem in case.DxItem_targets:
                messages = self._build_inference_messages(case=case, DxItem=DxItem)
                prompt = self._apply_chat_template(messages, add_generation_prompt=True)
                images = [roi.image for roi in case.rois if roi.image is not None]
                inputs = self._encode_inputs(text_prompt=prompt, images=images)

                generated = vlm.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
                # Decode only the newly generated tokens
                prompt_len = inputs["input_ids"].shape[1]
                new_token_ids = generated[0, prompt_len:]
                pred_txt = self.vlm_processor.tokenizer.decode(
                    new_token_ids, skip_special_tokens=True
                ).strip()

                output_case_dict[case.global_idx][DxItem] = {
                    "pred_txt": pred_txt,
                    "gt_txt": case.DxItem_targets[DxItem],
                }

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
            vlm.load_adapter(path, adapter_name="default")
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

        builder = modelBuilder(weight_path=cfg.weight_path)
        vlm_model, vlm_processor, _, _, vlm_max_senLen, _ = builder.create_language_model(
            model_name=cfg.model_name,
            project_name="DRGVLM",
            freeze_weight=True,
            load_visual_processor=True,
            torch_dtype=torch.bfloat16,
            config_dict={
                "use_bidirectional_attention": False,  # SFT requires causal (left-to-right) attention
                "attn_implementation": getattr(cfg, "attn_implementation", "sdpa"),
            },
            pp_num_gpus=(
                getattr(cfg, "pp_num_gpus", None)
                if getattr(cfg, "training_mode", None) == "PP"
                else None
            ),
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
        )
        model.init_sep(
            sep_str=getattr(cfg, "sep_str", "<unused0>"),
            boc_str=getattr(cfg, "boc_str", "<unused1>"),
        )

        model.use_shared_vision_cache = bool(getattr(cfg, "use_shared_vision_cache", True))
        log_print(f"use_shared_vision_cache = {model.use_shared_vision_cache}")

        if load_criterion:
            log_print("Loading Criterion...")
            from ..criterions.DRGVLMLoss import DRGVLMLoss
            criterion = DRGVLMLoss.from_config(cfg=cfg)
            model.init_criterion(criterion)
            log_print("...Done\n")

        log_print("...Done\n")
        return model















