"""
 SPDX-License-Identifier: MIT
 Copyright (c) 2026, yasaisen (clover)

 This file is part of a project licensed under the MIT License.
 See the LICENSE file in the project root for more information.

 Causal-compatible large-head fallback for FlashAttention interfaces.
"""


import functools
import inspect
import math
from typing import Any, Dict, Optional, Tuple

import torch


def _block_attention_mask(
    q_start: int,
    q_end: int,
    sq: int,
    sk: int,
    device: torch.device,
    causal: bool,
    window_left: int,
    window_right: int,
) -> Optional[torch.Tensor]:
    """Return a bottom-right-aligned FlashAttention-compatible mask."""
    has_window = window_left >= 0 or window_right >= 0
    if not causal and not has_window:
        return None

    # FlashAttention aligns the causal mask to the bottom-right when Sq != Sk.
    query_positions = (
        torch.arange(q_start, q_end, device=device, dtype=torch.long)
        + (sk - sq)
    ).view(1, 1, -1, 1)
    key_positions = torch.arange(
        sk,
        device=device,
        dtype=torch.long,
    ).view(1, 1, 1, -1)
    mask = torch.ones(
        (1, 1, q_end - q_start, sk),
        device=device,
        dtype=torch.bool,
    )
    if causal:
        mask &= key_positions <= query_positions
    if window_left >= 0:
        mask &= key_positions >= query_positions - window_left
    if window_right >= 0:
        mask &= key_positions <= query_positions + window_right
    return mask


def _masked_softmax(
    scores: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stable float32 softmax with well-defined all-masked rows."""
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
        row_has_value = mask.any(dim=-1, keepdim=True)
    else:
        row_has_value = torch.ones_like(
            scores[..., :1],
            dtype=torch.bool,
        )

    max_scores = scores.max(dim=-1, keepdim=True).values
    safe_max = torch.where(
        row_has_value,
        max_scores,
        torch.zeros_like(max_scores),
    )
    exp_scores = torch.exp(scores - safe_max)
    if mask is not None:
        exp_scores = torch.where(mask, exp_scores, torch.zeros_like(exp_scores))
    denominator = exp_scores.sum(dim=-1, keepdim=True)
    probabilities = exp_scores / denominator.clamp_min(
        torch.finfo(exp_scores.dtype).tiny
    )
    logsumexp = torch.where(
        row_has_value,
        safe_max + torch.log(denominator.clamp_min(
            torch.finfo(exp_scores.dtype).tiny
        )),
        torch.full_like(safe_max, float("-inf")),
    )
    return probabilities, logsumexp


class CausalBlockAttention(torch.autograd.Function):
    """Block-query exact attention fallback for head dimensions above 256."""

    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool,
        softmax_scale: Optional[float],
        window_left: int,
        window_right: int,
        block_size: int,
    ) -> torch.Tensor:
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                "CausalBlockAttention expects q/k/v in [B, H, S, D] layout."
            )
        if k.shape != v.shape:
            raise ValueError(
                f"k and v must have identical shapes, got {k.shape} and {v.shape}."
            )
        if q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
            raise ValueError(
                "q/k/v must have matching batch, head, and head-dim dimensions."
            )

        _, _, sq, head_dim = q.shape
        sk = k.shape[-2]
        scale = (
            float(softmax_scale)
            if softmax_scale is not None
            else 1.0 / math.sqrt(head_dim)
        )
        block_size = max(1, int(block_size))
        output = torch.zeros_like(q)
        logsumexp = torch.empty(
            (*q.shape[:-1], 1),
            device=q.device,
            dtype=torch.float32,
        )
        k_float_t = k.float().transpose(-2, -1)
        v_float = v.float()

        for q_start in range(0, sq, block_size):
            q_end = min(q_start + block_size, sq)
            q_block = q[:, :, q_start:q_end, :].float()
            scores = torch.matmul(q_block, k_float_t) * scale
            mask = _block_attention_mask(
                q_start=q_start,
                q_end=q_end,
                sq=sq,
                sk=sk,
                device=q.device,
                causal=bool(causal),
                window_left=int(window_left),
                window_right=int(window_right),
            )
            probabilities, block_lse = _masked_softmax(scores, mask)
            output[:, :, q_start:q_end, :] = torch.matmul(
                probabilities,
                v_float,
            ).to(q.dtype)
            logsumexp[:, :, q_start:q_end, :] = block_lse

        ctx.save_for_backward(q, k, v, output, logsumexp)
        ctx.causal = bool(causal)
        ctx.scale = scale
        ctx.window_left = int(window_left)
        ctx.window_right = int(window_right)
        ctx.block_size = block_size
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, output, logsumexp = ctx.saved_tensors
        _, _, sq, _ = q.shape
        sk = k.shape[-2]

        grad_q = torch.zeros_like(q, dtype=torch.float32)
        grad_k = torch.zeros_like(k, dtype=torch.float32)
        grad_v = torch.zeros_like(v, dtype=torch.float32)
        k_float = k.float()
        k_float_t = k_float.transpose(-2, -1)
        v_float = v.float()
        v_float_t = v_float.transpose(-2, -1)
        grad_output_float = grad_output.float()
        output_float = output.float()
        output_dot = torch.sum(
            grad_output_float * output_float,
            dim=-1,
            keepdim=True,
        )

        for q_start in range(0, sq, ctx.block_size):
            q_end = min(q_start + ctx.block_size, sq)
            q_block = q[:, :, q_start:q_end, :].float()
            grad_output_block = grad_output_float[:, :, q_start:q_end, :]
            scores = torch.matmul(q_block, k_float_t) * ctx.scale
            mask = _block_attention_mask(
                q_start=q_start,
                q_end=q_end,
                sq=sq,
                sk=sk,
                device=q.device,
                causal=ctx.causal,
                window_left=ctx.window_left,
                window_right=ctx.window_right,
            )
            if mask is not None:
                scores = scores.masked_fill(~mask, float("-inf"))

            lse_block = logsumexp[:, :, q_start:q_end, :]
            safe_lse = torch.where(
                torch.isfinite(lse_block),
                lse_block,
                torch.zeros_like(lse_block),
            )
            probabilities = torch.exp(scores - safe_lse)
            if mask is not None:
                probabilities = torch.where(
                    mask,
                    probabilities,
                    torch.zeros_like(probabilities),
                )
            probabilities = torch.where(
                torch.isfinite(lse_block),
                probabilities,
                torch.zeros_like(probabilities),
            )

            grad_v += torch.matmul(
                probabilities.transpose(-2, -1),
                grad_output_block,
            )
            grad_probabilities = torch.matmul(
                grad_output_block,
                v_float_t,
            )
            grad_scores = probabilities * (
                grad_probabilities - output_dot[:, :, q_start:q_end, :]
            )
            grad_scores *= ctx.scale
            grad_q[:, :, q_start:q_end, :] = torch.matmul(
                grad_scores,
                k_float,
            )
            grad_k += torch.matmul(
                grad_scores.transpose(-2, -1),
                q_block,
            )

        return (
            grad_q.to(q.dtype),
            grad_k.to(k.dtype),
            grad_v.to(v.dtype),
            None,
            None,
            None,
            None,
            None,
        )


def _bind_arguments(
    original,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    required_positional_names: Tuple[str, ...],
) -> Dict[str, Any]:
    try:
        signature = inspect.signature(original)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except (TypeError, ValueError):
        if len(args) > len(required_positional_names):
            raise RuntimeError(
                "Cannot safely bind optional positional FlashAttention arguments "
                "for this interface version. Pass causal/softmax/window options "
                "by keyword or use the native attention implementation."
            )
        values = dict(kwargs)
        for name, value in zip(required_positional_names, args):
            values[name] = value
        return values


def _window_from_arguments(arguments: Dict[str, Any]) -> Tuple[int, int]:
    window = arguments.get("window_size", (-1, -1))
    if window is None:
        window = (-1, -1)
    if isinstance(window, int):
        window = (window, window)
    if len(window) != 2:
        raise ValueError(f"window_size must contain two integers, got {window!r}.")
    left, right = int(window[0]), int(window[1])
    if arguments.get("window_size_left") is not None:
        left = int(arguments["window_size_left"])
    if arguments.get("window_size_right") is not None:
        right = int(arguments["window_size_right"])
    return left, right


def _validate_supported_options(arguments: Dict[str, Any]):
    dropout_p = arguments.get("dropout_p", 0.0)
    if dropout_p is not None and float(dropout_p) != 0.0:
        raise NotImplementedError(
            "The large-head FlashAttention fallback does not silently ignore "
            "attention dropout; use dropout_p=0 or native SDPA."
        )

    softcap = arguments.get("softcap", 0.0)
    if softcap is not None and float(softcap) != 0.0:
        raise NotImplementedError(
            "The large-head FlashAttention fallback does not support softcap."
        )

    unsupported_non_null = (
        "alibi_slopes",
        "qv",
        "q_descale",
        "k_descale",
        "v_descale",
        "seqused_q",
        "seqused_k",
        "leftpad_k",
        "block_table",
    )
    for name in unsupported_non_null:
        if arguments.get(name) is not None:
            raise NotImplementedError(
                f"The large-head FlashAttention fallback does not support {name}."
            )

    if arguments.get("attention_chunk", 0) not in (None, 0):
        raise NotImplementedError(
            "The large-head FlashAttention fallback does not support attention_chunk."
        )
    if bool(arguments.get("pack_gqa", False)):
        raise NotImplementedError(
            "The large-head FlashAttention fallback does not support pack_gqa."
        )
    if bool(arguments.get("return_attn_probs", False)):
        raise NotImplementedError(
            "The large-head FlashAttention fallback returns only attention output."
        )


def _repeat_kv_heads(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_heads = q.shape[1]
    kv_heads = k.shape[1]
    if k.shape[1] != v.shape[1]:
        raise ValueError("k and v must use the same number of heads.")
    if q_heads == kv_heads:
        return q, k, v
    if q_heads % kv_heads != 0:
        raise ValueError(
            f"Query heads ({q_heads}) must be divisible by KV heads ({kv_heads})."
        )
    repeat_factor = q_heads // kv_heads
    return (
        q,
        k.repeat_interleave(repeat_factor, dim=1).contiguous(),
        v.repeat_interleave(repeat_factor, dim=1).contiguous(),
    )


def _run_padded_attention(arguments: Dict[str, Any]) -> torch.Tensor:
    _validate_supported_options(arguments)
    q = arguments["q"]
    k = arguments["k"]
    v = arguments["v"]
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("flash_attn_func expects q/k/v in [B, S, H, D].")

    q_heads_first = q.transpose(1, 2)
    k_heads_first = k.transpose(1, 2)
    v_heads_first = v.transpose(1, 2)
    q_heads_first, k_heads_first, v_heads_first = _repeat_kv_heads(
        q_heads_first,
        k_heads_first,
        v_heads_first,
    )
    window_left, window_right = _window_from_arguments(arguments)
    output = CausalBlockAttention.apply(
        q_heads_first,
        k_heads_first,
        v_heads_first,
        bool(arguments.get("causal", False)),
        arguments.get("softmax_scale"),
        window_left,
        window_right,
        1024,
    )
    return output.transpose(1, 2).contiguous()


def _run_varlen_attention(arguments: Dict[str, Any]) -> torch.Tensor:
    _validate_supported_options(arguments)
    q = arguments["q"]
    k = arguments["k"]
    v = arguments["v"]
    cu_seqlens_q = arguments["cu_seqlens_q"]
    cu_seqlens_k = arguments["cu_seqlens_k"]
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError(
            "flash_attn_varlen_func expects q/k/v in [total_tokens, H, D]."
        )
    if len(cu_seqlens_q) != len(cu_seqlens_k):
        raise ValueError("cu_seqlens_q and cu_seqlens_k must have equal lengths.")

    window_left, window_right = _window_from_arguments(arguments)
    outputs = []
    for sequence_idx in range(len(cu_seqlens_q) - 1):
        q_start = int(cu_seqlens_q[sequence_idx].item())
        q_end = int(cu_seqlens_q[sequence_idx + 1].item())
        k_start = int(cu_seqlens_k[sequence_idx].item())
        k_end = int(cu_seqlens_k[sequence_idx + 1].item())

        q_sequence = q[q_start:q_end].unsqueeze(0).transpose(1, 2)
        k_sequence = k[k_start:k_end].unsqueeze(0).transpose(1, 2)
        v_sequence = v[k_start:k_end].unsqueeze(0).transpose(1, 2)
        q_sequence, k_sequence, v_sequence = _repeat_kv_heads(
            q_sequence,
            k_sequence,
            v_sequence,
        )
        output = CausalBlockAttention.apply(
            q_sequence,
            k_sequence,
            v_sequence,
            bool(arguments.get("causal", False)),
            arguments.get("softmax_scale"),
            window_left,
            window_right,
            1024,
        )
        outputs.append(output.transpose(1, 2).squeeze(0))

    return torch.cat(outputs, dim=0) if outputs else torch.empty_like(q)


def _make_wrapper(original, function_name: str):
    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        q = args[0] if args else kwargs.get("q")
        if q is None or q.shape[-1] <= 256:
            return original(*args, **kwargs)

        if function_name == "flash_attn_func":
            arguments = _bind_arguments(
                original,
                args,
                kwargs,
                ("q", "k", "v"),
            )
            return _run_padded_attention(arguments)
        if function_name == "flash_attn_varlen_func":
            arguments = _bind_arguments(
                original,
                args,
                kwargs,
                (
                    "q",
                    "k",
                    "v",
                    "cu_seqlens_q",
                    "cu_seqlens_k",
                    "max_seqlen_q",
                    "max_seqlen_k",
                ),
            )
            return _run_varlen_attention(arguments)
        raise RuntimeError(f"Unsupported patched function: {function_name}")

    return wrapper


def install_large_head_flash_attention_patch(flash_attn_interface) -> bool:
    """Install an idempotent causal-compatible fallback for head_dim > 256."""
    if getattr(
        flash_attn_interface,
        "_drgvlm_large_head_causal_patch_installed",
        False,
    ):
        return False

    required = ("flash_attn_func", "flash_attn_varlen_func")
    missing = [
        name for name in required
        if not hasattr(flash_attn_interface, name)
    ]
    if missing:
        raise AttributeError(
            f"FlashAttention interface is missing required functions: {missing}"
        )

    for function_name in required:
        original = getattr(flash_attn_interface, function_name)
        setattr(
            flash_attn_interface,
            function_name,
            _make_wrapper(original, function_name),
        )

    flash_attn_interface._drgvlm_large_head_causal_patch_installed = True
    print(
        "[FA3 Patch] Installed causal-compatible large-head attention fallback "
        "for flash_attn_func and flash_attn_varlen_func (head_dim > 256).",
        flush=True,
    )
    return True

