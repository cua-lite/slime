"""
Qwen3.5 hybrid linear+full-attention layer spec for Megatron-Core.

This module replaces the old HuggingFace-shim approach (``Qwen3_5GatedDeltaNet``
+ ``Attention``) with the native Megatron-Core ``GatedDeltaNet`` implementation.
Setting ``config.experimental_attention_variant = "gated_delta_net"`` lets
``get_gpt_decoder_block_spec`` build GDN specs automatically — no per-layer
manual override needed, and no custom gradient hooks.

Why this is gradient-stable (unlike the old HF shim):
  - ``use_qk_l2norm_in_kernel=True`` (FLA Triton kernel default): Q/K
    normalisation happens inside the kernel; the ``1/‖q‖`` backward term never
    enters PyTorch autograd.
  - Standard ``RMSNorm`` + fp32 gate instead of ``FusedRMSNormGated``: the
    ``1/rms`` backward amplification that caused the old shim to diverge is
    absent.  No gradient hooks or clamping needed.

Usage (via slime's --spec argument):
    --spec "slime_plugins.models.qwen3_5" "get_qwen3_5_spec"
"""

from __future__ import annotations

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

from .hf_attention import _load_hf_config


def _get_text_config(hf_config):
    """Extract text config from a VLM config if needed."""
    if hasattr(hf_config, "text_config"):
        return hf_config.text_config
    return hf_config


def get_qwen3_5_spec(args, config, vp_stage):
    # Always treat dense models (num_experts=None) as MoE-disabled.
    if not args.num_experts:
        config.moe_layer_freq = [0] * config.num_layers

    # Load HF config to fill in native GDN fields not exposed via the slime
    # CLI args in scripts/train/models/Qwen3.5-*.sh.
    hf_config = _load_hf_config(args.hf_checkpoint)
    text_config = _get_text_config(hf_config)

    # Configure native Megatron GDN.  get_gpt_decoder_block_spec reads these
    # fields and automatically inserts GatedDeltaNet specs for linear-attention
    # layers determined by linear_attention_freq.
    config.experimental_attention_variant = "gated_delta_net"
    config.linear_attention_freq = getattr(text_config, "full_attention_interval", 4)
    config.linear_conv_kernel_dim = getattr(text_config, "linear_conv_kernel_dim", 4)
    config.linear_key_head_dim = getattr(text_config, "linear_key_head_dim", 128)
    config.linear_value_head_dim = getattr(text_config, "linear_value_head_dim", 128)
    config.linear_num_key_heads = getattr(text_config, "linear_num_key_heads", 16)
    config.linear_num_value_heads = getattr(text_config, "linear_num_value_heads", 32)

    kwargs = {"use_transformer_engine": True}
    if vp_stage is not None:
        kwargs["vp_stage"] = vp_stage
    return get_gpt_decoder_block_spec(config, **kwargs)
