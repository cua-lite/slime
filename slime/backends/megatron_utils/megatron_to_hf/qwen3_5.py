"""
Convert Qwen3.5 model parameters from Megatron to HuggingFace format.

Called by slime's ``--megatron-to-hf-mode raw`` path.  When using
``--megatron-to-hf-mode bridge`` the Bridge ``mapping_registry`` handles
weight conversion instead, so this file is only needed for raw-mode export
and debugging.
"""

import re

import torch


# ---------------------------------------------------------------------------
# Native GDN weight conversion helpers (raw-mode export)
# ---------------------------------------------------------------------------
# These mirror the logic in megatron.bridge's split_gdn_linear_weights and
# _split_gdn_grouped_to_separate, inlined here so raw-mode export works
# without the Bridge fork installed.
# ---------------------------------------------------------------------------

def _split_in_proj_to_hf(args, in_proj: torch.Tensor, prefix: str):
    """Split native Megatron fused in_proj into 4 separate HF tensors.

    in_proj layout (after TP gather): head-grouped across tp_size shards,
    each shard = [q_heads_r, k_heads_r, v_heads_r, z_heads_r, b_heads_r, a_heads_r].
    """
    hidden_size = args.hidden_size
    qk_head_dim = args.linear_key_head_dim if hasattr(args, "linear_key_head_dim") else 128
    v_head_dim = args.linear_value_head_dim if hasattr(args, "linear_value_head_dim") else 128
    num_qk_heads = args.linear_num_key_heads if hasattr(args, "linear_num_key_heads") else 16
    num_v_heads = args.linear_num_value_heads if hasattr(args, "linear_num_value_heads") else 32
    tp_size = args.tensor_model_parallel_size if hasattr(args, "tensor_model_parallel_size") else 1

    qk_dim_per_tp = qk_head_dim * (num_qk_heads // tp_size)
    v_dim_per_tp = v_head_dim * (num_v_heads // tp_size)
    nv_per_tp = num_v_heads // tp_size

    # Reshape to (tp_size, per_tp_rows, hidden)
    in_proj_t = in_proj.reshape(tp_size, -1, hidden_size)
    q_t, k_t, v_t, z_t, b_t, a_t = torch.split(
        in_proj_t,
        [qk_dim_per_tp, qk_dim_per_tp, v_dim_per_tp, v_dim_per_tp, nv_per_tp, nv_per_tp],
        dim=1,
    )

    # Reshape to (num_heads, per_head_dim, hidden) and flatten to (total_dim, hidden)
    q = q_t.reshape(num_qk_heads, qk_head_dim, hidden_size).reshape(-1, hidden_size)
    k = k_t.reshape(num_qk_heads, qk_head_dim, hidden_size).reshape(-1, hidden_size)
    v = v_t.reshape(num_v_heads, v_head_dim, hidden_size).reshape(-1, hidden_size)
    z = z_t.reshape(num_v_heads, v_head_dim, hidden_size).reshape(-1, hidden_size)
    b = b_t.reshape(num_v_heads, hidden_size)
    a = a_t.reshape(num_v_heads, hidden_size)

    qkv = torch.cat([q, k, v], dim=0)
    return [
        (f"{prefix}.linear_attn.in_proj_qkv.weight", qkv),
        (f"{prefix}.linear_attn.in_proj_z.weight", z),
        (f"{prefix}.linear_attn.in_proj_b.weight", b),
        (f"{prefix}.linear_attn.in_proj_a.weight", a),
    ]


def _split_conv1d_to_hf(args, conv1d: torch.Tensor, prefix: str):
    """Convert head-grouped Megatron conv1d back to flat HF conv1d.

    conv1d shape: (conv_dim, 1, kernel_size) where conv_dim = qk_dim*2 + v_dim,
    stored in head-grouped TP layout.
    """
    qk_head_dim = args.linear_key_head_dim if hasattr(args, "linear_key_head_dim") else 128
    v_head_dim = args.linear_value_head_dim if hasattr(args, "linear_value_head_dim") else 128
    num_qk_heads = args.linear_num_key_heads if hasattr(args, "linear_num_key_heads") else 16
    num_v_heads = args.linear_num_value_heads if hasattr(args, "linear_num_value_heads") else 32
    tp_size = args.tensor_model_parallel_size if hasattr(args, "tensor_model_parallel_size") else 1

    qk_dim = qk_head_dim * num_qk_heads
    v_dim = v_head_dim * num_v_heads
    kernel_size = conv1d.shape[-1]

    # Head-grouped layout: (tp_size, [q+k+v]_per_tp, 1, kernel)
    qk_per_tp = qk_dim // tp_size
    v_per_tp = v_dim // tp_size
    conv1d_t = conv1d.reshape(tp_size, qk_per_tp + qk_per_tp + v_per_tp, 1, kernel_size)
    q_c, k_c, v_c = torch.split(conv1d_t, [qk_per_tp, qk_per_tp, v_per_tp], dim=1)

    # Flatten to (qk_dim, 1, k), (qk_dim, 1, k), (v_dim, 1, k)
    q_c = q_c.reshape(qk_dim, 1, kernel_size)
    k_c = k_c.reshape(qk_dim, 1, kernel_size)
    v_c = v_c.reshape(v_dim, 1, kernel_size)
    flat = torch.cat([q_c, k_c, v_c], dim=0)

    return [(f"{prefix}.linear_attn.conv1d.weight", flat)]


def _convert_mtp_layer(args, name, param, layer_idx):
    """Convert MTP layer parameters from Megatron to HuggingFace format."""
    if "enorm.weight" in name:
        return [("mtp.pre_fc_norm_embedding.weight", param)]
    if "hnorm.weight" in name:
        return [("mtp.pre_fc_norm_hidden.weight", param)]
    if "final_layernorm.weight" in name:
        return [("mtp.norm.weight", param)]
    if "eh_proj.weight" in name:
        return [("mtp.fc.weight", param)]

    if "transformer_layer" in name:
        proxy_name = name.replace(f"mtp.layers.{layer_idx}.transformer_layer", f"decoder.layers.{layer_idx}")
        mapped_params = convert_qwen3_5_to_hf(args, proxy_name, param)

        final_params = []
        for hf_name, tensor in mapped_params:
            target_prefix = f"mtp.layers.{layer_idx}"
            if f"model.language_model.layers.{layer_idx}" in hf_name:
                new_hf_name = hf_name.replace(f"model.language_model.layers.{layer_idx}", target_prefix)
                final_params.append((new_hf_name, tensor))
            else:
                final_params.append((hf_name, tensor))
        return final_params

    return None


def convert_qwen3_5_to_hf(args, name, param):
    """Convert Qwen3.5 model parameters from Megatron to HuggingFace format.

    Qwen3.5 uses model.language_model.layers prefix and has separate
    in_proj_qkv, in_proj_z, in_proj_b, in_proj_a for linear attention.

    VLM wrapper handling (mirrors ``qwen3_vl.convert_qwen3vl_to_hf``): when
    the Qwen3.5 VL model wraps ``GPTModel`` inside ``self.language_model`` and
    adds a frozen ``vision_model``, mcore's named_parameters emit
    ``module.module.language_model.<rest>`` and ``module.module.vision_model.<rest>``.
    Strip the ``language_model.`` infix so the rest of the converter matches
    against the plain ``module.module.<rest>`` paths it was originally written
    for, and route ``vision_model.*`` directly to ``model.visual.*``.
    """
    if name.startswith("module.module.language_model."):
        name = "module.module." + name[len("module.module.language_model.") :]

    while name.startswith("module.module.module."):
        name = name.replace("module.module.module.", "module.module.", 1)

    if name.startswith("module.module.vision_model."):
        hf_name = "model.visual." + name[len("module.module.vision_model.") :]
        return [(hf_name, param)]

    # Handle MTP layers
    if "mtp.layers" in name:
        parts = name.split(".")
        try:
            layer_idx_loc = parts.index("layers") + 1
            layer_idx = parts[layer_idx_loc]
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid MTP layer name format: {name}") from e

        result = _convert_mtp_layer(args, name, param, layer_idx)
        if result is not None:
            return result

    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.language_model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.language_model.norm.weight", param)]

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        prefix = f"model.language_model.layers.{layer_idx}"

        # experts (grouped gemm - fused format)
        if rest == "mlp.experts.linear_fc1":
            return [(f"{prefix}.mlp.experts.gate_up_proj", param)]
        elif rest == "mlp.experts.linear_fc2":
            return [(f"{prefix}.mlp.experts.down_proj", param)]

        # experts (ungrouped - individual expert format)
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"{prefix}.mlp.experts.{expert_idx}.gate_proj.weight", gate_weight),
                    (f"{prefix}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2":
                return [(f"{prefix}.mlp.experts.{expert_idx}.down_proj.weight", param)]
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        # shared expert
        shared_expert_pattern = r"mlp.shared_experts\.(.+)"
        match = re.match(shared_expert_pattern, rest)
        if match:
            rest = match.groups()[0]
            if rest == "linear_fc1.weight":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"{prefix}.mlp.shared_expert.gate_proj.weight", gate_weight),
                    (f"{prefix}.mlp.shared_expert.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2.weight":
                return [(f"{prefix}.mlp.shared_expert.down_proj.weight", param)]
            elif rest == "gate_weight":
                return [(f"{prefix}.mlp.shared_expert_gate.weight", param)]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        if rest == "self_attention.linear_proj.weight":
            return [(f"{prefix}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(
                param, split_size_or_sections=[2 * value_num_per_group, 1, 1], dim=1
            )
            q_param = (
                q_param.reshape(args.num_query_groups, 2, value_num_per_group, head_dim, args.hidden_size)
                .transpose(1, 2)
                .reshape(-1, args.hidden_size)
            )
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"{prefix}.self_attn.q_proj.weight", q_param),
                (f"{prefix}.self_attn.k_proj.weight", k_param),
                (f"{prefix}.self_attn.v_proj.weight", v_param),
            ]
        elif rest == "self_attention.linear_qkv.bias":
            param = param.view(args.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[value_num_per_group * head_dim, head_dim, head_dim],
                dim=1,
            )
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"{prefix}.self_attn.q_proj.bias", q_bias),
                (f"{prefix}.self_attn.k_proj.bias", k_bias),
                (f"{prefix}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"{prefix}.mlp.gate_proj.weight", gate_weight),
                (f"{prefix}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"{prefix}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"{prefix}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"{prefix}.post_attention_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"{prefix}.post_attention_layernorm.weight", param)]
        elif rest == "mlp.router.weight":
            return [(f"{prefix}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"{prefix}.mlp.gate.e_score_correction_bias", param)]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"{prefix}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"{prefix}.self_attn.k_norm.weight", param)]

        # Native GDN (Gated DeltaNet) parameters
        elif rest == "self_attention.in_proj.layer_norm_weight":
            return [(f"{prefix}.input_layernorm.weight", param)]
        elif rest == "self_attention.A_log":
            return [(f"{prefix}.linear_attn.A_log", param)]
        elif rest == "self_attention.dt_bias":
            return [(f"{prefix}.linear_attn.dt_bias", param)]
        elif rest == "self_attention.out_proj.weight":
            return [(f"{prefix}.linear_attn.out_proj.weight", param)]
        elif rest == "self_attention.out_norm.weight":
            # GDN out_norm: keep raw export consistent with bridge mode, which
            # currently uses plain AutoMapping (no ±1 offset). See the comment
            # in slime/slime_plugins/megatron_bridge/qwen3_5.py around the
            # `out_norm.weight` mapping for the open question.
            return [(f"{prefix}.linear_attn.norm.weight", param)]
        elif rest == "self_attention.in_proj.weight":
            # Fused head-grouped in_proj → 4 separate HF tensors.
            return _split_in_proj_to_hf(args, param, prefix)
        elif rest == "self_attention.conv1d.weight":
            # Head-grouped conv1d → flat HF conv1d.
            return _split_conv1d_to_hf(args, param, prefix)

    raise ValueError(f"Unknown parameter name: {name}")
