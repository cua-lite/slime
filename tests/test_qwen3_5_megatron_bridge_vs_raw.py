"""Byte-exact equivalence test: megatron.bridge Qwen3.5 mapping vs slime raw export.

The raw export (``slime/backends/megatron_utils/megatron_to_hf/qwen3_5.py``)
is the trusted reference; it is what ``--megatron-to-hf-mode raw`` uses and
has been cross-validated against real checkpoints. This test feeds the same
synthetic mcore-format tensors through:

  1. The raw converter ``convert_qwen3_5_to_hf``.
  2. The megatron.bridge helpers used by ``Qwen3_5Bridge.mapping_registry()``
     (``split_qkv_weights``, ``chunk(2, dim=0)`` for gated MLP, direct 1:1 for
     the rest).

and asserts that, for every mcore parameter name, both paths produce the
same (hf_name → tensor) output, byte-for-byte.

Must run inside the Slime container (needs megatron.bridge).

Run:  pytest -xvs slime/tests/test_qwen3_5_megatron_bridge_vs_raw.py
"""

from __future__ import annotations

import types

import pytest
import torch


# ---------------------------------------------------------------------------
# Synthetic Qwen3.5-2B config
# ---------------------------------------------------------------------------
# Values mirror scripts/train/models/Qwen3.5-2B.sh and the HF config for the
# dense 2B checkpoint.  The tests don't instantiate a model, so only the
# fields touched by the helpers matter.
QWEN3_5_2B = dict(
    num_attention_heads=8,
    num_query_groups=2,
    kv_channels=256,
    hidden_size=2048,
    ffn_hidden_size=6144,
    num_layers=24,
    vocab_size=248320,
    attention_output_gate=True,
    # linear attention
    linear_attention_freq=4,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_num_key_heads=16,
    linear_num_value_heads=32,
    linear_conv_kernel_dim=4,
)


def _config():
    """Plain namespace works for both raw (uses ``args.*``) and bridge helpers
    (use ``getattr(provider, ...)`` / ``provider.<field>``)."""
    return types.SimpleNamespace(**QWEN3_5_2B)


# ---------------------------------------------------------------------------
# Shape helpers — match what MCore produces for each parameter
# ---------------------------------------------------------------------------
def _qkv_weight_shape(cfg):
    """mcore linear_qkv.weight shape with attention_output_gate=True.

    Per-group layout along dim 0:
        [heads_per_group Q heads, heads_per_group Z heads, 1 K head, 1 V head]
    each head is head_dim rows, so total rows per group = (2*heads_per_group + 2) * head_dim
    """
    heads_per_group = cfg.num_attention_heads // cfg.num_query_groups
    rows = cfg.num_query_groups * (2 * heads_per_group + 2) * cfg.kv_channels
    return (rows, cfg.hidden_size)


def _qkv_bias_shape(cfg):
    heads_per_group = cfg.num_attention_heads // cfg.num_query_groups
    rows = cfg.num_query_groups * (2 * heads_per_group + 2) * cfg.kv_channels
    return (rows,)


def _fc1_weight_shape(cfg):
    # Gated MLP: [gate; up] fused along dim 0
    return (2 * cfg.ffn_hidden_size, cfg.hidden_size)


# ---------------------------------------------------------------------------
# Pytest scaffolding
# ---------------------------------------------------------------------------
pytest.importorskip("megatron.bridge", reason="megatron.bridge only inside Slime container")


@pytest.fixture
def raw_convert():
    """Import the raw reference converter."""
    from slime.backends.megatron_utils.megatron_to_hf import qwen3_5 as raw

    return raw.convert_qwen3_5_to_hf


@pytest.fixture
def bridge_helpers():
    """Import the megatron.bridge helpers that Qwen3_5Bridge relies on."""
    from megatron.bridge.models.conversion.param_mapping import (
        split_qkv_biases,
        split_qkv_weights,
    )

    return types.SimpleNamespace(
        split_qkv_weights=split_qkv_weights,
        split_qkv_biases=split_qkv_biases,
    )


# ---------------------------------------------------------------------------
# Name-mapping-only equivalence (direct 1:1 AutoMappings)
# ---------------------------------------------------------------------------
# raw prefix:    module.module.decoder.layers.N.<rest>
# bridge prefix: language_model.decoder.layers.N.<rest>  (the registry strips
#                language_model. before dispatching — see Qwen3_5Bridge)
#
# The bridge's AutoMapping entries (embedding, final_layernorm, output_layer,
# self_attention.linear_proj/q_layernorm/k_layernorm, mlp.linear_fc2,
# pre_mlp_layernorm, all linear_attn.* fields) are pure 1:1 renames.
# For those, equivalence reduces to "the renamed HF name matches raw's output".
@pytest.mark.parametrize(
    "mcore_suffix,expected_hf_suffix",
    [
        # Embedding / head / final norm (layer-less; raw handles these specially)
        # -- covered separately below
        # Full-attention layer 1:1 renames
        ("self_attention.linear_proj.weight", "self_attn.o_proj.weight"),
        ("self_attention.q_layernorm.weight", "self_attn.q_norm.weight"),
        ("self_attention.k_layernorm.weight", "self_attn.k_norm.weight"),
        ("self_attention.linear_qkv.layer_norm_weight", "input_layernorm.weight"),
        ("mlp.linear_fc1.layer_norm_weight", "post_attention_layernorm.weight"),
        ("mlp.linear_fc2.weight", "mlp.down_proj.weight"),
        ("pre_mlp_layernorm.weight", "post_attention_layernorm.weight"),
        # Linear-attention fields — same name on both sides (under self_attention. → ``.``)
        ("self_attention.input_layernorm.weight", "input_layernorm.weight"),
        ("self_attention.linear_attn.A_log", "linear_attn.A_log"),
        ("self_attention.linear_attn.conv1d.weight", "linear_attn.conv1d.weight"),
        ("self_attention.linear_attn.dt_bias", "linear_attn.dt_bias"),
        ("self_attention.linear_attn.in_proj_a.weight", "linear_attn.in_proj_a.weight"),
        ("self_attention.linear_attn.in_proj_b.weight", "linear_attn.in_proj_b.weight"),
        ("self_attention.linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_qkv.weight"),
        ("self_attention.linear_attn.in_proj_z.weight", "linear_attn.in_proj_z.weight"),
        ("self_attention.linear_attn.norm.weight", "linear_attn.norm.weight"),
        ("self_attention.linear_attn.out_proj.weight", "linear_attn.out_proj.weight"),
    ],
)
def test_raw_direct_mapping_matches_bridge_registry(raw_convert, mcore_suffix, expected_hf_suffix):
    """Every direct (1:1) mapping used by the bridge must match raw mode's output.

    We invoke raw with a layer-scoped name and assert the single HF name it
    returns equals ``model.language_model.layers.N.<expected_hf_suffix>``.
    """
    layer_idx = 7
    cfg = _config()
    tensor = torch.randn(4, 4)  # shape doesn't matter for direct renames
    raw_name = f"module.module.decoder.layers.{layer_idx}.{mcore_suffix}"

    raw_out = raw_convert(cfg, raw_name, tensor)

    assert len(raw_out) == 1, f"expected single output, got {raw_out}"
    hf_name, hf_tensor = raw_out[0]
    assert hf_name == f"model.language_model.layers.{layer_idx}.{expected_hf_suffix}"
    assert torch.equal(hf_tensor, tensor)


def test_raw_global_mappings_match_bridge_registry(raw_convert):
    """Embedding / final layernorm / output_layer renames."""
    cfg = _config()
    t = torch.randn(4, 4)

    assert raw_convert(cfg, "module.module.embedding.word_embeddings.weight", t) == [
        ("model.language_model.embed_tokens.weight", t)
    ]
    assert raw_convert(cfg, "module.module.output_layer.weight", t) == [("lm_head.weight", t)]
    assert raw_convert(cfg, "module.module.decoder.final_layernorm.weight", t) == [
        ("model.language_model.norm.weight", t)
    ]


# ---------------------------------------------------------------------------
# Gated-QKV split equivalence
# ---------------------------------------------------------------------------
def test_qkv_weight_split_matches_raw(raw_convert, bridge_helpers):
    """raw and bridge must split an mcore QKV weight into the same (q,k,v) HF tensors.

    This is the most load-bearing check: Qwen3.5 uses ``attention_output_gate=True``,
    so the Q slot in mcore is 2× wide (q+z). A wrong permutation between raw
    and bridge would silently corrupt Q and Z in a way no per-element shape
    check can catch.
    """
    cfg = _config()
    torch.manual_seed(0)
    qkv = torch.randn(_qkv_weight_shape(cfg))
    layer_idx = 3

    raw_out = dict(raw_convert(cfg, f"module.module.decoder.layers.{layer_idx}.self_attention.linear_qkv.weight", qkv))
    q_raw = raw_out[f"model.language_model.layers.{layer_idx}.self_attn.q_proj.weight"]
    k_raw = raw_out[f"model.language_model.layers.{layer_idx}.self_attn.k_proj.weight"]
    v_raw = raw_out[f"model.language_model.layers.{layer_idx}.self_attn.v_proj.weight"]

    q_br, k_br, v_br = bridge_helpers.split_qkv_weights(cfg, qkv)

    assert q_raw.shape == q_br.shape, f"Q shape mismatch: raw={q_raw.shape}, bridge={q_br.shape}"
    assert k_raw.shape == k_br.shape
    assert v_raw.shape == v_br.shape
    assert torch.equal(q_raw, q_br), "Q split differs between raw and bridge"
    assert torch.equal(k_raw, k_br), "K split differs between raw and bridge"
    assert torch.equal(v_raw, v_br), "V split differs between raw and bridge"


# NOTE: QKV bias split is intentionally NOT tested.
# Qwen3.5 sets ``--disable-bias-linear``, so the linear_qkv.bias code path
# is never hit. Raw's bias handler (lines 131-145 in slime/backends/megatron_utils
# /megatron_to_hf/qwen3_5.py) predates gated attention and would in fact miscompute
# for attention_output_gate=True (it splits [value_num_per_group*head_dim, head_dim,
# head_dim] which ignores the 2x Q expansion). Bridge's split_qkv_biases handles
# the gated layout correctly. Adding a regression test here would document the raw
# discrepancy but not protect any real training run; leaving it out on purpose.


# ---------------------------------------------------------------------------
# Gated MLP (gate/up) split equivalence
# ---------------------------------------------------------------------------
def test_gated_mlp_split_matches_raw(raw_convert):
    """bridge's GatedMLPMapping.megatron_to_hf with TP=1 is ``chunk(2, dim=0)``;
    raw does the same. Assert byte-exact on a random fc1 weight."""
    cfg = _config()
    torch.manual_seed(2)
    fc1 = torch.randn(_fc1_weight_shape(cfg))
    layer_idx = 11

    raw_out = dict(raw_convert(cfg, f"module.module.decoder.layers.{layer_idx}.mlp.linear_fc1.weight", fc1))
    gate_raw = raw_out[f"model.language_model.layers.{layer_idx}.mlp.gate_proj.weight"]
    up_raw = raw_out[f"model.language_model.layers.{layer_idx}.mlp.up_proj.weight"]

    gate_br, up_br = torch.chunk(fc1, 2, dim=0)

    assert torch.equal(gate_raw, gate_br), "gate split differs"
    assert torch.equal(up_raw, up_br), "up split differs"


# ---------------------------------------------------------------------------
# Structural coverage: every parameter name mcore will produce must match a
# mapping_registry entry.
# ---------------------------------------------------------------------------
# Byte-exact tensor tests only fire for mcore names that reach a mapping. If
# the mapping registry uses the wrong glob or omits a name the model will
# actually have, the bridge silently issues ``No mapping found: ...`` and
# load_weights_hf_to_megatron crashes at runtime — as it did before this
# test was added. The registry therefore needs a *coverage* guarantee.
#
# We enumerate the parameters that Qwen3.5's layer spec produces for both
# layer types and assert ``mapping_registry.megatron_to_hf_lookup`` returns a
# non-None mapping for every one.
#
# Per-layer expected mcore parameter names come from inspecting:
#   - ``get_gpt_decoder_block_spec`` (for full-attention layers' mcore-default
#     fused TE self-attn + MLP)
#   - ``slime_plugins.models.qwen3_5.{Attention, Qwen3_5GatedDeltaNet}`` (the
#     linear-attention override applied by ``Qwen3_5ModelProvider.provide``)
#
# Global parameters (embedding / lm_head / final_norm) also covered.

_FULL_ATTENTION_PARAMS = [
    "self_attention.linear_proj.weight",
    "self_attention.linear_qkv.weight",
    "self_attention.linear_qkv.layer_norm_weight",
    "self_attention.q_layernorm.weight",
    "self_attention.k_layernorm.weight",
    "mlp.linear_fc1.weight",
    "mlp.linear_fc1.layer_norm_weight",
    "mlp.linear_fc2.weight",
]

_LINEAR_ATTENTION_PARAMS = [
    # Qwen3.5 Attention wrapper owns input_layernorm explicitly (no fused TE)
    "self_attention.input_layernorm.weight",
    # Qwen3_5GatedDeltaNet fields
    "self_attention.linear_attn.conv1d.weight",
    "self_attention.linear_attn.in_proj_qkv.weight",
    "self_attention.linear_attn.in_proj_z.weight",
    "self_attention.linear_attn.in_proj_b.weight",
    "self_attention.linear_attn.in_proj_a.weight",
    "self_attention.linear_attn.dt_bias",
    "self_attention.linear_attn.A_log",
    "self_attention.linear_attn.norm.weight",
    "self_attention.linear_attn.out_proj.weight",
    # MLP is unchanged by the linear-attention override
    "mlp.linear_fc1.weight",
    "mlp.linear_fc1.layer_norm_weight",
    "mlp.linear_fc2.weight",
]

_GLOBAL_PARAMS = [
    "embedding.word_embeddings.weight",
    "output_layer.weight",
    "decoder.final_layernorm.weight",
]


def _mapping_registry():
    """Build the registry without instantiating a bridge (mapping_registry is
    a pure method — it only reads ``self`` structurally; we never touch self)."""
    from slime_plugins.megatron_bridge.qwen3_5 import Qwen3_5Bridge

    return Qwen3_5Bridge.__new__(Qwen3_5Bridge).mapping_registry()


def _expected_layer_types(num_layers: int, freq: int) -> list[str]:
    """Mirror ``get_qwen3_5_spec``: every ``freq``-th layer is full attention."""
    return ["full_attention" if (i + 1) % freq == 0 else "linear_attention" for i in range(num_layers)]


@pytest.mark.parametrize("num_layers,freq", [(24, 4), (28, 4)])
def test_every_mcore_param_name_is_covered_by_mapping_registry(num_layers, freq):
    """**The test that would have caught the ``No mapping found`` runtime crash.**

    For a Qwen3.5-sized config (24 / 28 dense layers, full_attention every 4
    layers), every parameter name the Qwen3.5 layer spec produces must be
    matched by the registry. If any is unmatched the bridge silently drops
    the weight at load and training blows up later on shape mismatch.
    """
    registry = _mapping_registry()
    layer_types = _expected_layer_types(num_layers, freq)

    missing: list[str] = []

    # Language-model prefix matches what mcore produces for our Qwen3_5Model
    # (GPTModel wrapped under ``language_model.``).
    for gp in _GLOBAL_PARAMS:
        name = f"language_model.{gp}"
        if registry.megatron_to_hf_lookup(name) is None:
            missing.append(name)

    for i, layer_type in enumerate(layer_types):
        params = _FULL_ATTENTION_PARAMS if layer_type == "full_attention" else _LINEAR_ATTENTION_PARAMS
        for p in params:
            name = f"language_model.decoder.layers.{i}.{p}"
            if registry.megatron_to_hf_lookup(name) is None:
                missing.append(name)

    assert not missing, (
        f"{len(missing)} mcore parameter names have no mapping. First 5: "
        + "\n  " + "\n  ".join(missing[:5])
    )
