"""
Qwen3.5 VLM bridge for megatron.bridge.

Registers `Qwen3_5ForConditionalGeneration` so that `AutoBridge.from_hf_pretrained`
recognises Qwen3.5 checkpoints and can provide a Megatron-compatible VL model +
weight mappings.

Architecture:
  HF vision encoder (Qwen3VLVisionModel, replicated on first PP stage)
  + Megatron GPTModel (dense language model with hybrid linear+full attention)

============================================================================
KNOWN OPEN ISSUE — diverges from upstream Megatron-Bridge-slime/qwen35
============================================================================

Upstream references (the things this section is being compared against):

  Repo:    https://github.com/coding-famer/Megatron-Bridge-slime/tree/qwen35
  Bridge:  src/megatron/bridge/models/qwen_vl/qwen35_vl_bridge.py
             - `Qwen35VLMoEBridge` (MoE variant):  lines  68-398
             - `Qwen35VLBridge`    (DENSE variant; matches our use case):
                                                   lines 400-606
             - dense `mapping_registry()`:         lines 478-606
             - `RMSNorm2ZeroCenteredRMSNormMapping(out_norm.weight,
                linear_attn.norm.weight)` for the GDN out_norm: line 583
  Provider: src/megatron/bridge/models/qwen_vl/qwen35_vl_provider.py
             - `Qwen35VLMoEModelProvider.layernorm_zero_centered_gamma = True`:
                                                   line 121
             - `Qwen35VLModelProvider.layernorm_zero_centered_gamma = True`:
                                                   line 267
  Slime training scripts (cross-checked for `--apply-layernorm-1p` and
  the spec/MoE flags):
             - slime/scripts/run-qwen3.5-27B.sh          (dense 27B GRPO)
             - slime/scripts/models/qwen3.5-27B.sh       (model args)
             - slime/scripts/models/qwen3.5-35B-A3B.sh   (MoE 35B-A3B args)
             - slime/examples/geo3k_vlm/run_geo3k_qwen35.sh  (VLM example)
  HF model:  https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
             - `Qwen3_5RMSNormGated` (GDN out_norm class, standard
                convention init=ones, forward = w*x):  lines 175-188
             - `Qwen3_5RMSNorm` (everything else, 1p convention init=zeros,
                forward = (1 + w)*x):                 lines 711-722

Layernorm convention recap (what we *think* is true, sourced from the HF
model file referenced above):
  - HF `Qwen3_5RMSNorm` (input/post-attn/q/k/final norms): 1p convention.
    Stored weight is centered around 0; forward = `(1 + w) * x / rms(x)`.
  - HF `Qwen3_5RMSNormGated` (GDN out_norm only):          standard convention.
    Stored weight is centered around 1; forward = `w * x / rms(x)`.
  - Megatron TENorm with `layernorm_zero_centered_gamma=True` (set on the
    provider — see `Qwen3_5ModelProvider.layernorm_zero_centered_gamma`
    below) applies `(1 + w) * x / rms(x)` for *every* RMSNorm. There is no
    per-module override; the forward formula is fixed at config-time inside
    `transformer_engine.pytorch.RMSNorm` (see `weight + 1 if self.zero_centered_gamma`
    in transformer_engine/pytorch/ops/basic/rmsnorm.py:251).

Implications for the bridge mappings:
  - Regular layernorms: both sides 1p → plain `AutoMapping` is correct
    (direct copy preserves the centered-around-0 stored weight).
  - GDN out_norm: HF stores standard-convention values (≈ 1) but Megatron's
    TENorm interprets them as 1p (effective scale ≈ 2). On paper this calls
    for `RMSNorm2ZeroCenteredRMSNormMapping` (`w_M = w_HF - 1` on load,
    `w_HF = w_M + 1` on save) — exactly what upstream uses on
    qwen35_vl_bridge.py:583.

Differences vs upstream (in *decreasing* suspected importance):

  (A) GDN `out_norm.weight`:
        upstream (qwen35_vl_bridge.py:583):
            `RMSNorm2ZeroCenteredRMSNormMapping`   (theoretically correct)
        ours (this file, around the `out_norm.weight` mapping below):
            `AutoMapping`                          (theoretically wrong)
      Empirically: switching to `RMSNorm2ZeroCenteredRMSNormMapping`
      *worsens* HF/SGLang teacher-forcing NLL on the single-sample overfit
      ckpt from ≈ 0.14 → ≈ 0.53. So a second, compensating bug must exist
      somewhere — the naive direct copy in (A) accidentally cancels it.
      Until the second bug is found, (A) stays as `AutoMapping`.

  (B) Vision-model mappings:
        upstream (qwen35_vl_bridge.py:516-541, dense `Qwen35VLBridge`):
            ~28 explicit per-tensor mappings; vision QKV uses
            `ConcatenatedQKVMapping` (lines 590-595);
            patch_embed/pos_embed use `ReplicatedMapping` (lines 601-605);
            the rest (linear_proj/mlp/norms, incl. `.bias` tensors) use
            `AutoMapping` via the param_mappings dict.
        ours (this file, the `vision_model.** ↔ model.visual.**` block
            below): a single wildcard `ReplicatedMapping`. This is
            identity-copy only — if HF and Megatron disagree on QKV
            concatenation order or any other layout, we silently corrupt
            vision weights.
      Currently dormant: in our SFT overfit run vision weights are
      byte-identical between trained and base ckpts (vision encoder barely
      gradient-updated on a single sample), so any latent corruption hasn't
      surfaced. **Strong candidate for the missing piece in (A)** — switching
      vision to upstream's explicit mappings (with `ConcatenatedQKVMapping`)
      should be tried *together with* fixing (A).

  (C) Stale `pre_mlp_layernorm.weight` mapping:
        upstream dense (qwen35_vl_bridge.py:498-499):
            not present — dense uses the fused
            `mlp.linear_fc1.layer_norm_weight` only.
        upstream MoE (qwen35_vl_bridge.py:229):
            present — MoE has a standalone `pre_mlp_layernorm`.
        ours: still listed in `param_mappings` regardless of dense/MoE.
      Pure no-op for dense models — the parameter doesn't exist in the
      Megatron state dict, the mapping framework just skips it. Cosmetic
      cleanup; no behavioral impact.

TODO(audit, 2026-04-24):
  This is **not** a "just port upstream" task. We already tried the obvious
  one-line port — switching the GDN out_norm mapping from `AutoMapping` to
  `RMSNorm2ZeroCenteredRMSNormMapping` *on its own* — and it caused a
  regression on the single-sample overfit eval:

    | bridge config                                    | HF NLL | SGLang NLL | greedy chars |
    |--------------------------------------------------|--------|------------|--------------|
    | AutoMapping (current, theoretically wrong)       |  0.14  |    0.14    |  235 / 291   |
    | RMSNorm2ZeroCenteredRMSNormMapping (upstream)    |  0.53  |    0.54    |  143 / 291   |

  The numbers above are recorded so nobody redoes that experiment expecting
  a different result. The asymmetry tells us there is a *second*,
  compensating mismatch elsewhere in the bridge (most likely (B) vision
  layout or a head-grouping subtlety in the locally-defined GDN mappings)
  that the naive direct copy on out_norm currently happens to cancel.
  Fixing (A) without also fixing the second mismatch makes things worse.

  So the real task is to *find and fix the second bug* before — or, more
  safely, in the same change as — flipping (A) back to upstream:

    1. Diff the locally-defined `GDNConv1dMapping` and
       `GDNLinearMappingSeparate` (in this file, above the bridge class)
       against the upstream Qwen3-Next versions
       (megatron.bridge.models.conversion.param_mapping.GDNConv1dMapping /
       GDNLinearMapping) to look for sign/permutation/TP-interleave
       differences that would scale-flip the GDN block at the same point
       where out_norm currently absorbs the error.
    2. Replace the wildcard vision `ReplicatedMapping` with upstream's
       explicit per-tensor block (cf. qwen35_vl_bridge.py:516-541 for
       AutoMapping and lines 588-605 for `ConcatenatedQKVMapping` +
       `ReplicatedMapping`). Verify byte-equivalence vs the wildcard on a
       fresh load to confirm no behaviour change for vision; if there *is*
       a difference, that's likely the compensating bug.
    3. Only after (1)/(2) reveal the second bug, switch (A) to
       `RMSNorm2ZeroCenteredRMSNormMapping` (cf. qwen35_vl_bridge.py:583)
       in the same commit, and re-run the single-sample overfit +
       HF/SGLang teacher-forcing eval (see /.tools/overfit/check_hf.py and
       check_sglang.py). Success criterion: HF NLL drops *below* the
       current 0.14 plateau, ideally close to Megatron's training-side loss
       (~ 1e-6).
    4. If (1)/(2) don't surface anything, instrument the Megatron forward
       to dump the GDN block's out_norm input and output activations on
       the overfit sample, and compare against the same activations from
       HF's `Qwen3_5RMSNormGated` on identical input — that should pin the
       residual factor exactly.
    5. While you're in here, drop the `pre_mlp_layernorm.weight` line from
       `param_mappings` (divergence (C)).

  See /overfit.md (section "GDN out_norm bridge conversion — open issue")
  for the empirical numbers and the original investigation transcript.
============================================================================
"""

from __future__ import annotations

import itertools
from copy import deepcopy
from dataclasses import dataclass, field

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple, MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import (
    AutoMapping,
    ColumnParallelMapping,
    GatedMLPMapping,
    MegatronParamMapping,
    QKVMapping,
    ReplicatedMapping,
    RMSNorm2ZeroCenteredRMSNormMapping,
)
from megatron.bridge.models.conversion.utils import remove_non_pickleables
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.utils.common_utils import hook_hf_module_setattr_for_tp_grad_sync
from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt import GPTModel as MCoreGPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule


# ---------------------------------------------------------------------------
# Qwen3.5 GDN mapping helpers and classes
#
# The upstream megatron-bridge (dev_rl branch) has GDNLinearMapping (for
# Qwen3-Next) but not the Qwen3.5-specific variants. We define them here
# instead of patching the installed package.
#
# Key difference from upstream helpers: merge/split functions take a tp_size
# argument so the TP-interleaved layout (blocks per rank) is preserved when
# round-tripping through Megatron.
# ---------------------------------------------------------------------------

def _merge_gdn_linear_weights_tp(
    config,
    qkvz: torch.Tensor,
    ba: torch.Tensor,
    tp_size: int,
) -> torch.Tensor:
    """Merge head-grouped QKVZ + BA into a TP-interleaved in_proj tensor.

    Layout: [tp0: q|k|v|z|b|a per-head-group, tp1: q|k|v|z|b|a, ...]
    """
    hidden_size = config.hidden_size
    qk_head_dim = config.linear_key_head_dim
    v_head_dim = config.linear_value_head_dim
    num_qk_heads = config.linear_num_key_heads
    num_v_heads = config.linear_num_value_heads
    v_per_group = num_v_heads // num_qk_heads

    qkvz_r = qkvz.reshape(num_qk_heads, -1, hidden_size)
    ba_r = ba.reshape(num_qk_heads, -1, hidden_size)
    q, k, v, z = torch.split(
        qkvz_r, [qk_head_dim, qk_head_dim, v_per_group * v_head_dim, v_per_group * v_head_dim], dim=1
    )
    b, a = torch.split(ba_r, [v_per_group, v_per_group], dim=1)

    q, k, v, z, b, a = [w.reshape(tp_size, -1, hidden_size) for w in [q, k, v, z, b, a]]
    return torch.cat([q, k, v, z, b, a], dim=1).reshape(-1, hidden_size)


def _split_gdn_linear_weights_tp(
    config,
    in_proj: torch.Tensor,
    tp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reverse of _merge_gdn_linear_weights_tp: split TP-interleaved in_proj into QKVZ and BA."""
    hidden_size = config.hidden_size
    qk_head_dim = config.linear_key_head_dim
    v_head_dim = config.linear_value_head_dim
    num_qk_heads = config.linear_num_key_heads
    num_v_heads = config.linear_num_value_heads
    qk_dim_local = qk_head_dim * (num_qk_heads // tp_size)
    v_dim_local = v_head_dim * (num_v_heads // tp_size)
    nv_local = num_v_heads // tp_size

    in_proj = in_proj.reshape(tp_size, -1, hidden_size)
    q, k, v, z, b, a = torch.split(
        in_proj, [qk_dim_local, qk_dim_local, v_dim_local, v_dim_local, nv_local, nv_local], dim=1
    )
    q, k, v, z, b, a = [w.reshape(num_qk_heads, -1, hidden_size) for w in [q, k, v, z, b, a]]
    qkvz = torch.cat([q, k, v, z], dim=1).reshape(-1, hidden_size)
    ba = torch.cat([b, a], dim=1).reshape(-1, hidden_size)
    return qkvz, ba


def _fuse_gdn_separate_to_grouped(
    config,
    qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert four flat HF tensors (qkv, z, b, a) into head-grouped (qkvz, ba)."""
    hidden_size = config.hidden_size
    qk_head_dim = config.linear_key_head_dim
    v_head_dim = config.linear_value_head_dim
    num_qk_heads = config.linear_num_key_heads
    num_v_heads = config.linear_num_value_heads
    qk_dim = qk_head_dim * num_qk_heads
    v_dim = v_head_dim * num_v_heads
    v_per_group = num_v_heads // num_qk_heads

    q_f, k_f, v_f = torch.split(qkv, [qk_dim, qk_dim, v_dim], dim=0)
    q_g = q_f.reshape(num_qk_heads, qk_head_dim, hidden_size)
    k_g = k_f.reshape(num_qk_heads, qk_head_dim, hidden_size)
    v_g = v_f.reshape(num_qk_heads, v_per_group * v_head_dim, hidden_size)
    z_g = z.reshape(num_qk_heads, v_per_group * v_head_dim, hidden_size)
    b_g = b.reshape(num_qk_heads, v_per_group, hidden_size)
    a_g = a.reshape(num_qk_heads, v_per_group, hidden_size)

    qkvz = torch.cat([q_g, k_g, v_g, z_g], dim=1).reshape(-1, hidden_size)
    ba = torch.cat([b_g, a_g], dim=1).reshape(-1, hidden_size)
    return qkvz, ba


def _split_gdn_grouped_to_separate(
    config,
    qkvz: torch.Tensor,
    ba: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reverse of _fuse_gdn_separate_to_grouped: head-grouped → four flat tensors."""
    hidden_size = config.hidden_size
    qk_head_dim = config.linear_key_head_dim
    v_head_dim = config.linear_value_head_dim
    num_qk_heads = config.linear_num_key_heads
    v_per_group = config.linear_num_value_heads // num_qk_heads

    qkvz_g = qkvz.reshape(num_qk_heads, -1, hidden_size)
    q_g, k_g, v_g, z_g = torch.split(
        qkvz_g, [qk_head_dim, qk_head_dim, v_per_group * v_head_dim, v_per_group * v_head_dim], dim=1
    )
    qkv = torch.cat([q_g.reshape(-1, hidden_size), k_g.reshape(-1, hidden_size), v_g.reshape(-1, hidden_size)], dim=0)

    ba_g = ba.reshape(num_qk_heads, -1, hidden_size)
    b_g, a_g = torch.split(ba_g, [v_per_group, v_per_group], dim=1)
    return qkv, z_g.reshape(-1, hidden_size), b_g.reshape(-1, hidden_size), a_g.reshape(-1, hidden_size)


class GDNConv1dMapping(MegatronParamMapping):
    """conv1d weight mapping for Qwen3.5 GDN layers.

    HF layout: flat (qk_dim*2 + v_dim, 1, kernel_size) — q, k, v concatenated.
    Megatron layout: TP-interleaved; each rank holds its q/k/v slice head-grouped.

    Follows the same pattern as MambaConv1dMapping.
    """

    def __init__(self, megatron_param: str, hf_param: str):
        super().__init__(megatron_param=megatron_param, hf_param=hf_param)
        self._tp_mapping = ColumnParallelMapping(megatron_param, megatron_param)

    def hf_to_megatron(self, hf_weights: torch.Tensor, megatron_module) -> torch.Tensor:
        if self.tp_rank == 0:
            config = self._get_config(megatron_module)
            qk_dim = config.linear_key_head_dim * config.linear_num_key_heads
            v_dim = config.linear_value_head_dim * config.linear_num_value_heads
            kernel_size = hf_weights.shape[-1]
            shape = (self.tp_size, -1, 1, kernel_size)

            q = hf_weights[:qk_dim].reshape(shape)
            k = hf_weights[qk_dim : 2 * qk_dim].reshape(shape)
            v = hf_weights[2 * qk_dim :].reshape(shape)
            merged = torch.cat([q, k, v], dim=1).reshape(-1, 1, kernel_size)
        else:
            merged = None
        return self._tp_mapping.hf_to_megatron(merged, megatron_module)

    def megatron_to_hf(self, megatron_weights, megatron_module) -> dict:
        megatron_weights = self.broadcast_from_pp_rank(megatron_weights, cache_key=str(self.hf_param))
        if megatron_weights is None:
            return {}
        megatron_weights = self.maybe_dequantize(megatron_weights)

        if megatron_module is None:
            config = self.broadcast_obj_from_pp_rank(None)
        else:
            config = self._get_config(megatron_module)
            config = remove_non_pickleables(config, max_depth=3)
            config = self.broadcast_obj_from_pp_rank(config)

        qk_dim_local = (config.linear_key_head_dim * config.linear_num_key_heads) // self.tp_size
        v_dim_local = (config.linear_value_head_dim * config.linear_num_value_heads) // self.tp_size

        q_local = megatron_weights[:qk_dim_local]
        k_local = megatron_weights[qk_dim_local : 2 * qk_dim_local]
        v_local = megatron_weights[2 * qk_dim_local :]

        full_parts = []
        for comp in [q_local, k_local, v_local]:
            if self.tp_size == 1:
                full_parts.append(comp)
            else:
                gathered = self.gather_from_tp_ranks(comp)
                full_parts.append(torch.cat(gathered, dim=0))

        return {self.hf_param: torch.cat(full_parts, dim=0)}


class GDNLinearMappingSeparate(MegatronParamMapping):
    """in_proj mapping for Qwen3.5 GDN: merges 4 separate HF tensors.

    HF stores four weight matrices (in_proj_qkv, in_proj_z, in_proj_b, in_proj_a).
    Megatron stores them fused in a single TP-interleaved in_proj tensor.
    """

    def __init__(self, megatron_param: str, qkv: str, z: str, b: str, a: str):
        super().__init__(megatron_param, {"qkv": qkv, "z": z, "b": b, "a": a})
        self._tp_mapping = AutoMapping(megatron_param, megatron_param)

    def hf_to_megatron(self, hf_weights: dict, megatron_module) -> torch.Tensor:
        if self.tp_rank == 0:
            config = self._get_config(megatron_module)
            qkvz, ba = _fuse_gdn_separate_to_grouped(
                config, hf_weights["qkv"], hf_weights["z"], hf_weights["b"], hf_weights["a"]
            )
            merged = _merge_gdn_linear_weights_tp(config, qkvz, ba, tp_size=self.tp_size)
        else:
            merged = None
        return self._tp_mapping.hf_to_megatron(merged, megatron_module)

    def megatron_to_hf(self, megatron_weights, megatron_module) -> dict:
        if megatron_weights is not None:
            megatron_weights = self.maybe_dequantize(megatron_weights)
        if megatron_module is None:
            config = self.broadcast_obj_from_pp_rank(None)
        else:
            config = self._get_config(megatron_module)
            config = remove_non_pickleables(config, max_depth=3)
            config = self.broadcast_obj_from_pp_rank(config)

        packed_dict = self._tp_mapping.megatron_to_hf(megatron_weights, megatron_module)
        if not packed_dict:
            return {}

        packed = next(iter(packed_dict.values()))
        qkvz, ba = _split_gdn_linear_weights_tp(config, packed, tp_size=self.tp_size)
        qkv, z, b, a = _split_gdn_grouped_to_separate(config, qkvz, ba)
        return {
            self.hf_param["qkv"]: qkv,
            self.hf_param["z"]: z,
            self.hf_param["b"]: b,
            self.hf_param["a"]: a,
        }

    def resolve(self, captures):
        resolved_megatron_param, resolved_hf_param = self._resolve_names(captures)
        return type(self)(
            resolved_megatron_param,
            resolved_hf_param["qkv"],
            resolved_hf_param["z"],
            resolved_hf_param["b"],
            resolved_hf_param["a"],
        )


# ---------------------------------------------------------------------------
# THD ↔ BSHD helpers (identical to glm4v_moe.py)
# ---------------------------------------------------------------------------
def _thd_to_bshd(packed: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    max_seq = seqlens.max().item()
    bs = len(cu_seqlens) - 1
    out = packed.new_zeros(bs, max_seq, *packed.shape[2:])
    for i, sl in enumerate(seqlens):
        out[i, :sl] = packed[0, cu_seqlens[i] : cu_seqlens[i] + sl]
    return out


def _bshd_to_thd(unpacked: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    total = cu_seqlens[-1].item()
    out = unpacked.new_zeros(1, total, *unpacked.shape[2:])
    for i, sl in enumerate(seqlens):
        out[0, cu_seqlens[i] : cu_seqlens[i] + sl] = unpacked[i, :sl]
    return out


def _gather_input_ids_from_cp(input_ids: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    cp_size = parallel_state.get_context_parallel_world_size()
    if cp_size <= 1:
        return input_ids

    gathered = torch.distributed.nn.all_gather(input_ids, group=parallel_state.get_context_parallel_group())
    local_cu_seqlens = cu_seqlens // cp_size
    num_seqs = len(cu_seqlens) - 1
    whole_list = []
    for i in range(num_seqs):
        seqlen = (cu_seqlens[i + 1] - cu_seqlens[i]).item()
        chunk_size = seqlen // 2 // cp_size
        whole_list.extend(
            gathered[cp_rank][0, local_cu_seqlens[i] : local_cu_seqlens[i] + chunk_size] for cp_rank in range(cp_size)
        )
        whole_list.extend(
            [
                gathered[cp_rank][0, local_cu_seqlens[i] + chunk_size : local_cu_seqlens[i + 1]]
                for cp_rank in range(cp_size)
            ][::-1]
        )
    return torch.cat(whole_list).unsqueeze(0)


def _select_local_image_embeds(
    full_input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    image_token_id: int,
    image_embeds: torch.Tensor,
    cp_rank: int,
    cp_size: int,
) -> torch.Tensor:
    device = full_input_ids.device
    full_flat = full_input_ids[0]
    full_mask = full_flat == image_token_id

    T_global = full_flat.shape[0]
    rank_mask = torch.zeros(T_global, dtype=torch.bool, device=device)

    num_seqs = len(cu_seqlens) - 1
    for i in range(num_seqs):
        seq_start = cu_seqlens[i].item()
        seqlen = (cu_seqlens[i + 1] - cu_seqlens[i]).item()
        chunk_size = seqlen // (2 * cp_size)
        first_start = seq_start + cp_rank * chunk_size
        rank_mask[first_start : first_start + chunk_size] = True
        second_end = seq_start + seqlen - cp_rank * chunk_size
        rank_mask[second_end - chunk_size : second_end] = True

    local_image_mask = full_mask & rank_mask
    n_local = local_image_mask.sum().item()

    if n_local == 0:
        return image_embeds[:0]
    if n_local == image_embeds.shape[0]:
        return image_embeds

    image_cumsum = full_mask.long().cumsum(0)
    local_positions = local_image_mask.nonzero(as_tuple=True)[0]
    embed_indices = image_cumsum[local_positions] - 1
    return image_embeds[embed_indices]


# ---------------------------------------------------------------------------
# Megatron VL Model
# ---------------------------------------------------------------------------
class Qwen3_5Model(MegatronModule):
    """Qwen3.5 vision-language model for Megatron training.

    Wraps a frozen HF Qwen3-VL vision encoder (first PP stage only) together
    with a Megatron Core GPTModel configured for hybrid linear+full attention
    and M-RoPE.
    """

    def __init__(
        self,
        language_transformer_config,
        language_transformer_layer_spec,
        hf_vision_config,
        parallel_output: bool = True,
        pre_process: bool = True,
        post_process: bool = True,
    ) -> None:
        super().__init__(config=language_transformer_config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.image_token_id = language_transformer_config.image_token_id
        self.spatial_merge_size = language_transformer_config.spatial_merge_size

        self.share_embeddings_and_output_weights = False

        self.vision_model = None
        if self.pre_process:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionConfig, Qwen3VLVisionModel

            vision_cfg = Qwen3VLVisionConfig(
                depth=hf_vision_config.depth,
                hidden_size=hf_vision_config.hidden_size,
                hidden_act=hf_vision_config.hidden_act,
                intermediate_size=hf_vision_config.intermediate_size,
                num_heads=hf_vision_config.num_heads,
                in_channels=hf_vision_config.in_channels,
                patch_size=hf_vision_config.patch_size,
                spatial_merge_size=hf_vision_config.spatial_merge_size,
                temporal_patch_size=hf_vision_config.temporal_patch_size,
                out_hidden_size=hf_vision_config.out_hidden_size,
                num_position_embeddings=hf_vision_config.num_position_embeddings,
                deepstack_visual_indexes=getattr(hf_vision_config, "deepstack_visual_indexes", []),
            )
            if hasattr(hf_vision_config, "torch_dtype"):
                vision_cfg.torch_dtype = hf_vision_config.torch_dtype
            # Force flash_attention_2 to avoid CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED
            # when the default SDPA backend tries to use cuDNN.
            vision_cfg._attn_implementation = "flash_attention_2"

            self.vision_model = Qwen3VLVisionModel._from_config(vision_cfg)
            self.vision_model.requires_grad_(False)
            self.vision_model.eval()
            hook_hf_module_setattr_for_tp_grad_sync(self.vision_model)
            if torch.cuda.is_available():
                self.vision_model = self.vision_model.to("cuda")

        self.language_model = MCoreGPTModel(
            config=language_transformer_config,
            transformer_layer_spec=language_transformer_layer_spec,
            vocab_size=language_transformer_config.vocab_size,
            max_sequence_length=language_transformer_config.language_max_sequence_length,
            parallel_output=parallel_output,
            position_embedding_type="mrope",
            rotary_percent=language_transformer_config.rotary_percent,
            pre_process=self.pre_process,
            post_process=self.post_process,
            rotary_base=language_transformer_config.rotary_base,
            fp16_lm_cross_entropy=language_transformer_config.fp16_lm_cross_entropy,
            share_embeddings_and_output_weights=language_transformer_config.share_embeddings_and_output_weights,
            scatter_embedding_sequence_parallel=False,
        )

        self.share_embeddings_and_output_weights = self.language_model.share_embeddings_and_output_weights

    def shared_embedding_or_output_weight(self):
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor):
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1
        if self.pre_process:
            self.encoder_hidden_state = input_tensor[0]
        else:
            self.language_model.set_input_tensor(input_tensor[0])

    def _get_image_features(self, pixel_values, image_grid_thw):
        pixel_values = pixel_values.to(dtype=self.vision_model.dtype)
        with torch.no_grad():
            result = self.vision_model(pixel_values, grid_thw=image_grid_thw)
            # Qwen3.5 vision model returns a tuple (BaseModelOutput); extract the tensor.
            return result[0] if isinstance(result, tuple) else result

    @staticmethod
    def _get_vision_position_ids(start_position, grid_thw, temp_merge_size, spatial_merge_size, device):
        llm_grid_t = grid_thw[0].item() // temp_merge_size
        llm_grid_h = grid_thw[1].item() // spatial_merge_size
        llm_grid_w = grid_thw[2].item() // spatial_merge_size
        n_tokens = llm_grid_h * llm_grid_w * llm_grid_t

        pos_w = torch.arange(start_position, start_position + llm_grid_w, device=device)
        pos_w = pos_w.repeat(llm_grid_h * llm_grid_t)
        pos_h = torch.arange(start_position, start_position + llm_grid_h, device=device)
        pos_h = pos_h.repeat_interleave(llm_grid_w * llm_grid_t)
        pos_t = torch.full((n_tokens,), start_position, device=device, dtype=torch.long)
        return torch.stack([pos_t, pos_h, pos_w], dim=0)

    def _compute_mrope_position_ids(self, input_ids_bshd, image_grid_thw):
        bs, seq_len = input_ids_bshd.shape
        device = input_ids_bshd.device
        spatial_merge_size = self.spatial_merge_size

        position_ids = torch.zeros(3, bs, seq_len, dtype=torch.long, device=device)

        if image_grid_thw is None or image_grid_thw.numel() == 0:
            pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bs, -1)
            position_ids[0] = pos
            position_ids[1] = pos
            position_ids[2] = pos
            return position_ids

        grid_iter = iter(image_grid_thw)

        for b in range(bs):
            ids = input_ids_bshd[b]
            is_image = ids == self.image_token_id
            token_types = is_image.long()
            groups = []
            for key, group in itertools.groupby(enumerate(token_types.tolist()), lambda x: x[1]):
                g = list(group)
                groups.append((key, g[0][0], g[-1][0] + 1))

            current_pos = 0
            pos_list = []
            for modality, start, end in groups:
                if modality == 0:
                    n = end - start
                    pos_list.append(torch.arange(n, device=device).view(1, -1).expand(3, -1) + current_pos)
                    current_pos += n
                else:
                    grid_thw = next(grid_iter)
                    temp_merge_size = grid_thw[0]
                    vis_pos = self._get_vision_position_ids(
                        current_pos, grid_thw, temp_merge_size, spatial_merge_size, device
                    )
                    pos_list.append(vis_pos)
                    current_pos += max(grid_thw[1], grid_thw[2]) // spatial_merge_size

            all_pos = torch.cat(pos_list, dim=1)
            position_ids[:, b, : all_pos.shape[1]] = all_pos

        return position_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        loss_mask: torch.Tensor = None,
        inference_params=None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
        pixel_values: torch.Tensor = None,
        image_grid_thw: torch.Tensor = None,
        pixel_values_videos: torch.Tensor = None,
        video_grid_thw: torch.Tensor = None,
        mm_token_type_ids: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        assert pixel_values_videos is None, "Video not supported yet"
        assert inference_params is None, "Inference not supported"

        cu_seqlens = None
        if packed_seq_params is not None:
            cu_seqlens = (
                packed_seq_params.cu_seqlens_q_padded
                if packed_seq_params.cu_seqlens_q_padded is not None
                else packed_seq_params.cu_seqlens_q
            )
        cp_size = parallel_state.get_context_parallel_world_size()
        full_input_ids = None

        combined_embeddings = None

        if self.pre_process:
            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids,
                position_ids=None,
            ).clone()

            if pixel_values is not None and image_grid_thw is not None:
                image_embeds = self._get_image_features(pixel_values, image_grid_thw)
                image_embeds = image_embeds.to(combined_embeddings.device, combined_embeddings.dtype)

                if cp_size > 1 and cu_seqlens is not None:
                    full_input_ids = _gather_input_ids_from_cp(input_ids, cu_seqlens)
                    cp_rank = parallel_state.get_context_parallel_rank()
                    image_embeds = _select_local_image_embeds(
                        full_input_ids, cu_seqlens, self.image_token_id, image_embeds, cp_rank, cp_size
                    )

                image_mask = (input_ids == self.image_token_id).contiguous()
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()
                if image_mask.any():
                    combined_embeddings[image_mask] = image_embeds
                combined_embeddings = combined_embeddings.transpose(0, 1).contiguous()

            if self.config.sequence_parallel:
                combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeddings)
                combined_embeddings = combined_embeddings.contiguous()

        pp_size = parallel_state.get_pipeline_model_parallel_world_size()

        if position_ids is None:
            if self.pre_process:
                if cu_seqlens is not None:
                    if cp_size > 1:
                        if full_input_ids is None:
                            full_input_ids = _gather_input_ids_from_cp(input_ids, cu_seqlens)
                    else:
                        full_input_ids = input_ids
                    input_ids_bshd = _thd_to_bshd(full_input_ids, cu_seqlens)
                    pos_bshd = self._compute_mrope_position_ids(input_ids_bshd, image_grid_thw)
                    pos_packed = _bshd_to_thd(pos_bshd.permute(1, 2, 0), cu_seqlens)
                    position_ids = pos_packed.permute(2, 0, 1).contiguous()
                else:
                    position_ids = self._compute_mrope_position_ids(input_ids, image_grid_thw)
            else:
                if cu_seqlens is not None:
                    T = cu_seqlens[-1].item()
                    position_ids = torch.zeros(3, 1, T, dtype=torch.long, device=torch.cuda.current_device())
                else:
                    raise NotImplementedError(
                        "Non-THD position_ids broadcast not yet supported for non-first PP stages"
                    )

            if pp_size > 1:
                src = parallel_state.get_pipeline_model_parallel_first_rank()
                torch.distributed.broadcast(
                    position_ids,
                    src=src,
                    group=parallel_state.get_pipeline_model_parallel_group(),
                )

        output = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            loss_mask=loss_mask,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            **(extra_block_kwargs or {}),
        )

        return output


# ---------------------------------------------------------------------------
# Model Provider
# ---------------------------------------------------------------------------
@dataclass
class Qwen3_5ModelProvider(GPTModelProvider):
    """Provider that creates Qwen3_5Model.

    Inherits directly from ``GPTModelProvider`` (dense defaults, no MoE).
    We used to inherit from ``Qwen3NextModelProvider`` because Qwen3.5
    shares the same hybrid linear+full attention + GatedDeltaNet
    architecture, but that provider is parameterised for the 80B-A3B MoE
    checkpoint and silently carries ``num_moe_experts=512`` and other MoE
    defaults into the spec. With a dense ``GPTModelProvider`` base we only
    need to declare the fields Qwen3.5 actually uses.

    Defined at module level (not inside a function) so that it is picklable
    — megatron-bridge broadcasts config objects across PP ranks via
    ``torch.distributed.broadcast_object_list``.
    """

    # --- defaults that differ from GPTModelProvider for all Qwen3.5 dense sizes ---
    bf16: bool = True
    params_dtype: torch.dtype = torch.bfloat16
    autocast_dtype: torch.dtype = torch.bfloat16
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    layernorm_epsilon: float = 1e-6
    # HF Qwen3_5RMSNorm uses 1p convention: stored weight is centered around 0
    # and the effective scale is (1 + weight). Megatron must apply the same
    # convention; otherwise the bridge's AutoMapping (direct copy) silently
    # produces a forward pass with effective scale = weight (≈ 0 at init),
    # which collapses the residual stream and forces training to compensate
    # by pushing weights to ≈ 1 — making the saved HF ckpt unusable for
    # inference (HF reads them as 1+weight ≈ 2). See modeling_qwen3_5.py
    # line 711-722: `output * (1.0 + self.weight)`.
    #
    # CAVEAT: this flag is global — it switches *every* TENorm in the model
    # to the (1 + w) forward, including the GDN out_norm. HF's GDN out_norm
    # is `Qwen3_5RMSNormGated` (modeling_qwen3_5.py:175), which is *standard*
    # convention (init = ones, forward = w * x), so by setting this flag we
    # introduce a per-tensor convention mismatch on out_norm.weight that the
    # bridge mapping must compensate for. See module-level docstring (open
    # issue (A)) and the comment on the `out_norm.weight` mapping below.
    layernorm_zero_centered_gamma: bool = True
    add_bias_linear: bool = False
    gated_linear_unit: bool = True
    normalization: str = "RMSNorm"
    qk_layernorm: bool = True

    # Gated attention (Qwen3.5-specific; Q has 2× width = query + output gate)
    attention_output_gate: bool = True

    # Hybrid linear + full attention (GatedDeltaNet; same family as Qwen3-Next)
    # experimental_attention_variant is set in provide() to avoid triggering
    # TransformerConfig.__post_init__ validation before all GDN fields are ready.
    linear_attention_freq: int = 4
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 32

    # M-RoPE / partial RoPE
    position_embedding_type: str = "mrope"
    rotary_base: float = 10000000.0
    rotary_percent: float = 0.25
    mrope_section: list[int] = field(default_factory=lambda: [11, 11, 10])

    # Vision hooks (filled in by Qwen3_5Bridge.provider_bridge from hf_config)
    hf_vision_config: object = None
    hf_text_config: object = None
    image_token_id: int = 248056
    video_token_id: int = 248057
    spatial_merge_size: int = 2

    # Long-context sequence length (Qwen3.5 supports up to 262 144)
    language_max_sequence_length: int = 262144
    seq_length: int = 262144

    # GPTModelProvider defaults to learned_absolute / scatter_embedding_sp=True; we need neither
    scatter_embedding_sequence_parallel: bool = False

    def provide(self, pre_process=None, post_process=None, vp_stage=None):
        """Build a Qwen3.5 VL model using native Megatron-Core GatedDeltaNet.

        Uses ``get_gpt_decoder_block_spec`` with
        ``experimental_attention_variant="gated_delta_net"`` so that Megatron
        builds the correct GDN spec automatically for each linear-attention layer
        — no per-layer override loop needed.  The resulting parameter names
        (``in_proj.weight``, ``out_norm.weight``, ``conv1d.weight``, …) match
        the native GDN names expected by our ``mapping_registry``.
        """
        if pre_process is None:
            pre_process = parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=vp_stage)
        if post_process is None:
            post_process = parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)

        # Force MoE off for dense Qwen3.5 (num_moe_experts=None).
        self.moe_layer_freq = [0] * self.num_layers

        # Activate native GDN; get_gpt_decoder_block_spec reads this field
        # and inserts GatedDeltaNet specs for layers determined by linear_attention_freq.
        self.experimental_attention_variant = "gated_delta_net"

        transformer_layer_spec = get_gpt_decoder_block_spec(
            config=self,
            use_transformer_engine=True,
            vp_stage=vp_stage,
        )

        return Qwen3_5Model(
            language_transformer_config=self,
            language_transformer_layer_spec=transformer_layer_spec,
            hf_vision_config=self.hf_vision_config,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------
# Qwen3.5 (model_type "qwen3_5") is not in transformers 4.57.1 (which ships with
# slime v0.2.4).  We need three stubs so AutoBridge can load and save checkpoints:
#
#  1. Qwen3_5TextConfig  → overrides model_type to "qwen3_5_text" (the type that
#     SGLang's qwen3_5.py expects for the text sub-config).  Without this,
#     Qwen3_5Config inherits Qwen3VLConfig.sub_configs which uses Qwen3VLTextConfig
#     (model_type="qwen3_vl_text"), causing SGLang to reject the saved checkpoint.
#
#  2. Qwen3_5Config  → registered with AutoConfig so AutoConfig.from_pretrained
#     doesn't raise "model type qwen3_5 not recognised".  Subclasses Qwen3VLConfig
#     so text_config / vision_config are parsed into proper sub-config objects, but
#     overrides sub_configs["text_config"] to use Qwen3_5TextConfig.
#
#  3. Qwen3_5ForConditionalGeneration  → stub class registered in the transformers
#     namespace; AutoBridge._causal_lm_architecture does getattr(transformers, arch)
#     and dispatches to get_model_bridge via this class object.
import transformers as _transformers  # noqa: E402
from transformers import AutoConfig  # noqa: E402

if not hasattr(_transformers, "Qwen3_5ForConditionalGeneration"):
    from transformers import Qwen3VLConfig
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig

    class Qwen3_5TextConfig(Qwen3VLTextConfig):
        # Override the inherited "qwen3_vl_text" so that saved HF checkpoints
        # carry the correct type that SGLang's qwen3_5.py expects.
        model_type = "qwen3_5_text"

    class Qwen3_5Config(Qwen3VLConfig):
        model_type = "qwen3_5"
        sub_configs = {**Qwen3VLConfig.sub_configs, "text_config": Qwen3_5TextConfig}

    AutoConfig.register("qwen3_5", Qwen3_5Config)

    class Qwen3_5ForConditionalGeneration:
        """Stub — enables megatron.bridge dispatch for Qwen3.5 checkpoints."""

    _transformers.Qwen3_5ForConditionalGeneration = Qwen3_5ForConditionalGeneration

    # megatron.bridge imports its own _LazyModule copy of transformers that is a
    # different object from the host's `transformers`.  Patch both so that
    # auto_bridge._validate_config and _causal_lm_architecture can find the stub.
    try:
        import megatron.bridge.models.conversion.auto_bridge as _ab
        _ab_tf = _ab.__dict__.get("transformers")
        if _ab_tf is not None and _ab_tf is not _transformers:
            setattr(_ab_tf, "Qwen3_5ForConditionalGeneration", Qwen3_5ForConditionalGeneration)
    except Exception:
        pass


# Register parallelism type for the native Megatron GatedDeltaNet module.
# AutoMapping._detect_parallelism_type inspects the leaf module class name;
# GatedDeltaNet uses ColumnParallelLinear for in_proj, so "column" is correct.
AutoMapping.register_module_type("GatedDeltaNet", "column")


@MegatronModelBridge.register_bridge(
    source=_transformers.Qwen3_5ForConditionalGeneration, target=Qwen3_5Model
)
class Qwen3_5Bridge(MegatronModelBridge):
    """Bridge between HuggingFace Qwen3.5 and the Megatron VL model."""

    def provider_bridge(self, hf_pretrained):
        hf_config = hf_pretrained.config
        text_config = hf_config.text_config
        vision_config = deepcopy(hf_config.vision_config)

        model_dtype = self.dtype_from_hf(text_config, default=torch.bfloat16)
        vision_config.torch_dtype = model_dtype

        rope_params = getattr(text_config, "rope_parameters", {}) or {}
        mrope_section = rope_params.get("mrope_section", [11, 11, 10])
        rotary_base = rope_params.get("rope_theta", 10000000)
        partial_rotary_factor = rope_params.get("partial_rotary_factor", 0.25)

        num_layers = text_config.num_hidden_layers

        provider = Qwen3_5ModelProvider(
            # Language model
            num_layers=num_layers,
            hidden_size=text_config.hidden_size,
            ffn_hidden_size=text_config.intermediate_size,
            num_attention_heads=text_config.num_attention_heads,
            num_query_groups=text_config.num_key_value_heads,
            kv_channels=getattr(text_config, "head_dim", 256),
            init_method_std=text_config.initializer_range,
            layernorm_epsilon=text_config.rms_norm_eps,
            gated_linear_unit=True,
            make_vocab_size_divisible_by=self.make_vocab_size_divisible_by(text_config.vocab_size),
            rotary_base=rotary_base,
            rotary_percent=partial_rotary_factor,
            share_embeddings_and_output_weights=getattr(text_config, "tie_word_embeddings", True),
            vocab_size=text_config.vocab_size,
            seq_length=text_config.max_position_embeddings,
            fp16=(model_dtype == torch.float16),
            bf16=(model_dtype == torch.bfloat16),
            params_dtype=model_dtype,
            # Hybrid attention
            attention_output_gate=getattr(text_config, "attn_output_gate", True),
            linear_attention_freq=getattr(text_config, "full_attention_interval", 4),
            linear_conv_kernel_dim=getattr(text_config, "linear_conv_kernel_dim", 4),
            linear_key_head_dim=getattr(text_config, "linear_key_head_dim", 128),
            linear_value_head_dim=getattr(text_config, "linear_value_head_dim", 128),
            linear_num_key_heads=getattr(text_config, "linear_num_key_heads", 16),
            linear_num_value_heads=getattr(text_config, "linear_num_value_heads", 32),
            # MTP
            mtp_num_layers=getattr(text_config, "mtp_num_hidden_layers", None),
            # M-RoPE
            mrope_section=mrope_section,
            position_embedding_type="mrope",
            scatter_embedding_sequence_parallel=False,
            # Vision
            hf_vision_config=vision_config,
            hf_text_config=text_config,
            image_token_id=getattr(hf_config, "image_token_id", 248056),
            video_token_id=getattr(hf_config, "video_token_id", 248057),
            spatial_merge_size=getattr(hf_config.vision_config, "spatial_merge_size", 2),
            language_max_sequence_length=text_config.max_position_embeddings,
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        """Weight mappings from HF Qwen3.5 to Megatron format (native GDN).

        Uses `*` wildcard for all layers — the mapping framework skips
        non-existent parameters gracefully, so linear-attention layers ignore
        QKV mappings and full-attention layers ignore GDN mappings.

        See the module-level docstring for the three known divergences from
        upstream Megatron-Bridge-slime/qwen35 and the open issue around the
        GDN out_norm convention.
        """
        # Layernorm conventions (with layernorm_zero_centered_gamma=True):
        #   * HF Qwen3_5RMSNorm (input/post-attn/q/k/final norms): 1p
        #     convention, same as Megatron → plain AutoMapping is correct,
        #     both sides store the centered-around-0 weight.
        #   * HF Qwen3_5RMSNormGated (GDN out_norm only): standard
        #     convention (init=1, forward = w*x/rms(x)). On paper this needs
        #     RMSNorm2ZeroCenteredRMSNormMapping (subtract 1 on HF→M, add 1
        #     on M→HF), matching upstream qwen35_vl_bridge.py. Empirically
        #     that *worsens* eval; we keep AutoMapping for now. See the
        #     module-level TODO and the comment on the out_norm.weight
        #     mapping in the special-mappings list below.
        param_mappings = {
            # Embeddings and output
            "language_model.embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
            "language_model.output_layer.weight": "lm_head.weight",
            # Final layernorm
            "language_model.decoder.final_layernorm.weight": "model.language_model.norm.weight",
            # Full attention: input layernorm (fused into linear_qkv by TE)
            "language_model.decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.language_model.layers.*.input_layernorm.weight",
            # Full attention: output projection
            "language_model.decoder.layers.*.self_attention.linear_proj.weight": "model.language_model.layers.*.self_attn.o_proj.weight",
            # Full attention: QK norms
            "language_model.decoder.layers.*.self_attention.q_layernorm.weight": "model.language_model.layers.*.self_attn.q_norm.weight",
            "language_model.decoder.layers.*.self_attention.k_layernorm.weight": "model.language_model.layers.*.self_attn.k_norm.weight",
            # Post-attention layernorm: dense Qwen3.5 fuses it into linear_fc1
            # (linear_fc1.layer_norm_weight). The pre_mlp_layernorm.weight
            # entry below is a no-op for dense models — the param doesn't
            # exist in the Megatron state dict so the framework skips it. It
            # is kept for the MoE variant where the pre-MLP norm is a
            # standalone module. Upstream's dense bridge omits this line; see
            # divergence (C) in the module-level docstring. TODO: drop when
            # the next out_norm/vision audit lands.
            "language_model.decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.language_model.layers.*.post_attention_layernorm.weight",
            "language_model.decoder.layers.*.pre_mlp_layernorm.weight": "model.language_model.layers.*.post_attention_layernorm.weight",
            # MLP output (all layers)
            "language_model.decoder.layers.*.mlp.linear_fc2.weight": "model.language_model.layers.*.mlp.down_proj.weight",
            # Native GDN: input layernorm (fused into in_proj by TE)
            "language_model.decoder.layers.*.self_attention.in_proj.layer_norm_weight": "model.language_model.layers.*.input_layernorm.weight",
            # Native GDN: scalar parameters (not TP-sharded)
            "language_model.decoder.layers.*.self_attention.A_log": "model.language_model.layers.*.linear_attn.A_log",
            "language_model.decoder.layers.*.self_attention.dt_bias": "model.language_model.layers.*.linear_attn.dt_bias",
            # Native GDN: output projection
            "language_model.decoder.layers.*.self_attention.out_proj.weight": "model.language_model.layers.*.linear_attn.out_proj.weight",
        }

        mapping_list = []
        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        mapping_list.extend(
            [
                # Vision model weights — single wildcard ReplicatedMapping
                # (identity copy after `vision_model.** ↔ model.visual.**`
                # name substitution).
                #
                # Diverges from upstream qwen35_vl_bridge.py (divergence (B)
                # in the module-level docstring), which spells out ~28
                # explicit per-tensor mappings, including `ConcatenatedQKVMapping`
                # for vision QKV (HF stores `attn.qkv.weight`/.bias as a
                # single concatenated tensor; Megatron's `linear_qkv` may use
                # a different concat order). If those layouts disagree, this
                # wildcard silently corrupts vision weights — but it has
                # stayed dormant on the SFT overfit path because vision is
                # essentially frozen on a single-image sample.
                #
                # TODO: replace with upstream's explicit mappings when
                # revisiting the GDN out_norm open issue (the two are
                # suspected to be related — see module-level docstring).
                ReplicatedMapping(
                    megatron_param="vision_model.**",
                    hf_param="model.visual.**",
                ),
                # Full attention: QKV weight (attention_output_gate=True handled by merge_qkv_weights)
                QKVMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.linear_qkv.weight",
                    q="model.language_model.layers.*.self_attn.q_proj.weight",
                    k="model.language_model.layers.*.self_attn.k_proj.weight",
                    v="model.language_model.layers.*.self_attn.v_proj.weight",
                ),
                # MLP gate+up (all layers)
                GatedMLPMapping(
                    megatron_param="language_model.decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.language_model.layers.*.mlp.gate_proj.weight",
                    up="model.language_model.layers.*.mlp.up_proj.weight",
                ),
                # Native GDN: conv1d — head-grouped TP scatter (q/k/v
                # interleaved per rank). NOTE: GDNConv1dMapping is defined
                # locally in this file (not imported from megatron.bridge);
                # subtle TP-grouping differences vs upstream are possible and
                # are listed as a follow-up in the module-level TODO.
                GDNConv1dMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.conv1d.weight",
                    hf_param="model.language_model.layers.*.linear_attn.conv1d.weight",
                ),
                # Native GDN: fused in_proj — merge 4 separate HF tensors into
                # Megatron's single head-grouped column-parallel in_proj.
                # NOTE: like GDNConv1dMapping above, this class is defined
                # locally; included in the module-level TODO audit.
                GDNLinearMappingSeparate(
                    megatron_param="language_model.decoder.layers.*.self_attention.in_proj.weight",
                    qkv="model.language_model.layers.*.linear_attn.in_proj_qkv.weight",
                    z="model.language_model.layers.*.linear_attn.in_proj_z.weight",
                    b="model.language_model.layers.*.linear_attn.in_proj_b.weight",
                    a="model.language_model.layers.*.linear_attn.in_proj_a.weight",
                ),
                # Native GDN: output norm — KNOWN OPEN ISSUE, see (A) in the
                # module-level docstring.
                #
                # Theoretical note: HF Qwen3_5RMSNormGated
                # (modeling_qwen3_5.py:175-188) stores standard-convention
                # weights (init=ones, forward = w*x/rms(x)) while Megatron's
                # TENorm with layernorm_zero_centered_gamma=True applies
                # (1+w)*x/rms(x). On paper this calls for
                # RMSNorm2ZeroCenteredRMSNormMapping (subtract 1 on HF→M,
                # add 1 on M→HF), matching upstream qwen35_vl_bridge.py:583.
                #
                # Empirically that one-line change *worsens* HF/SGLang
                # teacher-forcing NLL on the single-sample overfit ckpt
                # from ~0.14 to ~0.53 (greedy chars 235 → 143). Until the
                # second compensating bug is located, leaving this as plain
                # AutoMapping happens to give the better number. **Do not
                # naively switch to RMSNorm2ZeroCenteredRMSNormMapping in
                # isolation** — see the TODO in the module-level docstring
                # for the full debugging plan.
                AutoMapping(
                    megatron_param="language_model.decoder.layers.*.self_attention.out_norm.weight",
                    hf_param="model.language_model.layers.*.linear_attn.norm.weight",
                ),
            ]
        )

        return MegatronMappingRegistry(*mapping_list)

    def stream_weights_megatron_to_hf(
        self,
        megatron_model,
        hf_pretrained,
        cpu: bool = True,
        show_progress: bool = True,
        conversion_tasks=None,
    ):
        """Yield Megatron weights in HF format, with MTP pass-through.

        The Qwen3.5-2B HF checkpoint includes 15 `mtp.*` tensors for the
        Multi-Token Prediction head.  The Megatron model for SFT training does
        not build an MTP block (mtp_block_spec=None), so those tensors are
        absent from the Megatron state dict and would cause the safetensors
        shard to be incomplete.

        This override appends the original MTP tensors unchanged from the
        source HF checkpoint so that the saved checkpoint is complete and
        loadable.  All TP/PP ranks read from disk independently (no collective
        needed); rank 0 writes to disk while other ranks exhaust the generator.
        """
        yield from super().stream_weights_megatron_to_hf(
            megatron_model,
            hf_pretrained,
            cpu=cpu,
            show_progress=show_progress,
            conversion_tasks=conversion_tasks,
        )
        if not (hasattr(hf_pretrained, "state") and hasattr(hf_pretrained.state, "source")):
            return
        source = hf_pretrained.state.source
        mtp_keys = [k for k in source.get_all_keys() if k.startswith("mtp.")]
        if not mtp_keys:
            return
        mtp_tensors = source.load_tensors(mtp_keys)
        for key in mtp_keys:
            tensor = mtp_tensors[key]
            if cpu:
                tensor = tensor.cpu()
            yield HFWeightTuple(param_name=key, weight=tensor, megatron_param_name=key)
