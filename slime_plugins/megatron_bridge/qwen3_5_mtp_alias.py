"""Fix megatron-bridge 0.5.0's dense/MoE Qwen3_5 MTP submodule-naming skew.

Importing this module patches ``MegatronMappingRegistry`` so that every bridge
registry also resolves the MTP inner block under the name megatron-core actually
builds it with.

Background
----------
megatron-bridge 0.5.0's ``Qwen3_5`` bridges (``qwen_vl/qwen35_vl_bridge.py``)
register their MTP weight mappings under ``language_model.mtp.layers.*.mtp_model_layer.*``,
but the model that the *same* bridge provider builds (via megatron-core 0.16's
``get_gpt_mtp_block_spec``) names that submodule ``...transformer_layer.*``. The
8 inner MTP params (attention + MLP) therefore have no mapping, which breaks BOTH
directions:

  * **load** — ``build_conversion_tasks`` leaves those slots ``None`` → the
    consumer loop dereferences ``None.megatron_module`` and crashes with
    ``AttributeError: 'NoneType' object has no attribute 'megatron_module'``.
  * **save** — ``stream_weights_megatron_to_hf`` never yields the MTP HF keys, so
    for a single-shard checkpoint (e.g. Qwen3.5-2B) the shard never completes and
    ``save_generator`` silently writes nothing under ``strict=True``.

Fix
---
For every registered mapping whose ``megatron_param`` contains ``mtp_model_layer``,
add a shallow-copied alias with that segment renamed to ``transformer_layer``.
``copy.copy`` preserves the concrete mapping type (``AutoMapping`` / ``QKVMapping`` /
``GatedMLPMapping`` / …) and its ``hf_param`` (the HF target names do not change —
only the Megatron-side submodule name does), so this is type-agnostic and covers
both the dense and MoE Qwen3_5 bridges. The original ``mtp_model_layer`` mappings
are left in place (harmless: no built param uses that name).

This is registered automatically via ``slime_plugins.megatron_bridge``; no model
arg or script change is needed.
"""

from __future__ import annotations

import copy

from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry

_MTP_BUILT_NAME = "transformer_layer"
_MTP_MAPPED_NAME = "mtp_model_layer"

# Idempotent AND reload-safe: everything lives inside the guard, and the
# original __init__ is frozen as a default argument (not a module global —
# ``importlib.reload`` re-executes the module body, which would rebind a
# global ``_orig_init`` to the installed wrapper itself and recurse forever).
if getattr(MegatronMappingRegistry.__init__, "_mtp_alias_patched", False) is False:

    def _init_with_mtp_alias(self, *mappings, _orig_init=MegatronMappingRegistry.__init__):
        extra = []
        seen = {getattr(m, "megatron_param", None) for m in mappings}
        for m in mappings:
            mp = getattr(m, "megatron_param", None)
            if not mp or _MTP_MAPPED_NAME not in mp:
                continue
            aliased = mp.replace(_MTP_MAPPED_NAME, _MTP_BUILT_NAME)
            if aliased in seen:
                continue
            clone = copy.copy(m)  # preserves type + hf_param; only megatron_param differs
            clone.megatron_param = aliased
            extra.append(clone)
            seen.add(aliased)
        _orig_init(self, *mappings, *extra)

    _init_with_mtp_alias._mtp_alias_patched = True
    MegatronMappingRegistry.__init__ = _init_with_mtp_alias
