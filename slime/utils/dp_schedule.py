"""Per-group DP/microbatch scheduling.

Pure-Python logic that decides, for one rollout batch's worth of sample lengths,
how to group samples into micro-batches and which DP rank owns each mbs.
Lives outside the ray/sglang-importing modules so it can be unit-tested
under CPU-only CI.

The scheduling philosophy is **pack first, distribute second**:

  1. Group samples by training group id (``group_indices[i]`` =
     ``samples[i].group_id`` with a fallback to ``samples[i].index``) and
     split groups into steps of ``global_batch_size`` groups each. In the
     common case one rollout emits one training sample so this is the same as
     a contiguous chunk; under compact / subagent one rollout may emit
     multiple training samples, in which case all of those samples stay in the
     same step.
  2. For each step, pack its samples into ``K`` micro-batches with a
     single first-fit pass (dynamic batch) or fixed-size chunking
     (static batch).
  3. Adjust ``K`` to a multiple of ``dp_size * (mb_group if vpp>1 else 1)``
     by splitting the largest multi-sample bins (dynamic only). If splitting
     saturates (every bin already a singleton — ``max_tokens_per_gpu`` < 2x
     sample length), round DOWN to the previous multiple by merging the
     smallest bins instead (<= align_to - 1 merges).
  4. Distribute the ``K`` mbs across ``dp_size`` ranks, ``K / dp_size``
     each, with either a strided round-robin or a Karmarkar-Karp pass on
     mbs token sums.

Invariants guaranteed by :func:`build_dp_schedule` (asserted by the tests):
  - every DP rank runs the **same** ``num_microbatches`` per training step
    (required for PP sync);
  - every mbs (dynamic path) holds ``<= max_tokens_per_gpu * cp_size``
    tokens, with two exceptions — an individual sample larger than that cap
    lands alone in its own mbs, and up to ``align_to - 1`` merged bins per
    step (the splitting-saturated fallback above) may modestly exceed the
    cap;
  - the union of per-rank sample indices equals the set of samples kept
    after trimming trailing groups (every kept sample placed exactly
    once);
  - flattening ``micro_batch_indices`` for a rank yields
    ``range(num_samples_rank)`` (each rank's samples are tiled exactly
    once by its mbs schedule).
"""

from __future__ import annotations

import logging
from typing import Any

from slime.utils.seqlen_balancing import (
    expand_bins_by_splitting,
    first_fit_pack,
    get_seqlen_balanced_partitions,
    merge_bins_down,
)

logger = logging.getLogger(__name__)


def _pack_step_into_mbs(
    step_lengths: list[int],
    *,
    use_dynamic_batch_size: bool,
    max_per_bin: int | None,
    micro_batch_size: int | None,
) -> list[list[int]]:
    """Group a step's samples into mbs. Returns ``mbs[k]`` = local indices into ``step_lengths``."""
    if use_dynamic_batch_size:
        assert max_per_bin is not None
        return first_fit_pack(step_lengths, max_per_bin)
    assert micro_batch_size is not None
    n = len(step_lengths)
    return [list(range(i, min(i + micro_batch_size, n))) for i in range(0, n, micro_batch_size)]


def build_dp_schedule(
    args: Any,
    train_parallel_config: dict,
    total_lengths: list[int],
    *,
    global_batch_size: int,
    group_indices: list[int],
) -> tuple[list[list[int]], list[list[list[int]]], list[int], list[int]]:
    """Compute the per-rank DP partition and micro-batch schedule.

    See module docstring for the pack-first-distribute-second strategy.

    Args:
        args: Namespace with ``micro_batch_size``, ``use_dynamic_batch_size``,
            ``max_tokens_per_gpu``, ``balance_data``.
        train_parallel_config: ``{"dp_size", "cp_size", "vpp_size",
            "microbatch_group_size_per_vp_stage"}``.
        total_lengths: token count per sample, indexed globally.
        global_batch_size: number of groups (NOT training samples) per
            training step. Number of training steps =
            ``num_groups // global_batch_size``; trailing groups whose
            samples don't fit are dropped.
        group_indices: group id for each sample. Samples sharing the same id
            are kept together in one step.

    Returns:
        ``(partitions, micro_batch_indices, num_microbatches, global_batch_sizes)``.
        ``global_batch_sizes[s]`` = group count for step s (constant
        ``global_batch_size`` for every step).
    """
    dp_size = train_parallel_config["dp_size"]
    cp_size = train_parallel_config["cp_size"]
    vpp_size = train_parallel_config["vpp_size"]
    mb_group = train_parallel_config["microbatch_group_size_per_vp_stage"]

    max_per_bin = None
    if args.use_dynamic_batch_size:
        assert args.max_tokens_per_gpu is not None
        max_per_bin = args.max_tokens_per_gpu * cp_size

    # mbs count per step must be divisible by (dp_size * mb_group_for_vpp) so
    # every rank ends up with the same num_mbs and (for VPP) the per-rank mbs
    # count is a multiple of mb_group.
    align_to = dp_size * (mb_group if vpp_size > 1 else 1)

    # Group samples by group id (preserve first-occurrence order). All samples
    # from one group stay in a single step so the per-group loss reducer is
    # well-defined.
    group_id_to_samples: dict[int, list[int]] = {}
    for sample_pos, group_id in enumerate(group_indices):
        group_id_to_samples.setdefault(group_id, []).append(sample_pos)
    group_ids = list(group_id_to_samples.keys())

    num_steps = len(group_ids) // global_batch_size
    assert num_steps >= 1, (
        f"num_groups ({len(group_ids)}) < global_batch_size ({global_batch_size}); "
        f"need at least one group per step."
    )

    partitions: list[list[int]] = [[] for _ in range(dp_size)]
    micro_batch_indices: list[list[list[int]]] = [[] for _ in range(dp_size)]
    num_microbatches: list[int] = []
    global_batch_sizes: list[int] = []

    for step_i in range(num_steps):
        step_groups = group_ids[step_i * global_batch_size : (step_i + 1) * global_batch_size]
        sample_indices = [pos for group_id in step_groups for pos in group_id_to_samples[group_id]]
        step_lengths = [total_lengths[i] for i in sample_indices]
        global_batch_sizes.append(global_batch_size)
        assert len(sample_indices) >= dp_size, (
            f"step {step_i}: {len(sample_indices)} samples < dp_size {dp_size}; "
            f"each step needs at least one sample per rank."
        )

        # 1. Pack samples in this step into mbs with one global pass.
        # ``step_mbs`` indices are LOCAL into ``sample_indices``.
        step_mbs = _pack_step_into_mbs(
            step_lengths,
            use_dynamic_batch_size=args.use_dynamic_batch_size,
            max_per_bin=max_per_bin,
            micro_batch_size=getattr(args, "micro_batch_size", None),
        )

        # 2. Align mbs count to a multiple of ``align_to``.
        target_K = max(((len(step_mbs) + align_to - 1) // align_to) * align_to, align_to)
        if target_K != len(step_mbs):
            if args.use_dynamic_batch_size:
                expand_bins_by_splitting(step_mbs, target_K, step_lengths)
                if len(step_mbs) != target_K:
                    # Splitting saturated (all bins singletons — happens when
                    # ``max_tokens_per_gpu`` < 2x sample length, so first-fit
                    # packing is already 1 sample/bin). Round DOWN to the
                    # previous multiple by merging the smallest bins instead:
                    # <= align_to - 1 merges, each bounded by the step's two
                    # smallest samples.
                    target_K_down = (len(step_mbs) // align_to) * align_to
                    assert target_K_down >= align_to, (
                        f"dynamic path: step {step_i} produced {len(step_mbs)} mbs from "
                        f"{len(sample_indices)} samples — fewer than dp_size * mb_group "
                        f"({align_to}); neither splitting up nor merging down can align. "
                        f"Lower vpp/mb_group or raise global_batch_size."
                    )
                    logger.info(
                        f"dynamic path: step {step_i} cannot split {len(step_mbs)} mbs up to "
                        f"{target_K} (all bins single-sample); merging down to {target_K_down}."
                    )
                    merge_bins_down(step_mbs, target_K_down, step_lengths)
                    target_K = target_K_down
                assert len(step_mbs) == target_K
            else:
                raise AssertionError(
                    f"static path: num_mbs ({len(step_mbs)}) is not a multiple of "
                    f"dp_size * mb_group ({align_to}); got "
                    f"step_size={len(sample_indices)}, micro_batch_size={args.micro_batch_size}, "
                    f"dp_size={dp_size}, mb_group={mb_group if vpp_size > 1 else 1}. "
                    f"Splitting static mbs would break the fixed-size invariant; adjust the config "
                    f"so step_size % (dp_size * micro_batch_size * mb_group) == 0."
                )

        K = len(step_mbs)
        num_mbs_per_rank = K // dp_size
        num_microbatches.append(num_mbs_per_rank)

        # 3. Distribute mbs across ranks: KK on mbs token sums when balance_data is on,
        # otherwise a strided round-robin. Both produce ``num_mbs_per_rank`` mbs per
        # rank (equal_size=True is what KK needs for PP to stay synced).
        if args.balance_data:
            mbs_token_sums = [sum(step_lengths[i] for i in bin_) for bin_ in step_mbs]
            rank_mbs_idx = get_seqlen_balanced_partitions(mbs_token_sums, dp_size, equal_size=True)
        else:
            rank_mbs_idx = [list(range(r, K, dp_size)) for r in range(dp_size)]

        # 4. Build per-rank partitions (global sample indices) and micro_batch_indices
        # (local indices into partitions[r]).
        for r in range(dp_size):
            for mbs_idx in rank_mbs_idx[r]:
                mbs_locals = step_mbs[mbs_idx]  # local indices into sample_indices
                local_start = len(partitions[r])
                partitions[r].extend(sample_indices[i] for i in mbs_locals)
                micro_batch_indices[r].append(list(range(local_start, local_start + len(mbs_locals))))

    return partitions, micro_batch_indices, num_microbatches, global_batch_sizes


#: Per-sample-list fillers for :func:`pad_static_groups` dummy rows. Each
#: callable receives the index of the group's FIRST real row and returns the
#: dummy's value for that key. Keys absent from the train-data dict are
#: simply skipped; a per-sample key present in the dict but missing here
#: raises (extend the map rather than guessing a neutral value).
_STATIC_PAD_FILLERS: dict[str, Any] = {
    "tokens": lambda data, i: [0, 0],  # 1 prompt + 1 response token
    "response_lengths": lambda data, i: 1,
    "loss_masks": lambda data, i: [0],  # zero gradient
    "rewards": lambda data, i: 0.0,
    "truncated": lambda data, i: 0,
    "sample_indices": lambda data, i: data["sample_indices"][i],
    "group_ids": lambda data, i: data["group_ids"][i],
    # The dummy's mask sum is 0, so the group's total is unchanged — reuse it.
    "group_mask_sums": lambda data, i: data["group_mask_sums"][i],
    "rollout_log_probs": lambda data, i: [0.0],
    "teacher_log_probs": lambda data, i: [0.0],
    "multimodal_train_inputs": lambda data, i: None,
    "multimodal_lazy_payloads": lambda data, i: None,
    "rollout_routed_experts": lambda data, i: None,
    "prompt": lambda data, i: data["prompt"][i],
    "round_number": lambda data, i: data["round_number"][i],
    "metadata": lambda data, i: None,
}

#: Keys that are per-sample-length in some converters but consumed WHOLE with
#: their own shape contract — never padded. (``raw_reward`` feeds pass-rate
#: logging's ``[rollout_batch_size, n_samples_per_prompt]`` reshape.)
_STATIC_PAD_SKIP = {"raw_reward"}


def pad_static_groups(args: Any, train_parallel_config: dict, data: dict) -> None:
    """Pad every group in ``data`` IN PLACE with zero-loss dummy rows so its
    sample count is a multiple of ``dp_size * micro_batch_size * mb_group``.

    The STATIC (fixed ``micro_batch_size``) path of :func:`build_dp_schedule`
    can neither split nor merge bins, so per-step micro-batch alignment must
    hold in the data itself; padding per GROUP keeps any whole-group step
    composition aligned. Dummy rows share their group's id (same training
    step, same loss group) and carry ``loss_mask=[0]`` — zero gradient, and
    the group's ``group_mask_sums`` denominator is unchanged.

    No-op on the dynamic path (elastic bins split/merge to align — see
    ``expand_bins_by_splitting`` / ``merge_bins_down``) and at
    ``dp * micro_batch_size * mb_group == 1``.
    """
    if getattr(args, "use_dynamic_batch_size", False):
        return
    dp_size = train_parallel_config["dp_size"]
    vpp_size = train_parallel_config["vpp_size"]
    mb_group = train_parallel_config["microbatch_group_size_per_vp_stage"] if vpp_size > 1 else 1
    unit = dp_size * (getattr(args, "micro_batch_size", 1) or 1) * mb_group
    if unit <= 1:
        return

    group_ids = data["group_ids"]
    n = len(group_ids)
    counts: dict[int, int] = {}
    first_row: dict[int, int] = {}
    for i, gid in enumerate(group_ids):
        counts[gid] = counts.get(gid, 0) + 1
        first_row.setdefault(gid, i)

    per_sample_keys = [
        k for k, v in data.items() if k not in _STATIC_PAD_SKIP and isinstance(v, list) and len(v) == n
    ]
    unknown = [k for k in per_sample_keys if k not in _STATIC_PAD_FILLERS]
    assert not unknown, (
        f"pad_static_groups: no dummy filler for per-sample key(s) {unknown}; "
        f"extend _STATIC_PAD_FILLERS (or _STATIC_PAD_SKIP) in dp_schedule.py."
    )

    n_pad = 0
    for gid, count in counts.items():
        short = -count % unit
        for _ in range(short):
            i = first_row[gid]
            for key in per_sample_keys:
                data[key].append(_STATIC_PAD_FILLERS[key](data, i))
        n_pad += short
    if n_pad:
        logger.info(
            f"static path: padded {n_pad} zero-loss dummy rows across {len(counts)} groups "
            f"(unit={unit}) to satisfy fixed-size micro-batch alignment."
        )
