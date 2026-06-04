"""CPU unit tests for slime.utils.dp_schedule.pad_static_groups.

The STATIC (fixed ``micro_batch_size``) path of ``build_dp_schedule`` can
neither split nor merge bins, so each step's micro-batch count must divide
``dp_size * mb_group`` in the DATA itself. ``pad_static_groups`` is the
convert-boundary chokepoint (``_split_train_data_by_dp``) that pads every
group with zero-loss dummy rows up to ``dp * micro_batch_size * mb_group`` —
covering the native converter and any custom convert function alike.

Production repro: qwen3.5 mobilegym dp=4 emitted a step of 415 segments →
"static path" AssertionError before this existed.
"""

from types import SimpleNamespace

import pytest

from slime.utils.dp_schedule import build_dp_schedule, pad_static_groups


def make_args(*, use_dynamic_batch_size=False, micro_batch_size=1, max_tokens_per_gpu=None):
    return SimpleNamespace(
        micro_batch_size=micro_batch_size,
        use_dynamic_batch_size=use_dynamic_batch_size,
        max_tokens_per_gpu=max_tokens_per_gpu,
        balance_data=False,
    )


def make_tp(dp_size=4, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=1):
    return {
        "dp_size": dp_size,
        "cp_size": cp_size,
        "vpp_size": vpp_size,
        "microbatch_group_size_per_vp_stage": microbatch_group_size_per_vp_stage,
    }


def make_data(group_sizes: list[int]) -> dict:
    """Native-converter-shaped train_data: one group per entry with that many
    rows; per-group raw_reward stays one entry per ROW (native convention)."""
    data = {
        "tokens": [],
        "response_lengths": [],
        "rewards": [],
        "raw_reward": [],
        "truncated": [],
        "sample_indices": [],
        "group_ids": [],
        "loss_masks": [],
        "group_mask_sums": [],
        "rollout_log_probs": [],
    }
    for gid, size in enumerate(group_sizes):
        for _ in range(size):
            data["tokens"].append([1] * 8)
            data["response_lengths"].append(4)
            data["rewards"].append(1.0)
            data["raw_reward"].append(1.0)
            data["truncated"].append(0)
            data["sample_indices"].append(gid)
            data["group_ids"].append(gid)
            data["loss_masks"].append([1] * 4)
            data["group_mask_sums"].append(4 * size)
            data["rollout_log_probs"].append([-0.1] * 4)
    return data


@pytest.mark.unit
def test_pads_each_group_to_unit_multiple():
    data = make_data([3, 5, 8])  # 3→4, 5→8, 8→8 at unit 4
    pad_static_groups(make_args(), make_tp(dp_size=4), data)

    counts: dict[int, int] = {}
    for gid in data["group_ids"]:
        counts[gid] = counts.get(gid, 0) + 1
    assert counts == {0: 4, 1: 8, 2: 8}
    # Per-sample lists stay aligned in length; raw_reward is untouched.
    n = len(data["group_ids"])
    for key in ("tokens", "loss_masks", "rewards", "group_mask_sums", "rollout_log_probs"):
        assert len(data[key]) == n
    assert len(data["raw_reward"]) == 3 + 5 + 8

    # Dummy rows: zero loss mask, group's mask total reused, 2-token sequence.
    for i in range(3 + 5 + 8, n):
        assert sum(data["loss_masks"][i]) == 0
        assert data["tokens"][i] == [0, 0]
        gid = data["group_ids"][i]
        assert data["group_mask_sums"][i] == 4 * [3, 5, 8][gid]


@pytest.mark.unit
def test_padded_data_schedules_on_static_path():
    """End-to-end with build_dp_schedule: ragged groups that previously raised
    the static-path assert must schedule cleanly after padding."""
    args = make_args()
    tp = make_tp(dp_size=4)
    data = make_data([3, 5, 6, 7])  # sums 21 → unpadded step would be 21 % 4 != 0
    pad_static_groups(args, tp, data)

    total_lengths = [len(t) for t in data["tokens"]]
    partitions, mbi, nmb, gbs = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=4, group_indices=data["group_ids"]
    )
    n = len(total_lengths)
    assert sorted(i for p in partitions for i in p) == list(range(n))
    assert nmb == [n // 4]


@pytest.mark.unit
def test_dynamic_path_is_noop():
    data = make_data([3, 5])
    before = len(data["group_ids"])
    pad_static_groups(
        make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=64), make_tp(dp_size=4), data
    )
    assert len(data["group_ids"]) == before


@pytest.mark.unit
def test_dp1_is_noop():
    data = make_data([3, 5])
    before = len(data["group_ids"])
    pad_static_groups(make_args(), make_tp(dp_size=1), data)
    assert len(data["group_ids"]) == before


@pytest.mark.unit
def test_unit_includes_micro_batch_size():
    """dp=2, mbs=2 → unit 4: group of 5 pads to 8."""
    data = make_data([5])
    pad_static_groups(make_args(micro_batch_size=2), make_tp(dp_size=2), data)
    assert len(data["group_ids"]) == 8


@pytest.mark.unit
def test_unknown_per_sample_key_raises():
    data = make_data([3])
    data["my_custom_key"] = [0.5] * 3
    with pytest.raises(AssertionError, match="my_custom_key"):
        pad_static_groups(make_args(), make_tp(dp_size=4), data)


@pytest.mark.unit
def test_production_repro_415_segments():
    """qwen3.5 mobilegym dp=4: 16 groups, 415 segments total. After padding,
    every group is a multiple of 4 and the static schedule builds."""
    sizes = [26] * 15 + [25]  # 415
    args = make_args()
    tp = make_tp(dp_size=4)
    data = make_data(sizes)
    pad_static_groups(args, tp, data)

    total_lengths = [len(t) for t in data["tokens"]]
    partitions, mbi, nmb, gbs = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=16, group_indices=data["group_ids"]
    )
    n = len(total_lengths)
    assert n % 4 == 0
    assert sorted(i for p in partitions for i in p) == list(range(n))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
