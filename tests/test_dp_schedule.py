"""CPU unit tests for slime.utils.dp_schedule.build_dp_schedule.

The tests assert the invariants documented at the top of dp_schedule.py against
a range of static / dynamic / VPP / oversize / balance / uneven scenarios.
"""

from types import SimpleNamespace

import pytest

from slime.utils.dp_schedule import build_dp_schedule


NUM_GPUS = 0


def make_args(
    *,
    micro_batch_size=1,
    use_dynamic_batch_size=False,
    max_tokens_per_gpu=None,
    balance_data=False,
):
    return SimpleNamespace(
        micro_batch_size=micro_batch_size,
        use_dynamic_batch_size=use_dynamic_batch_size,
        max_tokens_per_gpu=max_tokens_per_gpu,
        balance_data=balance_data,
    )


def make_tp(dp_size=1, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=1):
    return {
        "dp_size": dp_size,
        "cp_size": cp_size,
        "vpp_size": vpp_size,
        "microbatch_group_size_per_vp_stage": microbatch_group_size_per_vp_stage,
    }


def assert_invariants(
    partitions,
    micro_batch_indices,
    num_microbatches,
    *,
    dp_size,
    expected_global_sample_indices,
    total_lengths,
    max_per_bin=None,
    allowed_merged_overshoot=0,
):
    """Check the invariants documented at the top of dp_schedule.py.

    ``expected_global_sample_indices`` is the set of global sample indices
    that should end up covered (after trim). Trailing groups that don't fit
    are excluded.
    """
    seen_global: set[int] = set()
    for r in range(dp_size):
        partition = partitions[r]
        mbi = micro_batch_indices[r]

        # Same num_mbs per rank (PP sync).
        assert len(mbi) == sum(num_microbatches), f"rank {r}: mbs count mismatch"

        # Flattened micro_batch_indices == range(len(partition)).
        flat = [i for mbs in mbi for i in mbs]
        assert flat == list(range(len(partition))), f"rank {r}: micro_batch_indices don't tile [0, n)"

        # Disjoint partitions whose union covers every kept sample.
        assert seen_global.isdisjoint(partition), f"rank {r}: overlap with other ranks"
        seen_global.update(partition)
    assert seen_global == set(expected_global_sample_indices), "covered sample set mismatch"

    if max_per_bin is None:
        return

    # Every mbs <= max_per_bin tokens, EXCEPT singleton bins holding an
    # oversized sample and up to ``allowed_merged_overshoot`` multi-sample bins
    # produced by the merge-down alignment fallback.
    overshoot_multi = 0
    for r in range(dp_size):
        partition = partitions[r]
        for mbs in micro_batch_indices[r]:
            bin_total = sum(total_lengths[partition[i]] for i in mbs)
            if bin_total > max_per_bin and len(mbs) > 1:
                overshoot_multi += 1
    assert overshoot_multi <= allowed_merged_overshoot, (
        f"{overshoot_multi} multi-sample mbs exceed max_per_bin "
        f"(allowed: {allowed_merged_overshoot})"
    )


@pytest.mark.unit
def test_static_stride_single_step():
    """Static + strided DP split, single step (1 rollout = 1 sample)."""
    total_lengths = [10] * 16
    group_indices = list(range(16))
    args = make_args(micro_batch_size=2)
    tp = make_tp(dp_size=4)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=16, group_indices=group_indices
    )

    assert nmb == [2]
    assert gbs_per_step == [16]
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=4,
        expected_global_sample_indices=range(16),
        total_lengths=total_lengths,
    )


@pytest.mark.unit
def test_static_balance_multi_step():
    """Static + balance_data + 2 training steps."""
    total_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1]
    group_indices = list(range(16))
    args = make_args(micro_batch_size=2, balance_data=True)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=8, group_indices=group_indices
    )

    assert nmb == [2, 2]
    assert gbs_per_step == [8, 8]
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        expected_global_sample_indices=range(16),
        total_lengths=total_lengths,
    )


@pytest.mark.unit
def test_dynamic_uniform():
    """Dynamic mbs on uniform-length samples."""
    total_lengths = [5] * 8
    group_indices = list(range(8))
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=8, group_indices=group_indices
    )

    assert gbs_per_step == [8]
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        expected_global_sample_indices=range(8),
        total_lengths=total_lengths,
        max_per_bin=10,
    )


@pytest.mark.unit
def test_dynamic_oversized_sample_lands_alone():
    """A sample larger than max_per_bin must end up alone in its mbs."""
    total_lengths = [15, 3, 3, 3, 3, 3, 3, 3]
    group_indices = list(range(8))
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=10)
    tp = make_tp(dp_size=2)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=8, group_indices=group_indices
    )

    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        expected_global_sample_indices=range(8),
        total_lengths=total_lengths,
        max_per_bin=10,
    )
    oversize_idx = total_lengths.index(15)
    found = False
    for r in range(2):
        if oversize_idx not in partitions[r]:
            continue
        local = partitions[r].index(oversize_idx)
        for mbs in mbi[r]:
            if local in mbs:
                assert mbs == [local], f"oversized sample shares an mbs: {mbs}"
                found = True
    assert found


@pytest.mark.unit
def test_dynamic_with_vpp_rounds_to_mb_group():
    """num_microbatches per rank should be a multiple of mb_group when vpp_size > 1."""
    total_lengths = [4] * 32
    group_indices = list(range(32))
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=8)
    tp = make_tp(dp_size=2, vpp_size=2, microbatch_group_size_per_vp_stage=2)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=16, group_indices=group_indices
    )

    for n in nmb:
        assert n % 2 == 0, f"num_microbatches {n} is not a multiple of mb_group=2"
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=2,
        expected_global_sample_indices=range(32),
        total_lengths=total_lengths,
        max_per_bin=8,
    )


@pytest.mark.unit
def test_grouping_keeps_samples_together():
    """compact / subagent simulation: group 0 emits 3 samples, group 1 emits 2,
    group 2 emits 4. Splitter keeps every group's samples in a single step."""
    group_indices = [0, 0, 0, 1, 1, 2, 2, 2, 2]
    total_lengths = [3] * 9
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=12)
    tp = make_tp(dp_size=1)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=1, group_indices=group_indices
    )

    # 3 groups / 1 per step -> 3 steps, gbs constant.
    assert gbs_per_step == [1, 1, 1]
    # For each step, collect the samples (global indices) that landed in that step's mbs
    # on rank 0, then verify they exactly equal the rollout's sample positions.
    expected_per_step = [[0, 1, 2], [3, 4], [5, 6, 7, 8]]
    rank0_partition = partitions[0]
    mbs_cursor = 0
    for step_i, n_mbs in enumerate(nmb):
        step_locals = sorted(j for mbs in mbi[0][mbs_cursor : mbs_cursor + n_mbs] for j in mbs)
        step_globals = [rank0_partition[j] for j in step_locals]
        assert (
            sorted(step_globals) == expected_per_step[step_i]
        ), f"step {step_i} samples = {step_globals}, expected {expected_per_step[step_i]}"
        mbs_cursor += n_mbs
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=1,
        expected_global_sample_indices=range(9),
        total_lengths=total_lengths,
        max_per_bin=12,
    )


@pytest.mark.unit
def test_trims_trailing_groups_that_dont_fill_a_step():
    """5 groups, gbs=2 -> 2 steps x 2 groups; trailing group 4 (sample positions 6, 7)
    is dropped."""
    group_indices = [0, 0, 1, 2, 2, 3, 4, 4]
    total_lengths = [3] * 8
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=12)
    tp = make_tp(dp_size=1)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=2, group_indices=group_indices
    )

    assert gbs_per_step == [2, 2]
    # Sample positions 6 and 7 belong to the trimmed rollout 4 and must be absent.
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=1,
        expected_global_sample_indices=range(6),
        total_lengths=total_lengths,
        max_per_bin=12,
    )


@pytest.mark.unit
def test_dynamic_singleton_bins_merge_down_to_align():
    """Splitting-saturated fallback: max_per_bin < 2x sample length packs 1
    sample/bin, so 11 bins cannot split UP to 12 (dp=4); the scheduler must
    merge DOWN to 8 instead of asserting."""
    total_lengths = [1900] * 11
    group_indices = list(range(11))
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=2048)
    tp = make_tp(dp_size=4)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=11, group_indices=group_indices
    )

    assert nmb == [2]  # 8 mbs / 4 ranks
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=4,
        expected_global_sample_indices=range(11),
        total_lengths=total_lengths,
        max_per_bin=2048,
        allowed_merged_overshoot=3,  # 11 -> 8 needs 3 merges
    )


@pytest.mark.unit
def test_dynamic_partial_split_then_merge_down():
    """Mixed initial bins: expand splits the multi-sample bins first, SATURATES
    below the up-target (every bin now a singleton), then the merge fallback
    rounds down. 2 small samples pack into 1 bin + 9 singletons = 10 bins;
    up-target 12 is unreachable (max 11 after splitting the pair), so expect
    merge down to 8 with all 11 samples kept."""
    total_lengths = [500, 500] + [1900] * 9
    group_indices = list(range(11))
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=2048)
    tp = make_tp(dp_size=4)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=11, group_indices=group_indices
    )

    assert nmb == [2]  # 11 (post-split) -> merge down to 8 -> 2 per rank
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=4,
        expected_global_sample_indices=range(11),
        total_lengths=total_lengths,
        max_per_bin=2048,
        allowed_merged_overshoot=3,
    )


@pytest.mark.unit
def test_dynamic_merge_only_in_unaligned_steps():
    """Multi-step rollout where step 0 aligns by itself (8 segments) and step 1
    needs the merge fallback (7 segments -> 4 bins): per-step counts must be
    independent and every sample kept."""
    # 2 steps x 2 groups: groups 0,1 have 4 segments each (step 0: 8 = aligned);
    # groups 2,3 have 4 + 3 (step 1: 7 -> merge down to 4).
    group_indices = [0] * 4 + [1] * 4 + [2] * 4 + [3] * 3
    n = len(group_indices)
    total_lengths = [1900] * n
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=2048)
    tp = make_tp(dp_size=4)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=2, group_indices=group_indices
    )

    assert gbs_per_step == [2, 2]
    assert nmb == [2, 1]  # step 0: 8 bins untouched; step 1: 7 -> 4
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=4,
        expected_global_sample_indices=range(n),
        total_lengths=total_lengths,
        max_per_bin=2048,
        allowed_merged_overshoot=3,  # step 1: 7 -> 4 needs 3 merges
    )


@pytest.mark.unit
def test_dynamic_merge_down_production_repro():
    """Regression for the android_world dp=4 crash: 203 ~1.9k-token segments
    at max_tokens_per_gpu=2048 -> 203 singleton bins, up-target 204 unreachable.
    Expect merge down to 200 (50 mbs/rank), all samples kept, <= 3 merged bins."""
    n = 203
    total_lengths = [1900] * n
    group_indices = list(range(n))
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=2048)
    tp = make_tp(dp_size=4)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=n, group_indices=group_indices
    )

    assert nmb == [50]
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=4,
        expected_global_sample_indices=range(n),
        total_lengths=total_lengths,
        max_per_bin=2048,
        allowed_merged_overshoot=3,
    )


@pytest.mark.unit
def test_dynamic_splittable_bins_still_split_not_merge():
    """When bins hold 2 samples (max_per_bin >= 2x sample length), the up-split
    path must still win: no bin may exceed the cap."""
    total_lengths = [1900] * 10
    group_indices = list(range(10))
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=4096)
    tp = make_tp(dp_size=4)

    partitions, mbi, nmb, gbs_per_step = build_dp_schedule(
        args, tp, total_lengths, global_batch_size=10, group_indices=group_indices
    )

    assert nmb == [2]  # 5 bins split up to 8 -> 2 per rank
    assert_invariants(
        partitions,
        mbi,
        nmb,
        dp_size=4,
        expected_global_sample_indices=range(10),
        total_lengths=total_lengths,
        max_per_bin=4096,
        allowed_merged_overshoot=0,
    )


@pytest.mark.unit
def test_dynamic_merge_down_below_align_asserts():
    """vpp edge: align_to = dp * mb_group = 4 but only 3 singleton bins ->
    neither splitting up nor merging down can align; expect a loud error."""
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=2048)
    tp = make_tp(dp_size=2, vpp_size=2, microbatch_group_size_per_vp_stage=2)
    with pytest.raises(AssertionError, match="neither splitting up nor merging down"):
        build_dp_schedule(
            args, tp, [1900] * 3, global_batch_size=3, group_indices=[0, 1, 2]
        )


@pytest.mark.unit
def test_rejects_when_fewer_groups_than_gbs():
    """gbs=4 with only 3 distinct groups -> cannot form one step."""
    args = make_args(use_dynamic_batch_size=True, max_tokens_per_gpu=12)
    tp = make_tp(dp_size=1)
    with pytest.raises(AssertionError, match="num_groups"):
        build_dp_schedule(args, tp, [3] * 6, global_batch_size=4, group_indices=[0, 0, 1, 1, 2, 2])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
