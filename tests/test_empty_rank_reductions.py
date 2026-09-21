"""CPU unit tests for the empty-rank reductions on the training path.

``pad_static_groups`` (dp_schedule.py, 9e567d7) pads data-parallel groups with
zero-loss dummy rows, so a rank can reach the loss with no *masked* tokens at
all. ``policy_loss_function``'s per-token diagnostics were added a day earlier
(a7260ef) and took ``Tensor.max()`` on those tensors, which raises on
``numel() == 0`` — a pure metric aborting the whole train step.

These tests pin the two behaviors that fix relies on:
  * an empty diagnostic tensor reports 0 instead of raising;
  * a non-empty one is bit-identical to plain ``max()``.
"""

import pytest

torch = pytest.importorskip("torch")


def _empty_safe_max(t: "torch.Tensor") -> "torch.Tensor":
    """Mirror of the helper in ``megatron_utils/loss.py``.

    Kept as a copy because importing loss.py pulls in megatron/transformer_engine,
    which is unavailable on the CPU-only CI runner these tests target.
    """
    if t.numel() == 0:
        return torch.zeros((), dtype=t.dtype, device=t.device)
    return t.max()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_empty_reports_zero_and_keeps_dtype(dtype):
    out = _empty_safe_max(torch.empty(0, dtype=dtype))
    assert out.numel() == 1 and out.ndim == 0
    assert out.dtype is dtype
    assert out.item() == 0.0


@pytest.mark.parametrize(
    "values",
    [[1.5, -3.0, 2.0], [-7.0], [0.0, 0.0], [1e4, -1e4]],
)
def test_non_empty_matches_plain_max(values):
    t = torch.tensor(values)
    assert _empty_safe_max(t).item() == t.max().item()
    assert _empty_safe_max(t.abs()).item() == t.abs().max().item()


def test_plain_max_is_what_breaks():
    """The regression this guards against, stated as a test."""
    with pytest.raises(RuntimeError, match="Expected reduction dim"):
        torch.empty(0).max()


def test_empty_safe_max_survives_the_same_input():
    assert _empty_safe_max(torch.empty(0)).item() == 0.0
