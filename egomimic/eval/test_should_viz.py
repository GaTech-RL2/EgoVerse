"""Unit tests for EvalVideo._should_viz epoch gating.

Validation runs only when (current_epoch + 1) % check_val_every_n_epoch == 0,
plus epoch 0 (val_at_start). The viz gate must align with those epochs so that
viz_every_n_epochs is honored relative to the validation cadence.
"""

from types import SimpleNamespace

from egomimic.eval.eval_video import EvalVideo


def _should_viz_at(epoch: int, viz_every_n_epochs: int) -> bool:
    fake = SimpleNamespace(
        viz_every_n_epochs=viz_every_n_epochs,
        trainer=SimpleNamespace(current_epoch=epoch),
    )
    return EvalVideo._should_viz(fake)


# Validation epochs for check_val_every_n_epoch=50: epoch 0 (val_at_start) + 49, 99, 149, 199
VAL_EPOCHS_50 = [0, 49, 99, 149, 199]


def test_viz_every_100_with_check_val_50():
    # Should viz at start (0) and every 100 epochs => 99, 199. Not at 49, 149.
    got = {e: _should_viz_at(e, 100) for e in VAL_EPOCHS_50}
    assert got == {0: True, 49: False, 99: True, 149: False, 199: True}, got


def test_viz_every_50_with_check_val_50():
    # viz_every == check_val => viz on every validation pass.
    got = {e: _should_viz_at(e, 50) for e in VAL_EPOCHS_50}
    assert all(got.values()), got


def test_viz_disabled():
    assert _should_viz_at(99, 0) is False
    assert _should_viz_at(99, -1) is False


def test_viz_every_1_always_on():
    assert all(_should_viz_at(e, 1) for e in range(5))
