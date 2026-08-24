import builtins

import numpy as np
import pytest

from egomimic.robot import rollout
from egomimic.robot.safety import (
    CartesianTranslationConfirmationRequired,
    validate_cartesian_command,
)


class _FakeLiveInterface:
    def __init__(self):
        self.current = {
            "left": np.asarray([0, 0, 0, 0, 0, 0, 0.5], dtype=np.float64),
            "right": np.asarray([0, 0, 0, 0, 0, 0, 0.5], dtype=np.float64),
        }
        self.validation_calls = []
        self.sent = []

    def validate_pose_command(
        self,
        pose,
        arm,
        *,
        allow_soft_translation_jump=False,
    ):
        self.validation_calls.append((arm, allow_soft_translation_jump))
        return validate_cartesian_command(
            pose,
            self.current[arm],
            max_translation_step_m=0.08,
            hard_max_translation_step_m=0.15,
            max_rotation_step_rad=0.5,
            allow_soft_translation_jump=allow_soft_translation_jump,
        )

    def set_pose(self, pose, arm, *, allow_soft_translation_jump=False):
        pose = self.validate_pose_command(
            pose,
            arm,
            allow_soft_translation_jump=allow_soft_translation_jump,
        )
        self.sent.append((arm, pose.copy(), allow_soft_translation_jump))


def _bimanual_action(*, left_x=0.0, right_x=0.0):
    action = np.zeros(14, dtype=np.float64)
    action[0] = left_x
    action[6] = 0.5
    action[7] = right_x
    action[13] = 0.5
    return action


def test_bimanual_preflight_aggregates_soft_jumps_before_any_send():
    interface = _FakeLiveInterface()
    action = _bimanual_action(left_x=0.10, right_x=0.09)

    validated, confirmations = rollout.validate_rollout_action(
        interface,
        action,
        "both",
        ["right", "left"],
        True,
    )

    np.testing.assert_array_equal(validated, action)
    assert [arm for arm, _ in confirmations] == ["right", "left"]
    assert [warning.translation_step_m for _, warning in confirmations] == [
        pytest.approx(0.09),
        pytest.approx(0.10),
    ]
    assert interface.sent == []


def test_hard_violation_pauses_without_prompting_or_sending(monkeypatch, capsys):
    interface = _FakeLiveInterface()
    action = _bimanual_action(left_x=0.15, right_x=0.09)
    prompts = []
    monkeypatch.setattr(
        rollout,
        "_confirm_cartesian_action",
        lambda *args: prompts.append(args) or True,
    )

    authorization = rollout.authorize_rollout_action(
        object(),
        interface,
        action,
        "both",
        ["right", "left"],
        True,
    )

    assert authorization is None
    assert prompts == []
    assert interface.sent == []
    assert "CARTESIAN SAFETY PAUSE" in capsys.readouterr().out


def test_explicit_confirmation_authorizes_only_one_action(monkeypatch):
    interface = _FakeLiveInterface()
    action = _bimanual_action(left_x=0.10, right_x=0.09)
    prompts = []
    monkeypatch.setattr(
        rollout,
        "_confirm_cartesian_action",
        lambda *args: prompts.append(args) or True,
    )

    first = rollout.authorize_rollout_action(
        object(), interface, action, "both", ["right", "left"], True
    )
    assert first is not None
    validated, allowed_soft_translation_arms = first
    assert allowed_soft_translation_arms == frozenset({"left", "right"})
    rollout.dispatch_rollout_action(
        interface,
        validated,
        "both",
        ["right", "left"],
        True,
        allowed_soft_translation_arms=allowed_soft_translation_arms,
    )

    assert [arm for arm, *_ in interface.sent] == ["right", "left"]
    assert all(allow for *_, allow in interface.sent)
    assert len(prompts) == 1

    second = rollout.authorize_rollout_action(
        object(), interface, action, "both", ["right", "left"], True
    )
    assert second is not None
    assert len(prompts) == 2


def test_rejecting_confirmation_returns_without_sending(monkeypatch):
    interface = _FakeLiveInterface()
    monkeypatch.setattr(rollout, "_confirm_cartesian_action", lambda *args: False)

    authorization = rollout.authorize_rollout_action(
        object(),
        interface,
        _bimanual_action(right_x=0.09),
        "both",
        ["right", "left"],
        True,
    )

    assert authorization is None
    assert interface.sent == []


def test_newly_risky_unapproved_arm_requires_its_own_confirmation(monkeypatch):
    interface = _FakeLiveInterface()
    action = _bimanual_action(left_x=0.07, right_x=0.09)
    prompted_arms = []

    def confirm(kp, confirmations):
        del kp
        prompted_arms.append([arm for arm, _ in confirmations])
        if len(prompted_arms) == 1:
            interface.current["left"][0] = -0.03
            return True
        return False

    monkeypatch.setattr(rollout, "_confirm_cartesian_action", confirm)

    authorization = rollout.authorize_rollout_action(
        object(), interface, action, "both", ["right", "left"], True
    )

    assert authorization is None
    assert prompted_arms == [["right"], ["left"]]
    assert interface.sent == []


def test_dispatch_preflights_every_arm_before_sending_any_command():
    interface = _FakeLiveInterface()
    action = _bimanual_action(left_x=0.15, right_x=0.07)

    dispatched = rollout.dispatch_rollout_action(
        interface,
        action,
        "both",
        ["right", "left"],
        True,
    )

    assert dispatched is False
    assert interface.sent == []


def test_dispatch_revalidation_hard_jump_pauses_instead_of_raising(capsys):
    class MovingInterface(_FakeLiveInterface):
        def set_pose(self, pose, arm, *, allow_soft_translation_jump=False):
            self.current[arm][0] = -0.15
            return super().set_pose(
                pose,
                arm,
                allow_soft_translation_jump=allow_soft_translation_jump,
            )

    interface = MovingInterface()

    dispatched = rollout.dispatch_rollout_action(
        interface,
        _bimanual_action(),
        "both",
        ["right", "left"],
        True,
    )

    assert dispatched is False
    assert interface.sent == []
    assert "CARTESIAN SAFETY PAUSE" in capsys.readouterr().out


def test_dispatch_reports_an_arm_sent_before_second_arm_revalidation_pause(capsys):
    class MovingLeftInterface(_FakeLiveInterface):
        def set_pose(self, pose, arm, *, allow_soft_translation_jump=False):
            if arm == "left":
                self.current[arm][0] = -0.15
            return super().set_pose(
                pose,
                arm,
                allow_soft_translation_jump=allow_soft_translation_jump,
            )

    interface = MovingLeftInterface()

    dispatched = rollout.dispatch_rollout_action(
        interface,
        _bimanual_action(),
        "both",
        ["right", "left"],
        True,
    )

    assert dispatched is False
    assert [arm for arm, *_ in interface.sent] == ["right"]
    assert "after commanding: right" in capsys.readouterr().out


def test_safe_action_dispatches_without_prompt_or_override(monkeypatch):
    interface = _FakeLiveInterface()
    prompts = []
    monkeypatch.setattr(
        rollout,
        "_confirm_cartesian_action",
        lambda *args: prompts.append(args) or True,
    )

    authorization = rollout.authorize_rollout_action(
        object(),
        interface,
        _bimanual_action(left_x=0.08, right_x=0.07),
        "both",
        ["right", "left"],
        True,
    )
    assert authorization is not None
    action, approved_arms = authorization
    assert approved_arms == frozenset()
    rollout.dispatch_rollout_action(
        interface,
        action,
        "both",
        ["right", "left"],
        True,
        allowed_soft_translation_arms=approved_arms,
    )

    assert prompts == []
    assert [arm for arm, *_ in interface.sent] == ["right", "left"]
    assert not any(allow for *_, allow in interface.sent)


def test_shadow_warmup_reports_soft_jump_without_prompt_or_send(capsys):
    interface = _FakeLiveInterface()
    interface.get_obs = lambda: {"observation": True}

    class Policy:
        def __init__(self):
            self.reset_calls = 0

        def act(self, obs):
            assert obs == {"observation": True}
            return _bimanual_action(right_x=0.09)

        def reset(self):
            self.reset_calls += 1

    policy = Policy()

    assert rollout.warmup_policy(
        interface,
        policy,
        "both",
        ["right", "left"],
        True,
    )

    assert policy.reset_calls == 1
    assert interface.sent == []
    output = capsys.readouterr().out
    assert "Shadow-only right translation 0.0900 m" in output
    assert "no command was sent" in output


def test_shadow_warmup_hard_jump_pauses_without_sending(capsys):
    interface = _FakeLiveInterface()
    interface.get_obs = lambda: {"observation": True}

    class Policy:
        def __init__(self):
            self.reset_calls = 0

        def act(self, obs):
            return _bimanual_action(right_x=0.15)

        def reset(self):
            self.reset_calls += 1

    policy = Policy()

    assert not rollout.warmup_policy(
        interface,
        policy,
        "both",
        ["right", "left"],
        True,
    )
    assert policy.reset_calls == 1
    assert interface.sent == []
    assert "CARTESIAN SAFETY PAUSE" in capsys.readouterr().out


def test_confirmation_revalidates_hard_limit_against_latest_pose(monkeypatch):
    interface = _FakeLiveInterface()
    action = _bimanual_action(right_x=0.09)

    def move_while_paused(*args):
        interface.current["right"][0] = -0.06
        return True

    monkeypatch.setattr(rollout, "_confirm_cartesian_action", move_while_paused)

    authorization = rollout.authorize_rollout_action(
        object(),
        interface,
        action,
        "both",
        ["right", "left"],
        True,
    )
    assert authorization is None
    assert interface.sent == []


@pytest.mark.parametrize("response", ["", "n", "no"])
def test_confirmation_defaults_to_no_and_restores_cbreak(monkeypatch, response):
    terminal_events = []
    kp = type("KeyPoll", (), {"fd": 7, "old": object()})()
    warning = CartesianTranslationConfirmationRequired(0.0873, 0.08, 0.15)
    monkeypatch.setattr(
        rollout.termios,
        "tcsetattr",
        lambda *args: terminal_events.append("cooked"),
    )
    monkeypatch.setattr(
        rollout.tty,
        "setcbreak",
        lambda *args: terminal_events.append("cbreak"),
    )
    monkeypatch.setattr(builtins, "input", lambda prompt: response)

    assert not rollout._confirm_cartesian_action(kp, [("right", warning)])
    assert terminal_events == ["cooked", "cbreak"]


def test_confirmation_keyboard_interrupt_restores_cbreak(monkeypatch):
    terminal_events = []
    kp = type("KeyPoll", (), {"fd": 7, "old": object()})()
    warning = CartesianTranslationConfirmationRequired(0.0873, 0.08, 0.15)
    monkeypatch.setattr(
        rollout.termios,
        "tcsetattr",
        lambda *args: terminal_events.append("cooked"),
    )
    monkeypatch.setattr(
        rollout.tty,
        "setcbreak",
        lambda *args: terminal_events.append("cbreak"),
    )

    def interrupt(prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr(builtins, "input", interrupt)

    with pytest.raises(KeyboardInterrupt):
        rollout._confirm_cartesian_action(kp, [("right", warning)])
    assert terminal_events == ["cooked", "cbreak"]


def test_replay_cursor_survives_continue_and_resets_only_with_home():
    replay = rollout.ReplayRollout.__new__(rollout.ReplayRollout)
    replay.actions = np.arange(21, dtype=np.float32).reshape(3, 7)
    replay._action_index = 0
    homes = []
    interface = type("Interface", (), {"set_home": lambda self: homes.append(True)})()

    np.testing.assert_array_equal(replay.rollout_step(0), replay.actions[0])
    np.testing.assert_array_equal(replay.rollout_step(999), replay.actions[0])
    assert replay._action_index == 0
    replay.commit_step()
    np.testing.assert_array_equal(replay.rollout_step(0), replay.actions[1])
    replay.commit_step()
    assert replay._action_index == 2

    rollout.reset_rollout(interface, replay)

    assert homes == [True]
    assert replay._action_index == 0
    np.testing.assert_array_equal(replay.rollout_step(999), replay.actions[0])


def test_restart_intervention_stays_paused_until_explicit_continue():
    commands = iter(["restart", "restart", "continue"])
    events = []

    result = rollout._run_intervention_loop(
        lambda: events.append("prompt") or next(commands),
        lambda: events.append("reset"),
        ensure_reset_before_continue=True,
    )

    assert result == "continue"
    assert events == ["prompt", "reset", "prompt", "reset", "prompt"]


def test_start_or_finished_continue_resets_once_before_motion():
    events = []

    result = rollout._run_intervention_loop(
        lambda: events.append("prompt") or "continue",
        lambda: events.append("reset"),
        ensure_reset_before_continue=True,
    )

    assert result == "continue"
    assert events == ["prompt", "reset"]


def test_confirmation_accepts_yes_and_reprompts_invalid_input(monkeypatch, capsys):
    terminal_events = []
    responses = iter(["maybe", "yes"])
    kp = type("KeyPoll", (), {"fd": 7, "old": object()})()
    warning = CartesianTranslationConfirmationRequired(0.0873, 0.08, 0.15)
    monkeypatch.setattr(
        rollout.termios,
        "tcsetattr",
        lambda *args: terminal_events.append("cooked"),
    )
    monkeypatch.setattr(
        rollout.tty,
        "setcbreak",
        lambda *args: terminal_events.append("cbreak"),
    )
    monkeypatch.setattr(builtins, "input", lambda prompt: next(responses))

    assert rollout._confirm_cartesian_action(kp, [("right", warning)])
    assert terminal_events == ["cooked", "cbreak"]
    output = capsys.readouterr().out
    assert "right: requested jump 0.0873 m" in output
    assert "automatic limit 0.0800 m" in output
    assert "hard limit 0.1500 m" in output


def test_confirmation_eof_fails_closed_and_restores_cbreak(monkeypatch):
    terminal_events = []
    kp = type("KeyPoll", (), {"fd": 7, "old": object()})()
    warning = CartesianTranslationConfirmationRequired(0.0873, 0.08, 0.15)
    monkeypatch.setattr(
        rollout.termios,
        "tcsetattr",
        lambda *args: terminal_events.append("cooked"),
    )
    monkeypatch.setattr(
        rollout.tty,
        "setcbreak",
        lambda *args: terminal_events.append("cbreak"),
    )

    def eof(prompt):
        raise EOFError

    monkeypatch.setattr(builtins, "input", eof)

    assert not rollout._confirm_cartesian_action(kp, [("right", warning)])
    assert terminal_events == ["cooked", "cbreak"]
