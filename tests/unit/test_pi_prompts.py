"""PI._build_prompts warns when samples fall back to default_prompt, loudly when
that prompt is empty, instead of silently training on it."""

import logging

import pytest

pi_mod = pytest.importorskip("egomimic.algo.pi")


def _pi(annotation_key="annotations", default_prompt=""):
    pi = object.__new__(pi_mod.PI)
    pi.annotation_key = annotation_key
    pi.default_prompt = default_prompt
    pi.sampling_mode = "first"
    pi.proprio_in_prompt = False
    pi.embodiment_label = False
    pi.control_mode = None
    pi._empty_prompt_warned = set()
    return pi


def _fallback_warnings(caplog):
    return [r for r in caplog.records if "PI prompt fallback" in r.getMessage()]


def test_missing_annotation_key_warns_once_per_embodiment(caplog):
    pi = _pi()
    with caplog.at_level(logging.WARNING, logger=pi_mod.__name__):
        for _ in range(3):
            assert pi._build_prompts({}, "eva_bimanual", 2) == ["", ""]
        pi._build_prompts({}, "aria_bimanual", 2)
    warnings = _fallback_warnings(caplog)
    assert len(warnings) == 2
    assert "not in the batch" in warnings[0].getMessage()
    assert "EMPTY prompt" in warnings[0].getMessage()


def test_empty_annotation_sample_warns(caplog):
    pi = _pi()
    with caplog.at_level(logging.WARNING, logger=pi_mod.__name__):
        prompts = pi._build_prompts({"annotations": [["pick"], []]}, "eva", 2)
    assert prompts == ["pick", ""]
    assert len(_fallback_warnings(caplog)) == 1


def test_annotated_batch_does_not_warn(caplog):
    pi = _pi()
    with caplog.at_level(logging.WARNING, logger=pi_mod.__name__):
        pi._build_prompts({"annotations": [["pick"], ["place"]]}, "eva", 2)
    assert _fallback_warnings(caplog) == []


def test_no_annotation_key_with_explicit_prompt_does_not_warn(caplog):
    pi = _pi(annotation_key=None, default_prompt="fold the towel")
    with caplog.at_level(logging.WARNING, logger=pi_mod.__name__):
        assert pi._build_prompts({}, "eva", 1) == ["fold the towel"]
    assert _fallback_warnings(caplog) == []
