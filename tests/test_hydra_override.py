from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from egomimic.utils.hydra_override import (
    encode_hydra_string_override,
    quote_hydra_string,
)

REPO_ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "value",
    [
        "",
        "plain checkpoint",
        "/scratch/run=name with spaces,part[4]{raw}/'single'/\"double\"",
        "internal\\backslash",
        'backslash-before-quote\\"and-trailing\\',
        "café/雪",
    ],
)
def test_string_override_round_trips_through_hydra_and_omegaconf(
    tmp_path: Path, value: str
) -> None:
    (tmp_path / "config.yaml").write_text("target: original\n", encoding="utf-8")
    token = encode_hydra_string_override("target", value)

    with initialize_config_dir(config_dir=str(tmp_path), version_base="1.3"):
        cfg = compose(config_name="config", overrides=[token])

    assert OmegaConf.select(cfg, "target") == value


def test_encoder_uses_one_double_quoted_override_token() -> None:
    value = "/scratch/checkpoint=epoch 4,step[80]{ema}.ckpt"

    assert encode_hydra_string_override("paths.ckpt", value) == (
        'paths.ckpt="/scratch/checkpoint=epoch 4,step[80]{ema}.ckpt"'
    )


@pytest.mark.parametrize(
    "value",
    [
        "${unsafe}",
        "prefix/${oc.env:HOME}/suffix",
        "nul\x00byte",
        "line\nbreak",
        "tab\tcharacter",
        "delete\x7fcharacter",
        "c1\x85control",
        "line\u2028separator",
        "paragraph\u2029separator",
    ],
)
def test_quote_rejects_interpolation_and_control_characters(value: str) -> None:
    with pytest.raises(ValueError):
        quote_hydra_string(value)


@pytest.mark.parametrize(
    "key",
    ["", "white space", "two=assignments", "group/name", "${unsafe}", "-flag"],
)
def test_encoder_rejects_non_scalar_keys(key: str) -> None:
    with pytest.raises(ValueError, match="dot-separated scalar key"):
        encode_hydra_string_override(key, "value")


def test_cli_prints_exactly_one_round_trippable_token(tmp_path: Path) -> None:
    value = 'C:\\scratch\\run=a b,[c]{d}\\"quoted"\\'
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "egomimic.utils.hydra_override",
            "++resume.ckpt_path",
            value,
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stderr == ""
    assert completed.stdout == (
        encode_hydra_string_override("++resume.ckpt_path", value) + "\n"
    )
    assert completed.stdout.count("\n") == 1


def test_cli_rejects_unsafe_interpolation_without_stdout() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "egomimic.utils.hydra_override",
            "resume.ckpt_path",
            "${oc.env:HOME}",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "must not contain interpolation" in completed.stderr
