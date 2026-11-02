"""Explicit synthetic IDs for composing a template that requires private splits."""


def split_overrides():
    return [
        f"data.{head}_datasets.human_bimanual.filters.episode_hashes=[fixture_{head}]"
        for head in ("train", "valid", "extra_valid", "unseen_op_valid")
    ]
