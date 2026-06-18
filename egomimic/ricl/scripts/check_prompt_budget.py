"""CPU token-budget check for the eva RICL prompt (mirrors DROID's stage_cpu).

The k retrieved (state, action) blocks are discretized into the prompt *text*. They
are kept at the full 32-D openpi layout (data/cotrain_pi_ricl.yaml `state_dim: 32`,
`action_dim: 32`) so the bins stay in pi0.5's pretrained distribution — which makes
the prompt long: at k=4 the worst case is ~1150 tokens, so model/pi0.5_ricl.yaml
sets `max_token_len: 1280`. If a prompt exceeds it, the PaliGemma collate truncates
from the right and drops the trailing `;\\nAction:` anchor — so the budget must
always exceed the worst case. This builds the WORST-CASE eva prompt the same way
`PIRicl._build_prompts` + `build_tokenized_collate` do, tokenizes it with the real
tokenizer, and checks it fits. The defaults below mirror those configs, so a no-arg
run validates exactly what ships (non-zero exit = a config that would truncate).

Worst case = every retrieved/state bin at the top id (`num_bins - 1`, the widest,
3-digit token) so the count is a genuine upper bound for the configured dims.

CPU-only / login-node-safe: needs only the tokenizer (egomimic/ricl/pg_tokenizer)
and the import-light `egomimic.ricl.conditioning`. No model, no zarr, no cache.

Run:
  source emimic/bin/activate            # transformers only; conditioning is light
  python -m egomimic.ricl.scripts.check_prompt_budget                # validates the shipped config
  python -m egomimic.ricl.scripts.check_prompt_budget --k 8          # what a bigger k would cost
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from egomimic.ricl import conditioning as C

RICL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOKENIZER_DIR = os.path.join(RICL_DIR, "pg_tokenizer")


def _worst_case_block(k, state_dim, action_dim, action_steps, num_bins, prompt) -> str:
    """k retrieved demos with every bin at the top id (widest token => upper bound)."""
    states = np.ones((k, state_dim), dtype=np.float32)  # 1.0 -> bin num_bins-1
    actions = np.ones((k, action_steps, action_dim), dtype=np.float32)
    valid = np.ones(k, dtype=bool)
    return C.build_retrieved_prompt_block(
        states,
        actions,
        valid,
        prompt=prompt,
        num_bins=num_bins,
        action_steps=action_steps,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # Defaults mirror data/cotrain_pi_ricl.yaml + model/pi0.5_ricl.yaml so a no-arg
    # run validates the shipped config.
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--state-dim", type=int, default=32, help="retrieved state width")
    p.add_argument("--action-dim", type=int, default=32, help="retrieved action width")
    p.add_argument("--action-steps", type=int, default=1)
    p.add_argument("--num-bins", type=int, default=256)
    p.add_argument(
        "--query-state-dim",
        type=int,
        default=14,
        help="eva observations.state.ee_pose width (bimanual xyzypr+gripper = 14)",
    )
    p.add_argument("--max-token-len", type=int, default=1280)
    p.add_argument(
        "--task",
        default="pick up the blue marker and place it in the tray on the right",
        help="a representative (long-ish) task string",
    )
    p.add_argument("--tokenizer", default=TOKENIZER_DIR)
    args = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    block = _worst_case_block(
        args.k,
        args.state_dim,
        args.action_dim,
        args.action_steps,
        args.num_bins,
        args.task,
    )
    # Base prompt mirrors build_tokenized_collate: Task, Embodiment, State + anchor.
    query_state = " ".join([str(args.num_bins - 1)] * args.query_state_dim)
    base = (
        f"Task: {args.task}, Embodiment: eva bimanual, "
        f"State: {query_state}" + C.PI05_ANCHOR
    )
    spliced = C.splice_retrieved_into_prompt(base, block)
    assert spliced.endswith(C.PI05_ANCHOR), "anchor must stay terminal"

    n = len(tok(spliced)["input_ids"])
    est = C.estimate_prompt_tokens(
        args.k, args.action_steps, args.state_dim, args.action_dim
    )
    print(
        f"[budget] k={args.k} state_dim={args.state_dim} action_dim={args.action_dim} "
        f"action_steps={args.action_steps} query_state_dim={args.query_state_dim} "
        f"num_bins={args.num_bins}"
    )
    print(f"[budget] heuristic estimate_prompt_tokens = {est}")
    print(
        f"[budget] worst-case tokenized prompt = {n} tokens "
        f"(max_token_len={args.max_token_len})"
    )
    if n < args.max_token_len:
        print(
            f"[budget] OK: {n} < {args.max_token_len} (margin {args.max_token_len - n})"
        )
    else:
        print(
            f"[budget] OVER BUDGET: {n} >= {args.max_token_len} -> bump "
            f"model.max_token_len / tokenizer_max_length, or compact retrieved dims"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
