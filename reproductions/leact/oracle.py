"""CPU CFR+ teacher construction with a measured, nonzero solver residual."""

from __future__ import annotations

import time
from pathlib import Path

from .data import file_sha256, fingerprint, write_json, write_jsonl


def leduc_parameters(ranks: int = 6, suits: int = 2) -> dict:
    if not 2 <= ranks <= 13 or not 1 <= suits <= 4:
        raise ValueError("OpenSpiel supports 2..13 ranks and 1..4 suits here")
    return {
        "betting": "limit", "numPlayers": 2, "numRounds": 2,
        "blind": "1 1", "raiseSize": "2 4", "firstPlayer": "1 1",
        "maxRaises": "2 2", "numSuits": suits, "numRanks": ranks,
        "numHoleCards": 1, "numBoardCards": "0 1", "stack": "100 100",
    }


def build_oracle(output: Path, *, ranks=6, suits=2, iterations=2000,
                 tolerance=0.005, check_every=100) -> dict:
    import pyspiel
    from open_spiel.python import policy

    if iterations <= 0 or check_every <= 0 or not 0 < tolerance < 1:
        raise ValueError("Invalid solver iteration/tolerance settings")
    if output.exists() or output.with_suffix(".meta.json").exists():
        raise FileExistsError(f"Choose a fresh oracle output: {output}")
    started = time.monotonic()
    params = leduc_parameters(ranks, suits)
    game = pyspiel.load_game("universal_poker", params)
    tabular = policy.TabularPolicy(game)
    expected = {(6, 2): 4032, (10, 4): 47040, (13, 4): 79872}.get((ranks, suits))
    if expected is not None and len(tabular.state_lookup) != expected:
        raise ValueError(f"Game mismatch: {len(tabular.state_lookup)} infosets, expected {expected}")
    solver = pyspiel.CFRPlusSolver(game)
    history = []
    for step in range(1, iterations + 1):
        solver.evaluate_and_update_policy()
        if step % check_every == 0 or step == iterations:
            residual = pyspiel.exploitability(game, solver.average_policy())
            history.append({"iteration": step, "exploitability": residual})
            print(f"CFR+ {step}: exploitability={residual:.8f}", flush=True)
            if residual <= tolerance:
                break
    teacher = solver.average_policy()
    names = {0: "fold", 1: "call", 2: "raise"}
    records = []
    for info_state, index in sorted(tabular.state_lookup.items()):
        state = tabular.states[index]
        action_ids = state.legal_actions()
        if not set(action_ids).issubset(names):
            raise ValueError("Unexpected universal_poker action encoding")
        probabilities = teacher.action_probabilities(state)
        records.append({
            "state_id": fingerprint([str(game), info_state]),
            "info_state": info_state,
            "player": state.current_player(),
            "legal_actions": [names[a] for a in action_ids],
            "action_ids": {names[a]: a for a in action_ids},
            "expert_policy": {names[a]: probabilities[a] for a in action_ids},
        })
    write_jsonl(output, records)
    metadata = {
        "schema_version": 1, "source": "independent OpenSpiel CFR+ reproduction",
        "paper": "https://arxiv.org/abs/2607.21856v1",
        "open_spiel_version": pyspiel.__version__, "game_string": str(game),
        "parameters": params, "num_infosets": len(records),
        "iterations": step, "target_exploitability": tolerance,
        "exploitability": residual, "converged": residual <= tolerance,
        "history": history, "wall_seconds": time.monotonic() - started,
        "sha256": file_sha256(output),
        "rules": (
            f"Two-player limit Leduc with {ranks} ranks and {suits} suits. "
            "Each player antes one chip and receives one private card. "
            "One public card is dealt after the first betting round. "
            "A pair with the public card beats an unpaired hand; otherwise the higher "
            "private rank wins. Suits do not break ties. Ranks ascend 2 through A. "
            "The bet size is two chips in round zero and four in round one, with "
            "at most two raises per round. Player zero acts first in each round. "
            "The call action checks when nothing is owed; raise also opens betting. "
            "Sequence symbols c, r, and f denote call/check, raise, and fold."
        ),
    }
    write_json(output.with_suffix(".meta.json"), metadata)
    return metadata
