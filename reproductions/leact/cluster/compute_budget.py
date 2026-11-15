"""Shared reservation cap for the authorized LeAct study workflows."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]


def node_limit():
    path = BASE / 'artifacts/embodied-first-exam/compute-authorization.json'
    return int(json.loads(path.read_text())['max_concurrent_nodes_total']) if path.exists() else 2


def reserved_nodes(workflows):
    # Current jobs each request one node. Multinode launchers must register
    # their workflow while holding builderbench-submission.lock.
    path = BASE / 'artifacts/embodied-first-exam/node-reservations.json'
    reservations = json.loads(path.read_text()) if path.exists() else {}
    return sum(int(reservations.get(w['name'], 1)) for w in workflows)


def require_capacity(workflows, additional_nodes=1):
    if reserved_nodes(workflows) + additional_nodes > node_limit():
        raise RuntimeError(f'Authorized {node_limit()}-node study budget is already reserved')
