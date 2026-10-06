# This file is interpreted by the bounded harness language, never imported.
# ruff: noqa: F821
def harness(observation, history, cards, memory):
    due = observation["action"] - get(memory, "last_request", -40) >= 40
    retrieved = rank(cards, observation["original_goal"])[:6]
    identifiers = []
    for card in retrieved:
        identifiers = append(identifiers, card["skill_id"])
    if due:
        memory["last_request"] = observation["action"]
    return {
        "request": due,
        "card_ids": identifiers,
        "history_ids": [],
        "instruction": "Use the current live scene to decide whether to set, keep or clear a policy program. Preserve useful native behavior.",
        "memory": memory,
    }
