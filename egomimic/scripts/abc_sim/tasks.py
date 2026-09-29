"""The ABC tasks the sim rollout pipeline runs, by short name.

``dataset``: the sim_224 ``task_name`` (convert_sim_to_zarr --task);
``sim``: the abc_sim task spec (the rollout env); ``real``: the matching REAL
ABC task on the Phoenix mirror (data=abc_real_joints data.abc_task=...), or
None. The first three are the tasks the ABC paper measured sim-real
correlation on (Pearson r = 0.85 success / 0.91 progress, 12 checkpoints).
The short name is also the zarr folder name under ABC_SIM_ZARR_ROOT.
"""

from __future__ import annotations

TASKS: dict[str, dict] = {
    "put_bottles": {
        "dataset": "sim_put_the_plastic_bottles_in_the_bin",
        "sim": "put_plastic_bottles_in_bin",
        "real": "put the plastic bottles in the bin",
    },
    "dishrack": {
        "dataset": "sim_load_the_plates_into_the_dish_rack",
        "sim": "load_plates_into_dish_rack",
        # real ABC loads mixed dishes; the sim rack holds plates only
        "real": "load the mixed dishes into the dish rack",
    },
    "turn_mug": {
        "dataset": "sim_turn_the_mug_right_side_up",
        "sim": "turn_mug_right_side_up",
        "real": "turn the mug right side up",
    },
    "hang_mug": {
        "dataset": "sim_hang_the_mug_on_the_mug_rack",
        "sim": "hang_mug_on_mug_rack",
        "real": None,
    },
    "pour": {
        "dataset": "sim_pouring_beads",
        "sim": "pouring",
        "real": None,
    },
    "sweep": {
        "dataset": "sim_sweep_away_paper_scraps_from_the_table",
        "sim": "sweep_away_paper_scraps_from_table",
        "real": "sweep away the paper scraps from the table",
    },
    "inhand_transfer": {
        "dataset": "sim_inhand_transfer_the_item_to_other_side",
        "sim": "inhand_transfer_item_to_other_side",
        "real": None,
    },
}


def task(short_name: str) -> dict:
    try:
        return TASKS[short_name]
    except KeyError:
        raise KeyError(
            f"unknown ABC task '{short_name}'; known: {', '.join(sorted(TASKS))}"
        ) from None


if __name__ == "__main__":
    import sys

    # `python -m egomimic.scripts.abc_sim.tasks <short> <field>` for shell use.
    print(task(sys.argv[1])[sys.argv[2]])
