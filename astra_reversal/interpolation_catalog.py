"""Frozen standard-demo donors and the released LIBERO-OOD oracle mapping.

This catalog describes a paper-form PI05 port, not a bitwise reproduction of
the released PI0 implementation. Only :func:`donor_catalog` is shown to Astra;
the target-to-donor mapping is reserved for the separately labelled oracle.
"""

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

PAPER_URL = "https://arxiv.org/html/2505.03500v5"
UPSTREAM_REVISION = "587a6cbf64f16c7b87fa5805dc0ed934192239a4"
UPSTREAM_URL = f"https://github.com/QuanyiLi/pi0-text-latent/tree/{UPSTREAM_REVISION}"
DATASET_REPO = "physical-intelligence/libero"
DATASET_REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"
DATASET_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/tree/{DATASET_REVISION}"
REPLAN_ACTIONS = 5
METADATA_SHA256 = {
    "info.json": "77c0da93c67960bac803a8ea25dabf380d752c26c1d91c54151121dd10e7d82f",
    "tasks.jsonl": "e72ef3cf7ca7a3abe70b3e142a0d1acf2719d2a3f27598869092cc4f1762bb99",
    "episodes.jsonl": "0479a2ced6117700e4638a1da54dccd9f60ae5f47485ea6d91390c869befcf08",
}


@dataclass(frozen=True)
class DonorTask:
    source_id: str
    dataset_task_index: int
    suite: str
    task_id: int
    prompt: str
    episode_indices: tuple[int, ...]
    all_frame_count: int
    release_frame_count: int
    episode_frame_counts: tuple[int, ...]


DONORS = (
    DonorTask(
        "10",
        10,
        "libero_goal",
        8,
        "put the bowl on the plate",
        (
            379,
            422,
            426,
            431,
            433,
            447,
            448,
            451,
            459,
            466,
            481,
            483,
            488,
            507,
            511,
            513,
            522,
            532,
            537,
            549,
        ),
        1909,
        1903,
        (
            112,
            87,
            103,
            106,
            105,
            90,
            92,
            90,
            84,
            93,
            91,
            89,
            94,
            103,
            90,
            79,
            84,
            88,
            126,
            103,
        ),
    ),
    DonorTask(
        "13",
        13,
        "libero_goal",
        6,
        "put the cream cheese in the bowl",
        (
            384,
            391,
            393,
            413,
            434,
            439,
            457,
            464,
            491,
            492,
            508,
            510,
            518,
            527,
            536,
            540,
            562,
            564,
            581,
            587,
        ),
        2070,
        2039,
        (
            112,
            135,
            92,
            107,
            106,
            96,
            105,
            90,
            96,
            86,
            103,
            113,
            109,
            103,
            85,
            107,
            136,
            99,
            95,
            95,
        ),
    ),
    DonorTask(
        "14",
        14,
        "libero_goal",
        2,
        "put the wine bottle on top of the cabinet",
        (
            385,
            389,
            396,
            397,
            404,
            417,
            419,
            423,
            427,
            430,
            432,
            436,
            454,
            468,
            470,
            475,
            476,
            479,
            480,
            482,
        ),
        2027,
        2027,
        (
            90,
            114,
            113,
            101,
            116,
            93,
            99,
            100,
            101,
            99,
            104,
            105,
            116,
            92,
            107,
            92,
            97,
            87,
            109,
            92,
        ),
    ),
    DonorTask(
        "17",
        17,
        "libero_goal",
        1,
        "put the bowl on the stove",
        (
            392,
            394,
            395,
            401,
            415,
            424,
            445,
            455,
            495,
            500,
            524,
            530,
            544,
            545,
            550,
            557,
            558,
            565,
            580,
            598,
        ),
        1977,
        1977,
        (
            103,
            111,
            104,
            104,
            94,
            91,
            99,
            95,
            92,
            92,
            102,
            106,
            102,
            98,
            99,
            112,
            91,
            92,
            91,
            99,
        ),
    ),
    DonorTask(
        "18",
        18,
        "libero_goal",
        4,
        "put the bowl on top of the cabinet",
        (
            398,
            402,
            438,
            444,
            446,
            452,
            456,
            458,
            474,
            493,
            501,
            505,
            514,
            521,
            570,
            573,
            590,
            597,
            601,
            608,
        ),
        1967,
        1967,
        (
            100,
            110,
            86,
            96,
            100,
            93,
            98,
            91,
            103,
            97,
            96,
            99,
            103,
            102,
            98,
            91,
            90,
            90,
            120,
            104,
        ),
    ),
    DonorTask(
        "32",
        32,
        "libero_spatial",
        5,
        "pick up the black bowl on the ramekin and place it on the plate",
        (
            1264,
            1265,
            1266,
            1271,
            1288,
            1313,
            1341,
            1343,
            1388,
            1396,
            1404,
            1433,
            1443,
            1462,
            1465,
            1469,
            1481,
            1482,
            1496,
            1498,
        ),
        2286,
        2185,
        (
            100,
            116,
            107,
            101,
            86,
            113,
            154,
            108,
            106,
            110,
            91,
            142,
            128,
            129,
            111,
            137,
            118,
            98,
            100,
            131,
        ),
    ),
    DonorTask(
        "35",
        35,
        "libero_spatial",
        3,
        "pick up the black bowl on the cookie box and place it on the plate",
        (
            1278,
            1279,
            1303,
            1319,
            1321,
            1324,
            1328,
            1357,
            1366,
            1367,
            1378,
            1386,
            1391,
            1405,
            1437,
            1452,
            1457,
            1463,
            1468,
            1475,
        ),
        2012,
        2012,
        (
            101,
            96,
            119,
            102,
            111,
            99,
            111,
            95,
            94,
            89,
            115,
            97,
            96,
            92,
            100,
            96,
            96,
            97,
            95,
            111,
        ),
    ),
    DonorTask(
        "36",
        36,
        "libero_spatial",
        8,
        "pick up the black bowl next to the plate and place it on the plate",
        (
            1280,
            1285,
            1293,
            1307,
            1332,
            1342,
            1346,
            1348,
            1363,
            1372,
            1387,
            1392,
            1393,
            1403,
            1409,
            1413,
            1417,
            1438,
            1451,
            1459,
        ),
        2384,
        2265,
        (
            107,
            125,
            121,
            138,
            125,
            126,
            123,
            114,
            164,
            115,
            138,
            122,
            109,
            137,
            110,
            97,
            96,
            106,
            111,
            100,
        ),
    ),
    DonorTask(
        "38",
        38,
        "libero_spatial",
        2,
        "pick up the black bowl from table center and place it on the plate",
        (
            1283,
            1287,
            1289,
            1299,
            1310,
            1315,
            1316,
            1317,
            1320,
            1345,
            1353,
            1354,
            1358,
            1377,
            1380,
            1383,
            1385,
            1426,
            1428,
            1432,
        ),
        2315,
        2253,
        (
            135,
            100,
            120,
            121,
            111,
            106,
            103,
            114,
            123,
            109,
            107,
            138,
            105,
            134,
            110,
            114,
            108,
            115,
            111,
            131,
        ),
    ),
)


@dataclass(frozen=True)
class OracleTask:
    suite: str
    task_id: int
    task_name: str
    source_a_id: str
    source_b_id: str
    lambda_calls: int
    release_lambda: int

    def metadata(self):
        return {
            **asdict(self),
            "lambda_value_source": (
                "paper_v5_section_3.3_wine_to_bowl_example"
                if self.suite == "libero_goal_ood" and self.task_id == 6
                else "released_per_task_mapping_not_specified_numerically_in_paper"
            ),
            "paper_form_port": {
                "alpha": "min(zero_based_policy_call_index / lambda_policy_calls, 1)",
                "lambda_policy_calls": self.lambda_calls,
                "first_full_second_donor_action_step": REPLAN_ACTIONS
                * self.lambda_calls,
            },
            "released_implementation": {
                "alpha": "bfloat16 linspace(0, 1, release_lambda + 2)[clamped_call_index]",
                "lambda_argument": self.release_lambda,
                "first_full_second_donor_action_step": REPLAN_ACTIONS
                * (self.release_lambda + 1),
            },
        }


_GOAL = (
    ("put_the_cream_cheese_in_the_basket", "13", "17", 24, 24),
    ("put_the_orange_juice_on_the_stove", "14", "17", 24, 24),
    ("put_the_bbq_source_on_the_plate", "13", "10", 24, 24),
    ("put_the_tomato_sauce_on_top_of_the_cabinet", "13", "18", 24, 24),
    ("put_the_wine_bottle_on_the_stove", "14", "17", 24, 24),
    ("put_the_wine_bottle_on_the_plate", "14", "10", 24, 24),
    ("put_the_wine_bottle_in_the_bowl", "14", "13", 14, 12),
    ("put_the_cream_cheese_on_the_plate", "13", "10", 24, 24),
    ("put_the_cream_cheese_on_the_stove", "13", "17", 24, 24),
    ("put_the_cream_cheese_on_top_of_the_cabinet", "13", "18", 24, 24),
)
_SPATIAL = (
    ("put_the_butter_on_the_plate", "13", "35", 24, 24),
    ("put_the_chocolate_pudding_on_the_plate", "13", "35", 24, 24),
    ("put_the_milk_on_the_plate", "14", "32", 24, 24),
    ("put_the_orange_juice_on_the_plate", "14", "32", 24, 24),
    ("put_the_bowl_on_cookie_box_on_the_stove", "35", "17", 24, 24),
    ("put_the_bowl_on_cookie_box_on_the_cabinet", "35", "18", 14, 14),
    ("put_the_bowl_next_to_the_plate_on_the_cabinet", "36", "18", 24, 24),
    ("put_the_bowl_next_to_the_plate_on_the_stove", "36", "17", 30, 30),
    ("put_the_bowl_at_table_center_on_the_cabinet", "38", "18", 24, 24),
    ("put_the_bowl_at_table_center_on_the_stove", "38", "17", 24, 24),
)
ORACLES = tuple(
    OracleTask(suite, task_id, *row)
    for suite, rows in (("libero_goal_ood", _GOAL), ("libero_spatial_ood", _SPATIAL))
    for task_id, row in enumerate(rows)
)


def donor_catalog():
    """Return only donor identities/prompts, with no oracle target information."""
    return [{"source_id": row.source_id, "prompt": row.prompt} for row in DONORS]


def donor_for(source_id):
    for row in DONORS:
        if source_id == row.source_id:
            return row
    raise ValueError(f"Unknown standard-demo source ID: {source_id!r}")


def oracle_for(suite, task_id):
    if type(task_id) is not int:
        raise ValueError("Task ID must be an integer")
    for row in ORACLES:
        if (suite, task_id) == (row.suite, row.task_id):
            return row
    raise ValueError(f"No oracle mapping for {suite!r} task {task_id}")


def paper_alpha(policy_call_index, lambda_policy_calls):
    """Paper-form schedule; time counts policy calls, each executing 5 actions."""
    if type(policy_call_index) is not int or policy_call_index < 0:
        raise ValueError("Policy call index must be a nonnegative integer")
    if type(lambda_policy_calls) is not int or lambda_policy_calls <= 0:
        raise ValueError("Lambda must be a positive integer number of policy calls")
    return min(policy_call_index / lambda_policy_calls, 1.0)


def catalog_metadata():
    return {
        "schema": "astra-interpolation-catalog-1",
        "paper": PAPER_URL,
        "released_source": UPSTREAM_URL,
        "released_mapping_path": "examples/libero/main.py:74-97,139-158",
        "released_schedule_path": "src/openpi/policies/policy.py:161-208",
        "released_extraction_path": "scripts/text_latent.py:38-80,86-145",
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "dataset_url": DATASET_URL,
        "metadata_sha256": dict(METADATA_SHA256),
        "donors": [asdict(row) for row in DONORS],
        "oracles": [row.metadata() for row in ORACLES],
        "extraction": {
            "episodes_per_source": 20,
            "episode_selection": "first 20 episodes in pinned dataset metadata order",
            "frame_selection": "all timesteps",
            "averaging": "one equally weighted contribution per observation frame",
            "source_episodes": 180,
            "source_frames": 18947,
            "released_frame_selection": "frame_index in range(120)",
            "released_source_frames": 18628,
            "ood_demonstrations": 0,
        },
        "port": {
            "model": "pi05 checkpoint; not the paper's pi0 weights",
            "instruction_only": True,
            "alignment": "left align instruction slots; truncate or zero-pad donors",
            "tli_effective_boundaries": 17,
            "tli_layer_indices": list(range(17)),
            "tli_boundary_convention": "post-block residuals 0..16 / pre-block inputs 1..17",
            "tei": "replace instruction embeddings with convex donor mixture",
            "tli": "add (1 - 2 * alpha) * (source1 - source2)",
            "released_token_selection": "first 9 text positions, hardcoded",
            "replan_actions": REPLAN_ACTIONS,
            "astra_donor_library": "nine standard tasks; oracle mapping withheld",
        },
    }


def demonstration_plan(metadata_dir):
    """Validate immutable dataset metadata and select all 180 exact episodes."""
    directory = Path(metadata_dir)
    raw = {}
    for name, expected in METADATA_SHA256.items():
        data = (directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Frozen dataset metadata checksum mismatch: {name}")
        raw[name] = data.decode("utf-8")
    info = json.loads(raw["info.json"])
    tasks = {
        row["task_index"]: row["task"]
        for row in map(json.loads, raw["tasks.jsonl"].splitlines())
    }
    episodes = list(map(json.loads, raw["episodes.jsonl"].splitlines()))
    plans = []
    for donor in DONORS:
        if tasks.get(donor.dataset_task_index) != donor.prompt:
            raise ValueError(f"Donor prompt changed for {donor.source_id}")
        selected = [row for row in episodes if row["tasks"] == [donor.prompt]][:20]
        if tuple(row["episode_index"] for row in selected) != donor.episode_indices:
            raise ValueError(f"Donor episode selection changed for {donor.source_id}")
        if sum(row["length"] for row in selected) != donor.all_frame_count:
            raise ValueError(f"Donor frame count changed for {donor.source_id}")
        if tuple(row["length"] for row in selected) != donor.episode_frame_counts:
            raise ValueError(
                f"Donor per-episode frame counts changed for {donor.source_id}"
            )
        if (
            sum(min(row["length"], 120) for row in selected)
            != donor.release_frame_count
        ):
            raise ValueError(f"Released frame count changed for {donor.source_id}")
        plans.append(
            {
                "source_id": donor.source_id,
                "prompt": donor.prompt,
                "dataset_task_index": donor.dataset_task_index,
                "frame_count": donor.all_frame_count,
                "episodes": [
                    {
                        "episode_index": row["episode_index"],
                        "frame_count": row["length"],
                        "relative_path": info["data_path"].format(
                            episode_chunk=row["episode_index"] // info["chunks_size"],
                            episode_index=row["episode_index"],
                        ),
                    }
                    for row in selected
                ],
            }
        )
    return {
        "dataset_repo": DATASET_REPO,
        "dataset_revision": DATASET_REVISION,
        "metadata_sha256": dict(METADATA_SHA256),
        "sources": plans,
    }
