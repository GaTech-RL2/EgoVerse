"""Rollout as a PIPELINE OF NODES, mirroring the training graph.

Rollout policy -- when to query the model, how many predicted actions to
commit, and whether to blend overlapping plans -- is represented as an ordered
list of nodes over one state dictionary. PipelineAlgo's module-local Policy
assembles these generic nodes with embodiment-specific codecs.

CONTRACT (identical to pipeline.core.Stage):
    node(state: dict) -> state: dict

The rollout STATE dict is the single carrier, exactly as `batch` is in training:

    t              int      env step index
    obs            dict     raw observation from the env/robot THIS step
    obs_norm       dict     normalized obs, model-ready            (ObsAssemble)
    should_query   bool     query the policy this step?            (ObsCadence)
    chunk          (C,D)    latest predicted action chunk          (PolicyStep)
    chunk_t        int      env step `chunk` was predicted at      (PolicyStep)
    prev_chunk     (C,D)    the chunk before it (for ensembling)   (PolicyStep)
    queue          list     committed actions not yet executed     (ChunkCommit)
    action         (D,)     the action to execute THIS step        (ActionDequeue)
    command        Any      env/robot-native command               (ActionDecode)
    policy_state   dict     optional algorithm-owned persistent state

Nodes declare ``reads``/``writes`` so a missing producer is caught at build
time instead of surfacing as a None halfway through an episode on hardware.
"""

from typing import Any, Dict, List, Optional, Sequence


class RolloutNode:
    """One rollout step-policy operation. Subclasses override ``__call__``."""

    reads: Sequence[str] = ()
    writes: Sequence[str] = ()

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError

    def reset(self, state: Dict[str, Any]) -> None:
        """Per-episode reset. Default: nothing to clear."""
        return None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(reads={list(self.reads)}, "
            f"writes={list(self.writes)})"
        )


class RolloutPipeline:
    """Ordered rollout nodes + a build-time key-flow check.

    ``seed_keys`` are what the env adapter guarantees before the first node
    runs (typically ``t`` and ``obs``). validate() replays the declared
    reads/writes so an ordering mistake is a startup error, not a mid-episode
    surprise on a real robot.
    """

    def __init__(
        self, nodes: List[RolloutNode], seed_keys: Sequence[str] = ("t", "obs")
    ):
        self.nodes = list(nodes)
        self.seed_keys = list(seed_keys)
        self.validate()

    def validate(self) -> None:
        have = set(self.seed_keys) | {"policy_state", "queue"}
        problems = []
        for n in self.nodes:
            missing = [r for r in n.reads if r not in have]
            if missing:
                problems.append((type(n).__name__, missing, sorted(have)))
            have.update(n.writes)
        if problems:
            lines = [
                f"  {name}: missing {miss} (available: {avail})"
                for name, miss, avail in problems
            ]
            raise ValueError(
                "RolloutPipeline: node inputs are not produced by anything "
                "earlier in the list.\n"
                + "\n".join(lines)
                + "\nReorder the nodes, or seed the key from the env adapter."
            )

    def reset(self, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Start a fresh episode; returns the initial state."""
        state = state if state is not None else {}
        state.update(
            {
                "t": 0,
                "queue": [],
                "chunk": None,
                "chunk_t": None,
                "prev_chunk": None,
                "action": None,
                "command": None,
            }
        )
        for n in self.nodes:
            n.reset(state)
        return state

    def step(self, state: Dict[str, Any], obs: Any) -> Dict[str, Any]:
        """One ENV step: run every node in order, return the updated state.

        The env adapter reads ``state["command"]`` afterwards. Nodes that do
        not apply on a given step (e.g. PolicyStep when should_query is False)
        pass the state through untouched -- the branch lives in the node, so
        the sequence stays the same every step and is readable from config.
        """
        state["obs"] = obs
        for n in self.nodes:
            state = n(state)
        state["t"] = int(state["t"]) + 1
        return state

    def explain(self) -> str:
        have = set(self.seed_keys) | {"policy_state", "queue"}
        out = []
        for n in self.nodes:
            miss = [r for r in n.reads if r not in have]
            tag = f"   MISSING {miss}" if miss else ""
            out.append(
                "%-22s reads=%-42s writes=%s%s"
                % (type(n).__name__, str(list(n.reads)), str(list(n.writes)), tag)
            )
            have.update(n.writes)
        return "\n".join(out)
