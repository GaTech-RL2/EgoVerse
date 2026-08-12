"""Build a routing graph from a pipeline config, for inspection and validation.

WHY INSTANTIATE RATHER THAN PARSE
---------------------------------
Every ``Stage`` declares ``self.reads`` / ``self.writes`` in its constructor,
and several stages *derive* those from defaults -- ``StreamTrunk`` and ``Apex``
both fall back to writing their input key when no output is named. A parser
that reads only the literal yaml fields therefore reports them as writing
nothing, which is exactly the bug this module replaced. Instantiating the
stages and reading the declared keys is ground truth and cannot drift from the
code.

The cost is that this needs the real environment (torch + egomimic importable);
that is the right trade for a tool whose job is to tell you whether a config is
wired correctly.

USAGE
    python config_graph.py graph.json  path/to/model/*.yaml
    python config_graph.py --html viz.html --template tpl.html graph.json  *.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, Iterable, List, Sequence

# Stage-name -> section of the pipeline. Sections are the coarse reading order
# of a config (obs -> trunk -> head); anything unlisted lands in OTHER so a new
# stage shows up as unclassified rather than being silently grouped wrong.
SECTIONS: Dict[str, str] = {}
for _sec, _names in (
    ("OBS", ("TargetBuilder", "ObsNoise", "TimePos", "VisualEncode", "ObsEmbed",
             "Fuse", "ObsEncoders", "SharedObsEncoders")),
    ("MAIN TRUNK", ("StreamTrunk", "Framewise", "Chunk", "Dechunk", "Apex", "Mix",
                    "Rename", "DualstreamTrunk")),
    ("ACTION PREDICTION", ("SDPHead", "FlowHead", "DiffusionHead", "GMMHead",
                           "RegressionHead", "RatioLoss", "ScheduledRatioLoss")),
):
    for _n in _names:
        SECTIONS[_n] = _sec

# Which stream a batch key belongs to. Order matters: the agnostic/specific
# prefixes are checked before the generic fallbacks.
_STREAM_RULES = (
    ("route", (r"^route",)),
    ("A", (r"^A(_|$)", r"^a_top$", r"^emb_object$", r"^feat_img_a$", r"^apex/")),
    ("S", (r"^S(_|$)", r"^s$", r"^emb_pusher$", r"^feat_img_s$")),
)


def stream_of(key: str) -> str:
    """Classify a batch key into A / S / route / shared."""
    for name, pats in _STREAM_RULES:
        if any(re.search(p, key) for p in pats):
            return name
    return "shared"


def depth_of(key: str) -> int:
    """Pyramid level encoded in the key name (``A_L1`` -> 1). 0 when absent."""
    m = re.search(r"_L(\d+)", key)
    return int(m.group(1)) if m else 0


def _expand(keys: Iterable[str], batch_keys: Iterable[str]) -> List[str]:
    """Resolve glob declarations like ``obs/*`` against keys seen so far.

    ``ObsNoise`` declares ``obs/*`` because it operates on whatever image keys
    the config gave it. Left unexpanded, such a node would connect to nothing.
    """
    out, pool = [], list(batch_keys)
    for k in keys:
        if k.endswith("*"):
            pre = k[:-1]
            out.extend(sorted(b for b in pool if b.startswith(pre)))
        else:
            out.append(k)
    seen, uniq = set(), []
    for k in out:
        if k not in seen:
            seen.add(k)
            uniq.append(k)
    return uniq


# Constructor arguments that are large nested blocks: keep them, but as compact
# JSON so the hover panel stays readable instead of dumping a whole sub-tree.
_NESTED = ("encoder", "main_network", "streams_cfg", "residual_mixer_kwargs",
           "mixer_kwargs", "transforms", "obs_specs", "img_encoders", "mapping")


def hyperparams(cfg_stage: Any, limit: int = 400) -> Dict[str, Any]:
    """Config-declared arguments for one stage, flattened for display."""
    out: Dict[str, Any] = {}
    for k, v in dict(cfg_stage).items():
        if k == "_target_":
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = json.dumps(_plain(v), separators=(",", ":"))[:limit]
    return out


def _plain(v: Any) -> Any:
    """OmegaConf containers -> plain python, so json can serialise them."""
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(v):
            return OmegaConf.to_container(v, resolve=False)
    except Exception:
        pass
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


def build_graph(path: str) -> Dict[str, Any]:
    """Instantiate a config's stages and return its routing graph.

    Nodes carry the stage's DECLARED reads/writes (not the yaml literals), its
    section, stream, pyramid depth and hyperparameters. Edges connect the
    most recent writer of a key to each subsequent reader -- i.e. the actual
    dataflow order, so a key read before anything writes it produces no edge
    and shows up as a dangling input.
    """
    import yaml
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    raw = yaml.safe_load(open(path))
    cfg = OmegaConf.create(raw)
    cfg_stages = list(cfg.robomimic_model.stages)

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    producer: Dict[str, int] = {}
    seen_keys: List[str] = ["cu_seqlens", "max_seq_len", "embodiment", "actions",
                            "obs/front_img_1", "obs/object_pose", "obs/pusher_pose",
                            "obs/state_agent_obj", "aux/chunker"]

    for i, cs in enumerate(cfg_stages):
        stage = instantiate(cs)
        name = str(cs["_target_"]).split(".")[-1]
        reads = _expand(list(getattr(stage, "reads", []) or []), seen_keys)
        writes = _expand(list(getattr(stage, "writes", []) or []), seen_keys)
        streams = {stream_of(k) for k in (writes or reads)} or {"shared"}
        stream = streams.pop() if len(streams) == 1 else "both"
        depth = max([depth_of(k) for k in (writes or reads)] or [0])

        nodes.append({
            "i": i, "t": name, "sec": SECTIONS.get(name, "OTHER"),
            "stream": stream, "depth": depth,
            "in": reads, "out": writes, "p": hyperparams(cs),
        })
        for k in reads:
            if k in producer:
                edges.append({"a": producer[k], "b": i, "k": k, "s": stream_of(k)})
        for k in writes:
            producer[k] = i
            if k not in seen_keys:
                seen_keys.append(k)

    return {"nodes": nodes, "edges": edges,
            "domains": _plain(cfg.robomimic_model.get("domains", [])),
            "source": os.path.basename(path)}


def lint(graph: Dict[str, Any]) -> List[str]:
    """Structural problems worth failing a config review over."""
    problems: List[str] = []
    written = {k for n in graph["nodes"] for k in n["out"]}
    ambient = {"cu_seqlens", "max_seq_len", "embodiment", "actions", "target",
               "aux/chunker", "time_pos"}
    for n in graph["nodes"]:
        if not n["out"]:
            problems.append(f"stage {n['i']} {n['t']}: declares no writes")
        for k in n["in"]:
            if k not in written and not k.startswith("obs/") and k not in ambient:
                problems.append(f"stage {n['i']} {n['t']}: reads {k!r}, never written")
    # Terminal outputs are consumed OUTSIDE the stage list -- by the lightning
    # module, the loss aggregator or an eval hook -- so an unread write here is
    # only suspicious for intermediate keys.
    TERMINAL = ("apex/", "log/", "loss/", "aux/", "chunk/")
    TERMINAL_EXACT = {"pred_action", "target", "frame_idx", "max_seq_len",
                      "time_pos", "cu_seqlens"}
    consumed = {e["k"] for e in graph["edges"]}
    for n in graph["nodes"]:
        for k in n["out"]:
            if (k not in consumed and not k.startswith(TERMINAL)
                    and k not in TERMINAL_EXACT and not k.startswith("obs/")):
                problems.append(f"stage {n['i']} {n['t']}: writes {k!r}, never read")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out_json")
    ap.add_argument("configs", nargs="+")
    ap.add_argument("--html", help="also render this standalone viz file")
    ap.add_argument("--template", help="HTML template containing __DATA__")
    ap.add_argument("--lint", action="store_true", help="print structural problems")
    a = ap.parse_args(argv)

    graphs: Dict[str, Any] = {}
    for path in a.configs:
        name = os.path.basename(path)[:-5]
        try:
            graphs[name] = build_graph(path)
        except Exception as exc:                      # keep going; report at the end
            print(f"  {name:40s} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        g = graphs[name]
        print(f"  {name:40s} {len(g['nodes']):2d} nodes {len(g['edges']):3d} edges")
        if a.lint:
            for p in lint(g):
                print(f"      ! {p}")

    json.dump(graphs, open(a.out_json, "w"), separators=(",", ":"))
    print(f"wrote {a.out_json} ({os.path.getsize(a.out_json)} bytes)")

    if a.html:
        if not a.template:
            ap.error("--html requires --template")
        tpl = open(a.template).read()
        if "__DATA__" not in tpl:
            ap.error("template has no __DATA__ placeholder")
        open(a.html, "w").write(
            tpl.replace("__DATA__", json.dumps(graphs, separators=(",", ":"))))
        print(f"wrote {a.html} ({os.path.getsize(a.html)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
