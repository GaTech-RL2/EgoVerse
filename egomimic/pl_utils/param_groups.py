"""Named-parameter group resolution for warm-start / freeze / stage-EMA.

A single source of truth for "which params are the encoder / trunk-compute /
chunker / apex / head". Experiments declare the groups ONCE as glob patterns
(fnmatch over `model.named_parameters()` names); callbacks then reference groups
by NAME. Composite groups reference other groups via a leading '@'.

    param_groups:
      encoder:   ["*.stages.1.*"]
      trunk:     ["*.stages.2.levels.0.*", "*.stages.2.levels.2.*"]
      chunker:   ["*.stages.2.levels.1.*", "*.stages.2.levels.3.*"]  # router lives here
      apex:      ["*.stages.2.levels.4.*"]
      head:      ["*.stages.3.*"]
      structure: ["@chunker", "@apex"]     # the transplant/EMA target
      fresh:     ["@encoder", "@trunk", "@head"]
"""
import fnmatch


def _leaf_patterns(name, specs, _seen=None):
    _seen = _seen or set()
    if name in _seen:
        raise ValueError(f"cyclic param-group reference at {name!r}")
    _seen = _seen | {name}
    pats = []
    for entry in specs[name]:
        if entry.startswith("@"):
            pats.extend(_leaf_patterns(entry[1:], specs, _seen))
        else:
            pats.append(entry)
    return pats


def resolve_param_groups(named_params, specs):
    """named_params: iterable of (name, param). specs: {group: [patterns|@refs]}.
    Returns {group: [(name, param), ...]} for every group in specs."""
    named = list(named_params)
    out = {}
    for g in specs:
        pats = _leaf_patterns(g, specs)
        out[g] = [(n, p) for (n, p) in named
                  if any(fnmatch.fnmatch(n, pat) for pat in pats)]
    return out


def summarize(named_params, specs):
    """Human-readable per-group param count + overlap/coverage audit."""
    groups = resolve_param_groups(named_params, specs)
    named = list(named_params)
    total = sum(p.numel() for _, p in named)
    lines, covered = [], {}
    for g, items in groups.items():
        n = sum(p.numel() for _, p in items)
        lines.append(f"  {g:10s}: {len(items):4d} tensors  {n/1e6:7.2f}M")
        for nm, _ in items:
            covered.setdefault(nm, []).append(g)
    # leaf groups only for coverage/overlap (skip composites made of @refs)
    leaf = [g for g in specs if all(e.startswith("@") is False for e in specs[g])]
    leaf_hit = set()
    for g in leaf:
        leaf_hit |= {nm for nm, _ in groups[g]}
    uncovered = [nm for (nm, _) in named if nm not in leaf_hit]
    overlap = {nm: gs for nm, gs in covered.items()
               if len([g for g in gs if g in leaf]) > 1}
    return {
        "lines": lines, "total_M": total / 1e6,
        "uncovered": uncovered, "overlap": overlap,
    }
