"""Map an OLD-code prdec ckpt into the BATCHFLOW module tree.

Template = a batchflow smoke ckpt (provides the new key/shape table AND the
correct hyper_parameters/config_tree). Output = template with state_dict
values replaced by the mapped old weights. 100% new-key coverage required.
"""
import sys, re, torch

old_p, tpl_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
old = torch.load(old_p, map_location="cpu", weights_only=False)
tpl = torch.load(tpl_p, map_location="cpu", weights_only=False)
osd, tsd = old["state_dict"], tpl["state_dict"]

def old_to_new(k):
    if not k.startswith("nets.outer_stage."):
        return None
    r = k[len("nets.outer_stage."):]
    # skip nested duplicate registrations (stage.inner_stage chains)
    if r.startswith("inner_stage.stages.") and ".inner_stage." in r[len("inner_stage.stages.0"):]:
        return None
    if r.startswith("agnostic_input."):
        return "nets.policy.stages.1.agnostic." + r[len("agnostic_input."):]
    if r.startswith("input_modules.0."):
        return "nets.policy.stages.1.specific.0." + r[len("input_modules.0."):]
    m = re.match(r"inner_stage\.stages\.(\d)\.(.*)", r)
    if m:
        return f"nets.policy.stages.2.levels.{m.group(1)}." + m.group(2)
    if r.startswith("action_out."):
        h = r[len("action_out."):]
        # _ModeHead trunk.{i} -> Sequential index i; proj -> final index
        h = re.sub(r"^A_head\.trunk\.(\d+)\.", lambda m: f"A_head.{m.group(1)}.", h)
        h = re.sub(r"^A_head\.proj\.", "A_head.6.", h)
        h = re.sub(r"^S_head\.table\.([^.]+)\.trunk\.(\d+)\.", lambda m: f"S_head.{m.group(1)}.{m.group(2)}.", h)
        h = re.sub(r"^S_head\.table\.([^.]+)\.proj\.", lambda m: f"S_head.{m.group(1)}.6.", h)
        h = re.sub(r"^gate\.table\.([^.]+)\.trunk\.(\d+)\.", lambda m: f"gate.{m.group(1)}.{m.group(2)}.", h)
        h = re.sub(r"^gate\.table\.([^.]+)\.proj\.", lambda m: f"gate.{m.group(1)}.6.", h)
        h = re.sub(r"^gate\.table\.([^.]+)\.net\.", lambda m: f"gate.{m.group(1)}.", h)
        return "nets.policy.stages.3." + h
    return None

mapped, misses, shape_bad = {}, [], []
for ok, v in osd.items():
    nk = old_to_new(ok)
    if nk is None:
        continue
    if nk in tsd:
        if tsd[nk].shape == v.shape:
            mapped[nk] = v
        else:
            shape_bad.append((ok, nk, tuple(v.shape), tuple(tsd[nk].shape)))
    else:
        misses.append((ok, nk))

# canonical new keys only: templates built pre-fix carry nested duplicates
# (levels.{i}.inner.*) — drop them from the requirement AND the output.
dupes = [k for k in tsd if ".levels." in k and ".inner." in k]
for k in dupes:
    del tsd[k]
print(f"dropped {len(dupes)} duplicate template keys (pre-fix registration)")
uncovered = [k for k in tsd if k not in mapped]
print(f"new keys total={len(tsd)} mapped={len(mapped)} uncovered={len(uncovered)}")
print(f"old->new misses={len(misses)} shape_mismatch={len(shape_bad)}")
for x in misses[:15]: print("  MISS:", x)
for x in shape_bad[:15]: print("  SHAPE:", x)
for x in uncovered[:25]: print("  UNCOVERED:", x)
if uncovered or shape_bad:
    print("MAPPING INCOMPLETE — not writing output")
    sys.exit(1)
tsd.update(mapped)
tpl["state_dict"] = tsd
torch.save(tpl, out_p)
print("WROTE", out_p)
