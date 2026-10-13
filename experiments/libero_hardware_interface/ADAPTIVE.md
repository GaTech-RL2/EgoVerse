# F-adaptive: Astra evolves its hardware interface

F-adaptive starts with F's typed hardware interface and the same Astra model.
During an episode, Astra can extend the interface, test its hypotheses through
robot actions, revise its tools, and revert an earlier interface version.

The initial comparison is a separate **F versus F-adaptive** cohort: the same
10 LIBERO-10 tasks × five official initial states × two methods = **100 trials**.
The original F/B0/B cohort continues unchanged. The model, native action limits,
1,000-step horizon, image resolution, initial-state protocol and private success
evaluator are shared. There are no experiment token, time or call caps.

## What Astra can change

| Operation | Behavior |
| --- | --- |
| `interface_catalog` | Discover available optional hardware information and current tool versions. |
| `enable_capability` / `read_capability` | Enable additional native robot encoders when available, or request camera calibration. |
| `define_interface_tool` | Create or replace a reusable Python tool with a description and a declared read-only or control role. |
| `call_interface_tool` | Run that version with JSON parameters in isolated scratch. Tools can process readings, implement coordinate conversions, action mappings, feedback control or identification procedures. |
| `probe_dynamics` | Apply a bounded native command and record numeric robot sensors before and after. All executed steps count toward the episode horizon. |
| `revise_interface_notes` | Retain measured response models, conventions, corrections or hypotheses in the actor's current interface context. |
| `restore_interface` | Restore previous capabilities, tools and notes. Physical motion and usage remain spent. |

Programs receive `params`, `read(channel, max_age_ms)`, `act(envelope)` and
`read_capability(name)`, and return JSON through `result`. Python runs in the
existing isolated standard-library scratch environment. Read-only tools cannot
dispatch actions. Every control program still reaches the same bounds,
freshness, repeated-step and horizon checks as ordinary F.

For example, Astra can first probe a small x command, estimate the observed
end-effector response per step, store that estimate, then define a tool that
converts a requested displacement into a bounded command. That tool can later
be revised when contact or robot configuration changes the observed response.
The implementation does not prescribe that procedure or supply a task solution.

Additional sensors are explicitly limited to native robot proprioception.
Camera information contains ideal intrinsics and current camera extrinsics,
with coordinate conventions and provenance. This is extra calibration
information, not a measurement learned by Astra. Task labels, object-state
aggregates, success predicates, demonstrations and arbitrary simulator queries
are unavailable. The native controller and evaluator are outside the editable
interface. MAX is not implicitly given raw sensor access that bypasses its
benchmark disturbances.

This first variant combines interface evolution **and** optional additional
information. An improvement cannot by itself isolate the benefit of code
evolution from the benefit of extra calibration or encoders.

## Evidence and reset behavior

Each episode starts with an empty extension: no tools or calibration knowledge
transfer from previous trials. Changes remain reusable throughout that episode.
Every revision saves full code, its hash, enabled capabilities, notes, reason and
native step under `interface-evolution/`. The event stream retains tool calls,
sensor results, probe measurements and revisions. Outcome records include
calibration steps, probe count, tool executions and the final interface hash.
The existing independent replay checks the complete native action sequence,
including calibration and actions executed by learned tools.

The Linux commissioning check executes the same three native commands through F
and F-adaptive, tests the extra hardware reads, actual sandboxed tool execution,
the read-only guard and version rollback, and requires identical terminal state.
It makes zero Astra calls and produces no benchmark score. Live transport and
two short actor/replay checks must then pass before the pilot begins.

## Launch and transfer

```bash
source emimic/bin/activate
python -m experiments.libero_hardware_interface.tools.prepare_osmo \
  /path/to/new/adaptive-pilot --adaptive --stage pilot --pool groot-l40-05
osmo workflow submit /path/to/new/adaptive-pilot/workflow.yaml --pool groot-l40-05
```

`preregistration-adaptive.yaml` freezes the companion comparison. RoboCasa has
its own companion manifest and `--adaptive` launcher using the same layer with
the 12-dimensional mobile controller, left/right/wrist cameras and its native
extra encoders. Its full pilot remains dependent on resolving the existing
paired-reset discrepancies. No adaptive RoboCasa score is claimed.

The process dashboard accepts `--cohorts metadata.json` for additional workflow
snapshots. Each entry specifies its title, benchmark, stage, planned trial count,
source commit, note and assigned `conditions`. Matching uses those assigned
methods, and interface changes/calibration have explicit timeline markers.
