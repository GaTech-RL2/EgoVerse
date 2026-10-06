# Astra / frozen pi0.5 Meta-Harness experiment

This implements the supplied [October 6 proposal](PROPOSAL.md) on the recovered
study source `d4b2b690ac8992137f35af566429f6241c39afca`. The source was available in
the local `astra/demo-skill-library-20260930` checkout. That runner lives in
EgoVerse's `astra_reversal` package. The implementation reuses its policy loader,
input hooks, matched reset runner, action decoder, archival system and OSMO setup.
It does not update Astra or pi0.5 weights or resume the separate policy-learning
experiment.

## Status and first experiment

The controller and bounded search infrastructure are implemented and CPU tested.
No new GPU rollout, Astra latency measurement, search improvement or held-out
success result has been obtained. Passing synthetic tests is not M0 completion.

The prepared first experiment uses one OSMO L40S, at most two hours, Goal-OOD task
6 and Spatial-OOD task 2, reset 0 and seed 137. It compares native, synchronous
interface debugging, and actual asynchronous runtime Astra. This is at most six
episodes, 1,800 primitive actions and 32 runtime Astra requests. The same captured
reset and deterministic policy-noise schedule are used across arms. It saves
three example videos and raw observation traces. There is no automatic larger
launch after the pilot.

Before rollouts, the worker runs existing native integration tests and a
weighted checkpoint gate: native sampler parity, neutral VEI/VLI, protected
token slots, TLI's neutral midpoint, and TEI's source-A endpoint with matching
source masks. It verifies actual parameter hashes before and after execution.
The pilot uses the nine locally cached standard sources, labelled incomplete;
the main study must freeze its complete eligible source pool separately.

## Timing and tool contract

- Prediction horizon 50; execute `[0, 5)` and replan; Euler 10; episode limit 300.
- At most one pending Astra request and eight dispatches per episode. Failed
  requests consume the cap. No automatic retries or unbudgeted repair loop.
- Responses commit only at five-action replan boundaries. The initial freshness
  limits are 20 actions and 10 wall-clock seconds, plus an unchanged stage ID.
  These are proposed limits to profile, not evidence they are sufficient.
- `set_policy_program` resolves a retrieved, hashed standard-demo segment.
  `keep_policy_program` preserves both cursor and expiry. Clear returns to native
  conditioning. Invalid calls retain a valid program or fall back to native.
- Only native, TEI and VEI are enabled for control. TEI zero selects source A;
  it is not native text. Simultaneous TEI/VEI is rejected. Live proprioception is
  mandatory. Exclusive source bounds and hold/advance behavior are unchanged.
- System 2 always gets raw live paired cameras. Simulator success, object poses
  and privileged state never enter runtime context. The fixed evaluator alone
  receives the original environment success criterion.

## Astra serving and the token gate

The existing authenticated Codex relay is reusable for the initial interface and
latency pilot. It pins `gpt-6-astra` and medium effort and preserves full provider
receipts, but does **not** expose a hard completion-token cap or an immutable
served model snapshot. Its runs are explicitly excluded from main selection.

`astra_worker.py` provides a separate Responses transport: count the complete
input, including images/tools, before generation; reject inputs above 8,192
tokens; set `max_output_tokens=256`; validate reported usage and model identity;
never retry or substitute models. This limit includes reasoning tokens and may
be too small for medium reasoning. The pilot must resolve that experimentally
before freezing main-study settings. The transport requires an actual
`OPENAI_API_KEY`; a Codex subscription/login is not silently reused as one.
Neither live API availability nor the capped transport has been verified here.

Official interface references:
[function calling](https://developers.openai.com/api/docs/guides/function-calling),
[token counting](https://developers.openai.com/api/docs/guides/token-counting),
and [Astra model settings](https://developers.openai.com/api/docs/models/gpt-6-astra).

## Search isolation and evidence

`harness.py` interprets a restricted executable Python subset. It never uses
Python `exec`/`eval`. Only JSON copies of public observations, bounded history,
cards and memory enter the interpreter; no imports, attributes, filesystem,
network, evaluator handle or arbitrary calls are available. Instruction and
intermediate-size limits prevent unbounded execution. This is a deliberately
smaller search space than arbitrary Python.

The searchable function can alter retrieval, prior-frame selection, memory,
instructions and call scheduling. The fixed runtime enforces mandatory current
observations, source eligibility, context bounds and call limits. Memory is
labelled as hypotheses; it cannot upgrade a model assertion into ground truth.

`proposer.py` implements a separate development-only Astra role that reads
candidate code, search metrics and traces through a confined file/image API.
Failed candidates are retained. Prompt-only search may change only the literal
instruction, with an AST comparison preventing code changes. Each arm gets eight
candidate attempts in four rounds and identical proposer limits. The search
driver accepts a trusted evaluator callback; the pilot does not launch it.

`search.py` rejects incomplete cohorts, changed cards/checkpoints, execution
contract violations, uncapped pilot results and unavailable usage. A selected
bundle can be frozen only after valid search evaluation. Final evidence is kept
outside the proposer's search capability. The provisional 12/4/4 task split is
explicitly **reset generalization on previously inspected compositions**, not
unseen-task transfer. New, uninspected compositions remain necessary for the
stronger claim. Fixed-library/selector baselines, main search execution and final
confirmation remain to be commissioned after the pilot gate.

The method follows the proposal's adaptation of
[Meta-Harness](https://arxiv.org/abs/2603.28052); no robotics performance claim is
inherited from that paper.

## Local validation and launch preparation

Activate an existing `emimic` environment before project Python commands:

```bash
source /path/to/emimic/bin/activate
python -m pytest --confcutdir=tests/unit/astra tests/unit/astra -q
python -m astra_reversal.osmo.prepare_meta_harness_launch \
  /path/to/verified/demo-skills/bundle-v4 /path/to/new/pilot-bundle \
  --activation /path/to/emimic/bin/activate
```

Preparation requires committed sources. It verifies the prior payload hash and
copies only explicitly allowed public tokenizer/reference/demo-cache assets.
It creates an isolated private relay token outside the source payload. Existing
checkouts, jobs and experiment archives are preserved.

Once OSMO/network access is available:

```bash
osmo workflow submit /path/to/new/pilot-bundle/workflow.yaml \
  --pool groot-l40s-01 --format-type json
bash /path/to/new/pilot-bundle/connect.sh WORKFLOW_ID
```

The resulting `pilot_report.json`, weighted interface gate, model hashes, reset
manifests, provider receipts, videos and observation/tool/timing traces establish
whether to proceed. `main_launch_allowed` remains false until actual latency,
token accounting and model-version checks justify a fixed main protocol.
