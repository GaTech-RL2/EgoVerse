# Observed phase interpolation in frozen π0.5

This follow-up tests whether Astra can select standard-task donors and change
their interpolation from fresh camera observations during a rollout. It keeps
the static intervention experiment and its results unchanged. The configuration
is [phase_interpolation_v1.json](configs/phase_interpolation_v1.json).
The [experiment record](reports/phase_interpolation/README.md) links measured
outcomes, iteration/token costs, observation examples and archive audits.

## What is being ported

The [LIBERO-OOD paper, v5](https://arxiv.org/html/2505.03500v5) describes tokenwise
text embedding interpolation (TEI) and a per-layer text latent residual (TLI).
This implementation uses the frozen `lerobot/pi05_libero_base` checkpoint at
`a217bfd3b14673cf2ce597e69997ab21866438dd`, with the previously verified OpenPI
LIBERO input profile. It is a π0.5 port, not a reproduction of the paper's π0
success rate.

For source tasks A and B and weight α:

| Operator | Update |
| --- | --- |
| TEI | Replace eligible target input embeddings with `(1-α) E(A) + α E(B)`. |
| TLI | Add `(1-2α) (T_A - T_B)` to eligible target hidden states at each effective boundary. |
| TEI+TLI | Apply both updates with the same pair and α. |

The active input profile uses plain SentencePiece instructions. Direct edits
cover instruction slots only; BOS, terminal newline, padding, masks and positions
are preserved. Source instruction positions are left-aligned to target positions,
with truncation or zero-padding. All alignment counts and positions are recorded.
The saved export's separate `Task/State/Action` processor is deliberately rejected
by this interpolation interface because its instruction boundary is unverified.

The native prefix has 18 decoder blocks. Banks retain all 18 post-block text
states. Residuals are injected after blocks 0–16, before the next block builds its
K/V cache; an edit after the last block cannot affect the action expert. Hooks are
removed before denoising. Direct writes leave vision slots unchanged, although
subsequent attention can change vision hidden states. A fresh cache is built at
every policy replan.

Identity gates check TEI with A=B=target and TLI with α=0.5 against native action
sampling. Real-checkpoint probes also require nonzero TEI and TLI to change the
seven decoded action channels before any rollout. Native A/B endpoint parity requires matching token layouts; unequal
prompt lengths do not imply identical conditioning under target-mask alignment.

## Donor data and oracle

Nine standard LIBERO donor tasks cover the released oracle mapping. Their IDs,
prompts, exact first 20 metadata episode IDs, frame counts and revisions are pinned
in [interpolation_catalog.py](interpolation_catalog.py). Extraction uses all 18,947
frames from these 180 standard demonstrations in public
[`physical-intelligence/libero`](https://huggingface.co/datasets/physical-intelligence/libero/tree/a4336d589d589045d1c56423ffdf3b88a0e19b1f).
The frame-weighted mean accumulates in float64 and stores float32. Files are
verified against their published LFS hashes, and each captured frame is logged.
No OOD demonstration is used. Donor images already contain the upstream 180-degree
rotation; the adapter handles resizing and normalization.

The oracle gets the released source-task mapping. Astra sees only the restricted
nine-donor catalog, not target-to-source mappings or demonstrations. This library
is informed by the benchmark and must not be described as an unrestricted or
held-out task library.

The declared oracle schedule is `α=clip(i/λ,0,1)` for zero-based policy-call index
`i`. Each call executes 5 environment actions. Wine-to-bowl uses the paper's λ=14;
other per-task λ values come from the released mapping. The release instead uses
BF16 `linspace(0,1,λ+2)`, wine λ=12, and a hardcoded 9-token span. Those differences
are preserved in oracle metadata rather than described as exact replication.

## Comparison and feedback

Development uses the previously inspected wine-to-bowl, milk-to-plate and
center-bowl-to-cabinet cases at seed 19. Evaluation uses all 20 OOD compositions
with seed 29 resets, captured before loading the model and restored identically
for every attempt. Tasks have already been inspected, and checkpoint training
overlap remains unknown. This is exploratory online rescue with reset access.

Native fresh-noise and known-noise controls accompany the common recovered-noise
baseline. Conditional rescue arms are random noise, oracle TEI/TLI/TEI+TLI, and
Astra TEI/TLI/TLI+vision. All interpolation arms reuse the same recovered baseline
noise. They test conditioning transfer; gains are not attributed to redundant
inversion of a policy-generated reference.

Each arm receives the common baseline. A successful baseline needs no rescue.
On failure, the oracle has one intervention rollout; random noise and Astra have
at most two. Each rollout is capped at 300 actions. Development forces one
intervention after a successful baseline to exercise every path; these additional
costs are recorded and development is not included in evaluation success rates.

Astra is called at actions 0,25,…,275. It receives up to four recent paired raw
camera observations, including the current state, all decisions from the current
attempt, previous outcomes, and four raw snapshots plus decisions from its last
completed attempt. It chooses two catalog IDs and α. It can change the pair or
move α backward based on observed failures. Simulator object poses and dense
goal predicates are not supplied.

The vision arm adds fresh translucent magenta marks to existing camera images.
Marks last only the current 5-action policy chunk and expire at the next replan;
there is no tracker. Astra always sees raw images. Rejected calls consume their
decision slot, hold the last valid text parameters (or native target if absent),
expire vision, and cause no hidden retry or synthesized proposal.

## Accounting and execution

Report full-rollout revisions and within-rollout decisions separately. Record
success curves by total rollout budget, failures censored at the cap, actions,
velocity evaluations, policy/HTTP time, every physical provider call, rejected
responses, and missing usage. Reasoning tokens are a subset of output tokens.
Monetary cost is unavailable without a verified provider rate.

Bank extraction, native-weight gates and rollouts run on OSMO L40S. Each worker
checks the hardware, frozen checkpoint inventory, tests, payload hash, donor
artifact hashes, reset hashes and native sampler parity. Workers use separate
artifact prefixes; uploaded source snapshots are committed and immutable.

Three offline checks accompany the results. `interpolation_audit.py` checks every
recorded array, noise construction, numerical gate, reset, decoded action and
interpolation mask against the frozen inputs. `interpolation_feedback_audit.py`
binds the actual camera arrays to Astra's image requests, decisions, text holds
and vision expiry. `interpolation_report.py` requires complete development or
evaluation coverage and reconciles provider records with success and cost tables.
These checks verify saved evidence; they do not replay model inference or the
simulator. Full arrays remain in the worker archives, with hashes in the receipts.
