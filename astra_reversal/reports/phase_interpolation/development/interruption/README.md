# Interrupted development workers

The original development workflow ended with OSMO control failures on workers 0/1. Both were on `w-7183`; scheduler events record repipelining to `w-2132` around 17:45 UTC on 2026-09-25. The deeper infrastructure cause is unverified. Worker 2 completed and its separate audits are retained.

Published provider records contain **63 calls and 519,361 tokens of known additional interruption overhead** for workers 0/1. This is **not an exact total overhead**: unpublished tails and in-flight calls are unknown. The original 77-call/598,528-token snapshot also includes worker 2's 14 calls/79,167 tokens, which belong to final complete-case cost and must not be added again as overhead.

Workers 0/1 are rerun in full with the same source payload, protocol and prescribed resets, independently of partial outcomes. Partial attempts here are incomplete recordings, not audited success evidence. Recovery metadata compatibility is checked separately once available.

`known_costs.json` and `published_calls.csv` preserve counts, provider-reported input/output/total/reasoning costs, source hashes, metadata identities and prefix coverage. Reasoning tokens are included within output tokens. No dollar price is assumed. `workflow.json` preserves normalized status/failure/scheduler evidence. `event_prefixes/` omits image payloads and hashes unexpanded event kinds. `providers/` contains deterministic compressed copies of the original provider records with only the endpoint URL replaced by its digest; exact original-file and original-line hashes remain available. No signed catalog URLs or credentials are published.

`source/build_interruption.py.txt` is the exact operational source. Its docstring records the argument-based invocation. It performs no simulator, GPU, model or provider call; optional network reads download already-published event artifacts. These receipts do not replace the complete-case, numerical-array or feedback audits.

Worker 1's event prefix contains one additional accepted decision whose provider sidecar row was not published. A later remote sidecar read still contains only the two earlier calls in that attempt. Its tokens are unknown, giving concrete evidence that the preserved usage ledger is incomplete.

`deployment_only_attempt.json` separately records the canceled first recovery attempt: worker 0 could not mount the missing Lustre CSI driver on `w-2132`; worker 1 started only container bootstrap after its image pull. Both payload uploads failed, and no study module ran. No study API/model cost is assigned to that deployment attempt; infrastructure allocation and bootstrap cost is unpriced. The second recovery workflow is the replacement named in `workflow.json`.
