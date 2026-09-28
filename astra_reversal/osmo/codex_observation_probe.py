"""Exercise the remote Codex mailbox on recorded observations, without a GPU.

These are reasoner/transport checks, never benchmark success-rate observations.
Requests are supplied separately and contain only the existing reasoner inputs.
"""

import json
import os
from pathlib import Path

from astra_reversal.codex_relay import CodexRelayClient, ensure_server
from astra_reversal.provider_stop import require_provider_available


def main():
    directory = Path(os.environ["ASTRA_CODEX_PROBE_DIRECTORY"])
    ensure_server()
    results = []
    for name, family in (
        ("frs", "frs"),
        ("vei", "representation"),
        ("vli", "representation"),
    ):
        request = json.loads((directory / f"{name}.json").read_text())
        client = CodexRelayClient(
            family=family,
            model="gpt-6-astra",
            reasoning_effort="medium",
            max_completion_tokens=None,
            timeout=300,
            response_log=directory / "provider.jsonl",
        )
        proposal = client.propose(request)
        record = client.records[-1]
        require_provider_available(record)
        result = {
            "method": name,
            "request_fingerprint": request["request_fingerprint"],
            "proposal": proposal,
            "record": record,
            "scope": "Recorded-observation reasoner test; no rollout or success rate",
        }
        results.append(result)
        (directory / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
