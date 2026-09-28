"""Provider-budget failures must interrupt rather than generate fallback results."""

import pytest

from astra_reversal.astra_client import ClientError
from astra_reversal.provider_stop import ProviderUnavailable, require_provider_available


@pytest.mark.parametrize(
    "record,reason",
    [
        ({"provider_call": True, "http_status": 429}, "provider_rate_limited"),
        (
            {
                "provider_call": True,
                "http_status": 429,
                "provider_error": {"type": "budget_exceeded", "message": "private"},
            },
            "provider_budget_exhausted",
        ),
        (
            {"provider_call": True, "provider_error": {"code": "insufficient_quota"}},
            "provider_budget_exhausted",
        ),
        ({"provider_call": True, "http_status": 401}, "provider_access_unavailable"),
        ({"provider_call": True, "http_status": 403}, "provider_access_unavailable"),
        ({"provider_call": False}, "provider_preflight_failed"),
        (None, "provider_ledger_missing"),
    ],
)
def test_unavailability_cannot_be_caught_as_ordinary_proposal_rejection(record, reason):
    with pytest.raises(ProviderUnavailable, match=reason) as stopped:
        require_provider_available(record)
    assert not isinstance(stopped.value, ClientError)
    assert set(stopped.value.receipt) == {"reason", "http_status"}
    assert "private" not in str(stopped.value.receipt)


@pytest.mark.parametrize("status", [200, 503])
def test_normal_response_or_transient_failure_retains_existing_trial_policy(status):
    require_provider_available({"provider_call": True, "http_status": status})
