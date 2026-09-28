"""Stop a study when the provider cannot serve its authorized intervention calls."""


class ProviderUnavailable(RuntimeError):
    """Outside ClientError so ordinary rejected-proposal fallback cannot catch it."""

    def __init__(self, reason, http_status=None):
        self.receipt = {"reason": reason, "http_status": http_status}
        super().__init__(reason)


def require_provider_available(record):
    """Inspect a retained ledger row without exposing raw provider error text.

    A rejected proposal from an available provider keeps the existing fallback
    behavior. Quota, authorization, rate-limit and preflight failures interrupt
    the study; unfinished rollouts must not become benchmark failures or successes.
    """
    if not isinstance(record, dict):
        raise ProviderUnavailable("provider_ledger_missing")
    status = record.get("http_status")
    status = status if type(status) is int else None
    if record.get("provider_unavailable") is True:
        raise ProviderUnavailable("reasoner_backend_unavailable", status)
    error = record.get("provider_error", {})
    error = error if isinstance(error, dict) else {}
    categories = (error.get("type"), error.get("code"))
    if any(c in ("budget_exceeded", "insufficient_quota") for c in categories):
        raise ProviderUnavailable("provider_budget_exhausted", status)
    if status in (401, 402, 403):
        raise ProviderUnavailable("provider_access_unavailable", status)
    if status == 429:
        raise ProviderUnavailable("provider_rate_limited", status)
    if not record.get("provider_call"):
        raise ProviderUnavailable("provider_preflight_failed", status)
