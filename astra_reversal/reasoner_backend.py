"""Explicit reasoner transport selection; never substitute models after failure."""

from pathlib import Path


def initialize_worker_backend(settings):
    """Expose a worker mailbox before GPU loading; no credentials cross the relay."""
    if settings.get("backend") == "codex_relay":
        from .codex_relay import ensure_server

        ensure_server()


def validate_backend(settings, *, codex=False):
    backend = settings.get("backend", "nvidia_http")
    if codex:
        if (
            backend != "codex_relay"
            or settings["model"] != "gpt-6-astra"
            or settings["reasoning_effort"] != "medium"
            or settings.get("max_completion_tokens") is not None
            or not settings.get("stop_on_provider_unavailable")
            or settings.get("retries") != 0
        ):
            raise ValueError("Codex studies require their declared Astra harness")
    elif backend != "nvidia_http":
        raise ValueError("A different reasoner backend requires its own study version")


def make_client(family, settings, response_log: Path):
    kwargs = {
        "model": settings["model"],
        "response_log": response_log,
        "reasoning_effort": settings["reasoning_effort"],
        "max_completion_tokens": settings.get("max_completion_tokens"),
        "timeout": settings["timeout_seconds"],
    }
    if family not in ("frs", "representation"):
        raise ValueError("Unknown reasoner family")
    if settings.get("backend", "nvidia_http") == "codex_relay":
        from .codex_relay import CodexRelayClient

        return CodexRelayClient(family=family, **kwargs)
    if settings.get("backend", "nvidia_http") != "nvidia_http":
        raise ValueError("Unknown reasoner backend")
    if family == "frs":
        from .frs_agent import FRSClient

        return FRSClient(**kwargs)
    from .representation_agent import RepresentationClient

    return RepresentationClient(**kwargs)
