"""Teacher-only guard for an atomically replaced model-head heartbeat file."""
from pathlib import Path
import time


VERSION = 'astra-native-head-heartbeat-v1'


def head_heartbeat_expired(path, *, now=None):
    """Keep existing missing-file semantics without an exists/stat race."""
    try:
        stamp = Path(path).stat().st_mtime
    except FileNotFoundError:
        return False
    return (time.time() if now is None else now)-stamp > 300
