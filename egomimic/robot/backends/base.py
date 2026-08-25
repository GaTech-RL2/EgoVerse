"""Small structural contract shared by live and offline robot backends."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RobotBackend(Protocol):
    """The hardware-independent surface consumed by the rollout loop."""

    def get_obs(self) -> dict[str, Any]: ...

    def set_joints(self, desired_position: Any, arm: str) -> None: ...

    def set_pose(self, desired_position: Any, arm: str) -> Any: ...

    def set_home(self) -> None: ...
