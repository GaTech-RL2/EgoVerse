"""Executable experiment settings and deliberately separate benchmark protocols."""

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Literal

OPENPI_REVISION = "981483dca0fd9acba698fea00aa6e52d56a66c58"
LIBERO_REVISION = "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"
OOD_REVISION = "587a6cbf64f16c7b87fa5805dc0ed934192239a4"

METHODS = (
    "policy_fresh",
    "policy_reused",
    "subgoal",
    "direct_astra",
    "same_condition",
    "reversal",
    "random_latent",
    "compute_matched",
    "augmentation_only",
    "inversion_only",
    "stage2",
    "language_only",
    "observation_only",
    "conditioning_transfer",
    "known_noise_transfer",
)
STAGE2_METHODS = frozenset(METHODS[8:])
AGENT_FREE_METHODS = frozenset(
    ("policy_fresh", "policy_reused", "compute_matched", "inversion_only")
)


@dataclass(frozen=True)
class BenchmarkConfig:
    suite: str = "libero_10"
    control_frequency_hz: int = 20
    task_action_budget: int = 520
    stabilization_steps: int = 10
    trials_per_task: int = 50
    protocol: str = "standard_libero_10"
    environment_revision: str = LIBERO_REVISION
    reset_source: str = "prescribed_initial_states"

    @classmethod
    def preset(cls, suite: str):
        if suite == "libero_10":
            return cls()
        if suite in ("libero_goal_ood", "libero_spatial_ood"):
            return cls(
                suite=suite,
                task_action_budget=300,
                trials_per_task=10,
                protocol="released_modified_libero",
                environment_revision=OOD_REVISION,
                reset_source="seeded_reset_stream",
            )
        raise ValueError(f"Unsupported suite: {suite}")


@dataclass(frozen=True)
class AgentConfig:
    model_version: str | None = None
    sampling_settings: dict = field(default_factory=dict)
    refresh_env_steps: int = 20
    invalid_response_retries: int = 1
    subgoal_timeout_env_steps: int = 60
    subgoal_recovery_replans: int = 2
    history_limit: int = 40
    request_timeout_seconds: float = 120.0


@dataclass(frozen=True)
class PolicyConfig:
    config_name: str = "pi05_libero"
    checkpoint: str | None = None
    checkpoint_provenance: str = "unknown"
    training_overlap: str = "unknown"
    execute_steps: int = 5
    device: str = "cpu"
    tokenizer_path: str | None = None
    input_profile: str = "checkpoint"
    reference_assets: str | None = None


@dataclass(frozen=True)
class FlowConfig:
    integrator: Literal["euler", "heun", "rk4"] = "euler"
    inversion_steps: int = 10
    generation_steps: int = 10
    noise_mix_rho: float = 0.0
    time_power: float = 1.0

    @property
    def solver_options(self):
        return {"time_power": self.time_power} if self.time_power != 1.0 else {}


@dataclass(frozen=True)
class ControllerConfig:
    latent_max_age_env_steps: int = 20
    completion_positive_checks: int = 2
    output_bounds: Literal["clip", "reject"] = "clip"
    compute_matched_candidates: int = 2


@dataclass(frozen=True)
class EvaluationConfig:
    split: Literal["development", "test"] = "development"
    save_flow_traces: bool = True
    save_rollout_videos: bool = True
    bootstrap_samples: int = 2000


@dataclass(frozen=True)
class RunConfig:
    seed: int = 0
    method: str = "reversal"
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def from_dict(cls, data: dict):
        values = dict(data)
        sections = {
            "benchmark": BenchmarkConfig,
            "agent": AgentConfig,
            "policy": PolicyConfig,
            "flow": FlowConfig,
            "controller": ControllerConfig,
            "evaluation": EvaluationConfig,
        }
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        for name, section in sections.items():
            if name in values:
                nested = values[name]
                if name == "benchmark":
                    nested = {
                        **asdict(
                            BenchmarkConfig.preset(nested.get("suite", "libero_10"))
                        ),
                        **nested,
                    }
                values[name] = section(**nested)
        config = cls(**values)
        config.validate()
        return config

    def validate(self, horizon: int | None = None):
        if self.policy.input_profile not in ("checkpoint", "openpi_libero"):
            raise ValueError("Unknown policy input profile")
        if (self.policy.input_profile == "openpi_libero") != bool(
            self.policy.reference_assets
        ):
            raise ValueError("openpi_libero inputs require explicit reference_assets")
        if self.method not in METHODS:
            raise ValueError(f"Unknown method {self.method!r}; choose from {METHODS}")
        # Changing these together is necessary to keep the published protocols distinct.
        if self.benchmark != BenchmarkConfig.preset(self.benchmark.suite):
            raise ValueError(
                "Benchmark settings must match the complete suite preset; custom protocols need a separate implementation"
            )
        positive = (
            self.agent.refresh_env_steps,
            self.agent.subgoal_timeout_env_steps,
            self.agent.history_limit,
            self.agent.request_timeout_seconds,
            self.policy.execute_steps,
            self.flow.inversion_steps,
            self.flow.generation_steps,
            self.controller.latent_max_age_env_steps,
            self.controller.completion_positive_checks,
            self.controller.compute_matched_candidates,
            self.evaluation.bootstrap_samples,
        )
        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not math.isfinite(x)
            or x <= 0
            for x in positive
        ):
            raise ValueError(
                "Periods, horizons, solver steps, and limits must be positive"
            )
        integer_values = positive[:3] + positive[4:]
        if any(not isinstance(x, int) for x in integer_values):
            raise ValueError("Step counts and sample counts must be integers")
        for count in (
            self.seed,
            self.agent.invalid_response_retries,
            self.agent.subgoal_recovery_replans,
        ):
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("Seeds and retry counts must be nonnegative integers")
        if (
            self.flow.integrator not in ("euler", "heun", "rk4")
            or not 0 <= self.flow.noise_mix_rho <= 1
        ):
            raise ValueError("Unsupported solver or mixing coefficient")
        if not math.isfinite(self.flow.time_power) or self.flow.time_power <= 0:
            raise ValueError("Flow time_power must be finite and positive")
        if self.controller.output_bounds not in ("clip", "reject"):
            raise ValueError("Output bounds rule must be clip or reject")
        if self.evaluation.split not in ("development", "test"):
            raise ValueError("A development or test split is required")
        if horizon is not None:
            if self.policy.execute_steps > horizon:
                raise ValueError("execute_steps exceeds the checkpoint horizon")
            if (
                self.method == "direct_astra"
                and self.agent.refresh_env_steps != horizon
            ):
                raise ValueError(
                    "Direct Astra must refresh at H steps; it cannot repeat or stretch its chunk"
                )

    @property
    def stage(self):
        return 2 if self.method in STAGE2_METHODS else 1
