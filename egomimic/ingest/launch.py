"""Where conversion jobs run: AWS Batch, or in this process (tests and local runs).

A launcher submits one egomimic.ingest.convert_job per delivery and reports whether a
submitted job is still running. The job records its own outcome in the store, so the
watcher only asks a launcher about jobs that ended without one (killed, out of memory,
a Spot interruption past its retries).
"""

from __future__ import annotations

import shlex
import uuid
from abc import ABC, abstractmethod

from egomimic.ingest.store import Ref, Store

RUNNING, ENDED = "running", "ended"


class Launcher(ABC):
    @abstractmethod
    def submit(self, ref: Ref, fingerprint: str) -> str:
        """Start converting `ref` at `fingerprint`; returns a job id."""

    @abstractmethod
    def state(self, job_id: str) -> str:
        """RUNNING, or ENDED (whether or not it recorded an outcome)."""


class LocalLauncher(Launcher):
    """Runs each job to completion inside submit."""

    def __init__(self, store: Store, lance_root, work, **settings):
        self.store, self.lance_root, self.work = store, lance_root, work
        self.settings = settings

    def submit(self, ref: Ref, fingerprint: str) -> str:
        from egomimic.ingest.convert_job import run_job

        run_job(
            self.store, ref, fingerprint, self.lance_root, self.work, **self.settings
        )
        return f"local-{uuid.uuid4().hex[:12]}"

    def state(self, job_id: str) -> str:
        return ENDED


class BatchLauncher(Launcher):
    """AWS Batch: one job per delivery on the queue and job definition that
    egomimic/ingest/aws/setup_batch.py creates. `job_args` are the convert_job flags
    every job shares (store, Lance root, work dir, conversion settings)."""

    def __init__(self, queue: str, definition: str, job_args: list[str], client=None):
        if client is None:
            import boto3

            client = boto3.client("batch")
        self.batch, self.queue, self.definition = client, queue, definition
        self.job_args = list(job_args)

    def command(self, ref: Ref, fingerprint: str) -> list[str]:
        return [
            "python", "-m", "egomimic.ingest.convert_job",
            "--vendor", ref.vendor, "--delivery", ref.delivery_id,
            "--fingerprint", fingerprint, *self.job_args,
        ]  # fmt: skip

    def submit(self, ref: Ref, fingerprint: str) -> str:
        # Batch job names: letters, digits, hyphens and underscores, up to 128.
        name = f"convert-{ref.vendor}-{ref.delivery_id}"[:110]
        name = "".join(c if c.isalnum() or c in "-_" else "-" for c in name)
        r = self.batch.submit_job(
            jobName=f"{name}-{fingerprint[:12]}",
            jobQueue=self.queue,
            jobDefinition=self.definition,
            containerOverrides={"command": self.command(ref, fingerprint)},
            tags={"vendor": ref.vendor},
        )
        return r["jobId"]

    def state(self, job_id: str) -> str:
        jobs = self.batch.describe_jobs(jobs=[job_id])["jobs"]
        if not jobs:  # Batch forgets jobs after a while: treat as ended
            return ENDED
        return ENDED if jobs[0]["status"] in ("SUCCEEDED", "FAILED") else RUNNING

    def __repr__(self) -> str:
        return f"BatchLauncher({self.queue}, {shlex.join(self.job_args)})"
