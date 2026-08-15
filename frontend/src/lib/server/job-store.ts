import "server-only";

import { randomUUID } from "node:crypto";
import { createWriteStream } from "node:fs";
import { copyFile, mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { spawn, spawnSync } from "node:child_process";

import { jobStatusSchema, type AnalysisResult, type JobStatus } from "@/lib/contracts";
import {
  adaptCurrentCsvResult,
  readArtifactMtimes,
  REPO_ROOT,
  RESULT_PATHS,
} from "@/lib/server/csv-adapter";

const JOBS_ROOT = path.join(process.cwd(), "data/jobs");
const JOB_ID_PATTERN = /^[0-9a-f-]{36}$/;

type InternalJob = JobStatus & {
  pid?: number;
  artifactMtimesBefore?: Awaited<ReturnType<typeof readArtifactMtimes>>;
};

export class ActiveJobError extends Error {
  constructor(public readonly activeJobId: string) {
    super("The fixed-output backend already has an active analysis.");
  }
}

export class BackendEnvironmentError extends Error {}

function now() {
  return new Date().toISOString();
}

function jobDirectory(jobId: string) {
  if (!JOB_ID_PATTERN.test(jobId)) throw new Error("Invalid job ID.");
  return path.join(JOBS_ROOT, jobId);
}

async function writeJson(filePath: string, value: unknown) {
  await mkdir(path.dirname(filePath), { recursive: true });
  const temporary = `${filePath}.tmp`;
  await writeFile(temporary, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  await import("node:fs/promises").then(({ rename }) => rename(temporary, filePath));
}

async function readInternalJob(jobId: string): Promise<InternalJob> {
  const contents = await readFile(path.join(jobDirectory(jobId), "status.json"), "utf8");
  return JSON.parse(contents) as InternalJob;
}

function publicStatus(status: InternalJob): JobStatus {
  return jobStatusSchema.parse(status);
}

function isProcessAlive(pid?: number) {
  if (!pid) return false;
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function resolvePythonExecutable() {
  const configured = process.env.PYTHON_EXECUTABLE?.trim();
  const candidates = [
    configured,
    path.join(REPO_ROOT, "emimic/bin/python"),
    "python",
    "python3",
  ].filter((candidate, index, all): candidate is string => Boolean(candidate) && all.indexOf(candidate) === index);
  const required = [
    "numpy",
    "pandas",
    "torch",
    "cv2",
    "scipy",
    "sklearn",
    "sentence_transformers",
    "transformers",
    "boto3",
    "sqlalchemy",
    "psycopg",
  ];
  const check = [
    "import importlib.util, sys",
    `required = ${JSON.stringify(required)}`,
    "missing = [name for name in required if importlib.util.find_spec(name) is None]",
    "print(','.join(missing))",
    "sys.exit(1 if missing else 0)",
  ].join("; ");
  const failures: string[] = [];

  for (const candidate of candidates) {
    const probe = spawnSync(candidate, ["-c", check], {
      cwd: REPO_ROOT,
      env: process.env,
      encoding: "utf8",
      timeout: 10_000,
    });
    if (!probe.error && probe.status === 0) return candidate;
    if (configured && candidate === configured) {
      failures.push(
        probe.error
          ? `${candidate}: ${probe.error.message}`
          : `${candidate}: missing ${probe.stdout.trim() || "required packages"}`,
      );
    }
  }

  const configuredDetail = failures.length ? ` (${failures.join("; ")})` : "";
  throw new BackendEnvironmentError(
    "Track 2 Python environment is unavailable. Start Next.js after `conda activate emimic`, " +
      "or set PYTHON_EXECUTABLE to that environment's Python path." +
      configuredDetail,
  );
}

async function updateJob(jobId: string, patch: Partial<InternalJob>) {
  const current = await readInternalJob(jobId);
  const updated: InternalJob = { ...current, ...patch, updatedAt: now() };
  await writeJson(path.join(jobDirectory(jobId), "status.json"), updated);
  return updated;
}

async function artifactsAreFresh(status: InternalJob) {
  const after = await readArtifactMtimes();
  const before = status.artifactMtimesBefore;
  if (!after.scores || !after.manifestA || !after.manifestB || !before) return false;
  return (
    (before.scores === null || after.scores > before.scores) &&
    (before.manifestA === null || after.manifestA > before.manifestA) &&
    (before.manifestB === null || after.manifestB > before.manifestB)
  );
}

async function completeJob(jobId: string, exitCode?: number | null) {
  const status = await readInternalJob(jobId);
  if (status.state === "completed" || status.state === "failed") return;

  if (exitCode !== undefined && exitCode !== 0) {
    await updateJob(jobId, {
      state: "failed",
      completedAt: now(),
      error: `Track 2 exited with code ${exitCode ?? "unknown"}. Inspect worker.log for details.`,
    });
    return;
  }

  if (!(await artifactsAreFresh(status))) {
    await updateJob(jobId, {
      state: "failed",
      completedAt: now(),
      error: "Track 2 stopped without producing fresh CSV outputs.",
    });
    return;
  }

  try {
    const result = await adaptCurrentCsvResult(jobId);
    const directory = jobDirectory(jobId);
    await Promise.all([
      writeJson(path.join(directory, "result.json"), result),
      copyFile(
        /* turbopackIgnore: true */ RESULT_PATHS.scores,
        path.join(directory, "single_random_120h_results.csv"),
      ),
      copyFile(
        /* turbopackIgnore: true */ RESULT_PATHS.manifests["subset-a"],
        path.join(directory, "mecka_seed_42.csv"),
      ),
      copyFile(
        /* turbopackIgnore: true */ RESULT_PATHS.manifests["subset-b"],
        path.join(directory, "scale_seed_42.csv"),
      ),
    ]);
    await updateJob(jobId, {
      state: "completed",
      completedAt: now(),
      error: null,
    });
  } catch (error) {
    await updateJob(jobId, {
      state: "failed",
      completedAt: now(),
      error: error instanceof Error ? error.message : "Unable to adapt Track 2 CSV outputs.",
    });
  }
}

async function findActiveJob(): Promise<InternalJob | null> {
  await mkdir(JOBS_ROOT, { recursive: true });
  const entries = await readdir(JOBS_ROOT, { withFileTypes: true });
  for (const entry of entries) {
    if (!entry.isDirectory() || !JOB_ID_PATTERN.test(entry.name)) continue;
    try {
      const status = await readInternalJob(entry.name);
      if (status.state === "queued" || status.state === "running") {
        if (status.state === "running" && status.pid && !isProcessAlive(status.pid)) {
          await completeJob(status.jobId);
          const reconciled = await readInternalJob(status.jobId);
          if (reconciled.state !== "queued" && reconciled.state !== "running") continue;
        }
        return status;
      }
    } catch {
      // Ignore malformed old job folders; their errors remain local to that folder.
    }
  }
  return null;
}

export async function createAnalysisJob(): Promise<JobStatus> {
  const active = await findActiveJob();
  if (active) throw new ActiveJobError(active.jobId);
  const python = resolvePythonExecutable();

  const jobId = randomUUID();
  const directory = jobDirectory(jobId);
  await mkdir(directory, { recursive: false });
  const timestamp = now();
  const status: InternalJob = {
    jobId,
    state: "queued",
    createdAt: timestamp,
    updatedAt: timestamp,
    error: null,
    artifactMtimesBefore: await readArtifactMtimes(),
  };
  await writeJson(path.join(directory, "status.json"), status);

  const log = createWriteStream(path.join(directory, "worker.log"), { flags: "a" });
  const child = spawn(
    /* turbopackIgnore: true */ python,
    ["-m", "track2.run_lab_random_120h_single"],
    {
    cwd: REPO_ROOT,
    env: process.env,
    stdio: ["ignore", "pipe", "pipe"],
    },
  );
  child.stdout.pipe(log);
  child.stderr.pipe(log);

  const running = await updateJob(jobId, {
    state: "running",
    startedAt: now(),
    pid: child.pid,
  });

  child.once("error", async (error) => {
    log.end();
    await updateJob(jobId, {
      state: "failed",
      completedAt: now(),
      error: `Could not start ${python}: ${error.message}`,
    });
  });
  child.once("close", async (code) => {
    log.end();
    await completeJob(jobId, code);
  });

  return publicStatus(running);
}

export async function getAnalysisJob(jobId: string): Promise<JobStatus> {
  let status = await readInternalJob(jobId);
  const millisecondsSinceUpdate = Date.now() - Date.parse(status.updatedAt);
  if (
    status.state === "running" &&
    status.pid &&
    !isProcessAlive(status.pid) &&
    millisecondsSinceUpdate > 5_000
  ) {
    await completeJob(jobId);
    status = await readInternalJob(jobId);
  }
  return publicStatus(status);
}

export async function getAnalysisResult(jobId: string): Promise<AnalysisResult> {
  const status = await getAnalysisJob(jobId);
  if (status.state !== "completed") {
    throw new Error(`Result is unavailable while job state is ${status.state}.`);
  }
  const contents = await readFile(path.join(jobDirectory(jobId), "result.json"), "utf8");
  return JSON.parse(contents) as AnalysisResult;
}
