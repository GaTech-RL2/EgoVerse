"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  ArrowRight,
  Check,
  CircleDot,
  Play,
  RotateCw,
  X,
} from "lucide-react";

import { DimensionChart } from "@/components/comparison/dimension-chart";
import { WeightControls } from "@/components/controls/weight-controls";
import { EvidencePanel } from "@/components/evidence/evidence-panel";
import { Button } from "@/components/ui/button";
import {
  BACKEND_CONFIGURATION,
  DIMENSIONS,
  type SubsetId,
} from "@/lib/backend-config";
import {
  analysisResultSchema,
  jobStatusSchema,
  type AnalysisResult,
  type JobStatus,
} from "@/lib/contracts";
import {
  calculateWeightedScores,
  DEFAULT_WEIGHTS,
  equalWeights,
  type Weights,
} from "@/lib/scoring";

type DatasetSnapshot = Partial<
  Record<SubsetId, { label: string; episodeCount: number; estimatedHours: number }>
> | null;

const statusCopy = {
  idle: "READY",
  queued: "QUEUED",
  running: "RUNNING",
  completed: "COMPLETE",
  failed: "FAILED",
} as const;

function AnimatedNumber({ value, fallback = "—" }: { value?: number; fallback?: string }) {
  const [display, setDisplay] = useState(value ?? 0);
  const previous = useRef(value);

  useEffect(() => {
    if (value === undefined) {
      previous.current = undefined;
      return;
    }
    const start = previous.current ?? value;
    const difference = value - start;
    const duration = 180;
    const startedAt = performance.now();
    let frame = 0;

    const tick = (now: number) => {
      const progress = Math.min((now - startedAt) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(start + difference * eased);
      if (progress < 1) frame = requestAnimationFrame(tick);
    };

    frame = requestAnimationFrame(tick);
    previous.current = value;
    return () => cancelAnimationFrame(frame);
  }, [value]);

  return <>{value === undefined ? fallback : display.toFixed(1)}</>;
}

function formatElapsed(startedAt?: string) {
  if (!startedAt) return "00:00";
  const seconds = Math.max(0, Math.floor((Date.now() - new Date(startedAt).getTime()) / 1000));
  const minutes = Math.floor(seconds / 60);
  return `${String(minutes).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

function DatasetIdentity({
  side,
  name,
  score,
  hours,
  episodes,
}: {
  side: "A" | "B";
  name: string;
  score?: number;
  hours?: number;
  episodes?: number;
}) {
  return (
    <div className={`dataset-identity dataset-${side.toLowerCase()}`}>
      <div className="dataset-name-row">
        <span className={`subset-mark subset-mark-${side.toLowerCase()}`}>{side}</span>
        <div>
          <span>Dataset {side}</span>
          <strong>{name}</strong>
        </div>
      </div>
      <div className="hero-score">
        <AnimatedNumber value={score} />
      </div>
      <div className="dataset-meta">
        <span>{hours === undefined ? "120h target" : `${hours.toFixed(2)}h`}</span>
        <i />
        <span>{episodes === undefined ? "episodes after run" : `${episodes.toLocaleString()} episodes`}</span>
      </div>
    </div>
  );
}

export function AnalysisConsole({ initialSnapshot }: { initialSnapshot: DatasetSnapshot }) {
  const [weights, setWeights] = useState<Weights>(DEFAULT_WEIGHTS);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedDimension, setSelectedDimension] = useState<string>("visual");
  const [, setClock] = useState(0);

  const state = job?.state ?? (result ? "completed" : "idle");
  const isRunning = state === "queued" || state === "running";

  useEffect(() => {
    if (!isRunning) return;
    const interval = window.setInterval(() => setClock((value) => value + 1), 1000);
    return () => window.clearInterval(interval);
  }, [isRunning]);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    let timeout: ReturnType<typeof setTimeout> | undefined;

    const poll = async () => {
      try {
        const response = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Unable to read job state.");
        const status = jobStatusSchema.parse(payload);
        if (cancelled) return;
        setJob(status);

        if (status.state === "completed") {
          const resultResponse = await fetch(`/api/jobs/${jobId}/result`, { cache: "no-store" });
          const resultPayload = await resultResponse.json();
          if (!resultResponse.ok) {
            throw new Error(resultPayload.error || "Unable to load completed result.");
          }
          const parsedResult = analysisResultSchema.parse(resultPayload);
          const dimensionIds = parsedResult.dimensions.map((dimension) => dimension.id);
          setResult(parsedResult);
          setWeights(equalWeights(dimensionIds));
          setSelectedDimension(dimensionIds.includes("visual") ? "visual" : dimensionIds[0]);
          setError(null);
          setJobId(null);
          return;
        }
        if (status.state === "failed") {
          setError(status.error || "The backend analysis failed.");
          setJobId(null);
          return;
        }
        timeout = setTimeout(poll, 1500);
      } catch (pollError) {
        if (!cancelled) {
          setError(pollError instanceof Error ? pollError.message : "Polling failed.");
          setJobId(null);
        }
      }
    };

    void poll();
    return () => {
      cancelled = true;
      if (timeout) clearTimeout(timeout);
    };
  }, [jobId]);

  const weightedScores = useMemo(() => {
    if (!result) return null;
    return calculateWeightedScores(result, weights);
  }, [result, weights]);

  const subsetA = result?.subsets.find((subset) => subset.id === "subset-a");
  const subsetB = result?.subsets.find((subset) => subset.id === "subset-b");
  const scoreA = weightedScores?.["subset-a"];
  const scoreB = weightedScores?.["subset-b"];
  const delta = scoreA !== undefined && scoreB !== undefined ? scoreB - scoreA : null;
  const winner = delta === null || Math.abs(delta) < 0.005 ? null : delta > 0 ? "B" : "A";
  const dimensionComparisons = result && subsetA && subsetB
    ? result.dimensions.map((dimension) => ({
        ...dimension,
        delta: subsetB.dimensions[dimension.id].display - subsetA.dimensions[dimension.id].display,
      }))
    : [];
  const strongestA = dimensionComparisons
    .filter((dimension) => dimension.delta < 0)
    .sort((left, right) => left.delta - right.delta)[0];
  const strongestB = dimensionComparisons
    .filter((dimension) => dimension.delta > 0)
    .sort((left, right) => right.delta - left.delta)[0];
  const dimensionDefinitions = result?.dimensions ?? DIMENSIONS;

  async function runAnalysis() {
    setError(null);
    try {
      const response = await fetch("/api/jobs", { method: "POST" });
      const payload = await response.json();
      if (response.status === 409 && payload.activeJobId) {
        setJobId(payload.activeJobId);
        return;
      }
      if (!response.ok) throw new Error(payload.error || "Unable to start analysis.");
      const status = jobStatusSchema.parse(payload);
      setJob(status);
      setJobId(status.jobId);
    } catch (runError) {
      setError(runError instanceof Error ? runError.message : "Unable to start analysis.");
    }
  }

  const hoursA = subsetA?.dataset.estimatedHours ?? initialSnapshot?.["subset-a"]?.estimatedHours;
  const hoursB = subsetB?.dataset.estimatedHours ?? initialSnapshot?.["subset-b"]?.estimatedHours;
  const episodesA = subsetA?.dataset.episodeCount ?? initialSnapshot?.["subset-a"]?.episodeCount;
  const episodesB = subsetB?.dataset.episodeCount ?? initialSnapshot?.["subset-b"]?.episodeCount;
  const visualFailures = result
    ? result.subsets.reduce(
        (sum, subset) => sum + (subset.evidence.visual?.failedEpisodes ?? 0),
        0,
      )
    : null;

  return (
    <main className="research-console">
      <header className="console-header">
        <div className="brand-lockup">
          <span className="brand-symbol"><CircleDot aria-hidden="true" /></span>
          <div>
            <p>EgoVerse <span>/</span> Track 2</p>
            <h1>Diversity measurement</h1>
          </div>
        </div>

        <div className="run-controls">
          <div className={`run-status status-${state}`}>
            <span className="status-dot" />
            <div>
              <strong>{statusCopy[state]}</strong>
              <small>
                {isRunning
                  ? `Track 2 pipeline · ${formatElapsed(job?.startedAt)}`
                  : state === "completed"
                    ? "Results generated"
                    : state === "failed"
                      ? "Pipeline stopped"
                      : "Fixed A/B configuration"}
              </small>
            </div>
          </div>
          <Button onClick={runAnalysis} disabled={isRunning} className="run-button">
            {isRunning ? <RotateCw className="animate-spin" /> : <Play className="fill-current" />}
            {isRunning ? "Running" : result ? "Run again" : "Run analysis"}
          </Button>
        </div>
      </header>

      {error ? (
        <div className="inline-error" role="alert">
          <AlertTriangle aria-hidden="true" />
          <strong>Run failed</strong>
          <span>{error}</span>
          <button type="button" onClick={() => setError(null)} aria-label="Dismiss error"><X /></button>
        </div>
      ) : null}

      <section className="comparison-stage" aria-labelledby="comparison-title">
        <div className="comparison-heading">
          <p className="section-kicker">Weighted diversity score</p>
          <h2 id="comparison-title">{subsetA?.source ?? "Mecka"} <span>versus</span> {subsetB?.source ?? "Scale"}</h2>
          <p>
            {result && delta !== null
              ? Math.abs(delta) < 5
                ? "Similar totals. Fundamentally different diversity profiles."
                : `${winner === "A" ? subsetA!.source : subsetB!.source} leads under the current weighting.`
              : "Run the fixed pipeline to compare the configured subsets."}
          </p>
        </div>

        <div className="score-comparison">
          <DatasetIdentity side="A" name={subsetA?.source ?? "Mecka"} score={scoreA} hours={hoursA} episodes={episodesA} />

          <div className={`comparison-delta comparison-delta-${winner?.toLowerCase() ?? "none"}`}>
            <span className="delta-rule" />
            {delta === null ? (
              <div><strong>—</strong><small>AWAITING RESULT</small></div>
            ) : (
              <div>
                <small>{winner ? `${winner} LEADS` : "EVEN"}</small>
                <strong><AnimatedNumber value={Math.abs(delta)} /></strong>
                <span>points</span>
              </div>
            )}
            <span className="delta-rule" />
          </div>

          <DatasetIdentity side="B" name={subsetB?.source ?? "Scale"} score={scoreB} hours={hoursB} episodes={episodesB} />
        </div>

        {result && strongestA && strongestB ? (
          <div className="comparison-thesis">
            <span className="thesis-a">
              A +{Math.abs(strongestA.delta).toFixed(1)} {strongestA.shortLabel ?? strongestA.label}
            </span>
            <ArrowRight aria-hidden="true" />
            <p>
              {winner === "A"
                ? `${subsetA!.source} leads under the current weights; ${subsetB!.source} remains stronger in ${strongestB.shortLabel ?? strongestB.label}.`
                : winner === "B"
                  ? `${subsetB!.source} leads under the current weights; ${subsetA!.source} remains stronger in ${strongestA.shortLabel ?? strongestA.label}.`
                  : `The current weights produce an even ranking, while the dimension profiles remain different.`}
            </p>
            <ArrowRight aria-hidden="true" />
            <span className="thesis-b">
              B +{strongestB.delta.toFixed(1)} {strongestB.shortLabel ?? strongestB.label}
            </span>
          </div>
        ) : null}
      </section>

      <section className="profile-workspace" aria-labelledby="profile-title">
        <div className="profile-main">
          <div className="section-heading">
            <div>
              <p className="section-kicker">Diversity profile</p>
              <h2 id="profile-title">{dimensionDefinitions.length} measurements, one ranking</h2>
            </div>
            <p>Each definition is shown below. Select a row to inspect its method and backend evidence.</p>
          </div>

          {result ? (
            <DimensionChart
              result={result}
              selectedDimension={selectedDimension}
              onSelect={setSelectedDimension}
            />
          ) : (
            <div className="profile-skeleton" aria-label="Awaiting validated CSV output">
              {dimensionDefinitions.map((dimension) => (
                <div key={dimension.id}><span>{dimension.shortLabel ?? dimension.label}</span><i /><i /></div>
              ))}
              <p>Awaiting validated CSV output</p>
            </div>
          )}
        </div>

        <WeightControls
          weights={weights}
          dimensions={dimensionDefinitions}
          onChange={setWeights}
        />
      </section>

      {result ? (
        <EvidencePanel
          result={result}
          selectedDimension={selectedDimension}
          onSelect={setSelectedDimension}
        />
      ) : (
        <section className="evidence-empty">
          <div>
            <p className="section-kicker">Supporting evidence</p>
            <h2>Evidence appears after analysis</h2>
          </div>
          <p>Only richness, coverage, evenness, visual distance, episode count, and duration from the validated CSV outputs will appear here.</p>
        </section>
      )}

      <section className="method-strip" aria-label="Method and run context">
        <div className="method-heading">
          <span>METHOD / RUN CONTEXT</span>
          <strong>{job?.jobId ? `RUN ${job.jobId.slice(0, 8).toUpperCase()}` : "NO ACTIVE RUN"}</strong>
        </div>
        <div><span>Pipeline</span><strong>{BACKEND_CONFIGURATION.command}</strong></div>
        <div><span>Visual sample</span><strong>{result?.backend.visualSampleSize ?? BACKEND_CONFIGURATION.visualSampleSize} / subset</strong></div>
        <div><span>Failures</span><strong>{visualFailures ?? "—"}</strong></div>
        <div><span>Duration</span><strong>{result?.backend.durationMethod ?? BACKEND_CONFIGURATION.durationMethod}</strong></div>
        <div><span>Selection</span><strong>Oldest-first</strong></div>
      </section>

      <footer className="console-footer">
        <span><Check aria-hidden="true" /> Scores are read from backend CSV outputs</span>
        <span>Frontend weighting only · no ML rerun</span>
      </footer>
    </main>
  );
}
