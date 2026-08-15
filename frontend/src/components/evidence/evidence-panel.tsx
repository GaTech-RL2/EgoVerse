"use client";

import { Info } from "lucide-react";

import { getDimensionGuide } from "@/lib/backend-config";
import type { AnalysisResult } from "@/lib/contracts";

function DimensionGuidePanel({
  dimensionId,
  label,
}: {
  dimensionId: string;
  label: string;
}) {
  const guide = getDimensionGuide(dimensionId);

  if (!guide) {
    return null;
  }

  return (
    <div className="dimension-guide" aria-label={`How to read ${label}`}>
      <div className="dimension-guide-definition">
        <Info aria-hidden="true" />
        <div>
          <span>What it measures</span>
          <p>{guide.definition}</p>
        </div>
      </div>
      <dl className="dimension-guide-details">
        <div>
          <dt>Input</dt>
          <dd>{guide.input}</dd>
        </div>
        <div>
          <dt>Score</dt>
          <dd>{guide.score}</dd>
        </div>
        <div>
          <dt>Does not mean</dt>
          <dd>{guide.boundary}</dd>
        </div>
      </dl>
    </div>
  );
}

function MetricBar({
  label,
  a,
  b,
  formatter,
  max = Math.max(a, b, 1),
}: {
  label: string;
  a: number;
  b: number;
  formatter: (value: number) => string;
  max?: number;
}) {
  return (
    <div className="evidence-metric">
      <div className="evidence-metric-label">{label}</div>
      <div className="evidence-bars">
        <div className="evidence-bar-row">
          <span className="subset-mark subset-mark-a">A</span>
          <span className="evidence-bar-track">
            <span className="evidence-bar-fill evidence-bar-a" style={{ width: `${Math.min((a / max) * 100, 100)}%` }} />
          </span>
          <strong className="metric-a">{formatter(a)}</strong>
        </div>
        <div className="evidence-bar-row">
          <span className="subset-mark subset-mark-b">B</span>
          <span className="evidence-bar-track">
            <span className="evidence-bar-fill evidence-bar-b" style={{ width: `${Math.min((b / max) * 100, 100)}%` }} />
          </span>
          <strong className="metric-b">{formatter(b)}</strong>
        </div>
      </div>
    </div>
  );
}

function BehaviorEvidence({ result }: { result: AnalysisResult }) {
  const a = result.subsets.find((subset) => subset.id === "subset-a")!;
  const b = result.subsets.find((subset) => subset.id === "subset-b")!;
  const evidenceA = a.evidence.behavior!;
  const evidenceB = b.evidence.behavior!;

  return (
    <div className="evidence-layout evidence-enter">
      <div className="evidence-story">
        <p className="evidence-overline">Observed behavior space</p>
        <h3>A reaches more concepts. B distributes its smaller set more evenly.</h3>
        <p>
          Mecka observes {evidenceA.richness.toLocaleString()} behavior clusters versus Scale&apos;s {evidenceB.richness.toLocaleString()}, while Scale has the higher evenness score.
        </p>
        <div className="reference-note">
          <span>Reference space</span>
          <strong>{evidenceA.referenceClusterCount.toLocaleString()} clusters</strong>
        </div>
      </div>
      <div className="evidence-chart-stack">
        <MetricBar label="Richness" a={evidenceA.richness} b={evidenceB.richness} formatter={(value) => value.toLocaleString()} />
        <MetricBar label="Coverage" a={evidenceA.coverage} b={evidenceB.coverage} max={1} formatter={(value) => `${(value * 100).toFixed(2)}%`} />
        <MetricBar label="Evenness" a={evidenceA.evenness} b={evidenceB.evenness} max={1} formatter={(value) => `${(value * 100).toFixed(1)}%`} />
      </div>
    </div>
  );
}

function VisualEvidence({ result }: { result: AnalysisResult }) {
  const a = result.subsets.find((subset) => subset.id === "subset-a")!;
  const b = result.subsets.find((subset) => subset.id === "subset-b")!;
  const evidenceA = a.evidence.visual!;
  const evidenceB = b.evidence.visual!;
  const reference = evidenceA.referenceDistance;

  return (
    <div className="evidence-layout evidence-enter">
      <div className="evidence-story">
        <p className="evidence-overline">Calibrated visual distance</p>
        <h3>Mecka retains more of the reference visual spread.</h3>
        <p>
          Both subsets use {evidenceA.successfulEpisodes} sampled episodes. Mecka reaches {(evidenceA.relativeSpread * 100).toFixed(1)}% of the reference distance; Scale reaches {(evidenceB.relativeSpread * 100).toFixed(1)}%.
        </p>
        <div className="reference-note">
          <span>Reference distance</span>
          <strong>{reference.toFixed(3)}</strong>
        </div>
      </div>
      <div className="visual-distance-stack">
        <MetricBar label="Mean pairwise distance" a={evidenceA.meanPairwiseDistance} b={evidenceB.meanPairwiseDistance} max={reference} formatter={(value) => value.toFixed(3)} />
        <div className="visual-diagnostics">
          <div><span>Relative spread · A</span><strong className="metric-a">{(evidenceA.relativeSpread * 100).toFixed(1)}%</strong></div>
          <div><span>Relative spread · B</span><strong className="metric-b">{(evidenceB.relativeSpread * 100).toFixed(1)}%</strong></div>
          <div><span>Sample / subset</span><strong>{evidenceA.successfulEpisodes}</strong></div>
          <div><span>Visual failures</span><strong>{evidenceA.failedEpisodes + evidenceB.failedEpisodes}</strong></div>
        </div>
      </div>
    </div>
  );
}

function EmbodimentEvidence({ result }: { result: AnalysisResult }) {
  const a = result.subsets.find((subset) => subset.id === "subset-a")!;
  const b = result.subsets.find((subset) => subset.id === "subset-b")!;
  const evidenceA = a.evidence.embodiment!;
  const evidenceB = b.evidence.embodiment!;

  return (
    <div className="evidence-layout evidence-enter">
      <div className="evidence-story">
        <p className="evidence-overline">Observed embodiment space</p>
        <h3>Scale&apos;s broader embodiment mix creates the decisive advantage.</h3>
        <p>
          Scale observes {evidenceB.richness} embodiment types with {(evidenceB.coverage * 100).toFixed(1)}% coverage. Mecka observes {evidenceA.richness} type with {(evidenceA.coverage * 100).toFixed(1)}% coverage.
        </p>
        <div className="reference-note">
          <span>Observed types</span>
          <strong>{evidenceA.richness} A · {evidenceB.richness} B</strong>
        </div>
      </div>
      <div className="evidence-chart-stack">
        <MetricBar label="Richness" a={evidenceA.richness} b={evidenceB.richness} formatter={(value) => `${value} ${value === 1 ? "type" : "types"}`} />
        <MetricBar label="Coverage" a={evidenceA.coverage} b={evidenceB.coverage} max={1} formatter={(value) => `${(value * 100).toFixed(1)}%`} />
        <MetricBar label="Evenness" a={evidenceA.evenness} b={evidenceB.evenness} max={1} formatter={(value) => `${(value * 100).toFixed(1)}%`} />
      </div>
    </div>
  );
}

export function EvidencePanel({
  result,
  selectedDimension,
  onSelect,
}: {
  result: AnalysisResult;
  selectedDimension: string;
  onSelect: (dimension: string) => void;
}) {
  const a = result.subsets.find((subset) => subset.id === "subset-a")!;
  const b = result.subsets.find((subset) => subset.id === "subset-b")!;
  const selectedDefinition = result.dimensions.find(
    (dimension) => dimension.id === selectedDimension,
  );
  const hasBehaviorEvidence = Boolean(a.evidence.behavior && b.evidence.behavior);
  const hasVisualEvidence = Boolean(a.evidence.visual && b.evidence.visual);
  const hasEmbodimentEvidence = Boolean(a.evidence.embodiment && b.evidence.embodiment);

  return (
    <section className="evidence-panel" aria-labelledby="evidence-title">
      <div className="evidence-header">
        <div>
          <p className="section-kicker">Supporting evidence</p>
          <h2 id="evidence-title">Why the profiles differ</h2>
        </div>
        <div className="dimension-tabs" role="tablist" aria-label="Evidence dimension">
          {result.dimensions.map((dimension) => (
            <button
              key={dimension.id}
              type="button"
              role="tab"
              aria-selected={selectedDimension === dimension.id}
              className={selectedDimension === dimension.id ? "is-active" : ""}
              onClick={() => onSelect(dimension.id)}
            >
              {dimension.shortLabel ?? dimension.label}
            </button>
          ))}
        </div>
      </div>

      {selectedDefinition ? (
        <DimensionGuidePanel
          dimensionId={selectedDimension}
          label={selectedDefinition.label}
        />
      ) : null}

      <div key={selectedDimension} role="tabpanel">
        {selectedDimension === "behavior" && hasBehaviorEvidence ? <BehaviorEvidence result={result} /> : null}
        {selectedDimension === "visual" && hasVisualEvidence ? <VisualEvidence result={result} /> : null}
        {selectedDimension === "embodiment" && hasEmbodimentEvidence ? <EmbodimentEvidence result={result} /> : null}
        {(
          (selectedDimension === "behavior" && !hasBehaviorEvidence) ||
          (selectedDimension === "visual" && !hasVisualEvidence) ||
          (selectedDimension === "embodiment" && !hasEmbodimentEvidence) ||
          !["behavior", "visual", "embodiment"].includes(selectedDimension)
        ) ? (
          <div className="evidence-unavailable evidence-enter">
            <p className="evidence-overline">Evidence unavailable</p>
            <h3>{selectedDefinition?.label ?? selectedDimension}</h3>
            <p>
              This result declares a score for the dimension, but its artifact does not include a
              compatible evidence payload. Missing evidence is not treated as zero.
            </p>
          </div>
        ) : null}
      </div>
    </section>
  );
}
