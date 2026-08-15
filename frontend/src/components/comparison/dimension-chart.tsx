"use client";

import { ArrowLeft, ArrowRight } from "lucide-react";

import { getDimensionGuide } from "@/lib/backend-config";
import type { AnalysisResult } from "@/lib/contracts";

export function DimensionChart({
  result,
  selectedDimension,
  onSelect,
}: {
  result: AnalysisResult;
  selectedDimension: string;
  onSelect: (dimension: string) => void;
}) {
  const subsetA = result.subsets.find((subset) => subset.id === "subset-a")!;
  const subsetB = result.subsets.find((subset) => subset.id === "subset-b")!;

  return (
    <div className="profile-rows" aria-label="Diversity profile comparison">
      <div className="profile-axis" aria-hidden="true">
        <span>A / Mecka</span>
        <span>score / 100</span>
        <span>B / Scale</span>
      </div>

      {result.dimensions.map((dimension) => {
        const guide = getDimensionGuide(dimension.id);
        const a = subsetA.dimensions[dimension.id].display;
        const b = subsetB.dimensions[dimension.id].display;
        const delta = b - a;
        const winner = Math.abs(delta) < 0.05 ? null : delta > 0 ? "B" : "A";

        return (
          <button
            key={dimension.id}
            type="button"
            className={`profile-row ${selectedDimension === dimension.id ? "is-active" : ""}`}
            onClick={() => onSelect(dimension.id)}
            aria-pressed={selectedDimension === dimension.id}
          >
            <span className="profile-label">
              <strong>{dimension.shortLabel ?? dimension.label}</strong>
              <small>{guide?.definition ?? dimension.description}</small>
            </span>

            <span className="profile-plot">
              <span className="profile-side profile-side-a">
                <span className="profile-score profile-score-a">{a.toFixed(1)}</span>
                <span className="profile-track">
                  <span className="profile-fill profile-fill-a" style={{ width: `${a}%` }} />
                </span>
              </span>

              <span className="profile-center" aria-hidden="true" />

              <span className="profile-side profile-side-b">
                <span className="profile-track">
                  <span className="profile-fill profile-fill-b" style={{ width: `${b}%` }} />
                </span>
                <span className="profile-score profile-score-b">{b.toFixed(1)}</span>
              </span>
            </span>

            <span className={`profile-delta profile-delta-${winner?.toLowerCase() ?? "tie"}`}>
              {winner === "A" ? <ArrowLeft aria-hidden="true" /> : null}
              {winner ? `${winner} +${Math.abs(delta).toFixed(1)}` : "EVEN"}
              {winner === "B" ? <ArrowRight aria-hidden="true" /> : null}
            </span>
          </button>
        );
      })}
    </div>
  );
}
