"use client";

import { RotateCcw } from "lucide-react";

import { Slider } from "@/components/ui/slider";
import type { AnalysisResult } from "@/lib/contracts";
import { equalWeights, rebalanceWeights, type Weights, weightTotal } from "@/lib/scoring";

export function WeightControls({
  weights,
  dimensions,
  onChange,
}: {
  weights: Weights;
  dimensions: readonly AnalysisResult["dimensions"][number][];
  onChange: (weights: Weights) => void;
}) {
  return (
    <aside className="weight-inspector" aria-labelledby="weighting-title">
      <div className="inspector-heading">
        <div>
          <p className="section-kicker">Interactive model</p>
          <h2 id="weighting-title">Model weighting</h2>
        </div>
        <span className="weight-total">{weightTotal(weights).toFixed(0)}%</span>
      </div>

      <p className="inspector-copy">
        Rebalance the measured dimensions. Scores update locally without rerunning Track 2.
      </p>

      <div className="weight-list">
        {dimensions.map((dimension) => (
          <div className="weight-row" key={dimension.id}>
            <div className="weight-row-label">
              <label htmlFor={`weight-${dimension.id}`}>{dimension.shortLabel ?? dimension.label}</label>
              <output htmlFor={`weight-${dimension.id}`}>
                {weights[dimension.id].toFixed(0)}
              </output>
            </div>
            <Slider
              id={`weight-${dimension.id}`}
              min={0}
              max={100}
              step={1}
              value={[weights[dimension.id]]}
              onValueChange={([value]) =>
                onChange(rebalanceWeights(weights, dimension.id, value))
              }
              aria-label={`${dimension.label} weight`}
            />
          </div>
        ))}
      </div>

      <div className="inspector-footer">
        <span><i /> Total locked at 100%</span>
        <button
          type="button"
          onClick={() => onChange(equalWeights(dimensions.map((dimension) => dimension.id)))}
        >
          <RotateCcw aria-hidden="true" /> Equal weights
        </button>
      </div>
    </aside>
  );
}
