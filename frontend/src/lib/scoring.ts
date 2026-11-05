import type { AnalysisResult } from "@/lib/contracts";
import type { SubsetId } from "@/lib/backend-config";

export type Weights = Record<string, number>;

export function equalWeights(dimensionIds: readonly string[]): Weights {
  if (dimensionIds.length === 0) throw new Error("At least one dimension is required.");
  const share = 100 / dimensionIds.length;
  return Object.fromEntries(dimensionIds.map((id) => [id, share]));
}

export const DEFAULT_WEIGHTS: Weights = equalWeights([
  "behavior",
  "visual",
  "embodiment",
]);

export function weightTotal(weights: Weights): number {
  return Object.values(weights).reduce((sum, value) => sum + value, 0);
}

export function validateWeights(
  weights: Weights,
  dimensionIds: readonly string[] = Object.keys(weights),
): boolean {
  const weightIds = Object.keys(weights);
  const declaredIds = new Set(dimensionIds);
  return (
    weightIds.length === dimensionIds.length &&
    weightIds.every((id) => declaredIds.has(id)) &&
    Object.values(weights).every((value) => Number.isFinite(value) && value >= 0 && value <= 100) &&
    Math.abs(weightTotal(weights) - 100) < 0.01
  );
}

export function calculateWeightedScores(
  result: AnalysisResult,
  weights: Weights,
): Record<SubsetId, number> {
  const dimensionIds = result.dimensions.map((dimension) => dimension.id);
  if (!validateWeights(weights, dimensionIds)) {
    throw new Error("Weights must match all returned dimensions and sum to 100%.");
  }

  return Object.fromEntries(
    result.subsets.map((subset) => {
      const score = dimensionIds.reduce(
        (sum, dimensionId) =>
          sum + subset.dimensions[dimensionId].display * (weights[dimensionId] / 100),
        0,
      );
      return [subset.id, score];
    }),
  ) as Record<SubsetId, number>;
}

export function rebalanceWeights(
  current: Weights,
  changed: string,
  nextValue: number,
): Weights {
  if (!(changed in current)) throw new Error(`Unknown dimension weight: ${changed}`);

  const clamped = Math.max(0, Math.min(100, nextValue));
  const others = Object.keys(current).filter((id) => id !== changed);
  if (others.length === 0) return { [changed]: 100 };

  const remaining = 100 - clamped;
  const otherTotal = others.reduce((sum, id) => sum + current[id], 0);
  const next: Weights = { [changed]: clamped };
  let allocated = 0;

  others.forEach((id, index) => {
    const value =
      index === others.length - 1
        ? remaining - allocated
        : otherTotal === 0
          ? remaining / others.length
          : remaining * (current[id] / otherTotal);
    next[id] = value;
    allocated += value;
  });

  return next;
}
