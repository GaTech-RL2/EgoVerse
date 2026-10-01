import { describe, expect, it } from "vitest";

import type { AnalysisResult } from "./contracts";
import {
  calculateWeightedScores,
  equalWeights,
  rebalanceWeights,
  validateWeights,
} from "./scoring";

const result = {
  dimensions: [
    { id: "behavior" },
    { id: "visual" },
    { id: "embodiment" },
  ],
  subsets: [
    {
      id: "subset-a",
      dimensions: {
        behavior: { display: 20 },
        visual: { display: 80 },
        embodiment: { display: 50 },
      },
    },
    {
      id: "subset-b",
      dimensions: {
        behavior: { display: 70 },
        visual: { display: 40 },
        embodiment: { display: 60 },
      },
    },
  ],
} as unknown as AnalysisResult;

describe("frontend-only weighted scoring", () => {
  it("recomputes both subsets from immutable dimension scores", () => {
    const scores = calculateWeightedScores(result, {
      behavior: 50,
      visual: 30,
      embodiment: 20,
    });
    expect(scores["subset-a"]).toBe(44);
    expect(scores["subset-b"]).toBe(59);
  });

  it("keeps rebalanced weights at 100 percent", () => {
    const next = rebalanceWeights(
      { behavior: 100 / 3, visual: 100 / 3, embodiment: 100 / 3 },
      "behavior",
      50,
    );
    expect(validateWeights(next)).toBe(true);
    expect(next.behavior).toBe(50);
    expect(next.visual).toBeCloseTo(25);
    expect(next.embodiment).toBeCloseTo(25);
  });

  it("rejects invalid totals", () => {
    expect(() =>
      calculateWeightedScores(result, {
        behavior: 50,
        visual: 50,
        embodiment: 50,
      }),
    ).toThrow("Weights must match all returned dimensions and sum to 100%");
  });

  it("creates equal weights for any returned dimension count", () => {
    const weights = equalWeights(["behavior", "visual", "embodiment", "temporal"]);
    expect(Object.values(weights)).toEqual([25, 25, 25, 25]);
    expect(validateWeights(weights, ["behavior", "visual", "embodiment", "temporal"])).toBe(true);
  });

  it("rejects weights for unknown or missing dimensions", () => {
    expect(validateWeights(
      { behavior: 50, visual: 25, unknown: 25 },
      ["behavior", "visual", "embodiment"],
    )).toBe(false);
  });
});
