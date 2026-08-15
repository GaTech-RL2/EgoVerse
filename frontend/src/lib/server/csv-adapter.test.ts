import { describe, expect, it } from "vitest";

import { adaptCsvContents } from "@/lib/server/csv-adapter";

const summaryCsv = `dataset,episodes,hours,raw_unique_tasks,labs,embodiments,unique_video_paths
Dataset A — Mecka,11103,120.025,738,1,1,11103
Dataset B — Scale,6089,120.010,60,1,3,6089
`;

const resultCsv = `rank,subset,overall_diversity,behavior_diversity,context_visual_diversity,embodiment_diversity,behavior_richness,behavior_coverage,behavior_evenness,reference_behavior_clusters,embodiment_richness,embodiment_coverage,embodiment_evenness,visual_raw_distance,context_relative_spread,reference_visual_distance,visual_n,visual_failures
1,Dataset B — Scale,0.332,0.0068,0.7936,0.1956,55,0.009,0.753,10772,3,0.5,0.391,0.514,0.794,0.6474,200,0
2,Dataset A — Mecka,0.3164,0.0258,0.9236,0.0,605,0.056,0.459,10772,1,0.1667,0.0,0.598,0.924,0.6474,200,0
`;

describe("Track 2 CSV adapter", () => {
  it("matches A/B by labels and converts only bounded scores to 0-100", () => {
    const result = adaptCsvContents(
      "job-fixture",
      summaryCsv,
      resultCsv,
      "2026-08-15T18:00:00.000Z",
    );

    const a = result.subsets.find((subset) => subset.id === "subset-a")!;
    const b = result.subsets.find((subset) => subset.id === "subset-b")!;
    expect(a.source).toBe("Mecka");
    expect(b.backendRank).toBe(1);
    expect(a.dimensions.visual.raw).toBe(0.9236);
    expect(a.dimensions.visual.display).toBeCloseTo(92.36);
    expect(a.evidence.visual!.meanPairwiseDistance).toBe(0.598);
    expect(a.evidence.visual!.relativeSpread).toBe(0.924);
  });

  it("rejects missing or unknown subset identities", () => {
    expect(() =>
      adaptCsvContents(
        "job-fixture",
        summaryCsv.replace("Dataset A — Mecka", "Unknown subset"),
        resultCsv,
        "2026-08-15T18:00:00.000Z",
      ),
    ).toThrow("Unsupported backend subset label");
  });
});
