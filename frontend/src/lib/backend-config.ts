export const BACKEND_CONFIGURATION = {
  track: "Track 2 · Quantitative Diversity Measurement",
  command: "python -m track2.run_track2",
  targetHours: 120,
  visualSampleSize: 200,
  visualFramesPerEpisode: 5,
  randomState: 42,
  durationMethod: "num_frames / 30 FPS",
  selectionStrategy: "Deterministic oldest-first within each configured lab",
  subsets: [
    {
      id: "subset-a" as const,
      shortLabel: "A",
      label: "Dataset A — Mecka",
      source: "Mecka",
    },
    {
      id: "subset-b" as const,
      shortLabel: "B",
      label: "Dataset B — Scale",
      source: "Scale",
    },
  ],
} as const;

export const DIMENSIONS = [
  {
    id: "behavior" as const,
    label: "Behavior Diversity",
    shortLabel: "Behavior",
    description: "Semantic task-cluster coverage × evenness",
  },
  {
    id: "visual" as const,
    label: "Context / Visual Diversity",
    shortLabel: "Visual",
    description: "Calibrated mean pairwise DINOv2 distance",
  },
  {
    id: "embodiment" as const,
    label: "Embodiment Diversity",
    shortLabel: "Embodiment",
    description: "Embodiment-category coverage × evenness",
  },
] as const;

export type SubsetId = (typeof BACKEND_CONFIGURATION.subsets)[number]["id"];
export type DimensionId = (typeof DIMENSIONS)[number]["id"];
