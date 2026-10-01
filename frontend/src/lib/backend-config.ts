export const BACKEND_CONFIGURATION = {
  track: "Track 2 · Quantitative Diversity Measurement",
  command: "python -m track2.run_lab_random_120h_single",
  targetHours: 120,
  visualSampleSize: 100,
  visualFramesPerEpisode: 5,
  randomState: 42,
  durationMethod: "num_frames / 30 FPS",
  selectionStrategy: "Single-seed random shuffle within each configured lab",
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
    description: "Variety and balance of semantic task groups",
  },
  {
    id: "visual" as const,
    label: "Context / Visual Diversity",
    shortLabel: "Visual",
    description: "Visual dissimilarity across sampled episodes",
  },
  {
    id: "embodiment" as const,
    label: "Embodiment Diversity",
    shortLabel: "Embodiment",
    description: "Variety and balance of embodiment categories",
  },
] as const;

export type SubsetId = (typeof BACKEND_CONFIGURATION.subsets)[number]["id"];
export type DimensionId = (typeof DIMENSIONS)[number]["id"];

export type DimensionGuide = {
  definition: string;
  input: string;
  score: string;
  boundary: string;
};

export const DIMENSION_GUIDES: Record<DimensionId, DimensionGuide> = {
  behavior: {
    definition: "How many semantic task groups are represented, and how evenly episodes are spread across those groups.",
    input: "Episode task labels · MiniLM semantic clusters",
    score: "Behavior-cluster coverage × normalized evenness",
    boundary: "Not action quality or motion extracted from the video.",
  },
  visual: {
    definition: "How visually different 100 sampled episode videos are from one another.",
    input: "100 videos / subset · 5 frames each · DINOv2-small",
    score: "Mean pairwise cosine distance ÷ reference distance",
    boundary: "Not named scene categories, PCA, or proof of which visual factor caused the difference.",
  },
  embodiment: {
    definition: "How many embodiment categories are represented, and how evenly episodes are spread across them.",
    input: "The dataset's embodiment metadata field",
    score: "Embodiment-category coverage × normalized evenness",
    boundary: "Not robot performance. A zero can mean every episode uses one embodiment category.",
  },
};

export function getDimensionGuide(id: string): DimensionGuide | undefined {
  return DIMENSION_GUIDES[id as DimensionId];
}
