import { z } from "zod";

const boundedScore = z.coerce.number().min(0).max(1);
const nonNegativeNumber = z.coerce.number().nonnegative();
const nonNegativeInteger = z.coerce.number().int().nonnegative();

export const datasetSummaryCsvRowSchema = z.object({
  dataset: z.string().min(1),
  episodes: nonNegativeInteger,
  hours: nonNegativeNumber,
  raw_unique_tasks: nonNegativeInteger,
  labs: nonNegativeInteger,
  embodiments: nonNegativeInteger,
  unique_video_paths: nonNegativeInteger,
});

export const diversityResultCsvRowSchema = z.object({
  rank: z.coerce.number().int().positive(),
  subset: z.string().min(1),
  overall_diversity: boundedScore,
  behavior_diversity: boundedScore,
  context_visual_diversity: boundedScore,
  embodiment_diversity: boundedScore,
  behavior_richness: nonNegativeInteger,
  behavior_coverage: boundedScore,
  behavior_evenness: boundedScore,
  reference_behavior_clusters: nonNegativeInteger,
  embodiment_richness: nonNegativeInteger,
  embodiment_coverage: boundedScore,
  embodiment_evenness: boundedScore,
  visual_raw_distance: nonNegativeNumber,
  context_relative_spread: nonNegativeNumber,
  reference_visual_distance: nonNegativeNumber,
  visual_n: nonNegativeInteger,
  visual_failures: nonNegativeInteger,
});

const subsetIdSchema = z.enum(["subset-a", "subset-b"]);
const dimensionIdSchema = z
  .string()
  .min(1)
  .regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/, "Dimension IDs must be stable kebab-case identifiers");

const scorePairSchema = z.object({
  raw: boundedScore,
  display: z.number().min(0).max(100),
});

export const subsetResultSchema = z.object({
  id: subsetIdSchema,
  label: z.string(),
  shortLabel: z.enum(["A", "B"]),
  source: z.string(),
  backendRank: z.number().int().positive(),
  backendOverall: scorePairSchema,
  dimensions: z.record(dimensionIdSchema, scorePairSchema),
  dataset: z.object({
    episodeCount: nonNegativeInteger,
    estimatedHours: nonNegativeNumber,
    rawUniqueTasks: nonNegativeInteger,
    labCount: nonNegativeInteger,
    embodimentCount: nonNegativeInteger,
    uniqueVideoPaths: nonNegativeInteger,
  }),
  evidence: z.object({
    behavior: z.object({
      richness: nonNegativeInteger,
      coverage: boundedScore,
      evenness: boundedScore,
      referenceClusterCount: nonNegativeInteger,
    }).optional(),
    visual: z.object({
      meanPairwiseDistance: nonNegativeNumber,
      relativeSpread: nonNegativeNumber,
      referenceDistance: nonNegativeNumber,
      successfulEpisodes: nonNegativeInteger,
      failedEpisodes: nonNegativeInteger,
    }).optional(),
    embodiment: z.object({
      richness: nonNegativeInteger,
      coverage: boundedScore,
      evenness: boundedScore,
    }).optional(),
  }),
});

export const analysisResultSchema = z.object({
  schemaVersion: z.literal("egoverse-diversity-ui/v1"),
  jobId: z.string().min(1),
  generatedAt: z.string().datetime(),
  comparison: z.object({
    leftSubsetId: z.literal("subset-a"),
    rightSubsetId: z.literal("subset-b"),
  }),
  subsets: z.array(subsetResultSchema).length(2),
  dimensions: z.array(
    z.object({
      id: dimensionIdSchema,
      label: z.string(),
      shortLabel: z.string().optional(),
      description: z.string(),
    }),
  ).min(1),
  backend: z.object({
    command: z.string(),
    targetHours: z.number(),
    randomState: z.number(),
    visualSampleSize: z.number(),
    durationMethod: z.string(),
    selectionStrategy: z.string(),
  }),
  warnings: z.array(z.string()),
}).superRefine((result, context) => {
  const dimensionIds = result.dimensions.map((dimension) => dimension.id);
  const uniqueIds = new Set(dimensionIds);
  if (uniqueIds.size !== dimensionIds.length) {
    context.addIssue({
      code: "custom",
      path: ["dimensions"],
      message: "Dimension IDs must be unique.",
    });
  }

  for (const [subsetIndex, subset] of result.subsets.entries()) {
    const measurementIds = Object.keys(subset.dimensions);
    const missing = dimensionIds.filter((id) => !(id in subset.dimensions));
    const unknown = measurementIds.filter((id) => !uniqueIds.has(id));
    if (missing.length || unknown.length) {
      context.addIssue({
        code: "custom",
        path: ["subsets", subsetIndex, "dimensions"],
        message: `Dimension measurements do not match the declared result dimensions. Missing: ${missing.join(", ") || "none"}; unknown: ${unknown.join(", ") || "none"}.`,
      });
    }
  }
});

export const jobStateSchema = z.enum(["queued", "running", "completed", "failed"]);

export const jobStatusSchema = z.object({
  jobId: z.string(),
  state: jobStateSchema,
  createdAt: z.string().datetime(),
  updatedAt: z.string().datetime(),
  startedAt: z.string().datetime().optional(),
  completedAt: z.string().datetime().optional(),
  error: z.string().nullable().optional(),
});

export type DatasetSummaryCsvRow = z.infer<typeof datasetSummaryCsvRowSchema>;
export type DiversityResultCsvRow = z.infer<typeof diversityResultCsvRowSchema>;
export type AnalysisResult = z.infer<typeof analysisResultSchema>;
export type SubsetResult = z.infer<typeof subsetResultSchema>;
export type JobStatus = z.infer<typeof jobStatusSchema>;
