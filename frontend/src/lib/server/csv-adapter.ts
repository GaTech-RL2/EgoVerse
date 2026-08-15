import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { parse } from "csv-parse/sync";

import { BACKEND_CONFIGURATION, DIMENSIONS, type SubsetId } from "@/lib/backend-config";
import {
  analysisResultSchema,
  datasetManifestCsvRowSchema,
  diversityResultCsvRowSchema,
  type AnalysisResult,
  type DatasetManifestCsvRow,
  type DiversityResultCsvRow,
} from "@/lib/contracts";

export const REPO_ROOT = path.resolve(
  process.env.EGOVERSE_REPO_ROOT ?? path.join(process.cwd(), ".."),
);

const RANDOM_RESULTS_ROOT = path.join(REPO_ROOT, "track2/results/single_random_120h");

export const RESULT_PATHS = {
  scores: path.join(RANDOM_RESULTS_ROOT, "single_random_120h_results.csv"),
  manifests: {
    "subset-a": path.join(RANDOM_RESULTS_ROOT, "manifests/mecka_seed_42.csv"),
    "subset-b": path.join(RANDOM_RESULTS_ROOT, "manifests/scale_seed_42.csv"),
  },
} as const;

function parseCsv<T>(contents: string, rowParser: (row: unknown) => T): T[] {
  const rows = parse(contents, {
    columns: true,
    skip_empty_lines: true,
    trim: true,
  }) as unknown[];
  return rows.map(rowParser);
}

function subsetIdForLabel(label: string): SubsetId {
  if (/^Dataset A\b/.test(label)) return "subset-a";
  if (/^Dataset B\b/.test(label)) return "subset-b";
  throw new Error(`Unsupported backend subset label: ${label}`);
}

function score(raw: number) {
  return { raw, display: raw * 100 };
}

function summarizeManifest(
  rows: DatasetManifestCsvRow[],
): AnalysisResult["subsets"][number]["dataset"] {
  if (rows.length === 0) throw new Error("Track 2 produced an empty dataset manifest.");

  return {
    episodeCount: rows.length,
    estimatedHours: rows.reduce((total, row) => total + row.duration_hours, 0),
    rawUniqueTasks: new Set(rows.map((row) => row.task)).size,
    labCount: new Set(rows.map((row) => row.lab)).size,
    embodimentCount: new Set(rows.map((row) => row.embodiment)).size,
    uniqueVideoPaths: new Set(rows.map((row) => row.zarr_mp4_path)).size,
  };
}

function adaptSubset(
  id: SubsetId,
  dataset: AnalysisResult["subsets"][number]["dataset"],
  result: DiversityResultCsvRow,
): AnalysisResult["subsets"][number] {
  if (subsetIdForLabel(result.subset) !== id) throw new Error(`Manifest/result subset mismatch for ${id}`);
  const configured = BACKEND_CONFIGURATION.subsets.find((subset) => subset.id === id);
  if (!configured) throw new Error(`Missing frontend configuration for ${id}`);

  return {
    id,
    label: result.subset,
    shortLabel: configured.shortLabel,
    source: configured.source,
    backendRank: result.rank,
    backendOverall: score(result.overall_diversity),
    dimensions: {
      behavior: score(result.behavior_diversity),
      visual: score(result.context_visual_diversity),
      embodiment: score(result.embodiment_diversity),
    },
    dataset,
    evidence: {
      behavior: {
        richness: result.behavior_richness,
        coverage: result.behavior_coverage,
        evenness: result.behavior_evenness,
        referenceClusterCount: result.reference_behavior_clusters,
      },
      visual: {
        meanPairwiseDistance: result.visual_raw_distance,
        relativeSpread: result.context_relative_spread,
        referenceDistance: result.reference_visual_distance,
        successfulEpisodes: result.visual_n,
        failedEpisodes: result.visual_failures,
      },
      embodiment: {
        richness: result.embodiment_richness,
        coverage: result.embodiment_coverage,
        evenness: result.embodiment_evenness,
      },
    },
  };
}

export async function readArtifactMtimes() {
  const [scores, manifestA, manifestB] = await Promise.all([
    stat(/* turbopackIgnore: true */ RESULT_PATHS.scores).catch(() => null),
    stat(/* turbopackIgnore: true */ RESULT_PATHS.manifests["subset-a"]).catch(() => null),
    stat(/* turbopackIgnore: true */ RESULT_PATHS.manifests["subset-b"]).catch(() => null),
  ]);
  return {
    scores: scores?.mtimeMs ?? null,
    manifestA: manifestA?.mtimeMs ?? null,
    manifestB: manifestB?.mtimeMs ?? null,
  };
}

export async function adaptCurrentCsvResult(jobId: string): Promise<AnalysisResult> {
  const [manifestAContents, manifestBContents, scoreContents, mtimes] = await Promise.all([
    readFile(/* turbopackIgnore: true */ RESULT_PATHS.manifests["subset-a"], "utf8"),
    readFile(/* turbopackIgnore: true */ RESULT_PATHS.manifests["subset-b"], "utf8"),
    readFile(/* turbopackIgnore: true */ RESULT_PATHS.scores, "utf8"),
    readArtifactMtimes(),
  ]);

  const artifactGeneratedAt = Math.max(
    mtimes.scores ?? 0,
    mtimes.manifestA ?? 0,
    mtimes.manifestB ?? 0,
  );
  return adaptRandomCsvContents(
    jobId,
    manifestAContents,
    manifestBContents,
    scoreContents,
    new Date(artifactGeneratedAt || Date.now()).toISOString(),
  );
}

export function adaptRandomCsvContents(
  jobId: string,
  manifestAContents: string,
  manifestBContents: string,
  scoreContents: string,
  generatedAt: string,
): AnalysisResult {
  const manifestById = new Map<SubsetId, DatasetManifestCsvRow[]>([
    [
      "subset-a",
      parseCsv(manifestAContents, (row) => datasetManifestCsvRowSchema.parse(row)),
    ],
    [
      "subset-b",
      parseCsv(manifestBContents, (row) => datasetManifestCsvRowSchema.parse(row)),
    ],
  ]);
  const results = parseCsv(scoreContents, (row) => diversityResultCsvRowSchema.parse(row));
  if (results.length !== 2) throw new Error("Expected exactly two rows in the Track 2 result CSV.");

  const resultById = new Map(results.map((row) => [subsetIdForLabel(row.subset), row]));
  const subsets = (["subset-a", "subset-b"] as const).map((id) => {
    const manifest = manifestById.get(id);
    const result = resultById.get(id);
    if (!manifest || !result) throw new Error(`Missing backend output for ${id}`);
    return adaptSubset(id, summarizeManifest(manifest), result);
  });

  const warnings = [
    "Subset duration is estimated from num_frames at a fixed 30 FPS.",
    "Each lab is randomly shuffled with seed 42, then episodes are selected until cumulative duration reaches 120 hours.",
    "Visual diversity is calculated from a random sample of 100 unique episode videos per subset using seed 42.",
  ];
  for (const subset of subsets) {
    const visualEvidence = subset.evidence.visual;
    if (visualEvidence && visualEvidence.failedEpisodes > 0) {
      warnings.push(
        `${subset.shortLabel}: ${visualEvidence.failedEpisodes} visual episodes failed during embedding.`,
      );
    }
  }

  return analysisResultSchema.parse({
    schemaVersion: "egoverse-diversity-ui/v1",
    jobId,
    generatedAt,
    comparison: { leftSubsetId: "subset-a", rightSubsetId: "subset-b" },
    subsets,
    dimensions: DIMENSIONS.map(({ id, label, shortLabel, description }) => ({
      id,
      label,
      shortLabel,
      description,
    })),
    backend: {
      command: BACKEND_CONFIGURATION.command,
      targetHours: BACKEND_CONFIGURATION.targetHours,
      randomState: BACKEND_CONFIGURATION.randomState,
      visualSampleSize: BACKEND_CONFIGURATION.visualSampleSize,
      durationMethod: BACKEND_CONFIGURATION.durationMethod,
      selectionStrategy: BACKEND_CONFIGURATION.selectionStrategy,
    },
    warnings,
  });
}

export async function readCurrentDatasetSnapshot() {
  try {
    const [manifestAContents, manifestBContents] = await Promise.all([
      readFile(/* turbopackIgnore: true */ RESULT_PATHS.manifests["subset-a"], "utf8"),
      readFile(/* turbopackIgnore: true */ RESULT_PATHS.manifests["subset-b"], "utf8"),
    ]);
    const manifests = new Map<SubsetId, DatasetManifestCsvRow[]>([
      ["subset-a", parseCsv(manifestAContents, (row) => datasetManifestCsvRowSchema.parse(row))],
      ["subset-b", parseCsv(manifestBContents, (row) => datasetManifestCsvRowSchema.parse(row))],
    ]);

    return Object.fromEntries(
      BACKEND_CONFIGURATION.subsets.map((subset) => {
        const manifest = manifests.get(subset.id);
        if (!manifest) throw new Error(`Missing manifest for ${subset.id}`);
        const summary = summarizeManifest(manifest);
        return [
          subset.id,
          {
            label: subset.label,
            episodeCount: summary.episodeCount,
            estimatedHours: summary.estimatedHours,
          },
        ];
      }),
    );
  } catch {
    return null;
  }
}
