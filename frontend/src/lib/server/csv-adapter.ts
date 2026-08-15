import { readFile, stat } from "node:fs/promises";
import path from "node:path";
import { parse } from "csv-parse/sync";

import { BACKEND_CONFIGURATION, DIMENSIONS, type SubsetId } from "@/lib/backend-config";
import {
  analysisResultSchema,
  datasetSummaryCsvRowSchema,
  diversityResultCsvRowSchema,
  type AnalysisResult,
  type DatasetSummaryCsvRow,
  type DiversityResultCsvRow,
} from "@/lib/contracts";

export const REPO_ROOT = path.resolve(
  process.env.EGOVERSE_REPO_ROOT ?? path.join(process.cwd(), ".."),
);

export const RESULT_PATHS = {
  summary: path.join(REPO_ROOT, "track2/results/final_dataset_summary.csv"),
  scores: path.join(REPO_ROOT, "track2/results/final_two_dataset_results.csv"),
};

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

function adaptSubset(
  summary: DatasetSummaryCsvRow,
  result: DiversityResultCsvRow,
): AnalysisResult["subsets"][number] {
  const id = subsetIdForLabel(summary.dataset);
  if (subsetIdForLabel(result.subset) !== id) {
    throw new Error(`Summary/result subset mismatch for ${summary.dataset}`);
  }
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
    dataset: {
      episodeCount: summary.episodes,
      estimatedHours: summary.hours,
      rawUniqueTasks: summary.raw_unique_tasks,
      labCount: summary.labs,
      embodimentCount: summary.embodiments,
      uniqueVideoPaths: summary.unique_video_paths,
    },
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
  const [summary, scores] = await Promise.all([
    stat(/* turbopackIgnore: true */ RESULT_PATHS.summary).catch(() => null),
    stat(/* turbopackIgnore: true */ RESULT_PATHS.scores).catch(() => null),
  ]);
  return {
    summary: summary?.mtimeMs ?? null,
    scores: scores?.mtimeMs ?? null,
  };
}

export async function adaptCurrentCsvResult(jobId: string): Promise<AnalysisResult> {
  const [summaryContents, scoreContents, mtimes] = await Promise.all([
    readFile(/* turbopackIgnore: true */ RESULT_PATHS.summary, "utf8"),
    readFile(/* turbopackIgnore: true */ RESULT_PATHS.scores, "utf8"),
    readArtifactMtimes(),
  ]);

  const artifactGeneratedAt = Math.max(mtimes.summary ?? 0, mtimes.scores ?? 0);
  return adaptCsvContents(
    jobId,
    summaryContents,
    scoreContents,
    new Date(artifactGeneratedAt || Date.now()).toISOString(),
  );
}

export function adaptCsvContents(
  jobId: string,
  summaryContents: string,
  scoreContents: string,
  generatedAt: string,
): AnalysisResult {
  const summaries = parseCsv(summaryContents, (row) => datasetSummaryCsvRowSchema.parse(row));
  const results = parseCsv(scoreContents, (row) => diversityResultCsvRowSchema.parse(row));
  if (summaries.length !== 2 || results.length !== 2) {
    throw new Error("Expected exactly two subset rows in each Track 2 CSV output.");
  }

  const resultById = new Map(results.map((row) => [subsetIdForLabel(row.subset), row]));
  const summaryById = new Map(summaries.map((row) => [subsetIdForLabel(row.dataset), row]));
  const subsets = (["subset-a", "subset-b"] as const).map((id) => {
    const summary = summaryById.get(id);
    const result = resultById.get(id);
    if (!summary || !result) throw new Error(`Missing backend output for ${id}`);
    return adaptSubset(summary, result);
  });

  const warnings = [
    "Subset duration is estimated from num_frames at a fixed 30 FPS.",
    "The backend uses fixed Mecka and Scale subsets selected oldest-first within each lab.",
    "Visual diversity is calculated from a matched sample of 200 unique episode videos per subset.",
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
    const contents = await readFile(
      /* turbopackIgnore: true */ RESULT_PATHS.summary,
      "utf8",
    );
    const rows = parseCsv(contents, (row) => datasetSummaryCsvRowSchema.parse(row));
    return Object.fromEntries(
      rows.map((row) => [
        subsetIdForLabel(row.dataset),
        {
          label: row.dataset,
          episodeCount: row.episodes,
          estimatedHours: row.hours,
        },
      ]),
    );
  } catch {
    return null;
  }
}
