// Claude Code WORKFLOW (run via the Workflow tool; not part of the Python package).
//
// Distills the verbose, compound freeform `annotations` of mecka folding_clothes
// episodes into a small set of canonical task labels, in two phases with a human
// review gate between them. The on-disk glue (chunking + merge/validate) is done by
// egomimic/scripts/data_quality/relabel_utils.py — workflow scripts have no filesystem
// access, so the AGENTS here do the Read/Write while this script orchestrates.
//
// End-to-end:
//   1. python -m egomimic.scripts.data_quality.export_segments --out segments.json
//   2. Workflow({ scriptPath: ".../relabel_segments.workflow.js",
//                 args: { phase: "propose", segments: "/abs/segments.json",
//                         out: "/abs/taxonomy.json" } })
//      -> review/edit taxonomy.json
//   3. python -m egomimic.scripts.data_quality.relabel_utils split \
//          --segments segments.json --out-dir relabel_work/ --chunk-size 25
//   4. Workflow({ scriptPath: ".../relabel_segments.workflow.js",
//                 args: { phase: "map", chunks: [<seg_chunk paths>],
//                         taxonomy: "/abs/taxonomy.json", outDir: "/abs/relabel_work",
//                         taxonomyLabels: [...], segCounts: {<hash>: <#segments>} } })
//   5. python -m egomimic.scripts.data_quality.relabel_utils merge \
//          --segments segments.json --labels-dir relabel_work/ \
//          --taxonomy taxonomy.json --out relabel_map.json
//
// `taxonomyLabels`/`segCounts` in the map args are optional; if provided, the workflow
// validates each chunk and retries a bad chunk once. relabel_utils.py merge is the final
// authoritative validation (and falls back to original text on any remaining mismatch).

export const meta = {
  name: 'relabel-segments',
  description:
    'Distill verbose episode annotations into canonical task labels (propose taxonomy, then map each segment)',
  phases: [
    { title: 'Propose taxonomy', detail: 'one agent proposes a compact canonical label set' },
    { title: 'Map segments', detail: 'fan out one agent per chunk to assign one label per segment' },
  ],
}

const TAXONOMY_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['labels'],
  properties: {
    labels: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['label', 'description'],
        properties: {
          label: { type: 'string' },
          description: { type: 'string' },
        },
      },
    },
  },
}

const MAP_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  required: ['episodes'],
  properties: {
    episodes: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        required: ['episode_hash', 'labels'],
        properties: {
          episode_hash: { type: 'string' },
          labels: { type: 'array', items: { type: 'string' } },
        },
      },
    },
  },
}

// `args` may arrive as a JS object or a JSON-encoded string depending on caller.
const A = typeof args === 'string' ? JSON.parse(args) : args || {}
const phaseArg = A.phase || 'map'

// ---------------------------------------------------------------------------
// Phase 1: propose a compact canonical taxonomy (single agent).
// ---------------------------------------------------------------------------
if (phaseArg === 'propose') {
  phase('Propose taxonomy')
  const segmentsPath = A.segments
  const outPath = A.out
  const result = await agent(
    `You are designing a compact, canonical task-label taxonomy for a BIMANUAL robot
folding-clothes dataset.

Read the list of distinct annotation phrases at: ${segmentsPath}
It is a text file, one phrase per line as "<count>\\t<phrase>" (count = how often the phrase
occurs across the dataset), sorted by frequency. The file has ~2500 lines — READ THE WHOLE
FILE (pass a high limit to Read, e.g. limit=3000). The phrases are verbose, compound,
inconsistent freeform (e.g. "smooth out green t-shirt on table, flip green t-shirt and smooth
out green t-shirt on table") and include actions like ironing and bagging, not just folding.

Propose a SMALL canonical taxonomy (target 8-15 labels) of short, atomic, snake_case
task labels covering the dominant actions across these segments (e.g. smooth_garment,
fold_garment, flip_garment, unfold_garment, pick_up_garment, place_on_stack, open_bag,
insert_in_bag, adjust_garment). Each segment will later be mapped to EXACTLY ONE label,
so the set must cover the common cases; include a catch-all like "other_manipulation"
for rare/ambiguous segments. Keep labels garment-agnostic (don't bake in colors/types).

Write the taxonomy as JSON to: ${outPath}
in the shape {"labels": [{"label": "...", "description": "..."}, ...]}
Then return that same object as your result.`,
    { label: 'propose-taxonomy', phase: 'Propose taxonomy', schema: TAXONOMY_SCHEMA },
  )
  log(`Proposed ${result && result.labels ? result.labels.length : 0} labels -> ${outPath}`)
  return result
}

// ---------------------------------------------------------------------------
// Phase 2: map each segment to one taxonomy label (fan out over chunks).
// ---------------------------------------------------------------------------
phase('Map segments')
const chunks = A.chunks || []
const taxonomyPath = A.taxonomy
const outDir = A.outDir
const taxonomyLabels = A.taxonomyLabels || null
const segCounts = A.segCounts || null

if (!chunks.length) {
  log('No chunks provided in args.chunks; nothing to map.')
  return { mapped: 0 }
}

function chunkTag(p, idx) {
  const m = String(p).match(/seg_chunk_(\d+)\.json/)
  return m ? m[1] : String(idx)
}

function mapPrompt(chunkPath, idx, extra) {
  const tag = String(idx).padStart(3, '0')
  const outPath = `${outDir}/relabel_chunk_${tag}.json`
  return `You are mapping robot folding-clothes episode segments to canonical task labels.

Read the approved taxonomy at: ${taxonomyPath}  (JSON {"labels": [{label, description}]})
Read the segment chunk at:      ${chunkPath}    (JSON list of episodes, each with an ordered "segments" array)

For EVERY episode in the chunk, output one label PER segment, IN THE SAME ORDER as the
segments appear:
- Each label MUST be exactly one of the taxonomy "label" values (verbatim, snake_case).
- Pick the single DOMINANT action of the (possibly compound) segment "text".
- Do NOT add, drop, reorder, merge, or split segments. For each episode,
  labels.length MUST equal that episode's segments.length.

Write the result as JSON to: ${outPath}
in the shape {"episodes": [{"episode_hash": "...", "labels": ["...", ...]}, ...]}
Then return that same object as your result.${extra || ''}`
}

const results = await pipeline(
  chunks,
  // Stage 1 — map this chunk.
  (chunkPath, _orig, idx) =>
    agent(mapPrompt(chunkPath, idx), {
      label: `map:${chunkTag(chunkPath, idx)}`,
      phase: 'Map segments',
      schema: MAP_SCHEMA,
    }),
  // Stage 2 — lightweight validation (if args supplied counts/labels) + one retry.
  (mapped, chunkPath, idx) => {
    if (!mapped) return null
    const bad = []
    for (const ep of mapped.episodes || []) {
      if (
        segCounts &&
        segCounts[ep.episode_hash] != null &&
        ep.labels.length !== segCounts[ep.episode_hash]
      ) {
        bad.push(
          `${ep.episode_hash}: produced ${ep.labels.length} labels but episode has ${segCounts[ep.episode_hash]} segments`,
        )
      }
      if (taxonomyLabels) {
        const unknown = ep.labels.filter((l) => !taxonomyLabels.includes(l))
        if (unknown.length) bad.push(`${ep.episode_hash}: labels not in taxonomy: ${unknown.join(', ')}`)
      }
    }
    if (!bad.length) return { chunk: idx, episodes: mapped.episodes.length }
    log(`chunk ${idx}: ${bad.length} issue(s); retrying once`)
    return agent(
      mapPrompt(chunkPath, idx, `\n\nA previous attempt had these problems — fix them exactly:\n- ${bad.join('\n- ')}`),
      { label: `map-retry:${chunkTag(chunkPath, idx)}`, phase: 'Map segments', schema: MAP_SCHEMA },
    ).then((r) => ({ chunk: idx, episodes: r && r.episodes ? r.episodes.length : 0, retried: true }))
  },
)

const done = results.filter(Boolean)
log(`Mapped ${done.length}/${chunks.length} chunks -> ${outDir}/relabel_chunk_*.json`)
return { chunks: chunks.length, done: done.length, retried: done.filter((d) => d.retried).length }
