# FOLIO-inspired focused semantic memory

This extension independently implements the public algorithms and record
interfaces described in [FOLIO: Focused Semantic Memory for Streaming Video
Understanding](https://arxiv.org/abs/2607.13298). It is not an official FOLIO
port: the paper currently links no runtime repository, and several numerical
thresholds are not disclosed. The implementation is therefore labeled
`folio-paper-reimplementation-v1`. Configurable defaults live in `FolioConfig`;
the remaining bounded adapter constants are listed below and in code.

## Memory model

Each source video owns one `FolioMemorySession` with three parts:

```text
S  short-term visual buffer = unchanged SimpleStream recent window
O  long-term memory         = persistent entity, observation, and event chains
B  visual evidence cache    = selected timestamped keyframes linked to O
```

At the default 1 fps, a completed eight-second segment is written once. An
incomplete segment is not forced into long-term memory when a question arrives;
the recent window still covers it.

For every completed segment, the memory system:

1. selects the boundary frames, lower middle frame, and frame after the largest
   adjacent pixel change;
2. keeps only the boundary frames when change is low and no high-focus entity
   needs extra evidence;
3. asks the same Qwen VLM for detailed entities, compact entities, and events;
4. validates the whole response before atomically merging stable entity IDs,
   aliases, observations, state/location chains, events, and evidence links;
5. updates deterministic focus scores for visibility, first appearance,
   persistence, reappearance, events, movement, state change, absence, and
   static background behavior.

The VLM writes segment records only. Python code owns entity identity, merging,
focus updates, evidence links, budgets, and transaction boundaries.

## Query flow

The query parser extracts targets, anchors, verbs, temporal scope, answer slot,
option cues, and whether an unavailable-evidence option exists. Direct ranking
matches those fields against entity names, aliases, categories, attributes,
head nouns, and action chains. It does not turn an arbitrary location, state,
or answer-option word into an entity hit. Evidence is then assembled according
to the paper's fixed taxonomy: current or historical location, interaction,
attribute, spatial, yes/no, hallucination detection, or concept.

The `full` profile implements the paper's S/O/B control flow with explicit
local defaults where the paper does not publish constants:

- Direct structured matching keeps at most six entities and an 8 KiB evidence
  block.
- If direct entity retrieval is empty, one text-only Qwen SemLink call receives
  the complete tracked entity/event catalog, selects existing IDs only, and
  cannot write facts. A matching event alone does not suppress this fallback.
  SemLink does not relax factual constraints: only concept queries use Mode B;
  location, attribute, spatial, and hallucination queries remain in Mode A.
- If the selected structured evidence is below the sufficiency threshold, at
  most two frames linked to the top-ranked observation or event records, and
  older than the recent window, are recovered. `first`, `last`, `before`, and
  `after` cues select from the complete observation chain before record limits
  are applied. Sufficiency is checked against the requested temporal evidence.
- Recovered historical frames are sorted before the unchanged recent frames;
  the answer still uses one ordinary `generate_from_frames` call.
- After a successful answer, matched entities and event participants receive a
  focus boost that can affect later segments only. Replaying the same query ID
  is idempotent.
- Each committed turn stores the model's predicted label and option text, never
  the ground truth. Up to eight previous turns are available to later parsing
  and answering, including pronoun and event-relative follow-ups.

The `compat` profile disables SemLink, historical-frame replay, interaction
focus, and the structured answer wrapper. It retains FOLIO writing and direct
structured retrieval while preserving the original answer prompt protocol.

## Run

Use a fresh output directory:

```bash
CUDA_VISIBLE_DEVICES=0 python main_experiments/eval_streamingbench.py \
  --anno-path data/streamingbench/questions_real.json \
  --video-dir data/streamingbench/videos \
  --top-k 0 \
  --recent-frames-only 4 \
  --chunk-duration 1.0 \
  --fps 1.0 \
  --folio-memory \
  --folio-profile full \
  --folio-segment-seconds 8 \
  --output-dir main_experiments/results/streamingbench_folio
```

`--folio-memory` and the earlier `--video-memory` mode are mutually exclusive.
With neither flag, the baseline prompt, frames, calls, and result schema are
unchanged.

FOLIO reuses the existing SimpleStream decoder, recent-frame selection, Qwen
model, and generation wrapper. Its writer and answer calls use the processor's
standard multi-image input path, so each image receives its own visual block
and the model retains its native visual features. The legacy cached-prefix
path remains the default when FOLIO is disabled. The result config records
`folio_standard_multimodal=true`; accuracy comparisons should use consistent
image formatting so an input-format correction is not mistaken for a memory
gain.

## Inspect memory use

Each video is mirrored after a successful query:

```text
<output-dir>/folio_memory/0001_<video-name>/
├── MEMORY.md
├── state.json
├── focus.json
├── queries.jsonl
├── usage.json
├── entities/
│   └── entity-0001.md
├── evidence/
│   ├── index.json
│   └── segment-00001-frame-00-<content-hash>.jpg
└── pending/
```

`queries.jsonl` and each result's `folio` object identify the retrieval mode,
matched and assembled entity/event IDs, selected observation/event records,
relevance scores, dialogue-turn count, recovered frame IDs/timestamps, actual
historical-frame count, and memory-prefix size. This makes it possible to
verify that a question used memory rather than inferring use from the answer.

`state.json` plus the evidence and pending images contains a machine-restorable
state accepted by `FolioMemory.load_snapshot`. Pending frames use lossless PNG.
The state manifest is committed last, so a failed save leaves the previous
machine snapshot loadable. The benchmark command still requires a fresh output
directory because automatic checkpoint/session resume is not yet wired into
the multi-video runner.

The result config distinguishes SimpleStream's feature cache from FOLIO's
evidence cache: `feature_cache_enabled` remains false, while `cache_enabled`
and `folio_evidence_cache` report whether historical evidence replay is active.

## Prompt source and adapters

The writer and SemLink schemas track the templates published in the paper's
`9_prompts.tex`. The answer reader retains the paper's Mode A/Mode B policy.
Explicit adapters add selected-frame timestamps, alias/visible-text/event-frame
fields, cache-frame ordering, prior model-prediction history, and protection
against instructions embedded in stored text. This is an independent paper
reimplementation rather than a claim of verbatim official runtime code.

## Local defaults not published by FOLIO

The paper specifies the formulas and behavior but not the visual-change
threshold, focus coefficients/thresholds, retrieval weights, Top-K, or cache
frame count. This implementation uses:

- visual-change threshold `0.03` on 64x64 grayscale images in `[0,1]`;
- focus decay `0.85`, with levels at `0.70 / 0.40 / 0.15`;
- deterministic merge scores `1.0 / 0.95 / 0.80 / 0.75 / 0.55` for canonical
  name, alias, head, substring, and category, with an acceptance threshold of
  `0.80` plus distinctive-attribute and shared-generic-alias guards;
- a small, explicit synonym table for common entity-name variants and action
  inflections used by direct retrieval and delayed focus matching;
- at most `12` objects and `6` events per writer call;
- writer focus slots of `4` detailed and `8` compact entities/actions;
- direct retrieval Top-`6`, evidence text capped at `8192` UTF-8 bytes;
- sufficiency score `8.0` and at most `2` recovered cache frames;
- SemLink receives the complete tracked entity/event catalog; only the
  query-time answer evidence block uses the `8192`-byte packing budget;
- writer output limit `3072` tokens and SemLink limit `512` tokens;
- at most `8` prior dialogue turns / `4096` dialogue bytes in an answer prompt;
- unresolved prior-query terms have no time-to-live; they remain pending until
  a uniquely matched later entity or action receives the delayed focus boost;
- the semantic bank and evidence cache retain the full stream history; the
  query-time 8 KiB block is packed from the highest-ranked concrete records,
  with temporal cue records selected before lower-ranked chain entries.

These are reproducible engineering defaults, not claimed FOLIO hyperparameters.

## Scope

The first release integrates StreamingBench. The OVO entry points operate on
independent prefix clips rather than a verified shared source stream, so they
are left unchanged to avoid mixing or replaying video state across clips.

Run the checks with:

```bash
python -m unittest discover -s tests -v
```

The tests cover keyframe selection, atomic writes, merge guards, structured and
semantic retrieval, temporal evidence recovery, multi-turn causality, snapshot
failure recovery, the existing Claude-style memory, and the unchanged baseline
path. They do not measure real-model accuracy; run one real-checkpoint smoke
test in the target Qwen/PyAV environment before a benchmark campaign.
