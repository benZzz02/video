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

## OVO shared memory and task routing

The original OVO baseline is unchanged. `eval_qwen3vl_ovo_folio.py` builds
independent prefix memories; `eval_qwen3vl_ovo_folio_fast.py` groups annotations
by their source `video`, opens the longest available causal prefix, and shares
causal memory state across that video's questions in timestamp order. Under
`task_state`, each required task memory has its own session.

The fast runner defaults to `--folio_query_policy task_state`. Its routes are:

| Tasks | `memory_route` | State and answer |
| --- | --- | --- |
| RT (OCR/ACR/ATR/STU/FPD/OJR), SSR | `recent_only` | Original OVO prompt and recent frames; no history writing or retrieval |
| BT (EPM/ASI/HLD) | `text_memory` | Shared FOLIO historical text plus recent frames |
| REC | `rec_count` | Shared repetition-completion state; answer is `str(count)` with no additional QA model call |
| CRR | `evidence_memory` | Shared short event records, lexical retrieval for the current question, and a new Yes/No decision using text plus recent frames |

REC uses `RecCountSession` in `lib/ovo_task_memory.py`, shared by source video
and normalized activity. It sends **all sampled frames** at the default
`--rec_fps 2`, rather than selecting FOLIO keyframes. Four-second windows keep
one second of earlier overlap and actor/phase state for cross-boundary motions.
The writer returns visible completion events with an actor and an exact supplied
`end_frame_index`; Python maps the frame to its timestamp, enforces ownership by
the current window, and deduplicates by actor and completion frame time.
Completions become committed when a window closes. At an earlier question time,
the unfinished window is recomputed and its provisional completions **replace**
the previous provisional result, so repeated questions do not add the same count
again. Only frames at or before the query time are supplied. If a processing gap
remains, the route records `count_complete=false` and returns an error rather
than presenting an incomplete count as a complete answer. A successful
`count_complete` status means processing succeeded, not that visual counting
has been proven correct.

CRR uses `CrrEvidenceSession` at the main `--fps` (default 1 fps), with
eight-second windows and one second of earlier overlap. Its question-independent
writer produces short JSON event records anchored to supplied frames. Completed
windows are committed and an unfinished window's provisional records are
replaced as it grows. At query time, lightweight lexical matching against the
current question packs timestamped observations into at most 8 KiB; the QA model
uses those observations and recent frames to decide Yes/No again. Previous
predicted Yes/No answers and sufficiency decisions are not saved as visual facts.
Failed intervals are marked as incomplete evidence.

**CRR uses a bounded-start protocol, not complete-prefix memory.** For each
source video, its memory begins one recent window before the earliest CRR
`ask_time`, clamped to zero. Missing `ask_time` falls back to query times, giving
the earliest query time when it is absent throughout the group. An arrival time
after its query is clamped to that query. The default recent window is four
seconds (`recent_frames_only * chunk_duration`). Earlier events are unavailable
to this route, which can affect questions that need pre-arrival evidence. The
start is recorded as `crr_memory_start_time`; annotated answers and clue times
do not set it. The frame filter still decodes the source sequentially from the
beginning and discards samples before that start. It saves VLM writing over
that earlier interval; it does not eliminate the prefix decoding cost. This
protocol difference must be stated when comparing results.

The earlier policies remain available through explicit CLI selection:

| Policy | RT | BT and FT (REC/SSR/CRR) |
| --- | --- | --- |
| `all` | Advance memory and retrieve using the selected FOLIO profile | Same |
| `task_routed` | Original OVO prompt and recent frames only; no memory advance or retrieval | Advance shared FOLIO memory to the question time, retrieve historical text, answer with text plus recent frames |

Both `task_routed` and `task_state` override the profile's SemLink, historical-image replay,
interaction focus, and structured answer wrapper to **off**. They preserve the
original letter/number/Yes-No answer contract. Retrieval uses the existing
deterministic entity/event matcher and its 8 KiB text budget for FOLIO text
routes; CRR uses the lexical event matcher described above. This is an
explicit speed/accuracy trade-off; conceptual matches may be missed without
SemLink and details absent from text cannot be recovered from historical images.

For example, from the `video` directory, with dataset/model paths set for the
machine:

```bash
python main_experiments/eval_qwen3vl_ovo_folio_fast.py \
  --model_path /path/to/Qwen3-VL-8B \
  --anno_path /path/to/ovo_bench_new.json \
  --chunked_dir /path/to/chunked_videos \
  --result_dir main_experiments/results/ovo_folio_task_state_v1 \
  --folio_query_policy task_state \
  --recent_frames_only 4 \
  --fps 1 \
  --rec_fps 2 \
  --rec_window_seconds 4 \
  --crr_window_seconds 8 \
  --task_memory_tokens 384 \
  --folio_segment_seconds 16 \
  --folio_generation_cap 1024
```

`--task_memory_tokens` caps each REC/CRR writer response; it does not change the
BT FOLIO writer cap. REC and CRR overlap is one second in their session defaults.
Use a fresh result directory when changing policies or processing settings. The
runner rejects checkpoints from a different policy, including legacy `all`
checkpoints, and rejects an incompatible `task_state_protocol` schema
(currently `ovo-task-state-v1`). At startup, `task_state` also saves
`task_state_run_config.json`: all CLI settings except `result_dir`, including
the model path, sampling rates, windows, token budgets, task selection/sharding
settings, plus the protocol and SHA256 hashes of the annotation, runner, and
task-memory implementation. Resuming requires an
identical manifest; existing task-state results without a manifest are rejected.
This prevents changed settings or annotations from being mixed into the same
checkpoint. The manifest records paths rather than hashing model weights or
video contents, so replacing those files in place still requires a fresh run.
Updating code does not switch processes that are already running;
existing runs continue with the policy loaded at process startup.
Per-video sharding remains available with `--num_shards` and `--shard_index`;
its cost estimate follows the selected task routes.

Memory is built lazily: an RT question, or SSR under `task_state`, does not
advance a history decoder. A later memory-routed query consumes only the
outstanding history through its timestamp. Committed segments/windows are
shared across questions; REC/CRR may revisit only an unfinished window. BT and
CRR writers are question-independent; REC receives the activity to count, but
no ground-truth count. Writers receive no answers or future frames. In a mixed
video, an earlier history query can still delay a later RT query in this
sequential runner; this is not a background writer service.

Each result records `folio_query_policy`, `task_state_protocol`, `memory_route`,
and `memory_used`. The latter identifies selected FOLIO records, use of the REC
counter, or nonempty CRR evidence text; it is not a correctness flag. Timings
include:

- `memory_advance_seconds`: prefix decoding, segment selection, and writing;
- `retrieval_seconds`: query preparation and retrieval;
- `answer_seconds`: recent-window decode and answer generation, including retries
  (zero for REC's direct count answer);
- `query_wall_seconds`: total time inside this query, including the above;
- `write_calls_delta`, `write_seconds_delta`, `write_errors_delta`, and
  `semantic_link_calls_delta`: new work performed for this question.

REC also records `count_events` and CRR records the retrieved `evidence_text`
so the observations behind an answer can be inspected. `writer_events_emitted`
and `writer_events_kept` count parsed writer records before and after overlap
ownership/deduplication; provisional rewrites can contribute more than once to
these diagnostics, so they are not repetition counts.

Sum the **delta** fields across questions. Existing `write_seconds` and
`write_calls` fields are cumulative within a session and must not be summed over
every question. REC/CRR additionally expose processing `status`, `gaps`, and
provisional record counts; REC exposes `cumulative_count` and `count_complete`.
Baseline runners have different `generate_time` boundaries;
compare end-to-end wall time with the same samples and image formatting.

These routes remove memory work from the RT path, but do not remove the first
historical query's observation cost. The BT FOLIO structured writer remains
unchanged (and is still used for FT under the earlier policies). Lowering a
writer's token cap can truncate JSON and lose a segment/window; monitor
`write_errors_delta` as well as speed. At 1 fps with 16-frame segments and a
four-frame recent window, the BT FOLIO route can leave roughly 12 seconds of
pending history outside both committed text and recent visual evidence.
REC's 2 fps sampling can miss fast repetitions or confuse actors, and CRR's
compact event text can omit decisive details. The new visual perception,
sampling choices, accuracy, and full-model speed have not been established by
state-machine tests and need separate measurement.

The fast runner shares memory **within one run**; it does not save/load memory
snapshots. Resuming an unfinished video rebuilds its causal history. Persistent
cross-run caching would require compatible model/config keys and timestamped
states; loading a final-video state for an earlier question would leak future
information.

Run the checks with:

```bash
python -m unittest discover -s tests -v
```

The tests cover keyframe selection, atomic writes, merge guards, structured and
semantic retrieval, temporal evidence recovery, multi-turn causality, snapshot
failure recovery, the existing Claude-style memory, and the unchanged baseline
path. They do not measure real-model accuracy; run one real-checkpoint smoke
test in the target Qwen/PyAV environment before a benchmark campaign.
