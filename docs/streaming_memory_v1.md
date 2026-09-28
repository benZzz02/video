# Streaming video memory v1

The extension adapts Claude Code's file-based memory behavior to a causal
video stream:

```text
Claude Code:  one project -> MEMORY.md index -> multiple semantic topic files
SimpleStream: one video   -> memory index    -> multiple semantic video records
```

This follows Claude Code auto-memory's multi-record store, rather than its
single context-compaction summary.

Each source video gets a fresh `VideoMemorySession`. Consecutive frames update
that video's store without seeing any question, option, answer, task label, or
future question time. A store contains multiple records rather than one rolling
summary. Each record holds one arbitrary important fact or key state change,
including its original observation time.

The writer can atomically `upsert` a semantic record, `delete` a record proved
wrong or fully merged into another record without losing historical evidence,
or make no change. A state that changed stays in the topic's timestamped
history. Updating an existing topic leaves unrelated records untouched.
Invalid model output and writer failures
keep the last valid store and do not stop the normal question-answering path;
the failed frame segment is retried once with the next segment. The retry
buffer is capped at two segments, and any frames dropped after repeated errors
are counted in the result metadata.

## Source basis

The behavior was checked directly in Anthropic's official distribution:

- `@anthropic-ai/claude-code@2.1.32/cli.js`, SHA-256
  `b551568d8671a4456d745212d38cb95787132071177385f5f7c2e7df5b38c778`,
  contains the memory-directory prompt and implementation. It keeps a concise
  `MEMORY.md`, moves detail into separate topic files, organizes memory by
  semantic topic, and uses ordinary Read/Write/Edit operations to update or
  remove wrong information.
- Claude Code 2.1.278's official native package was also inspected locally.
  It retains separate Markdown memory files and a bounded index, and adds
  structured frontmatter plus relevance-based recall.
- The public contract is documented in
  [Claude Code memory](https://code.claude.com/docs/en/memory): one memory
  directory per repository, a `MEMORY.md` index limited to 200 lines or 25KB,
  and topic files read on demand.

Anthropic's package is proprietary, so no Claude Code implementation code is
copied into this repository. This is an independent Python adaptation of the
verified interface and behavior.

Qwen's existing `generate_from_frames` method has no file-tool loop. The video
adapter therefore represents a batch of file operations as one JSON
transaction, validates the complete resulting store, and then applies it. At
answer time it supplies the bounded index and records in one text prefix. This
keeps the original one-call SimpleStream answer path instead of adding a second
query-time model or changing Qwen internals. In particular, v1 intentionally
injects the entire bounded store instead of porting Claude Code's selective
topic recall.

## SimpleStream compatibility

`--video-memory` only wraps the prompt passed to the existing inference path.
The following behavior remains unchanged:

- `query_recent_window` performs the original decode and selects the last N
  chunks.
- The original current frames, chunk IDs, decoder backend, greedy generation,
  QA token limit, and answer extraction are unchanged.
- Memory uses its own fixed source-clock sampler and advances only through
  frames whose timestamps are no later than the current question.
- Questions from the same video share one store; another video starts with an
  empty store.
- The three `recent_window_eval` implementations and both OVO entry points are
  unchanged. OVO's independent prefix clips do not share this memory.
- With the flag absent, the baseline result schema and inference calls are
  unchanged.

The frame batch is chosen to fit within the configured recent-window duration.
Thus frames waiting for the next memory update remain covered by the original
recent window. Memory runs require a fresh output directory because v1 does not
persist the store or stream cursor in the JSONL checkpoint.

## Inspect local memory

No extra flag is needed. While `--video-memory` is active, the latest memory
for every video is mirrored after each question:

```text
<output-dir>/video_memory/0001_<video-name>/
├── MEMORY.md
├── <semantic-topic>.md
├── ...
└── usage.json
```

`MEMORY.md` is the readable topic index. Each topic file contains `name`,
`description`, and `type` frontmatter followed by timestamped evidence.
`usage.json` reports record count, writer/stream errors, dropped frames, and
the managed topic filenames. Deleted or safely merged records are removed from
the mirror. The snapshot directory is program-managed; keep personal Markdown
notes elsewhere. Snapshot writes use atomic file replacement and are fail-open:
a local I/O error increments `snapshot_errors` but does not stop inference.

Every result also contains `memory.snapshot_dir` and
`memory.snapshot_saved`. A successful question with `record_count > 0` and
`snapshot_saved: true` therefore used a memory prefix and refreshed the local
mirror successfully. The directory is a live latest-state view, so later
questions from the same video update it in place.

## Run and check

Add the flag to the original StreamingBench command:

```bash
--video-memory
```

Run the checks with:

```bash
python -m unittest discover -s tests -v
```

The tests cover multiple records, in-place topic updates, deletion, atomic
fail-open behavior, question-independent writing, causal advancement, strict
timestamps, and fixed-clock sampling with a synthesized video. Real Qwen/GPU
accuracy is not measured by these unit tests.
