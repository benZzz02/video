"""Accelerated FOLIO-style OVO-Bench adapter.

This runner is deliberately separate from the independent-prefix adapter.  It
groups OVO questions by their original video, advances one FOLIO memory state
chronologically, and reuses that state for all questions from the video.  The
base OVO entrypoints and the FOLIO implementation are not modified.

The speed/accuracy trade-offs are explicit command-line settings:
``--folio_segment_seconds 16``, ``--folio_generation_cap 1024`` and
``--max_qa_tokens 128`` are the balanced defaults used for the full run.
Interaction focus is disabled in shared mode so one question cannot change the
memory retrieval of another question from the same video.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse the standalone adapter's environment-compatible OpenCV/video path and
# OVO prompt helpers without changing that adapter's default behavior.
from main_experiments import eval_qwen3vl_ovo_folio as base  # noqa: E402
from lib.folio_memory import FolioConfig, FolioMemorySession  # noqa: E402
from lib.recent_window_eval import (  # noqa: E402
    RecentWindowQAModel,
    build_ovo_prompt,
    calculate_ovo_scores,
    extract_mcq_answer,
    load_jsonl_results,
    print_ovo_results,
    query_recent_window,
)
from ovo_constants import BACKWARD_TASKS, FORWARD_TASKS, REAL_TIME_TASKS  # noqa: E402


LOGGER = logging.getLogger("ovo_folio_fast")
ALL_BR_TASKS = BACKWARD_TASKS + REAL_TIME_TASKS


def _query_specs(anno: dict[str, Any], chunked_dir: str) -> list[dict[str, Any]]:
    """Expand one annotation into causal video-query records."""
    task = str(anno["task"])
    specs: list[dict[str, Any]] = []
    if task in ALL_BR_TASKS:
        specs.append(
            {
                "annotation_key": base.make_key(anno),
                "query_id": base.make_key(anno),
                "query_index": None,
                "task": task,
                "source_video": str(anno["video"]),
                "query_time": float(anno["realtime"]),
                "clip_path": os.path.join(chunked_dir, f"{anno['id']}.mp4"),
                "question": str(anno["question"]),
                "options": [str(item) for item in anno.get("options", [])],
                "original_prompt": build_ovo_prompt(task, anno),
            }
        )
        return specs

    for index, test_info in enumerate(anno.get("test_info", [])):
        specs.append(
            {
                "annotation_key": base.make_key(anno),
                "query_id": f"{base.make_key(anno)}:{index}",
                "query_index": index,
                "task": task,
                "source_video": str(anno["video"]),
                "query_time": float(test_info["realtime"]),
                "clip_path": os.path.join(chunked_dir, f"{anno['id']}_{index}.mp4"),
                "question": base._forward_question(anno, index),
                "options": [],
                "original_prompt": build_ovo_prompt(task, anno, index=index),
            }
        )
    return specs


def _record_template(anno: dict[str, Any]) -> dict[str, Any]:
    task = str(anno["task"])
    if task in ALL_BR_TASKS:
        return {
            "id": anno["id"],
            "video": anno["video"],
            "task": task,
            "question": anno["question"],
            "response": None,
            "ground_truth": chr(65 + anno["gt"]),
        }
    result = copy.deepcopy(anno)
    for test_info in result.get("test_info", []):
        test_info["response"] = None
    return result


def _prediction_text(response: str, prediction: str | None, options: Sequence[str]) -> str:
    if prediction is not None:
        index = ord(prediction) - ord("A")
        if 0 <= index < len(options):
            return str(options[index])
    return str(response).strip()[:2048]


def _set_query_result(
    record: dict[str, Any],
    spec: dict[str, Any],
    response: str | None,
    metadata: dict[str, Any],
    error: str | None,
) -> None:
    if spec["query_index"] is None:
        prediction = extract_mcq_answer(response or "")
        record.update(
            {
                "response": response,
                "folio": metadata,
                **metadata,
            }
        )
        if error:
            record["error"] = error
        return

    test_info = record["test_info"][int(spec["query_index"])]
    test_info["response"] = response
    test_info["folio"] = metadata
    test_info.update(metadata)
    if error:
        test_info["error"] = error


def _format_prompt(
    task: str,
    folio_config: FolioConfig,
    plan: Any,
    original_prompt: str,
) -> tuple[str, str, Sequence[Any]]:
    if plan is None:
        return original_prompt, original_prompt, ()
    if task in ALL_BR_TASKS and folio_config.structured_answer:
        # The full FOLIO answer schema is useful for qualitative inspection,
        # but it is wasteful for OVO scoring and can be truncated by a small
        # answer budget.  Keep the full evidence/reasoning prompt and override
        # only the output contract to a parseable one-line label.
        compact_suffix = (
            "\n\nOUTPUT FORMAT OVERRIDE: Return exactly one valid JSON object and no "
            "other text or keys: {\"prediction_label\":\"A\"}. Replace A "
            "with exactly one of A, B, C, or D."
        )
        return (
            plan.prompt + compact_suffix,
            plan.recent_only_prompt + compact_suffix,
            plan.evidence_frames,
        )
    if task in {"REC", "SSR", "CRR"} and folio_config.structured_answer:
        prompt = base._format_non_mcq_folio_prompt(plan.memory_text, original_prompt)
        return prompt, original_prompt, plan.evidence_frames
    return plan.prompt, plan.recent_only_prompt, plan.evidence_frames


def run_shared_query(
    *,
    qa: RecentWindowQAModel,
    session: FolioMemorySession,
    source_path: str,
    spec: dict[str, Any],
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    folio_config: FolioConfig,
) -> tuple[str | None, dict[str, Any], str | None]:
    """Answer one query using an already advanced shared video session."""
    query_time = float(spec["query_time"])
    recent_start = max(0.0, query_time - recent_frames_only * chunk_duration)
    plan = None
    historical_frames: Sequence[Any] = ()
    history_frames_used = 0
    try:
        session.advance_to(query_time)
        try:
            plan = session.prepare_query(
                spec["query_id"],
                spec["question"],
                spec["options"],
                query_time=query_time,
                recent_start=recent_start,
                original_prompt=spec["original_prompt"],
                query_type_hint=spec["task"],
            )
        except Exception as exc:
            session.memory.query_errors += 1
            LOGGER.warning("FOLIO retrieval failed open for %s: %s", spec["query_id"], exc)

        prompt, recent_only_prompt, historical_frames = _format_prompt(
            spec["task"], folio_config, plan, spec["original_prompt"]
        )
        history_frames_used = len(historical_frames)
        query_kwargs = {
            "qa": qa,
            "video_path": source_path,
            "prompt": prompt,
            "chunk_duration": chunk_duration,
            "fps": fps,
            "recent_frames_only": max(1, int(recent_frames_only)),
            "video_start": recent_start,
            "video_end": query_time + 1e-4,
            "historical_frames": historical_frames,
        }
        try:
            result, decode_backend = query_recent_window(**query_kwargs)
        except Exception:
            if not historical_frames:
                raise
            LOGGER.warning("FOLIO evidence answer failed for %s; retrying recent window", spec["query_id"])
            query_kwargs.pop("historical_frames", None)
            query_kwargs["prompt"] = recent_only_prompt
            history_frames_used = 0
            result, decode_backend = query_recent_window(**query_kwargs)

        metadata = {
            **session.usage(),
            "snapshot_dir": None,
            "snapshot_saved": False,
            "decode_backend": decode_backend,
            "final_chunk_ids": result.final_chunk_ids,
            "generate_time": result.generate_time,
            "ttft_seconds": result.ttft_seconds,
            "num_vision_tokens": result.num_vision_tokens,
            "num_vision_tokens_before": result.num_vision_tokens_before,
            "num_vision_tokens_after": result.num_vision_tokens_after,
            "num_frames": result.num_frames,
            "historical_frame_count": history_frames_used,
            "recent_frame_count": max(0, int(result.num_frames) - history_frames_used),
            **(plan.to_metadata() if plan is not None else {}),
            "shared_video_memory": True,
        }
        # Do not commit dialogue focus: this keeps each question independent
        # while still reusing the expensive visual memory built for the video.
        return result.answer, metadata, None
    except Exception as exc:
        LOGGER.exception("Shared FOLIO query failed: %s", spec["query_id"])
        metadata = {
            **session.usage(),
            "snapshot_dir": None,
            "snapshot_saved": False,
            "historical_frame_count": history_frames_used,
            **(plan.to_metadata() if plan is not None else {}),
            "shared_video_memory": True,
        }
        return None, metadata, str(exc)


def _choose_source_path(specs: Sequence[dict[str, Any]]) -> str | None:
    available = [item for item in specs if os.path.exists(str(item["clip_path"]))]
    if not available:
        return None
    # OVO chunk files are causal prefixes.  The longest requested prefix is a
    # valid source for every earlier query from the same original video.
    return str(max(available, key=lambda item: float(item["query_time"]))["clip_path"])


def _partition_groups(
    groups: dict[str, list[dict[str, Any]]],
    *,
    num_shards: int,
    segment_seconds: float,
) -> list[list[tuple[str, list[dict[str, Any]]]]]:
    """Greedily balance video groups by estimated causal work.

    A group is intrinsically sequential within one video, but independent
    videos can run on separate model replicas.  Longest-prefix work is the
    dominant cost, so use it together with query count as a static estimate.
    """
    buckets: list[list[tuple[str, list[dict[str, Any]]]]] = [
        [] for _ in range(num_shards)
    ]
    loads = [0.0] * num_shards
    weighted: list[tuple[float, str, list[dict[str, Any]]]] = []
    divisor = max(float(segment_seconds), 1.0)
    for source_video, specs in groups.items():
        max_time = max((float(item["query_time"]) for item in specs), default=0.0)
        estimated = max_time / divisor + 0.5 * len(specs)
        weighted.append((estimated, source_video, specs))
    weighted.sort(key=lambda item: (-item[0], item[1]))
    for estimated, source_video, specs in weighted:
        shard = min(range(num_shards), key=lambda index: loads[index])
        buckets[shard].append((source_video, specs))
        loads[shard] += estimated
    return buckets


def _write_record(checkpoint: Any, record: dict[str, Any], key: str, done_keys: set[str]) -> None:
    record["_key"] = key
    checkpoint.write(json.dumps(record, ensure_ascii=False) + "\n")
    checkpoint.flush()
    done_keys.add(key)


def run_group(
    *,
    group_specs: Sequence[dict[str, Any]],
    annotations_by_key: dict[str, dict[str, Any]],
    records_by_key: dict[str, dict[str, Any]],
    done_keys: set[str],
    checkpoint: Any,
    qa: RecentWindowQAModel,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    folio_config: FolioConfig,
) -> None:
    new_specs = [item for item in group_specs if item["annotation_key"] not in done_keys]
    if not new_specs:
        return
    source_path = _choose_source_path(group_specs)
    grouped_by_key: dict[str, list[dict[str, Any]]] = {}
    for spec in new_specs:
        grouped_by_key.setdefault(str(spec["annotation_key"]), []).append(spec)

    if source_path is None:
        for key, specs in grouped_by_key.items():
            for spec in specs:
                _set_query_result(
                    records_by_key[key],
                    spec,
                    None,
                    {"shared_video_memory": True},
                    "Missing all chunked videos for source: " + str(spec["source_video"]),
                )
            _write_record(checkpoint, records_by_key[key], key, done_keys)
        return

    session = FolioMemorySession(
        path=source_path,
        fps=fps,
        generate=qa._folio_generate_memory,
        config=folio_config,
        frame_source=base._opencv_frame_source(source_path, fps),
    )
    try:
        ordered = sorted(new_specs, key=lambda item: (float(item["query_time"]), str(item["query_id"])))
        remaining = {key: len(value) for key, value in grouped_by_key.items()}
        for spec in ordered:
            answer, metadata, error = run_shared_query(
                qa=qa,
                session=session,
                source_path=source_path,
                spec=spec,
                chunk_duration=chunk_duration,
                fps=fps,
                recent_frames_only=recent_frames_only,
                folio_config=folio_config,
            )
            key = str(spec["annotation_key"])
            _set_query_result(records_by_key[key], spec, answer, metadata, error)
            remaining[key] -= 1
            if remaining[key] == 0:
                _write_record(checkpoint, records_by_key[key], key, done_keys)
                LOGGER.info(
                    "completed %s from %s (memory writes=%s, write_seconds=%.1f)",
                    key,
                    spec["source_video"],
                    session.memory.write_calls,
                    session.memory.write_seconds,
                )
    finally:
        session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Accelerated shared-memory FOLIO OVO-Bench adapter")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--anno_path", required=True)
    parser.add_argument("--chunked_dir", required=True)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--recent_frames_only", type=int, default=4)
    parser.add_argument("--chunk_duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_qa_tokens", type=int, default=128)
    parser.add_argument("--folio_profile", choices=("compat", "full"), default="full")
    parser.add_argument("--folio_segment_seconds", type=float, default=16.0)
    parser.add_argument("--folio_generation_cap", type=int, default=1024)
    parser.add_argument("--qa_device", default="auto")
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    args = parser.parse_args()

    if args.recent_frames_only < 1 or args.fps <= 0 or args.chunk_duration <= 0:
        raise ValueError("recent_frames_only, fps, and chunk_duration must be positive")
    if args.folio_segment_seconds <= 0 or args.folio_generation_cap < 1 or args.max_qa_tokens < 1:
        raise ValueError("segment seconds, generation cap, and QA tokens must be positive")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = result_dir / "results_incremental.jsonl"
    _, done_keys = load_jsonl_results(str(checkpoint_path))

    with open(args.anno_path, encoding="utf-8") as handle:
        annotations = json.load(handle)
    splits = {
        "backward": [item for item in annotations if item["task"] in BACKWARD_TASKS],
        "realtime": [item for item in annotations if item["task"] in REAL_TIME_TASKS],
        "forward": [item for item in annotations if item["task"] in FORWARD_TASKS],
    }
    rng = random.Random(args.seed)
    for values in splits.values():
        rng.shuffle(values)
        if args.max_samples_per_split is not None:
            if args.max_samples_per_split < 1:
                raise ValueError("max_samples_per_split must be >= 1")
            del values[args.max_samples_per_split :]
    selected = [item for values in splits.values() for item in values]
    total_queries = sum(
        len(item.get("test_info", [])) if item["task"] in FORWARD_TASKS else 1
        for item in selected
    )
    groups: dict[str, list[dict[str, Any]]] = {}
    annotations_by_key: dict[str, dict[str, Any]] = {}
    records_by_key: dict[str, dict[str, Any]] = {}
    for anno in selected:
        key = base.make_key(anno)
        annotations_by_key[key] = anno
        records_by_key[key] = _record_template(anno)
        for spec in _query_specs(anno, args.chunked_dir):
            groups.setdefault(str(anno["video"]), []).append(spec)

    full_group_count = len(groups)
    if args.num_shards > 1:
        group_buckets = _partition_groups(
            groups,
            num_shards=args.num_shards,
            segment_seconds=args.folio_segment_seconds,
        )
        groups = dict(group_buckets[args.shard_index])

    LOGGER.info(
        "FOLIO OVO shared-video run: %d annotations, %d video queries, "
        "%d unique source videos (shard %d/%d: %d groups)",
        len(selected),
        total_queries,
        full_group_count,
        args.shard_index,
        args.num_shards,
        len(groups),
    )

    folio_segment_frames = max(1, int(math.ceil(args.folio_segment_seconds * args.fps)))
    folio_config = FolioConfig.profile(args.folio_profile, segment_frames=folio_segment_frames)
    # Preserve per-question independence while sharing only visual memory.
    folio_config = replace(folio_config, interaction_focus=False)

    qa = RecentWindowQAModel(
        model_name=args.model_path,
        device=args.qa_device,
        max_new_tokens=args.max_qa_tokens,
        attn_implementation=args.attn_implementation,
        standard_multimodal=True,
    )

    def generate_memory(images: list[Any], text: str, limit: int) -> str:
        previous = qa.max_new_tokens
        try:
            qa.max_new_tokens = min(int(limit), int(args.folio_generation_cap))
            if images:
                return qa.generate_from_frames(images, text)
            return qa.generate_from_text(text)
        finally:
            qa.max_new_tokens = previous

    # Keep the callback on the model object only for the shared session
    # constructor; this avoids changing FolioMemory's public API.
    qa._folio_generate_memory = generate_memory

    selected_keys = set(records_by_key)
    with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
        for group_index, (source_video, group_specs) in enumerate(groups.items(), start=1):
            group_keys = {str(item["annotation_key"]) for item in group_specs}
            if not group_keys.intersection(selected_keys - done_keys):
                continue
            started = time.perf_counter()
            LOGGER.info(
                "[%d/%d groups] start %s (%d annotations, %d queries)",
                group_index,
                len(groups),
                source_video,
                len(group_keys),
                len(group_specs),
            )
            run_group(
                group_specs=group_specs,
                annotations_by_key=annotations_by_key,
                records_by_key=records_by_key,
                done_keys=done_keys,
                checkpoint=checkpoint,
                qa=qa,
                chunk_duration=args.chunk_duration,
                fps=args.fps,
                recent_frames_only=args.recent_frames_only,
                folio_config=folio_config,
            )
            LOGGER.info(
                "[%d/%d groups] done %s in %.1fs",
                group_index,
                len(groups),
                source_video,
                time.perf_counter() - started,
            )

    backward, realtime, forward = base._merge_results(result_dir)
    print_ovo_results("Qwen3-VL + FOLIO (shared video memory, accelerated)", backward, realtime, forward)
    summary = calculate_ovo_scores(backward, realtime, forward)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_path = result_dir / f"qwen3vl_folio_fast_ovo_results_{timestamp}.json"
    final_path.write_text(
        json.dumps(
            {
                "config": {
                    "model_path": args.model_path,
                    "recent_frames_only": args.recent_frames_only,
                    "chunk_duration": args.chunk_duration,
                    "fps": args.fps,
                    "max_qa_tokens": args.max_qa_tokens,
                    "folio_memory": True,
                    "folio_profile": args.folio_profile,
                    "folio_segment_seconds": args.folio_segment_seconds,
                    "folio_segment_frames": folio_segment_frames,
                    "folio_generation_cap": args.folio_generation_cap,
                    "folio_scope": "shared_video_chronological",
                    "interaction_focus": False,
                    "folio_standard_multimodal": True,
                    "memory_protocol": "folio-paper-reimplementation-v1",
                    "snapshots": False,
                    "num_shards": args.num_shards,
                    "shard_index": args.shard_index,
                },
                "summary": summary,
                "backward": backward,
                "realtime": realtime,
                "forward": forward,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Results saved to %s", final_path)


if __name__ == "__main__":
    main()
