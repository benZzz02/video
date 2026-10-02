"""
StreamingBench recent-window evaluation aligned with the no-cache OVO-stack baseline.

By default, this script runs the recency baseline (`--top-k 0`):
decode the time window ending at each question timestamp and answer from the
most recent N chunks without feature cache or retrieval.
Optional --video-memory adds bounded semantic records from earlier video frames.
Optional --folio-memory adds entity/event chains and linked visual evidence.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.recent_window_eval import (
    RecentWindowQAModel,
    extract_mcq_answer,
    load_jsonl_results,
    query_recent_window,
    save_json,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = (
    "You are an advanced video question-answering AI assistant. "
    "You have been provided with some frames from the video and a multiple-choice question. "
    "Your task is to analyze the video and provide the best answer.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Only give the best option's letter (A, B, C, or D) directly."
)


def timestamp_to_seconds(ts: str) -> int:
    parts = ts.split(":")
    return sum(int(x) * 60 ** i for i, x in enumerate(reversed(parts)))


def make_key(video_basename: str, q: dict[str, Any], question_limit: int = 80) -> str:
    return f"{video_basename}_{q.get('time_stamp', '')}_{q.get('question', '')[:question_limit]}"


def format_options(options: list[str]) -> str:
    formatted: list[str] = []
    for index, option in enumerate(options):
        text = str(option).strip()
        if not text.startswith(("A.", "B.", "C.", "D.")):
            text = f"{chr(65 + index)}. {text}"
        formatted.append(text)
    return "\n".join(formatted)


def build_prompt(q: dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(
        question=q.get("question", ""),
        options=format_options(q.get("options", [])),
    )


def resolve_video_path(video_path: str, video_dir: str) -> str:
    if video_path.startswith("./videos/"):
        return os.path.join(video_dir, os.path.basename(video_path))
    if not os.path.isabs(video_path):
        return os.path.join(video_dir, video_path)
    return video_path


def compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    total = 0
    correct = 0
    errors = 0
    for result in results:
        task_type = str(result.get("task_type", "unknown")).strip() or "unknown"
        by_type[task_type]["total"] += 1
        if result.get("correct", False):
            by_type[task_type]["correct"] += 1
            correct += 1
        total += 1
        if result.get("error"):
            errors += 1

    task_rows = []
    for task_type in sorted(by_type):
        row = by_type[task_type]
        task_total = int(row["total"])
        task_correct = int(row["correct"])
        task_rows.append(
            {
                "task_type": task_type,
                "total": task_total,
                "correct": task_correct,
                "accuracy": (100.0 * task_correct / task_total) if task_total else 0.0,
            }
        )

    overall_accuracy = (100.0 * correct / total) if total else 0.0
    return {
        "overall": {"total": total, "correct": correct, "accuracy": overall_accuracy},
        "error_count": errors,
        "tasks": task_rows,
    }


def print_summary(results: list[dict[str, Any]]) -> None:
    summary = compute_summary(results)
    print("\n" + "=" * 60)
    print("StreamingBench Recent-Window Results")
    print("=" * 60)
    for row in summary["tasks"]:
        print(f"  {row['task_type']}: {row['accuracy']:.2f}% ({row['correct']}/{row['total']})")
    overall = summary["overall"]
    print(f"\n  Overall: {overall['accuracy']:.2f}% ({overall['correct']}/{overall['total']})")
    print(f"  Errors: {summary['error_count']}")
    print("=" * 60)


def run_benchmark(
    anno_path: str,
    video_dir: str,
    output_dir: str,
    qa_model: str,
    qa_device: str,
    chunk_duration: float,
    fps: float,
    top_k: int,
    max_qa_tokens: int,
    recent_frames_only: int,
    context_time: int,
    video_memory: bool = False,
    folio_memory: bool = False,
    folio_profile: str = "full",
    folio_segment_seconds: float = 8.0,
) -> None:
    if video_memory and folio_memory:
        raise ValueError("--video-memory and --folio-memory are mutually exclusive")
    if folio_memory and (
        not math.isfinite(folio_segment_seconds) or folio_segment_seconds <= 0
    ):
        raise ValueError("folio_segment_seconds must be positive and finite")
    if top_k != 0:
        raise ValueError(
            "This release script only supports the no-retrieval baseline. "
            "Please run with --top-k 0."
        )

    with open(anno_path) as handle:
        all_data = json.load(handle)

    video_questions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    video_categories: dict[str, str] = {}
    for entry in all_data:
        video_path = entry["video_path"]
        video_categories[video_path] = entry.get("video_categories", "")
        for question in entry["questions"]:
            video_questions[video_path].append(question)

    for video_path in video_questions:
        video_questions[video_path].sort(key=lambda item: timestamp_to_seconds(item["time_stamp"]))

    total_questions = sum(len(items) for items in video_questions.values())
    logger.info("Loaded %d videos and %d questions", len(video_questions), total_questions)

    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "results_incremental.jsonl")
    memory_enabled = video_memory or folio_memory
    if memory_enabled and os.path.exists(ckpt_path) and os.path.getsize(ckpt_path):
        raise ValueError("Memory runs require a fresh output directory; stateful resume is unsupported.")
    all_results, done_keys = load_jsonl_results(ckpt_path)

    qa = RecentWindowQAModel(
        model_name=qa_model,
        device=qa_device,
        max_new_tokens=max_qa_tokens,
        **({"standard_multimodal": True} if folio_memory else {}),
    )

    memory_segment_frames = None
    if video_memory:
        memory_window_seconds = (
            float(context_time)
            if context_time > 0
            else float(recent_frames_only) * float(chunk_duration)
        )
        memory_segment_frames = max(1, int(math.ceil(memory_window_seconds * float(fps))))
    folio_config = None
    if folio_memory:
        from lib.folio_memory import FolioConfig

        folio_segment_frames = max(
            1, int(math.ceil(float(folio_segment_seconds) * float(fps)))
        )
        folio_config = FolioConfig.profile(
            folio_profile, segment_frames=folio_segment_frames
        )

    def generate_memory(images, text, limit):
        saved = qa.max_new_tokens
        try:
            qa.max_new_tokens = limit
            if images:
                return qa.generate_from_frames(images, text)
            return qa.generate_from_text(text)
        finally:
            qa.max_new_tokens = saved

    legacy_done_keys = {key for key in done_keys if isinstance(key, str)}

    def is_done(video_basename: str, question: dict[str, Any]) -> bool:
        return (
            make_key(video_basename, question, question_limit=80) in done_keys
            or make_key(video_basename, question, question_limit=50) in legacy_done_keys
        )

    with open(ckpt_path, "a") as ckpt_file:
        processed = 0
        for video_index, (video_path_raw, questions) in enumerate(video_questions.items(), start=1):
            video_path = resolve_video_path(video_path_raw, video_dir)
            video_basename = os.path.basename(video_path)
            logger.info("[video %d/%d] %s (%d questions)", video_index, len(video_questions), video_basename, len(questions))

            if not os.path.exists(video_path):
                logger.warning("Missing video: %s", video_path)
                processed += len(questions)
                continue

            memory_session = None
            memory_snapshot_dir = None
            memory_snapshot_rel = None
            memory_kind = None
            if video_memory:
                from lib.streaming_memory import VideoMemorySession

                snapshot_stem = Path(video_basename).stem or "video"
                memory_snapshot_rel = (
                    Path("video_memory") / f"{video_index:04d}_{snapshot_stem}"
                ).as_posix()
                memory_snapshot_dir = Path(output_dir) / memory_snapshot_rel
                memory_session = VideoMemorySession(
                    path=video_path,
                    fps=fps,
                    generate=generate_memory,
                    segment_frames=memory_segment_frames,
                )
                memory_kind = "semantic"
            elif folio_memory:
                from lib.folio_memory import FolioMemorySession

                snapshot_stem = Path(video_basename).stem or "video"
                memory_snapshot_rel = (
                    Path("folio_memory") / f"{video_index:04d}_{snapshot_stem}"
                ).as_posix()
                memory_snapshot_dir = Path(output_dir) / memory_snapshot_rel
                memory_session = FolioMemorySession(
                    path=video_path,
                    fps=fps,
                    generate=generate_memory,
                    config=folio_config,
                )
                memory_kind = "folio"

            for question in questions:
                processed += 1
                if is_done(video_basename, question):
                    logger.info("  [%d/%d] skip %s %s", processed, total_questions, question["time_stamp"], question.get("task_type", ""))
                    continue

                ts_sec = float(timestamp_to_seconds(question["time_stamp"]))
                window_seconds = float(context_time) if context_time > 0 else float(recent_frames_only) * float(chunk_duration)
                video_start = max(0.0, ts_sec - max(window_seconds, float(chunk_duration)))
                effective_recent_chunks = max(
                    int(recent_frames_only),
                    int(math.ceil(window_seconds / max(float(chunk_duration), 1e-6))),
                )
                prompt = build_prompt(question)
                memory_snapshot_saved = None
                folio_plan = None
                folio_history_frames_used = 0

                try:
                    if memory_session is not None:
                        memory_session.advance_to(ts_sec)
                        if memory_kind == "semantic":
                            memory_snapshot_saved = memory_session.save_snapshot(
                                memory_snapshot_dir
                            )
                            prompt = memory_session.augment(prompt)
                        else:
                            try:
                                folio_plan = memory_session.prepare_query(
                                    make_key(video_basename, question, question_limit=80),
                                    str(question.get("question", "")),
                                    [str(item) for item in question.get("options", [])],
                                    query_time=ts_sec,
                                    recent_start=video_start,
                                    original_prompt=prompt,
                                    query_type_hint=str(question.get("task_type", "")),
                                )
                                prompt = folio_plan.prompt
                            except Exception as memory_exc:
                                memory_session.memory.query_errors += 1
                                logger.warning(
                                    "FOLIO retrieval failed open for %s: %s",
                                    question["time_stamp"],
                                    memory_exc,
                                )
                    query_kwargs = dict(
                        qa=qa,
                        video_path=video_path,
                        prompt=prompt,
                        chunk_duration=chunk_duration,
                        fps=fps,
                        recent_frames_only=effective_recent_chunks,
                        video_start=video_start,
                        video_end=ts_sec + 1e-4,
                    )
                    if folio_plan is not None:
                        query_kwargs["historical_frames"] = folio_plan.evidence_frames
                        folio_history_frames_used = len(folio_plan.evidence_frames)
                    try:
                        result, decode_backend = query_recent_window(**query_kwargs)
                    except Exception:
                        if folio_plan is None or not folio_plan.evidence_frames:
                            raise
                        logger.warning(
                            "FOLIO history-frame answer failed; retrying the unchanged recent window"
                        )
                        query_kwargs.pop("historical_frames", None)
                        query_kwargs["prompt"] = folio_plan.recent_only_prompt
                        folio_history_frames_used = 0
                        result, decode_backend = query_recent_window(**query_kwargs)
                    response = result.answer
                    pred = extract_mcq_answer(response)
                    answer_gt = extract_mcq_answer(str(question.get("answer", ""))) or str(question.get("answer", "")).strip().upper()
                    correct = bool(pred is not None and pred == answer_gt)
                    if folio_plan is not None:
                        if memory_session.memory.query_traces:
                            memory_session.memory.query_traces[-1][
                                "historical_frames_used"
                            ] = folio_history_frames_used
                        memory_session.commit_interaction(
                            folio_plan.query_id,
                            folio_plan,
                            str(question.get("question", "")),
                            [str(item) for item in question.get("options", [])],
                            predicted_label=pred or "",
                            predicted_text=(
                                str(question.get("options", [])[ord(pred) - ord("A")])
                                if pred is not None
                                and ord(pred) - ord("A") < len(question.get("options", []))
                                else ""
                            ),
                        )
                    if memory_kind == "folio" and memory_session is not None:
                        memory_snapshot_saved = memory_session.save_snapshot(
                            memory_snapshot_dir
                        )
                    memory_metadata = None
                    if memory_session is not None:
                        memory_metadata = {
                            **memory_session.usage(),
                            "snapshot_dir": memory_snapshot_rel,
                            "snapshot_saved": memory_snapshot_saved,
                            **(
                                folio_plan.to_metadata()
                                if folio_plan is not None
                                else {}
                            ),
                            **(
                                {
                                    "historical_frame_count": folio_history_frames_used,
                                    "recent_frame_count": max(
                                        0,
                                        int(result.num_frames)
                                        - folio_history_frames_used,
                                    ),
                                }
                                if folio_plan is not None
                                else {}
                            ),
                        }
                    record = {
                        "_key": make_key(video_basename, question, question_limit=80),
                        "video": video_basename,
                        "video_categories": video_categories.get(video_path_raw, ""),
                        "task_type": question.get("task_type", ""),
                        "time_stamp": question["time_stamp"],
                        "question": question["question"],
                        "answer_gt": answer_gt,
                        "response": response,
                        "correct": correct,
                        "decode_backend": decode_backend,
                        "final_chunk_ids": result.final_chunk_ids,
                        "generate_time": result.generate_time,
                        "ttft_seconds": result.ttft_seconds,
                        "num_vision_tokens": result.num_vision_tokens,
                        "num_vision_tokens_before": result.num_vision_tokens_before,
                        "num_vision_tokens_after": result.num_vision_tokens_after,
                        **(
                            {"folio": memory_metadata}
                            if memory_kind == "folio" and memory_metadata is not None
                            else {"memory": memory_metadata}
                            if memory_metadata is not None
                            else {}
                        ),
                    }
                    logger.info(
                        "  [%d/%d] %s %s -> %s (gt=%s)",
                        processed,
                        total_questions,
                        question["time_stamp"],
                        question.get("task_type", ""),
                        response[:80] if response else "None",
                        answer_gt,
                    )
                except Exception as exc:
                    if memory_kind == "folio" and memory_session is not None:
                        if folio_plan is not None and memory_session.memory.query_traces:
                            memory_session.memory.query_traces[-1][
                                "historical_frames_used"
                            ] = folio_history_frames_used
                        memory_snapshot_saved = memory_session.save_snapshot(
                            memory_snapshot_dir
                        )
                    memory_metadata = None
                    if memory_session is not None:
                        memory_metadata = {
                            **memory_session.usage(),
                            "snapshot_dir": memory_snapshot_rel,
                            "snapshot_saved": memory_snapshot_saved,
                            **(
                                folio_plan.to_metadata()
                                if folio_plan is not None
                                else {}
                            ),
                            **(
                                {"historical_frame_count": folio_history_frames_used}
                                if folio_plan is not None
                                else {}
                            ),
                        }
                    record = {
                        "_key": make_key(video_basename, question, question_limit=80),
                        "video": video_basename,
                        "video_categories": video_categories.get(video_path_raw, ""),
                        "task_type": question.get("task_type", ""),
                        "time_stamp": question["time_stamp"],
                        "question": question["question"],
                        "answer_gt": extract_mcq_answer(str(question.get("answer", ""))) or str(question.get("answer", "")).strip().upper(),
                        "response": None,
                        "correct": False,
                        "error": str(exc),
                        **(
                            {"folio": memory_metadata}
                            if memory_kind == "folio" and memory_metadata is not None
                            else {"memory": memory_metadata}
                            if memory_metadata is not None
                            else {}
                        ),
                    }
                    logger.error("  [%d/%d] %s failed: %s", processed, total_questions, question["time_stamp"], exc)

                if memory_session is not None:
                    memory_stats = memory_session.usage()
                    logger.info(
                        "    memory: records=%d writes=%d write_errors=%d "
                        "stream_errors=%d snapshot=%s",
                        memory_stats["record_count"],
                        memory_stats["write_calls"],
                        memory_stats["write_errors"],
                        memory_stats["stream_errors"],
                        memory_snapshot_rel if memory_snapshot_saved else "failed",
                    )

                all_results.append(record)
                done_keys.add(record["_key"])
                ckpt_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                ckpt_file.flush()

            if memory_session is not None:
                memory_session.close()

    print_summary(all_results)
    summary = compute_summary(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {
        "qa_model": qa_model,
        "chunk_duration": chunk_duration,
        "fps": fps,
        "top_k": top_k,
        "recent_frames_only": recent_frames_only,
        "context_time": context_time,
        "cache_enabled": False,
    }
    if video_memory:
        config.update(
            {
                "video_memory": True,
                "memory_protocol": "claude-style-semantic-records",
                "memory_segment_frames": memory_segment_frames,
            }
        )
    elif folio_memory:
        config.update(
            {
                "cache_enabled": folio_config.cache_replay,
                "feature_cache_enabled": False,
                "folio_memory": True,
                "folio_standard_multimodal": True,
                "memory_protocol": "folio-paper-reimplementation-v1",
                "folio_profile": folio_profile,
                "folio_segment_seconds": float(folio_segment_seconds),
                "folio_segment_frames": folio_config.segment_frames,
                "folio_semantic_expansion": folio_config.semantic_link,
                "folio_evidence_cache": folio_config.cache_replay,
                "folio_interaction_focus": folio_config.interaction_focus,
                "folio_top_entities": folio_config.query_top_k,
                "folio_max_evidence_frames": folio_config.max_cache_frames_per_query,
            }
        )
    save_json(
        os.path.join(output_dir, f"streaming_bench_results_{timestamp}.json"),
        {
            "config": config,
            "summary": summary,
            "results": all_results,
        },
    )
    save_json(os.path.join(output_dir, "scores_report.json"), summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="StreamingBench recent-window evaluation")
    parser.add_argument("--anno-path", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--clip-device", default="cuda:0")
    parser.add_argument("--qa-model", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--qa-device", default="auto")
    parser.add_argument("--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-qa-tokens", type=int, default=256)
    parser.add_argument("--recent-frames-only", "--recent-frames-buffer", dest="recent_frames_only", type=int, default=4)
    parser.add_argument("--context-time", type=int, default=-1)
    memory_group = parser.add_mutually_exclusive_group()
    memory_group.add_argument("--video-memory", action="store_true")
    memory_group.add_argument("--folio-memory", action="store_true")
    parser.add_argument("--folio-profile", choices=("compat", "full"), default="full")
    parser.add_argument("--folio-segment-seconds", type=float, default=8.0)
    args = parser.parse_args()

    if args.clip_model or args.clip_device:
        logger.info("CLIP arguments are ignored in the release recent-window baseline.")

    if args.output_dir:
        output_dir = args.output_dir
    else:
        root_dir = Path(__file__).resolve().parents[3]
        model_tag = Path(str(args.qa_model).rstrip("/")).name.lower().replace("-instruct", "")
        run_tag = (
            f"streamingbench_release_{model_tag}"
            f"_recent{int(args.recent_frames_only)}"
            f"_chunk{str(args.chunk_duration).replace('.', 'p')}"
            f"_fps{str(args.fps).replace('.', 'p')}"
            f"{'_folio_' + args.folio_profile if args.folio_memory else ''}"
            f"{'_memory_v1' if args.video_memory else ''}"
            f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        output_dir = str(root_dir / "eval" / "StreamingBench" / "results" / run_tag)
    run_benchmark(
        anno_path=args.anno_path,
        video_dir=args.video_dir,
        output_dir=output_dir,
        qa_model=args.qa_model,
        qa_device=args.qa_device,
        chunk_duration=args.chunk_duration,
        fps=args.fps,
        top_k=args.top_k,
        max_qa_tokens=args.max_qa_tokens,
        recent_frames_only=args.recent_frames_only,
        context_time=args.context_time,
        video_memory=args.video_memory,
        folio_memory=args.folio_memory,
        folio_profile=args.folio_profile,
        folio_segment_seconds=args.folio_segment_seconds,
    )


if __name__ == "__main__":
    main()
