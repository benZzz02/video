"""Small OVO-Bench adapter for the repository's causal video-memory module.

The released OVO entry points only support the Qwen recent-window path.  This
adapter keeps the OVO annotation format, uses ``VideoMemorySession`` for
question-independent causal memory writing, and delegates both memory writing
and answer generation to the configured GPT-5.6-Luna Codex model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.streaming_memory import VideoMemorySession


CODEX = "/home/zhangnuohua/.nvm/versions/node/v24.15.0/bin/codex"


def base_prompt(item: dict[str, Any]) -> str:
    options = "\n".join(
        f"{chr(65 + index)}. {option}"
        for index, option in enumerate(item["options"])
    )
    return (
        "You are evaluating one OVO-Bench video question. "
        "Use the attached four most recent frames and any VIDEO_MEMORY evidence.\n\n"
        f"Question: {item['question']}\n\n"
        f"Options:\n{options}\n\n"
        "Output ONLY one compact JSON object with keys choice (A/B/C/D), "
        "answer (the exact option text), and confidence (0 to 1)."
    )


def _write_contact_pages(
    images: list[Image.Image],
    output_dir: Path,
    call_index: int,
    page_size: int = 16,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    cell = 256
    columns = 4
    rows = 4
    for page_index, start in enumerate(range(0, len(images), page_size)):
        page_images = images[start : start + page_size]
        sheet = Image.new("RGB", (columns * cell, rows * cell), "white")
        draw = ImageDraw.Draw(sheet)
        for offset, image in enumerate(page_images):
            thumbnail = image.convert("RGB").copy()
            thumbnail.thumbnail((cell - 8, cell - 28), Image.Resampling.LANCZOS)
            x = (offset % columns) * cell + (cell - thumbnail.width) // 2
            y = (offset // columns) * cell + 24 + (cell - 24 - thumbnail.height) // 2
            sheet.paste(thumbnail, (x, y))
            draw.text(
                ((offset % columns) * cell + 6, (offset // columns) * cell + 5),
                f"frame {start + offset}",
                fill="black",
            )
        path = output_dir / f"memory_{call_index:04d}_{page_index:02d}.jpg"
        sheet.save(path, quality=82, optimize=True)
        paths.append(path)
    return paths


def call_luna(
    images: list[Image.Image],
    prompt: str,
    output_dir: Path,
    call_index: int,
    codex_home: str,
    image_paths: list[Path] | None = None,
) -> str:
    if image_paths is None:
        image_paths = _write_contact_pages(images, output_dir / "images", call_index)
    output_path = output_dir / f"call_{call_index:04d}.txt"
    log_path = output_dir / f"call_{call_index:04d}.log"
    command = [
        CODEX,
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--color",
        "never",
        "--ignore-user-config",
        "--model",
        "gpt-5.6-luna",
        "--sandbox",
        "read-only",
    ]
    for image_path in image_paths:
        command.extend(["--image", str(image_path)])
    command.extend(["-o", str(output_path), "-"])
    environment = dict(os.environ)
    environment["CODEX_HOME"] = codex_home
    completed = subprocess.run(
        command,
        input=prompt,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
        cwd=str(Path(__file__).resolve().parents[1]),
        check=False,
    )
    log_path.write_text(completed.stdout or "", encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"GPT-5.6-Luna call failed ({completed.returncode}); see {log_path}")
    if not output_path.exists():
        raise RuntimeError(f"GPT-5.6-Luna produced no final response; see {log_path}")
    return output_path.read_text(encoding="utf-8").strip()


def parse_json_response(text: str) -> dict[str, Any]:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.DOTALL).strip()
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("GPT answer is not a JSON object")
    choice = str(parsed.get("choice", "")).strip().upper()
    if choice not in {"A", "B", "C", "D"}:
        raise ValueError(f"Invalid GPT choice: {choice!r}")
    parsed["choice"] = choice
    return parsed


def recent_sheet_path(frames_dir: Path, sample_id: int) -> Path:
    path = frames_dir / f"{sample_id}_recent4.jpg"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    annotations = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    if args.max_samples is not None:
        annotations = annotations[: args.max_samples]
    annotations = sorted(annotations, key=lambda item: (str(item["video"]), item["realtime"]))

    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in annotations:
        by_source[str(item["video"])].append(item)

    run_dir = Path(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    calls_dir = run_dir / "luna_calls"
    calls_dir.mkdir(parents=True, exist_ok=True)
    call_counter = 0
    records: list[dict[str, Any]] = []

    for source_video, items in by_source.items():
        local_paths = [Path(args.video_dir) / f"{item['id']}.mp4" for item in items]
        source_path = max(local_paths, key=lambda path: path.stat().st_size)
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        def generate_memory(images: list[Image.Image], text: str, _limit: int) -> str:
            nonlocal call_counter
            result = call_luna(
                images,
                text,
                calls_dir,
                call_counter,
                args.codex_home,
            )
            call_counter += 1
            return result

        session = VideoMemorySession(
            path=str(source_path),
            fps=args.fps,
            generate=generate_memory,
            segment_frames=args.memory_segment_frames,
        )
        try:
            for item in sorted(items, key=lambda row: row["realtime"]):
                timestamp = float(item["realtime"])
                session.advance_to(timestamp)
                answer_prompt = session.augment(base_prompt(item))
                answer_text = call_luna(
                    [],
                    answer_prompt,
                    calls_dir,
                    call_counter,
                    args.codex_home,
                    image_paths=[recent_sheet_path(args.frames_dir, int(item["id"]))],
                )
                call_counter += 1
                parsed = parse_json_response(answer_text)
                ground_truth = chr(65 + int(item["gt"]))
                record = {
                    "id": item["id"],
                    "task": item["task"],
                    "video": item["video"],
                    "question": item["question"],
                    "response": parsed["choice"],
                    "answer": parsed.get("answer"),
                    "confidence": parsed.get("confidence"),
                    "ground_truth": ground_truth,
                    "correct": parsed["choice"] == ground_truth,
                    "memory_source_path": str(source_path),
                    "memory": session.usage(),
                    "raw_response": parsed,
                }
                records.append(record)
                print(
                    f"id={item['id']} choice={parsed['choice']} gt={ground_truth} "
                    f"correct={record['correct']} records={record['memory']['record_count']} "
                    f"writes={record['memory']['write_calls']}"
                )
        finally:
            session.close()

    records.sort(key=lambda row: row["id"])
    correct = sum(bool(row["correct"]) for row in records)
    result = {
        "config": {
            "model": "gpt-5.6-luna",
            "recent_frames_only": 4,
            "fps": args.fps,
            "memory_segment_frames": args.memory_segment_frames,
            "memory_module": "lib.streaming_memory.VideoMemorySession",
            "num_samples": len(records),
            "codex_calls": call_counter,
        },
        "summary": {
            "total": len(records),
            "correct": correct,
            "accuracy": (100.0 * correct / len(records)) if records else 0.0,
        },
        "results": records,
    }
    (run_dir / "gpt56_luna_ovo_memory_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--codex-home", default="/tmp/codex-home-ovo")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--memory-segment-frames", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    args.frames_dir = Path(args.frames_dir)
    run(args)


if __name__ == "__main__":
    main()
