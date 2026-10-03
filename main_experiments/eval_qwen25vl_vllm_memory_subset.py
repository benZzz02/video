"""Qwen2.5-VL vLLM evaluation with the repository's VideoMemorySession."""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import time
from collections import defaultdict, deque
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

import cv2
from openai import OpenAI
from PIL import Image, ImageDraw

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lib.streaming_memory import VideoMemorySession


def cv2_frames(path: str | Path, fps: float) -> Iterable[tuple[float, Image.Image]]:
    """Fixed-clock RGB frame stream compatible with VideoMemorySession."""
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    source_fps = max(float(cap.get(cv2.CAP_PROP_FPS)), 1.0)
    frame_index = 0
    next_tick = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_index / source_fps
            frame_index += 1
            if timestamp * fps + 1e-9 < next_tick:
                continue
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image.thumbnail((448, 448), Image.Resampling.LANCZOS)
            yield timestamp, image
            next_tick = int(timestamp * fps) + 1
    finally:
        cap.release()


def recent_images(path: str | Path, count: int, fps: float) -> list[Image.Image]:
    window: deque[Image.Image] = deque(maxlen=max(1, int(count)))
    for _, image in cv2_frames(path, fps):
        window.append(image)
    return list(window)


def contact_sheets(images: list[Image.Image], page_size: int = 16) -> list[Image.Image]:
    pages: list[Image.Image] = []
    cell = 256
    columns = 4
    for start in range(0, len(images), page_size):
        page_images = images[start : start + page_size]
        rows = max(1, math.ceil(len(page_images) / columns))
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
        pages.append(sheet)
    return pages


def data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class VLLMGenerator:
    def __init__(self, base_url: str, model: str, timeout: float):
        self.client = OpenAI(base_url=base_url, api_key="EMPTY", timeout=timeout)
        self.model = model
        self.calls = 0
        self.memory_calls = 0
        self.answer_calls = 0

    def generate(self, images: list[Image.Image], prompt: str, max_tokens: int, memory: bool = False) -> str:
        visual_inputs = contact_sheets(images) if memory else images
        content = [
            {"type": "image_url", "image_url": {"url": data_uri(image)}}
            for image in visual_inputs
        ]
        content.append({"type": "text", "text": prompt})
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": min(int(max_tokens), 3072),
            "temperature": 0.0,
            "top_p": 1.0,
        }
        if memory:
            kwargs["response_format"] = {"type": "json_object"}
        self.calls += 1
        if memory:
            self.memory_calls += 1
        else:
            self.answer_calls += 1
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception:
            if not memory:
                raise
            # Keep the memory module fail-open if this vLLM build rejects JSON mode.
            kwargs.pop("response_format", None)
            response = self.client.chat.completions.create(**kwargs)
        return (response.choices[0].message.content or "").strip()


def base_prompt(item: dict[str, Any]) -> str:
    options = "; ".join(
        f"{chr(65 + index)}. {option}" for index, option in enumerate(item["options"])
    )
    return f'{item["question"]}\nOptions: {options};\nOnly give the best option\'s letter directly.'


def extract_choice(text: str) -> str | None:
    match = re.search(r"(?<![A-Z])([A-D])(?![A-Z])", text.upper())
    if match:
        return match.group(1)
    match = re.search(r"\b([1-4])\b", text)
    return chr(64 + int(match.group(1))) if match else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    annotations = json.loads(Path(args.annotations).read_text(encoding="utf-8"))[: args.limit]
    annotations = sorted(annotations, key=lambda item: (str(item["video"]), item["realtime"]))
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in annotations:
        by_source[str(item["video"])].append(item)

    generator = VLLMGenerator(args.base_url, args.model, args.timeout)
    records: list[dict[str, Any]] = []
    for source_video, items in by_source.items():
        paths = [Path(args.video_dir) / f"{item['id']}.mp4" for item in items]
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise FileNotFoundError(missing[0])
        source_path = max(paths, key=lambda path: path.stat().st_size)
        session = VideoMemorySession(
            path=str(source_path),
            fps=args.fps,
            generate=lambda images, prompt, limit: generator.generate(images, prompt, limit, memory=True),
            segment_frames=args.memory_segment_frames,
            frame_source=cv2_frames(source_path, args.fps),
        )
        try:
            for item in sorted(items, key=lambda row: row["realtime"]):
                session.advance_to(float(item["realtime"]))
                prompt = session.augment(base_prompt(item))
                frames = recent_images(paths[items.index(item)], args.recent_frames, args.fps)
                started = time.perf_counter()
                raw = generator.generate(frames, prompt, args.answer_tokens, memory=False)
                elapsed = time.perf_counter() - started
                choice = extract_choice(raw)
                ground_truth = chr(65 + int(item["gt"]))
                record = {
                    "id": item["id"],
                    "task": item["task"],
                    "video": item["video"],
                    "question": item["question"],
                    "response": raw,
                    "choice": choice,
                    "ground_truth": ground_truth,
                    "correct": choice == ground_truth,
                    "latency_seconds": elapsed,
                    "memory": session.usage(),
                    "memory_index": session.memory.memory_index(),
                    "memory_records": [asdict(value) for value in session.memory.records.values()],
                }
                records.append(record)
                print(
                    f"id={item['id']} choice={choice or '-'} gt={ground_truth} "
                    f"correct={record['correct']} records={record['memory']['record_count']} "
                    f"writes={record['memory']['write_calls']} errors={record['memory']['write_errors']} "
                    f"answer={elapsed:.2f}s response={raw!r}",
                    flush=True,
                )
        finally:
            session.close()

    correct = sum(bool(record["correct"]) for record in records)
    result = {
        "config": {
            "model": args.model,
            "base_url": args.base_url,
            "memory_module": "lib.streaming_memory.VideoMemorySession",
            "memory_segment_frames": args.memory_segment_frames,
            "memory_writer_input": "16-frame contact sheets",
            "recent_frames": args.recent_frames,
            "fps": args.fps,
            "video_dir": str(args.video_dir),
            "num_samples": len(records),
            "vllm_calls": generator.calls,
            "memory_writer_calls": generator.memory_calls,
            "answer_calls": generator.answer_calls,
        },
        "summary": {"total": len(records), "correct": correct, "accuracy": correct / len(records) if records else 0.0},
        "results": sorted(records, key=lambda record: record["id"]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"accuracy={correct}/{len(records)} ({100.0 * result['summary']['accuracy']:.1f}%)", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8011/v1")
    parser.add_argument("--model", default="qwen2.5-vl")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--recent-frames", type=int, default=4)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--memory-segment-frames", type=int, default=32)
    parser.add_argument("--answer-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
