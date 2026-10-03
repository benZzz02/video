"""Small OpenAI-compatible vLLM runner for a local OVO-Bench subset."""

from __future__ import annotations

import argparse
import base64
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import cv2
from openai import OpenAI


def jpeg_data_uri(video_path: Path, timestamp: float, max_side: int = 672) -> str:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = max(float(cap.get(cv2.CAP_PROP_FPS)), 1.0)
    frame_count = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    index = min(max(int(round(timestamp * fps)), 0), frame_count - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot read frame {index} from {video_path}")
    height, width = frame.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        frame = cv2.resize(frame, (max(1, int(width * scale)), max(1, int(height * scale))), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        raise RuntimeError(f"Cannot encode frame from {video_path}")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def recent_timestamps(video_path: Path, count: int = 4, fps: float = 1.0) -> list[float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    raw_fps = max(float(cap.get(cv2.CAP_PROP_FPS)), 1.0)
    frame_count = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    cap.release()
    duration = frame_count / raw_fps
    end_frame = max(frame_count - 1, 0)
    end_time = end_frame / raw_fps
    start_time = max(0.0, end_time - (count - 1) / fps)
    return [start_time + i / fps for i in range(count)]


def build_prompt(item: dict[str, Any]) -> str:
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
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=args.timeout)
    records: list[dict[str, Any]] = []
    for item in annotations:
        video_path = Path(args.video_dir) / f"{item['id']}.mp4"
        timestamps = recent_timestamps(video_path, count=args.recent_frames, fps=args.fps)
        content = [
            {"type": "image_url", "image_url": {"url": jpeg_data_uri(video_path, timestamp)}}
            for timestamp in timestamps
        ]
        content.append({"type": "text", "text": build_prompt(item)})
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
        )
        elapsed = time.perf_counter() - started
        raw = response.choices[0].message.content or ""
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
            "timestamps": timestamps,
            "latency_seconds": elapsed,
        }
        records.append(record)
        print(
            f"id={item['id']} choice={choice or '-'} gt={ground_truth} "
            f"correct={record['correct']} latency={elapsed:.2f}s response={raw!r}",
            flush=True,
        )
    correct = sum(bool(record["correct"]) for record in records)
    result = {
        "config": {
            "model": args.model,
            "base_url": args.base_url,
            "recent_frames": args.recent_frames,
            "fps": args.fps,
            "video_dir": str(args.video_dir),
        },
        "summary": {"total": len(records), "correct": correct, "accuracy": correct / len(records) if records else 0.0},
        "results": records,
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
    parser.add_argument("--timeout", type=float, default=300.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
