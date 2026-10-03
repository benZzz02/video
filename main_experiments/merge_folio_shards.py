"""Merge independently evaluated FOLIO OVO video shards."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main_experiments import eval_qwen3vl_ovo_folio as base  # noqa: E402
from lib.recent_window_eval import calculate_ovo_scores, print_ovo_results  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge FOLIO OVO shard checkpoints")
    parser.add_argument("--shard_dir", action="append", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged: dict[str, dict] = {}
    for shard_name in args.shard_dir:
        path = Path(shard_name) / "results_incremental.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                key = str(item.get("_key") or base.make_key(item))
                if key in merged:
                    raise ValueError(f"Duplicate shard result: {key}")
                merged[key] = item

    checkpoint = output_dir / "results_incremental.jsonl"
    with checkpoint.open("w", encoding="utf-8") as handle:
        for key in sorted(merged):
            handle.write(json.dumps(merged[key], ensure_ascii=False) + "\n")

    backward, realtime, forward = base._merge_results(output_dir)
    summary = calculate_ovo_scores(backward, realtime, forward)
    print_ovo_results("Qwen3-VL + FOLIO (shared video memory, sharded)", backward, realtime, forward)
    final_path = output_dir / "qwen3vl_folio_sharded_results.json"
    final_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "backward": backward,
                "realtime": realtime,
                "forward": forward,
                "shard_dirs": args.shard_dir,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Merged {len(merged)} annotations into {final_path}")


if __name__ == "__main__":
    main()
