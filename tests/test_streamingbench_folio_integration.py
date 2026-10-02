"""End-to-end contract for the optional FOLIO StreamingBench mode.

The recent-window model and video decoder are stubbed so this test exercises
the benchmark wiring, causal session order, and persisted artifacts without a
GPU or a real video file.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image


def _frame(value: int) -> Image.Image:
    return Image.new("RGB", (12, 8), (value, value, value))


def _writer_output(start: int, end: int) -> str:
    return json.dumps(
        {
            "time": f"{start}.0-{end}.0s",
            "detailed_objects": [
                {
                    "name": "bookshelf",
                    "aliases": [],
                    "category": "furniture",
                    "attributes": ["wooden"],
                    "location": "study wall",
                    "holder": "",
                    "state": "filled with books",
                    "relations": [],
                    "interactions": [],
                    "state_change": "newly visible" if start == 0 else "stable",
                    "visible_text": "",
                    "evidence_summary": "A wooden bookshelf is visible by the study wall.",
                    "evidence_frames": [0],
                    "confidence": 0.95,
                }
            ],
            "compact_objects": [],
            "events": [],
        }
    )


class StreamingBenchFolioIntegrationCheck(unittest.TestCase):
    def test_folio_is_causal_and_keeps_the_baseline_contract_unchanged(self):
        from lib import folio_memory

        # Import the benchmark against a tiny recent-window module so importing
        # this integration test never pulls in torch or a real Qwen checkpoint.
        recent_window_stub = ModuleType("lib.recent_window_eval")
        recent_window_stub.RecentWindowQAModel = object
        recent_window_stub.extract_mcq_answer = (
            lambda value: str(value).strip().upper()[:1]
            if value and str(value).strip().upper()[:1] in "ABCD"
            else None
        )
        recent_window_stub.load_jsonl_results = lambda _path: ([], set())
        recent_window_stub.query_recent_window = lambda **_kwargs: None

        def save_json(path, value):
            Path(path).write_text(json.dumps(value), encoding="utf-8")

        recent_window_stub.save_json = save_json
        sys.modules.pop("main_experiments.eval_streamingbench", None)
        with patch.dict(sys.modules, {"lib.recent_window_eval": recent_window_stub}):
            benchmark = importlib.import_module("main_experiments.eval_streamingbench")

        questions = [
            {
                "time_stamp": "00:00:07",
                "question": "Which description best fits this person?",
                "options": ["Avid reader", "Musician", "Athlete", "Unable to answer"],
                "answer": "A",
                "task_type": "concept",
            },
            {
                "time_stamp": "00:00:15",
                "question": "Where is the bookshelf now?",
                "options": ["Study wall", "Kitchen", "Outside", "Unknown"],
                "answer": "A",
                "task_type": "location",
            },
        ]
        writer_outputs = iter([_writer_output(0, 7), _writer_output(8, 15)])
        writer_prompts: list[str] = []
        writer_frames: list[list[Image.Image]] = []
        semlink_prompts: list[str] = []
        query_calls: list[dict] = []
        source_calls: list[tuple[str, float]] = []

        class FakeQA:
            def __init__(self, **kwargs):
                self.max_new_tokens = kwargs["max_new_tokens"]

            def generate_from_frames(self, frames, prompt):
                writer_frames.append(list(frames))
                writer_prompts.append(prompt)
                return next(writer_outputs)

            def generate_from_text(self, prompt):
                semlink_prompts.append(prompt)
                return json.dumps(
                    {
                        "relevant_entity_ids": ["entity-0001"],
                        "relevant_event_ids": [],
                        "reason": "The bookshelf supports the reading description.",
                        "suggested_option": "A",
                    }
                )

        def fake_query_recent_window(**kwargs):
            query_calls.append(kwargs)
            history_count = len(kwargs.get("historical_frames", ()))
            result = SimpleNamespace(
                answer="A",
                final_chunk_ids=[0, 1, 2, 3],
                generate_time=0.01,
                ttft_seconds=0.001,
                num_frames=4 + history_count,
                num_vision_tokens=4,
                num_vision_tokens_before=4,
                num_vision_tokens_after=4,
            )
            return result, "fake_window"

        def fake_video_frames(path, fps):
            source_calls.append((path, fps))
            # One look-ahead frame lets advance_to(15) stop without exhausting
            # the stream while still committing exactly two eight-frame segments.
            for timestamp in range(17):
                yield float(timestamp), _frame(timestamp)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_dir = root / "videos"
            video_dir.mkdir()
            video_path = video_dir / "sample.mp4"
            video_path.touch()
            anno_path = root / "annotations.json"
            anno_path.write_text(
                json.dumps(
                    [
                        {
                            "video_path": "sample.mp4",
                            "video_categories": "test",
                            "questions": questions,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            common = dict(
                anno_path=str(anno_path),
                video_dir=str(video_dir),
                qa_model="fake",
                qa_device="cpu",
                chunk_duration=1.0,
                fps=1.0,
                top_k=0,
                max_qa_tokens=16,
                recent_frames_only=4,
                context_time=-1,
            )

            with patch.object(benchmark, "RecentWindowQAModel", FakeQA), patch.object(
                benchmark, "query_recent_window", fake_query_recent_window
            ), patch.object(folio_memory, "video_frames", fake_video_frames):
                baseline_dir = root / "baseline"
                # This intentionally uses the pre-FOLIO call shape.
                benchmark.run_benchmark(
                    output_dir=str(baseline_dir), video_memory=False, **common
                )
                baseline_calls = list(query_calls)
                query_calls.clear()

                folio_dir = root / "folio"
                benchmark.run_benchmark(
                    output_dir=str(folio_dir),
                    video_memory=False,
                    folio_memory=True,
                    folio_profile="full",
                    folio_segment_seconds=8.0,
                    **common,
                )
                folio_calls = list(query_calls)

            self.assertEqual(len(baseline_calls), 2)
            self.assertEqual(len(folio_calls), 2)
            baseline_query_keys = {
                "qa",
                "video_path",
                "prompt",
                "chunk_duration",
                "fps",
                "recent_frames_only",
                "video_start",
                "video_end",
            }
            for baseline, folio, question in zip(
                baseline_calls, folio_calls, questions
            ):
                self.assertEqual(set(baseline), baseline_query_keys)
                self.assertEqual(baseline["prompt"], benchmark.build_prompt(question))
                self.assertEqual(set(folio), baseline_query_keys | {"historical_frames"})
                self.assertIn(question["question"], folio["prompt"])
                for option in question["options"]:
                    self.assertIn(option, folio["prompt"])
                self.assertIn("## PREVIOUS DIALOGUE", folio["prompt"])
                self.assertIn("## CONSOLIDATED MEMORY", folio["prompt"])
                for key in (
                    "video_path",
                    "chunk_duration",
                    "fps",
                    "recent_frames_only",
                    "video_start",
                    "video_end",
                ):
                    self.assertEqual(baseline[key], folio[key])

            # The first conceptual query replays evidence older than the recent
            # window. The second direct query still receives the optional field,
            # but needs no historical image.
            self.assertGreaterEqual(len(folio_calls[0]["historical_frames"]), 1)
            self.assertTrue(
                all(
                    isinstance(frame, Image.Image)
                    for frame in folio_calls[0]["historical_frames"]
                )
            )
            self.assertEqual(tuple(folio_calls[1]["historical_frames"]), ())

            # A FOLIO session is opened only for the FOLIO run. It observes eight
            # source frames per writer transaction. Adaptive selection sends two
            # boundaries initially and four keyframes once interaction focus is set.
            self.assertEqual(source_calls, [(str(video_path), 1.0)])
            self.assertEqual(len(writer_prompts), 2)
            self.assertEqual([len(frames) for frames in writer_frames], [2, 4])
            self.assertIn("Time range: 0.000s - 7.000s", writer_prompts[0])
            self.assertIn("Time range: 8.000s - 15.000s", writer_prompts[1])
            self.assertEqual(len(semlink_prompts), 1)

            # Questions are prepared only after the writer has committed the
            # current segment. The first interaction therefore cannot affect its
            # own writer prompt, but its committed entity focus affects segment 2.
            for question in questions:
                self.assertNotIn(question["question"], "\n".join(writer_prompts))
            self.assertIn("DETAILED existing entities: none", writer_prompts[0])
            self.assertIn(
                "DETAILED existing entities: entity-0001:bookshelf",
                writer_prompts[1],
            )
            self.assertIn("(no previous turns)", folio_calls[0]["prompt"])
            self.assertIn(
                "model prediction: A Avid reader", folio_calls[1]["prompt"]
            )

            baseline_result = json.loads(
                next(baseline_dir.glob("streaming_bench_results_*.json")).read_text(
                    encoding="utf-8"
                )
            )
            folio_result = json.loads(
                next(folio_dir.glob("streaming_bench_results_*.json")).read_text(
                    encoding="utf-8"
                )
            )

            baseline_config = {
                "qa_model": "fake",
                "chunk_duration": 1.0,
                "fps": 1.0,
                "top_k": 0,
                "recent_frames_only": 4,
                "context_time": -1,
                "cache_enabled": False,
            }
            self.assertEqual(set(baseline_result), {"config", "summary", "results"})
            self.assertEqual(baseline_result["config"], baseline_config)
            baseline_record_keys = {
                "_key",
                "video",
                "video_categories",
                "task_type",
                "time_stamp",
                "question",
                "answer_gt",
                "response",
                "correct",
                "decode_backend",
                "final_chunk_ids",
                "generate_time",
                "ttft_seconds",
                "num_vision_tokens",
                "num_vision_tokens_before",
                "num_vision_tokens_after",
            }
            self.assertTrue(
                all(set(record) == baseline_record_keys for record in baseline_result["results"])
            )
            self.assertFalse((baseline_dir / "folio_memory").exists())

            expected_folio_config = {
                **baseline_config,
                "cache_enabled": True,
                "feature_cache_enabled": False,
                "folio_memory": True,
                "memory_protocol": "folio-paper-reimplementation-v1",
                "folio_profile": "full",
                "folio_segment_seconds": 8.0,
                "folio_segment_frames": 8,
                "folio_semantic_expansion": True,
                "folio_evidence_cache": True,
                "folio_interaction_focus": True,
                "folio_top_entities": 6,
                "folio_max_evidence_frames": 2,
            }
            self.assertEqual(folio_result["config"], expected_folio_config)
            first_folio = folio_result["results"][0]["folio"]
            second_folio = folio_result["results"][1]["folio"]
            self.assertNotIn("memory", folio_result["results"][0])
            self.assertEqual(first_folio["observed_frames"], 8)
            self.assertEqual(first_folio["segments_written"], 1)
            self.assertEqual(first_folio["write_calls"], 1)
            self.assertEqual(first_folio["retrieval_mode"], "semlink")
            self.assertEqual(first_folio["selected_entity_ids"], ["entity-0001"])
            self.assertEqual(first_folio["recovered_evidence_timestamps"], [0.0])
            self.assertEqual(first_folio["historical_frame_count"], 1)
            self.assertEqual(first_folio["recent_frame_count"], 4)
            self.assertTrue(first_folio["snapshot_saved"])
            self.assertEqual(second_folio["observed_frames"], 16)
            self.assertEqual(second_folio["segments_written"], 2)
            self.assertEqual(second_folio["write_calls"], 2)
            self.assertEqual(second_folio["entity_count"], 1)
            self.assertEqual(second_folio["query_count"], 2)
            self.assertEqual(second_folio["historical_frame_count"], 0)
            self.assertEqual(second_folio["recent_frame_count"], 4)

            snapshot_rel = second_folio["snapshot_dir"]
            self.assertEqual(snapshot_rel, "folio_memory/0001_sample")
            snapshot_dir = folio_dir / snapshot_rel
            for relative in (
                "MEMORY.md",
                "state.json",
                "focus.json",
                "evidence/index.json",
                "usage.json",
                "queries.jsonl",
                "entities/entity-0001.md",
            ):
                self.assertTrue((snapshot_dir / relative).is_file(), relative)

            state = json.loads((snapshot_dir / "state.json").read_text(encoding="utf-8"))
            focus = json.loads((snapshot_dir / "focus.json").read_text(encoding="utf-8"))
            usage = json.loads((snapshot_dir / "usage.json").read_text(encoding="utf-8"))
            queries = [
                json.loads(line)
                for line in (snapshot_dir / "queries.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(state["schema"], "folio-memory-state-v1")
            self.assertEqual(state["config"]["segment_frames"], 8)
            self.assertEqual(state["segment_index"], 2)
            self.assertEqual(len(state["committed_query_ids"]), 2)
            self.assertEqual(focus["entities"]["entity-0001"]["level"], "focus")
            self.assertEqual(usage["observed_frames"], 16)
            self.assertEqual(usage["write_calls"], 2)
            self.assertEqual(len(queries), 2)
            self.assertEqual([item["question"] for item in queries], [
                question["question"] for question in questions
            ])
            self.assertGreaterEqual(len(list((snapshot_dir / "evidence").glob("*.jpg"))), 1)


if __name__ == "__main__":
    unittest.main()
