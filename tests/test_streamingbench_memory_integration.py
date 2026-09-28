import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType
import tempfile
import unittest
from unittest.mock import patch


class StreamingBenchMemoryIntegrationCheck(unittest.TestCase):
    def test_memory_wraps_only_the_original_question_prompt(self):
        from lib import streaming_memory

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
            Path(path).write_text(json.dumps(value))

        recent_window_stub.save_json = save_json
        sys.modules.pop("main_experiments.eval_streamingbench", None)
        with patch.dict(
            sys.modules, {"lib.recent_window_eval": recent_window_stub}
        ):
            benchmark = importlib.import_module("main_experiments.eval_streamingbench")

        questions = [
            {
                "time_stamp": "00:00:03",
                "question": "Which code is visible?",
                "options": ["4821", "1234", "0000", "9999"],
                "answer": "A",
                "task_type": "memory",
            },
            {
                "time_stamp": "00:00:07",
                "question": "What happened to the door?",
                "options": ["Opened", "Closed", "Vanished", "Unknown"],
                "answer": "A",
                "task_type": "memory",
            },
        ]
        writer_outputs = iter(
            [
                json.dumps(
                    {
                        "operations": [
                            {
                                "op": "upsert",
                                "name": "access-code",
                                "description": "Code 4821 is visible at 0–3s.",
                                "type": "fact",
                                "content": "[0–3s] The panel shows 4821.",
                            }
                        ]
                    }
                ),
                json.dumps(
                    {
                        "operations": [
                            {
                                "op": "upsert",
                                "name": "door-state",
                                "description": "The door opens at 4–7s.",
                                "type": "change",
                                "content": "[4s] Closed; [7s] open.",
                            }
                        ]
                    }
                ),
            ]
        )
        writer_prompts = []
        query_calls = []

        class FakeQA:
            def __init__(self, **kwargs):
                self.max_new_tokens = kwargs["max_new_tokens"]

            def generate_from_frames(self, frames, prompt):
                writer_prompts.append(prompt)
                self.assert_not_used = frames
                return next(writer_outputs)

        def fake_query_recent_window(**kwargs):
            query_calls.append(kwargs)
            result = SimpleNamespace(
                answer="A",
                final_chunk_ids=[0, 1, 2, 3],
                generate_time=0.01,
                ttft_seconds=0.001,
                num_vision_tokens=4,
                num_vision_tokens_before=4,
                num_vision_tokens_after=4,
            )
            return result, "fake_window"

        def fake_video_frames(_path, _fps):
            for timestamp in range(9):
                yield float(timestamp), SimpleNamespace(mode="RGB", size=(2, 2))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video_dir = root / "videos"
            video_dir.mkdir()
            (video_dir / "sample.mp4").touch()
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
                )
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
            ), patch.object(streaming_memory, "video_frames", fake_video_frames):
                baseline_dir = root / "baseline"
                benchmark.run_benchmark(
                    output_dir=str(baseline_dir), video_memory=False, **common
                )
                baseline_calls = list(query_calls)
                query_calls.clear()

                memory_dir = root / "memory"
                benchmark.run_benchmark(
                    output_dir=str(memory_dir), video_memory=True, **common
                )
                memory_calls = list(query_calls)

            self.assertEqual(len(baseline_calls), 2)
            self.assertEqual(len(memory_calls), 2)
            self.assertEqual(len(writer_prompts), 2)
            self.assertNotIn("Which code", "\n".join(writer_prompts))
            self.assertNotIn("What happened", "\n".join(writer_prompts))

            for baseline, memory, question in zip(
                baseline_calls, memory_calls, questions
            ):
                for key in (
                    "video_path",
                    "chunk_duration",
                    "fps",
                    "recent_frames_only",
                    "video_start",
                    "video_end",
                ):
                    self.assertEqual(baseline[key], memory[key])
                original = benchmark.build_prompt(question)
                self.assertEqual(baseline["prompt"], original)
                self.assertTrue(memory["prompt"].endswith(original))

            baseline_result = json.loads(
                next(baseline_dir.glob("streaming_bench_results_*.json")).read_text()
            )
            memory_result = json.loads(
                next(memory_dir.glob("streaming_bench_results_*.json")).read_text()
            )
            self.assertNotIn("video_memory", baseline_result["config"])
            self.assertEqual(memory_result["config"]["video_memory"], True)
            self.assertEqual(
                memory_result["config"]["memory_protocol"],
                "claude-style-semantic-records",
            )
            self.assertEqual(memory_result["results"][0]["memory"]["record_count"], 1)
            self.assertEqual(memory_result["results"][1]["memory"]["record_count"], 2)


if __name__ == "__main__":
    unittest.main()
