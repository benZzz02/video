"""CPU contracts for shared OVO memory and task-specific query routing.

The writer, answer model, and video decoder are replaced with small fakes.
The chronological integration check uses the real FOLIO session and retrieval
implementation, without importing torch, transformers, or a model checkpoint.
"""
from __future__ import annotations

from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from lib.folio_memory import FolioConfig, FolioMemorySession
from ovo_constants import FORWARD_TASKS, REAL_TIME_TASKS


def _load_runner():
    """Isolate heavy dependencies without polluting other tests' imports."""
    import main_experiments

    recent = ModuleType("lib.recent_window_eval")
    recent.RecentWindowQAModel = object
    recent.build_ovo_prompt = lambda *_args, **_kwargs: "original OVO prompt"
    recent.calculate_ovo_scores = lambda *_args: None
    recent.extract_mcq_answer = lambda text: str(text).strip()[:1] or None
    recent.load_jsonl_results = lambda _path: ([], set())
    recent.print_ovo_results = lambda *_args: None
    recent.query_recent_window = lambda **_kwargs: None
    base = ModuleType("main_experiments.eval_qwen3vl_ovo_folio")
    base.make_key = lambda anno: f"{anno['task']}:{anno['id']}"
    base._opencv_frame_source = lambda *_args: iter(())
    base._forward_question = lambda anno, _index: anno.get("question", "Which step?")
    base._format_non_mcq_folio_prompt = (
        lambda memory, original: f"<FOCUSED_VIDEO_MEMORY>\n{memory}\n</FOCUSED_VIDEO_MEMORY>\n\n{original}"
    )
    path = Path(__file__).resolve().parents[1] / "main_experiments/eval_qwen3vl_ovo_folio_fast.py"
    spec = importlib.util.spec_from_file_location("_ovo_folio_routing_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {
        "lib.recent_window_eval": recent,
        "main_experiments.eval_qwen3vl_ovo_folio": base,
    }), patch.object(main_experiments, "eval_qwen3vl_ovo_folio", base, create=True):
        spec.loader.exec_module(module)
    return module


def _result(answer="A"):
    return SimpleNamespace(
        answer=answer,
        final_chunk_ids=[0, 1],
        generate_time=0.01,
        ttft_seconds=0.002,
        num_vision_tokens=20,
        num_vision_tokens_before=20,
        num_vision_tokens_after=20,
        num_frames=2,
    ), "fake"


def _mock_session(config, write_calls=0, write_seconds=0.0):
    session = Mock()
    session.memory.config = config
    session.memory.write_calls = write_calls
    session.memory.write_seconds = write_seconds
    session.usage.return_value = {
        "write_calls": write_calls,
        "write_seconds": write_seconds,
        "write_errors": 0,
        "semantic_link_calls": 0,
    }
    return session


def _query(task="OCR", timestamp=7.0, identifier="sample"):
    key = f"{task}:{identifier}"
    forward = task in FORWARD_TASKS
    original = {
        "REC": "How many repetitions?\nOnly give a number as answer.",
        "SSR": "Is the person moving the bookshelf?\nAnswer Yes or No only.",
        "CRR": "Is there enough information?\nAnswer Yes or No only.",
    }.get(task, "Where is the bookshelf?\nOptions: A: study, B: outside\nOnly give the best option's letter directly.")
    return {
        "annotation_key": key,
        "query_id": f"{key}:0" if forward else key,
        "query_index": 0 if forward else None,
        "task": task,
        "source_video": "shared-video",
        "query_time": timestamp,
        "clip_path": "/unused/prefix.mp4",
        "question": "Where is the bookshelf?",
        "options": [] if forward else ["study", "outside"],
        "original_prompt": original,
    }


def _annotation(spec):
    annotation = {
        "id": spec["annotation_key"].split(":", 1)[1],
        "video": spec["source_video"],
        "task": spec["task"],
        "question": spec["question"],
        "gt": 0,
    }
    if spec["query_index"] is not None:
        annotation["test_info"] = [{"realtime": spec["query_time"], "gt": 1}]
    return annotation


def _writer_json(end):
    return json.dumps({
        "detailed_objects": [{
            "name": "bookshelf",
            "aliases": [],
            "category": "furniture",
            "attributes": ["wooden"],
            "location": "study",
            "state": f"seen-at-{end}",
            "evidence_summary": f"bookshelf observed at timestamp {end}",
            "evidence_frames": [0],
            "confidence": 0.9,
        }],
        "compact_objects": [],
        "events": [],
    })


class OvoFolioTaskRoutingCheck(unittest.TestCase):
    def setUp(self):
        self.runner = _load_runner()

    def _call(self, spec, session, config, **kwargs):
        return self.runner.run_shared_query(
            qa=object(),
            session=session,
            source_path="source.mp4",
            spec=spec,
            chunk_duration=1.0,
            fps=1.0,
            recent_frames_only=2,
            folio_config=config,
            **kwargs,
        )

    def _assert_timings(self, metadata):
        for key in (
            "memory_advance_seconds", "retrieval_seconds", "answer_seconds",
            "query_wall_seconds", "write_calls_delta", "write_seconds_delta",
        ):
            self.assertIn(key, metadata)
            self.assertGreaterEqual(metadata[key], 0, key)
        self.assertGreaterEqual(metadata["query_wall_seconds"], metadata["answer_seconds"])

    def test_task_policy_disables_extra_inference_without_mutating_config(self):
        original = FolioConfig.profile("full", segment_frames=16)
        configured = self.runner.configure_query_policy(original, "task_routed")
        for field in ("semantic_link", "cache_replay", "interaction_focus", "structured_answer"):
            self.assertFalse(getattr(configured, field), field)
            self.assertTrue(getattr(original, field), field)
        self.assertEqual(configured.segment_frames, 16)
        self.assertEqual(configured.memory_bytes, original.memory_bytes)
        legacy = self.runner.configure_query_policy(original, "all")
        for field in ("semantic_link", "cache_replay", "structured_answer"):
            self.assertEqual(getattr(legacy, field), getattr(original, field), field)
        with self.assertRaises(ValueError):
            self.runner.configure_query_policy(original, "unsupported")

    def test_every_realtime_task_bypasses_writer_and_retrieval(self):
        config = self.runner.configure_query_policy(FolioConfig(), "task_routed")
        for task in REAL_TIME_TASKS:
            with self.subTest(task=task):
                spec = _query(task)
                session = _mock_session(config, write_calls=8, write_seconds=4.0)
                with patch.object(self.runner, "query_recent_window", return_value=_result()) as answer:
                    response, metadata, error = self._call(
                        spec, session, config, query_policy="task_routed"
                    )
                self.assertEqual(response, "A")
                self.assertIsNone(error)
                session.advance_to.assert_not_called()
                session.prepare_query.assert_not_called()
                session.commit_interaction.assert_not_called()
                call = answer.call_args.kwargs
                self.assertEqual(call["prompt"], spec["original_prompt"])
                self.assertEqual(call["video_start"], 5.0)
                self.assertLessEqual(call["video_end"], 7.0001)
                self.assertEqual(tuple(call.get("historical_frames", ())), ())
                self.assertEqual(metadata["memory_route"], "recent_only")
                self.assertFalse(metadata["memory_used"])
                self.assertEqual(metadata["write_calls_delta"], 0)
                self.assertEqual(metadata["write_seconds_delta"], 0)
                self.assertEqual(metadata["memory_advance_seconds"], 0)
                self.assertEqual(metadata["retrieval_seconds"], 0)
                self._assert_timings(metadata)

    def test_all_policy_keeps_full_memory_and_structured_answer_for_realtime(self):
        config = replace(FolioConfig(), interaction_focus=False)
        spec = _query("OCR")
        session = _mock_session(config)
        plan = SimpleNamespace(
            prompt="full FOLIO memory prompt",
            recent_only_prompt="memory prompt without replay",
            memory_text="bookshelf in study",
            evidence_frames=(Image.new("RGB", (2, 2)),),
            selected_records=("bookshelf in study",),
            to_metadata=lambda: {"retrieval_mode": "direct"},
        )
        session.prepare_query.return_value = plan
        with patch.object(self.runner, "query_recent_window", return_value=_result()) as answer:
            # The old call shape (without query_policy) must retain all routing.
            response, metadata, error = self._call(spec, session, config)
        self.assertEqual(response, "A")
        self.assertIsNone(error)
        session.advance_to.assert_called_once_with(7.0)
        session.prepare_query.assert_called_once()
        self.assertIn("full FOLIO memory prompt", answer.call_args.kwargs["prompt"])
        self.assertIn("prediction_label", answer.call_args.kwargs["prompt"])
        self.assertEqual(answer.call_args.kwargs["historical_frames"], plan.evidence_frames)
        self.assertEqual(metadata["memory_route"], "full_memory")
        self.assertTrue(metadata["memory_used"])
        self._assert_timings(metadata)

    def test_forward_tasks_keep_the_number_or_yes_no_answer_protocol(self):
        config = self.runner.configure_query_policy(FolioConfig(), "task_routed")
        for task in FORWARD_TASKS:
            with self.subTest(task=task):
                spec = _query(task)
                session = _mock_session(config)
                session.prepare_query.return_value = SimpleNamespace(
                    prompt="unused memory answer template",
                    recent_only_prompt=spec["original_prompt"],
                    memory_text="bookshelf was moved twice",
                    evidence_frames=(),
                    selected_records=("bookshelf was moved twice",),
                    to_metadata=lambda: {"retrieval_mode": "direct"},
                )
                expected_answer = "2" if task == "REC" else "Yes"
                with patch.object(self.runner, "query_recent_window", return_value=_result(expected_answer)) as answer:
                    response, metadata, error = self._call(
                        spec, session, config, query_policy="task_routed"
                    )
                self.assertIsNone(error)
                self.assertEqual(response, expected_answer)
                self.assertTrue(answer.call_args.kwargs["prompt"].endswith(spec["original_prompt"]))
                self.assertIn("bookshelf was moved twice", answer.call_args.kwargs["prompt"])
                self.assertNotIn("prediction_label", answer.call_args.kwargs["prompt"])
                self.assertEqual(metadata["memory_route"], "text_memory")
                self.assertTrue(metadata["memory_used"])
                session.advance_to.assert_called_once_with(7.0)
                session.prepare_query.assert_called_once()

    def _run_group(self, specs, qa, config):
        annotations = {spec["annotation_key"]: _annotation(spec) for spec in specs}
        records = {key: self.runner._record_template(anno) for key, anno in annotations.items()}
        checkpoint = io.StringIO()
        done = set()
        self.runner.run_group(
            group_specs=specs,
            annotations_by_key=annotations,
            records_by_key=records,
            done_keys=done,
            checkpoint=checkpoint,
            qa=qa,
            chunk_duration=1.0,
            fps=1.0,
            recent_frames_only=2,
            folio_config=config,
            query_policy="task_routed",
        )
        self.assertEqual(done, set(records))
        self.assertEqual(len(checkpoint.getvalue().splitlines()), len(records))
        return records

    def test_realtime_only_video_never_consumes_a_memory_frame_source(self):
        config = self.runner.configure_query_policy(FolioConfig(), "task_routed")
        specs = [_query("OCR", 7.0, "one"), _query("ATR", 700.0, "two")]
        def source_frames(*_args):
            raise AssertionError("RT must not consume the video source")
            yield  # Make the decoder lazy, as the real OpenCV generator is.

        writer = Mock(side_effect=AssertionError("RT must not invoke the writer"))
        with patch.object(self.runner, "_choose_source_path", return_value="source.mp4"), \
                patch.object(self.runner.base, "_opencv_frame_source", side_effect=source_frames), \
                patch.object(self.runner, "query_recent_window", return_value=_result()) as answer:
            records = self._run_group(specs, SimpleNamespace(_folio_generate_memory=writer), config)
        writer.assert_not_called()
        self.assertEqual(answer.call_count, 2)
        for record in records.values():
            self.assertEqual(record["response"], "A")
            self.assertEqual(record["folio"]["memory_route"], "recent_only")
            self.assertEqual(record["folio"]["write_calls_delta"], 0)
            self.assertEqual(record["folio"]["observed_frames"], 0)
            self.assertEqual(record["folio"]["stream_errors"], 0)
            self._assert_timings(record["folio"])

    def test_backward_and_forward_share_one_causal_memory_across_realtime_queries(self):
        config = self.runner.configure_query_policy(FolioConfig(segment_frames=4), "task_routed")
        specs = [
            _query("OCR", 20.0, "late-rt"),
            _query("EPM", 7.0, "second-bt"),
            _query("SSR", 11.0, "forward"),
            _query("HLD", 3.0, "first-bt"),
            _query("ATR", 5.0, "middle-rt"),
            _query("ASI", 7.0, "same-time-bt"),
        ]
        sessions = []
        writes = []
        answers = []
        yielded = []

        def frames(_path, _fps):
            for timestamp in range(30):
                yielded.append(timestamp)
                yield float(timestamp), Image.new("RGB", (8, 8), (timestamp, 0, 0))

        def writer(images, prompt, _limit):
            self.assertTrue(images, "semantic linking must not make a text-only model call")
            end = max(image.getpixel((0, 0))[0] for image in images)
            writes.append((end, prompt))
            return _writer_json(end)

        def create_session(**kwargs):
            session = FolioMemorySession(**kwargs)
            sessions.append(session)
            return session

        def answer(**kwargs):
            current = float(kwargs["video_end"]) - 1e-4
            session = sessions[0]
            self.assertLessEqual(session.memory.last_time, current + 1e-6)
            self.assertEqual(tuple(kwargs.get("historical_frames", ())), ())
            self.assertNotIn("prediction_label", kwargs["prompt"])
            answers.append((round(current), session.memory.write_calls, kwargs["prompt"]))
            return _result("Yes" if "Answer Yes or No only." in kwargs["prompt"] else "A")

        with patch.object(self.runner, "_choose_source_path", return_value="source.mp4"), \
                patch.object(self.runner.base, "_opencv_frame_source", side_effect=frames) as source, \
                patch.object(self.runner, "FolioMemorySession", side_effect=create_session), \
                patch.object(self.runner, "query_recent_window", side_effect=answer):
            records = self._run_group(specs, SimpleNamespace(_folio_generate_memory=writer), config)

        source.assert_called_once_with("source.mp4", 1.0)
        self.assertEqual(len(sessions), 1)
        self.assertTrue(sessions[0]._closed)
        self.assertEqual([item[0] for item in answers], [3, 5, 7, 7, 11, 20])
        self.assertEqual([item[1] for item in answers], [1, 1, 2, 2, 3, 3])
        self.assertEqual([end for end, _prompt in writes], [3, 7, 11])
        # Session prefetch may decode one frame beyond the cutoff, but that
        # frame must never enter observations or any memory writer transaction.
        self.assertLessEqual(max(yielded), 12)
        self.assertEqual(sessions[0].memory.last_time, 11)
        self.assertEqual(sessions[0].memory.observed_frames, 12)
        self.assertEqual(sessions[0].memory.semantic_link_calls, 0)
        self.assertEqual(sessions[0].memory.cache_recovery_count, 0)
        self.assertEqual(sessions[0].memory.dialogue_history, [])
        for spec in specs:
            record = records[spec["annotation_key"]]
            if spec["query_index"] is not None:
                record = record["test_info"][0]
                self.assertEqual(record["response"], "Yes")
            metadata = record["folio"]
            self._assert_timings(metadata)
            if spec["task"] in REAL_TIME_TASKS:
                self.assertFalse(metadata["memory_used"])
                self.assertEqual(metadata["memory_route"], "recent_only")
                self.assertEqual(metadata["write_calls_delta"], 0)
            else:
                self.assertTrue(metadata["memory_used"])
                self.assertEqual(metadata["memory_route"], "text_memory")
                self.assertIn("<FOCUSED_VIDEO_MEMORY>", next(
                    prompt for stamp, _writes, prompt in answers if stamp == spec["query_time"]
                ))
        self.assertEqual(sum(
            (record["test_info"][0] if "test_info" in record else record)["folio"]["write_calls_delta"]
            for record in records.values()
        ), 3)

    def test_realtime_answer_failure_still_reports_route_and_cost(self):
        config = self.runner.configure_query_policy(FolioConfig(), "task_routed")
        session = _mock_session(config)
        with self.assertLogs(self.runner.LOGGER, level="ERROR"), \
                patch.object(self.runner, "query_recent_window", side_effect=RuntimeError("decode failed")):
            response, metadata, error = self._call(
                _query(), session, config, query_policy="task_routed"
            )
        self.assertIsNone(response)
        self.assertIn("decode failed", error)
        self.assertEqual(metadata["memory_route"], "recent_only")
        self.assertFalse(metadata["memory_used"])
        self.assertEqual(metadata["write_calls_delta"], 0)
        session.advance_to.assert_not_called()
        session.prepare_query.assert_not_called()
        self._assert_timings(metadata)

    def test_missing_clips_keep_policy_metadata_for_resume_validation(self):
        config = self.runner.configure_query_policy(FolioConfig(), "task_routed")
        specs = [_query("OCR"), _query("SSR", identifier="forward")]
        with patch.object(self.runner, "_choose_source_path", return_value=None), \
                patch.object(self.runner, "FolioMemorySession") as session:
            records = self._run_group(specs, object(), config)
        session.assert_not_called()
        for record in records.values():
            if "test_info" in record:
                record = record["test_info"][0]
            self.assertIsNone(record["response"])
            self.assertIn("Missing all chunked videos", record["error"])
            self.assertEqual(record["folio"]["folio_query_policy"], "task_routed")
            self.assertEqual(record["folio_query_policy"], "task_routed")


if __name__ == "__main__":
    unittest.main()
