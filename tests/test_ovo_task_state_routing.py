"""CPU dispatch and annotation-boundary contracts for OVO task state.

State algorithms have their own tests. These checks isolate model inference
and exercise the adapter's route selection, sharing, output, and resume rules.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from lib.folio_memory import FolioConfig
from ovo_constants import BACKWARD_TASKS, REAL_TIME_TASKS
from test_ovo_folio_task_routing import _annotation, _load_runner, _mock_session, _query, _result


class FakeTaskSession:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.last_time = -1.0
        self.advances = []
        self.questions = []
        self.count = 0
        self.events = []
        self.status = "complete"
        self.closed = False

    def advance_to(self, timestamp):
        if timestamp < self.last_time:
            raise ValueError("state cannot see future observations")
        self.advances.append(timestamp)
        self.last_time = timestamp
        self.count = int(timestamp // 4)

    def memory_text(self, question, max_bytes=8192):
        self.questions.append(question)
        return f"Observed door opening at or before {self.last_time}s."

    def usage(self):
        return {
            "write_calls": len(set(self.advances)),
            "write_seconds": len(set(self.advances)) * 0.1,
            "write_errors": 0,
            "semantic_link_calls": 0,
            "observed_frames": max(0, int(self.last_time)),
            "stream_errors": 0,
            "record_count": len(set(self.advances)),
            "status": self.status,
            "gaps": [],
        }

    def close(self):
        self.closed = True


class OvoTaskStateRoutingCheck(unittest.TestCase):
    def setUp(self):
        self.runner = _load_runner()
        self.config = self.runner.configure_query_policy(FolioConfig(), "task_state")

    def _call(self, spec, *, session=None, task_session=None):
        return self.runner.run_shared_query(
            qa=object(), session=session, task_session=task_session,
            source_path="source.mp4", spec=spec,
            chunk_duration=1.0, fps=1.0, recent_frames_only=2,
            folio_config=self.config, query_policy="task_state",
        )

    def _assert_metadata(self, metadata, route):
        self.assertEqual(metadata["folio_query_policy"], "task_state")
        self.assertEqual(metadata["task_state_protocol"], "ovo-task-state-v1")
        self.assertEqual(metadata["memory_route"], route)
        for key in (
            "memory_advance_seconds", "retrieval_seconds", "answer_seconds",
            "query_wall_seconds", "write_calls_delta", "write_seconds_delta",
        ):
            self.assertGreaterEqual(metadata[key], 0, key)

    def _run_group(self, specs):
        annotations = {}
        for spec in specs:
            key = spec["annotation_key"]
            if key not in annotations:
                annotations[key] = _annotation(spec)
            if spec["query_index"] is not None:
                while len(annotations[key]["test_info"]) <= spec["query_index"]:
                    annotations[key]["test_info"].append({"gt": 0})
                annotations[key]["test_info"][spec["query_index"]]["realtime"] = spec["query_time"]
        records = {key: self.runner._record_template(anno) for key, anno in annotations.items()}
        checkpoint = io.StringIO()
        done = set()
        self.runner.run_group(
            group_specs=specs, annotations_by_key=annotations,
            records_by_key=records, done_keys=done, checkpoint=checkpoint,
            qa=SimpleNamespace(_folio_generate_memory=Mock()),
            chunk_duration=1.0, fps=1.0, recent_frames_only=2,
            folio_config=self.config, query_policy="task_state",
        )
        self.assertEqual(done, set(records))
        self.assertEqual(len(checkpoint.getvalue().splitlines()), len(records))
        return records

    def test_task_state_config_and_cli_default(self):
        for name in ("semantic_link", "cache_replay", "interaction_focus", "structured_answer"):
            self.assertFalse(getattr(self.config, name), name)

        class ParsedEnough(Exception):
            pass

        defaults = {}

        def capture(parser, *_args, **_kwargs):
            for action in parser._actions:
                if action.dest in {"folio_query_policy", "rec_fps", "rec_window_seconds", "crr_window_seconds"}:
                    defaults[action.dest] = action.default
                    if action.dest == "folio_query_policy":
                        self.assertEqual(set(action.choices), {"all", "task_routed", "task_state"})
            raise ParsedEnough()

        with patch.object(self.runner.argparse.ArgumentParser, "parse_args", capture):
            with self.assertRaises(ParsedEnough):
                self.runner.main()
        self.assertEqual(defaults, {
            "folio_query_policy": "task_state", "rec_fps": 2.0,
            "rec_window_seconds": 4.0, "crr_window_seconds": 8.0,
        })

    def test_rt_and_ssr_use_only_recent_frames_even_with_unrelated_sessions(self):
        for task in [*REAL_TIME_TASKS, "SSR"]:
            with self.subTest(task=task):
                spec = _query(task)
                folio = _mock_session(self.config)
                unrelated = FakeTaskSession()
                with patch.object(self.runner, "query_recent_window", return_value=_result("Yes")) as answer:
                    response, metadata, error = self._call(spec, session=folio, task_session=unrelated)
                self.assertEqual(response, "Yes")
                self.assertIsNone(error)
                folio.advance_to.assert_not_called()
                folio.prepare_query.assert_not_called()
                self.assertEqual(unrelated.advances, [])
                self.assertEqual(unrelated.questions, [])
                self.assertEqual(answer.call_args.kwargs["prompt"], spec["original_prompt"])
                self.assertEqual(tuple(answer.call_args.kwargs.get("historical_frames", ())), ())
                self._assert_metadata(metadata, "recent_only")
                self.assertEqual(metadata["write_calls_delta"], 0)
                self.assertFalse(metadata["memory_used"])

    def test_bt_still_uses_shared_folio_text_without_task_state_inference(self):
        for task in BACKWARD_TASKS:
            with self.subTest(task=task):
                spec = _query(task)
                folio = _mock_session(self.config)
                folio.prepare_query.return_value = SimpleNamespace(
                    prompt="original FOLIO template", recent_only_prompt=spec["original_prompt"],
                    memory_text="bookshelf in study", evidence_frames=(),
                    selected_records=("bookshelf in study",), to_metadata=lambda: {},
                )
                unrelated = FakeTaskSession()
                with patch.object(self.runner, "query_recent_window", return_value=_result()) as answer:
                    response, metadata, error = self._call(spec, session=folio, task_session=unrelated)
                self.assertEqual(response, "A")
                self.assertIsNone(error)
                folio.advance_to.assert_called_once_with(7.0)
                folio.prepare_query.assert_called_once()
                self.assertEqual(unrelated.advances, [])
                self.assertIn("bookshelf in study", answer.call_args.kwargs["prompt"])
                self.assertTrue(answer.call_args.kwargs["prompt"].endswith(spec["original_prompt"]))
                self._assert_metadata(metadata, "text_memory")

    def test_rec_returns_accumulated_count_without_a_second_answer_model(self):
        spec = {**_query("REC", 12), "activity": "clap"}
        state = FakeTaskSession()
        folio = _mock_session(self.config)
        with patch.object(self.runner, "query_recent_window") as answer:
            response, metadata, error = self._call(spec, session=folio, task_session=state)
        self.assertEqual(response, "3")
        self.assertIsNone(error)
        answer.assert_not_called()
        folio.advance_to.assert_not_called()
        folio.prepare_query.assert_not_called()
        self.assertEqual(state.advances, [12])
        self._assert_metadata(metadata, "rec_count")
        self.assertEqual(metadata["write_calls_delta"], 1)

    def test_degraded_rec_does_not_publish_a_reliable_zero(self):
        state = FakeTaskSession()
        state.status = "degraded"
        with self.assertLogs(self.runner.LOGGER, level="ERROR"), \
                patch.object(self.runner, "query_recent_window") as answer:
            response, metadata, error = self._call(
                {**_query("REC", 0), "activity": "clap"}, task_session=state
            )
        self.assertIsNone(response)
        self.assertTrue(error)
        answer.assert_not_called()
        self._assert_metadata(metadata, "rec_count")

    def test_crr_rechecks_current_question_without_latching_a_previous_yes(self):
        state = FakeTaskSession()
        folio = _mock_session(self.config)
        specs = [
            _query("CRR", 7, "first-question"),
            {**_query("CRR", 11, "second-question"), "question": "Who closed the door?"},
        ]
        with patch.object(self.runner, "query_recent_window", side_effect=[_result("Yes"), _result("No")]) as answer:
            replies = [self._call(spec, session=folio, task_session=state) for spec in specs]
        self.assertEqual([reply[0] for reply in replies], ["Yes", "No"])
        self.assertEqual(state.advances, [7, 11])
        self.assertEqual(state.questions, [spec["question"] for spec in specs])
        folio.advance_to.assert_not_called()
        folio.prepare_query.assert_not_called()
        for spec, call, (_response, metadata, error) in zip(specs, answer.call_args_list, replies):
            self.assertIsNone(error)
            self.assertIn("Observed door opening", call.kwargs["prompt"])
            self.assertTrue(call.kwargs["prompt"].endswith(spec["original_prompt"]))
            self.assertEqual(tuple(call.kwargs.get("historical_frames", ())), ())
            self._assert_metadata(metadata, "evidence_memory")

    def test_specs_only_expose_activity_and_query_times_to_state_writers(self):
        rec = {
            "id": "rec", "video": "same", "task": "REC", "activity": "clap",
            "start_times": [1111], "end_times": [2222], "count": 3333,
            "test_info": [{"realtime": 7, "count": 4444, "gt": 5555}],
        }
        crr = {
            "id": "crr", "video": "same", "task": "CRR", "question": "Who opened the door?",
            "answer": "SECRET_LABEL", "clue_time": 6666, "ask_time": 4,
            "test_info": [{"realtime": 10, "type": 7777, "gt": 8888}],
        }
        rec_spec = self.runner._query_specs(rec, "/clips")[0]
        crr_spec = self.runner._query_specs(crr, "/clips")[0]
        self.assertEqual(rec_spec["activity"], "clap")
        self.assertEqual(crr_spec["memory_start_time"], 4)
        forbidden = {"start_times", "end_times", "count", "gt", "answer", "clue_time", "type"}
        for spec in (rec_spec, crr_spec):
            self.assertFalse(forbidden.intersection(spec))
            serialized = json.dumps(spec)
            for value in ("1111", "2222", "3333", "4444", "5555", "6666", "7777", "8888", "SECRET_LABEL"):
                self.assertNotIn(value, serialized)

    def test_rt_and_ssr_only_group_constructs_no_memory_session(self):
        specs = [_query("SSR", 5), _query("OCR", 500)]
        with patch.object(self.runner, "_choose_source_path", return_value="source.mp4"), \
                patch.object(self.runner, "FolioMemorySession") as folio, \
                patch.object(self.runner, "RecCountSession") as rec, \
                patch.object(self.runner, "CrrEvidenceSession") as crr, \
                patch.object(self.runner.base, "_opencv_frame_source") as frames, \
                patch.object(self.runner, "query_recent_window", return_value=_result()):
            records = self._run_group(specs)
        folio.assert_not_called()
        rec.assert_not_called()
        crr.assert_not_called()
        frames.assert_not_called()
        for record in records.values():
            query = record["test_info"][0] if "test_info" in record else record
            self._assert_metadata(query["folio"], "recent_only")

    def test_group_shares_rec_by_activity_and_crr_across_questions(self):
        specs = [
            {**_query("REC", 12, "clap-again"), "activity": "clap"},
            {**_query("CRR", 11, "crr-second"), "question": "Who closed the door?", "memory_start_time": 9},
            {**_query("REC", 10, "jump"), "activity": "jump"},
            {**_query("REC", 8, "clap"), "query_index": 1, "query_id": "REC:clap:1", "activity": "clap"},
            _query("SSR", 6),
            {**_query("CRR", 5, "crr-first"), "memory_start_time": 3},
            {**_query("REC", 4, "clap"), "activity": "clap"},
            _query("OCR", 20),
        ]
        rec_sessions, crr_sessions = [], []

        def create_rec(**kwargs):
            session = FakeTaskSession(**kwargs)
            rec_sessions.append(session)
            return session

        def create_crr(**kwargs):
            session = FakeTaskSession(**kwargs)
            crr_sessions.append(session)
            return session

        with patch.object(self.runner, "_choose_source_path", return_value="source.mp4"), \
                patch.object(self.runner, "FolioMemorySession") as folio, \
                patch.object(self.runner, "RecCountSession", side_effect=create_rec), \
                patch.object(self.runner, "CrrEvidenceSession", side_effect=create_crr), \
                patch.object(self.runner.base, "_opencv_frame_source", side_effect=lambda *_args: iter(())), \
                patch.object(self.runner, "query_recent_window", return_value=_result("Yes")) as answer:
            records = self._run_group(specs)
        folio.assert_not_called()
        self.assertEqual(len(rec_sessions), 2)
        self.assertEqual(len(crr_sessions), 1)
        by_activity = {session.kwargs["activity"]: session for session in rec_sessions}
        self.assertEqual(by_activity["clap"].advances, [4, 8, 12])
        self.assertEqual(by_activity["jump"].advances, [10])
        for session in rec_sessions:
            self.assertEqual(session.kwargs["fps"], 2.0)
            self.assertEqual(session.kwargs["window_seconds"], 4.0)
        self.assertEqual(crr_sessions[0].advances, [5, 11])
        self.assertEqual(crr_sessions[0].questions, ["Where is the bookshelf?", "Who closed the door?"])
        self.assertEqual(crr_sessions[0].kwargs["window_seconds"], 8.0)
        self.assertTrue(all(session.closed for session in [*rec_sessions, *crr_sessions]))
        self.assertEqual(answer.call_count, 4)  # Two CRR, one SSR, one RT.
        self.assertEqual([query["response"] for query in records["REC:clap"]["test_info"]], ["1", "2"])
        self.assertEqual(records["REC:clap-again"]["test_info"][0]["response"], "3")

    def test_group_advances_shared_folio_only_for_backward_queries(self):
        specs = [
            _query("EPM", 3, "first-bt"),
            {**_query("REC", 8), "activity": "clap"},
            _query("SSR", 9),
            _query("HLD", 11, "second-bt"),
            {**_query("CRR", 20), "memory_start_time": 18},
            _query("OCR", 100),
        ]
        folio = _mock_session(self.config)

        def retrieve(_query_id, _question, _options, **kwargs):
            return SimpleNamespace(
                prompt="folio text", recent_only_prompt=kwargs["original_prompt"],
                memory_text="bookshelf in study", evidence_frames=(),
                selected_records=("bookshelf in study",), to_metadata=lambda: {},
            )

        folio.prepare_query.side_effect = retrieve
        with patch.object(self.runner, "_choose_source_path", return_value="source.mp4"), \
                patch.object(self.runner, "FolioMemorySession", return_value=folio) as constructor, \
                patch.object(self.runner, "RecCountSession", side_effect=FakeTaskSession), \
                patch.object(self.runner, "CrrEvidenceSession", side_effect=FakeTaskSession), \
                patch.object(self.runner.base, "_opencv_frame_source", side_effect=lambda *_args: iter(())), \
                patch.object(self.runner, "query_recent_window", return_value=_result()) as answer:
            self._run_group(specs)
        constructor.assert_called_once()
        self.assertEqual([call.args[0] for call in folio.advance_to.call_args_list], [3, 11])
        self.assertEqual(folio.prepare_query.call_count, 2)
        folio.close.assert_called_once()
        self.assertEqual(answer.call_count, 5)

    def test_missing_video_results_retain_current_resume_protocol(self):
        specs = [_query("OCR"), {**_query("REC"), "activity": "clap"}]
        with patch.object(self.runner, "_choose_source_path", return_value=None):
            records = self._run_group(specs)
        self.runner.validate_checkpoint_policy(list(records.values()), "task_state")
        for record in records.values():
            query = record["test_info"][0] if "test_info" in record else record
            self.assertIsNone(query["response"])
            self.assertTrue(query["error"])
            self.assertEqual(query["folio"]["task_state_protocol"], "ovo-task-state-v1")

    def test_resume_rejects_prior_policies_and_unversioned_task_state(self):
        for record in (
            {"task": "OCR", "folio": {}},
            {"task": "OCR", "folio": {"folio_query_policy": "task_routed"}},
            {"task": "REC", "test_info": [{"folio": {"folio_query_policy": "task_state"}}]},
            {"task": "CRR", "test_info": [{"folio": {
                "folio_query_policy": "task_state", "task_state_protocol": "old-schema",
            }}]},
        ):
            with self.subTest(record=record), self.assertRaises(ValueError):
                self.runner.validate_checkpoint_policy([record], "task_state")
        valid = {"folio_query_policy": "task_state", "task_state_protocol": "ovo-task-state-v1"}
        self.runner.validate_checkpoint_policy([
            {"task": "OCR", "folio": valid},
            {"task": "REC", "test_info": [{"folio": valid}]},
        ], "task_state")
        self.runner.validate_checkpoint_policy([{"task": "OCR", "folio": {}}], "all")
        self.runner.validate_checkpoint_policy([
            {"task": "OCR", "folio": {"folio_query_policy": "task_routed"}},
        ], "task_routed")

    def test_run_manifest_allows_same_configuration_to_resume(self):
        configuration = {
            "task_state_protocol": "ovo-task-state-v1", "rec_fps": 2.0,
            "rec_window_seconds": 4.0, "crr_window_seconds": 8.0,
            "task_memory_tokens": 384, "anno_sha256": "original-annotations",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.runner.validate_run_manifest(root, configuration, has_records=False)
            path = root / "task_state_run_config.json"
            original = path.read_bytes()
            self.assertEqual(json.loads(original), configuration)
            # Dictionary insertion order does not change the experiment.
            resumed = dict(reversed(list(configuration.items())))
            self.runner.validate_run_manifest(root, resumed, has_records=True)
            self.assertEqual(path.read_bytes(), original)

    def test_run_manifest_rejects_changed_settings_without_overwriting_original(self):
        configuration = {
            "task_state_protocol": "ovo-task-state-v1", "model_path": "model-one",
            "rec_fps": 2.0, "rec_window_seconds": 4.0,
            "crr_window_seconds": 8.0, "task_memory_tokens": 384,
            "anno_sha256": "original-annotations",
        }
        changes = {
            "rec_fps": 4.0, "rec_window_seconds": 6.0,
            "crr_window_seconds": 12.0, "task_memory_tokens": 512,
            "model_path": "model-two", "anno_sha256": "changed-annotations",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.runner.validate_run_manifest(root, configuration, has_records=False)
            path = root / "task_state_run_config.json"
            original = path.read_bytes()
            for setting, value in changes.items():
                with self.subTest(setting=setting), self.assertRaises(ValueError):
                    self.runner.validate_run_manifest(
                        root, {**configuration, setting: value}, has_records=True
                    )
                self.assertEqual(path.read_bytes(), original)

    def test_existing_results_without_manifest_cannot_be_relabelled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "results_incremental.jsonl"
            original = '{"task":"REC","test_info":[{"response":"2"}]}\n'
            checkpoint.write_text(original, encoding="utf-8")
            with self.assertRaises(ValueError):
                self.runner.validate_run_manifest(
                    root, {"rec_fps": 2.0, "task_state_protocol": "ovo-task-state-v1"},
                    has_records=True,
                )
            self.assertFalse((root / "task_state_run_config.json").exists())
            self.assertEqual(checkpoint.read_text(encoding="utf-8"), original)


if __name__ == "__main__":
    unittest.main()
