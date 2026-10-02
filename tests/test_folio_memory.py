"""Behavioral contract for the FOLIO-inspired streaming memory.

These tests intentionally use only synthetic Pillow images and scripted model
responses.  They verify the online memory protocol without requiring a video
decoder, a GPU, or a real VLM.
"""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

import lib.folio_memory as folio_module
from lib.folio_memory import FolioConfig, FolioMemory, FolioMemorySession, FolioQueryPlan


def make_frame(value: int) -> Image.Image:
    """Make a frame whose first pixel identifies it in writer call traces."""
    return Image.new("RGB", (12, 8), (value, value, value))


def observe_segment(memory: FolioMemory, start: int, values: list[int]) -> None:
    for offset, value in enumerate(values):
        memory.observe(float(start + offset), make_frame(value))


def detailed_object(
    name: str,
    *,
    category: str,
    location: str,
    state: str,
    evidence_frames: list[int] | None = None,
    state_change: str = "newly visible",
) -> dict:
    return {
        "name": name,
        "category": category,
        "attributes": [],
        "location": location,
        "holder": "none",
        "state": state,
        "relations": [],
        "interactions": [],
        "state_change": state_change,
        "evidence_summary": f"{name} is visible at {location}",
        "evidence_frames": [0] if evidence_frames is None else evidence_frames,
        "confidence": 0.95,
    }


def event(
    event_type: str,
    summary: str,
    participants: list[str],
    changed_objects: list[str],
) -> dict:
    return {
        "event_type": event_type,
        "summary": summary,
        "participants": participants,
        "changed_objects": changed_objects,
        "confidence": 0.9,
    }


def writer_response(
    time_range: str,
    *,
    detailed: list[dict] | None = None,
    compact: list[dict] | None = None,
    events: list[dict] | None = None,
) -> str:
    return json.dumps(
        {
            "time": time_range,
            "detailed_objects": detailed or [],
            "compact_objects": compact or [],
            "events": events or [],
        }
    )


class ScriptedGenerate:
    """Serve writer responses and, when needed, one semantic-link response."""

    def __init__(self, writer_outputs: list[str], semlink_output=None):
        self.writer_outputs = iter(writer_outputs)
        self.semlink_output = semlink_output
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs):
        if args and isinstance(args[0], (list, tuple)):
            frames = list(args[0])
            prompt = str(args[1])
        else:
            frames = []
            prompt = str(args[0] if args else kwargs.get("prompt", ""))

        upper = prompt.upper()
        is_semlink = (
            "SEMANTIC REASONING" in upper
            or "RELEVANT_OBJECT_IDS" in upper
            or "SEMANTIC QUERY EXPANSION" in upper
        )
        kind = "semlink" if is_semlink else "writer"
        self.calls.append({"kind": kind, "frames": frames, "prompt": prompt})
        if is_semlink:
            output = self.semlink_output
            if callable(output):
                output = output(prompt)
            if isinstance(output, BaseException):
                raise output
            if output is None:
                raise AssertionError("Unexpected SemLink call")
            return output
        return next(self.writer_outputs)

    @property
    def writer_calls(self) -> list[dict]:
        return [call for call in self.calls if call["kind"] == "writer"]

    @property
    def semlink_calls(self) -> list[dict]:
        return [call for call in self.calls if call["kind"] == "semlink"]


class FolioMemoryContract(unittest.TestCase):
    def test_adaptive_keyframes_use_boundaries_middle_and_largest_change(self):
        high_change = ScriptedGenerate([writer_response("0.0-7.0s")])
        memory = FolioMemory(
            high_change,
            segment_frames=8,
            change_threshold=0.01,
        )
        # Candidate indices are 0 (start), 2 (largest adjacent change),
        # 3 (lower middle), and 7 (end). They must be de-duplicated and ordered.
        observe_segment(memory, 0, [0, 0, 255, 254, 100, 100, 100, 101])
        self.assertEqual(len(high_change.writer_calls), 1)
        selected = [image.getpixel((0, 0))[0] for image in high_change.writer_calls[0]["frames"]]
        self.assertEqual(selected, [0, 255, 254, 101])

        low_change = ScriptedGenerate([writer_response("0.0-7.0s")])
        memory = FolioMemory(
            low_change,
            segment_frames=8,
            change_threshold=1.0,
        )
        observe_segment(memory, 0, list(range(8)))
        selected = [image.getpixel((0, 0))[0] for image in low_change.writer_calls[0]["frames"]]
        self.assertEqual(selected, [0, 7])

    def test_same_entity_builds_one_time_chain_and_direct_hit_skips_semlink(self):
        generate = ScriptedGenerate(
            [
                writer_response(
                    "0.0-7.0s",
                    detailed=[
                        detailed_object(
                            "red mug",
                            category="container",
                            location="counter",
                            state="empty",
                        )
                    ],
                ),
                writer_response(
                    "8.0-15.0s",
                    detailed=[
                        detailed_object(
                            "red mug",
                            category="container",
                            location="sink",
                            state="being washed",
                            state_change="moved from the counter to the sink",
                        )
                    ],
                    events=[
                        event(
                            "move",
                            "The red mug is moved from the counter to the sink.",
                            ["red mug"],
                            ["red mug"],
                        )
                    ],
                ),
            ]
        )
        memory = FolioMemory(generate, segment_frames=8, change_threshold=1.0)
        observe_segment(memory, 0, list(range(8)))
        observe_segment(memory, 8, list(range(8, 16)))

        plan = memory.prepare_query(
            "q-chain",
            "Where was the red mug before it reached the sink?",
            ["On the counter", "In a drawer", "Outside", "Unknown"],
            query_time=15.0,
            recent_start=12.0,
        )

        self.assertIsInstance(plan, FolioQueryPlan)
        self.assertEqual(plan.retrieval_mode, "direct")
        self.assertEqual(plan.semlink_status, "not_needed")
        self.assertEqual(len(plan.selected_entity_ids), 1)
        self.assertGreaterEqual(len(plan.selected_event_ids), 1)
        self.assertIn("counter", plan.memory_text.lower())
        self.assertIn("sink", plan.memory_text.lower())
        self.assertIn("empty", plan.memory_text.lower())
        self.assertIn("being washed", plan.memory_text.lower())
        self.assertIn("0.0", plan.memory_text)
        self.assertIn("8.0", plan.memory_text)
        self.assertIn("Where was the red mug", plan.prompt)
        self.assertEqual(len(generate.writer_calls), 2)
        self.assertEqual(generate.semlink_calls, [])

    def test_committed_question_changes_only_future_focus_and_is_idempotent(self):
        question = "Where is the red mug now?"
        options = ["Counter", "Sink", "Shelf", "Unknown"]

        def run(commit_count: int) -> list[dict]:
            generate = ScriptedGenerate(
                [
                    writer_response(
                        "0.0-7.0s",
                        detailed=[
                            detailed_object(
                                "red mug",
                                category="container",
                                location="counter",
                                state="empty",
                            )
                        ],
                    ),
                    writer_response(
                        "8.0-15.0s",
                        detailed=[
                            detailed_object(
                                "red mug",
                                category="container",
                                location="counter",
                                state="empty",
                                state_change="stable",
                            )
                        ],
                    ),
                ]
            )
            memory = FolioMemory(generate, segment_frames=8, change_threshold=1.0)
            observe_segment(memory, 0, list(range(8)))
            plan = memory.prepare_query(
                "q-focus",
                question,
                options,
                query_time=7.0,
                recent_start=4.0,
            )
            for _ in range(commit_count):
                memory.commit_interaction("q-focus", plan, question, options)
            observe_segment(memory, 8, list(range(8, 16)))
            return generate.writer_calls

        no_commit = run(0)
        one_commit = run(1)
        duplicate_commit = run(2)

        # The segment that existed before the question is written identically.
        self.assertEqual(no_commit[0]["prompt"], one_commit[0]["prompt"])
        # A committed turn changes the next segment's writing-level budget.
        self.assertNotEqual(no_commit[1]["prompt"], one_commit[1]["prompt"])
        # Replaying the same query id is a no-op.
        self.assertEqual(one_commit[1]["prompt"], duplicate_commit[1]["prompt"])
        # Future writing receives resolved focus, not the raw question text.
        self.assertNotIn(question, one_commit[1]["prompt"])

    def test_semlink_filters_unknown_ids_and_recovers_linked_evidence(self):
        generate = ScriptedGenerate(
            [
                writer_response(
                    "0.0-7.0s",
                    detailed=[
                        detailed_object(
                            "bookshelf",
                            category="furniture",
                            location="study wall",
                            state="filled with books",
                            evidence_frames=[0],
                        )
                    ],
                )
            ]
        )
        memory = FolioMemory(
            generate,
            segment_frames=8,
            change_threshold=1.0,
            max_cache_frames_per_query=2,
        )
        observe_segment(memory, 0, list(range(8)))

        direct = memory.prepare_query(
            "q-id",
            "Where is the bookshelf?",
            ["Study wall", "Kitchen", "Outside", "Unknown"],
            query_time=7.0,
            recent_start=4.0,
        )
        self.assertEqual(len(direct.selected_entity_ids), 1)
        valid_id = direct.selected_entity_ids[0]
        generate.semlink_output = lambda _prompt: json.dumps(
            {
                "relevant_object_ids": [valid_id, "obj_does_not_exist", valid_id],
                "reasoning": "Books and a bookshelf support the reading concept.",
                "suggested_option": "A",
            }
        )

        plan = memory.prepare_query(
            "q-concept",
            "Which description best fits this person?",
            ["Avid reader", "Musician", "Athlete", "Unable to answer"],
            query_time=10.0,
            recent_start=8.0,
        )

        self.assertEqual(plan.retrieval_mode, "semlink")
        self.assertEqual(plan.semlink_status, "ok")
        self.assertEqual(list(plan.selected_entity_ids), [valid_id])
        self.assertEqual(len(generate.semlink_calls), 1)
        self.assertGreaterEqual(len(plan.evidence_frames), 1)
        self.assertLessEqual(len(plan.evidence_frames), 2)
        self.assertEqual(len(plan.evidence_frames), len(plan.evidence_timestamps))
        self.assertTrue(all(isinstance(frame, Image.Image) for frame in plan.evidence_frames))
        self.assertTrue(all(timestamp <= 10.0 for timestamp in plan.evidence_timestamps))
        self.assertTrue(all(timestamp < 8.0 for timestamp in plan.evidence_timestamps))
        self.assertIn("bookshelf", plan.memory_text.lower())
        self.assertNotIn("obj_does_not_exist", plan.memory_text)
        self.assertIn("The first 1 image(s)", plan.prompt)
        self.assertIn("The first 0 image(s)", plan.recent_only_prompt)

    def test_invalid_writer_transaction_keeps_last_complete_memory(self):
        invalid_transaction = json.dumps(
            {
                "time": "8.0-15.0s",
                "detailed_objects": [
                    detailed_object(
                        "red mug",
                        category="container",
                        location="sink",
                        state="being washed",
                    ),
                    {"category": "tool", "location": "sink"},
                ],
                "compact_objects": [],
                "events": [],
            }
        )
        generate = ScriptedGenerate(
            [
                writer_response(
                    "0.0-7.0s",
                    detailed=[
                        detailed_object(
                            "red mug",
                            category="container",
                            location="counter",
                            state="empty",
                        )
                    ],
                ),
                invalid_transaction,
            ]
        )
        memory = FolioMemory(generate, segment_frames=8, change_threshold=1.0)
        observe_segment(memory, 0, list(range(8)))
        observe_segment(memory, 8, list(range(8, 16)))

        plan = memory.prepare_query(
            "q-after-error",
            "Where is the red mug?",
            ["Counter", "Sink", "Shelf", "Unknown"],
            query_time=15.0,
            recent_start=12.0,
        )
        self.assertEqual(plan.retrieval_mode, "direct")
        self.assertIn("counter", plan.memory_text.lower())
        self.assertNotIn("being washed", plan.memory_text.lower())
        self.assertNotIn("location=sink", plan.memory_text.lower())
        self.assertEqual(len(plan.selected_entity_ids), 1)

    def test_semlink_failure_falls_back_without_blocking_the_answer_plan(self):
        generate = ScriptedGenerate(
            [
                writer_response(
                    "0.0-7.0s",
                    detailed=[
                        detailed_object(
                            "bookshelf",
                            category="furniture",
                            location="study wall",
                            state="filled with books",
                        )
                    ],
                )
            ],
            semlink_output=RuntimeError("injected semantic-link failure"),
        )
        memory = FolioMemory(generate, segment_frames=8, change_threshold=1.0)
        observe_segment(memory, 0, list(range(8)))

        plan = memory.prepare_query(
            "q-semlink-error",
            "Which personality description fits this person?",
            ["Curious", "Reserved", "Outgoing", "Unable to answer"],
            query_time=10.0,
            recent_start=8.0,
        )

        self.assertEqual(len(generate.semlink_calls), 1)
        self.assertNotEqual(plan.semlink_status, "ok")
        self.assertNotEqual(plan.retrieval_mode, "semlink")
        self.assertEqual(list(plan.selected_entity_ids), [])
        self.assertEqual(list(plan.selected_event_ids), [])
        self.assertEqual(list(plan.evidence_frames), [])
        self.assertIn("Which personality description", plan.prompt)

    def test_snapshot_is_machine_restorable_and_human_inspectable(self):
        generate = ScriptedGenerate(
            [
                writer_response(
                    "0.0-7.0s",
                    detailed=[
                        detailed_object(
                            "red mug",
                            category="container",
                            location="counter",
                            state="empty",
                            evidence_frames=[0],
                        )
                    ],
                )
            ]
        )
        memory = FolioMemory(generate, segment_frames=8, change_threshold=1.0)
        observe_segment(memory, 0, list(range(8)))
        question = "Where is the red mug?"
        options = ["Counter", "Sink", "Shelf", "Unknown"]
        plan = memory.prepare_query(
            "q-snapshot",
            question,
            options,
            query_time=7.0,
            recent_start=4.0,
        )
        memory.commit_interaction("q-snapshot", plan, question, options)

        metadata = plan.to_metadata()
        json.dumps(metadata)
        for key in (
            "selected_entity_ids",
            "selected_event_ids",
            "retrieval_mode",
            "semlink_status",
            "recovered_evidence_timestamps",
        ):
            self.assertIn(key, metadata)

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "folio"
            self.assertTrue(memory.save_snapshot(target))

            for relative in (
                "MEMORY.md",
                "state.json",
                "focus.json",
                "evidence/index.json",
                "usage.json",
            ):
                self.assertTrue((target / relative).is_file(), relative)

            self.assertIn("red mug", (target / "MEMORY.md").read_text().lower())
            self.assertIn("counter", (target / "MEMORY.md").read_text().lower())
            json.loads((target / "state.json").read_text())
            json.loads((target / "focus.json").read_text())
            json.loads((target / "evidence/index.json").read_text())
            json.loads((target / "usage.json").read_text())

            entity_notes = list((target / "entities").glob("*.md"))
            self.assertGreaterEqual(len(entity_notes), 1)
            self.assertIn("red mug", entity_notes[0].read_text().lower())

            image_files = [
                *target.glob("evidence/**/*.jpg"),
                *target.glob("evidence/**/*.jpeg"),
                *target.glob("evidence/**/*.png"),
            ]
            self.assertGreaterEqual(len(image_files), 1)
            with Image.open(image_files[0]) as saved:
                saved.verify()

            restored = FolioMemory.load_snapshot(
                target, ScriptedGenerate([], semlink_output=None)
            )
            self.assertEqual(restored.segment_index, memory.segment_index)
            self.assertEqual(set(restored.entities), set(memory.entities))
            self.assertEqual(
                restored.entities[next(iter(restored.entities))].canonical_name,
                "red mug",
            )
            self.assertIn("q-snapshot", restored.committed_query_ids)

    def test_snapshot_restores_an_incomplete_segment(self):
        initial_generate = ScriptedGenerate([writer_response("0.0-3.0s")])
        memory = FolioMemory(initial_generate, segment_frames=4, change_threshold=1.0)
        observe_segment(memory, 0, [0, 1, 2, 3])
        memory.observe(4.0, make_frame(4))
        memory.observe(5.0, make_frame(5))

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "folio"
            self.assertTrue(memory.save_snapshot(target))
            resumed_generate = ScriptedGenerate([writer_response("4.0-7.0s")])
            restored = FolioMemory.load_snapshot(target, resumed_generate)
            restored.observe(6.0, make_frame(6))
            restored.observe(7.0, make_frame(7))

        self.assertEqual(restored.segment_index, 2)
        self.assertEqual(restored.usage()["pending_frames"], 0)
        selected = [
            image.getpixel((0, 0))[0]
            for image in resumed_generate.writer_calls[0]["frames"]
        ]
        self.assertEqual(selected, [4, 7])

    def test_entity_merge_uses_identity_features_without_category_veto(self):
        generate = ScriptedGenerate(
            [
                writer_response(
                    "0.0-1.0s",
                    detailed=[
                        detailed_object(
                            "red cup", category="container", location="counter", state="empty"
                        )
                    ],
                ),
                writer_response(
                    "2.0-3.0s",
                    detailed=[
                        detailed_object(
                            "red cup", category="cup", location="counter", state="full"
                        )
                    ],
                ),
                writer_response(
                    "4.0-5.0s",
                    detailed=[
                        detailed_object(
                            "blue cup", category="cup", location="table", state="empty"
                        )
                    ],
                ),
            ]
        )
        memory = FolioMemory(generate, segment_frames=2, change_threshold=1.0)
        observe_segment(memory, 0, [0, 1])
        observe_segment(memory, 2, [2, 3])
        observe_segment(memory, 4, [4, 5])

        self.assertEqual(len(memory.entities), 2)
        red = next(item for item in memory.entities.values() if item.canonical_name == "red cup")
        blue = next(item for item in memory.entities.values() if item.canonical_name == "blue cup")
        self.assertEqual(len(red.observations), 2)
        self.assertNotEqual(red.id, blue.id)

    def test_query_type_and_word_boundaries_do_not_suppress_semlink(self):
        generate = ScriptedGenerate(
            [
                writer_response(
                    "0.0-1.0s",
                    detailed=[
                        detailed_object(
                            "man", category="person", location="room", state="standing"
                        )
                    ],
                )
            ],
            semlink_output=json.dumps(
                {
                    "relevant_object_ids": [],
                    "reasoning": "No matching woman is stored.",
                    "suggested_option": "",
                }
            ),
        )
        memory = FolioMemory(generate, segment_frames=2, change_threshold=1.0)
        observe_segment(memory, 0, [0, 1])

        location = memory.prepare_query(
            "q-location-type",
            "Where is the man now?",
            ["Room", "Outside", "Unable to answer", "Kitchen"],
            query_time=1.0,
            recent_start=0.5,
        )
        self.assertEqual(location.query_type, "current-location")

        missing = memory.prepare_query(
            "q-boundary",
            "What is the woman doing?",
            ["Standing", "Sitting", "Walking", "Unable to answer"],
            query_time=1.0,
            recent_start=0.5,
        )
        self.assertEqual(missing.selected_entity_ids, ())
        self.assertEqual(len(generate.semlink_calls), 1)

    def test_event_match_still_invokes_entity_semlink_on_direct_entity_miss(self):
        generate = ScriptedGenerate(
            [
                writer_response(
                    "0.0-1.0s",
                    events=[event("juggling", "A juggling action occurs.", [], [])],
                )
            ],
            semlink_output=json.dumps(
                {
                    "relevant_object_ids": [],
                    "reasoning": "No object entry is available.",
                    "suggested_option": "",
                }
            ),
        )
        memory = FolioMemory(generate, segment_frames=2, change_threshold=1.0)
        observe_segment(memory, 0, [0, 1])
        plan = memory.prepare_query(
            "q-event-only",
            "What happened during the juggling?",
            ["Juggling", "Reading", "Cooking", "Unknown"],
            query_time=1.0,
            recent_start=0.5,
        )

        self.assertEqual(len(generate.semlink_calls), 1)
        self.assertIn("event-0001", plan.selected_event_ids)

    def test_cache_recovery_uses_record_immediately_before_anchor(self):
        outputs = []
        for index, location in enumerate(("counter", "table", "drawer")):
            outputs.append(
                writer_response(
                    f"{2 * index}.0-{2 * index + 1}.0s",
                    detailed=[
                        detailed_object(
                            "red mug",
                            category="container",
                            location=location,
                            state="empty",
                            evidence_frames=[0],
                            state_change="moved" if index else "newly visible",
                        )
                    ],
                )
            )
        memory = FolioMemory(
            ScriptedGenerate(outputs),
            config=FolioConfig(
                segment_frames=2,
                change_threshold=1.0,
                max_cache_frames_per_query=1,
                sufficient_relevance=100.0,
            ),
        )
        observe_segment(memory, 0, [0, 1])
        observe_segment(memory, 2, [2, 3])
        observe_segment(memory, 4, [4, 5])

        plan = memory.prepare_query(
            "q-before-anchor",
            "Where was the red mug just before it reached the drawer?",
            ["Counter", "Table", "Drawer", "Unknown"],
            query_time=6.0,
            recent_start=6.0,
        )
        self.assertEqual(plan.evidence_timestamps, (2.0,))

    def test_cache_recovery_falls_back_to_image_when_slot_text_is_missing(self):
        sign = {
            "name": "street sign",
            "category": "sign",
            "attributes": [],
            "location": "roadside",
            "holder": "none",
            "state": "",
            "relations": [],
            "interactions": [],
            "state_change": "newly visible",
            "evidence_summary": "",
            "evidence_frames": [0],
            "confidence": 0.95,
        }
        memory = FolioMemory(
            ScriptedGenerate(
                [writer_response("0.0-1.0s", detailed=[sign])]
            ),
            segment_frames=2,
            change_threshold=1.0,
            max_cache_frames_per_query=1,
        )
        observe_segment(memory, 0, [0, 1])

        plan = memory.prepare_query(
            "q-sign-text",
            "What text is written on the street sign?",
            ["STOP", "YIELD", "SCHOOL", "Unknown"],
            query_time=10.0,
            recent_start=8.0,
        )

        self.assertEqual(plan.query_type, "attribute")
        self.assertEqual(plan.selected_records, ())
        self.assertGreater(len(plan.candidate_records), 0)
        self.assertEqual(plan.best_grounding_score, 0.0)
        self.assertEqual(plan.evidence_timestamps, (0.0,))

    def test_explicit_hld_hint_is_preserved_independently_of_question_wording(self):
        memory = FolioMemory(
            ScriptedGenerate(
                [
                    writer_response(
                        "0.0-1.0s",
                        detailed=[
                            detailed_object(
                                "red mug",
                                category="container",
                                location="counter",
                                state="empty",
                            )
                        ],
                    )
                ]
            ),
            segment_frames=2,
            change_threshold=1.0,
        )
        observe_segment(memory, 0, [0, 1])

        plan = memory.prepare_query(
            "q-hld-hint",
            "Which item matches the scene?",
            ["Red mug", "Blue bowl", "Green plate", "None"],
            query_time=1.0,
            recent_start=0.5,
            query_type_hint="HLD",
        )

        self.assertEqual(plan.query_type, "hallucination-detection")

    def test_latest_action_is_not_dropped_from_a_long_event_chain(self):
        outputs = []
        for index in range(10):
            outputs.append(
                writer_response(
                    f"{2 * index}.0-{2 * index + 1}.0s",
                    detailed=[
                        detailed_object(
                            "red mug",
                            category="container",
                            location="counter",
                            state=f"step {index + 1}",
                            state_change="touched",
                        )
                    ],
                    events=[
                        event(
                            "touch",
                            f"The person touched the red mug in step {index + 1}.",
                            ["red mug"],
                            ["red mug"],
                        )
                    ],
                )
            )
        memory = FolioMemory(
            ScriptedGenerate(outputs),
            segment_frames=2,
            change_threshold=1.0,
        )
        for start in range(0, 20, 2):
            observe_segment(memory, start, [start, start + 1])

        plan = memory.prepare_query(
            "q-last-action",
            "What happened last to the red mug?",
            ["An early action", "A middle action", "The final action", "Unknown"],
            query_time=19.0,
            recent_start=18.0,
        )

        self.assertIn("event-0010", plan.selected_event_ids)
        self.assertIn("step 10", plan.memory_text.lower())

    def test_direct_retrieval_uses_common_synonyms_and_option_identities(self):
        generate = ScriptedGenerate(
            [
                writer_response(
                    "0.0-1.0s",
                    detailed=[
                        detailed_object(
                            "couch",
                            category="furniture",
                            location="living room",
                            state="empty",
                        ),
                        detailed_object(
                            "glass bottle",
                            category="container",
                            location="table",
                            state="closed",
                        ),
                    ],
                )
            ]
        )
        memory = FolioMemory(generate, segment_frames=2, change_threshold=1.0)
        observe_segment(memory, 0, [0, 1])

        synonym_plan = memory.prepare_query(
            "q-sofa",
            "Where is the sofa?",
            ["Living room", "Kitchen", "Garden", "Unknown"],
            query_time=1.0,
            recent_start=0.5,
        )
        option_plan = memory.prepare_query(
            "q-option-identity",
            "Which object was visible?",
            ["Glass bottle", "Spoon", "Plate", "Unable to answer"],
            query_time=1.0,
            recent_start=0.5,
        )

        self.assertEqual(synonym_plan.retrieval_mode, "direct")
        self.assertIn("couch", synonym_plan.memory_text.lower())
        self.assertEqual(option_plan.retrieval_mode, "direct")
        self.assertIn("glass bottle", option_plan.memory_text.lower())
        self.assertEqual(generate.semlink_calls, [])

    def test_pending_query_term_survives_long_delay_and_becomes_alias(self):
        outputs = [writer_response(f"{2 * index}.0-{2 * index + 1}.0s") for index in range(33)]
        outputs.append(
            writer_response(
                "66.0-67.0s",
                detailed=[
                    detailed_object(
                        "car",
                        category="vehicle",
                        location="road",
                        state="parked",
                    )
                ],
            )
        )
        memory = FolioMemory(
            ScriptedGenerate(outputs), segment_frames=2, change_threshold=1.0
        )
        plan = memory.prepare_query(
            "q-delayed-focus",
            "Where is the automobile?",
            ["Garage", "Road", "Driveway", "Unknown"],
            query_time=0.0,
            recent_start=0.0,
        )
        memory.commit_interaction(
            plan.query_id,
            plan,
            "Where is the automobile?",
            ["Garage", "Road", "Driveway", "Unknown"],
            predicted_label="D",
            predicted_text="Unknown",
        )
        for start in range(0, 68, 2):
            observe_segment(memory, start, [start, start + 1])

        entity = next(item for item in memory.entities.values() if item.canonical_name == "car")
        self.assertIn("automobile", entity.aliases)
        self.assertNotIn("automobile", memory.pending_focus_terms)

    def test_semlink_receives_complete_catalog_in_one_call(self):
        outputs = []
        for index in range(14):
            item = detailed_object(
                f"relic {index}",
                category="artifact",
                location=f"shelf {index}",
                state="stored",
            )
            item["attributes"] = [f"descriptor-{index}-" + ("x" * 700)]
            outputs.append(
                writer_response(
                    f"{2 * index}.0-{2 * index + 1}.0s", detailed=[item]
                )
            )

        def link_oldest(prompt):
            self.assertIn("entity-0001", prompt)
            self.assertIn("entity-0014", prompt)
            self.assertGreater(len(prompt.encode("utf-8")), 8192)
            return json.dumps(
                {
                    "relevant_object_ids": ["entity-0001"],
                    "reasoning": "The oldest catalog entry is relevant.",
                    "suggested_option": "A",
                }
            )

        generate = ScriptedGenerate(outputs, semlink_output=link_oldest)
        memory = FolioMemory(generate, segment_frames=2, change_threshold=1.0)
        for start in range(0, 28, 2):
            observe_segment(memory, start, [start, start + 1])

        plan = memory.prepare_query(
            "q-full-catalog",
            "Which profession best fits this scene?",
            ["Teacher", "Chef", "Pilot", "Unknown"],
            query_time=27.0,
            recent_start=26.0,
        )

        self.assertEqual(plan.retrieval_mode, "semlink")
        self.assertEqual(plan.selected_entity_ids, ("entity-0001",))
        self.assertEqual(len(generate.semlink_calls), 1)

    def test_sufficient_semlink_record_does_not_force_cache_replay(self):
        generate = ScriptedGenerate(
            [
                writer_response(
                    "0.0-1.0s",
                    detailed=[
                        detailed_object(
                            "bookshelf",
                            category="furniture",
                            location="study",
                            state="filled with books",
                            evidence_frames=[0],
                        )
                    ],
                )
            ]
        )
        memory = FolioMemory(
            generate,
            config=FolioConfig(
                segment_frames=2,
                change_threshold=1.0,
                sufficient_relevance=6.0,
            ),
        )
        observe_segment(memory, 0, [0, 1])
        generate.semlink_output = json.dumps(
            {
                "relevant_object_ids": ["entity-0001"],
                "reasoning": "A bookshelf supports the reading concept.",
                "suggested_option": "A",
            }
        )

        plan = memory.prepare_query(
            "q-sufficient-semlink",
            "Which description best fits this person?",
            ["Avid reader", "Musician", "Athlete", "Unable to answer"],
            query_time=10.0,
            recent_start=8.0,
        )

        self.assertEqual(plan.retrieval_mode, "semlink")
        self.assertEqual(plan.best_grounding_score, 6.0)
        self.assertEqual(plan.evidence_frames, ())

    def test_hidden_high_grounding_record_still_triggers_cache_recovery(self):
        outputs = []
        for index in range(7):
            sign = {
                "name": "street sign",
                "category": "sign",
                "attributes": [],
                "location": "roadside",
                "holder": "none",
                "state": "" if index == 0 else f"weathered {index}",
                "relations": [],
                "interactions": [],
                "state_change": f"view {index}",
                "visible_text": "STOP" if index == 0 else "",
                "evidence_summary": "",
                "evidence_frames": [0],
                "confidence": 0.95,
            }
            outputs.append(
                writer_response(
                    f"{2 * index}.0-{2 * index + 1}.0s", detailed=[sign]
                )
            )
        memory = FolioMemory(
            ScriptedGenerate(outputs),
            config=FolioConfig(
                segment_frames=2,
                change_threshold=1.0,
                max_cache_frames_per_query=1,
            ),
        )
        for start in range(0, 14, 2):
            observe_segment(memory, start, [start, start + 1])

        plan = memory.prepare_query(
            "q-old-sign-text",
            "What text was written on the street sign?",
            ["STOP", "YIELD", "SCHOOL", "Unknown"],
            query_time=14.0,
            recent_start=14.0,
        )

        self.assertEqual(plan.best_grounding_score, 6.0)
        self.assertEqual(plan.evidence_timestamps, (0.0,))

    def test_dialogue_history_uses_only_committed_model_predictions_and_restores(self):
        generate = ScriptedGenerate(
            [
                writer_response(
                    "0.0-1.0s",
                    detailed=[
                        detailed_object(
                            "red mug", category="container", location="counter", state="empty"
                        )
                    ],
                )
            ]
        )
        memory = FolioMemory(generate, segment_frames=2, change_threshold=1.0)
        observe_segment(memory, 0, [0, 1])
        first = memory.prepare_query(
            "q-dialogue-1",
            "Where is the red mug?",
            ["Counter", "Sink", "Shelf", "Unknown"],
            query_time=1.0,
            recent_start=0.5,
        )
        self.assertIn("(no previous turns)", first.prompt)
        memory.commit_interaction(
            first.query_id,
            first,
            "Where is the red mug?",
            ["Counter", "Sink", "Shelf", "Unknown"],
            predicted_label="A",
            predicted_text="Counter",
        )
        second = memory.prepare_query(
            "q-dialogue-2",
            "Where is it now?",
            ["Counter", "Sink", "Shelf", "Unknown"],
            query_time=1.0,
            recent_start=0.5,
        )
        self.assertEqual(second.dialogue_turns_used, 1)
        self.assertIn("Where is the red mug?", second.prompt)
        self.assertIn("model prediction: A Counter", second.prompt)
        self.assertEqual(second.selected_entity_ids, first.selected_entity_ids)

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "folio"
            self.assertTrue(memory.save_snapshot(target))
            restored = FolioMemory.load_snapshot(target, ScriptedGenerate([]))
        self.assertEqual(restored.dialogue_history, memory.dialogue_history)

    def test_followup_that_uses_previous_event_as_temporal_anchor(self):
        outputs = []
        actions = (
            ("pick", "The person picked up the red mug."),
            ("pour", "The person poured water from the red mug."),
            ("wash", "The person washed the red mug."),
        )
        for index, (event_type, summary) in enumerate(actions):
            outputs.append(
                writer_response(
                    f"{2 * index}.0-{2 * index + 1}.0s",
                    detailed=[
                        detailed_object(
                            "red mug",
                            category="container",
                            location="sink" if index == 2 else "counter",
                            state=event_type,
                            evidence_frames=[0],
                            state_change=event_type,
                        )
                    ],
                    events=[event(event_type, summary, ["red mug"], ["red mug"])],
                )
            )
        memory = FolioMemory(
            ScriptedGenerate(outputs),
            config=FolioConfig(
                segment_frames=2,
                change_threshold=1.0,
                max_cache_frames_per_query=1,
                sufficient_relevance=100.0,
            ),
        )
        observe_segment(memory, 0, [0, 1])
        observe_segment(memory, 2, [2, 3])
        observe_segment(memory, 4, [4, 5])

        first = memory.prepare_query(
            "q-event-anchor",
            "What did the person pour from the red mug?",
            ["Water", "Juice", "Nothing", "Unknown"],
            query_time=5.0,
            recent_start=5.0,
        )
        self.assertEqual(first.matched_event_ids, ("event-0002",))
        memory.commit_interaction(
            first.query_id,
            first,
            "What did the person pour from the red mug?",
            ["Water", "Juice", "Nothing", "Unknown"],
            predicted_label="A",
            predicted_text="Water",
        )

        followup = memory.prepare_query(
            "q-event-after",
            "What happened after that?",
            ["The mug was washed", "The mug was picked up", "Nothing", "Unknown"],
            query_time=6.0,
            recent_start=6.0,
        )
        self.assertIn("event-0003", followup.matched_event_ids)
        self.assertIn("washed", followup.memory_text.lower())
        self.assertEqual(followup.evidence_timestamps, (4.0,))

    def test_failed_snapshot_commit_preserves_previous_machine_state(self):
        generate = ScriptedGenerate(
            [
                writer_response(
                    "0.0-1.0s",
                    detailed=[
                        detailed_object(
                            "red mug", category="container", location="counter", state="empty"
                        )
                    ],
                )
            ]
        )
        memory = FolioMemory(generate, segment_frames=2, change_threshold=1.0)
        observe_segment(memory, 0, [0, 1])
        memory.observe(2.0, make_frame(2))

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "folio"
            self.assertTrue(memory.save_snapshot(target))
            original_state = (target / "state.json").read_bytes()
            real_write = folio_module._atomic_write_text

            def fail_state(path, text):
                if Path(path).name == "state.json":
                    raise OSError("injected state commit failure")
                return real_write(path, text)

            memory.pending.clear()
            memory.evidence.clear()
            with patch.object(folio_module, "_atomic_write_text", side_effect=fail_state):
                self.assertFalse(memory.save_snapshot(target))

            self.assertEqual((target / "state.json").read_bytes(), original_state)
            restored = FolioMemory.load_snapshot(target, ScriptedGenerate([]))
            self.assertEqual(len(restored.pending), 1)
            self.assertGreaterEqual(len(restored.evidence), 1)

    def test_reused_snapshot_directory_keeps_evidence_content_transactional(self):
        def make_memory(value):
            memory = FolioMemory(
                ScriptedGenerate([writer_response("0.0-1.0s")]),
                segment_frames=2,
                change_threshold=1.0,
            )
            observe_segment(memory, 0, [value, value])
            return memory

        old = make_memory(0)
        new = make_memory(1)
        old_frame = next(iter(old.evidence.values()))
        new_frame = next(iter(new.evidence.values()))
        self.assertEqual(len(old_frame.jpeg), len(new_frame.jpeg))
        self.assertNotEqual(old_frame.jpeg, new_frame.jpeg)

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "folio"
            self.assertTrue(old.save_snapshot(target))
            real_write = folio_module._atomic_write_text

            def fail_state(path, text):
                if Path(path).name == "state.json":
                    raise OSError("injected manifest failure")
                return real_write(path, text)

            with patch.object(folio_module, "_atomic_write_text", side_effect=fail_state):
                self.assertFalse(new.save_snapshot(target))
            restored = FolioMemory.load_snapshot(target, ScriptedGenerate([]))
            self.assertEqual(next(iter(restored.evidence.values())).jpeg, old_frame.jpeg)

            self.assertTrue(new.save_snapshot(target))
            restored = FolioMemory.load_snapshot(target, ScriptedGenerate([]))
            self.assertEqual(next(iter(restored.evidence.values())).jpeg, new_frame.jpeg)

    def test_restored_session_rejects_queries_before_snapshot_time(self):
        memory = FolioMemory(
            ScriptedGenerate([writer_response("0.0-1.0s")]),
            segment_frames=2,
            change_threshold=1.0,
        )
        observe_segment(memory, 0, [0, 1])
        session = FolioMemorySession(
            path="unused.mp4",
            fps=1.0,
            generate=ScriptedGenerate([]),
            config=memory.config,
            frame_source=[],
            memory=memory,
        )
        with self.assertRaises(ValueError):
            session.advance_to(0.5)
        with self.assertRaises(ValueError):
            memory.prepare_query(
                "q-too-early",
                "What happened?",
                ["A", "B", "C", "D"],
                query_time=0.5,
                recent_start=0.0,
            )

    def test_shared_generic_alias_does_not_merge_distinct_cups(self):
        red = detailed_object("red mug", category="container", location="counter", state="empty")
        blue = detailed_object("blue mug", category="container", location="sink", state="full")
        red.update(aliases=["cup"], attributes=["red"])
        blue.update(aliases=["cup"], attributes=["blue"])
        memory = FolioMemory(
            ScriptedGenerate([
                writer_response("0.0-1.0s", detailed=[red]),
                writer_response("2.0-3.0s", detailed=[blue]),
            ]),
            segment_frames=2,
            change_threshold=1.0,
        )
        observe_segment(memory, 0, [0, 1])
        observe_segment(memory, 2, [2, 3])
        self.assertEqual(len(memory.entities), 2)
        self.assertEqual(
            {entity.canonical_name: len(entity.observations) for entity in memory.entities.values()},
            {"red mug": 1, "blue mug": 1},
        )

        red["aliases"] = ["coffee mug"]
        renamed = detailed_object("coffee mug", category="container", location="sink", state="full")
        renamed.update(aliases=["red mug"], attributes=["red"])
        memory = FolioMemory(
            ScriptedGenerate([
                writer_response("0.0-1.0s", detailed=[red]),
                writer_response("2.0-3.0s", detailed=[renamed]),
            ]),
            segment_frames=2,
            change_threshold=1.0,
        )
        observe_segment(memory, 0, [0, 1])
        observe_segment(memory, 2, [2, 3])
        self.assertEqual(len(memory.entities), 1)
        self.assertEqual(len(next(iter(memory.entities.values())).observations), 2)

        for canonical, alias in (("bicycle", "bike"), ("sofa", "couch"), ("refrigerator", "fridge")):
            with self.subTest(canonical=canonical, alias=alias):
                original = detailed_object(canonical, category=canonical, location="left", state="visible")
                original["aliases"] = [alias]
                renamed = detailed_object(alias, category=canonical, location="right", state="visible")
                memory = FolioMemory(
                    ScriptedGenerate([
                        writer_response("0.0-1.0s", detailed=[original]),
                        writer_response("2.0-3.0s", detailed=[renamed]),
                    ]),
                    segment_frames=2,
                )
                observe_segment(memory, 0, [0, 1])
                observe_segment(memory, 2, [2, 3])
                self.assertEqual(len(memory.entities), 1)
                self.assertEqual(len(next(iter(memory.entities.values())).observations), 2)

    def test_temporal_attribute_and_spatial_targets_survive_recent_record_limits(self):
        for kind in ("attribute", "spatial"):
            for relation in ("first", "before", "after"):
                for weak in (False, True):
                    with self.subTest(kind=kind, relation=relation, weak=weak):
                        target = {"first": 0, "before": 1, "after": 2}[relation]
                        anchor = 2 if relation == "before" else 1
                        name = "street sign" if kind == "attribute" else "red mug"
                        outputs = []
                        for index in range(12):
                            item = detailed_object(
                                name,
                                category="sign" if kind == "attribute" else "container",
                                location="roadside" if kind == "attribute" else "counter",
                                state="checkpoint" if index == anchor else f"view {index}",
                                state_change=f"view {index}",
                            )
                            item["evidence_summary"] = ""
                            if kind == "attribute":
                                item["visible_text"] = "" if weak and index == target else (
                                    "STOP" if index == target else "YIELD"
                                )
                            else:
                                item["relations"] = [] if weak and index == target else [
                                    {"relation": "left" if index == target else "right", "target": "plate"}
                                ]
                            outputs.append(writer_response(f"{2 * index}.0-{2 * index + 1}.0s", detailed=[item]))
                        memory = FolioMemory(
                            ScriptedGenerate(outputs),
                            config=FolioConfig(segment_frames=2, change_threshold=1.0, max_cache_frames_per_query=1),
                        )
                        for index in range(12):
                            observe_segment(memory, 2 * index, [2 * index, 2 * index + 1])
                        cue = "first" if relation == "first" else f"{relation} the checkpoint"
                        question = (
                            f"What text was on the street sign {cue}?" if kind == "attribute" else
                            f"Was the red mug left or right of the plate {cue}?"
                        )
                        options = ["STOP", "YIELD", "SCHOOL", "Unknown"] if kind == "attribute" else [
                            "Left", "Right", "Above", "Unknown"
                        ]
                        plan = memory.prepare_query("q-temporal", question, options, query_time=24.0, recent_start=24.0)
                        self.assertEqual(plan.query_type, kind)
                        if weak:
                            # Strong recent distractors cannot substitute for the requested time's missing slot.
                            self.assertEqual(plan.evidence_timestamps, (float(2 * target),))
                        else:
                            selected = [record for record in plan.selected_records if record.observation_index == target]
                            self.assertTrue(selected)
                            self.assertIn("STOP" if kind == "attribute" else "left", selected[0].field_text)

    def test_semlink_keeps_hld_strict_but_allows_actual_concept_inference(self):
        generate = ScriptedGenerate([
            writer_response("0.0-1.0s", detailed=[
                detailed_object("bookshelf", category="furniture", location="study wall", state="filled with books")
            ])
        ])
        memory = FolioMemory(generate, segment_frames=2, change_threshold=1.0)
        observe_segment(memory, 0, [0, 1])
        entity_id = next(iter(memory.entities))
        generate.semlink_output = json.dumps({"relevant_object_ids": [entity_id], "suggested_option": "A"})
        hld = memory.prepare_query(
            "q-hld-semlink", "Where was the bicycle seat before I opened it?",
            ["On the bicycle", "On the floor", "In the garage", "Unable to answer"],
            query_time=2.0, recent_start=2.0, query_type_hint="HLD",
        )
        self.assertEqual(hld.direct_scores, ())
        self.assertEqual(hld.retrieval_mode, "semlink")
        self.assertEqual(hld.query_type, "hallucination-detection")
        self.assertNotIn("CONCEPT QUESTION", hld.memory_text)
        self.assertNotIn("LLM-SUGGESTED OPTION", hld.memory_text)
        concept = memory.prepare_query(
            "q-concept-semlink", "Which description best fits this person?",
            ["Avid reader", "Musician", "Athlete", "Unable to answer"],
            query_time=2.0, recent_start=2.0,
        )
        self.assertEqual(concept.retrieval_mode, "semlink")
        self.assertEqual(concept.query_type, "concept")
        self.assertIn("CONCEPT QUESTION", concept.memory_text)
        self.assertIn("LLM-SUGGESTED OPTION", concept.memory_text)

    def test_compressed_observation_updates_quality_and_current_event_frames(self):
        brief = "red mug rests on counter"
        compact = {"name": "red mug", "category": "container", "location": "counter", "state": "empty",
                   "brief": brief, "confidence": 0.3, "evidence_frames": [0]}
        outputs = [writer_response("0.0-1.0s", compact=[compact])]
        for index, confidence in ((1, 0.95), (2, 0.99)):
            item = detailed_object("red mug", category="container", location="counter", state="empty", state_change="stable")
            item.update(holder="", evidence_summary=brief, confidence=confidence)
            outputs.append(writer_response(f"{2 * index}.0-{2 * index + 1}.0s", detailed=[item], events=[
                event("touch", "The red mug was touched.", ["red mug"], [])
            ]))
        memory = FolioMemory(ScriptedGenerate(outputs), segment_frames=2, change_threshold=1.0)
        for index in range(3):
            observe_segment(memory, 2 * index, [2 * index, 2 * index + 1])
        observation = next(iter(memory.entities.values())).observations
        self.assertEqual(len(observation), 1)
        self.assertEqual((observation[0].start_time, observation[0].end_time), (0.0, 5.0))
        self.assertEqual(observation[0].detail, "detailed")
        self.assertEqual(observation[0].confidence, 0.99)
        for item in memory.events.values():
            self.assertTrue(item.evidence_frame_ids)
            frames = [memory.evidence[frame_id] for frame_id in item.evidence_frame_ids]
            self.assertTrue(all(frame.segment_id == item.segment_id for frame in frames))
            self.assertTrue(all(item.start_time <= frame.timestamp <= item.end_time for frame in frames))


if __name__ == "__main__":
    unittest.main()
