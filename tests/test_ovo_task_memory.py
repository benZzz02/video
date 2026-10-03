"""CPU tests for causal counting state and question-independent evidence logs."""
import json
import re
import unittest

from PIL import Image

from lib.ovo_task_memory import CrrEvidenceSession, RecCountSession


def frames(times):
    return [(float(t), Image.new("RGB", (2, 2), (int(t) % 256, 0, 0))) for t in times]


def times_in(prompt):
    line = next(line for line in prompt.splitlines() if line.startswith("Frame timestamps"))
    return [item["time"] for item in json.loads(line.split(": ", 1)[1])]


class Writer:
    def __init__(self, outputs=None, callback=None):
        self.outputs = iter(outputs or [])
        self.callback = callback
        self.calls = []

    def __call__(self, images, prompt, max_tokens):
        self.calls.append((list(images), prompt, max_tokens))
        value = self.callback(times_in(prompt), prompt) if self.callback else next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, str) else json.dumps(value)


def count_events(*indices, actor="person_1", state=None):
    result = {"events": [{"actor": actor, "end_frame_index": index} for index in indices]}
    if state is not None:
        result["active_states"] = [{"actor": actor, "state": state}]
    return result


class RecCountSessionCheck(unittest.TestCase):
    def test_query_cutoff_is_causal_and_same_query_does_no_work(self):
        consumed = []

        def source():
            for item in frames(range(11)):
                consumed.append(item[0])
                yield item

        writer = Writer(callback=lambda times, _prompt: count_events(len(times) - 1))
        session = RecCountSession(frame_source=source(), fps=1, generate=writer, activity="jump")
        session.advance_to(2)
        self.assertEqual(times_in(writer.calls[0][1]), [0, 1, 2])
        self.assertEqual(session.count, 1)
        self.assertEqual(session.usage()["observed_frames"], 3)
        before = (list(consumed), session.usage())
        session.advance_to(2)
        self.assertEqual((consumed, session.usage()), before)
        session.advance_to(7)
        for _, prompt, _ in writer.calls:
            cutoff = float(re.search(r"Current interval: \([^,]+, ([^)\]]+)\]", prompt).group(1))
            self.assertTrue(all(timestamp <= cutoff for timestamp in times_in(prompt)))
        self.assertEqual(session.last_time, 7)
        self.assertEqual(session.status, "complete")

    def test_unfinished_window_replaces_events_instead_of_union(self):
        writer = Writer(outputs=[count_events(1), count_events(2), count_events(2, 4)])
        session = RecCountSession(frame_source=frames(range(5)), fps=1, generate=writer, activity="jump")
        session.advance_to(1)
        self.assertEqual([event["time"] for event in session.events], [1])
        session.advance_to(2)
        self.assertEqual([event["time"] for event in session.events], [2])
        session.advance_to(4)
        self.assertEqual([event["time"] for event in session.events], [2, 4])
        self.assertEqual(session.count, 2)
        self.assertEqual(session.usage()["provisional_record_count"], 0)

    def test_overlap_dedup_and_cross_window_phase(self):
        writer = Writer(outputs=[
            count_events(4, state="arms down, next repetition beginning"),
            count_events(1, 4, state="resting"),
        ])
        session = RecCountSession(frame_source=frames(range(9)), fps=1, generate=writer, activity="jump")
        session.advance_to(4)
        session.advance_to(8)
        self.assertEqual(times_in(writer.calls[1][1]), [3, 4, 5, 6, 7, 8])
        self.assertEqual([event["time"] for event in session.events], [4, 7])
        self.assertIn("next repetition beginning", writer.calls[1][1])
        self.assertIn('"time": 4.0', writer.calls[1][1])
        self.assertEqual(session.count, 2)

    def test_partial_flush_does_not_drop_prefix_frames(self):
        writer = Writer(callback=lambda _times, _prompt: {"events": []})
        session = RecCountSession(frame_source=frames(range(13)), fps=1, generate=writer, activity="jump")
        for timestamp in (2, 6, 10, 12):
            session.advance_to(timestamp)
        observed = set()
        for images, prompt, _ in writer.calls:
            timestamps = times_in(prompt)
            self.assertEqual(len(images), len(timestamps))
            observed.update(timestamps)
        self.assertEqual(observed, set(range(13)))
        self.assertEqual(session.usage()["observed_frames"], 13)
        self.assertEqual(session.status, "complete")

    def test_bad_json_is_atomic_and_failed_closed_interval_is_reported(self):
        writer = Writer(outputs=[count_events(2), {
            "events": [{"actor": "person_1", "end_frame_index": 2},
                       {"actor": "person_1", "end_frame_index": 999}],
        }, count_events(4)])
        session = RecCountSession(frame_source=frames(range(13)), fps=1, generate=writer, activity="jump")
        session.advance_to(4)
        before = session.events
        session.advance_to(8)
        self.assertEqual(session.events, before)
        self.assertEqual(session.count, 1)
        self.assertEqual(session.status, "degraded")
        session.advance_to(12)
        self.assertEqual(session.count, 2)
        self.assertEqual(session.status, "degraded")
        gap = session.usage()["gaps"][0]
        self.assertEqual((gap["start_time"], gap["end_time"]), (4, 8))
        self.assertEqual(session.usage()["write_errors"], 1)

    def test_failed_first_window_does_not_claim_reliable_zero(self):
        writer = Writer(outputs=["not json"])
        session = RecCountSession(frame_source=frames(range(5)), fps=1, generate=writer, activity="jump")
        session.advance_to(4)
        self.assertEqual(session.count, 0)
        self.assertEqual(session.status, "degraded")
        self.assertEqual(session.usage()["record_count"], 0)
        self.assertTrue(session.usage()["gaps"])

    def test_partial_failure_can_be_recovered_by_recomputing_same_window(self):
        writer = Writer(outputs=[count_events(1), "broken", count_events(1, 4)])
        session = RecCountSession(frame_source=frames(range(5)), fps=1, generate=writer, activity="jump")
        session.advance_to(1)
        session.advance_to(2)
        self.assertEqual(session.count, 1)
        self.assertEqual(session.status, "degraded")
        session.advance_to(4)
        self.assertEqual(session.count, 2)
        self.assertEqual(session.status, "complete")
        self.assertEqual(session.usage()["write_errors"], 1)
        self.assertEqual(session.usage()["gaps"], [])

    def test_frames_are_not_sparsified_and_distinct_actors_count(self):
        writer = Writer(outputs=[{"events": [
            {"actor": "person_1", "end_frame_index": 2},
            {"actor": "PERSON_1", "end_frame_index": 2},
            {"actor": "person_2", "end_frame_index": 2},
        ]}])
        sampled = [i / 4 for i in range(17)]
        session = RecCountSession(frame_source=frames(sampled), fps=4, generate=writer, activity="jump")
        session.advance_to(4)
        self.assertEqual(times_in(writer.calls[0][1]), sampled)
        self.assertEqual(len(writer.calls[0][0]), 17)
        self.assertEqual(session.count, 2)

    def test_reverse_queries_and_nonfinite_values_are_rejected(self):
        writer = Writer(callback=lambda _times, _prompt: {"events": []})
        session = RecCountSession(frame_source=frames(range(4)), fps=1, generate=writer, activity="jump")
        session.advance_to(2)
        for timestamp in (1, -1, float("nan"), float("inf")):
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                session.advance_to(timestamp)
        self.assertEqual(session.last_time, 2)

    def test_decode_failure_is_reported_and_close_closes_source(self):
        closed = []

        def source():
            try:
                yield from frames([0, 1])
                raise OSError("decode failure")
            finally:
                closed.append(True)

        writer = Writer(outputs=[count_events(1)])
        session = RecCountSession(frame_source=source(), fps=1, generate=writer, activity="jump")
        session.advance_to(3)
        self.assertEqual(session.count, 1)
        self.assertEqual(session.status, "degraded")
        self.assertEqual(session.usage()["stream_errors"], 1)
        self.assertEqual(closed, [True])
        session.close()
        session.close()
        with self.assertRaises(RuntimeError):
            session.advance_to(4)

    def test_nonmonotonic_frames_fail_closed(self):
        writer = Writer(outputs=[count_events(1)])
        session = RecCountSession(frame_source=frames([0, 1, 1, 2]), fps=1, generate=writer, activity="jump")
        session.advance_to(4)
        self.assertEqual(session.usage()["observed_frames"], 2)
        self.assertEqual(session.status, "degraded")
        self.assertEqual(session.usage()["stream_errors"], 1)

    def test_empty_and_truncated_sources_do_not_report_complete_count(self):
        for supplied in ([], [0, 1, 2]):
            with self.subTest(supplied=supplied):
                writer = Writer(callback=lambda _times, _prompt: {"events": []})
                session = RecCountSession(frame_source=frames(supplied), fps=1,
                                          generate=writer, activity="jump")
                session.advance_to(10)
                self.assertEqual(session.count, 0)
                self.assertEqual(session.status, "degraded")
                self.assertEqual(session.usage()["gaps"][0]["end_time"], 10)
                self.assertEqual(session.usage()["stream_errors"], 0)

    def test_normal_unsampled_video_tail_is_not_a_coverage_gap(self):
        writer = Writer(callback=lambda _times, _prompt: {"events": []})
        session = RecCountSession(frame_source=frames([0, 1, 2]), fps=1,
                                  generate=writer, activity="jump")
        session.advance_to(2.9)
        self.assertEqual(session.status, "complete")
        session.advance_to(3.01)
        self.assertEqual(session.status, "degraded")
        self.assertEqual(session.usage()["last_observed_time"], 2)

    def test_strict_schema_rejects_boolean_indices_and_cumulative_numbers(self):
        invalid = [
            {"events": [{"actor": "person_1", "end_frame_index": True}]},
            {"events": [], "count": 3},
            {"events": [{"actor": "person_1", "end_frame_index": 0.5}]},
            '{"events":[],"events":[]}',
            {"events": [], "active_states": [{"actor": "p", "state": ""}]},
        ]
        for output in invalid:
            with self.subTest(output=output):
                session = RecCountSession(frame_source=frames([0, 1]), fps=1,
                                          generate=Writer(outputs=[output]), activity="jump")
                session.advance_to(1)
                self.assertEqual(session.count, 0)
                self.assertEqual(session.status, "degraded")


class CrrEvidenceSessionCheck(unittest.TestCase):
    def test_old_evidence_is_retained_and_retrieved_without_question_in_writer(self):
        def observe(times, _prompt):
            if 2 in times:
                return {"events": [{"frame_index": times.index(2), "text": "A red umbrella is placed in the blue locker."}]}
            return {"events": [{"frame_index": len(times) - 1, "text": "A person stands near a chair."}]}

        writer = Writer(callback=observe)
        session = CrrEvidenceSession(frame_source=frames(range(49)), fps=1, generate=writer)
        session.advance_to(48)
        events_before = session.events
        question = "Which locker contains the red umbrella? SECRET_QUESTION_GT_YES"
        memory = session.memory_text(question, max_bytes=100)
        self.assertIn("red umbrella", memory)
        self.assertIn("[2s]", memory)
        self.assertEqual(session.events, events_before)
        for _, prompt, _ in writer.calls:
            self.assertNotIn("SECRET_QUESTION_GT_YES", prompt)
            self.assertNotIn("Which locker", prompt)
        self.assertLessEqual(len(memory.encode("utf-8")), 100)

    def test_byte_limits_support_multibyte_facts(self):
        writer = Writer(outputs=[{"events": [{"frame_index": 0, "text": "一位穿蓝衣服的人把红色杯子放在桌子上。"}]}])
        session = CrrEvidenceSession(frame_source=frames([0]), fps=1, generate=writer)
        session.advance_to(0)
        for budget in range(0, 95):
            with self.subTest(budget=budget):
                memory = session.memory_text("杯子", max_bytes=budget)
                self.assertLessEqual(len(memory.encode("utf-8")), budget)
                self.assertNotIn("\ufffd", memory)
        with self.assertRaises(ValueError):
            session.memory_text("cup", max_bytes=-1)

    def test_answers_and_invalid_evidence_are_not_persisted_atomically(self):
        for text in ("Yes", "No.", "There is sufficient information to answer.", "The answer is yes."):
            with self.subTest(text=text):
                writer = Writer(outputs=[{"events": [
                    {"frame_index": 0, "text": "A red cup is visible."},
                    {"frame_index": 1, "text": text},
                ]}])
                session = CrrEvidenceSession(frame_source=frames([0, 1]), fps=1, generate=writer)
                session.advance_to(1)
                self.assertEqual(session.events, [])
                self.assertEqual(session.status, "degraded")
                self.assertIn("Memory incomplete", session.memory_text("cup"))

    def test_evidence_overlap_is_deduplicated(self):
        writer = Writer(outputs=[
            {"events": [{"frame_index": 8, "text": "A cup falls."}]},
            {"events": [{"frame_index": 1, "text": "A cup falls."},
                        {"frame_index": 3, "text": "The person picks up the cup."}]},
        ])
        session = CrrEvidenceSession(frame_source=frames(range(17)), fps=1, generate=writer)
        session.advance_to(8)
        session.advance_to(16)
        self.assertEqual([event["time"] for event in session.events], [8, 10])

    def test_partial_evidence_is_replaced_and_no_prefix_is_lost(self):
        writer = Writer(outputs=[
            {"events": [{"frame_index": 1, "text": "A person reaches for a cup."}]},
            {"events": [{"frame_index": 1, "text": "A person reaches for a cup."},
                        {"frame_index": 5, "text": "The person places the cup on a shelf."}]},
        ])
        session = CrrEvidenceSession(frame_source=frames(range(9)), fps=1, generate=writer)
        session.advance_to(2)
        session.advance_to(8)
        self.assertEqual([event["time"] for event in session.events], [1, 5])
        self.assertEqual(times_in(writer.calls[1][1]), list(range(9)))

    def test_late_start_and_boundary_first_frame_are_valid(self):
        for start in (80, 81):
            with self.subTest(start=start):
                writer = Writer(callback=lambda times, _prompt: {"events": [
                    {"frame_index": i, "text": f"An object is visible at {t}."} for i, t in enumerate(times)
                ]})
                session = CrrEvidenceSession(frame_source=frames([start, start + 1]), fps=1, generate=writer)
                session.advance_to(start + 1)
                self.assertEqual([event["time"] for event in session.events], [start, start + 1])
                self.assertEqual(session.status, "complete")
                self.assertEqual(session.usage()["observed_frames"], 2)

    def test_temporal_retrieval_prefers_requested_time(self):
        writer = Writer(callback=lambda times, _prompt: {"events": [{
            "frame_index": len(times) - 1, "text": f"The person holds a cup labeled {int(times[-1])}."
        }]})
        session = CrrEvidenceSession(frame_source=frames(range(25)), fps=1, generate=writer)
        session.advance_to(24)
        memory = session.memory_text("Which cup at 8 seconds?", max_bytes=56)
        self.assertIn("[8s]", memory)


if __name__ == "__main__":
    unittest.main()
