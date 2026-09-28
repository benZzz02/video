"""Run with: python -m unittest discover -s tests -v"""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from lib.streaming_memory import (
    WRITER_TOKENS,
    VideoMemory,
    VideoMemorySession,
    video_frames,
)


def image():
    return SimpleNamespace(mode="RGB", size=(2, 2))


def upsert(name, description, kind, content):
    return {
        "op": "upsert",
        "name": name,
        "description": description,
        "type": kind,
        "content": content,
    }


def response(*operations):
    return json.dumps({"operations": list(operations)})


class VideoMemoryCheck(unittest.TestCase):
    def test_multiple_records_update_without_question_conditioning(self):
        outputs = [
            response(
                upsert(
                    "access-code",
                    "Code 4821 appears on the panel at 0–3s.",
                    "fact",
                    "[0–3s] The panel visibly shows the digits 4821.",
                )
            ),
            response(
                upsert(
                    "door-state",
                    "The door changes from closed to opening at 4–7s.",
                    "change",
                    "[4s] The door is closed. [7s] It begins to open.",
                )
            ),
            response(
                upsert(
                    "door-state",
                    "The door changes from closed to fully open by 11s.",
                    "change",
                    "[4s] Closed; [7s] opening; [11s] fully open.",
                )
            ),
        ]
        prompts = [[], []]

        def writer(index):
            calls = iter(outputs)

            def generate(frames, prompt, limit):
                prompts[index].append(prompt)
                self.assertEqual(len(frames), 4)
                self.assertEqual(limit, WRITER_TOKENS)
                return next(calls)

            return generate

        silent = VideoMemory(writer(0))
        queried = VideoMemory(writer(1))
        for timestamp in range(12):
            silent.observe(timestamp, image())
            queried.observe(timestamp, image())
            queried.augment(f"UNIQUE QUESTION {timestamp}")

        self.assertEqual(prompts[0], prompts[1])
        self.assertNotIn("UNIQUE QUESTION", "\n".join(prompts[0]))
        self.assertIn("access-code", prompts[0][1])
        self.assertEqual(set(silent.records), {"access-code", "door-state"})
        self.assertIn("[11s] fully open", silent.records["door-state"].content)
        self.assertEqual(silent.usage()["record_count"], 2)
        self.assertEqual(silent.usage()["write_calls"], 3)

        original = "Question: Which code?\n\nOnly give A, B, C, or D."
        answer_prompt = silent.augment(original)
        self.assertIn("[access-code](access-code.md)", answer_prompt)
        self.assertIn("## door-state.md", answer_prompt)
        self.assertTrue(answer_prompt.endswith(original))

    def test_updates_are_atomic_and_fail_open(self):
        outputs = iter(
            [
                response(
                    upsert(
                        "object-color",
                        "The object appears red at 0s.",
                        "fact",
                        "[0s] The visible object appears red.",
                    )
                ),
                response(
                    upsert("object-color", "first", "fact", "[1s] first"),
                    upsert("object-color", "second", "fact", "[1s] second"),
                ),
                response(
                    {"op": "delete", "name": "object-color"},
                    upsert(
                        "object-description",
                        "The red object is identified as a ball by 2s.",
                        "fact",
                        "[0s] The object appears red. [2s] Its round shape identifies it as a ball.",
                    ),
                ),
            ]
        )
        memory = VideoMemory(lambda *_: next(outputs), segment_frames=1)

        memory.observe(0, image())
        original = dict(memory.records)
        memory.observe(1, image())
        self.assertEqual(memory.records, original)
        self.assertEqual(memory.usage()["write_errors"], 1)
        self.assertEqual(memory.usage()["pending_frames"], 1)
        self.assertIn("object-color", memory.augment("Question?"))

        memory.observe(2, image())
        self.assertEqual(set(memory.records), {"object-description"})
        self.assertIn("[0s] The object appears red", memory.records["object-description"].content)
        self.assertEqual(memory.usage()["pending_frames"], 0)
        self.assertEqual(memory.usage()["write_calls"], 3)
        self.assertEqual(memory.usage()["write_errors"], 1)

    def test_snapshot_writes_readable_files_and_removes_merged_topic(self):
        outputs = iter(
            [
                response(
                    upsert(
                        "object-color",
                        "The object appears red at 0s.",
                        "fact",
                        "[0s] The visible object appears red.",
                    )
                ),
                response(
                    {"op": "delete", "name": "object-color"},
                    upsert(
                        "object-description",
                        "The red object is identified as a ball by 1s.",
                        "fact",
                        "[0s] The object appears red. [1s] It is a ball.",
                    ),
                ),
            ]
        )
        memory = VideoMemory(lambda *_: next(outputs), segment_frames=1)

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "memory"
            memory.observe(0, image())
            self.assertTrue(memory.save_snapshot(target))
            self.assertIn("object-color.md", (target / "MEMORY.md").read_text())
            self.assertIn(
                'description: "The object appears red at 0s."',
                (target / "object-color.md").read_text(),
            )

            memory.observe(1, image())
            self.assertTrue(memory.save_snapshot(target))
            self.assertFalse((target / "object-color.md").exists())
            self.assertTrue((target / "object-description.md").exists())
            index = (target / "MEMORY.md").read_text()
            self.assertNotIn("object-color.md", index)
            self.assertIn("object-description.md", index)
            usage = json.loads((target / "usage.json").read_text())
            self.assertEqual(usage["record_count"], 1)
            self.assertEqual(usage["record_files"], ["object-description.md"])
            self.assertEqual(usage["snapshot_errors"], 0)
            self.assertFalse(any(target.glob(".*.tmp")))

    def test_snapshot_failure_does_not_raise_or_change_memory(self):
        memory = VideoMemory(
            lambda *_: response(
                upsert("event", "An event occurs at 0s.", "fact", "[0s] Event.")
            ),
            segment_frames=1,
        )
        memory.observe(0, image())
        original = dict(memory.records)
        with tempfile.TemporaryDirectory() as directory:
            blocked = Path(directory) / "not-a-directory"
            blocked.write_text("file")
            self.assertFalse(memory.save_snapshot(blocked))
        self.assertEqual(memory.records, original)
        self.assertEqual(memory.usage()["snapshot_errors"], 1)

    def test_snapshot_recovers_and_cleans_orphan_after_partial_failure(self):
        outputs = iter(
            [
                response(
                    upsert(
                        "old-topic",
                        "An old topic is visible at 0s.",
                        "fact",
                        "[0s] Old topic.",
                    )
                ),
                response({"op": "delete", "name": "old-topic"}),
            ]
        )
        memory = VideoMemory(lambda *_: next(outputs), segment_frames=1)

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "memory"
            memory.observe(0, image())
            from lib import streaming_memory

            original_write = streaming_memory._atomic_write_text

            def fail_index(path, text):
                if path.name == "MEMORY.md":
                    raise OSError("injected index failure")
                return original_write(path, text)

            with patch("lib.streaming_memory._atomic_write_text", side_effect=fail_index):
                self.assertFalse(memory.save_snapshot(target))
            self.assertTrue((target / "old-topic.md").exists())

            memory.observe(1, image())
            self.assertTrue(memory.save_snapshot(target))
            self.assertFalse((target / "old-topic.md").exists())
            self.assertIn(
                "No retained memory records", (target / "MEMORY.md").read_text()
            )

    def test_repeated_writer_failures_keep_a_bounded_retry_buffer(self):
        memory = VideoMemory(lambda *_: "not json", segment_frames=2)
        for timestamp in range(10):
            memory.observe(timestamp, image())

        self.assertEqual(memory.usage()["write_calls"], 5)
        self.assertEqual(memory.usage()["write_errors"], 5)
        self.assertEqual(memory.usage()["pending_frames"], 4)
        self.assertEqual(memory.usage()["dropped_frames"], 6)

    def test_frame_times_must_increase(self):
        memory = VideoMemory(lambda *_: response(), segment_frames=2)
        memory.observe(0, image())
        with self.assertRaises(ValueError):
            memory.observe(0, image())
        memory.observe(1, image())
        self.assertEqual(memory.usage()["write_calls"], 1)
        self.assertEqual(memory.last_time, 1)

    def test_session_advances_causally_and_reuses_equal_cutoff(self):
        prompts = []
        outputs = iter(
            [
                response(
                    upsert(
                        "first-event",
                        "An event is observed from 0–1s.",
                        "fact",
                        "[0–1s] First event.",
                    )
                ),
                response(
                    upsert(
                        "second-event",
                        "A second event is observed from 2–3s.",
                        "fact",
                        "[2–3s] Second event.",
                    )
                ),
            ]
        )

        def generate(_frames, prompt, _limit):
            prompts.append(prompt)
            return next(outputs)

        source = ((float(timestamp), image()) for timestamp in range(6))
        session = VideoMemorySession(
            path="unused.mp4",
            fps=1.0,
            generate=generate,
            segment_frames=2,
            frame_source=source,
        )
        session.advance_to(1.0)
        session.advance_to(1.0)
        self.assertEqual(len(prompts), 1)
        self.assertIn("[0.0, 1.0]", prompts[0])
        self.assertNotIn("2.0, 3.0", prompts[0])

        session.advance_to(3.0)
        self.assertEqual(len(prompts), 2)
        self.assertIn("[2.0, 3.0]", prompts[1])
        self.assertEqual(session.usage()["record_count"], 2)
        with self.assertRaises(ValueError):
            session.advance_to(2.0)
        session.close()
        session.close()

    def test_session_stream_failure_does_not_block_answering(self):
        def broken_source():
            yield 0.0, image()
            raise RuntimeError("decode failed")

        session = VideoMemorySession(
            path="unused.mp4",
            fps=1.0,
            generate=lambda *_: response(),
            segment_frames=2,
            frame_source=broken_source(),
        )
        session.advance_to(3.0)
        self.assertEqual(session.usage()["stream_errors"], 1)
        self.assertEqual(session.augment("Question?"), "Question?")
        session.advance_to(4.0)
        self.assertEqual(session.usage()["stream_errors"], 1)

    @unittest.skipUnless(
        importlib.util.find_spec("av") and importlib.util.find_spec("PIL"),
        "requires av and Pillow",
    )
    def test_video_sampling_uses_one_fixed_source_clock(self):
        import av
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "video.mp4")
            with av.open(path, "w") as container:
                stream = container.add_stream("mpeg4", rate=10)
                stream.width, stream.height, stream.pix_fmt = 32, 16, "yuv420p"
                for index in range(60):
                    frame = av.VideoFrame.from_image(
                        Image.new("RGB", (32, 16), (index, 20, 30))
                    )
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)

            timestamps = [timestamp for timestamp, _ in video_frames(path, 1.0)]
            self.assertEqual(timestamps, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


if __name__ == "__main__":
    unittest.main()
