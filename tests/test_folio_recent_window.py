import importlib.util
import unittest
from unittest.mock import patch

from PIL import Image

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
if TORCH_AVAILABLE:
    import torch

    from lib.recent_window_eval import EvalChunk, RecentWindowQAModel, query_recent_window


def frame(value: int) -> Image.Image:
    return Image.new("RGB", (2, 2), (value, value, value))


@unittest.skipUnless(TORCH_AVAILABLE, "requires torch")
class FolioRecentWindowCheck(unittest.TestCase):
    def test_text_only_generation_builds_no_visual_input(self):
        class Processor:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                return {
                    "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
                    "attention_mask": torch.ones((1, 3), dtype=torch.long),
                }

        qa = object.__new__(RecentWindowQAModel)
        qa.processor = Processor()
        qa._last_num_vision_tokens = 99
        qa._last_num_vision_frames = 9
        qa._get_text_input_device = lambda: torch.device("cpu")
        captured = {}

        def generate(**kwargs):
            captured.update(kwargs)
            return "linked"

        qa._generate_from_model_inputs = generate
        result = qa.generate_from_text("catalog query")
        self.assertEqual(result, "linked")
        self.assertEqual(qa._last_num_vision_tokens, 0)
        self.assertEqual(qa._last_num_vision_frames, 0)
        self.assertEqual(captured["prompt_length"], 3)
        self.assertNotIn("pixel_values", captured)

    def test_historical_frames_prepend_without_changing_recent_selection(self):
        chunks = [
            EvalChunk([frame(10)], [1.0], 1.0, 2.0, 1, 1.0),
            EvalChunk([frame(20)], [2.0], 2.0, 3.0, 2, 1.0),
            EvalChunk([frame(30)], [3.0], 3.0, 4.0, 3, 1.0),
        ]

        class QA:
            def __init__(self):
                self.calls = []
                self._last_ttft_seconds = 0.01
                self._last_num_vision_tokens = 0
                self._last_num_vision_frames = 0

            def generate_from_frames(self, frames, prompt):
                self.calls.append((list(frames), prompt))
                self._last_num_vision_frames = len(frames)
                self._last_num_vision_tokens = len(frames) * 10
                return "A"

        qa = QA()
        with patch(
            "lib.recent_window_eval.decode_video_to_chunks_qwen",
            return_value=(chunks, "fake"),
        ):
            baseline, _ = query_recent_window(
                qa=qa,
                video_path="unused.mp4",
                prompt="question",
                chunk_duration=1.0,
                fps=1.0,
                recent_frames_only=2,
            )
            folio, _ = query_recent_window(
                qa=qa,
                video_path="unused.mp4",
                prompt="question",
                chunk_duration=1.0,
                fps=1.0,
                recent_frames_only=2,
                historical_frames=[frame(5)],
            )

        baseline_values = [item.getpixel((0, 0))[0] for item in qa.calls[0][0]]
        folio_values = [item.getpixel((0, 0))[0] for item in qa.calls[1][0]]
        self.assertEqual(baseline_values, [20, 30])
        self.assertEqual(folio_values, [5, 20, 30])
        self.assertEqual(baseline.final_chunk_ids, [2, 3])
        self.assertEqual(folio.final_chunk_ids, [2, 3])
        self.assertEqual(baseline.num_frames, 2)
        self.assertEqual(folio.num_frames, 3)


if __name__ == "__main__":
    unittest.main()
