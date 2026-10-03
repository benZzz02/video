"""Small causal memories for OVO counting and evidence-retention tasks.

Writers receive sampled images, timestamps and observation instructions only.
They never receive benchmark answers, and the CRR writer never receives a
question.  A fixed time window is committed once; an unfinished window is
recomputed and replaced as more frames arrive.  This avoids counting the same
completion twice when successive queries land inside one window.
"""
from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Callable, Iterable

from PIL import Image

Generate = Callable[[list[Image.Image], str, int], str]
Frame = tuple[float, Image.Image]
_TOKEN = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]", re.IGNORECASE)
_STOP = set("a an and are as at be by did do does for from has have how in is it of on or that the their there this to was were what when where which who with you your".split())


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _text(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be nonempty text of at most {limit} characters")
    return " ".join(value.split())


def _index(value: Any, length: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < length:
        raise ValueError("frame index must select one of the supplied images")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json_object(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise ValueError("writer must return a JSON string")
    value = raw.strip()
    if value.startswith("```json\n") and value.endswith("\n```"):
        value = value[8:-4]
    parsed = json.loads(value, object_pairs_hook=_unique_object)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("events"), list):
        raise ValueError("writer must return an object containing an events list")
    return parsed


class _WindowSession:
    """Common streaming, transaction and fixed-window ownership machinery."""

    def __init__(
        self, *, frame_source: Iterable[Frame], fps: float, generate: Generate,
        window_seconds: float, overlap_seconds: float, max_tokens: int,
    ) -> None:
        self.fps = _finite(fps, "fps")
        self.window_seconds = _finite(window_seconds, "window_seconds")
        self.overlap_seconds = _finite(overlap_seconds, "overlap_seconds")
        if self.fps <= 0 or self.window_seconds <= 0:
            raise ValueError("fps and window_seconds must be positive")
        if not 0 <= self.overlap_seconds < self.window_seconds:
            raise ValueError("overlap_seconds must be nonnegative and less than window_seconds")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        self.max_tokens = max_tokens
        self.generate = generate
        self._source = iter(frame_source)
        self._lookahead: Frame | None = None
        self._eof = False
        self._closed = False
        self._source_time = -math.inf
        self._window_start: float | None = None
        self._frames: list[Frame] = []
        self._new_frames = 0
        self._committed: list[dict[str, Any]] = []
        self._provisional: list[dict[str, Any]] = []
        self._state: list[dict[str, str]] = []
        self._provisional_state: list[dict[str, str]] = []
        self._attempt_signature: tuple[float, int, float] | None = None
        self._failures: dict[float, dict[str, Any]] = {}
        self._stream_failure: dict[str, Any] | None = None
        self._coverage_failure: dict[str, Any] | None = None
        self._last_observed_time: float | None = None
        self.last_time = -math.inf
        self._stats: dict[str, int | float] = {
            "write_calls": 0, "write_errors": 0, "write_seconds": 0.0,
            "observed_frames": 0, "stream_errors": 0, "committed_windows": 0,
            "writer_events_emitted": 0, "writer_events_kept": 0,
        }

    @property
    def status(self) -> str:
        """Complete means processing succeeded, not that model observations are exact."""
        return "degraded" if self._failures or self._stream_failure or self._coverage_failure else "complete"

    @property
    def events(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._committed + self._provisional]

    def usage(self) -> dict[str, Any]:
        gaps = [dict(item) for _, item in sorted(self._failures.items())]
        if self._stream_failure:
            gaps.append(dict(self._stream_failure))
        if self._coverage_failure:
            gaps.append(dict(self._coverage_failure))
        return {
            **self._stats, "status": self.status, "gaps": gaps,
            "record_count": len(self._committed) + len(self._provisional),
            "provisional_record_count": len(self._provisional),
            "last_time": self.last_time if math.isfinite(self.last_time) else None,
            "last_observed_time": self._last_observed_time,
        }

    def _peek(self) -> Frame | None:
        if self._lookahead is not None or self._eof:
            return self._lookahead
        try:
            timestamp, image = next(self._source)
            timestamp = _finite(timestamp, "frame timestamp")
            if timestamp < 0 or timestamp <= self._source_time:
                raise ValueError("frame timestamps must be nonnegative and strictly increasing")
            if not isinstance(image, Image.Image):
                raise ValueError("frame source must yield PIL images")
            self._source_time = timestamp
            self._lookahead = (timestamp, image)
        except StopIteration:
            self._eof = True
        except Exception as exc:
            self._eof = True
            self._stats["stream_errors"] += 1
            self._stream_failure = {
                "start_time": self._last_observed_time,
                "end_time": None, "error": f"{type(exc).__name__}: {exc}",
            }
        return self._lookahead

    def advance_to(self, timestamp: float) -> None:
        if self._closed:
            raise RuntimeError("session is closed")
        target = _finite(timestamp, "query timestamp")
        if target < 0 or target < self.last_time:
            raise ValueError("queries must have nonnegative, nondecreasing timestamps")
        if target == self.last_time:
            return
        while True:
            item = self._peek()
            if item is None or item[0] > target:
                break
            frame_time, image = item
            self._lookahead = None
            if self._window_start is None:
                # A CRR source may deliberately start after the question is asked.
                self._window_start = (
                    max(0, math.ceil(frame_time / self.window_seconds) - 1)
                    * self.window_seconds
                )
            if frame_time > self._window_start + self.window_seconds:
                self._finish_window()
                # Empty sampling intervals need no model call and are not errors.
                self._window_start = max(
                    self._window_start,
                    (math.ceil(frame_time / self.window_seconds) - 1) * self.window_seconds,
                )
                self._trim_overlap()
            self._frames.append((frame_time, image))
            self._new_frames += 1
            self._stats["observed_frames"] += 1
            self._last_observed_time = frame_time
        if self._window_start is not None:
            if target >= self._window_start + self.window_seconds:
                self._finish_window()
                # Move to the current fixed window, retaining only its overlap.
                self._window_start = max(
                    self._window_start,
                    math.floor(target / self.window_seconds) * self.window_seconds,
                )
                self._trim_overlap()
            elif self._new_frames:
                self._analyse(target)
        if self._eof and not self._stream_failure:
            tolerance = 1.0 / self.fps + 1e-6
            if self._last_observed_time is None or target > self._last_observed_time + tolerance:
                self._coverage_failure = {
                    "start_time": self._last_observed_time,
                    "end_time": target,
                    "error": "frame source ended before covering the query cutoff",
                }
        self.last_time = target

    def _trim_overlap(self) -> None:
        assert self._window_start is not None
        self._frames = [
            item for item in self._frames
            if item[0] >= self._window_start - self.overlap_seconds
        ]

    def _finish_window(self) -> None:
        assert self._window_start is not None
        if self._new_frames:
            self._analyse(self._window_start + self.window_seconds)
            self._committed.extend(self._provisional)
            self._state = self._provisional_state
            self._stats["committed_windows"] += 1
        self._provisional = []
        self._provisional_state = list(self._state)
        self._attempt_signature = None
        self._new_frames = 0
        self._window_start += self.window_seconds
        self._trim_overlap()

    def _analyse(self, cutoff: float) -> None:
        assert self._window_start is not None
        signature = (self._window_start, self._new_frames, self._frames[-1][0])
        if signature == self._attempt_signature:
            return
        self._attempt_signature = signature
        self._stats["write_calls"] += 1
        started = time.perf_counter()
        try:
            raw = self.generate([image for _, image in self._frames], self._prompt(cutoff), self.max_tokens)
            payload = _json_object(raw)
            records, state = self._parse(payload)
            # Parsing and ownership validation finish before any state is changed.
            self._provisional = records
            self._provisional_state = state
            self._stats["writer_events_emitted"] += len(payload["events"])
            self._stats["writer_events_kept"] += len(records)
            self._failures.pop(self._window_start, None)
        except Exception as exc:
            self._stats["write_errors"] += 1
            self._failures[self._window_start] = {
                "start_time": self._window_start, "end_time": cutoff,
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            self._stats["write_seconds"] += time.perf_counter() - started

    def _frame_info(self, cutoff: float) -> str:
        assert self._window_start is not None
        return (
            f"Frame indices are zero-based. Sampling rate: {self.fps:g} fps.\n"
            f"Current interval: ({self._window_start:g}, {cutoff:g}] seconds "
            "(include time 0 if supplied). Older frames are overlap context only.\n"
            "Frame timestamps (seconds): "
            + json.dumps([{"frame_index": i, "time": t} for i, (t, _) in enumerate(self._frames)])
            + "\nEligible indices for NEW events: "
            + json.dumps([i for i, (t, _) in enumerate(self._frames) if self._owned(t)])
        )

    def _owned(self, timestamp: float) -> bool:
        assert self._window_start is not None
        return timestamp > self._window_start or (timestamp == 0 and self._window_start == 0)

    def _prompt(self, cutoff: float) -> str:
        raise NotImplementedError

    def _parse(self, payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        raise NotImplementedError

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self._source, "close", None)
        try:
            if close is not None:
                close()
        finally:
            self._lookahead = None
            self._frames.clear()


class RecCountSession(_WindowSession):
    """Count visually completed repetitions, retaining actor/phase boundary state.

    ``count`` is the number of extracted events. If ``status == 'degraded'`` it
    is incomplete and must not be presented as a reliable cumulative count.
    """

    def __init__(
        self, *, frame_source: Iterable[Frame], fps: float, generate: Generate,
        activity: str, window_seconds: float = 4.0, overlap_seconds: float = 1.0,
        max_tokens: int = 384,
    ) -> None:
        self.activity = _text(activity, "activity", 2048)
        super().__init__(frame_source=frame_source, fps=fps, generate=generate,
                         window_seconds=window_seconds, overlap_seconds=overlap_seconds,
                         max_tokens=max_tokens)

    @property
    def count(self) -> int:
        return len(self._committed) + len(self._provisional)

    def _prompt(self, cutoff: float) -> str:
        recent = self._committed[-12:]
        return (
            "Observe every supplied sampled frame and extract completed repetitions of this activity: "
            + json.dumps(self.activity, ensure_ascii=False)
            + ". Count each distinct occurrence once when the activity's observable goal is reached. "
            "For showing or presenting an object, a deliberate completed presentation counts even "
            "if the object is still held up; holding the same presentation across more frames is "
            "not another occurrence. For lifting or placing, count the achieved goal, without "
            "waiting for the person to return to a resting pose. For cyclic exercise, require "
            "a full motion cycle. Do not count a preparation that has not achieved the activity. "
            "A repetition may start in an earlier window: use ongoing actor phases and overlap "
            "to recognize its completion now. Use stable actor IDs (e.g. person_1), preserving "
            "existing IDs. Different people completing the activity count separately. "
            "Choose the exact supplied frame at which each completion is first visible; do not "
            "estimate times or emit a cumulative number. Emit only completions whose end frame "
            "is inside the current interval. Do not repeat any recent completed event. "
            "The current unfinished interval is recomputed from its beginning, so include all "
            "completions in that interval. Describe the actual object, actor phase and whether "
            "the current occurrence has already been counted, so the next window can distinguish "
            "a continuation from a new occurrence.\n"
            + self._frame_info(cutoff)
            + "\nRecent committed completions: " + json.dumps(recent, ensure_ascii=False)
            + "\nActor phases at the start of this interval: " + json.dumps(self._state, ensure_ascii=False)
            + '\nReturn only a JSON object with two fields: '
              '"events": a list of objects with "actor" (stable person ID string), '
              '"end_frame_index" (an integer from the eligible indices), and "description" '
              '(the observed completed action and object); '
              '"active_states": a list of objects with "actor" and "state" (both strings). '
              'Use empty lists only when no corresponding event or actor state is visible. '
              'Fill all descriptions from the images; do not copy these field instructions.'
        )

    def _parse(self, payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        if set(payload) - {"events", "active_states"}:
            raise ValueError("unexpected counting writer fields")
        states = payload.get("active_states", [])
        if not isinstance(states, list):
            raise ValueError("active_states must be a list")
        parsed_states = []
        seen_actors: set[str] = set()
        for state in states:
            if not isinstance(state, dict) or set(state) != {"actor", "state"}:
                raise ValueError("each actor state must contain actor and state")
            actor = _text(state["actor"], "actor", 96).casefold()
            if actor in seen_actors:
                raise ValueError("duplicate actor state")
            seen_actors.add(actor)
            parsed_states.append({"actor": actor, "state": _text(state["state"], "state", 384)})
        records = []
        seen: set[tuple[str, float]] = set()
        for event in payload["events"]:
            if not isinstance(event, dict) or set(event) - {"actor", "end_frame_index", "description"}:
                raise ValueError("unexpected completion event fields")
            actor = _text(event.get("actor"), "actor", 96).casefold()
            index = _index(event.get("end_frame_index"), len(self._frames))
            description = ""
            if "description" in event:
                description = _text(event["description"], "description", 512)
            timestamp = self._frames[index][0]
            key = (actor, timestamp)
            if self._owned(timestamp) and key not in seen:
                records.append({"actor": actor, "time": timestamp, "description": description})
                seen.add(key)
        records.sort(key=lambda event: (event["time"], event["actor"]))
        return records, parsed_states


class CrrEvidenceSession(_WindowSession):
    """Question-independent factual event log, with inexpensive lexical retrieval."""

    def __init__(
        self, *, frame_source: Iterable[Frame], fps: float, generate: Generate,
        window_seconds: float = 8.0, overlap_seconds: float = 1.0, max_tokens: int = 384,
    ) -> None:
        super().__init__(frame_source=frame_source, fps=fps, generate=generate,
                         window_seconds=window_seconds, overlap_seconds=overlap_seconds,
                         max_tokens=max_tokens)

    def _prompt(self, cutoff: float) -> str:
        return (
            "Write a short factual video event log for later retrieval. Observe all supplied "
            "images and record visible actions, objects, interactions, ongoing states and readable text. "
            "Each entry must be a self-contained visual observation, tied to a supplied frame. "
            "Use precise actor/object descriptions. Record facts even if they leave the frame later. "
            "Do not answer questions, output Yes/No decisions, judge whether information is "
            "sufficient, or infer anything outside the images. Emit events only in the current "
            "interval; older overlap frames provide context. Recompute the entire unfinished "
            "interval when it is supplied again.\n"
            + self._frame_info(cutoff)
            + '\nReturn only a JSON object with an "events" list. Each entry has exactly '
              '"frame_index" (an integer from the eligible indices) and "text" '
              '(a short specific description of what is visible in that frame). '
              'A visible ongoing state is valid evidence even without a scene change. '
              'Use an empty list only if no factual visual observation can be made. '
              'Fill the text from the images, not from an example or these instructions.'
        )

    def _parse(self, payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        if set(payload) != {"events"}:
            raise ValueError("unexpected evidence writer fields")
        records = []
        seen: set[tuple[float, str]] = set()
        for event in payload["events"]:
            if not isinstance(event, dict) or set(event) != {"frame_index", "text"}:
                raise ValueError("each evidence event must contain frame_index and text")
            index = _index(event["frame_index"], len(self._frames))
            fact = _text(event["text"], "evidence text", 1024)
            if (
                re.fullmatch(r"(?:yes|no|true|false)[.!\s]*", fact, re.IGNORECASE)
                or re.search(r"\b(?:answer\s*(?:is|:)|(?:enough|sufficient)\s+(?:information|evidence))\b", fact, re.IGNORECASE)
            ):
                raise ValueError("answer/sufficiency decisions are not visual evidence")
            timestamp = self._frames[index][0]
            key = (timestamp, fact.casefold())
            if self._owned(timestamp) and key not in seen:
                records.append({"time": timestamp, "text": fact})
                seen.add(key)
        records.sort(key=lambda event: event["time"])
        return records, []

    def memory_text(self, question: str, max_bytes: int = 8192) -> str:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        if max_bytes == 0:
            return ""
        query_terms = set(_TOKEN.findall(str(question).casefold())) - _STOP
        requested_times = [float(value) for value in re.findall(r"\b(\d+(?:\.\d+)?)\s*(?:s\b|sec|second)", str(question), re.IGNORECASE)]
        records = self._committed + self._provisional

        def rank(event: dict[str, Any]) -> tuple[float, float, float]:
            terms = set(_TOKEN.findall(event["text"].casefold())) - _STOP
            relevance = len(query_terms & terms)
            proximity = -min(abs(event["time"] - t) for t in requested_times) if requested_times else 0.0
            return float(relevance), proximity, event["time"]

        prefix = ""
        if self.status != "complete":
            prefix = "[Memory incomplete: some video intervals could not be processed.]\n"
        budget = max_bytes - len(prefix.encode("utf-8"))
        if budget <= 0:
            return prefix.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
        selected: list[tuple[float, str]] = []
        seen: set[str] = set()
        for event in sorted(records, key=rank, reverse=True):
            key = event["text"].casefold()
            if key in seen:
                continue
            seen.add(key)
            line = f"[{event['time']:g}s] {event['text']}\n"
            encoded = line.encode("utf-8")
            if len(encoded) <= budget:
                selected.append((event["time"], line))
                budget -= len(encoded)
            elif not selected:
                selected.append((event["time"], encoded[:budget].decode("utf-8", errors="ignore")))
                budget = 0
            if budget == 0:
                break
        selected.sort(key=lambda item: item[0])
        return (prefix + "".join(line for _, line in selected)).rstrip()
