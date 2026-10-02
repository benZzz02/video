"""Paper-based FOLIO-style focused semantic memory for streaming video.

This is an independent implementation of the algorithms and public record
schemas in arXiv:2607.13298.  The paper does not publish runtime source code or
all numerical hyperparameters, so the defaults below are explicit local
choices rather than claimed official values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from io import BytesIO
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
import unicodedata
from typing import Any, Callable, Iterable, Iterator, Sequence

import numpy as np
from PIL import Image

from lib.streaming_memory import video_frames


WRITER_TOKENS = 3_072
SEMLINK_TOKENS = 512
DEFAULT_CHANGE_THRESHOLD = 0.03
DEFAULT_TOP_K_OBJECTS = 12
DEFAULT_TOP_M_EVENTS = 6
DEFAULT_QUERY_TOP_K = 6
DEFAULT_MEMORY_BYTES = 8_192
DEFAULT_CACHE_FRAMES_PER_QUERY = 2
MAX_DIALOGUE_TURNS = 8
MAX_DIALOGUE_BYTES = 4_096
MAX_TEXT_FIELD = 1_024

PROMPT_DIR = Path(__file__).parent / "prompts"
WRITER_PROMPT = (PROMPT_DIR / "folio_writer.md").read_text(encoding="utf-8")
SEMLINK_PROMPT = (PROMPT_DIR / "folio_semantic_link.md").read_text(encoding="utf-8")
ANSWER_PROMPT = (PROMPT_DIR / "folio_answer.md").read_text(encoding="utf-8")

TOKEN_RE = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does",
    "for", "from", "had", "has", "have", "he", "her", "his", "how", "i",
    "in", "is", "it", "its", "of", "on", "or", "she", "that", "the", "their",
    "there", "they", "this", "to", "was", "were", "what", "when", "where",
    "which", "who", "why", "with", "would", "you", "your",
}
GENERIC_NAMES = {"person", "man", "woman", "object", "item", "thing", "container"}
BACKGROUND_CATEGORIES = {"background", "wall", "floor", "ceiling", "scenery"}
MANIPULABLE_TERMS = {
    "tool", "container", "appliance", "screen", "text", "bottle", "cup", "knife",
    "pan", "pot", "drawer", "door", "phone", "book", "ball", "box", "bag",
}
ACTION_TERMS = {
    "action", "activity", "carry", "carried", "close", "closed", "cook",
    "cooked", "doing", "drink", "drinking", "drop", "dropped", "eat",
    "eating", "grab", "grabbed", "happen", "happened", "hold", "held",
    "move", "moved", "open", "opened", "pick", "picked", "place", "placed",
    "pour", "poured", "put", "read", "reading", "remove", "removed",
    "reach", "reached", "take", "took", "touch", "touched", "use", "used", "using", "wear",
    "wearing", "wash", "washed", "write", "writing",
}
QUERY_CONTROL_TERMS = {
    "answer", "best", "choose", "current", "currently", "describe",
    "description", "first", "last", "now", "option", "previous",
    "previously", "seen", "shown", "video", "visible",
}
REFERENCE_TERMS = {
    "he", "her", "hers", "him", "his", "it", "its", "she", "that", "them",
    "then", "they", "this", "those",
}
DISTINCTIVE_PREFIXES = {
    "black", "blue", "brown", "green", "grey", "gray", "orange", "pink", "purple",
    "red", "white", "yellow", "large", "small", "striped", "round", "square",
}
SYNONYM_GROUPS = (
    frozenset({"automobile", "car"}),
    frozenset({"bicycle", "bike"}),
    frozenset({"cellphone", "mobile", "phone"}),
    frozenset({"couch", "sofa"}),
    frozenset({"fridge", "refrigerator"}),
    frozenset({"television", "tv"}),
    frozenset({"carry", "carried"}),
    frozenset({"close", "closed"}),
    frozenset({"cook", "cooked"}),
    frozenset({"drink", "drinking"}),
    frozenset({"drop", "dropped"}),
    frozenset({"eat", "eating"}),
    frozenset({"grab", "grabbed"}),
    frozenset({"happen", "happened"}),
    frozenset({"hold", "held"}),
    frozenset({"move", "moved"}),
    frozenset({"open", "opened"}),
    frozenset({"pick", "picked"}),
    frozenset({"place", "placed"}),
    frozenset({"pour", "poured"}),
    frozenset({"reach", "reached"}),
    frozenset({"read", "reading"}),
    frozenset({"remove", "removed"}),
    frozenset({"take", "took"}),
    frozenset({"touch", "touched"}),
    frozenset({"use", "used", "using"}),
    frozenset({"wash", "washed"}),
    frozenset({"wear", "wearing"}),
    frozenset({"write", "writing"}),
)
SYNONYM_MAP = {
    token: group
    for group in SYNONYM_GROUPS
    for token in group
}


@dataclass(frozen=True)
class FolioConfig:
    segment_frames: int = 8
    change_threshold: float = DEFAULT_CHANGE_THRESHOLD
    top_k_objects: int = DEFAULT_TOP_K_OBJECTS
    top_m_events: int = DEFAULT_TOP_M_EVENTS
    query_top_k: int = DEFAULT_QUERY_TOP_K
    memory_bytes: int = DEFAULT_MEMORY_BYTES
    max_cache_frames_per_query: int = DEFAULT_CACHE_FRAMES_PER_QUERY
    semantic_link: bool = True
    cache_replay: bool = True
    interaction_focus: bool = True
    structured_answer: bool = True
    focus_decay: float = 0.85
    focus_threshold: float = 0.70
    support_threshold: float = 0.40
    context_threshold: float = 0.15
    sufficient_relevance: float = 8.0

    def validate(self) -> None:
        if self.segment_frames < 1:
            raise ValueError("segment_frames must be positive")
        if not 0.0 <= self.change_threshold <= 1.0:
            raise ValueError("change_threshold must be between zero and one")
        if self.top_k_objects < 1 or self.top_m_events < 0 or self.query_top_k < 1:
            raise ValueError("FOLIO object/event/query budgets must be positive")
        if self.memory_bytes < 1 or self.max_cache_frames_per_query < 0:
            raise ValueError("FOLIO memory and cache budgets must be nonnegative")
        if not 0.0 <= self.focus_decay <= 1.0:
            raise ValueError("focus_decay must be between zero and one")
        if not (
            0.0 <= self.context_threshold <= self.support_threshold
            <= self.focus_threshold <= 1.0
        ):
            raise ValueError("FOLIO focus thresholds must be ordered")

    @classmethod
    def profile(cls, name: str, *, segment_frames: int) -> "FolioConfig":
        if name == "compat":
            return cls(
                segment_frames=segment_frames,
                semantic_link=False,
                cache_replay=False,
                interaction_focus=False,
                structured_answer=False,
            )
        if name == "full":
            return cls(segment_frames=segment_frames)
        raise ValueError("FOLIO profile must be 'compat' or 'full'")


@dataclass
class FolioObservation:
    segment_id: int
    start_time: float
    end_time: float
    detail: str
    location: str = ""
    holder: str = ""
    state: str = ""
    relations: list[dict[str, str]] = field(default_factory=list)
    interactions: list[dict[str, str]] = field(default_factory=list)
    state_change: str = ""
    visible_text: str = ""
    evidence_summary: str = ""
    confidence: float = 0.0
    evidence_frame_ids: list[str] = field(default_factory=list)


@dataclass
class FolioEntity:
    id: str
    canonical_name: str
    aliases: list[str]
    category: str
    attributes: list[str]
    first_seen: float
    last_seen: float
    last_seen_segment: int
    focus_score: float = 0.0
    observations: list[FolioObservation] = field(default_factory=list)
    event_ids: list[str] = field(default_factory=list)


@dataclass
class FolioEvent:
    id: str
    segment_id: int
    start_time: float
    end_time: float
    event_type: str
    summary: str
    participants: list[str]
    participant_ids: list[str]
    changed_objects: list[str]
    changed_entity_ids: list[str]
    confidence: float
    evidence_frame_ids: list[str]
    focus_score: float = 0.0


@dataclass
class EvidenceFrame:
    id: str
    segment_id: int
    timestamp: float
    segment_frame_index: int
    change_score: float
    jpeg: bytes = field(repr=False)
    entity_ids: list[str] = field(default_factory=list)
    event_ids: list[str] = field(default_factory=list)

    def image(self) -> Image.Image:
        with Image.open(BytesIO(self.jpeg)) as image:
            return image.convert("RGB")

    def metadata(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("jpeg", None)
        digest = hashlib.sha256(self.jpeg).hexdigest()[:16]
        value["jpeg_file"] = f"{self.id}-{digest}.jpg"
        return value


@dataclass(frozen=True)
class QueryIntent:
    query_type: str
    normalized: str
    tokens: tuple[str, ...]
    target_terms: tuple[str, ...]
    anchor_terms: tuple[str, ...]
    verb_terms: tuple[str, ...]
    option_tokens: tuple[str, ...]
    key_option_terms: tuple[str, ...]
    temporal_scope: str
    temporal_relation: str
    answer_slot: str
    coverage_required: bool
    history_entity_ids: tuple[str, ...]
    history_event_ids: tuple[str, ...]


@dataclass(frozen=True)
class SelectedRecord:
    kind: str
    identifier: str
    entity_id: str
    observation_index: int | None
    event_id: str
    start_time: float
    end_time: float
    score: float
    grounding_score: float
    confidence: float
    evidence_frame_ids: tuple[str, ...]
    field_text: str = field(repr=False, compare=False)

    def metadata(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("field_text", None)
        return value


@dataclass(frozen=True)
class DialogueTurn:
    query_id: str
    timestamp: float
    question: str
    options: tuple[str, ...]
    predicted_label: str
    predicted_text: str
    selected_entity_ids: tuple[str, ...]
    selected_event_ids: tuple[str, ...]


@dataclass(frozen=True)
class FolioQueryPlan:
    query_id: str
    query_time: float
    prompt: str
    recent_only_prompt: str
    evidence_frames: tuple[Image.Image, ...]
    evidence_timestamps: tuple[float, ...]
    evidence_frame_ids: tuple[str, ...]
    selected_entity_ids: tuple[str, ...]
    selected_event_ids: tuple[str, ...]
    matched_event_ids: tuple[str, ...]
    selected_records: tuple[SelectedRecord, ...]
    candidate_records: tuple[SelectedRecord, ...] = field(repr=False, compare=False)
    direct_scores: tuple[tuple[str, float], ...]
    retrieval_mode: str
    semlink_status: str
    query_type: str
    memory_text: str
    memory_bytes: int
    dialogue_turns_used: int
    best_grounding_score: float
    suggested_option: str = ""

    def to_metadata(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query_time": self.query_time,
            "selected_entity_ids": list(self.selected_entity_ids),
            "selected_event_ids": list(self.selected_event_ids),
            "matched_event_ids": list(self.matched_event_ids),
            "selected_records": [item.metadata() for item in self.selected_records],
            "cache_candidate_record_count": len(self.candidate_records),
            "direct_scores": [list(item) for item in self.direct_scores],
            "retrieval_mode": self.retrieval_mode,
            "semlink_status": self.semlink_status,
            "query_type": self.query_type,
            "recovered_evidence_ids": list(self.evidence_frame_ids),
            "recovered_evidence_timestamps": list(self.evidence_timestamps),
            "memory_bytes": self.memory_bytes,
            "dialogue_turns_used": self.dialogue_turns_used,
            "best_grounding_score": self.best_grounding_score,
            "suggested_option": self.suggested_option,
        }


@dataclass(frozen=True)
class KeyframeSelection:
    indices: tuple[int, ...]
    max_change: float
    full_candidates: bool


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _atomic_write_text(path: Path, text: str) -> None:
    _atomic_write(path, text.encode("utf-8"))


def _json_object(output: str) -> dict[str, Any]:
    if not isinstance(output, str):
        raise ValueError("VLM returned non-text output")
    text = output.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("VLM output must be one JSON object")
    return value


def _clean_text(value: Any, *, required: bool = False, limit: int = MAX_TEXT_FIELD) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise ValueError("FOLIO text fields must be strings")
    cleaned = " ".join(value.strip().split())
    if required and not cleaned:
        raise ValueError("Required FOLIO text field is empty")
    if len(cleaned) > limit:
        raise ValueError("FOLIO text field exceeds its size limit")
    return cleaned


def _string_list(value: Any, *, limit: int = 32) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > limit:
        raise ValueError("Expected a bounded string list")
    output: list[str] = []
    for item in value:
        text = _clean_text(item)
        if text and text not in output:
            output.append(text)
    return output


def _confidence(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("confidence must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError("confidence must be between zero and one")
    return number


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    tokens = TOKEN_RE.findall(value)
    while tokens and tokens[0] in {"a", "an", "the"}:
        tokens.pop(0)
    return " ".join(tokens)


def _tokens(text: str) -> set[str]:
    return {token for token in TOKEN_RE.findall(_normalize(text)) if token not in STOPWORDS}


def _expand_synonyms(tokens: Iterable[str]) -> set[str]:
    expanded: set[str] = set(tokens)
    for token in tuple(expanded):
        expanded.update(SYNONYM_MAP.get(token, ()))
    return expanded


def _format_options(options: Sequence[str]) -> str:
    formatted = []
    for index, option in enumerate(options):
        text = str(option).strip()
        if not text.startswith(("A.", "B.", "C.", "D.")):
            text = f"{chr(65 + index)}. {text}"
        formatted.append(text)
    return "\n".join(formatted)


def _head(text: str) -> str:
    parts = _normalize(text).split()
    return parts[-1] if parts else ""


def _phrase_in(phrase: str, text: str) -> bool:
    normalized_phrase = _normalize(phrase)
    normalized_text = _normalize(text)
    if not normalized_phrase or not normalized_text:
        return False
    if any("\u3400" <= character <= "\u9fff" for character in normalized_phrase):
        return normalized_phrase in normalized_text
    return f" {normalized_phrase} " in f" {normalized_text} "


def _distinctive(attributes: Sequence[str]) -> set[str]:
    result: set[str] = set()
    for attribute in attributes:
        for token in _tokens(attribute):
            if token in DISTINCTIVE_PREFIXES or any(character.isdigit() for character in token):
                result.add(token)
    return result


def _image_probe(image: Image.Image) -> np.ndarray:
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    return np.asarray(image.convert("L").resize((64, 64), resampling), dtype=np.float32) / 255.0


def select_keyframes(
    frames: Sequence[tuple[float, Image.Image]],
    *,
    change_threshold: float = DEFAULT_CHANGE_THRESHOLD,
    focus_risk: bool = False,
) -> KeyframeSelection:
    """Select boundary/middle/high-change candidates using the FOLIO rule."""
    if not frames:
        raise ValueError("Cannot select keyframes from an empty segment")
    if len(frames) == 1:
        return KeyframeSelection((0,), 0.0, False)
    probes = [_image_probe(image) for _, image in frames]
    changes = [float(np.mean(np.abs(right - left))) for left, right in zip(probes, probes[1:])]
    max_change = max(changes, default=0.0)
    change_index = changes.index(max_change) + 1 if changes else 0
    candidates = sorted({0, len(frames) - 1, (len(frames) - 1) // 2, change_index})
    full = bool(focus_risk or max_change >= change_threshold)
    indices = tuple(candidates if full else sorted({0, len(frames) - 1}))
    return KeyframeSelection(indices, max_change, full)


def _encode_jpeg(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85, optimize=True)
    return buffer.getvalue()


def _encode_png(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _render(template: str, replacements: dict[str, Any]) -> str:
    text = template
    for key, value in replacements.items():
        text = text.replace("{{ " + key + " }}", str(value))
    return text


def _parse_frame_indices(value: Any, frame_count: int) -> list[int]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > frame_count:
        raise ValueError("evidence_frames must be a bounded list")
    output: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError("evidence frame indices must be integers")
        if item < 0 or item >= frame_count:
            raise ValueError("evidence frame index is outside the selected frames")
        if item not in output:
            output.append(item)
    return output


def _parse_relations(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 24:
        raise ValueError("relations must be a bounded list")
    output: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("relation entries must be objects")
        relation = _clean_text(item.get("relation"), required=True, limit=128)
        target = _clean_text(item.get("target"), required=True, limit=256)
        output.append({"relation": relation, "target": target})
    return output


def _parse_interactions(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list) or len(value) > 24:
        raise ValueError("interactions must be a bounded list")
    output: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("interaction entries must be objects")
        output.append(
            {
                "type": _clean_text(item.get("type"), required=True, limit=128),
                "with": _clean_text(item.get("with"), limit=256),
                "summary": _clean_text(item.get("summary"), limit=512),
            }
        )
    return output


def _parse_writer_output(
    output: str,
    *,
    frame_count: int,
    max_objects: int,
    max_events: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    value = _json_object(output)
    allowed_root = {"time", "detailed_objects", "compact_objects", "events"}
    if not set(value).issubset(allowed_root):
        raise ValueError("Writer output has unsupported root keys")
    detailed = value.get("detailed_objects", [])
    compact = value.get("compact_objects", [])
    events = value.get("events", [])
    if not isinstance(detailed, list) or not isinstance(compact, list) or not isinstance(events, list):
        raise ValueError("Writer object and event fields must be lists")
    if len(detailed) + len(compact) > max_objects or len(events) > max_events:
        raise ValueError("Writer output exceeds the configured semantic budget")

    parsed_detailed: list[dict[str, Any]] = []
    parsed_compact: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for source, detail, target in (
        (detailed, "detailed", parsed_detailed),
        (compact, "compact", parsed_compact),
    ):
        for item in source:
            if not isinstance(item, dict):
                raise ValueError("Writer object entries must be objects")
            name = _clean_text(item.get("name"), required=True, limit=256)
            normalized_name = _normalize(name)
            if not normalized_name or normalized_name in seen_names:
                raise ValueError("Writer duplicated an entity within one segment")
            seen_names.add(normalized_name)
            record: dict[str, Any] = {
                "name": name,
                "aliases": _string_list(item.get("aliases"), limit=16),
                "category": _clean_text(item.get("category"), limit=128),
                "location": _clean_text(item.get("location"), limit=512),
                "state": _clean_text(item.get("state"), limit=512),
                "confidence": _confidence(item.get("confidence", 0.0)),
                "evidence_frames": _parse_frame_indices(item.get("evidence_frames"), frame_count),
                "detail": detail,
            }
            if detail == "detailed":
                record.update(
                    {
                        "attributes": _string_list(item.get("attributes"), limit=32),
                        "holder": _clean_text(item.get("holder"), limit=256),
                        "relations": _parse_relations(item.get("relations")),
                        "interactions": _parse_interactions(item.get("interactions")),
                        "state_change": _clean_text(item.get("state_change"), limit=512),
                        "visible_text": _clean_text(item.get("visible_text"), limit=512),
                        "evidence_summary": _clean_text(item.get("evidence_summary"), limit=768),
                    }
                )
            else:
                relation = _clean_text(item.get("relation"), limit=512)
                record.update(
                    {
                        "attributes": [],
                        "holder": "",
                        "relations": ([{"relation": relation, "target": ""}] if relation else []),
                        "interactions": [],
                        "state_change": "",
                        "visible_text": "",
                        "evidence_summary": _clean_text(item.get("brief"), limit=768),
                    }
                )
            target.append(record)

    parsed_events: list[dict[str, Any]] = []
    for item in events:
        if not isinstance(item, dict):
            raise ValueError("Writer event entries must be objects")
        parsed_events.append(
            {
                "event_type": _clean_text(item.get("event_type"), required=True, limit=128),
                "summary": _clean_text(item.get("summary"), required=True, limit=768),
                "participants": _string_list(item.get("participants"), limit=24),
                "changed_objects": _string_list(item.get("changed_objects"), limit=24),
                "confidence": _confidence(item.get("confidence", 0.0)),
                "evidence_frames": _parse_frame_indices(item.get("evidence_frames"), frame_count),
            }
        )
    return parsed_detailed, parsed_compact, parsed_events


class FolioMemory:
    """Maintain FOLIO's entity memory, focus state, and visual evidence cache."""

    def __init__(
        self,
        generate: Callable[[list[Image.Image], str, int], str],
        segment_frames: int = 8,
        *,
        config: FolioConfig | None = None,
        change_threshold: float | None = None,
        max_cache_frames_per_query: int | None = None,
    ) -> None:
        selected_config = config or FolioConfig(segment_frames=int(segment_frames))
        if change_threshold is not None:
            selected_config = replace(selected_config, change_threshold=float(change_threshold))
        if max_cache_frames_per_query is not None:
            selected_config = replace(
                selected_config,
                max_cache_frames_per_query=int(max_cache_frames_per_query),
            )
        selected_config.validate()
        self.generate = generate
        self.config = selected_config
        self.segment_frames = selected_config.segment_frames
        self.entities: dict[str, FolioEntity] = {}
        self.events: dict[str, FolioEvent] = {}
        self.evidence: dict[str, EvidenceFrame] = {}
        self.pending: list[tuple[float, Image.Image]] = []
        self.pending_focus_terms: dict[str, int] = {}
        self.committed_query_ids: set[str] = set()
        self.dialogue_history: list[DialogueTurn] = []
        self.query_traces: list[dict[str, Any]] = []
        self.last_time = -1.0
        self.segment_index = 0
        self.next_entity_index = 1
        self.next_event_index = 1
        self._new_frames_since_attempt = 0
        self.observed_frames = 0
        self.selected_frames = 0
        self.write_calls = 0
        self.write_errors = 0
        self.write_seconds = 0.0
        self.semantic_link_calls = 0
        self.semantic_link_errors = 0
        self.semantic_link_seconds = 0.0
        self.query_count = 0
        self.query_errors = 0
        self.cache_recovery_count = 0
        self.cache_decode_errors = 0
        self.dropped_frames = 0
        self.snapshot_errors = 0
        self.last_write_error: str | None = None
        self.last_snapshot_error: str | None = None

    def observe(self, timestamp: float, image: Image.Image) -> None:
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Frame timestamps must be finite and nonnegative")
        if timestamp <= self.last_time:
            raise ValueError("Frame timestamps must be strictly increasing")
        if image.mode != "RGB" or min(image.size) < 1:
            raise ValueError("Expected nonempty RGB frames")
        self.last_time = float(timestamp)
        self.observed_frames += 1
        self.pending.append((float(timestamp), image.copy()))
        self._new_frames_since_attempt += 1
        max_pending = 2 * self.segment_frames
        if len(self.pending) > max_pending:
            dropped = min(self.segment_frames, len(self.pending))
            self.pending = self.pending[dropped:]
            self.dropped_frames += dropped
        if self._new_frames_since_attempt >= self.segment_frames:
            self._new_frames_since_attempt = 0
            while len(self.pending) >= self.segment_frames:
                if not self._update():
                    break

    def _focus_level(self, score: float) -> str:
        if score >= self.config.focus_threshold:
            return "focus"
        if score >= self.config.support_threshold:
            return "support"
        if score >= self.config.context_threshold:
            return "context"
        return "drop"

    def _focus_risk(self) -> bool:
        return any(
            entity.focus_score >= self.config.focus_threshold
            and _normalize(entity.category) not in BACKGROUND_CATEGORIES
            for entity in self.entities.values()
        )

    def _entity_catalog(self, entities: dict[str, FolioEntity] | None = None) -> str:
        source = self.entities if entities is None else entities
        lines: list[str] = []
        ordered = sorted(
            source.values(),
            key=lambda item: (-item.focus_score, -item.last_seen, item.id),
        )
        for entity in ordered[:64]:
            latest = entity.observations[-1] if entity.observations else None
            lines.append(
                " | ".join(
                    [
                        entity.id,
                        entity.canonical_name,
                        f"aliases={','.join(entity.aliases) or '-'}",
                        f"category={entity.category or '-'}",
                        f"attributes={','.join(entity.attributes) or '-'}",
                        f"latest_location={latest.location if latest else '-'}",
                        f"latest_state={latest.state if latest else '-'}",
                        f"last_seen={entity.last_seen:.1f}s",
                        f"focus={entity.focus_score:.2f}/{self._focus_level(entity.focus_score)}",
                    ]
                )
            )
        return "\n".join(lines) if lines else "(empty; discover visible entities)"

    def _writing_budget(self) -> str:
        detailed: list[FolioEntity] = []
        compact: list[FolioEntity] = []
        ordered = sorted(
            self.entities.values(),
            key=lambda item: (-item.focus_score, -item.last_seen, item.id),
        )
        for entity in ordered:
            level = self._focus_level(entity.focus_score)
            if level == "focus" and len(detailed) < 4:
                detailed.append(entity)
            elif level in {"support", "context"} and len(compact) < 8:
                compact.append(entity)
        detailed_text = ", ".join(f"{item.id}:{item.canonical_name}" for item in detailed) or "none"
        compact_text = ", ".join(f"{item.id}:{item.canonical_name}" for item in compact) or "none"
        ordered_events = sorted(
            self.events.values(),
            key=lambda item: (-item.focus_score, -item.end_time, item.id),
        )
        detailed_actions = [
            item
            for item in ordered_events
            if self._focus_level(item.focus_score) == "focus"
        ][:4]
        compact_actions = [
            item
            for item in ordered_events
            if self._focus_level(item.focus_score) in {"support", "context"}
        ][:8]
        detailed_action_text = ", ".join(
            f"{item.id}:{item.event_type}" for item in detailed_actions
        ) or "none"
        compact_action_text = ", ".join(
            f"{item.id}:{item.event_type}" for item in compact_actions
        ) or "none"
        pending = ", ".join(sorted(self.pending_focus_terms)) or "none"
        return (
            f"DETAILED existing entities: {detailed_text}\n"
            f"COMPACT existing entities: {compact_text}\n"
            f"DETAILED existing actions: {detailed_action_text}\n"
            f"COMPACT existing actions: {compact_action_text}\n"
            f"Unresolved prior-query terms to notice if visible: {pending}"
        )

    def _update(self) -> bool:
        frames = list(self.pending[: self.segment_frames])
        if not frames:
            return True
        started = time.perf_counter()
        self.write_calls += 1
        try:
            selection = select_keyframes(
                frames,
                change_threshold=self.config.change_threshold,
                focus_risk=self._focus_risk(),
            )
            selected = [frames[index] for index in selection.indices]
            frame_table = ", ".join(
                f"{local_index}={timestamp:.3f}s(segment_index={segment_index})"
                for local_index, (segment_index, (timestamp, _)) in enumerate(
                    zip(selection.indices, selected)
                )
            )
            prompt = _render(
                WRITER_PROMPT,
                {
                    "segment_range": f"{frames[0][0]:.3f}s - {frames[-1][0]:.3f}s",
                    "frame_table": frame_table,
                    "writing_budget": self._writing_budget(),
                    "entity_catalog": self._entity_catalog(),
                    "max_objects": self.config.top_k_objects,
                    "max_events": self.config.top_m_events,
                },
            )
            if self.last_write_error:
                prompt += "\n\nPrevious rejected transaction: " + self.last_write_error[:500]
            output = self.generate([image for _, image in selected], prompt, WRITER_TOKENS)
            detailed, compact, events = _parse_writer_output(
                output,
                frame_count=len(selected),
                max_objects=self.config.top_k_objects,
                max_events=self.config.top_m_events,
            )
            self._commit_segment(
                frames=frames,
                selected=selected,
                selected_segment_indices=selection.indices,
                max_change=selection.max_change,
                object_records=[*detailed, *compact],
                event_records=events,
            )
            del self.pending[: len(frames)]
            self.selected_frames += len(selected)
            self.last_write_error = None
            return True
        except Exception as exc:
            self.write_errors += 1
            self.last_write_error = str(exc)
            return False
        finally:
            self.write_seconds += time.perf_counter() - started

    @staticmethod
    def _latest(entity: FolioEntity) -> FolioObservation | None:
        return entity.observations[-1] if entity.observations else None

    def _match_entity(
        self,
        record: dict[str, Any],
        entities: dict[str, FolioEntity],
        claimed: set[str],
    ) -> str | None:
        incoming_names = {_normalize(record["name"]), *(_normalize(item) for item in record["aliases"])}
        incoming_names.discard("")
        incoming_category = _normalize(record["category"])
        incoming_distinctive = _distinctive(
            [record["name"], *record["aliases"], *record["attributes"]]
        )
        candidates: list[tuple[float, float, str]] = []
        for entity in entities.values():
            names = {_normalize(entity.canonical_name), *(_normalize(item) for item in entity.aliases)}
            names.discard("")
            category = _normalize(entity.category)
            old_distinctive = _distinctive(
                [entity.canonical_name, *entity.aliases, *entity.attributes]
            )
            score = 0.0
            winner = ""
            if _normalize(record["name"]) == _normalize(entity.canonical_name):
                score, winner = 1.0, "name"
            elif incoming_names & names:
                score, winner = 0.95, "alias"
            elif _head(record["name"]) and _head(record["name"]) == _head(entity.canonical_name):
                score, winner = 0.80, "head"
            elif any(left in right or right in left for left in incoming_names for right in names):
                score, winner = 0.75, "substring"
            if incoming_category and category and incoming_category == category and score < 0.55:
                score, winner = 0.55, "category"
            if score < 0.80:
                continue
            if (
                winner in {"head", "substring", "category"}
                and incoming_distinctive
                and old_distinctive
                and not (incoming_distinctive & old_distinctive)
            ):
                continue
            if _head(record["name"]) in GENERIC_NAMES and score < 0.95 and not incoming_distinctive:
                continue
            if entity.id in claimed and score < 0.95:
                continue
            temporal_consistency = 1.0 / (1.0 + max(0, self.segment_index - entity.last_seen_segment))
            candidates.append((score, temporal_consistency, entity.id))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
        return candidates[0][2]

    @staticmethod
    def _same_observation(left: FolioObservation, right: FolioObservation) -> bool:
        return (
            _normalize(left.location) == _normalize(right.location)
            and _normalize(left.holder) == _normalize(right.holder)
            and _normalize(left.state) == _normalize(right.state)
            and _normalize(left.visible_text) == _normalize(right.visible_text)
            and _normalize(left.evidence_summary) == _normalize(right.evidence_summary)
            and left.relations == right.relations
            and not right.interactions
            and _normalize(right.state_change) in {"", "stable", "unchanged", "unknown"}
        )

    @staticmethod
    def _resolve_name(name: str, entities: dict[str, FolioEntity]) -> str | None:
        normalized = _normalize(name)
        if not normalized:
            return None
        exact: list[str] = []
        head_matches: list[str] = []
        for entity in entities.values():
            names = {_normalize(entity.canonical_name), *(_normalize(item) for item in entity.aliases)}
            if normalized in names:
                exact.append(entity.id)
            elif _head(normalized) and _head(normalized) == _head(entity.canonical_name):
                head_matches.append(entity.id)
        if len(exact) == 1:
            return exact[0]
        if not exact and len(head_matches) == 1 and _head(normalized) not in GENERIC_NAMES:
            return head_matches[0]
        return None

    def _commit_segment(
        self,
        *,
        frames: list[tuple[float, Image.Image]],
        selected: list[tuple[float, Image.Image]],
        selected_segment_indices: tuple[int, ...],
        max_change: float,
        object_records: list[dict[str, Any]],
        event_records: list[dict[str, Any]],
    ) -> None:
        # ponytail: copy-on-write keeps rollback simple; use staged deltas if
        # full-history copies become costly on long streams.
        entities = copy.deepcopy(self.entities)
        events = copy.deepcopy(self.events)
        evidence = copy.deepcopy(self.evidence)
        pending_terms = dict(self.pending_focus_terms)
        segment_id = self.segment_index + 1
        next_entity_index = self.next_entity_index
        next_event_index = self.next_event_index

        def new_entity_id() -> str:
            nonlocal next_entity_index
            identifier = f"entity-{next_entity_index:04d}"
            next_entity_index += 1
            return identifier

        def new_event_id() -> str:
            nonlocal next_event_index
            identifier = f"event-{next_event_index:04d}"
            next_event_index += 1
            return identifier

        evidence_ids: list[str] = []
        for local_index, ((timestamp, image), segment_frame_index) in enumerate(
            zip(selected, selected_segment_indices)
        ):
            identifier = f"segment-{segment_id:05d}-frame-{local_index:02d}"
            evidence_ids.append(identifier)
            evidence[identifier] = EvidenceFrame(
                id=identifier,
                segment_id=segment_id,
                timestamp=float(timestamp),
                segment_frame_index=int(segment_frame_index),
                change_score=float(max_change),
                jpeg=_encode_jpeg(image),
            )

        for entity in entities.values():
            entity.focus_score = max(0.0, min(1.0, entity.focus_score * self.config.focus_decay))
        for event in events.values():
            event.focus_score = max(
                0.0, min(1.0, event.focus_score * self.config.focus_decay)
            )

        claimed: set[str] = set()
        observed_flags: dict[str, dict[str, bool]] = {}
        for record in object_records:
            entity_id = self._match_entity(record, entities, claimed)
            is_new = entity_id is None
            if is_new:
                entity_id = new_entity_id()
                entity = FolioEntity(
                    id=entity_id,
                    canonical_name=record["name"],
                    aliases=list(record["aliases"]),
                    category=record["category"],
                    attributes=list(record["attributes"]),
                    first_seen=float(frames[0][0]),
                    last_seen=float(frames[-1][0]),
                    last_seen_segment=segment_id,
                )
                entities[entity_id] = entity
                previous = None
                previous_segment = -1
            else:
                entity = entities[entity_id]
                previous = self._latest(entity)
                previous_segment = entity.last_seen_segment
            claimed.add(entity_id)

            for alias in [record["name"], *record["aliases"]]:
                if (
                    _normalize(alias) != _normalize(entity.canonical_name)
                    and alias not in entity.aliases
                ):
                    entity.aliases.append(alias)
            if not entity.category and record["category"]:
                entity.category = record["category"]
            for attribute in record["attributes"]:
                if attribute not in entity.attributes:
                    entity.attributes.append(attribute)

            linked = [evidence_ids[index] for index in record["evidence_frames"]]
            observation = FolioObservation(
                segment_id=segment_id,
                start_time=float(frames[0][0]),
                end_time=float(frames[-1][0]),
                detail=record["detail"],
                location=record["location"],
                holder=record["holder"],
                state=record["state"],
                relations=copy.deepcopy(record["relations"]),
                interactions=copy.deepcopy(record["interactions"]),
                state_change=record["state_change"],
                visible_text=record["visible_text"],
                evidence_summary=record["evidence_summary"],
                confidence=record["confidence"],
                evidence_frame_ids=linked,
            )
            moved = bool(previous and _normalize(previous.location) != _normalize(observation.location))
            state_changed = bool(
                previous and _normalize(previous.state) != _normalize(observation.state)
            ) or _normalize(observation.state_change) not in {"", "stable", "unchanged", "unknown"}
            if previous and previous_segment == segment_id:
                previous.detail = (
                    "detailed"
                    if "detailed" in {previous.detail, observation.detail}
                    else previous.detail
                )
                for field_name in (
                    "location", "holder", "state", "state_change", "visible_text",
                    "evidence_summary",
                ):
                    if not getattr(previous, field_name) and getattr(observation, field_name):
                        setattr(previous, field_name, getattr(observation, field_name))
                for relation in observation.relations:
                    if relation not in previous.relations:
                        previous.relations.append(relation)
                for interaction in observation.interactions:
                    if interaction not in previous.interactions:
                        previous.interactions.append(interaction)
                previous.confidence = max(previous.confidence, observation.confidence)
                for frame_id in linked:
                    if frame_id not in previous.evidence_frame_ids:
                        previous.evidence_frame_ids.append(frame_id)
            elif previous and previous_segment == segment_id - 1 and self._same_observation(previous, observation):
                previous.end_time = observation.end_time
                for frame_id in linked:
                    if frame_id not in previous.evidence_frame_ids:
                        previous.evidence_frame_ids.append(frame_id)
            else:
                entity.observations.append(observation)
            entity.last_seen = float(frames[-1][0])
            entity.last_seen_segment = segment_id
            reappeared = bool(not is_new and previous_segment < segment_id - 1)
            prior_flags = observed_flags.get(entity_id, {})
            observed_flags[entity_id] = {
                "new": bool(prior_flags.get("new", False) or is_new),
                "reappeared": bool(prior_flags.get("reappeared", False) or reappeared),
                "persistent": bool(
                    prior_flags.get("persistent", False)
                    or (not is_new and previous_segment == segment_id - 1)
                ),
                "moved": bool(prior_flags.get("moved", False) or moved),
                "state_changed": bool(
                    prior_flags.get("state_changed", False) or state_changed
                ),
                "event": bool(prior_flags.get("event", False)),
            }
            for frame_id in linked:
                if entity_id not in evidence[frame_id].entity_ids:
                    evidence[frame_id].entity_ids.append(entity_id)

        for raw in event_records:
            event_id = new_event_id()
            participant_ids = [
                resolved
                for name in raw["participants"]
                if (resolved := self._resolve_name(name, entities)) is not None
            ]
            changed_ids = [
                resolved
                for name in raw["changed_objects"]
                if (resolved := self._resolve_name(name, entities)) is not None
            ]
            linked = [evidence_ids[index] for index in raw["evidence_frames"]]
            if not linked:
                for entity_id in participant_ids:
                    entity = entities[entity_id]
                    latest = self._latest(entity)
                    if latest and latest.segment_id == segment_id:
                        linked.extend(latest.evidence_frame_ids)
                linked = list(dict.fromkeys(linked))
            event = FolioEvent(
                id=event_id,
                segment_id=segment_id,
                start_time=float(frames[0][0]),
                end_time=float(frames[-1][0]),
                event_type=raw["event_type"],
                summary=raw["summary"],
                participants=raw["participants"],
                participant_ids=list(dict.fromkeys(participant_ids)),
                changed_objects=raw["changed_objects"],
                changed_entity_ids=list(dict.fromkeys(changed_ids)),
                confidence=raw["confidence"],
                evidence_frame_ids=linked,
                focus_score=0.50,
            )
            events[event_id] = event
            for entity_id in dict.fromkeys([*participant_ids, *changed_ids]):
                entity = entities[entity_id]
                if event_id not in entity.event_ids:
                    entity.event_ids.append(event_id)
                if entity_id in observed_flags:
                    observed_flags[entity_id]["event"] = True
            for frame_id in linked:
                if frame_id not in evidence:
                    raise ValueError("Event linked an unknown evidence frame")
                if event_id not in evidence[frame_id].event_ids:
                    evidence[frame_id].event_ids.append(event_id)

        pending_entity_matches: dict[str, list[str]] = {}
        for term in pending_terms:
            term_tokens = _expand_synonyms(_tokens(term))
            if not term_tokens:
                continue
            candidates: list[str] = []
            for entity in entities.values():
                name_tokens = _expand_synonyms(
                    _tokens(" ".join([entity.canonical_name, *entity.aliases]))
                )
                if term_tokens & name_tokens:
                    candidates.append(entity.id)
            if len(candidates) == 1:
                pending_entity_matches.setdefault(candidates[0], []).append(term)

        observed_ids = set(observed_flags)
        for entity in entities.values():
            if entity.id not in observed_ids:
                gap = segment_id - entity.last_seen_segment
                entity.focus_score = max(0.0, entity.focus_score - (0.15 if gap >= 3 else 0.08))
                continue
            flags = observed_flags[entity.id]
            delta = 0.10
            if flags["new"]:
                delta += 0.35
            if flags["reappeared"]:
                delta += 0.25
            if flags["persistent"]:
                delta += 0.05
            if flags["event"]:
                delta += 0.20
            if flags["moved"]:
                delta += 0.20
            if flags["state_changed"]:
                delta += 0.25
            if _normalize(entity.category) in MANIPULABLE_TERMS or _head(entity.canonical_name) in MANIPULABLE_TERMS:
                delta += 0.10
            if (
                _normalize(entity.category) in BACKGROUND_CATEGORIES
                and not flags["moved"]
                and not flags["state_changed"]
                and not flags["event"]
            ):
                delta -= 0.10
            entity.focus_score = max(0.0, min(1.0, entity.focus_score + delta))

            matched_pending = pending_entity_matches.get(entity.id, [])
            if matched_pending:
                entity.focus_score = min(1.0, entity.focus_score + 0.20)
                for term in matched_pending:
                    normalized_names = {
                        _normalize(entity.canonical_name),
                        *(_normalize(item) for item in entity.aliases),
                    }
                    if (
                        term not in GENERIC_NAMES
                        and term not in DISTINCTIVE_PREFIXES
                        and _normalize(term) not in normalized_names
                    ):
                        entity.aliases.append(term)
                    pending_terms.pop(term, None)

        current_events = [
            item for item in events.values() if item.segment_id == segment_id
        ]
        for term in list(pending_terms):
            term_tokens = _expand_synonyms(_tokens(term))
            matched_events = [
                item
                for item in current_events
                if term_tokens
                & _expand_synonyms(
                    _tokens(
                        " ".join(
                            [
                                item.event_type,
                                item.summary,
                                *item.participants,
                                *item.changed_objects,
                            ]
                        )
                    )
                )
            ]
            if not matched_events:
                continue
            for item in matched_events:
                item.focus_score = min(1.0, item.focus_score + 0.20)
                for entity_id in [*item.participant_ids, *item.changed_entity_ids]:
                    if entity_id in entities:
                        entities[entity_id].focus_score = min(
                            1.0, entities[entity_id].focus_score + 0.20
                        )
            pending_terms.pop(term, None)

        self.entities = entities
        self.events = events
        self.evidence = evidence
        self.pending_focus_terms = pending_terms
        self.segment_index = segment_id
        self.next_entity_index = next_entity_index
        self.next_event_index = next_event_index

    def _parse_intent(
        self,
        question: str,
        options: Sequence[str],
        query_type_hint: str | None = None,
    ) -> QueryIntent:
        question_norm = _normalize(question)
        option_text = " ".join(str(option) for option in options)
        query_tokens = _tokens(question)
        option_tokens = _tokens(option_text)
        raw_tokens = TOKEN_RE.findall(question_norm)
        temporal_words = {"after", "before", "earlier", "first", "previously"}
        is_historical = bool(set(raw_tokens) & temporal_words)
        if "where" in raw_tokens:
            is_past_location = bool(
                is_historical
                or "was" in raw_tokens
                or "did" in raw_tokens
                or set(raw_tokens) & ACTION_TERMS
            )
            query_type = "historical-location" if is_past_location else "current-location"
            scope = "historical" if is_past_location else "current"
        elif query_tokens & {"color", "colour", "number", "text", "wearing", "shape", "attribute"}:
            query_type, scope = "attribute", "historical" if is_historical else "mixed"
        elif query_tokens & {"left", "right", "inside", "beside", "near", "under", "above", "behind", "front"}:
            query_type, scope = "spatial", "historical" if is_historical else "mixed"
        elif set(raw_tokens) & ACTION_TERMS:
            query_type, scope = "interaction", "historical"
        elif question_norm.startswith(("is ", "was ", "did ", "does ", "has ", "have ")):
            query_type, scope = "yes/no", "mixed"
        else:
            query_type, scope = "concept", "mixed"
        normalized_hint = _normalize(query_type_hint or "").replace(" ", "-")
        hint_aliases = {
            "hld": "hallucination-detection",
            "hallucination": "hallucination-detection",
            "hallucination-detection": "hallucination-detection",
            "current-location": "current-location",
            "historical-location": "historical-location",
            "interaction": "interaction",
            "attribute": "attribute",
            "spatial": "spatial",
            "yes-no": "yes/no",
            "concept": "concept",
        }
        explicit_absence_query = any(
            phrase in question_norm
            for phrase in (
                "never appear", "never appeared", "not appear", "not visible",
                "was not seen", "were not seen", "did not appear",
            )
        )
        if normalized_hint in hint_aliases:
            query_type = hint_aliases[normalized_hint]
            if query_type == "current-location":
                scope = "current"
            elif query_type in {"historical-location", "interaction"}:
                scope = "historical"
        elif explicit_absence_query:
            query_type = "hallucination-detection"
        option_norm = _normalize(option_text)
        coverage_required = query_type == "hallucination-detection" or any(
            term in option_norm
            for term in ("cannot determine", "unable to answer", "not visible", "never appeared")
        )
        if "before" in question_norm:
            temporal_relation = "before"
        elif "after" in question_norm:
            temporal_relation = "after"
        elif any(term in question_norm for term in ("first", "initially", "at the beginning")):
            temporal_relation = "first"
        elif any(term in question_norm for term in ("last", "latest", "most recent")):
            temporal_relation = "last"
        else:
            temporal_relation = "none"

        verb_terms = tuple(sorted(set(raw_tokens) & ACTION_TERMS))
        target_terms = set(query_tokens) - set(verb_terms) - QUERY_CONTROL_TERMS
        anchor_terms: set[str] = set()
        if temporal_relation in {"before", "after"}:
            marker = temporal_relation
            try:
                marker_index = raw_tokens.index(marker)
            except ValueError:
                marker_index = -1
            if marker_index >= 0:
                anchor_terms = {
                    token
                    for token in raw_tokens[marker_index + 1 :]
                    if token not in STOPWORDS
                    and token not in ACTION_TERMS
                    and token not in QUERY_CONTROL_TERMS
                    and token not in REFERENCE_TERMS
                }
                target_terms -= anchor_terms

        answer_slot = {
            "current-location": "location",
            "historical-location": "location",
            "interaction": "event",
            "attribute": "attribute",
            "spatial": "relation",
            "yes/no": "boolean",
            "hallucination-detection": "coverage",
            "concept": "concept",
        }[query_type]
        history_entity_ids: list[str] = []
        history_event_ids: list[str] = []
        if set(raw_tokens) & REFERENCE_TERMS:
            for turn in reversed(self.dialogue_history[-8:]):
                for entity_id in turn.selected_entity_ids:
                    if entity_id in self.entities and entity_id not in history_entity_ids:
                        history_entity_ids.append(entity_id)
                for event_id in turn.selected_event_ids:
                    if event_id in self.events and event_id not in history_event_ids:
                        history_event_ids.append(event_id)
                if history_entity_ids or history_event_ids:
                    break
        return QueryIntent(
            query_type=query_type,
            normalized=question_norm,
            tokens=tuple(sorted(query_tokens)),
            target_terms=tuple(sorted(target_terms)),
            anchor_terms=tuple(sorted(anchor_terms)),
            verb_terms=verb_terms,
            option_tokens=tuple(sorted(option_tokens)),
            key_option_terms=tuple(sorted(option_tokens - QUERY_CONTROL_TERMS)),
            temporal_scope=scope,
            temporal_relation=temporal_relation,
            answer_slot=answer_slot,
            coverage_required=coverage_required,
            history_entity_ids=tuple(history_entity_ids),
            history_event_ids=tuple(history_event_ids),
        )

    def _entity_score(
        self,
        entity: FolioEntity,
        intent: QueryIntent,
    ) -> float:
        target_terms = _expand_synonyms(intent.target_terms)
        anchor_terms = _expand_synonyms(intent.anchor_terms)
        option_tokens = set(intent.key_option_terms)
        canonical = _normalize(entity.canonical_name)
        aliases = [_normalize(item) for item in entity.aliases]
        category = _normalize(entity.category)
        identity_tokens = _expand_synonyms(
            _tokens(" ".join([entity.canonical_name, *entity.aliases, entity.category]))
        )
        attribute_tokens = _tokens(" ".join(entity.attributes))
        head = _head(entity.canonical_name)
        score = 0.0
        if canonical and _phrase_in(canonical, intent.normalized):
            score += 8.0
        for alias in aliases:
            if alias and _phrase_in(alias, intent.normalized):
                score = max(score, 7.0)
        if category and _phrase_in(category, intent.normalized):
            score += 3.0
        score += 3.0 * len(target_terms & identity_tokens)
        score += 2.0 * len(anchor_terms & identity_tokens)
        if head and head in target_terms:
            score += 4.0
        score += 2.0 * len(target_terms & attribute_tokens)
        score += 1.5 * len(_expand_synonyms(option_tokens) & identity_tokens)
        event_text = " ".join(
            events_text
            for event_id in entity.event_ids
            if (events_text := (
                f"{self.events[event_id].event_type} {self.events[event_id].summary} "
                f"{' '.join(self.events[event_id].participants)}"
                if event_id in self.events
                else ""
            ))
        )
        event_tokens = _tokens(event_text)
        score += 1.75 * len(set(intent.verb_terms) & event_tokens)
        if entity.id in intent.history_entity_ids:
            score += 6.0
        if _normalize(entity.category) in BACKGROUND_CATEGORIES:
            score -= 0.5
        return max(0.0, score)

    def _event_score(self, event: FolioEvent, intent: QueryIntent) -> float:
        text = " ".join([event.event_type, event.summary, *event.participants, *event.changed_objects])
        event_tokens = _tokens(text)
        return (
            3.0 * len(set(intent.verb_terms) & event_tokens)
            + 2.0 * len(set(intent.target_terms) & event_tokens)
            + 1.0 * len(set(intent.anchor_terms) & event_tokens)
            + 0.25 * len(set(intent.key_option_terms) & event_tokens)
        )

    def _semantic_link(
        self,
        question: str,
        options: Sequence[str],
    ) -> tuple[list[str], list[str], str, str]:
        if not self.config.semantic_link or (not self.entities and not self.events):
            return [], [], "disabled", ""
        entity_lines = [
            f"{item.id} | {item.canonical_name} | aliases={','.join(item.aliases[:8])} | "
            f"category={item.category} | attributes={','.join(item.attributes[:12])}"
            for item in sorted(
                self.entities.values(),
                key=lambda entity: (-entity.focus_score, -entity.last_seen, entity.id),
            )
        ]
        event_lines = [
            f"{item.id} | {item.event_type} | {item.summary} | "
            f"participants={','.join(item.participants)}"
            for item in sorted(
                self.events.values(), key=lambda event: (-event.end_time, event.id)
            )
        ]
        entity_catalog = "\n".join(entity_lines) or "(empty)"
        event_catalog = "\n".join(event_lines) or "(empty)"
        formatted_options = _format_options(options)
        prompt = _render(
            SEMLINK_PROMPT,
            {
                "question": question,
                "options": formatted_options,
                "options_text": formatted_options,
                "entity_catalog": entity_catalog,
                "object_list": entity_catalog,
                "event_catalog": event_catalog,
                "action_list": event_catalog,
            },
        )
        self.semantic_link_calls += 1
        started = time.perf_counter()
        try:
            value = _json_object(self.generate([], prompt, SEMLINK_TOKENS))
            raw_entities = value.get(
                "relevant_entity_ids", value.get("relevant_object_ids", [])
            )
            raw_events = value.get("relevant_event_ids", [])
            entity_ids = [
                item
                for item in _string_list(raw_entities, limit=7)
                if item in self.entities
            ]
            event_ids = [
                item
                for item in _string_list(raw_events, limit=7)
                if item in self.events
            ]
            entity_ids = list(dict.fromkeys(entity_ids))
            event_ids = list(dict.fromkeys(event_ids))
            suggested = _clean_text(value.get("suggested_option"), limit=1).upper()
            if suggested not in {"A", "B", "C", "D"}:
                suggested = ""
            if not entity_ids and not event_ids:
                return [], [], "empty", suggested
            return entity_ids, event_ids, "ok", suggested
        except Exception as exc:
            self.semantic_link_errors += 1
            return [], [], f"error:{type(exc).__name__}", ""
        finally:
            self.semantic_link_seconds += time.perf_counter() - started

    @staticmethod
    def _observation_line(observation: FolioObservation) -> str:
        fields = [
            f"[{observation.start_time:.1f}-{observation.end_time:.1f}s]",
            f"location={observation.location or 'unknown'}",
            f"state={observation.state or 'unknown'}",
        ]
        if observation.holder:
            fields.append(f"holder={observation.holder}")
        if observation.state_change:
            fields.append(f"change={observation.state_change}")
        if observation.visible_text:
            fields.append(f"visible_text={observation.visible_text}")
        if observation.relations:
            fields.append(
                "relations="
                + "; ".join(
                    f"{item.get('relation', '')} {item.get('target', '')}".strip()
                    for item in observation.relations
                )
            )
        if observation.interactions:
            fields.append(
                "interactions="
                + "; ".join(
                    f"{item.get('type', '')} {item.get('with', '')}: {item.get('summary', '')}".strip()
                    for item in observation.interactions
                )
            )
        if observation.evidence_summary:
            fields.append(f"evidence={observation.evidence_summary}")
        fields.append(f"confidence={observation.confidence:.2f}")
        return " | ".join(fields)

    def _assemble_memory(
        self,
        intent: QueryIntent,
        entity_ids: Sequence[str],
        event_ids: Sequence[str],
        question: str,
        options: Sequence[str],
        *,
        concept: bool,
        suggested_option: str,
        entity_scores: dict[str, float],
    ) -> tuple[str, list[str], list[SelectedRecord], list[SelectedRecord]]:
        overview_entities = sorted(
            self.entities.values(),
            key=lambda item: (-item.focus_score, -item.last_seen, item.id),
        )[:6]
        overview_parts: list[str] = []
        for entity in overview_entities:
            latest = entity.observations[-1] if entity.observations else None
            details = [entity.canonical_name]
            if entity.category:
                details.append(entity.category)
            if latest is not None and latest.location:
                details.append(f"at {latest.location}")
            if latest is not None and latest.state:
                details.append(latest.state)
            overview_parts.append(" / ".join(details)[:240])
        header_lines = [
            f"QUERY TYPE: {intent.query_type}",
            f"TEMPORAL SCOPE: {intent.temporal_scope}",
            f"ANSWER SLOT: {intent.answer_slot}",
            f"TARGET COVERAGE: {'FOUND' if entity_ids or event_ids else 'TARGET NOT FOUND'}",
            f"QUESTION CONTEXT: {question}",
            "OPTION CHECK: " + " | ".join(
                f"{chr(65 + index)}={option}" for index, option in enumerate(options)
            ),
            "SCENE OVERVIEW: " + ("; ".join(overview_parts) or "(no stored entities)"),
        ]
        if concept:
            header_lines.append("## (!) CONCEPT QUESTION -- SPECIAL HANDLING REQUIRED")
            header_lines.append("Semantic catalog linking was used to select existing memory records.")
            if suggested_option:
                header_lines.append(f"LLM-SUGGESTED OPTION: {suggested_option}")
        candidate_records: list[SelectedRecord] = []
        packable_records: list[SelectedRecord] = []
        entity_headers: dict[str, str] = {}
        explicit_events = [item for item in event_ids if item in self.events]
        related_events: list[str] = []
        for entity_id in entity_ids:
            entity = self.entities[entity_id]
            entity_headers[entity_id] = (
                f"\nENTITY {entity.id}: {entity.canonical_name} | aliases={','.join(entity.aliases[:16]) or '-'} "
                f"| category={entity.category or '-'} | attributes={','.join(entity.attributes[:32]) or '-'} "
                f"| observed={entity.first_seen:.1f}-{entity.last_seen:.1f}s"
            )
            all_indexed_observations = list(enumerate(entity.observations))
            indexed_observations = list(all_indexed_observations)
            if intent.query_type == "current-location":
                indexed_observations = [
                    item for item in indexed_observations if item[1].location
                ][-1:]
            elif intent.query_type == "historical-location":
                indexed_observations = [
                    item for item in indexed_observations if item[1].location
                ]
            elif intent.query_type == "attribute":
                indexed_observations = [
                    item
                    for item in indexed_observations
                    if entity.attributes
                    or item[1].visible_text
                    or item[1].state
                    or item[1].evidence_summary
                ][-6:]
            elif intent.query_type == "spatial":
                indexed_observations = [
                    item for item in indexed_observations if item[1].relations
                ][-8:]
            elif intent.query_type in {"yes/no", "hallucination-detection"}:
                if indexed_observations:
                    indexed_observations = list(
                        dict.fromkeys(
                            [indexed_observations[0][0], indexed_observations[-1][0]]
                        )
                    )
                    indexed_observations = [
                        (index, entity.observations[index])
                        for index in indexed_observations
                    ]
            else:
                indexed_observations = indexed_observations[-6:]
            packable_indices = {index for index, _ in indexed_observations}
            for observation_index, observation in all_indexed_observations:
                observation_text = self._observation_line(observation)
                field_tokens = _tokens(observation_text)
                record_score = entity_scores.get(entity_id, 0.0)
                record_score += 1.5 * len(set(intent.target_terms) & field_tokens)
                record_score += 2.0 * len(set(intent.anchor_terms) & field_tokens)
                record_score += 0.5 * len(set(intent.key_option_terms) & field_tokens)
                record_score += 2.0 * len(set(intent.verb_terms) & field_tokens)
                if intent.query_type in {"current-location", "historical-location"}:
                    grounding_text = observation.location
                elif intent.query_type == "attribute":
                    grounding_text = " ".join(
                        [
                            *entity.attributes,
                            observation.visible_text,
                            observation.state,
                            observation.evidence_summary,
                        ]
                    )
                elif intent.query_type == "spatial":
                    grounding_text = " ".join(
                        item.get("relation", "") + " " + item.get("target", "")
                        for item in observation.relations
                    )
                elif intent.query_type == "interaction":
                    grounding_text = " ".join(
                        item.get("type", "") + " " + item.get("summary", "")
                        for item in observation.interactions
                    )
                elif intent.query_type in {"yes/no", "hallucination-detection"}:
                    grounding_text = entity.canonical_name
                else:
                    grounding_text = " ".join(
                        [
                            observation.state,
                            observation.visible_text,
                            observation.evidence_summary,
                        ]
                    )
                grounding_tokens = _tokens(grounding_text)
                grounding_score = 0.0
                if grounding_text.strip():
                    grounding_score = (
                        8.0
                        if intent.query_type in {"yes/no", "hallucination-detection"}
                        else 6.0
                    )
                    grounding_score += 2.0 * len(
                        set(intent.tokens) & grounding_tokens
                    )
                    grounding_score += 2.0 * len(
                        set(intent.key_option_terms) & grounding_tokens
                    )
                record = SelectedRecord(
                    kind="observation",
                    identifier=f"{entity_id}:observation:{observation_index}",
                    entity_id=entity_id,
                    observation_index=observation_index,
                    event_id="",
                    start_time=observation.start_time,
                    end_time=observation.end_time,
                    score=record_score,
                    grounding_score=grounding_score,
                    confidence=observation.confidence,
                    evidence_frame_ids=tuple(observation.evidence_frame_ids),
                    field_text=observation_text,
                )
                candidate_records.append(record)
                if observation_index in packable_indices:
                    packable_records.append(record)
            for related_event_id in entity.event_ids:
                if related_event_id in self.events and related_event_id not in related_events:
                    related_events.append(related_event_id)

        related_events.sort(
            key=lambda item: (
                -self._event_score(self.events[item], intent),
                -self.events[item].end_time,
                item,
            )
        )
        selected_events = list(dict.fromkeys(explicit_events))
        for event_id in related_events:
            if event_id not in selected_events:
                selected_events.append(event_id)
        for event_id in selected_events:
            event = self.events[event_id]
            event_text = (
                f"EVENT {event.id} [{event.start_time:.1f}-{event.end_time:.1f}s] "
                f"{event.event_type}: {event.summary} | participants={','.join(event.participants) or '-'} "
                f"| confidence={event.confidence:.2f}"
            )
            event_base = max(
                [
                    entity_scores.get(entity_id, 0.0)
                    for entity_id in [*event.participant_ids, *event.changed_entity_ids]
                ]
                or [0.0]
            )
            event_tokens = _tokens(event_text)
            event_grounding = 0.0
            if intent.query_type in {"interaction", "yes/no", "hallucination-detection"}:
                event_grounding = 6.0
                event_grounding += 2.0 * len(set(intent.verb_terms) & event_tokens)
                event_grounding += 2.0 * len(set(intent.key_option_terms) & event_tokens)
            record = SelectedRecord(
                kind="event",
                identifier=event.id,
                entity_id="",
                observation_index=None,
                event_id=event.id,
                start_time=event.start_time,
                end_time=event.end_time,
                score=event_base + self._event_score(event, intent),
                grounding_score=event_grounding,
                confidence=event.confidence,
                evidence_frame_ids=tuple(event.evidence_frame_ids),
                field_text=event_text,
            )
            candidate_records.append(record)
            packable_records.append(record)

        ranked_records = self._rank_cache_records(packable_records, intent)
        first_by_entity: list[SelectedRecord] = []
        seen_entities: set[str] = set()
        for record in ranked_records:
            if record.kind == "observation" and record.entity_id not in seen_entities:
                first_by_entity.append(record)
                seen_entities.add(record.entity_id)
        first_record_ids = {record.identifier for record in first_by_entity}
        ranked_records = [
            *first_by_entity,
            *[record for record in ranked_records if record.identifier not in first_record_ids],
        ]

        base_text = "\n".join(header_lines)
        budget = self.config.memory_bytes
        if len(base_text.encode("utf-8")) > budget:
            text = base_text.encode("utf-8")[:budget].decode("utf-8", errors="ignore")
            return text, [], [], candidate_records

        chosen: list[SelectedRecord] = []
        activated_entities: set[str] = set()
        used_bytes = len(base_text.encode("utf-8"))
        for record in ranked_records:
            addition = "\n" + record.field_text
            if record.kind == "observation":
                addition = "\n  " + record.field_text
                if record.entity_id not in activated_entities:
                    addition = entity_headers[record.entity_id] + addition
            addition_bytes = len(addition.encode("utf-8"))
            if used_bytes + addition_bytes > budget:
                continue
            chosen.append(record)
            used_bytes += addition_bytes
            if record.kind == "observation":
                activated_entities.add(record.entity_id)

        chosen_ids = {record.identifier for record in chosen}
        packed_lines = list(header_lines)
        for entity_id in entity_ids:
            entity_records = sorted(
                (
                    record
                    for record in chosen
                    if record.kind == "observation" and record.entity_id == entity_id
                ),
                key=lambda item: (
                    item.observation_index if item.observation_index is not None else -1
                ),
            )
            if not entity_records:
                continue
            packed_lines.append(entity_headers[entity_id])
            packed_lines.extend("  " + record.field_text for record in entity_records)
        for event_id in selected_events:
            for record in chosen:
                if record.kind == "event" and record.event_id == event_id:
                    packed_lines.append(record.field_text)
                    break
        text = "\n".join(packed_lines)
        if len(text.encode("utf-8")) > budget:
            raise AssertionError("FOLIO memory packer exceeded its byte budget")
        packed_records = [
            record for record in packable_records if record.identifier in chosen_ids
        ]
        packed_event_ids = [
            event_id
            for event_id in selected_events
            if any(record.event_id == event_id for record in packed_records)
        ]
        return text, packed_event_ids, packed_records, candidate_records

    def _rank_cache_records(
        self,
        records: Sequence[SelectedRecord],
        intent: QueryIntent,
    ) -> list[SelectedRecord]:
        cue_priority: dict[str, float] = {}
        observations_by_entity: dict[str, list[SelectedRecord]] = {}
        for record in records:
            if record.kind == "observation":
                observations_by_entity.setdefault(record.entity_id, []).append(record)

        for entity_records in observations_by_entity.values():
            entity_records.sort(
                key=lambda item: (
                    item.observation_index if item.observation_index is not None else -1
                )
            )
            if intent.temporal_relation in {"before", "after"} and intent.anchor_terms:
                anchor_terms = set(intent.anchor_terms)
                anchor_positions = [
                    index
                    for index, record in enumerate(entity_records)
                    if anchor_terms & _tokens(record.field_text)
                ]
                for position in anchor_positions:
                    target_position = position + (-1 if intent.temporal_relation == "before" else 1)
                    if 0 <= target_position < len(entity_records):
                        cue_priority[entity_records[target_position].identifier] = 100.0
            elif intent.temporal_relation == "first" and entity_records:
                cue_priority[entity_records[0].identifier] = 100.0
            elif intent.temporal_relation == "last" and entity_records:
                cue_priority[entity_records[-1].identifier] = 100.0

        anchor_events = [
            self.events[event_id]
            for event_id in intent.history_event_ids
            if event_id in self.events
        ]
        if anchor_events and intent.temporal_relation in {"before", "after"}:
            if intent.temporal_relation == "after":
                boundary = max(item.end_time for item in anchor_events)
                candidates = [
                    item
                    for item in records
                    if item.identifier not in intent.history_event_ids
                    and item.start_time >= boundary
                ]
                if candidates:
                    nearest = min(candidates, key=lambda item: (item.start_time, item.identifier))
                    cue_priority[nearest.identifier] = 120.0
            else:
                boundary = min(item.start_time for item in anchor_events)
                candidates = [
                    item
                    for item in records
                    if item.identifier not in intent.history_event_ids
                    and item.end_time <= boundary
                ]
                if candidates:
                    nearest = max(candidates, key=lambda item: (item.end_time, item.identifier))
                    cue_priority[nearest.identifier] = 120.0

        event_records = [item for item in records if item.kind == "event"]
        if intent.temporal_relation == "first" and event_records:
            first_event = min(
                event_records, key=lambda item: (item.start_time, item.identifier)
            )
            cue_priority[first_event.identifier] = max(
                cue_priority.get(first_event.identifier, 0.0), 100.0
            )
        elif intent.temporal_relation == "last" and event_records:
            last_event = max(
                event_records, key=lambda item: (item.end_time, item.identifier)
            )
            cue_priority[last_event.identifier] = max(
                cue_priority.get(last_event.identifier, 0.0), 100.0
            )

        return sorted(
            records,
            key=lambda item: (
                -cue_priority.get(item.identifier, 0.0),
                -item.score,
                -item.confidence,
                -item.end_time,
                item.identifier,
            ),
        )

    def _recover_evidence(
        self,
        records: Sequence[SelectedRecord],
        *,
        query_time: float,
        recent_start: float,
        intent: QueryIntent,
    ) -> tuple[list[Image.Image], list[float], list[str]]:
        if not self.config.cache_replay or self.config.max_cache_frames_per_query <= 0:
            return [], [], []
        ordered_records = self._rank_cache_records(records, intent)
        chosen: list[EvidenceFrame] = []
        seen: set[str] = set()
        for record in ordered_records:
            linked = [
                self.evidence[frame_id]
                for frame_id in record.evidence_frame_ids
                if frame_id in self.evidence
                and frame_id not in seen
                and self.evidence[frame_id].timestamp <= query_time
                and self.evidence[frame_id].timestamp < recent_start
            ]
            linked.sort(key=lambda item: (item.timestamp, item.id), reverse=True)
            for frame in linked:
                seen.add(frame.id)
                chosen.append(frame)
                if len(chosen) >= self.config.max_cache_frames_per_query:
                    break
            if len(chosen) >= self.config.max_cache_frames_per_query:
                break
        chosen.sort(key=lambda item: (item.timestamp, item.id))
        images: list[Image.Image] = []
        timestamps: list[float] = []
        identifiers: list[str] = []
        for frame in chosen:
            try:
                images.append(frame.image())
                timestamps.append(frame.timestamp)
                identifiers.append(frame.id)
            except Exception:
                self.cache_decode_errors += 1
        if images:
            self.cache_recovery_count += 1
        return images, timestamps, identifiers

    def prepare_query(
        self,
        query_id: str,
        question: str,
        options: Sequence[str],
        *,
        query_time: float,
        recent_start: float,
        original_prompt: str | None = None,
        query_type_hint: str | None = None,
    ) -> FolioQueryPlan:
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError("query_id must be nonempty")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be nonempty")
        if not math.isfinite(query_time) or query_time < 0:
            raise ValueError("query_time must be finite and nonnegative")
        if query_time + 1e-6 < self.last_time:
            raise ValueError("query_time precedes the latest frame already committed to memory")
        intent = self._parse_intent(question, options, query_type_hint)
        scores = [
            (entity.id, self._entity_score(entity, intent))
            for entity in self.entities.values()
        ]
        scores = [item for item in scores if item[1] > 0.0]
        scores.sort(key=lambda item: (-item[1], item[0]))
        direct_scores = scores[: self.config.query_top_k]
        selected_entity_ids = [entity_id for entity_id, _ in direct_scores]
        scored_events = []
        for event in self.events.values():
            score = self._event_score(event, intent)
            if score > 0.0:
                scored_events.append((event.id, score))
        scored_events.sort(key=lambda item: (-item[1], item[0]))
        if intent.verb_terms and scored_events:
            best_event_score = scored_events[0][1]
            scored_events = [
                item for item in scored_events if item[1] >= best_event_score
            ]
        scored_event_ids = [event_id for event_id, _ in scored_events]
        direct_event_ids = list(intent.history_event_ids)
        if intent.history_event_ids and intent.temporal_relation in {"before", "after"}:
            for anchor_id in intent.history_event_ids:
                anchor = self.events.get(anchor_id)
                if anchor is None:
                    continue
                anchor_entities = set(
                    [*anchor.participant_ids, *anchor.changed_entity_ids]
                )
                related = sorted(
                    (
                        event
                        for event in self.events.values()
                        if event.id != anchor_id
                        and anchor_entities.intersection(
                            [*event.participant_ids, *event.changed_entity_ids]
                        )
                    ),
                    key=lambda item: (item.start_time, item.id),
                )
                if intent.temporal_relation == "after":
                    neighbors = [item for item in related if item.start_time >= anchor.end_time]
                    if neighbors and neighbors[0].id not in direct_event_ids:
                        direct_event_ids.append(neighbors[0].id)
                else:
                    neighbors = [item for item in related if item.end_time <= anchor.start_time]
                    if neighbors and neighbors[-1].id not in direct_event_ids:
                        direct_event_ids.append(neighbors[-1].id)
        direct_event_ids.extend(
            event_id for event_id in scored_event_ids if event_id not in direct_event_ids
        )
        selected_event_ids: list[str] = []
        retrieval_mode = "direct" if selected_entity_ids else "recent_only"
        semlink_status = "not_needed"
        suggested_option = ""
        if not selected_entity_ids:
            linked_entities, linked_events, semlink_status, suggested_option = self._semantic_link(
                question, options
            )
            if linked_entities or linked_events:
                selected_entity_ids = linked_entities
                selected_event_ids.extend(linked_events)
                retrieval_mode = "semlink"
        selected_event_ids.extend(
            event_id for event_id in direct_event_ids if event_id not in selected_event_ids
        )
        if retrieval_mode == "recent_only" and selected_event_ids:
            retrieval_mode = "direct-event"

        entity_score_map = dict(direct_scores)
        (
            memory_text,
            assembled_event_ids,
            selected_records,
            candidate_records,
        ) = self._assemble_memory(
            intent,
            selected_entity_ids,
            selected_event_ids,
            question,
            options,
            concept=retrieval_mode == "semlink",
            suggested_option=suggested_option,
            entity_scores=entity_score_map,
        )
        best_grounding_score = max(
            (item.grounding_score for item in selected_records), default=0.0
        )
        should_recover = bool(
            candidate_records
            and (
                not selected_records
                or best_grounding_score < self.config.sufficient_relevance
            )
        )
        history_frames: list[Image.Image] = []
        history_times: list[float] = []
        history_ids: list[str] = []
        if should_recover:
            history_frames, history_times, history_ids = self._recover_evidence(
                candidate_records,
                query_time=query_time,
                recent_start=recent_start,
                intent=intent,
            )

        formatted_options = _format_options(options)
        if original_prompt is None:
            original_prompt = (
                f"Question: {question}\n\nOptions:\n{formatted_options}\n\n"
                "Only give the best option's letter (A, B, C, or D) directly."
            )
        dialogue_lines = [
            f"Turn {index + 1} @ {turn.timestamp:.1f}s | Q: {turn.question} | "
            f"model prediction: {turn.predicted_label or '?'} {turn.predicted_text}".rstrip()
            for index, turn in enumerate(self.dialogue_history[-MAX_DIALOGUE_TURNS:])
        ]
        dialogue_text = "\n".join(dialogue_lines) or "(no previous turns)"
        dialogue_encoded = dialogue_text.encode("utf-8")
        if len(dialogue_encoded) > MAX_DIALOGUE_BYTES:
            dialogue_text = dialogue_encoded[-MAX_DIALOGUE_BYTES:].decode(
                "utf-8", errors="ignore"
            )
        if self.config.structured_answer:
            answer_values = {
                "history_frame_count": len(history_frames),
                "history_timestamps": json.dumps(history_times),
                "concept_header": (
                    "Conceptual catalog linking was used; indirect evidence may support the answer."
                    if retrieval_mode == "semlink"
                    else "Use direct factual evidence."
                ),
                "memory_text": memory_text,
                "original_prompt": original_prompt,
                "question": question,
                "options_text": formatted_options,
                "dialogue_history": dialogue_text,
            }
            prompt = _render(ANSWER_PROMPT, answer_values)
            recent_only_prompt = _render(
                ANSWER_PROMPT,
                {
                    **answer_values,
                    "history_frame_count": 0,
                    "history_timestamps": "[]",
                },
            )
        else:
            prompt = (
                "Use FOCUSED_VIDEO_MEMORY as fallible historical evidence. "
                "Recovered historical frames, when present, appear before the unchanged recent window.\n\n"
                f"<FOCUSED_VIDEO_MEMORY>\n{memory_text}\n</FOCUSED_VIDEO_MEMORY>\n\n"
                + original_prompt
            )
            recent_only_prompt = prompt

        plan = FolioQueryPlan(
            query_id=query_id,
            query_time=float(query_time),
            prompt=prompt,
            recent_only_prompt=recent_only_prompt,
            evidence_frames=tuple(history_frames),
            evidence_timestamps=tuple(history_times),
            evidence_frame_ids=tuple(history_ids),
            selected_entity_ids=tuple(selected_entity_ids),
            selected_event_ids=tuple(assembled_event_ids),
            matched_event_ids=tuple(selected_event_ids),
            selected_records=tuple(selected_records),
            candidate_records=tuple(candidate_records),
            direct_scores=tuple(direct_scores),
            retrieval_mode=retrieval_mode,
            semlink_status=semlink_status,
            query_type=intent.query_type,
            memory_text=memory_text,
            memory_bytes=len(memory_text.encode("utf-8")),
            dialogue_turns_used=min(len(self.dialogue_history), MAX_DIALOGUE_TURNS),
            best_grounding_score=best_grounding_score,
            suggested_option=suggested_option,
        )
        self.query_count += 1
        self.query_traces.append(
            {
                **plan.to_metadata(),
                "question": question,
                "options": list(options),
                "query_time": query_time,
                "recent_start": recent_start,
            }
        )
        return plan

    def commit_interaction(
        self,
        query_id: str,
        plan: FolioQueryPlan,
        question: str,
        options: Sequence[str],
        predicted_label: str = "",
        predicted_text: str = "",
    ) -> bool:
        if query_id != plan.query_id:
            raise ValueError("Interaction query id does not match its query plan")
        if query_id in self.committed_query_ids:
            return False
        normalized_label = str(predicted_label).strip().upper()
        if normalized_label not in {"A", "B", "C", "D"}:
            normalized_label = ""
        matched = set(plan.selected_entity_ids)
        for event_id in plan.matched_event_ids:
            event = self.events.get(event_id)
            if event is not None:
                matched.update(event.participant_ids)
                matched.update(event.changed_entity_ids)
        self.committed_query_ids.add(query_id)
        self.dialogue_history.append(
            DialogueTurn(
                query_id=query_id,
                timestamp=plan.query_time,
                question=_clean_text(question, required=True),
                options=tuple(_clean_text(str(item)) for item in options),
                predicted_label=normalized_label,
                predicted_text=_clean_text(str(predicted_text), limit=2_048),
                selected_entity_ids=tuple(sorted(matched)),
                selected_event_ids=plan.matched_event_ids,
            )
        )
        for trace in reversed(self.query_traces):
            if trace.get("query_id") == query_id:
                trace["predicted_label"] = normalized_label
                trace["predicted_text"] = str(predicted_text)
                break
        if not self.config.interaction_focus:
            return True
        for event_id in plan.matched_event_ids:
            if event_id in self.events:
                self.events[event_id].focus_score = min(
                    1.0, self.events[event_id].focus_score + 0.25
                )
        for entity_id in matched:
            if entity_id in self.entities:
                self.entities[entity_id].focus_score = min(
                    1.0, self.entities[entity_id].focus_score + 0.25
                )
        query_terms = _tokens(question + " " + " ".join(str(item) for item in options))
        matched_terms: set[str] = set()
        for entity_id in matched:
            entity = self.entities.get(entity_id)
            if entity is not None:
                matched_terms.update(
                    _expand_synonyms(
                        _tokens(
                            " ".join(
                                [entity.canonical_name, *entity.aliases, entity.category]
                            )
                        )
                    )
                )
        for term in sorted(query_terms - matched_terms):
            if len(term) >= 3:
                self.pending_focus_terms.setdefault(term, self.segment_index)
        return True

    def usage(self) -> dict[str, int | float]:
        return {
            "record_count": len(self.entities),
            "entity_count": len(self.entities),
            "observation_count": sum(len(item.observations) for item in self.entities.values()),
            "event_count": len(self.events),
            "evidence_cache_count": len(self.evidence),
            "pending_frames": len(self.pending),
            "segments_written": self.segment_index,
            "observed_frames": self.observed_frames,
            "selected_frames": self.selected_frames,
            "write_calls": self.write_calls,
            "write_errors": self.write_errors,
            "write_seconds": self.write_seconds,
            "semantic_link_calls": self.semantic_link_calls,
            "semantic_link_errors": self.semantic_link_errors,
            "semantic_link_seconds": self.semantic_link_seconds,
            "query_count": self.query_count,
            "dialogue_turn_count": len(self.dialogue_history),
            "query_errors": self.query_errors,
            "cache_recovery_count": self.cache_recovery_count,
            "cache_decode_errors": self.cache_decode_errors,
            "dropped_frames": self.dropped_frames,
            "snapshot_errors": self.snapshot_errors,
        }

    def _state_dict(self, pending_files: Sequence[str] | None = None) -> dict[str, Any]:
        stats = self.usage()
        stats.pop("snapshot_errors", None)
        if pending_files is None:
            pending_files = [f"pending/frame-{index:04d}.png" for index in range(len(self.pending))]
        if len(pending_files) != len(self.pending):
            raise ValueError("Pending snapshot manifest does not match buffered frames")
        return {
            "schema": "folio-memory-state-v1",
            "config": asdict(self.config),
            "entities": [asdict(item) for item in self.entities.values()],
            "events": [asdict(item) for item in self.events.values()],
            "evidence": [item.metadata() for item in self.evidence.values()],
            "pending": [
                {
                    "timestamp": timestamp,
                    "file": pending_files[index],
                }
                for index, (timestamp, _) in enumerate(self.pending)
            ],
            "pending_focus_terms": self.pending_focus_terms,
            "committed_query_ids": sorted(self.committed_query_ids),
            "dialogue_history": [asdict(turn) for turn in self.dialogue_history],
            "query_traces": self.query_traces,
            "last_time": self.last_time,
            "segment_index": self.segment_index,
            "next_entity_index": self.next_entity_index,
            "next_event_index": self.next_event_index,
            "new_frames_since_attempt": self._new_frames_since_attempt,
            "stats": stats,
            "last_write_error": self.last_write_error,
        }

    def _memory_markdown(self) -> str:
        lines = ["# FOLIO focused semantic memory", ""]
        if not self.entities:
            lines.append("_No committed entities._")
        for entity in sorted(self.entities.values(), key=lambda item: item.id):
            lines.extend(
                [
                    f"## {entity.id}: {entity.canonical_name}",
                    "",
                    f"- Category: {entity.category or 'unknown'}",
                    f"- Aliases: {', '.join(entity.aliases) or 'none'}",
                    f"- Attributes: {', '.join(entity.attributes) or 'none'}",
                    f"- Seen: {entity.first_seen:.1f}s to {entity.last_seen:.1f}s",
                    f"- Focus: {entity.focus_score:.3f} ({self._focus_level(entity.focus_score)})",
                    "",
                ]
            )
            for observation in entity.observations:
                lines.append("- " + self._observation_line(observation))
            lines.append("")
        if self.events:
            lines.extend(["# Events", ""])
            for event in sorted(self.events.values(), key=lambda item: item.id):
                lines.append(
                    f"- **{event.id}** [{event.start_time:.1f}-{event.end_time:.1f}s] "
                    f"{event.event_type}: {event.summary}"
                )
        return "\n".join(lines).rstrip() + "\n"

    @staticmethod
    def _entity_markdown(entity: FolioEntity) -> str:
        lines = [
            "---",
            f"id: {json.dumps(entity.id)}",
            f"name: {json.dumps(entity.canonical_name, ensure_ascii=False)}",
            f"category: {json.dumps(entity.category, ensure_ascii=False)}",
            f"focus: {entity.focus_score:.6f}",
            "---",
            "",
            f"# {entity.canonical_name}",
            "",
            f"Aliases: {', '.join(entity.aliases) or 'none'}",
            "",
            f"Attributes: {', '.join(entity.attributes) or 'none'}",
            "",
            "## Observation chain",
            "",
        ]
        lines.extend("- " + FolioMemory._observation_line(item) for item in entity.observations)
        if entity.event_ids:
            lines.extend(["", "## Event links", "", *[f"- {item}" for item in entity.event_ids]])
        return "\n".join(lines).rstrip() + "\n"

    def save_snapshot(self, directory: str | Path) -> bool:
        target = Path(directory)
        try:
            target.mkdir(parents=True, exist_ok=True)
            evidence_dir = target / "evidence"
            entity_dir = target / "entities"
            pending_dir = target / "pending"
            evidence_dir.mkdir(exist_ok=True)
            entity_dir.mkdir(exist_ok=True)
            pending_dir.mkdir(exist_ok=True)

            expected_evidence: set[str] = set()
            for frame in self.evidence.values():
                filename = frame.metadata()["jpeg_file"]
                expected_evidence.add(filename)
                destination = evidence_dir / filename
                if not destination.exists() or destination.stat().st_size != len(frame.jpeg):
                    _atomic_write(destination, frame.jpeg)

            expected_entities: set[str] = set()
            for entity in self.entities.values():
                filename = f"{entity.id}.md"
                expected_entities.add(filename)
                _atomic_write_text(entity_dir / filename, self._entity_markdown(entity))

            expected_pending: set[str] = set()
            pending_files: list[str] = []
            for index, (_, image) in enumerate(self.pending):
                png = _encode_png(image)
                digest = hashlib.sha256(png).hexdigest()[:16]
                filename = f"frame-{index:04d}-{digest}.png"
                expected_pending.add(filename)
                pending_files.append(f"pending/{filename}")
                destination = pending_dir / filename
                if not destination.exists() or destination.stat().st_size != len(png):
                    _atomic_write(destination, png)

            _atomic_write_text(target / "MEMORY.md", self._memory_markdown())
            _atomic_write_text(
                target / "focus.json",
                json.dumps(
                    {
                        "entities": {
                            item.id: {
                                "name": item.canonical_name,
                                "score": item.focus_score,
                                "level": self._focus_level(item.focus_score),
                                "last_seen_segment": item.last_seen_segment,
                            }
                            for item in self.entities.values()
                        },
                        "events": {
                            item.id: {
                                "type": item.event_type,
                                "score": item.focus_score,
                                "level": self._focus_level(item.focus_score),
                                "end_time": item.end_time,
                            }
                            for item in self.events.values()
                        },
                        "pending_terms": self.pending_focus_terms,
                        "committed_query_ids": sorted(self.committed_query_ids),
                        "dialogue_turn_count": len(self.dialogue_history),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
            _atomic_write_text(
                evidence_dir / "index.json",
                json.dumps(
                    [item.metadata() for item in self.evidence.values()],
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
            )
            _atomic_write_text(
                target / "queries.jsonl",
                "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in self.query_traces),
            )
            _atomic_write_text(
                target / "usage.json",
                json.dumps(self.usage(), ensure_ascii=False, indent=2) + "\n",
            )
            # Commit the machine-restorable manifest last. Until this atomic
            # replacement succeeds, every file referenced by the previous
            # state.json remains in place.
            _atomic_write_text(
                target / "state.json",
                json.dumps(
                    self._state_dict(pending_files), ensure_ascii=False, indent=2
                )
                + "\n",
            )

            # Orphan cleanup happens only after the new manifest is durable.
            for candidate in evidence_dir.glob("*.jpg"):
                if candidate.name not in expected_evidence:
                    try:
                        candidate.unlink(missing_ok=True)
                    except OSError:
                        pass
            for candidate in entity_dir.glob("*.md"):
                if candidate.name not in expected_entities:
                    try:
                        candidate.unlink(missing_ok=True)
                    except OSError:
                        pass
            for candidate in pending_dir.iterdir():
                if candidate.is_file() and candidate.name not in expected_pending:
                    try:
                        candidate.unlink(missing_ok=True)
                    except OSError:
                        pass
            self.last_snapshot_error = None
            return True
        except Exception as exc:
            self.snapshot_errors += 1
            self.last_snapshot_error = str(exc)
            return False

    @classmethod
    def load_snapshot(
        cls,
        directory: str | Path,
        generate: Callable[[list[Image.Image], str, int], str],
    ) -> "FolioMemory":
        target = Path(directory)
        state = json.loads((target / "state.json").read_text(encoding="utf-8"))
        if state.get("schema") != "folio-memory-state-v1":
            raise ValueError("Unsupported FOLIO snapshot schema")
        config = FolioConfig(**state["config"])
        memory = cls(generate, config=config)
        entities: dict[str, FolioEntity] = {}
        for value in state.get("entities", []):
            observations = [FolioObservation(**item) for item in value.pop("observations", [])]
            entity = FolioEntity(**value, observations=observations)
            entities[entity.id] = entity
        events = {
            item["id"]: FolioEvent(**item)
            for item in state.get("events", [])
        }
        evidence: dict[str, EvidenceFrame] = {}
        for value in state.get("evidence", []):
            value = dict(value)
            filename = value.pop("jpeg_file")
            data = (target / "evidence" / filename).read_bytes()
            frame = EvidenceFrame(**value, jpeg=data)
            evidence[frame.id] = frame
        pending: list[tuple[float, Image.Image]] = []
        for value in state.get("pending", []):
            with Image.open(target / value["file"]) as image:
                pending.append((float(value["timestamp"]), image.convert("RGB")))
        memory.entities = entities
        memory.events = events
        memory.evidence = evidence
        memory.pending = pending
        memory.pending_focus_terms = {
            str(key): int(value) for key, value in state.get("pending_focus_terms", {}).items()
        }
        memory.committed_query_ids = set(state.get("committed_query_ids", []))
        memory.dialogue_history = [
            DialogueTurn(
                query_id=str(item["query_id"]),
                timestamp=float(item["timestamp"]),
                question=str(item["question"]),
                options=tuple(str(value) for value in item.get("options", [])),
                predicted_label=str(item.get("predicted_label", "")),
                predicted_text=str(item.get("predicted_text", "")),
                selected_entity_ids=tuple(
                    str(value) for value in item.get("selected_entity_ids", [])
                ),
                selected_event_ids=tuple(
                    str(value) for value in item.get("selected_event_ids", [])
                ),
            )
            for item in state.get("dialogue_history", [])
        ]
        memory.query_traces = list(state.get("query_traces", []))
        memory.last_time = float(state.get("last_time", -1.0))
        memory.segment_index = int(state.get("segment_index", 0))
        memory.next_entity_index = int(state.get("next_entity_index", 1))
        memory.next_event_index = int(state.get("next_event_index", 1))
        memory._new_frames_since_attempt = int(state.get("new_frames_since_attempt", len(pending)))
        for key, value in state.get("stats", {}).items():
            if hasattr(memory, key):
                setattr(memory, key, value)
        memory.last_write_error = state.get("last_write_error")
        return memory


class FolioMemorySession:
    """Advance one video's FOLIO state on a fixed causal source clock."""

    def __init__(
        self,
        path: str,
        fps: float,
        generate: Callable[[list[Image.Image], str, int], str],
        *,
        config: FolioConfig,
        frame_source: Iterable[tuple[float, Image.Image]] | None = None,
        memory: FolioMemory | None = None,
    ) -> None:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")
        self.path = path
        self.fps = float(fps)
        self.memory = memory or FolioMemory(generate, config=config)
        if self.memory.config != config:
            raise ValueError("Loaded FOLIO memory config does not match the session")
        self._frames: Iterator[tuple[float, Image.Image]] = iter(
            frame_source if frame_source is not None else video_frames(path, fps)
        )
        self._next_frame: tuple[float, Image.Image] | None = None
        self._exhausted = False
        self._closed = False
        self._last_cutoff = float(self.memory.last_time)
        self._stream_errors = 0

    def advance_to(self, timestamp: float) -> None:
        if self._closed:
            raise RuntimeError("FOLIO session is closed")
        if not math.isfinite(timestamp) or timestamp < 0:
            raise ValueError("Question timestamps must be finite and nonnegative")
        if timestamp < self._last_cutoff:
            raise ValueError("Question timestamps must be nondecreasing")
        self._last_cutoff = float(timestamp)
        while True:
            try:
                if self._next_frame is None and not self._exhausted:
                    self._next_frame = next(self._frames, None)
                    self._exhausted = self._next_frame is None
                if self._next_frame is None or self._next_frame[0] > timestamp:
                    return
                frame = self._next_frame
                self._next_frame = None
                if frame[0] <= self.memory.last_time:
                    continue
                self.memory.observe(*frame)
            except Exception:
                self._stream_errors += 1
                self._next_frame = None
                self._exhausted = True
                return

    def prepare_query(self, *args: Any, **kwargs: Any) -> FolioQueryPlan:
        return self.memory.prepare_query(*args, **kwargs)

    def commit_interaction(self, *args: Any, **kwargs: Any) -> bool:
        return self.memory.commit_interaction(*args, **kwargs)

    def usage(self) -> dict[str, int | float]:
        return {**self.memory.usage(), "stream_errors": self._stream_errors}

    def save_snapshot(self, directory: str | Path) -> bool:
        return self.memory.save_snapshot(directory)

    def close(self) -> None:
        if self._closed:
            return
        close = getattr(self._frames, "close", None)
        if close is not None:
            close()
        self._closed = True
