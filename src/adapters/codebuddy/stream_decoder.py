"""SSE decoding for CodeBuddyAdapterV2 (spec §9, §10).

CodeBuddy responds only with an OpenAI-style SSE stream. :class:`StreamDecoder`
turns a raw byte/text stream into discrete parsed chunk objects and offers two
consumption modes:

  * :meth:`iter_events` — line-oriented parsing that yields each decoded SSE
    ``data:`` object (skipping comments, blanks, and ``[DONE]``). The caller
    handles pass-through vs aggregation.
  * :meth:`aggregate` — consume the whole stream into a single non-streaming
    ``chat.completion`` message, reconstructing split tool-call arguments via
    :class:`ToolCallAccumulator`.

Parsing is incremental and buffered so a chunk boundary in the middle of a line
never corrupts a frame.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from .tool_call_state import ToolCallAccumulator

logger = logging.getLogger(__name__)


def parse_sse_data_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single ``data:`` SSE line into an object, or ``None``.

    Returns ``None`` for comments (``:`` prefix), blank lines, non-data lines,
    the ``[DONE]`` sentinel, and any line whose payload is not valid JSON.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        return None
    if not stripped.startswith("data:"):
        return None
    data = stripped[len("data:"):].strip()
    if not data or data == "[DONE]":
        return None
    try:
        return json.loads(data)
    except (ValueError, TypeError):
        return None


class SSELineBuffer:
    """Incrementally split a chunked text stream into complete lines."""

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, chunk: str) -> List[str]:
        """Add a chunk and return any complete lines it produced."""
        if not chunk:
            return []
        self._buffer += chunk
        lines: List[str] = []
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            lines.append(line)
        return lines

    def flush(self) -> Optional[str]:
        """Return any trailing partial line and clear the buffer."""
        if self._buffer.strip():
            line = self._buffer
            self._buffer = ""
            return line
        self._buffer = ""
        return None


def _extract_delta_content(obj: Dict[str, Any]) -> str:
    """Return the assistant ``delta.content`` string from a chunk object."""
    try:
        choices = obj.get("choices") or []
        if not choices:
            return ""
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        return content if isinstance(content, str) else ""
    except (AttributeError, IndexError, TypeError):
        return ""


class StreamAggregator:
    """Aggregate decoded SSE chunk objects into one ``chat.completion`` message.

    Content deltas are concatenated; tool-call fragments are routed to a
    :class:`ToolCallAccumulator` so split ``function.arguments`` are only parsed
    once complete. ``finalize`` returns the OpenAI non-streaming message body.
    """

    def __init__(self) -> None:
        self.id: Optional[str] = None
        self.model: Optional[str] = None
        self.system_fingerprint: Optional[str] = None
        self.content = ""
        self.finish_reason: Optional[str] = None
        self.usage: Optional[Dict[str, Any]] = None
        self.reasoning_content = ""
        self._tools = ToolCallAccumulator()

    def process_chunk(self, obj: Dict[str, Any]) -> None:
        if not isinstance(obj, dict):
            return
        self.id = self.id or obj.get("id")
        self.model = self.model or obj.get("model")
        self.system_fingerprint = obj.get("system_fingerprint") or self.system_fingerprint
        if obj.get("usage"):
            self.usage = obj.get("usage")

        choices = obj.get("choices") or []
        if not choices:
            return
        choice = choices[0]
        if choice.get("finish_reason"):
            self.finish_reason = choice.get("finish_reason")

        delta = choice.get("delta") or {}
        if isinstance(delta.get("content"), str):
            self.content += delta["content"]
        if isinstance(delta.get("reasoning_content"), str):
            self.reasoning_content += delta["reasoning_content"]
        if delta.get("tool_calls"):
            self._tools.process_delta_tool_calls(delta["tool_calls"])

    def finalize(self) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
        """Return ``(content, tool_calls, finish_reason)``.

        May raise :class:`InvalidToolArgumentsError` from the accumulator when a
        completed tool call has malformed JSON arguments.
        """
        tool_calls = self._tools.finalize() if self._tools.has_pending() else []
        finish_reason = "tool_calls" if tool_calls else (self.finish_reason or "stop")
        return self.content, tool_calls, finish_reason
