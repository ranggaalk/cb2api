"""Streaming tool-call reconstruction state machine (spec §9).

Upstream delivers a tool call across many SSE chunks, and ``function.arguments``
may be split at arbitrary byte boundaries:

    chunk 1: {"file_
    chunk 2: path":"port
    chunk 3: folio.html"}

Parsing any single fragment as JSON would fail or, worse, silently produce ``{}``
— the exact bug the legacy ``validate_and_fix_tool_call_args`` had. This state
machine instead:

  * groups fragments by tool-call ``index`` AND ``id`` (upstream reuses index 0
    for every call, so the ID is the real key once known);
  * appends every ``arguments`` fragment verbatim without parsing;
  * parses the joined JSON exactly once, when the call is finalized;
  * raises :class:`InvalidToolArgumentsError` if the *complete* JSON is invalid,
    rather than coercing it to ``{}``;
  * emits each tool call exactly once and never mutates its ID.

The machine is transport-agnostic: :class:`StreamDecoder` feeds it delta
fragments and asks it to finalize when ``finish_reason == "tool_calls"`` or the
stream ends.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from .errors import InvalidToolArgumentsError
from .models import ToolCallState

logger = logging.getLogger(__name__)


class ToolCallAccumulator:
    """Accumulates streaming tool-call fragments into complete tool calls."""

    def __init__(self) -> None:
        # Keyed by a stable slot key so index-0-repeated calls stay separate
        # once their IDs are known.
        self._states: Dict[str, ToolCallState] = {}
        # Preserve the order tool calls were first seen.
        self._order: List[str] = []
        # The slot key of the tool call currently receiving fragments; used for
        # incremental deltas that arrive without an id/index.
        self._current_key: Optional[str] = None

    @staticmethod
    def _slot_key(index: Optional[int], tool_id: Optional[str]) -> str:
        """Build a stable key. Prefer the ID; fall back to the index."""
        if tool_id:
            return f"id:{tool_id}"
        if index is not None:
            return f"idx:{index}"
        return "idx:0"

    def has_pending(self) -> bool:
        return bool(self._states)

    def process_delta_tool_calls(self, tool_calls: List[Dict[str, Any]]) -> None:
        """Consume the ``delta.tool_calls`` array from one SSE chunk.

        Each entry may carry an ``id``/``index``/``function.name`` (a new call or
        the first fragment) and/or a ``function.arguments`` fragment (an
        increment). Nothing is parsed here — fragments are only buffered.
        """
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            index = tc.get("index")
            tool_id = tc.get("id")
            func = tc.get("function") or {}
            name = func.get("name") if isinstance(func, dict) else None
            arguments = func.get("arguments") if isinstance(func, dict) else None

            key = self._resolve_key(index, tool_id)
            state = self._states.get(key)
            if state is None:
                state = ToolCallState(index=index if index is not None else 0)
                self._states[key] = state
                self._order.append(key)
                logger.debug("New streaming tool call slot=%s", key)

            # Fill in identity fields as they arrive.
            if tool_id and not state.id:
                state.id = tool_id
            if name:
                state.name = name
            if arguments:
                state.append_arguments(arguments)

            self._current_key = key

    def _resolve_key(self, index: Optional[int], tool_id: Optional[str]) -> str:
        """Map an incoming delta entry to an existing slot or a new one.

        When an id is present we key by id, merging any earlier index-only slot
        that was opened before the id was known. When neither id nor index is
        present, the fragment belongs to the call currently in progress.
        """
        if tool_id:
            id_key = f"id:{tool_id}"
            if id_key in self._states:
                return id_key
            # Promote a prior index-only slot to this id, if one exists and has
            # not yet been assigned an id.
            if index is not None:
                idx_key = f"idx:{index}"
                existing = self._states.get(idx_key)
                if existing is not None and not existing.id:
                    self._states[id_key] = existing
                    self._states.pop(idx_key, None)
                    self._order[:] = [
                        id_key if k == idx_key else k for k in self._order
                    ]
                    return id_key
            return id_key
        if index is not None:
            return f"idx:{index}"
        if self._current_key is not None:
            return self._current_key
        return "idx:0"

    def finalize(self) -> List[Dict[str, Any]]:
        """Join fragments, parse arguments once, and return complete tool calls.

        Raises :class:`InvalidToolArgumentsError` when a completed tool call's
        joined arguments are not valid JSON. Empty arguments are normalized to
        ``"{}"`` (a call with no arguments is valid); a NON-empty but malformed
        argument string is an error, never silently replaced.
        """
        result: List[Dict[str, Any]] = []
        for key in self._order:
            state = self._states.get(key)
            if state is None or state.emitted:
                continue

            raw = state.raw_arguments().strip()
            if not raw:
                arguments = "{}"
            else:
                try:
                    parsed = json.loads(raw)
                except (ValueError, TypeError) as exc:
                    logger.warning(
                        "Streaming tool call produced invalid JSON arguments "
                        "(slot=%s, name=%s)",
                        key,
                        state.name,
                    )
                    raise InvalidToolArgumentsError(state.name) from exc
                # Re-serialize compactly to guarantee a canonical JSON string.
                arguments = json.dumps(parsed, ensure_ascii=False)

            state.emitted = True
            result.append(
                {
                    "id": state.id or "",
                    "type": "function",
                    "function": {
                        "name": state.name or "",
                        "arguments": arguments,
                    },
                }
            )
        return result
