"""Dataclasses shared across CodeBuddyAdapterV2.

These are plain data holders — no I/O, no upstream calls — so they are trivially
unit-testable and safe to import from anywhere in the package.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ResolvedModel:
    """Outcome of resolving a client model label to an upstream model ID.

    ``source`` records how it was resolved (exact / alias / default_empty /
    passthrough / default) for safe diagnostics.
    """

    requested: str
    mapped: str
    source: str


@dataclass(repr=False)
class UpstreamCredential:
    """A request-local upstream credential.

    The raw ``bearer_token`` is never logged or serialized; ``repr`` is disabled
    so it cannot leak through incidental logging. ``fingerprint`` is a short,
    non-reversible SHA-256 prefix suitable for correlating logs.
    """

    bearer_token: str = field(repr=False)
    fingerprint: str = "none"
    source: str = "passthrough"
    user_id: Optional[str] = None
    key_id: Optional[str] = None


@dataclass
class ToolCallState:
    """Accumulates one streaming tool call across many SSE chunks (spec §9).

    Upstream may split ``function.arguments`` across arbitrary chunk boundaries
    (e.g. ``{"file_`` then ``path":"a.txt"}``). Fragments are buffered verbatim
    and only joined + parsed once the tool call is complete, so a partial JSON
    fragment is never parsed or emitted as tool input.
    """

    index: int
    id: Optional[str] = None
    name: Optional[str] = None
    argument_fragments: List[str] = field(default_factory=list)
    emitted: bool = False

    def append_arguments(self, fragment: str) -> None:
        if fragment:
            self.argument_fragments.append(fragment)

    def raw_arguments(self) -> str:
        """The concatenated argument fragments (may be incomplete mid-stream)."""
        return "".join(self.argument_fragments)


@dataclass
class TelemetryContext:
    """Per-request safe telemetry accumulator (spec §15).

    Holds only structural metadata — never prompts, file contents, tool
    argument values, or credentials. ``request_id`` correlates all stage logs.
    """

    request_id: str
    requested_model: str = "unknown"
    mapped_model: str = "unknown"
    mapping_source: str = "unknown"
    message_count: int = 0
    tool_count: int = 0
    total_content_length: int = 0
    key_fingerprint: str = "none"
    client_wants_stream: bool = False
    large_agentic_request: bool = False

    def as_log_fields(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "requested_model": self.requested_model,
            "mapped_model": self.mapped_model,
            "mapping_source": self.mapping_source,
            "message_count": self.message_count,
            "tool_count": self.tool_count,
            "total_content_length": self.total_content_length,
            "key_fingerprint": self.key_fingerprint,
            "stream": self.client_wants_stream,
            "large_agentic_request": self.large_agentic_request,
        }
