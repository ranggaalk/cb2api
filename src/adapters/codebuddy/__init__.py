"""CodeBuddyAdapterV2 — an isolated, testable CodeBuddy provider adapter.

This package converts OpenAI/Claude-compatible chat-completion requests into
CodeBuddy-compatible requests, streams the SSE response back, and maps it to the
OpenAI schema. It is stateless per request: the upstream API key is supplied by
the caller (9Router) on each request and is never stored or rotated internally.

The router selects this adapter when ``CODEBUDDY_ADAPTER_VERSION=v2``. The
public entry point is :func:`get_adapter`, which returns a process-wide adapter
instance sharing a single pooled ``httpx.AsyncClient``.
"""
from __future__ import annotations

from typing import Optional

from .adapter import CodeBuddyAdapterV2

_adapter_instance: Optional[CodeBuddyAdapterV2] = None


def get_adapter() -> CodeBuddyAdapterV2:
    """Return the process-wide :class:`CodeBuddyAdapterV2` singleton."""
    global _adapter_instance
    if _adapter_instance is None:
        _adapter_instance = CodeBuddyAdapterV2()
    return _adapter_instance


__all__ = ["CodeBuddyAdapterV2", "get_adapter"]
