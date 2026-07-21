"""Model resolution and upstream payload construction for CodeBuddyAdapterV2.

Two responsibilities (spec §3):

  * :func:`resolve_model` — map a client/UI model label to an upstream CodeBuddy
    model ID WITHOUT ever silently rewriting an unknown label to ``auto-chat``.
    An unknown label is handled per ``CODEBUDDY_UNKNOWN_MODEL_POLICY``.
  * :func:`build_payload` — assemble the strict, allowlisted upstream payload.
    Only fields CodeBuddy understands are forwarded; OpenAI-only and unknown UI
    fields are dropped so they cannot trigger an upstream rejection.

Both are pure functions of their inputs plus the injected settings/model list,
so they are directly unit-testable without touching global config.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import config as root_config

from .config import AdapterSettings
from .errors import UnknownModelError
from .models import ResolvedModel

logger = logging.getLogger(__name__)

# Top-level request fields forwarded to CodeBuddy. Everything else (OpenAI-only
# fields like response_format / reasoning_effort / stream_options, and unknown
# UI fields) is dropped. ``tools`` / ``tool_choice`` are forwarded only when the
# client actually sent them.
_UPSTREAM_ALLOWLIST = {"model", "messages", "stream", "tools", "tool_choice"}


def resolve_model(
    requested_model: Any,
    settings: AdapterSettings,
    available_models: Optional[List[str]] = None,
    aliases: Optional[Dict[str, str]] = None,
) -> ResolvedModel:
    """Resolve a model label to an upstream model ID and record how.

    ``source`` is one of:
      * ``exact``          — matched a configured available model ID
      * ``alias``          — matched a configured alias (case-insensitive)
      * ``default_empty``  — no model supplied; used the default
      * ``passthrough``    — unknown label forwarded verbatim
      * ``default``        — unknown label mapped to the default model

    An unknown label follows ``settings.unknown_model_policy``:
      * ``passthrough`` (default): forward it verbatim.
      * ``reject``: raise :class:`UnknownModelError` (local HTTP 400).
      * ``default``: fall back to ``settings.default_model``.
    """
    if available_models is None:
        try:
            available_models = root_config.get_available_models()
        except Exception:
            available_models = []
    if aliases is None:
        try:
            aliases = root_config.get_codebuddy_model_aliases()
        except Exception:
            aliases = {}

    available = set(available_models)
    default_model = settings.default_model or "auto-chat"

    # No usable model supplied: nothing to pass through, so use the default
    # regardless of policy.
    if not isinstance(requested_model, str) or not requested_model.strip():
        return ResolvedModel(requested="", mapped=default_model, source="default_empty")
    requested = requested_model.strip()

    # Exact match against a known upstream model ID (covers an explicit
    # "auto-chat" request, which is therefore "exact", not "default").
    if requested in available:
        return ResolvedModel(requested=requested, mapped=requested, source="exact")

    # Configured alias mapping (case-insensitive).
    mapped = aliases.get(requested.lower())
    if mapped:
        return ResolvedModel(requested=requested, mapped=mapped, source="alias")

    # Unknown label: governed by policy. Never silently rewrite to the default.
    policy = settings.unknown_model_policy
    if policy == "reject":
        logger.info("Unknown requested model rejected (policy=reject)")
        raise UnknownModelError(requested)
    if policy == "default":
        logger.info("Unknown requested model mapped to default (policy=default)")
        return ResolvedModel(requested=requested, mapped=default_model, source="default")
    # passthrough (default): forward verbatim so the real upstream model is
    # preserved and CodeBuddy decides whether it is valid.
    return ResolvedModel(requested=requested, mapped=requested, source="passthrough")


def dropped_fields(request_body: Dict[str, Any]) -> List[str]:
    """Return the sorted top-level field names that will NOT be forwarded."""
    return sorted(k for k in request_body.keys() if k not in _UPSTREAM_ALLOWLIST)


def build_payload(
    request_body: Dict[str, Any],
    mapped_model: str,
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the strict upstream payload.

    CodeBuddy is SSE-only, so ``stream`` is always ``True`` upstream regardless
    of the client's preference; the adapter decides separately whether to
    aggregate (non-stream client) or pass the stream through. ``tools`` and
    ``tool_choice`` are forwarded only when present.
    """
    payload: Dict[str, Any] = {
        "model": mapped_model,
        "messages": messages,
        "stream": True,
    }
    if request_body.get("tools"):
        payload["tools"] = request_body["tools"]
    if request_body.get("tool_choice") is not None:
        payload["tool_choice"] = request_body["tool_choice"]
    return payload
