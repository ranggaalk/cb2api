"""Settings snapshot for CodeBuddyAdapterV2.

The adapter reads configuration through :func:`load_adapter_settings`, which
pulls from the root layered config system (in-memory → config.json → env →
defaults). Bundling the values into one immutable dataclass keeps request
handling free of scattered ``get_*`` calls and makes tests trivial: build an
``AdapterSettings`` directly instead of monkeypatching many getters.
"""
from __future__ import annotations

from dataclasses import dataclass

import config as root_config


@dataclass(frozen=True)
class AdapterSettings:
    """Immutable per-request snapshot of adapter configuration."""

    adapter_version: str
    request_profile: str
    upstream_api_key_header: str
    sanitize_agent_prompt: bool
    max_system_prompt_length: int

    default_model: str
    unknown_model_policy: str

    connect_timeout: float
    pool_timeout: float
    write_timeout: float
    headers_timeout: float
    first_chunk_timeout: float
    stream_idle_timeout: float

    max_concurrent_upstream_requests: int
    upstream_queue_timeout: float

    warn_total_content_length: int
    warn_message_count: int
    warn_tool_count: int

    api_endpoint: str

    @property
    def chat_completions_url(self) -> str:
        return f"{self.api_endpoint}/v2/chat/completions"


def load_adapter_settings() -> AdapterSettings:
    """Build an :class:`AdapterSettings` from the current root configuration.

    Config-validation errors (e.g. an invalid profile or header mode) propagate
    to the caller, which surfaces them as a 500 misconfiguration error, exactly
    like the legacy path.
    """
    return AdapterSettings(
        adapter_version=root_config.get_codebuddy_adapter_version(),
        request_profile=root_config.get_codebuddy_request_profile(),
        upstream_api_key_header=root_config.get_upstream_api_key_header(),
        sanitize_agent_prompt=root_config.get_sanitize_agent_prompt(),
        max_system_prompt_length=root_config.get_max_system_prompt_length(),
        default_model=root_config.get_codebuddy_default_model(),
        unknown_model_policy=root_config.get_codebuddy_unknown_model_policy(),
        connect_timeout=root_config.get_codebuddy_connect_timeout_seconds(),
        pool_timeout=root_config.get_codebuddy_pool_timeout_seconds(),
        write_timeout=root_config.get_codebuddy_write_timeout_seconds(),
        headers_timeout=root_config.get_codebuddy_headers_timeout_seconds(),
        first_chunk_timeout=root_config.get_codebuddy_first_chunk_timeout_seconds(),
        stream_idle_timeout=root_config.get_codebuddy_stream_idle_timeout_seconds(),
        max_concurrent_upstream_requests=root_config.get_codebuddy_max_concurrent_upstream_requests(),
        upstream_queue_timeout=root_config.get_codebuddy_upstream_queue_timeout_seconds(),
        warn_total_content_length=root_config.get_codebuddy_warn_total_content_length(),
        warn_message_count=root_config.get_codebuddy_warn_message_count(),
        warn_tool_count=root_config.get_codebuddy_warn_tool_count(),
        api_endpoint=root_config.get_codebuddy_api_endpoint(),
    )
