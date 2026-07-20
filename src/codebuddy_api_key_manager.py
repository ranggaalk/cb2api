"""
CodeBuddy API Key Manager - TXT key pool, rotation, and runtime state
"""
import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Set

from config import (
    get_codebuddy_api_key_cooldown_seconds,
    get_codebuddy_api_key_reload_interval,
    get_codebuddy_api_key_rotation,
    get_codebuddy_api_keys_file,
)

logger = logging.getLogger(__name__)


class ApiKeyConfigurationError(Exception):
    """The API key file configuration is invalid."""


@dataclass(repr=False)
class ApiKeySelection:
    """A request-scoped key that must never be serialized or logged."""

    key: str = field(repr=False)
    key_id: str
    masked_key: str


@dataclass(repr=False)
class _ApiKeyState:
    key: str = field(repr=False)
    key_id: str
    masked_key: str
    status: str = "active"
    request_count: int = 0
    error_count: int = 0
    last_used_at: Optional[float] = None
    cooldown_until: Optional[float] = None
    last_error: Optional[str] = None


class CodeBuddyApiKeyManager:
    """Concurrency-safe CodeBuddy upstream API key pool."""

    def __init__(
        self,
        file_path: Optional[str] = None,
        cooldown_seconds: Optional[int] = None,
        reload_interval: Optional[int] = None,
        clock: Callable[[], float] = time.time,
    ):
        self._file_path = file_path
        self._cooldown_seconds = cooldown_seconds
        self._reload_interval = reload_interval
        self._clock = clock
        self._states: List[_ApiKeyState] = []
        self._cursor = 0
        self._lock = asyncio.Lock()
        self._reload_task: Optional[asyncio.Task] = None

    @staticmethod
    def parse_text(content: str) -> List[str]:
        """Parse TXT content while preserving order and removing duplicates."""
        keys = []
        seen = set()
        for raw_line in content.splitlines():
            key = raw_line.strip()
            if not key or key.startswith("#") or key in seen:
                continue
            seen.add(key)
            keys.append(key)
        return keys

    @staticmethod
    def _key_id(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    @staticmethod
    def mask_key(key: str) -> str:
        if len(key) < 9:
            return "****"
        return f"{key[:4]}...{key[-4:]}"

    def _get_file_path(self) -> str:
        path = self._file_path if self._file_path is not None else get_codebuddy_api_keys_file()
        if not path:
            raise ApiKeyConfigurationError("CODEBUDDY_API_KEYS_FILE is not configured")
        return os.path.abspath(os.path.expanduser(path))

    def _get_cooldown_seconds(self) -> int:
        if self._cooldown_seconds is not None:
            return self._cooldown_seconds
        return get_codebuddy_api_key_cooldown_seconds()

    def _get_reload_interval(self) -> int:
        if self._reload_interval is not None:
            return self._reload_interval
        return get_codebuddy_api_key_reload_interval()

    def _refresh_expired_cooldowns(self, now: float) -> None:
        for state in self._states:
            if (
                state.status == "cooldown"
                and state.cooldown_until is not None
                and state.cooldown_until <= now
            ):
                state.status = "active"
                state.cooldown_until = None

    def _read_keys(self) -> tuple[List[str], bool]:
        try:
            get_codebuddy_api_key_rotation()
            path = self._get_file_path()
            with open(path, "r", encoding="utf-8") as file:
                return self.parse_text(file.read()), True
        except (OSError, ValueError, ApiKeyConfigurationError) as exc:
            if isinstance(exc, ValueError):
                logger.error("Invalid API key rotation configuration")
            else:
                logger.warning("CodeBuddy API key file is unavailable")
            return [], False

    async def reload(self) -> Dict[str, object]:
        """Force a TXT file reload and merge the runtime state."""
        keys, source_available = await asyncio.to_thread(self._read_keys)
        async with self._lock:
            if not source_available:
                return {
                    "loaded": len(self._states),
                    "added": 0,
                    "removed": 0,
                    "available": bool(self._states),
                    "reloaded": False,
                }
            existing = {state.key_id: state for state in self._states}
            new_states = []
            added = 0
            for key in keys:
                key_id = self._key_id(key)
                state = existing.get(key_id)
                if state is None:
                    state = _ApiKeyState(
                        key=key,
                        key_id=key_id,
                        masked_key=self.mask_key(key),
                    )
                    added += 1
                new_states.append(state)

            new_ids = {state.key_id for state in new_states}
            removed = sum(1 for state in self._states if state.key_id not in new_ids)
            self._states = new_states
            self._cursor = self._cursor % len(self._states) if self._states else 0
            logger.info(
                "CodeBuddy API key pool reloaded: loaded=%d, added=%d, removed=%d",
                len(self._states),
                added,
                removed,
            )
            return {
                "loaded": len(self._states),
                "added": added,
                "removed": removed,
                "available": bool(self._states),
            }

    async def start_periodic_reload(self) -> None:
        """Perform the initial load and start configured background reloads."""
        await self.reload()
        interval = self._get_reload_interval()
        if interval <= 0 or (self._reload_task and not self._reload_task.done()):
            return
        self._reload_task = asyncio.create_task(
            self._periodic_reload_loop(), name="codebuddy-api-key-reload"
        )

    async def _periodic_reload_loop(self) -> None:
        try:
            while True:
                interval = self._get_reload_interval()
                if interval <= 0:
                    return
                await asyncio.sleep(interval)
                await self.reload()
        except asyncio.CancelledError:
            raise

    async def stop_periodic_reload(self) -> None:
        task = self._reload_task
        self._reload_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def eligible_count(self) -> int:
        async with self._lock:
            self._refresh_expired_cooldowns(self._clock())
            return sum(1 for state in self._states if state.status == "active")

    async def total_count(self) -> int:
        async with self._lock:
            return len(self._states)

    async def acquire(self, excluded_ids: Optional[Set[str]] = None) -> Optional[ApiKeySelection]:
        """Select a key not yet tried by this request using round-robin."""
        excluded_ids = excluded_ids or set()
        async with self._lock:
            now = self._clock()
            self._refresh_expired_cooldowns(now)
            if not self._states:
                return None

            for offset in range(len(self._states)):
                index = (self._cursor + offset) % len(self._states)
                state = self._states[index]
                if state.status != "active" or state.key_id in excluded_ids:
                    continue

                self._cursor = (index + 1) % len(self._states)
                state.request_count += 1
                state.last_used_at = now
                return ApiKeySelection(
                    key=state.key,
                    key_id=state.key_id,
                    masked_key=state.masked_key,
                )
            return None

    async def mark_success(self, key_id: str) -> None:
        async with self._lock:
            state = self._find_state(key_id)
            if state is not None:
                state.last_error = None

    async def mark_invalid(self, key_id: str) -> None:
        await self._mark_error(key_id, "invalid", "HTTP 401")

    async def mark_cooldown(self, key_id: str, status_code: int) -> None:
        async with self._lock:
            state = self._find_state(key_id)
            if state is None:
                return
            state.status = "cooldown"
            state.error_count += 1
            state.cooldown_until = self._clock() + self._get_cooldown_seconds()
            state.last_error = f"HTTP {status_code}"

    async def mark_transient_error(self, key_id: str, error_code: str) -> None:
        await self._mark_error(key_id, "active", error_code)

    async def _mark_error(self, key_id: str, status: str, error_code: str) -> None:
        async with self._lock:
            state = self._find_state(key_id)
            if state is None:
                return
            state.status = status
            state.error_count += 1
            state.cooldown_until = None
            state.last_error = error_code

    def _find_state(self, key_id: str) -> Optional[_ApiKeyState]:
        return next((state for state in self._states if state.key_id == key_id), None)

    @staticmethod
    def _format_timestamp(value: Optional[float]) -> Optional[str]:
        if value is None:
            return None
        return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()

    async def get_status(self) -> Dict[str, object]:
        """Return safe status data without raw keys or file paths."""
        async with self._lock:
            self._refresh_expired_cooldowns(self._clock())
            return {
                "keys": [
                    {
                        "masked_key": state.masked_key,
                        "status": state.status,
                        "request_count": state.request_count,
                        "error_count": state.error_count,
                        "last_used_at": self._format_timestamp(state.last_used_at),
                        "cooldown_until": self._format_timestamp(state.cooldown_until),
                    }
                    for state in self._states
                ],
            }


codebuddy_api_key_manager = CodeBuddyApiKeyManager()
