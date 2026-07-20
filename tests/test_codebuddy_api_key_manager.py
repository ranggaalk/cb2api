import asyncio
import logging

import pytest

from src.codebuddy_api_key_manager import CodeBuddyApiKeyManager


@pytest.fixture
def key_file(tmp_path):
    path = tmp_path / "keys.txt"
    path.write_text(
        "\n# comment\n  alpha-secret-0001  \n\nbeta-secret-0002\nalpha-secret-0001\n",
        encoding="utf-8",
    )
    return path


@pytest.mark.asyncio
async def test_parsing_dedup_and_round_robin(key_file):
    manager = CodeBuddyApiKeyManager(str(key_file), reload_interval=0)
    result = await manager.reload()

    assert result == {"loaded": 2, "added": 2, "removed": 0, "available": True}
    first = await manager.acquire()
    second = await manager.acquire()
    third = await manager.acquire()

    assert [first.key, second.key, third.key] == [
        "alpha-secret-0001",
        "beta-secret-0002",
        "alpha-secret-0001",
    ]


@pytest.mark.asyncio
async def test_concurrent_selection_is_even(key_file):
    manager = CodeBuddyApiKeyManager(str(key_file), reload_interval=0)
    await manager.reload()

    selections = await asyncio.gather(*(manager.acquire() for _ in range(20)))
    keys = [selection.key for selection in selections]
    assert keys.count("alpha-secret-0001") == 10
    assert keys.count("beta-secret-0002") == 10


@pytest.mark.asyncio
async def test_status_cooldown_invalid_and_no_raw_key(key_file, caplog):
    now = [1_700_000_000.0]
    manager = CodeBuddyApiKeyManager(
        str(key_file), cooldown_seconds=10, reload_interval=0, clock=lambda: now[0]
    )
    with caplog.at_level(logging.DEBUG):
        await manager.reload()
        first = await manager.acquire()
        second = await manager.acquire()
        await manager.mark_invalid(first.key_id)
        await manager.mark_cooldown(second.key_id, 429)
        status = await manager.get_status()

    serialized = str(status) + caplog.text
    assert "alpha-secret-0001" not in serialized
    assert "beta-secret-0002" not in serialized
    assert [item["status"] for item in status["keys"]] == ["invalid", "cooldown"]
    assert status["keys"][0]["request_count"] == 1
    assert status["keys"][0]["error_count"] == 1
    assert set(status["keys"][1]) == {
        "masked_key",
        "status",
        "request_count",
        "error_count",
        "last_used_at",
        "cooldown_until",
    }
    assert await manager.eligible_count() == 0

    now[0] += 11
    assert await manager.eligible_count() == 1


@pytest.mark.asyncio
async def test_hot_reload_preserves_state_and_removes_key(key_file):
    manager = CodeBuddyApiKeyManager(str(key_file), reload_interval=0)
    await manager.reload()
    first = await manager.acquire()
    await manager.mark_invalid(first.key_id)

    key_file.write_text("alpha-secret-0001\ngamma-secret-0003\n", encoding="utf-8")
    result = await manager.reload()
    status = await manager.get_status()

    assert result == {"loaded": 2, "added": 1, "removed": 1, "available": True}
    assert [item["status"] for item in status["keys"]] == ["invalid", "active"]
    selections = [await manager.acquire() for _ in range(4)]
    assert {selection.key for selection in selections if selection} == {"gamma-secret-0003"}
    assert "beta-secret-0002" not in {selection.key for selection in selections if selection}


@pytest.mark.asyncio
async def test_periodic_hot_reload_adds_key_without_restart(tmp_path):
    path = tmp_path / "keys.txt"
    path.write_text("first-secret-0001\n", encoding="utf-8")
    manager = CodeBuddyApiKeyManager(str(path), reload_interval=1)
    await manager.start_periodic_reload()
    try:
        path.write_text("first-secret-0001\nsecond-secret-0002\n", encoding="utf-8")
        await asyncio.sleep(1.1)
        assert await manager.total_count() == 2
    finally:
        await manager.stop_periodic_reload()


@pytest.mark.asyncio
async def test_missing_file_keeps_pool_safe_without_exposing_path(tmp_path):
    path = tmp_path / "missing-secret-name.txt"
    manager = CodeBuddyApiKeyManager(str(path), reload_interval=0)
    result = await manager.reload()
    status = await manager.get_status()

    assert result["loaded"] == 0
    assert result["reloaded"] is False
    assert status["keys"] == []
    assert "missing-secret-name" not in str(status)


@pytest.mark.asyncio
async def test_temporary_read_error_does_not_drop_loaded_keys(tmp_path):
    path = tmp_path / "keys.txt"
    path.write_text("stable-secret-0001\n", encoding="utf-8")
    manager = CodeBuddyApiKeyManager(str(path), reload_interval=0)
    await manager.reload()
    path.unlink()

    result = await manager.reload()
    selection = await manager.acquire()

    assert result["reloaded"] is False
    assert selection.key == "stable-secret-0001"
