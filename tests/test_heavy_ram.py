"""Unit tests for the heavy_ram preemption foundation (avatar#102, WP-102-01).

These tests cover the module-level `preempted_state` + `preempted_since`
ledgers, the config-load validation that guards against `heavy_ram: true`
clients without a `start_command`, and the loopback-gated
`GET /debug/preempted_state` endpoint.
"""

from __future__ import annotations

import sys

import pytest
from fastapi.testclient import TestClient

import gpu_manager as gm


@pytest.fixture(autouse=True)
def _pin_gpu_manager_module():
    """Ensure `sys.modules['gpu_manager']` points at the same module object
    referenced by our module-level `gm` binding.

    The `gpu_manager_module` fixture used by other test files does
    `del sys.modules['gpu_manager']; import gpu_manager`, which leaves
    `sys.modules` holding a fresh module instance while our `gm` here
    still points at the original one collected with this test file.
    Conftest fixtures (`clean_state`, `write_config`, `loopback_client`)
    resolve the module via `import gpu_manager`, so without pinning they'd
    act on the wrong instance and state wouldn't cross over. This autouse
    fixture re-pins `sys.modules` to our `gm` for every test in this file.
    """
    previous = sys.modules.get("gpu_manager")
    sys.modules["gpu_manager"] = gm
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop("gpu_manager", None)
        else:
            sys.modules["gpu_manager"] = previous


class TestFoundation:
    def test_preempted_state_initialised_empty(self, clean_state):
        assert gm.preempted_state == {}
        assert gm.preempted_since == {}

    def test_valid_heavy_ram_config_passes(self, monkeypatch, write_config):
        write_config({
            "clients": {
                "vllm": {
                    "cpu_contention": True,
                    "preemptible": True,
                    "heavy_ram": True,
                    "start_command": "bash /tmp/sglang-launch.sh",
                    "stop_command": "docker stop sglang-llm",
                    "pause_command": "docker pause sglang-llm",
                    "unpause_command": "docker unpause sglang-llm",
                    "description": "x",
                }
            }
        })
        cfg = gm.load_config()
        assert not cfg["clients"]["vllm"].get("_heavy_ram_invalid")

    def test_heavy_ram_without_start_command_is_excluded(self, caplog, write_config):
        write_config({
            "clients": {
                "broken": {
                    "cpu_contention": True,
                    "preemptible": True,
                    "heavy_ram": True,
                    # no start_command
                    "stop_command": "true",
                    "description": "x",
                }
            }
        })
        with caplog.at_level("CRITICAL"):
            cfg = gm.load_config()
        assert cfg["clients"]["broken"]["_heavy_ram_invalid"]
        assert any(
            "heavy_ram=true but no start_command" in r.getMessage()
            for r in caplog.records
        )

    def test_debug_preempted_state_requires_loopback(self):
        client = TestClient(gm.app)
        # TestClient default host is "testclient" (non-loopback)
        resp = client.get("/debug/preempted_state")
        assert resp.status_code == 403

    def test_debug_preempted_state_returns_snapshot(self, loopback_client, clean_state):
        gm.preempted_state["vllm"] = "paused"
        gm.preempted_since["vllm"] = 1234.5
        resp = loopback_client.get("/debug/preempted_state")
        assert resp.status_code == 200
        body = resp.json()
        assert body["preempted_state"] == {"vllm": "paused"}
        assert body["preempted_since"] == {"vllm": 1234.5}
        assert "heavy_ram_enabled" in body


# ---------------------------------------------------------------------------
# WP-102-02: schedule_restart + _delayed_restart
# ---------------------------------------------------------------------------


import asyncio  # noqa: E402


@pytest.fixture
def _clean_restart_tasks():
    """Cancel and discard any residual scheduled restart tasks on the module.

    Complements `clean_state` from conftest: that fixture doesn't know about
    `restart_tasks` (added in WP-102-02), so clean it here.
    """
    # Best-effort cancel anything left from a previous test.
    for task in list(gm.restart_tasks.values()):
        if not task.done():
            task.cancel()
    gm.restart_tasks.clear()
    yield
    for task in list(gm.restart_tasks.values()):
        if not task.done():
            task.cancel()
    gm.restart_tasks.clear()


class TestRestartScheduler:
    """Unit tests for `schedule_restart` + `_delayed_restart`."""

    @pytest.mark.asyncio
    async def test_schedule_restart_cancels_existing(
        self, clean_state, _clean_restart_tasks, monkeypatch
    ):
        """A second schedule_restart for the same client cancels the first."""
        monkeypatch.setattr(gm, "HEAVY_RAM_PREEMPTION_ENABLED", True)

        async def fake_shell(cmd, timeout=5.0):
            return (0, "", "")

        async def fake_health(url, timeout=3.0):
            return True

        monkeypatch.setattr(gm, "_run_shell", fake_shell)
        monkeypatch.setattr(gm, "_check_health", fake_health)
        monkeypatch.setattr(
            gm,
            "load_config",
            lambda: {
                "clients": {
                    "x": {
                        "start_command": "true",
                        "health_check": "http://x",
                        "startup_seconds": 5,
                    }
                }
            },
        )

        gm.schedule_restart("x", delay_s=10)
        t1 = gm.restart_tasks["x"]
        gm.schedule_restart("x", delay_s=10)
        t2 = gm.restart_tasks["x"]
        assert t1 is not t2
        # Give the event loop a tick so the cancellation propagates.
        try:
            await t1
        except asyncio.CancelledError:
            pass
        assert t1.cancelled() or t1.done()

        t2.cancel()
        try:
            await t2
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_happy_path_restarts_and_health_ok(
        self, clean_state, _clean_restart_tasks, monkeypatch
    ):
        """Start command runs, health comes up, preempted_state is cleared."""
        monkeypatch.setattr(gm, "HEAVY_RAM_PREEMPTION_ENABLED", True)
        calls: list[str] = []

        async def fake_shell(cmd, timeout=5.0):
            calls.append(cmd)
            return (0, "", "")

        async def fake_health(url, timeout=3.0):
            return True

        monkeypatch.setattr(gm, "_run_shell", fake_shell)
        monkeypatch.setattr(gm, "_check_health", fake_health)
        monkeypatch.setattr(
            gm,
            "load_config",
            lambda: {
                "clients": {
                    "x": {
                        "start_command": "/tmp/start.sh",
                        "health_check": "http://x",
                        "startup_seconds": 5,
                    }
                }
            },
        )

        gm.preempted_state["x"] = "stopped"
        gm.preempted_since["x"] = 0.0

        gm.schedule_restart("x", delay_s=0)
        await asyncio.wait_for(gm.restart_tasks["x"], timeout=3.0)

        assert "x" not in gm.preempted_state
        assert "x" not in gm.preempted_since
        assert "x" not in gm.restart_tasks
        assert any("/tmp/start.sh" in c for c in calls)
        # preemption_log should record the ok action.
        assert any(
            e.get("action") == "restart_ok" and e.get("client") == "x"
            for e in gm.preemption_log
        )

    @pytest.mark.asyncio
    async def test_feature_flag_off_is_noop(
        self, clean_state, _clean_restart_tasks, monkeypatch
    ):
        """With the flag disabled, schedule_restart does nothing."""
        monkeypatch.setattr(gm, "HEAVY_RAM_PREEMPTION_ENABLED", False)
        gm.schedule_restart("x", delay_s=0)
        assert "x" not in gm.restart_tasks

    @pytest.mark.asyncio
    async def test_cancel_before_fire_leaves_state_untouched(
        self, clean_state, _clean_restart_tasks, monkeypatch
    ):
        """Cancelling a pending (still-sleeping) task must not touch state."""
        monkeypatch.setattr(gm, "HEAVY_RAM_PREEMPTION_ENABLED", True)

        gm.schedule_restart("x", delay_s=60)
        task = gm.restart_tasks["x"]
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert "x" not in gm.preempted_state
        assert "x" not in gm.preempted_since

    @pytest.mark.asyncio
    async def test_start_command_failure_retries_then_succeeds(
        self, clean_state, _clean_restart_tasks, monkeypatch
    ):
        """First start_command returns non-zero, retry succeeds, state cleared.

        We patch the backoffs to near-zero so the test runs in < 1 s.
        """
        monkeypatch.setattr(gm, "HEAVY_RAM_PREEMPTION_ENABLED", True)

        # Speed up retries for the test.
        original_delayed = gm._delayed_restart

        async def fast_delayed(name: str, delay_s: int) -> None:
            # Shrink the backoff list by monkey-patching in a wrapper that
            # calls the real function but with a pre-patched module.
            await original_delayed(name, delay_s)

        # Patch asyncio.sleep used inside _delayed_restart to be instant for
        # sleeps >= 5 s (retry backoffs) but keep the 2 s health poll honest.
        real_sleep = asyncio.sleep

        async def fast_sleep(s):
            if s >= 5:
                return
            await real_sleep(0)

        monkeypatch.setattr(gm.asyncio, "sleep", fast_sleep)

        attempts = {"n": 0}

        async def fake_shell(cmd, timeout=5.0):
            attempts["n"] += 1
            if attempts["n"] == 1:
                return (1, "", "boom")
            return (0, "", "")

        async def fake_health(url, timeout=3.0):
            return True

        monkeypatch.setattr(gm, "_run_shell", fake_shell)
        monkeypatch.setattr(gm, "_check_health", fake_health)
        monkeypatch.setattr(
            gm,
            "load_config",
            lambda: {
                "clients": {
                    "x": {
                        "start_command": "/tmp/start.sh",
                        "health_check": "http://x",
                        "startup_seconds": 5,
                    }
                }
            },
        )

        gm.preempted_state["x"] = "stopped"
        gm.preempted_since["x"] = 0.0

        gm.schedule_restart("x", delay_s=0)
        await asyncio.wait_for(gm.restart_tasks["x"], timeout=3.0)

        assert attempts["n"] >= 2
        assert "x" not in gm.preempted_state
        assert any(
            e.get("action") == "restart_failed" and e.get("client") == "x"
            for e in gm.preemption_log
        )
        assert any(
            e.get("action") == "restart_ok" and e.get("client") == "x"
            for e in gm.preemption_log
        )

    @pytest.mark.asyncio
    async def test_exhausted_budget_marks_start_failed(
        self, clean_state, _clean_restart_tasks, monkeypatch
    ):
        """Every start attempt fails → final state is `start_failed`.

        We both collapse the backoff sleeps and shrink the 1 h deadline so the
        test finishes in under a second.
        """
        monkeypatch.setattr(gm, "HEAVY_RAM_PREEMPTION_ENABLED", True)

        real_sleep = asyncio.sleep

        async def fast_sleep(s):
            if s >= 5:
                return
            await real_sleep(0)

        monkeypatch.setattr(gm.asyncio, "sleep", fast_sleep)

        # Fake time.time so the 1 h deadline is reached after a few iterations.
        # We let the first attempt proceed, then jump forward past the deadline
        # so the while-loop exits via the "out of retries" branch.
        t0 = [1000.0]

        def fake_time():
            return t0[0]

        monkeypatch.setattr(gm.time, "time", fake_time)

        attempts = {"n": 0}

        async def fake_shell(cmd, timeout=5.0):
            attempts["n"] += 1
            # Advance time each retry so the deadline (3600 s) is exceeded
            # after a small number of attempts.
            t0[0] += 2000
            return (1, "", "boom")

        async def fake_health(url, timeout=3.0):
            return False

        monkeypatch.setattr(gm, "_run_shell", fake_shell)
        monkeypatch.setattr(gm, "_check_health", fake_health)
        monkeypatch.setattr(
            gm,
            "load_config",
            lambda: {
                "clients": {
                    "x": {
                        "start_command": "/tmp/start.sh",
                        "health_check": "http://x",
                        "startup_seconds": 5,
                    }
                }
            },
        )

        gm.schedule_restart("x", delay_s=0)
        await asyncio.wait_for(gm.restart_tasks["x"], timeout=3.0)

        assert gm.preempted_state.get("x") == "start_failed"
        assert "x" in gm.preempted_since
        assert "x" not in gm.restart_tasks
        assert any(
            e.get("action") == "start_failed" and e.get("client") == "x"
            for e in gm.preemption_log
        )
