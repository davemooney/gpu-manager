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


# ---------------------------------------------------------------------------
# WP-102-03: preempt_cpu_contention_clients dispatcher (stop-or-pause branch)
# ---------------------------------------------------------------------------


class TestPreemptDispatcher:
    """Branch tests for `preempt_cpu_contention_clients`: heavy_ram clients
    get `stop_command`, plain cpu_contention clients keep the legacy pause
    path, and already-preempted clients ref-count without re-firing shell.
    """

    @pytest.mark.asyncio
    async def test_heavy_client_gets_stopped(self, clean_state, monkeypatch, write_config):
        monkeypatch.setattr(gm, "HEAVY_RAM_PREEMPTION_ENABLED", True)
        calls: list[str] = []

        async def fake_shell(cmd, timeout=5.0):
            calls.append(cmd)
            return (0, "", "")

        monkeypatch.setattr(gm, "_run_shell", fake_shell)
        write_config({
            "clients": {
                "vllm": {
                    "cpu_contention": True,
                    "preemptible": True,
                    "heavy_ram": True,
                    "start_command": "launch",
                    "stop_command": "docker stop vllm",
                    "pause_command": "docker pause vllm",
                    "unpause_command": "docker unpause vllm",
                }
            }
        })
        config = gm.load_config()
        await gm.preempt_cpu_contention_clients("lease-abc", "forma-avatar", config)

        assert any("docker stop vllm" in c for c in calls)
        # Should NOT have issued a pause — heavy path took over.
        assert not any("docker pause vllm" in c for c in calls)
        assert gm.preempted_state["vllm"] == "stopped"
        assert "vllm" in gm.preempted_since
        assert "vllm" in gm.paused_by["lease-abc"]
        # Preemption log records a `stop` action for the client.
        assert any(
            e.get("action") == "stop" and e.get("client") == "vllm"
            for e in gm.preemption_log
        )

    @pytest.mark.asyncio
    async def test_non_heavy_client_gets_paused(self, clean_state, monkeypatch, write_config):
        monkeypatch.setattr(gm, "HEAVY_RAM_PREEMPTION_ENABLED", True)
        calls: list[str] = []

        async def fake_shell(cmd, timeout=5.0):
            calls.append(cmd)
            return (0, "", "")

        monkeypatch.setattr(gm, "_run_shell", fake_shell)
        write_config({
            "clients": {
                "lite": {
                    "cpu_contention": True,
                    "preemptible": True,
                    "pause_command": "docker pause lite",
                    "unpause_command": "docker unpause lite",
                    "stop_command": "docker stop lite",
                }
            }
        })
        config = gm.load_config()
        await gm.preempt_cpu_contention_clients("lease-abc", "forma-avatar", config)

        assert any("docker pause lite" in c for c in calls)
        assert not any("docker stop lite" in c for c in calls)
        assert gm.preempted_state["lite"] == "paused"
        assert "lite" in gm.preempted_since
        assert "lite" in gm.paused_by["lease-abc"]

    @pytest.mark.asyncio
    async def test_already_stopped_skips_action(self, clean_state, monkeypatch, write_config):
        monkeypatch.setattr(gm, "HEAVY_RAM_PREEMPTION_ENABLED", True)
        calls: list[str] = []

        async def fake_shell(cmd, timeout=5.0):
            calls.append(cmd)
            return (0, "", "")

        monkeypatch.setattr(gm, "_run_shell", fake_shell)
        write_config({
            "clients": {
                "vllm": {
                    "cpu_contention": True,
                    "preemptible": True,
                    "heavy_ram": True,
                    "start_command": "launch",
                    "stop_command": "docker stop vllm",
                }
            }
        })
        config = gm.load_config()
        gm.preempted_state["vllm"] = "stopped"  # pre-existing blocker

        await gm.preempt_cpu_contention_clients("lease-xyz", "forma-avatar", config)

        assert calls == []  # no shell called — short-circuited on pre-existing state
        assert "vllm" in gm.paused_by["lease-xyz"]
        # And preemption_log records stop_refcount for the ref-count path.
        assert any(
            e.get("action") == "stop_refcount" and e.get("client") == "vllm"
            for e in gm.preemption_log
        )

    @pytest.mark.asyncio
    async def test_already_restarting_skips_action(self, clean_state, monkeypatch, write_config):
        """A client currently in the `restarting` window (cooldown→restart
        coalesce) also short-circuits — we still want to enrol the new
        blocker so it can cancel the restart on its next release."""
        monkeypatch.setattr(gm, "HEAVY_RAM_PREEMPTION_ENABLED", True)
        calls: list[str] = []

        async def fake_shell(cmd, timeout=5.0):
            calls.append(cmd)
            return (0, "", "")

        monkeypatch.setattr(gm, "_run_shell", fake_shell)
        write_config({
            "clients": {
                "vllm": {
                    "cpu_contention": True,
                    "preemptible": True,
                    "heavy_ram": True,
                    "start_command": "launch",
                    "stop_command": "docker stop vllm",
                }
            }
        })
        config = gm.load_config()
        gm.preempted_state["vllm"] = "restarting"

        await gm.preempt_cpu_contention_clients("lease-new", "forma-avatar", config)

        assert calls == []
        assert "vllm" in gm.paused_by["lease-new"]
        assert any(
            e.get("action") == "pause_refcount" and e.get("client") == "vllm"
            for e in gm.preemption_log
        )

    @pytest.mark.asyncio
    async def test_mixed_heavy_and_light(self, clean_state, monkeypatch, write_config):
        """Config with both a heavy_ram and a plain cpu_contention client:
        each takes its own branch — heavy gets stopped, light gets paused."""
        monkeypatch.setattr(gm, "HEAVY_RAM_PREEMPTION_ENABLED", True)
        calls: list[str] = []

        async def fake_shell(cmd, timeout=5.0):
            calls.append(cmd)
            return (0, "", "")

        monkeypatch.setattr(gm, "_run_shell", fake_shell)
        write_config({
            "clients": {
                "vllm": {
                    "cpu_contention": True,
                    "preemptible": True,
                    "heavy_ram": True,
                    "start_command": "launch",
                    "stop_command": "docker stop vllm",
                    "pause_command": "docker pause vllm",
                },
                "sglang-vision": {
                    "cpu_contention": True,
                    "preemptible": True,
                    # No heavy_ram flag — pure light path.
                    "pause_command": "docker pause sglang-vision",
                    "unpause_command": "docker unpause sglang-vision",
                    "stop_command": "docker stop sglang-vision",
                },
            }
        })
        config = gm.load_config()
        await gm.preempt_cpu_contention_clients("lease-mix", "forma-avatar", config)

        # Heavy path hit stop, light path hit pause — no cross-contamination.
        assert any("docker stop vllm" in c for c in calls)
        assert any("docker pause sglang-vision" in c for c in calls)
        assert not any("docker pause vllm" in c for c in calls)
        assert not any("docker stop sglang-vision" in c for c in calls)

        assert gm.preempted_state["vllm"] == "stopped"
        assert gm.preempted_state["sglang-vision"] == "paused"
        assert {"vllm", "sglang-vision"} <= gm.paused_by["lease-mix"]

    @pytest.mark.asyncio
    async def test_feature_flag_off_falls_back_to_pause_for_heavy(
        self, clean_state, monkeypatch, write_config
    ):
        """Flag OFF + heavy_ram=True client should take the LIGHT path
        (pause) so operators can instantly roll back without a code change."""
        monkeypatch.setattr(gm, "HEAVY_RAM_PREEMPTION_ENABLED", False)
        calls: list[str] = []

        async def fake_shell(cmd, timeout=5.0):
            calls.append(cmd)
            return (0, "", "")

        monkeypatch.setattr(gm, "_run_shell", fake_shell)
        write_config({
            "clients": {
                "vllm": {
                    "cpu_contention": True,
                    "preemptible": True,
                    "heavy_ram": True,
                    "start_command": "launch",
                    "stop_command": "docker stop vllm",
                    "pause_command": "docker pause vllm",
                    "unpause_command": "docker unpause vllm",
                }
            }
        })
        config = gm.load_config()
        await gm.preempt_cpu_contention_clients("lease-flag-off", "forma-avatar", config)

        # With flag OFF the heavy branch is skipped; LIGHT path fires pause.
        assert any("docker pause vllm" in c for c in calls)
        assert not any("docker stop vllm" in c for c in calls)
        assert gm.preempted_state["vllm"] == "paused"
        assert "vllm" in gm.paused_by["lease-flag-off"]

    @pytest.mark.asyncio
    async def test_heavy_invalid_config_skipped_not_crashed(
        self, clean_state, monkeypatch, write_config
    ):
        """A heavy_ram client missing a start_command is flagged invalid at
        config load and must NOT be stopped — preempt_cpu_contention_clients
        skips it entirely (no shell, no ledger entry)."""
        monkeypatch.setattr(gm, "HEAVY_RAM_PREEMPTION_ENABLED", True)
        calls: list[str] = []

        async def fake_shell(cmd, timeout=5.0):
            calls.append(cmd)
            return (0, "", "")

        monkeypatch.setattr(gm, "_run_shell", fake_shell)
        write_config({
            "clients": {
                "broken": {
                    "cpu_contention": True,
                    "preemptible": True,
                    "heavy_ram": True,
                    # No start_command — load_config sets _heavy_ram_invalid.
                    "stop_command": "docker stop broken",
                    "pause_command": "docker pause broken",
                }
            }
        })
        config = gm.load_config()
        assert config["clients"]["broken"]["_heavy_ram_invalid"]

        await gm.preempt_cpu_contention_clients("lease-bad", "forma-avatar", config)

        # Heavy branch is gated out; LIGHT branch still runs pause_command.
        # The invariant we MUST preserve is: no `docker stop broken` fires, so
        # we don't strand a client with no way to restart.
        assert not any("docker stop broken" in c for c in calls)
        # Light fallback still paused it (preserves forward progress).
        assert any("docker pause broken" in c for c in calls)
        assert gm.preempted_state.get("broken") == "paused"

    @pytest.mark.asyncio
    async def test_heavy_stop_failure_is_best_effort(
        self, clean_state, monkeypatch, write_config
    ):
        """A failing stop_command must not crash the dispatcher; the client
        is NOT enrolled on the blocker (nothing to reverse later), and the
        failure is recorded in preemption_log."""
        monkeypatch.setattr(gm, "HEAVY_RAM_PREEMPTION_ENABLED", True)

        async def failing_shell(cmd, timeout=5.0):
            return (1, "", "Error: no such container")

        monkeypatch.setattr(gm, "_run_shell", failing_shell)
        write_config({
            "clients": {
                "vllm": {
                    "cpu_contention": True,
                    "preemptible": True,
                    "heavy_ram": True,
                    "start_command": "launch",
                    "stop_command": "docker stop vllm",
                }
            }
        })
        config = gm.load_config()
        await gm.preempt_cpu_contention_clients("lease-fail", "forma-avatar", config)

        # State left clean: nothing recorded as stopped, no ledger entry.
        assert "vllm" not in gm.preempted_state
        assert "vllm" not in gm.paused_by.get("lease-fail", set())
        # But the failure was recorded.
        assert any(
            e.get("action") == "stop_failed" and e.get("client") == "vllm"
            for e in gm.preemption_log
        )

    @pytest.mark.asyncio
    async def test_blocker_client_skipped_even_if_cpu_contention(
        self, clean_state, monkeypatch, write_config
    ):
        """The blocker itself must not be preempted even if it's flagged
        cpu_contention — protects against self-preemption."""
        monkeypatch.setattr(gm, "HEAVY_RAM_PREEMPTION_ENABLED", True)
        calls: list[str] = []

        async def fake_shell(cmd, timeout=5.0):
            calls.append(cmd)
            return (0, "", "")

        monkeypatch.setattr(gm, "_run_shell", fake_shell)
        write_config({
            "clients": {
                "vllm": {
                    "cpu_contention": True,
                    "preemptible": True,
                    "heavy_ram": True,
                    "start_command": "launch",
                    "stop_command": "docker stop vllm",
                    "pause_command": "docker pause vllm",
                },
            }
        })
        config = gm.load_config()
        await gm.preempt_cpu_contention_clients("lease-self", "vllm", config)

        assert calls == []
        assert "vllm" not in gm.preempted_state
        assert gm.paused_by.get("lease-self", set()) == set()
