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
