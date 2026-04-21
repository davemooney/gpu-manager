"""Test fixtures shared across the gpu-manager tests.

The top-level trick: we don't want to actually shell out to docker during
unit tests, so we replace `_run_shell` with an async recorder that logs what
would have been executed and returns a configurable (returncode, stdout,
stderr) tuple. Individual tests can override the return tuple per-command.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Callable, Optional

import pytest
import yaml


# Ensure repo root is importable.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def sample_config() -> dict:
    """A minimal but representative clients config for tests."""
    return {
        "clients": {
            "vllm": {
                "default_vram_mb": 29000,
                "priority": "idle",
                "preemptible": True,
                "cpu_contention": True,
                "pause_command": "docker pause sglang-llm",
                "unpause_command": "docker unpause sglang-llm",
                "stop_command": "docker stop sglang-llm",
            },
            "sglang-vision": {
                "default_vram_mb": 16000,
                "priority": "idle",
                "preemptible": True,
                "cpu_contention": True,
                "pause_command": "docker pause sglang-vision",
                "unpause_command": "docker unpause sglang-vision",
                "stop_command": "docker stop sglang-vision",
            },
            "forma-avatar": {
                "default_vram_mb": 28000,
                "priority": "normal",
                "preemptible": True,
                # Intentionally no cpu_contention — this is the blocker client.
            },
            "benchmarker": {
                "default_vram_mb": 40000,
                "priority": "high",
                "preemptible": False,
                # Intentionally no cpu_contention.
            },
        }
    }


@pytest.fixture
def config_file(tmp_path: Path, sample_config: dict) -> Path:
    """Write `sample_config` to a temp file and return the path."""
    p = tmp_path / "clients.yaml"
    p.write_text(yaml.safe_dump(sample_config))
    return p


class ShellRecorder:
    """Stand-in for `_run_shell` — records every invocation and returns a
    configurable response. Default is (0, '', '')."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._responses: dict[str, tuple[int, str, str]] = {}
        self._default: tuple[int, str, str] = (0, "", "")
        self._raise: dict[str, BaseException] = {}

    def set_default(self, rc: int, stdout: str = "", stderr: str = "") -> None:
        self._default = (rc, stdout, stderr)

    def set_response(self, cmd_substr: str, rc: int, stdout: str = "", stderr: str = "") -> None:
        """Set a specific response for any command containing `cmd_substr`."""
        self._responses[cmd_substr] = (rc, stdout, stderr)

    def set_raise(self, cmd_substr: str, exc: BaseException) -> None:
        """Force `_run_shell` to raise `exc` when the command contains cmd_substr."""
        self._raise[cmd_substr] = exc

    async def __call__(self, cmd: str, timeout: float = 5.0) -> tuple[int, str, str]:
        self.calls.append(cmd)
        for substr, exc in self._raise.items():
            if substr in cmd:
                raise exc
        for substr, resp in self._responses.items():
            if substr in cmd:
                return resp
        return self._default

    def commands_matching(self, substr: str) -> list[str]:
        return [c for c in self.calls if substr in c]


@pytest.fixture
def gpu_manager_module(monkeypatch, config_file: Path):
    """Import gpu_manager with CONFIG_PATH pointed at our temp file.

    Also:
    - Patches `_run_shell` to a ShellRecorder on the module.
    - Patches `get_gpu_vram` / `total_free_vram` so VRAM is effectively infinite.
    - Resets module-level mutable state between tests.

    Yields a tuple (module, recorder).
    """
    monkeypatch.setenv("GPU_MANAGER_CONFIG", str(config_file))

    # Fresh import per-test so the module picks up the env var.
    if "gpu_manager" in sys.modules:
        del sys.modules["gpu_manager"]
    import gpu_manager  # type: ignore

    # Reset mutable state.
    gpu_manager.active_leases.clear()
    gpu_manager.stopped_services.clear()
    gpu_manager.wait_queue.clear()
    gpu_manager.paused_by.clear()
    gpu_manager.preemption_log.clear()

    recorder = ShellRecorder()

    async def fake_run_shell(cmd: str, timeout: float = 5.0) -> tuple[int, str, str]:
        return await recorder(cmd, timeout)

    monkeypatch.setattr(gpu_manager, "_run_shell", fake_run_shell)

    # Effectively infinite VRAM.
    monkeypatch.setattr(gpu_manager, "get_gpu_vram", lambda: [
        {"index": 0, "total_mb": 32000, "used_mb": 0, "free_mb": 32000},
        {"index": 1, "total_mb": 32000, "used_mb": 0, "free_mb": 32000},
    ])
    monkeypatch.setattr(gpu_manager, "total_free_vram", lambda: 64000)

    # Don't actually shell out for start/stop in the acquire path.
    monkeypatch.setattr(gpu_manager, "start_service", lambda name, cfg: True)
    monkeypatch.setattr(gpu_manager, "stop_service", lambda name, cfg: True)
    monkeypatch.setattr(gpu_manager, "is_service_active", lambda name, cfg: False)
    monkeypatch.setattr(gpu_manager, "write_model_env", lambda name, model, cfg: True)

    async def fake_health(url, timeout=3.0):
        return True
    monkeypatch.setattr(gpu_manager, "_check_health", fake_health)

    yield gpu_manager, recorder
