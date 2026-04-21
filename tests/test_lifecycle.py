"""Unit tests for the Tier-3 lifecycle callback router (issue avatar#100).

These tests cover:
  - Envelope validation (registered source, protocol version, valid state)
  - Concurrent fan-out to peer callback_urls with per-target timeout
  - Legacy fallback (pause/unpause) for cpu_contention clients with no callback
  - Ref-counting between real `normal` leases AND lifecycle `generating` signals
  - Admin endpoints: /lifecycle_log and /lifecycle/state (loopback-only)

The HTTP POSTs are mocked by patching `_post_lifecycle_callback` on the
module — this keeps the tests deterministic while still exercising
`asyncio.gather` with realistic per-target latency.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reimport_with_config(monkeypatch, config_path: Path):
    """Reload `gpu_manager` against a different config file."""
    monkeypatch.setenv("GPU_MANAGER_CONFIG", str(config_path))
    if "gpu_manager" in sys.modules:
        del sys.modules["gpu_manager"]
    import gpu_manager  # noqa: E402
    gpu_manager.active_leases.clear()
    gpu_manager.stopped_services.clear()
    gpu_manager.wait_queue.clear()
    gpu_manager.paused_by.clear()
    gpu_manager.preemption_log.clear()
    gpu_manager.lifecycle_log.clear()
    return gpu_manager


@pytest.fixture
def lifecycle_config() -> dict:
    """Config covering all three scenarios the router needs to handle:
       - `forma-avatar` is the producer (has a callback_url).
       - `peer-a` has a callback_url — receives fan-out.
       - `peer-b` has a callback_url — receives fan-out.
       - `sglang-vision` has NO callback_url but cpu_contention=true — legacy fallback.
       - `vllm` has NO callback_url but cpu_contention=true — legacy fallback.
       - `no-cb-no-contention` has no callback_url and no cpu_contention — ignored.
    """
    return {
        "clients": {
            "forma-avatar": {
                "default_vram_mb": 28000,
                "priority": "normal",
                "preemptible": True,
                "callback_url": "http://localhost:8420/gpu/callback",
            },
            "peer-a": {
                "default_vram_mb": 4000,
                "priority": "normal",
                "preemptible": True,
                "callback_url": "http://localhost:9501/gpu/callback",
            },
            "peer-b": {
                "default_vram_mb": 4000,
                "priority": "normal",
                "preemptible": True,
                "callback_url": "http://localhost:9502/gpu/callback",
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
            "vllm": {
                "default_vram_mb": 29000,
                "priority": "idle",
                "preemptible": True,
                "cpu_contention": True,
                "pause_command": "docker pause sglang-llm",
                "unpause_command": "docker unpause sglang-llm",
                "stop_command": "docker stop sglang-llm",
            },
            "no-cb-no-contention": {
                "default_vram_mb": 2000,
                "priority": "idle",
                "preemptible": True,
            },
        }
    }


@pytest.fixture
def lifecycle_config_file(tmp_path: Path, lifecycle_config: dict) -> Path:
    p = tmp_path / "clients.yaml"
    p.write_text(yaml.safe_dump(lifecycle_config))
    return p


class _CallbackRecorder:
    """Mocks `_post_lifecycle_callback`. Records every invocation and returns
    a configurable per-target delivery record."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._delays: dict[str, float] = {}
        self._statuses: dict[str, str] = {}

    def set_delay(self, target: str, seconds: float) -> None:
        self._delays[target] = seconds

    def set_status(self, target: str, status: str, **extra: str) -> None:
        self._statuses[target] = status
        for k, v in extra.items():
            setattr(self, f"_{target}_{k}", v)

    async def __call__(self, target_name, callback_url, envelope, timeout=1.0):
        self.calls.append({
            "target": target_name,
            "callback_url": callback_url,
            "envelope": dict(envelope),
            "timeout": timeout,
        })
        delay = self._delays.get(target_name, 0.0)
        if delay > 0:
            try:
                await asyncio.wait_for(asyncio.sleep(delay), timeout=timeout)
            except asyncio.TimeoutError:
                return {
                    "target": target_name,
                    "callback_url": callback_url,
                    "status": "timeout",
                }
        status = self._statuses.get(target_name, "ok")
        record = {"target": target_name, "callback_url": callback_url, "status": status}
        if status == "ok":
            record["http_status"] = 200
        return record


@pytest.fixture
def lifecycle_module(monkeypatch, lifecycle_config_file: Path):
    """Import gpu_manager with the Tier-3 fixture config + stub out shell calls
    and HTTP callbacks.

    Yields (module, shell_recorder, callback_recorder)."""
    mod = _reimport_with_config(monkeypatch, lifecycle_config_file)

    # Shell recorder — reused shape from conftest.ShellRecorder but inlined so
    # we don't reach across test files.
    shell_calls: list[str] = []
    shell_responses: dict[str, tuple[int, str, str]] = {}

    async def fake_run_shell(cmd: str, timeout: float = 5.0) -> tuple[int, str, str]:
        shell_calls.append(cmd)
        for substr, resp in shell_responses.items():
            if substr in cmd:
                return resp
        return (0, "", "")

    monkeypatch.setattr(mod, "_run_shell", fake_run_shell)
    monkeypatch.setattr(mod, "get_gpu_vram", lambda: [
        {"index": 0, "total_mb": 32000, "used_mb": 0, "free_mb": 32000},
        {"index": 1, "total_mb": 32000, "used_mb": 0, "free_mb": 32000},
    ])
    monkeypatch.setattr(mod, "total_free_vram", lambda: 64000)
    monkeypatch.setattr(mod, "start_service", lambda name, cfg: True)
    monkeypatch.setattr(mod, "stop_service", lambda name, cfg: True)
    monkeypatch.setattr(mod, "is_service_active", lambda name, cfg: False)
    monkeypatch.setattr(mod, "write_model_env", lambda name, model, cfg: True)

    async def fake_health(url, timeout=3.0):
        return True
    monkeypatch.setattr(mod, "_check_health", fake_health)

    callback_recorder = _CallbackRecorder()
    monkeypatch.setattr(mod, "_post_lifecycle_callback", callback_recorder)

    # Expose the shell recorder as a simple object.
    class _ShellHandle:
        calls = shell_calls
        responses = shell_responses

        def matching(self, substr: str) -> list[str]:
            return [c for c in shell_calls if substr in c]
    shell_handle = _ShellHandle()

    yield mod, shell_handle, callback_recorder


def _post_envelope(
    client: TestClient,
    source: str,
    state: str,
    *,
    version: int = 1,
    context: dict | None = None,
) -> "TestClient.Response":
    body = {
        "source": source,
        "state": state,
        "version": version,
        "context": context or {},
    }
    return client.post(f"/lifecycle/{source}", json=body)


# ---------------------------------------------------------------------------
# Envelope validation
# ---------------------------------------------------------------------------

def test_valid_envelope_registered_source_200(lifecycle_module):
    mod, _shell, callbacks = lifecycle_module
    c = TestClient(mod.app)

    r = _post_envelope(c, "forma-avatar", "preflight")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert body["source"] == "forma-avatar"
    assert body["state"] == "preflight"

    # Event landed in the lifecycle log.
    assert len(mod.lifecycle_log) == 1
    entry = list(mod.lifecycle_log)[0]
    assert entry["source"] == "forma-avatar"
    assert entry["state"] == "preflight"
    # Fan-out targets recorded (peer-a, peer-b have callback_urls).
    assert set(entry["targets"]) == {"peer-a", "peer-b"}


def test_unregistered_source_404(lifecycle_module):
    mod, _shell, _cb = lifecycle_module
    c = TestClient(mod.app)

    r = _post_envelope(c, "not-a-real-client", "preflight")
    assert r.status_code == 404
    assert "not-a-real-client" in r.text or "Unknown" in r.text


def test_version_mismatch_400(lifecycle_module):
    mod, _shell, _cb = lifecycle_module
    c = TestClient(mod.app)

    r = _post_envelope(c, "forma-avatar", "preflight", version=2)
    assert r.status_code == 400
    assert "version" in r.text.lower()


def test_invalid_state_422(lifecycle_module):
    mod, _shell, _cb = lifecycle_module
    c = TestClient(mod.app)

    r = _post_envelope(c, "forma-avatar", "exploding")
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------

def test_fanout_delivers_to_all_peers_and_not_source(lifecycle_module):
    """Every registered client with a callback_url EXCEPT the source must
    receive the envelope. The source must NOT receive its own event."""
    mod, _shell, callbacks = lifecycle_module
    c = TestClient(mod.app)

    r = _post_envelope(
        c, "forma-avatar", "generating",
        context={"session_id": "abc123"},
    )
    assert r.status_code == 200

    # Two peers got the callback, and the source did not.
    delivered_to = [call["target"] for call in callbacks.calls]
    assert set(delivered_to) == {"peer-a", "peer-b"}
    assert "forma-avatar" not in delivered_to

    # Envelope preserved (source stamped from URL path, context forwarded).
    for call in callbacks.calls:
        assert call["envelope"]["source"] == "forma-avatar"
        assert call["envelope"]["state"] == "generating"
        assert call["envelope"]["version"] == 1
        assert call["envelope"]["context"] == {"session_id": "abc123"}
        # Router stamps a timestamp if producer didn't set one.
        assert call["envelope"].get("timestamp")


def test_fanout_concurrent_one_slow_target_does_not_block(lifecycle_module):
    """Make peer-a hang 5s (longer than the 1s timeout). peer-b must still get
    delivered promptly; total request time must be bounded by the 1s timeout,
    not the full 5s delay."""
    mod, _shell, callbacks = lifecycle_module
    callbacks.set_delay("peer-a", 5.0)  # exceeds 1s per-target timeout

    c = TestClient(mod.app)
    import time
    t0 = time.monotonic()
    r = _post_envelope(c, "forma-avatar", "generating")
    elapsed = time.monotonic() - t0

    assert r.status_code == 200
    # Must return within ~1.5s (1s timeout + overhead), not 5s.
    assert elapsed < 3.0, f"fan-out took too long: {elapsed:.2f}s"

    body = r.json()
    records = {rec["target"]: rec for rec in body["fanout"]["deliveries"]}
    assert records["peer-a"]["status"] == "timeout"
    assert records["peer-b"]["status"] == "ok"
    assert body["fanout"]["failures"] == 1


# ---------------------------------------------------------------------------
# Legacy fallback (clients without callback_url)
# ---------------------------------------------------------------------------

def test_legacy_fallback_generating_pauses_cpu_contention_clients(lifecycle_module):
    mod, shell, _cb = lifecycle_module
    c = TestClient(mod.app)

    r = _post_envelope(c, "forma-avatar", "generating")
    assert r.status_code == 200

    # sglang-vision + vllm (no callback, cpu_contention=true) were paused.
    pauses = shell.matching("docker pause")
    assert any("sglang-vision" in cmd for cmd in pauses)
    assert any("sglang-llm" in cmd for cmd in pauses)

    # Ledger records under the synthetic lifecycle key.
    blocker_key = mod._lifecycle_blocker_key("forma-avatar")
    assert blocker_key in mod.paused_by
    assert mod.paused_by[blocker_key] == {"sglang-vision", "vllm"}


def test_legacy_fallback_cooldown_unpauses(lifecycle_module):
    mod, shell, _cb = lifecycle_module
    c = TestClient(mod.app)

    # Pause via generating, then unpause via cooldown.
    r1 = _post_envelope(c, "forma-avatar", "generating")
    assert r1.status_code == 200
    shell.calls.clear()

    r2 = _post_envelope(c, "forma-avatar", "cooldown")
    assert r2.status_code == 200

    unpauses = shell.matching("docker unpause")
    assert any("sglang-vision" in cmd for cmd in unpauses)
    assert any("sglang-llm" in cmd for cmd in unpauses)

    # Ledger entry gone.
    blocker_key = mod._lifecycle_blocker_key("forma-avatar")
    assert blocker_key not in mod.paused_by


def test_legacy_fallback_idle_and_preflight_are_noops(lifecycle_module):
    mod, shell, _cb = lifecycle_module
    c = TestClient(mod.app)

    r1 = _post_envelope(c, "forma-avatar", "idle")
    assert r1.status_code == 200
    r2 = _post_envelope(c, "forma-avatar", "preflight")
    assert r2.status_code == 200

    assert shell.matching("docker pause") == []
    assert shell.matching("docker unpause") == []


def test_clients_with_callback_url_bypass_legacy_fallback(lifecycle_module):
    """peer-a and peer-b each have a callback_url, so they should receive the
    fan-out POST and NOT be touched by the legacy pause path (they're not
    cpu_contention anyway, but double-check no shell calls fire for them)."""
    mod, shell, callbacks = lifecycle_module
    c = TestClient(mod.app)

    r = _post_envelope(c, "forma-avatar", "generating")
    assert r.status_code == 200
    # Only cpu_contention clients (sglang-vision, vllm) get pause commands —
    # peer-a and peer-b have no pause_command anyway.
    for cmd in shell.calls:
        assert "peer-a" not in cmd
        assert "peer-b" not in cmd
    # Both peers received their fan-out callbacks.
    delivered = [call["target"] for call in callbacks.calls]
    assert set(delivered) == {"peer-a", "peer-b"}


# ---------------------------------------------------------------------------
# Mixed blockers: real lease + lifecycle signal
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mixed_lease_then_lifecycle_refcount(lifecycle_module):
    """Acquire a real `normal` lease → sglang-vision paused by lease.
    Then receive `generating` from another source → paused by TWO blockers.
    Release the lease → still paused. `cooldown` from the other → unpaused."""
    mod, shell, _cb = lifecycle_module

    # Step 1: acquire a real lease for forma-avatar (priority=normal).
    acquire_resp = await mod.acquire_lease(mod.AcquireRequest(
        client="forma-avatar", priority="normal", vram_mb=28000, preemptible=True,
    ))
    assert acquire_resp.granted is True
    lease_id = acquire_resp.lease_id
    # sglang-vision + vllm paused under the lease blocker.
    assert mod.paused_by[lease_id] == {"sglang-vision", "vllm"}
    initial_pause_count = len(shell.matching("docker pause"))
    assert initial_pause_count == 2

    # Step 2: lifecycle `generating` from a DIFFERENT source (peer-a) — which
    # has no cpu_contention itself but is treated as an external busy source.
    # We expect the legacy path to enroll sglang-vision + vllm under the new
    # blocker WITHOUT re-issuing pause commands (ref-count only).
    c = TestClient(mod.app)
    r = _post_envelope(c, "peer-a", "generating")
    assert r.status_code == 200
    blocker_key = mod._lifecycle_blocker_key("peer-a")
    assert blocker_key in mod.paused_by
    assert mod.paused_by[blocker_key] == {"sglang-vision", "vllm"}
    # No additional pause shell calls — already held.
    assert len(shell.matching("docker pause")) == initial_pause_count

    # Step 3: release the real lease. Clients remain paused (still held by
    # the lifecycle blocker).
    shell.calls.clear()
    await mod.release_lease(mod.ReleaseRequest(lease_id=lease_id))
    assert shell.matching("docker unpause") == []
    assert lease_id not in mod.paused_by
    assert mod.paused_by[blocker_key] == {"sglang-vision", "vllm"}

    # Step 4: lifecycle `cooldown` from peer-a → unpause finally fires.
    r2 = _post_envelope(c, "peer-a", "cooldown")
    assert r2.status_code == 200
    unpauses = shell.matching("docker unpause")
    assert len(unpauses) == 2
    assert blocker_key not in mod.paused_by


@pytest.mark.asyncio
async def test_mixed_lifecycle_then_lease_refcount(lifecycle_module):
    """Reverse order: lifecycle `generating` first, THEN acquire a real lease.
    Cooldown from the lifecycle source must NOT unpause while the lease is
    still held; releasing the lease finally unpauses."""
    mod, shell, _cb = lifecycle_module
    c = TestClient(mod.app)

    # Step 1: lifecycle generating → paused via lifecycle blocker.
    r = _post_envelope(c, "peer-a", "generating")
    assert r.status_code == 200
    blocker_key = mod._lifecycle_blocker_key("peer-a")
    assert mod.paused_by[blocker_key] == {"sglang-vision", "vllm"}
    initial_pause_count = len(shell.matching("docker pause"))
    assert initial_pause_count == 2

    # Step 2: acquire a real normal lease. It's a normal priority acquire, so
    # it will try to pause cpu_contention clients — they're already paused,
    # so only the ledger ref-count grows (no new shell calls).
    acquire_resp = await mod.acquire_lease(mod.AcquireRequest(
        client="forma-avatar", priority="normal", vram_mb=28000, preemptible=True,
    ))
    assert acquire_resp.granted is True
    lease_id = acquire_resp.lease_id
    assert mod.paused_by[lease_id] == {"sglang-vision", "vllm"}
    assert len(shell.matching("docker pause")) == initial_pause_count

    # Step 3: lifecycle `cooldown` from peer-a. Lease still holds → no unpause.
    shell.calls.clear()
    r2 = _post_envelope(c, "peer-a", "cooldown")
    assert r2.status_code == 200
    assert shell.matching("docker unpause") == []
    assert blocker_key not in mod.paused_by
    assert mod.paused_by[lease_id] == {"sglang-vision", "vllm"}

    # Step 4: release the real lease → unpause finally fires.
    await mod.release_lease(mod.ReleaseRequest(lease_id=lease_id))
    unpauses = shell.matching("docker unpause")
    assert len(unpauses) == 2
    assert lease_id not in mod.paused_by


# ---------------------------------------------------------------------------
# Admin endpoints
# ---------------------------------------------------------------------------

def test_lifecycle_log_endpoint_rejects_non_loopback(lifecycle_module):
    mod, _shell, _cb = lifecycle_module
    # TestClient sets client.host to "testclient" — loopback guard returns 403.
    c = TestClient(mod.app)
    r = c.get("/lifecycle_log")
    assert r.status_code == 403


def test_lifecycle_log_endpoint_loopback_returns_entries(lifecycle_module):
    mod, _shell, _cb = lifecycle_module
    # Bypass loopback guard for the test.
    mod._require_loopback = lambda req: None

    c = TestClient(mod.app)
    _post_envelope(c, "forma-avatar", "preflight")
    _post_envelope(c, "forma-avatar", "generating")

    r = c.get("/lifecycle_log?limit=10")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert len(body["entries"]) == 2
    assert body["entries"][0]["state"] == "preflight"
    assert body["entries"][1]["state"] == "generating"
    # Deliveries are recorded per-target.
    assert all("deliveries" in e for e in body["entries"])


def test_lifecycle_state_endpoint_loopback(lifecycle_module):
    mod, _shell, _cb = lifecycle_module
    mod._require_loopback = lambda req: None
    c = TestClient(mod.app)

    # No events yet → empty.
    r = c.get("/lifecycle/state")
    assert r.status_code == 200
    assert r.json() == {"sources": {}}

    # After generating → busy.
    _post_envelope(c, "forma-avatar", "generating")
    r = c.get("/lifecycle/state")
    assert r.json() == {"sources": {"forma-avatar": "generating"}}

    # After cooldown → cleared.
    _post_envelope(c, "forma-avatar", "cooldown")
    r = c.get("/lifecycle/state")
    assert r.json() == {"sources": {}}


def test_lifecycle_state_endpoint_rejects_non_loopback(lifecycle_module):
    mod, _shell, _cb = lifecycle_module
    c = TestClient(mod.app)
    r = c.get("/lifecycle/state")
    assert r.status_code == 403
