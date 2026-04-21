"""Unit tests for CPU-contention preemption (issue #99)."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Core acquire/release ref-counting
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_normal_acquire_pauses_cpu_contention_clients(gpu_manager_module):
    mod, recorder = gpu_manager_module

    req = mod.AcquireRequest(
        client="forma-avatar",
        priority="normal",
        vram_mb=28000,
        description="test",
        preemptible=True,
    )
    resp = await mod.acquire_lease(req)

    assert resp.granted is True
    assert resp.lease_id is not None
    # Both cpu_contention clients should be paused.
    assert set(resp.paused) == {"vllm", "sglang-vision"}

    pause_calls = recorder.commands_matching("docker pause")
    assert any("sglang-vision" in c for c in pause_calls)
    assert any("sglang-llm" in c for c in pause_calls)

    # Ledger records them against the blocker lease.
    assert mod.paused_by[resp.lease_id] == {"vllm", "sglang-vision"}


@pytest.mark.asyncio
async def test_release_unpauses(gpu_manager_module):
    mod, recorder = gpu_manager_module

    acquire_req = mod.AcquireRequest(
        client="forma-avatar", priority="normal", vram_mb=28000, preemptible=True
    )
    acquire_resp = await mod.acquire_lease(acquire_req)
    lease_id = acquire_resp.lease_id

    recorder.calls.clear()

    rel = mod.ReleaseRequest(lease_id=lease_id)
    result = await mod.release_lease(rel)

    assert result["status"] == "released"
    assert set(result["unpaused"]) == {"vllm", "sglang-vision"}
    unpause_calls = recorder.commands_matching("docker unpause")
    assert any("sglang-vision" in c for c in unpause_calls)
    assert any("sglang-llm" in c for c in unpause_calls)

    # Ledger is cleaned up.
    assert lease_id not in mod.paused_by


@pytest.mark.asyncio
async def test_overlapping_leases_refcount(gpu_manager_module):
    """Two overlapping `normal` leases → only one shell pause, only one
    shell unpause per client (ref-counting)."""
    mod, recorder = gpu_manager_module

    # First acquire → pauses everything.
    r1 = await mod.acquire_lease(mod.AcquireRequest(
        client="forma-avatar", priority="normal", vram_mb=28000, preemptible=True,
    ))
    initial_pause_count = len(recorder.commands_matching("docker pause"))
    assert initial_pause_count == 2  # vllm + sglang-vision

    # Second acquire (different client, same pause targets). forma-avatar is
    # already leased so we swap to theear for the 2nd blocker — but our
    # config only has forma-avatar as the `normal` client. So instead fake a
    # second lease against a client that's not itself in the cpu_contention
    # list: use forma-avatar variant by adding a lease directly via the
    # internal pause function (matches the acquire path exactly).
    r2_lease_id = "fake-lease-2"
    mod.active_leases[r2_lease_id] = mod.Lease(
        lease_id=r2_lease_id,
        client="some-other-caller",
        priority="normal",
        vram_mb=1000,
        description="overlap",
        granted_at="2026-04-20T00:00:00",
        preemptible=True,
    )
    paused_2 = await mod.pause_cpu_contention_clients(r2_lease_id, "some-other-caller", mod.load_config())

    # Both targets recorded under BOTH blockers.
    assert set(paused_2) == {"vllm", "sglang-vision"}
    assert mod.paused_by[r1.lease_id] == {"vllm", "sglang-vision"}
    assert mod.paused_by[r2_lease_id] == {"vllm", "sglang-vision"}

    # No additional docker pause calls fired (they were already paused).
    assert len(recorder.commands_matching("docker pause")) == initial_pause_count

    recorder.calls.clear()

    # Release the first blocker. Unpause should NOT fire — the second blocker still holds.
    await mod.release_lease(mod.ReleaseRequest(lease_id=r1.lease_id))
    assert len(recorder.commands_matching("docker unpause")) == 0
    assert mod.paused_by.get(r2_lease_id) == {"vllm", "sglang-vision"}

    # Release the second blocker. Now unpause fires.
    await mod.unpause_for_released_lease(r2_lease_id, mod.load_config())
    unpauses = recorder.commands_matching("docker unpause")
    assert len(unpauses) == 2  # one per client


@pytest.mark.asyncio
async def test_pause_failure_does_not_abort_acquire(gpu_manager_module):
    """If the pause_command fails, the acquire still succeeds (best-effort),
    and the failure is logged in preemption_log."""
    mod, recorder = gpu_manager_module

    recorder.set_response("docker pause sglang-vision", 1, "", "Error: no such container")

    r = await mod.acquire_lease(mod.AcquireRequest(
        client="forma-avatar", priority="normal", vram_mb=28000, preemptible=True,
    ))

    assert r.granted is True
    # sglang-vision did NOT get enrolled (pause failed); vllm did.
    assert "sglang-vision" not in mod.paused_by[r.lease_id]
    assert "vllm" in mod.paused_by[r.lease_id]

    # preemption_log captured the failure.
    failure_entries = [e for e in mod.preemption_log if e["action"] == "pause_failed"]
    assert any(e["client"] == "sglang-vision" for e in failure_entries)


@pytest.mark.asyncio
async def test_release_survives_config_reload(gpu_manager_module, tmp_path, monkeypatch):
    """Config may have been edited/reloaded between acquire and release. As
    long as the current config still has unpause_command for the tracked
    clients, release must still unpause them correctly."""
    mod, recorder = gpu_manager_module

    r = await mod.acquire_lease(mod.AcquireRequest(
        client="forma-avatar", priority="normal", vram_mb=28000, preemptible=True,
    ))

    # Simulate a config reload by writing a slightly modified YAML at the same
    # path — e.g. description changes, but pause/unpause commands intact.
    import yaml
    cfg_path = mod.CONFIG_PATH
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["clients"]["vllm"]["description"] = "edited"
    cfg["clients"]["sglang-vision"]["description"] = "edited"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    # Release → should still issue unpause for both.
    recorder.calls.clear()
    await mod.release_lease(mod.ReleaseRequest(lease_id=r.lease_id))
    unpauses = recorder.commands_matching("docker unpause")
    assert len(unpauses) == 2


@pytest.mark.asyncio
async def test_idle_priority_does_not_pause(gpu_manager_module):
    """A lease at `idle` priority must NOT trigger cpu_contention pausing."""
    mod, recorder = gpu_manager_module

    # vllm itself is priority=idle. Acquire a lease for it.
    r = await mod.acquire_lease(mod.AcquireRequest(
        client="vllm", priority="idle", vram_mb=29000, preemptible=True,
    ))

    assert r.granted is True
    assert r.paused == []
    assert mod.paused_by.get(r.lease_id, set()) == set()
    assert len(recorder.commands_matching("docker pause")) == 0


# ---------------------------------------------------------------------------
# Preemption log
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_preemption_log_structure(gpu_manager_module):
    mod, recorder = gpu_manager_module

    r = await mod.acquire_lease(mod.AcquireRequest(
        client="forma-avatar", priority="normal", vram_mb=28000, preemptible=True,
    ))

    entries = list(mod.preemption_log)
    assert len(entries) >= 2
    pause_entries = [e for e in entries if e["action"] == "pause"]
    assert len(pause_entries) == 2
    for e in pause_entries:
        assert "timestamp" in e
        assert "action" in e
        assert "client" in e
        assert "reason" in e
        assert "lease_id" in e
        assert e["lease_id"] == r.lease_id


# ---------------------------------------------------------------------------
# Boot-time defensive unpause
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_boot_sweep_unpauses_each_cpu_contention_client_once(gpu_manager_module):
    mod, recorder = gpu_manager_module

    await mod.boot_time_unpause_sweep(mod.load_config())

    unpauses = recorder.commands_matching("docker unpause")
    # Exactly one unpause per cpu_contention client with an unpause_command.
    assert len(unpauses) == 2
    assert any("sglang-vision" in c for c in unpauses)
    assert any("sglang-llm" in c for c in unpauses)

    # Sweep entries appear in the preemption log.
    boot_entries = [e for e in mod.preemption_log if e["action"] == "boot_unpause"]
    assert len(boot_entries) == 2


# ---------------------------------------------------------------------------
# Admin / HTTP endpoints (loopback + state)
# ---------------------------------------------------------------------------

def _client(gpu_manager_module):
    mod, _ = gpu_manager_module
    return TestClient(mod.app)


def test_admin_pause_rejects_non_loopback(gpu_manager_module):
    mod, recorder = gpu_manager_module
    # Starlette's TestClient sets client.host to "testclient" by default —
    # NOT a loopback. So the guard should return 403.
    c = TestClient(mod.app)
    r = c.post("/clients/vllm/pause")
    assert r.status_code == 403
    # No pause call fired.
    assert len(recorder.commands_matching("docker pause")) == 0


def test_admin_pause_from_loopback(gpu_manager_module):
    """Simulate a loopback call by patching the endpoint's guard."""
    mod, recorder = gpu_manager_module
    # Swap the guard to a no-op so we can exercise the handler body.
    mod._require_loopback = lambda req: None

    c = TestClient(mod.app)
    r = c.post("/clients/vllm/pause")
    assert r.status_code == 200
    assert r.json() == {"client": "vllm", "paused": True}
    assert any("docker pause sglang-llm" in cmd for cmd in recorder.calls)


def test_admin_unpause_from_loopback(gpu_manager_module):
    mod, recorder = gpu_manager_module
    mod._require_loopback = lambda req: None

    c = TestClient(mod.app)
    r = c.post("/clients/sglang-vision/unpause")
    assert r.status_code == 200
    assert r.json() == {"client": "sglang-vision", "paused": False}
    assert any("docker unpause sglang-vision" in cmd for cmd in recorder.calls)


def test_preemption_log_endpoint(gpu_manager_module):
    mod, _ = gpu_manager_module

    # Seed some entries directly.
    mod._log_preemption("pause", "vllm", "test", "lease-1")
    mod._log_preemption("unpause", "vllm", "test", "lease-1")

    c = TestClient(mod.app)
    r = c.get("/preemption_log?limit=5")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 2
    assert len(body["entries"]) == 2
    assert body["entries"][-1]["action"] == "unpause"


def test_client_state_endpoint_reports_paused_from_ledger(gpu_manager_module):
    # WP-102-06: endpoint is now loopback-gated + returns a richer shape.
    # `blocker_leases` replaces the old `source` field.
    mod, _ = gpu_manager_module
    mod.paused_by["lease-xyz"] = {"sglang-vision"}
    mod.preempted_state["sglang-vision"] = "paused"

    c = TestClient(mod.app, client=("127.0.0.1", 0))
    r = c.get("/clients/sglang-vision/state")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "paused"
    assert "lease-xyz" in body["blocker_leases"]


def test_client_state_unknown_client_404(gpu_manager_module):
    mod, _ = gpu_manager_module
    c = TestClient(mod.app, client=("127.0.0.1", 0))
    r = c.get("/clients/nonexistent/state")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# YAML backwards compatibility
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cpu_contention_absent_defaults_to_false(tmp_path, monkeypatch):
    """A client config without `cpu_contention` must load cleanly, and the
    manager must NOT pause it on acquire."""
    import yaml
    cfg = {
        "clients": {
            "vllm": {
                "default_vram_mb": 29000,
                "priority": "idle",
                "preemptible": True,
                # NO cpu_contention key
                "stop_command": "docker stop sglang-llm",
            },
            "forma-avatar": {
                "default_vram_mb": 28000,
                "priority": "normal",
                "preemptible": True,
            },
        }
    }
    p = tmp_path / "clients.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("GPU_MANAGER_CONFIG", str(p))

    import sys
    if "gpu_manager" in sys.modules:
        del sys.modules["gpu_manager"]
    import gpu_manager  # noqa: E402

    gpu_manager.active_leases.clear()
    gpu_manager.paused_by.clear()
    gpu_manager.preemption_log.clear()

    calls: list[str] = []

    async def fake_run_shell(cmd, timeout=5.0):
        calls.append(cmd)
        return (0, "", "")

    monkeypatch.setattr(gpu_manager, "_run_shell", fake_run_shell)
    monkeypatch.setattr(gpu_manager, "total_free_vram", lambda: 64000)
    monkeypatch.setattr(gpu_manager, "start_service", lambda n, c: True)
    monkeypatch.setattr(gpu_manager, "stop_service", lambda n, c: True)

    async def fake_health(url, timeout=3.0):
        return True
    monkeypatch.setattr(gpu_manager, "_check_health", fake_health)

    r = await gpu_manager.acquire_lease(gpu_manager.AcquireRequest(
        client="forma-avatar", priority="normal", vram_mb=28000, preemptible=True,
    ))

    assert r.granted is True
    assert r.paused == []
    # No shell pause was invoked.
    assert not any("docker pause" in c for c in calls)
