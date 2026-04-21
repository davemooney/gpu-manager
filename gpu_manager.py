"""
GPU Resource Manager - Centralized VRAM allocation for aidin (dual RTX 5090).

A lightweight daemon that coordinates GPU usage across multiple services:
- noSonix (Parakeet transcription)
- vLLM (Qwen LLM inference)
- TheEar (audio processing)
- Ollama (local LLM)
- sglang-vision (Qwen2.5-VL captioning)
- forma-avatar (Flux Kontext + PuLID)

Runs on port 9090, manages service lifecycle via systemctl.

ENFORCEMENT: Only services with active leases are allowed to run.
On startup, all managed services are stopped to establish a clean baseline.
A watchdog task periodically stops any service running without a lease.

CPU CONTENTION (Tier 2): Clients with `cpu_contention: true` in their config
get auto-paused when a priority >= normal lease is granted, and auto-unpaused
when the last blocking lease releases. Ref-counted by blocker lease_id so
overlapping high-priority leases don't accidentally unpause early.

LIFECYCLE ROUTER (Tier 3): Clients POST their lifecycle state transitions
(`idle` / `preflight` / `generating` / `cooldown`) to
`POST /lifecycle/{source}`. The manager fans out the envelope concurrently to
every OTHER registered client with a `callback_url`, and falls back to the
Tier-2 pause/unpause path for clients that haven't implemented their own
callback (gated on `preemptible + cpu_contention`). Lifecycle-driven pauses
share the same ref-counted `paused_by` ledger as real leases via synthetic
keys of the form `lifecycle:{source}`.
"""

import os
import uuid
import time
import logging
import subprocess
import asyncio
from collections import deque
from datetime import datetime
from typing import Optional
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger("gpu_manager")

app = FastAPI(title="GPU Resource Manager", version="2.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFIG_PATH = os.getenv(
    "GPU_MANAGER_CONFIG",
    str(Path(__file__).parent / "clients.yaml"),
)

# Feature flag for the heavy_ram preemption path (davemooney/avatar#102).
# Default on; flip to "false" on the systemd unit to instantly roll back to
# pure-pause semantics without a code change.
HEAVY_RAM_PREEMPTION_ENABLED = os.environ.get(
    "HEAVY_RAM_PREEMPTION_ENABLED", "true"
).strip().lower() == "true"

# Coalesce window (seconds) for heavy_ram restart scheduling. A cooldown event
# schedules the restart this far in the future; a subsequent `generating` event
# within the window cancels the pending restart so we don't thrash the cold
# start cycle on back-to-back generates. Tunable per-deployment.
HEAVY_RAM_COALESCE_SECONDS = int(os.environ.get("HEAVY_RAM_COALESCE_SECONDS", "60"))

PRIORITY_ORDER = {"critical": 0, "high": 1, "normal": 2, "idle": 3}

# Threshold at which we pause cpu_contention clients. normal (2) and tighter
# trigger preemption; idle (3) does not.
CPU_CONTENTION_PRIORITY_THRESHOLD = PRIORITY_ORDER["normal"]

WATCHDOG_INTERVAL = 30  # seconds between enforcement checks

PREEMPTION_LOG_MAX = 500

# Tier-3 lifecycle router constants
LIFECYCLE_LOG_MAX = 500
LIFECYCLE_PROTOCOL_VERSION = 1
LIFECYCLE_VALID_STATES = {"idle", "preflight", "generating", "cooldown"}
# Per-target HTTP timeout when fanning out lifecycle events to peer callbacks.
LIFECYCLE_FANOUT_TIMEOUT_SECONDS = 1.0


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        data = yaml.safe_load(f) or {}

    # Schema validation: a client flagged `heavy_ram: true` MUST have a
    # `start_command`, otherwise the stop→restart dispatcher would leave it
    # stuck in the stopped state with no way to recover. Rather than crash the
    # manager, flag the client as invalid so the preemption path skips it —
    # the rest of the config still works.
    for name, cfg in data.get("clients", {}).items():
        if cfg.get("heavy_ram") and not cfg.get("start_command"):
            logger.critical(
                f"[config] client '{name}' has heavy_ram=true but no start_command — "
                "excluding from cpu_contention preemption to avoid stuck-stopped state"
            )
            cfg["_heavy_ram_invalid"] = True  # internal flag, skipped by dispatchers

    return data


def _lifecycle_blocker_key(source_client: str) -> str:
    """Synthetic paused_by ledger key for a lifecycle-driven blocker.

    Real leases are keyed by their UUID (e.g. `lease-abc123`); lifecycle
    signals use the format `lifecycle:{source_client}` so one ref-counting
    codepath can handle both."""
    return f"lifecycle:{source_client}"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class Lease(BaseModel):
    lease_id: str
    client: str
    priority: str
    vram_mb: int
    description: str
    granted_at: str
    preemptible: bool = True
    callback_url: Optional[str] = None
    model: Optional[str] = None


# Active leases: lease_id -> Lease
active_leases: dict[str, Lease] = {}

# Services stopped by the manager (to restart later)
stopped_services: list[str] = []

# Wait queue for denied requests
wait_queue: list[dict] = []

# CPU-contention pause ledger.
#   paused_by[blocker_lease_id] = set of client names paused on behalf of this lease.
# A client can appear under multiple blockers — the manager unpauses it only
# when NO blocker remains for that client.
paused_by: dict[str, set[str]] = {}

# preempted_state tracks what action was taken on each client by the preemption
# machinery, so the release-time dispatcher knows how to reverse it.
# Values: "paused" | "stopped" | "restarting" | "start_failed"
# Keys: client names as they appear in clients.yaml.
preempted_state: dict[str, str] = {}

# Parallel map recording when the state transitioned. Used by /clients/{name}/state
# to report `since_seconds`.
preempted_since: dict[str, float] = {}

# Per-client asyncio Task handles for coalesced restarts.
# Key = client name; value = the currently-scheduled restart task.
# Scheduling a new restart for a client cancels any existing pending task,
# so rapid-fire cooldown→generating→cooldown sequences collapse to a single
# eventual restart at the trailing edge of the burst.
restart_tasks: dict[str, "asyncio.Task"] = {}

# In-memory audit log of preemption actions. Oldest first, newest last.
preemption_log: deque = deque(maxlen=PREEMPTION_LOG_MAX)

# In-memory audit log of lifecycle events received + fan-out outcomes.
# Parallel to preemption_log but recording the router-side of the Tier 3
# protocol. Oldest first, newest last.
lifecycle_log: deque = deque(maxlen=LIFECYCLE_LOG_MAX)


def _log_preemption(action: str, client: str, reason: str, lease_id: Optional[str]) -> None:
    """Append an entry to the preemption audit log + stdout."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "client": client,
        "reason": reason,
        "lease_id": lease_id,
    }
    preemption_log.append(entry)
    print(f"[GPU-MGR][preempt] {action} client={client} reason={reason} lease={lease_id}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_gpu_vram() -> list[dict]:
    """Query nvidia-smi for VRAM on each GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        gpus = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                gpus.append({
                    "index": int(parts[0]),
                    "total_mb": int(parts[1]),
                    "used_mb": int(parts[2]),
                    "free_mb": int(parts[3]),
                })
        return gpus
    except Exception as e:
        print(f"[GPU-MGR] nvidia-smi error: {e}")
        return []


def total_free_vram() -> int:
    """Total free VRAM across all GPUs in MiB."""
    return sum(g["free_mb"] for g in get_gpu_vram())


def is_service_active(client_name: str, config: dict) -> bool:
    """Check if a systemd service is currently active (running)."""
    client_cfg = config.get("clients", {}).get(client_name, {})
    service_name = client_cfg.get("service")
    if not service_name:
        return False
    try:
        result = subprocess.run(
            ["systemctl", "is-active", service_name],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() == "active"
    except Exception:
        return False


def stop_service(client_name: str, config: dict) -> bool:
    """Stop a service via its configured stop command."""
    client_cfg = config.get("clients", {}).get(client_name, {})
    stop_cmd = client_cfg.get("stop_command")
    if not stop_cmd:
        print(f"[GPU-MGR] No stop_command for {client_name}")
        return False

    print(f"[GPU-MGR] Stopping {client_name}: {stop_cmd}")
    try:
        subprocess.run(stop_cmd, shell=True, capture_output=True, timeout=30)
        if client_name not in stopped_services:
            stopped_services.append(client_name)
        return True
    except Exception as e:
        print(f"[GPU-MGR] Failed to stop {client_name}: {e}")
        return False


def start_service(client_name: str, config: dict) -> bool:
    """Start a service via its configured start command."""
    client_cfg = config.get("clients", {}).get(client_name, {})
    start_cmd = client_cfg.get("start_command")
    if not start_cmd:
        return False

    print(f"[GPU-MGR] Starting {client_name}: {start_cmd}")
    try:
        subprocess.run(start_cmd, shell=True, capture_output=True, timeout=30)
        if client_name in stopped_services:
            stopped_services.remove(client_name)
        return True
    except Exception as e:
        print(f"[GPU-MGR] Failed to start {client_name}: {e}")
        return False


def write_model_env(client_name: str, model: str, config: dict) -> bool:
    """Write model selection to env file before starting a service.

    Clients with 'model_env_file' and 'model_env_var' in their config
    support dynamic model switching. The env file is read by the systemd
    service via EnvironmentFile= directive.
    """
    client_cfg = config.get("clients", {}).get(client_name, {})
    env_file = client_cfg.get("model_env_file")
    env_var = client_cfg.get("model_env_var", "VLLM_MODEL")

    if not env_file:
        print(f"[GPU-MGR] No model_env_file for {client_name}, skipping model config")
        return False

    print(f"[GPU-MGR] Writing {env_var}={model} to {env_file}")
    try:
        env_path = Path(env_file)
        env_path.parent.mkdir(parents=True, exist_ok=True)
        env_path.write_text(f"{env_var}={model}\n")
        return True
    except Exception as e:
        print(f"[GPU-MGR] Failed to write model env: {e}")
        return False


async def _run_shell(cmd: str, timeout: float = 5.0) -> tuple[int, str, str]:
    """Run a shell command asynchronously. Returns (returncode, stdout, stderr).

    Separate from subprocess.run so the pause/unpause code path is non-blocking
    and easy to mock in tests.
    """
    proc = await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise
    return proc.returncode or 0, stdout.decode(errors="replace"), stderr.decode(errors="replace")


async def _check_health(url: str, timeout: float = 3.0) -> bool:
    """Check if a service is healthy by hitting its health URL."""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                return resp.status == 200
    except Exception:
        return False


async def notify_client(callback_url: str, message: dict):
    """Send a notification to a client via webhook."""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            await session.post(callback_url, json=message, timeout=aiohttp.ClientTimeout(total=5))
    except Exception as e:
        print(f"[GPU-MGR] Callback failed ({callback_url}): {e}")


# ---------------------------------------------------------------------------
# CPU-contention preemption
# ---------------------------------------------------------------------------

def _cpu_contention_candidates(config: dict) -> list[str]:
    """Names of all clients flagged for CPU-contention preemption."""
    out = []
    for name, cfg in config.get("clients", {}).items():
        if cfg.get("cpu_contention") and cfg.get("preemptible", True):
            out.append(name)
    return out


def _is_currently_paused(client: str) -> bool:
    """True if any blocker lease is currently holding `client` paused."""
    for blockers in paused_by.values():
        if client in blockers:
            return True
    return False


async def pause_cpu_contention_clients(blocker_lease_id: str, blocker_client: str, config: dict) -> list[str]:
    """Pause all cpu_contention clients that are currently active, on behalf
    of a newly granted blocker lease. Best-effort: failures are logged, not
    fatal. Returns the list of client names successfully enrolled in the
    pause ledger for this blocker.

    Ref-counting: if a client is already paused by another blocker, we add
    ourselves as an additional blocker and skip the shell pause call.
    """
    clients_cfg = config.get("clients", {})
    candidates = _cpu_contention_candidates(config)
    enrolled: list[str] = []

    # Don't pause ourselves.
    candidates = [c for c in candidates if c != blocker_client]

    # Ensure the ledger entry exists even if no candidates — keeps release logic simple.
    paused_by.setdefault(blocker_lease_id, set())

    for name in candidates:
        cfg = clients_cfg.get(name, {})
        pause_cmd = cfg.get("pause_command")
        if not pause_cmd:
            print(f"[GPU-MGR] Skipping {name}: cpu_contention flagged but no pause_command")
            continue

        if _is_currently_paused(name):
            # Already paused by someone else — just add ourselves as a blocker.
            paused_by[blocker_lease_id].add(name)
            enrolled.append(name)
            _log_preemption(
                action="pause_refcount",
                client=name,
                reason=f"already paused, adding blocker={blocker_client}",
                lease_id=blocker_lease_id,
            )
            continue

        try:
            rc, _out, err = await _run_shell(pause_cmd, timeout=5.0)
            if rc == 0:
                paused_by[blocker_lease_id].add(name)
                enrolled.append(name)
                _log_preemption(
                    action="pause",
                    client=name,
                    reason=f"cpu_contention blocker={blocker_client}",
                    lease_id=blocker_lease_id,
                )
            else:
                _log_preemption(
                    action="pause_failed",
                    client=name,
                    reason=f"rc={rc} stderr={err.strip()[:120]}",
                    lease_id=blocker_lease_id,
                )
        except asyncio.TimeoutError:
            _log_preemption(
                action="pause_failed",
                client=name,
                reason="timeout",
                lease_id=blocker_lease_id,
            )
        except Exception as e:
            _log_preemption(
                action="pause_failed",
                client=name,
                reason=f"exception={e}",
                lease_id=blocker_lease_id,
            )

    return enrolled


async def unpause_for_released_lease(lease_id: str, config: dict) -> list[str]:
    """Remove a blocker from the pause ledger. For each client it was holding,
    if no other blocker remains, issue unpause_command. Returns the list of
    client names actually unpaused (not including still-held-by-another)."""
    clients_cfg = config.get("clients", {})
    formerly_held = paused_by.pop(lease_id, set())
    unpaused: list[str] = []

    for name in formerly_held:
        # If another blocker still holds this client, leave it paused.
        if _is_currently_paused(name):
            _log_preemption(
                action="unpause_skipped",
                client=name,
                reason="another blocker still active",
                lease_id=lease_id,
            )
            continue

        # Look up the unpause command from the CURRENT config (which may have
        # been reloaded since the pause fired — that's fine, we just need the
        # command to run).
        cfg = clients_cfg.get(name, {})
        unpause_cmd = cfg.get("unpause_command")
        if not unpause_cmd:
            _log_preemption(
                action="unpause_skipped",
                client=name,
                reason="no unpause_command in current config",
                lease_id=lease_id,
            )
            continue

        try:
            rc, _out, err = await _run_shell(unpause_cmd, timeout=5.0)
            if rc == 0:
                unpaused.append(name)
                _log_preemption(
                    action="unpause",
                    client=name,
                    reason="last blocker released",
                    lease_id=lease_id,
                )
            else:
                _log_preemption(
                    action="unpause_failed",
                    client=name,
                    reason=f"rc={rc} stderr={err.strip()[:120]}",
                    lease_id=lease_id,
                )
        except asyncio.TimeoutError:
            _log_preemption(
                action="unpause_failed",
                client=name,
                reason="timeout",
                lease_id=lease_id,
            )
        except Exception as e:
            _log_preemption(
                action="unpause_failed",
                client=name,
                reason=f"exception={e}",
                lease_id=lease_id,
            )

    return unpaused


async def boot_time_unpause_sweep(config: dict) -> None:
    """At manager startup, best-effort unpause every cpu_contention client so
    we recover from a crash that stranded containers paused. INFO on success,
    WARN on failure. Each client is unpaused at most once per sweep."""
    for name, cfg in config.get("clients", {}).items():
        if not cfg.get("cpu_contention"):
            continue
        unpause_cmd = cfg.get("unpause_command")
        if not unpause_cmd:
            continue
        try:
            rc, _out, err = await _run_shell(unpause_cmd, timeout=5.0)
            if rc == 0:
                print(f"[GPU-MGR][boot-sweep] INFO: unpaused {name}")
                _log_preemption(
                    action="boot_unpause",
                    client=name,
                    reason="manager startup defensive sweep",
                    lease_id=None,
                )
            else:
                print(f"[GPU-MGR][boot-sweep] WARN: {name} unpause rc={rc} stderr={err.strip()[:120]}")
                _log_preemption(
                    action="boot_unpause_failed",
                    client=name,
                    reason=f"rc={rc} stderr={err.strip()[:120]}",
                    lease_id=None,
                )
        except asyncio.TimeoutError:
            print(f"[GPU-MGR][boot-sweep] WARN: {name} unpause timed out")
            _log_preemption(
                action="boot_unpause_failed",
                client=name,
                reason="timeout",
                lease_id=None,
            )
        except Exception as e:
            print(f"[GPU-MGR][boot-sweep] WARN: {name} unpause raised {e}")
            _log_preemption(
                action="boot_unpause_failed",
                client=name,
                reason=f"exception={e}",
                lease_id=None,
            )


# ---------------------------------------------------------------------------
# heavy_ram restart scheduler (davemooney/avatar#102 / WP-102-02)
# ---------------------------------------------------------------------------
#
# Heavy_ram clients get `docker stop`-ed (not just paused) during a generate,
# which releases their ~15 GB RSS back to the host. On cooldown we don't want
# to start them back up immediately — if the user fires another generate
# within ~60 s, we'd thrash the cold-start cycle. Instead, schedule a delayed
# restart; a new schedule call cancels the existing pending one (coalesce).


def schedule_restart(name: str, delay_s: int | None = None) -> None:
    """Schedule a heavy_ram client to restart after `delay_s` seconds.

    Cancels any existing pending restart task for this client (coalesce).
    No-op if HEAVY_RAM_PREEMPTION_ENABLED is False so operators can instantly
    disable the path via env var without a code change.
    """
    if not HEAVY_RAM_PREEMPTION_ENABLED:
        return
    delay = delay_s if delay_s is not None else HEAVY_RAM_COALESCE_SECONDS
    existing = restart_tasks.get(name)
    if existing and not existing.done():
        existing.cancel()
    task = asyncio.create_task(_delayed_restart(name, delay))
    restart_tasks[name] = task


async def _delayed_restart(name: str, delay_s: int) -> None:
    """After `delay_s` seconds, run the client's `start_command` and poll
    `health_check` until it responds or `startup_seconds` elapses.

    On start_command / health-check failure, retries with exponential backoff
    (10/30/90/300 s) for up to 1 h total before giving up and marking
    `preempted_state[name] = "start_failed"`. Cancellation at any point (via
    `schedule_restart` coalesce) exits cleanly without touching state.
    """
    try:
        await asyncio.sleep(delay_s)
    except asyncio.CancelledError:
        return

    backoffs = [10, 30, 90, 300]  # seconds between retries; roughly a 1 h budget
    deadline = time.time() + 3600  # 1 hour total
    attempt = 0

    while time.time() < deadline:
        config = load_config()
        cfg = config.get("clients", {}).get(name, {})
        start_cmd = cfg.get("start_command")
        health = cfg.get("health_check")
        startup_seconds = int(cfg.get("startup_seconds", 180))

        if not start_cmd:
            _log_preemption(
                action="start_failed",
                client=name,
                reason="no start_command in current config",
                lease_id="",
            )
            preempted_state[name] = "start_failed"
            preempted_since[name] = time.time()
            restart_tasks.pop(name, None)
            return

        _log_preemption(
            action="restart_started",
            client=name,
            reason=start_cmd[:120],
            lease_id="",
        )
        try:
            rc, _out, err = await _run_shell(start_cmd, timeout=60.0)
            if rc != 0:
                raise RuntimeError(f"start_command rc={rc}: {err[:160]}")

            # Poll health until up or startup_seconds elapsed.
            health_start = time.time()
            hdeadline = health_start + startup_seconds
            while time.time() < hdeadline:
                if health and await _check_health(health):
                    _log_preemption(
                        action="restart_ok",
                        client=name,
                        reason=f"healthy in {int(time.time() - health_start)}s",
                        lease_id="",
                    )
                    preempted_state.pop(name, None)
                    preempted_since.pop(name, None)
                    restart_tasks.pop(name, None)
                    return
                await asyncio.sleep(2)

            raise RuntimeError(f"health_check timeout after {startup_seconds}s")
        except asyncio.CancelledError:
            # Somebody re-scheduled us; just exit without touching state.
            return
        except Exception as e:  # noqa: BLE001 — best-effort restart
            _log_preemption(
                action="restart_failed",
                client=name,
                reason=str(e)[:160],
                lease_id="",
            )
            if attempt < len(backoffs):
                wait = backoffs[attempt]
                attempt += 1
                try:
                    await asyncio.sleep(wait)
                except asyncio.CancelledError:
                    return
            else:
                break

    # Out of retries within the 1 h budget — mark start_failed and give up.
    _log_preemption(
        action="start_failed",
        client=name,
        reason="exhausted 1h retry budget",
        lease_id="",
    )
    preempted_state[name] = "start_failed"
    preempted_since[name] = time.time()
    restart_tasks.pop(name, None)


# ---------------------------------------------------------------------------
# Watchdog - Enforce lease-only GPU access
# ---------------------------------------------------------------------------

async def enforce_leases():
    """Stop any managed service running without an active lease."""
    config = load_config()
    leased_clients = {l.client for l in active_leases.values()}

    # Build set of services that have active leases (any client sharing
    # the same systemd service counts as having a lease for that service)
    leased_services = set()
    for client_name in leased_clients:
        svc = config.get("clients", {}).get(client_name, {}).get("service")
        if svc:
            leased_services.add(svc)

    for name, cfg in config.get("clients", {}).items():
        if name in leased_clients:
            continue  # Has a lease — allowed to run

        # Tier-3 integration (davemooney/avatar#100): services marked
        # `keepalive: true` are long-running baseline workloads that
        # should stay up between leases. They get paused/unpaused via
        # the CPU-contention + lifecycle-callback paths instead of
        # stopped. The original watchdog would fight the pause/unpause
        # cycle by recycling the whole container each time its lease
        # was released.
        if cfg.get("keepalive", False):
            continue

        if not cfg.get("stop_command"):
            continue  # Not a managed service (e.g. nosonix has no stop_command)

        # If another client sharing the same service has a lease, skip
        service_name = cfg.get("service")
        if service_name and service_name in leased_services:
            continue

        # Check if service is running via systemd
        if is_service_active(name, config):
            print(f"[GPU-MGR] Watchdog: {name} running without lease — stopping")
            stop_service(name, config)
            continue

        # Fallback: check via health endpoint
        health_url = cfg.get("health_check")
        if health_url and await _check_health(health_url):
            print(f"[GPU-MGR] Watchdog: {name} healthy without lease — stopping")
            stop_service(name, config)


async def watchdog_loop():
    """Periodic enforcement loop."""
    while True:
        await asyncio.sleep(WATCHDOG_INTERVAL)
        try:
            await enforce_leases()
        except Exception as e:
            print(f"[GPU-MGR] Watchdog error: {e}")


# ---------------------------------------------------------------------------
# API Models
# ---------------------------------------------------------------------------

class AcquireRequest(BaseModel):
    client: str
    priority: str = "normal"
    vram_mb: int = 4000
    description: str = ""
    callback_url: Optional[str] = None
    preemptible: bool = True
    model: Optional[str] = None


class AcquireResponse(BaseModel):
    lease_id: Optional[str]
    granted: bool
    vram_mb: int = 0
    preempted: list[str] = []
    paused: list[str] = []
    message: str = ""
    queue_position: Optional[int] = None
    service_started: bool = False
    service_healthy: bool = False
    model: Optional[str] = None
    model_switched: bool = False


class ReleaseRequest(BaseModel):
    lease_id: str


class LifecycleEnvelope(BaseModel):
    """Tier-3 lifecycle event envelope.

    Shape is the source-of-truth for both emitter (forma-avatar) and router
    (this manager). Matches `doc/design/gpu-lifecycle-protocol.md` in
    davemooney/avatar.
    """
    source: str
    state: str
    version: int = LIFECYCLE_PROTOCOL_VERSION
    timestamp: Optional[str] = None
    context: dict = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/status")
async def status():
    """Current GPU state, active leases, and stopped services."""
    gpus = get_gpu_vram()
    # Summarise paused ledger as {client: [blocker_lease_id, ...]}
    paused_summary: dict[str, list[str]] = {}
    for blocker, clients in paused_by.items():
        for c in clients:
            paused_summary.setdefault(c, []).append(blocker)
    return {
        "gpus": gpus,
        "total_free_mb": sum(g["free_mb"] for g in gpus),
        "active_leases": {k: v.model_dump() for k, v in active_leases.items()},
        "stopped_services": stopped_services,
        "paused_clients": paused_summary,
        "wait_queue_length": len(wait_queue),
        "watchdog_interval": WATCHDOG_INTERVAL,
    }


@app.get("/clients")
async def list_clients():
    """List configured clients."""
    config = load_config()
    return config.get("clients", {})


@app.post("/lease/acquire", response_model=AcquireResponse)
async def acquire_lease(req: AcquireRequest):
    """Request GPU resources. May preempt lower-priority services.

    If the client has a start_command, the service is auto-started after
    the lease is granted and we wait for it to become healthy.

    If the lease's priority is >= normal, any clients flagged
    `cpu_contention: true` in the config are paused (best-effort) before
    the acquire returns.
    """
    config = load_config()

    # Validate client
    if req.client not in config.get("clients", {}):
        raise HTTPException(400, f"Unknown client: {req.client}")

    # Check if client already has a lease
    for lid, lease in list(active_leases.items()):
        if lease.client == req.client:
            # If a model is requested and differs from current → switch models
            if req.model and lease.model and req.model != lease.model:
                print(f"[GPU-MGR] Model switch requested for {req.client}: {lease.model} → {req.model}")
                stop_service(req.client, config)
                await asyncio.sleep(3)  # Wait for VRAM to free
                write_model_env(req.client, req.model, config)
                lease.model = req.model
                lease.description = req.description or lease.description
                service_started = start_service(req.client, config)
                service_healthy = False
                if service_started:
                    client_cfg = config.get("clients", {}).get(req.client, {})
                    health_url = client_cfg.get("health_check")
                    if health_url:
                        startup_secs = client_cfg.get("startup_seconds", 30)
                        print(f"[GPU-MGR] Waiting up to {startup_secs}s for {req.client} to become healthy after model switch...")
                        for i in range(startup_secs):
                            await asyncio.sleep(1)
                            if await _check_health(health_url):
                                service_healthy = True
                                print(f"[GPU-MGR] {req.client} healthy after {i + 1}s")
                                break
                    else:
                        service_healthy = True
                return AcquireResponse(
                    lease_id=lease.lease_id,
                    granted=True,
                    vram_mb=lease.vram_mb,
                    message=f"Model switched to {req.model}",
                    model=req.model,
                    model_switched=True,
                    service_started=service_started,
                    service_healthy=service_healthy,
                )
            # Same model or no model specified → return existing lease
            return AcquireResponse(
                lease_id=lease.lease_id,
                granted=True,
                vram_mb=lease.vram_mb,
                message=f"Already have active lease {lease.lease_id}",
                model=lease.model,
            )

    free = total_free_vram()
    preempted = []

    # If not enough VRAM, try preempting lower-priority services
    if free < req.vram_mb:
        req_priority = PRIORITY_ORDER.get(req.priority, 2)

        # Find preemptible leases with lower priority
        preemptible = [
            (lid, l) for lid, l in active_leases.items()
            if l.preemptible and PRIORITY_ORDER.get(l.priority, 2) > req_priority
        ]
        # Sort by priority (lowest/most preemptible first)
        preemptible.sort(key=lambda x: -PRIORITY_ORDER.get(x[1].priority, 2))

        for lid, lease in preemptible:
            if free >= req.vram_mb:
                break
            # Notify client if it has a callback
            if lease.callback_url:
                await notify_client(lease.callback_url, {
                    "action": "preempted",
                    "by": req.client,
                    "priority": req.priority,
                })
            # Stop the service
            stop_service(lease.client, config)
            del active_leases[lid]
            # Also release any pause ledger entry held by the preempted lease.
            await unpause_for_released_lease(lid, config)
            preempted.append(lease.client)
            # Wait briefly for VRAM to free
            await asyncio.sleep(2)
            free = total_free_vram()

        # Also stop services that don't have leases but are using VRAM
        if free < req.vram_mb:
            for name, cfg in config.get("clients", {}).items():
                if free >= req.vram_mb:
                    break
                if name == req.client:
                    continue
                if cfg.get("preemptible", True) and name not in [l.client for l in active_leases.values()]:
                    default_priority = PRIORITY_ORDER.get(cfg.get("priority", "idle"), 3)
                    if default_priority > req_priority:
                        stop_service(name, config)
                        preempted.append(name)
                        await asyncio.sleep(2)
                        free = total_free_vram()

    # Final check
    free = total_free_vram()
    if free < req.vram_mb:
        return AcquireResponse(
            lease_id=None,
            granted=False,
            message=f"Not enough VRAM ({free} MiB free, {req.vram_mb} MiB needed)",
            queue_position=len(wait_queue) + 1,
        )

    # Grant lease
    lease_id = str(uuid.uuid4())[:8]
    lease = Lease(
        lease_id=lease_id,
        client=req.client,
        priority=req.priority,
        vram_mb=req.vram_mb,
        description=req.description,
        granted_at=datetime.utcnow().isoformat(),
        preemptible=req.preemptible,
        callback_url=req.callback_url,
        model=req.model,
    )
    active_leases[lease_id] = lease

    msg = f"Lease granted. {free} MiB free."
    if preempted:
        msg = f"Stopped {', '.join(preempted)} to free VRAM. {free} MiB available."

    print(f"[GPU-MGR] Lease {lease_id} granted to {req.client} ({req.vram_mb} MiB, priority={req.priority})")

    # CPU-contention preemption: if this lease is priority >= normal, pause
    # all cpu_contention:true clients. Best-effort — failures are logged, not
    # fatal.
    paused: list[str] = []
    req_priority = PRIORITY_ORDER.get(req.priority, 2)
    if req_priority <= CPU_CONTENTION_PRIORITY_THRESHOLD:
        try:
            paused = await pause_cpu_contention_clients(lease_id, req.client, config)
        except Exception as e:
            print(f"[GPU-MGR] pause_cpu_contention_clients error: {e}")

    # Write model env file if model specified and client supports it
    if req.model:
        write_model_env(req.client, req.model, config)

    # Auto-start service if it has a start_command
    service_started = False
    service_healthy = False
    client_cfg = config.get("clients", {}).get(req.client, {})
    if client_cfg.get("start_command"):
        service_started = start_service(req.client, config)
        if service_started:
            health_url = client_cfg.get("health_check")
            if health_url:
                startup_secs = client_cfg.get("startup_seconds", 30)
                print(f"[GPU-MGR] Waiting up to {startup_secs}s for {req.client} to become healthy...")
                for i in range(startup_secs):
                    await asyncio.sleep(1)
                    if await _check_health(health_url):
                        service_healthy = True
                        print(f"[GPU-MGR] {req.client} healthy after {i + 1}s")
                        break
                if not service_healthy:
                    print(f"[GPU-MGR] {req.client} health check timed out after {startup_secs}s")
            else:
                service_healthy = True  # No health check — assume healthy

    return AcquireResponse(
        lease_id=lease_id,
        granted=True,
        vram_mb=req.vram_mb,
        preempted=preempted,
        paused=paused,
        message=msg,
        service_started=service_started,
        service_healthy=service_healthy,
        model=req.model,
    )


@app.post("/lease/release")
async def release_lease(req: ReleaseRequest):
    """Release GPU resources. Stops the associated service."""
    lease = active_leases.pop(req.lease_id, None)
    if not lease:
        raise HTTPException(404, f"Lease not found: {req.lease_id}")

    print(f"[GPU-MGR] Lease {req.lease_id} released by {lease.client}")

    # Stop the service that lost its lease
    config = load_config()
    stop_service(lease.client, config)

    # Unpause any cpu_contention clients that were being held paused by this lease.
    unpaused = await unpause_for_released_lease(req.lease_id, config)

    return {
        "status": "released",
        "lease_id": req.lease_id,
        "client": lease.client,
        "unpaused": unpaused,
    }


@app.post("/lease/release-by-client/{client_name}")
async def release_by_client(client_name: str):
    """Release all leases for a given client (convenience endpoint)."""
    released = []
    for lid in [*active_leases.keys()]:
        if active_leases[lid].client == client_name:
            released.append(lid)
            del active_leases[lid]

    if not released:
        raise HTTPException(404, f"No active leases for client: {client_name}")

    print(f"[GPU-MGR] Released {len(released)} lease(s) for {client_name}")

    # Stop the service
    config = load_config()
    stop_service(client_name, config)

    # Unpause any cpu_contention clients that were being held paused by these leases.
    all_unpaused: list[str] = []
    for lid in released:
        all_unpaused.extend(await unpause_for_released_lease(lid, config))

    return {"status": "released", "leases": released, "unpaused": all_unpaused}


@app.get("/health")
async def health():
    return {"status": "ok", "service": "gpu-manager"}


# ---------------------------------------------------------------------------
# Service Lifecycle
# ---------------------------------------------------------------------------

@app.get("/service/health/{name}")
async def service_health(name: str):
    """Check if a registered service is healthy."""
    config = load_config()
    client_cfg = config.get("clients", {}).get(name)
    if not client_cfg:
        raise HTTPException(404, f"Unknown service: {name}")

    health_url = client_cfg.get("health_check")
    if not health_url:
        # Fall back to systemd status
        active = is_service_active(name, config)
        return {"service": name, "healthy": active, "message": "Checked via systemd"}

    healthy = await _check_health(health_url)
    return {"service": name, "healthy": healthy}


@app.post("/service/ensure/{name}")
async def ensure_service(name: str, model: Optional[str] = None):
    """Ensure a registered service is running via lease.

    This is a convenience endpoint that acquires a lease (if needed)
    and starts the service. Optionally specify a model for model-switchable
    services (e.g. vllm).
    """
    config = load_config()
    client_cfg = config.get("clients", {}).get(name)
    if not client_cfg:
        raise HTTPException(404, f"Unknown service: {name}")

    # Check if already has a lease and is running
    for lease in active_leases.values():
        if lease.client == name:
            # If model requested and different, delegate to acquire for model switch
            if model and lease.model and model != lease.model:
                break  # Fall through to acquire path for model switching
            health_url = client_cfg.get("health_check")
            if health_url and await _check_health(health_url):
                return {"status": "running", "service": name, "already_running": True, "lease_id": lease.lease_id, "model": lease.model}
            # Has lease but not healthy — try starting
            if client_cfg.get("start_command"):
                start_service(name, config)
                if health_url:
                    startup_secs = client_cfg.get("startup_seconds", 30)
                    for i in range(startup_secs):
                        await asyncio.sleep(1)
                        if await _check_health(health_url):
                            return {"status": "running", "service": name, "startup_seconds": i + 1, "lease_id": lease.lease_id, "model": lease.model}
            return {"status": "unknown", "service": name, "lease_id": lease.lease_id}

    # No lease (or model switch needed) — acquire one
    req = AcquireRequest(
        client=name,
        priority=client_cfg.get("priority", "normal"),
        vram_mb=client_cfg.get("default_vram_mb", 4000),
        description=f"Auto-ensure for {name}",
        model=model,
    )
    result = await acquire_lease(req)

    if not result.granted:
        return {
            "status": "failed",
            "service": name,
            "message": result.message,
        }

    return {
        "status": "running" if result.service_healthy else "started",
        "service": name,
        "lease_id": result.lease_id,
        "service_started": result.service_started,
        "service_healthy": result.service_healthy,
        "model": result.model,
        "model_switched": result.model_switched,
    }


# ---------------------------------------------------------------------------
# Tier 3 — Lifecycle callback protocol
# ---------------------------------------------------------------------------

def _log_lifecycle(entry: dict) -> None:
    """Append a lifecycle event entry to the audit deque + stdout."""
    lifecycle_log.append(entry)
    print(
        f"[GPU-MGR][lifecycle] source={entry.get('source')} "
        f"state={entry.get('state')} targets={entry.get('targets')} "
        f"failures={entry.get('failures')}"
    )


async def _post_lifecycle_callback(
    target_name: str,
    callback_url: str,
    envelope: dict,
    timeout: float = LIFECYCLE_FANOUT_TIMEOUT_SECONDS,
) -> dict:
    """POST the envelope to a peer's callback URL. Returns a per-target
    delivery record, never raises. Used inside `asyncio.gather` so one slow
    target doesn't stall the fan-out."""
    record = {"target": target_name, "callback_url": callback_url, "status": "ok"}
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.post(
                callback_url,
                json=envelope,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                record["http_status"] = resp.status
                if resp.status >= 400:
                    record["status"] = "http_error"
                    print(
                        f"[GPU-MGR][lifecycle] WARN: callback {target_name} "
                        f"returned HTTP {resp.status}"
                    )
    except asyncio.TimeoutError:
        record["status"] = "timeout"
        print(f"[GPU-MGR][lifecycle] WARN: callback {target_name} timed out")
    except Exception as e:  # noqa: BLE001 — best-effort delivery
        record["status"] = "error"
        record["error"] = str(e)[:200]
        print(f"[GPU-MGR][lifecycle] WARN: callback {target_name} raised {e}")
    return record


async def _pause_for_lifecycle(
    source_client: str, config: dict
) -> tuple[list[str], list[str]]:
    """Legacy fallback: pause cpu_contention clients on behalf of a lifecycle
    `generating` signal. Shares the `paused_by` ledger with real leases via
    a synthetic `lifecycle:{source}` key.

    Returns (newly_enrolled, already_held) where:
      - newly_enrolled: clients we acted upon for this blocker
      - already_held: clients that were already paused by another blocker
        (we just added ourselves to the ref-count)
    """
    clients_cfg = config.get("clients", {})
    candidates = _cpu_contention_candidates(config)
    enrolled: list[str] = []
    already_held: list[str] = []

    blocker_key = _lifecycle_blocker_key(source_client)

    # Don't pause the emitter itself if it's also in the candidate set.
    candidates = [c for c in candidates if c != source_client]

    # Ensure ledger entry exists so release works uniformly even with no targets.
    paused_by.setdefault(blocker_key, set())

    for name in candidates:
        cfg = clients_cfg.get(name, {})
        pause_cmd = cfg.get("pause_command")
        if not pause_cmd:
            continue

        if _is_currently_paused(name):
            paused_by[blocker_key].add(name)
            enrolled.append(name)
            already_held.append(name)
            _log_preemption(
                action="pause_refcount",
                client=name,
                reason=f"already paused, adding lifecycle blocker={source_client}",
                lease_id=blocker_key,
            )
            continue

        try:
            rc, _out, err = await _run_shell(pause_cmd, timeout=5.0)
            if rc == 0:
                paused_by[blocker_key].add(name)
                enrolled.append(name)
                _log_preemption(
                    action="pause",
                    client=name,
                    reason=f"lifecycle generating from {source_client}",
                    lease_id=blocker_key,
                )
            else:
                _log_preemption(
                    action="pause_failed",
                    client=name,
                    reason=f"rc={rc} stderr={err.strip()[:120]}",
                    lease_id=blocker_key,
                )
        except asyncio.TimeoutError:
            _log_preemption(
                action="pause_failed",
                client=name,
                reason="timeout",
                lease_id=blocker_key,
            )
        except Exception as e:  # noqa: BLE001
            _log_preemption(
                action="pause_failed",
                client=name,
                reason=f"exception={e}",
                lease_id=blocker_key,
            )

    return enrolled, already_held


async def _unpause_for_lifecycle(source_client: str, config: dict) -> list[str]:
    """Release the synthetic lifecycle blocker and unpause any clients whose
    ref-count hits zero. Reuses the exact same unpause path as real leases."""
    blocker_key = _lifecycle_blocker_key(source_client)
    return await unpause_for_released_lease(blocker_key, config)


@app.post("/lifecycle/{source_client}")
async def lifecycle_event(source_client: str, envelope: LifecycleEnvelope):
    """Receive a lifecycle transition from a registered client.

    Validates + logs the event, fans out the envelope concurrently to every
    OTHER registered client with a `callback_url` (per-target 1s timeout),
    and applies the legacy pause/unpause fallback for clients that haven't
    implemented their own callback.
    """
    config = load_config()
    clients = config.get("clients", {})

    if source_client not in clients:
        raise HTTPException(404, f"Unknown source client: {source_client}")

    if envelope.version != LIFECYCLE_PROTOCOL_VERSION:
        raise HTTPException(
            400,
            f"unsupported protocol version {envelope.version} "
            f"(expected {LIFECYCLE_PROTOCOL_VERSION})",
        )

    if envelope.state not in LIFECYCLE_VALID_STATES:
        raise HTTPException(
            422,
            f"Invalid state '{envelope.state}' "
            f"(expected one of {sorted(LIFECYCLE_VALID_STATES)})",
        )

    # Envelope we actually forward — always use the URL path source so peers
    # don't have to cross-check `source` against the URL.
    forwarded = envelope.model_dump()
    forwarded["source"] = source_client
    # Timestamp the router's view of the event if the producer didn't set one.
    if not forwarded.get("timestamp"):
        forwarded["timestamp"] = datetime.utcnow().isoformat()

    # Identify fan-out targets (every OTHER client with a callback_url).
    fanout_targets: list[tuple[str, str]] = []
    legacy_targets: list[str] = []
    for name, cfg in clients.items():
        if name == source_client:
            continue
        cb = cfg.get("callback_url")
        if cb:
            fanout_targets.append((name, cb))
        elif cfg.get("preemptible", True) and cfg.get("cpu_contention"):
            legacy_targets.append(name)

    # Concurrent fan-out — one slow target does not delay others.
    delivery_records: list[dict] = []
    if fanout_targets:
        delivery_records = await asyncio.gather(
            *(
                _post_lifecycle_callback(name, url, forwarded)
                for name, url in fanout_targets
            )
        )

    # Legacy fallback: translate generating/cooldown into pause/unpause for
    # cpu_contention clients that haven't adopted the protocol yet.
    legacy_actions: dict = {}
    if envelope.state == "generating" and legacy_targets:
        enrolled, already_held = await _pause_for_lifecycle(source_client, config)
        legacy_actions = {
            "mode": "pause",
            "enrolled": enrolled,
            "already_held": already_held,
        }
    elif envelope.state == "cooldown":
        # Always attempt an unpause — harmless if the blocker key isn't in the
        # ledger (release_lease handles that).
        unpaused = await _unpause_for_lifecycle(source_client, config)
        legacy_actions = {"mode": "unpause", "unpaused": unpaused}
    # idle / preflight → no-op for the legacy path.

    failures = [r for r in delivery_records if r["status"] != "ok"]
    entry = {
        "timestamp": forwarded["timestamp"],
        "source": source_client,
        "state": envelope.state,
        "targets": [name for name, _ in fanout_targets],
        "deliveries": delivery_records,
        "failures": len(failures),
        "legacy_targets": legacy_targets,
        "legacy_actions": legacy_actions,
        "context": forwarded.get("context", {}),
    }
    _log_lifecycle(entry)

    return {
        "status": "ok",
        "source": source_client,
        "state": envelope.state,
        "fanout": {
            "targets": [name for name, _ in fanout_targets],
            "deliveries": delivery_records,
            "failures": len(failures),
        },
        "legacy": legacy_actions,
    }


@app.get("/lifecycle_log")
async def get_lifecycle_log(request: Request, limit: int = 50):
    """Return the last `limit` entries from the lifecycle audit log.

    Loopback-only — the endpoint exposes per-target callback URLs and
    delivery outcomes, which aren't something we want leaking off-host.
    """
    _require_loopback(request)
    if limit <= 0:
        return {"entries": [], "total": len(lifecycle_log)}
    entries = list(lifecycle_log)[-limit:]
    return {"entries": entries, "total": len(lifecycle_log)}


@app.get("/lifecycle/state")
async def get_lifecycle_state(request: Request):
    """Return the manager's view of each source's current external-busy state.

    Derived from `lifecycle_log`: the latest non-`cooldown` state per source.
    Once a source transitions to `cooldown` it drops out of the busy map,
    since cooldown means \"no longer generating\".
    """
    _require_loopback(request)
    current: dict[str, str] = {}
    for entry in lifecycle_log:
        source = entry.get("source")
        state = entry.get("state")
        if not source:
            continue
        if state == "cooldown":
            current.pop(source, None)
        elif state in LIFECYCLE_VALID_STATES:
            current[source] = state
    return {"sources": current}


# ---------------------------------------------------------------------------
# Admin / debug endpoints for CPU contention
# ---------------------------------------------------------------------------

def _require_loopback(request: Request) -> None:
    """Raise 403 unless the caller is on loopback. Admin endpoints only."""
    host = request.client.host if request.client else None
    if host not in ("127.0.0.1", "::1"):
        raise HTTPException(403, f"Loopback only (saw {host})")


async def _docker_container_state(container_name: str) -> Optional[str]:
    """Query `docker inspect ... --format={{.State.Status}}`. Returns the
    raw state string (e.g. 'running', 'paused', 'exited') or None on error."""
    cmd = f"docker inspect {container_name} --format='{{{{.State.Status}}}}'"
    try:
        rc, stdout, _err = await _run_shell(cmd, timeout=3.0)
        if rc != 0:
            return None
        # strip quotes + whitespace
        return stdout.strip().strip("'").strip('"') or None
    except Exception:
        return None


def _infer_container_name(client: str, cfg: dict) -> Optional[str]:
    """Best-effort pluck the docker container name out of pause_command. Most
    of our clients use `docker pause <name>` so we parse that. Returns None
    if we can't figure it out."""
    pause_cmd = cfg.get("pause_command") or ""
    if "docker pause" in pause_cmd:
        parts = pause_cmd.replace("docker pause", "").strip().split()
        if parts:
            return parts[0]
    return None


@app.get("/clients/{name}/state")
async def client_state(name: str):
    """Return one of: running | paused | stopped | starting.

    Primary source of truth: our internal pause ledger. If the client is in
    the ledger → paused. Otherwise, if the client has docker-backed
    pause/unpause commands, we fall through to `docker inspect` so the answer
    stays truthful across manager restarts. Finally, fall back to the health
    check + lease state."""
    config = load_config()
    cfg = config.get("clients", {}).get(name)
    if not cfg:
        raise HTTPException(404, f"Unknown client: {name}")

    # Ledger check first — this is what WE know.
    if _is_currently_paused(name):
        return {"client": name, "state": "paused", "source": "ledger"}

    # Docker inspect fallback for clients with pause commands.
    container = _infer_container_name(name, cfg)
    if container:
        raw = await _docker_container_state(container)
        if raw == "paused":
            return {"client": name, "state": "paused", "source": "docker"}
        if raw == "running":
            return {"client": name, "state": "running", "source": "docker"}
        if raw in ("exited", "dead", "created"):
            return {"client": name, "state": "stopped", "source": "docker"}
        if raw == "restarting":
            return {"client": name, "state": "starting", "source": "docker"}

    # Last-resort: health check + lease state.
    has_lease = any(l.client == name for l in active_leases.values())
    health_url = cfg.get("health_check")
    if health_url:
        healthy = await _check_health(health_url)
        if healthy:
            return {"client": name, "state": "running", "source": "health"}
        if has_lease:
            return {"client": name, "state": "starting", "source": "health"}
        return {"client": name, "state": "stopped", "source": "health"}

    if is_service_active(name, config):
        return {"client": name, "state": "running", "source": "systemd"}
    return {"client": name, "state": "stopped", "source": "systemd"}


@app.post("/clients/{name}/pause")
async def admin_pause(name: str, request: Request):
    """Loopback-only: directly pause a client, no lease required."""
    _require_loopback(request)
    config = load_config()
    cfg = config.get("clients", {}).get(name)
    if not cfg:
        raise HTTPException(404, f"Unknown client: {name}")
    pause_cmd = cfg.get("pause_command")
    if not pause_cmd:
        raise HTTPException(400, f"No pause_command configured for {name}")
    try:
        rc, _out, err = await _run_shell(pause_cmd, timeout=5.0)
    except asyncio.TimeoutError:
        _log_preemption("admin_pause_failed", name, "timeout", None)
        raise HTTPException(504, "pause timed out")
    if rc != 0:
        _log_preemption("admin_pause_failed", name, f"rc={rc} stderr={err.strip()[:120]}", None)
        raise HTTPException(500, f"pause failed rc={rc}: {err.strip()[:200]}")
    _log_preemption("admin_pause", name, "manual loopback request", None)
    return {"client": name, "paused": True}


@app.post("/clients/{name}/unpause")
async def admin_unpause(name: str, request: Request):
    """Loopback-only: directly unpause a client, regardless of ledger state."""
    _require_loopback(request)
    config = load_config()
    cfg = config.get("clients", {}).get(name)
    if not cfg:
        raise HTTPException(404, f"Unknown client: {name}")
    unpause_cmd = cfg.get("unpause_command")
    if not unpause_cmd:
        raise HTTPException(400, f"No unpause_command configured for {name}")
    try:
        rc, _out, err = await _run_shell(unpause_cmd, timeout=5.0)
    except asyncio.TimeoutError:
        _log_preemption("admin_unpause_failed", name, "timeout", None)
        raise HTTPException(504, "unpause timed out")
    if rc != 0:
        _log_preemption("admin_unpause_failed", name, f"rc={rc} stderr={err.strip()[:120]}", None)
        raise HTTPException(500, f"unpause failed rc={rc}: {err.strip()[:200]}")
    _log_preemption("admin_unpause", name, "manual loopback request", None)
    return {"client": name, "paused": False}


@app.get("/debug/preempted_state")
async def get_preempted_state(request: Request):
    """Loopback-only: snapshot of the heavy_ram preemption ledger.

    Exposes the in-memory `preempted_state` + `preempted_since` maps plus the
    feature flag so operators can confirm whether the heavy_ram path is
    actively engaged mid-generate. Loopback-guarded because it reveals the
    internal preemption machinery shape."""
    client_host = request.client.host if request.client else None
    if client_host not in ("127.0.0.1", "::1"):
        raise HTTPException(status_code=403, detail="loopback only")
    return {
        "preempted_state": dict(preempted_state),
        "preempted_since": dict(preempted_since),
        "heavy_ram_enabled": HEAVY_RAM_PREEMPTION_ENABLED,
    }


@app.get("/preemption_log")
async def get_preemption_log(limit: int = 50):
    """Return the last `limit` entries from the preemption audit log.
    Newest last."""
    if limit <= 0:
        return {"entries": [], "total": len(preemption_log)}
    entries = list(preemption_log)[-limit:]
    return {"entries": entries, "total": len(preemption_log)}


# ---------------------------------------------------------------------------
# Startup & Watchdog
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def on_startup():
    config = load_config()
    clients = config.get("clients", {})
    print(f"[GPU-MGR] Loaded config with {len(clients)} clients")

    gpus = get_gpu_vram()
    for g in gpus:
        print(f"[GPU-MGR] GPU {g['index']}: {g['total_mb']} MiB total, {g['free_mb']} MiB free")

    # Defensive unpause — recover any containers stranded paused by a prior crash.
    await boot_time_unpause_sweep(config)

    # ENFORCEMENT: Stop all managed services to establish clean baseline
    print("[GPU-MGR] Enforcing clean baseline — stopping all managed services...")
    for name, cfg in clients.items():
        if cfg.get("stop_command"):
            if is_service_active(name, config):
                print(f"[GPU-MGR] Stopping {name} (was running without lease)")
                stop_service(name, config)
    stopped_services.clear()  # Clean slate — don't track as "manager-stopped"

    # Wait for VRAM to free up
    await asyncio.sleep(3)
    gpus = get_gpu_vram()
    for g in gpus:
        print(f"[GPU-MGR] GPU {g['index']}: {g['free_mb']} MiB free (after cleanup)")

    # Start watchdog
    asyncio.create_task(watchdog_loop())
    print(f"[GPU-MGR] Watchdog started (interval: {WATCHDOG_INTERVAL}s)")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9090)
