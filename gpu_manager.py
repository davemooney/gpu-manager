"""
GPU Resource Manager - Centralized VRAM allocation for aidin (dual RTX 5090).

A lightweight daemon that coordinates GPU usage across multiple services:
- noSonix (Parakeet transcription)
- vLLM (Qwen LLM inference)
- TheEar (audio processing)
- Ollama (local LLM)

Runs on port 9090, manages service lifecycle via systemctl.

ENFORCEMENT: Only services with active leases are allowed to run.
On startup, all managed services are stopped to establish a clean baseline.
A watchdog task periodically stops any service running without a lease.
"""

import os
import uuid
import time
import subprocess
import asyncio
from datetime import datetime
from typing import Optional
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="GPU Resource Manager", version="2.0.0")
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

PRIORITY_ORDER = {"critical": 0, "high": 1, "normal": 2, "idle": 3}

WATCHDOG_INTERVAL = 30  # seconds between enforcement checks


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


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
    message: str = ""
    queue_position: Optional[int] = None
    service_started: bool = False
    service_healthy: bool = False
    model: Optional[str] = None
    model_switched: bool = False


class ReleaseRequest(BaseModel):
    lease_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/status")
async def status():
    """Current GPU state, active leases, and stopped services."""
    gpus = get_gpu_vram()
    return {
        "gpus": gpus,
        "total_free_mb": sum(g["free_mb"] for g in gpus),
        "active_leases": {k: v.model_dump() for k, v in active_leases.items()},
        "stopped_services": stopped_services,
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

    return {
        "status": "released",
        "lease_id": req.lease_id,
        "client": lease.client,
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

    return {"status": "released", "leases": released}


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
