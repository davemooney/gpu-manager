# gpu-manager

Centralised GPU resource broker for aidin (dual RTX 5090). A lightweight
FastAPI daemon that coordinates VRAM and CPU time across services by issuing
leases: priority, preemptible flag, start/stop/pause commands, VRAM
accounting. Runs as `gpu-manager.service` under uvicorn on port `9090`.

## `clients.yaml` schema

Every entry under `clients:` describes one managed process.

| Field             | Type    | Required | Description |
| ----------------- | ------- | -------- | ----------- |
| `service`         | string \| null | no | systemd unit name (null when start/stop is an arbitrary shell command) |
| `default_vram_mb` | int     | yes      | baseline VRAM budget in MiB |
| `priority`        | string  | yes      | one of `critical`, `high`, `normal`, `idle` |
| `preemptible`     | bool    | yes      | may this client be preempted to free resources? |
| `cpu_contention`  | bool    | no (**NEW**) | if true, the manager auto-pauses this client when any `priority >= normal` lease is granted, and auto-unpauses when the last blocking lease releases. Defaults to `false`. |
| `start_command`   | string  | no       | shell command that launches the client |
| `stop_command`    | string  | no       | shell command that fully stops the client |
| `pause_command`   | string  | no (**NEW**) | shell command that pauses without teardown (e.g. `docker pause foo`) |
| `unpause_command` | string  | no (**NEW**) | shell command that resumes a paused client |
| `health_check`    | string  | no       | HTTP URL returning 200 when healthy |
| `startup_seconds` | int     | no       | max wait window for `health_check` to come up after start |
| `callback_url`    | string  | no       | webhook the manager POSTs to on preemption |
| `model_env_file`  | string  | no       | env file written before start for dynamic model selection |
| `model_env_var`   | string  | no       | var name inside `model_env_file` (default `VLLM_MODEL`) |
| `description`     | string  | no       | human label |

The schema is backwards-compatible — existing configs without
`cpu_contention`, `pause_command`, or `unpause_command` load fine.

## HTTP API

Public (bound `0.0.0.0:9090`):

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET    | `/health` | liveness |
| GET    | `/status` | GPU state, active leases, paused clients, stopped services, watchdog interval |
| GET    | `/clients` | list of configured clients (raw YAML) |
| POST   | `/lease/acquire` | request a lease. Body: `{client, priority, vram_mb, description, preemptible, callback_url?, model?}` |
| POST   | `/lease/release` | release a lease. Body: `{lease_id}` |
| POST   | `/lease/release-by-client/{client_name}` | release all leases for a client |
| GET    | `/service/health/{name}` | check if the named service is healthy |
| POST   | `/service/ensure/{name}?model=...` | acquire + start in one call |
| GET    | `/clients/{name}/state` | returns `running | paused | stopped | starting` (checks the pause ledger, then `docker inspect`, then health check) |
| GET    | `/preemption_log?limit=50` | audit trail of pause/unpause/fail events (last 500 kept in memory) |

Loopback-only admin (rejects non-`127.0.0.1` / `::1` with 403):

| Method | Path | Purpose |
| ------ | ---- | ------- |
| POST   | `/clients/{name}/pause`   | direct pause regardless of lease state |
| POST   | `/clients/{name}/unpause` | direct unpause regardless of ledger state |

### CPU-contention behaviour

When a lease is granted with `priority` in `{critical, high, normal}` (i.e.
anything stricter than `idle`):

1. Every client with `cpu_contention: true` AND `preemptible: true` is paused
   via its `pause_command` (best-effort, 5s timeout, failures logged not fatal).
2. The blocker lease is recorded in a ref-counted ledger. Overlapping
   high-priority leases will share ownership of the pause state — the
   client only gets unpaused when the LAST blocker releases.
3. On release, the ledger entry is removed and the client is unpaused (via
   `unpause_command`) only if no other blocker still holds it.

On manager startup, every `cpu_contention` client is unpaused defensively
(best-effort) to recover from any prior crash that stranded containers paused.

## Deploying on aidin

Place this repo at `/home/aidin/gpu-manager/` and manage via systemd:

```bash
# deploy
scp -r ./* aidin:/home/aidin/gpu-manager/

# on aidin
cd /home/aidin/gpu-manager
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# reload
sudo systemctl restart gpu-manager.service
journalctl -u gpu-manager.service -f
```

The systemd unit's `ExecStart` points at `uvicorn gpu_manager:app --host 0.0.0.0 --port 9090`.

### Smoke test

```bash
# Baseline: sglang-vision should be running
docker ps | grep sglang-vision

# Acquire a normal lease as forma-avatar
curl -X POST http://localhost:9090/lease/acquire \
  -H 'Content-Type: application/json' \
  -d '{"client":"forma-avatar","vram_mb":28000,"priority":"normal","preemptible":true,"description":"smoke"}'

# sglang-vision should now be in (Paused) state
docker ps | grep sglang-vision

# Release
curl -X POST http://localhost:9090/lease/release \
  -H 'Content-Type: application/json' \
  -d '{"lease_id":"<id-from-above>"}'

# Back to running
docker ps | grep sglang-vision
```

## Running tests

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pytest -q
```

Tests stub out `_run_shell`, `nvidia-smi`, `start_service`, and health checks
so nothing real is launched.

## Related issues

- [`davemooney/avatar#98`](https://github.com/davemooney/avatar/issues/98) —
  Tier 1 in-forma pause helper (fallback if the manager is unreachable)
- [`davemooney/avatar#99`](https://github.com/davemooney/avatar/issues/99) —
  this change: register sglang-vision + CPU-contention preemption
- [`davemooney/avatar#100`](https://github.com/davemooney/avatar/issues/100) —
  Tier 3 follow-ups (drain timeouts, observability)
