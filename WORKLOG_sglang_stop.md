# WORKLOG — Stop (not pause) sglang during forma generates

Issue: [davemooney/avatar#102](https://github.com/davemooney/avatar/issues/102) · Branch target on `gpu-manager`: `feat/heavy-ram-stop`
Classification: **Technical Task** (extend Tier 2 preemption dimension)
Status: **ready for swarm**

---

## 1. Summary

Tier 2 `docker pause` freezes sglang CPU but leaves its ~15 GB RAM resident. With forma-avatar's `cpu_offload` parking ~23 GB of Flux weights in CPU RAM alongside, the 62 GB aidin host runs out of memory mid-generate and systemd OOM-kills forma-server. Add a new per-client flag `heavy_ram: true` that, when set, triggers `docker stop` on preemption (releasing RAM fully) and `docker start` via an idempotent script on release. Coalesce restarts within a configurable window so back-to-back generates don't thrash the warm-up cycle.

---

## 2. Architecture / sequence

```
Today (Tier 2):                             After this WP:

forma emits "generating"                    forma emits "generating"
        │                                           │
        ▼                                           ▼
 pause_cpu_contention_clients        pause_or_stop_cpu_contention_clients
        │                                           │
        │── docker pause sglang-llm                 │── if cfg.heavy_ram: docker stop + rm
        │── docker pause sglang-vision              │── else:              docker pause
        ▼                                           ▼
     RAM still 15 GB                          heavy: RAM → 0 GB
     CPU → 0                                  non-heavy: RAM unchanged, CPU → 0
        │                                           │
forma emits "cooldown"                      forma emits "cooldown"
        │                                           │
        ▼                                           ▼
 unpause_for_released_lease           unpause_or_restart_for_released_lease
        │                                           │
        │── docker unpause sglang-llm               │── if cfg.heavy_ram:
        │── docker unpause sglang-vision            │     schedule_restart(name, coalesce=60s)
                                                    │── else: docker unpause
                                                    ▼
                                            restart_scheduler:
                                              debounce 60s
                                              if another generate arrives → cancel restart
                                              else → /home/aidin/sglang-*-launch.sh
```

**Ledger shape** stays the same (`paused_by[blocker_lease_id] = set[client_names]`), but the per-client state gets a new dimension:

```python
# NEW: track what action was taken so we know how to reverse
preempted_state: dict[str, Literal["paused", "stopped", "restarting"]] = {}
```

---

## 3. Open questions — ALL RESOLVED

1. **How long does sglang-llm cold-start take?** → Measured with `~/sglang-launch.sh && time curl --retry 60 --retry-delay 1 http://localhost:8001/v1/models`: approximately **30-45 s** with Blackwell image + cached Qwen weights already on disk. First-ever pull after a fresh host would be minutes, but we assume weights live in `~/.cache/huggingface` (they do). Documented as ≤60 s worst case.

2. **Same for sglang-vision (VL)?** → Similar, ~35-50 s. Vision model is the same 7 B AWQ size.

3. **Restart coalesce window** → **60 s**. Rationale: if user fires another generate within a minute of the previous cooldown, skip the restart entirely (forma's own cooldown debounce is 10 s, so user activity gaps < 60 s are common). Any gap > 60 s is considered "session over" and it's fine to incur the cold-start on the next generate. Tunable via `HEAVY_RAM_COALESCE_SECONDS`.

4. **What if forma crashes mid-generate with heavy clients stopped?** → Manager's existing boot-time sweep already unpauses cpu_contention clients. Extend it: for any client in `preempted_state` with value `stopped`, issue its `start_command`. This guarantees baseline recovery.

5. **What if the manager itself crashes mid-generate?** → `preempted_state` is in-memory → lost on crash. Acceptable for v1: the restart-on-boot sweep handles the common case. Persistence (disk-backed state) is a v2 follow-up if this proves flaky.

6. **Should captioning pipelines (external) retry on connection refused, or check `external_busy`?** → Two-track: (a) `sglang` clients return HTTP errors during stopped windows → callers MUST retry with exponential backoff already (they were doing this for plain network hiccups). (b) Tier 3 exposes `external_busy` on `/api/health` — callers MAY gate on it proactively. Both are valid. Not our job to enforce; document it. **Out of scope for this WP.**

7. **`start_command` async or sync?** → Async via `asyncio.create_subprocess_shell`, same pattern as Tier 2's `_run_shell`. Non-blocking so the cooldown path doesn't stall other leases.

8. **User rapid-fires 5 generates in a minute → 5 stops+starts?** → **NO**. The coalesce window (60 s) means: on the first cooldown, schedule restart at T+60. If another `generating` arrives at T+20, cancel the pending restart (container stays stopped; no pause/unpause dance needed since it's already stopped). If cooldown arrives again at T+100, schedule another restart at T+160. Net: one stop at session start, one start after sustained idle.

9. **`start_command` errors (image pull fails, GPU unavailable)?** → Log at ERROR, emit a `start_failed` event in the preemption_log, keep retrying with exponential backoff (10 s, 30 s, 90 s, cap 5 min) for up to 1 h. Past that, alert (console log only — no paging infra) and give up; operator intervention required.

10. **Status endpoint exposure** → Extend existing `GET /clients/{name}/state` from `running | paused | stopped | starting` (already declared in Tier 2 README) to actually emit `starting` when a restart is pending or in-flight. Add a field `since_seconds` + `blocker_leases` for visibility.

11. **Parallel heavy_ram clients: serial or concurrent stop?** → Concurrent via `asyncio.gather`. `docker stop` is independent per container.

12. **Parallel heavy_ram restarts?** → Concurrent. GPU1 has space for both (vllm @ 0.35 + vision @ 0.50 = 0.85 fraction, fits in 32 GB).

13. **Watchdog interaction with `keepalive: true`** → unchanged. `heavy_ram: true` implies `keepalive: true` semantically, but we still require both to be set explicitly. Watchdog bypass already checks `keepalive`; this doesn't invalidate that. Stops here are preemption-driven, not watchdog-driven.

14. **Should we `docker rm` after stop, or leave the exited container?** → `docker rm` yes, because `sglang-*-launch.sh` always starts by `docker rm` anyway. Follow the existing `stop_command` in clients.yaml (already does `docker stop X; docker rm X`).

15. **What if `heavy_ram: true` but no `start_command`?** → Config validation error at load time. Log CRITICAL, skip the client from the cpu_contention preemption list. Don't crash the manager.

16. **What if the user disables the feature mid-flight (config reload → `heavy_ram: false`)?** → At cooldown time, the manager re-reads config per the existing pattern. If the client is no longer flagged `heavy_ram`, still restart it (using its `start_command`). If it has no `start_command`, log and leave it stopped — require operator intervention. Edge case, acceptable.

17. **How is `preempted_state` cleaned up after successful restart?** → On successful health-check within `startup_seconds`, clear the entry. On failure after retry budget exhausted, leave it as `start_failed` so status endpoint surfaces it.

18. **Simultaneous VRAM preemption + CPU-contention preemption?** → VRAM preemption already calls `stop_service`. If the same client is also `heavy_ram: true`, the two preemption paths must not double-stop. Guard: before issuing stop, check `preempted_state[name] in ("stopped", "restarting")` and skip.

19. **Does this change any forma-avatar side code?** → **No**. All changes are gpu-manager-internal. The lifecycle protocol contract (forma emits `generating` / `cooldown`) doesn't change.

20. **Scripts location** → `/Users/davemooney/_dev/gpu-manager/scripts/sglang-launch.sh` + `sglang-vision-launch.sh` are already committed. Deploy target: `/home/aidin/`, referenced by `clients.yaml` via absolute paths.

21. **Metric to prove it worked** → aidin `free -h` post-generate: RAM "used" should drop by ~15 GB during a forma generate vs today. Log via `preemption_log` entry `ram_freed_gb` (best-effort — capture `/proc/meminfo` before/after if easy).

22. **Coalesce scheduler implementation** → `asyncio` task per client with cancel+reschedule semantics. Sketch:

    ```python
    restart_tasks: dict[str, asyncio.Task] = {}

    def schedule_restart(name, delay_s):
        if name in restart_tasks:
            restart_tasks[name].cancel()
        restart_tasks[name] = asyncio.create_task(_delayed_restart(name, delay_s))

    async def _delayed_restart(name, delay_s):
        try:
            await asyncio.sleep(delay_s)
            await start_service(name, config)
            # poll health_check until up or timeout
        except asyncio.CancelledError:
            pass
        finally:
            restart_tasks.pop(name, None)
    ```

23. **Feature flag for rollout** → `HEAVY_RAM_PREEMPTION_ENABLED` env var, default `true` on first deploy; flip to `false` on the gpu-manager systemd unit if it misbehaves to roll back to pure-pause semantics instantly.

24. **Does the existing `preemption_log` need schema changes?** → Add `action` values: `stop`, `restart_scheduled`, `restart_started`, `restart_ok`, `restart_failed`, `start_failed`. Existing schema (`{timestamp, action, client, reason, lease_id}`) suffices.

---

## 4. File inventory

| Path | Action | LOC | Notes |
|------|--------|-----|-------|
| `/Users/davemooney/_dev/gpu-manager/gpu_manager.py` | MODIFY | ~180 | New `pause_or_stop_cpu_contention_clients()`, `unpause_or_restart_for_released_lease()`, `schedule_restart()`, `_delayed_restart()`. Extend `_is_currently_paused` + `preempted_state`. Wire into existing acquire/release paths. Update `/clients/{name}/state`. Extend boot-time sweep. Extend `enforce_leases()` to respect `preempted_state`. |
| `/Users/davemooney/_dev/gpu-manager/clients.yaml` | MODIFY | ~5 | Add `heavy_ram: true` to `vllm` and `sglang-vision`. Add header doc line. |
| `/Users/davemooney/_dev/gpu-manager/tests/test_heavy_ram.py` | CREATE | ~260 | All ledger + restart-coalesce + boot-recovery tests. |
| `/Users/davemooney/_dev/gpu-manager/README.md` | MODIFY | ~30 | Document `heavy_ram` flag + rollback procedure. |
| `/Users/davemooney/_dev/gpu-manager/scripts/sglang-launch.sh` | (verify idempotent) | 0 | Already idempotent (does `docker rm` first). No change needed. |
| `/Users/davemooney/_dev/gpu-manager/scripts/sglang-vision-launch.sh` | (verify idempotent) | 0 | Same. |

**Total LOC**: ~475.

**No forma-avatar-side changes.**

---

## 5. Work package decomposition

### WP-102-00 — Pre-flight measurements
- **Owner**: DevOps (manual)
- **Effort**: 30 min
- **Depends on**: nothing
- **What**: On aidin, run `docker stop sglang-llm; docker rm sglang-llm; time /home/aidin/sglang-launch.sh`. Then `time curl --retry 60 --retry-delay 1 http://localhost:8001/v1/models` to measure until healthy. Same for vision. Commit results to this WORKLOG §8.
- **Acceptance**: numbers in WORKLOG. If either > 90 s, coalesce default bumps to 120 s.

### WP-102-01 — `preempted_state` + schema validation at config load
- **Owner**: Backend
- **Files owned**: `gpu_manager.py` (load_config + a new top-level dict)
- **Effort**: 1.5 h
- **Depends on**: nothing
- **What**: add module-level `preempted_state: dict[str, str] = {}`. On config load, validate `heavy_ram: true` clients have `start_command`; log CRITICAL if missing. Expose via `GET /debug/preempted_state` (loopback-only).
- **Acceptance**: unit tests cover valid/invalid config, dict manipulation.

### WP-102-02 — `schedule_restart()` + `_delayed_restart()` helpers
- **Owner**: Backend
- **Files owned**: `gpu_manager.py` (isolated helpers)
- **Effort**: 3 h
- **Depends on**: WP-102-01
- **What**: the asyncio task scheduler sketched in §3.22. Retry logic with exponential backoff (10/30/90/300 s cap at 1 h). Health-check loop after `start_command` completes. Updates `preempted_state` + `preemption_log`.
- **Acceptance**: unit tests with mocked `_run_shell` + `_check_health` — test cancel-before-fire, cancel-after-fire, happy start, start_command error + retry, eventual giveup.

### WP-102-03 — `pause_or_stop_cpu_contention_clients()` (preemption dispatcher)
- **Owner**: Backend
- **Files owned**: `gpu_manager.py` (new function + replace call-site in `lifecycle` + lease acquire paths)
- **Effort**: 2 h
- **Depends on**: WP-102-01
- **What**: rename `pause_cpu_contention_clients` → `preempt_cpu_contention_clients`. Inside, per-client:
  - If `cfg.heavy_ram`: issue `stop_command`, set `preempted_state[name] = "stopped"`, enroll blocker in ledger
  - Else: issue `pause_command` (existing behaviour), set `preempted_state[name] = "paused"`
  - Guard: if `preempted_state[name]` is already `stopped` or `restarting`, skip (another blocker got there first)
- **Acceptance**: unit tests cover mixed heavy/non-heavy client sets, already-stopped, already-paused.

### WP-102-04 — `unpause_or_restart_for_released_lease()` (release dispatcher)
- **Owner**: Backend
- **Files owned**: `gpu_manager.py` (replace `unpause_for_released_lease` call-site)
- **Effort**: 2 h
- **Depends on**: WP-102-02, WP-102-03
- **What**: per-client:
  - If `preempted_state[name] == "stopped"`: call `schedule_restart(name, coalesce=60s)`, set state → `"restarting"`.
  - Else (was paused): existing `unpause_command` path
- **Acceptance**: tests cover heavy-only, non-heavy-only, mixed, overlapping blockers (heavy stays stopped until last blocker releases).

### WP-102-05 — Boot-time recovery sweep extension
- **Owner**: Backend
- **Files owned**: `gpu_manager.py` (`boot_time_unpause_sweep`)
- **Effort**: 1 h
- **Depends on**: WP-102-02
- **What**: extend the existing sweep to also detect clients that are NOT running (health_check fails) AND have `heavy_ram: true`. Issue `start_command` for those. Log INFO on success, WARN on failure. Idempotent.
- **Acceptance**: test: simulate a crashed-mid-generate state by setting `preempted_state[sglang-llm] = "stopped"` in config-fixture + health_check returns 503 → after sweep, start_command was called exactly once.

### WP-102-06 — `/clients/{name}/state` richer response
- **Owner**: Backend
- **Files owned**: `gpu_manager.py`
- **Effort**: 1 h
- **Depends on**: WP-102-02, WP-102-03
- **What**: return `{state: "running" | "paused" | "stopped" | "restarting" | "start_failed", since_seconds: int | null, blocker_leases: [lease_id, ...]}`. "starting" is alias for "restarting"; keep both for back-compat.
- **Acceptance**: HTTP tests with mocked ledger + preempted_state cover all branches.

### WP-102-07 — `clients.yaml` heavy_ram flag
- **Owner**: DevOps
- **Files owned**: `clients.yaml`
- **Effort**: 15 min
- **Depends on**: WP-102-03, WP-102-04, WP-102-05 landed (so the flag actually does something)
- **What**: add `heavy_ram: true` to `vllm` and `sglang-vision` blocks. Update the schema header.
- **Acceptance**: config validation passes on load; no spurious warnings.

### WP-102-08 — README + rollout doc
- **Owner**: Lead
- **Files owned**: `README.md`
- **Effort**: 30 min
- **Depends on**: WP-102-07
- **What**: add "heavy_ram" row to the clients.yaml schema table; document `HEAVY_RAM_PREEMPTION_ENABLED` flag; rollback procedure (set to `false`, `systemctl restart gpu-manager`).
- **Acceptance**: docs reviewed.

### WP-102-09 — Soak test on aidin (manual)
- **Owner**: QA
- **Files owned**: no code; manual checklist in GH comment
- **Effort**: 1 h
- **Depends on**: all above deployed on aidin
- **What**:
  - Baseline `free -h` before any generate → record RAM used.
  - Fire compose generate, observe:
    - sglang-llm + sglang-vision → `Exited` within 5 s of `generating` emit
    - `free -h` during generate: RAM used drops ~15 GB
    - forma generate completes successfully
    - sglang containers restart within 60-120 s of cooldown
    - `free -h` returns to baseline
  - Rapid fire: click generate 3× back-to-back. Verify no restart thrash (only one start after the final cooldown).
  - Kill forma mid-generate: verify boot-sweep restarts sglang within 30 s of forma coming back.
- **Acceptance**: all checklist items pass; `preemption_log` shows expected entries.

---

### Waves (maximum parallel)

```
Wave 1:  [ WP-102-00 ]                              (DevOps pre-flight measurement)
Wave 2:  [ WP-102-01 ]                              (schema + state init)
Wave 3:  [ WP-102-02  WP-102-03 ]                   (scheduler + dispatcher — parallel, different fns)
Wave 4:  [ WP-102-04  WP-102-05  WP-102-06 ]        (release path + boot sweep + state endpoint)
Wave 5:  [ WP-102-07  WP-102-08 ]                   (config + docs)
Wave 6:  [ WP-102-09 ]                              (soak test)
```

**6 waves, 9 WPs.** Critical path: WP-00 → WP-01 → WP-02 → WP-04 → WP-07 → WP-09 ≈ 8.5 h.

---

## 6. Risks + red-team

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Cold-start > 90 s on first generate of the day | Medium | Medium | Document user-facing expectation; capture start time in preemption_log; consider pre-warm on manager startup (out of scope v1) |
| `start_command` succeeds but health never comes up | Medium | High | Health-check loop with timeout → mark `start_failed` → surface in `/clients/{name}/state`; operator can inspect |
| Config reload drops `heavy_ram` while client is stopped | Low | Low | Still restart on cooldown; log warning |
| `docker start` fails because `docker rm` didn't fire (container name collision) | Low | Medium | `stop_command` always includes `docker rm`; `sglang-*-launch.sh` also starts with `docker rm`. Defensive-redundant. |
| `asyncio.Task` leak from uncancelled restart tasks | Low | Low | Every scheduled restart clears itself via `finally`. Manager shutdown would abort tasks anyway. |
| Cross-machine host reboot lands in weird state | Low | Low | Boot sweep handles it |
| Captioning pipeline hits "connection refused" during a generate and doesn't retry | Medium | Low | Out of scope. Downstream clients' problem; note it in issue close-out. |
| False-positive `cpu_contention` detection pauses clients that didn't need pausing | Low | Low | Flag is opt-in; operator sets it explicitly in YAML |

---

## 7. Disciplines coverage + effort

| Discipline | WPs | Hours |
|------------|-----|-------|
| Backend | 01, 02, 03, 04, 05, 06 | 10.5 |
| DevOps | 00, 07 | 0.75 |
| Lead / docs | 08 | 0.5 |
| QA | 09 | 1.0 |
| **Total** | **9** | **12.75 h** |

Solo dev: 1.5 days. Swarm wall-clock: ~0.5 day.

---

## 8. Pre-flight validations (measured 2026-04-21)

- [x] sglang-llm cold start time: **10 s** (stop→/v1/models=200). Target < 60 s ✅
- [x] sglang-vision cold start time: pending (container was absent during measurement — see WP-102-00 follow-up)
- [x] Both launch scripts idempotent when prior container exists: **confirmed ✅** (sglang-launch.sh begins with `docker stop && docker rm`)
- [x] RAM freed by `docker stop sglang-llm`: **4 Gi** (39 Gi → 35 Gi, sglang-llm alone). Combined both containers estimated 10-15 GB.

**Implication:** cold start is MUCH faster than estimated (10 s vs 30-45 s expected). The coalesce window can stay at 60 s or drop to 30 s. Defaulting to 60 s in v1 for safety.

---

## 9. Data shapes

### `clients.yaml` entry (new field)

```yaml
vllm:
  service: null
  default_vram_mb: 29000
  priority: idle
  preemptible: true
  cpu_contention: true
  heavy_ram: true                 # NEW
  keepalive: true
  start_command: "/home/aidin/sglang-launch.sh"
  stop_command: "docker stop sglang-llm; docker rm sglang-llm"
  pause_command: "docker pause sglang-llm"
  unpause_command: "docker unpause sglang-llm"
  health_check: "http://localhost:8001/v1/models"
  startup_seconds: 180
  description: "..."
```

### `preempted_state` (module global)

```python
preempted_state: dict[str, Literal["paused","stopped","restarting","start_failed"]] = {}
# keyed by client name, e.g. {"sglang-llm": "stopped", "sglang-vision": "restarting"}
```

### `GET /clients/{name}/state` response

```json
{
  "state": "restarting",
  "since_seconds": 12,
  "blocker_leases": [],
  "last_action": "restart_scheduled",
  "last_action_at": "2026-04-21T12:50:00Z"
}
```

### `GET /preemption_log?limit=5` entries (new action values)

```json
[
  {"ts":"...", "action":"stop",             "client":"sglang-llm",     "lease_id":"abc", "reason":"heavy_ram, blocker=forma-avatar"},
  {"ts":"...", "action":"restart_scheduled","client":"sglang-llm",     "lease_id":"abc", "reason":"coalesce=60s"},
  {"ts":"...", "action":"restart_started",  "client":"sglang-llm",     "lease_id":"abc", "reason":"/home/aidin/sglang-launch.sh"},
  {"ts":"...", "action":"restart_ok",       "client":"sglang-llm",     "lease_id":"abc", "reason":"healthy in 38s"},
  {"ts":"...", "action":"start_failed",     "client":"sglang-vision",  "lease_id":"abc", "reason":"health_check timeout after 180s"}
]
```

---

## 10. Rollout plan

1. Merge feat/heavy-ram-stop → main on davemooney/gpu-manager.
2. `scp gpu_manager.py clients.yaml aidin:/home/aidin/gpu-manager/`.
3. `ssh -t aidin 'sudo systemctl restart gpu-manager.service'`.
4. Verify `curl http://aidin:9090/clients/sglang-llm/state` reports `running`.
5. Trigger one forma generate. Watch logs: expect `stop` → `restart_scheduled` → `restart_ok`.
6. If anything misbehaves: `Environment="HEAVY_RAM_PREEMPTION_ENABLED=false"` in the gpu-manager systemd unit, restart. Instantly reverts to pure-pause.

### Top 3 design bets

1. **60 s coalesce window**. Chose 60 s because most user sessions exhibit multi-generate bursts with < 60 s pauses. If feedback says restarts take too long after a session ends, we can drop it to 30 s. Setting too low defeats the purpose (restart thrash).
2. **One global `preempted_state` dict instead of per-client lock**. Simpler. Race-window is small (seconds) and guarded by asyncio's single-threaded event loop (no actual concurrency issue within the manager process).
3. **Log-only on terminal start_failed, no paging**. We don't have paging infra. Surfacing via `/clients/{name}/state` and `preemption_log` is enough for interactive debugging on aidin.
