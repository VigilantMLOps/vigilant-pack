"""
orchestrator.py — deterministic state machine for the ML application run lifecycle.

Responsibilities:
  - Stage progression: VALIDATE → PREFLIGHT → INFRA → MODELS → RUNTIME → READY
  - StageEvent emission at each transition
  - Final failure classification (may agree with or override sub-executor suggestions)
  - Global deadline enforcement

Does NOT implement: Docker mechanics, Ollama API calls, health polling.
Those are delegated to services.py and models.py.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from . import manifest as manifest_mod
from . import models as models_mod
from . import preflight as preflight_mod
from . import services as services_mod
from .manifest import ManifestError


# ── Event contract ────────────────────────────────────────────────────────────

@dataclass
class StageEvent:
    stage: str
    event: str          # "started" | "succeeded" | "failed" | "warning"
    timestamp: str
    message: str
    metadata: dict = field(default_factory=dict)


_Emit = Callable[[StageEvent], None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _noop(_: StageEvent) -> None:
    pass


# ── Entry point ───────────────────────────────────────────────────────────────

def run(manifest_path: str, emit: _Emit | None = None) -> None:
    """
    Execute the full 6-stage run lifecycle.
    `emit` receives a StageEvent at each transition; defaults to no-op.
    Exits with code 1 on any HARD FAIL.
    """
    if emit is None:
        emit = _noop

    wall_start = time.monotonic()

    # ── Stage 1: VALIDATE ────────────────────────────────────────────────────
    manifest = _stage_validate(manifest_path, emit)

    compose       = manifest["compose"]
    ollama_url    = manifest_mod.ollama_url(manifest)
    global_budget = manifest.get("startup", {}).get("timeout", 300)
    deadline      = time.monotonic() + global_budget

    # ── Stage 2: PREFLIGHT ───────────────────────────────────────────────────
    _stage_preflight(manifest, emit)

    # ── Stage 3: INFRA ───────────────────────────────────────────────────────
    _stage_infra(manifest, compose, deadline, emit)

    # ── Stage 4: MODELS ──────────────────────────────────────────────────────
    degraded_models = _stage_models(manifest, ollama_url, deadline, emit)

    # ── Stage 5: RUNTIME ─────────────────────────────────────────────────────
    _stage_runtime(manifest, compose, deadline, emit)

    # ── Stage 6: READY ───────────────────────────────────────────────────────
    elapsed = int(time.monotonic() - wall_start)
    emit(StageEvent(
        stage="READY",
        event="succeeded",
        timestamp=_now(),
        message=f"System ready in {elapsed}s",
        metadata={
            "elapsed_seconds": elapsed,
            "degraded_models": degraded_models,
            "app_name":        manifest["app"]["name"],
            "app_version":     manifest["app"]["version"],
            "runtime_health":  manifest.get("runtime", {}).get("health", ""),
        },
    ))


# ── Stage implementations ─────────────────────────────────────────────────────

def _stage_validate(path: str, emit: _Emit) -> dict:
    emit(StageEvent("VALIDATE", "started", _now(), f"Loading {path}"))
    try:
        m = manifest_mod.load(path)
    except ManifestError as exc:
        emit(StageEvent("VALIDATE", "failed", _now(), str(exc),
                        {"failure_type": "hard"}))
        _abort(str(exc))

    emit(StageEvent("VALIDATE", "succeeded", _now(), "Manifest valid"))
    return m


def _stage_preflight(manifest: dict, emit: _Emit) -> None:
    emit(StageEvent("PREFLIGHT", "started", _now(), "Checking environment"))

    results = preflight_mod.run_checks(manifest)
    result_dicts = [
        {"name": r.name, "status": r.status, "message": r.message, "fix": r.fix}
        for r in results
    ]

    if preflight_mod.has_hard_failure(results):
        failures = [r.message for r in results if r.status == "FAIL"]
        emit(StageEvent(
            "PREFLIGHT", "failed", _now(),
            "; ".join(failures),
            {"failure_type": "hard", "results": result_dicts},
        ))
        _abort("Environment checks failed. Run `vigilantpack doctor` to see details and fixes.")

    warnings = [r for r in results if r.status == "WARN"]
    emit(StageEvent(
        "PREFLIGHT", "succeeded", _now(),
        "Environment ready",
        {"results": result_dicts, "warning_count": len(warnings)},
    ))


def _stage_infra(manifest: dict, compose: str, deadline: float, emit: _Emit) -> None:
    services = manifest.get("services", {})
    emit(StageEvent("INFRA", "started", _now(),
                    f"Starting {len(services)} infrastructure service(s)"))

    for name, svc in services.items():
        _assert_deadline(deadline, "INFRA", emit)

        start_r = services_mod.start(compose, name)
        if not start_r.success and start_r.error_type == "deterministic":
            emit(StageEvent("INFRA", "failed", _now(), start_r.message,
                            {"failure_type": "hard", "service": name}))
            _abort(start_r.message)

        health_url = svc["health"]
        svc_timeout = svc.get("timeout", 60)
        time_left   = max(5, int(deadline - time.monotonic()))
        poll_timeout = min(svc_timeout, time_left)

        emit(StageEvent("INFRA", "started", _now(), f"{name}: waiting for health",
                        {"service": name}))
        health_r = services_mod.poll_health(health_url, poll_timeout)

        if not health_r.success:
            # One restart attempt for transient failures (e.g., container exited early)
            services_mod.restart(compose, name)
            time_left    = max(5, int(deadline - time.monotonic()))
            poll_timeout = min(30, time_left)
            health_r     = services_mod.poll_health(health_url, poll_timeout)

        if not health_r.success:
            logs = services_mod.tail_logs(compose, name, lines=30)
            emit(StageEvent("INFRA", "failed", _now(), health_r.message,
                            {"failure_type": "hard", "service": name}))
            _abort(
                f"Service '{name}' did not become healthy.\n\n"
                f"--- recent logs ---\n{logs}\n---\n"
                f"Tip: run `vigilantpack logs {name}` for more."
            )

        emit(StageEvent("INFRA", "succeeded", _now(), f"{name}: healthy",
                        {"service": name}))

    emit(StageEvent("INFRA", "succeeded", _now(), "All infrastructure services healthy"))


def _stage_models(
    manifest: dict,
    ollama_url: str,
    deadline: float,
    emit: _Emit,
) -> list[str]:
    """
    Returns list of model names that ended in SOFT FAIL (degraded but not fatal).
    """
    model_list = manifest.get("models", [])
    if not model_list:
        emit(StageEvent("MODELS", "succeeded", _now(), "No models configured"))
        return []

    emit(StageEvent("MODELS", "started", _now(),
                    f"Managing {len(model_list)} model(s)"))

    present  = models_mod.list_present(ollama_url)
    degraded: list[str] = []

    # Sub-phase A: pull absent models
    for cfg in model_list:
        name     = cfg["name"]
        required = cfg.get("required", True)

        if name in present:
            emit(StageEvent("MODELS", "warning", _now(), f"{name}: already present",
                            {"model": name, "skipped": True}))
            continue

        _assert_deadline(deadline, "MODELS", emit)
        emit(StageEvent("MODELS", "started", _now(), f"{name}: pulling",
                        {"model": name}))

        last_pct = [-1]

        def progress(status: str, pct: int, _name: str = name) -> None:
            if pct != last_pct[0] and pct % 10 == 0:
                last_pct[0] = pct
                emit(StageEvent("MODELS", "started", _now(),
                                f"{_name}: {status} {pct}%",
                                {"model": _name, "pct": pct}))

        pull_r = models_mod.pull(name, ollama_url, progress_cb=progress)

        if pull_r.success:
            emit(StageEvent("MODELS", "succeeded", _now(), f"{name}: downloaded",
                            {"model": name}))
        elif required:
            emit(StageEvent("MODELS", "failed", _now(), pull_r.message,
                            {"failure_type": "hard", "model": name}))
            _abort(
                f"Required model '{name}' could not be pulled: {pull_r.message}"
            )
        else:
            # Optional model pull failure → SOFT FAIL, continue
            emit(StageEvent("MODELS", "warning", _now(),
                            f"{name}: pull failed (optional — continuing)",
                            {"failure_type": "soft", "model": name}))
            degraded.append(name)

    # Refresh presence after pulls
    present = models_mod.list_present(ollama_url)

    # Sub-phase B: warmup
    for cfg in model_list:
        name     = cfg["name"]
        required = cfg.get("required", True)

        if not cfg.get("warmup", False):
            continue
        if name not in present:
            continue   # couldn't pull optional model; skip its warmup

        emit(StageEvent("MODELS", "started", _now(), f"{name}: warming up",
                        {"model": name}))

        warmup_r = models_mod.warmup(name, ollama_url, required=required)

        if warmup_r.success:
            emit(StageEvent("MODELS", "succeeded", _now(), f"{name}: warm",
                            {"model": name}))
        elif warmup_r.suggested_severity == "soft":
            # Required model with API error → SOFT FAIL (first query may fail)
            emit(StageEvent("MODELS", "warning", _now(),
                            f"{name}: warmup failed — first query may fail ({warmup_r.message})",
                            {"failure_type": "soft", "model": name}))
            if name not in degraded:
                degraded.append(name)
        else:
            # Timeout or optional model → WARNING only
            emit(StageEvent("MODELS", "warning", _now(),
                            f"{name}: warmup incomplete — first query will be slower ({warmup_r.message})",
                            {"failure_type": "warn", "model": name}))

    emit(StageEvent("MODELS", "succeeded", _now(), "Model lifecycle complete",
                    {"degraded": degraded}))
    return degraded


def _stage_runtime(manifest: dict, compose: str, deadline: float, emit: _Emit) -> None:
    runtime = manifest.get("runtime", {})
    service = runtime["service"]
    health_url  = runtime["health"]
    svc_timeout = runtime.get("timeout", 45)

    emit(StageEvent("RUNTIME", "started", _now(), f"Starting {service}"))

    start_r = services_mod.start(compose, service)
    if not start_r.success and start_r.error_type == "deterministic":
        emit(StageEvent("RUNTIME", "failed", _now(), start_r.message,
                        {"failure_type": "hard"}))
        _abort(start_r.message)

    time_left    = max(5, int(deadline - time.monotonic()))
    poll_timeout = min(svc_timeout, time_left)
    health_r     = services_mod.poll_health(health_url, poll_timeout)

    if not health_r.success:
        services_mod.restart(compose, service)
        time_left    = max(5, int(deadline - time.monotonic()))
        poll_timeout = min(30, time_left)
        health_r     = services_mod.poll_health(health_url, poll_timeout)

    if not health_r.success:
        logs = services_mod.tail_logs(compose, service, lines=30)
        emit(StageEvent("RUNTIME", "failed", _now(), health_r.message,
                        {"failure_type": "hard", "service": service}))
        # Intentionally do NOT stop infra — it stays up for debugging
        _abort(
            f"Runtime service '{service}' did not become healthy.\n\n"
            f"--- recent logs ---\n{logs}\n---\n"
            f"Infrastructure services are still running.\n"
            f"Tip: run `vigilantpack logs {service}` to investigate."
        )

    emit(StageEvent("RUNTIME", "succeeded", _now(), f"{service}: healthy",
                    {"service": service}))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _assert_deadline(deadline: float, stage: str, emit: _Emit) -> None:
    if time.monotonic() > deadline:
        emit(StageEvent(stage, "failed", _now(),
                        "Global startup timeout exceeded",
                        {"failure_type": "hard"}))
        _abort(
            "Startup timeout exceeded. Increase `startup.timeout` in vigilant.yaml."
        )


def _abort(message: str) -> None:
    print(f"\n✗  {message}", file=sys.stderr)
    sys.exit(1)
