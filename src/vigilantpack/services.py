"""
services.py — Docker Compose service lifecycle.

Responsibilities: start, stop, health-poll, restart, log retrieval.
Does NOT know about models, warmup, or orchestration stage order.

StageResult is defined here because models.py also returns StageResult —
both modules are sub-executors that share the same result contract.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class StageResult:
    success: bool
    suggested_severity: str          # "hard" | "soft" | "warn"
    message: str
    recoverable: bool
    error_type: str | None = None    # "transient" | "deterministic" | "unknown"
    metadata: dict = field(default_factory=dict)


# ── Service lifecycle ─────────────────────────────────────────────────────────

def start(compose_file: str, service: str) -> StageResult:
    """Start a single service. Returns immediately; does not wait for health."""
    proc = subprocess.run(
        ["docker", "compose", "-f", compose_file, "up", "-d", "--no-deps", service],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return StageResult(True, "hard", f"{service}: container started", True)

    stderr = (proc.stderr or "").strip()
    if any(phrase in stderr for phrase in ("port is already allocated", "address already in use")):
        return StageResult(
            False, "hard",
            f"{service}: port conflict — {stderr}",
            False, "deterministic",
        )
    if any(phrase in stderr.lower() for phrase in ("pull", "network", "unable to find")):
        return StageResult(
            False, "hard",
            f"{service}: image pull failed — {stderr}",
            True, "transient",
        )
    return StageResult(False, "hard", f"{service}: {stderr}", False, "unknown")


def poll_health(url: str, timeout: int) -> StageResult:
    """
    Poll a health URL every 2 s until it returns < 500 or the timeout expires.
    Returns success=True on the first healthy response.
    """
    deadline = time.monotonic() + timeout
    last_error = "no response yet"

    while time.monotonic() < deadline:
        try:
            resp = httpx.get(url, timeout=3.0)
            if resp.status_code < 500:
                return StageResult(True, "hard", f"healthy (HTTP {resp.status_code})", True)
        except httpx.RequestError as exc:
            last_error = str(exc)
        time.sleep(2)

    return StageResult(
        False, "hard",
        f"health check timed out after {timeout}s — {last_error}",
        False, "transient",
    )


def restart(compose_file: str, service: str) -> None:
    """Best-effort restart; caller re-polls health after calling this."""
    subprocess.run(
        ["docker", "compose", "-f", compose_file, "restart", service],
        capture_output=True,
    )


def stop_all(compose_file: str) -> None:
    """Stop all services. Volumes are preserved (no --volumes flag)."""
    subprocess.run(
        ["docker", "compose", "-f", compose_file, "down"],
        capture_output=True,
    )


# ── Log access ────────────────────────────────────────────────────────────────

def tail_logs(compose_file: str, service: str | None, lines: int = 30) -> str:
    cmd = ["docker", "compose", "-f", compose_file, "logs", "--tail", str(lines)]
    if service:
        cmd.append(service)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return (proc.stdout + proc.stderr).strip()


def stream_logs(compose_file: str, service: str | None, follow: bool) -> None:
    """Stream logs to the terminal; blocks until the user interrupts."""
    cmd = ["docker", "compose", "-f", compose_file, "logs"]
    if follow:
        cmd.append("--follow")
    if service:
        cmd.append(service)
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        pass
