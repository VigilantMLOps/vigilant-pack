"""
preflight.py — pre-flight environment checks.

Used identically by `vigilantpack doctor` and Stage 2 of `vigilantpack run`.
Returns a list of CheckResult; has no side effects (never starts anything).

Check severity:
  PASS — everything is fine
  WARN — advisory, run proceeds
  FAIL — hard blocker, run aborts
"""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse


@dataclass
class CheckResult:
    name: str
    status: str       # "PASS" | "WARN" | "FAIL"
    message: str
    fix: str | None = None


# ── Public API ────────────────────────────────────────────────────────────────

def run_checks(manifest: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    results.extend(_check_docker())
    results.extend(_check_env(manifest))
    results.extend(_check_vault())
    results.extend(_check_ports(manifest))
    results.extend(_check_disk())
    return results


def has_hard_failure(results: list[CheckResult]) -> bool:
    return any(r.status == "FAIL" for r in results)


# ── Checks ────────────────────────────────────────────────────────────────────

def _check_docker() -> list[CheckResult]:
    if not shutil.which("docker"):
        return [CheckResult(
            name="Docker CLI",
            status="FAIL",
            message="docker command not found in PATH",
            fix="Install Docker Desktop: https://www.docker.com/products/docker-desktop",
        )]

    try:
        proc = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        return [CheckResult(
            name="Docker daemon",
            status="FAIL",
            message="docker info timed out — daemon may be starting",
            fix="Wait for Docker Desktop to finish starting, then retry",
        )]
    except FileNotFoundError:
        return [CheckResult(
            name="Docker daemon",
            status="FAIL",
            message="docker not found",
            fix="Install Docker Desktop",
        )]

    if proc.returncode != 0:
        return [CheckResult(
            name="Docker daemon",
            status="FAIL",
            message="Docker daemon is not running",
            fix="Start Docker Desktop and wait until the whale icon is steady",
        )]

    ver_proc = subprocess.run(
        ["docker", "--version"], capture_output=True, text=True, timeout=5
    )
    ver = ver_proc.stdout.strip() if ver_proc.returncode == 0 else "unknown version"
    return [CheckResult(name="Docker daemon", status="PASS", message=ver)]


def _check_env(manifest: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    required: list[str] = manifest.get("env", {}).get("require", [])
    env_file = manifest.get("env", {}).get("file", ".env")

    for var in required:
        if os.environ.get(var):
            results.append(CheckResult(
                name=f"env  {var}",
                status="PASS",
                message=f"{var} is set",
            ))
        else:
            results.append(CheckResult(
                name=f"env  {var}",
                status="FAIL",
                message=f"{var} is not set",
                fix=f"Add to {env_file}: {var}=<value>",
            ))
    return results


def _check_vault() -> list[CheckResult]:
    vault = os.environ.get("VAULT_PATH")
    if not vault:
        return []   # already caught by _check_env if it was required

    p = Path(vault)
    if p.is_dir():
        md_count = sum(1 for _ in p.rglob("*.md"))
        return [CheckResult(
            name="VAULT_PATH",
            status="PASS",
            message=f"{vault} ({md_count} markdown files)",
        )]

    return [CheckResult(
        name="VAULT_PATH",
        status="FAIL",
        message=f"Directory does not exist: {vault}",
        fix="Create the directory or correct VAULT_PATH in .env",
    )]


def _check_ports(manifest: dict) -> list[CheckResult]:
    """
    For each port derived from service health URLs:
    - If free: PASS
    - If occupied but health endpoint responds: PASS (service already running — idempotent)
    - If occupied and health endpoint silent: FAIL (foreign process on the port)
    """
    results: list[CheckResult] = []

    port_map: dict[int, tuple[str, str]] = {}   # port → (label, health_url)
    for name, svc in manifest.get("services", {}).items():
        port = _port_from_url(svc.get("health", ""))
        if port:
            port_map[port] = (name, svc["health"])

    runtime = manifest.get("runtime", {})
    rt_port = _port_from_url(runtime.get("health", ""))
    if rt_port:
        port_map[rt_port] = (runtime.get("service", "runtime"), runtime["health"])

    for port, (label, health_url) in sorted(port_map.items()):
        if not _port_in_use(port):
            results.append(CheckResult(
                name=f"port {port}",
                status="PASS",
                message=f"available  ({label})",
            ))
        elif _url_responds(health_url):
            results.append(CheckResult(
                name=f"port {port}",
                status="PASS",
                message=f"occupied by running {label} service",
            ))
        else:
            results.append(CheckResult(
                name=f"port {port}",
                status="FAIL",
                message=f"port {port} is in use by an unknown process",
                fix=f"Stop whatever is using port {port}, or change the port in docker-compose.yml",
            ))
    return results


def _check_disk() -> list[CheckResult]:
    free_gb = shutil.disk_usage("/").free / (1024 ** 3)
    if free_gb >= 10:
        return [CheckResult(
            name="disk space",
            status="PASS",
            message=f"{free_gb:.1f} GB free",
        )]
    if free_gb >= 5:
        return [CheckResult(
            name="disk space",
            status="WARN",
            message=f"{free_gb:.1f} GB free — model downloads require 2–5 GB",
            fix="Free up disk space before running",
        )]
    return [CheckResult(
        name="disk space",
        status="WARN",
        message=f"{free_gb:.1f} GB free — insufficient for model downloads",
        fix="Free at least 5 GB before running",
    )]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _port_from_url(url: str) -> int | None:
    if not url:
        return None
    try:
        parsed = urlparse(url)
        if parsed.port:
            return parsed.port
        return {"http": 80, "https": 443}.get(parsed.scheme)
    except Exception:
        return None


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        try:
            s.connect(("localhost", port))
            return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            return False


def _url_responds(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme == "tcp":
        return _port_in_use(parsed.port) if parsed.port else False
    import httpx
    try:
        resp = httpx.get(url, timeout=2.0)
        return resp.status_code < 500
    except Exception:
        return False
