"""
cli.py — Click commands and Rich terminal output.

No business logic here. Commands delegate entirely to orchestrator,
preflight, services, and manifest modules.
"""
from __future__ import annotations

import sys

import click
import httpx
from rich import box
from rich.console import Console
from rich.table import Table

from . import manifest as manifest_mod
from . import models as models_mod
from . import orchestrator
from . import preflight as preflight_mod
from . import services as services_mod
from .manifest import ManifestError
from .orchestrator import StageEvent

console     = Console()
err_console = Console(stderr=True, highlight=False)

_MANIFEST_OPT = click.option(
    "--file", "manifest_path",
    default="vigilant.yaml",
    show_default=True,
    help="Path to vigilant.yaml",
)


# ── Group ─────────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(package_name="vigilantpack")
def cli() -> None:
    """VigilantPack — deterministic ML application runtime."""


# ── doctor ────────────────────────────────────────────────────────────────────

@cli.command()
@_MANIFEST_OPT
def doctor(manifest_path: str) -> None:
    """Check prerequisites without starting anything."""
    try:
        m = manifest_mod.load(manifest_path)
    except ManifestError as exc:
        err_console.print(f"[red]✗ Manifest error:[/red] {exc}")
        sys.exit(1)

    results = preflight_mod.run_checks(m)

    table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    table.add_column(width=3)
    table.add_column(min_width=18)
    table.add_column()

    for r in results:
        if r.status == "PASS":
            icon = "[green]✓[/green]"
        elif r.status == "WARN":
            icon = "[yellow]⚠[/yellow]"
        else:
            icon = "[red]✗[/red]"

        detail = r.message
        if r.fix and r.status != "PASS":
            detail += f"\n    [dim]→ {r.fix}[/dim]"

        table.add_row(icon, r.name, detail)

    console.print(table)

    if preflight_mod.has_hard_failure(results):
        console.print("[red]Preflight failed.[/red] Fix the issues above, then retry.")
        sys.exit(1)

    console.print("[green]All checks passed — ready to run.[/green]")


# ── run ───────────────────────────────────────────────────────────────────────

@cli.command()
@_MANIFEST_OPT
def run(manifest_path: str) -> None:
    """Start the full ML application stack (6-stage lifecycle)."""
    orchestrator.run(manifest_path, emit=_make_run_emitter())


def _make_run_emitter() -> orchestrator._Emit:
    """Build the Rich-formatted emit function for `vigilantpack run`."""

    def emit(event: StageEvent) -> None:
        stage = event.stage
        etype = event.event
        msg   = event.message
        meta  = event.metadata

        if etype == "started":
            # Skip per-model pull-progress lines to keep output clean;
            # only log meaningful started events
            if meta.get("pct") is not None:
                pct = meta["pct"]
                console.print(f"   [dim]{msg}[/dim]", end="\r")
            elif meta.get("skipped"):
                console.print(f"  [dim]–[/dim] {msg}")
            else:
                console.print(f"  [dim]→[/dim] [bold]{stage}[/bold]  {msg}")

        elif etype == "succeeded":
            if stage == "READY":
                _render_ready(meta)
            elif stage in ("INFRA", "MODELS", "RUNTIME") and "service" not in meta and "model" not in meta:
                # Top-level stage completion
                console.print(f"  [green]✓[/green] [bold]{stage}[/bold]  {msg}")
            elif "service" in meta or "model" in meta:
                # Per-service / per-model confirmation.  For models, clear any
                # in-progress \r line before printing so no progress remnant shows.
                if "model" in meta:
                    sys.stdout.write("\r" + " " * 72 + "\r")
                    sys.stdout.flush()
                console.print(f"    [green]✓[/green] {msg}")

        elif etype == "failed":
            ft = meta.get("failure_type", "hard")
            color = "red" if ft == "hard" else "yellow"
            console.print(f"  [{color}]✗[/{color}] [bold]{stage}[/bold]  {msg}")
            if "results" in meta:
                _render_check_results(meta["results"])

        elif etype == "warning":
            console.print(f"  [yellow]⚠[/yellow] {msg}")

    return emit


def _render_ready(meta: dict) -> None:
    app     = meta.get("app_name", "app")
    version = meta.get("app_version", "")
    elapsed = meta.get("elapsed_seconds", 0)
    degraded = set(meta.get("degraded_models", []))
    rt_health = meta.get("runtime_health", "")
    api_base  = rt_health.rsplit("/", 1)[0] if rt_health else ""

    console.print()
    console.rule(f"[bold green]{app}  v{version}  ready  ({elapsed}s)[/bold green]")
    if api_base:
        console.print(f"\n  API   [link={api_base}]{api_base}[/link]")
    if degraded:
        console.print()
        console.print("  [yellow]Degraded (warmup failed — first query may be slow):[/yellow]")
        for name in sorted(degraded):
            console.print(f"    [yellow]⚠[/yellow]  {name}")
    console.print()


def _render_check_results(results: list[dict]) -> None:
    for r in results:
        s = r["status"]
        icon = {"PASS": "[green]✓[/green]", "WARN": "[yellow]⚠[/yellow]", "FAIL": "[red]✗[/red]"}.get(s, " ")
        console.print(f"    {icon}  {r['name']}: {r['message']}")
        if r.get("fix") and s != "PASS":
            console.print(f"       [dim]→ {r['fix']}[/dim]")


# ── stop ──────────────────────────────────────────────────────────────────────

@cli.command()
@_MANIFEST_OPT
def stop(manifest_path: str) -> None:
    """Stop all services. Volumes are preserved."""
    m = _load_or_exit(manifest_path)
    compose = m["compose"]
    console.print(f"Stopping services in [bold]{compose}[/bold] …")
    services_mod.stop_all(compose)
    console.print("[green]✓[/green] Stopped. Data volumes preserved.")


# ── logs ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("service", required=False)
@click.option("--follow", "-f", is_flag=True, help="Follow log output (Ctrl-C to stop)")
@_MANIFEST_OPT
def logs(service: str | None, follow: bool, manifest_path: str) -> None:
    """Stream logs for all services or a specific SERVICE."""
    m = _load_or_exit(manifest_path)
    services_mod.stream_logs(m["compose"], service, follow)


# ── status ────────────────────────────────────────────────────────────────────

@cli.command()
@_MANIFEST_OPT
def status(manifest_path: str) -> None:
    """Show health of all services and models."""
    m = _load_or_exit(manifest_path)

    table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
    table.add_column("", width=3, no_wrap=True)
    table.add_column("Name",     min_width=22)
    table.add_column("Type",     width=10)
    table.add_column("Status")

    # Infrastructure services
    for name, svc in m.get("services", {}).items():
        ok, label = _health_check(svc["health"])
        icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
        table.add_row(icon, name, "service", label)

    # Models
    ollama_url = manifest_mod.ollama_url(m)
    present    = models_mod.list_present(ollama_url)
    for cfg in m.get("models", []):
        name     = cfg["name"]
        required = cfg.get("required", True)
        if name in present:
            table.add_row("[green]✓[/green]", name, "model", "downloaded")
        else:
            icon  = "[red]✗[/red]" if required else "[yellow]⚠[/yellow]"
            label = "not downloaded (required)" if required else "not downloaded (optional)"
            table.add_row(icon, name, "model", label)

    # Runtime
    runtime = m.get("runtime", {})
    ok, label = _health_check(runtime.get("health", ""))
    icon = "[green]✓[/green]" if ok else "[red]✗[/red]"
    table.add_row(icon, runtime.get("service", "app"), "runtime", label)

    console.print(table)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_or_exit(path: str) -> dict:
    try:
        return manifest_mod.load(path)
    except ManifestError as exc:
        err_console.print(f"[red]✗ Manifest error:[/red] {exc}")
        sys.exit(1)


def _health_check(url: str) -> tuple[bool, str]:
    if not url:
        return False, "no URL configured"
    from urllib.parse import urlparse
    if urlparse(url).scheme == "tcp":
        import socket
        parsed = urlparse(url)
        host = parsed.hostname or "localhost"
        port = parsed.port
        if not port:
            return False, f"invalid TCP URL: {url}"
        try:
            with socket.create_connection((host, port), timeout=3.0):
                return True, f"TCP reachable  ({url})"
        except OSError:
            return False, f"unreachable  ({url})"
    try:
        resp = httpx.get(url, timeout=3.0)
        return resp.status_code < 500, f"HTTP {resp.status_code}  ({url})"
    except httpx.ConnectError:
        return False, f"unreachable  ({url})"
    except httpx.TimeoutException:
        return False, f"timeout  ({url})"
    except Exception as exc:
        return False, str(exc)
