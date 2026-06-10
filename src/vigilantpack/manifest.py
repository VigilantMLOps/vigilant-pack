"""
manifest.py — load and validate vigilant.yaml as a plain dict.

No Pydantic. Schema is not stable enough for typed models in V1.
Validation is explicit assertions with clear field-path error messages.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import yaml


class ManifestError(Exception):
    """Raised for any manifest parse, validation, or env-loading problem."""


# ── Public API ────────────────────────────────────────────────────────────────

def load(path: str | Path) -> dict:
    """
    Parse vigilant.yaml, load the .env file (if configured), resolve ${VAR}
    references, validate required structure, and return a plain dict.
    """
    path = Path(path)
    if not path.exists():
        raise ManifestError(f"manifest not found: {path}")

    try:
        data = yaml.safe_load(path.read_text())
    except yaml.YAMLError as exc:
        raise ManifestError(f"YAML parse error: {exc}") from exc

    if not isinstance(data, dict):
        raise ManifestError("manifest must be a YAML mapping at the top level")

    _load_env(data)     # populate os.environ from .env before resolving
    _resolve_env(data)  # substitute ${VAR} references throughout
    _validate(data)
    return data


def ollama_url(manifest: dict) -> str:
    """Extract the Ollama base URL from the manifest's ollama service health URL."""
    health = manifest.get("services", {}).get("ollama", {}).get("health", "")
    if health:
        from urllib.parse import urlparse
        parsed = urlparse(health)
        return f"{parsed.scheme}://{parsed.netloc}"
    return "http://localhost:11434"


# ── Internal ──────────────────────────────────────────────────────────────────

def _load_env(data: dict) -> None:
    env_file = data.get("env", {}).get("file")
    if not env_file:
        return
    env_path = Path(env_file)
    if not env_path.exists():
        return  # missing .env is a warning, not an error — surfaced by preflight
    from dotenv import dotenv_values
    for key, val in dotenv_values(env_path).items():
        if key not in os.environ:   # system env takes precedence
            os.environ[key] = val or ""


def _resolve_env(obj: object) -> None:
    """Recursively substitute ${VAR} in all string values. Unresolved refs are left as-is."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                obj[k] = _sub(v)
            else:
                _resolve_env(v)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, str):
                obj[i] = _sub(item)
            else:
                _resolve_env(item)


def _sub(s: str) -> str:
    def replace(m: re.Match) -> str:
        val = os.environ.get(m.group(1))
        return val if val is not None else m.group(0)
    return re.sub(r"\$\{([^}]+)\}", replace, s)


def _validate(data: dict) -> None:
    _req(data, "vigilantpack", "top-level `vigilantpack` version key")
    _req(data, "app",          "top-level `app` section")
    _req(data["app"], "name",    "app.name")
    _req(data["app"], "version", "app.version")
    _req(data, "compose", "top-level `compose` (path to docker-compose file)")

    compose_path = Path(data["compose"])
    if not compose_path.exists():
        raise ManifestError(f"compose file not found: {compose_path}")

    services = data.get("services", {})
    if not isinstance(services, dict) or not services:
        raise ManifestError("at least one entry is required under `services:`")
    for name, svc in services.items():
        if not isinstance(svc, dict):
            raise ManifestError(f"services.{name} must be a mapping")
        _req(svc, "health", f"services.{name}.health")

    for i, model in enumerate(data.get("models", [])):
        if not isinstance(model, dict):
            raise ManifestError(f"models[{i}] must be a mapping")
        _req(model, "name", f"models[{i}].name")

    runtime = data.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ManifestError("`runtime` must be a mapping")
    _req(runtime, "service", "runtime.service")
    _req(runtime, "health",  "runtime.health")


def _req(d: dict, key: str, label: str) -> None:
    if not d.get(key):
        raise ManifestError(f"required field missing or empty: {label}")
